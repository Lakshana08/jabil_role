"""IAS sign-in for the web page's chat (OpenID Connect authorization-code flow + PKCE).

  /auth/login     -> redirects the browser to the IAS sign-in page
  /auth/callback  -> IAS sends the user back here with a one-time code; the code is swapped
                     for an ID token (server-side, with the client secret), the token is
                     verified like any IAS token, and a signed session cookie is set
  /auth/me        -> who is signed in (the page uses it to show "Sign in" / the user's name)
  /auth/logout    -> clears the session

The A2A guard (app/core/ias_auth.py) accepts that session cookie in place of a Bearer token
for calls from this page, so the page never sees a client secret or token. Joule, the Teams
bot and Postman keep using Bearer tokens as before.
"""

import asyncio
import base64
import hashlib
import logging
import secrets
from urllib.parse import urlencode

import httpx
import jwt
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, RedirectResponse

from app.core.config import get_settings
from app.core.ias_auth import _issuer_forms, _signing_key
from app.core.web_session import LOGIN_COOKIE, SESSION_COOKIE, login_enabled, session_user, sign, unsign

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/auth", include_in_schema=False)

_LOGIN_TTL_SECONDS = 600  # time allowed to complete the IAS sign-in page


def _client_id() -> str:
    settings = get_settings()
    return settings.web_login_client_id.strip() or settings.ias_audience.split(",")[0].strip()


def _web_url() -> str:
    settings = get_settings()
    return (settings.web_public_url.strip() or settings.app_public_url).rstrip("/")


def _redirect_uri() -> str:
    return f"{_web_url()}/auth/callback"


def _secure_cookies() -> bool:
    return _web_url().startswith("https://")


def _pkce_challenge(verifier: str) -> str:
    return base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()


def _failed(reason: str) -> RedirectResponse:
    logger.warning("Web sign-in failed: %s", reason)
    response = RedirectResponse("/?login=failed", status_code=302)
    response.delete_cookie(LOGIN_COOKIE, path="/auth")
    return response


@router.get("/login")
def login() -> RedirectResponse:
    if not login_enabled():
        return RedirectResponse("/", status_code=302)
    base_url, _ = _issuer_forms(get_settings().ias_issuer)
    state, nonce, verifier = secrets.token_urlsafe(24), secrets.token_urlsafe(24), secrets.token_urlsafe(48)
    query = urlencode({
        "response_type": "code",
        "client_id": _client_id(),
        "redirect_uri": _redirect_uri(),
        "scope": "openid email profile",
        "state": state,
        "nonce": nonce,
        "code_challenge": _pkce_challenge(verifier),
        "code_challenge_method": "S256",
    })
    response = RedirectResponse(f"{base_url}/oauth2/authorize?{query}", status_code=302)
    response.set_cookie(
        LOGIN_COOKIE, sign({"state": state, "nonce": nonce, "verifier": verifier}, _LOGIN_TTL_SECONDS),
        max_age=_LOGIN_TTL_SECONDS, path="/auth", httponly=True, secure=_secure_cookies(), samesite="lax",
    )
    return response


def _exchange_code(code: str, verifier: str) -> str:
    """Blocking (calls IAS) - run via asyncio.to_thread. Returns the ID token."""
    settings = get_settings()
    base_url, _ = _issuer_forms(settings.ias_issuer)
    resp = httpx.post(
        f"{base_url}/oauth2/token",
        data={
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": _redirect_uri(),
            "client_id": _client_id(),
            "code_verifier": verifier,
        },
        auth=(_client_id(), settings.web_login_client_secret.strip()),
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["id_token"]


def _verify_id_token(id_token: str, nonce: str) -> dict:
    """Blocking (may fetch the IAS keys) - run via asyncio.to_thread."""
    base_url, issuers = _issuer_forms(get_settings().ias_issuer)
    claims = jwt.decode(
        id_token,
        _signing_key(id_token, base_url),
        algorithms=["RS256"],
        audience=_client_id(),
        issuer=issuers,
        options={"require": ["exp", "iss", "aud", "sub"]},
    )
    if claims.get("nonce") != nonce:
        raise jwt.InvalidTokenError("nonce mismatch")
    return claims


@router.get("/callback")
async def callback(request: Request) -> RedirectResponse:
    if not login_enabled():
        return RedirectResponse("/", status_code=302)
    pending = unsign(request.cookies.get(LOGIN_COOKIE))
    params = request.query_params
    if params.get("error"):
        return _failed(f"IAS returned {params.get('error')}: {params.get('error_description', '')}")
    if not pending or not params.get("code") or params.get("state") != pending.get("state"):
        return _failed("missing or mismatched state (expired, or not started from /auth/login)")
    try:
        id_token = await asyncio.to_thread(_exchange_code, params["code"], pending["verifier"])
        claims = await asyncio.to_thread(_verify_id_token, id_token, pending["nonce"])
    except httpx.HTTPStatusError as exc:
        return _failed(f"code exchange refused by IAS: HTTP {exc.response.status_code} {exc.response.text[:200]}")
    except (httpx.HTTPError, jwt.PyJWTError, KeyError, ValueError) as exc:
        return _failed(f"{type(exc).__name__}: {exc}")

    name = " ".join(p for p in (claims.get("given_name"), claims.get("family_name")) if p) or claims.get("email") or claims["sub"]
    user = {"sub": claims["sub"], "email": claims.get("email", ""), "name": name}
    logger.info("Web sign-in: %s", user["email"] or user["sub"])
    hours = get_settings().web_session_hours
    response = RedirectResponse("/", status_code=302)
    response.set_cookie(
        SESSION_COOKIE, sign(user, hours * 3600),
        max_age=hours * 3600, path="/", httponly=True, secure=_secure_cookies(), samesite="lax",
    )
    response.delete_cookie(LOGIN_COOKIE, path="/auth")
    return response


@router.get("/me")
def me(request: Request) -> JSONResponse:
    if not get_settings().ias_auth_enabled:
        # Agents are open - the page can chat without signing in.
        return JSONResponse({"auth_required": False, "login_enabled": False, "user": None})
    user = session_user(request)
    body = {
        "auth_required": True,
        "login_enabled": login_enabled(),
        "user": {"name": user["name"], "email": user["email"]} if user else None,
    }
    return JSONResponse(body, status_code=200 if user else 401)


@router.get("/logout")
def logout() -> RedirectResponse:
    response = RedirectResponse("/", status_code=302)
    response.delete_cookie(SESSION_COOKIE, path="/")
    return response
