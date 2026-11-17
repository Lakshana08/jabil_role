"""Signed, stateless cookies for the web page's sign-in (app/core/web_login.py).

A cookie value is base64url(JSON payload) + "." + base64url(HMAC-SHA256), signed with
settings.web_session_secret. Nothing is stored server-side, so it works the same across
uvicorn workers, instances and restarts. The payload carries its own "exp".
"""

import base64
import hashlib
import hmac
import json
import time

from fastapi import Request

from app.core.config import get_settings

SESSION_COOKIE = "sm_session"
LOGIN_COOKIE = "sm_login"
# Sent by the web page on every A2A call. A cross-site page cannot add a custom header
# without a CORS preflight (which this app never allows), so together with SameSite=Lax
# this stops other sites from using a signed-in user's cookie (CSRF).
WEB_CLIENT_HEADER = "X-Smart-Material-Client"
WEB_CLIENT_VALUE = "web"


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def _signature(body: str) -> str:
    key = get_settings().web_session_secret.encode()
    return _b64(hmac.new(key, body.encode(), hashlib.sha256).digest())


def login_enabled() -> bool:
    settings = get_settings()
    return bool(
        settings.ias_auth_enabled
        and settings.web_login_client_secret.strip()
        and len(settings.web_session_secret.strip()) >= 32
    )


def sign(payload: dict, ttl_seconds: int) -> str:
    body = _b64(json.dumps({**payload, "exp": int(time.time()) + ttl_seconds}, separators=(",", ":")).encode())
    return f"{body}.{_signature(body)}"


def unsign(value: str | None) -> dict | None:
    """The payload if the value is untampered and unexpired, else None."""
    if not value or "." not in value or not get_settings().web_session_secret.strip():
        return None
    body, sig = value.rsplit(".", 1)
    if not hmac.compare_digest(sig, _signature(body)):
        return None
    try:
        payload = json.loads(_unb64(body))
    except ValueError:
        return None
    if not isinstance(payload, dict) or payload.get("exp", 0) < time.time():
        return None
    return payload


def session_user(request: Request) -> dict | None:
    """The signed-in web user, or None. Always None while web sign-in is switched off."""
    if not login_enabled():
        return None
    return unsign(request.cookies.get(SESSION_COOKIE))
