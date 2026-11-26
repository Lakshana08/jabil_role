"""IAS app-to-app authentication for the A2A agent endpoints.

Follows SAP's "Joule A2A: Secure Code Based Agents with IAS" pattern: a caller
(Joule via an OAuth2ClientCredentials destination, or Postman) gets a
token from the IAS tenant with the client-credentials flow and sends it as
"Authorization: Bearer <token>". This checks that token before any agent runs:

  - signature: verified against the IAS tenant's public keys (<issuer>/oauth2/certs)
  - iss:       is our IAS tenant (settings.ias_issuer)
  - aud:       is our IAS application (settings.ias_audience)
  - exp:       not expired
  - ias_apis:  contains settings.ias_required_api (e.g. api_read_access)

The token identifies the calling *application*, not the end user - S/4 is still
called with the destination's technical user (Principal Propagation is a
separate pattern).

Only JSON-RPC calls (POST /a2a/...) are checked. Agent cards (GET
.../.well-known/agent-card.json) stay public for A2A discovery, as do /health,
the report download links and the web UI files. Switched by
settings.ias_auth_enabled - off by default, so nothing changes until IAS is set up.
"""

import asyncio
import logging

import jwt
from fastapi import Request
from fastapi.responses import JSONResponse

from app.core.config import get_settings

logger = logging.getLogger(__name__)

# PyJWKClient caches the IAS public keys, so the keys are fetched once an hour
# (or when IAS rotates to an unknown key id), not on every request.
_JWKS_CACHE_SECONDS = 3600
_jwks_clients: dict[str, jwt.PyJWKClient] = {}


class AuthError(Exception):
    """Rejected token; the message is safe to return to the caller."""


def _jwks_client(issuer: str) -> jwt.PyJWKClient:
    client = _jwks_clients.get(issuer)
    if client is None:
        client = jwt.PyJWKClient(f"{issuer}/oauth2/certs", cache_keys=True, lifespan=_JWKS_CACHE_SECONDS)
        _jwks_clients[issuer] = client
    return client


def _signing_key(token: str, issuer: str):
    """Blocking (may fetch the IAS key set) - called via asyncio.to_thread."""
    return _jwks_client(issuer).get_signing_key_from_jwt(token).key


def _issuer_forms(configured: str) -> tuple[str, list[str]]:
    """(tenant base URL, accepted iss values). IAS tenants differ in how they write their
    issuer - some as "https://<tenant>.accounts.ondemand.com", some (e.g. anjgnszvi) as the
    bare host "<tenant>.accounts.ondemand.com" - so both forms of the same tenant are
    accepted, and the keys are always fetched over https."""
    host = configured.strip().rstrip("/").removeprefix("https://").removeprefix("http://")
    return f"https://{host}", [f"https://{host}", host]


def verify_token(token: str) -> dict:
    """Validates an IAS token and returns its claims, or raises AuthError."""
    settings = get_settings()
    audiences = [a.strip() for a in settings.ias_audience.split(",") if a.strip()]
    if not settings.ias_issuer.strip() or not audiences:
        # Fail closed: auth switched on but not configured must never mean "let everyone in".
        logger.error("IAS auth is enabled but IAS_ISSUER / IAS_AUDIENCE are not set - rejecting all calls")
        raise AuthError("Authentication is not configured")
    base_url, accepted_issuers = _issuer_forms(settings.ias_issuer)
    try:
        key = _signing_key(token, base_url)
        claims = jwt.decode(
            token,
            key,
            algorithms=["RS256"],
            audience=audiences,
            issuer=accepted_issuers,
            options={"require": ["exp", "iss", "aud"]},
        )
    except jwt.ExpiredSignatureError:
        raise AuthError("Token expired") from None
    except jwt.PyJWTError as exc:
        raise AuthError(f"Invalid token: {type(exc).__name__}") from None
    # "none" (or empty) switches the permission check off. Cloud Foundry can't store an empty
    # env value (`cf set-env ... ''` is ignored), so "none" is the way to set it there.
    required_api = settings.ias_required_api.strip()
    if required_api.lower() in ("", "none", "off", "-"):
        required_api = ""
    if required_api and required_api not in (claims.get("ias_apis") or []):
        raise AuthError(f"Token is missing the '{required_api}' permission")
    return claims


def _needs_token(request: Request) -> bool:
    return request.method == "POST" and request.url.path.startswith("/a2a/")


async def ias_auth_middleware(request: Request, call_next):
    if not get_settings().ias_auth_enabled or not _needs_token(request):
        return await call_next(request)

    header = request.headers.get("Authorization", "")
    if not header.startswith("Bearer "):
        return JSONResponse(status_code=401, content={"detail": "Missing token"})
    try:
        claims = await asyncio.to_thread(verify_token, header[len("Bearer "):].strip())
    except AuthError as exc:
        logger.warning("Rejected A2A call to %s: %s", request.url.path, exc)
        return JSONResponse(status_code=401, content={"detail": str(exc)})
    except Exception:
        # e.g. the IAS key endpoint unreachable - reject, never let the call through unchecked.
        logger.exception("Could not verify IAS token for %s", request.url.path)
        return JSONResponse(status_code=401, content={"detail": "Token could not be verified"})

    request.state.caller = claims.get("azp") or claims.get("sub")
    return await call_next(request)
