"""Generic OData v2/v4 client for S/4HANA - handles CSRF token fetch/retry
for mutating calls and the on-premise Connectivity proxy transparently.

One instance is shared by every concurrent request, so it is built to be
thread-safe: a pooled requests.Session (connections reused instead of a new
TCP/TLS handshake per call), automatic retry with backoff on transient S/4
errors for reads only, and a lock around the shared CSRF token."""

import logging
import threading
from http.cookiejar import DefaultCookiePolicy
from typing import Any

import requests
import urllib3
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from app.core.config import get_settings
from app.services.destination import DestinationService, ResolvedDestination, get_destination_service

logger = logging.getLogger(__name__)


class S4ClientError(Exception):
    def __init__(self, status_code: int, detail: Any):
        self.status_code = status_code
        self.detail = detail  # full response, for logs
        super().__init__(f"S/4 request failed ({status_code}): {_readable(detail)}")


def _readable(detail: Any) -> Any:
    """Just SAP's own message and code from an OData error body (e.g. "Order 1000000 was not
    changed. (PP_ODATA_API_PRODORD/006)") instead of the whole JSON with inner error details."""
    if isinstance(detail, dict) and isinstance(detail.get("error"), dict):
        error = detail["error"]
        message = error.get("message")
        text = message.get("value") if isinstance(message, dict) else message
        if text:
            return f"{text} ({error['code']})" if error.get("code") else text
    return detail


_POOL_SIZE = 20

# GET only - a retried POST/PATCH/DELETE could apply the same change twice.
# raise_on_status=False hands the final response back to
# _raise_for_odata_error instead of raising urllib3's own exception.
_READ_RETRY = Retry(
    total=3,
    backoff_factor=0.5,
    status_forcelist=(429, 502, 503, 504),
    allowed_methods=frozenset({"GET"}),
    raise_on_status=False,
)


def _build_session() -> requests.Session:
    session = requests.Session()
    adapter = HTTPAdapter(pool_connections=_POOL_SIZE, pool_maxsize=_POOL_SIZE, max_retries=_READ_RETRY)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    # Never let the shared session pick up cookies on its own - the CSRF
    # session cookies are passed explicitly per mutating call (same as before),
    # so one user's GET can't silently swap the session another user's CSRF
    # token is bound to.
    session.cookies.set_policy(DefaultCookiePolicy(allowed_domains=[]))
    return session


class S4Client:
    def __init__(self, destination_service: DestinationService, destination_name: str):
        self._destination_service = destination_service
        self._destination_name = destination_name
        self._session = _build_session()
        # (token, cookies) kept as one tuple so a reader never sees a new
        # token paired with old cookies; guarded by _csrf_lock.
        self._csrf: tuple[str | None, Any] | None = None
        self._csrf_lock = threading.Lock()

    def get(self, path: str, params: dict | None = None) -> dict:
        resolved = self._resolve()
        kwargs = self._base_kwargs(resolved)
        kwargs["headers"]["Accept"] = "application/json"
        merged_params = {**self._format_params(path), **self._client_params(resolved), **(params or {})}
        resp = self._session.get(self._url(resolved, path), params=merged_params, **kwargs)
        self._raise_for_odata_error(resp)
        return resp.json()

    def get_text(self, path: str, accept: str = "application/xml") -> str:
        """Raw-text GET - used for a service's $metadata document (XML)."""
        resolved = self._resolve()
        kwargs = self._base_kwargs(resolved)
        kwargs["headers"]["Accept"] = accept
        resp = self._session.get(self._url(resolved, path), params=self._client_params(resolved), **kwargs)
        self._raise_for_odata_error(resp)
        return resp.text

    def post(self, path: str, json_body: dict | None = None) -> dict:
        return self._mutate("POST", path, json_body)

    def patch(self, path: str, json_body: dict | None = None, extra_headers: dict | None = None) -> dict:
        return self._mutate("PATCH", path, json_body, extra_headers=extra_headers)

    def delete(self, path: str, extra_headers: dict | None = None) -> dict:
        return self._mutate("DELETE", path, None, extra_headers=extra_headers)

    def get_etag(self, path: str) -> str | None:
        """GETs a keyed entity and returns its ETag, needed for If-Match on a PATCH/DELETE
        against an entity with optimistic concurrency control (e.g. A_ProductionOrder_2) -
        unlike A_Product, which exposes no ETag at all."""
        return self.get_entity_with_etag(path)[1]

    def get_entity_with_etag(self, path: str) -> tuple[dict, str | None]:
        """GETs a keyed entity: (its fields, its ETag). Prefers the HTTP ETag response header;
        falls back to the OData v2 JSON body's __metadata.etag if that's absent."""
        resolved = self._resolve()
        kwargs = self._base_kwargs(resolved)
        kwargs["headers"]["Accept"] = "application/json"
        params = {**self._format_params(path), **self._client_params(resolved)}
        resp = self._session.get(self._url(resolved, path), params=params, **kwargs)
        self._raise_for_odata_error(resp)
        entity = resp.json().get("d", {})
        etag = resp.headers.get("ETag") or entity.get("__metadata", {}).get("etag")
        return entity, etag

    def _mutate(self, method: str, path: str, json_body: dict | None, extra_headers: dict | None = None) -> dict:
        resolved = self._resolve()
        csrf = self._get_csrf(resolved, path)

        def attempt(token_and_cookies) -> requests.Response:
            token, cookies = token_and_cookies
            kwargs = self._base_kwargs(resolved)
            kwargs["headers"]["Accept"] = "application/json"
            kwargs["headers"]["X-CSRF-Token"] = token
            if extra_headers:
                kwargs["headers"].update(extra_headers)
            kwargs["cookies"] = cookies
            return self._session.request(
                method, self._url(resolved, path), params=self._client_params(resolved), json=json_body, **kwargs
            )

        resp = attempt(csrf)
        # SAP Gateway's actual signal for a rejected/expired CSRF token is the
        # literal header value "Required" on the 403 response - it never
        # contains the substring "csrf", so check for that instead.
        if resp.status_code == 403 and resp.headers.get("X-CSRF-Token", "").lower() == "required":
            resp = attempt(self._get_csrf(resolved, path, stale_token=csrf[0]))

        self._raise_for_odata_error(resp)
        return resp.json() if resp.content else {}

    def _get_csrf(self, resolved: ResolvedDestination, path: str, stale_token: str | None = None):
        """Returns the shared (token, cookies), fetching one if there is none
        yet - or if the caller's token was just rejected (stale_token) and no
        other thread has already replaced it. The lock means a burst of
        concurrent writes triggers one fetch, not one per request, and one
        thread can never wipe the token while another is mid-request."""
        with self._csrf_lock:
            if self._csrf is not None and self._csrf[0] != stale_token:
                return self._csrf
            self._csrf = self._fetch_csrf(resolved, path)
            return self._csrf

    def _fetch_csrf(self, resolved: ResolvedDestination, path: str):
        kwargs = self._base_kwargs(resolved)
        kwargs["headers"]["X-CSRF-Token"] = "Fetch"
        # Fetch against the entity SET (key predicate stripped), not the exact
        # keyed entity being mutated - the token isn't tied to a specific
        # entity, and the keyed path 404s whenever that entity doesn't exist
        # yet (e.g. PATCHing a Product/Language description combo that
        # hasn't been created), which would otherwise block CSRF fetch for
        # every subsequent POST/create fallback too. Must still be an actual
        # service path, not the bare host root - GETting resolved.url alone
        # 404s since there's no service mounted there.
        entity_set_path = path.split("(", 1)[0]
        resp = self._session.get(
            self._url(resolved, entity_set_path), params=self._client_params(resolved), **kwargs
        )
        resp.raise_for_status()
        return resp.headers.get("X-CSRF-Token"), resp.cookies

    def _resolve(self) -> ResolvedDestination:
        return self._destination_service.resolve(self._destination_name)

    @staticmethod
    def _url(resolved: ResolvedDestination, path: str) -> str:
        return f"{resolved.url.rstrip('/')}/{path.lstrip('/')}"

    @staticmethod
    def _format_params(path: str) -> dict:
        """OData v2 only: this system's SAP Gateway returns XML for v2 unless
        $format=json is passed as a query param, not just via the Accept header.
        OData v4 (/sap/opu/odata4/...) answers JSON by default."""
        return {} if "/odata4/" in path else {"$format": "json"}

    @staticmethod
    def _client_params(resolved: ResolvedDestination) -> dict:
        return {"sap-client": resolved.sap_client} if resolved.sap_client else {}

    @staticmethod
    def _base_kwargs(resolved: ResolvedDestination) -> dict:
        auth = None
        if resolved.authentication == "BasicAuthentication":
            auth = (resolved.username, resolved.password)
        return {
            "proxies": resolved.requests_proxies,
            "headers": dict(resolved.extra_headers),
            "auth": auth,
            "timeout": 30,
            "verify": get_settings().s4_verify_ssl,
        }

    @staticmethod
    def _raise_for_odata_error(resp: requests.Response) -> None:
        if resp.status_code >= 400:
            try:
                detail = resp.json()
            except ValueError:
                detail = resp.text
            raise S4ClientError(resp.status_code, detail)


if not get_settings().s4_verify_ssl:
    # One startup warning instead of urllib3's InsecureRequestWarning on every single call.
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    logger.warning("S4_VERIFY_SSL is false - S/4 TLS certificates are NOT verified. Dev systems only.")

_client = S4Client(get_destination_service(), get_settings().s4_destination_name)


def get_s4_client() -> S4Client:
    return _client