"""Read-only OData client for S/4HANA - handles sap-client, TLS settings and
the on-premise Connectivity proxy transparently.

One instance is shared by every concurrent request, so it is built to be
thread-safe: a pooled requests.Session (connections reused instead of a new
TCP/TLS handshake per call) with automatic retry and backoff on transient S/4
errors."""

import logging
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
    # Never let the shared session pick up cookies, so one user's call can't
    # carry state into another's.
    session.cookies.set_policy(DefaultCookiePolicy(allowed_domains=[]))
    return session


class S4Client:
    def __init__(self, destination_service: DestinationService, destination_name: str):
        self._destination_service = destination_service
        self._destination_name = destination_name
        self._session = _build_session()

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