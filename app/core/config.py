from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # BTP Destination resolving S/4HANA - used only when s4_base_url is empty.
    s4_destination_name: str = "S4HANA_MATERIAL_API"

    # Direct-connect mode for local dev - bypasses BTP Destination/Connectivity.
    s4_base_url: str = ""
    s4_username: str = ""
    s4_password: str = ""

    # Base URL this app is reachable at - used in AgentCard URLs and report links.
    app_public_url: str = "http://localhost:8000"

    report_download_ttl_seconds: int = 60 * 15

    # SAP AI Core - direct Foundation Model access (app/services/ai_core.py),
    # used by all 4 agents to (a) parse free-text task input into parameters
    # and (b) phrase natural-language response summaries. When False, agents
    # only accept structured JSON input and use their own template summaries
    # (current behavior, unaffected). NOTE: credentials themselves
    # (AICORE_CLIENT_ID/CLIENT_SECRET/AUTH_URL/BASE_URL/RESOURCE_GROUP) are
    # read directly from the process environment, not through this Settings
    # class - see main.py's load_dotenv() call.
    ai_core_enabled: bool = False
    ai_core_model_name: str = "gpt-4o-mini"

    # Preferred over ai_core_model_name when set - skips the catalog lookup.
    llm_deployment_id: str = ""

    # Max LLM tool-calling runs in flight at once, per worker process - extra
    # requests wait for a free slot instead of all hitting AI Core together
    # (which gets 429-throttled). A request that can't get a slot within
    # agent_queue_timeout_seconds gets a "busy, try again" reply.
    agent_max_concurrency: int = 5
    agent_queue_timeout_seconds: int = 60

    # IAS app-to-app authentication for the A2A agent endpoints (app/core/ias_auth.py).
    # Off by default: when off, /a2a/* accepts calls without a token (today's behaviour).
    # When on, every POST to /a2a/* needs "Authorization: Bearer <IAS token>" whose issuer
    # is ias_issuer, whose audience is ias_audience (the IAS application's client ID) and
    # whose ias_apis claim contains ias_required_api. Agent cards stay public for discovery.
    ias_auth_enabled: bool = False
    ias_issuer: str = ""  # e.g. https://<tenant>.accounts.ondemand.com
    ias_audience: str = ""  # client ID of the IAS application; comma-separate to allow several
    ias_required_api: str = "api_read_access"  # "none" = skip the permission check

    # Sign-in for the web page's chat (app/core/web_login.py): users log in with IAS
    # (OpenID Connect, authorization-code flow) and get a signed session cookie that the
    # A2A guard accepts in place of a token. Active only when IAS auth is on and both
    # secrets below are set. The IAS application must list
    # <app_public_url>/auth/callback as a redirect URI.
    web_login_client_id: str = ""  # empty = first value of ias_audience
    web_login_client_secret: str = ""
    web_session_secret: str = ""  # random string, at least 32 characters; signs the session cookie
    web_session_hours: int = 8

    # Logs full raw A2A request/response bodies - debugging only. Off by
    # default: it buffers every response in memory and writes S/4 data to logs.
    a2a_wire_debug: bool = False


@lru_cache
def get_settings() -> Settings:
    return Settings()