# Run: .venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8000
"""FastAPI entrypoint that hosts three independent A2A agents - querying,
acting on, and reporting on live S/4HANA material data - plus the assistant
router and a plain REST route for downloading generated report files.

Each agent is a separate A2A server (its own AgentCard/discovery endpoint,
its own JSON-RPC endpoint) mounted at its own path prefix within this one
deployable app:

  /a2a/planner -> Material Planner Agent -> /.well-known/agent-card.json, /
  /a2a/action  -> Material Action Agent  -> /.well-known/agent-card.json, /
  /a2a/report  -> Material Report Agent  -> /.well-known/agent-card.json, /
  /a2a/assistant -> Smart Material Assistant (router + confirm-before-change, used by teams_bot/)

so a caller (e.g. a Joule Studio A2A code-based agent registration) sees
distinct agents at
http://<host>/a2a/planner/.well-known/agent-card.json, etc.

/a2a/query is kept as an alias of /a2a/planner (the agent's former name), so
existing callers such as the Joule SMART_MATERIAL_QUERY destination keep working.
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles

logger = logging.getLogger("a2a_wire_debug")

# Must run before app.services.ai_core is imported: it reads
# AICORE_CLIENT_ID/CLIENT_SECRET/AUTH_URL/BASE_URL/RESOURCE_GROUP directly
# from os.environ rather than through app.core.config.Settings, so .env
# values need to land in the real process environment for local development.
load_dotenv()

from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import add_a2a_routes_to_fastapi, create_agent_card_routes, create_jsonrpc_routes
from a2a.utils.constants import DEFAULT_RPC_URL
from starlette.routing import Route

from app.a2a_agents.action_agent import ActionAgentExecutor, build_action_agent_card
from app.a2a_agents.assistant_agent import AssistantAgentExecutor, build_assistant_agent_card
from app.a2a_agents.common import BoundedTaskStore
from app.a2a_agents.planner_agent import PlannerAgentExecutor, build_planner_agent_card
from app.a2a_agents.report_agent import ReportAgentExecutor, build_report_agent_card
from app.core.config import get_settings
from app.core.ias_auth import ias_auth_middleware
from app.core.web_login import router as web_login_router
from app.routers.reports import router as reports_router

_AGENTS = {
    "planner": (build_planner_agent_card, PlannerAgentExecutor),
    "action": (build_action_agent_card, ActionAgentExecutor),
    "report": (build_report_agent_card, ReportAgentExecutor),
    # Router for chat channels without routing of their own (Teams bot); changes need confirmation.
    "assistant": (build_assistant_agent_card, AssistantAgentExecutor),
}

# Former path -> current agent. Served by the same agent app and handler.
_ALIASES = {"query": "planner"}


def _build_agent_app(path_segment: str, build_card, executor_cls) -> tuple[FastAPI, DefaultRequestHandler]:
    settings = get_settings()
    base_url = f"{settings.app_public_url}/a2a/{path_segment}/"
    card = build_card(base_url=base_url)
    handler = DefaultRequestHandler(
        agent_executor=executor_cls(),
        task_store=BoundedTaskStore(),
        agent_card=card,
    )
    agent_app = FastAPI(title=card.name)
    add_a2a_routes_to_fastapi(
        agent_app,
        agent_card_routes=create_agent_card_routes(card),
        jsonrpc_routes=create_jsonrpc_routes(handler, rpc_url=DEFAULT_RPC_URL, enable_v0_3_compat=True),
    )
    return agent_app, handler


# Every blocking call (S/4 HTTP, Excel parsing, report building, LangChain's
# sync tools) runs in the default thread pool via asyncio.to_thread /
# run_in_executor. Python's default is min(32, cpu+4) threads - only ~5-6 on a
# small CF container - so a handful of slow S/4 calls would queue everyone
# else behind them. Sized well above agent_max_concurrency.
_BLOCKING_IO_THREADS = 32
_WIRE_DEBUG_MAX_BODY = 2000


@asynccontextmanager
async def _lifespan(app: FastAPI):
    asyncio.get_running_loop().set_default_executor(
        ThreadPoolExecutor(max_workers=_BLOCKING_IO_THREADS, thread_name_prefix="blocking-io")
    )
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="Smart Material AI - A2A Agents", lifespan=_lifespan)

    @app.middleware("http")
    async def log_a2a_wire_traffic(request: Request, call_next):
        # Debug only (A2A_WIRE_DEBUG=true): logs raw request/response bodies
        # for /a2a/* calls. Off by default - it buffers every full response in
        # memory and writes S/4 data into the logs.
        if get_settings().a2a_wire_debug and request.url.path.startswith("/a2a/"):
            body = await request.body()
            logger.warning(
                "A2A REQUEST %s %s headers=%s body=%s",
                request.method,
                request.url.path,
                dict(request.headers),
                body.decode("utf-8", errors="replace")[:_WIRE_DEBUG_MAX_BODY],
            )
            response = await call_next(request)
            chunks = [chunk async for chunk in response.body_iterator]
            resp_body = b"".join(chunks)
            logger.warning("A2A RESPONSE %s status=%s body=%s", request.url.path, response.status_code, resp_body.decode("utf-8", errors="replace")[:_WIRE_DEBUG_MAX_BODY])
            from starlette.responses import Response as _Response

            return _Response(
                content=resp_body,
                status_code=response.status_code,
                headers=dict(response.headers),
                media_type=response.media_type,
            )
        return await call_next(request)

    # Registered after the wire-debug middleware, so it runs first: a call without a
    # valid IAS token is rejected before anything else touches it. No-op unless
    # IAS_AUTH_ENABLED=true (app/core/ias_auth.py).
    app.middleware("http")(ias_auth_middleware)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    app.include_router(reports_router)
    # IAS sign-in for the web page's chat (/auth/login, /auth/callback, /auth/me, /auth/logout).
    app.include_router(web_login_router)

    built = {}
    for path_segment, (build_card, executor_cls) in _AGENTS.items():
        built[path_segment] = _build_agent_app(path_segment, build_card, executor_cls)
    segments = list(_AGENTS) + list(_ALIASES)
    for path_segment in segments:
        agent_app, handler = built[_ALIASES.get(path_segment, path_segment)]
        app.mount(f"/a2a/{path_segment}", agent_app)
        # Mounting means a bare POST to /a2a/<segment> (no trailing slash) 307-
        # redirects to /a2a/<segment>/ - some A2A clients (e.g. Joule) don't
        # follow redirects on POST, so the request just fails. Register the
        # same JSON-RPC endpoint directly on the bare path too, bypassing the
        # Mount's redirect entirely for that case.
        bare_route = create_jsonrpc_routes(handler, rpc_url=f"/a2a/{path_segment}", enable_v0_3_compat=True)[0]
        app.routes.append(Route(bare_route.path, endpoint=bare_route.endpoint, methods=["POST"]))

    # Frontend UI (frontend/public/index.html). On Cloud Foundry it is its own
    # app (frontend/manifest.yml) and frontend/ is left out of this app's
    # upload (.cfignore), so this only serves the page for local development.
    # Registered last: Starlette matches mounts in registration order, and a
    # root-path Mount registered earlier would swallow /health, /a2a/*, etc.
    if Path("frontend/public").is_dir():
        app.mount("/", StaticFiles(directory="frontend/public", html=True), name="frontend")

    return app


app = create_app()
