"""Load test: fire many concurrent A2A requests at the agents and report
how the app holds up (status codes, "busy" replies, latency, /health
responsiveness while under load).

Runs only from your machine against the URL you give it - local or the
deployed CF app. Note the free-text mode calls real AI Core and S/4HANA,
so every request costs tokens and hits S/4 - start small.

Local runs need no separate terminal: if nothing is answering at the local
URL, this script starts the app itself (one uvicorn worker - multiple
workers crash on Windows with WinError 10022), runs the test, and stops it
again afterwards.

Examples (from the project root):

  # 1. Local, structured (no AI Core) - Planner Agent count via S/4
  .venv\\Scripts\\python.exe tools\\load_test.py --agent query --structured --users 50 --requests 500

  # 2. Local, real free-text path through AI Core + S/4
  .venv\\Scripts\\python.exe tools\\load_test.py --agent query --users 10 --requests 30 \\
      --question "How many production orders are there?"

  # 3. Deployed app
  .venv\\Scripts\\python.exe tools\\load_test.py --url https://smart-material-ai-role.cfapps.us10-001.hana.ondemand.com \\
      --agent query --structured --users 20 --requests 100
"""

import argparse
import asyncio
import os
import statistics
import subprocess
import sys
import time
import uuid
from collections import Counter
from urllib.parse import urlparse

import httpx

DEFAULT_QUESTIONS = {
    "planner": ["How many production orders are there?", "How many product masters are there in total?"],
    "report": ["Give me a CSV of production orders for plant 1010"],
}

DEFAULT_STRUCTURED = {
    "planner": [{"query_type": "production_order", "count_only": True}],
}

BUSY_MARKER = "handling a lot of requests"

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_STARTUP_TIMEOUT_SECONDS = 90


def _app_is_up(base: str) -> bool:
    try:
        return httpx.get(f"{base}/health", timeout=3).status_code == 200
    except httpx.HTTPError:
        return False


def _start_local_app(base: str) -> subprocess.Popen:
    """Starts the app in the background with the same Python running this script."""
    port = str(urlparse(base).port or 8000)
    log_path = os.path.join(_PROJECT_ROOT, "tools", "load_test_app.log")
    print(f"App not running - starting it on port {port} (log: {log_path}) ...")
    log = open(log_path, "w", encoding="utf-8")
    process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", port],
        cwd=_PROJECT_ROOT,
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    deadline = time.time() + _STARTUP_TIMEOUT_SECONDS
    while time.time() < deadline:
        if process.poll() is not None:
            raise SystemExit(f"The app exited during startup - see {log_path} for the error.")
        if _app_is_up(base):
            print("App is up.\n")
            return process
        time.sleep(1)
    process.terminate()
    raise SystemExit(f"The app didn't start within {_STARTUP_TIMEOUT_SECONDS}s - see {log_path}.")


def _body(text: str | None, data: dict | None) -> dict:
    part = {"data": data} if data is not None else {"text": text}
    return {
        "jsonrpc": "2.0",
        "id": str(uuid.uuid4()),
        "method": "SendMessage",
        "params": {"message": {"messageId": str(uuid.uuid4()), "role": "ROLE_USER", "parts": [part]}, "configuration": {}},
    }


def _reply_text(payload: dict) -> str:
    result = payload.get("result") or {}
    parts = (result.get("task") or {}).get("status", {}).get("message", {}).get("parts") or (
        result.get("message") or {}
    ).get("parts", [])
    return next((p.get("text", "") for p in parts if "text" in p), "")


async def _one(client: httpx.AsyncClient, url: str, body: dict, results: list) -> None:
    started = time.perf_counter()
    try:
        resp = await client.post(url, json=body, headers={"a2a-version": "1.0"})
        elapsed = time.perf_counter() - started
        outcome = f"HTTP {resp.status_code}"
        if resp.status_code == 200:
            payload = resp.json()
            if "error" in payload:
                outcome = "JSON-RPC error"
            elif BUSY_MARKER in _reply_text(payload):
                outcome = "busy reply"
            else:
                outcome = "ok"
        results.append((outcome, elapsed))
    except httpx.HTTPError as exc:
        results.append((f"{type(exc).__name__}", time.perf_counter() - started))


async def _watch_health(client: httpx.AsyncClient, base: str, stop: asyncio.Event, samples: list) -> None:
    """Pings /health during the run - if it slows down, something is blocking the event loop."""
    while not stop.is_set():
        started = time.perf_counter()
        try:
            await client.get(f"{base}/health", timeout=10)
            samples.append(time.perf_counter() - started)
        except httpx.HTTPError:
            samples.append(float("inf"))
        await asyncio.sleep(0.25)


def _pct(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * pct))]


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # 127.0.0.1, not "localhost": on Windows localhost resolves to IPv6 ::1
    # first, while uvicorn listens on IPv4 127.0.0.1 only.
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="App base URL")
    parser.add_argument("--agent", default="planner", choices=sorted(DEFAULT_QUESTIONS))
    parser.add_argument("--users", type=int, default=20, help="Requests in flight at the same time")
    parser.add_argument("--requests", type=int, default=100, help="Total requests to send")
    parser.add_argument("--question", action="append", help="Free-text question (repeatable)")
    parser.add_argument("--structured", action="store_true", help="Send JSON data parts - skips the LLM entirely")
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()

    base = args.url.rstrip("/")
    endpoint = f"{base}/a2a/{args.agent}/"
    if args.structured:
        payloads = [(None, d) for d in DEFAULT_STRUCTURED.get(args.agent, [])]
    else:
        payloads = [(q, None) for q in (args.question or DEFAULT_QUESTIONS[args.agent])]
    if not payloads:
        raise SystemExit(f"No default requests for agent '{args.agent}' in this mode - pass --question.")

    started_app = None
    if not _app_is_up(base):
        if urlparse(base).hostname not in ("127.0.0.1", "localhost"):
            raise SystemExit(f"Can't reach the app at {base} - check the URL and that the app is deployed and running.")
        started_app = _start_local_app(base)
    try:
        await _run(args, base, endpoint, payloads)
    finally:
        if started_app is not None:
            started_app.terminate()
            started_app.wait(timeout=15)
            print("\nStopped the app this script started.")


async def _run(args, base: str, endpoint: str, payloads: list) -> None:
    print(f"Target: {endpoint}")
    print(f"Mode: {'structured JSON (no LLM)' if args.structured else 'free text (AI Core + tools)'}")
    print(f"Sending {args.requests} requests, {args.users} at a time...\n")

    results: list = []
    health: list = []
    limits = httpx.Limits(max_connections=args.users + 5, max_keepalive_connections=args.users + 5)
    async with httpx.AsyncClient(timeout=args.timeout, limits=limits) as client:
        stop = asyncio.Event()
        watcher = asyncio.create_task(_watch_health(client, base, stop, health))
        gate = asyncio.Semaphore(args.users)

        async def worker(i: int) -> None:
            text, data = payloads[i % len(payloads)]
            async with gate:
                await _one(client, endpoint, _body(text, data), results)

        started = time.perf_counter()
        await asyncio.gather(*(worker(i) for i in range(args.requests)))
        wall = time.perf_counter() - started
        stop.set()
        await watcher

    outcomes = Counter(outcome for outcome, _ in results)
    latencies = [elapsed for outcome, elapsed in results if outcome in ("ok", "busy reply")]
    print("Outcomes:")
    for outcome, count in outcomes.most_common():
        print(f"  {outcome:<20} {count}")
    print(f"\nWall time: {wall:.1f}s   Throughput: {len(results) / wall:.1f} req/s")
    if latencies:
        print(
            f"Latency  p50 {statistics.median(latencies):.2f}s   p95 {_pct(latencies, 0.95):.2f}s   "
            f"max {max(latencies):.2f}s"
        )
    finite = [h for h in health if h != float("inf")]
    if health:
        print(
            f"/health during load: {len(health)} pings, max {max(finite) if finite else float('nan'):.3f}s, "
            f"failed {len(health) - len(finite)}  (should stay well under 1s)"
        )
    bad = sum(c for o, c in outcomes.items() if o not in ("ok", "busy reply"))
    print("\nRESULT:", "PASS - no errors" if bad == 0 else f"{bad} failed requests - check the app logs")


if __name__ == "__main__":
    asyncio.run(main())
