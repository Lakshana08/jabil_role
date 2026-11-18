"""IAS security check: proves that strangers are blocked and an IAS token gets through.

Gets a REAL token from the IAS tenant (client-credentials flow, exactly like the Joule
destination), then calls an agent with: no token, a forged token, a tampered token and the real
token. By default it starts a LOCAL copy of the app with the IAS check switched ON, so production
is not affected. The client secret is asked for without echoing (or read from IAS_TEST_SECRET),
so it never appears on screen or in shell history.

  # local test (safe - production untouched)
  .venv\\Scripts\\python.exe tools\\ias_check.py

  # test the deployed app (only after IAS_AUTH_ENABLED=true there)
  .venv\\Scripts\\python.exe tools\\ias_check.py --url https://smart-material-ai.cfapps.us10-001.hana.ondemand.com
"""

import argparse
import getpass
import os
import subprocess
import sys
import time
import uuid

import httpx
import jwt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCAL_PORT = 8020
BODY = {"jsonrpc": "2.0", "id": "1", "method": "message/send", "params": {"message": {
    "role": "user", "messageId": "", "kind": "message",
    # Query agent with no product given: answers "I need a 'product'" without S/4 or AI Core
    "parts": [{"kind": "data", "data": {"query_type": "material_master"}}]}}}


def start_local_app(tenant: str, client_id: str, required_api: str) -> subprocess.Popen:
    env = dict(os.environ, IAS_AUTH_ENABLED="true", IAS_ISSUER=tenant, IAS_AUDIENCE=client_id,
               IAS_REQUIRED_API=required_api, AI_CORE_ENABLED="false")
    proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1",
                             "--port", str(LOCAL_PORT)], cwd=ROOT, env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(60):
        try:
            if httpx.get(f"http://127.0.0.1:{LOCAL_PORT}/health", timeout=1).status_code == 200:
                return proc
        except httpx.HTTPError:
            time.sleep(1)
    proc.terminate()
    raise SystemExit("The local app did not start - is port 8020 free?")


def call(base: str, label: str, token: str | None, expect: int) -> bool:
    BODY["params"]["message"]["messageId"] = str(uuid.uuid4())
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    r = httpx.post(f"{base}/a2a/query/", json=BODY, headers=headers, timeout=60)
    detail = r.json().get("detail") if r.status_code != 200 else "agent answered"
    ok = r.status_code == expect
    print(f"  {'PASS' if ok else 'FAIL'}  {label:46} HTTP {r.status_code}  {detail}")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tenant", default="anjgnszvi.accounts.ondemand.com")
    parser.add_argument("--client-id", default="71d3f7f2-9dc5-4b1f-ad9c-3644a366f34a")
    parser.add_argument("--required-api", default="api_read_access", help="'' to skip the permission check")
    parser.add_argument("--url", help="test this deployed app instead of a local copy")
    args = parser.parse_args()

    secret = os.environ.get("IAS_TEST_SECRET") or getpass.getpass("IAS client secret (hidden): ")
    tenant = args.tenant.removeprefix("https://").rstrip("/")

    print("\n1) Getting a token from IAS (like the Joule destination does)...")
    r = httpx.post(f"https://{tenant}/oauth2/token", data={"grant_type": "client_credentials", "client_id": args.client_id},
                   auth=(args.client_id, secret), timeout=30)
    if r.status_code != 200:
        raise SystemExit(f"   IAS refused the Client ID / secret: HTTP {r.status_code} {r.text[:200]}")
    token = r.json()["access_token"]
    claims = jwt.decode(token, options={"verify_signature": False})
    print(f"   OK - issuer={claims.get('iss')}  audience={claims.get('aud')}  permissions={claims.get('ias_apis')}")
    if args.required_api and args.required_api not in (claims.get("ias_apis") or []):
        print(f"   NOTE: the token has no '{args.required_api}' - the real-token test below will be rejected "
              "until the IAS secret is given that permission.")

    forged = jwt.encode({"iss": tenant, "aud": args.client_id, "exp": int(time.time()) + 600,
                         "ias_apis": [args.required_api]}, "not-the-ias-key-but-long-enough-for-hs256!", algorithm="HS256")
    tampered = token[:-6] + ("AAAAAA" if not token.endswith("AAAAAA") else "BBBBBB")
    real_expected = 200 if not args.required_api or args.required_api in (claims.get("ias_apis") or []) else 401

    proc = None
    if args.url:
        base = args.url.rstrip("/")
        print(f"\n2) Testing the deployed app {base}")
    else:
        print("\n2) Starting a LOCAL copy of the app with the IAS check ON (production untouched)...")
        proc = start_local_app(tenant, args.client_id, args.required_api)
        base = f"http://127.0.0.1:{LOCAL_PORT}"
    try:
        card = httpx.get(f"{base}/a2a/query/.well-known/agent-card.json", timeout=10).status_code
        results = [card == 200]
        print(f"  {'PASS' if card == 200 else 'FAIL'}  {'agent card (public, for discovery)':46} HTTP {card}")
        results.append(call(base, "stranger: no token", None, 401))
        results.append(call(base, "stranger: forged token", forged, 401))
        results.append(call(base, "stranger: real token, tampered", tampered, 401))
        results.append(call(base, "Joule-like: real IAS token", token, real_expected))
    finally:
        if proc:
            proc.terminate()
            proc.wait(timeout=15)
    print("\nRESULT:", "ALL CHECKS PASSED" if all(results) else "SOME CHECKS FAILED - see above")


if __name__ == "__main__":
    main()
