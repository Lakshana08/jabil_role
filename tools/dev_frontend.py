"""Serve frontend/public locally and pass /a2a/, /auth/, /reports/ through to a
backend - the same thing frontend/nginx.conf does on Cloud Foundry. Lets you
test page changes against the deployed backend without running it locally.

Run from the repo root (standard library only, no venv needed):
  python tools/dev_frontend.py                       # deployed backend, port 8080
  python tools/dev_frontend.py http://localhost:8000 # local backend
"""

import http.server
import sys
import urllib.error
import urllib.request
from functools import partial

BACKEND_URL = (sys.argv[1] if len(sys.argv) > 1 else "https://smart-material-ai-role.cfapps.us10-001.hana.ondemand.com").rstrip("/")
PORT = 8080
PROXIED = ("/a2a/", "/auth/", "/reports/")
# Hop-by-hop or recomputed headers - not copied between the two connections.
SKIP_HEADERS = {"host", "connection", "keep-alive", "transfer-encoding", "content-length", "accept-encoding"}


class Handler(http.server.SimpleHTTPRequestHandler):
    def _proxy(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else None
        headers = {k: v for k, v in self.headers.items() if k.lower() not in SKIP_HEADERS}
        req = urllib.request.Request(BACKEND_URL + self.path, data=body, headers=headers, method=self.command)
        try:
            resp = urllib.request.urlopen(req, timeout=300)
        except urllib.error.HTTPError as err:
            resp = err  # 4xx/5xx still carry a body the page wants to show
        except urllib.error.URLError as err:
            self.send_error(502, f"Backend not reachable: {err.reason}")
            return
        data = resp.read()
        self.send_response(resp.status)
        for key, value in resp.headers.items():
            if key.lower() not in SKIP_HEADERS:
                self.send_header(key, value)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        if self.path.startswith(PROXIED):
            self._proxy()
        else:
            super().do_GET()

    def do_POST(self) -> None:
        if self.path.startswith(PROXIED):
            self._proxy()
        else:
            self.send_error(405)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    # Pass redirects (e.g. /auth/login -> IAS) back to the browser instead of following them here.
    def redirect_request(self, *args, **kwargs):
        return None


urllib.request.install_opener(urllib.request.build_opener(_NoRedirect))

if __name__ == "__main__":
    print(f"Page:    http://localhost:{PORT}\nBackend: {BACKEND_URL}")
    server = http.server.ThreadingHTTPServer(("", PORT), partial(Handler, directory="frontend/public"))
    server.serve_forever()
