"""Exercise a client against a NIM-shaped mock on an ephemeral loopback port."""

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread
from urllib.error import HTTPError
from urllib.request import Request, urlopen


class MockNim(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def reply(self, status, payload):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/v1/health/live":
            self.reply(200, {"status": "live"})
        elif self.path == "/v1/health/ready":
            ready = self.server.model_ready
            self.reply(200 if ready else 503, {"ready": ready})
        else:
            self.reply(404, {"error": "unknown path"})

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        if self.path != "/v1/chat/completions":
            self.reply(404, {"error": "unknown path"})
        elif not self.server.model_ready:
            self.reply(503, {"error": "mock loading"})
        else:
            self.reply(200, {"choices": [{"message": {"role": "assistant", "content": "mock result"}}]})


def call(base, path, payload=None):
    request = Request(base + path, data=json.dumps(payload).encode() if payload is not None else None,
                      headers={"Content-Type": "application/json"})
    try:
        with urlopen(request, timeout=5) as reply:
            return reply.status, json.load(reply)
    except HTTPError as error:
        with error:
            return error.code, json.load(error)


if __name__ == "__main__":
    with HTTPServer(("127.0.0.1", 0), MockNim) as server:
        server.model_ready = False
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            base = f"http://127.0.0.1:{server.server_port}"
            payload = {"model": "mock", "messages": [{"role": "user", "content": "status"}]}
            for label in ("loading", "loaded"):
                server.model_ready = label == "loaded"
                live, _ = call(base, "/v1/health/live")
                ready, _ = call(base, "/v1/health/ready")
                chat, reply = call(base, "/v1/chat/completions", payload)
                print(json.dumps({"mode": "mock", "state": label, "live": live, "ready": ready,
                                  "chat": chat, "response": reply}))
        finally:
            server.shutdown()
            thread.join()
