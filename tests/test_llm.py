import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from extract.llm import ExtractionError, make_llm, openai_compatible


def serve(responses):
    """Tiny fake chat-completions server. `responses` is consumed one per request."""
    seen = []

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append((self.path, self.headers.get("Authorization"), body))
            status, headers, payload = responses.pop(0)
            self.send_response(status)
            for k, v in headers.items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(json.dumps(payload).encode())

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_port}/v1", seen


OK = (200, {}, {"choices": [{"message": {"content": '{"decisions": []}'}}]})


def test_success_sends_key_and_json_mode():
    srv, url, seen = serve([OK])
    try:
        out = openai_compatible("m", url, "KEY")("sys", "usr")
    finally:
        srv.shutdown()
    assert out == '{"decisions": []}'
    path, auth, body = seen[0]
    assert path == "/v1/chat/completions" and auth == "Bearer KEY"
    assert body["response_format"] == {"type": "json_object"} and body["temperature"] == 0
    assert [m["role"] for m in body["messages"]] == ["system", "user"]


def test_rate_limit_waits_and_retries():
    srv, url, seen = serve([(429, {"Retry-After": "7"}, {"error": "slow down"}), OK])
    waits = []
    try:
        out = openai_compatible("m", url, "K", sleep=waits.append)("s", "u")
    finally:
        srv.shutdown()
    assert out and waits == [7.0] and len(seen) == 2


def test_response_format_rejected_falls_back():
    srv, url, seen = serve([(400, {}, {"error": "response_format unsupported"}), OK])
    try:
        openai_compatible("m", url, "K")("s", "u")
    finally:
        srv.shutdown()
    assert "response_format" in seen[0][2] and "response_format" not in seen[1][2]


def test_bad_key_gives_clear_error():
    srv, url, _ = serve([(401, {}, {"error": "invalid key"})])
    try:
        with pytest.raises(ExtractionError, match="API key"):
            openai_compatible("m", url, "bad")("s", "u")
    finally:
        srv.shutdown()


def test_make_llm_requires_key_and_base_url(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(ExtractionError, match="GROQ_API_KEY"):
        make_llm("groq", "m")
    with pytest.raises(ExtractionError, match="base-url"):
        make_llm("openai", "m")
    monkeypatch.setenv("GROQ_API_KEY", "x")
    assert callable(make_llm("groq", "m"))
