import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from extract.llm import ExtractionError, list_models, make_llm, openai_compatible


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


def test_user_agent_is_sent_and_models_are_listed(monkeypatch):
    seen = []

    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            seen.append((self.path, self.headers.get("Authorization"), self.headers.get("User-Agent")))
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps({"data": [{"id": "b-model"}, {"id": "a-model"}]}).encode())

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setenv("LLM_API_KEY", "K")
    try:
        names = list_models("openai", f"http://127.0.0.1:{srv.server_port}/v1")
    finally:
        srv.shutdown()
    assert names == ["a-model", "b-model"]
    assert seen[0][0] == "/v1/models" and seen[0][1] == "Bearer K" and "meeting-memory" in seen[0][2]


def test_list_models_needs_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(ExtractionError, match="OPENROUTER_API_KEY"):
        list_models("openrouter")


def test_json_validate_failed_falls_back_to_plain_mode():
    err = (400, {}, {"error": {"code": "json_validate_failed", "failed_generation": ""}})
    srv, url, seen = serve([err, OK])
    try:
        out = openai_compatible("m", url, "K")("s", "u")
    finally:
        srv.shutdown()
    assert out and "response_format" in seen[0][2] and "response_format" not in seen[1][2]


def test_retry_is_logged():
    srv, url, _ = serve([(429, {"Retry-After": "3"}, {}), OK])
    logs = []
    try:
        openai_compatible("m", url, "K", sleep=lambda s: None, log=logs.append)("s", "u")
    finally:
        srv.shutdown()
    assert any("rate limited" in m and "3s" in m for m in logs)


def test_max_tokens_and_reasoning_effort_are_sent():
    srv, url, seen = serve([OK])
    try:
        openai_compatible("m", url, "K", max_tokens=1234, reasoning_effort="low")("s", "u")
    finally:
        srv.shutdown()
    body = seen[0][2]
    assert body["max_tokens"] == 1234 and body["reasoning_effort"] == "low"


def test_reasoning_effort_rejected_is_dropped():
    srv, url, seen = serve([(400, {}, {"error": "unknown parameter: reasoning_effort"}), OK])
    try:
        openai_compatible("m", url, "K", reasoning_effort="low")("s", "u")
    finally:
        srv.shutdown()
    assert "reasoning_effort" in seen[0][2] and "reasoning_effort" not in seen[1][2]


def test_empty_reply_is_diagnosed():
    empty = (200, {}, {"choices": [{"message": {"content": ""}, "finish_reason": "length"}],
                       "usage": {"completion_tokens": 6000}})
    srv, url, _ = serve([empty])
    logs = []
    try:
        out = openai_compatible("m", url, "K", log=logs.append)("s", "u")
    finally:
        srv.shutdown()
    assert out == "" and any("finish_reason=length" in m and "6000" in m for m in logs)


def test_413_gives_actionable_hint():
    srv, url, _ = serve([(413, {}, {"error": "Request too large ... TPM"})])
    try:
        with pytest.raises(ExtractionError, match="max-tokens"):
            openai_compatible("m", url, "K")("s", "u")
    finally:
        srv.shutdown()


def test_daily_token_limit_stops_immediately():
    msg = {"error": {"message": "Rate limit reached ... on tokens per day (TPD): Limit 200000, Used 199464"}}
    srv, url, seen = serve([(429, {}, msg)])
    waits = []
    try:
        with pytest.raises(ExtractionError, match="Daily token limit"):
            openai_compatible("m", url, "K", sleep=waits.append)("s", "u")
    finally:
        srv.shutdown()
    assert waits == [] and len(seen) == 1
