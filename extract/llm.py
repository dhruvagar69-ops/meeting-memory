"""Model access (standard library only): local Ollama, or any OpenAI-compatible
chat API such as Groq, OpenRouter or LM Studio."""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Callable

LLM = Callable[[str, str], str]  # (system, user) -> reply text


class ExtractionError(RuntimeError):
    pass


def ollama(model: str, host: str = "http://localhost:11434",
           num_ctx: int = 8192, timeout: int = 900) -> LLM:
    def call(system: str, user: str) -> str:
        payload = {
            "model": model, "stream": False, "format": "json",
            "options": {"temperature": 0, "num_ctx": num_ctx},
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
        }
        req = urllib.request.Request(host + "/api/chat", data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read())["message"]["content"]
        except urllib.error.URLError as exc:
            raise ExtractionError(
                f"Could not reach Ollama at {host} ({exc}). Is it running (`ollama serve`) "
                f"and is the model pulled (`ollama pull {model}`)?") from exc
    return call


PROVIDERS = {  # backend name -> (base URL, environment variable holding the API key)
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
}


def openai_compatible(model: str, base_url: str, api_key: str, *, json_mode: bool = True,
                      timeout: int = 300, max_retries: int = 3, sleep=time.sleep) -> LLM:
    """Chat-completions client. Waits and retries on HTTP 429 (rate limits), and
    drops response_format if the provider rejects it."""
    url = base_url.rstrip("/") + "/chat/completions"

    def call(system: str, user: str) -> str:
        use_json = json_mode
        for attempt in range(max_retries + 1):
            body = {"model": model, "temperature": 0,
                    "messages": [{"role": "system", "content": system},
                                 {"role": "user", "content": user}]}
            if use_json:
                body["response_format"] = {"type": "json_object"}
            req = urllib.request.Request(
                url, data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}",
                         "User-Agent": "meeting-memory/0.1"})
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    return json.loads(resp.read())["choices"][0]["message"]["content"]
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", "replace")[:300]
                if exc.code == 429 and attempt < max_retries:
                    try:
                        wait = min(float(exc.headers.get("Retry-After") or 20), 60.0)
                    except ValueError:
                        wait = 20.0
                    sleep(wait)
                    continue
                if exc.code == 400 and use_json and "response_format" in detail:
                    use_json = False
                    continue
                hint = " (check the API key)" if exc.code in (401, 403) else ""
                raise ExtractionError(f"{url} returned HTTP {exc.code}{hint}: {detail}") from exc
            except urllib.error.URLError as exc:
                raise ExtractionError(f"Could not reach {url} ({exc})") from exc
            except (KeyError, IndexError, ValueError) as exc:
                raise ExtractionError(f"Unexpected reply format from {url}") from exc
        raise ExtractionError(f"{url} kept rate-limiting; wait a minute or use a smaller model")
    return call


def _endpoint(backend: str, base_url: str | None) -> tuple[str, str]:
    """(base URL, API key) for an OpenAI-compatible backend."""
    if backend in PROVIDERS:
        url, env = PROVIDERS[backend]
        key = os.environ.get(env)
        if not key:
            raise ExtractionError(f"Set the {env} environment variable to your {backend} API key.")
        return url, key
    if backend == "openai":  # any other OpenAI-compatible server, e.g. LM Studio
        if not base_url:
            raise ExtractionError("--backend openai needs --base-url (LM Studio: http://localhost:1234/v1)")
        return base_url, os.environ.get("LLM_API_KEY", "not-needed")
    raise ExtractionError(f"Unknown backend {backend!r}")


def make_llm(backend: str, model: str, host: str = "http://localhost:11434",
             base_url: str | None = None) -> LLM:
    if backend == "ollama":
        return ollama(model, host)
    url, key = _endpoint(backend, base_url)
    return openai_compatible(model, url, key)


def list_models(backend: str, base_url: str | None = None, timeout: int = 30) -> list[str]:
    """Model ids your key can use right now (GET <base>/models)."""
    url, key = _endpoint(backend, base_url)
    req = urllib.request.Request(url.rstrip("/") + "/models",
                                 headers={"Authorization": f"Bearer {key}", "User-Agent": "meeting-memory/0.1"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return sorted(m["id"] for m in json.loads(resp.read())["data"])
    except urllib.error.HTTPError as exc:
        hint = " (check the API key)" if exc.code in (401, 403) else ""
        raise ExtractionError(f"Model list request failed: HTTP {exc.code}{hint}") from exc
    except (urllib.error.URLError, KeyError, ValueError) as exc:
        raise ExtractionError(f"Could not list models from {url} ({exc})") from exc
