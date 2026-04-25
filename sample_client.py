
"""
Ollama LLM client using urllib3 for HTTP, with Traceloop + OpenTelemetry instrumentation.
"""
import os
import json
import time
from typing import Optional

import urllib3
from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import task, workflow
from opentelemetry.instrumentation.urllib3 import URLLib3Instrumentor


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OTEL_ENDPOINT = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")

# ---------------------------------------------------------------------------
# Traceloop / OpenTelemetry setup
# ---------------------------------------------------------------------------
Traceloop.init(
    app_name="llm-test-client-p",
    api_endpoint=OTEL_ENDPOINT,
    disable_batch=True,
)

URLLib3Instrumentor().instrument()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "qwen2.5:latest"

# ---------------------------------------------------------------------------
# Low-level urllib3 helpers  (no requests / httpx)
# ---------------------------------------------------------------------------
_http = urllib3.PoolManager()


def _check_status(resp: urllib3.HTTPResponse, url: str) -> None:
    if resp.status >= 400:
        raise RuntimeError(
            f"Ollama HTTP {resp.status} for {url}: {resp.data.decode()[:200]}"
        )


def _get(path: str) -> dict:
    url = f"{OLLAMA_BASE_URL}{path}"
    resp = _http.request("GET", url)
    _check_status(resp, url)
    return json.loads(resp.data.decode())


def _post(path: str, body: dict) -> dict:
    url = f"{OLLAMA_BASE_URL}{path}"
    encoded = json.dumps(body).encode()
    resp = _http.request(
        "POST",
        url,
        body=encoded,
        headers={"Content-Type": "application/json"},
    )
    _check_status(resp, url)
    return json.loads(resp.data.decode())


# ---------------------------------------------------------------------------
# Ollama API wrappers
# ---------------------------------------------------------------------------

@task(name="ollama.list_models")
def list_models() -> list[dict]:
    """Return all models registered in Ollama."""
    data = _get("/api/tags")
    return data.get("models", [])


@task(name="ollama.chat")
def chat(
    prompt: str,
    model: str = DEFAULT_MODEL,
    system_prompt: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
) -> str:
    """Send a chat request to Ollama and return the assistant reply."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    options: dict = {"temperature": temperature}
    if max_tokens is not None:
        options["num_predict"] = max_tokens

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": options,
    }

    data = _post("/api/chat", payload)
    return data["message"]["content"]


# ---------------------------------------------------------------------------
# Top-level workflow (shows up as a single root span in Traceloop)
# ---------------------------------------------------------------------------

@workflow(name="ollama_demo")
def run_demo() -> None:
    print("=== Available Ollama models ===")
    models = list_models()
    for m in models:
        size_gb = m.get("size", 0) / 1_073_741_824
        print(f"  • {m['name']:<35}  ({size_gb:.1f} GB)")

    print(f"\n=== Chat with {DEFAULT_MODEL} ===")
    reply = chat(
        prompt="Explain the difference between a process and a thread in one paragraph.",
        model=DEFAULT_MODEL,
        system_prompt="You are a helpful and concise software-engineering tutor.",
        temperature=0.5,
    )
    print(f"\nAssistant:\n{reply}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    for _ in range(20):  # Run the demo multiple times to generate more spans
        run_demo()
        time.sleep(1)
