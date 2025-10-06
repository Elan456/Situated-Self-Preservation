#!/usr/bin/env python3
"""
ollama.py

Thin wrapper around the Ollama chat API used by run_trials.py.
Kept minimal on purpose so it can be swapped or extended later.
"""

from typing import List, Optional
import requests

# Public so callers can reuse the same default
OLLAMA_HOST_DEFAULT = "http://localhost:11434"


class OllamaError(RuntimeError):
    """Raised when the Ollama API returns a non-200 or malformed response."""


def call_ollama_chat(
    host: str = OLLAMA_HOST_DEFAULT,
    model: str = "deepseek-r1:14b",
    system_prompt: str = "",
    user_prompt: str = "",
    temperature: float = 0.1,
    top_p: float = 0.4,
    num_predict: int = 2048,
    repeat_penalty: float = 1.2,
    seed: int = 42,
    num_ctx: int = 8192,
    stop: Optional[List[str]] = None,
    timeout_seconds: int = 600,
) -> str:
    """
    Call Ollama's chat endpoint and return the assistant content as a string.

    This mirrors the previous inline behavior in run_trials.py so callers do not need to change.
    """
    url = f"{host.rstrip('/')}/api/chat"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "options": {
            "temperature": temperature,
            "top_p": top_p,
            "num_predict": num_predict,
            "repeat_penalty": repeat_penalty,
            "repeat_last_n": 256,
            "seed": seed,
            "num_ctx": num_ctx,
        },
    }
    if stop:
        payload["stop"] = stop

    try:
        resp = requests.post(url, json=payload, timeout=timeout_seconds)
    except requests.RequestException as e:
        raise OllamaError(f"Failed to reach Ollama at {url}: {e}") from e

    if resp.status_code != 200:
        text = resp.text[:1000]
        raise OllamaError(f"Ollama HTTP {resp.status_code}: {text}")

    try:
        data = resp.json()
        return data["message"]["content"]
    except Exception as e:
        raise OllamaError(f"Unexpected response format: {resp.text[:500]}") from e
