#!/usr/bin/env python3
"""
gemini.py

Thin wrapper around the Google Gemini chat API used by run_trials.py.
Requires the environment variable GEMINI_API_KEY to be set.

Install:
  pip install google-genai

Docs quick start example:
  from google import genai
  client = genai.Client()
  resp = client.models.generate_content(model="gemini-2.5-pro", contents="hi")
  print(resp.text)
"""

from typing import List, Optional
from dataclasses import dataclass
import time 

try:
    from google import genai
    from google.genai import types as genai_types
except Exception as e:  # pragma: no cover
    raise ImportError(
        "google-genai is not installed. Run: pip install google-genai"
    ) from e


class GeminiError(RuntimeError):
    """Raised when the Gemini API fails or returns an unexpected format."""


# Keep a visible default model similar to OLLAMA_HOST_DEFAULT
GEMINI_MODEL_DEFAULT = "gemini-2.5-pro"


@dataclass
class GeminiClientFactory:
    """Factory so callers can pass a client if they want to reuse connections."""
    client: Optional["genai.Client"] = None

    api_key = open("gemini.key").read().strip()

    def get(self) -> "genai.Client":
        if self.client is None:
            self.client = genai.Client(api_key=self.api_key)
        return self.client


_factory = GeminiClientFactory()


def call_gemini_chat(
    model: str = GEMINI_MODEL_DEFAULT,
    system_prompt: str = "",
    user_prompt: str = "",
    temperature: float = 0.1,
    top_p: float = 0.4,
    num_predict: int = 2048,      # maps to max_output_tokens
    seed: int = 42,               # used if supported by SDK
    stop: Optional[List[str]] = None,
    client: Optional["genai.Client"] = None,
) -> str:
    """
    Call Gemini's generate_content and return the assistant content as a string.

    Args match the Ollama wrapper for easy swapping. Unsupported params are ignored.
    """
    cli = client or _factory.get()

    config = genai_types.GenerateContentConfig(
        temperature=temperature,
        top_p=top_p,
        max_output_tokens=num_predict,
        stop_sequences=stop or [],
        system_instruction=system_prompt if system_prompt else None,
        seed=seed,
        thinking_config=genai_types.ThinkingConfig(
            include_thoughts=True,
        )
    )

    # We pass system_prompt via system_instruction so it does not pollute the user content
    try:
        response = cli.models.generate_content(
            model=model,
            contents=[
                # Keep user content separate from system_instruction
                {"role": "user", "parts": [{"text": user_prompt}]}
            ],
            config=config,
        )
        time.sleep(15)
    except Exception as e:
        raise GeminiError(f"Gemini request failed: {e}") from e

    # Extract text
    all_out = ""
    try:
        if response is None or not response.candidates or len(response.candidates) == 0 or not response.candidates[0].content:
            raise GeminiError(f"Empty response: {response!r}")
        for part in response.candidates[0].content.parts:
            if not part.text:
                continue
            if part.thought:
                all_out += f"<think>{part.text}</think>\n"
            else:
                all_out += part.text
        return all_out
    except Exception as e:
        raise GeminiError(f"Unexpected response format: {response!r}") from e
