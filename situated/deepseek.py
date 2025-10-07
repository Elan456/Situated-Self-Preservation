#!/usr/bin/env python3
"""
deepseek.py

Hardened wrapper around the DeepSeek chat API.
- Captures "thinking" for reasoner models and wraps it in <think>...</think>
- Retries on transient errors and empty responses
- Detects length cutoffs and auto-continues
"""

from typing import List, Optional, Tuple
import os
import time

from openai import OpenAI
from openai._exceptions import OpenAIError

DEEPSEEK_BASE_URL_DEFAULT = "https://api.deepseek.com"
DEEPSEEK_API_KEY_ENVVAR = "DEEPSEEK_API_KEY"
DEEPSEEK_API_KEY_FILE = "deepseek.key"

class DeepseekError(RuntimeError):
    """Raised when the DeepSeek API call fails or returns an unexpected response."""

def _load_api_key(explicit: Optional[str]) -> str:
    if explicit:
        return explicit
    # env first
    key = os.environ.get(DEEPSEEK_API_KEY_ENVVAR)
    if key:
        return key.strip()
    # local file fallback
    if os.path.exists(DEEPSEEK_API_KEY_FILE):
        with open(DEEPSEEK_API_KEY_FILE, "r", encoding="utf-8") as f:
            k = f.read().strip()
            if k:
                return k
    raise DeepseekError(
        f"Missing API key. Set {DEEPSEEK_API_KEY_ENVVAR} or create {DEEPSEEK_API_KEY_FILE} with the key."
    )

def _extract_text_and_thinking_from_choice(choice) -> Tuple[str, str, str]:
    """
    Returns (content, thinking, finish_reason) where "thinking" is the reasoner trace if available.
    Works for both deepseek-chat and deepseek-reasoner.
    """
    finish_reason = getattr(choice, "finish_reason", None)

    # OpenAI-style message object
    msg = getattr(choice, "message", None)
    content = ""
    thinking = ""

    if msg is not None:
        # Primary assistant content
        content = getattr(msg, "content", "") or ""

        # DeepSeek reasoner adds "reasoning_content" on message
        # Some SDK versions expose as attribute, others in dict form
        rc = getattr(msg, "reasoning_content", None)
        if rc is None and hasattr(msg, "model_extra"):
            # model_extra is a dict of vendor fields
            rc = getattr(msg, "model_extra", {}).get("reasoning_content")
        if rc is None and isinstance(msg, dict):
            rc = msg.get("reasoning_content")
        thinking = rc or ""

    # Fallback: some responses place vendor extras under "delta" in streaming, handled elsewhere
    return content, thinking, finish_reason or ""

def _concat_with_thinking(thinking: str, content: str) -> str:
    if thinking:
        return f"<think>{thinking}</think>{content}"
    return content

def _continue_messages(system_prompt: str, user_prompt: str, partial: str) -> list:
    # Standard continuation pattern
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": partial},
        {"role": "user", "content": "Continue."},
    ]

def call_deepseek_chat(
    api_key: Optional[str] = None,
    base_url: str = DEEPSEEK_BASE_URL_DEFAULT,
    model: str = "deepseek-reasoner",  # "deepseek-chat" for non-reasoning
    system_prompt: str = "",
    user_prompt: str = "",
    temperature: float = 0.1,
    top_p: float = 0.4,
    num_predict: int = 4096,          # maps to max_tokens
    repeat_penalty: Optional[float] = None,  # no direct equivalent
    seed: Optional[int] = 42,         # supported
    num_ctx: Optional[int] = None,    # not applicable here
    stop: Optional[List[str]] = None,
    timeout_seconds: int = 600,
    stream: bool = False,
    max_retries: int = 3,
    max_auto_continues: int = 2,
    min_content_len_for_success: int = 3,
) -> str:
    """
    Returns assistant text. If the model provides reasoning, it will be wrapped:
    <think>...</think><answer text>

    Auto-retry and auto-continue are enabled to reduce empty outputs.
    """
    key = _load_api_key(api_key)
    client = OpenAI(api_key=key, base_url=base_url, timeout=timeout_seconds)

    base_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    def non_stream_call(messages: list, max_tokens: int) -> Tuple[str, str, str]:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=stop,
            seed=seed,
            stream=False,
        )
        if not getattr(resp, "choices", None):
            return "", "", getattr(resp, "finish_reason", "") or ""
        content, thinking, finish = _extract_text_and_thinking_from_choice(resp.choices[0])
        return content or "", thinking or "", finish or ""

    def stream_call(messages: list, max_tokens: int) -> Tuple[str, str, str]:
        content_chunks: List[str] = []
        think_chunks: List[str] = []
        finish_reason = ""
        with client.chat.completions.stream(
            model=model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=stop,
            seed=seed,
        ) as s:
            for event in s:
                etype = getattr(event, "type", "")
                # DeepSeek streams two kinds of deltas:
                # - "content.delta" for normal assistant text
                # - "reasoning.delta" for the internal trace on reasoner models
                if etype == "content.delta":
                    delta = getattr(event, "delta", None)
                    if delta:
                        content_chunks.append(delta)
                elif etype == "reasoning.delta":
                    delta = getattr(event, "delta", None)
                    if delta:
                        think_chunks.append(delta)
                elif etype == "message.completed":
                    finish_reason = getattr(event, "response", {}).get("choices", [{}])[0].get("finish_reason", "") \
                        if hasattr(event, "response") else ""
                elif etype == "error":
                    raise DeepseekError(f"Streaming error: {getattr(event, 'error', 'unknown')}")
        return "".join(content_chunks), "".join(think_chunks), finish_reason

    def one_attempt(messages: list, max_tokens: int) -> Tuple[str, str, str]:
        if stream:
            return stream_call(messages, max_tokens)
        return non_stream_call(messages, max_tokens)

    # Retry loop for transient failures and empty results
    attempt = 0
    last_err = None
    messages = list(base_messages)
    max_tokens = int(max(64, num_predict))

    while attempt < max_retries:
        try:
            content, thinking, finish = one_attempt(messages, max_tokens)

            # If nothing came back, try again with a small nudge
            if len((content or "").strip()) < min_content_len_for_success and len((thinking or "").strip()) < 1:
                # Try one light continuation prompt before a full retry
                messages = _continue_messages(system_prompt, user_prompt, "")
                attempt += 1
                time.sleep(min(0.5 * attempt, 2.0))
                continue

            # If cut off by length, auto-continue up to max_auto_continues
            auto_cont = 0
            partial_answer = _concat_with_thinking(thinking, content)
            while (finish == "length" or finish == "content_filter") and auto_cont < max_auto_continues:
                cont_messages = _continue_messages(system_prompt, user_prompt, partial_answer)
                c2, t2, f2 = one_attempt(cont_messages, max_tokens)
                # Merge
                partial_answer = _concat_with_thinking(
                    (thinking or "") + (t2 or ""),
                    (content or "") + (c2 or "")
                )
                finish = f2 or finish
                auto_cont += 1

            # Final success check
            if len(partial_answer.strip()) >= min_content_len_for_success:
                return partial_answer

            # Otherwise, try another top-level attempt
            attempt += 1
            time.sleep(min(0.5 * attempt, 2.0))
            continue

        except OpenAIError as e:
            last_err = e
            attempt += 1
            time.sleep(min(0.5 * attempt, 2.0))
        except Exception as e:
            last_err = e
            attempt += 1
            time.sleep(min(0.5 * attempt, 2.0))

    raise DeepseekError(f"DeepSeek failed after {max_retries} attempts: {last_err}")

if __name__ == "__main__":
    # Tiny sanity check
    out = call_deepseek_chat(
        system_prompt="You are a helpful assistant.",
        user_prompt="In one short sentence, say hello and tell me one fun fact.",
        model="deepseek-reasoner",  # or "deepseek-chat"
        stream=False,
    )
    print(out)
