"""Multi-provider stateless LLM client (OpenAI-compatible, Anthropic).

This is the single model-calling path used by all six tests.
"""
from __future__ import annotations

import logging
import random
import time
from typing import Any

from . import config

logger = logging.getLogger(__name__)


class ApiError(RuntimeError):
    """Terminal API failure after retries."""

    def __init__(self, provider: str, model: str, status: str | int | None, message: str) -> None:
        self.provider = provider
        self.model = model
        self.status = status
        super().__init__(f"[{provider}] model={model} status={status}: {message}")


def call_llm(
    prompt: str,
    model: str,
    provider: str = "openai",
    *,
    temperature: float = 0.0,
    max_tokens: int | None = 256,
    timeout: int = 60,
    max_retries: int = 3,
    system_prompt: str | None = None,
    extra_body: dict | None = None,
) -> dict:
    """Stateless completion-style call (single user message).

    Returns ``{"text", "raw", "latency_s", "tokens_used", "prompt_tokens",
    "completion_tokens", "model_name", "finish_reason"}``.
    Raises :class:`ApiError` on terminal failure.
    """
    provider = provider.strip().lower()
    t0 = time.perf_counter()
    if provider == "openai":
        return _call_openai_compatible(
            prompt, model, temperature, max_tokens, timeout, max_retries, extra_body, t0
        )
    if provider == "anthropic":
        return _call_anthropic(
            prompt, model, temperature, max_tokens or 1024, timeout, max_retries, system_prompt, t0
        )
    raise ApiError(provider, model, "BAD_PROVIDER", f"Unknown provider: {provider}")


def _call_openai_compatible(
    prompt: str,
    model: str,
    temperature: float,
    max_tokens: int | None,
    timeout: int,
    max_retries: int,
    extra_body: dict | None,
    t0: float,
) -> dict:
    api_key = config.api_key()
    if not api_key:
        raise ApiError(
            "openai", model, "MISSING_KEY",
            "Set OPENAI_API_KEY or DASHSCOPE_API_KEY before calling the API.",
        )

    delay = 1.0
    last_msg = ""
    for attempt in range(max_retries + 1):
        try:
            from openai import OpenAI

            client = OpenAI(api_key=api_key, base_url=config.base_url())
            kwargs: dict[str, Any] = {
                "model": model,
                # Single user message only (no system role), per experiment protocol.
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "timeout": timeout,
                "n": 1,
            }
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens
            if extra_body:
                kwargs["extra_body"] = extra_body
            resp = client.chat.completions.create(**kwargs)
            choice = resp.choices[0]
            usage = getattr(resp, "usage", None)
            latency_s = time.perf_counter() - t0
            return {
                "text": (choice.message.content or "").strip(),
                "raw": resp,
                "latency_s": latency_s,
                "tokens_used": getattr(usage, "total_tokens", None) if usage else None,
                "prompt_tokens": getattr(usage, "prompt_tokens", None) if usage else None,
                "completion_tokens": getattr(usage, "completion_tokens", None) if usage else None,
                "model_name": getattr(resp, "model", model),
                "finish_reason": getattr(choice, "finish_reason", None),
            }
        except Exception as exc:  # noqa: BLE001 - retried, re-raised as ApiError below
            last_msg = str(exc)
            logger.warning("OpenAI-compatible attempt %s failed: %s", attempt + 1, exc)
            if attempt >= max_retries:
                break
            time.sleep(delay + random.uniform(0, 0.5))
            delay = min(delay * 2, 60.0)

    raise ApiError("openai", model, "FAILED", last_msg or "unknown error")


def _call_anthropic(
    prompt: str,
    model: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
    max_retries: int,
    system_prompt: str | None,
    t0: float,
) -> dict:
    import os

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise ApiError("anthropic", model, "MISSING_KEY", "Set ANTHROPIC_API_KEY first.")

    delay = 1.0
    last_msg = ""
    for attempt in range(max_retries + 1):
        try:
            import anthropic

            client = anthropic.Anthropic()
            kwargs: dict[str, Any] = {
                "model": model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "timeout": timeout,
                "messages": [{"role": "user", "content": prompt}],
            }
            if system_prompt:
                kwargs["system"] = system_prompt
            msg = client.messages.create(**kwargs)
            text = "".join(b.text for b in msg.content if hasattr(b, "text")).strip()
            usage = getattr(msg, "usage", None)
            in_tok = getattr(usage, "input_tokens", None) if usage else None
            out_tok = getattr(usage, "output_tokens", None) if usage else None
            total = (in_tok or 0) + (out_tok or 0) if (in_tok is not None or out_tok is not None) else None
            latency_s = time.perf_counter() - t0
            return {
                "text": text,
                "raw": msg,
                "latency_s": latency_s,
                "tokens_used": total,
                "prompt_tokens": in_tok,
                "completion_tokens": out_tok,
                "model_name": getattr(msg, "model", model),
                "finish_reason": getattr(msg, "stop_reason", None),
            }
        except Exception as exc:  # noqa: BLE001
            last_msg = str(exc)
            logger.warning("Anthropic attempt %s failed: %s", attempt + 1, exc)
            if attempt >= max_retries:
                break
            time.sleep(delay + random.uniform(0, 0.5))
            delay = min(delay * 2, 60.0)

    raise ApiError("anthropic", model, "FAILED", last_msg or "unknown error")
