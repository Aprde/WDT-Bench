"""Runtime configuration shared by all model-querying code.

No API key is stored in the repository.  Set credentials through environment
variables before running any script that calls a model API:

    export OPENAI_API_KEY="your_key"
    export OPENAI_BASE_URL="https://your-api-endpoint/v1"

DashScope-compatible variables (``DASHSCOPE_API_KEY`` / ``DASHSCOPE_BASE_URL``)
are accepted as fallbacks; Anthropic models additionally require
``ANTHROPIC_API_KEY``.
"""
from __future__ import annotations

import os

_DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
_DEFAULT_MODEL = "qwen-plus"


def api_key() -> str:
    return (os.environ.get("OPENAI_API_KEY") or os.environ.get("DASHSCOPE_API_KEY") or "").strip()


def base_url() -> str:
    return (
        os.environ.get("OPENAI_BASE_URL")
        or os.environ.get("DASHSCOPE_BASE_URL")
        or _DEFAULT_BASE_URL
    ).strip()


def default_chat_model() -> str:
    """Read at call time so wrapper scripts and CLI flags can override it."""
    return (os.environ.get("DEFAULT_CHAT_MODEL") or "").strip() or _DEFAULT_MODEL


def default_provider() -> str:
    """``openai`` (any OpenAI-compatible endpoint) or ``anthropic``."""
    return (os.environ.get("DEFAULT_LLM_PROVIDER") or "openai").strip()
