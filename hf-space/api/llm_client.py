"""LLM transport: OpenRouter (production) with optional local Ollama fallback."""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from typing import Optional

logger = logging.getLogger(__name__)

OPENROUTER_BASE = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL = (
    os.environ.get("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct").strip()
    or "meta-llama/llama-3.1-8b-instruct"
)


def _openrouter_key() -> str:
    return os.environ.get("OPENROUTER_API_KEY", "").strip()


def llm_available() -> bool:
    if _openrouter_key():
        return True
    return os.environ.get("CAMBRIDGE_USE_OLLAMA", "").strip().lower() in ("1", "true", "yes")


def _openrouter_chat(
    system: str,
    user: str,
    *,
    max_tokens: int,
    temperature: float,
) -> str:
    key = _openrouter_key()
    if not key:
        return ""
    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    req = urllib.request.Request(
        f"{OPENROUTER_BASE}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "HTTP-Referer": os.environ.get("OPENROUTER_REFERER", "http://localhost:5173"),
            "X-Title": os.environ.get("OPENROUTER_APP_TITLE", "EF-CamDAT L2 Profiler"),
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return (data["choices"][0]["message"]["content"] or "").strip()
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:300]
        logger.warning("OpenRouter HTTP %s: %s", exc.code, body)
        return ""
    except Exception as exc:
        logger.warning("OpenRouter chat failed: %s", exc)
        return ""


def llm_chat(
    system: str,
    user: str,
    *,
    max_tokens: int = 60,
    temperature: float = 0,
) -> str:
    """Chat completion — OpenRouter if keyed, else Ollama when CAMBRIDGE_USE_OLLAMA=1."""
    if _openrouter_key():
        return _openrouter_chat(system, user, max_tokens=max_tokens, temperature=temperature)

    if os.environ.get("CAMBRIDGE_USE_OLLAMA", "").strip().lower() in ("1", "true", "yes"):
        try:
            from .ollama_client import ollama_chat

            return ollama_chat(system, user, temperature=temperature, num_predict=max_tokens)
        except Exception as exc:
            logger.warning("Ollama chat failed: %s", exc)
            return ""

    return ""


def llm_complete(
    prompt: str,
    *,
    system: Optional[str] = None,
    max_tokens: int = 512,
    temperature: float = 0,
) -> str:
    """Single-shot completion (error analysis). Uses OpenRouter or Ollama when enabled."""
    sys_msg = system or "You are a helpful assistant. Reply with valid JSON only."
    return llm_chat(sys_msg, prompt, max_tokens=max_tokens, temperature=temperature)
