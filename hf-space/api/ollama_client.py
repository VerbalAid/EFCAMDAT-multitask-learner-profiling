"""Shared Ollama client (/api/chat and /api/generate)."""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "mistral"


def _model_name() -> str:
    return os.environ.get("CAMBRIDGE_OLLAMA_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL


def _base_url() -> str:
    return os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")


def ollama_chat(
    system: str,
    user: str,
    *,
    temperature: float = 0,
    num_predict: int = 60,
) -> str:
    """POST /api/chat — num_predict is passed in options (hard token cap)."""
    options = {"temperature": temperature, "num_predict": num_predict}
    payload = json.dumps(
        {
            "model": _model_name(),
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": options,
        }
    ).encode()
    logger.debug("Ollama chat model=%s num_predict=%s", _model_name(), num_predict)
    req = urllib.request.Request(
        f"{_base_url()}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read())
    message = data.get("message") or {}
    return (message.get("content") or "").strip()


def ollama_generate(
    prompt: str,
    *,
    system: str | None = None,
    temperature: float = 0,
    num_predict: int = 512,
) -> str:
    body: dict = {
        "model": _model_name(),
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": num_predict},
    }
    if system:
        body["system"] = system
    payload = json.dumps(body).encode()
    req = urllib.request.Request(
        f"{_base_url()}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read())
    return (data.get("response") or "").strip()
