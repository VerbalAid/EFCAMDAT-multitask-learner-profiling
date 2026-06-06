"""Small local instruct LM for explanation text (no API keys; optional vs classification RoBERTa)."""

from __future__ import annotations

import logging
import os
from typing import Optional

import torch

logger = logging.getLogger(__name__)

_MODEL = None
_TOKENIZER = None

# Apache-2.0, ~0.5B params — enough for short structured explanations from JSON + SHAP lists.
DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def narrator_device() -> torch.device:
    d = os.environ.get("CAMBRIDGE_NARRATOR_DEVICE", "").strip().lower()
    if d == "cpu":
        return torch.device("cpu")
    if d == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_narrator() -> None:
    global _MODEL, _TOKENIZER
    if _MODEL is not None:
        return
    from transformers import AutoModelForCausalLM, AutoTokenizer

    mid = os.environ.get("CAMBRIDGE_NARRATOR_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
    dev = narrator_device()
    logger.info("Loading narrator model %s on %s", mid, dev)

    _TOKENIZER = AutoTokenizer.from_pretrained(mid, trust_remote_code=True)
    if getattr(_TOKENIZER, "pad_token", None) is None and getattr(_TOKENIZER, "eos_token", None):
        _TOKENIZER.pad_token = _TOKENIZER.eos_token

    dtype = torch.float16 if dev.type == "cuda" else torch.float32
    _MODEL = AutoModelForCausalLM.from_pretrained(
        mid,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    _MODEL.eval()
    _MODEL.to(dev)


def hf_generate_chat(system: str, user: str, max_new_tokens: int = 384) -> Optional[str]:
    """
    Run a tiny instruct model. Returns None on failure (caller can omit narrative).
    """
    try:
        _load_narrator()
    except Exception as e:
        logger.exception("Failed to load narrator: %s", e)
        return None

    assert _TOKENIZER is not None and _MODEL is not None
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    if getattr(_TOKENIZER, "chat_template", None):
        prompt = _TOKENIZER.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        prompt = f"System:\n{system}\n\nUser:\n{user}\n\nAssistant:\n"

    dev = next(_MODEL.parameters()).device
    inputs = _TOKENIZER(prompt, return_tensors="pt", truncation=True, max_length=3072)
    inputs = {k: v.to(dev) for k, v in inputs.items()}
    eos = getattr(_TOKENIZER, "eos_token_id", None)
    pad = getattr(_TOKENIZER, "pad_token_id", None) or eos

    with torch.inference_mode():
        out = _MODEL.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=pad,
            eos_token_id=eos,
        )

    gen = out[0, inputs["input_ids"].shape[1] :]
    text = _TOKENIZER.decode(gen, skip_special_tokens=True).strip()
    return text or None
