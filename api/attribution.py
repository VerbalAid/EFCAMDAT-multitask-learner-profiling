"""Unified attribution: SHAP (default, notebook method) with gradient fallback."""

from __future__ import annotations

import logging
import os
from typing import List, Tuple

import torch

logger = logging.getLogger(__name__)


def head_token_attribution(
    dual_text: str,
    model: torch.nn.Module,
    tokenizer,
    device: torch.device,
    head_key: str,
    pred_class_i: int,
    max_length: int,
) -> Tuple[List[dict], str]:
    """
    Score learner tokens for one head's predicted class.
    Default: SHAP (same as Cambridge_Models_final.ipynb).
    Fallback: integrated-gradient if SHAP unavailable or CAMBRIDGE_ATTR=gradient.
    """
    prefer = os.environ.get("CAMBRIDGE_ATTR", "shap").strip().lower()

    if prefer != "gradient":
        try:
            from .shap_narrate import top_token_shap

            scored = top_token_shap(
                dual_text, model, tokenizer, device, head_key, pred_class_i, max_length
            )
            if scored and max(abs(t["shap"]) for t in scored) > 1e-12:
                return scored, "shap"
            logger.warning("shap %s returned near-zero values — trying gradient", head_key)
        except ImportError:
            logger.warning("shap not installed — pip install shap; using gradient fallback")
        except Exception as exc:
            logger.warning("shap %s failed (%s) — using gradient fallback", head_key, exc)

    from .gradient_attr import head_token_attribution as gradient_attr

    scored, method = gradient_attr(
        dual_text, model, tokenizer, device, head_key, pred_class_i, max_length
    )
    return scored, method
