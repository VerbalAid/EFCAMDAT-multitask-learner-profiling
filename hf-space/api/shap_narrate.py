"""SHAP token attribution for multitask heads (original notebook method)."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np
import torch

from .sentence_shap import _to_learner_span, raw_token_indices_from_offsets

logger = logging.getLogger(__name__)


def _shap_vec_for_class(shap_row: Any, class_i: int, n_tokens: int) -> np.ndarray:
    V = np.asarray(shap_row.values, dtype=float)
    if V.ndim == 3:
        V = V[0]
    if V.ndim == 2:
        vec = V[:, class_i] if V.shape[1] > 1 else V[:, 0]
    else:
        vec = V.ravel()
    vec = vec[:n_tokens]
    if len(vec) < n_tokens:
        vec = np.pad(vec, (0, n_tokens - len(vec)))
    return vec


def _is_skip_token(lab: str) -> bool:
    if lab in ("<s>", "</s>", "<pad>", "", "Ċ"):
        return True
    return "[RAW]" in lab or lab == "[RAW]" or "[CORRECTED]" in lab


def top_token_shap(
    dual_str: str,
    model: torch.nn.Module,
    tokenizer,
    device: torch.device,
    head_key: str,
    pred_class_i: int,
    max_length: int,
) -> List[Dict[str, Any]]:
    """
    Per-head SHAP on the predicted class logit — separate explainer per head.
    Returns learner-token scores with character offsets when available.
    """
    import shap

    def predict_logit(texts):
        enc = tokenizer(
            list(texts),
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            return model(**enc)[head_key].detach().cpu().numpy()

    explainer = shap.Explainer(predict_logit, tokenizer)
    row = explainer([dual_str])[0]

    enc = tokenizer(
        dual_str,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_offsets_mapping=True,
    )
    toks = tokenizer.convert_ids_to_tokens(enc["input_ids"][0].tolist())
    offset_mapping = enc["offset_mapping"][0]
    svals = _shap_vec_for_class(row, pred_class_i, len(toks))

    raw_i = raw_token_indices_from_offsets(offset_mapping, dual_str)
    if not raw_i:
        raw_i = []
        after_corrected = False
        for ti, w in enumerate(toks):
            if "[CORRECTED]" in w:
                after_corrected = True
                continue
            if after_corrected:
                continue
            if ti < len(svals):
                raw_i.append(ti)

    scored: List[Dict[str, Any]] = []
    for ti in raw_i:
        if ti >= len(svals):
            continue
        lab = toks[ti].replace("Ġ", " ").strip()
        if _is_skip_token(lab):
            continue
        item: Dict[str, Any] = {"token": lab, "shap": float(svals[ti])}
        if ti < len(offset_mapping):
            a, b = offset_mapping[ti]
            if a is not None and b is not None and not (a == 0 and b == 0):
                span = _to_learner_span(int(a), int(b), dual_str)
                if span is not None:
                    item["start"], item["end"] = span
        scored.append(item)

    logger.debug("shap %s class=%d n_tokens=%d peak=%.4f", head_key, pred_class_i, len(scored),
                 max((abs(t["shap"]) for t in scored), default=0.0))
    return scored
