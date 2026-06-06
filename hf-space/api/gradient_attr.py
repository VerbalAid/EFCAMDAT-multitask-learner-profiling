"""Head-specific gradient / integrated-gradient attribution on input embeddings."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np
import torch

logger = logging.getLogger(__name__)

_RAW_PREFIX = "[RAW] "
_CORRECTED_SEP = " [CORRECTED] "
_IG_STEPS = 8
_ATTR_EPS = 1e-7


def _is_skip_token(lab: str) -> bool:
    if lab in ("<s>", "</s>", "<pad>", "", "Ċ"):
        return True
    return "[RAW]" in lab or lab == "[RAW]" or "[CORRECTED]" in lab


def _raw_token_indices_from_offsets(offset_mapping: list, dual_text: str) -> list[int]:
    sep_start = dual_text.find(_CORRECTED_SEP)
    if sep_start < 0:
        sep_start = len(dual_text)
    raw_start = len(_RAW_PREFIX) if dual_text.startswith(_RAW_PREFIX) else 0
    raw_i: list[int] = []
    for ti, span in enumerate(offset_mapping):
        a, b = span[0], span[1]
        if a is None or b is None:
            continue
        a, b = int(a), int(b)
        if a == 0 and b == 0:
            continue
        if b <= sep_start and a >= raw_start:
            raw_i.append(ti)
    return raw_i


def _head_logit_from_embeds(
    model: torch.nn.Module,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    head_key: str,
    pred_class_i: int,
) -> torch.Tensor:
    heads = {"cefr": model.cefr_head, "l1": model.l1_head, "nat": model.nat_head}
    encoder_out = model.encoder(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
    pooled = model.dropout(encoder_out.last_hidden_state[:, 0, :])
    return heads[head_key](pooled)[0, pred_class_i]


def _gradient_attr(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    head_key: str,
    pred_class_i: int,
) -> np.ndarray:
    emb = model.encoder.embeddings
    inputs_embeds = emb(input_ids=input_ids)
    inputs_embeds.retain_grad()
    logit = _head_logit_from_embeds(model, inputs_embeds, attention_mask, head_key, pred_class_i)
    logit.backward(retain_graph=False)
    grad = inputs_embeds.grad
    if grad is None:
        return np.zeros(inputs_embeds.shape[1], dtype=np.float64)
    return (grad[0] * inputs_embeds.detach()[0]).sum(dim=-1).detach().cpu().numpy()


def _integrated_grad_attr(
    model: torch.nn.Module,
    tokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    head_key: str,
    pred_class_i: int,
    steps: int = _IG_STEPS,
) -> np.ndarray:
    emb = model.encoder.embeddings
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id or 1

    input_embeds = emb(input_ids=input_ids)
    baseline = emb(input_ids=torch.full_like(input_ids, pad_id))

    total_grad = torch.zeros_like(input_embeds)
    for step in range(1, steps + 1):
        alpha = step / steps
        interp = baseline + alpha * (input_embeds - baseline)
        interp = interp.detach().requires_grad_(True)
        logit = _head_logit_from_embeds(model, interp, attention_mask, head_key, pred_class_i)
        grad = torch.autograd.grad(logit, interp, retain_graph=False, create_graph=False)[0]
        total_grad = total_grad + grad

    avg_grad = total_grad / steps
    integrated = (input_embeds - baseline).detach() * avg_grad
    return integrated[0].sum(dim=-1).detach().cpu().numpy()


def _extract_scored_tokens(
    token_attr: np.ndarray,
    tokenizer,
    input_ids: torch.Tensor,
    dual_text: str,
    offset_mapping: list,
) -> List[Dict[str, Any]]:
    from .sentence_shap import _to_learner_span, raw_token_indices_from_offsets

    raw_i = _raw_token_indices_from_offsets(offset_mapping, dual_text)
    toks = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())

    if not raw_i:
        raw_i = [
            ti
            for ti in range(len(toks))
            if not _is_skip_token(toks[ti].replace("Ġ", " ").strip())
        ]

    scored: List[Dict[str, Any]] = []
    for ti in raw_i:
        if ti >= len(token_attr):
            continue
        lab = toks[ti].replace("Ġ", " ").strip()
        if _is_skip_token(lab):
            continue
        item: Dict[str, Any] = {"token": lab, "shap": float(token_attr[ti])}
        if ti < len(offset_mapping):
            a, b = offset_mapping[ti]
            if a is not None and b is not None and not (a == 0 and b == 0):
                span = _to_learner_span(int(a), int(b), dual_text)
                if span is not None:
                    item["start"], item["end"] = span
        scored.append(item)
    return scored


def _raw_peak(scored: List[Dict[str, Any]]) -> float:
    if not scored:
        return 0.0
    return max(abs(float(t["shap"])) for t in scored)


def head_token_attribution(
    dual_text: str,
    model: torch.nn.Module,
    tokenizer,
    device: torch.device,
    head_key: str,
    pred_class_i: int,
    max_length: int,
) -> tuple[List[Dict[str, Any]], str]:
    """
    Attribution for the predicted class logit on one head.
    Returns (scored_tokens, method) where method is 'gradient' or 'integrated_gradients'.
    """
    enc = tokenizer(
        dual_text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_offsets_mapping=True,
    )
    offset_mapping = enc.pop("offset_mapping")[0]
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    was_training = model.training
    model.eval()
    method = "gradient"

    try:
        with torch.enable_grad():
            model.zero_grad(set_to_none=True)
            token_attr = _gradient_attr(model, input_ids, attention_mask, head_key, pred_class_i)
            scored = _extract_scored_tokens(token_attr, tokenizer, input_ids, dual_text, offset_mapping)
            peak = _raw_peak(scored)

            if peak < _ATTR_EPS:
                logger.warning(
                    "gradient_attr %s: near-zero gradient (peak=%.2e) — using integrated gradients",
                    head_key,
                    peak,
                )
                model.zero_grad(set_to_none=True)
                token_attr = _integrated_grad_attr(
                    model, tokenizer, input_ids, attention_mask, head_key, pred_class_i
                )
                scored = _extract_scored_tokens(
                    token_attr, tokenizer, input_ids, dual_text, offset_mapping
                )
                method = "integrated_gradients"
                peak = _raw_peak(scored)

            logger.debug(
                "gradient_attr %s class=%d method=%s peak=%.4f n_tokens=%d",
                head_key,
                pred_class_i,
                method,
                peak,
                len(scored),
            )
    finally:
        model.train(was_training)
        model.zero_grad(set_to_none=True)

    return scored, method
