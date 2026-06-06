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


def _pad_id(tokenizer) -> int:
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id or 1
    return int(pad_id)


def _token_attr_from_grad(
    grad: torch.Tensor,
    input_embeds: torch.Tensor,
    baseline_embeds: torch.Tensor,
) -> np.ndarray:
    """
    Signed attribution for the predicted-class logit.
    Use grad * (input - baseline), not grad * input: post-LN embeddings are
    zero-centred so grad*input often flips sign vs the true toward-class direction.
    """
    delta = input_embeds.detach()[0] - baseline_embeds.detach()[0]
    return (grad[0] * delta).sum(dim=-1).detach().cpu().numpy()


def _calibrate_attribution_sign(
    token_attr: np.ndarray,
    raw_indices: list[int],
    logit_input: float,
    logit_baseline: float,
) -> np.ndarray:
    """Flip attribution if learner-token mass disagrees with logit(input) - logit(baseline)."""
    if not raw_indices:
        return token_attr
    expected = float(logit_input - logit_baseline)
    raw_sum = float(token_attr[raw_indices].sum())
    if expected == 0.0 or raw_sum == 0.0:
        return token_attr
    if (expected > 0) != (raw_sum > 0):
        return -token_attr
    return token_attr


def _gradient_attr(
    model: torch.nn.Module,
    tokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    head_key: str,
    pred_class_i: int,
) -> np.ndarray:
    emb = model.encoder.embeddings
    pad_id = _pad_id(tokenizer)
    input_embeds = emb(input_ids=input_ids)
    baseline_embeds = emb(input_ids=torch.full_like(input_ids, pad_id))
    input_embeds.retain_grad()
    logit = _head_logit_from_embeds(model, input_embeds, attention_mask, head_key, pred_class_i)
    logit.backward(retain_graph=False)
    grad = input_embeds.grad
    if grad is None:
        return np.zeros(input_embeds.shape[1], dtype=np.float64)
    return _token_attr_from_grad(grad, input_embeds, baseline_embeds)


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
    pad_id = _pad_id(tokenizer)

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
    return _token_attr_from_grad(avg_grad, input_embeds, baseline)


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

    raw_i = _raw_token_indices_from_offsets(offset_mapping, dual_text)

    try:
        with torch.enable_grad():
            with torch.no_grad():
                emb = model.encoder.embeddings
                input_embeds_ref = emb(input_ids=input_ids)
                baseline_embeds_ref = emb(input_ids=torch.full_like(input_ids, _pad_id(tokenizer)))
                logit_input = float(
                    _head_logit_from_embeds(
                        model, input_embeds_ref, attention_mask, head_key, pred_class_i
                    ).item()
                )
                logit_baseline = float(
                    _head_logit_from_embeds(
                        model, baseline_embeds_ref, attention_mask, head_key, pred_class_i
                    ).item()
                )

            model.zero_grad(set_to_none=True)
            token_attr = _gradient_attr(
                model, tokenizer, input_ids, attention_mask, head_key, pred_class_i
            )
            token_attr = _calibrate_attribution_sign(
                token_attr, raw_i, logit_input, logit_baseline
            )
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
                token_attr = _calibrate_attribution_sign(
                    token_attr, raw_i, logit_input, logit_baseline
                )
                scored = _extract_scored_tokens(
                    token_attr, tokenizer, input_ids, dual_text, offset_mapping
                )
                method = "integrated_gradients"
                peak = _raw_peak(scored)

            logger.debug(
                "gradient_attr %s class=%d method=%s peak=%.4f n_tokens=%d logit=%.3f baseline=%.3f",
                head_key,
                pred_class_i,
                method,
                peak,
                len(scored),
                logit_input,
                logit_baseline,
            )
    finally:
        model.train(was_training)
        model.zero_grad(set_to_none=True)

    return scored, method
