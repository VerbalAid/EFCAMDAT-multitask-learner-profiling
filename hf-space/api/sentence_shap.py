"""Aggregate token-level gradient attribution to sentences for the predicted class."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence

HEAD_LABELS = {"cefr": "CEFR", "l1": "L1", "nat": "Nationality"}

_RAW_PREFIX = "[RAW] "
_CORRECTED_SEP = " [CORRECTED] "
_MIN_SENTENCE_FRAC = 0.12
_MIN_TOKEN_FRAC = 0.08
_MAX_SENTENCES = 5


def raw_token_indices_from_offsets(offset_mapping: list, dual_text: str) -> list[int]:
    """Learner (RAW) token indices from tokenizer offset mapping."""
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


def split_sentences(text: str) -> List[str]:
    text = text.strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def _learner_text_base(dual_text: str) -> int:
    if dual_text.startswith(_RAW_PREFIX):
        return len(_RAW_PREFIX)
    return 0


def _corrected_sep_start(dual_text: str) -> int | None:
    sep = " [CORRECTED] "
    j = dual_text.find(sep)
    return j if j >= 0 else None


def _to_learner_span(
    char_start: int,
    char_end: int,
    dual_text: str,
) -> tuple[int, int] | None:
    base = _learner_text_base(dual_text)
    sep = _corrected_sep_start(dual_text)
    if sep is not None and char_start >= sep:
        return None
    if char_end <= base:
        return None
    return max(0, char_start - base), max(0, char_end - base)


def _merge_subword_spans(scored: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge adjacent subword pieces into surface spans; sum signed attribution."""
    with_span = [t for t in scored if t.get("start") is not None and t.get("end") is not None]
    without = [t for t in scored if t.get("start") is None or t.get("end") is None]
    with_span.sort(key=lambda x: (x["start"], x["end"]))

    merged: List[Dict[str, Any]] = []
    for t in with_span:
        shap = float(t["shap"])
        if (
            merged
            and merged[-1]["end"] == t["start"]
            and (merged[-1]["shap"] >= 0) == (shap >= 0)
        ):
            merged[-1]["end"] = t["end"]
            merged[-1]["token"] += t["token"].lstrip()
            merged[-1]["shap"] += shap
        else:
            merged.append(
                {
                    "token": t["token"],
                    "shap": shap,
                    "start": int(t["start"]),
                    "end": int(t["end"]),
                }
            )

    return merged + without


def _sentence_spans(learner_text: str, sentences: List[str]) -> List[tuple[int, int]]:
    spans: List[tuple[int, int]] = []
    pos = 0
    for sent in sentences:
        idx = learner_text.find(sent, pos)
        if idx < 0:
            idx = pos
        spans.append((idx, idx + len(sent)))
        pos = idx + len(sent)
    return spans


def _sentence_for_span(start: int, end: int, spans: Sequence[tuple[int, int]]) -> int | None:
    mid = (start + end) // 2
    for i, (a, b) in enumerate(spans):
        if a <= mid < b:
            return i
    best_i: int | None = None
    best_overlap = 0
    for i, (a, b) in enumerate(spans):
        overlap = max(0, min(end, b) - max(start, a))
        if overlap > best_overlap:
            best_overlap = overlap
            best_i = i
    return best_i if best_overlap > 0 else None


def _assign_word_to_sentence(word: str, sentences: List[str]) -> int | None:
    w = word.lower().strip()
    if len(w) < 2:
        return None
    for i, sent in enumerate(sentences):
        if w in sent.lower():
            return i
    return None


def _toward_tokens_for_sentence(
    sentence: str,
    sent_abs_start: int,
    groups: List[Dict[str, Any]],
    sentence_idx: int,
    all_sentences: List[str],
) -> List[Dict[str, Any]]:
    """Tokens that increase the predicted-class logit within one sentence."""
    sent_end = sent_abs_start + len(sentence)
    raw: List[Dict[str, Any]] = []

    for g in groups:
        shap = float(g["shap"])
        if shap <= 0:
            continue
        start = g.get("start")
        end = g.get("end")
        if start is not None and end is not None:
            if end <= sent_abs_start or start >= sent_end:
                continue
            rel_start = max(0, int(start) - sent_abs_start)
            rel_end = min(len(sentence), int(end) - sent_abs_start)
            if rel_end <= rel_start:
                continue
            raw.append(
                {
                    "text": sentence[rel_start:rel_end],
                    "start": rel_start,
                    "end": rel_end,
                    "signed_mass": shap,
                }
            )
            continue

        si = _assign_word_to_sentence(g["token"], all_sentences)
        if si != sentence_idx:
            continue
        tok = str(g["token"]).strip().lstrip()
        if len(tok) < 2:
            continue
        for m in re.finditer(re.escape(tok), sentence, re.I):
            raw.append(
                {
                    "text": sentence[m.start() : m.end()],
                    "start": m.start(),
                    "end": m.end(),
                    "signed_mass": shap,
                }
            )
            break

    if not raw:
        return []

    raw.sort(key=lambda x: x["signed_mass"], reverse=True)
    occupied = [False] * len(sentence)
    spans: List[Dict[str, Any]] = []
    for tok in raw:
        start, end = tok["start"], tok["end"]
        if any(occupied[start:end]):
            continue
        for i in range(start, end):
            occupied[i] = True
        spans.append(
            {
                "text": tok["text"],
                "start": start,
                "end": end,
                "signed_mass": round(tok["signed_mass"], 5),
                "direction": "positive",
            }
        )

    spans.sort(key=lambda x: x["start"])
    peak = max(s["signed_mass"] for s in spans) or 1.0
    total = sum(s["signed_mass"] for s in spans) or 1.0
    out: List[Dict[str, Any]] = []
    for s in spans:
        if s["signed_mass"] / peak < _MIN_TOKEN_FRAC:
            continue
        out.append({**s, "attribution": round(s["signed_mass"] / total, 4)})
    return out


def aggregate_to_sentences(
    learner_text: str,
    scored_tokens: List[Dict[str, Any]],
    *,
    sort_by_attribution: bool = True,
) -> List[Dict[str, Any]]:
    """
    Map token attributions to sentences for the predicted class.
    Sentence score = net signed mass (toward − against) in that sentence.
    Only sentences with positive net toward signal are returned by default.
    """
    sentences = split_sentences(learner_text)
    if not sentences:
        return []

    groups = _merge_subword_spans(scored_tokens)
    spans = _sentence_spans(learner_text, sentences)
    toward_mass = [0.0] * len(sentences)
    against_mass = [0.0] * len(sentences)

    for g in groups:
        shap = float(g["shap"])
        start = g.get("start")
        end = g.get("end")
        if start is not None and end is not None:
            si = _sentence_for_span(int(start), int(end), spans)
        else:
            si = _assign_word_to_sentence(g["token"], sentences)
        if si is None:
            continue
        if shap >= 0:
            toward_mass[si] += shap
        else:
            against_mass[si] += -shap

    total_toward = sum(toward_mass) or 1.0
    total_signed = sum(max(0.0, toward_mass[i] - against_mass[i]) for i in range(len(sentences))) or 1.0
    out: List[Dict[str, Any]] = []
    for i, sent in enumerate(sentences):
        toward = toward_mass[i]
        against = against_mass[i]
        signed = toward - against
        if toward <= 0 and against <= 0:
            continue
        sent_start = spans[i][0]
        tokens = _toward_tokens_for_sentence(sent, sent_start, groups, i, sentences)
        attr = toward / total_toward if toward > 0 else 0.0
        signed_attr = signed / total_signed if total_signed > 0 else 0.0
        out.append(
            {
                "sentence": sent,
                "attribution": round(attr, 4),
                "signed_attribution": round(signed_attr, 4),
                "direction": "positive" if signed > 0 else "negative",
                "signed_mass": round(signed, 4),
                "toward_mass": round(toward, 4),
                "tokens": tokens,
            }
        )

    if not out:
        return []

    peak_signed = max((r["signed_mass"] for r in out), default=0.0)
    cutoff = peak_signed * _MIN_SENTENCE_FRAC
    filtered = [r for r in out if r["signed_mass"] >= cutoff]
    if not filtered:
        filtered = sorted(out, key=lambda x: x["signed_mass"], reverse=True)[:3]

    if sort_by_attribution:
        filtered.sort(key=lambda x: (x["signed_mass"], x["signed_attribution"]), reverse=True)

    supporting = [r for r in filtered if r["signed_mass"] > 0]
    if supporting:
        peak = supporting[0]["signed_mass"]
        cutoff = peak * _MIN_SENTENCE_FRAC
        supporting = [r for r in supporting if r["signed_mass"] >= cutoff]
        return supporting[:_MAX_SENTENCES]

    return filtered[: min(3, _MAX_SENTENCES)]


def _truncate_words(s: str, max_len: int = 80) -> str:
    s = s.strip()
    if len(s) <= max_len:
        return s
    words = s.split()
    parts: List[str] = []
    length = 0
    for word in words:
        add = len(word) + (1 if parts else 0)
        if length + add + 1 > max_len:
            break
        parts.append(word)
        length += add
    if not parts:
        return s[: max_len - 1].rstrip() + "…"
    return " ".join(parts) + "…"


def _phrase_label(row: Dict[str, Any]) -> str:
    tok_preview = ", ".join(t["text"] for t in (row.get("tokens") or [])[:3])
    extra = f"; key words: {tok_preview}" if tok_preview else ""
    return f'"{_truncate_words(row["sentence"])}"{extra}'


def _quote_phrases(sentences: List[Dict[str, Any]], k: int = 2) -> List[str]:
    ranked = sorted(sentences, key=lambda x: x.get("signed_mass", 0), reverse=True)
    phrases: List[str] = []
    for row in ranked[:k]:
        if row.get("signed_mass", 0) <= 0:
            break
        phrases.append(_phrase_label(row))
    return phrases


def _describe_sentence_roles(sentences: List[Dict[str, Any]]) -> str:
    """Heuristic label for where flat attribution concentrates."""
    if not sentences:
        return "several passages"
    ranked = sorted(sentences, key=lambda x: x.get("signed_mass", 0), reverse=True)
    roles: List[str] = []
    all_sents = [r["sentence"] for r in ranked]
    for row in ranked[:3]:
        sent = row["sentence"]
        lower = sent.lower()
        if sent == all_sents[-1] or any(
            m in lower for m in ("in conclusion", "to sum up", "in summary", "finally")
        ):
            roles.append("the conclusion")
        elif sent == all_sents[0]:
            roles.append("topic introduction")
        elif any(m in lower for m in ("because", "therefore", "although", "however")):
            roles.append("causal reasoning passages")
        else:
            roles.append("topic-development sentences")
    # dedupe preserving order
    seen: set[str] = set()
    unique_roles: List[str] = []
    for r in roles:
        if r not in seen:
            seen.add(r)
            unique_roles.append(r)
    if len(unique_roles) == 1:
        return unique_roles[0]
    return " and ".join(unique_roles[:2])


def is_flat_attribution(sentences: List[Dict[str, Any]], threshold: int = 10) -> bool:
    """True when top 3 net-supporting sentences are within `threshold` relative % points."""
    ranked = sorted(
        [s for s in sentences if float(s.get("signed_mass") or 0) > 0],
        key=lambda x: float(x.get("signed_mass") or 0),
        reverse=True,
    )
    if len(ranked) < 3:
        return False
    peak = float(ranked[0].get("signed_mass") or 0)
    if peak <= 0:
        return False
    pcts = [
        max(0, min(100, round(100 * float(r.get("signed_mass") or 0) / peak)))
        for r in ranked[:3]
    ]
    return max(pcts) - min(pcts) <= threshold


def template_narrative(
    head_key: str,
    predicted_label: str,
    sentences: List[Dict[str, Any]],
) -> str:
    label = HEAD_LABELS.get(head_key, head_key.upper())
    if not sentences:
        return f"{label} → {predicted_label}: no supporting sentences found."

    ranked = sorted(sentences, key=lambda x: x.get("signed_mass", 0), reverse=True)
    top = ranked[0]
    phrases = _quote_phrases(sentences, k=2)

    if top.get("signed_mass", 0) <= 0:
        return f"{label} → {predicted_label}: no clear token evidence for this class."

    if is_flat_attribution(sentences):
        phrases = _quote_phrases(sentences, k=3)
        roles = _describe_sentence_roles(ranked)
        quoted = "; ".join(phrases[:2]) if phrases else ""
        lead = (
            f"{label} → {predicted_label}: strongest evidence comes from "
            f"{roles}. The model appears to rely on overall essay organisation "
            f"rather than a single sentence."
        )
        if quoted:
            lead += f" Top passages: {quoted}."
        if len(phrases) > 2:
            lead += f" Also: {phrases[2]}."
        return lead

    peak_signed = max(float(s.get("signed_mass") or 0) for s in ranked) or 1.0
    share = float(top.get("signed_mass") or 0) / peak_signed
    lead = (
        f"{label} → {predicted_label}: model leans on {phrases[0]} "
        f"({share:.0%} net toward-class attribution)."
    )
    if len(phrases) > 1:
        lead += f" Also: {phrases[1]}."
    return lead


def _top_row(sentences: List[Dict[str, Any]]) -> tuple[int, Dict[str, Any] | None]:
    if not sentences:
        return 0, None
    top = max(sentences, key=lambda x: x.get("signed_mass", 0))
    idx = next((i for i, s in enumerate(sentences) if s["sentence"] == top["sentence"]), 0)
    return idx + 1, top


def build_head_comparison(
    per_head_sentences: Dict[str, List[Dict[str, Any]]],
    head_keys: List[str],
) -> str | None:
    if "l1" not in head_keys or "nat" not in head_keys:
        return None
    l1_rows = per_head_sentences.get("l1") or []
    nat_rows = per_head_sentences.get("nat") or []
    l1_idx, l1_top = _top_row(l1_rows)
    nat_idx, nat_top = _top_row(nat_rows)
    if not l1_top or not nat_top:
        return None
    if l1_top["sentence"] == nat_top["sentence"]:
        return (
            "L1 and nationality heads share the same top supporting sentence, consistent with "
            "correlation between these tasks."
        )
    return (
        f"L1 top evidence: “{_truncate_words(l1_top['sentence'], 60)}”; "
        f"nationality top evidence: “{_truncate_words(nat_top['sentence'], 60)}”."
    )


def narrate_shap_per_head(
    predictions: Dict[str, str],
    per_head_sentences: Dict[str, List[Dict[str, Any]]],
    head_keys: List[str],
) -> Dict[str, str]:
    label_for = {"cefr": predictions["cefr"], "l1": predictions["l1"], "nat": predictions["nationality"]}
    return {
        hk: template_narrative(hk, label_for[hk], per_head_sentences.get(hk) or [])
        for hk in head_keys
    }


_UNIQUE_MIN = 0.02
_SHARED_MIN = 0.02


def _row_from_mass(sentence: str, mass: float, source_row: Dict[str, Any] | None) -> Dict[str, Any]:
    base = dict(source_row) if source_row else {"sentence": sentence, "tokens": []}
    base["sentence"] = sentence
    base["signed_mass"] = round(mass, 4)
    base["signed_attribution"] = round(max(mass, 0.0), 4)
    base["direction"] = "positive" if mass > 0 else "negative"
    base["attribution"] = round(max(mass, 0.0), 4)
    return base


def decompose_head_attribution(
    per_head_sentences: Dict[str, List[Dict[str, Any]]],
    head_keys: List[str],
) -> Dict[str, Any]:
    """
    Split attribution into shared (mean across heads) vs head-specific (residual).
    unique_attr[head] = attr[head] - mean(attr_all_heads)
    """
    if len(head_keys) < 2:
        return {"shared_signals": [], "head_specific_signals": {}}

    lookup: Dict[str, Dict[str, Dict[str, Any]]] = {}
    all_sentences: set[str] = set()
    for hk in head_keys:
        lookup[hk] = {}
        for row in per_head_sentences.get(hk) or []:
            sent = row["sentence"]
            lookup[hk][sent] = row
            all_sentences.add(sent)

    shared_scores: Dict[str, float] = {}
    unique_scores: Dict[str, Dict[str, float]] = {hk: {} for hk in head_keys}

    for sent in all_sentences:
        masses = {
            hk: float(lookup.get(hk, {}).get(sent, {}).get("signed_mass") or 0.0)
            for hk in head_keys
        }
        mean_mass = sum(masses.values()) / len(head_keys)
        shared_scores[sent] = mean_mass
        for hk in head_keys:
            unique_scores[hk][sent] = masses[hk] - mean_mass

    def _top_by(scores: Dict[str, float], min_val: float, k: int = 3) -> List[Dict[str, Any]]:
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        out: List[Dict[str, Any]] = []
        for sent, mass in ranked:
            if mass < min_val:
                break
            source = None
            for hk in head_keys:
                if sent in lookup.get(hk, {}):
                    source = lookup[hk][sent]
                    break
            out.append(_row_from_mass(sent, mass, source))
            if len(out) >= k:
                break
        return out

    shared = _top_by(shared_scores, _SHARED_MIN)
    head_specific = {
        hk: _top_by(unique_scores[hk], _UNIQUE_MIN)
        for hk in head_keys
    }
    return {"shared_signals": shared, "head_specific_signals": head_specific}
