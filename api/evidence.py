"""Rule-based linguistic evidence lists for interpretability (no LLM)."""

from __future__ import annotations

import re
from typing import Any, Dict, List

from .sentence_shap import split_sentences

_DISCOURSE_MARKERS = (
    "however",
    "in conclusion",
    "furthermore",
    "therefore",
    "although",
    "moreover",
    "first",
    "second",
    "third",
    "nevertheless",
    "in addition",
    "on the other hand",
)

_CAUSAL = ("because", "therefore", "since", "as a result", "consequently", "thus")

_L1_CALQUES = (
    "for the work",
    "for the communication",
    "more open mind",
    "decide to study",
    "every time more",
    "companies international",
    "the learning of",
    "open mind and respectful",
)

_ARTICLE_OVERUSE = (
    "the work",
    "the communication",
    "the languages",
    "the learning",
)

_NAT_TOPIC_HINTS: Dict[str, List[str]] = {
    "br": ["brazil", "brazilian", "rio", "são paulo", "sao paulo"],
    "pt": ["portugal", "portuguese", "lisbon", "lisboa"],
    "de": ["germany", "german", "berlin", "munich"],
    "it": ["italy", "italian", "rome", "milano", "modena", "emilia"],
    "es": ["spain", "spanish", "madrid", "barcelona"],
    "mx": ["mexico", "mexican"],
    "jp": ["japan", "japanese", "tokyo"],
    "cn": ["china", "chinese", "beijing"],
    "fr": ["france", "french", "paris"],
}


def _find_quoted(text: str, patterns: tuple[str, ...]) -> List[str]:
    lower = text.lower()
    found: List[str] = []
    for p in patterns:
        if p in lower:
            for m in re.finditer(re.escape(p), text, re.I):
                found.append(text[m.start() : m.end()])
                break
    return found[:4]


def _discourse_examples(text: str) -> List[str]:
    found: List[str] = []
    for marker in _DISCOURSE_MARKERS:
        for m in re.finditer(rf"\b{re.escape(marker)}\b", text, re.I):
            found.append(text[m.start() : m.end()])
            break
    return found[:4]


def _causal_examples(text: str) -> List[str]:
    return _find_quoted(text, _CAUSAL)


def build_head_evidence(
    head: str,
    predicted_label: str,
    essay_text: str,
) -> List[Dict[str, Any]]:
    """Return matched evidence items for one head."""
    text = essay_text.strip()
    if not text:
        return []

    sentences = split_sentences(text)
    items: List[Dict[str, Any]] = []

    if head == "cefr":
        discourse = _discourse_examples(text)
        if discourse:
            items.append(
                {
                    "feature": "discourse markers",
                    "matched": True,
                    "examples": discourse,
                }
            )
        if len(sentences) >= 4:
            items.append(
                {
                    "feature": "multi-sentence essay structure",
                    "matched": True,
                    "examples": [f"{len(sentences)} sentences"],
                }
            )
        causal = _causal_examples(text)
        if causal:
            items.append(
                {
                    "feature": "causal / logical connectors",
                    "matched": True,
                    "examples": causal,
                }
            )
        avg_len = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
        if avg_len >= 14:
            items.append(
                {
                    "feature": "longer, more complex sentences",
                    "matched": True,
                    "examples": [f"avg {avg_len:.0f} words per sentence"],
                }
            )
        elif avg_len <= 8 and sentences:
            items.append(
                {
                    "feature": "short, simple sentences",
                    "matched": True,
                    "examples": [f"avg {avg_len:.0f} words per sentence"],
                }
            )

    elif head == "l1":
        calques = _find_quoted(text, _L1_CALQUES)
        if calques:
            items.append(
                {
                    "feature": "possible L1 calque structures",
                    "matched": True,
                    "examples": calques,
                }
            )
        articles = _find_quoted(text, _ARTICLE_OVERUSE)
        if articles:
            items.append(
                {
                    "feature": "article patterns typical of L2 transfer",
                    "matched": True,
                    "examples": articles,
                }
            )
        non_en = re.findall(r"\b(und|oder|avec|porque|muy|très|anche|sehr)\b", text, re.I)
        if non_en:
            items.append(
                {
                    "feature": "untranslated L1 tokens in English text",
                    "matched": True,
                    "examples": list(dict.fromkeys(non_en))[:4],
                }
            )

    elif head == "nat":
        nat_key = predicted_label.lower().strip()
        hints = _NAT_TOPIC_HINTS.get(nat_key, [])
        topic_hits = _find_quoted(text, tuple(hints))
        if topic_hits:
            items.append(
                {
                    "feature": "topic / place references linked to predicted nationality",
                    "matched": True,
                    "examples": topic_hits,
                }
            )
        else:
            items.append(
                {
                    "feature": "writing-style cues from learner corpus (no strong topic keyword)",
                    "matched": True,
                    "examples": [],
                }
            )

    return items


def build_all_evidence(
    predictions: Dict[str, str],
    head_keys: List[str],
    essay_text: str,
) -> Dict[str, List[Dict[str, Any]]]:
    label_for = {
        "cefr": predictions["cefr"],
        "l1": predictions["l1"],
        "nat": predictions["nationality"],
    }
    return {
        hk: build_head_evidence(hk, label_for[hk], essay_text)
        for hk in head_keys
    }
