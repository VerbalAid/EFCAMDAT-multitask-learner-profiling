"""Per-head one-sentence summary via OpenRouter/Ollama (constrained prompt)."""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from .llm_client import llm_chat, llm_available
from .sentence_shap import is_flat_attribution

logger = logging.getLogger(__name__)

SUMMARY_MODE = "openrouter_v1"
OLLAMA_NUM_PREDICT = 60

HEAD_LABEL = {"cefr": "CEFR", "l1": "L1", "nat": "Nationality"}

MAX_WORDS = 30
MAX_SUMMARY_WORDS = 35
QUOTED_WORDS_RE = re.compile(r"['\"](\w+)['\"]")
BANNED_RE = re.compile(
    r"\b(might|could|may|likely|uncertain|contributed|influenced|suggests|prevalence|cholesterol)\b",
    re.I,
)


def _clean_token(text: str) -> str:
    return str(text).strip().strip("\"'").strip(".,;:!?")


def build_ollama_prompt(head_name, label, top_sentence, top_pct, top_words, second_sentence, second_pct):
    clean_words = [_clean_token(w) for w in top_words[:3]]
    clean_words = [w for w in clean_words if w]
    system = (
        "You rewrite structured data into one English sentence. "
        "Do NOT translate words into any other language. "
        "Do NOT use the words might, could, may, likely, uncertain, "
        "contributed, influenced, or suggests. "
        "Do NOT explain grammar or linguistics. "
        "Only cite the Key words listed below — do not summarise essay content. "
        "Max 30 words."
    )
    user = (
        f"Head: {head_name}, Prediction: {label}\n"
        f"Top sentence (100%): \"{top_sentence[:80]}\"\n"
        f"Key words: {', '.join(clean_words)}\n"
        f"Second sentence ({second_pct}%): \"{second_sentence[:80]}\"\n\n"
        f"Write ONE sentence starting with: "
        f"'The model predicted {label} mainly because of'"
    )
    return system, user


def _ranked_supporting(sentences: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    supporting = [r for r in sentences if float(r.get("signed_mass") or 0) > 0]
    supporting.sort(key=lambda x: float(x.get("signed_mass") or 0), reverse=True)
    return supporting


def _relative_pct(row: Dict[str, Any], peak: float) -> int:
    if peak <= 0:
        return 0
    mass = float(row.get("signed_mass") or 0)
    return max(0, min(100, round(100 * mass / peak)))


_SUMMARY_STOP_WORDS = frozenset(
    {
        "the", "a", "an", "is", "in", "of", "to", "and", "for", "it", "on", "at", "as", "or",
        "be", "he", "we", "so", "no", "do", "are", "was", "were", "been", "being", "this",
        "that", "these", "those", "its", "by", "with", "from", "not", "if", "but", "or",
        "all", "recent", "first",
    }
)


def _is_summary_stopword(word: str) -> bool:
    clean = _clean_token(word).lower()
    return not clean or clean in _SUMMARY_STOP_WORDS


def _summary_key_words(row: Dict[str, Any], n: int = 3) -> List[str]:
    """Content words for LLM prompt and validation (stop words excluded)."""
    pool = _key_words(row, 5)
    out: List[str] = []
    seen: set[str] = set()
    for word in pool:
        clean = _clean_token(word)
        low = clean.lower()
        if not clean or low in seen or low in _SUMMARY_STOP_WORDS:
            continue
        seen.add(low)
        out.append(clean)
        if len(out) >= n:
            break
    return out


_STOPWORDS = frozenset(
    {
        "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "of",
        "is", "are", "was", "were", "be", "been", "being", "this", "that", "these",
        "those", "it", "its", "as", "by", "with", "from", "not", "no", "so", "if",
    }
)


def _key_words(row: Dict[str, Any], n: int = 3) -> List[str]:
    tokens = row.get("tokens") or []
    ranked = sorted(tokens, key=lambda t: float(t.get("attribution") or t.get("signed_mass") or 0), reverse=True)
    words: List[str] = []
    seen: set[str] = set()
    for tok in ranked:
        text = _clean_token(tok.get("text", ""))
        if not text or text.lower() in seen:
            continue
        seen.add(text.lower())
        words.append(text)
        if len(words) >= n:
            break

    if len(words) < n:
        sentence = str(row.get("sentence", ""))
        candidates: List[str] = []
        for match in re.finditer(r"[A-Za-z']+", sentence):
            text = _clean_token(match.group(0))
            low = text.lower()
            if not text or low in seen or low in _STOPWORDS:
                continue
            candidates.append(text)
        candidates.sort(key=lambda w: (len(w) >= 2, len(w)), reverse=True)
        for text in candidates:
            seen.add(text.lower())
            words.append(text)
            if len(words) >= n:
                break
    return words


def _first_sentence(text: str) -> str:
    text = re.sub(r"\s+", " ", (text or "").strip()).strip('"\'')
    if not text:
        return ""
    match = re.match(r"^(.+?[.!?])(?:\s+|$)", text)
    return match.group(1).strip() if match else text


def _quoted_words(text: str) -> List[str]:
    return [_clean_token(w).lower() for w in QUOTED_WORDS_RE.findall(text) if _clean_token(w)]


def _quoted_key_word_hits(text: str, key_words: List[str]) -> int:
    quoted_lower = _quoted_words(text)
    cited = [
        _clean_token(w)
        for w in key_words
        if _clean_token(w) and not _is_summary_stopword(w)
    ]
    if len(cited) < 2:
        return 0
    return sum(1 for w in cited if w.lower() in quoted_lower)


def _validate_summary(text: str, label: str, top_words: List[str]) -> Optional[str]:
    text = _first_sentence(text)
    if not text:
        return None
    if BANNED_RE.search(text):
        return None
    if len(text.split()) > MAX_SUMMARY_WORDS:
        return None
    starter = f"The model predicted {label} mainly because of"
    if not text.lower().startswith(starter.lower()):
        return None

    if _quoted_key_word_hits(text, top_words) < 2:
        return None

    if text[-1] not in ".!?":
        text += "."
    return text


def _deterministic_fallback(label: str, top: Dict[str, Any], second: Optional[Dict[str, Any]]) -> str:
    top_words = _summary_key_words(top, 3) or _key_words(top, 3)
    if len(top_words) >= 3:
        lead = (
            f"The model predicted {label} mainly because of "
            f"'{top_words[0]}', '{top_words[1]}', and '{top_words[2]}' in the top sentence"
        )
    elif len(top_words) == 2:
        lead = (
            f"The model predicted {label} mainly because of "
            f"'{top_words[0]}' and '{top_words[1]}' in the top sentence"
        )
    elif len(top_words) == 1:
        lead = f"The model predicted {label} mainly because of '{top_words[0]}' in the top sentence"
    else:
        sent = str(top.get("sentence", "")).strip()[:80]
        lead = f'The model predicted {label} mainly because of "{sent}" in the top sentence'

    if second:
        second_words = _key_words(second, 1)
        if second_words:
            lead += f", and '{second_words[0]}' in the second"
    return lead + "."


def narrate_attribution_per_head(
    predictions: Dict[str, str],
    per_head_sentences: Dict[str, List[Dict[str, Any]]],
    head_keys: List[str],
    *,
    essay_text: str = "",
) -> Dict[str, str]:
    del essay_text
    label_for = {
        "cefr": predictions["cefr"],
        "l1": predictions["l1"],
        "nat": predictions["nationality"],
    }
    out: Dict[str, str] = {}
    for hk in head_keys:
        ranked = _ranked_supporting(per_head_sentences.get(hk) or [])
        if not ranked:
            out[hk] = ""
            continue

        label = label_for[hk]
        top = ranked[0]
        second = ranked[1] if len(ranked) > 1 else None
        peak = float(top.get("signed_mass") or 0)

        if is_flat_attribution(ranked):
            out[hk] = ""
            continue

        top_words = _summary_key_words(top, 3)
        if len(top_words) < 2:
            out[hk] = _deterministic_fallback(label, top, second)
            continue

        if not llm_available():
            out[hk] = _deterministic_fallback(label, top, second)
            continue

        head_name = HEAD_LABEL.get(hk, hk)
        top_sentence = str(top.get("sentence", "")).strip()
        second_sentence = str(second.get("sentence", "")).strip() if second else ""
        second_pct = _relative_pct(second, peak) if second else 0

        system, user = build_ollama_prompt(
            head_name,
            label,
            top_sentence,
            100,
            top_words,
            second_sentence,
            second_pct,
        )

        raw = llm_chat(system, user, max_tokens=OLLAMA_NUM_PREDICT, temperature=0)
        if raw:
            logger.debug("LLM raw summary for %s (%d chars): %s", hk, len(raw), raw[:120])
        validated = _validate_summary(raw, label, top_words) if raw else None
        out[hk] = validated if validated else _deterministic_fallback(label, top, second)
        if raw and not validated:
            logger.info("LLM summary rejected for %s; using fallback", hk)
    return out
