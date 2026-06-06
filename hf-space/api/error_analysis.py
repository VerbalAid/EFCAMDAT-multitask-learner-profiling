"""Essay-level L2 error analysis via OpenRouter/Ollama (one call per essay)."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from .llm_client import llm_available, llm_complete

logger = logging.getLogger(__name__)

PROMPT = """You are an L2 error analyst specialising in {l1} learners of English.

Essay:
{essay}

Predicted L1: {l1}
Predicted CEFR: {cefr}

Sentence attributions (sentences the classifiers weighted most):
{attributions}

Review the full essay. Identify ONLY errors where you can
confidently explain the cause. For each error:
- Quote the exact error span (2-5 words max)
- Give the correction
- Classify as ONE of: l1_transfer, developmental, orthographic
- For l1_transfer: name the specific L1 rule causing it
  (e.g. 'Portuguese uses definite articles before abstract
  nouns: a comunicação → the communication')
- For developmental: state what the learner is overgeneralising

If an error is just a typo with no linguistic explanation,
classify as orthographic and move on. Do not explain typos.

If you cannot confidently classify an error, SKIP IT.
It is better to return 2 real errors than 6 forced ones.

Return JSON array only:
[{{"span": "...", "correction": "...", "type": "...",
  "explanation": "..."}}]

If no clear errors, return [].
JSON only, no markdown."""


def _format_attributions(attributions: Optional[List[Dict[str, Any]]]) -> str:
    if not attributions:
        return "Not provided."
    lines: List[str] = []
    for row in attributions:
        head = str(row.get("head", "")).strip() or "?"
        sent = str(row.get("sentence", "")).strip()
        if len(sent) > 220:
            sent = sent[:217] + "…"
        score = float(row.get("toward_mass") or row.get("attribution") or 0)
        toks = row.get("tokens") or []
        if toks and isinstance(toks[0], dict):
            tok_str = ", ".join(str(t.get("text", "")) for t in toks[:6])
        else:
            tok_str = ", ".join(str(t) for t in toks[:6])
        cue = f" cues: {tok_str}" if tok_str else ""
        lines.append(f'- [{head}] "{sent}" (attribution {score:.3f}{cue})')
    return "\n".join(lines) if lines else "Not provided."


def _parse_json_array(raw: str) -> List[Dict[str, Any]]:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    if text.lower().startswith("no errors"):
        return []
    data = json.loads(text)
    if isinstance(data, dict) and "errors" in data:
        data = data["errors"]
    if not isinstance(data, list):
        return []
    allowed_types = {"l1_transfer", "developmental", "orthographic"}
    out: List[Dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        span = str(item.get("span", "")).strip()
        if not span:
            continue
        err_type = str(item.get("type", "")).strip().lower().replace(" ", "_")
        if err_type not in allowed_types:
            err_type = err_type.split("|")[0] if "|" in err_type else err_type
        if err_type not in allowed_types:
            continue
        explanation = str(item.get("explanation", "")).strip()
        if err_type == "orthographic":
            explanation = ""
        out.append(
            {
                "span": span,
                "correction": str(item.get("correction", "")).strip(),
                "type": err_type,
                "explanation": explanation,
            }
        )
    return out


def analyze_errors(
    text: str,
    cefr: str,
    l1: str,
    attributions: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """One LLM call over the full essay."""
    essay = text.strip()
    if not essay:
        return []

    if not llm_available():
        logger.info("LLM unavailable — skipping error analysis")
        return []

    prompt = PROMPT.format(
        l1=l1,
        cefr=cefr,
        essay=essay.replace("{", "{{").replace("}", "}}"),
        attributions=_format_attributions(attributions).replace("{", "{{").replace("}", "}}"),
    )
    raw = llm_complete(prompt, max_tokens=1024, temperature=0)
    if not raw:
        return []
    try:
        return _parse_json_array(raw)
    except json.JSONDecodeError:
        logger.warning("LLM returned non-JSON for essay errors: %s", raw[:300])
        return []
