from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import pandas as pd

_PACKAGE_DIR = Path(__file__).resolve().parent
FAMILIES_PATH = _PACKAGE_DIR / "topic_families.json"

# Optional heuristic: drop prompts that often elicit heavy geo/culture content.
# Keep this tight so curated topic families are not accidentally emptied.
_PLACE_HINT_SUBSTRINGS = (
    "weather guide for your city",
    "holiday postcard",
    "safari",
    "labeling photos from a safari",
)


def load_families(path: Path = FAMILIES_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def family_topic_strings(family_key: str, path: Path = FAMILIES_PATH) -> List[str]:
    data = load_families(path)
    if family_key not in data or family_key.startswith("_"):
        raise KeyError(f"Unknown topic family {family_key!r}. Keys: {sorted(k for k in data if not k.startswith('_'))}")
    block = data[family_key]
    m = block.get("topics_by_band") or {}
    return [str(v) for v in m.values()]


def mask_place_heavy_topics(series: pd.Series) -> pd.Series:
    s = series.fillna("").astype(str).str.lower()
    bad = False
    for sub in _PLACE_HINT_SUBSTRINGS:
        bad = bad | s.str.contains(sub, regex=False)
    return ~bad


def filter_by_family(df: pd.DataFrame, family_key: str, path: Path = FAMILIES_PATH) -> pd.DataFrame:
    topics = family_topic_strings(family_key, path)
    return df[df["topic"].astype(str).isin(topics)].copy()


def filter_exclude_place_heavy(df: pd.DataFrame) -> pd.DataFrame:
    return df[mask_place_heavy_topics(df["topic"])].copy()


def combined_topic_mask(
    df: pd.DataFrame,
    family_key: Optional[str] = None,
    exclude_place_heavy: bool = False,
    path: Path = FAMILIES_PATH,
) -> pd.Series:
    m = pd.Series(True, index=df.index)
    if family_key:
        topics = frozenset(family_topic_strings(family_key, path))
        m &= df["topic"].astype(str).isin(topics)
    if exclude_place_heavy:
        m &= mask_place_heavy_topics(df["topic"])
    return m
