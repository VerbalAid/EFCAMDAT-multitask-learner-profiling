"""Aggregate filled-in annotation_label counts from shap_export CSV."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ALLOWED = frozenset({"transfer_error", "named_entity", "register_marker", "noise", ""})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("csv", type=Path)
    p.add_argument("--head", type=str, default=None, help="Filter to cefr | l1 | nat")
    args = p.parse_args()
    df = pd.read_csv(args.csv)
    if args.head:
        df = df[df["head"] == args.head]
    col = df["annotation_label"].fillna("").astype(str).str.strip()
    unk = set(col.unique()) - ALLOWED
    if unk - {""}:
        print("Warning: non-standard labels present:", sorted(unk - {""}))
    filled = col[col != ""]
    print(f"Annotated rows: {len(filled)} / {len(df)}")
    if len(filled) == 0:
        return
    vc = filled.value_counts()
    print(vc.to_string())
    print("\nPercent of annotated:")
    for k, v in vc.items():
        print(f"  {k}: {100 * v / len(filled):.1f}%")


if __name__ == "__main__":
    main()
