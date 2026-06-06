"""
Export stratified test indices from the *unbalanced* prepared frame (same logic as training).

Use this so baseline and balanced checkpoints can be evaluated on the **same** essays.

  PYTHONPATH=. python -m cambridge_exp.export_test_split \\
      --csv efcamdat_full_with_corrected.csv \\
      --out splits/full_dual_test_indices.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from cambridge_exp.config import TrainConfig
from cambridge_exp.data_pipeline import attach_label_ids, fit_encoders, load_and_prepare_frame


def parse_args():
    p = argparse.ArgumentParser(description="Export test row indices for fair cross-checkpoint eval")
    p.add_argument("--csv", type=Path, default=Path("efcamdat_full_with_corrected.csv"))
    p.add_argument("--out", type=Path, required=True, help="JSON path to write")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--dual-mode", choices=("dual", "raw_only"), default="dual")
    p.add_argument("--train-topic-family", type=str, default=None)
    p.add_argument("--exclude-place-heavy-topics", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = TrainConfig(
        csv_path=args.csv.resolve(),
        checkpoint_dir=args.out.parent,
        seed=args.seed,
        test_size=args.test_size,
        dual_mode=args.dual_mode,
        balance_cefr=False,
        train_topic_family=args.train_topic_family,
        exclude_place_heavy_topics=args.exclude_place_heavy_topics,
    )
    df = load_and_prepare_frame(cfg)
    cefr_enc, l1_enc, nat_enc = fit_encoders(df)
    df_l = attach_label_ids(df, cefr_enc, l1_enc, nat_enc)

    idx = np.arange(len(df_l))
    _, te = train_test_split(
        idx,
        test_size=cfg.test_size,
        random_state=cfg.seed,
        stratify=df_l["cefr_label_id"],
    )
    te_list = [int(x) for x in sorted(te.tolist())]

    payload = {
        "version": 1,
        "indices": te_list,
        "n_prepared_rows": len(df_l),
        "csv": str(cfg.csv_path.resolve()),
        "dual_mode": cfg.dual_mode,
        "balance_cefr": False,
        "seed": cfg.seed,
        "test_size": cfg.test_size,
        "train_topic_family": cfg.train_topic_family,
        "exclude_place_heavy_topics": cfg.exclude_place_heavy_topics,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {len(te_list)} test indices to {args.out.resolve()}")


if __name__ == "__main__":
    main()
