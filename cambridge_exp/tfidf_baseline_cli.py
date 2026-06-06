"""
TF-IDF + logistic regression baseline on the same train/test split as RoBERTa training.

  PYTHONPATH=. python -m cambridge_exp.tfidf_baseline_cli \\
    --csv efcamdat_full_with_corrected.csv

Fair eval (same essays as a saved RoBERTa checkpoint):

  PYTHONPATH=. python -m cambridge_exp.tfidf_baseline_cli \\
    --test-indices splits/test_indices.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from cambridge_exp.config import TrainConfig
from cambridge_exp.data_pipeline import (
    attach_label_ids,
    fit_encoders,
    load_and_prepare_frame,
    split_train_test_frames,
)

TextSource = str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TF-IDF + LR baseline (CEFR / L1 / nationality)")
    p.add_argument("--csv", type=Path, default=Path("efcamdat_full_with_corrected.csv"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument(
        "--text-source",
        choices=("raw", "corrected", "dual"),
        default="raw",
        help="raw=learner text; corrected=teacher text; dual=concatenate both",
    )
    p.add_argument(
        "--test-indices",
        type=Path,
        default=None,
        help="JSON from export_test_split.py — train on all other rows, eval on these",
    )
    p.add_argument("--train-topic-family", type=str, default=None)
    p.add_argument("--exclude-place-heavy-topics", action="store_true")
    p.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Optional path to write accuracy / macro-F1 summary JSON",
    )
    return p.parse_args()


def baseline_text(row, text_source: TextSource) -> str:
    raw = str(row["text"])
    corrected = str(row["text_corrected"])
    if text_source == "raw":
        return raw
    if text_source == "corrected":
        return corrected
    return raw + " " + corrected


def make_tfidf_pipeline() -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    min_df=5,
                    max_df=0.95,
                    sublinear_tf=True,
                    max_features=50_000,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=500,
                    class_weight="balanced",
                    solver="saga",
                    tol=1e-3,
                ),
            ),
        ]
    )


def load_test_indices(path: Path, csv: Path) -> tuple[list[int], dict]:
    with open(path, encoding="utf-8") as f:
        meta = json.load(f)
    if meta.get("version") != 1:
        raise ValueError("test-indices JSON must have version 1")
    split_csv = Path(meta["csv"]).resolve()
    if split_csv != csv.resolve():
        raise ValueError(f"--csv {csv.resolve()} does not match split file ({split_csv})")
    return [int(i) for i in meta["indices"]], meta


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    cfg = TrainConfig(
        csv_path=args.csv.resolve(),
        checkpoint_dir=Path("."),
        seed=args.seed,
        test_size=args.test_size,
        dual_mode="raw_only",
        balance_cefr=False,
        train_topic_family=args.train_topic_family,
        exclude_place_heavy_topics=args.exclude_place_heavy_topics,
    )

    df = load_and_prepare_frame(cfg)
    cefr_enc, l1_enc, nat_enc = fit_encoders(df)
    df_l = attach_label_ids(df, cefr_enc, l1_enc, nat_enc)
    df_l["baseline_text"] = [baseline_text(df_l.loc[i], args.text_source) for i in df_l.index]

    split_meta = None
    if args.test_indices is not None:
        test_ix, split_meta = load_test_indices(args.test_indices, cfg.csv_path)
        if split_meta.get("n_prepared_rows") is not None and int(split_meta["n_prepared_rows"]) != len(df_l):
            raise ValueError(
                f"Prepared frame has {len(df_l)} rows but split file expects {split_meta['n_prepared_rows']}"
            )
        if max(test_ix, default=-1) >= len(df_l):
            raise ValueError("test-indices out of range for current prepared frame")
        test_df = df_l.iloc[test_ix].reset_index(drop=True)
        train_df = df_l.drop(index=test_ix).reset_index(drop=True)
        print(f"Fair eval: {len(test_df)} test rows from {args.test_indices}")
    else:
        train_df, test_df = split_train_test_frames(df_l, cfg)

    x_train = train_df["baseline_text"].tolist()
    x_test = test_df["baseline_text"].tolist()

    tasks = (
        ("cefr", "cefr_label_id", cefr_enc),
        ("l1", "l1_label_id", l1_enc),
        ("nat", "nat_label_id", nat_enc),
    )

    summary: dict = {
        "text_source": args.text_source,
        "n_train": len(train_df),
        "n_test": len(test_df),
        "heads": {},
    }

    print(f"TF-IDF + LR baseline  |  text={args.text_source}  |  train={len(train_df)}  test={len(test_df)}")

    for head, label_col, enc in tasks:
        print(f"Training {head.upper()}...", flush=True)
        pipe = make_tfidf_pipeline()
        y_train = train_df[label_col].to_numpy()
        y_test = test_df[label_col].to_numpy()
        pipe.fit(x_train, y_train)
        y_pred = pipe.predict(x_test)

        acc = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        print(f"\n{head.upper()} accuracy: {acc:.4f}  |  macro-F1: {macro_f1:.4f}")
        print(classification_report(y_test, y_pred, target_names=list(enc.classes_), zero_division=0))

        summary["heads"][head] = {
            "accuracy": float(acc),
            "macro_f1": float(macro_f1),
            "classes": list(enc.classes_),
        }

    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"\nWrote summary to {args.out_json.resolve()}")


if __name__ == "__main__":
    main()
