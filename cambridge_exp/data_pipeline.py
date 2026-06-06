from __future__ import annotations

from typing import Mapping, Optional, Tuple, Union

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer

from cambridge_exp.config import DualMode, TrainConfig
from cambridge_exp.topics import combined_topic_mask


def build_model_input_text(row: Union[pd.Series, Mapping[str, str]], dual_mode: DualMode) -> str:
    if dual_mode == "dual":
        return "[RAW] " + str(row["text"]) + " [CORRECTED] " + str(row["text_corrected"])
    if dual_mode == "raw_only":
        return str(row["text"])
    raise ValueError(dual_mode)


def balance_cefr_undersample(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Per-CEFR random undersample to the smallest band count."""
    sizes = df.groupby("cefr").size()
    target = int(sizes.min())
    parts = []
    for c in df["cefr"].unique():
        sub = df[df["cefr"] == c]
        n = min(len(sub), target)
        parts.append(sub.sample(n=n, random_state=seed))
    out = pd.concat(parts, axis=0)
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def load_and_prepare_frame(cfg: TrainConfig) -> pd.DataFrame:
    df = pd.read_csv(cfg.csv_path)
    required = {"text", "text_corrected", "cefr", "l1", "nationality", "topic"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {sorted(missing)}")

    mask = combined_topic_mask(
        df,
        family_key=cfg.train_topic_family,
        exclude_place_heavy=cfg.exclude_place_heavy_topics,
    )
    df = df.loc[mask].copy()
    if len(df) == 0:
        raise RuntimeError("No rows left after topic filters; relax train_topic_family / exclude_place_heavy.")

    df["model_input"] = [build_model_input_text(df.loc[i], cfg.dual_mode) for i in df.index]

    if cfg.balance_cefr:
        df = balance_cefr_undersample(df, cfg.seed)

    return df.reset_index(drop=True)


def fit_encoders(df: pd.DataFrame) -> Tuple[LabelEncoder, LabelEncoder, LabelEncoder]:
    cefr_enc = LabelEncoder()
    l1_enc = LabelEncoder()
    nat_enc = LabelEncoder()
    cefr_enc.fit(df["cefr"])
    l1_enc.fit(df["l1"])
    nat_enc.fit(df["nationality"])
    return cefr_enc, l1_enc, nat_enc


def attach_label_ids(df: pd.DataFrame, cefr_enc: LabelEncoder, l1_enc: LabelEncoder, nat_enc: LabelEncoder) -> pd.DataFrame:
    out = df.copy()
    out["cefr_label_id"] = cefr_enc.transform(out["cefr"])
    out["l1_label_id"] = l1_enc.transform(out["l1"])
    out["nat_label_id"] = nat_enc.transform(out["nationality"])
    return out


def make_tokenizer(model_name: str) -> AutoTokenizer:
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.add_special_tokens({"additional_special_tokens": ["[RAW]", "[CORRECTED]"]})
    return tok


def split_train_test_frames(
    df: pd.DataFrame,
    cfg: TrainConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified 80/20 split by CEFR (same logic as training)."""
    idx = np.arange(len(df))
    tr, te = train_test_split(
        idx,
        test_size=cfg.test_size,
        random_state=cfg.seed,
        stratify=df["cefr_label_id"],
    )
    return df.iloc[tr].reset_index(drop=True), df.iloc[te].reset_index(drop=True)


def build_datasets(
    df: pd.DataFrame,
    tokenizer: AutoTokenizer,
    cfg: TrainConfig,
) -> Tuple[Dataset, Dataset]:
    """`df` must already contain cefr_label_id, l1_label_id, nat_label_id."""

    train_df, test_df = split_train_test_frames(df, cfg)

    keep = ["model_input", "text", "text_corrected", "topic", "cefr", "l1", "nationality",
            "cefr_label_id", "l1_label_id", "nat_label_id"]

    def tokenize(batch):
        return tokenizer(
            batch["model_input"],
            truncation=True,
            padding="max_length",
            max_length=cfg.max_length,
        )

    train_ds = Dataset.from_pandas(train_df[keep])
    test_ds = Dataset.from_pandas(test_df[keep])
    train_ds = train_ds.map(tokenize, batched=True)
    test_ds = test_ds.map(tokenize, batched=True)
    train_ds = train_ds.rename_column("model_input", "dual_text")
    test_ds = test_ds.rename_column("model_input", "dual_text")
    train_ds.set_format("torch")
    test_ds.set_format("torch")

    return train_ds, test_ds


def build_eval_dataset(
    df: pd.DataFrame,
    tokenizer: AutoTokenizer,
    cfg: TrainConfig,
) -> Dataset:
    """Tokenize every row of `df` for evaluation (no train/test split)."""
    keep = [
        "model_input",
        "text",
        "text_corrected",
        "topic",
        "cefr",
        "l1",
        "nationality",
        "cefr_label_id",
        "l1_label_id",
        "nat_label_id",
    ]

    def tokenize(batch):
        return tokenizer(
            batch["model_input"],
            truncation=True,
            padding="max_length",
            max_length=cfg.max_length,
        )

    ds = Dataset.from_pandas(df[keep])
    ds = ds.map(tokenize, batched=True)
    ds = ds.rename_column("model_input", "dual_text")
    ds.set_format("torch")
    return ds
