"""
Export top SHAP tokens for ~N essays into a CSV for the four-way annotation task.

Example:
  python -m cambridge_exp.shap_export \\
    --checkpoint checkpoints/baseline_dual/best_model.pt \\
    --csv efcamdat_full_with_corrected.csv \\
    --n 100 --seed 42 \\
    --topic-family workplace_across_levels \\
    --split test \\
    --output annotation/shap_top_tokens.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from cambridge_exp.data_pipeline import attach_label_ids, build_model_input_text, make_tokenizer
from cambridge_exp.model import MultiTaskRoberta
from cambridge_exp.topics import combined_topic_mask


def raw_corr_token_indices(tokenizer, dual_text: str, max_length: int):
    sep = " [CORRECTED] "
    j = dual_text.find(sep)
    if j < 0:
        return None, None
    sep_start, sep_end = j, j + len(sep)
    batch = tokenizer(
        [dual_text],
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
        add_special_tokens=True,
    )
    pairs = batch["offset_mapping"][0]
    raw_i, corr_i = [], []
    for ti, span in enumerate(pairs):
        a, b = span[0], span[1]
        if a is None or b is None:
            continue
        a, b = int(a), int(b)
        if a == 0 and b == 0:
            continue
        if b <= sep_start:
            raw_i.append(ti)
        elif a >= sep_end:
            corr_i.append(ti)
    return raw_i, corr_i


def shap_vec_for_class(shap_row, class_i: int, n_tokens: int) -> np.ndarray:
    V = np.asarray(shap_row.values, dtype=float)
    if V.ndim == 3:
        V = V[0]
    if V.ndim == 2:
        vec = V[:, class_i] if V.shape[1] > 1 else V[:, 0]
    else:
        vec = V.ravel()
    vec = vec[:n_tokens]
    if len(vec) < n_tokens:
        vec = np.pad(vec, (0, n_tokens - len(vec)))
    return vec


def logits_head(model, device, tokenizer, texts, head_key: str, max_length: int) -> np.ndarray:
    enc = tokenizer(
        list(texts),
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        return model(**enc)[head_key].cpu().numpy()


def side_for_index(i: int, raw_i, corr_i) -> str:
    if raw_i is None or not corr_i:
        return "text"
    if i in raw_i:
        return "raw"
    if i in corr_i:
        return "corrected"
    return "marker_or_pad"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--csv", type=Path, default=Path("efcamdat_full_with_corrected.csv"))
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split", choices=("test", "train", "all"), default="test")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--topic-family", type=str, default=None)
    p.add_argument("--exclude-place-heavy-topics", action="store_true")
    p.add_argument("--dual-mode", choices=("dual", "raw_only"), default=None,
                   help="If set, overrides value stored in checkpoint")
    p.add_argument("--top-k", type=int, default=15)
    p.add_argument("--output", type=Path, default=Path("annotation/shap_top_tokens.csv"))
    p.add_argument("--heads", type=str, default="l1", help="Comma-separated: cefr,l1,nat")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    dual_mode = args.dual_mode or ckpt.get("dual_mode", "dual")
    max_length = int(ckpt.get("max_length", 128))
    model_name = ckpt.get("model_name", "roberta-base")

    cefr_enc = LabelEncoder()
    l1_enc = LabelEncoder()
    nat_enc = LabelEncoder()
    cefr_enc.fit(ckpt["cefr_classes"])
    l1_enc.fit(ckpt["l1_classes"])
    nat_enc.fit(ckpt["nat_classes"])

    tokenizer = make_tokenizer(model_name)
    num_cefr, num_l1, num_nat = len(cefr_enc.classes_), len(l1_enc.classes_), len(nat_enc.classes_)
    model = MultiTaskRoberta(model_name, num_cefr, num_l1, num_nat).to(device)
    model.encoder.resize_token_embeddings(len(tokenizer))
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    df = pd.read_csv(args.csv)
    mask = combined_topic_mask(
        df,
        family_key=args.topic_family,
        exclude_place_heavy=args.exclude_place_heavy_topics,
    )
    df = df.loc[mask].reset_index(drop=True)
    if len(df) == 0:
        raise SystemExit("No rows after topic filters.")

    df["model_input"] = [build_model_input_text(df.loc[i], dual_mode) for i in df.index]
    df_l = attach_label_ids(df, cefr_enc, l1_enc, nat_enc)

    idx = np.arange(len(df_l))
    tr, te = train_test_split(idx, test_size=args.test_size, random_state=args.seed, stratify=df_l["cefr_label_id"])
    if args.split == "test":
        pick = te
    elif args.split == "train":
        pick = tr
    else:
        pick = idx

    sub = df_l.iloc[pick].copy()
    sub["_df_ix"] = sub.index.to_numpy()
    n_take = min(args.n, len(sub))
    sub = sub.sample(n=n_take, random_state=args.seed).reset_index(drop=True)

    head_list = tuple(h.strip() for h in args.heads.split(",") if h.strip())
    explainers = {}
    for h in head_list:
        hk = h

        def make_predictor(key=hk):
            return lambda texts: logits_head(model, device, tokenizer, texts, key, max_length)

        explainers[h] = shap.Explainer(make_predictor(), tokenizer)

    rows_out = []
    for row_i, (_, row) in enumerate(tqdm(sub.iterrows(), total=len(sub), desc="SHAP")):
        dual = row["model_input"]
        enc = tokenizer(
            dual,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        toks = tokenizer.convert_ids_to_tokens(enc["input_ids"][0].tolist())
        raw_i, corr_i = raw_corr_token_indices(tokenizer, dual, max_length)

        batch = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**batch)

        essay_key = f"E{row_i:04d}"
        for h in head_list:
            pred_i = int(out[h].argmax(-1).item())
            classes = {"cefr": cefr_enc, "l1": l1_enc, "nat": nat_enc}[h].classes_
            pred_lbl = classes[pred_i]
            shap_row = explainers[h]([dual])[0]
            svals = shap_vec_for_class(shap_row, pred_i, len(toks))

            scored = []
            for ti, w in enumerate(toks):
                if ti >= len(svals):
                    break
                lab = w.replace("Ġ", " ").strip()
                if lab in ("<s>", "</s>", "<pad>", "", "Ċ"):
                    continue
                if "[CORRECTED]" in lab or lab == "[RAW]" or lab.startswith("[RAW]"):
                    continue
                side = side_for_index(ti, raw_i, corr_i)
                if side == "marker_or_pad":
                    continue
                scored.append((ti, lab, float(svals[ti]), side))

            scored.sort(key=lambda x: abs(x[2]), reverse=True)
            for rank, (ti, lab, sv, side) in enumerate(scored[: args.top_k], start=1):
                rows_out.append(
                    {
                        "essay_key": essay_key,
                        "dataframe_index": int(row["_df_ix"]),
                        "topic": row["topic"],
                        "cefr": row["cefr"],
                        "l1": row["l1"],
                        "nationality": row["nationality"],
                        "head": h,
                        "pred_label": pred_lbl,
                        "token_rank": rank,
                        "token_text": lab,
                        "shap_value": sv,
                        "side": side,
                        "annotation_label": "",
                        "annotator_notes": "",
                    }
                )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows_out).to_csv(args.output, index=False)
    print(f"Wrote {len(rows_out)} rows to {args.output}")


if __name__ == "__main__":
    main()
