"""
Evaluate a saved multitask checkpoint on the held-out test split (same 80/20 seed as training).

  PYTHONPATH=. python -m cambridge_exp.eval_cli \\
    --checkpoint checkpoints/baseline_dual/best_model.pt \\
    --csv efcamdat_full_with_corrected.csv

Fair comparison (same essays for baseline vs balanced): export indices once, then pass to each run:

  PYTHONPATH=. python -m cambridge_exp.export_test_split --out splits/test_indices.json
  PYTHONPATH=. python -m cambridge_exp.eval_cli -c checkpoints/baseline/best_model.pt --test-indices splits/test_indices.json
  PYTHONPATH=. python -m cambridge_exp.eval_cli -c checkpoints/balanced_cefr/best_model.pt --test-indices splits/test_indices.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from torch.amp import autocast
from torch.utils.data import DataLoader

from cambridge_exp.config import TrainConfig
from cambridge_exp.data_pipeline import (
    attach_label_ids,
    build_datasets,
    build_eval_dataset,
    load_and_prepare_frame,
    make_tokenizer,
)
from cambridge_exp.metrics import accuracy_from_loader
from cambridge_exp.model import DEFAULT_HEADS, MultiTaskRoberta


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate multitask checkpoint on test split")
    p.add_argument("--checkpoint", "-c", type=Path, required=True)
    p.add_argument("--csv", type=Path, default=Path("efcamdat_full_with_corrected.csv"))
    p.add_argument(
        "--test-indices",
        type=Path,
        default=None,
        help="JSON from export_test_split.py — evaluate only those rows in the unbalanced prepared frame",
    )
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    use_cuda = torch.cuda.is_available() and not args.cpu
    device = torch.device("cuda" if use_cuda else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    dual_mode = ckpt.get("dual_mode", "dual")
    active_heads = tuple(ckpt.get("active_heads", DEFAULT_HEADS))
    max_length = int(ckpt.get("max_length", 128))
    model_name = ckpt.get("model_name", "roberta-base")

    split_meta = None
    if args.test_indices is not None:
        with open(args.test_indices, encoding="utf-8") as f:
            split_meta = json.load(f)
        if split_meta.get("version") != 1:
            raise ValueError("test-indices JSON must have version 1")
        if split_meta.get("dual_mode") != dual_mode:
            raise ValueError(
                f"split dual_mode={split_meta.get('dual_mode')!r} != checkpoint dual_mode={dual_mode!r}"
            )
        ckpt_topic = ckpt.get("train_topic_family")
        meta_topic = split_meta.get("train_topic_family")
        if ckpt_topic != meta_topic:
            raise ValueError(
                f"split train_topic_family={meta_topic!r} != checkpoint {ckpt_topic!r}"
            )
        if bool(split_meta.get("exclude_place_heavy_topics", False)) != bool(
            ckpt.get("exclude_place_heavy_topics", False)
        ):
            raise ValueError("split exclude_place_heavy_topics flag does not match checkpoint")
        split_csv = Path(split_meta["csv"]).resolve()
        if split_csv != args.csv.resolve():
            raise ValueError(
                f"--csv {args.csv.resolve()} does not match split file ({split_csv}); use the same CSV."
            )

    cfg = TrainConfig(
        csv_path=args.csv.resolve(),
        checkpoint_dir=args.checkpoint.parent,
        dual_mode=dual_mode,
        max_length=max_length,
        model_name=model_name,
        balance_cefr=False if args.test_indices else bool(ckpt.get("balance_cefr", False)),
        train_topic_family=ckpt.get("train_topic_family"),
        exclude_place_heavy_topics=bool(ckpt.get("exclude_place_heavy_topics", False)),
    )

    df = load_and_prepare_frame(cfg)
    if args.test_indices is not None and split_meta is not None:
        expected_n = split_meta.get("n_prepared_rows")
        if expected_n is not None and int(expected_n) != len(df):
            raise ValueError(
                f"Prepared frame has {len(df)} rows but split file expects {expected_n} "
                "(CSV path, filters, or code changed since export_test_split)."
            )

    cefr_enc = LabelEncoder()
    l1_enc = LabelEncoder()
    nat_enc = LabelEncoder()
    cefr_enc.fit(ckpt["cefr_classes"])
    l1_enc.fit(ckpt["l1_classes"])
    nat_enc.fit(ckpt["nat_classes"])
    df_l = attach_label_ids(df, cefr_enc, l1_enc, nat_enc)

    tokenizer = make_tokenizer(model_name)
    if args.test_indices is not None:
        assert split_meta is not None
        ix = split_meta["indices"]
        if max(ix, default=-1) >= len(df_l):
            raise ValueError("test-indices out of range for current prepared frame (check CSV / filters)")
        df_eval = df_l.iloc[ix].reset_index(drop=True)
        test_ds = build_eval_dataset(df_eval, tokenizer, cfg)
        print(f"Fair eval: {len(df_eval)} rows from --test-indices ({args.test_indices})")
    else:
        _, test_ds = build_datasets(df_l, tokenizer, cfg)

    num_c = len(cefr_enc.classes_)
    num_l = len(l1_enc.classes_)
    num_n = len(nat_enc.classes_)
    model = MultiTaskRoberta(model_name, num_c, num_l, num_n, heads=active_heads).to(device)
    model.encoder.resize_token_embeddings(len(tokenizer))
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    bf16_ok = use_cuda and torch.cuda.is_bf16_supported()
    fp16_ok = use_cuda and not bf16_ok
    amp_dtype = torch.bfloat16 if bf16_ok else torch.float16
    use_amp = bf16_ok or fp16_ok

    cols = ["input_ids", "attention_mask", "cefr_label_id", "l1_label_id", "nat_label_id"]
    loader = DataLoader(test_ds.select_columns(cols), batch_size=args.batch_size)

    for h in active_heads:
        acc = accuracy_from_loader(model, loader, device, h, amp_dtype, use_amp)
        print(f"{h.upper()} accuracy: {acc:.4f}")

    # Full classification reports (CPU, small tensors)
    head_preds: dict[str, tuple[list, list]] = {}
    label_keys = {"cefr": "cefr_label_id", "l1": "l1_label_id", "nat": "nat_label_id"}
    encoders = {"cefr": cefr_enc, "l1": l1_enc, "nat": nat_enc}
    with torch.no_grad():
        for batch in loader:
            enc = {k: batch[k].to(device) for k in ("input_ids", "attention_mask")}
            with autocast(device.type, dtype=amp_dtype, enabled=use_amp):
                out = model(**enc)
            for h in active_heads:
                if h not in head_preds:
                    head_preds[h] = ([], [])
                y_true, y_pred = head_preds[h]
                y_true.extend(batch[label_keys[h]].tolist())
                y_pred.extend(out[h].argmax(-1).cpu().tolist())

    for h in active_heads:
        y_true, y_pred = head_preds[h]
        print(f"\n--- {h.upper()} report ---")
        print(classification_report(y_true, y_pred, target_names=list(encoders[h].classes_), zero_division=0))


if __name__ == "__main__":
    main()
