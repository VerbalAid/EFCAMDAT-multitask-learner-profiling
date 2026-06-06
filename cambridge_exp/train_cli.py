"""
Train multitask RoBERTa from the CLI.

Examples:
  python -m cambridge_exp.train_cli --csv efcamdat_full_with_corrected.csv \\
      --checkpoint-dir checkpoints/baseline_dual

  python -m cambridge_exp.train_cli --dual-mode raw_only --balance-cefr \\
      --checkpoint-dir checkpoints/ablation_raw_only_balanced

  python -m cambridge_exp.train_cli --epochs 5 --early-stop-patience 3 \\
      --checkpoint-dir checkpoints/ep5_earlystop

  python -m cambridge_exp.train_cli --heads l1 \\
      --checkpoint-dir checkpoints/single_task_l1
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from sklearn.utils import class_weight
from torch import nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from cambridge_exp.config import TrainConfig
from cambridge_exp.data_pipeline import (
    attach_label_ids,
    build_datasets,
    fit_encoders,
    load_and_prepare_frame,
    make_tokenizer,
)
from cambridge_exp.metrics import accuracy_from_loader
from cambridge_exp.model import DEFAULT_HEADS, MultiTaskRoberta


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train EF-CamDAT multitask model")
    p.add_argument("--csv", type=Path, default=Path("efcamdat_full_with_corrected.csv"))
    p.add_argument("--checkpoint-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Per-step batch size (default 16 for ~8GB GPUs; increase if you have headroom)",
    )
    p.add_argument(
        "--eval-batch-size",
        type=int,
        default=32,
        help="Validation batch size (lower if CUDA OOM during val)",
    )
    p.add_argument(
        "--grad-accum-steps",
        type=int,
        default=1,
        help="Gradient accumulation steps (effective batch ≈ batch-size × this). Use 4 with batch 16 to mimic 64.",
    )
    p.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU training (slow; avoids CUDA OOM)",
    )
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--early-stop-patience", type=int, default=3)
    p.add_argument("--dual-mode", choices=("dual", "raw_only"), default="dual")
    p.add_argument("--balance-cefr", action="store_true")
    p.add_argument("--train-topic-family", type=str, default=None,
                   help="Restrict training rows to a key from topic_families.json")
    p.add_argument("--exclude-place-heavy-topics", action="store_true")
    p.add_argument("--model-name", type=str, default="roberta-base")
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument(
        "--heads",
        type=str,
        default="cefr,l1,nat",
        help="Comma-separated task heads to train (e.g. cefr for single-task CEFR-only)",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Load best_model.pt from --checkpoint-dir and continue (optimizer + scheduler restored if present)",
    )
    return p.parse_args()


def parse_heads(raw: str) -> tuple[str, ...]:
    heads = tuple(h.strip() for h in raw.split(",") if h.strip())
    if not heads:
        raise ValueError("--heads must list at least one of: cefr, l1, nat")
    unknown = set(heads) - set(DEFAULT_HEADS)
    if unknown:
        raise ValueError(f"Unknown --heads values: {sorted(unknown)}")
    return heads


def main() -> None:
    args = parse_args()
    active_heads = parse_heads(args.heads)
    cfg = TrainConfig(
        csv_path=args.csv.resolve(),
        checkpoint_dir=args.checkpoint_dir.resolve(),
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        grad_accum_steps=max(1, args.grad_accum_steps),
        lr=args.lr,
        early_stop_patience=args.early_stop_patience,
        dual_mode=args.dual_mode,
        balance_cefr=args.balance_cefr,
        train_topic_family=args.train_topic_family,
        exclude_place_heavy_topics=args.exclude_place_heavy_topics,
        model_name=args.model_name,
        max_length=args.max_length,
    )

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    df = load_and_prepare_frame(cfg)
    cefr_enc, l1_enc, nat_enc = fit_encoders(df)
    df_l = attach_label_ids(df, cefr_enc, l1_enc, nat_enc)
    tokenizer = make_tokenizer(cfg.model_name)
    train_ds, test_ds = build_datasets(df_l, tokenizer, cfg)

    use_cuda = torch.cuda.is_available() and not args.cpu
    device = torch.device("cuda" if use_cuda else "cpu")
    if torch.cuda.is_available() and not args.cpu:
        try:
            free_b, total_b = torch.cuda.mem_get_info()
            print(
                f"GPU mem: {free_b / 1e9:.2f} GiB free / {total_b / 1e9:.2f} GiB total "
                "(close other GPU jobs if OOM)"
            )
        except Exception:
            pass
    elif torch.cuda.is_available() and args.cpu:
        print("Using CPU (--cpu); CUDA is available but ignored.")

    bf16_ok = use_cuda and torch.cuda.is_bf16_supported()
    fp16_ok = use_cuda and not bf16_ok
    amp_dtype = torch.bfloat16 if bf16_ok else torch.float16
    use_amp = bf16_ok or fp16_ok
    scaler = GradScaler(device.type, enabled=fp16_ok)

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    best_path = cfg.checkpoint_dir / "best_model.pt"
    meta_path = cfg.checkpoint_dir / "experiment.json"

    num_cefr = len(cefr_enc.classes_)
    num_l1 = len(l1_enc.classes_)
    num_nat = len(nat_enc.classes_)

    model = MultiTaskRoberta(
        cfg.model_name, num_cefr, num_l1, num_nat, heads=active_heads
    ).to(device)
    model.encoder.resize_token_embeddings(len(tokenizer))
    # foreach=False lowers peak VRAM during optimizer.step() (multi-tensor Adam uses extra buffers).
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, foreach=False)

    # Class weights from training split (dataframe not in scope — recompute from train_ds is awkward).
    # Use full prepared df label distribution (matches notebook which used full df).
    cefr_weights = class_weight.compute_class_weight(
        class_weight="balanced", classes=np.unique(df_l["cefr_label_id"]), y=df_l["cefr_label_id"]
    )
    l1_weights = class_weight.compute_class_weight(
        class_weight="balanced", classes=np.unique(df_l["l1_label_id"]), y=df_l["l1_label_id"]
    )
    nat_weights = class_weight.compute_class_weight(
        class_weight="balanced", classes=np.unique(df_l["nat_label_id"]), y=df_l["nat_label_id"]
    )

    cefr_w = torch.tensor(cefr_weights, dtype=torch.float32).to(device)
    l1_w = torch.tensor(l1_weights, dtype=torch.float32).to(device)
    nat_w = torch.tensor(nat_weights, dtype=torch.float32).to(device)

    cefr_loss_fn = nn.CrossEntropyLoss(weight=cefr_w) if "cefr" in active_heads else None
    l1_loss_fn = nn.CrossEntropyLoss(weight=l1_w) if "l1" in active_heads else None
    nat_loss_fn = nn.CrossEntropyLoss(weight=nat_w) if "nat" in active_heads else None

    def task_loss(outputs, cefr_y, l1_y, nat_y):
        parts = []
        if cefr_loss_fn is not None:
            parts.append(cefr_loss_fn(outputs["cefr"], cefr_y))
        if l1_loss_fn is not None:
            parts.append(l1_loss_fn(outputs["l1"], l1_y))
        if nat_loss_fn is not None:
            parts.append(nat_loss_fn(outputs["nat"], nat_y))
        return sum(parts)

    train_cols = ["input_ids", "attention_mask", "cefr_label_id", "l1_label_id", "nat_label_id"]
    accum = max(1, cfg.grad_accum_steps)
    train_loader = DataLoader(
        train_ds.select_columns(train_cols),
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=(accum > 1),
    )
    val_loader = DataLoader(
        test_ds.select_columns(train_cols),
        batch_size=cfg.eval_batch_size,
    )

    updates_per_epoch = math.ceil(len(train_loader) / accum)
    total_steps = updates_per_epoch * cfg.epochs
    warmup_steps = math.ceil(total_steps * cfg.warmup_pct)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    best_val = float("inf")
    stalled = 0
    start_epoch = 0

    if args.resume:
        if not best_path.is_file():
            raise FileNotFoundError(f"--resume needs an existing checkpoint: {best_path}")
        ck = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state_dict"])
        if "optimizer_state_dict" in ck:
            optimizer.load_state_dict(ck["optimizer_state_dict"])
        if "scheduler_state_dict" in ck:
            scheduler.load_state_dict(ck["scheduler_state_dict"])
        # Saved `epoch` is 1-based count of completed epochs (see torch.save below).
        start_epoch = int(ck.get("epoch", 0))
        best_val = float(ck.get("val_loss", float("inf")))
        stalled = 0
        print(
            f"Resume: {best_path.name} — completed {start_epoch} epoch(s), "
            f"best val loss {best_val:.4f}; continuing from epoch {start_epoch + 1}/{cfg.epochs}"
        )

    if start_epoch >= cfg.epochs:
        print(f"Checkpoint already at epoch {start_epoch}; nothing left for --epochs {cfg.epochs}.")
        return

    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        train_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        for step, batch in tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Epoch {epoch + 1}/{cfg.epochs} [train]",
        ):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            cefr_y = batch["cefr_label_id"].long().to(device)
            l1_y = batch["l1_label_id"].long().to(device)
            nat_y = batch["nat_label_id"].long().to(device)

            with autocast(device.type, dtype=amp_dtype, enabled=use_amp):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = task_loss(outputs, cefr_y, l1_y, nat_y) / accum

            if fp16_ok:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            train_loss += loss.item() * accum

            stepped = (step + 1) % accum == 0 or (step + 1) == len(train_loader)
            if stepped:
                if (step + 1) == len(train_loader) and (step + 1) % accum != 0 and not fp16_ok:
                    rem = (step + 1) % accum
                    factor = accum / rem
                    for p in model.parameters():
                        if p.grad is not None:
                            p.grad.mul_(factor)
                if fp16_ok:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

        model.eval()
        val_loss = 0.0
        if use_cuda:
            torch.cuda.empty_cache()
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{cfg.epochs} [val]  "):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                cefr_y = batch["cefr_label_id"].long().to(device)
                l1_y = batch["l1_label_id"].long().to(device)
                nat_y = batch["nat_label_id"].long().to(device)
                with autocast(device.type, dtype=amp_dtype, enabled=use_amp):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = task_loss(outputs, cefr_y, l1_y, nat_y)
                val_loss += loss.item()

        avg_train = train_loss / max(len(train_loader), 1)
        avg_val = val_loss / len(val_loader)
        print(f"Epoch {epoch + 1}/{cfg.epochs}  |  train {avg_train:.4f}  |  val {avg_val:.4f}")

        if avg_val < best_val:
            best_val = avg_val
            stalled = 0
            payload = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "train_loss": avg_train,
                "val_loss": avg_val,
                "cefr_classes": list(cefr_enc.classes_),
                "l1_classes": list(l1_enc.classes_),
                "nat_classes": list(nat_enc.classes_),
                "dual_mode": cfg.dual_mode,
                "balance_cefr": cfg.balance_cefr,
                "train_topic_family": cfg.train_topic_family,
                "exclude_place_heavy_topics": cfg.exclude_place_heavy_topics,
                "tokenizer_extra": ["[RAW]", "[CORRECTED]"],
                "max_length": cfg.max_length,
                "model_name": cfg.model_name,
                "active_heads": list(active_heads),
                "batch_size": cfg.batch_size,
                "grad_accum_steps": accum,
                "eval_batch_size": cfg.eval_batch_size,
            }
            torch.save(payload, best_path)
            print(f"  saved {best_path}")
        else:
            stalled += 1
            print(f"  no val improvement ({stalled}/{cfg.early_stop_patience})")
            if stalled >= cfg.early_stop_patience:
                print("  early stopping.")
                break

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "csv": str(cfg.csv_path),
                "dual_mode": cfg.dual_mode,
                "balance_cefr": cfg.balance_cefr,
                "train_topic_family": cfg.train_topic_family,
                "exclude_place_heavy": cfg.exclude_place_heavy_topics,
                "active_heads": list(active_heads),
                "epochs_ran": epoch + 1,
                "best_val_loss": best_val,
                "checkpoint": str(best_path),
                "batch_size": cfg.batch_size,
                "grad_accum_steps": accum,
                "eval_batch_size": cfg.eval_batch_size,
            },
            f,
            indent=2,
        )

    # Quick accuracies on val
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        for h in active_heads:
            acc = accuracy_from_loader(model, val_loader, device, h, amp_dtype, use_amp)
            print(f"  val acc {h}: {acc:.4f}")


if __name__ == "__main__":
    main()
