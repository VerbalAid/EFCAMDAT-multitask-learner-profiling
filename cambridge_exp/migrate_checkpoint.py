"""
Add missing metadata keys to a notebook-trained best_model.pt so API / shap_export / eval_cli work.

  PYTHONPATH=. python -m cambridge_exp.migrate_checkpoint checkpoints/multitask_model/best_model.pt

Writes alongside the original unless you pass --in-place (overwrites after backup optional).
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch

DEFAULT_CEFR = ["A1", "A2", "B1", "B2", "C1"]
DEFAULT_L1 = [
    "Arabic", "French", "German", "Italian", "Japanese", "Mandarin",
    "Portuguese", "Russian", "Spanish", "Turkish",
]
DEFAULT_NAT = ["br", "cn", "de", "fr", "it", "jp", "mx", "ru", "sa", "tr", "tw"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--in-place", action="store_true", help="Overwrite file (creates .bak first)")
    p.add_argument("--out", type=Path, default=None, help="Output path if not --in-place")
    return p.parse_args()


def main():
    args = parse_args()
    path = args.checkpoint.resolve()
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    ckpt.setdefault("cefr_classes", DEFAULT_CEFR)
    ckpt.setdefault("l1_classes", DEFAULT_L1)
    ckpt.setdefault("nat_classes", DEFAULT_NAT)
    ckpt.setdefault("dual_mode", "dual")
    ckpt.setdefault("max_length", 128)
    ckpt.setdefault("model_name", "roberta-base")
    ckpt.setdefault("balance_cefr", False)
    ckpt.setdefault("train_topic_family", None)
    ckpt.setdefault("exclude_place_heavy_topics", False)

    if args.in_place:
        bak = path.with_suffix(path.suffix + ".bak")
        shutil.copy2(path, bak)
        torch.save(ckpt, path)
        print(f"Updated {path} (backup {bak})")
    else:
        out = args.out or path.with_name(path.stem + "_migrated.pt")
        torch.save(ckpt, out)
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
