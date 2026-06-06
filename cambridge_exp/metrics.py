"""Shared evaluation helpers for training and eval_cli."""

from __future__ import annotations

import torch
from torch.amp import autocast


def accuracy_from_loader(model, loader, device, head: str, amp_dtype, use_amp: bool) -> float:
    key = {"cefr": "cefr_label_id", "l1": "l1_label_id", "nat": "nat_label_id"}[head]
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            y = batch[key].long().to(device)
            with autocast(device.type, dtype=amp_dtype, enabled=use_amp):
                out = model(input_ids=input_ids, attention_mask=attention_mask)
            pred = out[head].argmax(-1)
            correct += (pred == y).sum().item()
            total += y.numel()
    return correct / max(total, 1)
