from __future__ import annotations

from typing import Sequence

import torch
from torch import nn
from transformers import AutoModel

DEFAULT_HEADS: tuple[str, ...] = ("cefr", "l1", "nat")


class MultiTaskRoberta(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_cefr: int,
        num_l1: int,
        num_nat: int,
        heads: Sequence[str] | None = None,
    ):
        super().__init__()
        self.active_heads = tuple(heads or DEFAULT_HEADS)
        unknown = set(self.active_heads) - set(DEFAULT_HEADS)
        if unknown:
            raise ValueError(f"Unknown heads: {sorted(unknown)}")

        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        if "cefr" in self.active_heads:
            self.cefr_head = nn.Linear(hidden, num_cefr)
        if "l1" in self.active_heads:
            self.l1_head = nn.Linear(hidden, num_l1)
        if "nat" in self.active_heads:
            self.nat_head = nn.Linear(hidden, num_nat)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(out.last_hidden_state[:, 0, :])
        logits = {}
        if "cefr" in self.active_heads:
            logits["cefr"] = self.cefr_head(pooled)
        if "l1" in self.active_heads:
            logits["l1"] = self.l1_head(pooled)
        if "nat" in self.active_heads:
            logits["nat"] = self.nat_head(pooled)
        return logits
