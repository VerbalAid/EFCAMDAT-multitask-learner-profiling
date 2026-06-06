from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

DualMode = Literal["dual", "raw_only"]


@dataclass
class TrainConfig:
    csv_path: Path
    checkpoint_dir: Path
    seed: int = 42
    test_size: float = 0.2
    epochs: int = 3
    batch_size: int = 16
    eval_batch_size: int = 32
    grad_accum_steps: int = 1
    lr: float = 2e-5
    warmup_pct: float = 0.1
    early_stop_patience: int = 3
    dual_mode: DualMode = "dual"
    balance_cefr: bool = False
    max_length: int = 128
    model_name: str = "roberta-base"
    train_topic_family: Optional[str] = None
    exclude_place_heavy_topics: bool = False
    extra: dict = field(default_factory=dict)
