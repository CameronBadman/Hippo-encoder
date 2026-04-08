from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DistillConfig:
    teacher_model_name: str
    student_model_name: str
    dataset_jsonl: str
    output_dir: str
    max_text_length: int = 64
    batch_size: int = 8
    num_epochs: int = 1
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    log_every: int = 10
    save_every: int = 500
    num_workers: int = 2
    teacher_text_weight: float = 1.0
    hidden_state_weight: float = 0.2
    contrastive_weight: float = 0.2
    contrastive_temperature: float = 0.07
    normalize_targets: bool = True
    gradient_clip_norm: float = 1.0
    warmup_steps: int = 100
    seed: int = 42

    @classmethod
    def from_json(cls, path: str | Path) -> "DistillConfig":
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return cls(**payload)
