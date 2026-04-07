from __future__ import annotations

import json
from pathlib import Path

from torch.utils.data import Dataset


class TextJsonlDataset(Dataset):
    """Local text-only dataset backed by a JSONL manifest."""

    def __init__(self, jsonl_path: str | Path):
        self.jsonl_path = Path(jsonl_path)
        self.rows = []

        with self.jsonl_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if "text" not in row:
                    raise ValueError("Each JSONL row must contain a `text` key.")
                self.rows.append(row)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        return {
            "text": row["text"],
        }
