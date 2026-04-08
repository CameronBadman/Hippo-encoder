from __future__ import annotations

import json
from pathlib import Path

from torch.utils.data import Dataset


class DistillJsonlDataset(Dataset):
    """JSONL dataset supporting text-only rows or anchor/positive[/negative] rows."""

    def __init__(self, jsonl_path: str | Path):
        self.jsonl_path = Path(jsonl_path)
        self.rows = []
        self.schema: str | None = None

        with self.jsonl_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                schema = self._infer_schema(row)
                if self.schema is None:
                    self.schema = schema
                elif schema != self.schema:
                    raise ValueError(f"Mixed dataset schemas are not supported: {self.schema} vs {schema}")
                self.rows.append(row)

        if self.schema is None:
            raise ValueError("Dataset is empty.")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        if self.schema == "text":
            return {"text": row["text"]}

        item = {
            "anchor": row["anchor"],
            "positive": row["positive"],
        }
        if "negative" in row and isinstance(row["negative"], str) and row["negative"].strip():
            item["negative"] = row["negative"]
        return item

    @staticmethod
    def _infer_schema(row: dict) -> str:
        if "text" in row and isinstance(row["text"], str):
            return "text"
        if "anchor" in row and "positive" in row and isinstance(row["anchor"], str) and isinstance(row["positive"], str):
            return "pair"
        raise ValueError("Each JSONL row must contain either `text` or `anchor` and `positive`.")
