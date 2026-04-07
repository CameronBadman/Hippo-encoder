from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


class ImageTextJsonlDataset(Dataset):
    """Local image-text dataset backed by a JSONL manifest."""

    def __init__(self, jsonl_path: str | Path, image_root: str | Path):
        self.jsonl_path = Path(jsonl_path)
        self.image_root = Path(image_root)
        self.rows = []

        with self.jsonl_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if "image" not in row or "text" not in row:
                    raise ValueError("Each JSONL row must contain `image` and `text` keys.")
                self.rows.append(row)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        image_path = self.image_root / row["image"]
        image = Image.open(image_path).convert("RGB")
        return {
            "image": image,
            "text": row["text"],
            "image_path": str(image_path),
        }
