from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset


COMMON_TEXT_COLUMNS = (
    "text",
    "sentence",
    "caption",
    "content",
    "description",
)


def infer_text_column(sample: dict, explicit_column: str | None) -> str:
    if explicit_column is not None:
        if explicit_column not in sample:
            raise ValueError(f"Requested column `{explicit_column}` not found in dataset row.")
        return explicit_column

    for column in COMMON_TEXT_COLUMNS:
        if column in sample and isinstance(sample[column], str):
            return column

    for key, value in sample.items():
        if isinstance(value, str):
            return key

    raise ValueError("Could not infer a text column from the dataset sample.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a text-only JSONL file from a Hugging Face dataset.")
    parser.add_argument("--dataset", default="ag_news", help="Dataset name to load with datasets.load_dataset.")
    parser.add_argument("--split", default="train", help="Dataset split to read.")
    parser.add_argument("--config-name", default=None, help="Optional dataset config name.")
    parser.add_argument("--text-column", default=None, help="Optional explicit text column.")
    parser.add_argument("--limit", type=int, default=5000, help="Maximum number of rows to export.")
    parser.add_argument("--prefix", default="passage: ", help="Prefix added to each exported row.")
    parser.add_argument("--output", required=True, help="Path to the output JSONL file.")
    args = parser.parse_args()

    dataset = load_dataset(args.dataset, args.config_name, split=args.split)
    sample = dataset[0]
    text_column = infer_text_column(sample, args.text_column)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows_written = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for row in dataset:
            text = row[text_column]
            if not isinstance(text, str):
                continue
            text = text.strip()
            if not text:
                continue
            handle.write(json.dumps({"text": f"{args.prefix}{text}"}, ensure_ascii=True) + "\n")
            rows_written += 1
            if rows_written >= args.limit:
                break

    print(
        json.dumps(
            {
                "dataset": args.dataset,
                "split": args.split,
                "text_column": text_column,
                "rows_written": rows_written,
                "output": str(output_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
