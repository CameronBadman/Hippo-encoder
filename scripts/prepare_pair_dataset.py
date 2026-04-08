from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset


def export_all_nli_pair(
    split: str,
    limit: int,
    prefix: str,
    output: Path,
) -> int:
    dataset = load_dataset("sentence-transformers/all-nli", "pair", split=split)
    rows_written = 0
    with output.open("w", encoding="utf-8") as handle:
        for row in dataset:
            anchor = row.get("anchor")
            positive = row.get("positive")
            if not isinstance(anchor, str) or not isinstance(positive, str):
                continue
            anchor = anchor.strip()
            positive = positive.strip()
            if not anchor or not positive:
                continue
            handle.write(
                json.dumps(
                    {
                        "anchor": f"{prefix}{anchor}",
                        "positive": f"{prefix}{positive}",
                    },
                    ensure_ascii=True,
                )
                + "\n"
            )
            rows_written += 1
            if rows_written >= limit:
                break
    return rows_written


def export_all_nli_triplet(
    split: str,
    limit: int,
    prefix: str,
    output: Path,
) -> int:
    dataset = load_dataset("sentence-transformers/all-nli", "triplet", split=split)
    rows_written = 0
    with output.open("w", encoding="utf-8") as handle:
        for row in dataset:
            anchor = row.get("anchor")
            positive = row.get("positive")
            negative = row.get("negative")
            if not all(isinstance(item, str) for item in (anchor, positive, negative)):
                continue
            anchor = anchor.strip()
            positive = positive.strip()
            negative = negative.strip()
            if not anchor or not positive or not negative:
                continue
            handle.write(
                json.dumps(
                    {
                        "anchor": f"{prefix}{anchor}",
                        "positive": f"{prefix}{positive}",
                        "negative": f"{prefix}{negative}",
                    },
                    ensure_ascii=True,
                )
                + "\n"
            )
            rows_written += 1
            if rows_written >= limit:
                break
    return rows_written


def export_msmarco_triplet(
    split: str,
    limit: int,
    prefix: str,
    output: Path,
) -> int:
    dataset = load_dataset("sentence-transformers/msmarco", "triplet", split=split)
    rows_written = 0
    with output.open("w", encoding="utf-8") as handle:
        for row in dataset:
            query = row.get("query")
            positive = row.get("positive")
            negative = row.get("negative")
            if not all(isinstance(item, str) for item in (query, positive, negative)):
                continue
            query = query.strip()
            positive = positive.strip()
            negative = negative.strip()
            if not query or not positive or not negative:
                continue
            handle.write(
                json.dumps(
                    {
                        "anchor": f"query: {query}" if not query.startswith("query: ") else query,
                        "positive": f"{prefix}{positive}",
                        "negative": f"{prefix}{negative}",
                    },
                    ensure_ascii=True,
                )
                + "\n"
            )
            rows_written += 1
            if rows_written >= limit:
                break
    return rows_written


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare pair/triplet JSONL for tiny encoder distillation.")
    parser.add_argument(
        "--source",
        choices=("all_nli_pair", "all_nli_triplet", "msmarco_triplet"),
        required=True,
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=50000)
    parser.add_argument("--prefix", default="passage: ")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.source == "all_nli_pair":
        rows_written = export_all_nli_pair(args.split, args.limit, args.prefix, output_path)
    elif args.source == "all_nli_triplet":
        rows_written = export_all_nli_triplet(args.split, args.limit, args.prefix, output_path)
    else:
        rows_written = export_msmarco_triplet(args.split, args.limit, args.prefix, output_path)

    print(
        json.dumps(
            {
                "source": args.source,
                "split": args.split,
                "rows_written": rows_written,
                "output": str(output_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
