from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


def masked_mean(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).float()
    masked = hidden_states * mask
    denom = mask.sum(dim=1).clamp(min=1.0)
    return masked.sum(dim=1) / denom


def load_rows(path: str | Path, limit: int | None) -> list[dict]:
    rows: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            rows.append(payload)
            if limit is not None and len(rows) >= limit:
                break
    return rows


def load_texts(path: str | Path, limit: int | None) -> list[str]:
    texts: list[str] = []
    seen: set[str] = set()
    for payload in load_rows(path, limit=limit):
        for key in ("text", "anchor", "positive", "negative"):
            value = payload.get(key)
            if not isinstance(value, str):
                continue
            value = value.strip()
            if not value or value in seen:
                continue
            texts.append(value)
            seen.add(value)
    return texts


@torch.no_grad()
def encode_texts(
    texts: list[str],
    model_name: str,
    device: torch.device,
    max_length: int,
    batch_size: int,
) -> torch.Tensor:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()

    chunks: list[torch.Tensor] = []
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        tokens = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        tokens = {name: tensor.to(device) for name, tensor in tokens.items()}
        outputs = model(**tokens, return_dict=True)
        embeds = masked_mean(outputs.last_hidden_state, tokens["attention_mask"])
        chunks.append(F.normalize(embeds, dim=-1).cpu())
    return torch.cat(chunks, dim=0)


def build_cases(
    texts: list[str],
    embeds: torch.Tensor,
    num_cases: int,
    positives_per_case: int,
    negatives_per_case: int,
    positive_pool: int,
    hard_negative_pool: int,
    hard_negative_fraction: float,
    seed: int,
) -> list[dict]:
    if embeds.dim() != 2 or embeds.shape[0] != len(texts):
        raise ValueError("Embeddings must have shape [N, D] matching the text rows.")

    generator = random.Random(seed)
    indices = list(range(len(texts)))
    generator.shuffle(indices)
    selected_indices = indices[: min(num_cases, len(indices))]

    sims = embeds @ embeds.T
    sims.fill_diagonal_(-1.0)

    cases: list[dict] = []
    for query_index in selected_indices:
        row = sims[query_index]
        ranking = torch.argsort(row, descending=True)

        positives: list[str] = []
        for candidate_index in ranking[:positive_pool].tolist():
            if candidate_index == query_index:
                continue
            candidate_text = texts[candidate_index]
            if candidate_text == texts[query_index]:
                continue
            positives.append(candidate_text)
            if len(positives) >= positives_per_case:
                break

        if len(positives) < positives_per_case:
            continue

        hard_needed = min(negatives_per_case, max(0, int(round(negatives_per_case * hard_negative_fraction))))
        easy_needed = negatives_per_case - hard_needed

        hard_slice_start = max(positive_pool, 1)
        hard_slice_end = min(len(texts), hard_slice_start + max(hard_negative_pool, hard_needed))
        hard_candidates = [
            candidate
            for candidate in ranking[hard_slice_start:hard_slice_end].tolist()
            if candidate != query_index and texts[candidate] not in positives
        ]
        generator.shuffle(hard_candidates)

        easy_candidates = [
            candidate
            for candidate in ranking.flip(0).tolist()
            if candidate != query_index and texts[candidate] not in positives
        ]
        generator.shuffle(easy_candidates)

        negatives: list[str] = []
        for candidate_index in hard_candidates:
            negatives.append(texts[candidate_index])
            if len(negatives) >= hard_needed:
                break
        for candidate_index in easy_candidates:
            candidate_text = texts[candidate_index]
            if candidate_text in negatives:
                continue
            negatives.append(candidate_text)
            if len(negatives) >= negatives_per_case:
                break

        if len(negatives) < negatives_per_case:
            continue

        cases.append(
            {
                "query": texts[query_index],
                "positives": positives,
                "negatives": negatives,
            }
        )

    return cases


def build_cases_from_pairs(
    rows: list[dict],
    texts: list[str],
    embeds: torch.Tensor,
    positives_per_case: int,
    negatives_per_case: int,
    positive_pool: int,
    hard_negative_pool: int,
    hard_negative_fraction: float,
    seed: int,
) -> list[dict]:
    text_to_index = {text: idx for idx, text in enumerate(texts)}
    if len(text_to_index) != len(texts):
        raise ValueError("Texts must be unique when building pair-derived region cases.")

    generator = random.Random(seed)
    sims = embeds @ embeds.T
    sims.fill_diagonal_(-1.0)

    cases: list[dict] = []
    for row in rows:
        query = row.get("anchor")
        positive = row.get("positive")
        negative = row.get("negative")
        if not isinstance(query, str) or not isinstance(positive, str):
            continue
        query = query.strip()
        positive = positive.strip()
        if not query or not positive or query not in text_to_index or positive not in text_to_index:
            continue

        query_index = text_to_index[query]
        ranking = torch.argsort(sims[query_index], descending=True)

        positives = [positive]
        for candidate_index in ranking[:positive_pool].tolist():
            candidate_text = texts[candidate_index]
            if candidate_text == query or candidate_text in positives:
                continue
            positives.append(candidate_text)
            if len(positives) >= positives_per_case:
                break
        if len(positives) < positives_per_case:
            continue

        hard_needed = min(negatives_per_case, max(0, int(round(negatives_per_case * hard_negative_fraction))))
        easy_needed = negatives_per_case - hard_needed

        negatives: list[str] = []
        if isinstance(negative, str):
            negative = negative.strip()
            if negative and negative != query and negative not in positives:
                negatives.append(negative)

        hard_slice_start = max(positive_pool, 1)
        hard_slice_end = min(len(texts), hard_slice_start + max(hard_negative_pool, hard_needed + len(negatives)))
        hard_candidates = [
            candidate
            for candidate in ranking[hard_slice_start:hard_slice_end].tolist()
            if texts[candidate] != query and texts[candidate] not in positives and texts[candidate] not in negatives
        ]
        generator.shuffle(hard_candidates)

        easy_candidates = [
            candidate
            for candidate in ranking.flip(0).tolist()
            if texts[candidate] != query and texts[candidate] not in positives and texts[candidate] not in negatives
        ]
        generator.shuffle(easy_candidates)

        for candidate_index in hard_candidates:
            if len(negatives) >= hard_needed:
                break
            negatives.append(texts[candidate_index])
        for candidate_index in easy_candidates:
            if len(negatives) >= negatives_per_case:
                break
            negatives.append(texts[candidate_index])

        negatives = negatives[:negatives_per_case]
        if len(negatives) < negatives_per_case:
            continue

        cases.append(
            {
                "query": query,
                "positives": positives[:positives_per_case],
                "negatives": negatives,
            }
        )

    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description="Build query/positive/negative region cases from a text JSONL file.")
    parser.add_argument("--input-jsonl", required=True, help="Source JSONL with `text` rows.")
    parser.add_argument("--output", required=True, help="Destination JSON file for region cases.")
    parser.add_argument("--teacher-model", default="intfloat/e5-base-v2")
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--text-limit", type=int, default=None, help="Optional cap on source texts.")
    parser.add_argument("--num-cases", type=int, default=2000)
    parser.add_argument("--positives-per-case", type=int, default=2)
    parser.add_argument("--negatives-per-case", type=int, default=3)
    parser.add_argument("--positive-pool", type=int, default=16, help="Take positives from the top-N nearest neighbors.")
    parser.add_argument(
        "--hard-negative-pool",
        type=int,
        default=64,
        help="Sample hard negatives from the post-positive neighborhood slice.",
    )
    parser.add_argument(
        "--hard-negative-fraction",
        type=float,
        default=0.67,
        help="Fraction of negatives to draw from the hard-negative pool.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = load_rows(args.input_jsonl, limit=args.text_limit)
    texts = load_texts(args.input_jsonl, limit=args.text_limit)
    if len(texts) < args.positives_per_case + args.negatives_per_case + 1:
        raise ValueError("Not enough texts to build region cases.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embeds = encode_texts(
        texts=texts,
        model_name=args.teacher_model,
        device=device,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )

    if rows and any("anchor" in row for row in rows):
        cases = build_cases_from_pairs(
            rows=rows[: args.num_cases],
            texts=texts,
            embeds=embeds,
            positives_per_case=args.positives_per_case,
            negatives_per_case=args.negatives_per_case,
            positive_pool=args.positive_pool,
            hard_negative_pool=args.hard_negative_pool,
            hard_negative_fraction=args.hard_negative_fraction,
            seed=args.seed,
        )
    else:
        cases = build_cases(
            texts=texts,
            embeds=embeds,
            num_cases=args.num_cases,
            positives_per_case=args.positives_per_case,
            negatives_per_case=args.negatives_per_case,
            positive_pool=args.positive_pool,
            hard_negative_pool=args.hard_negative_pool,
            hard_negative_fraction=args.hard_negative_fraction,
            seed=args.seed,
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(cases, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "input_jsonl": str(args.input_jsonl),
                "teacher_model": args.teacher_model,
                "texts_loaded": len(texts),
                "cases_written": len(cases),
                "positives_per_case": args.positives_per_case,
                "negatives_per_case": args.negatives_per_case,
                "output": str(output_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
