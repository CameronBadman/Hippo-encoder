from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from hippo_encoder.rope_region import DualRopeFormulaProgram
from hippo_encoder.student import TinyEncoderStudent


DEFAULT_SCORE_THRESHOLDS = [0.001, 0.002, 0.003, 0.004, 0.005, 0.0075, 0.01]
DEFAULT_TOP_K = [1, 2, 3, 5, 10]
SCORE_MODES = ("mean", "inv_radius_weighted", "topk_overflow", "mean_plus_max", "mean_plus_l2")
DEFAULT_PRESET = {
    "terms_per_side": 32,
    "min_radius": 0.015,
    "radius_scale": 0.85,
    "negative_weight": 0.8,
    "size_weight": 0.012,
    "teacher_weight": 0.25,
    "student_weight": 1.5,
}


def masked_mean(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).float()
    masked = hidden_states * mask
    denom = mask.sum(dim=1).clamp(min=1.0)
    return masked.sum(dim=1) / denom


class TeacherEncoder:
    def __init__(self, model_name: str, device: torch.device, max_length: int):
        self.device = device
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()

    @torch.no_grad()
    def encode(self, texts: list[str], batch_size: int) -> torch.Tensor:
        outputs = []
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            batch = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            batch = {name: tensor.to(self.device) for name, tensor in batch.items()}
            model_outputs = self.model(**batch, return_dict=True)
            embeds = masked_mean(model_outputs.last_hidden_state, batch["attention_mask"])
            outputs.append(F.normalize(embeds, dim=-1).cpu())
        return torch.cat(outputs, dim=0)


class StudentEncoder:
    def __init__(self, checkpoint_dir: str | Path, device: torch.device, max_length: int):
        self.device = device
        self.max_length = max_length
        self.student = TinyEncoderStudent.load_checkpoint(checkpoint_dir, device=device)
        self.student.eval()

    @torch.no_grad()
    def encode(self, texts: list[str], batch_size: int) -> torch.Tensor:
        outputs = []
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            batch_outputs = self.student(texts=batch_texts, device=self.device, max_length=self.max_length)
            outputs.append(batch_outputs["projected_embeds"].cpu())
        return torch.cat(outputs, dim=0)


@dataclass(frozen=True)
class RetrievalCase:
    query: str
    positives: list[str]
    negatives: list[str]
    distractors: list[str]


def load_cases(path: str | Path, seed: int) -> list[dict]:
    with open(path, "r", encoding="utf-8") as handle:
        cases = json.load(handle)
    if not isinstance(cases, list):
        raise ValueError("Cases file must be a JSON list.")
    cases = list(cases)
    random.Random(seed).shuffle(cases)
    return cases


def collect_text_pool(cases: Iterable[dict]) -> list[str]:
    texts: list[str] = []
    seen: set[str] = set()
    for case in cases:
        for text in [case["query"], *case.get("positives", []), *case.get("negatives", [])]:
            if text not in seen:
                seen.add(text)
                texts.append(text)
    return texts


def build_retrieval_cases(
    cases: list[dict],
    text_pool: list[str],
    distractors_per_case: int,
    positives_per_case: int,
    teacher_embeds: torch.Tensor | None,
    text_to_index: dict[str, int] | None,
    seed: int,
) -> list[RetrievalCase]:
    rng = random.Random(seed)
    retrieval_cases = []
    for case in cases:
        query = case["query"]
        positives = list(dict.fromkeys(case.get("positives", [])))
        negatives = list(dict.fromkeys(case.get("negatives", [])))

        if positives_per_case > len(positives):
            if teacher_embeds is None or text_to_index is None:
                raise ValueError("Teacher embeddings are required to expand positives.")
            positives = expand_positives_by_teacher_neighbors(
                query=query,
                positives=positives,
                negatives=negatives,
                text_pool=text_pool,
                text_to_index=text_to_index,
                teacher_embeds=teacher_embeds,
                positives_per_case=positives_per_case,
            )

        blocked = {query, *positives, *negatives}
        candidates = [text for text in text_pool if text not in blocked]
        if distractors_per_case > len(candidates):
            raise ValueError(
                f"Requested {distractors_per_case} distractors, but only {len(candidates)} are available. "
                "Use more source cases or reduce --distractors-per-case."
            )
        distractors = rng.sample(candidates, distractors_per_case)
        retrieval_cases.append(
            RetrievalCase(
                query=query,
                positives=positives,
                negatives=negatives,
                distractors=distractors,
            )
        )
    return retrieval_cases


def expand_positives_by_teacher_neighbors(
    query: str,
    positives: list[str],
    negatives: list[str],
    text_pool: list[str],
    text_to_index: dict[str, int],
    teacher_embeds: torch.Tensor,
    positives_per_case: int,
) -> list[str]:
    expanded = list(dict.fromkeys(positives))
    blocked = {query, *expanded, *negatives}
    query_embed = teacher_embeds[text_to_index[query]]
    pool_indices = [text_to_index[text] for text in text_pool]
    sims = teacher_embeds[pool_indices] @ query_embed
    ranked = sorted(
        zip(text_pool, sims.tolist(), strict=True),
        key=lambda item: item[1],
        reverse=True,
    )
    for text, _score in ranked:
        if text in blocked:
            continue
        expanded.append(text)
        blocked.add(text)
        if len(expanded) >= positives_per_case:
            break
    if len(expanded) < positives_per_case:
        raise ValueError(
            f"Could only find {len(expanded)} positives for query {query!r}; "
            f"requested {positives_per_case}."
        )
    return expanded


def encode_texts(
    texts: list[str],
    teacher: TeacherEncoder,
    student: StudentEncoder,
    batch_size: int,
) -> tuple[dict[str, int], torch.Tensor, torch.Tensor]:
    text_to_index = {text: index for index, text in enumerate(texts)}
    teacher_embeds = teacher.encode(texts, batch_size=batch_size)
    student_embeds = student.encode(texts, batch_size=batch_size)
    return text_to_index, teacher_embeds, student_embeds


def tensor_for(texts: list[str], text_to_index: dict[str, int], embeds: torch.Tensor) -> torch.Tensor:
    return embeds[[text_to_index[text] for text in texts]]


def soft_box_scores(
    vectors: torch.Tensor,
    query: torch.Tensor,
    minus: torch.Tensor,
    plus: torch.Tensor,
    mode: str,
    overflow_topk: int,
    max_overflow_alpha: float,
    l2_alpha: float,
    distances: torch.Tensor,
) -> torch.Tensor:
    lower = query - minus
    upper = query + plus
    overflow = torch.relu(lower.unsqueeze(0) - vectors) + torch.relu(vectors - upper.unsqueeze(0))
    mean_score = overflow.mean(dim=-1)
    if mode == "mean":
        return mean_score
    if mode == "inv_radius_weighted":
        weights = 1.0 / (minus + plus + 1e-6)
        return (overflow * weights.unsqueeze(0)).sum(dim=-1) / weights.sum().clamp(min=1e-12)
    if mode == "topk_overflow":
        topk = min(max(1, overflow_topk), overflow.shape[-1])
        return torch.topk(overflow, k=topk, dim=-1).values.mean(dim=-1)
    if mode == "mean_plus_max":
        return mean_score + max_overflow_alpha * overflow.max(dim=-1).values
    if mode == "mean_plus_l2":
        return mean_score + l2_alpha * distances
    raise ValueError(f"Unsupported score mode: {mode}")


def average_precision_at_k(labels: list[int], k: int) -> float:
    hits = 0
    precision_sum = 0.0
    for rank, label in enumerate(labels[:k], start=1):
        if label:
            hits += 1
            precision_sum += hits / rank
    return precision_sum / max(1, sum(labels))


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": sum(values) / len(values),
        "min": min(values),
        "max": max(values),
    }


def evaluate_case(
    case: RetrievalCase,
    text_to_index: dict[str, int],
    teacher_embeds: torch.Tensor,
    student_embeds: torch.Tensor,
    top_k: list[int],
    score_thresholds: list[float],
    score_mode: str,
    radius_scale: float,
    args: argparse.Namespace,
) -> dict:
    query_teacher = tensor_for([case.query], text_to_index, teacher_embeds)[0]
    query_student = tensor_for([case.query], text_to_index, student_embeds)[0]
    positive_teacher = tensor_for(case.positives, text_to_index, teacher_embeds)
    negative_teacher = tensor_for(case.negatives, text_to_index, teacher_embeds)

    program = DualRopeFormulaProgram.from_transfer_case(
        teacher_anchor=query_teacher,
        student_anchor=query_student,
        positives=positive_teacher,
        negatives=negative_teacher,
        terms_per_side=args.terms_per_side,
        base_radius=args.min_radius,
        radius_scale=radius_scale,
        negative_weight=args.negative_weight,
        size_weight=args.size_weight,
        teacher_weight=args.teacher_weight,
        student_weight=args.student_weight,
    )
    region = program.hydrate(query_student)

    records = [*case.positives, *case.negatives, *case.distractors]
    labels = [1] * len(case.positives) + [0] * (len(case.negatives) + len(case.distractors))
    vectors = tensor_for(records, text_to_index, student_embeds)

    started = time.perf_counter()
    distances = torch.linalg.vector_norm(vectors - query_student.unsqueeze(0), dim=-1)
    scores = soft_box_scores(
        vectors,
        query_student,
        region["minus"],
        region["plus"],
        mode=score_mode,
        overflow_topk=args.overflow_topk,
        max_overflow_alpha=args.max_overflow_alpha,
        l2_alpha=args.l2_alpha,
        distances=distances,
    )
    order = sorted(range(len(records)), key=lambda index: (float(scores[index]), float(distances[index]), index))
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    ordered_labels = [labels[index] for index in order]
    ordered_scores = [float(scores[index]) for index in order]
    positive_count = len(case.positives)
    negative_count = len(labels) - positive_count

    topk_metrics = {}
    for k in top_k:
        k_labels = ordered_labels[:k]
        positives_found = sum(k_labels)
        topk_metrics[str(k)] = {
            "accuracy": 1.0 if k_labels and k_labels[0] == 1 else 0.0,
            "hit_rate": 1.0 if positives_found > 0 else 0.0,
            "precision": positives_found / max(1, k),
            "recall": positives_found / max(1, positive_count),
            "average_precision": average_precision_at_k(ordered_labels, k),
        }

    threshold_metrics = {}
    for threshold in score_thresholds:
        selected = [index for index, score in enumerate(ordered_scores) if score <= threshold]
        selected_labels = [ordered_labels[index] for index in selected]
        true_positive = sum(selected_labels)
        false_positive = len(selected_labels) - true_positive
        precision = true_positive / len(selected_labels) if selected_labels else 0.0
        recall = true_positive / max(1, positive_count)
        threshold_metrics[str(threshold)] = {
            "coverage": 1.0 if selected_labels else 0.0,
            "avg_results": float(len(selected_labels)),
            "precision": precision,
            "recall": recall,
            "false_positive_rate": false_positive / max(1, negative_count),
            "f1": (2 * precision * recall / (precision + recall)) if precision + recall > 0 else 0.0,
        }

    positive_scores = [float(scores[index]) for index, label in enumerate(labels) if label == 1]
    negative_scores = [float(scores[index]) for index, label in enumerate(labels) if label == 0]
    first_positive_rank = next((rank for rank, label in enumerate(ordered_labels, start=1) if label), None)

    return {
        "query": case.query,
        "record_count": len(records),
        "positive_count": positive_count,
        "negative_count": negative_count,
        "elapsed_ms": elapsed_ms,
        "mrr": 1.0 / first_positive_rank if first_positive_rank is not None else 0.0,
        "topk": topk_metrics,
        "thresholds": threshold_metrics,
        "score_stats": {
            "positive": summarize(positive_scores),
            "negative": summarize(negative_scores),
            "gap": (sum(negative_scores) / len(negative_scores)) - (sum(positive_scores) / len(positive_scores)),
        },
        "top_results": [
            {
                "rank": rank,
                "label": "positive" if labels[index] else "negative",
                "score": float(scores[index]),
                "distance": float(distances[index]),
                "text": records[index],
            }
            for rank, index in enumerate(order[: max(top_k)], start=1)
        ],
    }


def aggregate(case_results: list[dict], top_k: list[int], score_thresholds: list[float]) -> dict:
    summary = {
        "cases": len(case_results),
        "records_per_case_mean": sum(result["record_count"] for result in case_results) / len(case_results),
        "elapsed_ms_per_case_mean": sum(result["elapsed_ms"] for result in case_results) / len(case_results),
        "mrr": sum(result["mrr"] for result in case_results) / len(case_results),
        "score_gap_mean": sum(result["score_stats"]["gap"] for result in case_results) / len(case_results),
        "topk": {},
        "thresholds": {},
    }
    for k in top_k:
        key = str(k)
        summary["topk"][key] = {
            metric: sum(result["topk"][key][metric] for result in case_results) / len(case_results)
            for metric in ("accuracy", "hit_rate", "precision", "recall", "average_precision")
        }
    for threshold in score_thresholds:
        key = str(threshold)
        summary["thresholds"][key] = {
            metric: sum(result["thresholds"][key][metric] for result in case_results) / len(case_results)
            for metric in ("coverage", "avg_results", "precision", "recall", "false_positive_rate", "f1")
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Hippo-5 soft-box retrieval with large distractor pools.")
    parser.add_argument("--cases", required=True, help="Region case JSON file.")
    parser.add_argument("--student-checkpoint", required=True)
    parser.add_argument("--teacher-model", default="intfloat/e5-base-v2")
    parser.add_argument("--output", default=None)
    parser.add_argument("--case-limit", type=int, default=100)
    parser.add_argument("--distractors-per-case", type=int, default=1000)
    parser.add_argument(
        "--positives-per-case",
        type=int,
        default=2,
        help=(
            "Minimum relevant positives per query. Values above the positives already present in the case "
            "are filled with teacher-nearest neighbors from the corpus."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--top-k", type=int, nargs="+", default=DEFAULT_TOP_K)
    parser.add_argument("--score-thresholds", type=float, nargs="+", default=DEFAULT_SCORE_THRESHOLDS)
    parser.add_argument("--score-modes", nargs="+", choices=SCORE_MODES, default=["mean"])
    parser.add_argument("--overflow-topk", type=int, default=32)
    parser.add_argument("--max-overflow-alpha", type=float, default=0.25)
    parser.add_argument("--l2-alpha", type=float, default=0.01)
    parser.add_argument("--terms-per-side", type=int, default=DEFAULT_PRESET["terms_per_side"])
    parser.add_argument("--min-radius", type=float, default=DEFAULT_PRESET["min_radius"])
    parser.add_argument("--radius-scale", type=float, default=DEFAULT_PRESET["radius_scale"])
    parser.add_argument("--radius-scales", type=float, nargs="+", default=None)
    parser.add_argument("--negative-weight", type=float, default=DEFAULT_PRESET["negative_weight"])
    parser.add_argument("--size-weight", type=float, default=DEFAULT_PRESET["size_weight"])
    parser.add_argument("--teacher-weight", type=float, default=DEFAULT_PRESET["teacher_weight"])
    parser.add_argument("--student-weight", type=float, default=DEFAULT_PRESET["student_weight"])
    parser.add_argument("--include-cases", action="store_true", help="Include per-case results in the output JSON.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_source_cases = load_cases(args.cases, seed=args.seed)
    source_cases = all_source_cases[: args.case_limit] if args.case_limit is not None else all_source_cases
    text_pool = collect_text_pool(all_source_cases)
    teacher = TeacherEncoder(args.teacher_model, device=device, max_length=args.max_length)
    student = StudentEncoder(args.student_checkpoint, device=device, max_length=args.max_length)
    pool_teacher_embeds = None
    pool_text_to_index = None
    if args.positives_per_case > 2:
        pool_text_to_index = {text: index for index, text in enumerate(text_pool)}
        pool_teacher_embeds = teacher.encode(text_pool, batch_size=args.batch_size)
    retrieval_cases = build_retrieval_cases(
        cases=source_cases,
        text_pool=text_pool,
        distractors_per_case=args.distractors_per_case,
        positives_per_case=args.positives_per_case,
        teacher_embeds=pool_teacher_embeds,
        text_to_index=pool_text_to_index,
        seed=args.seed + 1,
    )

    all_texts = collect_text_pool(
        {
            "query": case.query,
            "positives": case.positives,
            "negatives": [*case.negatives, *case.distractors],
        }
        for case in retrieval_cases
    )
    text_to_index, teacher_embeds, student_embeds = encode_texts(
        all_texts,
        teacher=teacher,
        student=student,
        batch_size=args.batch_size,
    )

    radius_scales = args.radius_scales if args.radius_scales is not None else [args.radius_scale]
    variant_results = {}
    for radius_scale in radius_scales:
        for score_mode in args.score_modes:
            variant_key = f"{score_mode}_radius{radius_scale:g}"
            case_results = [
                evaluate_case(
                    case,
                    text_to_index=text_to_index,
                    teacher_embeds=teacher_embeds,
                    student_embeds=student_embeds,
                    top_k=args.top_k,
                    score_thresholds=args.score_thresholds,
                    score_mode=score_mode,
                    radius_scale=radius_scale,
                    args=args,
                )
                for case in retrieval_cases
            ]
            variant_results[variant_key] = {
                "score_mode": score_mode,
                "radius_scale": radius_scale,
                "summary": aggregate(case_results, top_k=args.top_k, score_thresholds=args.score_thresholds),
                "cases": case_results if args.include_cases else None,
            }

    primary_key = next(iter(variant_results))
    payload = {
        "config": {
            "cases": args.cases,
            "student_checkpoint": args.student_checkpoint,
            "teacher_model": args.teacher_model,
            "case_limit": args.case_limit,
            "distractors_per_case": args.distractors_per_case,
            "positives_per_case": args.positives_per_case,
            "top_k": args.top_k,
            "score_thresholds": args.score_thresholds,
            "score_modes": args.score_modes,
            "overflow_topk": args.overflow_topk,
            "max_overflow_alpha": args.max_overflow_alpha,
            "l2_alpha": args.l2_alpha,
            "preset": {
                "terms_per_side": args.terms_per_side,
                "min_radius": args.min_radius,
                "radius_scale": args.radius_scale,
                "radius_scales": radius_scales,
                "negative_weight": args.negative_weight,
                "size_weight": args.size_weight,
                "teacher_weight": args.teacher_weight,
                "student_weight": args.student_weight,
            },
        },
        "summary": variant_results[primary_key]["summary"],
        "variants": {
            key: {
                "score_mode": result["score_mode"],
                "radius_scale": result["radius_scale"],
                "summary": result["summary"],
            }
            for key, result in variant_results.items()
        },
    }
    if args.include_cases:
        payload["cases"] = {
            key: result["cases"]
            for key, result in variant_results.items()
        }

    encoded = json.dumps(payload, indent=2)
    if args.output:
        Path(args.output).write_text(encoded, encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
