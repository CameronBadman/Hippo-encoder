from __future__ import annotations

import argparse
import json

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from hippo_encoder.student import TinyEncoderStudent


DEFAULT_EVAL_TEXTS = [
    "query: red chair near the window",
    "query: a scarlet chair beside the window",
    "query: blue mug on the desk",
    "query: coffee cup sitting on the office desk",
    "query: dog running through the grass",
    "query: a small dog sprinting across a field",
    "query: stock market falls after weak earnings",
    "query: company shares drop after poor results",
    "query: basketball team wins championship game",
    "query: the team secured the title in the final match",
    "query: person standing beside a parked bicycle",
    "query: someone next to a bike on the street",
]

DEFAULT_PARAPHRASE_PAIRS = [
    (0, 1, "chair paraphrase"),
    (2, 3, "mug paraphrase"),
    (4, 5, "dog paraphrase"),
    (6, 7, "market paraphrase"),
    (8, 9, "basketball paraphrase"),
    (10, 11, "bicycle paraphrase"),
]


def pair_key(left: int, right: int) -> tuple[int, int]:
    return (left, right) if left < right else (right, left)


def masked_mean(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).float()
    masked = hidden_states * mask
    denom = mask.sum(dim=1).clamp(min=1.0)
    return masked.sum(dim=1) / denom


@torch.no_grad()
def teacher_embed(
    texts: list[str],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: torch.device,
    max_length: int,
) -> torch.Tensor:
    batch = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    batch = {name: tensor.to(device) for name, tensor in batch.items()}
    outputs = model(**batch, return_dict=True)
    embeds = masked_mean(outputs.last_hidden_state, batch["attention_mask"])
    return F.normalize(embeds, dim=-1)


@torch.no_grad()
def student_embed(
    texts: list[str],
    student: TinyEncoderStudent,
    device: torch.device,
    max_length: int,
) -> torch.Tensor:
    outputs = student(texts=texts, device=device, max_length=max_length)
    return outputs["projected_embeds"]


def topk_neighbors(similarity: torch.Tensor, k: int) -> torch.Tensor:
    indices = torch.topk(similarity, k=k + 1, dim=-1).indices
    return indices[:, 1:]


def load_eval_suite(path: str | None) -> tuple[list[str], list[tuple[int, int, str]]]:
    if path is None:
        return DEFAULT_EVAL_TEXTS, DEFAULT_PARAPHRASE_PAIRS
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list) and all(isinstance(item, str) for item in payload):
        return payload, DEFAULT_PARAPHRASE_PAIRS
    if not isinstance(payload, dict):
        raise ValueError("Eval file must be a JSON list of strings or an object with `texts`.")

    texts = payload.get("texts")
    if not isinstance(texts, list) or not all(isinstance(item, str) for item in texts):
        raise ValueError("Eval suite `texts` must be a JSON list of strings.")

    paraphrase_pairs = []
    for item in payload.get("paraphrase_pairs", []):
        if (
            not isinstance(item, list)
            or len(item) != 3
            or not isinstance(item[0], int)
            or not isinstance(item[1], int)
            or not isinstance(item[2], str)
        ):
            raise ValueError("Each paraphrase pair must be [left_index, right_index, label].")
        paraphrase_pairs.append((item[0], item[1], item[2]))
    return texts, paraphrase_pairs


def collect_pair_stats(
    similarity: torch.Tensor,
    paraphrase_pairs: list[tuple[int, int, str]],
    num_texts: int,
) -> dict[str, float]:
    paraphrase_keys = {
        pair_key(left, right)
        for left, right, _label in paraphrase_pairs
        if left < num_texts and right < num_texts
    }

    paraphrase_values = []
    non_paraphrase_values = []
    for left in range(num_texts):
        for right in range(left + 1, num_texts):
            value = float(similarity[left, right].item())
            if pair_key(left, right) in paraphrase_keys:
                paraphrase_values.append(value)
            else:
                non_paraphrase_values.append(value)

    paraphrase_mean = sum(paraphrase_values) / max(1, len(paraphrase_values))
    non_paraphrase_mean = sum(non_paraphrase_values) / max(1, len(non_paraphrase_values))

    return {
        "paraphrase_mean": paraphrase_mean,
        "non_paraphrase_mean": non_paraphrase_mean,
        "paraphrase_margin": paraphrase_mean - non_paraphrase_mean,
        "max_non_paraphrase": max(non_paraphrase_values) if non_paraphrase_values else 0.0,
        "min_paraphrase": min(paraphrase_values) if paraphrase_values else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a distilled tiny LLM encoder against the teacher encoder.")
    parser.add_argument("--student-checkpoint", required=True)
    parser.add_argument("--teacher-model", default="intfloat/e5-base-v2")
    parser.add_argument("--texts-file", default=None, help="Optional JSON file containing a list of eval texts.")
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    texts, paraphrase_pairs = load_eval_suite(args.texts_file)

    teacher_tokenizer = AutoTokenizer.from_pretrained(args.teacher_model)
    teacher_model = AutoModel.from_pretrained(args.teacher_model).to(device).eval()
    student = TinyEncoderStudent.load_checkpoint(args.student_checkpoint, device=device)
    student.eval()

    teacher_embeds = teacher_embed(texts, teacher_tokenizer, teacher_model, device, args.max_length)
    student_embeds = student_embed(texts, student, device, args.max_length)

    pair_cos = F.cosine_similarity(teacher_embeds, student_embeds, dim=-1)
    teacher_sim = teacher_embeds @ teacher_embeds.T
    student_sim = student_embeds @ student_embeds.T
    teacher_nn = topk_neighbors(teacher_sim, k=args.top_k)
    student_nn = topk_neighbors(student_sim, k=args.top_k)

    neighbor_overlap = []
    for index in range(len(texts)):
        teacher_set = set(teacher_nn[index].tolist())
        student_set = set(student_nn[index].tolist())
        neighbor_overlap.append(len(teacher_set & student_set) / max(1, len(teacher_set)))

    paraphrase_scores = []
    for left, right, label in paraphrase_pairs:
        if right >= len(texts):
            continue
        paraphrase_scores.append(
            {
                "label": label,
                "teacher": F.cosine_similarity(teacher_embeds[left : left + 1], teacher_embeds[right : right + 1]).item(),
                "student": F.cosine_similarity(student_embeds[left : left + 1], student_embeds[right : right + 1]).item(),
            }
        )

    valid_paraphrase_pairs = [
        (left, right, label)
        for left, right, label in paraphrase_pairs
        if right < len(texts)
    ]
    teacher_pair_stats = collect_pair_stats(teacher_sim, valid_paraphrase_pairs, len(texts))
    student_pair_stats = collect_pair_stats(student_sim, valid_paraphrase_pairs, len(texts))

    print(
        json.dumps(
            {
                "summary": {
                    "mean_teacher_student_cosine": pair_cos.mean().item(),
                    "mean_topk_overlap": sum(neighbor_overlap) / len(neighbor_overlap),
                    "teacher_paraphrase_mean": teacher_pair_stats["paraphrase_mean"],
                    "teacher_non_paraphrase_mean": teacher_pair_stats["non_paraphrase_mean"],
                    "teacher_paraphrase_margin": teacher_pair_stats["paraphrase_margin"],
                    "student_paraphrase_mean": student_pair_stats["paraphrase_mean"],
                    "student_non_paraphrase_mean": student_pair_stats["non_paraphrase_mean"],
                    "student_paraphrase_margin": student_pair_stats["paraphrase_margin"],
                    "student_collapse_gap": student_pair_stats["min_paraphrase"] - student_pair_stats["max_non_paraphrase"],
                    "num_texts": len(texts),
                    "top_k": args.top_k,
                },
                "pair_cosine": [
                    {"text": text, "teacher_student_cosine": score}
                    for text, score in zip(texts, pair_cos.tolist())
                ],
                "paraphrase_similarity": paraphrase_scores,
                "pair_stats": {
                    "teacher": teacher_pair_stats,
                    "student": student_pair_stats,
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
