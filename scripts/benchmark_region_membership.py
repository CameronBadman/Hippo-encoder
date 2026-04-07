from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


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
    def encode(self, texts: list[str]) -> torch.Tensor:
        batch = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        batch = {name: tensor.to(self.device) for name, tensor in batch.items()}
        outputs = self.model(**batch, return_dict=True)
        embeds = masked_mean(outputs.last_hidden_state, batch["attention_mask"])
        return F.normalize(embeds, dim=-1)


class StudentEncoder:
    def __init__(self, checkpoint_dir: str | Path, device: torch.device, max_length: int):
        self.device = device
        self.max_length = max_length
        checkpoint_dir = Path(checkpoint_dir)

        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir / "tokenizer")
        self.backbone = AutoModel.from_pretrained(checkpoint_dir / "backbone").to(device).eval()
        heads = torch.load(checkpoint_dir / "heads.pt", map_location=device)

        target_dim = heads["embed_head"]["weight"].shape[0]
        self.embed_head = torch.nn.Linear(self.backbone.config.hidden_size, target_dim).to(device)
        self.embed_head.load_state_dict(heads["embed_head"])
        self.embed_head.eval()

    @torch.no_grad()
    def encode(self, texts: list[str]) -> torch.Tensor:
        batch = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        batch = {name: tensor.to(self.device) for name, tensor in batch.items()}
        outputs = self.backbone(**batch, return_dict=True)
        pooled = masked_mean(outputs.last_hidden_state, batch["attention_mask"])
        embeds = self.embed_head(pooled)
        return F.normalize(embeds, dim=-1)


def compute_radius(
    center: torch.Tensor,
    positives: torch.Tensor,
    radius_scale: float,
    min_radius: float,
) -> torch.Tensor:
    deltas = (positives - center.unsqueeze(0)).abs()
    radius = deltas.max(dim=0).values * radius_scale
    return torch.clamp(radius, min=min_radius)


def inside_fraction(embeds: torch.Tensor, center: torch.Tensor, radius: torch.Tensor) -> torch.Tensor:
    deltas = (embeds - center.unsqueeze(0)).abs()
    inside = deltas <= radius.unsqueeze(0)
    return inside.float().mean(dim=-1)


def evaluate_case(
    case: dict,
    teacher: TeacherEncoder,
    student: StudentEncoder,
    inside_threshold: float,
    radius_scale: float,
    min_radius: float,
) -> dict:
    query = case["query"]
    positives = case["positives"]
    negatives = case["negatives"]

    teacher_query = teacher.encode([query])[0]
    student_query = student.encode([query])[0]
    teacher_positives = teacher.encode(positives)
    teacher_negatives = teacher.encode(negatives)

    radius = compute_radius(
        center=teacher_query,
        positives=teacher_positives,
        radius_scale=radius_scale,
        min_radius=min_radius,
    )

    teacher_pos_frac = inside_fraction(teacher_positives, teacher_query, radius)
    teacher_neg_frac = inside_fraction(teacher_negatives, teacher_query, radius)
    student_pos_frac = inside_fraction(teacher_positives, student_query, radius)
    student_neg_frac = inside_fraction(teacher_negatives, student_query, radius)

    student_teacher_cos = F.cosine_similarity(
        student_query.unsqueeze(0),
        teacher_query.unsqueeze(0),
        dim=-1,
    ).item()

    return {
        "query": query,
        "student_teacher_cosine": student_teacher_cos,
        "teacher_positive_hit_rate": (teacher_pos_frac >= inside_threshold).float().mean().item(),
        "teacher_negative_false_positive_rate": (teacher_neg_frac >= inside_threshold).float().mean().item(),
        "student_positive_hit_rate": (student_pos_frac >= inside_threshold).float().mean().item(),
        "student_negative_false_positive_rate": (student_neg_frac >= inside_threshold).float().mean().item(),
        "teacher_positive_inside_fraction_mean": teacher_pos_frac.mean().item(),
        "teacher_negative_inside_fraction_mean": teacher_neg_frac.mean().item(),
        "student_positive_inside_fraction_mean": student_pos_frac.mean().item(),
        "student_negative_inside_fraction_mean": student_neg_frac.mean().item(),
    }


def summarize(results: list[dict]) -> dict:
    keys = [
        "student_teacher_cosine",
        "teacher_positive_hit_rate",
        "teacher_negative_false_positive_rate",
        "student_positive_hit_rate",
        "student_negative_false_positive_rate",
        "teacher_positive_inside_fraction_mean",
        "teacher_negative_inside_fraction_mean",
        "student_positive_inside_fraction_mean",
        "student_negative_inside_fraction_mean",
    ]
    return {
        key: sum(result[key] for result in results) / len(results)
        for key in keys
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark hypercube-style IN region behavior.")
    parser.add_argument("--cases", required=True, help="Path to a JSON file with query/positives/negatives cases.")
    parser.add_argument("--teacher-model", default="intfloat/e5-base-v2")
    parser.add_argument("--student-checkpoint", required=True)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--inside-threshold", type=float, default=0.95)
    parser.add_argument("--radius-scale", type=float, default=1.10)
    parser.add_argument("--min-radius", type=float, default=0.01)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.cases, "r", encoding="utf-8") as handle:
        cases = json.load(handle)
    if not isinstance(cases, list) or not cases:
        raise ValueError("Cases file must be a non-empty JSON list.")

    teacher = TeacherEncoder(args.teacher_model, device=device, max_length=args.max_length)
    student = StudentEncoder(args.student_checkpoint, device=device, max_length=args.max_length)

    results = [
        evaluate_case(
            case=case,
            teacher=teacher,
            student=student,
            inside_threshold=args.inside_threshold,
            radius_scale=args.radius_scale,
            min_radius=args.min_radius,
        )
        for case in cases
    ]

    print(json.dumps({"summary": summarize(results), "cases": results}, indent=2))


if __name__ == "__main__":
    main()
