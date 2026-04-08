from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from hippo_encoder.rope_region import (
    DualRopeFormulaProgram,
    DualRopePointProgram,
    DualRopeShapeProgram,
    inside_fraction,
    soft_box_distance,
)
from hippo_encoder.student import TinyEncoderStudent


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
        self.student = TinyEncoderStudent.load_checkpoint(checkpoint_dir, device=device)
        self.student.eval()

    @torch.no_grad()
    def encode(self, texts: list[str]) -> torch.Tensor:
        outputs = self.student(texts=texts, device=self.device, max_length=self.max_length)
        return outputs["projected_embeds"]


def evaluate_case(
    case: dict,
    teacher: TeacherEncoder,
    student: StudentEncoder,
    inside_threshold: float,
    radius_scale: float,
    min_radius: float,
    terms_per_side: int,
    program_type: str,
) -> dict:
    query = case["query"]
    positives = case["positives"]
    negatives = case["negatives"]

    teacher_query = teacher.encode([query])[0]
    student_query = student.encode([query])[0]
    teacher_positives = teacher.encode(positives)
    teacher_negatives = teacher.encode(negatives)

    if program_type == "point":
        program = DualRopePointProgram.from_teacher_spread(
            anchor=teacher_query,
            positives=teacher_positives,
            terms_per_side=terms_per_side,
            base_radius=min_radius,
            radius_scale=radius_scale,
        )
    elif program_type == "shape":
        program = DualRopeShapeProgram.from_teacher_spread(
            anchor=teacher_query,
            positives=teacher_positives,
            terms_per_side=terms_per_side,
            base_radius=min_radius,
            radius_scale=radius_scale,
        )
    elif program_type == "formula":
        program = DualRopeFormulaProgram.from_teacher_spread(
            anchor=teacher_query,
            positives=teacher_positives,
            terms_per_side=terms_per_side,
            base_radius=min_radius,
            radius_scale=radius_scale,
        )
    else:
        raise ValueError(f"Unsupported program type: {program_type}")
    teacher_region = program.hydrate(teacher_query)
    student_region = program.hydrate(student_query)

    teacher_pos_frac = inside_fraction(teacher_positives, teacher_region["lower"], teacher_region["upper"])
    teacher_neg_frac = inside_fraction(teacher_negatives, teacher_region["lower"], teacher_region["upper"])
    student_pos_frac = inside_fraction(teacher_positives, student_region["lower"], student_region["upper"])
    student_neg_frac = inside_fraction(teacher_negatives, student_region["lower"], student_region["upper"])

    return {
        "query": query,
        "minus_point_count": len(program.minus_ops),
        "plus_point_count": len(program.plus_ops),
        "student_teacher_cosine": F.cosine_similarity(
            student_query.unsqueeze(0),
            teacher_query.unsqueeze(0),
            dim=-1,
        ).item(),
        "teacher_positive_hit_rate": (teacher_pos_frac >= inside_threshold).float().mean().item(),
        "teacher_negative_false_positive_rate": (teacher_neg_frac >= inside_threshold).float().mean().item(),
        "student_positive_hit_rate": (student_pos_frac >= inside_threshold).float().mean().item(),
        "student_negative_false_positive_rate": (student_neg_frac >= inside_threshold).float().mean().item(),
        "teacher_positive_inside_fraction_mean": teacher_pos_frac.mean().item(),
        "teacher_negative_inside_fraction_mean": teacher_neg_frac.mean().item(),
        "student_positive_inside_fraction_mean": student_pos_frac.mean().item(),
        "student_negative_inside_fraction_mean": student_neg_frac.mean().item(),
        "teacher_positive_soft_distance_mean": soft_box_distance(
            teacher_positives, teacher_region["lower"], teacher_region["upper"]
        ).mean().item(),
        "teacher_negative_soft_distance_mean": soft_box_distance(
            teacher_negatives, teacher_region["lower"], teacher_region["upper"]
        ).mean().item(),
        "student_positive_soft_distance_mean": soft_box_distance(
            teacher_positives, student_region["lower"], student_region["upper"]
        ).mean().item(),
        "student_negative_soft_distance_mean": soft_box_distance(
            teacher_negatives, student_region["lower"], student_region["upper"]
        ).mean().item(),
    }


def summarize(results: list[dict]) -> dict:
    keys = [
        "minus_point_count",
        "plus_point_count",
        "student_teacher_cosine",
        "teacher_positive_hit_rate",
        "teacher_negative_false_positive_rate",
        "student_positive_hit_rate",
        "student_negative_false_positive_rate",
        "teacher_positive_inside_fraction_mean",
        "teacher_negative_inside_fraction_mean",
        "student_positive_inside_fraction_mean",
        "student_negative_inside_fraction_mean",
        "teacher_positive_soft_distance_mean",
        "teacher_negative_soft_distance_mean",
        "student_positive_soft_distance_mean",
        "student_negative_soft_distance_mean",
    ]
    return {key: sum(result[key] for result in results) / len(results) for key in keys}


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark dual-rope region programs across term budgets.")
    parser.add_argument("--cases", required=True)
    parser.add_argument("--teacher-model", default="intfloat/e5-base-v2")
    parser.add_argument("--student-checkpoint", required=True)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--inside-threshold", type=float, default=0.9)
    parser.add_argument("--radius-scale", type=float, default=1.0)
    parser.add_argument("--min-radius", type=float, default=0.01)
    parser.add_argument("--budgets", type=int, nargs="+", default=[16, 32, 64, 128])
    parser.add_argument("--program-type", choices=("point", "shape", "formula"), default="point")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(args.cases, "r", encoding="utf-8") as handle:
        cases = json.load(handle)

    teacher = TeacherEncoder(args.teacher_model, device=device, max_length=args.max_length)
    student = StudentEncoder(args.student_checkpoint, device=device, max_length=args.max_length)

    payload: dict[str, object] = {"budgets": {}}
    for budget in args.budgets:
        results = [
            evaluate_case(
                case=case,
                teacher=teacher,
                student=student,
                inside_threshold=args.inside_threshold,
                radius_scale=args.radius_scale,
                min_radius=args.min_radius,
                terms_per_side=budget,
                program_type=args.program_type,
            )
            for case in cases
        ]
        payload["budgets"][str(budget)] = {
            "summary": summarize(results),
            "cases": results,
        }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
