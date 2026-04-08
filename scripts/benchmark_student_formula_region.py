from __future__ import annotations

import argparse
import json

import torch
import torch.nn.functional as F

from hippo_encoder.formula_region import soft_box_distance, inside_fraction
from hippo_encoder.student import TinyEncoderStudent
from hippo_encoder.teacher import TextTeacher


def evaluate_case(
    case: dict,
    student: TinyEncoderStudent,
    teacher: TextTeacher,
    device: torch.device,
    max_length: int,
    inside_threshold: float,
    base_radius: float,
    active_threshold: float,
) -> dict:
    query = case["query"]
    positives = case["positives"]
    negatives = case["negatives"]

    student_outputs = student(texts=[query], device=device, max_length=max_length)
    anchor = student_outputs["projected_embeds"][0]
    program = student.formula_head.decode_program(
        student_outputs["formula_outputs"],
        dimensions=anchor.shape[0],
        base_minus=base_radius,
        base_plus=base_radius,
        active_threshold=active_threshold,
    )
    region = program.hydrate(anchor)

    teacher_outputs = teacher.encode(
        [query] + positives + negatives,
        device=device,
        max_length=max_length,
        normalize=True,
    )
    teacher_query = teacher_outputs["text_embeds"][0:1]
    teacher_positives = teacher_outputs["text_embeds"][1 : 1 + len(positives)]
    teacher_negatives = teacher_outputs["text_embeds"][1 + len(positives) :]

    positive_frac = inside_fraction(teacher_positives, region["lower"], region["upper"])
    negative_frac = inside_fraction(teacher_negatives, region["lower"], region["upper"])
    query_cosine = F.cosine_similarity(anchor.unsqueeze(0), teacher_query, dim=-1).item()

    sample_terms = []
    for term in program.minus_terms[:6] + program.plus_terms[:6]:
        payload = f"{term.target} {term.term_type} {term.start} {term.end} amp={term.amplitude:.3f}"
        if term.term_type == "ramp":
            payload += f" start={term.start_value:.3f} end={term.end_value:.3f}"
        if term.term_type == "gaussian":
            payload += f" center={term.center_ratio:.3f} width={term.width_ratio:.3f}"
        sample_terms.append(payload)

    return {
        "query": query,
        "student_teacher_cosine": query_cosine,
        "minus_term_count": len(program.minus_terms),
        "plus_term_count": len(program.plus_terms),
        "positive_hit_rate": (positive_frac >= inside_threshold).float().mean().item(),
        "negative_false_positive_rate": (negative_frac >= inside_threshold).float().mean().item(),
        "positive_inside_fraction_mean": positive_frac.mean().item(),
        "negative_inside_fraction_mean": negative_frac.mean().item(),
        "positive_soft_distance_mean": soft_box_distance(
            teacher_positives, region["lower"], region["upper"]
        ).mean().item(),
        "negative_soft_distance_mean": soft_box_distance(
            teacher_negatives, region["lower"], region["upper"]
        ).mean().item(),
        "sample_terms": sample_terms,
    }


def summarize(results: list[dict]) -> dict:
    keys = [
        "student_teacher_cosine",
        "minus_term_count",
        "plus_term_count",
        "positive_hit_rate",
        "negative_false_positive_rate",
        "positive_inside_fraction_mean",
        "negative_inside_fraction_mean",
        "positive_soft_distance_mean",
        "negative_soft_distance_mean",
    ]
    return {key: sum(item[key] for item in results) / len(results) for key in keys}


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark a student-owned formula-region head.")
    parser.add_argument("--student-checkpoint", required=True)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--teacher-model", default="intfloat/e5-base-v2")
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--inside-threshold", type=float, default=0.9)
    parser.add_argument("--base-radius", type=float, default=0.01)
    parser.add_argument("--active-threshold", type=float, default=0.5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student = TinyEncoderStudent.load_checkpoint(args.student_checkpoint, device=device)
    if student.formula_head is None:
        raise ValueError("Checkpoint does not contain a formula head.")
    student.eval()
    teacher = TextTeacher(args.teacher_model).to(device)

    with open(args.cases, "r", encoding="utf-8") as handle:
        cases = json.load(handle)

    results = [
        evaluate_case(
            case=case,
            student=student,
            teacher=teacher,
            device=device,
            max_length=args.max_length,
            inside_threshold=args.inside_threshold,
            base_radius=args.base_radius,
            active_threshold=args.active_threshold,
        )
        for case in cases
    ]

    print(json.dumps({"summary": summarize(results), "cases": results}, indent=2))


if __name__ == "__main__":
    main()
