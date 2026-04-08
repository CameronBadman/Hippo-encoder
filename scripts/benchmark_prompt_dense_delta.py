from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from hippo_encoder.student import TinyEncoderStudent


DEFAULT_INSTRUCTION = (
    "You are shaping a latent region around the query. "
    "Include semantically matching paraphrases and close descriptions. "
    "Exclude unrelated concepts. Use enough margin to keep relevant items in."
)


def masked_mean(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).float()
    masked = hidden_states * mask
    denom = mask.sum(dim=1).clamp(min=1.0)
    return masked.sum(dim=1) / denom


def inside_fraction(embeds: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    inside = (embeds >= lower.unsqueeze(0)) & (embeds <= upper.unsqueeze(0))
    return inside.float().mean(dim=-1)


def soft_box_distance(embeds: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    below = torch.relu(lower.unsqueeze(0) - embeds)
    above = torch.relu(embeds - upper.unsqueeze(0))
    return (below + above).mean(dim=-1)


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


def build_prompted_query(instruction: str, query: str) -> str:
    return f"instruction: {instruction}\nquery: {query}"


def evaluate_case(
    case: dict,
    teacher: TeacherEncoder,
    student: TinyEncoderStudent,
    device: torch.device,
    max_length: int,
    instruction: str,
    inside_threshold: float,
    base_radius: float,
) -> dict:
    query = case["query"]
    positives = case["positives"]
    negatives = case["negatives"]

    prompted_query = build_prompted_query(instruction, query)
    query_outputs = student(texts=[query], device=device, max_length=max_length)
    prompted_outputs = student(texts=[prompted_query], device=device, max_length=max_length)
    anchor = query_outputs["projected_embeds"][0]
    region = student.dense_delta_head.hydrate_region(
        prompted_outputs["dense_delta_outputs"],
        anchor=anchor,
        base_minus=base_radius,
        base_plus=base_radius,
    )
    region = {name: tensor[0] if tensor.dim() == 2 else tensor for name, tensor in region.items()}

    teacher_outputs = teacher.encode([query] + positives + negatives)
    teacher_query = teacher_outputs[0]
    teacher_positives = teacher_outputs[1 : 1 + len(positives)]
    teacher_negatives = teacher_outputs[1 + len(positives) :]

    positive_frac = inside_fraction(teacher_positives, region["lower"], region["upper"])
    negative_frac = inside_fraction(teacher_negatives, region["lower"], region["upper"])

    return {
        "query": query,
        "student_teacher_cosine": F.cosine_similarity(anchor.unsqueeze(0), teacher_query.unsqueeze(0), dim=-1).item(),
        "positive_hit_rate": (positive_frac >= inside_threshold).float().mean().item(),
        "negative_false_positive_rate": (negative_frac >= inside_threshold).float().mean().item(),
        "positive_inside_fraction_mean": positive_frac.mean().item(),
        "negative_inside_fraction_mean": negative_frac.mean().item(),
        "positive_soft_distance_mean": soft_box_distance(teacher_positives, region["lower"], region["upper"]).mean().item(),
        "negative_soft_distance_mean": soft_box_distance(teacher_negatives, region["lower"], region["upper"]).mean().item(),
        "mean_minus": region["minus"].mean().item(),
        "mean_plus": region["plus"].mean().item(),
    }


def summarize(results: list[dict]) -> dict:
    keys = [
        "student_teacher_cosine",
        "positive_hit_rate",
        "negative_false_positive_rate",
        "positive_inside_fraction_mean",
        "negative_inside_fraction_mean",
        "positive_soft_distance_mean",
        "negative_soft_distance_mean",
        "mean_minus",
        "mean_plus",
    ]
    return {key: sum(result[key] for result in results) / len(results) for key in keys}


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark a prompt-conditioned dense delta head.")
    parser.add_argument("--student-checkpoint", required=True)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--teacher-model", default="intfloat/e5-base-v2")
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--inside-threshold", type=float, default=0.9)
    parser.add_argument("--base-radius", type=float, default=0.01)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student = TinyEncoderStudent.load_checkpoint(args.student_checkpoint, device=device)
    if student.dense_delta_head is None:
        raise ValueError("Checkpoint does not contain a dense delta head.")
    student.eval()
    teacher = TeacherEncoder(args.teacher_model, device=device, max_length=args.max_length)

    with open(args.cases, "r", encoding="utf-8") as handle:
        cases = json.load(handle)

    results = [
        evaluate_case(
            case=case,
            teacher=teacher,
            student=student,
            device=device,
            max_length=args.max_length,
            instruction=args.instruction,
            inside_threshold=args.inside_threshold,
            base_radius=args.base_radius,
        )
        for case in cases
    ]
    print(json.dumps({"summary": summarize(results), "cases": results}, indent=2))


if __name__ == "__main__":
    main()
