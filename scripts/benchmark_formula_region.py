from __future__ import annotations

import argparse
import json

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from hippo_encoder.formula_region import FormulaRegionProgram, inside_fraction, soft_box_distance


def masked_mean(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).float()
    masked = hidden_states * mask
    denom = mask.sum(dim=1).clamp(min=1.0)
    return masked.sum(dim=1) / denom


class Encoder:
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


def term_lines(program: FormulaRegionProgram) -> list[str]:
    lines = []
    for term in program.minus_terms + program.plus_terms:
        payload = f"{term.target} {term.term_type} {term.start} {term.end} amp={term.amplitude:.3f}"
        if term.term_type == "ramp":
            payload += f" start={term.start_value:.3f} end={term.end_value:.3f}"
        if term.term_type == "gaussian":
            payload += f" center={term.center_ratio:.3f} width={term.width_ratio:.3f}"
        lines.append(payload)
    return lines


def approx_tokens(lines: list[str]) -> int:
    return sum(max(1, (len(line) + 3) // 4) for line in lines)


def evaluate_case(
    case: dict,
    encoder: Encoder,
    inside_threshold: float,
    radius_scale: float,
    min_radius: float,
    max_terms_per_side: int,
) -> dict:
    query = case["query"]
    positives = case["positives"]
    negatives = case["negatives"]

    query_embed = encoder.encode([query])[0]
    positive_embeds = encoder.encode(positives)
    negative_embeds = encoder.encode(negatives)

    program = FormulaRegionProgram.from_teacher_spread(
        anchor=query_embed,
        positives=positive_embeds,
        negatives=negative_embeds,
        base_radius=min_radius,
        radius_scale=radius_scale,
        max_terms_per_side=max_terms_per_side,
    )
    region = program.hydrate(query_embed)

    positive_frac = inside_fraction(positive_embeds, region["lower"], region["upper"])
    negative_frac = inside_fraction(negative_embeds, region["lower"], region["upper"])
    lines = term_lines(program)

    return {
        "query": query,
        "minus_term_count": len(program.minus_terms),
        "plus_term_count": len(program.plus_terms),
        "formula_token_estimate": approx_tokens(lines),
        "positive_hit_rate": (positive_frac >= inside_threshold).float().mean().item(),
        "negative_false_positive_rate": (negative_frac >= inside_threshold).float().mean().item(),
        "positive_inside_fraction_mean": positive_frac.mean().item(),
        "negative_inside_fraction_mean": negative_frac.mean().item(),
        "positive_soft_distance_mean": soft_box_distance(
            positive_embeds,
            region["lower"],
            region["upper"],
        ).mean().item(),
        "negative_soft_distance_mean": soft_box_distance(
            negative_embeds,
            region["lower"],
            region["upper"],
        ).mean().item(),
        "sample_terms": lines[:12],
    }


def summarize(results: list[dict]) -> dict:
    keys = [
        "minus_term_count",
        "plus_term_count",
        "formula_token_estimate",
        "positive_hit_rate",
        "negative_false_positive_rate",
        "positive_inside_fraction_mean",
        "negative_inside_fraction_mean",
        "positive_soft_distance_mean",
        "negative_soft_distance_mean",
    ]
    return {key: sum(item[key] for item in results) / len(results) for key in keys}


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark ranged-formula region programs.")
    parser.add_argument("--cases", required=True)
    parser.add_argument("--teacher-model", default="intfloat/e5-base-v2")
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--inside-threshold", type=float, default=0.75)
    parser.add_argument("--radius-scale", type=float, default=1.5)
    parser.add_argument("--min-radius", type=float, default=0.01)
    parser.add_argument("--max-terms-per-side", type=int, default=12)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = Encoder(args.teacher_model, device=device, max_length=args.max_length)

    with open(args.cases, "r", encoding="utf-8") as handle:
        cases = json.load(handle)

    results = [
        evaluate_case(
            case=case,
            encoder=encoder,
            inside_threshold=args.inside_threshold,
            radius_scale=args.radius_scale,
            min_radius=args.min_radius,
            max_terms_per_side=args.max_terms_per_side,
        )
        for case in cases
    ]

    print(json.dumps({"summary": summarize(results), "cases": results}, indent=2))


if __name__ == "__main__":
    main()
