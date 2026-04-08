from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from hippo_encoder.formula_region import (
    FormulaRegionProgram,
    extract_json_object,
    inside_fraction,
    soft_box_distance,
)


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


class ProgramGenerator:
    def __init__(self, model_name: str, device: torch.device, max_new_tokens: int):
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device).eval()

    @torch.no_grad()
    def generate(self, prompt: str) -> str:
        tokens = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        output = self.model.generate(
            **tokens,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return text[len(prompt) :].strip() if text.startswith(prompt) else text


def build_prompt(spec: str, query: str, positives: list[str], negatives: list[str], dimensions: int) -> str:
    return (
        f"{spec}\n\n"
        "Task:\n"
        f"- Dimensions: 0..{dimensions - 1}\n"
        f"- Query: {query}\n"
        f"- Positives that should be IN: {json.dumps(positives)}\n"
        f"- Negatives that should be OUT: {json.dumps(negatives)}\n\n"
        "Return JSON only.\n"
    )


def evaluate_case(
    case: dict,
    encoder: Encoder,
    generator: ProgramGenerator,
    prompt_spec: str,
    inside_threshold: float,
    base_radius: float,
) -> dict:
    query = case["query"]
    positives = case["positives"]
    negatives = case["negatives"]

    query_embed = encoder.encode([query])[0]
    positive_embeds = encoder.encode(positives)
    negative_embeds = encoder.encode(negatives)

    prompt = build_prompt(prompt_spec, query, positives, negatives, dimensions=query_embed.shape[0])
    raw_output = generator.generate(prompt)
    payload = extract_json_object(raw_output)
    program = FormulaRegionProgram.from_generated_dict(
        payload,
        dimensions=query_embed.shape[0],
        base_minus=base_radius,
        base_plus=base_radius,
    )
    region = program.hydrate(query_embed)

    positive_frac = inside_fraction(positive_embeds, region["lower"], region["upper"])
    negative_frac = inside_fraction(negative_embeds, region["lower"], region["upper"])
    token_estimate = max(1, (len(raw_output) + 3) // 4)

    return {
        "query": query,
        "minus_term_count": len(program.minus_terms),
        "plus_term_count": len(program.plus_terms),
        "generated_token_estimate": token_estimate,
        "positive_hit_rate": (positive_frac >= inside_threshold).float().mean().item(),
        "negative_false_positive_rate": (negative_frac >= inside_threshold).float().mean().item(),
        "positive_inside_fraction_mean": positive_frac.mean().item(),
        "negative_inside_fraction_mean": negative_frac.mean().item(),
        "positive_soft_distance_mean": soft_box_distance(
            positive_embeds, region["lower"], region["upper"]
        ).mean().item(),
        "negative_soft_distance_mean": soft_box_distance(
            negative_embeds, region["lower"], region["upper"]
        ).mean().item(),
        "raw_output": raw_output[:1200],
    }


def summarize(results: list[dict]) -> dict:
    keys = [
        "minus_term_count",
        "plus_term_count",
        "generated_token_estimate",
        "positive_hit_rate",
        "negative_false_positive_rate",
        "positive_inside_fraction_mean",
        "negative_inside_fraction_mean",
        "positive_soft_distance_mean",
        "negative_soft_distance_mean",
    ]
    return {key: sum(item[key] for item in results) / len(results) for key in keys}


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark direct generation of ranged-formula region programs.")
    parser.add_argument("--cases", required=True)
    parser.add_argument("--teacher-model", default="intfloat/e5-base-v2")
    parser.add_argument("--generator-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--prompt-file", default="prompts/ranged_formula_region_prompt.md")
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--inside-threshold", type=float, default=0.9)
    parser.add_argument("--base-radius", type=float, default=0.01)
    parser.add_argument("--max-new-tokens", type=int, default=600)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = Encoder(args.teacher_model, device=device, max_length=args.max_length)
    generator = ProgramGenerator(args.generator_model, device=device, max_new_tokens=args.max_new_tokens)
    prompt_spec = Path(args.prompt_file).read_text(encoding="utf-8")

    with open(args.cases, "r", encoding="utf-8") as handle:
        cases = json.load(handle)

    results = [
        evaluate_case(
            case=case,
            encoder=encoder,
            generator=generator,
            prompt_spec=prompt_spec,
            inside_threshold=args.inside_threshold,
            base_radius=args.base_radius,
        )
        for case in cases
    ]

    print(json.dumps({"summary": summarize(results), "cases": results}, indent=2))


if __name__ == "__main__":
    main()
