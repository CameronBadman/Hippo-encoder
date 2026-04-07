from __future__ import annotations

import argparse
import json

import torch
from transformers import AutoModel, AutoTokenizer

from hippo_encoder.region import SparseRegionProgram


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
        return torch.nn.functional.normalize(embeds, dim=-1)


def dense_program_lines(program: SparseRegionProgram) -> tuple[list[str], list[str]]:
    zeros = torch.zeros(program.dimensions)
    hydrated = program.hydrate(zeros)
    minus = hydrated["minus"].tolist()
    plus = hydrated["plus"].tolist()

    minus_lines = [f"minus {index} {value:.2f}" for index, value in enumerate(minus)]
    plus_lines = [f"plus {index} {value:.2f}" for index, value in enumerate(plus)]
    return minus_lines, plus_lines


def sparse_program_lines(program: SparseRegionProgram) -> tuple[list[str], list[str]]:
    minus_lines = [
        f"{op.target} {op.mode} {op.start} {op.end} {op.value:+.2f}"
        for op in program.minus_ops
    ]
    plus_lines = [
        f"{op.target} {op.mode} {op.start} {op.end} {op.value:+.2f}"
        for op in program.plus_ops
    ]
    return minus_lines, plus_lines


def approx_tokens(lines: list[str]) -> int:
    # Rough estimate: one token per ~4 characters is a decent fast proxy.
    return sum(max(1, (len(line) + 3) // 4) for line in lines)


def evaluate_case(
    case: dict,
    teacher: TeacherEncoder,
    radius_scale: float,
    min_radius: float,
) -> dict:
    teacher_query = teacher.encode([case["query"]])[0]
    teacher_positives = teacher.encode(case["positives"])
    program = SparseRegionProgram.from_teacher_spread(
        anchor=teacher_query,
        positives=teacher_positives,
        base_radius=min_radius,
        radius_scale=radius_scale,
    )

    dense_minus, dense_plus = dense_program_lines(program)
    sparse_minus, sparse_plus = sparse_program_lines(program)

    dense_lines = dense_minus + dense_plus
    sparse_lines = sparse_minus + sparse_plus

    return {
        "query": case["query"],
        "dimensions": program.dimensions,
        "dense_line_count": len(dense_lines),
        "sparse_line_count": len(sparse_lines),
        "dense_token_estimate": approx_tokens(dense_lines),
        "sparse_token_estimate": approx_tokens(sparse_lines),
        "compression_ratio_lines": len(dense_lines) / max(1, len(sparse_lines)),
        "compression_ratio_tokens": approx_tokens(dense_lines) / max(1, approx_tokens(sparse_lines)),
        "minus_op_count": len(program.minus_ops),
        "plus_op_count": len(program.plus_ops),
        "sample_sparse_program": sparse_lines[:10],
    }


def summarize(results: list[dict]) -> dict:
    keys = [
        "dense_line_count",
        "sparse_line_count",
        "dense_token_estimate",
        "sparse_token_estimate",
        "compression_ratio_lines",
        "compression_ratio_tokens",
        "minus_op_count",
        "plus_op_count",
    ]
    return {key: sum(item[key] for item in results) / len(results) for key in keys}


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate dense-vs-sparse region program size.")
    parser.add_argument("--cases", required=True)
    parser.add_argument("--teacher-model", default="intfloat/e5-base-v2")
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--radius-scale", type=float, default=1.5)
    parser.add_argument("--min-radius", type=float, default=0.01)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher = TeacherEncoder(args.teacher_model, device=device, max_length=args.max_length)

    with open(args.cases, "r", encoding="utf-8") as handle:
        cases = json.load(handle)

    results = [
        evaluate_case(
            case=case,
            teacher=teacher,
            radius_scale=args.radius_scale,
            min_radius=args.min_radius,
        )
        for case in cases
    ]

    print(json.dumps({"summary": summarize(results), "cases": results}, indent=2))


if __name__ == "__main__":
    main()
