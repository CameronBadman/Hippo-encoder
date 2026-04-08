from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup

from hippo_encoder.student import TinyEncoderStudent
from hippo_encoder.teacher import TextTeacher


DEFAULT_INSTRUCTION = (
    "You are shaping a latent region around the query. "
    "Include semantically matching paraphrases and close descriptions. "
    "Exclude unrelated concepts. Use enough margin to keep relevant items in."
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class RegionCaseDataset(Dataset):
    def __init__(self, path: str | Path):
        with open(path, "r", encoding="utf-8") as handle:
            self.rows = json.load(handle)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        return self.rows[index]


def collate_fn(rows: list[dict]) -> list[dict]:
    return rows


def batched_soft_box_distance(embeds: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    below = torch.relu(lower.unsqueeze(1) - embeds)
    above = torch.relu(embeds - upper.unsqueeze(1))
    return (below + above).mean(dim=-1)


def batched_inside_fraction(embeds: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    inside = (embeds >= lower.unsqueeze(1)) & (embeds <= upper.unsqueeze(1))
    return inside.float().mean(dim=-1)


def build_prompted_inputs(batch: list[dict], instruction: str) -> list[str]:
    return [f"instruction: {instruction}\nquery: {row['query']}" for row in batch]


def build_query_inputs(batch: list[dict]) -> list[str]:
    return [row["query"] for row in batch]


def encode_case_batch(
    teacher: TextTeacher,
    batch: list[dict],
    device: torch.device,
    max_length: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_pos = len(batch[0]["positives"])
    num_neg = len(batch[0]["negatives"])
    flat_texts = []
    for row in batch:
        flat_texts.append(row["query"])
        flat_texts.extend(row["positives"])
        flat_texts.extend(row["negatives"])

    outputs = teacher.encode(flat_texts, device=device, max_length=max_length, normalize=True)["text_embeds"]
    dim = outputs.shape[-1]
    stride = 1 + num_pos + num_neg
    outputs = outputs.view(len(batch), stride, dim)
    return outputs[:, 0], outputs[:, 1 : 1 + num_pos], outputs[:, 1 + num_pos :]


def save_checkpoint(path: Path, student: TinyEncoderStudent) -> None:
    path.mkdir(parents=True, exist_ok=True)
    student.backbone.save_pretrained(path / "backbone")
    student.tokenizer.save_pretrained(path / "tokenizer")
    torch.save(
        {
            "embed_head": student.embed_head.state_dict(),
            "hidden_head": student.hidden_head.state_dict(),
            "formula_head": student.formula_head.state_dict() if student.formula_head is not None else None,
            "formula_terms_per_side": student.formula_head.terms_per_side if student.formula_head is not None else 0,
            "dense_delta_head": student.dense_delta_head.state_dict() if student.dense_delta_head is not None else None,
        },
        path / "heads.pt",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a prompt-conditioned dense delta head on region behavior.")
    parser.add_argument("--student-checkpoint", required=True)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--teacher-model", default="intfloat/e5-base-v2")
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--base-radius", type=float, default=0.02)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--save-every-epochs", type=int, default=5)
    parser.add_argument("--negative-margin", type=float, default=0.02)
    parser.add_argument("--negative-weight", type=float, default=2.0)
    parser.add_argument("--positive-inside-target", type=float, default=0.75)
    parser.add_argument("--positive-coverage-weight", type=float, default=1.0)
    parser.add_argument("--size-weight", type=float, default=0.0)
    parser.add_argument("--anchor-weight", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freeze-backbone", action="store_true")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = RegionCaseDataset(args.cases)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    base_student = TinyEncoderStudent.load_checkpoint(args.student_checkpoint, device=device)
    student = TinyEncoderStudent(
        model_name=str(Path(args.student_checkpoint) / "backbone"),
        target_dim=base_student.embed_head.weight.shape[0],
        hidden_target_dim=base_student.hidden_head.weight.shape[0],
        enable_dense_delta_head=True,
        tokenizer_name=str(Path(args.student_checkpoint) / "tokenizer"),
        backbone_name=str(Path(args.student_checkpoint) / "backbone"),
    ).to(device)
    student.embed_head.load_state_dict(base_student.embed_head.state_dict())
    student.hidden_head.load_state_dict(base_student.hidden_head.state_dict())
    student.train()

    if args.freeze_backbone:
        for module in (student.backbone, student.embed_head, student.hidden_head):
            for parameter in module.parameters():
                parameter.requires_grad = False

    teacher = TextTeacher(args.teacher_model).to(device)
    optimizer = AdamW(
        [parameter for parameter in student.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    total_steps = max(1, len(loader) * args.num_epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=min(args.warmup_steps, total_steps),
        num_training_steps=total_steps,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    for epoch in range(args.num_epochs):
        for batch in loader:
            teacher_query, teacher_positives, teacher_negatives = encode_case_batch(
                teacher=teacher,
                batch=batch,
                device=device,
                max_length=args.max_length,
            )
            query_inputs = build_query_inputs(batch)
            prompted_inputs = build_prompted_inputs(batch, args.instruction)
            query_outputs = student(texts=query_inputs, device=device, max_length=args.max_length)
            prompted_outputs = student(texts=prompted_inputs, device=device, max_length=args.max_length)
            anchor = query_outputs["projected_embeds"]
            region = student.dense_delta_head.hydrate_region(
                prompted_outputs["dense_delta_outputs"],
                anchor=anchor,
                base_minus=args.base_radius,
                base_plus=args.base_radius,
            )

            pos_dist = batched_soft_box_distance(teacher_positives, region["lower"], region["upper"])
            neg_dist = batched_soft_box_distance(teacher_negatives, region["lower"], region["upper"])
            pos_inside = batched_inside_fraction(teacher_positives, region["lower"], region["upper"])

            positive_loss = pos_dist.mean()
            positive_coverage_loss = torch.relu(args.positive_inside_target - pos_inside).mean()
            negative_loss = torch.relu(args.negative_margin - neg_dist).mean()
            size_loss = (region["minus"].mean() + region["plus"].mean())
            anchor_loss = (1.0 - F.cosine_similarity(anchor, teacher_query, dim=-1)).mean()
            loss = (
                positive_loss
                + args.positive_coverage_weight * positive_coverage_loss
                + args.negative_weight * negative_loss
                + args.size_weight * size_loss
                + args.anchor_weight * anchor_loss
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            global_step += 1
            if global_step % 10 == 0:
                print(
                    f"epoch={epoch} step={global_step} "
                    f"loss={loss.detach().item():.4f} "
                    f"pos={positive_loss.detach().item():.4f} "
                    f"cover={positive_coverage_loss.detach().item():.4f} "
                    f"neg={negative_loss.detach().item():.4f} "
                    f"size={size_loss.detach().item():.4f} "
                    f"anchor={anchor_loss.detach().item():.4f}"
                )

        should_save = ((epoch + 1) % max(1, args.save_every_epochs) == 0) or (epoch == args.num_epochs - 1)
        if should_save:
            save_checkpoint(output_dir / f"epoch-{epoch}", student)


if __name__ == "__main__":
    main()
