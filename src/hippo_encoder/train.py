from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from hippo_encoder.config import DistillConfig
from hippo_encoder.data import DistillJsonlDataset
from hippo_encoder.losses import (
    pair_distillation_loss,
    text_distillation_loss,
    triplet_distillation_loss,
)
from hippo_encoder.student import TinyEncoderStudent
from hippo_encoder.teacher import TextTeacher


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collate_fn(rows: list[dict]) -> dict:
    batch: dict = {
        "texts": [],
        "pairs": {"anchors": [], "positives": []},
        "triplets": {"anchors": [], "positives": [], "negatives": []},
    }
    for row in rows:
        if "text" in row:
            batch["texts"].append(row["text"])
        elif "negative" in row:
            batch["triplets"]["anchors"].append(row["anchor"])
            batch["triplets"]["positives"].append(row["positive"])
            batch["triplets"]["negatives"].append(row["negative"])
        else:
            batch["pairs"]["anchors"].append(row["anchor"])
            batch["pairs"]["positives"].append(row["positive"])
    return batch


def merge_weighted_metrics(metric_parts: list[tuple[dict, int]]) -> dict:
    total_weight = sum(weight for _, weight in metric_parts)
    if total_weight <= 0:
        raise ValueError("Cannot merge metrics from an empty batch.")

    keys = set().union(*(metrics.keys() for metrics, _ in metric_parts))
    merged = {}
    for key in keys:
        merged[key] = sum(metrics.get(key, 0.0) * weight for metrics, weight in metric_parts) / total_weight
    return merged


def train(config: DistillConfig) -> None:
    seed_everything(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = DistillJsonlDataset(config.dataset_jsonl)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
    )

    teacher = TextTeacher(config.teacher_model_name).to(device)
    if config.init_student_checkpoint:
        student = TinyEncoderStudent.load_checkpoint(
            config.init_student_checkpoint,
            device=device,
        )
    else:
        student = TinyEncoderStudent(
            model_name=config.student_model_name,
            target_dim=teacher.embedding_dim,
            hidden_target_dim=teacher.hidden_dim,
        ).to(device)

    optimizer = AdamW(
        student.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    total_steps = max(1, math.ceil(len(loader) * config.num_epochs))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=min(config.warmup_steps, total_steps),
        num_training_steps=total_steps,
    )

    global_step = 0
    for epoch in range(config.num_epochs):
        student.train()
        for batch in loader:
            loss_parts: list[tuple[torch.Tensor, int]] = []
            metric_parts: list[tuple[dict, int]] = []

            if batch["texts"]:
                teacher_outputs = teacher.encode(
                    texts=batch["texts"],
                    device=device,
                    max_length=config.max_text_length,
                    normalize=config.normalize_targets,
                )
                student_outputs = student(
                    texts=batch["texts"],
                    device=device,
                    max_length=config.max_text_length,
                )
                loss, metrics = text_distillation_loss(
                    student_outputs=student_outputs,
                    teacher_outputs=teacher_outputs,
                    teacher_text_weight=config.teacher_text_weight,
                    hidden_state_weight=config.hidden_state_weight,
                    contrastive_weight=config.contrastive_weight,
                    contrastive_temperature=config.contrastive_temperature,
                )
                loss_parts.append((loss, len(batch["texts"])))
                metric_parts.append((metrics, len(batch["texts"])))

            if batch["pairs"]["anchors"]:
                anchor_teacher = teacher.encode(
                    texts=batch["pairs"]["anchors"],
                    device=device,
                    max_length=config.max_text_length,
                    normalize=config.normalize_targets,
                )
                positive_teacher = teacher.encode(
                    texts=batch["pairs"]["positives"],
                    device=device,
                    max_length=config.max_text_length,
                    normalize=config.normalize_targets,
                )
                anchor_student = student(
                    texts=batch["pairs"]["anchors"],
                    device=device,
                    max_length=config.max_text_length,
                )
                positive_student = student(
                    texts=batch["pairs"]["positives"],
                    device=device,
                    max_length=config.max_text_length,
                )
                loss, metrics = pair_distillation_loss(
                    anchor_student=anchor_student,
                    positive_student=positive_student,
                    anchor_teacher=anchor_teacher,
                    positive_teacher=positive_teacher,
                    teacher_text_weight=config.teacher_text_weight,
                    hidden_state_weight=config.hidden_state_weight,
                    contrastive_weight=config.contrastive_weight,
                    contrastive_temperature=config.contrastive_temperature,
                )
                weight = len(batch["pairs"]["anchors"])
                loss_parts.append((loss, weight))
                metric_parts.append((metrics, weight))

            if batch["triplets"]["anchors"]:
                anchor_teacher = teacher.encode(
                    texts=batch["triplets"]["anchors"],
                    device=device,
                    max_length=config.max_text_length,
                    normalize=config.normalize_targets,
                )
                positive_teacher = teacher.encode(
                    texts=batch["triplets"]["positives"],
                    device=device,
                    max_length=config.max_text_length,
                    normalize=config.normalize_targets,
                )
                negative_teacher = teacher.encode(
                    texts=batch["triplets"]["negatives"],
                    device=device,
                    max_length=config.max_text_length,
                    normalize=config.normalize_targets,
                )
                anchor_student = student(
                    texts=batch["triplets"]["anchors"],
                    device=device,
                    max_length=config.max_text_length,
                )
                positive_student = student(
                    texts=batch["triplets"]["positives"],
                    device=device,
                    max_length=config.max_text_length,
                )
                negative_student = student(
                    texts=batch["triplets"]["negatives"],
                    device=device,
                    max_length=config.max_text_length,
                )
                loss, metrics = triplet_distillation_loss(
                    anchor_student=anchor_student,
                    positive_student=positive_student,
                    negative_student=negative_student,
                    anchor_teacher=anchor_teacher,
                    positive_teacher=positive_teacher,
                    negative_teacher=negative_teacher,
                    teacher_text_weight=config.teacher_text_weight,
                    hidden_state_weight=config.hidden_state_weight,
                    contrastive_weight=config.contrastive_weight,
                    contrastive_temperature=config.contrastive_temperature,
                    triplet_weight=config.triplet_weight,
                    triplet_margin=config.triplet_margin,
                )
                weight = len(batch["triplets"]["anchors"])
                loss_parts.append((loss, weight))
                metric_parts.append((metrics, weight))

            total_weight = sum(weight for _, weight in loss_parts)
            loss = sum(loss_part * weight for loss_part, weight in loss_parts) / total_weight
            metrics = merge_weighted_metrics(metric_parts)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), config.gradient_clip_norm)
            optimizer.step()
            scheduler.step()

            global_step += 1
            if global_step % config.log_every == 0:
                print(
                    f"epoch={epoch} step={global_step} "
                    f"loss={metrics['loss']:.4f} "
                    f"text={metrics['text_loss']:.4f} "
                    f"hidden={metrics['hidden_loss']:.4f} "
                    f"contrastive={metrics['contrastive_loss']:.4f} "
                    f"triplet={metrics.get('triplet_loss', 0.0):.4f}"
                )

            if global_step % config.save_every == 0:
                save_checkpoint(output_dir / f"step-{global_step}", student, config)

        save_checkpoint(output_dir / f"epoch-{epoch}", student, config)


def save_checkpoint(path: Path, student: TinyEncoderStudent, config: DistillConfig) -> None:
    path.mkdir(parents=True, exist_ok=True)
    student.backbone.save_pretrained(path / "backbone")
    student.tokenizer.save_pretrained(path / "tokenizer")
    torch.save(
        {
            "embed_head": student.embed_head.state_dict(),
            "hidden_head": student.hidden_head.state_dict(),
            "formula_head": student.formula_head.state_dict() if student.formula_head is not None else None,
            "formula_terms_per_side": student.formula_head.terms_per_side if student.formula_head is not None else 0,
            "config": config.__dict__,
        },
        path / "heads.pt",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Distill a text embedding teacher into a tiny text model.")
    parser.add_argument("--config", required=True, help="Path to a JSON config file.")
    args = parser.parse_args()
    train(DistillConfig.from_json(args.config))


if __name__ == "__main__":
    main()
