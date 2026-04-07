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
from hippo_encoder.data import TextJsonlDataset
from hippo_encoder.losses import text_distillation_loss
from hippo_encoder.student import TinyEncoderStudent
from hippo_encoder.teacher import TextTeacher


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collate_fn(rows: list[dict]) -> dict:
    return {
        "texts": [row["text"] for row in rows],
    }


def train(config: DistillConfig) -> None:
    seed_everything(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = TextJsonlDataset(config.dataset_jsonl)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
    )

    teacher = TextTeacher(config.teacher_model_name).to(device)
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
            )

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
                    f"contrastive={metrics['contrastive_loss']:.4f}"
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
