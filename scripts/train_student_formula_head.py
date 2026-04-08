from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from hippo_encoder.formula_region import FormulaRegionProgram, encode_program_slots
from hippo_encoder.student import TinyEncoderStudent
from hippo_encoder.teacher import TextTeacher


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class RegionCaseDataset(torch.utils.data.Dataset):
    def __init__(self, path: str | Path):
        with open(path, "r", encoding="utf-8") as handle:
            self.rows = json.load(handle)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        return self.rows[index]


def collate_fn(rows: list[dict]) -> list[dict]:
    return rows


def build_target_programs(
    batch: list[dict],
    teacher: TextTeacher,
    device: torch.device,
    max_length: int,
    base_radius: float,
    radius_scale: float,
    formula_terms_per_side: int,
) -> list[FormulaRegionProgram]:
    programs: list[FormulaRegionProgram] = []
    for case in batch:
        query_embed = teacher.encode([case["query"]], device=device, max_length=max_length, normalize=True)["text_embeds"][0]
        positives = teacher.encode(case["positives"], device=device, max_length=max_length, normalize=True)["text_embeds"]
        negatives = teacher.encode(case["negatives"], device=device, max_length=max_length, normalize=True)["text_embeds"]
        programs.append(
            FormulaRegionProgram.from_teacher_spread(
                anchor=query_embed,
                positives=positives,
                negatives=negatives,
                base_radius=base_radius,
                radius_scale=radius_scale,
                max_terms_per_side=formula_terms_per_side,
            )
        )
    return programs


def formula_head_loss(
    predicted: dict[str, dict[str, torch.Tensor]],
    targets: dict[str, dict[str, torch.Tensor]],
) -> tuple[torch.Tensor, dict[str, float]]:
    total = predicted["minus"]["active_logits"].new_tensor(0.0)
    metrics: dict[str, float] = {}

    for side in ("minus", "plus"):
        pred = predicted[side]
        tgt = targets[side]
        active = tgt["active"].to(pred["active_logits"].device)
        type_id = tgt["type_id"].to(pred["type_logits"].device)

        active_loss = F.binary_cross_entropy_with_logits(pred["active_logits"], active)

        type_loss_raw = F.cross_entropy(
            pred["type_logits"].transpose(1, 2),
            type_id,
            reduction="none",
        )
        type_loss = (type_loss_raw * active).sum() / active.sum().clamp(min=1.0)

        start_loss = (F.l1_loss(torch.sigmoid(pred["start"]), tgt["start"].to(pred["start"].device), reduction="none") * active).sum()
        end_loss = (F.l1_loss(torch.sigmoid(pred["end"]), tgt["end"].to(pred["end"].device), reduction="none") * active).sum()
        span_loss = (start_loss + end_loss) / active.sum().clamp(min=1.0)

        amplitude_mask = tgt["amplitude_mask"].to(pred["amplitude"].device)
        amplitude_loss = (
            F.l1_loss(F.softplus(pred["amplitude"]), tgt["amplitude"].to(pred["amplitude"].device), reduction="none")
            * amplitude_mask
        ).sum() / amplitude_mask.sum().clamp(min=1.0)

        ramp_mask = tgt["ramp_mask"].to(pred["start_value"].device)
        ramp_loss = (
            (
                F.l1_loss(
                    F.softplus(pred["start_value"]),
                    tgt["start_value"].to(pred["start_value"].device),
                    reduction="none",
                )
                + F.l1_loss(
                    F.softplus(pred["end_value"]),
                    tgt["end_value"].to(pred["end_value"].device),
                    reduction="none",
                )
            )
            * ramp_mask
        ).sum() / ramp_mask.sum().clamp(min=1.0)

        gaussian_mask = tgt["gaussian_mask"].to(pred["center_ratio"].device)
        gaussian_loss = (
            (
                F.l1_loss(
                    torch.sigmoid(pred["center_ratio"]),
                    tgt["center_ratio"].to(pred["center_ratio"].device),
                    reduction="none",
                )
                + F.l1_loss(
                    torch.sigmoid(pred["width_ratio"]),
                    tgt["width_ratio"].to(pred["width_ratio"].device),
                    reduction="none",
                )
            )
            * gaussian_mask
        ).sum() / gaussian_mask.sum().clamp(min=1.0)

        side_loss = active_loss + type_loss + span_loss + amplitude_loss + ramp_loss + gaussian_loss
        total = total + side_loss

        metrics[f"{side}_active_loss"] = float(active_loss.detach().item())
        metrics[f"{side}_type_loss"] = float(type_loss.detach().item())
        metrics[f"{side}_span_loss"] = float(span_loss.detach().item())
        metrics[f"{side}_amplitude_loss"] = float(amplitude_loss.detach().item())
        metrics[f"{side}_ramp_loss"] = float(ramp_loss.detach().item())
        metrics[f"{side}_gaussian_loss"] = float(gaussian_loss.detach().item())

    metrics["loss"] = float(total.detach().item())
    return total, metrics


def save_checkpoint(
    path: Path,
    student: TinyEncoderStudent,
    formula_terms_per_side: int,
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    student.backbone.save_pretrained(path / "backbone")
    student.tokenizer.save_pretrained(path / "tokenizer")
    torch.save(
        {
            "embed_head": student.embed_head.state_dict(),
            "hidden_head": student.hidden_head.state_dict(),
            "formula_head": student.formula_head.state_dict() if student.formula_head is not None else None,
            "formula_terms_per_side": formula_terms_per_side,
        },
        path / "heads.pt",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a native student formula-region head from query/positive/negative cases.")
    parser.add_argument("--student-checkpoint", required=True)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--teacher-model", default="intfloat/e5-base-v2")
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--formula-terms-per-side", type=int, default=12)
    parser.add_argument("--base-radius", type=float, default=0.01)
    parser.add_argument("--radius-scale", type=float, default=1.5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--num-epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--save-every-epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freeze-backbone", action="store_true")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = RegionCaseDataset(args.cases)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    student = TinyEncoderStudent.load_checkpoint(
        args.student_checkpoint,
        device=device,
        formula_terms_per_side=args.formula_terms_per_side,
    )
    if student.formula_head is None:
        raise ValueError("Formula head failed to initialize.")
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
            target_programs = build_target_programs(
                batch=batch,
                teacher=teacher,
                device=device,
                max_length=args.max_length,
                base_radius=args.base_radius,
                radius_scale=args.radius_scale,
                formula_terms_per_side=args.formula_terms_per_side,
            )
            target_slots = [encode_program_slots(program, args.formula_terms_per_side) for program in target_programs]
            student_outputs = student(
                texts=[row["query"] for row in batch],
                device=device,
                max_length=args.max_length,
            )
            predicted = student_outputs["formula_outputs"]

            stacked_targets = {
                side: {
                    name: torch.stack([slots[side][name] for slots in target_slots]).to(device)
                    for name in target_slots[0][side]
                }
                for side in ("minus", "plus")
            }

            loss, metrics = formula_head_loss(predicted, stacked_targets)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            global_step += 1
            if global_step % 10 == 0:
                print(
                    f"epoch={epoch} step={global_step} loss={metrics['loss']:.4f} "
                    f"minus_active={metrics['minus_active_loss']:.4f} "
                    f"plus_active={metrics['plus_active_loss']:.4f}"
                )

        should_save = ((epoch + 1) % max(1, args.save_every_epochs) == 0) or (epoch == args.num_epochs - 1)
        if should_save:
            save_checkpoint(output_dir / f"epoch-{epoch}", student, args.formula_terms_per_side)


if __name__ == "__main__":
    main()
