from __future__ import annotations

import torch
import torch.nn.functional as F


def text_distillation_loss(
    student_outputs: dict,
    teacher_outputs: dict,
    teacher_text_weight: float,
    hidden_state_weight: float,
    contrastive_weight: float,
    contrastive_temperature: float,
) -> tuple[torch.Tensor, dict]:
    text_loss = 1.0 - F.cosine_similarity(
        student_outputs["projected_embeds"],
        teacher_outputs["text_embeds"],
        dim=-1,
    ).mean()
    hidden_loss = 1.0 - F.cosine_similarity(
        student_outputs["predicted_hidden"],
        teacher_outputs["text_hidden"],
        dim=-1,
    ).mean()

    logits = student_outputs["projected_embeds"] @ teacher_outputs["text_embeds"].T
    logits = logits / max(contrastive_temperature, 1e-6)
    labels = torch.arange(logits.size(0), device=logits.device)
    contrastive_loss = F.cross_entropy(logits, labels)

    total = (
        teacher_text_weight * text_loss
        + hidden_state_weight * hidden_loss
        + contrastive_weight * contrastive_loss
    )

    metrics = {
        "loss": total.detach().item(),
        "text_loss": text_loss.detach().item(),
        "hidden_loss": hidden_loss.detach().item(),
        "contrastive_loss": contrastive_loss.detach().item(),
    }
    return total, metrics


def pair_distillation_loss(
    anchor_student: dict,
    positive_student: dict,
    anchor_teacher: dict,
    positive_teacher: dict,
    teacher_text_weight: float,
    hidden_state_weight: float,
    contrastive_weight: float,
    contrastive_temperature: float,
) -> tuple[torch.Tensor, dict]:
    anchor_text_loss = 1.0 - F.cosine_similarity(
        anchor_student["projected_embeds"],
        anchor_teacher["text_embeds"],
        dim=-1,
    ).mean()
    positive_text_loss = 1.0 - F.cosine_similarity(
        positive_student["projected_embeds"],
        positive_teacher["text_embeds"],
        dim=-1,
    ).mean()
    text_loss = 0.5 * (anchor_text_loss + positive_text_loss)

    anchor_hidden_loss = 1.0 - F.cosine_similarity(
        anchor_student["predicted_hidden"],
        anchor_teacher["text_hidden"],
        dim=-1,
    ).mean()
    positive_hidden_loss = 1.0 - F.cosine_similarity(
        positive_student["predicted_hidden"],
        positive_teacher["text_hidden"],
        dim=-1,
    ).mean()
    hidden_loss = 0.5 * (anchor_hidden_loss + positive_hidden_loss)

    logits = (anchor_student["projected_embeds"] @ positive_teacher["text_embeds"].T) / contrastive_temperature
    labels = torch.arange(logits.size(0), device=logits.device)
    forward_loss = F.cross_entropy(logits, labels)
    backward_logits = (positive_student["projected_embeds"] @ anchor_teacher["text_embeds"].T) / contrastive_temperature
    backward_loss = F.cross_entropy(backward_logits, labels)
    contrastive_loss = 0.5 * (forward_loss + backward_loss)

    total = (
        teacher_text_weight * text_loss
        + hidden_state_weight * hidden_loss
        + contrastive_weight * contrastive_loss
    )

    metrics = {
        "loss": total.detach().item(),
        "text_loss": text_loss.detach().item(),
        "hidden_loss": hidden_loss.detach().item(),
        "contrastive_loss": contrastive_loss.detach().item(),
    }
    return total, metrics
