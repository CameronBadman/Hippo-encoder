from __future__ import annotations

import torch
import torch.nn.functional as F


def text_distillation_loss(
    student_outputs: dict,
    teacher_outputs: dict,
    teacher_text_weight: float,
    hidden_state_weight: float,
    contrastive_weight: float,
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
