from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


class TinyEncoderStudent(nn.Module):
    def __init__(self, model_name: str, target_dim: int, hidden_target_dim: int):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.backbone = AutoModel.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size
        self.embed_head = nn.Linear(hidden_size, target_dim)
        self.hidden_head = nn.Linear(hidden_size, hidden_target_dim)

    def forward(self, texts, device: torch.device, max_length: int) -> dict:
        tokens = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        tokens = {name: tensor.to(device) for name, tensor in tokens.items()}
        outputs = self.backbone(**tokens, return_dict=True)
        hidden = outputs.last_hidden_state
        pooled = self._masked_mean(hidden, tokens["attention_mask"])

        return {
            "tokens": tokens,
            "pooled_hidden": pooled,
            "projected_embeds": F.normalize(self.embed_head(pooled), dim=-1),
            "predicted_hidden": F.normalize(self.hidden_head(pooled), dim=-1),
        }

    @staticmethod
    def _masked_mean(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1)
        masked_hidden = hidden_states * mask
        denom = mask.sum(dim=1).clamp(min=1)
        return masked_hidden.sum(dim=1) / denom
