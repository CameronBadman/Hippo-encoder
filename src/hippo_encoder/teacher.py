from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


class TextTeacher(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False

    @property
    def embedding_dim(self) -> int:
        return self.model.config.hidden_size

    @property
    def hidden_dim(self) -> int:
        return self.model.config.hidden_size

    @torch.no_grad()
    def encode(self, texts, device: torch.device, max_length: int, normalize: bool = True) -> dict:
        batch = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        batch = {name: tensor.to(device) for name, tensor in batch.items()}
        outputs = self.model(
            **batch,
            return_dict=True,
        )

        hidden_states = outputs.last_hidden_state
        text_embeds = self._masked_mean(hidden_states, batch["attention_mask"])
        if normalize:
            text_embeds = F.normalize(text_embeds, dim=-1)

        text_hidden = hidden_states[:, 0]
        if normalize:
            text_hidden = F.normalize(text_hidden, dim=-1)

        return {
            "text_embeds": text_embeds,
            "text_hidden": text_hidden,
        }

    @staticmethod
    def _masked_mean(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1)
        masked_hidden = hidden_states * mask
        denom = mask.sum(dim=1).clamp(min=1)
        return masked_hidden.sum(dim=1) / denom
