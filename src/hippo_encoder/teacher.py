from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoProcessor, CLIPModel


class ClipTeacher(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False

    @property
    def embedding_dim(self) -> int:
        return self.model.config.projection_dim

    @property
    def hidden_dim(self) -> int:
        return self.model.config.text_config.hidden_size

    @torch.no_grad()
    def encode(self, images, texts, device: torch.device, normalize: bool = True) -> dict:
        batch = self.processor(
            text=texts,
            images=images,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        batch = {name: tensor.to(device) for name, tensor in batch.items()}
        outputs = self.model(
            **batch,
            output_hidden_states=True,
            return_dict=True,
        )

        image_embeds = outputs.image_embeds
        text_embeds = outputs.text_embeds
        if normalize:
            image_embeds = F.normalize(image_embeds, dim=-1)
            text_embeds = F.normalize(text_embeds, dim=-1)

        # CLIP exposes pooled text encoder state through text_model_output.
        text_hidden = outputs.text_model_output.pooler_output
        if normalize:
            text_hidden = F.normalize(text_hidden, dim=-1)

        return {
            "image_embeds": image_embeds,
            "text_embeds": text_embeds,
            "text_hidden": text_hidden,
        }
