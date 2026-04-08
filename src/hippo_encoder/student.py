from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from hippo_encoder.formula_region import FormulaRegionProgram, RangedFormulaTerm, TERM_TYPES


class FormulaRegionHead(nn.Module):
    param_dim = 12

    def __init__(self, hidden_size: int, terms_per_side: int):
        super().__init__()
        self.terms_per_side = terms_per_side
        inner = max(128, min(hidden_size, 512))
        self.minus_head = nn.Sequential(
            nn.Linear(hidden_size, inner),
            nn.Tanh(),
            nn.Linear(inner, terms_per_side * self.param_dim),
        )
        self.plus_head = nn.Sequential(
            nn.Linear(hidden_size, inner),
            nn.Tanh(),
            nn.Linear(inner, terms_per_side * self.param_dim),
        )

    def forward(self, pooled_hidden: torch.Tensor) -> dict[str, dict[str, torch.Tensor]]:
        return {
            "minus": self._split_side(self.minus_head(pooled_hidden)),
            "plus": self._split_side(self.plus_head(pooled_hidden)),
        }

    def hydrate_soft_region(
        self,
        outputs: dict[str, dict[str, torch.Tensor]],
        anchor: torch.Tensor,
        base_minus: float,
        base_plus: float,
        sharpness: float = 40.0,
    ) -> dict[str, torch.Tensor]:
        if anchor.dim() == 1:
            anchor = anchor.unsqueeze(0)
        dimensions = anchor.shape[-1]
        x = torch.linspace(0.0, 1.0, steps=dimensions, device=anchor.device, dtype=anchor.dtype).view(1, 1, -1)

        minus = anchor.new_full(anchor.shape, base_minus)
        plus = anchor.new_full(anchor.shape, base_plus)
        minus = minus + self._soft_side_contrib(outputs["minus"], x, sharpness)
        plus = plus + self._soft_side_contrib(outputs["plus"], x, sharpness)
        minus = torch.clamp(minus, min=0.0)
        plus = torch.clamp(plus, min=0.0)

        return {
            "minus": minus,
            "plus": plus,
            "lower": anchor - minus,
            "upper": anchor + plus,
        }

    def decode_program(
        self,
        outputs: dict[str, dict[str, torch.Tensor]],
        dimensions: int,
        base_minus: float,
        base_plus: float,
        active_threshold: float = 0.5,
    ) -> FormulaRegionProgram:
        return FormulaRegionProgram(
            dimensions=dimensions,
            base_minus=base_minus,
            base_plus=base_plus,
            minus_terms=self._decode_side(outputs["minus"], dimensions, "minus", active_threshold),
            plus_terms=self._decode_side(outputs["plus"], dimensions, "plus", active_threshold),
        )

    def _decode_side(
        self,
        side: dict[str, torch.Tensor],
        dimensions: int,
        target_name: str,
        active_threshold: float,
    ) -> list[RangedFormulaTerm]:
        if side["active_logits"].dim() == 2:
            side = {name: tensor[0] for name, tensor in side.items()}

        active = torch.sigmoid(side["active_logits"])
        type_ids = side["type_logits"].argmax(dim=-1)
        denom = max(1, dimensions - 1)

        terms: list[RangedFormulaTerm] = []
        for slot_index in range(self.terms_per_side):
            if float(active[slot_index].item()) < active_threshold:
                continue

            start = int(round(float(torch.sigmoid(side["start"][slot_index]).item()) * denom))
            end = int(round(float(torch.sigmoid(side["end"][slot_index]).item()) * denom))
            start, end = sorted((start, end))
            term_type = TERM_TYPES[int(type_ids[slot_index].item())]

            amplitude = float(F.softplus(side["amplitude"][slot_index]).item())
            start_value = float(F.softplus(side["start_value"][slot_index]).item())
            end_value = float(F.softplus(side["end_value"][slot_index]).item())
            center_ratio = float(torch.sigmoid(side["center_ratio"][slot_index]).item())
            width_ratio = max(0.05, float(torch.sigmoid(side["width_ratio"][slot_index]).item()))

            term = RangedFormulaTerm(
                target=target_name,
                term_type=term_type,
                start=start,
                end=end,
                amplitude=amplitude,
                start_value=start_value if term_type == "ramp" else None,
                end_value=end_value if term_type == "ramp" else None,
                center_ratio=center_ratio if term_type == "gaussian" else None,
                width_ratio=width_ratio if term_type == "gaussian" else None,
            )
            terms.append(term)
        return terms

    def _split_side(self, tensor: torch.Tensor) -> dict[str, torch.Tensor]:
        tensor = tensor.view(tensor.shape[0], self.terms_per_side, self.param_dim)
        return {
            "active_logits": tensor[..., 0],
            "type_logits": tensor[..., 1:5],
            "start": tensor[..., 5],
            "end": tensor[..., 6],
            "amplitude": tensor[..., 7],
            "start_value": tensor[..., 8],
            "end_value": tensor[..., 9],
            "center_ratio": tensor[..., 10],
            "width_ratio": tensor[..., 11],
        }

    def _soft_side_contrib(
        self,
        side: dict[str, torch.Tensor],
        x: torch.Tensor,
        sharpness: float,
    ) -> torch.Tensor:
        active = torch.sigmoid(side["active_logits"]).unsqueeze(-1)
        type_probs = torch.softmax(side["type_logits"], dim=-1)

        raw_start = torch.sigmoid(side["start"]).unsqueeze(-1)
        raw_end = torch.sigmoid(side["end"]).unsqueeze(-1)
        start = torch.minimum(raw_start, raw_end)
        end = torch.maximum(raw_start, raw_end)
        span = (end - start).clamp(min=1e-3)

        left = torch.sigmoid(sharpness * (x - start))
        right = torch.sigmoid(sharpness * (end - x))
        mask = left * right

        amplitude = F.softplus(side["amplitude"]).unsqueeze(-1)
        start_value = F.softplus(side["start_value"]).unsqueeze(-1)
        end_value = F.softplus(side["end_value"]).unsqueeze(-1)
        center_ratio = torch.sigmoid(side["center_ratio"]).unsqueeze(-1)
        width_ratio = torch.sigmoid(side["width_ratio"]).unsqueeze(-1).clamp(min=0.05)

        local = ((x - start) / span).clamp(0.0, 1.0)
        box_values = amplitude * mask
        const_values = amplitude * mask
        ramp_values = (start_value + (end_value - start_value) * local) * mask
        gaussian_values = amplitude * torch.exp(-0.5 * ((local - center_ratio) / width_ratio) ** 2) * mask

        values = (
            type_probs[..., 0].unsqueeze(-1) * box_values
            + type_probs[..., 1].unsqueeze(-1) * const_values
            + type_probs[..., 2].unsqueeze(-1) * ramp_values
            + type_probs[..., 3].unsqueeze(-1) * gaussian_values
        )
        return (active * values).sum(dim=1)


class DenseDeltaHead(nn.Module):
    def __init__(self, hidden_size: int, target_dim: int):
        super().__init__()
        inner = max(256, min(hidden_size * 2, 1024))
        self.minus_head = nn.Sequential(
            nn.Linear(hidden_size, inner),
            nn.Tanh(),
            nn.Linear(inner, target_dim),
        )
        self.plus_head = nn.Sequential(
            nn.Linear(hidden_size, inner),
            nn.Tanh(),
            nn.Linear(inner, target_dim),
        )

    def forward(self, pooled_hidden: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "minus_raw": self.minus_head(pooled_hidden),
            "plus_raw": self.plus_head(pooled_hidden),
        }

    def hydrate_region(
        self,
        outputs: dict[str, torch.Tensor],
        anchor: torch.Tensor,
        base_minus: float,
        base_plus: float,
    ) -> dict[str, torch.Tensor]:
        if anchor.dim() == 1:
            anchor = anchor.unsqueeze(0)
        minus = torch.clamp(F.softplus(outputs["minus_raw"]) + base_minus, min=0.0)
        plus = torch.clamp(F.softplus(outputs["plus_raw"]) + base_plus, min=0.0)
        return {
            "minus": minus,
            "plus": plus,
            "lower": anchor - minus,
            "upper": anchor + plus,
        }


class TinyEncoderStudent(nn.Module):
    def __init__(
        self,
        model_name: str,
        target_dim: int,
        hidden_target_dim: int,
        formula_terms_per_side: int = 0,
        enable_dense_delta_head: bool = False,
        tokenizer_name: str | None = None,
        backbone_name: str | None = None,
    ):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name or model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.backbone = AutoModel.from_pretrained(backbone_name or model_name)
        hidden_size = self.backbone.config.hidden_size
        self.embed_head = nn.Linear(hidden_size, target_dim)
        self.hidden_head = nn.Linear(hidden_size, hidden_target_dim)
        self.formula_head = FormulaRegionHead(hidden_size, formula_terms_per_side) if formula_terms_per_side > 0 else None
        self.dense_delta_head = DenseDeltaHead(hidden_size, target_dim) if enable_dense_delta_head else None

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

        outputs = {
            "tokens": tokens,
            "pooled_hidden": pooled,
            "projected_embeds": F.normalize(self.embed_head(pooled), dim=-1),
            "predicted_hidden": F.normalize(self.hidden_head(pooled), dim=-1),
        }
        if self.formula_head is not None:
            outputs["formula_outputs"] = self.formula_head(pooled)
        if self.dense_delta_head is not None:
            outputs["dense_delta_outputs"] = self.dense_delta_head(pooled)
        return outputs

    @staticmethod
    def _masked_mean(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1)
        masked_hidden = hidden_states * mask
        denom = mask.sum(dim=1).clamp(min=1)
        return masked_hidden.sum(dim=1) / denom

    @classmethod
    def load_checkpoint(
        cls,
        checkpoint_dir: str | Path,
        device: torch.device,
        formula_terms_per_side: int | None = None,
    ) -> "TinyEncoderStudent":
        checkpoint_dir = Path(checkpoint_dir)
        heads = torch.load(checkpoint_dir / "heads.pt", map_location=device)
        target_dim = heads["embed_head"]["weight"].shape[0]
        hidden_target_dim = heads["hidden_head"]["weight"].shape[0]
        saved_terms = int(heads.get("formula_terms_per_side", 0))
        term_budget = saved_terms if formula_terms_per_side is None else formula_terms_per_side
        enable_dense_delta_head = heads.get("dense_delta_head") is not None

        student = cls(
            model_name=str(checkpoint_dir / "backbone"),
            target_dim=target_dim,
            hidden_target_dim=hidden_target_dim,
            formula_terms_per_side=term_budget,
            enable_dense_delta_head=enable_dense_delta_head,
            tokenizer_name=str(checkpoint_dir / "tokenizer"),
            backbone_name=str(checkpoint_dir / "backbone"),
        ).to(device)
        student.embed_head.load_state_dict(heads["embed_head"])
        student.hidden_head.load_state_dict(heads["hidden_head"])
        if student.formula_head is not None and heads.get("formula_head") is not None:
            student.formula_head.load_state_dict(heads["formula_head"])
        if student.dense_delta_head is not None and heads.get("dense_delta_head") is not None:
            student.dense_delta_head.load_state_dict(heads["dense_delta_head"])
        return student
