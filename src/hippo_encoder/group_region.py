from __future__ import annotations

from dataclasses import asdict, dataclass

import torch


@dataclass
class GroupRegionProgram:
    dimensions: int
    group_size: int
    base_minus: float
    base_plus: float
    minus_groups: list[float]
    plus_groups: list[float]

    @property
    def num_groups(self) -> int:
        return len(self.minus_groups)

    def hydrate(self, anchor: torch.Tensor) -> dict[str, torch.Tensor]:
        if anchor.dim() != 1 or anchor.shape[0] != self.dimensions:
            raise ValueError("Anchor must be a 1D tensor matching program dimensions.")

        minus = self._expand_groups(self.minus_groups, device=anchor.device, dtype=anchor.dtype)
        plus = self._expand_groups(self.plus_groups, device=anchor.device, dtype=anchor.dtype)
        minus = minus[: self.dimensions]
        plus = plus[: self.dimensions]

        return {
            "minus": minus,
            "plus": plus,
            "lower": anchor - minus,
            "upper": anchor + plus,
        }

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_teacher_spread(
        cls,
        anchor: torch.Tensor,
        positives: torch.Tensor,
        group_size: int = 16,
        base_radius: float = 0.01,
        radius_scale: float = 1.0,
        quantize_step: float = 0.01,
    ) -> "GroupRegionProgram":
        if anchor.dim() != 1:
            raise ValueError("Anchor must be a 1D tensor.")
        if positives.dim() != 2 or positives.shape[1] != anchor.shape[0]:
            raise ValueError("Positives must have shape [N, D] with same D as anchor.")

        deltas = positives - anchor.unsqueeze(0)
        minus = torch.clamp((-deltas).max(dim=0).values * radius_scale, min=base_radius)
        plus = torch.clamp(deltas.max(dim=0).values * radius_scale, min=base_radius)

        minus_groups = _pool_groups(minus, group_size=group_size, reduction="max")
        plus_groups = _pool_groups(plus, group_size=group_size, reduction="max")

        minus_groups = [round(float(v) / quantize_step) * quantize_step for v in minus_groups]
        plus_groups = [round(float(v) / quantize_step) * quantize_step for v in plus_groups]

        return cls(
            dimensions=anchor.shape[0],
            group_size=group_size,
            base_minus=base_radius,
            base_plus=base_radius,
            minus_groups=minus_groups,
            plus_groups=plus_groups,
        )

    def _expand_groups(self, values: list[float], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        chunks = [
            torch.full((self.group_size,), value, device=device, dtype=dtype)
            for value in values
        ]
        return torch.cat(chunks, dim=0)


def _pool_groups(values: torch.Tensor, group_size: int, reduction: str) -> list[float]:
    pooled: list[float] = []
    for start in range(0, values.shape[0], group_size):
        chunk = values[start : start + group_size]
        if reduction == "max":
            pooled.append(float(chunk.max().item()))
        elif reduction == "mean":
            pooled.append(float(chunk.mean().item()))
        else:
            raise ValueError(f"Unsupported reduction: {reduction}")
    return pooled


def inside_fraction(embeds: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    inside = (embeds >= lower.unsqueeze(0)) & (embeds <= upper.unsqueeze(0))
    return inside.float().mean(dim=-1)


def soft_box_distance(embeds: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    below = torch.relu(lower.unsqueeze(0) - embeds)
    above = torch.relu(embeds - upper.unsqueeze(0))
    return (below + above).mean(dim=-1)
