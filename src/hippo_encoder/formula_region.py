from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch


@dataclass
class FormulaTerm:
    target: str
    amplitude: float
    center: float
    width: float

    def to_dict(self) -> dict:
        payload = asdict(self)
        return payload

    def support_interval(self, dimensions: int) -> dict:
        start = math.floor(max(0.0, self.center - self.width) * (dimensions - 1))
        end = math.ceil(min(1.0, self.center + self.width) * (dimensions - 1))
        return {"start": start, "end": end}


@dataclass
class FormulaRegionProgram:
    dimensions: int
    base_minus: float
    base_plus: float
    minus_terms: list[FormulaTerm]
    plus_terms: list[FormulaTerm]

    def hydrate(self, anchor: torch.Tensor) -> dict[str, torch.Tensor]:
        if anchor.dim() != 1 or anchor.shape[0] != self.dimensions:
            raise ValueError("Anchor must be a 1D tensor matching program dimensions.")

        x = torch.linspace(0.0, 1.0, self.dimensions, device=anchor.device, dtype=anchor.dtype)
        minus = torch.full_like(anchor, self.base_minus)
        plus = torch.full_like(anchor, self.base_plus)

        minus = minus + self._evaluate_terms(self.minus_terms, x)
        plus = plus + self._evaluate_terms(self.plus_terms, x)
        minus = torch.clamp(minus, min=0.0)
        plus = torch.clamp(plus, min=0.0)

        return {
            "minus": minus,
            "plus": plus,
            "lower": anchor - minus,
            "upper": anchor + plus,
        }

    def to_dict(self) -> dict:
        return {
            "dimensions": self.dimensions,
            "base_minus": self.base_minus,
            "base_plus": self.base_plus,
            "minus_terms": [term.to_dict() for term in self.minus_terms],
            "plus_terms": [term.to_dict() for term in self.plus_terms],
        }

    @classmethod
    def from_teacher_spread(
        cls,
        anchor: torch.Tensor,
        positives: torch.Tensor,
        base_radius: float = 0.01,
        radius_scale: float = 1.0,
        num_terms: int = 8,
    ) -> "FormulaRegionProgram":
        if anchor.dim() != 1:
            raise ValueError("Anchor must be a 1D tensor.")
        if positives.dim() != 2 or positives.shape[1] != anchor.shape[0]:
            raise ValueError("Positives must have shape [N, D] with same D as anchor.")

        deltas = positives - anchor.unsqueeze(0)
        minus = torch.clamp((-deltas).max(dim=0).values * radius_scale, min=base_radius)
        plus = torch.clamp(deltas.max(dim=0).values * radius_scale, min=base_radius)

        minus_terms = fit_formula_terms(minus, base_radius=base_radius, num_terms=num_terms)
        plus_terms = fit_formula_terms(plus, base_radius=base_radius, num_terms=num_terms)
        return cls(
            dimensions=anchor.shape[0],
            base_minus=base_radius,
            base_plus=base_radius,
            minus_terms=minus_terms,
            plus_terms=plus_terms,
        )

    @staticmethod
    def _evaluate_terms(terms: list[FormulaTerm], x: torch.Tensor) -> torch.Tensor:
        values = torch.zeros_like(x)
        for term in terms:
            width = max(term.width, 1e-4)
            values = values + term.amplitude * torch.exp(-0.5 * ((x - term.center) / width) ** 2)
        return values


def fit_formula_terms(
    target: torch.Tensor,
    base_radius: float,
    num_terms: int,
    num_centers: int = 24,
    widths: tuple[float, ...] = (0.03, 0.06, 0.12, 0.24),
) -> list[FormulaTerm]:
    if target.dim() != 1:
        raise ValueError("Target must be 1D.")

    residual = torch.clamp(target - base_radius, min=0.0)
    if residual.max().item() <= 1e-8:
        return []

    x = torch.linspace(0.0, 1.0, target.shape[0], dtype=target.dtype, device=target.device)
    centers = torch.linspace(0.0, 1.0, num_centers, dtype=target.dtype, device=target.device)

    basis_columns: list[torch.Tensor] = []
    basis_specs: list[tuple[float, float]] = []
    for width in widths:
        for center in centers:
            column = torch.exp(-0.5 * ((x - center) / width) ** 2)
            basis_columns.append(column)
            basis_specs.append((float(center.item()), float(width)))

    basis = torch.stack(basis_columns, dim=1)
    solution = torch.linalg.lstsq(basis, residual.unsqueeze(1)).solution.squeeze(1)
    solution = torch.clamp(solution, min=0.0)

    topk = min(num_terms, solution.shape[0])
    values, indices = torch.topk(solution, k=topk)
    terms: list[FormulaTerm] = []
    for amplitude_tensor, index_tensor in zip(values, indices):
        amplitude = float(amplitude_tensor.item())
        if amplitude <= 1e-6:
            continue
        center, width = basis_specs[int(index_tensor.item())]
        terms.append(
            FormulaTerm(
                target="radius",
                amplitude=amplitude,
                center=center,
                width=width,
            )
        )
    return terms


def inside_fraction(embeds: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    inside = (embeds >= lower.unsqueeze(0)) & (embeds <= upper.unsqueeze(0))
    return inside.float().mean(dim=-1)


def soft_box_distance(embeds: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    below = torch.relu(lower.unsqueeze(0) - embeds)
    above = torch.relu(embeds - upper.unsqueeze(0))
    return (below + above).mean(dim=-1)
