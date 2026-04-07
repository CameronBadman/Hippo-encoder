from __future__ import annotations

from dataclasses import asdict, dataclass

import torch


@dataclass
class RangeOp:
    target: str
    start: int
    end: int
    value: float
    mode: str = "set"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SparseRegionProgram:
    dimensions: int
    base_minus: float
    base_plus: float
    minus_ops: list[RangeOp]
    plus_ops: list[RangeOp]

    def hydrate(self, anchor: torch.Tensor) -> dict[str, torch.Tensor]:
        if anchor.dim() != 1 or anchor.shape[0] != self.dimensions:
            raise ValueError("Anchor must be a 1D tensor matching program dimensions.")

        minus = torch.full_like(anchor, self.base_minus)
        plus = torch.full_like(anchor, self.base_plus)

        self._apply_ops(minus, self.minus_ops)
        self._apply_ops(plus, self.plus_ops)

        lower = anchor - minus
        upper = anchor + plus
        return {
            "minus": minus,
            "plus": plus,
            "lower": lower,
            "upper": upper,
        }

    def to_dict(self) -> dict:
        return {
            "dimensions": self.dimensions,
            "base_minus": self.base_minus,
            "base_plus": self.base_plus,
            "minus_ops": [op.to_dict() for op in self.minus_ops],
            "plus_ops": [op.to_dict() for op in self.plus_ops],
        }

    @classmethod
    def from_teacher_spread(
        cls,
        anchor: torch.Tensor,
        positives: torch.Tensor,
        base_radius: float = 0.01,
        radius_scale: float = 1.0,
        quantize_step: float = 0.01,
        change_threshold: float = 0.005,
    ) -> "SparseRegionProgram":
        if anchor.dim() != 1:
            raise ValueError("Anchor must be a 1D tensor.")
        if positives.dim() != 2 or positives.shape[1] != anchor.shape[0]:
            raise ValueError("Positives must have shape [N, D] with same D as anchor.")

        deltas = positives - anchor.unsqueeze(0)
        minus = torch.clamp((-deltas).max(dim=0).values * radius_scale, min=base_radius)
        plus = torch.clamp(deltas.max(dim=0).values * radius_scale, min=base_radius)

        minus_ops = _compress_dense_array(
            dense=minus,
            base_value=base_radius,
            target="minus",
            quantize_step=quantize_step,
            change_threshold=change_threshold,
        )
        plus_ops = _compress_dense_array(
            dense=plus,
            base_value=base_radius,
            target="plus",
            quantize_step=quantize_step,
            change_threshold=change_threshold,
        )
        return cls(
            dimensions=anchor.shape[0],
            base_minus=base_radius,
            base_plus=base_radius,
            minus_ops=minus_ops,
            plus_ops=plus_ops,
        )

    @staticmethod
    def _apply_ops(target: torch.Tensor, ops: list[RangeOp]) -> None:
        for op in ops:
            start = max(0, op.start)
            end = min(target.shape[0] - 1, op.end)
            if end < start:
                continue
            if op.mode == "set":
                target[start : end + 1] = op.value
            elif op.mode == "add":
                target[start : end + 1] += op.value
            else:
                raise ValueError(f"Unsupported op mode: {op.mode}")


def _compress_dense_array(
    dense: torch.Tensor,
    base_value: float,
    target: str,
    quantize_step: float,
    change_threshold: float,
) -> list[RangeOp]:
    if dense.dim() != 1:
        raise ValueError("Dense array must be 1D.")

    quantized = torch.round(dense / quantize_step) * quantize_step
    ops: list[RangeOp] = []
    run_start = None
    run_value = None

    for index, value_tensor in enumerate(quantized):
        value = float(value_tensor.item())
        if abs(value - base_value) < change_threshold:
            if run_start is not None:
                ops.append(RangeOp(target=target, start=run_start, end=index - 1, value=run_value))
                run_start = None
                run_value = None
            continue

        if run_start is None:
            run_start = index
            run_value = value
            continue

        if abs(value - run_value) >= change_threshold:
            ops.append(RangeOp(target=target, start=run_start, end=index - 1, value=run_value))
            run_start = index
            run_value = value

    if run_start is not None:
        ops.append(RangeOp(target=target, start=run_start, end=dense.shape[0] - 1, value=run_value))

    return ops


def inside_fraction(embeds: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    inside = (embeds >= lower.unsqueeze(0)) & (embeds <= upper.unsqueeze(0))
    return inside.float().mean(dim=-1)


def soft_box_distance(embeds: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    below = torch.relu(lower.unsqueeze(0) - embeds)
    above = torch.relu(embeds - upper.unsqueeze(0))
    return (below + above).mean(dim=-1)
