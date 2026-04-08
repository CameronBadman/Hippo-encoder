from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch


@dataclass
class RopeBoxOp:
    target: str
    rope: int
    x0: int
    y0: int
    x1: int
    y1: int
    value: float
    mode: str = "add"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DualRopeRegionProgram:
    dimensions: int
    base_minus: float
    base_plus: float
    minus_ops: list[RopeBoxOp]
    plus_ops: list[RopeBoxOp]

    def hydrate(self, anchor: torch.Tensor) -> dict[str, torch.Tensor]:
        if anchor.dim() != 1 or anchor.shape[0] != self.dimensions:
            raise ValueError("Anchor must be a 1D tensor matching program dimensions.")

        minus = torch.full_like(anchor, self.base_minus)
        plus = torch.full_like(anchor, self.base_plus)
        rope_ids, xs, ys = _layout_tensors(self.dimensions, device=anchor.device)

        self._apply_ops(minus, self.minus_ops, rope_ids=rope_ids, xs=xs, ys=ys)
        self._apply_ops(plus, self.plus_ops, rope_ids=rope_ids, xs=xs, ys=ys)

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
    ) -> "DualRopeRegionProgram":
        if anchor.dim() != 1:
            raise ValueError("Anchor must be a 1D tensor.")
        if positives.dim() != 2 or positives.shape[1] != anchor.shape[0]:
            raise ValueError("Positives must have shape [N, D] with same D as anchor.")

        deltas = positives - anchor.unsqueeze(0)
        minus = torch.clamp((-deltas).max(dim=0).values * radius_scale, min=base_radius)
        plus = torch.clamp(deltas.max(dim=0).values * radius_scale, min=base_radius)

        minus_ops = _compress_rope_dense(
            dense=minus,
            base_value=base_radius,
            target="minus",
            quantize_step=quantize_step,
            change_threshold=change_threshold,
            mode="add",
        )
        plus_ops = _compress_rope_dense(
            dense=plus,
            base_value=base_radius,
            target="plus",
            quantize_step=quantize_step,
            change_threshold=change_threshold,
            mode="add",
        )
        return cls(
            dimensions=anchor.shape[0],
            base_minus=base_radius,
            base_plus=base_radius,
            minus_ops=minus_ops,
            plus_ops=plus_ops,
        )

    @staticmethod
    def _apply_ops(
        target: torch.Tensor,
        ops: list[RopeBoxOp],
        rope_ids: torch.Tensor,
        xs: torch.Tensor,
        ys: torch.Tensor,
    ) -> None:
        for op in ops:
            mask = (
                (rope_ids == op.rope)
                & (xs >= op.x0)
                & (xs <= op.x1)
                & (ys >= op.y0)
                & (ys <= op.y1)
            )
            if op.mode == "set":
                target[mask] = op.value
            elif op.mode == "add":
                target[mask] += op.value
            else:
                raise ValueError(f"Unsupported op mode: {op.mode}")


def _rope_count(dimensions: int, rope: int) -> int:
    return (dimensions + (1 if rope == 0 else 0)) // 2


def _rope_width(count: int) -> int:
    return max(1, math.ceil(math.sqrt(count)))


def _layout_tensors(dimensions: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    indices = torch.arange(dimensions, device=device)
    rope_ids = indices % 2
    positions = indices // 2
    widths = torch.tensor(
        [_rope_width(_rope_count(dimensions, 0)), _rope_width(_rope_count(dimensions, 1))],
        device=device,
    )
    xs = positions % widths[rope_ids]
    ys = torch.div(positions, widths[rope_ids], rounding_mode="floor")
    return rope_ids, xs, ys


def _compress_rope_dense(
    dense: torch.Tensor,
    base_value: float,
    target: str,
    quantize_step: float,
    change_threshold: float,
    mode: str,
) -> list[RopeBoxOp]:
    if dense.dim() != 1:
        raise ValueError("Dense array must be 1D.")

    ops: list[RopeBoxOp] = []
    for rope in (0, 1):
        rope_values = dense[rope::2]
        if rope_values.numel() == 0:
            continue
        width = _rope_width(rope_values.numel())
        height = math.ceil(rope_values.numel() / width)
        grid = torch.zeros((height, width), dtype=rope_values.dtype)
        flat = rope_values - base_value
        quantized = torch.round(flat / quantize_step) * quantize_step
        grid.view(-1)[: rope_values.numel()] = quantized
        valid = torch.zeros((height, width), dtype=torch.bool)
        valid.view(-1)[: rope_values.numel()] = True
        ops.extend(
            _grid_to_box_ops(
                grid=grid,
                valid=valid,
                target=target,
                rope=rope,
                change_threshold=change_threshold,
                mode=mode,
            )
        )
    return ops


def _grid_to_box_ops(
    grid: torch.Tensor,
    valid: torch.Tensor,
    target: str,
    rope: int,
    change_threshold: float,
    mode: str,
) -> list[RopeBoxOp]:
    active: dict[tuple[int, int, float], tuple[int, int]] = {}
    finished: list[RopeBoxOp] = []

    for y in range(grid.shape[0]):
        runs: list[tuple[int, int, float]] = []
        x = 0
        while x < grid.shape[1]:
            if not bool(valid[y, x]):
                x += 1
                continue
            value = float(grid[y, x].item())
            if abs(value) < change_threshold:
                x += 1
                continue
            start = x
            while (
                x + 1 < grid.shape[1]
                and bool(valid[y, x + 1])
                and abs(float(grid[y, x + 1].item()) - value) < change_threshold
            ):
                x += 1
            runs.append((start, x, value))
            x += 1

        current_keys = {(x0, x1, value) for x0, x1, value in runs}
        for key, (y0, y1) in list(active.items()):
            if key not in current_keys:
                x0, x1, value = key
                finished.append(
                    RopeBoxOp(
                        target=target,
                        rope=rope,
                        x0=x0,
                        y0=y0,
                        x1=x1,
                        y1=y1,
                        value=value,
                        mode=mode,
                    )
                )
                del active[key]

        for x0, x1, value in runs:
            key = (x0, x1, value)
            if key in active:
                y0, _ = active[key]
                active[key] = (y0, y)
            else:
                active[key] = (y, y)

    for key, (y0, y1) in active.items():
        x0, x1, value = key
        finished.append(
            RopeBoxOp(
                target=target,
                rope=rope,
                x0=x0,
                y0=y0,
                x1=x1,
                y1=y1,
                value=value,
                mode=mode,
            )
        )
    return finished


def inside_fraction(embeds: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    inside = (embeds >= lower.unsqueeze(0)) & (embeds <= upper.unsqueeze(0))
    return inside.float().mean(dim=-1)


def soft_box_distance(embeds: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    below = torch.relu(lower.unsqueeze(0) - embeds)
    above = torch.relu(embeds - upper.unsqueeze(0))
    return (below + above).mean(dim=-1)
