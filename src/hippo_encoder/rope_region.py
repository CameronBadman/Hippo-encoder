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
class RopePointOp:
    target: str
    rope: int
    x: int
    y: int
    value: float
    mode: str = "add"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RopeShapeOp:
    target: str
    rope: int
    shape_type: str
    x: int
    y: int
    dx: int
    dy: int
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


@dataclass
class DualRopePointProgram:
    dimensions: int
    base_minus: float
    base_plus: float
    minus_ops: list[RopePointOp]
    plus_ops: list[RopePointOp]

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
        terms_per_side: int,
        base_radius: float = 0.01,
        radius_scale: float = 1.0,
        quantize_step: float = 0.01,
        change_threshold: float = 0.005,
    ) -> "DualRopePointProgram":
        if anchor.dim() != 1:
            raise ValueError("Anchor must be a 1D tensor.")
        if positives.dim() != 2 or positives.shape[1] != anchor.shape[0]:
            raise ValueError("Positives must have shape [N, D] with same D as anchor.")

        deltas = positives - anchor.unsqueeze(0)
        minus = torch.clamp((-deltas).max(dim=0).values * radius_scale, min=base_radius)
        plus = torch.clamp(deltas.max(dim=0).values * radius_scale, min=base_radius)

        minus_ops = _compress_rope_points(
            dense=minus,
            base_value=base_radius,
            target="minus",
            terms=terms_per_side,
            quantize_step=quantize_step,
            change_threshold=change_threshold,
            mode="add",
        )
        plus_ops = _compress_rope_points(
            dense=plus,
            base_value=base_radius,
            target="plus",
            terms=terms_per_side,
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
        ops: list[RopePointOp],
        rope_ids: torch.Tensor,
        xs: torch.Tensor,
        ys: torch.Tensor,
    ) -> None:
        for op in ops:
            mask = (rope_ids == op.rope) & (xs == op.x) & (ys == op.y)
            if op.mode == "set":
                target[mask] = op.value
            elif op.mode == "add":
                target[mask] += op.value
            else:
                raise ValueError(f"Unsupported op mode: {op.mode}")


@dataclass
class DualRopeShapeProgram:
    dimensions: int
    base_minus: float
    base_plus: float
    minus_ops: list[RopeShapeOp]
    plus_ops: list[RopeShapeOp]

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
        terms_per_side: int,
        base_radius: float = 0.01,
        radius_scale: float = 1.0,
        quantize_step: float = 0.01,
        change_threshold: float = 0.005,
    ) -> "DualRopeShapeProgram":
        if anchor.dim() != 1:
            raise ValueError("Anchor must be a 1D tensor.")
        if positives.dim() != 2 or positives.shape[1] != anchor.shape[0]:
            raise ValueError("Positives must have shape [N, D] with same D as anchor.")

        deltas = positives - anchor.unsqueeze(0)
        minus = torch.clamp((-deltas).max(dim=0).values * radius_scale, min=base_radius)
        plus = torch.clamp(deltas.max(dim=0).values * radius_scale, min=base_radius)

        minus_ops = _compress_rope_shapes(
            dense=minus,
            base_value=base_radius,
            target="minus",
            terms=terms_per_side,
            quantize_step=quantize_step,
            change_threshold=change_threshold,
            mode="add",
        )
        plus_ops = _compress_rope_shapes(
            dense=plus,
            base_value=base_radius,
            target="plus",
            terms=terms_per_side,
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
        ops: list[RopeShapeOp],
        rope_ids: torch.Tensor,
        xs: torch.Tensor,
        ys: torch.Tensor,
    ) -> None:
        for op in ops:
            if op.shape_type == "point":
                mask = (rope_ids == op.rope) & (xs == op.x) & (ys == op.y)
            elif op.shape_type == "hline":
                mask = (rope_ids == op.rope) & (ys == op.y) & (xs >= op.x) & (xs <= op.x + op.dx)
            elif op.shape_type == "vline":
                mask = (rope_ids == op.rope) & (xs == op.x) & (ys >= op.y) & (ys <= op.y + op.dy)
            elif op.shape_type == "box":
                mask = (
                    (rope_ids == op.rope)
                    & (xs >= op.x)
                    & (xs <= op.x + op.dx)
                    & (ys >= op.y)
                    & (ys <= op.y + op.dy)
                )
            else:
                raise ValueError(f"Unsupported shape type: {op.shape_type}")
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


def _compress_rope_points(
    dense: torch.Tensor,
    base_value: float,
    target: str,
    terms: int,
    quantize_step: float,
    change_threshold: float,
    mode: str,
) -> list[RopePointOp]:
    if dense.dim() != 1:
        raise ValueError("Dense array must be 1D.")
    if terms <= 0:
        return []

    device = dense.device
    rope_ids, xs, ys = _layout_tensors(dense.shape[0], device=device)
    delta = dense - base_value
    quantized = torch.round(delta / quantize_step) * quantize_step
    active = torch.nonzero(torch.abs(quantized) >= change_threshold, as_tuple=False).flatten()
    if active.numel() == 0:
        return []

    values = torch.abs(quantized[active])
    topk = min(terms, active.numel())
    _, order = torch.topk(values, k=topk, largest=True)
    selected = active[order].tolist()

    ops: list[RopePointOp] = []
    for index in selected:
        ops.append(
            RopePointOp(
                target=target,
                rope=int(rope_ids[index].item()),
                x=int(xs[index].item()),
                y=int(ys[index].item()),
                value=float(quantized[index].item()),
                mode=mode,
            )
        )
    return ops


def _compress_rope_shapes(
    dense: torch.Tensor,
    base_value: float,
    target: str,
    terms: int,
    quantize_step: float,
    change_threshold: float,
    mode: str,
) -> list[RopeShapeOp]:
    if dense.dim() != 1:
        raise ValueError("Dense array must be 1D.")
    if terms <= 0:
        return []

    ops: list[RopeShapeOp] = []
    for rope in (0, 1):
        rope_values = dense[rope::2]
        if rope_values.numel() == 0:
            continue
        width = _rope_width(rope_values.numel())
        height = math.ceil(rope_values.numel() / width)
        residual = torch.zeros((height, width), dtype=rope_values.dtype)
        valid = torch.zeros((height, width), dtype=torch.bool)
        residual.view(-1)[: rope_values.numel()] = rope_values - base_value
        residual = torch.round(residual / quantize_step) * quantize_step
        valid.view(-1)[: rope_values.numel()] = True

        for _ in range(terms):
            mask = valid & (torch.abs(residual) >= change_threshold)
            if not bool(mask.any()):
                break
            op, op_mask, op_value = _best_shape_op(
                residual=residual,
                valid=valid,
                target=target,
                rope=rope,
                mode=mode,
            )
            if op is None or op_mask is None or abs(op_value) < change_threshold:
                break
            ops.append(op)
            residual[op_mask] = residual[op_mask] - op_value
            residual = torch.round(residual / quantize_step) * quantize_step

    return ops


def _best_shape_op(
    residual: torch.Tensor,
    valid: torch.Tensor,
    target: str,
    rope: int,
    mode: str,
) -> tuple[RopeShapeOp | None, torch.Tensor | None, float]:
    candidates: list[tuple[str, int, int, int, int]] = []
    best_index = torch.argmax(torch.abs(torch.where(valid, residual, torch.zeros_like(residual))))
    y = int((best_index // residual.shape[1]).item())
    x = int((best_index % residual.shape[1]).item())

    for dx in range(0, 3):
        candidates.append(("hline" if dx > 0 else "point", x, y, dx, 0))
    for dy in range(1, 3):
        candidates.append(("vline", x, y, 0, dy))
    for dx in range(1, 3):
        for dy in range(1, 3):
            candidates.append(("box", x, y, dx, dy))

    best_score = -1.0
    best_op: RopeShapeOp | None = None
    best_mask: torch.Tensor | None = None
    best_value = 0.0
    for shape_type, x0, y0, dx, dy in candidates:
        mask = _shape_mask(valid, shape_type, x0, y0, dx, dy)
        if mask is None or not bool(mask.any()):
            continue
        values = residual[mask]
        avg = float(values.mean().item())
        score = abs(avg) * values.numel()
        if score > best_score:
            best_score = score
            best_value = avg
            best_mask = mask
            best_op = RopeShapeOp(
                target=target,
                rope=rope,
                shape_type=shape_type,
                x=x0,
                y=y0,
                dx=dx,
                dy=dy,
                value=avg,
                mode=mode,
            )
    return best_op, best_mask, best_value


def _shape_mask(valid: torch.Tensor, shape_type: str, x: int, y: int, dx: int, dy: int) -> torch.Tensor | None:
    h, w = valid.shape
    if x < 0 or y < 0 or x >= w or y >= h:
        return None
    if shape_type == "point":
        mask = torch.zeros_like(valid)
        if bool(valid[y, x]):
            mask[y, x] = True
        return mask
    if shape_type == "hline":
        x1 = min(w - 1, x + dx)
        mask = torch.zeros_like(valid)
        mask[y, x : x1 + 1] = valid[y, x : x1 + 1]
        return mask
    if shape_type == "vline":
        y1 = min(h - 1, y + dy)
        mask = torch.zeros_like(valid)
        mask[y : y1 + 1, x] = valid[y : y1 + 1, x]
        return mask
    if shape_type == "box":
        x1 = min(w - 1, x + dx)
        y1 = min(h - 1, y + dy)
        mask = torch.zeros_like(valid)
        mask[y : y1 + 1, x : x1 + 1] = valid[y : y1 + 1, x : x1 + 1]
        return mask
    return None


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
