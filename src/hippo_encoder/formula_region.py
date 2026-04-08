from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch


@dataclass
class RangedFormulaTerm:
    target: str
    term_type: str
    start: int
    end: int
    amplitude: float
    start_value: float | None = None
    end_value: float | None = None
    center_ratio: float | None = None
    width_ratio: float | None = None
    margin: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    def support_interval(self) -> dict:
        return {"start": self.start, "end": self.end}


@dataclass
class FormulaRegionProgram:
    dimensions: int
    base_minus: float
    base_plus: float
    minus_terms: list[RangedFormulaTerm]
    plus_terms: list[RangedFormulaTerm]

    def hydrate(self, anchor: torch.Tensor) -> dict[str, torch.Tensor]:
        if anchor.dim() != 1 or anchor.shape[0] != self.dimensions:
            raise ValueError("Anchor must be a 1D tensor matching program dimensions.")

        minus = torch.full_like(anchor, self.base_minus)
        plus = torch.full_like(anchor, self.base_plus)

        self._apply_terms(minus, self.minus_terms)
        self._apply_terms(plus, self.plus_terms)
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
    def from_generated_dict(
        cls,
        payload: dict,
        dimensions: int,
        base_minus: float = 0.01,
        base_plus: float = 0.01,
    ) -> "FormulaRegionProgram":
        minus_terms = _parse_generated_terms(payload.get("minus_terms", []), dimensions, "minus")
        plus_terms = _parse_generated_terms(payload.get("plus_terms", []), dimensions, "plus")
        return cls(
            dimensions=dimensions,
            base_minus=base_minus,
            base_plus=base_plus,
            minus_terms=minus_terms,
            plus_terms=plus_terms,
        )

    @classmethod
    def from_teacher_spread(
        cls,
        anchor: torch.Tensor,
        positives: torch.Tensor,
        negatives: torch.Tensor | None = None,
        base_radius: float = 0.01,
        radius_scale: float = 1.0,
        max_terms_per_side: int = 24,
    ) -> "FormulaRegionProgram":
        if anchor.dim() != 1:
            raise ValueError("Anchor must be a 1D tensor.")
        if positives.dim() != 2 or positives.shape[1] != anchor.shape[0]:
            raise ValueError("Positives must have shape [N, D] with same D as anchor.")

        deltas = positives - anchor.unsqueeze(0)
        minus = torch.clamp((-deltas).max(dim=0).values * radius_scale, min=base_radius)
        plus = torch.clamp(deltas.max(dim=0).values * radius_scale, min=base_radius)

        minus_terms = fit_ranged_formula_terms(
            target=minus,
            base_radius=base_radius,
            max_terms=max_terms_per_side,
            target_name="minus",
            negatives=negatives,
            anchor=anchor,
            side="minus",
        )
        plus_terms = fit_ranged_formula_terms(
            target=plus,
            base_radius=base_radius,
            max_terms=max_terms_per_side,
            target_name="plus",
            negatives=negatives,
            anchor=anchor,
            side="plus",
        )
        return cls(
            dimensions=anchor.shape[0],
            base_minus=base_radius,
            base_plus=base_radius,
            minus_terms=minus_terms,
            plus_terms=plus_terms,
        )

    @staticmethod
    def _apply_terms(target: torch.Tensor, terms: list[RangedFormulaTerm]) -> None:
        for term in terms:
            start = max(0, min(target.shape[0] - 1, term.start))
            end = max(start, min(target.shape[0] - 1, term.end))
            span = end - start + 1
            if span <= 0:
                continue

            if term.term_type == "const":
                values = torch.full_like(target[start : end + 1], term.amplitude)
            elif term.term_type == "ramp":
                start_value = term.start_value if term.start_value is not None else 0.0
                end_value = term.end_value if term.end_value is not None else term.amplitude
                values = torch.linspace(
                    start_value,
                    end_value,
                    steps=span,
                    device=target.device,
                    dtype=target.dtype,
                )
            elif term.term_type == "gaussian":
                center_ratio = term.center_ratio if term.center_ratio is not None else 0.5
                width_ratio = max(term.width_ratio if term.width_ratio is not None else 0.25, 1e-4)
                local_x = torch.linspace(0.0, 1.0, steps=span, device=target.device, dtype=target.dtype)
                values = term.amplitude * torch.exp(-0.5 * ((local_x - center_ratio) / width_ratio) ** 2)
            elif term.term_type == "box":
                values = torch.full_like(target[start : end + 1], term.amplitude)
            else:
                raise ValueError(f"Unsupported term type: {term.term_type}")

            target[start : end + 1] += values


def fit_ranged_formula_terms(
    target: torch.Tensor,
    base_radius: float,
    max_terms: int,
    target_name: str,
    negatives: torch.Tensor | None,
    anchor: torch.Tensor,
    side: str,
    min_segment_length: int = 8,
    top_segments: int = 4,
) -> list[RangedFormulaTerm]:
    if target.dim() != 1:
        raise ValueError("Target must be 1D.")

    residual = torch.clamp(target - base_radius, min=0.0)
    if residual.max().item() <= 1e-8:
        return []

    segments = _find_top_segments(residual, min_segment_length=min_segment_length, top_k=top_segments)
    terms: list[RangedFormulaTerm] = []

    candidates: list[tuple[float, RangedFormulaTerm]] = []

    for start, end in segments:
        if len(terms) >= max_terms:
            break
        segment = residual[start : end + 1]
        if segment.numel() == 0:
            continue

        mean_value = float(segment.mean().item())
        start_value = float(segment[0].item())
        end_value = float(segment[-1].item())
        peak_index = int(torch.argmax(segment).item())
        peak_value = float(segment[peak_index].item())
        center_ratio = peak_index / max(1, segment.numel() - 1)

        weights = segment / segment.sum().clamp(min=1e-8)
        positions = torch.linspace(0.0, 1.0, steps=segment.numel(), dtype=segment.dtype, device=segment.device)
        mean_pos = float((weights * positions).sum().item())
        variance = float((weights * (positions - mean_pos) ** 2).sum().item())
        width_ratio = max(0.08, min(0.5, math.sqrt(max(variance, 1e-5)) * 1.5))

        candidates.append(
            _score_candidate(
                term=RangedFormulaTerm(
                    target=target_name,
                    term_type="const",
                    start=start,
                    end=end,
                    amplitude=mean_value,
                ),
                negatives=negatives,
                anchor=anchor,
                side=side,
            )
        )

        candidates.append(
            _score_candidate(
                term=RangedFormulaTerm(
                    target=target_name,
                    term_type="box",
                    start=start,
                    end=end,
                    amplitude=max(start_value, end_value, mean_value),
                ),
                negatives=negatives,
                anchor=anchor,
                side=side,
            )
        )

        if abs(end_value - start_value) >= 0.01:
            candidates.append(
                _score_candidate(
                    term=RangedFormulaTerm(
                        target=target_name,
                        term_type="ramp",
                        start=start,
                        end=end,
                        amplitude=end_value,
                        start_value=start_value,
                        end_value=end_value,
                    ),
                    negatives=negatives,
                    anchor=anchor,
                    side=side,
                )
            )

        if peak_value >= mean_value + 0.01:
            candidates.append(
                _score_candidate(
                    term=RangedFormulaTerm(
                        target=target_name,
                        term_type="gaussian",
                        start=start,
                        end=end,
                        amplitude=max(0.0, peak_value - mean_value),
                        center_ratio=center_ratio,
                        width_ratio=width_ratio,
                    ),
                    negatives=negatives,
                    anchor=anchor,
                    side=side,
                )
            )

    candidates.sort(key=lambda item: item[0], reverse=True)
    selected = [term for _, term in candidates[:max_terms]]
    return selected


def _score_candidate(
    term: RangedFormulaTerm,
    negatives: torch.Tensor | None,
    anchor: torch.Tensor,
    side: str,
) -> tuple[float, RangedFormulaTerm]:
    # Prefer terms with large positive support, but penalize terms that likely widen regions
    # along dimensions where negatives already sit close to the anchor.
    length = term.end - term.start + 1
    base_score = term.amplitude * length
    if negatives is None or negatives.numel() == 0:
        return base_score, term

    negative_slice = negatives[:, term.start : term.end + 1]
    anchor_slice = anchor[term.start : term.end + 1].unsqueeze(0)
    if side == "minus":
        closeness = torch.relu(anchor_slice - negative_slice)
    else:
        closeness = torch.relu(negative_slice - anchor_slice)
    penalty = float(closeness.mean().item()) * length
    return base_score - penalty, term


def _find_top_segments(residual: torch.Tensor, min_segment_length: int, top_k: int) -> list[tuple[int, int]]:
    smoothed = _smooth(residual, kernel_size=max(5, min_segment_length))
    values, indices = torch.topk(smoothed, k=min(top_k, smoothed.shape[0]))

    segments: list[tuple[int, int, float]] = []
    half = max(1, min_segment_length // 2)
    for value_tensor, index_tensor in zip(values, indices):
        score = float(value_tensor.item())
        center = int(index_tensor.item())
        start = max(0, center - half)
        end = min(residual.shape[0] - 1, center + half)
        merged = False
        for seg_index, (seg_start, seg_end, seg_score) in enumerate(segments):
            if not (end < seg_start or start > seg_end):
                segments[seg_index] = (
                    min(seg_start, start),
                    max(seg_end, end),
                    max(seg_score, score),
                )
                merged = True
                break
        if not merged:
            segments.append((start, end, score))

    segments.sort(key=lambda item: item[2], reverse=True)
    return [(start, end) for start, end, _ in segments[:top_k]]


def _smooth(values: torch.Tensor, kernel_size: int) -> torch.Tensor:
    kernel_size = max(1, kernel_size)
    padding = kernel_size // 2
    kernel = torch.ones(1, 1, kernel_size, dtype=values.dtype, device=values.device) / kernel_size
    padded = torch.nn.functional.pad(values.view(1, 1, -1), (padding, padding), mode="replicate")
    return torch.nn.functional.conv1d(padded, kernel).view(-1)[: values.shape[0]]


def inside_fraction(embeds: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    inside = (embeds >= lower.unsqueeze(0)) & (embeds <= upper.unsqueeze(0))
    return inside.float().mean(dim=-1)


def soft_box_distance(embeds: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    below = torch.relu(lower.unsqueeze(0) - embeds)
    above = torch.relu(embeds - upper.unsqueeze(0))
    return (below + above).mean(dim=-1)


def extract_json_object(text: str) -> dict:
    import json

    decoder = json.JSONDecoder()
    for start, char in enumerate(text):
        if char != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise ValueError("No valid JSON object found in generated text.")


def _parse_generated_terms(items: list[dict], dimensions: int, target_name: str) -> list[RangedFormulaTerm]:
    parsed: list[RangedFormulaTerm] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        term_type = str(item.get("term_type", "")).strip()
        if term_type not in {"box", "const", "ramp", "gaussian"}:
            continue

        start = max(0, min(dimensions - 1, int(item.get("start", 0))))
        end = max(start, min(dimensions - 1, int(item.get("end", dimensions - 1))))
        amplitude = float(item.get("amplitude", 0.0))
        start_value = item.get("start_value")
        end_value = item.get("end_value")
        center_ratio = item.get("center_ratio")
        width_ratio = item.get("width_ratio")

        parsed.append(
            RangedFormulaTerm(
                target=target_name,
                term_type=term_type,
                start=start,
                end=end,
                amplitude=amplitude,
                start_value=float(start_value) if start_value is not None else None,
                end_value=float(end_value) if end_value is not None else None,
                center_ratio=float(center_ratio) if center_ratio is not None else None,
                width_ratio=float(width_ratio) if width_ratio is not None else None,
            )
        )
    return parsed
