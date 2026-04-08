"""Hippo encoder distillation package."""

from hippo_encoder.region import RangeOp, SparseRegionProgram, inside_fraction, soft_box_distance
from hippo_encoder.formula_region import FormulaRegionProgram, RangedFormulaTerm
from hippo_encoder.group_region import GroupRegionProgram
from hippo_encoder.rope_region import (
    DualRopeFormulaProgram,
    DualRopePointProgram,
    DualRopeRegionProgram,
    DualRopeShapeProgram,
    RopeBoxOp,
    RopeFormulaTerm,
    RopePointOp,
    RopeShapeOp,
)

__all__ = [
    "FormulaRegionProgram",
    "GroupRegionProgram",
    "DualRopeFormulaProgram",
    "DualRopePointProgram",
    "DualRopeRegionProgram",
    "DualRopeShapeProgram",
    "RangedFormulaTerm",
    "RangeOp",
    "RopeBoxOp",
    "RopeFormulaTerm",
    "RopePointOp",
    "RopeShapeOp",
    "SparseRegionProgram",
    "inside_fraction",
    "soft_box_distance",
]
