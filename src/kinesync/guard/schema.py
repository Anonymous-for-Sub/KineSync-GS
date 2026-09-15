"""Immutable schemas for conditional physical updates."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class UpdateEvidence:
    view_count: int
    visual_gain_ratio: float
    gradient_cosine: float
    correction_cosine: float
    relative_correction_disagreement: float
    candidate_bound_fraction: float
    finite: bool

    def __post_init__(self) -> None:
        if self.view_count < 0:
            raise ValueError("view_count must be nonnegative")
        values = (
            self.visual_gain_ratio,
            self.gradient_cosine,
            self.correction_cosine,
            self.relative_correction_disagreement,
            self.candidate_bound_fraction,
        )
        if self.finite and not all(math.isfinite(value) for value in values):
            raise ValueError("finite=True requires finite evidence values")
        if self.finite and self.relative_correction_disagreement < 0:
            raise ValueError("relative correction disagreement must be nonnegative")
        if self.finite and self.candidate_bound_fraction < 0:
            raise ValueError("candidate bound fraction must be nonnegative")


@dataclass(frozen=True)
class GuardThresholds:
    min_visual_gain_ratio: float
    min_gradient_cosine: float
    min_correction_cosine: float
    max_relative_correction_disagreement: float

    def __post_init__(self) -> None:
        values = (
            self.min_visual_gain_ratio,
            self.min_gradient_cosine,
            self.min_correction_cosine,
            self.max_relative_correction_disagreement,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Guard thresholds must be finite")
        if self.min_visual_gain_ratio < 0:
            raise ValueError("Minimum visual gain must be nonnegative")
        if not -1 <= self.min_gradient_cosine <= 1:
            raise ValueError("Gradient cosine threshold must be in [-1, 1]")
        if not -1 <= self.min_correction_cosine <= 1:
            raise ValueError("Correction cosine threshold must be in [-1, 1]")
        if self.max_relative_correction_disagreement < 0:
            raise ValueError("Maximum correction disagreement must be nonnegative")


@dataclass(frozen=True)
class GuardDecision:
    accepted: bool
    reason: str
    minimum_margin: float


__all__ = ["GuardDecision", "GuardThresholds", "UpdateEvidence"]
