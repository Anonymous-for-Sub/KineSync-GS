"""Deterministic commit or rollback decisions from update evidence."""

from __future__ import annotations

import math

from .schema import GuardDecision, GuardThresholds, UpdateEvidence


def decide_update(
    evidence: UpdateEvidence, thresholds: GuardThresholds
) -> GuardDecision:
    values = (
        evidence.visual_gain_ratio,
        evidence.gradient_cosine,
        evidence.correction_cosine,
        evidence.relative_correction_disagreement,
        evidence.candidate_bound_fraction,
    )
    if not evidence.finite or not all(math.isfinite(value) for value in values):
        return GuardDecision(False, "nonfinite", float("-inf"))
    if evidence.view_count < 2:
        return GuardDecision(False, "insufficient_views", -1.0)

    margins = (
        evidence.visual_gain_ratio - thresholds.min_visual_gain_ratio,
        evidence.gradient_cosine - thresholds.min_gradient_cosine,
        evidence.correction_cosine - thresholds.min_correction_cosine,
        thresholds.max_relative_correction_disagreement
        - evidence.relative_correction_disagreement,
    )
    reason = "accepted"
    if margins[0] < 0:
        reason = "visual_gain"
    elif margins[1] < 0:
        reason = "gradient_agreement"
    elif margins[2] < 0:
        reason = "correction_direction"
    elif margins[3] < 0:
        reason = "correction_disagreement"
    return GuardDecision(reason == "accepted", reason, min(margins))


__all__ = ["decide_update"]
