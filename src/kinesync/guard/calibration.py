"""Deterministic train-dev calibration for conditional update guards."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import itertools
import json
import math
from typing import Iterable, Sequence

import numpy as np

from .decision import decide_update
from .schema import GuardThresholds, UpdateEvidence


@dataclass(frozen=True)
class CalibrationRecord:
    evidence: UpdateEvidence
    trial_type: str
    candidate_success: bool
    state_id: str

    def __post_init__(self) -> None:
        if self.trial_type not in {"controlled", "zero"}:
            raise ValueError("trial_type must be 'controlled' or 'zero'")
        if not self.state_id:
            raise ValueError("state_id must be nonempty")


@dataclass(frozen=True)
class CalibrationCandidate:
    thresholds: GuardThresholds
    controlled_coverage: float
    zero_stability: float
    guarded_success: float
    minimum_accepted_margin: float | None
    feasible: bool


@dataclass(frozen=True)
class GuardCalibration:
    thresholds: GuardThresholds
    candidate_table: tuple[CalibrationCandidate, ...]
    controlled_coverage: float
    zero_stability: float
    guarded_success: float
    source_state_ids: tuple[str, ...]
    fingerprint: str
    threshold_generation: str = "quantile_grid_v1"
    evaluated_candidate_count: int = 0


def _quantile_values(
    values: Sequence[float],
    quantiles: Sequence[float],
    *,
    lower: float,
    upper: float | None = None,
) -> tuple[float, ...]:
    candidates = []
    for quantile in quantiles:
        value = float(np.quantile(values, quantile))
        value = max(lower, value)
        if upper is not None:
            value = min(upper, value)
        candidates.append(value)
    return tuple(sorted(set(candidates)))


def _score_candidate(
    records: Sequence[CalibrationRecord],
    thresholds: GuardThresholds,
    *,
    minimum_zero_stability: float,
    minimum_controlled_coverage: float,
) -> CalibrationCandidate:
    decisions = [decide_update(record.evidence, thresholds) for record in records]
    controlled = [
        (record, decision)
        for record, decision in zip(records, decisions, strict=True)
        if record.trial_type == "controlled"
    ]
    zeros = [
        (record, decision)
        for record, decision in zip(records, decisions, strict=True)
        if record.trial_type == "zero"
    ]
    coverage = sum(decision.accepted for _, decision in controlled) / len(controlled)
    zero_stability = sum(
        (not decision.accepted) or record.candidate_success
        for record, decision in zeros
    ) / len(zeros)
    final_success = sum(
        (
            decision.accepted and record.candidate_success
            if record.trial_type == "controlled"
            else (not decision.accepted) or record.candidate_success
        )
        for record, decision in zip(records, decisions, strict=True)
    ) / len(records)
    accepted_margins = [
        decision.minimum_margin for decision in decisions if decision.accepted
    ]
    minimum_margin = min(accepted_margins) if accepted_margins else None
    return CalibrationCandidate(
        thresholds=thresholds,
        controlled_coverage=coverage,
        zero_stability=zero_stability,
        guarded_success=final_success,
        minimum_accepted_margin=minimum_margin,
        feasible=(
            zero_stability >= minimum_zero_stability
            and coverage >= minimum_controlled_coverage
        ),
    )


def _threshold_tuple(thresholds: GuardThresholds) -> tuple[float, ...]:
    return (
        thresholds.min_visual_gain_ratio,
        thresholds.min_gradient_cosine,
        thresholds.min_correction_cosine,
        thresholds.max_relative_correction_disagreement,
    )


def _candidate_payload(candidate: CalibrationCandidate) -> dict[str, object]:
    return {
        "thresholds": asdict(candidate.thresholds),
        "controlled_coverage": candidate.controlled_coverage,
        "zero_stability": candidate.zero_stability,
        "guarded_success": candidate.guarded_success,
        "minimum_accepted_margin": candidate.minimum_accepted_margin,
        "feasible": candidate.feasible,
    }


def _validate_records(
    records: Iterable[CalibrationRecord],
    *,
    minimum_zero_stability: float,
    minimum_controlled_coverage: float,
) -> tuple[CalibrationRecord, ...]:
    ordered_records = tuple(
        sorted(records, key=lambda record: (record.state_id, record.trial_type))
    )
    if not ordered_records:
        raise ValueError("Calibration records must be nonempty")
    trial_types = {record.trial_type for record in ordered_records}
    if trial_types != {"controlled", "zero"}:
        raise ValueError("Calibration requires controlled and zero trial types")
    if not 0 <= minimum_zero_stability <= 1:
        raise ValueError("minimum_zero_stability must lie in [0, 1]")
    if not 0 <= minimum_controlled_coverage <= 1:
        raise ValueError("minimum_controlled_coverage must lie in [0, 1]")
    if not any(record.evidence.finite for record in ordered_records):
        raise ValueError("Calibration requires finite evidence")
    return ordered_records


def _candidate_rank(candidate: CalibrationCandidate) -> tuple[float, ...]:
    margin = (
        candidate.minimum_accepted_margin
        if candidate.minimum_accepted_margin is not None
        else float("-inf")
    )
    return (
        -candidate.guarded_success,
        -candidate.controlled_coverage,
        -margin,
        *_threshold_tuple(candidate.thresholds),
    )


def _build_calibration(
    ordered_records: tuple[CalibrationRecord, ...],
    candidates: tuple[CalibrationCandidate, ...],
    *,
    threshold_generation: str,
    payload: dict[str, object],
    include_metadata_in_fingerprint: bool,
) -> GuardCalibration:
    feasible = [candidate for candidate in candidates if candidate.feasible]
    if not feasible:
        raise ValueError("No feasible guard threshold tuple satisfies calibration constraints")
    selected = min(feasible, key=_candidate_rank)
    source_state_ids = tuple(record.state_id for record in ordered_records)
    evaluated_candidate_count = len(candidates)
    payload.update(
        {
            "thresholds": asdict(selected.thresholds),
            "controlled_coverage": selected.controlled_coverage,
            "zero_stability": selected.zero_stability,
            "guarded_success": selected.guarded_success,
            "source_state_ids": source_state_ids,
            "candidate_table": [_candidate_payload(candidate) for candidate in candidates],
        }
    )
    if include_metadata_in_fingerprint:
        payload.update(
            {
                "threshold_generation": threshold_generation,
                "evaluated_candidate_count": evaluated_candidate_count,
            }
        )
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return GuardCalibration(
        thresholds=selected.thresholds,
        candidate_table=candidates,
        controlled_coverage=selected.controlled_coverage,
        zero_stability=selected.zero_stability,
        guarded_success=selected.guarded_success,
        source_state_ids=source_state_ids,
        threshold_generation=threshold_generation,
        evaluated_candidate_count=evaluated_candidate_count,
        fingerprint=hashlib.sha256(canonical).hexdigest(),
    )


def calibrate_guard(
    records: Iterable[CalibrationRecord],
    quantiles: Sequence[float],
    *,
    minimum_zero_stability: float = 0.9,
    minimum_controlled_coverage: float = 0.6,
) -> GuardCalibration:
    """Select and fingerprint one feasible guard from train-dev evidence."""

    ordered_records = tuple(
        sorted(records, key=lambda record: (record.state_id, record.trial_type))
    )
    if not ordered_records:
        raise ValueError("Calibration records must be nonempty")
    trial_types = {record.trial_type for record in ordered_records}
    if trial_types != {"controlled", "zero"}:
        raise ValueError("Calibration requires controlled and zero trial types")
    normalized_quantiles = tuple(sorted(set(float(value) for value in quantiles)))
    if not normalized_quantiles:
        raise ValueError("quantiles must be nonempty")
    if not all(math.isfinite(value) and 0 <= value <= 1 for value in normalized_quantiles):
        raise ValueError("quantiles must lie in [0, 1]")
    if not 0 <= minimum_zero_stability <= 1:
        raise ValueError("minimum_zero_stability must lie in [0, 1]")
    if not 0 <= minimum_controlled_coverage <= 1:
        raise ValueError("minimum_controlled_coverage must lie in [0, 1]")

    finite_evidence = [record.evidence for record in ordered_records if record.evidence.finite]
    if not finite_evidence:
        raise ValueError("Calibration requires finite evidence")
    gains = _quantile_values(
        [item.visual_gain_ratio for item in finite_evidence],
        normalized_quantiles,
        lower=0.0,
    )
    gradients = _quantile_values(
        [item.gradient_cosine for item in finite_evidence],
        normalized_quantiles,
        lower=-1.0,
        upper=1.0,
    )
    corrections = _quantile_values(
        [item.correction_cosine for item in finite_evidence],
        normalized_quantiles,
        lower=-1.0,
        upper=1.0,
    )
    disagreements = _quantile_values(
        [item.relative_correction_disagreement for item in finite_evidence],
        normalized_quantiles,
        lower=0.0,
    )

    candidates = tuple(
        _score_candidate(
            ordered_records,
            GuardThresholds(gain, gradient, correction, disagreement),
            minimum_zero_stability=minimum_zero_stability,
            minimum_controlled_coverage=minimum_controlled_coverage,
        )
        for gain, gradient, correction, disagreement in itertools.product(
            gains, gradients, corrections, disagreements
        )
    )
    return _build_calibration(
        ordered_records,
        candidates,
        threshold_generation="quantile_grid_v1",
        payload={
            "quantiles": normalized_quantiles,
            "constraints": {
                "minimum_zero_stability": minimum_zero_stability,
                "minimum_controlled_coverage": minimum_controlled_coverage,
            },
        },
        include_metadata_in_fingerprint=False,
    )


def calibrate_event_gain_guard(
    records: Iterable[CalibrationRecord],
    *,
    fixed_thresholds: GuardThresholds,
    minimum_zero_stability: float = 0.9,
    minimum_controlled_coverage: float = 0.6,
) -> GuardCalibration:
    """Select a feasible guard from every finite train-only visual-gain event."""

    ordered_records = _validate_records(
        records,
        minimum_zero_stability=minimum_zero_stability,
        minimum_controlled_coverage=minimum_controlled_coverage,
    )
    gains = tuple(
        sorted(
            {
                0.0,
                *(
                    max(0.0, record.evidence.visual_gain_ratio)
                    for record in ordered_records
                    if record.evidence.finite
                ),
            }
        )
    )
    candidates = tuple(
        _score_candidate(
            ordered_records,
            GuardThresholds(
                gain,
                fixed_thresholds.min_gradient_cosine,
                fixed_thresholds.min_correction_cosine,
                fixed_thresholds.max_relative_correction_disagreement,
            ),
            minimum_zero_stability=minimum_zero_stability,
            minimum_controlled_coverage=minimum_controlled_coverage,
        )
        for gain in gains
    )
    return _build_calibration(
        ordered_records,
        candidates,
        threshold_generation="event_gain_v1",
        payload={
            "fixed_thresholds": asdict(fixed_thresholds),
            "constraints": {
                "minimum_zero_stability": minimum_zero_stability,
                "minimum_controlled_coverage": minimum_controlled_coverage,
            },
        },
        include_metadata_in_fingerprint=True,
    )


__all__ = [
    "CalibrationCandidate",
    "CalibrationRecord",
    "GuardCalibration",
    "calibrate_event_gain_guard",
    "calibrate_guard",
]
