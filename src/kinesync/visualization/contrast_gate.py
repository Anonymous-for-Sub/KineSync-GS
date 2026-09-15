"""Fail-closed, pixel-bound receipts for paper comparison frames.

Receipts are deliberately not user constructible.  The validator owns their
construction and hashes the materialized RGB/ROI arrays itself; callers can
only provide frozen provenance and recorded metric values.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np


class ContrastGateError(ValueError):
    """Raised when a proposed comparison cannot be used as paper evidence."""


class SelectionClass(str, Enum):
    MEDIAN_POSITIVE = "median-positive"
    HIGH_GAIN = "high-gain"
    STRESS_POSITIVE = "stress-positive"


_TOKEN = object()
_ROI_MAD_THRESHOLD = 0.01
_RELATIVE_REDUCTION = 0.25
_OVERLAP_GAIN = 0.03
_DIRECTIONS = frozenset({"lower", "higher"})


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ContrastGateError(f"{field} must be a full lowercase SHA-256")
    return value


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContrastGateError(f"{field} must be a nonempty string")
    return value


def pixel_sha256(array: object) -> str:
    """Hash array dtype, shape and contiguous bytes, avoiding ambiguous pixels."""

    value = np.ascontiguousarray(np.asarray(array))
    return _sha256_bytes(_canonical({"dtype": value.dtype.str, "shape": list(value.shape)}).encode("ascii") + value.tobytes())


def _rgb(image: object, field: str) -> np.ndarray:
    value = np.ascontiguousarray(np.asarray(image))
    if value.ndim != 3 or value.shape[2] != 3 or value.shape[0] < 2 or value.shape[1] < 2:
        raise ContrastGateError(f"{field} must be an HWC RGB image")
    if value.dtype != np.uint8 or not np.isfinite(value).all():
        raise ContrastGateError(f"{field} must be finite uint8 RGB")
    luminance = value.astype(np.float64).mean(axis=2)
    if float(luminance.max() - luminance.min()) < 8.0 or float(luminance.std()) < 1.0:
        raise ContrastGateError(f"{field} is blank or near-uniform")
    return value


def _roi(value: object, shape: tuple[int, int]) -> np.ndarray:
    mask = np.ascontiguousarray(np.asarray(value).astype(bool))
    if mask.shape != shape or not mask.any():
        raise ContrastGateError("ROI must be aligned, nonempty, and match RGB dimensions")
    return mask


@dataclass(frozen=True)
class MetricBinding:
    identity: str
    direction: str
    baseline_value: float
    ours_value: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "identity", _text(self.identity, "metric identity"))
        if self.direction not in _DIRECTIONS:
            raise ContrastGateError("metric direction must be lower or higher")
        for field in ("baseline_value", "ours_value"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ContrastGateError(f"{field} must be a finite numeric value")
            object.__setattr__(self, field, float(value))

    @property
    def improvement(self) -> float:
        return self.baseline_value - self.ours_value if self.direction == "lower" else self.ours_value - self.baseline_value

    @property
    def qualifies(self) -> bool:
        metric = self.identity.lower()
        if metric in {"iou", "boundary_f1", "bf1"}:
            return self.direction == "higher" and self.improvement >= _OVERLAP_GAIN
        if metric in {"success", "recovery_success", "guard_success"}:
            return self.direction == "higher" and self.baseline_value <= 0.0 and self.ours_value >= 1.0
        return self.direction == "lower" and self.baseline_value > 0.0 and self.improvement / self.baseline_value >= _RELATIVE_REDUCTION

    def as_dict(self) -> dict[str, object]:
        return {"identity": self.identity, "direction": self.direction, "baseline_value": self.baseline_value, "ours_value": self.ours_value}


@dataclass(frozen=True)
class ComparisonBinding:
    case_id: str
    camera: str
    timepoint: str
    crop_geometry: Mapping[str, object]
    source_row_id: str
    frozen_result_sha256: str
    result_row_sha256: str
    related_result_row_sha256s: Mapping[str, str]
    related_result_file_sha256s: Mapping[str, str]
    baseline_containers: Mapping[str, str]
    ours_containers: Mapping[str, str]
    selection_decision: Mapping[str, object]
    rollback_binding: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        for field in ("case_id", "camera", "timepoint", "source_row_id"):
            object.__setattr__(self, field, _text(getattr(self, field), field))
        object.__setattr__(self, "frozen_result_sha256", _sha256(self.frozen_result_sha256, "frozen_result_sha256"))
        object.__setattr__(self, "result_row_sha256", _sha256(self.result_row_sha256, "result_row_sha256"))
        if not isinstance(self.related_result_row_sha256s, Mapping) or not self.related_result_row_sha256s:
            raise ContrastGateError("related_result_row_sha256s must be nonempty")
        object.__setattr__(self, "related_result_row_sha256s", MappingProxyType({str(k): _sha256(v, "related_result_row_sha256s") for k, v in sorted(self.related_result_row_sha256s.items())}))
        if not isinstance(self.related_result_file_sha256s, Mapping) or not self.related_result_file_sha256s:
            raise ContrastGateError("related_result_file_sha256s must be nonempty")
        object.__setattr__(self, "related_result_file_sha256s", MappingProxyType({str(k): _sha256(v, "related_result_file_sha256s") for k, v in sorted(self.related_result_file_sha256s.items())}))
        for field in ("baseline_containers", "ours_containers"):
            hashes = getattr(self, field)
            if not isinstance(hashes, Mapping) or not hashes:
                raise ContrastGateError(f"{field} must be a nonempty source hash mapping")
            object.__setattr__(self, field, MappingProxyType({str(k): _sha256(v, field) for k, v in sorted(hashes.items())}))
        if not isinstance(self.crop_geometry, Mapping) or not self.crop_geometry:
            raise ContrastGateError("crop_geometry must be a nonempty mapping")
        if not isinstance(self.selection_decision, Mapping) or "selection_class" not in self.selection_decision:
            raise ContrastGateError("selection_decision must contain selection_class")
        try:
            SelectionClass(str(self.selection_decision["selection_class"]))
        except ValueError as error:
            raise ContrastGateError("selection decision has an unsupported class") from error
        if self.rollback_binding is not None:
            required = {"candidate_role", "final_role", "candidate_committed", "final_committed", "decision_reason"}
            if not isinstance(self.rollback_binding, Mapping) or set(self.rollback_binding) != required:
                raise ContrastGateError("rollback_binding must bind frozen candidate/final roles and decision")
            if self.rollback_binding["candidate_role"] != "candidate" or self.rollback_binding["final_role"] != "final":
                raise ContrastGateError("rollback binding has invalid candidate/final roles")
            if self.rollback_binding["candidate_committed"] is not True or self.rollback_binding["final_committed"] is not False:
                raise ContrastGateError("rollback binding does not contain frozen rejected candidate/final state")
            object.__setattr__(self, "rollback_binding", MappingProxyType(dict(self.rollback_binding)))

    @property
    def crop_geometry_sha256(self) -> str:
        return _sha256_bytes(_canonical(dict(self.crop_geometry)).encode("ascii"))

    @property
    def selection_sha256(self) -> str:
        return _sha256_bytes(_canonical(dict(self.selection_decision)).encode("ascii"))


@dataclass(frozen=True)
class RollbackEvidence:
    candidate_committed: bool
    final_committed: bool
    decision_reason: str

    def __post_init__(self) -> None:
        if self.candidate_committed is not True or self.final_committed is not False:
            raise ContrastGateError("rollback requires frozen candidate=true and final=false")
        if self.decision_reason in {"", "accepted", "unprotected_baseline"}:
            raise ContrastGateError("rollback must have a frozen rejecting decision_reason")


@dataclass(frozen=True, init=False)
class ContrastReceipt:
    """Immutable receipt issued only by :func:`validate_informative_comparison`."""

    binding: ComparisonBinding
    baseline_crop_sha256: str
    ours_crop_sha256: str
    roi_pixel_sha256: str
    roi_mad: float
    metrics: tuple[MetricBinding, ...]
    recommended: bool
    rollback_evidence: RollbackEvidence | None

    def __init__(self, *args: object, _token: object | None = None, **kwargs: object) -> None:
        if _token is not _TOKEN:
            raise TypeError("ContrastReceipt is issued only by validate_informative_comparison")
        for name, value in kwargs.items():
            object.__setattr__(self, name, value)

    @property
    def selection_class(self) -> SelectionClass:
        return SelectionClass(str(self.binding.selection_decision["selection_class"]))

    @property
    def semantic_role(self) -> str:
        return "rollback_mechanism" if self.rollback_evidence else "ours_improves_baseline"

    @property
    def sha256(self) -> str:
        return _sha256_bytes(self.canonical_json().encode("ascii"))

    def canonical_json(self) -> str:
        return _canonical({
            "baseline_container_sha256": dict(self.binding.baseline_containers),
            "baseline_crop_pixel_sha256": self.baseline_crop_sha256,
            "camera": self.binding.camera,
            "case_id": self.binding.case_id,
            "crop_geometry_sha256": self.binding.crop_geometry_sha256,
            "frozen_result_sha256": self.binding.frozen_result_sha256,
            "result_row_sha256": self.binding.result_row_sha256,
            "related_result_row_sha256s": dict(self.binding.related_result_row_sha256s),
            "related_result_file_sha256s": dict(self.binding.related_result_file_sha256s),
            "metrics": [metric.as_dict() for metric in self.metrics],
            "ours_container_sha256": dict(self.binding.ours_containers),
            "ours_crop_pixel_sha256": self.ours_crop_sha256,
            "recommended": self.recommended,
            "rollback": None if self.rollback_evidence is None else {"candidate_committed": True, "final_committed": False, "decision_reason": self.rollback_evidence.decision_reason},
            "rollback_binding": None if self.binding.rollback_binding is None else dict(self.binding.rollback_binding),
            "roi_mad": self.roi_mad,
            "roi_pixel_sha256": self.roi_pixel_sha256,
            "selection_decision_sha256": self.binding.selection_sha256,
            "source_row_id": self.binding.source_row_id,
            "timepoint": self.binding.timepoint,
        })


def measure_roi_difference(baseline: object, ours: object, roi: object) -> float:
    baseline_array, ours_array = _rgb(baseline, "baseline"), _rgb(ours, "ours")
    if baseline_array.shape != ours_array.shape:
        raise ContrastGateError("aligned RGB dimensions differ; stretched images are forbidden")
    mask = _roi(roi, baseline_array.shape[:2])
    return float(np.abs(baseline_array.astype(np.float64) - ours_array.astype(np.float64))[mask].mean() / 255.0)


def validate_informative_comparison(
    baseline: object, ours: object, roi: object, *, binding: ComparisonBinding, metrics: Sequence[MetricBinding],
    rollback_evidence: RollbackEvidence | None = None,
) -> ContrastReceipt:
    """Hash real arrays and issue a receipt after fail-closed gate evaluation."""

    baseline_array, ours_array = _rgb(baseline, "baseline"), _rgb(ours, "ours")
    if baseline_array.shape != ours_array.shape:
        raise ContrastGateError("stretch or resize mismatch is forbidden")
    if np.array_equal(baseline_array, ours_array):
        raise ContrastGateError("byte-identical images are forbidden")
    mask = _roi(roi, baseline_array.shape[:2])
    if not metrics or not all(isinstance(metric, MetricBinding) for metric in metrics):
        raise ContrastGateError("metrics must be nonempty MetricBinding values")
    ordered = tuple(sorted(metrics, key=lambda metric: metric.identity))
    if not any(metric.improvement > 0.0 for metric in ordered):
        raise ContrastGateError("ours is not better on a frozen metric")
    roi_mad = measure_roi_difference(baseline_array, ours_array, mask)
    recommended = any(metric.qualifies for metric in ordered)
    if roi_mad <= _ROI_MAD_THRESHOLD and not recommended:
        raise ContrastGateError("low ROI MAD requires a formal qualifying threshold")
    if rollback_evidence is not None:
        rollback = binding.rollback_binding
        if rollback is None or rollback["decision_reason"] != rollback_evidence.decision_reason:
            raise ContrastGateError("rollback evidence must be bound to the frozen candidate/final decision row")
        unsafe = [metric for metric in ordered if metric.identity == "unsafe_commit"]
        if len(unsafe) != 1 or unsafe[0].direction != "lower" or unsafe[0].baseline_value != 1.0 or unsafe[0].ours_value != 0.0:
            raise ContrastGateError("rollback requires frozen unsafe_commit 1->0 evidence")
    return ContrastReceipt(_token=_TOKEN, binding=binding, baseline_crop_sha256=pixel_sha256(baseline_array), ours_crop_sha256=pixel_sha256(ours_array), roi_pixel_sha256=pixel_sha256(mask), roi_mad=roi_mad, metrics=ordered, recommended=recommended, rollback_evidence=rollback_evidence)


def rank_contrast_receipts(receipts: Sequence[ContrastReceipt]) -> tuple[ContrastReceipt, ...]:
    return tuple(sorted(receipts, key=lambda receipt: (receipt.selection_class.value, receipt.binding.source_row_id, receipt.binding.camera)))


__all__ = ["ComparisonBinding", "ContrastGateError", "ContrastReceipt", "MetricBinding", "RollbackEvidence", "SelectionClass", "measure_roi_difference", "pixel_sha256", "rank_contrast_receipts", "validate_informative_comparison"]
