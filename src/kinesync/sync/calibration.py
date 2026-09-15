"""Train-only calibration of physically detectable joint corrections."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .schema import DetectabilityProfile, JointDetectability


ALGORITHM_VERSION = "rt6-detectability-v2"
_MINIMUM_STATE_REDUCTION = 0.60
_ZERO_QUANTILE = 0.95
_ZERO_TOLERANCE = 1e-12
_PIPER_ARM_JOINT_COUNT = 6
_PIPER_FULL_JOINT_COUNT = 8


def _boolean(value: Any, *, field: str) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise ValueError(f"{field} must be boolean")


def _mapping(value: Any, *, field: str) -> dict[str, float]:
    parsed = json.loads(value) if isinstance(value, str) else value
    if not isinstance(parsed, Mapping) or not parsed:
        raise ValueError(f"{field} must be a nonempty mapping")
    result = {str(name): float(number) for name, number in parsed.items()}
    if not all(math.isfinite(number) for number in result.values()):
        raise ValueError(f"{field} values must be finite")
    return result


def _vector(value: Any, *, expected: int) -> tuple[float, ...]:
    parsed = json.loads(value) if isinstance(value, str) else value
    if not isinstance(parsed, Sequence) or isinstance(parsed, (str, bytes)):
        raise ValueError("joint_correction_rad must be a vector")
    result = tuple(float(number) for number in parsed)
    if len(result) != expected:
        raise ValueError(
            f"joint_correction_rad must contain {expected} values, got {len(result)}"
        )
    if not all(math.isfinite(number) for number in result):
        raise ValueError("joint_correction_rad values must be finite")
    return result


def _validate_sha256(value: str) -> None:
    if len(value) != 64:
        raise ValueError("source_sha256 must be a SHA-256 hex digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError("source_sha256 must be a SHA-256 hex digest") from error


def _finite_metric(row: Mapping[str, Any], field: str) -> float:
    if field not in row:
        raise ValueError("calibration rows require complete visual metrics")
    value = float(row[field])
    if not math.isfinite(value):
        raise ValueError("calibration rows require finite visual metrics")
    return value


def calibration_rows_sha256(rows: Iterable[Mapping[str, Any]]) -> str:
    """Hash calibration row content using a CSV-round-trip-stable encoding."""

    canonical_rows: list[dict[str, str]] = []
    for source in rows:
        canonical: dict[str, str] = {}
        for key, value in source.items():
            if value is None:
                encoded = ""
            elif isinstance(value, Mapping) or (
                isinstance(value, Sequence) and not isinstance(value, (str, bytes))
            ):
                encoded = json.dumps(
                    value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
                )
            else:
                encoded = str(value)
            canonical[str(key)] = encoded
        canonical_rows.append(canonical)
    if not canonical_rows:
        raise ValueError("calibration rows cannot be empty")
    canonical_rows.sort(
        key=lambda row: json.dumps(
            row, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )
    )
    payload = json.dumps(
        canonical_rows, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def derive_per_joint_recovery(
    row: Mapping[str, Any], *, joint_names: Sequence[str]
) -> dict[str, dict[str, float | bool]]:
    """Derive component physical recovery under one trial-level visual gate."""

    names = tuple(str(name) for name in joint_names)
    offsets = _mapping(row.get("offsets_rad"), field="offsets_rad")
    unknown = set(offsets) - set(names)
    if unknown:
        raise ValueError(f"Unknown joint in offsets_rad: {sorted(unknown)}")
    correction = _vector(row.get("joint_correction_rad"), expected=len(names))
    evidence_finite = _boolean(
        row.get("evidence_finite"), field="evidence_finite"
    )
    iou_before = _finite_metric(row, "mask_iou_before")
    iou_candidate = _finite_metric(row, "mask_iou_candidate")
    boundary_before = _finite_metric(row, "boundary_f1_before")
    boundary_candidate = _finite_metric(row, "boundary_f1_candidate")
    trial_visual_success = bool(
        evidence_finite
        and iou_candidate > iou_before
        and boundary_candidate > boundary_before
    )

    result: dict[str, dict[str, float | bool]] = {}
    for joint_name, injected_value in offsets.items():
        injected = float(injected_value)
        index = names.index(joint_name)
        applied = float(correction[index])
        initial_error = abs(injected)
        final_error = abs(injected + applied)
        if initial_error <= _ZERO_TOLERANCE:
            reduction = 0.0
            physical_success = False
        else:
            reduction = 1.0 - final_error / initial_error
            physical_success = reduction >= _MINIMUM_STATE_REDUCTION
        result[joint_name] = {
            "injected_offset_rad": injected,
            "applied_correction_rad": applied,
            "initial_error_rad": initial_error,
            "final_error_rad": final_error,
            "state_error_reduction": reduction,
            "physical_success": physical_success,
            "trial_visual_success": trial_visual_success,
            "candidate_success": bool(physical_success and trial_visual_success),
        }
    return result


def _validated_v2_recovery(
    row: Mapping[str, Any], *, joint_names: Sequence[str], joint_order_source_sha256: str
) -> dict[str, dict[str, float | bool]]:
    try:
        schema_version = int(row.get("schema_version"))
    except (TypeError, ValueError) as error:
        raise ValueError("calibration rows require schema_version=2") from error
    if schema_version != 2:
        raise ValueError("calibration rows require schema_version=2")
    raw_order = row.get("joint_order")
    order = json.loads(raw_order) if isinstance(raw_order, str) else raw_order
    expected_order = tuple(str(name) for name in joint_names)
    if not isinstance(order, Sequence) or isinstance(order, (str, bytes)):
        raise ValueError("joint_order must be a sequence")
    if tuple(str(name) for name in order) != expected_order:
        raise ValueError("joint_order must match the requested joint names")
    if str(row.get("joint_order_source_sha256", "")) != joint_order_source_sha256:
        raise ValueError("joint_order_source_sha256 does not match the frozen factor")

    raw_recorded = row.get("per_joint_recovery")
    recorded = (
        json.loads(raw_recorded) if isinstance(raw_recorded, str) else raw_recorded
    )
    if not isinstance(recorded, Mapping):
        raise ValueError("per_joint_recovery must be a mapping")
    derived = derive_per_joint_recovery(row, joint_names=expected_order)
    if set(recorded) != set(derived):
        raise ValueError("per_joint_recovery joints must match offsets_rad")
    for joint_name, expected in derived.items():
        item = recorded[joint_name]
        if not isinstance(item, Mapping) or set(item) != set(expected):
            raise ValueError("per_joint_recovery schema does not match v2")
        for field, expected_value in expected.items():
            actual_value = item[field]
            if isinstance(expected_value, bool):
                if _boolean(actual_value, field=field) != expected_value:
                    raise ValueError("per_joint_recovery does not match raw evidence")
            else:
                actual = float(actual_value)
                if not math.isfinite(actual) or not math.isclose(
                    actual, float(expected_value), rel_tol=0.0, abs_tol=1e-12
                ):
                    raise ValueError("per_joint_recovery does not match raw evidence")
    return derived


def migrate_legacy_calibration_rows(
    rows: Iterable[Mapping[str, Any]], *, joint_names: Sequence[str], factors_path: str | Path
) -> list[dict[str, Any]]:
    """Create explicit v2 rows from legacy raw offsets, corrections, and visuals."""

    names = tuple(str(name) for name in joint_names)
    if not names or len(set(names)) != len(names):
        raise ValueError("joint_names must be nonempty and unique")
    path = Path(factors_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"frozen factor file does not exist: {path}")
    factor_bytes = path.read_bytes()
    factor_payload = json.loads(factor_bytes)
    if not isinstance(factor_payload, Mapping):
        raise ValueError("frozen factor file must contain a mapping")
    if factor_payload.get("schema_version") != 1:
        raise ValueError("joint-order factor must be an RT2 schema-v1 file")
    if factor_payload.get("frozen") is not True or factor_payload.get("fit_split") != "train":
        raise ValueError("joint-order factor must be frozen and fit on train")
    raw_factor_names = factor_payload.get("joint_names")
    if not isinstance(raw_factor_names, Sequence) or isinstance(
        raw_factor_names, (str, bytes)
    ):
        raise ValueError("frozen factor joint_names must be a sequence")
    factor_names = tuple(str(name) for name in raw_factor_names)
    if len(factor_names) != _PIPER_ARM_JOINT_COUNT:
        raise ValueError("RT2 factor must contain exactly six PiPER arm joints")
    if len(names) != _PIPER_FULL_JOINT_COUNT:
        raise ValueError("calibration joint order must contain all eight PiPER joints")
    if names[:_PIPER_ARM_JOINT_COUNT] != factor_names:
        raise ValueError("factor joint order must match the six-joint PiPER arm prefix")
    raw_joint_zero = factor_payload.get("joint_zero_rad")
    if not isinstance(raw_joint_zero, Sequence) or isinstance(
        raw_joint_zero, (str, bytes)
    ) or len(raw_joint_zero) != _PIPER_ARM_JOINT_COUNT:
        raise ValueError("RT2 factor joint_zero_rad must contain six values")
    if not all(math.isfinite(float(value)) for value in raw_joint_zero):
        raise ValueError("RT2 factor joint_zero_rad must be finite")
    factor_sha256 = hashlib.sha256(factor_bytes).hexdigest()
    migrated: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        row["schema_version"] = 2
        row["joint_order"] = json.dumps(list(names))
        row["joint_order_source_sha256"] = factor_sha256
        row["per_joint_recovery"] = json.dumps(
            derive_per_joint_recovery(row, joint_names=names), sort_keys=True
        )
        migrated.append(row)
    if not migrated:
        raise ValueError("legacy calibration rows cannot be empty")
    return migrated


def calibrate_detectability(
    rows: Iterable[Mapping[str, Any]],
    *,
    joint_names: Sequence[str],
    source_sha256: str,
    joint_order_source_sha256: str,
) -> DetectabilityProfile:
    """Build a per-joint profile from immutable train calibration rows."""

    names = tuple(str(name) for name in joint_names)
    if not names or len(set(names)) != len(names):
        raise ValueError("joint_names must be nonempty and unique")
    _validate_sha256(source_sha256)
    _validate_sha256(joint_order_source_sha256)

    records = list(rows)
    if not records:
        raise ValueError("detectability calibration rows cannot be empty")
    if source_sha256 != calibration_rows_sha256(records):
        raise ValueError("source_sha256 does not match calibration row content")
    seen_states: set[str] = set()
    controlled: dict[tuple[str, float], list[tuple[bool, float, float]]] = defaultdict(list)
    zero_corrections: dict[str, list[float]] = defaultdict(list)
    controlled_counts: dict[str, int] = defaultdict(int)
    has_controlled = False

    for row in records:
        state_id = str(row.get("state_id", "")).strip()
        if not state_id:
            raise ValueError("calibration row has empty state_id")
        if state_id in seen_states:
            raise ValueError(f"Duplicate state_id: {state_id}")
        seen_states.add(state_id)
        if str(row.get("split", "")) != "train":
            raise ValueError("detectability calibration requires split=train")

        offsets = _mapping(row.get("offsets_rad"), field="offsets_rad")
        recovery = _validated_v2_recovery(
            row,
            joint_names=names,
            joint_order_source_sha256=joint_order_source_sha256,
        )
        _boolean(row.get("candidate_success"), field="candidate_success")
        evidence_finite = _boolean(row.get("evidence_finite"), field="evidence_finite")
        reduction = float(row.get("state_error_reduction"))
        if not evidence_finite or not math.isfinite(reduction):
            raise ValueError("calibration rows must contain finite evidence and metrics")

        for joint_name, injected in offsets.items():
            index = names.index(joint_name)
            component = recovery[joint_name]
            candidate_magnitude = abs(float(component["applied_correction_rad"]))
            if abs(injected) <= _ZERO_TOLERANCE:
                zero_corrections[joint_name].append(candidate_magnitude)
                continue
            has_controlled = True
            controlled_counts[joint_name] += 1
            magnitude = round(abs(injected), 12)
            controlled[(joint_name, magnitude)].append(
                (
                    bool(component["candidate_success"]),
                    float(component["state_error_reduction"]),
                    candidate_magnitude,
                )
            )

    if not has_controlled:
        raise ValueError("detectability calibration requires controlled examples")

    joint_profiles: list[JointDetectability] = []
    for joint_name in names:
        supported_magnitude: float | None = None
        signal_floor: float | None = None
        supporting_count = 0
        magnitudes = sorted(
            magnitude for name, magnitude in controlled if name == joint_name
        )
        for magnitude in magnitudes:
            group = controlled[(joint_name, magnitude)]
            if all(
                success and reduction >= _MINIMUM_STATE_REDUCTION
                for success, reduction, _ in group
            ):
                supported_magnitude = magnitude
                supporting_count = len(group)
                controlled_floor = min(value for _, _, value in group)
                zeros = zero_corrections.get(joint_name, [])
                zero_floor = (
                    float(np.quantile(zeros, _ZERO_QUANTILE)) if zeros else 0.0
                )
                signal_floor = max(controlled_floor, zero_floor)
                break

        supported = supported_magnitude is not None
        joint_profiles.append(
            JointDetectability(
                joint_name=joint_name,
                supported=supported,
                minimum_magnitude_rad=supported_magnitude,
                minimum_candidate_correction_rad=signal_floor,
                controlled_count=controlled_counts[joint_name],
                supporting_count=supporting_count,
                zero_count=len(zero_corrections.get(joint_name, [])),
            )
        )

    state_ids = tuple(sorted(seen_states))
    payload = {
        "algorithm_version": ALGORITHM_VERSION,
        "calibration_state_ids": state_ids,
        "joint_names": names,
        "joint_order_source_sha256": joint_order_source_sha256,
        "joints": [
            {
                "joint_name": item.joint_name,
                "supported": item.supported,
                "minimum_magnitude_rad": item.minimum_magnitude_rad,
                "minimum_candidate_correction_rad": (
                    item.minimum_candidate_correction_rad
                ),
                "controlled_count": item.controlled_count,
                "supporting_count": item.supporting_count,
                "zero_count": item.zero_count,
            }
            for item in joint_profiles
        ],
        "source_sha256": source_sha256,
        "source_split": "train",
    }
    fingerprint = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return DetectabilityProfile(
        joint_names=names,
        joints=tuple(joint_profiles),
        source_sha256=source_sha256,
        joint_order_source_sha256=joint_order_source_sha256,
        source_split="train",
        calibration_state_ids=state_ids,
        algorithm_version=ALGORITHM_VERSION,
        fingerprint=fingerprint,
    )


__all__ = [
    "ALGORITHM_VERSION",
    "calibration_rows_sha256",
    "calibrate_detectability",
    "derive_per_joint_recovery",
    "migrate_legacy_calibration_rows",
]
