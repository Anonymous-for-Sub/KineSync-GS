"""Validated, frozen experiment matrices for RT3 conditional guards."""

from __future__ import annotations

from collections import Counter
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from kinesync.config import load_config


_ROLES = {"paired_controlled", "paired_zero", "head_stress", "extra_stress"}
_ROLE_CAMERAS = {
    "paired_controlled": ["head", "extra"],
    "paired_zero": ["head", "extra"],
    "head_stress": ["head"],
    "extra_stress": ["extra"],
}
_JOINTS = {f"joint{index}" for index in range(1, 7)}


def _provenance_ids(payload: Mapping[str, Any], name: str) -> tuple[str, ...]:
    values = payload.get(name)
    if not isinstance(values, list) or not values:
        raise ValueError(f"RT3 matrix provenance requires {name}")
    normalized = tuple(str(value) for value in values)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"RT3 matrix provenance {name} contains duplicates")
    return normalized


def load_rt3_cases(
    path: str | Path,
    dataset,
    factors: Mapping[str, Any],
    old_validation_states: Sequence[str],
) -> list[dict[str, Any]]:
    """Load RT3 cases only after validating their immutable data boundaries."""

    matrix = load_config(path)
    split = str(matrix.get("split", ""))
    if split not in {"train", "validation"}:
        raise ValueError("RT3 matrix split must be train or validation")
    provenance = matrix.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("RT3 matrix requires provenance")
    declared_factor_states = _provenance_ids(provenance, "factor_train_state_ids")
    actual_factor_states = tuple(str(value) for value in factors["train_state_ids"])
    if declared_factor_states != actual_factor_states:
        raise ValueError("RT3 factor-fit provenance does not match factors")
    declared_old_states = set(
        _provenance_ids(provenance, "old_validation_state_ids")
    )
    if declared_old_states != set(str(value) for value in old_validation_states):
        raise ValueError("RT3 old validation provenance does not match caller")

    raw_cases = matrix.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("RT3 matrix requires explicit cases")
    cases: list[dict[str, Any]] = []
    seen_states: set[str] = set()
    for raw in raw_cases:
        if not isinstance(raw, Mapping):
            raise ValueError("Each RT3 case must be a mapping")
        state_id = str(raw["state_id"])
        if state_id in seen_states:
            raise ValueError(f"RT3 matrix reuses state {state_id}")
        seen_states.add(state_id)
        role = str(raw["case_role"])
        if role not in _ROLES:
            raise ValueError(f"Unknown RT3 case role: {role}")
        cameras = [str(value) for value in raw["cameras"]]
        if cameras != _ROLE_CAMERAS[role]:
            raise ValueError(f"Camera mode does not match RT3 role {role}")
        available = set(dataset.state_ids(split=split, cameras=cameras))
        if state_id not in available:
            raise ValueError(f"State {state_id} is unavailable in {split} for {cameras}")
        offsets = {str(name): float(value) for name, value in raw["offsets_rad"].items()}
        if not offsets or set(offsets) - _JOINTS:
            raise ValueError(f"Invalid physical offsets for {state_id}")
        if not all(math.isfinite(value) for value in offsets.values()):
            raise ValueError(f"Nonfinite physical offset for {state_id}")
        magnitude_deg = float(raw["magnitude_deg"])
        actual_magnitude = math.degrees(max(abs(value) for value in offsets.values()))
        if not math.isclose(magnitude_deg, actual_magnitude, abs_tol=1e-5):
            raise ValueError(f"Magnitude metadata mismatch for {state_id}")
        is_zero = actual_magnitude <= 1e-12
        if (role == "paired_zero") != is_zero:
            raise ValueError(f"Zero-control role mismatch for {state_id}")
        cases.append(
            {
                "state_id": state_id,
                "split": split,
                "camera_names": cameras,
                "offset_name": str(raw["offset_name"]),
                "offsets_rad": offsets,
                "case_role": role,
                "magnitude_deg": magnitude_deg,
                "seed": int(raw.get("seed", matrix.get("seed", 41))),
            }
        )

    factor_states = set(actual_factor_states)
    if split == "train" and seen_states & factor_states:
        raise ValueError("RT3 guard-dev overlaps factor-fit states")
    if split == "validation" and seen_states & declared_old_states:
        raise ValueError("RT3 guard-test overlaps old validation states")
    expected_counts = Counter(
        {str(name): int(value) for name, value in matrix["expected_counts"].items()}
    )
    actual_counts = Counter(case["case_role"] for case in cases)
    if actual_counts != expected_counts:
        raise ValueError(
            f"RT3 role counts differ from frozen expectation: {actual_counts}"
        )
    return cases


__all__ = ["load_rt3_cases"]
