"""Frozen RT6 validation matrix contract.

The YAML ``cases`` sequence is the canonical order.  Reordering otherwise
identical cases changes the fingerprint because it creates a distinct frozen
trial schedule; repeated loads of the same file remain byte-for-byte stable.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from kinesync.config import load_config


_ARM_JOINTS = frozenset(f"joint{index}" for index in range(1, 7))
_CAMERAS = ("head", "extra")
_ROLES = frozenset({"single_joint", "zero_control", "multi_joint_diagnostic"})
_UNTOUCHED_VALIDATION_IDS = frozenset(
    {
        "state_0359", "state_0361", "state_0362", "state_0364",
        "state_0365", "state_0366", "state_0367", "state_0368", "state_0369",
        "state_0371", "state_0372", "state_0373", "state_0374", "state_0375",
        "state_0378", "state_0379",
    }
)
_PREVIOUSLY_USED_IDS = frozenset(
    {f"state_{index:04d}" for index in range(0, 44)}
    | {f"state_{index:04d}" for index in range(299, 359)}
    | {f"state_{index:04d}" for index in range(52, 287, 13)}
    | {
        "state_0100", "state_0199", "state_0360", "state_0363",
        "state_0370", "state_0376", "state_0377",
    }
)
_EXPECTED_MULTI = {
    "diagnostic_multi_2p5": {"joint2": 2.5, "joint4": -2.5},
    "diagnostic_multi_5": {"joint2": 5.0, "joint5": -5.0},
}


@dataclass(frozen=True)
class RT6MatrixCase:
    """One predeclared paired-camera perturbation trial."""

    case_id: str
    state_id: str
    split: str
    cameras: tuple[str, ...]
    role: str
    magnitude_deg: float
    offsets_deg: tuple[tuple[str, float], ...]
    offsets_rad: tuple[tuple[str, float], ...]
    seed: int


@dataclass(frozen=True)
class RT6MatrixSummary:
    """Trial and state counts without treating camera observations as scenes."""

    trial_count: int
    unique_state_count: int
    max_trials_per_state: int
    role_counts: Mapping[str, int]
    single_joint_counts_deg: Mapping[float, int]


@dataclass(frozen=True)
class RT6Matrix:
    """Immutable, validated RT6 formal matrix for CLI consumption."""

    split: str
    cases: tuple[RT6MatrixCase, ...]
    summary: RT6MatrixSummary
    fingerprint: str
    source_file_sha256: str

    @property
    def trial_count(self) -> int:
        return self.summary.trial_count

    @property
    def unique_state_count(self) -> int:
        return self.summary.unique_state_count


def _canonical_cases(cases: tuple[RT6MatrixCase, ...]) -> list[dict[str, Any]]:
    return [
        {
            "case_id": case.case_id,
            "state_id": case.state_id,
            "split": case.split,
            "cameras": list(case.cameras),
            "role": case.role,
            "magnitude_deg": case.magnitude_deg,
            "offsets_deg": list(case.offsets_deg),
            "offsets_rad": list(case.offsets_rad),
            "seed": case.seed,
        }
        for case in cases
    ]


def _canonical_payload(
    matrix: Mapping[str, Any],
    cases: tuple[RT6MatrixCase, ...],
    untouched_ids: tuple[str, ...],
    previous_ids: tuple[str, ...],
) -> dict[str, Any]:
    contract = matrix["matrix_contract"]
    return {
        "experiment": str(matrix["experiment"]),
        "split": str(matrix["split"]),
        "canonical_case_order": str(matrix["canonical_case_order"]),
        "matrix_contract": {
            "trial_count": int(contract["trial_count"]),
            "unique_state_count": int(contract["unique_state_count"]),
            "max_trials_per_state": int(contract["max_trials_per_state"]),
            "cameras": [str(value) for value in contract["cameras"]],
            "role_counts": {
                str(name): int(value) for name, value in contract["role_counts"].items()
            },
            "single_joint_counts_deg": sorted(
                (float(name), int(value))
                for name, value in contract["single_joint_counts_deg"].items()
            ),
        },
        "untouched_validation_state_ids": list(untouched_ids),
        "exclusion_provenance": {
            "previously_used_state_ids": list(previous_ids),
        },
        "cases": _canonical_cases(cases),
    }


def _fingerprint(
    matrix: Mapping[str, Any],
    cases: tuple[RT6MatrixCase, ...],
    untouched_ids: tuple[str, ...],
    previous_ids: tuple[str, ...],
) -> str:
    encoded = json.dumps(
        _canonical_payload(matrix, cases, untouched_ids, previous_ids),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _as_mapping(value: Any, field: str, case_id: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"RT6 {field} must be a mapping for {case_id}")
    return value


def _parse_offsets(raw: Any, field: str, case_id: str) -> dict[str, float]:
    mapping = _as_mapping(raw, field, case_id)
    offsets = {str(joint): float(value) for joint, value in mapping.items()}
    if set(offsets) - _ARM_JOINTS:
        raise ValueError(f"RT6 {field} has unknown joint for {case_id}")
    if not all(math.isfinite(value) for value in offsets.values()):
        raise ValueError(f"RT6 {field} has nonfinite offset for {case_id}")
    return offsets


def _parse_case(raw: Any, matrix_split: str) -> RT6MatrixCase:
    if not isinstance(raw, Mapping):
        raise ValueError("Each RT6 case must be a mapping")
    case_id = str(raw.get("case_id", ""))
    if not case_id:
        raise ValueError("RT6 case requires a case_id")
    state_id = str(raw.get("state_id", ""))
    if not state_id:
        raise ValueError(f"RT6 case {case_id} requires a state_id")
    split = str(raw.get("split", ""))
    if split != matrix_split:
        raise ValueError(f"RT6 case split mismatch for {case_id}")
    cameras = tuple(str(value) for value in raw.get("cameras", ()))
    if cameras != _CAMERAS:
        raise ValueError(f"RT6 case cameras must be head+extra for {case_id}")
    role = str(raw.get("role", ""))
    if role not in _ROLES:
        raise ValueError(f"RT6 case has unknown role for {case_id}")
    magnitude_deg = float(raw.get("magnitude_deg", float("nan")))
    if not math.isfinite(magnitude_deg):
        raise ValueError(f"RT6 magnitude is nonfinite for {case_id}")
    degrees = _parse_offsets(raw.get("offsets_deg"), "offsets_deg", case_id)
    radians = _parse_offsets(raw.get("offsets_rad"), "offsets_rad", case_id)
    if set(degrees) != set(radians):
        raise ValueError(f"RT6 degree/radian joints differ for {case_id}")
    for joint, degrees_value in degrees.items():
        if not math.isclose(radians[joint], math.radians(degrees_value), abs_tol=1e-12):
            raise ValueError(f"RT6 degree/radian mismatch for {case_id}")
    actual_magnitude = max((abs(value) for value in degrees.values()), default=0.0)
    if not math.isclose(magnitude_deg, actual_magnitude, abs_tol=1e-12):
        raise ValueError(f"RT6 magnitude metadata mismatch for {case_id}")
    seed = raw.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError(f"RT6 case needs a deterministic nonnegative integer seed: {case_id}")
    return RT6MatrixCase(
        case_id=case_id,
        state_id=state_id,
        split=split,
        cameras=cameras,
        role=role,
        magnitude_deg=magnitude_deg,
        offsets_deg=tuple(sorted(degrees.items())),
        offsets_rad=tuple(sorted(radians.items())),
        seed=seed,
    )


def _ordered_unique_ids(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"RT6 provenance requires {name}")
    ids = tuple(str(item) for item in value)
    if len(set(ids)) != len(ids):
        raise ValueError(f"RT6 provenance {name} contains duplicates")
    return ids


def _validate_provenance(matrix: Mapping[str, Any]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    untouched_ids = _ordered_unique_ids(
        matrix.get("untouched_validation_state_ids"),
        "untouched_validation_state_ids",
    )
    if frozenset(untouched_ids) != _UNTOUCHED_VALIDATION_IDS:
        raise ValueError("RT6 untouched validation state provenance does not match frozen set")
    provenance = matrix.get("exclusion_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("RT6 matrix requires exclusion provenance")
    previous_ids = _ordered_unique_ids(
        provenance.get("previously_used_state_ids"),
        "previously_used_state_ids",
    )
    if frozenset(previous_ids) != _PREVIOUSLY_USED_IDS:
        raise ValueError("RT6 exclusion provenance does not cover exact previously used states")
    return untouched_ids, previous_ids


def _validate_contract(matrix: Mapping[str, Any], cases: tuple[RT6MatrixCase, ...]) -> RT6MatrixSummary:
    contract = matrix.get("matrix_contract")
    if not isinstance(contract, Mapping):
        raise ValueError("RT6 matrix requires matrix_contract")
    if tuple(str(value) for value in contract.get("cameras", ())) != _CAMERAS:
        raise ValueError("RT6 matrix contract cameras must be exactly head+extra")
    trial_count = int(contract.get("trial_count", -1))
    if trial_count != 32 or len(cases) != 32:
        raise ValueError("RT6 matrix must contain exactly 32 trials")
    state_counts = Counter(case.state_id for case in cases)
    if set(state_counts) & _PREVIOUSLY_USED_IDS:
        raise ValueError("RT6 matrix overlaps previously used state IDs")
    if len(state_counts) != 16 or int(contract.get("unique_state_count", -1)) != 16:
        raise ValueError("RT6 matrix must contain exactly 16 unique states")
    if any(count > 2 for count in state_counts.values()):
        raise ValueError("RT6 matrix reuses a state more than two times")
    if any(count != 2 for count in state_counts.values()):
        raise ValueError("RT6 matrix must use every state exactly two times")
    if set(state_counts) != _UNTOUCHED_VALIDATION_IDS:
        raise ValueError("RT6 cases must use every and only untouched validation state")
    case_ids = [case.case_id for case in cases]
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("RT6 case IDs must be unique")
    if len({case.seed for case in cases}) != len(cases):
        raise ValueError("RT6 deterministic seeds must be unique per trial")
    role_counts = Counter(case.role for case in cases)
    expected_roles = {str(name): int(value) for name, value in contract.get("role_counts", {}).items()}
    if dict(role_counts) != expected_roles:
        raise ValueError("RT6 role counts differ from frozen contract")
    single = [case for case in cases if case.role == "single_joint"]
    single_counts = Counter(case.magnitude_deg for case in single)
    expected_single = {float(name): int(value) for name, value in contract.get("single_joint_counts_deg", {}).items()}
    if dict(single_counts) != expected_single:
        raise ValueError("RT6 single-joint magnitude counts differ from frozen contract")
    coverage = {magnitude: {joint: set() for joint in _ARM_JOINTS} for magnitude in (2.5, 5.0)}
    for case in single:
        degrees = dict(case.offsets_deg)
        if len(degrees) != 1 or case.magnitude_deg not in coverage:
            raise ValueError("RT6 single-joint case shape differs from frozen contract")
        joint, value = next(iter(degrees.items()))
        if not math.isclose(abs(value), case.magnitude_deg, abs_tol=1e-12):
            raise ValueError("RT6 single-joint offset magnitude mismatch")
        coverage[case.magnitude_deg][joint].add(1 if value > 0 else -1)
    if any(signs != {-1, 1} for by_joint in coverage.values() for signs in by_joint.values()):
        raise ValueError("RT6 single-joint signed coverage differs from frozen contract")
    zeros = [case for case in cases if case.role == "zero_control"]
    zero_joint_counts: Counter[str] = Counter()
    for case in zeros:
        degrees = dict(case.offsets_deg)
        radians = dict(case.offsets_rad)
        if (
            case.magnitude_deg != 0.0
            or len(degrees) != 1
            or len(radians) != 1
            or set(degrees) != set(radians)
            or next(iter(degrees.values())) != 0.0
            or next(iter(radians.values())) != 0.0
        ):
            raise ValueError("RT6 zero control must select exactly one zero-valued joint")
        zero_joint_counts[next(iter(degrees))] += 1
    if zero_joint_counts != Counter({joint: 1 for joint in _ARM_JOINTS}):
        raise ValueError("RT6 zero control joint coverage must include each arm joint once")
    diagnostics = [case for case in cases if case.role == "multi_joint_diagnostic"]
    actual_multi = {case.case_id: dict(case.offsets_deg) for case in diagnostics}
    if actual_multi != _EXPECTED_MULTI:
        raise ValueError("RT6 multi-joint diagnostics differ from frozen contract")
    max_trials = max(state_counts.values())
    if max_trials != int(contract.get("max_trials_per_state", -1)):
        raise ValueError("RT6 maximum state repetitions differ from frozen contract")
    return RT6MatrixSummary(
        trial_count=len(cases),
        unique_state_count=len(state_counts),
        max_trials_per_state=max_trials,
        role_counts=MappingProxyType(dict(sorted(role_counts.items()))),
        single_joint_counts_deg=MappingProxyType(dict(sorted(single_counts.items()))),
    )


def load_rt6_matrix(path: str | Path) -> RT6Matrix:
    """Load the predeclared 32-trial RT6 validation matrix.

    No dataset, model, or outcome is used by this loader.  It validates the
    frozen schedule and returns immutable records for the later formal CLI.
    """

    matrix_path = Path(path).expanduser().resolve()
    source_file_sha256 = hashlib.sha256(matrix_path.read_bytes()).hexdigest()
    matrix = load_config(matrix_path)
    if str(matrix.get("experiment", "")) != "rt6_piper_fresh_matrix":
        raise ValueError("Unexpected RT6 matrix experiment")
    split = str(matrix.get("split", ""))
    if split != "validation":
        raise ValueError("RT6 primary matrix split must be validation")
    if str(matrix.get("canonical_case_order", "")) != "declared_cases":
        raise ValueError("RT6 matrix must declare canonical case order")
    untouched_ids, previous_ids = _validate_provenance(matrix)
    raw_cases = matrix.get("cases")
    if not isinstance(raw_cases, list):
        raise ValueError("RT6 matrix requires explicit cases")
    cases = tuple(_parse_case(raw, split) for raw in raw_cases)
    summary = _validate_contract(matrix, cases)
    return RT6Matrix(
        split=split,
        cases=cases,
        summary=summary,
        fingerprint=_fingerprint(matrix, cases, untouched_ids, previous_ids),
        source_file_sha256=source_file_sha256,
    )


__all__ = ["RT6Matrix", "RT6MatrixCase", "RT6MatrixSummary", "load_rt6_matrix"]
