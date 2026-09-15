"""Deterministic causal windows derived from frozen RT3 cases."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from kinesync.config import load_config
from kinesync.experiments.rt3_matrix import load_rt3_cases


def causal_state_window(
    center_state_id: str, ordered_state_ids: Sequence[str], size: int
) -> list[str]:
    """Return a fixed causal window ending at the requested center state."""

    if size <= 0:
        raise ValueError("Temporal window size must be positive")
    ordered = [str(value) for value in ordered_state_ids]
    if len(set(ordered)) != len(ordered):
        raise ValueError("Ordered state IDs contain duplicates")
    center = str(center_state_id)
    try:
        center_index = ordered.index(center)
    except ValueError as error:
        raise ValueError(f"Center state {center} is not available") from error
    start = center_index - size + 1
    if start < 0:
        raise ValueError(f"Center state {center} has fewer than five causal states")
    return ordered[start : center_index + 1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rt4_case_fingerprint(cases: Sequence[Mapping[str, Any]]) -> str:
    """Hash the ordered, normalized derived cases."""

    encoded = json.dumps(
        list(cases),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_rt4_cases(
    path: str | Path,
    dataset,
    factors: Mapping[str, Any],
    guard_source_states: Sequence[str],
) -> list[dict[str, Any]]:
    """Derive frozen causal windows while preserving RT3 data boundaries."""

    matrix_path = Path(path).expanduser().resolve()
    matrix = load_config(matrix_path)
    if str(matrix.get("split", "")) != "validation":
        raise ValueError("RT4 development matrix must use validation centers")
    source_value = Path(str(matrix["source_matrix"]))
    source_path = (
        source_value.resolve()
        if source_value.is_absolute()
        else (matrix_path.parent / source_value).resolve()
    )
    if _sha256(source_path) != str(matrix["source_matrix_sha256"]):
        raise ValueError("RT4 source matrix hash mismatch")
    source_payload = load_config(source_path)
    provenance = source_payload.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("RT4 source matrix lacks RT3 provenance")
    old_states = [str(value) for value in provenance["old_validation_state_ids"]]
    source_cases = load_rt3_cases(source_path, dataset, factors, old_states)

    eligible_roles = [str(value) for value in matrix["eligible_roles"]]
    if not eligible_roles or set(eligible_roles) - {
        "paired_controlled",
        "paired_zero",
    }:
        raise ValueError("RT4 supports paired controlled and zero roles only")
    window = matrix.get("window")
    if not isinstance(window, Mapping):
        raise ValueError("RT4 matrix requires a window mapping")
    if str(window.get("direction")) != "causal":
        raise ValueError("RT4 currently supports causal windows only")
    size = int(window["size"])
    if size != 5:
        raise ValueError("RT4 development requires a five-state window")
    minimum_center = str(window["minimum_center_state_id"])

    guard_states = {str(value) for value in guard_source_states}
    ordered = dataset.state_ids(
        split="validation", cameras=["head", "extra"]
    )
    cases: list[dict[str, Any]] = []
    for source in source_cases:
        if source["case_role"] not in eligible_roles:
            continue
        center = str(source["state_id"])
        if center < minimum_center:
            continue
        context = causal_state_window(center, ordered, size)
        overlap = set(context) & guard_states
        if overlap:
            raise ValueError(
                "RT4 temporal window overlaps guard calibration states: "
                f"{sorted(overlap)}"
            )
        case = dict(source)
        case.update(
            {
                "center_state_id": center,
                "context_state_ids": context,
                "window_size": size,
                "source_trial_id": f"{center}_{source['offset_name']}",
            }
        )
        cases.append(case)

    actual = Counter(case["case_role"] for case in cases)
    expected = Counter(
        {str(name): int(value) for name, value in matrix["expected_counts"].items()}
    )
    if actual != expected:
        raise ValueError(f"RT4 role counts differ from frozen expectation: {actual}")
    return cases


__all__ = [
    "causal_state_window",
    "load_rt4_cases",
    "rt4_case_fingerprint",
]
