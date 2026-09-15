"""Frozen RT5 observation-audit cases derived from RT3 validation data."""

from __future__ import annotations

from collections import Counter
import hashlib
from pathlib import Path
from typing import Any, Mapping

from kinesync.config import load_config
from kinesync.experiments.rt3_matrix import load_rt3_cases


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rt5_cases(
    path: str | Path,
    dataset,
    factors: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Select the frozen paired one-degree and zero-control audit."""

    matrix_path = Path(path).expanduser().resolve()
    matrix = load_config(matrix_path)
    if str(matrix.get("split", "")) != "validation":
        raise ValueError("RT5 observation audit must use validation cases")
    source_value = Path(str(matrix["source_matrix"]))
    source_path = (
        source_value.resolve()
        if source_value.is_absolute()
        else (matrix_path.parent / source_value).resolve()
    )
    if _sha256(source_path) != str(matrix["source_matrix_sha256"]):
        raise ValueError("RT5 source matrix hash mismatch")
    source = load_config(source_path)
    provenance = source.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("RT5 source matrix lacks RT3 provenance")
    source_cases = load_rt3_cases(
        source_path,
        dataset,
        factors,
        provenance["old_validation_state_ids"],
    )
    magnitude = float(matrix["selection"]["controlled_magnitude_deg"])
    include_zero = bool(matrix["selection"]["include_paired_zero"])
    cases: list[dict[str, Any]] = []
    for source_case in source_cases:
        role = str(source_case["case_role"])
        audit_role = None
        if role == "paired_controlled" and abs(
            float(source_case["magnitude_deg"]) - magnitude
        ) <= 1e-9:
            audit_role = "paired_one_degree"
        elif role == "paired_zero" and include_zero:
            audit_role = "paired_zero"
        if audit_role is None:
            continue
        case = dict(source_case)
        case["audit_role"] = audit_role
        case["trial_id"] = (
            f"{source_case['state_id']}_{source_case['offset_name']}"
        )
        cases.append(case)

    actual = Counter(case["audit_role"] for case in cases)
    expected = Counter(
        {str(name): int(value) for name, value in matrix["expected_counts"].items()}
    )
    if actual != expected:
        raise ValueError(f"RT5 case counts differ from frozen expectation: {actual}")
    if len({case["trial_id"] for case in cases}) != len(cases):
        raise ValueError("RT5 trial IDs are not unique")
    return cases


__all__ = ["load_rt5_cases"]
