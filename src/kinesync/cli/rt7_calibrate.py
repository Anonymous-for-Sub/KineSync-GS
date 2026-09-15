"""Emit an immutable train-only RT7 event-complete guard artifact."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from kinesync.config import load_config
from kinesync.guard.calibration import CalibrationRecord, calibrate_event_gain_guard
from kinesync.guard.schema import GuardThresholds, UpdateEvidence
from kinesync.runs.artifacts import RunArtifacts


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("candidate table must be nonempty")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _parse_bool(value: str, *, field: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"RT3 calibration row has invalid {field}: {value!r}")


def _load_rt3_records(path: Path) -> tuple[CalibrationRecord, ...]:
    records: list[CalibrationRecord] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("split") != "train":
                raise ValueError("RT7 calibration accepts train rows only")
            role = row.get("case_role")
            if role == "paired_zero":
                trial_type = "zero"
            elif role == "paired_controlled":
                trial_type = "controlled"
            else:
                raise ValueError(f"RT3 calibration row has unsupported case role: {role!r}")
            try:
                evidence = UpdateEvidence(
                    view_count=int(row["view_count"]),
                    visual_gain_ratio=float(row["visual_gain_ratio"]),
                    gradient_cosine=float(row["gradient_cosine"]),
                    correction_cosine=float(row["correction_cosine"]),
                    relative_correction_disagreement=float(
                        row["relative_correction_disagreement"]
                    ),
                    candidate_bound_fraction=float(row["candidate_bound_fraction"]),
                    finite=_parse_bool(row["evidence_finite"], field="evidence_finite"),
                )
                records.append(
                    CalibrationRecord(
                        evidence=evidence,
                        trial_type=trial_type,
                        candidate_success=_parse_bool(
                            row["candidate_success"], field="candidate_success"
                        ),
                        state_id=str(row["state_id"]),
                    )
                )
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError("Malformed RT3 calibration row") from error
    if not records:
        raise ValueError("RT3 calibration rows must be nonempty")
    return tuple(records)


def _load_frozen_rt6_guard(path: Path) -> GuardThresholds:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("frozen") is not True:
            raise ValueError("RT7 calibration requires a frozen RT6 guard")
        if payload.get("source_split") != "train":
            raise ValueError("RT7 calibration requires a train-derived RT6 guard")
        return GuardThresholds(**dict(payload["thresholds"]))
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Malformed frozen RT6 guard") from error


def _candidate_rows(calibration) -> list[dict[str, Any]]:
    return [
        {
            **asdict(item.thresholds),
            "controlled_coverage": item.controlled_coverage,
            "zero_stability": item.zero_stability,
            "guarded_success": item.guarded_success,
            "minimum_accepted_margin": item.minimum_accepted_margin,
            "feasible": item.feasible,
        }
        for item in calibration.candidate_table
    ]


def execute_rt7_calibration(
    config: Mapping[str, Any], *, run_id: str | None = None
) -> Path:
    """Calibrate a development-only RT7 guard from immutable train evidence."""

    try:
        inputs = config["inputs"]
        rows_path = Path(inputs["calibration_rows"]).expanduser().resolve()
        guard_path = Path(inputs["frozen_rt6_guard"]).expanduser().resolve()
    except (KeyError, TypeError) as error:
        raise ValueError("RT7 calibration config requires immutable input paths") from error
    if not rows_path.is_file() or not guard_path.is_file():
        raise FileNotFoundError("RT7 calibration input artifact does not exist")

    records = _load_rt3_records(rows_path)
    fixed_thresholds = _load_frozen_rt6_guard(guard_path)
    calibration_config = config.get("calibration", {})
    minimum_zero_stability = float(calibration_config.get("minimum_zero_stability", 0.9))
    minimum_controlled_coverage = float(
        calibration_config.get("minimum_controlled_coverage", 0.6)
    )
    calibration = calibrate_event_gain_guard(
        records,
        fixed_thresholds=fixed_thresholds,
        minimum_zero_stability=minimum_zero_stability,
        minimum_controlled_coverage=minimum_controlled_coverage,
    )
    rows_sha256 = _sha256(rows_path)
    guard_sha256 = _sha256(guard_path)
    hashes = {
        "source_calibration_rows_sha256": rows_sha256,
        "frozen_rt6_guard_sha256": guard_sha256,
    }
    candidate_rows = _candidate_rows(calibration)
    fixed_non_gain_thresholds = {
        "min_gradient_cosine": fixed_thresholds.min_gradient_cosine,
        "min_correction_cosine": fixed_thresholds.min_correction_cosine,
        "max_relative_correction_disagreement": (
            fixed_thresholds.max_relative_correction_disagreement
        ),
    }
    constraints = {
        "minimum_zero_stability": minimum_zero_stability,
        "minimum_controlled_coverage": minimum_controlled_coverage,
    }
    guard_payload = {
        "schema_version": 1,
        "frozen": True,
        "formal_evidence": False,
        "source_split": "train",
        "threshold_generation": calibration.threshold_generation,
        "evaluated_candidate_count": calibration.evaluated_candidate_count,
        "source_calibration_rows": str(rows_path),
        "frozen_rt6_guard": str(guard_path),
        **hashes,
        "source_state_ids": list(calibration.source_state_ids),
        "thresholds": asdict(calibration.thresholds),
        "fixed_non_gain_thresholds": fixed_non_gain_thresholds,
        "constraints": constraints,
        "controlled_coverage": calibration.controlled_coverage,
        "zero_stability": calibration.zero_stability,
        "guarded_success": calibration.guarded_success,
        "calibration_fingerprint": calibration.fingerprint,
    }
    metrics = {
        "formal_evidence": False,
        "source_split": "train",
        "source_row_count": len(records),
        "source_state_ids": list(calibration.source_state_ids),
        "threshold_generation": calibration.threshold_generation,
        "evaluated_candidate_count": calibration.evaluated_candidate_count,
        "thresholds": asdict(calibration.thresholds),
        "fixed_non_gain_thresholds": fixed_non_gain_thresholds,
        "constraints": constraints,
        "controlled_coverage": calibration.controlled_coverage,
        "zero_stability": calibration.zero_stability,
        "guarded_success": calibration.guarded_success,
        "calibration_fingerprint": calibration.fingerprint,
        **hashes,
    }
    run = RunArtifacts.create(
        root=config.get("runs_root", "runs"),
        experiment=str(config.get("experiment", "rt7_event_guard_calibration")),
        run_id=run_id,
        config=config,
        assets={
            "rt3_calibration_rows": rows_path,
            "frozen_rt6_guard": guard_path,
        },
        target_provenance="real_observation_guard_calibration",
        observation_backend="rt7_event_gain_train_calibration",
    )
    _write_json(run.path / "guard.json", guard_payload)
    _write_csv(run.path / "candidate_table.csv", candidate_rows)
    run.write_metrics(metrics)
    _write_json(run.path / "hashes.json", hashes)
    return run.path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id")
    arguments = parser.parse_args()
    print(execute_rt7_calibration(load_config(arguments.config), run_id=arguments.run_id))


if __name__ == "__main__":
    main()
