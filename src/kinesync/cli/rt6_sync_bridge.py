"""Freeze an RT6 detectability profile and audit RT3 validation traces."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

from kinesync.config import load_config
from kinesync.data.real_observation import RouteARealObservationDataset
from kinesync.geometry.urdf import TorchURDFKinematics
from kinesync.guard.schema import GuardDecision
from kinesync.runs.artifacts import RunArtifacts
from kinesync.sync.calibration import (
    calibration_rows_sha256,
    calibrate_detectability,
    migrate_legacy_calibration_rows,
)
from kinesync.sync.commit import commit_synchronized_state
from kinesync.sync.mujoco_asset import convert_urdf_for_mujoco
from kinesync.sync.mujoco_bridge import MujocoStateBridge, compare_fk_parity
from kinesync.sync.schema import SynchronizedStateFrame


_ZERO_TOLERANCE = 1e-12
_RECOVERY_THRESHOLD = 0.60
_DEVELOPMENT_AUDIT = "development_audit_only"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _path(value: object, *, field: str) -> Path:
    path = Path(str(value)).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{field} does not exist: {path}")
    return path


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    return value


def _boolean(value: object, *, field: str) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise ValueError(f"{field} must be boolean")


def _finite(value: object, *, field: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    return number


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError(f"CSV has no rows: {path}")
    return rows


def _write_csv(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    materialized = [dict(row) for row in rows]
    if not materialized:
        raise ValueError("cannot write an empty CSV")
    fieldnames: list[str] = []
    for row in materialized:
        for field in row:
            if field not in fieldnames:
                fieldnames.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(materialized)


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _load_factor(path: Path, full_joint_order: Sequence[str]) -> tuple[dict[str, Any], str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    factor = dict(_mapping(payload, field="frozen factor file"))
    if factor.get("schema_version") != 1:
        raise ValueError("frozen factor must use schema_version=1")
    if factor.get("frozen") is not True or factor.get("fit_split") != "train":
        raise ValueError("frozen factor must be frozen and fit on train")
    raw_names = factor.get("joint_names")
    if not isinstance(raw_names, Sequence) or isinstance(raw_names, (str, bytes)):
        raise ValueError("frozen factor joint_names must be a sequence")
    factor_names = tuple(str(name) for name in raw_names)
    if len(full_joint_order) != 8:
        raise ValueError("full_joint_order must contain exactly eight PiPER joints")
    if len(factor_names) != 6 or tuple(full_joint_order[:6]) != factor_names:
        raise ValueError("frozen factor joint order must be the six-joint arm prefix")
    raw_joint_zero = factor.get("joint_zero_rad")
    if not isinstance(raw_joint_zero, Sequence) or isinstance(raw_joint_zero, (str, bytes)):
        raise ValueError("frozen factor joint_zero_rad must be a six-value sequence")
    if len(raw_joint_zero) != 6:
        raise ValueError("frozen factor joint_zero_rad must contain six values")
    if not all(math.isfinite(float(value)) for value in raw_joint_zero):
        raise ValueError("frozen factor joint_zero_rad must be finite")
    return factor, _sha256(path)


def _real_dataset(config: Mapping[str, Any]) -> RouteARealObservationDataset:
    source = _mapping(config.get("real_observations"), field="real_observations")
    return RouteARealObservationDataset(
        records=_mapping(source.get("records"), field="real_observations.records"),
        image_roots=_mapping(
            source.get("image_roots"), field="real_observations.image_roots"
        ),
        mask_roots=_mapping(
            source.get("mask_roots"), field="real_observations.mask_roots"
        ),
        max_pair_qpos_delta_rad=_finite(
            source.get("max_pair_qpos_delta_rad", 0.02),
            field="real_observations.max_pair_qpos_delta_rad",
        ),
    )


def _load_cases(path: Path, full_joint_order: Sequence[str]) -> list[dict[str, Any]]:
    matrix = load_config(path)
    if matrix.get("split") != "validation":
        raise ValueError("RT3 audit matrix must have split=validation")
    raw_cases = matrix.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("RT3 audit matrix requires nonempty cases")
    known = set(full_joint_order)
    cases: list[dict[str, Any]] = []
    seen_trials: set[str] = set()
    seen_states: set[str] = set()
    for raw in raw_cases:
        item = _mapping(raw, field="RT3 audit case")
        state_id = str(item.get("state_id", "")).strip()
        offset_name = str(item.get("offset_name", "")).strip()
        if not state_id or not offset_name:
            raise ValueError("RT3 audit cases require state_id and offset_name")
        raw_cameras = item.get("cameras")
        if not isinstance(raw_cameras, Sequence) or isinstance(
            raw_cameras, (str, bytes)
        ):
            raise ValueError("RT3 audit cases require cameras")
        cameras = tuple(str(camera).strip() for camera in raw_cameras)
        if not cameras or not all(cameras) or len(set(cameras)) != len(cameras):
            raise ValueError("RT3 audit case cameras must be nonempty and unique")
        if state_id in seen_states:
            raise ValueError(f"RT3 audit matrix reuses state_id: {state_id}")
        seen_states.add(state_id)
        trial_id = f"{state_id}_{offset_name}"
        if trial_id in seen_trials:
            raise ValueError(f"RT3 audit matrix reuses trial_id: {trial_id}")
        seen_trials.add(trial_id)
        raw_offsets = _mapping(item.get("offsets_rad"), field="offsets_rad")
        offsets = {str(name): _finite(value, field="offsets_rad") for name, value in raw_offsets.items()}
        if not offsets or set(offsets) - known:
            raise ValueError(f"RT3 audit case has unknown or empty offsets: {trial_id}")
        cases.append(
            {
                "trial_id": trial_id,
                "state_id": state_id,
                "case_role": str(item.get("case_role", "")),
                "cameras": cameras,
                "offsets_rad": offsets,
            }
        )
    return cases


def _guard_decisions(
    rows: Iterable[Mapping[str, str]],
    *,
    method: str,
    cases: Sequence[Mapping[str, Any]],
) -> dict[str, GuardDecision]:
    expected = {str(case["trial_id"]): str(case["state_id"]) for case in cases}
    decisions: dict[str, GuardDecision] = {}
    for row in rows:
        if row.get("method") != method:
            continue
        trial_id = str(row.get("trial_id", ""))
        if trial_id not in expected:
            continue
        if row.get("split") != "validation":
            raise ValueError("RT3 development audit rows require split=validation")
        if row.get("state_id") != expected[trial_id]:
            raise ValueError(f"RT3 result state_id mismatch: {trial_id}")
        if trial_id in decisions:
            raise ValueError(f"RT3 development audit has duplicate result: {trial_id}")
        accepted = _boolean(row.get("candidate_committed"), field="candidate_committed")
        reason = str(row.get("decision_reason", "")).strip()
        if not reason:
            raise ValueError("RT3 development audit requires decision_reason")
        margin = _finite(row.get("minimum_margin"), field="minimum_margin")
        decisions[trial_id] = GuardDecision(accepted, reason, margin)
    if set(decisions) != set(expected):
        missing = sorted(set(expected) - set(decisions))
        raise ValueError(f"RT3 development audit lacks {method} results: {missing}")
    return decisions


def _candidate_recovery_success(
    rows: Iterable[Mapping[str, str]],
    *,
    method: str,
    cases: Sequence[Mapping[str, Any]],
) -> dict[str, bool]:
    expected = {str(case["trial_id"]): str(case["state_id"]) for case in cases}
    successes: dict[str, bool] = {}
    for row in rows:
        if row.get("method") != method:
            continue
        trial_id = str(row.get("trial_id", ""))
        if trial_id not in expected:
            continue
        if row.get("split") != "validation":
            raise ValueError("candidate recovery rows require split=validation")
        if row.get("state_id") != expected[trial_id]:
            raise ValueError(f"candidate recovery state_id mismatch: {trial_id}")
        if trial_id in successes:
            raise ValueError(f"candidate recovery has duplicate result: {trial_id}")
        successes[trial_id] = _boolean(
            row.get("recovery_success"), field="candidate recovery_success"
        )
    if set(successes) != set(expected):
        missing = sorted(set(expected) - set(successes))
        raise ValueError(f"RT3 development audit lacks {method} results: {missing}")
    return successes


def _candidate_corrections(
    rows: Iterable[Mapping[str, str]],
    *,
    candidate_source: str,
    trial_ids: set[str],
    joint_names: Sequence[str],
) -> dict[str, tuple[float, ...]]:
    selected: dict[str, tuple[int, tuple[float, ...]]] = {}
    for row in rows:
        if row.get("candidate_source") != candidate_source:
            continue
        trial_id = str(row.get("trial_id", ""))
        if trial_id not in trial_ids:
            continue
        step = int(row.get("step", ""))
        correction = tuple(
            float(row.get(f"correction_{name}", "")) for name in joint_names
        )
        existing = selected.get(trial_id)
        if existing is not None and step == existing[0]:
            raise ValueError(f"RT3 trace has duplicate final candidate step: {trial_id}")
        if existing is None or step > existing[0]:
            selected[trial_id] = (step, correction)
    if set(selected) != trial_ids:
        missing = sorted(trial_ids - set(selected))
        raise ValueError(f"RT3 development trace lacks {candidate_source}: {missing}")
    return {trial_id: values for trial_id, (_, values) in selected.items()}


def _component_outcomes(
    *,
    offsets: Mapping[str, float],
    candidate_correction: torch.Tensor,
    joint_names: Sequence[str],
    recorded_recovery_success: bool,
) -> dict[str, dict[str, float | bool | None]]:
    outcomes: dict[str, dict[str, float | bool | None]] = {}
    for name, offset in offsets.items():
        index = joint_names.index(name)
        correction = float(candidate_correction[index])
        initial_error = abs(offset)
        if initial_error <= _ZERO_TOLERANCE:
            outcomes[name] = {
                "candidate_success": False,
                "state_error_reduction": None,
            }
            continue
        reduction = 1.0 - abs(offset + correction) / initial_error
        outcomes[name] = {
            "candidate_success": bool(
                recorded_recovery_success and reduction >= _RECOVERY_THRESHOLD
            ),
            "state_error_reduction": reduction,
        }
    return outcomes


def _bridge_exclusion(
    *,
    frame: SynchronizedStateFrame,
    joint_limits: Mapping[str, tuple[float, float]],
) -> dict[str, object] | None:
    violations: list[dict[str, object]] = []
    for index, name in enumerate(frame.joint_names):
        value = float(frame.measured_qpos[index])
        lower, upper = joint_limits[name]
        outside = (lower is not None and value < lower) or (
            upper is not None and value > upper
        )
        if outside:
            violations.append(
                {
                    "joint_name": name,
                    "measured_value": value,
                    "original_limit": [lower, upper],
                }
            )
    if not violations:
        return None
    return {
        "case_id": frame.case_id,
        "state_id": frame.state_id,
        "reason": "bridge_ineligible_measured_outside_limit",
        "violations": violations,
    }


def _validate_input_provenance(
    *,
    guard_path: Path,
    factor_sha256: str,
) -> str:
    payload = json.loads(guard_path.read_text(encoding="utf-8"))
    guard = _mapping(payload, field="RT3 guard")
    if guard.get("frozen") is not True or guard.get("source_split") != "train":
        raise ValueError("RT3 guard must be frozen from train evidence")
    if guard.get("factor_sha256") != factor_sha256:
        raise ValueError("RT3 guard factor hash does not match frozen factor")
    return _sha256(guard_path)


def execute_rt6_sync_bridge(
    config: Mapping[str, Any], *, run_id: str | None = None
) -> Path:
    """Migrate train rows, calibrate RT6, and replay RT3 validation as development only."""

    inputs = _mapping(config.get("inputs"), field="inputs")
    assets = _mapping(config.get("assets"), field="assets")
    audit = _mapping(config.get("audit"), field="audit")
    full_joint_order = tuple(str(name) for name in config.get("full_joint_order", ()))
    if not full_joint_order or len(set(full_joint_order)) != len(full_joint_order):
        raise ValueError("full_joint_order must be nonempty and unique")

    calibration_path = _path(inputs.get("calibration_rows"), field="calibration_rows")
    factors_path = _path(inputs.get("factors"), field="factors")
    guard_path = _path(inputs.get("guard"), field="guard")
    audit_results_path = _path(inputs.get("audit_results"), field="audit_results")
    audit_trace_path = _path(inputs.get("audit_trace"), field="audit_trace")
    audit_matrix_path = _path(inputs.get("audit_matrix"), field="audit_matrix")
    urdf_path = _path(assets.get("urdf"), field="assets.urdf")
    package_root = Path(str(assets.get("package_root", ""))).expanduser().resolve()
    if not package_root.is_dir():
        raise NotADirectoryError(f"assets.package_root is not a directory: {package_root}")

    factor, factor_sha256 = _load_factor(factors_path, full_joint_order)
    guard_sha256 = _validate_input_provenance(
        guard_path=guard_path, factor_sha256=factor_sha256
    )
    legacy_rows = _load_csv(calibration_path)
    cases = _load_cases(audit_matrix_path, full_joint_order)
    calibration_states = {str(row.get("state_id", "")) for row in legacy_rows}
    audit_states = {str(case["state_id"]) for case in cases}
    if not all(calibration_states):
        raise ValueError("calibration rows require state_id")
    if overlap := sorted(calibration_states & audit_states):
        raise ValueError(f"calibration and audit state overlap: {overlap}")

    audit_method = str(audit.get("method", "rt3_full"))
    candidate_method = str(audit.get("candidate_method", "rt2f_unguarded"))
    candidate_source = str(audit.get("candidate_source", "rt2_joint"))
    result_rows = _load_csv(audit_results_path)
    trace_rows = _load_csv(audit_trace_path)
    guard_decisions = _guard_decisions(
        result_rows, method=audit_method, cases=cases
    )
    candidate_recovery_success = _candidate_recovery_success(
        result_rows, method=candidate_method, cases=cases
    )
    candidate_corrections = _candidate_corrections(
        trace_rows,
        candidate_source=candidate_source,
        trial_ids={str(case["trial_id"]) for case in cases},
        joint_names=full_joint_order,
    )
    dataset = _real_dataset(config)
    unknown_cameras = sorted(
        {
            camera
            for case in cases
            for camera in case["cameras"]
            if camera not in dataset.cameras
        }
    )
    if unknown_cameras:
        raise ValueError(f"RT3 audit matrix has unknown cameras: {unknown_cameras}")
    measured_by_trial: dict[str, torch.Tensor] = {}
    for case in cases:
        observations = dataset.load_state(
            str(case["state_id"]), cameras=case["cameras"]
        )
        if any(item.split != "validation" for item in observations.values()):
            raise ValueError("RT3 development observations require split=validation")
        reference = torch.as_tensor(
            np.mean(
                np.stack([item.qpos for item in observations.values()]), axis=0
            ),
            dtype=torch.float32,
        )
        shared_joint = torch.zeros(len(full_joint_order), dtype=torch.float32)
        shared_joint[:6] = torch.as_tensor(
            factor["joint_zero_rad"], dtype=torch.float32
        )
        injected = torch.zeros(len(full_joint_order), dtype=torch.float32)
        for name, value in case["offsets_rad"].items():
            injected[full_joint_order.index(name)] = value
        measured_by_trial[str(case["trial_id"])] = reference + shared_joint + injected

    real_source = _mapping(config.get("real_observations"), field="real_observations")
    observation_records = _mapping(
        real_source.get("records"), field="real_observations.records"
    )
    run = RunArtifacts.create(
        root=config.get("runs_root", "runs"),
        experiment=str(config.get("experiment", "rt6_sync_bridge")),
        run_id=run_id,
        config=config,
        assets={
            "legacy_calibration_rows": calibration_path,
            "frozen_factor": factors_path,
            "frozen_guard": guard_path,
            "rt3_development_results": audit_results_path,
            "rt3_development_trace": audit_trace_path,
            "rt3_validation_matrix": audit_matrix_path,
            "source_urdf": urdf_path,
            **{
                f"real_observation_records_{camera}": observation_records[camera]
                for camera in dataset.cameras
            },
        },
        target_provenance="real_observation_guard_evaluation",
        observation_backend="rt6_synchronized_trace_reconstruction",
    )

    migrated_rows = migrate_legacy_calibration_rows(
        legacy_rows, joint_names=full_joint_order, factors_path=factors_path
    )
    migrated_path = run.path / "migrated_calibration_rows.csv"
    _write_csv(migrated_path, migrated_rows)
    migrated_file_sha256 = _sha256(migrated_path)
    migrated_content_sha256 = calibration_rows_sha256(migrated_rows)
    profile = calibrate_detectability(
        migrated_rows,
        joint_names=full_joint_order,
        source_sha256=migrated_content_sha256,
        joint_order_source_sha256=factor_sha256,
    )
    detectability_payload = profile.to_dict()
    detectability_payload["migration"] = {
        "legacy_calibration_rows_sha256": _sha256(calibration_path),
        "migrated_calibration_rows_file_sha256": migrated_file_sha256,
        "migrated_calibration_rows_content_sha256": migrated_content_sha256,
        "joint_order_source_sha256": factor_sha256,
        "full_joint_order": list(full_joint_order),
    }
    _write_json(run.path / "detectability.json", detectability_payload)

    converted_urdf = run.path / "converted_piper.urdf"
    conversion = convert_urdf_for_mujoco(urdf_path, converted_urdf, package_root)
    bridge = MujocoStateBridge.from_urdf(converted_urdf, full_joint_order)
    fk = TorchURDFKinematics(urdf_path)
    if tuple(fk.active_joint_names) != full_joint_order:
        raise ValueError("URDF joint order does not match full_joint_order")

    results: list[dict[str, object]] = []
    frames: list[SynchronizedStateFrame] = []
    parity_cases: list[Mapping[str, Any]] = []
    parity_frames: list[SynchronizedStateFrame] = []
    bridge_parity_exclusions: list[dict[str, object]] = []
    controlled_components: list[tuple[bool, bool]] = []
    committed_components: list[bool] = []
    zero_case_updates: list[bool] = []
    for index, case in enumerate(cases):
        trial_id = str(case["trial_id"])
        offsets = dict(case["offsets_rad"])
        selected_joints = tuple(name for name in full_joint_order if name in offsets)
        correction = torch.tensor(
            candidate_corrections[trial_id], dtype=torch.float32
        )
        measured = measured_by_trial[trial_id]
        candidate = measured + correction
        frame = commit_synchronized_state(
            profile=profile,
            guard_decision=guard_decisions[trial_id],
            measured_qpos=measured,
            candidate_qpos=candidate,
            selected_joints=selected_joints,
            joint_limits=bridge.joint_limits,
            timestamp_ns=index,
            state_id=str(case["state_id"]),
            case_id=trial_id,
            factor_fingerprint=factor_sha256,
            guard_fingerprint=guard_sha256,
        )
        frame.validate_integrity()
        frames.append(frame)
        bridge_exclusion = _bridge_exclusion(
            frame=frame, joint_limits=bridge.joint_limits
        )
        if bridge_exclusion is None:
            bridge.write(frame)
            parity_cases.append(case)
            parity_frames.append(frame)
        else:
            bridge_parity_exclusions.append(bridge_exclusion)
        outcomes = _component_outcomes(
            offsets=offsets,
            candidate_correction=correction,
            joint_names=full_joint_order,
            recorded_recovery_success=candidate_recovery_success[trial_id],
        )
        decisions = {item.joint_name: item for item in frame.decisions}
        for name, offset in offsets.items():
            accepted = decisions[name].accepted
            committed_components.append(accepted)
            if abs(offset) > _ZERO_TOLERANCE:
                controlled_components.append((accepted, bool(outcomes[name]["candidate_success"])))
        if all(abs(value) <= _ZERO_TOLERANCE for value in offsets.values()):
            zero_case_updates.append(any(item.accepted for item in frame.decisions))
        results.append(
            {
                "trial_id": trial_id,
                "state_id": case["state_id"],
                "audit_designation": _DEVELOPMENT_AUDIT,
                "formal_evidence": False,
                "case_role": case["case_role"],
                "rt3_guard_accepted": guard_decisions[trial_id].accepted,
                "rt3_guard_reason": guard_decisions[trial_id].reason,
                "candidate_method": candidate_method,
                "candidate_recovery_success": candidate_recovery_success[trial_id],
                "candidate_committed": any(item.accepted for item in frame.decisions),
                "bridge_eligible": bridge_exclusion is None,
                "bridge_exclusion": json.dumps(
                    bridge_exclusion, sort_keys=True
                ),
                "measured_qpos": json.dumps(frame.measured_qpos.cpu().tolist()),
                "candidate_qpos": json.dumps(frame.candidate_qpos.cpu().tolist()),
                "synchronized_qpos": json.dumps(
                    frame.synchronized_qpos.cpu().tolist()
                ),
                "component_decisions": json.dumps(
                    [asdict(item) for item in frame.decisions], sort_keys=True
                ),
                "component_outcomes": json.dumps(outcomes, sort_keys=True),
                "frame_fingerprint": frame.fingerprint,
            }
        )

    if not controlled_components or not zero_case_updates:
        raise ValueError("RT3 development audit requires controlled and zero cases")
    committed_with_success = [
        success for accepted, success in controlled_components if accepted
    ]
    candidate_successes = [
        success for _, success in controlled_components if success
    ]
    retained_successes = [
        accepted and success for accepted, success in controlled_components if success
    ]
    metrics = {
        "audit_designation": _DEVELOPMENT_AUDIT,
        "formal_evidence": False,
        "calibration_state_ids": list(profile.calibration_state_ids),
        "audit_state_ids": sorted(audit_states),
        "calibration_audit_overlap": [],
        "profile_fingerprint": profile.fingerprint,
        "controlled_component_count": len(controlled_components),
        "committed_component_count": sum(committed_components),
        "committed_precision": (
            sum(committed_with_success) / len(committed_with_success)
            if committed_with_success
            else None
        ),
        "controlled_commit_coverage": sum(
            accepted for accepted, _ in controlled_components
        )
        / len(controlled_components),
        "successful_candidate_retention": (
            sum(retained_successes) / len(candidate_successes)
            if candidate_successes
            else None
        ),
        "zero_false_update_rate": sum(zero_case_updates) / len(zero_case_updates),
        "zero_stability": 1.0 - sum(zero_case_updates) / len(zero_case_updates),
        "bridge_parity_exclusions": bridge_parity_exclusions,
        "source_hashes": {
            "legacy_calibration_rows": _sha256(calibration_path),
            "migrated_calibration_rows": migrated_content_sha256,
            "migrated_calibration_rows_file": migrated_file_sha256,
            "frozen_factor": factor_sha256,
            "frozen_guard": guard_sha256,
            "rt3_development_results": _sha256(audit_results_path),
            "rt3_development_trace": _sha256(audit_trace_path),
            "rt3_validation_matrix": _sha256(audit_matrix_path),
            **{
                f"real_observation_records_{camera}": _sha256(dataset.records[camera])
                for camera in dataset.cameras
            },
        },
    }
    parity = compare_fk_parity(
        bridge,
        fk,
        torch.stack(
            [frame.synchronized_qpos for frame in parity_frames]
        ).numpy(),
    )
    parity_payload = {
        **asdict(parity),
        "audit_designation": _DEVELOPMENT_AUDIT,
        "formal_evidence": False,
        "conversion": conversion.to_dict(),
        "exclusions": bridge_parity_exclusions,
        "state_fingerprints": [frame.fingerprint for frame in parity_frames],
        "states": [
            {
                "trial_id": case["trial_id"],
                "state_id": case["state_id"],
                "measured_qpos": frame.measured_qpos.cpu().tolist(),
                "candidate_qpos": frame.candidate_qpos.cpu().tolist(),
                "synchronized_qpos": frame.synchronized_qpos.cpu().tolist(),
            }
            for case, frame in zip(parity_cases, parity_frames, strict=True)
        ],
    }
    metrics["parity_report"] = asdict(parity)
    _write_csv(run.path / "results.csv", results)
    _write_json(run.path / "metrics.json", metrics)
    _write_json(run.path / "parity_report.json", parity_payload)

    manifest_path = run.path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["audit_designation"] = _DEVELOPMENT_AUDIT
    manifest["formal_evidence"] = False
    manifest["bridge_parity_exclusions"] = bridge_parity_exclusions
    manifest["source_hashes"] = metrics["source_hashes"]
    manifest["derived_artifacts"] = {
        "migrated_calibration_rows": {
            "content_sha256": migrated_content_sha256,
            "sha256": migrated_file_sha256,
        },
        "converted_piper_urdf": {"sha256": _sha256(converted_urdf)},
        "conversion_manifest": {"sha256": _sha256(conversion.path)},
        "detectability": {"sha256": _sha256(run.path / "detectability.json")},
        "metrics": {"sha256": _sha256(run.path / "metrics.json")},
        "parity_report": {"sha256": _sha256(run.path / "parity_report.json")},
        "results": {"sha256": _sha256(run.path / "results.csv")},
    }
    _write_json(manifest_path, manifest)
    return run.path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id")
    arguments = parser.parse_args()
    print(execute_rt6_sync_bridge(load_config(arguments.config), run_id=arguments.run_id))


if __name__ == "__main__":
    main()
