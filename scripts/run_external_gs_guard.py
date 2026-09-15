#!/usr/bin/env python3
"""Run frozen two-view Franka same-GS guard development or heldout analysis."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any, Mapping

import cv2
import numpy as np
import torch
import yaml

from kinesync.correction.joint_offset import JointOffsetRecovery
from kinesync.external_abc import require_gpu_allocation
from kinesync.external_gs.franka import load_franka_external_gs
from kinesync.external_gs.guard import (
    ExternalRGBObservationBackend,
    apply_project_global_guard,
    build_development_detectability_profile,
)
from kinesync.guard.calibration import CalibrationRecord, GuardCalibration, calibrate_guard
from kinesync.guard.evidence import assemble_update_evidence
from kinesync.guard.schema import GuardThresholds
from kinesync.replay.rt7_policy import apply_component_verified_policy
from kinesync.sync.schema import DetectabilityProfile, JointDetectability


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _digest(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty JSONL: {path}")
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _persist_candidate_diagnostics(
    output: Path,
    *,
    candidate_rows: list[dict[str, object]],
    trace_rows: list[dict[str, object]],
) -> dict[str, object]:
    """Flush generated candidates before any development-only calibration."""
    _write_csv(output / "candidate_evidence.csv", candidate_rows)
    _write_csv(output / "optimizer_trace_qpos.csv", trace_rows)
    receipt = {
        "status": "candidate_evidence_persisted_before_calibration",
        "candidate_row_count": len(candidate_rows),
        "trace_row_count": len(trace_rows),
        "candidate_evidence_sha256": _sha256(output / "candidate_evidence.csv"),
        "optimizer_trace_sha256": _sha256(output / "optimizer_trace_qpos.csv"),
    }
    _write_json(output / "candidate_flush.json", receipt)
    return receipt


def _tensor_json(qpos: torch.Tensor) -> str:
    return json.dumps([float(value) for value in qpos.detach().cpu().tolist()])


def _save_rgb(path: Path, rgb: torch.Tensor) -> None:
    image = np.clip(np.round(rgb.detach().cpu().numpy() * 255.0), 0, 255).astype(np.uint8)
    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def _validate_config(config: Mapping[str, Any]) -> None:
    if config["experiment"]["target_provenance"] != "analysis_same_gs":
        raise ValueError("external GS guard runner requires analysis_same_gs targets")
    if len(config["conditions"]) != 4:
        raise ValueError("Stage 2 requires four predeclared observation conditions")
    expected = {"clean_nonzero", "clean_zero", "one_view_stale_nonzero", "one_view_stale_zero"}
    if {item["id"] for item in config["conditions"]} != expected:
        raise ValueError("unexpected Stage 2 condition set")
    if len(config["development_cases"]) != 6 or len(config["heldout_cases"]) != 12:
        raise ValueError("Stage 2 requires six development and twelve heldout states")


def _all_states(config: Mapping[str, Any]) -> np.ndarray:
    return np.asarray(
        [case["qpos_rad"] for key in ("development_cases", "heldout_cases") for case in config[key]],
        dtype=np.float64,
    )


def _frozen_state_ids(config: Mapping[str, Any], split: str) -> set[str]:
    cases = config[f"{split}_cases"]
    return {
        f"{case['id']}__{condition['id']}"
        for case in cases
        for condition in config["conditions"]
    }


def _validate_heldout_calibration_provenance(
    config: Mapping[str, Any],
    *,
    config_path: Path,
    development_root: Path,
    calibration: Any,
    profile: Any,
) -> None:
    """Reject a heldout run unless it uses this config's frozen development guard."""
    expected_development = _frozen_state_ids(config, "development")
    heldout_ids = _frozen_state_ids(config, "heldout")
    calibration_ids = set(calibration.source_state_ids)
    profile_ids = set(profile.calibration_state_ids)
    if calibration_ids != expected_development or profile_ids != expected_development:
        raise ValueError("heldout calibration must use exactly this development state IDs")
    if calibration_ids & heldout_ids or profile_ids & heldout_ids:
        raise ValueError("heldout calibration state IDs overlap heldout IDs")
    manifest_path = development_root / "manifest.json"
    source_config_path = development_root / "config_source.yaml"
    heldout_manifest_path = development_root / "heldout_manifest.json"
    if not all(path.is_file() for path in (manifest_path, source_config_path, heldout_manifest_path)):
        raise ValueError("heldout calibration provenance receipt is incomplete")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    heldout_manifest = json.loads(heldout_manifest_path.read_text(encoding="utf-8"))
    source_hash = _sha256(config_path)
    if (
        manifest.get("split") != "development"
        or manifest.get("target_provenance") != "analysis_same_gs"
        or manifest.get("config_source_sha256") != source_hash
        or _sha256(source_config_path) != source_hash
    ):
        raise ValueError("heldout calibration config source hash does not match")
    if (
        heldout_manifest.get("config_sha256") != source_hash
        or set(heldout_manifest.get("case_ids", ()))
        != {str(case["id"]) for case in config["heldout_cases"]}
        or set(heldout_manifest.get("condition_ids", ()))
        != {str(condition["id"]) for condition in config["conditions"]}
    ):
        raise ValueError("heldout manifest does not match frozen config")


def _condition_measurement(case: Mapping[str, Any], condition: Mapping[str, Any], *, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    true = torch.tensor(case["qpos_rad"], dtype=torch.float32, device=device)
    if condition["measured_offset"] == "inherited":
        offset = torch.tensor(case["injected_measurement_offset_rad"], dtype=torch.float32, device=device)
    elif condition["measured_offset"] == "zero":
        offset = torch.zeros_like(true)
    else:
        raise ValueError(f"unknown measurement condition: {condition['measured_offset']}")
    return true, true + offset


def _targets_for_condition(backend, true_q: torch.Tensor, condition: Mapping[str, Any], stale_delta: torch.Tensor):
    with torch.no_grad():
        targets = dict(backend.render(true_q))
        if condition["view_b"] == "stale":
            targets["view_b"] = backend.render(true_q + stale_delta)["view_b"]
        elif condition["view_b"] != "current":
            raise ValueError(f"unknown view_b mode: {condition['view_b']}")
    return targets


def _profile_from_json(payload: Mapping[str, Any]) -> DetectabilityProfile:
    return DetectabilityProfile(
        joint_names=tuple(payload["joint_names"]),
        joints=tuple(JointDetectability(**item) for item in payload["joints"]),
        source_sha256=payload["source_sha256"],
        joint_order_source_sha256=payload["joint_order_source_sha256"],
        source_split=payload["source_split"],
        calibration_state_ids=tuple(payload["calibration_state_ids"]),
        algorithm_version=payload["algorithm_version"],
        fingerprint=payload["fingerprint"],
    )


def _calibration_from_json(payload: Mapping[str, Any]) -> GuardCalibration:
    thresholds = GuardThresholds(**payload["thresholds"])
    return GuardCalibration(
        thresholds=thresholds,
        candidate_table=(),
        controlled_coverage=float(payload["controlled_coverage"]),
        zero_stability=float(payload["zero_stability"]),
        guarded_success=float(payload["guarded_success"]),
        source_state_ids=tuple(payload["source_state_ids"]),
        fingerprint=payload["fingerprint"],
        threshold_generation=str(payload["threshold_generation"]),
        evaluated_candidate_count=int(payload["evaluated_candidate_count"]),
    )


def _record_image_set(frames_dir: Path, *, state_id: str, method: str, backend, qpos: torch.Tensor) -> dict[str, str]:
    with torch.no_grad():
        buffers = backend.renderer.render_buffers(qpos)
    paths: dict[str, str] = {}
    for camera, buffer in buffers.items():
        path = frames_dir / f"{state_id}__{method}__{camera}.png"
        _save_rgb(path, buffer.rgb)
        paths[camera] = str(path.relative_to(frames_dir.parent))
    return paths


def _receipt(backend, qpos: torch.Tensor) -> Mapping[str, Any]:
    with torch.no_grad():
        buffers = backend.renderer.render_buffers(qpos)
    return {
        "qpos": [float(value) for value in qpos.detach().cpu().tolist()],
        "camera_render_sha256": {
            name: hashlib.sha256(buffer.rgb.detach().cpu().numpy().tobytes()).hexdigest()
            for name, buffer in buffers.items()
        },
    }


def _receipt_with_native_frames(
    backend,
    qpos: torch.Tensor,
    *,
    frames_dir: Path,
    state_id: str,
    receipt_label: str,
) -> Mapping[str, Any]:
    """Persist the exact RT7 receipt render and return its small receipt payload."""
    with torch.no_grad():
        buffers = backend.renderer.render_buffers(qpos)
    paths: dict[str, str] = {}
    hashes: dict[str, str] = {}
    for camera, buffer in buffers.items():
        path = frames_dir / f"{state_id}__{receipt_label}__{camera}.png"
        _save_rgb(path, buffer.rgb)
        paths[camera] = str(path.relative_to(frames_dir.parent))
        hashes[camera] = hashlib.sha256(buffer.rgb.detach().cpu().numpy().tobytes()).hexdigest()
    return {
        "qpos": [float(value) for value in qpos.detach().cpu().tolist()],
        "camera_render_sha256": hashes,
        "native_frame_paths": paths,
    }


def _evaluation_rows(
    *,
    split: str,
    state_id: str,
    case_id: str,
    condition_id: str,
    method: str,
    measured: torch.Tensor,
    candidate: torch.Tensor,
    final: torch.Tensor,
    true_q: torch.Tensor,
    global_guard_accepted: bool | None,
    component_guard_accepted: Mapping[str, bool] | None,
    latency_ms: float,
    frame_paths: Mapping[str, str],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    initial = (measured - true_q).abs()
    final_error = (final - true_q).abs()
    correction = final - measured
    tolerance = 1e-6
    joint_rows = []
    for index in range(measured.numel()):
        needed = bool(float(initial[index]) > tolerance)
        changed = bool(float(correction[index].abs()) > tolerance)
        harmful = bool(float(final_error[index]) > float(initial[index]) + tolerance)
        improved = bool(float(final_error[index]) + tolerance < float(initial[index]))
        joint_rows.append({
            "split": split,
            "state_id": state_id,
            "case_id": case_id,
            "condition_id": condition_id,
            "method": method,
            "joint_name": f"joint{index + 1}",
            "update_applied": changed,
            "global_guard_accepted": global_guard_accepted,
            "component_guard_accepted": (
                None
                if component_guard_accepted is None
                else bool(component_guard_accepted[f"joint{index + 1}"])
            ),
            "needed_correction": needed,
            "changed": changed,
            "improved": improved,
            "harmful_update": harmful,
            "initial_abs_error_rad": float(initial[index]),
            "final_abs_error_rad": float(final_error[index]),
            "applied_correction_rad": float(correction[index]),
        })
    row = {
        "split": split,
        "state_id": state_id,
        "case_id": case_id,
        "condition_id": condition_id,
        "method": method,
        "target_provenance": "analysis_same_gs",
        "evaluation_reference_only": True,
        "measured_qpos_rad": _tensor_json(measured),
        "candidate_qpos_rad": _tensor_json(candidate),
        "final_qpos_rad": _tensor_json(final),
        "joint_mae_deg": float(final_error.mean() * 180.0 / np.pi),
        "measured_joint_mae_deg": float(initial.mean() * 180.0 / np.pi),
        "harmful_update_count": sum(bool(item["harmful_update"]) for item in joint_rows),
        "update_applied_joint_count": sum(bool(item["update_applied"]) for item in joint_rows),
        "global_guard_accepted": global_guard_accepted,
        "component_guard_accepted_joint_count": sum(
            bool(item["component_guard_accepted"])
            for item in joint_rows
            if item["component_guard_accepted"] is not None
        ),
        "latency_ms": latency_ms,
        "native_frames": json.dumps(frame_paths, sort_keys=True),
    }
    return row, joint_rows


def _aggregate_rows(joint_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, str], list[dict[str, object]]] = {}
    for row in joint_rows:
        groups.setdefault((str(row["split"]), str(row["condition_id"]), str(row["method"])), []).append(row)
    result = []
    for (split, condition, method), rows in sorted(groups.items()):
        applied = [row for row in rows if bool(row["update_applied"])]
        needed = [row for row in rows if bool(row["needed_correction"])]
        global_decisions = [
            row for row in rows if row["global_guard_accepted"] is not None
        ]
        component_decisions = [
            row for row in rows if row["component_guard_accepted"] is not None
        ]
        result.append({
            "split": split,
            "condition_id": condition,
            "method": method,
            "joint_count": len(rows),
            "update_applied_count": len(applied),
            "update_applied_coverage": len(applied) / len(rows),
            "needed_count": len(needed),
            "needed_update_coverage": (sum(bool(row["update_applied"]) for row in needed) / len(needed)) if needed else 0.0,
            "update_applied_precision_nonharmful": (sum(not bool(row["harmful_update"]) for row in applied) / len(applied)) if applied else 1.0,
            "global_guard_decision_count": len(global_decisions),
            "global_guard_accept_count": sum(bool(row["global_guard_accepted"]) for row in global_decisions),
            "component_guard_decision_count": len(component_decisions),
            "component_guard_accept_count": sum(bool(row["component_guard_accepted"]) for row in component_decisions),
            "harmful_update_count": sum(bool(row["harmful_update"]) for row in rows),
        })
    return result


def _cpu_preflight(config: Mapping[str, Any], output: Path, *, split: str) -> None:
    _validate_config(config)
    asset = load_franka_external_gs(config["assets"]["root"])
    cameras = asset.framed_analysis_cameras(
        _all_states(config),
        camera_specs=config["cameras"]["specs"],
        width=int(config["render"]["preflight_width"]),
        height=int(config["render"]["preflight_height"]),
    )
    cases = config[f"{split}_cases"]
    rows = []
    for case in cases:
        state = torch.tensor(case["qpos_rad"], dtype=torch.float64)
        for name, camera in cameras.items():
            fraction = asset.point_visibility_fraction(state, camera)
            rows.append({"split": split, "case_id": case["id"], "camera": name, "point_visibility_fraction": fraction})
    minimum = float(config["cameras"]["minimum_point_visibility"])
    if any(float(row["point_visibility_fraction"]) < minimum for row in rows):
        raise RuntimeError("CPU camera visibility preflight failed")
    output.mkdir(parents=True, exist_ok=False)
    _write_csv(output / "cpu_camera_visibility.csv", rows)
    _write_json(output / "heldout_manifest.json", {
        "split": "heldout",
        "case_count": len(config["heldout_cases"]),
        "condition_ids": [item["id"] for item in config["conditions"]],
        "case_ids": [item["id"] for item in config["heldout_cases"]],
        "config_sha256": _sha256(Path(config["_config_path"])),
        "target_provenance": "analysis_same_gs",
        "frozen_before_heldout_execution": True,
    })
    print(json.dumps({"cpu_preflight": str(output), "rows": len(rows), "minimum_visibility": minimum}, sort_keys=True))


def run(config_path: Path, output: Path, *, split: str, preflight_only: bool = False, development_calibration: Path | None = None) -> None:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["_config_path"] = str(config_path)
    _validate_config(config)
    if split not in {"development", "heldout"}:
        raise ValueError("split must be development or heldout")
    calibration: GuardCalibration | None = None
    profile: DetectabilityProfile | None = None
    if split == "heldout":
        if development_calibration is None:
            raise ValueError("heldout execution requires --development-calibration")
        calibration = _calibration_from_json(
            json.loads(
                (development_calibration / "guard_calibration.json").read_text(
                    encoding="utf-8"
                )
            )
        )
        profile = _profile_from_json(
            json.loads(
                (development_calibration / "detectability_profile.json").read_text(
                    encoding="utf-8"
                )
            )
        )
        _validate_heldout_calibration_provenance(
            config,
            config_path=config_path,
            development_root=development_calibration,
            calibration=calibration,
            profile=profile,
        )
    job_id = require_gpu_allocation()
    if not torch.cuda.is_available():
        raise RuntimeError("native external GS guard rendering requires a Slurm CUDA allocation")
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(config_path, output / "config_source.yaml")
    (output / "config.yaml").write_text(yaml.safe_dump({key: value for key, value in config.items() if key != "_config_path"}, sort_keys=False), encoding="utf-8")
    (output / "script.sha256").write_text(_sha256(Path(__file__).resolve()) + "\n", encoding="ascii")
    _write_json(output / "heldout_manifest.json", {
        "split": "heldout", "case_count": len(config["heldout_cases"]),
        "condition_ids": [item["id"] for item in config["conditions"]],
        "case_ids": [item["id"] for item in config["heldout_cases"]],
        "config_sha256": _sha256(config_path), "target_provenance": "analysis_same_gs",
        "frozen_before_heldout_execution": True,
    })
    asset = load_franka_external_gs(config["assets"]["root"])
    render = config["render"]
    if preflight_only:
        height, width, per_link = int(render["preflight_height"]), int(render["preflight_width"]), int(render["preflight_gaussians_per_link"])
    else:
        height, width, per_link = int(render["height"]), int(render["width"]), int(render["gaussians_per_link"])
    cameras = asset.framed_analysis_cameras(_all_states(config), camera_specs=config["cameras"]["specs"], width=width, height=height)
    device = torch.device("cuda")
    renderer = asset.cuda_backend(output_size=(height, width), per_link=per_link, device=device, cameras=cameras)
    renderer.background = tuple(float(value) for value in render["background"])
    backend = ExternalRGBObservationBackend(renderer)
    cases = config[f"{split}_cases"]
    if preflight_only:
        rows = []
        preflight_dir = output / "native_frames"
        preflight_dir.mkdir()
        for case in cases[:1]:
            qpos = torch.tensor(case["qpos_rad"], dtype=torch.float32, device=device)
            with torch.no_grad():
                buffers = renderer.render_buffers(qpos)
            for name, buffer in buffers.items():
                path = preflight_dir / f"{case['id']}__{name}.png"
                _save_rgb(path, buffer.rgb)
                rows.append({"case_id": case["id"], "camera": name, "width": width, "height": height, "pixel_std": float(buffer.rgb.std()), "nonblank": bool(float(buffer.rgb.std()) > 1e-6), "path": str(path.relative_to(output))})
        if not all(bool(row["nonblank"]) for row in rows):
            raise RuntimeError("GPU small-frame preflight produced blank native image")
        _write_csv(output / "gpu_small_frame_preflight.csv", rows)
        _write_json(output / "manifest.json", {"schema": "kinesync.external_gs.guard.preflight.v1", "split": split, "slurm_job_id": job_id, "target_provenance": "analysis_same_gs", "config_sha256": _sha256(output / "config.yaml"), "script_sha256": _sha256(Path(__file__).resolve()), "camera_names": list(cameras)})
        print(json.dumps({"gpu_preflight": str(output), "frames": len(rows)}, sort_keys=True))
        return

    camera_backends = {}
    for name, camera in cameras.items():
        single = asset.cuda_backend(output_size=(height, width), per_link=per_link, device=device, cameras={name: camera})
        single.background = renderer.background
        camera_backends[name] = ExternalRGBObservationBackend(single)
    stale_delta = torch.tensor(config["stale_view_delta_rad"], dtype=torch.float32, device=device)
    frames_dir = output / "native_frames"
    frames_dir.mkdir()
    candidate_rows: list[dict[str, object]] = []
    trace_rows: list[dict[str, object]] = []
    runtimes: list[dict[str, Any]] = []
    for case_index, case in enumerate(cases):
        for condition_index, condition in enumerate(config["conditions"]):
            true_q, measured = _condition_measurement(case, condition, device=device)
            target = _targets_for_condition(backend, true_q, condition, stale_delta)
            started = time.perf_counter()
            recovery = JointOffsetRecovery(
                backend, joint_names=asset.joint_names, selected_joints=config["recovery"]["selected_joints"],
                steps=int(config["recovery"]["steps"]), learning_rate=float(config["recovery"]["learning_rate"]),
                offset_bound=float(config["recovery"]["offset_bound_rad"]), seed=int(config["experiment"]["seed"]) + case_index * 10 + condition_index,
            ).recover(measured, target)
            torch.cuda.synchronize()
            candidate_latency_ms = (time.perf_counter() - started) * 1000.0
            candidate = recovery.corrected_qpos.to(device)
            state_id = f"{case['id']}__{condition['id']}"
            camera_corrections: dict[str, torch.Tensor] = {}
            camera_recovery_latency_ms: dict[str, float] = {}
            for camera_name, camera_backend in camera_backends.items():
                started = time.perf_counter()
                single_view_recovery = JointOffsetRecovery(
                    camera_backend,
                    joint_names=asset.joint_names,
                    selected_joints=config["recovery"]["selected_joints"],
                    steps=int(config["recovery"]["steps"]),
                    learning_rate=float(config["recovery"]["learning_rate"]),
                    offset_bound=float(config["recovery"]["offset_bound_rad"]),
                    seed=(
                        int(config["experiment"]["seed"])
                        + case_index * 100
                        + condition_index * 10
                        + list(cameras).index(camera_name)
                        + 1
                    ),
                ).recover(measured, {camera_name: target[camera_name]})
                torch.cuda.synchronize()
                camera_recovery_latency_ms[camera_name] = (
                    time.perf_counter() - started
                ) * 1000.0
                camera_corrections[camera_name] = single_view_recovery.applied_correction.to(device)
                for trace in single_view_recovery.trace:
                    row = {
                        "split": split,
                        "state_id": state_id,
                        "case_id": case["id"],
                        "condition_id": condition["id"],
                        "target_provenance": "analysis_same_gs",
                        "optimizer_source": f"{camera_name}_evidence_recovery",
                        **trace,
                    }
                    for joint_index, name in enumerate(asset.joint_names):
                        row[f"qpos_{name}"] = float(
                            measured[joint_index]
                            + float(trace[f"correction_{name}"])
                        )
                    trace_rows.append(row)
            evidence = assemble_update_evidence(
                joint_backend=backend, joint_target=target, camera_backends=camera_backends,
                camera_targets={name: {name: target[name]} for name in cameras}, measured_qpos=measured,
                candidate_qpos=candidate, camera_corrections=camera_corrections,
                selected_indices=tuple(range(len(asset.joint_names))), candidate_bound=float(config["recovery"]["offset_bound_rad"]),
            )
            initial = (measured - true_q).abs()
            final_error = (candidate - true_q).abs()
            candidate_rows.append({
                "split": split, "state_id": state_id, "case_id": case["id"], "condition_id": condition["id"],
                "trial_type": condition["trial_type"], "target_provenance": "analysis_same_gs",
                "evaluation_reference_only": True, "measured_qpos_rad": _tensor_json(measured),
                "candidate_qpos_rad": _tensor_json(candidate), "true_qpos_rad": _tensor_json(true_q),
                "initial_abs_error_rad": json.dumps([float(value) for value in initial]),
                "candidate_abs_error_rad": json.dumps([float(value) for value in final_error]),
                "candidate_correction_rad": _tensor_json(candidate - measured), "candidate_latency_ms": candidate_latency_ms,
                "camera_corrections_rad": json.dumps({name: json.loads(_tensor_json(value)) for name, value in camera_corrections.items()}, sort_keys=True),
                "camera_recovery_latency_ms": json.dumps(camera_recovery_latency_ms, sort_keys=True),
                "camera_recovery_total_latency_ms": sum(camera_recovery_latency_ms.values()),
                **asdict(evidence),
            })
            for trace in recovery.trace:
                row = {"split": split, "state_id": state_id, "case_id": case["id"], "condition_id": condition["id"], "target_provenance": "analysis_same_gs", "optimizer_source": "fused_candidate", **trace}
                for index, name in enumerate(asset.joint_names):
                    row[f"qpos_{name}"] = float(measured[index] + float(trace[f"correction_{name}"]))
                trace_rows.append(row)
            runtimes.append({"case": case, "condition": condition, "state_id": state_id, "true_q": true_q, "measured": measured, "candidate": candidate, "target": target, "candidate_latency_ms": candidate_latency_ms, "camera_corrections": camera_corrections, "camera_recovery_latency_ms": camera_recovery_latency_ms, "evidence": evidence})

    candidate_flush = _persist_candidate_diagnostics(
        output, candidate_rows=candidate_rows, trace_rows=trace_rows
    )
    if split == "development":
        calibration_records = [
            CalibrationRecord(
                evidence=item["evidence"], trial_type=item["condition"]["trial_type"], state_id=item["state_id"],
                candidate_success=(
                    bool(float((item["candidate"] - item["true_q"]).abs().mean()) < float((item["measured"] - item["true_q"]).abs().mean()))
                    if item["condition"]["trial_type"] == "controlled"
                    else bool(float((item["candidate"] - item["true_q"]).abs().mean()) <= float((item["measured"] - item["true_q"]).abs().mean()) + float(config["guard"]["zero_tolerance_rad"]))
                ),
            )
            for item in runtimes
        ]
        try:
            calibration = calibrate_guard(calibration_records, config["guard"]["calibration_quantiles"], minimum_zero_stability=float(config["guard"]["minimum_zero_stability"]), minimum_controlled_coverage=float(config["guard"]["minimum_controlled_coverage"]))
            profile = build_development_detectability_profile(
                [
                    {
                        "split": row["split"],
                        "state_id": row["state_id"],
                        "initial_abs_error_rad": json.loads(row["initial_abs_error_rad"]),
                        "candidate_abs_error_rad": json.loads(row["candidate_abs_error_rad"]),
                        "candidate_correction_rad": json.loads(row["candidate_correction_rad"]),
                    }
                    for row in candidate_rows
                ],
                joint_names=asset.joint_names, minimum_state_reduction=float(config["guard"]["minimum_state_reduction"]), zero_tolerance_rad=float(config["guard"]["zero_tolerance_rad"]),
            )
        except Exception as error:
            _write_json(output / "calibration_failure.json", {
                "status": "development_calibration_failed_after_candidate_flush",
                "error_type": type(error).__name__,
                "error": str(error),
                "candidate_evidence_sha256": candidate_flush["candidate_evidence_sha256"],
                "optimizer_trace_sha256": candidate_flush["optimizer_trace_sha256"],
                "candidate_row_count": candidate_flush["candidate_row_count"],
                "trace_row_count": candidate_flush["trace_row_count"],
            })
            _write_json(output / "manifest.json", {
                "schema": "kinesync.external_gs.guard.analysis.v1",
                "status": "development_calibration_failed_after_candidate_flush",
                "split": split,
                "target_provenance": "analysis_same_gs",
                "slurm_job_id": job_id,
                "config_source_sha256": _sha256(config_path),
                "config_sha256": _sha256(output / "config.yaml"),
                "script_sha256": _sha256(Path(__file__).resolve()),
                "source_asset_sha256": asset.provenance.asset_sha256,
            })
            raise
        _write_json(output / "guard_calibration.json", {**asdict(calibration), "thresholds": asdict(calibration.thresholds), "candidate_table": [asdict(item) for item in calibration.candidate_table]})
        _write_json(output / "detectability_profile.json", profile.to_dict())
    if calibration is None or profile is None:
        raise RuntimeError("guard calibration/profile was not resolved")

    result_rows: list[dict[str, object]] = []
    joint_rows: list[dict[str, object]] = []
    component_rows: list[dict[str, object]] = []
    receipt_rows: list[dict[str, object]] = []
    for index, item in enumerate(runtimes):
        case, condition, state_id = item["case"], item["condition"], item["state_id"]
        measured, candidate, true_q, target = item["measured"], item["candidate"], item["true_q"], item["target"]
        shared_paths = _record_image_set(frames_dir, state_id=state_id, method="measurement_raw", backend=backend, qpos=measured)
        for method, qpos, latency_ms, paths in [
            ("measurement_raw", measured, 0.0, shared_paths),
            ("gaussian_inverse_unguarded", candidate, item["candidate_latency_ms"], _record_image_set(frames_dir, state_id=state_id, method="gaussian_inverse_unguarded", backend=backend, qpos=candidate)),
        ]:
            row, per_joint = _evaluation_rows(split=split, state_id=state_id, case_id=case["id"], condition_id=condition["id"], method=method, measured=measured, candidate=candidate, final=qpos, true_q=true_q, global_guard_accepted=None, component_guard_accepted=None, latency_ms=latency_ms, frame_paths=paths)
            result_rows.append(row); joint_rows.extend(per_joint)
        camera_targets = {name: {name: target[name]} for name in cameras}
        camera_corrections = item["camera_corrections"]
        camera_recovery_total_latency_ms = sum(
            item["camera_recovery_latency_ms"].values()
        )
        started = time.perf_counter()
        global_result = apply_project_global_guard(joint_backend=backend, joint_target=target, camera_backends=camera_backends, camera_targets=camera_targets, measured_qpos=measured, candidate_qpos=candidate, camera_corrections=camera_corrections, selected_indices=tuple(range(len(asset.joint_names))), candidate_bound=float(config["recovery"]["offset_bound_rad"]), thresholds=calibration.thresholds)
        torch.cuda.synchronize(); global_latency_ms = (time.perf_counter() - started) * 1000.0
        paths = _record_image_set(frames_dir, state_id=state_id, method="global_guard", backend=backend, qpos=global_result.committed_qpos)
        row, per_joint = _evaluation_rows(split=split, state_id=state_id, case_id=case["id"], condition_id=condition["id"], method="global_guard", measured=measured, candidate=candidate, final=global_result.committed_qpos, true_q=true_q, global_guard_accepted=global_result.decision.accepted, component_guard_accepted=None, latency_ms=item["candidate_latency_ms"] + camera_recovery_total_latency_ms + global_latency_ms, frame_paths=paths)
        row.update({"fused_candidate_latency_ms": item["candidate_latency_ms"], "single_view_evidence_recovery_latency_ms": camera_recovery_total_latency_ms, "guard_additional_latency_ms": global_latency_ms, "guard_reason": global_result.decision.reason, "guard_minimum_margin": global_result.decision.minimum_margin, **{f"global_{key}": value for key, value in asdict(global_result.evidence).items()}})
        result_rows.append(row); joint_rows.extend(per_joint)
        started = time.perf_counter()
        receipt_order = ["synchronized", *config["recovery"]["selected_joints"]]
        receipt_counter = 0

        def persistent_render_receipt(receipt_backend, receipt_qpos):
            nonlocal receipt_counter
            receipt_label = receipt_order[receipt_counter]
            receipt_counter += 1
            return _receipt_with_native_frames(
                receipt_backend,
                receipt_qpos,
                frames_dir=frames_dir,
                state_id=state_id,
                receipt_label=f"rt7_{receipt_label}_receipt",
            )

        application = apply_component_verified_policy(profile=profile, joint_backend=backend, joint_target=target, camera_backends=camera_backends, camera_targets=camera_targets, measured_qpos=measured, candidate_qpos=candidate, camera_corrections=camera_corrections, joint_names=asset.joint_names, selected_joints=config["recovery"]["selected_joints"], candidate_bound=float(config["recovery"]["offset_bound_rad"]), thresholds=calibration.thresholds, joint_limits={name: tuple(config["joint_limits_rad"][name]) for name in asset.joint_names}, timestamp_ns=1_726_000_000_000_000_000 + index, state_id=state_id, case_id=case["id"], factor_fingerprint=_digest({"split": split, "state_id": state_id, "candidate": _tensor_json(candidate)}), guard_fingerprint=calibration.fingerprint, render_receipt=persistent_render_receipt)
        if receipt_counter != len(receipt_order):
            raise RuntimeError("RT7 receipt count does not match synchronized/component renders")
        torch.cuda.synchronize(); component_latency_ms = (time.perf_counter() - started) * 1000.0
        paths = dict(application.synchronized_render_receipt["native_frame_paths"])
        decisions = {decision.joint_name: decision for decision in application.frame.decisions}
        row, per_joint = _evaluation_rows(split=split, state_id=state_id, case_id=case["id"], condition_id=condition["id"], method="component_guard", measured=measured, candidate=candidate, final=application.frame.synchronized_qpos, true_q=true_q, global_guard_accepted=None, component_guard_accepted={name: decision.accepted for name, decision in decisions.items()}, latency_ms=item["candidate_latency_ms"] + camera_recovery_total_latency_ms + component_latency_ms, frame_paths=paths)
        row.update({"fused_candidate_latency_ms": item["candidate_latency_ms"], "single_view_evidence_recovery_latency_ms": camera_recovery_total_latency_ms, "guard_additional_latency_ms": component_latency_ms, "latency_definition": "instrumented_wall_clock_including_rt7_receipt_render_and_png_io", "guard_reason": application.frame.guard_reason, "guard_fingerprint": calibration.fingerprint, "profile_fingerprint": profile.fingerprint, "accepted_joint_names": json.dumps(application.frame.accepted_joint_names)})
        result_rows.append(row)
        for item_row in per_joint:
            decision = decisions[item_row["joint_name"]]
            item_row.update({"decision_reason": decision.reason, "signal_floor_rad": decision.signal_floor_rad, "decision_margin_rad": decision.margin_rad})
        joint_rows.extend(per_joint)
        for verification in application.verifications.verifications:
            component_rows.append({"split": split, "state_id": state_id, "case_id": case["id"], "condition_id": condition["id"], "joint_name": verification.joint_name, "candidate_qpos_rad": _tensor_json(verification.candidate_qpos), "accepted": verification.decision.accepted, "decision_reason": verification.decision.reason, "decision_minimum_margin": verification.decision.minimum_margin, "verification_fingerprint": verification.fingerprint, **asdict(verification.evidence), "before_metrics": json.dumps(dict(verification.before_metrics), sort_keys=True), "after_metrics": json.dumps(dict(verification.after_metrics), sort_keys=True), "per_view_visual_gain": json.dumps(dict(verification.per_view_visual_gain), sort_keys=True)})
        receipt_rows.append({
            "split": split,
            "state_id": state_id,
            "case_id": case["id"],
            "condition_id": condition["id"],
            "receipt_kind": "synchronized",
            "joint_name": None,
            "frame_fingerprint": application.frame.fingerprint,
            "verification_fingerprint": None,
            **dict(application.synchronized_render_receipt),
        })
        for verification in application.verifications.verifications:
            receipt_rows.append({
                "split": split,
                "state_id": state_id,
                "case_id": case["id"],
                "condition_id": condition["id"],
                "receipt_kind": "component",
                "joint_name": verification.joint_name,
                "frame_fingerprint": application.frame.fingerprint,
                "verification_fingerprint": verification.fingerprint,
                **dict(application.component_render_receipts[verification.joint_name]),
            })

    _write_csv(output / "candidate_evidence.csv", candidate_rows)
    _write_csv(output / "optimizer_trace_qpos.csv", trace_rows)
    _write_csv(output / "results.csv", result_rows)
    _write_csv(output / "per_joint_decisions.csv", joint_rows)
    _write_csv(output / "component_evidence.csv", component_rows)
    _write_csv(output / "guard_summary.csv", _aggregate_rows(joint_rows))
    _write_jsonl(output / "rt7_render_receipts.jsonl", receipt_rows)
    _write_json(output / "manifest.json", {"schema": "kinesync.external_gs.guard.analysis.v1", "split": split, "target_provenance": "analysis_same_gs", "claim_boundary": config["experiment"]["claim_boundary"], "methods": ["measurement_raw", "gaussian_inverse_unguarded", "global_guard", "component_guard"], "candidate_shared_across_guard_methods": True, "independent_single_view_corrections_cached_for_guards": True, "guard_latency_accounting": "global/component latency includes fused candidate recovery, both single-view evidence recoveries, and guard application", "component_guard_latency_definition": "instrumented_wall_clock_including_rt7_receipt_render_and_png_io; not raw online compute latency", "reference_qpos_use": "evaluation and development calibration only; never candidate generation or deployed guard evidence", "camera_provenance": "two fixed framed analysis cameras built from predeclared states; not real calibration", "slurm_job_id": job_id, "config_source_sha256": _sha256(config_path), "config_sha256": _sha256(output / "config.yaml"), "script_sha256": _sha256(Path(__file__).resolve()), "source_asset_sha256": asset.provenance.asset_sha256, "gaussian_count_loaded": int(asset.local_xyz.shape[0]), "gaussians_per_link": per_link, "camera_names": list(cameras), "calibration_fingerprint": calibration.fingerprint, "detectability_profile_fingerprint": profile.fingerprint, "case_count": len(cases), "condition_count": len(config["conditions"])})
    print(json.dumps({"run": str(output), "split": split, "cases": len(cases), "conditions": len(config["conditions"]), "methods": 4}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("development", "heldout"), required=True)
    parser.add_argument("--cpu-preflight", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--development-calibration", type=Path)
    args = parser.parse_args()
    if args.cpu_preflight and args.preflight_only:
        raise ValueError("choose only one preflight mode")
    config_path, output = args.config.resolve(), args.output.resolve()
    if args.cpu_preflight:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        config["_config_path"] = str(config_path)
        _cpu_preflight(config, output, split=args.split)
    else:
        run(config_path, output, split=args.split, preflight_only=args.preflight_only, development_calibration=(None if args.development_calibration is None else args.development_calibration.resolve()))


if __name__ == "__main__":
    main()
