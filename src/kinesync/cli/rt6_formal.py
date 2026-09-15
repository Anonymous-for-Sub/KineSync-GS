"""Run the one-shot RT6-DM formal evaluation on untouched Route-A states."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
from statistics import mean
import subprocess
from typing import Any, Iterable, Mapping
import xml.etree.ElementTree as ET

import torch
import numpy as np
import cv2

from kinesync.cli.rt1_real_observation import _anchor_indices, _real_dataset
from kinesync.cli.rt2_evaluate import _load_factors, _validate_factor_bounds
from kinesync.cli.rt3_calibrate import _prior_backend
from kinesync.cli.rt3_evaluate import _load_guard, _outcome, _validate_optimizer
from kinesync.data.route_a import RouteAAssets, RouteAPaths
from kinesync.experiments.rt6_matrix import load_rt6_matrix
from kinesync.factorization import FrozenCameraObservationBackend
from kinesync.geometry.urdf import TorchURDFKinematics
from kinesync.guard.candidate import run_guard_candidate
from kinesync.guard.decision import decide_update
from kinesync.guard.schema import GuardThresholds
from kinesync.observation.cuda_gaussian import CudaGaussianBackend
from kinesync.observation.real_route_a import prepare_real_targets
from kinesync.runs.artifacts import RunArtifacts
from kinesync.sync.calibration import ALGORITHM_VERSION
from kinesync.sync.commit import commit_synchronized_state
from kinesync.sync.mujoco_asset import convert_urdf_for_mujoco
from kinesync.sync.mujoco_asset import _resolved_mesh_path
from kinesync.sync.mujoco_bridge import MujocoStateBridge, compare_fk_parity
from kinesync.sync.schema import DetectabilityProfile, JointDetectability
from kinesync.visualization.synchronization import (
    write_synchronization_comparison,
    write_synchronization_video,
)


_METHODS = ("rt2f_unguarded", "rt3c_guarded", "rt6dm_synchronized")
_REQUIRED_DERIVED_ARTIFACTS = {
    "component_results",
    "conversion_manifest",
    "converted_urdf",
    "metrics",
    "optimizer_executions",
    "parity_report",
    "results",
    "state_frames",
    "synchronization_image",
    "synchronization_video",
    "trace",
}
_REQUIRED_SOURCE_HASHES = {
    "anchor_sha256",
    "conversion_manifest",
    "converted_urdf",
    "extra_camera_sha256",
    "factors_sha256",
    "formal_config",
    "guard_sha256",
    "head_camera_sha256",
    "matrix_file_sha256",
    "matrix_semantic_fingerprint",
    "metadata_sha256",
    "observation_bundle_sha256",
    "profile_fingerprint",
    "profile_sha256",
    "records_extra_sha256",
    "records_head_sha256",
    "trajectory_h5_sha256",
    "urdf_mesh_bundle_sha256",
    "urdf_sha256",
}


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _git_provenance(repo: str | Path) -> dict[str, Any]:
    root = Path(repo).expanduser().resolve()
    commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=normal"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if status:
        raise ValueError("RT6 formal execution requires a clean tracked worktree")
    return {"commit_sha": commit, "tracked_worktree_dirty": False}


def _named_file_bundle(
    paths: Mapping[str, str | Path],
) -> tuple[str, dict[str, dict[str, Any]]]:
    records: dict[str, dict[str, Any]] = {}
    for name, raw_path in sorted(paths.items()):
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Frozen formal asset does not exist: {name}={path}")
        records[str(name)] = {
            "path": str(path),
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
        }
    digest = hashlib.sha256(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return digest, records


def _selected_observation_assets(
    config: Mapping[str, Any], state_ids: set[str]
) -> dict[str, Path]:
    source = config["real_observations"]
    selected: dict[str, Path] = {}
    for camera in sorted(source["records"]):
        records_path = Path(source["records"][camera]).expanduser().resolve()
        image_root = Path(source["image_roots"][camera]).expanduser().resolve()
        mask_root = Path(source["mask_roots"][camera]).expanduser().resolve()
        records = {}
        for line in records_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                state_id = str(record["state_id"])
                if state_id in records:
                    raise ValueError(f"Duplicate frozen observation: {camera}/{state_id}")
                records[state_id] = record
        missing = state_ids - set(records)
        if missing:
            raise ValueError(
                f"Frozen observation records lack {camera} states: {sorted(missing)}"
            )
        for state_id in sorted(state_ids):
            record = records[state_id]
            if str(record["camera_id"]) != camera or str(record["split"]) != "validation":
                raise ValueError(f"Invalid frozen observation record: {camera}/{state_id}")
            image = (image_root / str(record["image_path"])).resolve()
            mask = (mask_root / f"{record['sample_id']}.png").resolve()
            if not image.is_relative_to(image_root) or not mask.is_relative_to(mask_root):
                raise ValueError(f"Frozen observation escapes declared root: {camera}/{state_id}")
            selected[f"observation/{camera}/{state_id}/rgb"] = image
            selected[f"observation/{camera}/{state_id}/mask"] = mask
    return selected


def _urdf_mesh_assets(urdf: Path, package_root: Path) -> dict[str, Path]:
    root = ET.parse(urdf).getroot()
    assets: dict[str, Path] = {}
    for index, mesh in enumerate(root.findall(".//mesh")):
        filename = mesh.attrib.get("filename")
        if filename is None:
            raise ValueError("URDF mesh lacks a filename")
        assets[f"urdf_mesh/{index:03d}/{filename}"] = _resolved_mesh_path(
            filename,
            source_directory=urdf.parent,
            package_root=package_root,
        )
    if not assets:
        raise ValueError("Formal PiPER URDF contains no mesh assets")
    return assets


def _formal_source_hashes(
    *,
    config: Mapping[str, Any],
    matrix,
    frozen: Mapping[str, Any],
    factor_sha256: str,
    guard_sha256: str,
    profile_sha256: str,
    profile_fingerprint: str,
) -> tuple[dict[str, str], dict[str, Path]]:
    paths = RouteAPaths.from_mapping(config["assets"])
    state_ids = {case.state_id for case in matrix.cases}
    observations = _selected_observation_assets(config, state_ids)
    meshes = _urdf_mesh_assets(paths.urdf, frozen["package_root"])
    observation_digest, _ = _named_file_bundle(observations)
    mesh_digest, _ = _named_file_bundle(meshes)
    records = config["real_observations"]["records"]
    hashes = {
        "anchor_sha256": _sha256(paths.anchor),
        "extra_camera_sha256": _sha256(paths.extra_camera),
        "factors_sha256": factor_sha256,
        "guard_sha256": guard_sha256,
        "head_camera_sha256": _sha256(paths.head_camera),
        "matrix_file_sha256": matrix.source_file_sha256,
        "matrix_semantic_fingerprint": matrix.fingerprint,
        "metadata_sha256": _sha256(paths.metadata),
        "observation_bundle_sha256": observation_digest,
        "profile_fingerprint": profile_fingerprint,
        "profile_sha256": profile_sha256,
        "records_extra_sha256": _sha256(records["extra"]),
        "records_head_sha256": _sha256(records["head"]),
        "trajectory_h5_sha256": _sha256(paths.trajectory_h5),
        "urdf_mesh_bundle_sha256": mesh_digest,
        "urdf_sha256": _sha256(paths.urdf),
    }
    return hashes, {**observations, **meshes}


def _source_frame_provenance(
    observations: Mapping[str, Any], *, max_timestamp_delta_ns: int
) -> tuple[int, list[dict[str, Any]]]:
    if not observations or isinstance(max_timestamp_delta_ns, bool) or max_timestamp_delta_ns < 0:
        raise ValueError("source observations and timestamp tolerance are required")
    records = []
    for camera, observation in sorted(observations.items()):
        timestamp_ns = observation.timestamp_ns
        if isinstance(timestamp_ns, bool) or not isinstance(timestamp_ns, int):
            raise ValueError("source observation timestamps must be integers")
        qpos = np.asarray(observation.qpos, dtype=np.float64)
        if qpos.shape != (8,) or not np.isfinite(qpos).all():
            raise ValueError("source observation qpos must be finite PiPER states")
        if str(observation.camera) != str(camera):
            raise ValueError("source observation camera key mismatch")
        records.append(
            {
                "camera": str(camera),
                "frame_index": int(observation.frame_index),
                "qpos": qpos.tolist(),
                "sample_id": str(observation.sample_id),
                "timestamp_ns": timestamp_ns,
            }
        )
    timestamps = [row["timestamp_ns"] for row in records]
    span = max(timestamps) - min(timestamps)
    if span > max_timestamp_delta_ns:
        raise ValueError(
            f"source observation timestamp span {span} exceeds {max_timestamp_delta_ns}"
        )
    return sum(timestamps) // len(timestamps), records


def _optimizer_execution_records(case_id: str, candidate: Any) -> list[dict[str, Any]]:
    completed = [("rt2_joint", candidate.joint_result)] + [
        (f"rt2_{camera}", result)
        for camera, result in sorted(candidate.camera_results.items())
    ]
    return [
        {
            "candidate_source": source,
            "case_id": str(case_id),
            "completed": result is not None,
            "trace_step_count": len(getattr(result, "trace", ())),
        }
        for source, result in completed
    ]


def _gaussian_render_receipt(backend: Any, qpos: torch.Tensor) -> dict[str, Any]:
    values = qpos.detach().cpu().float().contiguous()
    with torch.no_grad():
        rendered = backend.render(qpos)
    camera_hashes: dict[str, str] = {}
    camera_shapes: dict[str, list[int]] = {}
    for camera, output in sorted(rendered.items()):
        alpha = output.values
        if isinstance(alpha, torch.Tensor):
            array = alpha.detach().cpu().float().contiguous().numpy()
        else:
            array = np.asarray(alpha, dtype=np.float32)
        if array.ndim != 2 or not np.isfinite(array).all():
            raise ValueError("Gaussian audit render must contain finite alpha images")
        camera_hashes[str(camera)] = hashlib.sha256(
            np.ascontiguousarray(array).tobytes()
        ).hexdigest()
        camera_shapes[str(camera)] = list(array.shape)
    if not camera_hashes:
        raise ValueError("Gaussian audit render returned no cameras")
    return {
        "qpos": [float(value) for value in values.tolist()],
        "camera_render_sha256": camera_hashes,
        "camera_shapes": camera_shapes,
    }


def _profile_payload(profile: DetectabilityProfile) -> dict[str, Any]:
    return {
        "algorithm_version": profile.algorithm_version,
        "calibration_state_ids": profile.calibration_state_ids,
        "joint_names": profile.joint_names,
        "joint_order_source_sha256": profile.joint_order_source_sha256,
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
            for item in profile.joints
        ],
        "source_sha256": profile.source_sha256,
        "source_split": profile.source_split,
    }


def _load_detectability_profile(
    path: str | Path, *, factor_sha256: str
) -> tuple[Path, DetectabilityProfile]:
    profile_path = Path(path).expanduser().resolve()
    payload = json.loads(profile_path.read_text(encoding="utf-8"))
    if payload.get("algorithm_version") != ALGORITHM_VERSION:
        raise ValueError("RT6 formal evaluation requires the frozen v2 profile")
    if payload.get("source_split") != "train":
        raise ValueError("RT6 detectability profile must originate from train")
    if payload.get("joint_order_source_sha256") != factor_sha256:
        raise ValueError("RT6 profile factor hash does not match evaluation factors")
    try:
        profile = DetectabilityProfile(
            joint_names=tuple(str(value) for value in payload["joint_names"]),
            joints=tuple(JointDetectability(**dict(value)) for value in payload["joints"]),
            source_sha256=str(payload["source_sha256"]),
            joint_order_source_sha256=str(payload["joint_order_source_sha256"]),
            source_split=str(payload["source_split"]),
            calibration_state_ids=tuple(
                str(value) for value in payload["calibration_state_ids"]
            ),
            algorithm_version=str(payload["algorithm_version"]),
            fingerprint=str(payload["fingerprint"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Malformed RT6 detectability profile") from error
    expected = hashlib.sha256(
        json.dumps(
            _profile_payload(profile), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    if profile.fingerprint != expected:
        raise ValueError("RT6 detectability profile fingerprint mismatch")
    return profile_path, profile


def _candidate_fingerprint(
    *,
    case_id: str,
    state_id: str,
    candidate_qpos: torch.Tensor,
    factor_sha256: str,
    guard_sha256: str,
) -> str:
    values = [float(value) for value in candidate_qpos.detach().cpu().double().tolist()]
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("candidate_qpos must be a finite nonempty vector")
    payload = {
        "candidate_qpos": values,
        "case_id": str(case_id),
        "factor_sha256": factor_sha256,
        "guard_sha256": guard_sha256,
        "state_id": str(state_id),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _rate(values: Iterable[bool]) -> float | None:
    materialized = [bool(value) for value in values]
    return mean(materialized) if materialized else None


def _component_group_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    committed = [row for row in rows if bool(row["accepted"])]
    successful = [row for row in rows if bool(row["candidate_success"])]
    return {
        "count": len(rows),
        "candidate_success_rate": _rate(row["candidate_success"] for row in rows),
        "commit_coverage": _rate(row["accepted"] for row in rows),
        "committed_precision": _rate(row["candidate_success"] for row in committed),
        "successful_candidate_retention": _rate(
            row["accepted"] for row in successful
        ),
    }


def _component_subgroups(rows: list[dict[str, Any]]) -> dict[str, Any]:
    controlled = [row for row in rows if row.get("role") == "single_joint"]
    groups: dict[str, dict[str, list[dict[str, Any]]]] = {
        "by_joint": defaultdict(list),
        "by_magnitude_deg": defaultdict(list),
        "by_sign": defaultdict(list),
    }
    for row in controlled:
        groups["by_joint"][str(row["joint_name"])].append(row)
        groups["by_magnitude_deg"][f"{float(row['magnitude_deg']):.1f}"].append(row)
        sign = "positive" if float(row["offset_rad"]) > 0 else "negative"
        groups["by_sign"][sign].append(row)
    return {
        family: {
            name: _component_group_summary(group)
            for name, group in sorted(values.items())
        }
        for family, values in groups.items()
    }


def _summarize_formal(
    *,
    method_rows: list[dict[str, Any]],
    component_rows: list[dict[str, Any]],
    parity_passed: bool,
) -> dict[str, Any]:
    method_summaries: dict[str, dict[str, Any]] = {}
    for method in _METHODS:
        rows = [row for row in method_rows if row.get("method") == method]
        if not rows:
            raise ValueError(f"formal results lack method rows: {method}")
        by_state: dict[str, list[bool]] = defaultdict(list)
        for row in rows:
            by_state[str(row["state_id"])].append(bool(row["recovery_success"]))
        method_summaries[method] = {
            "trial_count": len(rows),
            "unique_state_count": len(by_state),
            "trial_success_rate": _rate(row["recovery_success"] for row in rows),
            "state_clustered_success_rate": mean(
                mean(values) for values in by_state.values()
            ),
            "mean_mask_iou_gain": mean(float(row["mask_iou_gain"]) for row in rows),
            "mean_boundary_f1_gain": mean(
                float(row["boundary_f1_gain"]) for row in rows
            ),
        }

    controlled = [row for row in component_rows if row.get("role") == "single_joint"]
    zeros = [row for row in component_rows if row.get("role") == "zero_control"]
    if not controlled or not zeros:
        raise ValueError("formal component results require controlled and zero rows")
    committed = [row for row in controlled if bool(row["accepted"])]
    successful = [row for row in controlled if bool(row["candidate_success"])]
    primary = {
        "committed_precision": _rate(row["candidate_success"] for row in committed),
        "controlled_commit_coverage": _rate(row["accepted"] for row in controlled),
        "successful_candidate_retention": _rate(row["accepted"] for row in successful),
        "zero_false_update_rate": _rate(row["accepted"] for row in zeros),
    }
    primary["zero_stability"] = (
        None
        if primary["zero_false_update_rate"] is None
        else 1.0 - float(primary["zero_false_update_rate"])
    )
    rt6_rows = [
        row
        for row in method_rows
        if row.get("method") == "rt6dm_synchronized"
        and row.get("role") == "single_joint"
        and bool(row.get("any_component_accepted", True))
    ]
    finite_pass = all(
        bool(row.get("evidence_finite"))
        and all(
            math.isfinite(float(row[field]))
            for field in ("mask_iou_gain", "boundary_f1_gain")
        )
        for row in method_rows
    )
    positive_visual = bool(
        rt6_rows
        and mean(float(row["mask_iou_gain"]) for row in rt6_rows) > 0.0
        and mean(float(row["boundary_f1_gain"]) for row in rt6_rows) > 0.0
    )
    gates = {
        "committed_precision_at_least_85pct": bool(
            primary["committed_precision"] is not None
            and primary["committed_precision"] >= 0.85
        ),
        "controlled_commit_coverage_at_least_40pct": bool(
            primary["controlled_commit_coverage"] is not None
            and primary["controlled_commit_coverage"] >= 0.40
        ),
        "successful_candidate_retention_at_least_75pct": bool(
            primary["successful_candidate_retention"] is not None
            and primary["successful_candidate_retention"] >= 0.75
        ),
        "zero_stability_at_least_90pct": bool(
            primary["zero_stability"] is not None
            and primary["zero_stability"] >= 0.90
        ),
        "zero_false_update_at_most_10pct": bool(
            primary["zero_false_update_rate"] is not None
            and primary["zero_false_update_rate"] <= 0.10
        ),
        "positive_committed_visual_gains": positive_visual,
        "finite_evidence_and_fk_parity": bool(finite_pass and parity_passed),
    }
    return {
        "method_summaries": method_summaries,
        "primary_metrics": primary,
        "component_subgroups": _component_subgroups(component_rows),
        "advance_gates": gates,
        "all_advance_gates_passed": all(gates.values()),
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write empty formal result rows")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write empty formal state frames")
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
                + "\n"
            )


def _validate_formal_run_contract(path: Path) -> None:
    metrics = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
    if (
        metrics.get("trial_count") != 32
        or metrics.get("unique_state_count") != 16
        or metrics.get("method_row_count") != 96
        or metrics.get("component_row_count") != 34
        or metrics.get("candidate_procedure_runs") != 32
        or metrics.get("optimizer_execution_count") != 96
        or metrics.get("optimizer_executions_per_candidate") != 3
        or int(metrics.get("max_source_timestamp_delta_ns", -1)) <= 0
    ):
        raise ValueError("RT6 formal run count contract failed")
    with (path / "results.csv").open(encoding="utf-8", newline="") as handle:
        results = list(csv.DictReader(handle))
    by_case: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in results:
        by_case[row["case_id"]].append(row)
    if len(by_case) != 32:
        raise ValueError("RT6 formal results do not contain 32 matched cases")
    for rows in by_case.values():
        if {row["method"] for row in rows} != set(_METHODS) or len(rows) != 3:
            raise ValueError("RT6 formal case lacks matched method rows")
        if len({row["candidate_fingerprint"] for row in rows}) != 1:
            raise ValueError("RT6 compared methods did not reuse one candidate")
    optimizer_records = [
        json.loads(line)
        for line in (path / "optimizer_executions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    optimizers_by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in optimizer_records:
        optimizers_by_case[str(row["case_id"])].append(row)
    if len(optimizer_records) != 96 or set(optimizers_by_case) != set(by_case):
        raise ValueError("RT6 optimizer execution receipt contract failed")
    for rows in optimizers_by_case.values():
        if (
            len(rows) != 3
            or {row["candidate_source"] for row in rows}
            != {"rt2_joint", "rt2_head", "rt2_extra"}
            or not all(row["completed"] for row in rows)
            or not all(int(row.get("trace_step_count", 0)) > 0 for row in rows)
        ):
            raise ValueError("RT6 case lacks three completed optimizer receipts")
    state_frames = [
        json.loads(line)
        for line in (path / "state_frames.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(state_frames) != 32 or len({row["case_id"] for row in state_frames}) != 32:
        raise ValueError("RT6 formal state frame contract failed")
    for row in state_frames:
        if row["synchronized_qpos"] != row["mujoco_qpos"] or row[
            "synchronized_qpos"
        ] != row["gaussian_render_qpos"]:
            raise ValueError("RT6 synchronized qpos routing contract failed")
        source = row.get("source_observations", [])
        source_span = max(int(item["timestamp_ns"]) for item in source) - min(
            int(item["timestamp_ns"]) for item in source
        ) if source else -1
        if (
            {item.get("camera") for item in source} != {"head", "extra"}
            or len(source) != 2
            or row.get("source_fusion_policy") != "paired_qpos_mean_v1"
            or row.get("timestamp_ns")
            != sum(int(item["timestamp_ns"]) for item in source) // 2
            or row.get("source_timestamp_span_ns") != source_span
            or source_span > int(metrics["max_source_timestamp_delta_ns"])
            or any(
                not isinstance(item.get("frame_index"), int)
                or not item.get("sample_id")
                or len(item.get("qpos", [])) != 8
                or not all(math.isfinite(float(value)) for value in item["qpos"])
                for item in source
            )
        ):
            raise ValueError("RT6 source-frame provenance contract failed")
        receipt = row.get("gaussian_render_receipt", {})
        if (
            receipt.get("qpos") != row["synchronized_qpos"]
            or set(receipt.get("camera_render_sha256", {})) != {"head", "extra"}
            or set(receipt.get("camera_shapes", {})) != {"head", "extra"}
            or any(
                len(value) != 64
                for value in receipt.get("camera_render_sha256", {}).values()
            )
        ):
            raise ValueError("RT6 Gaussian render receipt contract failed")
    image = cv2.imread(
        str(path / "images" / "rt6_formal_synchronization.png"), cv2.IMREAD_COLOR
    )
    if image is None or image.shape != (1080, 1920, 3) or float(image.std()) <= 1.0:
        raise ValueError("RT6 formal image contract failed")
    capture = cv2.VideoCapture(str(path / "videos" / "rt6_formal_synchronization.mp4"))
    try:
        if (
            not capture.isOpened()
            or int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) != 1920
            or int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) != 1080
            or int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) <= 0
        ):
            raise ValueError("RT6 formal video contract failed")
    finally:
        capture.release()
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    git_provenance = metrics.get("git_provenance")
    source_hashes = metrics.get("source_hashes")
    if (
        not isinstance(git_provenance, Mapping)
        or len(str(git_provenance.get("commit_sha", ""))) != 40
        or git_provenance.get("tracked_worktree_dirty") is not False
        or manifest.get("git_provenance") != git_provenance
        or not isinstance(source_hashes, Mapping)
        or set(source_hashes) != _REQUIRED_SOURCE_HASHES
        or any(len(str(value)) != 64 for value in source_hashes.values())
        or manifest.get("source_hashes") != source_hashes
    ):
        raise ValueError("RT6 formal Git/source provenance contract failed")
    derived = manifest.get("derived_artifacts")
    if not isinstance(derived, Mapping) or set(derived) != _REQUIRED_DERIVED_ARTIFACTS:
        raise ValueError("RT6 formal manifest lacks required derived artifacts")
    for record in derived.values():
        artifact = Path(record["path"])
        if not artifact.is_file() or _sha256(artifact) != record["sha256"]:
            raise ValueError("RT6 formal derived artifact hash mismatch")


def _component_rows(
    *,
    case,
    frame,
    candidate_before: Mapping[str, float],
    candidate_after: Mapping[str, float],
    evidence_finite: bool,
) -> list[dict[str, Any]]:
    offsets = dict(case.offsets_rad)
    decisions = {item.joint_name: item for item in frame.decisions}
    trial_visual_success = bool(
        evidence_finite
        and float(candidate_after["mean_mask_iou"])
        > float(candidate_before["mean_mask_iou"])
        and float(candidate_after["mean_boundary_f1"])
        > float(candidate_before["mean_boundary_f1"])
    )
    rows: list[dict[str, Any]] = []
    for joint_name, offset in offsets.items():
        index = frame.joint_names.index(joint_name)
        correction = float(frame.candidate_qpos[index] - frame.measured_qpos[index])
        initial_error = abs(float(offset))
        if initial_error <= 1e-12:
            reduction = None
            physical_success = False
        else:
            reduction = 1.0 - abs(float(offset) + correction) / initial_error
            physical_success = reduction >= 0.60
        decision = decisions[joint_name]
        rows.append(
            {
                "case_id": case.case_id,
                "state_id": case.state_id,
                "role": case.role,
                "magnitude_deg": case.magnitude_deg,
                "joint_name": joint_name,
                "offset_rad": float(offset),
                "candidate_correction_rad": correction,
                "state_error_reduction": reduction,
                "physical_success": physical_success,
                "trial_visual_success": trial_visual_success,
                "candidate_success": bool(physical_success and trial_visual_success),
                "accepted": decision.accepted,
                "decision_reason": decision.reason,
                "signal_floor_rad": decision.signal_floor_rad,
                "margin_rad": decision.margin_rad,
                "frame_fingerprint": frame.fingerprint,
            }
        )
    return rows


def _formal_renderer_factory(*, fk, assets, indices, config, device):
    height, width = map(int, config["observation"]["image_size"])
    cache: dict[tuple[str, ...], CudaGaussianBackend] = {}

    def renderer_for(names: list[str]) -> CudaGaussianBackend:
        key = tuple(names)
        if key not in cache:
            cache[key] = CudaGaussianBackend.from_route_a(
                kinematics=fk,
                anchors=assets.load_anchors(),
                anchor_indices=indices,
                link_names=assets.link_names,
                calibrations={name: assets.load_camera(name) for name in names},
                output_size=(height, width),
                observation_mode="alpha",
                device=device,
            )
        return cache[key]

    return renderer_for


def _frozen_formal_inputs(config: Mapping[str, Any]) -> dict[str, Any]:
    formal = config.get("formal")
    if not isinstance(formal, Mapping) or formal.get("frozen") is not True:
        raise ValueError("RT6 formal execution requires formal.frozen=true")
    inputs = formal.get("inputs")
    hashes = formal.get("expected_hashes")
    if not isinstance(inputs, Mapping) or not isinstance(hashes, Mapping):
        raise ValueError("RT6 formal config requires inputs and expected_hashes")
    resolved: dict[str, Any] = {}
    for name in ("factors", "guard", "profile", "matrix"):
        path = Path(str(inputs.get(name, ""))).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"RT6 frozen input does not exist: {name}={path}")
        resolved[name] = path
    package_root = Path(str(formal.get("package_root", ""))).expanduser().resolve()
    if not package_root.is_dir():
        raise NotADirectoryError(f"PiPER package root does not exist: {package_root}")
    config_path = Path(str(config.get("_config_path", ""))).expanduser().resolve()
    if not config_path.is_file():
        raise ValueError("RT6 formal config must originate from a file")
    resolved["package_root"] = package_root
    resolved["config_path"] = config_path
    resolved["hashes"] = {str(name): str(value) for name, value in hashes.items()}
    resolved["expected_trial_count"] = int(formal.get("expected_trial_count", -1))
    resolved["expected_unique_state_count"] = int(
        formal.get("expected_unique_state_count", -1)
    )
    resolved["optimizer_executions_per_candidate"] = int(
        formal.get("expected_optimizer_executions_per_candidate", -1)
    )
    resolved["max_source_timestamp_delta_ns"] = int(
        formal.get("max_source_timestamp_delta_ns", -1)
    )
    if resolved["optimizer_executions_per_candidate"] != 3:
        raise ValueError("RT6 candidate procedure requires three optimizer executions")
    if resolved["max_source_timestamp_delta_ns"] <= 0:
        raise ValueError("RT6 formal source timestamp tolerance must be positive")
    return resolved


def execute_rt6_formal(
    config: Mapping[str, Any],
    *,
    run_id: str | None = None,
) -> Path:
    """Execute the frozen formal matrix with one shared candidate per case."""

    device = torch.device(str(config.get("device", "cuda")))
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("RT6 formal evaluation requires the native CUDA rasterizer")
    frozen = _frozen_formal_inputs(config)
    factor_path, factors = _load_factors(frozen["factors"])
    _validate_factor_bounds(factors)
    factor_sha = _sha256(factor_path)
    guard_path, guard = _load_guard(frozen["guard"], factor_sha256=factor_sha)
    guard_sha = _sha256(guard_path)
    evaluation = config["evaluation"]
    _validate_optimizer(guard, evaluation)
    thresholds = GuardThresholds(**guard["thresholds"])
    profile_path, profile = _load_detectability_profile(
        frozen["profile"], factor_sha256=factor_sha
    )
    profile_sha = _sha256(profile_path)
    matrix_path = frozen["matrix"]
    matrix = load_rt6_matrix(matrix_path)
    expected_hashes = frozen["hashes"]
    if (
        matrix.trial_count != frozen["expected_trial_count"]
        or matrix.unique_state_count != frozen["expected_unique_state_count"]
    ):
        raise ValueError("RT6 matrix counts do not match executable config")

    paths = RouteAPaths.from_mapping(config["assets"])
    actual_frozen, frozen_file_assets = _formal_source_hashes(
        config=config,
        matrix=matrix,
        frozen=frozen,
        factor_sha256=factor_sha,
        guard_sha256=guard_sha,
        profile_sha256=profile_sha,
        profile_fingerprint=profile.fingerprint,
    )
    if expected_hashes != actual_frozen:
        raise ValueError("RT6 frozen input hashes do not match executable config")
    git_provenance = _git_provenance(frozen["config_path"].parents[1])
    assets = RouteAAssets(paths)
    dataset = _real_dataset(config)
    case_states = {case.state_id for case in matrix.cases}
    available = set(dataset.state_ids(split="validation", cameras=["head", "extra"]))
    if not case_states.issubset(available):
        raise ValueError("RT6 matrix requests unavailable paired validation states")
    forbidden = (
        set(str(value) for value in factors.get("train_state_ids", ()))
        | set(str(value) for value in guard.get("source_state_ids", ()))
        | set(profile.calibration_state_ids)
    )
    if overlap := sorted(case_states & forbidden):
        raise ValueError(f"RT6 formal matrix overlaps calibration sources: {overlap}")

    fk = TorchURDFKinematics(paths.urdf)
    if tuple(fk.active_joint_names) != profile.joint_names:
        raise ValueError("RT6 profile joint order does not match the PiPER URDF")
    factor_joint_names = tuple(str(value) for value in factors["joint_names"])
    if factor_joint_names != profile.joint_names[: len(factor_joint_names)]:
        raise ValueError("RT6 frozen factor joint order does not match profile")
    shared_joint = torch.zeros(
        len(profile.joint_names), dtype=torch.float32, device=device
    )
    shared_joint[: len(factor_joint_names)] = torch.as_tensor(
        factors["joint_zero_rad"], dtype=torch.float32, device=device
    )
    camera_factors = {
        name: torch.as_tensor(value, dtype=torch.float32, device=device)
        for name, value in factors["camera_twists"].items()
    }
    height, width = map(int, config["observation"]["image_size"])
    boundary_weight = float(config["observation"]["boundary_weight"])
    indices = _anchor_indices(
        assets, config["data"]["anchors_per_link"], seed=int(config.get("seed", 41))
    )
    renderer_for = _formal_renderer_factory(
        fk=fk, assets=assets, indices=indices, config=config, device=device
    )

    package_root = frozen["package_root"]
    real_source = config["real_observations"]
    run = RunArtifacts.create(
        root=config.get("runs_root", "runs"),
        experiment="rt6_piper_formal",
        run_id=run_id,
        config=config,
        assets={
            **paths.as_assets(),
            "factors": factor_path,
            "guard": guard_path,
            "detectability_profile": profile_path,
            "formal_matrix": matrix_path,
            "formal_config": frozen["config_path"],
            **frozen_file_assets,
            **{
                f"records_{camera}": real_source["records"][camera]
                for camera in dataset.cameras
            },
        },
        target_provenance="real_observation_guard_evaluation",
        observation_backend="native_cuda_rt6_synchronized_formal",
    )
    converted_path = run.path / "converted_piper.urdf"
    conversion = convert_urdf_for_mujoco(paths.urdf, converted_path, package_root)
    bridge = MujocoStateBridge.from_urdf(converted_path, profile.joint_names)

    method_rows: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    optimizer_execution_rows: list[dict[str, Any]] = []
    frames = []
    state_frame_rows: list[dict[str, Any]] = []
    media_contexts: list[dict[str, Any]] = []
    candidate_fingerprints: dict[str, str] = {}
    for case in matrix.cases:
        cameras = list(case.cameras)
        observations = dataset.load_state(case.state_id, cameras=cameras)
        if any(value.split != "validation" for value in observations.values()):
            raise ValueError("RT6 formal evaluation loaded a non-validation observation")
        targets = prepare_real_targets(
            observations, output_size=(height, width), device=device
        )
        source_timestamp_ns, source_observations = _source_frame_provenance(
            observations,
            max_timestamp_delta_ns=int(
                frozen["max_source_timestamp_delta_ns"]
            ),
        )
        reference = torch.as_tensor(
            np.mean(np.stack([value.qpos for value in observations.values()]), axis=0),
            dtype=torch.float32,
            device=device,
        )
        calibrated_reference = reference + shared_joint
        offsets = dict(case.offsets_rad)
        selected_joints = [name for name in profile.joint_names if name in offsets]
        selected_indices = [profile.joint_names.index(name) for name in selected_joints]
        injected = torch.zeros_like(calibrated_reference)
        for name, value in offsets.items():
            injected[profile.joint_names.index(name)] = float(value)
        measured = calibrated_reference + injected

        renderer = renderer_for(cameras)
        joint_twists = {name: camera_factors[name] for name in cameras}
        visual = FrozenCameraObservationBackend(
            renderer, camera_twists=joint_twists, boundary_weight=boundary_weight
        )
        backend = _prior_backend(
            visual,
            renderer,
            targets,
            camera_twists=joint_twists,
            measured_qpos=measured,
            selected_indices=selected_indices,
            evaluation=evaluation,
            boundary_weight=boundary_weight,
        )
        camera_targets = {name: {name: targets[name]} for name in cameras}
        camera_visual = {}
        camera_backends = {}
        for camera in cameras:
            one_renderer = renderer_for([camera])
            twists = {camera: camera_factors[camera]}
            one_visual = FrozenCameraObservationBackend(
                one_renderer, camera_twists=twists, boundary_weight=boundary_weight
            )
            camera_visual[camera] = one_visual
            camera_backends[camera] = _prior_backend(
                one_visual,
                one_renderer,
                camera_targets[camera],
                camera_twists=twists,
                measured_qpos=measured,
                selected_indices=selected_indices,
                evaluation=evaluation,
                boundary_weight=boundary_weight,
            )
        candidate = run_guard_candidate(
            joint_backend=backend,
            joint_visual_backend=visual,
            joint_target=targets,
            camera_backends=camera_backends,
            camera_visual_backends=camera_visual,
            camera_targets=camera_targets,
            measured_qpos=measured,
            reference_qpos=calibrated_reference,
            joint_names=profile.joint_names,
            selected_joints=selected_joints,
            steps=int(evaluation["steps"]),
            learning_rate=float(evaluation["learning_rate"]),
            offset_bound=float(evaluation["offset_bound_rad"]),
            seed=case.seed,
            acceptance=config.get("acceptance", {}),
        )
        case_optimizer_records = _optimizer_execution_records(case.case_id, candidate)
        if len(case_optimizer_records) != frozen["optimizer_executions_per_candidate"]:
            raise ValueError("RT6 candidate returned an unexpected optimizer count")
        optimizer_execution_rows.extend(case_optimizer_records)
        candidate_qpos = candidate.joint_result.corrected_qpos
        candidate_fp = _candidate_fingerprint(
            case_id=case.case_id,
            state_id=case.state_id,
            candidate_qpos=candidate_qpos,
            factor_sha256=factor_sha,
            guard_sha256=guard_sha,
        )
        if candidate_fp in candidate_fingerprints.values():
            raise ValueError("RT6 candidates unexpectedly share a fingerprint")
        candidate_fingerprints[case.case_id] = candidate_fp
        guard_decision = decide_update(candidate.evidence, thresholds)
        frame = commit_synchronized_state(
            profile=profile,
            guard_decision=guard_decision,
            measured_qpos=measured,
            candidate_qpos=candidate_qpos,
            selected_joints=selected_joints,
            joint_limits=bridge.joint_limits,
            timestamp_ns=source_timestamp_ns,
            state_id=case.state_id,
            case_id=case.case_id,
            factor_fingerprint=factor_sha,
            guard_fingerprint=guard_sha,
        )
        bridge.write(frame)
        frames.append(frame)
        frame.validate_integrity()
        synchronized_values = [
            float(value) for value in frame.synchronized_qpos.detach().cpu().tolist()
        ]
        mujoco_values = [
            float(value)
            for value in bridge.data.qpos[list(bridge.qpos_addresses)].tolist()
        ]
        if mujoco_values != synchronized_values:
            raise ValueError("MuJoCo qpos differs from RT6 synchronized qpos")
        gaussian_receipt = _gaussian_render_receipt(
            visual, frame.synchronized_qpos
        )
        gaussian_values = gaussian_receipt["qpos"]
        if gaussian_values != synchronized_values:
            raise ValueError("Gaussian qpos differs from RT6 synchronized qpos")
        state_frame_rows.append(
            {
                "case_id": frame.case_id,
                "state_id": frame.state_id,
                "timestamp_ns": frame.timestamp_ns,
                "joint_names": list(frame.joint_names),
                "measured_qpos": [
                    float(value) for value in frame.measured_qpos.detach().cpu().tolist()
                ],
                "candidate_qpos": [
                    float(value) for value in frame.candidate_qpos.detach().cpu().tolist()
                ],
                "synchronized_qpos": synchronized_values,
                "mujoco_qpos": mujoco_values,
                "gaussian_render_qpos": gaussian_values,
                "gaussian_render_receipt": gaussian_receipt,
                "source_fusion_policy": "paired_qpos_mean_v1",
                "source_observations": source_observations,
                "source_timestamp_span_ns": max(
                    item["timestamp_ns"] for item in source_observations
                )
                - min(item["timestamp_ns"] for item in source_observations),
                "decisions": [asdict(value) for value in frame.decisions],
                "factor_fingerprint": frame.factor_fingerprint,
                "guard_fingerprint": frame.guard_fingerprint,
                "profile_fingerprint": frame.profile_fingerprint,
                "frame_fingerprint": frame.fingerprint,
            }
        )

        method_states = {
            "rt2f_unguarded": candidate_qpos,
            "rt3c_guarded": candidate_qpos if guard_decision.accepted else measured,
            "rt6dm_synchronized": frame.synchronized_qpos,
        }
        candidate_before = visual.metrics(measured, targets)
        candidate_after = visual.metrics(candidate_qpos, targets)
        component_rows.extend(
            _component_rows(
                case=case,
                frame=frame,
                candidate_before=candidate_before,
                candidate_after=candidate_after,
                evidence_finite=candidate.evidence.finite,
            )
        )
        for method, final_qpos in method_states.items():
            outcome, before, after = _outcome(
                backend=visual,
                target=targets,
                measured=measured,
                final=final_qpos,
                reference=calibrated_reference,
                selected_indices=selected_indices,
                acceptance=config.get("acceptance", {}),
            )
            method_rows.append(
                {
                    "case_id": case.case_id,
                    "state_id": case.state_id,
                    "split": case.split,
                    "role": case.role,
                    "magnitude_deg": case.magnitude_deg,
                    "method": method,
                    "candidate_fingerprint": candidate_fp,
                    "frame_fingerprint": frame.fingerprint,
                    "guard_accepted": guard_decision.accepted,
                    "guard_reason": guard_decision.reason,
                    "any_component_accepted": bool(frame.accepted_joint_names),
                    "state_error_reduction": outcome["state_error_reduction"],
                    "recovery_success": bool(outcome["recovery_success"]),
                    "mask_iou_before": before["mean_mask_iou"],
                    "mask_iou_after": after["mean_mask_iou"],
                    "mask_iou_gain": after["mean_mask_iou"] - before["mean_mask_iou"],
                    "boundary_f1_before": before["mean_boundary_f1"],
                    "boundary_f1_after": after["mean_boundary_f1"],
                    "boundary_f1_gain": (
                        after["mean_boundary_f1"] - before["mean_boundary_f1"]
                    ),
                    "evidence_finite": candidate.evidence.finite,
                    "visual_gain_ratio": candidate.evidence.visual_gain_ratio,
                    "gradient_cosine": candidate.evidence.gradient_cosine,
                    "correction_cosine": candidate.evidence.correction_cosine,
                    "relative_correction_disagreement": (
                        candidate.evidence.relative_correction_disagreement
                    ),
                }
            )
        for source, result in {
            "rt2_joint": candidate.joint_result,
            **{f"rt2_{name}": value for name, value in candidate.camera_results.items()},
        }.items():
            traces.extend(
                {
                    "case_id": case.case_id,
                    "state_id": case.state_id,
                    "candidate_source": source,
                    **trace,
                }
                for trace in result.trace
            )
        media_contexts.append(
            {"frame": frame, "observations": observations, "backend": visual}
        )

    parity = compare_fk_parity(
        bridge,
        fk,
        torch.stack([frame.synchronized_qpos for frame in frames]).cpu().numpy(),
    )
    summary = _summarize_formal(
        method_rows=method_rows,
        component_rows=component_rows,
        parity_passed=parity.passed,
    )
    selected_media = sorted(
        media_contexts,
        key=lambda context: (
            0
            if 0 < len(context["frame"].accepted_joint_names) < len(context["frame"].decisions)
            else 1
            if context["frame"].accepted_joint_names
            else 2,
            context["frame"].case_id,
        ),
    )[:2]
    image_path = run.path / "images" / "rt6_formal_synchronization.png"
    video_path = run.path / "videos" / "rt6_formal_synchronization.mp4"
    write_synchronization_comparison(image_path, selected_media)
    write_synchronization_video(
        video_path,
        selected_media,
        frame_count=int(evaluation.get("video_frames", 36)),
        fps=int(evaluation.get("video_fps", 12)),
    )

    _write_csv(run.path / "results.csv", method_rows)
    _write_csv(run.path / "component_results.csv", component_rows)
    _write_jsonl(run.path / "optimizer_executions.jsonl", optimizer_execution_rows)
    _write_jsonl(run.path / "state_frames.jsonl", state_frame_rows)
    run.write_trace(traces)
    source_hashes = {
        **actual_frozen,
        "formal_config": _sha256(frozen["config_path"]),
        "converted_urdf": _sha256(converted_path),
        "conversion_manifest": _sha256(run.path / "conversion_manifest.json"),
    }
    metrics = {
        "formal_evidence": True,
        "evaluation_split": "validation",
        "trial_count": matrix.trial_count,
        "unique_state_count": matrix.unique_state_count,
        "method_row_count": len(method_rows),
        "component_row_count": len(component_rows),
        "validation_state_ids": sorted(case_states),
        "candidate_procedure_runs": len(candidate_fingerprints),
        "optimizer_executions_per_candidate": frozen[
            "optimizer_executions_per_candidate"
        ],
        "optimizer_execution_count": len(optimizer_execution_rows),
        "max_source_timestamp_delta_ns": frozen["max_source_timestamp_delta_ns"],
        "candidate_fingerprints": candidate_fingerprints,
        "profile_fingerprint": profile.fingerprint,
        "matrix_fingerprint": matrix.fingerprint,
        "git_provenance": git_provenance,
        "source_hashes": source_hashes,
        "parity_report": asdict(parity),
        "selected_media_case_ids": [context["frame"].case_id for context in selected_media],
        **summary,
    }
    _write_json(run.path / "metrics.json", metrics)
    _write_json(
        run.path / "parity_report.json",
        {
            "formal_evidence": True,
            **asdict(parity),
            "conversion": conversion.to_dict(),
            "state_fingerprints": [frame.fingerprint for frame in frames],
        },
    )
    manifest_path = run.path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["formal_evidence"] = True
    manifest["git_provenance"] = git_provenance
    manifest["source_hashes"] = source_hashes
    manifest["derived_artifacts"] = {
        name: {"path": str(path), "sha256": _sha256(path)}
        for name, path in {
            "component_results": run.path / "component_results.csv",
            "conversion_manifest": run.path / "conversion_manifest.json",
            "converted_urdf": converted_path,
            "metrics": run.path / "metrics.json",
            "optimizer_executions": run.path / "optimizer_executions.jsonl",
            "parity_report": run.path / "parity_report.json",
            "results": run.path / "results.csv",
            "state_frames": run.path / "state_frames.jsonl",
            "synchronization_image": image_path,
            "synchronization_video": video_path,
            "trace": run.path / "trace.csv",
        }.items()
    }
    _write_json(manifest_path, manifest)
    _validate_formal_run_contract(run.path)
    return run.path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id")
    arguments = parser.parse_args()
    from kinesync.config import load_config

    print(
        execute_rt6_formal(
            load_config(arguments.config),
            run_id=arguments.run_id,
        )
    )


if __name__ == "__main__":
    main()
