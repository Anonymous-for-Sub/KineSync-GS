"""Calibrate an immutable RT3 conditional-update guard on train states."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from kinesync.cli.rt1_real_observation import (
    _anchor_indices,
    _observability_prior_mask,
    _real_dataset,
)
from kinesync.cli.rt2_evaluate import _load_factors, _validate_factor_bounds
from kinesync.config import load_config
from kinesync.data.route_a import RouteAAssets, RouteAPaths
from kinesync.experiments.rt3_matrix import load_rt3_cases
from kinesync.factorization import FrozenCameraObservationBackend
from kinesync.geometry.urdf import TorchURDFKinematics
from kinesync.guard.calibration import CalibrationRecord, calibrate_guard
from kinesync.guard.candidate import run_guard_candidate
from kinesync.observation.cuda_gaussian import CudaGaussianBackend
from kinesync.observation.real_route_a import prepare_real_targets
from kinesync.runs.artifacts import RunArtifacts
from kinesync.sync.calibration import derive_per_joint_recovery


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
        raise ValueError("Cannot write an empty calibration table")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _prior_backend(
    visual_backend,
    renderer,
    target,
    *,
    camera_twists: Mapping[str, torch.Tensor],
    measured_qpos: torch.Tensor,
    selected_indices: list[int],
    evaluation: Mapping[str, Any],
    boundary_weight: float,
):
    prior_kind = str(evaluation.get("prior_kind", "adaptive"))
    prior_mask = None
    if prior_kind == "adaptive":
        probe = measured_qpos.detach().clone().requires_grad_(True)
        probe_loss, _ = visual_backend.loss(probe, target)
        probe_loss.backward()
        prior_mask = _observability_prior_mask(
            probe.grad.detach().abs(),
            selected_indices=selected_indices,
            minimum_gradient=float(evaluation.get("minimum_gradient", 0.05)),
            minimum_relative=float(evaluation.get("minimum_relative", 0.10)),
        )
    return FrozenCameraObservationBackend(
        renderer,
        camera_twists=camera_twists,
        boundary_weight=boundary_weight,
        sdf_weight=float(evaluation.get("sdf_weight", 0.0)),
        sdf_radii=tuple(
            float(value) for value in evaluation.get("sdf_radii_px", (4.0, 12.0))
        ),
        prior_weight=float(evaluation.get("prior_weight", 5.0)),
        measured_qpos=measured_qpos,
        prior_kind=prior_kind,
        prior_delta_rad=float(evaluation.get("prior_delta_rad", 0.02)),
        prior_cauchy_mask=prior_mask,
    )


def execute_rt3_calibrate(
    config: Mapping[str, Any],
    *,
    factors_path: str | Path,
    matrix_path: str | Path,
    run_id: str | None = None,
) -> Path:
    device = torch.device(str(config.get("device", "cuda")))
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("RT3 calibration requires the native CUDA rasterizer")
    factor_path, factors = _load_factors(factors_path)
    _validate_factor_bounds(factors)
    factor_sha = _sha256(factor_path)
    matrix_path = Path(matrix_path).expanduser().resolve()
    matrix_payload = load_config(matrix_path)
    old_states = matrix_payload["provenance"]["old_validation_state_ids"]
    paths = RouteAPaths.from_mapping(config["assets"])
    assets = RouteAAssets(paths)
    dataset = _real_dataset(config)
    cases = load_rt3_cases(matrix_path, dataset, factors, old_states)
    if any(case["split"] != "train" for case in cases):
        raise ValueError("RT3 calibration accepts train-dev cases only")

    fk = TorchURDFKinematics(paths.urdf)
    joint_names = [str(value) for value in factors["joint_names"]]
    if joint_names != fk.active_joint_names[: len(joint_names)]:
        raise ValueError("Frozen joint factors do not match the PiPER URDF")
    shared_joint = torch.zeros(
        len(fk.active_joint_names), dtype=torch.float32, device=device
    )
    shared_joint[: len(joint_names)] = torch.as_tensor(
        factors["joint_zero_rad"], dtype=torch.float32, device=device
    )
    camera_factors = {
        name: torch.as_tensor(value, dtype=torch.float32, device=device)
        for name, value in factors["camera_twists"].items()
    }
    height, width = map(int, config["observation"]["image_size"])
    boundary_weight = float(config["observation"]["boundary_weight"])
    seed = int(config.get("seed", 41))
    indices = _anchor_indices(assets, config["data"]["anchors_per_link"], seed=seed)
    renderer_cache: dict[tuple[str, ...], CudaGaussianBackend] = {}

    def renderer_for(names: list[str]) -> CudaGaussianBackend:
        key = tuple(names)
        if key not in renderer_cache:
            renderer_cache[key] = CudaGaussianBackend.from_route_a(
                kinematics=fk,
                anchors=assets.load_anchors(),
                anchor_indices=indices,
                link_names=assets.link_names,
                calibrations={name: assets.load_camera(name) for name in names},
                output_size=(height, width),
                observation_mode="alpha",
                device=device,
            )
        return renderer_cache[key]

    evaluation = config["evaluation"]
    rows: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    records: list[CalibrationRecord] = []
    for case in cases:
        state_id = case["state_id"]
        camera_names = case["camera_names"]
        observations = dataset.load_state(state_id, cameras=camera_names)
        if any(value.split != "train" for value in observations.values()):
            raise ValueError("RT3 calibration read a non-train observation")
        targets = prepare_real_targets(
            observations, output_size=(height, width), device=device
        )
        reference = torch.as_tensor(
            np.mean(np.stack([value.qpos for value in observations.values()]), axis=0),
            dtype=torch.float32,
            device=device,
        )
        calibrated_reference = reference + shared_joint
        injected = torch.zeros_like(calibrated_reference)
        selected_joints = list(case["offsets_rad"])
        selected_indices = []
        for name, value in case["offsets_rad"].items():
            index = fk.active_joint_names.index(name)
            selected_indices.append(index)
            injected[index] = value
        measured = calibrated_reference + injected
        joint_renderer = renderer_for(camera_names)
        joint_twists = {name: camera_factors[name] for name in camera_names}
        joint_visual = FrozenCameraObservationBackend(
            joint_renderer,
            camera_twists=joint_twists,
            boundary_weight=boundary_weight,
        )
        joint_backend = _prior_backend(
            joint_visual,
            joint_renderer,
            targets,
            camera_twists=joint_twists,
            measured_qpos=measured,
            selected_indices=selected_indices,
            evaluation=evaluation,
            boundary_weight=boundary_weight,
        )
        camera_targets = {name: {name: targets[name]} for name in camera_names}
        camera_visual = {}
        camera_backends = {}
        for name in camera_names:
            renderer = renderer_for([name])
            twists = {name: camera_factors[name]}
            visual = FrozenCameraObservationBackend(
                renderer,
                camera_twists=twists,
                boundary_weight=boundary_weight,
            )
            camera_visual[name] = visual
            camera_backends[name] = _prior_backend(
                visual,
                renderer,
                camera_targets[name],
                camera_twists=twists,
                measured_qpos=measured,
                selected_indices=selected_indices,
                evaluation=evaluation,
                boundary_weight=boundary_weight,
            )
        candidate = run_guard_candidate(
            joint_backend=joint_backend,
            joint_visual_backend=joint_visual,
            joint_target=targets,
            camera_backends=camera_backends,
            camera_visual_backends=camera_visual,
            camera_targets=camera_targets,
            measured_qpos=measured,
            reference_qpos=calibrated_reference,
            joint_names=fk.active_joint_names,
            selected_joints=selected_joints,
            steps=int(evaluation["steps"]),
            learning_rate=float(evaluation["learning_rate"]),
            offset_bound=float(evaluation["offset_bound_rad"]),
            seed=int(case["seed"]),
            acceptance=config.get("acceptance", {}),
        )
        evidence = candidate.evidence
        trial_type = "zero" if case["case_role"] == "paired_zero" else "controlled"
        records.append(
            CalibrationRecord(
                evidence=evidence,
                trial_type=trial_type,
                candidate_success=bool(candidate.evaluation["recovery_success"]),
                state_id=state_id,
            )
        )
        trial_id = f"{state_id}_{case['offset_name']}"
        calibration_row = {
                "schema_version": 2,
                "trial_id": trial_id,
                "state_id": state_id,
                "split": "train",
                "case_role": case["case_role"],
                "offset_name": case["offset_name"],
                "offsets_rad": json.dumps(case["offsets_rad"], sort_keys=True),
                "joint_order": json.dumps(list(fk.active_joint_names)),
                "joint_order_source_sha256": factor_sha,
                "candidate_success": bool(candidate.evaluation["recovery_success"]),
                "trial_candidate_success": bool(candidate.evaluation["recovery_success"]),
                "state_error_reduction": candidate.evaluation["state_error_reduction"],
                "trial_state_error_reduction": candidate.evaluation["state_error_reduction"],
                "mask_iou_before": candidate.before_metrics["mean_mask_iou"],
                "mask_iou_candidate": candidate.candidate_metrics["mean_mask_iou"],
                "boundary_f1_before": candidate.before_metrics["mean_boundary_f1"],
                "boundary_f1_candidate": candidate.candidate_metrics["mean_boundary_f1"],
                "evidence_finite": evidence.finite,
                "view_count": evidence.view_count,
                "visual_gain_ratio": evidence.visual_gain_ratio,
                "gradient_cosine": evidence.gradient_cosine,
                "correction_cosine": evidence.correction_cosine,
                "relative_correction_disagreement": evidence.relative_correction_disagreement,
                "candidate_bound_fraction": evidence.candidate_bound_fraction,
                "joint_correction_rad": json.dumps(
                    candidate.joint_result.applied_correction.cpu().tolist()
                ),
            }
        calibration_row["per_joint_recovery"] = json.dumps(
            derive_per_joint_recovery(
                calibration_row,
                joint_names=fk.active_joint_names,
            ),
            sort_keys=True,
        )
        rows.append(calibration_row)
        for source, result in {
            "joint": candidate.joint_result,
            **candidate.camera_results,
        }.items():
            traces.extend(
                {"trial_id": trial_id, "candidate_source": source, **trace}
                for trace in result.trace
            )

    guard_config = config.get("guard", {})
    calibration = calibrate_guard(
        records,
        quantiles=guard_config.get("quantiles", [0.0, 0.25, 0.5, 0.75, 1.0]),
        minimum_zero_stability=float(
            guard_config.get("minimum_zero_stability", 0.9)
        ),
        minimum_controlled_coverage=float(
            guard_config.get("minimum_controlled_coverage", 0.6)
        ),
    )
    matrix_sha = _sha256(matrix_path)
    candidate_table = [
        {
            **asdict(item),
            "thresholds": asdict(item.thresholds),
        }
        for item in calibration.candidate_table
    ]
    guard_payload = {
        "schema_version": 1,
        "frozen": True,
        "source_split": "train",
        "factor_file": str(factor_path),
        "factor_sha256": factor_sha,
        "matrix_file": str(matrix_path),
        "matrix_sha256": matrix_sha,
        "source_state_ids": list(calibration.source_state_ids),
        "thresholds": asdict(calibration.thresholds),
        "controlled_coverage": calibration.controlled_coverage,
        "zero_stability": calibration.zero_stability,
        "guarded_success": calibration.guarded_success,
        "calibration_fingerprint": calibration.fingerprint,
        "candidate_table": candidate_table,
        "optimizer": {
            "steps": int(evaluation["steps"]),
            "learning_rate": float(evaluation["learning_rate"]),
            "offset_bound_rad": float(evaluation["offset_bound_rad"]),
            "prior_weight": float(evaluation.get("prior_weight", 5.0)),
            "prior_kind": str(evaluation.get("prior_kind", "adaptive")),
        },
    }
    metrics = {
        "source_split": "train",
        "case_count": len(rows),
        "train_state_ids": list(calibration.source_state_ids),
        "validation_state_ids": [],
        "factor_sha256": factor_sha,
        "matrix_sha256": matrix_sha,
        "controlled_coverage": calibration.controlled_coverage,
        "zero_stability": calibration.zero_stability,
        "guarded_success": calibration.guarded_success,
        "calibration_fingerprint": calibration.fingerprint,
    }
    real_source = config["real_observations"]
    run_assets: dict[str, str | Path] = {
        **paths.as_assets(),
        "factors": factor_path,
        "guard_dev_matrix": matrix_path,
        **{
            f"records_{camera}": real_source["records"][camera]
            for camera in dataset.cameras
        },
    }
    run = RunArtifacts.create(
        root=config.get("runs_root", "runs"),
        experiment=str(config.get("experiment", "rt3_guard_calibration")),
        run_id=run_id,
        config=config,
        assets=run_assets,
        target_provenance="real_observation_guard_calibration",
        observation_backend="native_cuda_frozen_factor_guard_dev",
    )
    run.write_metrics(metrics)
    run.write_trace(traces)
    _write_csv(run.path / "calibration_rows.csv", rows)
    _write_json(run.path / "guard.json", guard_payload)
    return run.path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--factors", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--run-id")
    arguments = parser.parse_args()
    print(
        execute_rt3_calibrate(
            load_config(arguments.config),
            factors_path=arguments.factors,
            matrix_path=arguments.matrix,
            run_id=arguments.run_id,
        )
    )


if __name__ == "__main__":
    main()
