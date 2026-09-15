"""Evaluate frozen RT2-F factors on untouched Route-A validation cases."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, median
from typing import Any, Mapping

import numpy as np
import torch

from kinesync.cli.rt1_real_matrix import _matrix_trials
from kinesync.cli.rt1_real_observation import (
    _anchor_indices,
    _evaluate_trial,
    _observability_prior_mask,
    _real_dataset,
)
from kinesync.config import load_config
from kinesync.correction.joint_offset import JointOffsetRecovery
from kinesync.data.route_a import RouteAAssets, RouteAPaths
from kinesync.factorization import FrozenCameraObservationBackend
from kinesync.geometry.urdf import TorchURDFKinematics
from kinesync.observation.cuda_gaussian import CudaGaussianBackend
from kinesync.observation.real_route_a import prepare_real_targets
from kinesync.runs.artifacts import RunArtifacts
from kinesync.visualization.factorization import (
    write_frozen_validation_comparison,
    write_frozen_validation_video,
)


def _load_factors(path: str | Path) -> tuple[Path, dict[str, Any]]:
    factor_path = Path(path).expanduser().resolve()
    payload = json.loads(factor_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or payload.get("frozen") is not True:
        raise ValueError("RT2-F evaluation requires a frozen schema-v1 factor file")
    if payload.get("fit_split") != "train":
        raise ValueError("RT2-F factors must originate from train-only fitting")
    return factor_path, payload


def _resolve_path(config: Mapping[str, Any], raw_path: str | Path) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path.resolve()
    config_path = config.get("_config_path")
    base = Path(config_path).parent if config_path else Path.cwd()
    return (base / path).resolve()


def _evaluation_cases(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    evaluation = config["evaluation"]
    if "cases" in evaluation:
        cases = [dict(value) for value in evaluation["cases"]]
    elif "matrix" in evaluation:
        matrix = load_config(_resolve_path(config, evaluation["matrix"]))
        cases = _matrix_trials(matrix)
    else:
        raise ValueError("evaluation requires cases or a matrix path")
    if not cases:
        raise ValueError("RT2-F evaluation has no cases")
    return cases


def _media_priority(row: Mapping[str, Any]) -> tuple[int, int, int, float, float]:
    """Rank validation cases for review media without changing evaluation."""

    return (
        int(bool(row["recovery_success"])),
        int(str(row["camera_mode"]) == "head+extra"),
        int(str(row["trial_type"]) == "controlled_offset"),
        float(row["state_error_reduction"]),
        float(row["mask_iou_gain"]),
    )


def _validate_factor_bounds(factors: Mapping[str, Any]) -> None:
    bounds = factors["bounds"]
    joint = np.abs(np.asarray(factors["joint_zero_rad"], dtype=np.float64))
    if float(joint.max(initial=0.0)) > float(bounds["joint_rad"]) + 1e-6:
        raise ValueError("Frozen joint factors exceed their declared bound")
    for value in factors["camera_twists"].values():
        twist = np.abs(np.asarray(value, dtype=np.float64))
        if twist.shape != (6,):
            raise ValueError("Each frozen camera twist must have six values")
        if float(twist[:3].max(initial=0.0)) > float(
            bounds["camera_rotation_rad"]
        ) + 1e-6:
            raise ValueError("Frozen camera rotation exceeds its declared bound")
        if float(twist[3:].max(initial=0.0)) > float(
            bounds["camera_translation_m"]
        ) + 1e-6:
            raise ValueError("Frozen camera translation exceeds its declared bound")


def execute_rt2_evaluate(
    config: Mapping[str, Any],
    *,
    factors_path: str | Path,
    run_id: str | None = None,
    pilot_cases: int | None = None,
) -> Path:
    device = torch.device(str(config.get("device", "cuda")))
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("RT2-F evaluation requires the native CUDA rasterizer")
    factor_path, factors = _load_factors(factors_path)
    _validate_factor_bounds(factors)
    cases = _evaluation_cases(config)
    if pilot_cases is not None:
        if pilot_cases <= 0:
            raise ValueError("pilot_cases must be positive")
        cases = cases[:pilot_cases]
    evaluation = config["evaluation"]
    paths = RouteAPaths.from_mapping(config["assets"])
    assets = RouteAAssets(paths)
    dataset = _real_dataset(config)
    fk = TorchURDFKinematics(paths.urdf)
    joint_names = [str(value) for value in factors["joint_names"]]
    if joint_names != fk.active_joint_names[: len(joint_names)]:
        raise ValueError("Frozen joint-factor names do not match the PiPER URDF")
    shared_joint = torch.zeros(
        len(fk.active_joint_names), dtype=torch.float32, device=device
    )
    shared_joint[: len(joint_names)] = torch.as_tensor(
        factors["joint_zero_rad"], dtype=torch.float32, device=device
    )
    available_camera_factors = {
        name: torch.as_tensor(value, dtype=torch.float32, device=device)
        for name, value in factors["camera_twists"].items()
    }
    height, width = map(int, config["observation"]["image_size"])
    seed = int(config.get("seed", 0))
    indices = _anchor_indices(
        assets, config["data"]["anchors_per_link"], seed=seed
    )
    renderer_cache: dict[tuple[str, ...], CudaGaussianBackend] = {}
    rows: list[dict[str, Any]] = []
    all_trace: list[dict[str, Any]] = []
    media_context = None
    source_observations: dict[tuple[str, str], object] = {}

    def renderer_for(camera_names: list[str]) -> CudaGaussianBackend:
        key = tuple(camera_names)
        if key not in renderer_cache:
            renderer_cache[key] = CudaGaussianBackend.from_route_a(
                kinematics=fk,
                anchors=assets.load_anchors(),
                anchor_indices=indices,
                link_names=assets.link_names,
                calibrations={name: assets.load_camera(name) for name in camera_names},
                output_size=(height, width),
                observation_mode="alpha",
                device=device,
            )
        return renderer_cache[key]

    for case_index, case in enumerate(cases):
        state_id = str(case["state_id"])
        camera_names = [str(value) for value in case["cameras"]]
        if not set(camera_names).issubset(available_camera_factors):
            raise ValueError("Evaluation case requests a camera without a frozen factor")
        observations = dataset.load_state(state_id, cameras=camera_names)
        if any(value.split != "validation" for value in observations.values()):
            raise ValueError("RT2-F evaluation may only read validation observations")
        source_observations.update(
            {(state_id, camera): value for camera, value in observations.items()}
        )
        targets = prepare_real_targets(
            observations, output_size=(height, width), device=device
        )
        reference = torch.as_tensor(
            np.mean(np.stack([value.qpos for value in observations.values()]), axis=0),
            dtype=torch.float32,
            device=device,
        )
        calibrated_reference = reference + shared_joint
        offsets = {
            str(name): float(value)
            for name, value in case["offsets_rad"].items()
        }
        unknown = set(offsets) - set(fk.active_joint_names)
        if unknown:
            raise ValueError(f"Unknown evaluation joints: {sorted(unknown)}")
        injected = torch.zeros_like(calibrated_reference)
        selected_indices = []
        for name, value in offsets.items():
            index = fk.active_joint_names.index(name)
            selected_indices.append(index)
            injected[index] = value
        measured = calibrated_reference + injected
        camera_twists = {
            name: available_camera_factors[name] for name in camera_names
        }
        renderer = renderer_for(camera_names)
        prior_kind = str(evaluation.get("prior_kind", "adaptive"))
        prior_mask = None
        if prior_kind == "adaptive":
            probe = FrozenCameraObservationBackend(
                renderer,
                camera_twists=camera_twists,
                boundary_weight=float(config["observation"]["boundary_weight"]),
            )
            probe_qpos = measured.detach().clone().requires_grad_(True)
            probe_loss, _ = probe.loss(probe_qpos, targets)
            probe_loss.backward()
            prior_mask = _observability_prior_mask(
                probe_qpos.grad.detach().abs(),
                selected_indices=selected_indices,
                minimum_gradient=float(evaluation.get("minimum_gradient", 0.05)),
                minimum_relative=float(evaluation.get("minimum_relative", 0.10)),
            )
        backend = FrozenCameraObservationBackend(
            renderer,
            camera_twists=camera_twists,
            boundary_weight=float(config["observation"]["boundary_weight"]),
            prior_weight=float(evaluation.get("prior_weight", 5.0)),
            measured_qpos=measured,
            prior_kind=prior_kind,
            prior_delta_rad=float(evaluation.get("prior_delta_rad", 0.02)),
            prior_cauchy_mask=prior_mask,
        )
        before_metrics = backend.metrics(measured, targets)
        recovery = JointOffsetRecovery(
            backend,
            joint_names=fk.active_joint_names,
            selected_joints=list(offsets),
            steps=int(evaluation["steps"]),
            learning_rate=float(evaluation["learning_rate"]),
            offset_bound=float(evaluation["offset_bound_rad"]),
            seed=int(case.get("seed", seed)),
        )
        result = recovery.recover(
            measured, targets, reference_qpos=calibrated_reference
        )
        after_metrics = backend.metrics(result.corrected_qpos, targets)
        initial_mae = float(
            (measured[selected_indices] - calibrated_reference[selected_indices])
            .abs()
            .mean()
        )
        final_mae = float(
            (
                result.corrected_qpos[selected_indices]
                - calibrated_reference[selected_indices]
            )
            .abs()
            .mean()
        )
        applied_mae = float(result.applied_correction[selected_indices].abs().mean())
        decision = _evaluate_trial(
            initial_selected_mae=initial_mae,
            final_selected_mae=final_mae,
            applied_correction_mae=applied_mae,
            before_metrics=before_metrics,
            after_metrics=after_metrics,
            acceptance=config.get("acceptance", {}),
        )
        magnitude_deg = math.degrees(
            max((abs(value) for value in offsets.values()), default=0.0)
        )
        trial_id = (
            f"{state_id}_{'-'.join(camera_names)}_"
            f"{case.get('offset_name', case_index)}_s{int(case.get('seed', seed))}"
        )
        row = {
            "trial_id": trial_id,
            "state_id": state_id,
            "split": "validation",
            "camera_mode": "+".join(camera_names),
            "offset_name": str(case.get("offset_name", case_index)),
            "offsets_rad": json.dumps(offsets, sort_keys=True),
            "magnitude_deg": magnitude_deg,
            "trial_type": decision["trial_type"],
            "state_error_reduction": float(decision["state_error_reduction"]),
            "natural_state_correction_rad": decision[
                "natural_state_correction_rad"
            ],
            "mask_iou_before": before_metrics["mean_mask_iou"],
            "mask_iou_after": after_metrics["mean_mask_iou"],
            "mask_iou_gain": (
                after_metrics["mean_mask_iou"] - before_metrics["mean_mask_iou"]
            ),
            "boundary_f1_before": before_metrics["mean_boundary_f1"],
            "boundary_f1_after": after_metrics["mean_boundary_f1"],
            "boundary_f1_gain": (
                after_metrics["mean_boundary_f1"]
                - before_metrics["mean_boundary_f1"]
            ),
            "recovery_success": bool(decision["recovery_success"]),
        }
        rows.append(row)
        for trace_row in result.trace:
            all_trace.append({"trial_id": trial_id, **trace_row})
        if media_context is None or _media_priority(row) > media_context["priority"]:
            with torch.no_grad():
                before_render = backend.render(measured)
                after_render = backend.render(result.corrected_qpos)
            media_context = {
                "priority": _media_priority(row),
                "backend": backend,
                "observations": observations,
                "targets": targets,
                "measured": measured,
                "before": before_render,
                "after": after_render,
                "trace": result.trace,
                "metrics": {
                    **row,
                    "mean_mask_iou_before": before_metrics["mean_mask_iou"],
                    "mean_mask_iou_after": after_metrics["mean_mask_iou"],
                    "mean_boundary_f1_before": before_metrics["mean_boundary_f1"],
                    "mean_boundary_f1_after": after_metrics["mean_boundary_f1"],
                },
            }

    controlled = [row for row in rows if row["trial_type"] == "controlled_offset"]
    zero_controls = [
        row for row in rows if row["trial_type"] == "zero_injection_control"
    ]
    one_degree = [
        row for row in controlled if abs(float(row["magnitude_deg"]) - 1.0) <= 0.1
    ]
    paired = [row for row in controlled if "+" in str(row["camera_mode"])]
    controlled_success = (
        mean(float(row["recovery_success"]) for row in controlled)
        if controlled
        else None
    )
    zero_success = (
        mean(float(row["recovery_success"]) for row in zero_controls)
        if zero_controls
        else None
    )
    one_degree_success = (
        mean(float(row["recovery_success"]) for row in one_degree)
        if one_degree
        else None
    )
    paired_median = (
        median(float(row["state_error_reduction"]) for row in paired)
        if paired
        else None
    )
    mask_gain = mean(float(row["mask_iou_gain"]) for row in rows)
    boundary_gain = mean(float(row["boundary_f1_gain"]) for row in rows)
    gates = {
        "zero_control_success_at_least_80pct": bool(
            zero_success is not None and zero_success >= 0.80
        ),
        "one_degree_success_at_least_50pct": bool(
            one_degree_success is not None and one_degree_success >= 0.50
        ),
        "controlled_success_at_least_70pct": bool(
            controlled_success is not None and controlled_success >= 0.70
        ),
        "positive_mean_visual_gains": bool(mask_gain > 0 and boundary_gain > 0),
        "positive_paired_median_reduction": bool(
            paired_median is not None and paired_median > 0
        ),
        "factors_within_declared_bounds": True,
    }
    metrics = {
        "target_provenance": "real_observation_factorization_evaluation",
        "evaluation_split": "validation",
        "factor_source_run": str(factors["source_run"]),
        "factor_file": str(factor_path),
        "case_count": len(rows),
        "validation_state_ids": sorted({str(row["state_id"]) for row in rows}),
        "controlled_case_count": len(controlled),
        "controlled_success_rate": controlled_success,
        "zero_control_count": len(zero_controls),
        "zero_control_success_rate": zero_success,
        "one_degree_case_count": len(one_degree),
        "one_degree_success_rate": one_degree_success,
        "paired_camera_median_state_error_reduction": paired_median,
        "mean_mask_iou_gain": mask_gain,
        "mean_boundary_f1_gain": boundary_gain,
        "acceptance_gates": gates,
        "all_acceptance_gates_passed": all(gates.values()),
    }

    run_assets: dict[str, str | Path] = {**paths.as_assets(), "factors": factor_path}
    real_source = config["real_observations"]
    for camera in dataset.cameras:
        run_assets[f"records_{camera}"] = real_source["records"][camera]
    for (state_id, camera), observation in source_observations.items():
        run_assets[f"rgb_{state_id}_{camera}"] = (
            Path(real_source["image_roots"][camera])
            / "images"
            / f"{observation.sample_id}.jpg"
        )
        run_assets[f"mask_{state_id}_{camera}"] = (
            Path(real_source["mask_roots"][camera])
            / f"{observation.sample_id}.png"
        )
    run = RunArtifacts.create(
        root=config.get("runs_root", "runs"),
        experiment=str(config.get("experiment", "rt2_piper_factorization"))
        + "_evaluation",
        run_id=run_id,
        config=config,
        assets=run_assets,
        target_provenance="real_observation_factorization_evaluation",
        observation_backend="native_cuda_frozen_factor_mask",
    )
    run.write_trace(all_trace)
    run.write_metrics(metrics)
    with (run.path / "evaluation_rows.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    assert media_context is not None
    write_frozen_validation_comparison(
        run.path / "images" / "frozen_validation_before_after.png",
        observations=media_context["observations"],
        targets=media_context["targets"],
        before=media_context["before"],
        after=media_context["after"],
        metrics=media_context["metrics"],
    )
    write_frozen_validation_video(
        run.path / "videos" / "frozen_validation_recovery.mp4",
        backend=media_context["backend"],
        observations=media_context["observations"],
        targets=media_context["targets"],
        measured_qpos=media_context["measured"],
        trace=media_context["trace"],
        joint_names=fk.active_joint_names,
        frame_count=int(evaluation.get("video_frames", 36)),
        fps=int(evaluation.get("video_fps", 12)),
    )
    return run.path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--factors", type=Path, required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--pilot-cases", type=int)
    arguments = parser.parse_args()
    print(
        execute_rt2_evaluate(
            load_config(arguments.config),
            factors_path=arguments.factors,
            run_id=arguments.run_id,
            pilot_cases=arguments.pilot_cases,
        )
    )


if __name__ == "__main__":
    main()
