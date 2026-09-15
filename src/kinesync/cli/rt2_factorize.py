"""Fit shared PiPER joint-zero and camera factors from Route-A train frames."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from kinesync.cli.rt1_real_observation import _anchor_indices, _real_dataset
from kinesync.config import load_config
from kinesync.data.route_a import RouteAAssets, RouteAPaths
from kinesync.factorization import (
    FactorFrame,
    FactorizationConfig,
    PhysicalFactorOptimizer,
    RouteAFactorizationBackend,
    select_evenly,
)
from kinesync.geometry.urdf import TorchURDFKinematics
from kinesync.observation.cuda_gaussian import CudaGaussianBackend
from kinesync.observation.real_route_a import prepare_real_targets
from kinesync.runs.artifacts import RunArtifacts
from kinesync.visualization.factorization import (
    write_factorization_comparison,
    write_factorization_video,
)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _factorization_config(
    config: Mapping[str, Any], camera_names: list[str]
) -> FactorizationConfig:
    values = config["factorization"]
    return FactorizationConfig(
        joint_count=int(values["joint_count"]),
        camera_names=tuple(camera_names),
        joint_bound=float(values["joint_bound_rad"]),
        camera_rotation_bound=float(values["camera_rotation_bound_rad"]),
        camera_translation_bound=float(values["camera_translation_bound_m"]),
        joint_steps=int(values["joint_steps"]),
        camera_steps=int(values["camera_steps"]),
        refinement_steps=int(values["refinement_steps"]),
        joint_learning_rate=float(values["joint_learning_rate"]),
        camera_learning_rate=float(values["camera_learning_rate"]),
        batch_size=int(values["batch_size"]),
        seed=int(config.get("seed", 0)),
        rms_gradient_min=float(values.get("rms_gradient_min", 1e-4)),
        relative_parameter_min=float(values.get("relative_parameter_min", 0.1)),
        singular_ratio_min=float(values.get("singular_ratio_min", 1e-3)),
        joint_prior_weight=float(values.get("joint_prior_weight", 0.0)),
        joint_prior_delta=float(values.get("joint_prior_delta_rad", 0.02)),
        camera_rotation_prior_weight=float(
            values.get("camera_rotation_prior_weight", 0.0)
        ),
        camera_translation_prior_weight=float(
            values.get("camera_translation_prior_weight", 0.0)
        ),
    )


def _observability_payload(report) -> dict[str, Any]:
    return {
        "frame_count": report.frame_count,
        "rms_gradient_norm": report.rms_gradient_norm,
        "median_gradient_norm": report.median_gradient_norm,
        "singular_values": list(report.singular_values),
        "retained_rank": report.retained_rank,
        "median_absolute_gradient": report.median_absolute_gradient.cpu().tolist(),
        "sign_agreement": report.sign_agreement.cpu().tolist(),
        "parameter_mask": report.parameter_mask.cpu().tolist(),
        "accepted": report.accepted,
    }


def execute_rt2_factorize(
    config: Mapping[str, Any],
    *,
    run_id: str | None = None,
    pilot_frames: int | None = None,
) -> Path:
    device = torch.device(str(config.get("device", "cuda")))
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("RT2-F requires the native CUDA Gaussian rasterizer")
    seed = int(config.get("seed", 0))
    torch.manual_seed(seed)
    np.random.seed(seed)
    camera_names = [str(name) for name in config["data"]["cameras"]]
    paths = RouteAPaths.from_mapping(config["assets"])
    assets = RouteAAssets(paths)
    dataset = _real_dataset(config)
    available_ids = dataset.state_ids(split="train", cameras=camera_names)
    requested_count = int(
        pilot_frames
        if pilot_frames is not None
        else config["data"]["train_frame_count"]
    )
    state_ids = select_evenly(available_ids, requested_count)
    height, width = map(int, config["observation"]["image_size"])
    frame_observations = {}
    frames = []
    for state_id in state_ids:
        observations = dataset.load_state(state_id, cameras=camera_names)
        if any(value.split != "train" for value in observations.values()):
            raise ValueError("RT2-F factor fitting may only read train observations")
        qpos = torch.as_tensor(
            np.mean(np.stack([value.qpos for value in observations.values()]), axis=0),
            dtype=torch.float32,
            device=device,
        )
        targets = prepare_real_targets(
            observations, output_size=(height, width), device=device
        )
        frames.append(
            FactorFrame(
                state_id=state_id,
                split="train",
                qpos=qpos,
                targets=targets,
            )
        )
        frame_observations[state_id] = observations

    fk = TorchURDFKinematics(paths.urdf)
    factor_config = _factorization_config(config, camera_names)
    if factor_config.joint_count > len(fk.active_joint_names):
        raise ValueError("Configured joint factor count exceeds active URDF joints")
    indices = _anchor_indices(
        assets, config["data"]["anchors_per_link"], seed=seed
    )
    renderer = CudaGaussianBackend.from_route_a(
        kinematics=fk,
        anchors=assets.load_anchors(),
        anchor_indices=indices,
        link_names=assets.link_names,
        calibrations={name: assets.load_camera(name) for name in camera_names},
        output_size=(height, width),
        observation_mode="alpha",
        device=device,
    )
    backend = RouteAFactorizationBackend(
        renderer,
        arm_joint_count=factor_config.joint_count,
        boundary_weight=float(config["observation"]["boundary_weight"]),
    )
    zero_joint = torch.zeros(
        factor_config.joint_count, dtype=torch.float32, device=device
    )
    zero_cameras = {
        name: torch.zeros(6, dtype=torch.float32, device=device)
        for name in camera_names
    }
    before_metrics = backend.metrics(frames, zero_joint, zero_cameras)
    result = PhysicalFactorOptimizer(backend, factor_config).fit(frames)
    joint_zero = result.joint_zero.to(device)
    camera_twists = {
        name: value.to(device) for name, value in result.camera_twists.items()
    }
    after_metrics = backend.metrics(frames, joint_zero, camera_twists)

    run_assets: dict[str, str | Path] = paths.as_assets()
    real_source = config["real_observations"]
    for camera in camera_names:
        run_assets[f"records_{camera}"] = real_source["records"][camera]
    for state_id in state_ids:
        for camera in camera_names:
            observation = frame_observations[state_id][camera]
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
        experiment=str(config.get("experiment", "rt2_piper_factorization")),
        run_id=run_id,
        config=config,
        assets=run_assets,
        target_provenance="real_observation_factorization",
        observation_backend="native_cuda_dynamic_camera_mask",
    )
    metrics: dict[str, Any] = {
        "target_provenance": "real_observation_factorization",
        "fit_split": "train",
        "train_state_ids": state_ids,
        "train_frame_count": len(state_ids),
        "camera_names": camera_names,
        "anchor_count": int(len(indices)),
        "output_size": [height, width],
        "stage_steps": [
            factor_config.joint_steps,
            factor_config.camera_steps,
            factor_config.refinement_steps,
        ],
        "initial_loss": result.initial_loss,
        "final_loss": result.final_loss,
        "loss_reduction_ratio": result.final_loss / max(result.initial_loss, 1e-12),
        "nonfinite_skips": result.nonfinite_skips,
        "mean_mask_iou_before": before_metrics["mean_mask_iou"],
        "mean_mask_iou_after": after_metrics["mean_mask_iou"],
        "mean_mask_iou_gain": (
            after_metrics["mean_mask_iou"] - before_metrics["mean_mask_iou"]
        ),
        "mean_boundary_f1_before": before_metrics["mean_boundary_f1"],
        "mean_boundary_f1_after": after_metrics["mean_boundary_f1"],
        "mean_boundary_f1_gain": (
            after_metrics["mean_boundary_f1"]
            - before_metrics["mean_boundary_f1"]
        ),
        "per_frame_before": {
            key: value
            for key, value in before_metrics.items()
            if not key.startswith("mean_")
        },
        "per_frame_after": {
            key: value
            for key, value in after_metrics.items()
            if not key.startswith("mean_")
        },
    }
    factor_payload = {
        "schema_version": 1,
        "frozen": True,
        "source_run": run.path.name,
        "fit_split": "train",
        "train_state_ids": state_ids,
        "joint_names": fk.active_joint_names[: factor_config.joint_count],
        "joint_zero_rad": result.joint_zero.tolist(),
        "camera_twist_order": ["rx", "ry", "rz", "tx", "ty", "tz"],
        "camera_twists": {
            name: value.tolist() for name, value in result.camera_twists.items()
        },
        "bounds": {
            "joint_rad": factor_config.joint_bound,
            "camera_rotation_rad": factor_config.camera_rotation_bound,
            "camera_translation_m": factor_config.camera_translation_bound,
        },
    }
    run.write_trace(result.trace)
    run.write_metrics(metrics)
    _write_json(run.path / "factors.json", factor_payload)
    _write_json(
        run.path / "observability.json",
        {
            name: _observability_payload(report)
            for name, report in result.observability.items()
        },
    )
    visual_config = config.get("visualization", {})
    comparison_index = len(frames) // 2
    comparison_frame = frames[comparison_index]
    write_factorization_comparison(
        run.path / "images" / "factorization_before_after.png",
        renderer=renderer,
        frame=comparison_frame,
        observations=frame_observations[comparison_frame.state_id],
        joint_zero=result.joint_zero,
        camera_twists=result.camera_twists,
        metrics=metrics,
    )
    video_frame_count = int(visual_config.get("video_frames", 36))
    comparison_count = min(
        int(visual_config.get("comparison_frames", len(frames))), len(frames)
    )
    visual_ids = set(select_evenly(state_ids, comparison_count))
    visual_frames = [frame for frame in frames if frame.state_id in visual_ids]
    write_factorization_video(
        run.path / "videos" / "factorization_progress.mp4",
        renderer=renderer,
        frames=visual_frames,
        observations=frame_observations,
        joint_zero=result.joint_zero,
        camera_twists=result.camera_twists,
        frame_count=video_frame_count,
        fps=int(visual_config.get("video_fps", 12)),
    )
    return run.path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--pilot-frames", type=int)
    arguments = parser.parse_args()
    print(
        execute_rt2_factorize(
            load_config(arguments.config),
            run_id=arguments.run_id,
            pilot_frames=arguments.pilot_frames,
        )
    )


if __name__ == "__main__":
    main()
