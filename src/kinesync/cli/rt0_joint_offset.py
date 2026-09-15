"""Run PiPER RT0 visual joint-offset recovery."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from kinesync.config import load_config
from kinesync.correction.joint_offset import JointOffsetRecovery
from kinesync.data.route_a import RouteAAssets, RouteAPaths
from kinesync.geometry.urdf import TorchURDFKinematics
from kinesync.observation.camera import TorchCamera
from kinesync.observation.cuda_gaussian import CudaGaussianBackend
from kinesync.observation.projected_centers import ProjectedCentersBackend
from kinesync.observation.soft_occupancy import SoftOccupancyBackend
from kinesync.runs.artifacts import RunArtifacts
from kinesync.visualization.diagnostics import (
    write_before_target_after,
    write_native_before_target_after,
    write_native_recovery_video,
    write_recovery_curves,
    write_recovery_video,
)


def _pixel_rmse(prediction, target) -> float:
    squared = []
    for name in target:
        mask = prediction[name].valid & target[name].valid.to(prediction[name].valid)
        delta = prediction[name].values[mask] - target[name].values.to(prediction[name].values)[mask]
        squared.append(delta.square().sum(dim=-1))
    return float(torch.cat(squared).mean().sqrt())


def _tip_error_mm(
    fk: TorchURDFKinematics, qpos: torch.Tensor, reference: torch.Tensor
) -> float:
    tip = fk.joints[-1].child
    predicted = fk.forward(qpos)[tip][..., :3, 3]
    target = fk.forward(reference)[tip][..., :3, 3]
    return float(torch.linalg.vector_norm(predicted - target) * 1000.0)


def execute_rt0(config: Mapping[str, Any], *, run_id: str | None = None) -> Path:
    paths = RouteAPaths.from_mapping(config["assets"])
    assets = RouteAAssets(paths)
    requested_device = str(config.get("device", "cuda"))
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(requested_device)
    dtype = torch.float32
    seed = int(config.get("seed", 0))
    torch.manual_seed(seed)
    np.random.seed(seed)

    fk = TorchURDFKinematics(paths.urdf)
    camera_names = list(config["data"]["cameras"])
    frame_index = int(config["data"]["frame_index"])
    source_frame = assets.load_frame(camera_names[0], frame_index)
    true_qpos = torch.as_tensor(source_frame.qpos, dtype=dtype, device=device)
    if true_qpos.numel() != len(fk.active_joint_names):
        raise ValueError("Route-A state dimension does not match the PiPER URDF")

    sampled = assets.sample_anchor_indices(
        per_link=int(config["data"]["anchors_per_link"]), seed=seed
    )
    anchors = assets.load_anchors()
    local_xyz = torch.as_tensor(anchors.xyz[sampled], dtype=dtype, device=device)
    link_index = torch.as_tensor(anchors.link_index[sampled], dtype=torch.long, device=device)
    cameras = {
        name: TorchCamera.from_calibration(
            assets.load_camera(name), device=device, dtype=dtype
        )
        for name in camera_names
    }
    projected_backend = ProjectedCentersBackend(
        fk, local_xyz, link_index, assets.link_names, cameras
    )

    injected_mapping = {
        str(name): float(value)
        for name, value in config["target"]["injected_joint_offsets_rad"].items()
    }
    unknown = set(injected_mapping) - set(fk.active_joint_names)
    if unknown:
        raise ValueError(f"Unknown injected joints: {sorted(unknown)}")
    injected = torch.zeros_like(true_qpos)
    for name, value in injected_mapping.items():
        injected[fk.active_joint_names.index(name)] = value
    measured_qpos = true_qpos + injected

    backend_name = str(config["observation"]["backend"])
    if backend_name == "projected_centers":
        backend = projected_backend
    elif backend_name == "soft_occupancy":
        height, width = config["observation"]["image_size"]
        backend = SoftOccupancyBackend(
            fk,
            local_xyz,
            link_index,
            assets.link_names,
            cameras,
            output_size=(int(height), int(width)),
            sigma_px=float(config["observation"].get("sigma_px", 1.5)),
        )
    elif backend_name in {"cuda_gaussian_alpha", "cuda_gaussian_rgb"}:
        height, width = config["observation"]["image_size"]
        backend = CudaGaussianBackend.from_route_a(
            kinematics=fk,
            anchors=anchors,
            anchor_indices=sampled,
            link_names=assets.link_names,
            calibrations={name: assets.load_camera(name) for name in camera_names},
            output_size=(int(height), int(width)),
            observation_mode="alpha" if backend_name.endswith("alpha") else "rgb",
            device=device,
        )
    else:
        raise ValueError(f"Unsupported RT0 observation backend: {backend_name}")
    target = backend.render(true_qpos)
    projected_target = projected_backend.render(true_qpos)

    optimizer_config = config["optimizer"]
    recovery = JointOffsetRecovery(
        backend,
        joint_names=fk.active_joint_names,
        selected_joints=list(injected_mapping),
        steps=int(optimizer_config["steps"]),
        learning_rate=float(optimizer_config["learning_rate"]),
        offset_bound=float(optimizer_config["offset_bound_rad"]),
        seed=seed,
    )
    result = recovery.recover(measured_qpos, target, reference_qpos=true_qpos)

    before_projection = projected_backend.render(measured_qpos)
    after_projection = projected_backend.render(result.corrected_qpos)
    selected_indices = [fk.active_joint_names.index(name) for name in injected_mapping]
    offset_mae = float(
        (result.estimated_measurement_offset[selected_indices] - injected[selected_indices])
        .abs()
        .mean()
    )
    pixel_before = _pixel_rmse(before_projection, projected_target)
    pixel_after = _pixel_rmse(after_projection, projected_target)
    tip_before = _tip_error_mm(fk, measured_qpos, true_qpos)
    tip_after = _tip_error_mm(fk, result.corrected_qpos, true_qpos)
    metrics = {
        "target_provenance": "synthetic_state",
        "visual_background_source": "real_rgb_context_only",
        "observation_backend": backend_name,
        "frame_index": frame_index,
        "camera_names": camera_names,
        "anchor_count": int(len(sampled)),
        "selected_joints": list(injected_mapping),
        "injected_measurement_offsets_rad": injected_mapping,
        "estimated_measurement_offsets_rad": {
            name: float(result.estimated_measurement_offset[fk.active_joint_names.index(name)])
            for name in injected_mapping
        },
        "initial_loss": result.initial_loss,
        "final_loss": result.final_loss,
        "loss_reduction_ratio": result.final_loss / max(result.initial_loss, 1e-12),
        "offset_mae_rad": offset_mae,
        "state_mae_before_rad": result.initial_state_mae,
        "state_mae_after_rad": result.final_state_mae,
        "pixel_rmse_before": pixel_before,
        "pixel_rmse_after": pixel_after,
        "tip_error_before_mm": tip_before,
        "tip_error_after_mm": tip_after,
        "recovery_success": bool(offset_mae < 0.02 and pixel_after < pixel_before * 0.25),
    }

    run = RunArtifacts.create(
        root=config.get("runs_root", "runs"),
        experiment=str(config.get("experiment", "rt0_piper_joint_offset")),
        run_id=run_id,
        config=config,
        assets=paths.as_assets(),
        target_provenance="synthetic_state",
        observation_backend=backend_name,
    )
    run.write_trace(result.trace)
    run.write_metrics(metrics)
    real_frames = {
        name: assets.load_frame(name, frame_index).rgb for name in camera_names
    }
    write_before_target_after(
        run.path / "images" / "before_target_after.png",
        real_frames,
        before_projection,
        projected_target,
        after_projection,
        metrics=metrics,
    )
    write_recovery_curves(
        run.path / "images" / "recovery_curves.png", result.trace, injected_mapping
    )
    if isinstance(backend, CudaGaussianBackend):
        write_native_before_target_after(
            run.path / "images" / "native_gaussian_before_target_after.png",
            real_frames,
            backend.render(measured_qpos),
            target,
            backend.render(result.corrected_qpos),
            mode=backend.observation_mode,
        )
    visualization = config.get("visualization", {})
    write_recovery_video(
        run.path / "videos" / "joint_offset_recovery.mp4",
        backend=projected_backend,
        measured_qpos=measured_qpos,
        target=projected_target,
        real_frames=real_frames,
        trace=result.trace,
        joint_names=fk.active_joint_names,
        frame_count=int(visualization.get("video_frames", 60)),
        fps=int(visualization.get("video_fps", 12)),
    )
    if isinstance(backend, CudaGaussianBackend):
        write_native_recovery_video(
            run.path / "videos" / "native_gaussian_recovery.mp4",
            backend=backend,
            measured_qpos=measured_qpos,
            target=target,
            real_frames=real_frames,
            trace=result.trace,
            joint_names=fk.active_joint_names,
            frame_count=int(visualization.get("video_frames", 60)),
            fps=int(visualization.get("video_fps", 12)),
        )
    return run.path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    run_path = execute_rt0(load_config(args.config), run_id=args.run_id)
    print(run_path)


if __name__ == "__main__":
    main()
