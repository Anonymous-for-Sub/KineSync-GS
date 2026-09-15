"""Run RT1-O joint correction against real Route-A RGB and robot masks."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from kinesync.config import load_config
from kinesync.correction.joint_offset import JointOffsetRecovery
from kinesync.data.real_observation import RouteARealObservationDataset
from kinesync.data.route_a import RouteAAssets, RouteAPaths
from kinesync.geometry.urdf import TorchURDFKinematics
from kinesync.observation.cuda_gaussian import CudaGaussianBackend
from kinesync.observation.real_route_a import (
    RealObservationLossWeights,
    RealRouteAObservationBackend,
    prepare_real_targets,
)
from kinesync.runs.artifacts import RunArtifacts
from kinesync.visualization.real_observation import (
    write_real_observation_comparison,
    write_real_observation_curves,
    write_real_observation_video,
)


def _real_dataset(config: Mapping[str, Any]) -> RouteARealObservationDataset:
    source = config["real_observations"]
    return RouteARealObservationDataset(
        records=source["records"],
        image_roots=source["image_roots"],
        mask_roots=source["mask_roots"],
        max_pair_qpos_delta_rad=float(source.get("max_pair_qpos_delta_rad", 0.02)),
    )


def _anchor_indices(
    assets: RouteAAssets, setting: int | str, *, seed: int
) -> np.ndarray:
    anchors = assets.load_anchors()
    if str(setting).lower() == "all":
        return np.arange(len(anchors.xyz), dtype=np.int64)
    return assets.sample_anchor_indices(per_link=int(setting), seed=seed)


def _evaluate_trial(
    *,
    initial_selected_mae: float,
    final_selected_mae: float,
    applied_correction_mae: float,
    before_metrics: Mapping[str, float],
    after_metrics: Mapping[str, float],
    acceptance: Mapping[str, Any],
) -> dict[str, Any]:
    if initial_selected_mae <= 1e-12:
        correction_limit = float(
            acceptance.get("natural_correction_max_rad", np.deg2rad(0.5))
        )
        visual_tolerance = float(
            acceptance.get("natural_visual_drop_tolerance", 0.005)
        )
        success = bool(
            applied_correction_mae <= correction_limit
            and after_metrics["mean_mask_iou"]
            >= before_metrics["mean_mask_iou"] - visual_tolerance
            and after_metrics["mean_boundary_f1"]
            >= before_metrics["mean_boundary_f1"] - visual_tolerance
        )
        return {
            "trial_type": "zero_injection_control",
            "state_error_reduction": 0.0,
            "natural_state_correction_rad": applied_correction_mae,
            "recovery_success": success,
        }
    state_error_reduction = 1.0 - final_selected_mae / initial_selected_mae
    threshold = float(acceptance.get("state_error_reduction_min", 0.6))
    success = bool(
        state_error_reduction >= threshold
        and after_metrics["mean_mask_iou"] > before_metrics["mean_mask_iou"]
        and after_metrics["mean_boundary_f1"] > before_metrics["mean_boundary_f1"]
    )
    return {
        "trial_type": "controlled_offset",
        "state_error_reduction": state_error_reduction,
        "natural_state_correction_rad": None,
        "recovery_success": success,
    }


def _observability_prior_mask(
    visual_gradient: torch.Tensor,
    *,
    selected_indices: list[int],
    minimum_gradient: float,
    minimum_relative: float,
) -> torch.Tensor:
    if visual_gradient.ndim != 1:
        raise ValueError("visual_gradient must be one-dimensional")
    if minimum_gradient < 0 or not 0 <= minimum_relative <= 1:
        raise ValueError("Invalid observability thresholds")
    mask = torch.zeros_like(visual_gradient, dtype=torch.bool)
    if not selected_indices:
        return mask
    selected = visual_gradient[selected_indices].abs()
    maximum = float(selected.max())
    for index in selected_indices:
        absolute = float(visual_gradient[index].abs())
        relative = absolute / max(maximum, 1e-12)
        mask[index] = absolute >= minimum_gradient and relative >= minimum_relative
    return mask


def execute_rt1o(config: Mapping[str, Any], *, run_id: str | None = None) -> Path:
    device = torch.device(str(config.get("device", "cuda")))
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("RT1-O requires the native CUDA Gaussian rasterizer")
    seed = int(config.get("seed", 0))
    torch.manual_seed(seed)
    np.random.seed(seed)

    paths = RouteAPaths.from_mapping(config["assets"])
    assets = RouteAAssets(paths)
    real_dataset = _real_dataset(config)
    camera_names = list(config["data"]["cameras"])
    state_id = str(config["data"]["state_id"])
    observations = real_dataset.load_state(state_id, cameras=camera_names)
    if any(item.split != "validation" for item in observations.values()):
        raise ValueError("Reported RT1-O trials must use Route-A validation observations")

    fk = TorchURDFKinematics(paths.urdf)
    reference_qpos_np = np.mean(
        np.stack([item.qpos for item in observations.values()]), axis=0
    )
    reference_qpos = torch.as_tensor(
        reference_qpos_np, dtype=torch.float32, device=device
    )
    if reference_qpos.numel() != len(fk.active_joint_names):
        raise ValueError("Real Route-A state dimension does not match the PiPER URDF")

    injected_mapping = {
        str(name): float(value)
        for name, value in config["target"]["injected_joint_offsets_rad"].items()
    }
    unknown = set(injected_mapping) - set(fk.active_joint_names)
    if unknown:
        raise ValueError(f"Unknown injected joints: {sorted(unknown)}")
    injected = torch.zeros_like(reference_qpos)
    for name, value in injected_mapping.items():
        injected[fk.active_joint_names.index(name)] = value
    measured_qpos = reference_qpos + injected
    selected_indices = [fk.active_joint_names.index(name) for name in injected_mapping]

    indices = _anchor_indices(
        assets, config["data"]["anchors_per_link"], seed=seed
    )
    height, width = map(int, config["observation"]["image_size"])
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
    targets = prepare_real_targets(
        observations, output_size=(height, width), device=device
    )
    loss_config = config["observation"]["loss_weights"]
    loss_weights = RealObservationLossWeights(
        mask_iou=float(loss_config["mask_iou"]),
        boundary=float(loss_config["boundary"]),
        rgb=float(loss_config["rgb"]),
        prior=float(loss_config["prior"]),
        prior_kind=str(loss_config.get("prior_kind", "quadratic")),
        prior_delta_rad=float(loss_config.get("prior_delta_rad", 0.02)),
    )
    prior_cauchy_mask = None
    initial_visual_gradient = torch.zeros_like(measured_qpos)
    if loss_weights.prior_kind == "adaptive":
        probe_weights = RealObservationLossWeights(
            mask_iou=loss_weights.mask_iou,
            boundary=loss_weights.boundary,
            rgb=loss_weights.rgb,
            prior=0.0,
            prior_kind="quadratic",
            prior_delta_rad=loss_weights.prior_delta_rad,
        )
        probe_backend = RealRouteAObservationBackend(
            renderer, loss_weights=probe_weights
        )
        probe_qpos = measured_qpos.detach().clone().requires_grad_()
        probe_loss, _ = probe_backend.loss(probe_qpos, targets)
        probe_loss.backward()
        initial_visual_gradient = probe_qpos.grad.detach().abs()
        gate = config["observation"].get("observability_gate", {})
        prior_cauchy_mask = _observability_prior_mask(
            initial_visual_gradient,
            selected_indices=selected_indices,
            minimum_gradient=float(gate.get("minimum_gradient", 0.05)),
            minimum_relative=float(gate.get("minimum_relative", 0.10)),
        )
    backend = RealRouteAObservationBackend(
        renderer,
        loss_weights=loss_weights,
        measured_qpos=measured_qpos,
        prior_cauchy_mask=prior_cauchy_mask,
    )
    before_metrics = backend.metrics(measured_qpos, targets)
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
    result = recovery.recover(
        measured_qpos, targets, reference_qpos=reference_qpos
    )
    after_metrics = backend.metrics(result.corrected_qpos, targets)

    initial_selected_mae = float(
        (measured_qpos[selected_indices] - reference_qpos[selected_indices]).abs().mean()
    )
    final_selected_mae = float(
        (result.corrected_qpos[selected_indices] - reference_qpos[selected_indices]).abs().mean()
    )
    offset_mae = float(
        (result.estimated_measurement_offset[selected_indices] - injected[selected_indices])
        .abs()
        .mean()
    )
    applied_correction_mae = float(
        result.applied_correction[selected_indices].abs().mean()
    )
    decision = _evaluate_trial(
        initial_selected_mae=initial_selected_mae,
        final_selected_mae=final_selected_mae,
        applied_correction_mae=applied_correction_mae,
        before_metrics=before_metrics,
        after_metrics=after_metrics,
        acceptance=config.get("acceptance", {}),
    )
    metrics: dict[str, Any] = {
        "target_provenance": "real_observation_controlled_offset",
        "seed": seed,
        "trial_type": decision["trial_type"],
        "state_id": state_id,
        "split": "validation",
        "camera_names": camera_names,
        "camera_pair_max_qpos_delta_rad": float(
            np.max(
                np.ptp(np.stack([item.qpos for item in observations.values()]), axis=0)
            )
        ),
        "anchor_count": int(len(indices)),
        "output_size": [height, width],
        "selected_joints": list(injected_mapping),
        "initial_visual_gradient_abs": {
            name: float(initial_visual_gradient[fk.active_joint_names.index(name)])
            for name in injected_mapping
        },
        "prior_assignment": {
            name: (
                "cauchy"
                if prior_cauchy_mask is not None
                and bool(prior_cauchy_mask[fk.active_joint_names.index(name)])
                else "quadratic"
            )
            for name in injected_mapping
        },
        "injected_measurement_offsets_rad": injected_mapping,
        "estimated_measurement_offsets_rad": {
            name: float(result.estimated_measurement_offset[fk.active_joint_names.index(name)])
            for name in injected_mapping
        },
        "initial_loss": result.initial_loss,
        "final_loss": result.final_loss,
        "loss_reduction_ratio": result.final_loss / max(result.initial_loss, 1e-12),
        "state_mae_before_rad": initial_selected_mae,
        "state_mae_after_rad": final_selected_mae,
        "state_error_reduction": decision["state_error_reduction"],
        "natural_state_correction_rad": decision["natural_state_correction_rad"],
        "offset_mae_rad": offset_mae,
        "mean_mask_iou_before": before_metrics["mean_mask_iou"],
        "mean_mask_iou_after": after_metrics["mean_mask_iou"],
        "mean_boundary_f1_before": before_metrics["mean_boundary_f1"],
        "mean_boundary_f1_after": after_metrics["mean_boundary_f1"],
        "mean_masked_rgb_loss_before": before_metrics["mean_masked_rgb_loss"],
        "mean_masked_rgb_loss_after": after_metrics["mean_masked_rgb_loss"],
        "per_camera_before": {
            key: value
            for key, value in before_metrics.items()
            if not key.startswith("mean_")
        },
        "per_camera_after": {
            key: value
            for key, value in after_metrics.items()
            if not key.startswith("mean_")
        },
        "recovery_success": decision["recovery_success"],
    }

    real_source = config["real_observations"]
    run_assets: dict[str, str | Path] = paths.as_assets()
    for camera in camera_names:
        run_assets[f"records_{camera}"] = real_source["records"][camera]
        run_assets[f"real_rgb_{camera}"] = (
            Path(real_source["image_roots"][camera])
            / "images"
            / f"{observations[camera].sample_id}.jpg"
        )
        run_assets[f"real_mask_{camera}"] = (
            Path(real_source["mask_roots"][camera])
            / f"{observations[camera].sample_id}.png"
        )
    run = RunArtifacts.create(
        root=config.get("runs_root", "runs"),
        experiment=str(config.get("experiment", "rt1o_piper_real_observation")),
        run_id=run_id,
        config=config,
        assets=run_assets,
        target_provenance="real_observation_controlled_offset",
        observation_backend="native_cuda_real_rgb_mask",
    )
    run.write_trace(result.trace)
    run.write_metrics(metrics)
    with torch.no_grad():
        before_buffers = renderer.render_buffers(measured_qpos)
        after_buffers = renderer.render_buffers(result.corrected_qpos)
    write_real_observation_comparison(
        run.path / "images" / "real_observation_before_after.png",
        observations=observations,
        targets=targets,
        before_buffers=before_buffers,
        after_buffers=after_buffers,
        metrics=metrics,
    )
    write_real_observation_curves(
        run.path / "images" / "real_observation_recovery_curves.png",
        trace=result.trace,
        injected_offsets=injected_mapping,
    )
    visualization = config.get("visualization", {})
    write_real_observation_video(
        run.path / "videos" / "real_observation_recovery.mp4",
        renderer=renderer,
        observations=observations,
        targets=targets,
        measured_qpos=measured_qpos,
        trace=result.trace,
        joint_names=fk.active_joint_names,
        frame_count=int(visualization.get("video_frames", 36)),
        fps=int(visualization.get("video_fps", 12)),
    )
    return run.path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id")
    arguments = parser.parse_args()
    print(execute_rt1o(load_config(arguments.config), run_id=arguments.run_id))


if __name__ == "__main__":
    main()
