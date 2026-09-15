"""Review-ready diagnostics for multi-frame physical factorization."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import cv2
import numpy as np
import torch

from kinesync.data.schema import RealObservation
from kinesync.factorization.backend import FactorFrame

from .real_observation import (
    AFTER,
    BACKGROUND,
    BEFORE,
    MUTED,
    TARGET,
    TEXT,
    _finalize_mp4,
    _mask_overlay,
    _open_mp4_writer,
    _panel,
)


def _device_factors(
    renderer,
    joint_zero: torch.Tensor,
    camera_twists: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    reference = renderer.local_xyz
    return (
        joint_zero.to(reference),
        {name: value.to(reference) for name, value in camera_twists.items()},
    )


def _corrected_qpos(frame: FactorFrame, joint_zero: torch.Tensor) -> torch.Tensor:
    padding = torch.zeros(
        frame.qpos.numel() - joint_zero.numel(),
        dtype=joint_zero.dtype,
        device=joint_zero.device,
    )
    return frame.qpos.to(joint_zero) + torch.cat((joint_zero, padding))


def _render_pair(
    renderer,
    frame: FactorFrame,
    joint_zero: torch.Tensor,
    camera_twists: Mapping[str, torch.Tensor],
):
    zero_joint = torch.zeros_like(joint_zero)
    zero_twists = {name: torch.zeros_like(value) for name, value in camera_twists.items()}
    with torch.no_grad():
        nominal = renderer.render_alpha_with_camera_deltas(
            _corrected_qpos(frame, zero_joint), zero_twists
        )
        factorized = renderer.render_alpha_with_camera_deltas(
            _corrected_qpos(frame, joint_zero), camera_twists
        )
    return nominal, factorized


def _factorization_row(
    camera: str,
    observation: RealObservation,
    frame: FactorFrame,
    nominal: torch.Tensor,
    factorized: torch.Tensor,
) -> np.ndarray:
    target = frame.targets[camera].mask.detach().cpu().numpy()
    height, width = target.shape
    real = cv2.resize(observation.rgb, (width, height), interpolation=cv2.INTER_AREA)
    nominal_np = nominal.detach().cpu().numpy()
    factorized_np = factorized.detach().cpu().numpy()
    residual = np.abs(factorized_np - target)
    residual_rgb = cv2.cvtColor(
        cv2.applyColorMap(
            (np.clip(residual, 0.0, 1.0) * 255).astype(np.uint8),
            cv2.COLORMAP_MAGMA,
        ),
        cv2.COLOR_BGR2RGB,
    )
    panels = (
        _mask_overlay(real, target, TARGET),
        _mask_overlay(real, nominal_np, BEFORE),
        _mask_overlay(real, factorized_np, AFTER),
        residual_rgb,
    )
    labels = (
        f"{camera}: real target",
        f"{camera}: nominal GS",
        f"{camera}: factorized GS",
        f"{camera}: final residual",
    )
    return np.concatenate(
        [_panel(image, label) for image, label in zip(panels, labels)], axis=1
    )


def write_factorization_comparison(
    path: Path,
    *,
    renderer,
    frame: FactorFrame,
    observations: Mapping[str, RealObservation],
    joint_zero: torch.Tensor,
    camera_twists: Mapping[str, torch.Tensor],
    metrics: Mapping[str, float],
) -> None:
    joint_zero, camera_twists = _device_factors(
        renderer, joint_zero, camera_twists
    )
    nominal, factorized = _render_pair(
        renderer, frame, joint_zero, camera_twists
    )
    rows = [
        _factorization_row(
            camera,
            observations[camera],
            frame,
            nominal[camera].values,
            factorized[camera].values,
        )
        for camera in list(observations)[:2]
    ]
    body = np.concatenate(rows, axis=0)
    canvas = np.full((1080, 1920, 3), BACKGROUND, dtype=np.uint8)
    y0 = 130 if len(rows) > 1 else 310
    canvas[y0 : y0 + body.shape[0]] = body
    cv2.putText(
        canvas,
        "KineSync-GS multi-frame physical factorization",
        (48, 76),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.25,
        TEXT,
        3,
        cv2.LINE_AA,
    )
    summary = (
        f"train mask IoU {metrics['mean_mask_iou_before']:.3f} -> "
        f"{metrics['mean_mask_iou_after']:.3f}   |   boundary F1 "
        f"{metrics['mean_boundary_f1_before']:.3f} -> "
        f"{metrics['mean_boundary_f1_after']:.3f}"
    )
    cv2.putText(
        canvas,
        summary,
        (48, 1034),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        MUTED,
        2,
        cv2.LINE_AA,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(
        str(path),
        cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_PNG_COMPRESSION, 6],
    )


def write_factorization_video(
    path: Path,
    *,
    renderer,
    frames: Sequence[FactorFrame],
    observations: Mapping[str, Mapping[str, RealObservation]],
    joint_zero: torch.Tensor,
    camera_twists: Mapping[str, torch.Tensor],
    frame_count: int,
    fps: int,
) -> None:
    if not frames or frame_count <= 0 or fps <= 0:
        raise ValueError("frames, frame_count, and fps must be positive")
    joint_zero, camera_twists = _device_factors(
        renderer, joint_zero, camera_twists
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    writer, fallback = _open_mp4_writer(path, fps)
    try:
        for output_index in range(frame_count):
            frame = frames[output_index % len(frames)]
            fraction = (
                1.0 if frame_count == 1 else output_index / float(frame_count - 1)
            )
            current_joint = joint_zero * fraction
            current_cameras = {
                name: value * fraction for name, value in camera_twists.items()
            }
            with torch.no_grad():
                rendered = renderer.render_alpha_with_camera_deltas(
                    _corrected_qpos(frame, current_joint), current_cameras
                )
            camera_rows = []
            for camera in list(current_cameras)[:2]:
                target = frame.targets[camera].mask.detach().cpu().numpy()
                height, width = target.shape
                real = cv2.resize(
                    observations[frame.state_id][camera].rgb,
                    (width, height),
                    interpolation=cv2.INTER_AREA,
                )
                current = rendered[camera].values.detach().cpu().numpy()
                residual = np.abs(current - target)
                residual_rgb = cv2.cvtColor(
                    cv2.applyColorMap(
                        (np.clip(residual, 0.0, 1.0) * 255).astype(np.uint8),
                        cv2.COLORMAP_MAGMA,
                    ),
                    cv2.COLOR_BGR2RGB,
                )
                camera_rows.append(
                    np.concatenate(
                        (
                            _panel(
                                _mask_overlay(real, target, TARGET),
                                f"{camera}: real target",
                                width=640,
                            ),
                            _panel(
                                _mask_overlay(real, current, AFTER),
                                f"{camera}: factor scale {fraction:.2f}",
                                width=640,
                            ),
                            _panel(
                                residual_rgb,
                                f"{camera}: |alpha-mask|",
                                width=640,
                            ),
                        ),
                        axis=1,
                    )
                )
            body = np.concatenate(camera_rows, axis=0)
            canvas = np.full((1080, 1920, 3), BACKGROUND, dtype=np.uint8)
            y0 = 120 if len(camera_rows) > 1 else 300
            canvas[y0 : y0 + body.shape[0]] = body
            cv2.putText(
                canvas,
                "RT2-F shared joint/camera factorization",
                (48, 74),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.30,
                TEXT,
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                f"state {frame.state_id}   frozen factor scale {fraction:.2f}",
                (48, 930),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.86,
                MUTED,
                2,
                cv2.LINE_AA,
            )
            writer.write(cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    _finalize_mp4(fallback, path)


def _validation_row(
    camera: str,
    observation: RealObservation,
    target: torch.Tensor,
    before: torch.Tensor,
    after: torch.Tensor,
) -> np.ndarray:
    target_np = target.detach().cpu().numpy()
    height, width = target_np.shape
    real = cv2.resize(observation.rgb, (width, height), interpolation=cv2.INTER_AREA)
    before_np = before.detach().cpu().numpy()
    after_np = after.detach().cpu().numpy()
    residual = np.abs(after_np - target_np)
    residual_rgb = cv2.cvtColor(
        cv2.applyColorMap(
            (np.clip(residual, 0.0, 1.0) * 255).astype(np.uint8),
            cv2.COLORMAP_MAGMA,
        ),
        cv2.COLOR_BGR2RGB,
    )
    panels = (
        _mask_overlay(real, target_np, TARGET),
        _mask_overlay(real, before_np, BEFORE),
        _mask_overlay(real, after_np, AFTER),
        residual_rgb,
    )
    labels = (
        f"{camera}: validation target",
        f"{camera}: injected state",
        f"{camera}: recovered state",
        f"{camera}: final residual",
    )
    return np.concatenate(
        [_panel(image, label) for image, label in zip(panels, labels)], axis=1
    )


def write_frozen_validation_comparison(
    path: Path,
    *,
    observations: Mapping[str, RealObservation],
    targets: Mapping[str, object],
    before: Mapping[str, object],
    after: Mapping[str, object],
    metrics: Mapping[str, float],
) -> None:
    rows = [
        _validation_row(
            camera,
            observations[camera],
            targets[camera].mask,
            before[camera].values,
            after[camera].values,
        )
        for camera in list(observations)[:2]
    ]
    body = np.concatenate(rows, axis=0)
    canvas = np.full((1080, 1920, 3), BACKGROUND, dtype=np.uint8)
    y0 = 130 if len(rows) > 1 else 310
    canvas[y0 : y0 + body.shape[0]] = body
    cv2.putText(
        canvas,
        "KineSync-GS frozen-factor validation",
        (48, 76),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.32,
        TEXT,
        3,
        cv2.LINE_AA,
    )
    summary = (
        f"state error reduction {metrics['state_error_reduction'] * 100:.1f}%   |   "
        f"mask IoU {metrics['mean_mask_iou_before']:.3f} -> "
        f"{metrics['mean_mask_iou_after']:.3f}   |   boundary F1 "
        f"{metrics['mean_boundary_f1_before']:.3f} -> "
        f"{metrics['mean_boundary_f1_after']:.3f}"
    )
    cv2.putText(
        canvas,
        summary,
        (48, 1034),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.70,
        MUTED,
        2,
        cv2.LINE_AA,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(
        str(path),
        cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_PNG_COMPRESSION, 6],
    )


def write_frozen_validation_video(
    path: Path,
    *,
    backend,
    observations: Mapping[str, RealObservation],
    targets: Mapping[str, object],
    measured_qpos: torch.Tensor,
    trace: Sequence[Mapping[str, object]],
    joint_names: Sequence[str],
    frame_count: int,
    fps: int,
) -> None:
    if not trace or frame_count <= 0 or fps <= 0:
        raise ValueError("trace, frame_count, and fps must be positive")
    path.parent.mkdir(parents=True, exist_ok=True)
    writer, fallback = _open_mp4_writer(path, fps)
    indices = np.linspace(0, len(trace) - 1, frame_count).round().astype(int)
    try:
        for output_index, trace_index in enumerate(indices):
            row = trace[int(trace_index)]
            correction = torch.tensor(
                [float(row[f"correction_{name}"]) for name in joint_names],
                dtype=measured_qpos.dtype,
                device=measured_qpos.device,
            )
            with torch.no_grad():
                rendered = backend.render(measured_qpos + correction)
            camera_rows = []
            for camera in list(observations)[:2]:
                target = targets[camera].mask.detach().cpu().numpy()
                height, width = target.shape
                real = cv2.resize(
                    observations[camera].rgb,
                    (width, height),
                    interpolation=cv2.INTER_AREA,
                )
                current = rendered[camera].values.detach().cpu().numpy()
                residual = np.abs(current - target)
                residual_rgb = cv2.cvtColor(
                    cv2.applyColorMap(
                        (np.clip(residual, 0.0, 1.0) * 255).astype(np.uint8),
                        cv2.COLORMAP_MAGMA,
                    ),
                    cv2.COLOR_BGR2RGB,
                )
                camera_rows.append(
                    np.concatenate(
                        (
                            _panel(
                                _mask_overlay(real, target, TARGET),
                                f"{camera}: validation target",
                                width=640,
                            ),
                            _panel(
                                _mask_overlay(real, current, AFTER),
                                f"{camera}: recovered GS",
                                width=640,
                            ),
                            _panel(
                                residual_rgb,
                                f"{camera}: |alpha-mask|",
                                width=640,
                            ),
                        ),
                        axis=1,
                    )
                )
            body = np.concatenate(camera_rows, axis=0)
            canvas = np.full((1080, 1920, 3), BACKGROUND, dtype=np.uint8)
            y0 = 120 if len(camera_rows) > 1 else 300
            canvas[y0 : y0 + body.shape[0]] = body
            cv2.putText(
                canvas,
                "RT2-F frozen-factor state recovery",
                (48, 74),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.30,
                TEXT,
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                f"frame {output_index + 1:03d}/{frame_count:03d}   "
                f"loss {float(row['loss']):.6f}",
                (48, 930),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.86,
                MUTED,
                2,
                cv2.LINE_AA,
            )
            writer.write(cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    _finalize_mp4(fallback, path)


__all__ = [
    "write_factorization_comparison",
    "write_factorization_video",
    "write_frozen_validation_comparison",
    "write_frozen_validation_video",
]
