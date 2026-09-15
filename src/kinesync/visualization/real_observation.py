"""Review-ready 1080p diagnostics for real-observation state correction."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Mapping, Sequence

import cv2
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from kinesync.data.schema import RealObservation
from kinesync.observation.real_route_a import RealObservationTarget


BACKGROUND = (20, 24, 29)
TEXT = (245, 245, 245)
MUTED = (185, 200, 215)
TARGET = (40, 210, 120)
BEFORE = (245, 92, 72)
AFTER = (30, 180, 235)


def _u8(value: torch.Tensor) -> np.ndarray:
    return (value.detach().clamp(0.0, 1.0).cpu().numpy() * 255.0).astype(np.uint8)


def _mask_overlay(
    rgb: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int],
    *,
    opacity: float = 0.42,
) -> np.ndarray:
    canvas = rgb.copy()
    active = mask > 0.5
    colored = np.empty_like(canvas)
    colored[:] = color
    canvas[active] = (
        canvas[active].astype(np.float32) * (1.0 - opacity)
        + colored[active].astype(np.float32) * opacity
    ).astype(np.uint8)
    contours, _ = cv2.findContours(
        active.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(canvas, contours, -1, color, 2, cv2.LINE_AA)
    return canvas


def _panel(image: np.ndarray, label: str, *, width: int = 480, height: int = 360) -> np.ndarray:
    panel = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    cv2.rectangle(panel, (0, height - 44), (width, height), BACKGROUND, -1)
    cv2.putText(
        panel,
        label,
        (16, height - 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.66,
        TEXT,
        2,
        cv2.LINE_AA,
    )
    return panel


def _camera_row(
    camera: str,
    observation: RealObservation,
    target: RealObservationTarget,
    before_alpha: torch.Tensor,
    after_alpha: torch.Tensor,
) -> np.ndarray:
    height, width = target.mask.shape
    real = cv2.resize(observation.rgb, (width, height), interpolation=cv2.INTER_AREA)
    target_mask = target.mask.detach().cpu().numpy()
    before = before_alpha.detach().cpu().numpy()
    after = after_alpha.detach().cpu().numpy()
    residual = np.abs(after - target_mask)
    residual_rgb = cv2.cvtColor(
        cv2.applyColorMap((np.clip(residual, 0.0, 1.0) * 255).astype(np.uint8), cv2.COLORMAP_MAGMA),
        cv2.COLOR_BGR2RGB,
    )
    images = [
        _mask_overlay(real, target_mask, TARGET),
        _mask_overlay(real, before, BEFORE),
        _mask_overlay(real, after, AFTER),
        residual_rgb,
    ]
    labels = (
        f"{camera}: real target mask",
        f"{camera}: before correction",
        f"{camera}: after correction",
        f"{camera}: final |alpha-mask|",
    )
    return np.concatenate([_panel(image, label) for image, label in zip(images, labels)], axis=1)


def write_real_observation_comparison(
    path: Path,
    *,
    observations: Mapping[str, RealObservation],
    targets: Mapping[str, RealObservationTarget],
    before_buffers: Mapping[str, object],
    after_buffers: Mapping[str, object],
    metrics: Mapping[str, float | str | bool],
) -> None:
    rows = [
        _camera_row(
            camera,
            observations[camera],
            targets[camera],
            before_buffers[camera].alpha,
            after_buffers[camera].alpha,
        )
        for camera in observations
    ]
    body = np.concatenate(rows[:2], axis=0)
    canvas = np.full((1080, 1920, 3), BACKGROUND, dtype=np.uint8)
    y0 = 130 if len(rows) > 1 else 250
    canvas[y0 : y0 + body.shape[0]] = body
    cv2.putText(
        canvas,
        "KineSync-GS real-observation joint correction",
        (48, 76),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.35,
        TEXT,
        3,
        cv2.LINE_AA,
    )
    summary = (
        f"state error reduction {float(metrics['state_error_reduction']) * 100:.1f}%   |   "
        f"mask IoU {float(metrics['mean_mask_iou_before']):.3f} -> "
        f"{float(metrics['mean_mask_iou_after']):.3f}   |   "
        f"boundary F1 {float(metrics['mean_boundary_f1_before']):.3f} -> "
        f"{float(metrics['mean_boundary_f1_after']):.3f}"
    )
    cv2.putText(
        canvas, summary, (48, 1034), cv2.FONT_HERSHEY_SIMPLEX, 0.72, MUTED, 2, cv2.LINE_AA
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(
        str(path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 6]
    )


def write_real_observation_curves(
    path: Path,
    *,
    trace: Sequence[Mapping[str, float | int]],
    injected_offsets: Mapping[str, float],
) -> None:
    steps = np.asarray([row["step"] for row in trace])
    losses = np.asarray([row["loss"] for row in trace], dtype=np.float64)
    fig, axes = plt.subplots(1, 2, figsize=(19.2, 10.8), dpi=100)
    axes[0].plot(steps, losses, color="#2878B5", linewidth=3)
    axes[0].set_title("Real-observation objective", fontsize=20)
    axes[0].set_xlabel("Optimization step")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.25)
    palette = ["#E45756", "#2CA02C", "#17BECF", "#9467BD"]
    for color, (joint, injected) in zip(palette, injected_offsets.items()):
        estimate = -np.asarray(
            [row[f"correction_{joint}"] for row in trace], dtype=np.float64
        )
        axes[1].plot(steps, np.rad2deg(estimate), color=color, linewidth=3, label=joint)
        axes[1].axhline(np.rad2deg(injected), color=color, linestyle="--", linewidth=2)
    axes[1].set_title("Recovered measurement offsets", fontsize=20)
    axes[1].set_xlabel("Optimization step")
    axes[1].set_ylabel("Offset (degree)")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)
    fig.suptitle("RT1-O real RGB/mask correction", fontsize=24, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100, facecolor="white")
    plt.close(fig)


def _open_mp4_writer(path: Path, fps: int) -> tuple[cv2.VideoWriter, Path]:
    fallback = path.with_name(f".{path.stem}.fallback.mp4")
    writer = cv2.VideoWriter(
        str(fallback), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (1920, 1080)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Unable to open real-observation video writer: {fallback}")
    return writer, fallback


def _finalize_mp4(fallback: Path, path: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is not None:
        result = subprocess.run(
            [
                ffmpeg,
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(fallback),
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "23",
                "-pix_fmt",
                "yuv420p",
                str(path),
            ],
            check=False,
        )
        if result.returncode == 0:
            fallback.unlink()
            return
    fallback.replace(path)


def write_real_observation_video(
    path: Path,
    *,
    renderer,
    observations: Mapping[str, RealObservation],
    targets: Mapping[str, RealObservationTarget],
    measured_qpos: torch.Tensor,
    trace: Sequence[Mapping[str, float | int]],
    joint_names: Sequence[str],
    frame_count: int,
    fps: int,
) -> None:
    if frame_count <= 0 or fps <= 0:
        raise ValueError("frame_count and fps must be positive")
    path.parent.mkdir(parents=True, exist_ok=True)
    writer, fallback = _open_mp4_writer(path, fps)
    trace_indices = np.linspace(0, len(trace) - 1, frame_count).round().astype(int)
    try:
        for trace_index in trace_indices:
            row = trace[int(trace_index)]
            correction = torch.tensor(
                [row[f"correction_{joint}"] for joint in joint_names],
                dtype=measured_qpos.dtype,
                device=measured_qpos.device,
            )
            with torch.no_grad():
                buffers = renderer.render_buffers(measured_qpos + correction)
            camera_rows = []
            for camera, observation in list(observations.items())[:2]:
                target = targets[camera]
                height, width = target.mask.shape
                real = cv2.resize(observation.rgb, (width, height), interpolation=cv2.INTER_AREA)
                target_mask = target.mask.detach().cpu().numpy()
                current = buffers[camera].alpha.detach().cpu().numpy()
                residual = np.abs(current - target_mask)
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
                            _panel(_mask_overlay(real, target_mask, TARGET), f"{camera}: real target", width=640),
                            _panel(_mask_overlay(real, current, AFTER), f"{camera}: current GS state", width=640),
                            _panel(residual_rgb, f"{camera}: |alpha-mask|", width=640),
                        ),
                        axis=1,
                    )
                )
            body = np.concatenate(camera_rows, axis=0)
            canvas = np.full((1080, 1920, 3), BACKGROUND, dtype=np.uint8)
            y0 = 120 if len(camera_rows) > 1 else 290
            canvas[y0 : y0 + body.shape[0]] = body
            cv2.putText(
                canvas,
                "RT1-O real-observation correction",
                (48, 74),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.35,
                TEXT,
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                f"step {int(row['step']):04d}   loss {float(row['loss']):.6f}",
                (48, 930),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                MUTED,
                2,
                cv2.LINE_AA,
            )
            writer.write(cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    _finalize_mp4(fallback, path)
