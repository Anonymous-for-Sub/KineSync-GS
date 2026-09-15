"""Clear, storage-conscious visual diagnostics for state recovery."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import cv2
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from kinesync.observation.base import RenderedObservation
from kinesync.observation.projected_centers import ProjectedCentersBackend


TARGET_COLOR = (40, 210, 120)
BEFORE_COLOR = (245, 92, 72)
AFTER_COLOR = (30, 180, 235)
TEXT_COLOR = (245, 245, 245)
BACKGROUND = (20, 24, 29)


def _draw_layers(
    rgb: np.ndarray,
    layers: Sequence[tuple[RenderedObservation, tuple[int, int, int], str]],
) -> np.ndarray:
    canvas = np.asarray(rgb, dtype=np.uint8).copy()
    for observation, color, _ in layers:
        points = observation.values.detach().cpu().numpy()
        valid = observation.valid.detach().cpu().numpy().astype(bool)
        for u, v in points[valid]:
            cv2.circle(canvas, (int(round(u)), int(round(v))), 2, color, -1, cv2.LINE_AA)
    legend_x = 18
    for _, color, label in layers:
        cv2.circle(canvas, (legend_x + 7, 26), 6, color, -1, cv2.LINE_AA)
        cv2.putText(
            canvas,
            label,
            (legend_x + 20, 33),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            TEXT_COLOR,
            2,
            cv2.LINE_AA,
        )
        legend_x += 175
    return canvas


def write_before_target_after(
    path: Path,
    real_frames: Mapping[str, np.ndarray],
    before: Mapping[str, RenderedObservation],
    target: Mapping[str, RenderedObservation],
    after: Mapping[str, RenderedObservation],
    *,
    metrics: Mapping[str, float],
) -> None:
    camera_names = list(real_frames)
    rows = []
    for camera_name in camera_names:
        rgb = real_frames[camera_name]
        panels = [
            _draw_layers(
                rgb,
                [(target[camera_name], TARGET_COLOR, "target"), (before[camera_name], BEFORE_COLOR, "measured")],
            ),
            _draw_layers(rgb, [(target[camera_name], TARGET_COLOR, "target")]),
            _draw_layers(
                rgb,
                [(target[camera_name], TARGET_COLOR, "target"), (after[camera_name], AFTER_COLOR, "corrected")],
            ),
        ]
        labels = ("Before correction", "Synthetic visual target", "After correction")
        for panel, label in zip(panels, labels):
            cv2.putText(
                panel,
                f"{camera_name}: {label}",
                (18, panel.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                TEXT_COLOR,
                2,
                cv2.LINE_AA,
            )
        rows.append(np.concatenate(panels, axis=1))
    body = np.concatenate(rows, axis=0)
    body = cv2.resize(body, (1920, 900 if len(rows) > 1 else 720), interpolation=cv2.INTER_AREA)
    canvas = np.full((1080, 1920, 3), BACKGROUND, dtype=np.uint8)
    y0 = 100 if len(rows) > 1 else 180
    canvas[y0 : y0 + body.shape[0]] = body
    cv2.putText(
        canvas,
        "KineSync-GS joint-state correction",
        (48, 62),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.35,
        TEXT_COLOR,
        3,
        cv2.LINE_AA,
    )
    summary = (
        f"offset MAE {metrics['offset_mae_rad']:.4f} rad   |   "
        f"pixel RMSE {metrics['pixel_rmse_before']:.2f} -> {metrics['pixel_rmse_after']:.2f} px"
    )
    cv2.putText(
        canvas,
        summary,
        (48, 1050),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.78,
        (190, 205, 220),
        2,
        cv2.LINE_AA,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 6])


def write_recovery_curves(
    path: Path,
    trace: Sequence[Mapping[str, float | int]],
    injected_offsets: Mapping[str, float],
) -> None:
    steps = np.asarray([row["step"] for row in trace], dtype=np.int64)
    loss = np.asarray([row["loss"] for row in trace], dtype=np.float64)
    fig, axes = plt.subplots(1, 2, figsize=(19.2, 10.8), dpi=100)
    axes[0].plot(steps, loss, color="#1f77b4", linewidth=3)
    axes[0].set_yscale("log")
    axes[0].set_title("Visual observation loss", fontsize=20)
    axes[0].set_xlabel("Optimization step")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.25)
    palette = ["#e45756", "#2ca02c", "#17becf", "#9467bd"]
    for color, (joint, injected) in zip(palette, injected_offsets.items()):
        estimate = -np.asarray(
            [row[f"correction_{joint}"] for row in trace], dtype=np.float64
        )
        axes[1].plot(steps, estimate, color=color, linewidth=3, label=f"{joint} estimate")
        axes[1].axhline(injected, color=color, linestyle="--", linewidth=2, alpha=0.75)
    axes[1].set_title("Recovered measurement offsets", fontsize=20)
    axes[1].set_xlabel("Optimization step")
    axes[1].set_ylabel("Offset (rad)")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)
    fig.suptitle("KineSync-GS RT0 recovery diagnostics", fontsize=24, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100, facecolor="white")
    plt.close(fig)


def _observation_to_rgb(observation: RenderedObservation) -> np.ndarray:
    values = observation.values.detach().clamp(0.0, 1.0).cpu().numpy()
    if values.ndim == 2:
        grayscale = (values * 255.0).astype(np.uint8)
        colored_bgr = cv2.applyColorMap(grayscale, cv2.COLORMAP_VIRIDIS)
        return cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)
    if values.ndim == 3 and values.shape[-1] == 3:
        return (values * 255.0).astype(np.uint8)
    raise ValueError(f"Unsupported rendered observation shape: {values.shape}")


def write_native_before_target_after(
    path: Path,
    real_frames: Mapping[str, np.ndarray],
    before: Mapping[str, RenderedObservation],
    target: Mapping[str, RenderedObservation],
    after: Mapping[str, RenderedObservation],
    *,
    mode: str,
) -> None:
    rows = []
    labels = ("Real context", "Measured state", "Target state", "Corrected state")
    for camera_name, real in real_frames.items():
        panels = [
            real,
            _observation_to_rgb(before[camera_name]),
            _observation_to_rgb(target[camera_name]),
            _observation_to_rgb(after[camera_name]),
        ]
        formatted = []
        for panel, label in zip(panels, labels):
            panel = cv2.resize(panel, (480, 360), interpolation=cv2.INTER_AREA)
            cv2.putText(
                panel,
                f"{camera_name}: {label}",
                (16, 336),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                TEXT_COLOR,
                2,
                cv2.LINE_AA,
            )
            formatted.append(panel)
        rows.append(np.concatenate(formatted, axis=1))
    body = np.concatenate(rows, axis=0)
    if body.shape[0] > 780:
        body = cv2.resize(body, (1920, 780), interpolation=cv2.INTER_AREA)
    canvas = np.full((1080, 1920, 3), BACKGROUND, dtype=np.uint8)
    y0 = 130
    canvas[y0 : y0 + body.shape[0]] = body
    cv2.putText(
        canvas,
        f"Native 3DGS {mode} state correction",
        (48, 76),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.35,
        TEXT_COLOR,
        3,
        cv2.LINE_AA,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 6])


def write_recovery_video(
    path: Path,
    *,
    backend: ProjectedCentersBackend,
    measured_qpos: torch.Tensor,
    target: Mapping[str, RenderedObservation],
    real_frames: Mapping[str, np.ndarray],
    trace: Sequence[Mapping[str, float | int]],
    joint_names: Sequence[str],
    frame_count: int,
    fps: int,
) -> None:
    if frame_count <= 0 or fps <= 0:
        raise ValueError("frame_count and fps must be positive")
    trace_indices = np.linspace(0, len(trace) - 1, frame_count).round().astype(int)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (1920, 1080)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Unable to open video writer: {path}")
    try:
        for trace_index in trace_indices:
            row = trace[int(trace_index)]
            correction = torch.tensor(
                [row[f"correction_{joint}"] for joint in joint_names],
                dtype=measured_qpos.dtype,
                device=measured_qpos.device,
            )
            current = backend.render(measured_qpos + correction)
            panes = []
            for camera_name, rgb in real_frames.items():
                pane = _draw_layers(
                    rgb,
                    [
                        (target[camera_name], TARGET_COLOR, "target"),
                        (current[camera_name], AFTER_COLOR, "current"),
                    ],
                )
                pane = cv2.resize(pane, (960, 720), interpolation=cv2.INTER_AREA)
                cv2.putText(
                    pane,
                    camera_name,
                    (24, 690),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.85,
                    TEXT_COLOR,
                    2,
                    cv2.LINE_AA,
                )
                panes.append(pane)
            if len(panes) == 1:
                panes.append(np.full_like(panes[0], BACKGROUND))
            body = np.concatenate(panes[:2], axis=1)
            canvas = np.full((1080, 1920, 3), BACKGROUND, dtype=np.uint8)
            canvas[120:840] = body
            cv2.putText(
                canvas,
                "KineSync-GS visual state recovery",
                (48, 72),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.35,
                TEXT_COLOR,
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                f"step {int(row['step']):04d}   loss {float(row['loss']):.7f}",
                (48, 915),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (190, 205, 220),
                2,
                cv2.LINE_AA,
            )
            offset_text = "   ".join(
                f"{joint}: {-float(row[f'correction_{joint}']):+.3f} rad"
                for joint in joint_names
                if abs(float(row[f"correction_{joint}"])) > 1e-8
            )
            cv2.putText(
                canvas,
                f"estimated measurement offset   {offset_text}",
                (48, 980),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.78,
                (190, 205, 220),
                2,
                cv2.LINE_AA,
            )
            writer.write(cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def write_native_recovery_video(
    path: Path,
    *,
    backend,
    measured_qpos: torch.Tensor,
    target: Mapping[str, RenderedObservation],
    real_frames: Mapping[str, np.ndarray],
    trace: Sequence[Mapping[str, float | int]],
    joint_names: Sequence[str],
    frame_count: int,
    fps: int,
) -> None:
    trace_indices = np.linspace(0, len(trace) - 1, frame_count).round().astype(int)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (1920, 1080)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Unable to open native video writer: {path}")
    try:
        for trace_index in trace_indices:
            row = trace[int(trace_index)]
            correction = torch.tensor(
                [row[f"correction_{joint}"] for joint in joint_names],
                dtype=measured_qpos.dtype,
                device=measured_qpos.device,
            )
            current = backend.render(measured_qpos + correction)
            rows = []
            for camera_name, real in real_frames.items():
                panels = [
                    (real, "real context"),
                    (_observation_to_rgb(current[camera_name]), "current native render"),
                    (_observation_to_rgb(target[camera_name]), "target native render"),
                ]
                formatted = []
                for panel, label in panels:
                    panel = cv2.resize(panel, (640, 360), interpolation=cv2.INTER_AREA)
                    cv2.putText(
                        panel,
                        f"{camera_name}: {label}",
                        (16, 336),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.64,
                        TEXT_COLOR,
                        2,
                        cv2.LINE_AA,
                    )
                    formatted.append(panel)
                rows.append(np.concatenate(formatted, axis=1))
            body = np.concatenate(rows[:2], axis=0)
            canvas = np.full((1080, 1920, 3), BACKGROUND, dtype=np.uint8)
            canvas[120:840] = body
            cv2.putText(
                canvas,
                f"Native 3DGS {backend.observation_mode} recovery",
                (48, 74),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.35,
                TEXT_COLOR,
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                f"step {int(row['step']):04d}   loss {float(row['loss']):.8f}",
                (48, 920),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.95,
                (190, 205, 220),
                2,
                cv2.LINE_AA,
            )
            estimates = "   ".join(
                f"{joint}: {-float(row[f'correction_{joint}']):+.4f} rad"
                for joint in joint_names
                if abs(float(row[f"correction_{joint}"])) > 1e-8
            )
            cv2.putText(
                canvas,
                f"estimated offset   {estimates}",
                (48, 988),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.78,
                (190, 205, 220),
                2,
                cv2.LINE_AA,
            )
            writer.write(cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
