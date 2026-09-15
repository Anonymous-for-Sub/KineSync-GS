"""Fixed-layout 1080p evidence media for RT9 offline temporal telemetry."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from kinesync.temporal.telemetry import OffsetScore

from .real_observation import _finalize_mp4, _open_mp4_writer


WIDTH = 1920
HEIGHT = 1080
BACKGROUND = (245, 247, 250)
TEXT = (31, 37, 45)
MUTED = (88, 98, 112)
HEAD = (51, 116, 184)
WRIST = (216, 123, 42)
RAW = (211, 88, 76)
GUARDED = (48, 155, 98)
PROFILE = (89, 91, 195)
INJECTED = (210, 123, 42)
ESTIMATED = (89, 91, 195)
COMMITTED = (48, 155, 98)


@dataclass(frozen=True)
class TemporalTelemetryVisualFrame:
    """One recorded RGB telemetry state with matched raw and guarded errors."""

    trajectory_id: str
    condition_id: str
    frame_index: int
    frame_count: int
    head_rgb: np.ndarray
    wrist_rgb: np.ndarray
    score_profile: tuple[OffsetScore, ...]
    raw_joint_abs_error_rad: np.ndarray
    guarded_joint_abs_error_rad: np.ndarray
    injected_offset_ms: float
    estimated_offset_ms: float
    committed_offset_ms: float
    guarded_committed: bool

    def __post_init__(self) -> None:
        if not self.trajectory_id or not self.condition_id:
            raise ValueError("trajectory_id and condition_id must be nonempty")
        if self.frame_count < 1 or not 0 <= self.frame_index < self.frame_count:
            raise ValueError("frame_index must lie within frame_count")
        for name in ("head_rgb", "wrist_rgb"):
            image = np.asarray(getattr(self, name))
            if image.ndim != 3 or image.shape[2] != 3 or image.size == 0:
                raise ValueError(f"{name} must be a nonempty HWC RGB image")
        for name in ("raw_joint_abs_error_rad", "guarded_joint_abs_error_rad"):
            values = np.asarray(getattr(self, name), dtype=np.float64)
            if values.shape != (6,) or not np.isfinite(values).all() or np.any(values < 0):
                raise ValueError(f"{name} must contain six finite nonnegative values")
        if not self.score_profile:
            raise ValueError("score_profile must be nonempty")
        if not all(isinstance(score, OffsetScore) for score in self.score_profile):
            raise TypeError("score_profile entries must be OffsetScore values")
        if not np.isfinite(
            (self.injected_offset_ms, self.estimated_offset_ms, self.committed_offset_ms)
        ).all():
            raise ValueError("telemetry offsets must be finite")


def _put_text(
    canvas: np.ndarray,
    text: str,
    origin: tuple[int, int],
    *,
    scale: float,
    color: tuple[int, int, int] = TEXT,
    thickness: int = 2,
) -> None:
    cv2.putText(
        canvas, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA
    )


def _put_fitted(
    canvas: np.ndarray,
    text: str,
    origin: tuple[int, int],
    *,
    width: int,
    scale: float,
    color: tuple[int, int, int] = TEXT,
    thickness: int = 2,
) -> None:
    value = str(text)
    current = scale
    while current >= 0.36:
        extent = cv2.getTextSize(value, cv2.FONT_HERSHEY_SIMPLEX, current, thickness)[0][0]
        if extent <= width:
            _put_text(canvas, value, origin, scale=current, color=color, thickness=thickness)
            return
        current -= 0.04
    while value and cv2.getTextSize(value + "...", cv2.FONT_HERSHEY_SIMPLEX, 0.36, thickness)[0][0] > width:
        value = value[:-1]
    _put_text(canvas, value + "...", origin, scale=0.36, color=color, thickness=thickness)


def _image_panel(
    canvas: np.ndarray,
    image_rgb: np.ndarray,
    *,
    x: int,
    y: int,
    width: int,
    height: int,
    label: str,
    color: tuple[int, int, int],
) -> None:
    cv2.rectangle(canvas, (x, y), (x + width, y + height), color, 3)
    image = np.asarray(image_rgb)
    source_height, source_width = image.shape[:2]
    scale = min((width - 6) / source_width, (height - 6) / source_height)
    resized = cv2.resize(
        cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
        (max(1, round(source_width * scale)), max(1, round(source_height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    top = y + (height - resized.shape[0]) // 2
    left = x + (width - resized.shape[1]) // 2
    canvas[top : top + resized.shape[0], left : left + resized.shape[1]] = resized
    _put_text(canvas, label, (x + 9, y + height + 24), scale=0.58, color=color, thickness=2)


def _offset_x(offset_ms: float, minimum: float, maximum: float, x: int, width: int) -> int:
    if maximum <= minimum:
        return x + width // 2
    ratio = float(np.clip((offset_ms - minimum) / (maximum - minimum), 0.0, 1.0))
    return x + int(round(ratio * width))


def _score_profile(
    canvas: np.ndarray, context: TemporalTelemetryVisualFrame, *, x: int, y: int
) -> None:
    width, height = 550, 220
    cv2.rectangle(canvas, (x, y), (x + width, y + height), MUTED, 1)
    _put_text(
        canvas,
        "Full correlation score profile: relative correction",
        (x + 10, y + 22),
        scale=0.42,
        thickness=2,
    )
    left, right, top, bottom = x + 45, x + width - 12, y + 42, y + height - 48
    cv2.line(canvas, (left, bottom), (right, bottom), MUTED, 1, cv2.LINE_AA)
    cv2.line(canvas, (left, top), (left, bottom), MUTED, 1, cv2.LINE_AA)
    for correlation, label in ((-1.0, "-1"), (0.0, "0"), (1.0, "1")):
        plot_y = int(round(bottom - (correlation + 1.0) * 0.5 * (bottom - top)))
        cv2.line(canvas, (left, plot_y), (right, plot_y), (220, 224, 230), 1)
        _put_text(canvas, label, (x + 10, plot_y + 5), scale=0.38, color=MUTED, thickness=1)
    finite = [score for score in context.score_profile if score.correlation is not None]
    offsets = [score.offset_ms for score in context.score_profile]
    minimum, maximum = min(offsets), max(offsets)
    previous: tuple[int, int] | None = None
    for score in context.score_profile:
        if score.correlation is None:
            previous = None
            continue
        point = (
            _offset_x(score.offset_ms, minimum, maximum, left, right - left),
            int(round(bottom - (float(np.clip(score.correlation, -1.0, 1.0)) + 1.0) * 0.5 * (bottom - top))),
        )
        if previous is not None:
            cv2.line(canvas, previous, point, PROFILE, 2, cv2.LINE_AA)
        cv2.circle(canvas, point, 3, PROFILE, -1, cv2.LINE_AA)
        previous = point
    for offset, label, color in (
        (context.injected_offset_ms, "I", INJECTED),
        (context.estimated_offset_ms, "E", ESTIMATED),
        (context.committed_offset_ms, "C", COMMITTED if context.guarded_committed else MUTED),
    ):
        marker_x = _offset_x(offset, minimum, maximum, left, right - left)
        cv2.line(canvas, (marker_x, top), (marker_x, bottom), color, 2, cv2.LINE_AA)
        _put_text(canvas, label, (marker_x - 4, top - 5), scale=0.43, color=color, thickness=2)
    _put_text(canvas, f"{minimum:+.0f} ms", (left, bottom + 22), scale=0.38, color=MUTED, thickness=1)
    maximum_label = f"{maximum:+.0f} ms"
    label_width = cv2.getTextSize(maximum_label, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)[0][0]
    _put_text(canvas, maximum_label, (right - label_width, bottom + 22), scale=0.38, color=MUTED, thickness=1)
    _put_text(
        canvas,
        "relative correction; native delay removed",
        (x + 145, y + height - 8),
        scale=0.38,
        color=MUTED,
        thickness=1,
    )
    if not finite:
        _put_text(canvas, "No finite correlations", (left + 130, top + 65), scale=0.52, color=MUTED)


def _error_bars(
    canvas: np.ndarray, context: TemporalTelemetryVisualFrame, *, x: int, y: int
) -> None:
    width, height = 550, 220
    cv2.rectangle(canvas, (x, y), (x + width, y + height), MUTED, 1)
    _put_text(canvas, "Six-joint absolute error: raw vs guarded", (x + 10, y + 24), scale=0.54, thickness=2)
    raw = np.asarray(context.raw_joint_abs_error_rad, dtype=np.float64)
    guarded = np.asarray(context.guarded_joint_abs_error_rad, dtype=np.float64)
    maximum = max(float(raw.max()), float(guarded.max()), 1e-6)
    baseline, chart_top, chart_bottom = y + 45, y + 54, y + height - 35
    chart_height = chart_bottom - chart_top
    cv2.line(canvas, (x + 30, chart_bottom), (x + width - 12, chart_bottom), MUTED, 1)
    group_width = 78
    start = x + 48
    for joint_index, (raw_value, guarded_value) in enumerate(zip(raw, guarded, strict=True)):
        group_x = start + joint_index * group_width
        raw_height = int(round(chart_height * raw_value / maximum))
        guarded_height = int(round(chart_height * guarded_value / maximum))
        cv2.rectangle(canvas, (group_x, chart_bottom - raw_height), (group_x + 22, chart_bottom), RAW, -1)
        cv2.rectangle(canvas, (group_x + 28, chart_bottom - guarded_height), (group_x + 50, chart_bottom), GUARDED, -1)
        _put_text(canvas, f"J{joint_index + 1}", (group_x + 5, chart_bottom + 20), scale=0.39, color=MUTED, thickness=1)
    _put_text(canvas, "raw", (x + width - 132, baseline), scale=0.40, color=RAW, thickness=2)
    _put_text(canvas, "guarded", (x + width - 77, baseline), scale=0.40, color=GUARDED, thickness=2)
    _put_text(canvas, f"max {maximum:.4f} rad", (x + 10, chart_top + 13), scale=0.38, color=MUTED, thickness=1)


def _row(canvas: np.ndarray, context: TemporalTelemetryVisualFrame, *, y: int) -> None:
    _put_fitted(canvas, context.trajectory_id, (20, y + 25), width=330, scale=0.70, thickness=2)
    _put_fitted(canvas, context.condition_id, (20, y + 49), width=730, scale=0.40, color=MUTED, thickness=1)
    _put_text(canvas, f"recorded frame {context.frame_index + 1}/{context.frame_count}", (582, y + 25), scale=0.47, color=MUTED, thickness=1)
    _image_panel(canvas, context.head_rgb, x=20, y=y + 65, width=350, height=250, label="Recorded head RGB", color=HEAD)
    _image_panel(canvas, context.wrist_rgb, x=398, y=y + 65, width=350, height=250, label="Recorded wrist RGB", color=WRIST)
    _score_profile(canvas, context, x=776, y=y + 65)
    _error_bars(canvas, context, x=1350, y=y + 65)
    committed_label = "Committed" if context.guarded_committed else "Committed fallback raw"
    _put_text(canvas, f"Injected {context.injected_offset_ms:+.1f} ms", (20, y + 362), scale=0.50, color=INJECTED, thickness=2)
    _put_text(canvas, f"Estimated {context.estimated_offset_ms:+.1f} ms", (260, y + 362), scale=0.50, color=ESTIMATED, thickness=2)
    _put_text(canvas, f"{committed_label} {context.committed_offset_ms:+.1f} ms", (520, y + 362), scale=0.50, color=COMMITTED if context.guarded_committed else MUTED, thickness=2)


def compose_temporal_telemetry_frame(
    contexts: Sequence[TemporalTelemetryVisualFrame],
) -> np.ndarray:
    """Compose one fixed 1920x1080 BGR review frame from one or two trajectories."""

    if not contexts or len(contexts) > 2:
        raise ValueError("A telemetry review frame requires one or two contexts")
    canvas = np.full((HEIGHT, WIDTH, 3), BACKGROUND, dtype=np.uint8)
    _put_text(canvas, "RT9 Take-Pens temporal synchronization telemetry", (20, 43), scale=1.00, thickness=3)
    _put_text(canvas, "Offline recorded telemetry; no robot commands", (22, 78), scale=0.66, color=MUTED, thickness=2)
    row_y = (105,) if len(contexts) == 1 else (105, 580)
    for context, y in zip(contexts, row_y, strict=True):
        _row(canvas, context, y=y)
    return canvas


def write_temporal_telemetry_image(
    path: str | Path, contexts: Sequence[TemporalTelemetryVisualFrame]
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), compose_temporal_telemetry_frame(contexts)):
        raise RuntimeError(f"Unable to write image: {destination}")
    return destination


def write_temporal_telemetry_video(
    path: str | Path,
    sequence: Sequence[Sequence[TemporalTelemetryVisualFrame]],
    *,
    fps: int,
) -> Path:
    if not sequence or fps <= 0:
        raise ValueError("Video sequence and fps must be positive")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    writer, fallback = _open_mp4_writer(destination, int(fps))
    try:
        for contexts in sequence:
            writer.write(compose_temporal_telemetry_frame(contexts))
    finally:
        writer.release()
    _finalize_mp4(fallback, destination)
    return destination


__all__ = [
    "TemporalTelemetryVisualFrame",
    "compose_temporal_telemetry_frame",
    "write_temporal_telemetry_image",
    "write_temporal_telemetry_video",
]
