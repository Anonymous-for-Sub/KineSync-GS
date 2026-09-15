"""1080p review media for offline Real-to-GS shadow synchronization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np


WIDTH = 1920
HEIGHT = 1080
BACKGROUND = (245, 246, 248)
TEXT = (34, 37, 42)
MUTED = (92, 98, 108)
REAL = (43, 121, 188)
DELAYED = (226, 126, 34)
CORRECTED = (53, 155, 93)


@dataclass(frozen=True)
class OfflineShadowVisualFrame:
    backend: str
    episode_id: str
    source_pair_id: str
    lag: int
    committed: bool
    delayed_qmae_rad: float
    corrected_qmae_rad: float
    real_rgb: np.ndarray
    delayed_gaussian_rgb: np.ndarray
    corrected_gaussian_rgb: np.ndarray


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
        canvas,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def _fit_rgb(image: np.ndarray, *, width: int, height: int) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected HWC RGB image, got {image.shape}")
    source_height, source_width = image.shape[:2]
    scale = min(width / source_width, height / source_height)
    resized_width = max(1, int(round(source_width * scale)))
    resized_height = max(1, int(round(source_height * scale)))
    resized = cv2.resize(
        cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
        (resized_width, resized_height),
        interpolation=cv2.INTER_AREA,
    )
    panel = np.full((height, width, 3), 255, dtype=np.uint8)
    x = (width - resized_width) // 2
    y = (height - resized_height) // 2
    panel[y : y + resized_height, x : x + resized_width] = resized
    return panel


def compose_offline_shadow_frame(
    contexts: Sequence[OfflineShadowVisualFrame],
) -> np.ndarray:
    if not contexts or len(contexts) > 2:
        raise ValueError("A review frame requires one or two visual backends")
    canvas = np.full((HEIGHT, WIDTH, 3), BACKGROUND, dtype=np.uint8)
    _put_text(
        canvas,
        "RT8 Paired Real-to-GS Offline Shadow Replay",
        (42, 55),
        scale=1.25,
        thickness=3,
    )
    _put_text(
        canvas,
        "Recorded real observations; no live robot commands",
        (44, 91),
        scale=0.68,
        color=MUTED,
        thickness=2,
    )
    row_height = 440
    first_y = 122 if len(contexts) == 2 else 330
    column_x = (22, 656, 1290)
    panel_width = 608
    image_height = 340
    labels = (
        ("Real target", REAL),
        ("Delayed GS state", DELAYED),
        ("KineSync guarded GS", CORRECTED),
    )
    for row_index, context in enumerate(contexts):
        y = first_y + row_index * row_height
        _put_text(
            canvas,
            f"{context.backend} | episode {context.episode_id} | lag {context.lag:+d}",
            (28, y + 30),
            scale=0.72,
            thickness=2,
        )
        images = (
            context.real_rgb,
            context.delayed_gaussian_rgb,
            context.corrected_gaussian_rgb,
        )
        for x, image, (label, color) in zip(column_x, images, labels):
            cv2.rectangle(canvas, (x, y + 48), (x + panel_width, y + 48 + image_height), color, 3)
            canvas[y + 51 : y + 48 + image_height - 2, x + 3 : x + panel_width - 2] = _fit_rgb(
                image, width=panel_width - 5, height=image_height - 5
            )
            _put_text(canvas, label, (x + 10, y + 422), scale=0.68, color=color)
        status = "committed" if context.committed else "rollback"
        _put_text(
            canvas,
            f"{status}: qMAE {context.delayed_qmae_rad:.4f} -> {context.corrected_qmae_rad:.4f} rad",
            (1280, y + 30),
            scale=0.58,
            color=CORRECTED if context.committed else MUTED,
            thickness=2,
        )
    return canvas


def write_offline_shadow_image(
    path: str | Path, contexts: Sequence[OfflineShadowVisualFrame]
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), compose_offline_shadow_frame(contexts)):
        raise RuntimeError(f"Unable to write image: {destination}")
    return destination


def write_offline_shadow_video(
    path: str | Path,
    sequence: Sequence[Sequence[OfflineShadowVisualFrame]],
    *,
    fps: float,
) -> Path:
    if not sequence or fps <= 0:
        raise ValueError("Video sequence and fps must be positive")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(destination), cv2.VideoWriter_fourcc(*"mp4v"), fps, (WIDTH, HEIGHT)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Unable to open video writer: {destination}")
    try:
        for contexts in sequence:
            writer.write(compose_offline_shadow_frame(contexts))
    finally:
        writer.release()
    return destination


__all__ = [
    "OfflineShadowVisualFrame",
    "compose_offline_shadow_frame",
    "write_offline_shadow_image",
    "write_offline_shadow_video",
]
