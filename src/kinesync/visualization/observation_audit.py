"""Review media for the RT5 matched observation audit."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np
import torch

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


def _mask(backend, qpos: torch.Tensor, camera: str) -> np.ndarray:
    with torch.no_grad():
        rendered = backend.render(qpos)
    return rendered[camera].values.detach().cpu().numpy()


def _canvas(context: Mapping[str, Any], current: torch.Tensor) -> np.ndarray:
    camera = "head"
    observation = context["observations"][camera]
    target = context["targets"][camera].mask.detach().cpu().numpy()
    height, width = target.shape
    rgb = cv2.resize(observation.rgb, (width, height), interpolation=cv2.INTER_AREA)
    backend = context["backend"]
    panels = [
        _panel(_mask_overlay(rgb, target, TARGET), "real target", width=384, height=480),
        _panel(
            _mask_overlay(rgb, _mask(backend, context["measured"], camera), BEFORE),
            "measured state",
            width=384,
            height=480,
        ),
        _panel(
            _mask_overlay(
                rgb, _mask(backend, context["finals"]["standard_120"], camera), BEFORE
            ),
            "standard 120x160",
            width=384,
            height=480,
        ),
        _panel(
            _mask_overlay(
                rgb, _mask(backend, context["finals"]["standard_240"], camera), BEFORE
            ),
            "standard 240x320",
            width=384,
            height=480,
        ),
        _panel(
            _mask_overlay(rgb, _mask(backend, current, camera), AFTER),
            "TSDF 240x320",
            width=384,
            height=480,
        ),
    ]
    canvas = np.full((1080, 1920, 3), BACKGROUND, dtype=np.uint8)
    canvas[210:690] = np.concatenate(panels, axis=1)
    cv2.putText(
        canvas,
        "RT5-O matched silhouette observability audit",
        (48, 78),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.30,
        TEXT,
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "Same physical factors and optimizer; only resolution and TSDF evidence change",
        (48, 130),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        MUTED,
        2,
        cv2.LINE_AA,
    )
    rows = context["rows"]
    detail = "  |  ".join(
        f"{name}: reduction {float(rows[name]['state_error_reduction']) * 100:.1f}%, "
        f"grad {float(rows[name]['initial_selected_gradient_norm']):.4f}"
        for name in ("standard_120", "standard_240", "tsdf_240")
    )
    cv2.putText(
        canvas,
        detail,
        (48, 1015),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        MUTED,
        2,
        cv2.LINE_AA,
    )
    return canvas


def write_observation_audit(path: Path, *, context: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas = _canvas(context, context["finals"]["tsdf_240"])
    cv2.imwrite(
        str(path),
        cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_PNG_COMPRESSION, 6],
    )


def write_observation_audit_video(
    path: Path,
    *,
    context: Mapping[str, Any],
    frame_count: int,
    fps: int,
) -> None:
    if frame_count <= 0 or fps <= 0:
        raise ValueError("frame_count and fps must be positive")
    path.parent.mkdir(parents=True, exist_ok=True)
    writer, fallback = _open_mp4_writer(path, fps)
    measured = context["measured"]
    final = context["finals"]["tsdf_240"]
    try:
        for index in range(frame_count):
            progress = index / max(frame_count - 1, 1)
            current = measured + (final - measured) * progress
            writer.write(cv2.cvtColor(_canvas(context, current), cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    _finalize_mp4(fallback, path)


__all__ = ["write_observation_audit", "write_observation_audit_video"]
