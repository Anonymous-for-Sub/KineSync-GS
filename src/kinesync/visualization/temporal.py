"""Review-ready media for temporal shared-bias correction."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

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


def _real_panel(context: Mapping[str, Any], state_id: str, camera: str) -> np.ndarray:
    observation = context["context_observations"][state_id][camera]
    target = context["context_targets"][state_id][camera].mask.detach().cpu().numpy()
    height, width = target.shape
    rgb = cv2.resize(observation.rgb, (width, height), interpolation=cv2.INTER_AREA)
    label = "center real target" if state_id == context["center_state_id"] else state_id
    return _panel(
        _mask_overlay(rgb, target, TARGET),
        label,
        width=384,
        height=260,
    )


def _state_panel(
    context: Mapping[str, Any],
    qpos: torch.Tensor,
    label: str,
    color: tuple[int, int, int],
    *,
    camera: str,
) -> np.ndarray:
    center = context["center_state_id"]
    observation = context["context_observations"][center][camera]
    target = context["context_targets"][center][camera].mask.detach().cpu().numpy()
    height, width = target.shape
    rgb = cv2.resize(observation.rgb, (width, height), interpolation=cv2.INTER_AREA)
    return _panel(
        _mask_overlay(rgb, _mask(context["backend"], qpos, camera), color),
        label,
        width=384,
        height=360,
    )


def _canvas(context: Mapping[str, Any], temporal_qpos: torch.Tensor) -> np.ndarray:
    camera = str(context.get("media_camera", "head"))
    states = list(context["context_state_ids"])
    history = np.concatenate(
        [_real_panel(context, state_id, camera) for state_id in states], axis=1
    )
    accepted = bool(context["decision"].accepted)
    center = context["center_state_id"]
    target_panel = _real_panel(context, center, camera)
    target_panel = cv2.resize(target_panel, (384, 360), interpolation=cv2.INTER_AREA)
    outcomes = np.concatenate(
        [
            target_panel,
            _state_panel(
                context, context["measured"], "measured center", BEFORE, camera=camera
            ),
            _state_panel(
                context,
                context["single"],
                "single-frame estimate",
                BEFORE,
                camera=camera,
            ),
            _state_panel(
                context, temporal_qpos, "five-frame estimate", AFTER, camera=camera
            ),
            _state_panel(
                context,
                context["final"],
                "guarded commit" if accepted else "guarded rollback",
                AFTER if accepted else TARGET,
                camera=camera,
            ),
        ],
        axis=1,
    )
    canvas = np.full((1080, 1920, 3), BACKGROUND, dtype=np.uint8)
    canvas[120:380] = history
    canvas[500:860] = outcomes
    cv2.putText(
        canvas,
        "RT4-W causal visual evidence and persistent joint correction",
        (48, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.18,
        TEXT,
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "Five preceding real observations share one measurement-bias estimate",
        (48, 426),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        MUTED,
        2,
        cv2.LINE_AA,
    )
    evidence = context["evidence"]
    detail = (
        f"{center}  decision={context['decision'].reason}  "
        f"single reduction={float(context['single_reduction']) * 100:.1f}%  "
        f"temporal reduction={float(context['temporal_reduction']) * 100:.1f}%  "
        f"gain={evidence.visual_gain_ratio:.3f}  grad={evidence.gradient_cosine:.3f}"
    )
    cv2.putText(
        canvas,
        detail,
        (48, 1018),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.64,
        MUTED,
        2,
        cv2.LINE_AA,
    )
    return canvas


def write_temporal_comparison(path: Path, *, context: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas = _canvas(context, context["temporal"])
    cv2.imwrite(
        str(path),
        cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_PNG_COMPRESSION, 6],
    )


def write_temporal_video(
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
    temporal = context["temporal"]
    final = context["final"]
    try:
        for index in range(frame_count):
            progress = index / max(frame_count - 1, 1)
            if progress <= 0.75:
                current = measured + (temporal - measured) * (progress / 0.75)
            else:
                current = temporal + (final - temporal) * ((progress - 0.75) / 0.25)
            writer.write(cv2.cvtColor(_canvas(context, current), cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    _finalize_mp4(fallback, path)


__all__ = ["write_temporal_comparison", "write_temporal_video"]
