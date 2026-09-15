"""Review media for RT3 conditional commits and rollbacks."""

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


def _rendered_masks(backend, qpos: torch.Tensor) -> dict[str, np.ndarray]:
    with torch.no_grad():
        rendered = backend.render(qpos)
    return {
        name: value.values.detach().cpu().numpy() for name, value in rendered.items()
    }


def _comparison_row(context: Mapping[str, Any]) -> np.ndarray:
    camera = list(context["observations"])[0]
    observation = context["observations"][camera]
    target = context["targets"][camera].mask.detach().cpu().numpy()
    height, width = target.shape
    real = cv2.resize(observation.rgb, (width, height), interpolation=cv2.INTER_AREA)
    measured = _rendered_masks(context["backend"], context["measured"])[camera]
    candidate = _rendered_masks(context["backend"], context["candidate"])[camera]
    final = _rendered_masks(context["backend"], context["final"])[camera]
    accepted = bool(context["decision"].accepted)
    panels = (
        _mask_overlay(real, target, TARGET),
        _mask_overlay(real, measured, BEFORE),
        _mask_overlay(real, candidate, AFTER),
        _mask_overlay(real, final, AFTER if accepted else TARGET),
    )
    labels = (
        f"{camera}: real target",
        f"{camera}: measured state",
        f"{camera}: candidate update",
        "committed final" if accepted else "rolled-back final",
    )
    return np.concatenate(
        [_panel(image, label) for image, label in zip(panels, labels)], axis=1
    )


def write_guard_comparison(
    path: Path, *, contexts: Sequence[Mapping[str, Any]]
) -> None:
    if not contexts:
        raise ValueError("At least one guard media context is required")
    selected = list(contexts[:2])
    rows = [_comparison_row(context) for context in selected]
    body = np.concatenate(rows, axis=0)
    canvas = np.full((1080, 1920, 3), BACKGROUND, dtype=np.uint8)
    y0 = 130 if len(rows) > 1 else 310
    canvas[y0 : y0 + body.shape[0]] = body
    cv2.putText(
        canvas,
        "RT3-C evidence-gated physical updates",
        (48, 76),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.30,
        TEXT,
        3,
        cv2.LINE_AA,
    )
    summaries = []
    for context in selected:
        evidence = context["evidence"]
        summaries.append(
            f"{context['state_id']}: {context['decision'].reason}, "
            f"gain {evidence.visual_gain_ratio:.3f}, grad {evidence.gradient_cosine:.3f}"
        )
    cv2.putText(
        canvas,
        "   |   ".join(summaries),
        (48, 1034),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.66,
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


def _video_canvas(context: Mapping[str, Any], current: torch.Tensor) -> np.ndarray:
    current_masks = _rendered_masks(context["backend"], current)
    final_masks = _rendered_masks(context["backend"], context["final"])
    camera_rows = []
    for camera in list(context["observations"])[:2]:
        observation = context["observations"][camera]
        target = context["targets"][camera].mask.detach().cpu().numpy()
        height, width = target.shape
        real = cv2.resize(observation.rgb, (width, height), interpolation=cv2.INTER_AREA)
        camera_rows.append(
            np.concatenate(
                (
                    _panel(
                        _mask_overlay(real, target, TARGET),
                        f"{camera}: real target",
                        width=640,
                    ),
                    _panel(
                        _mask_overlay(real, current_masks[camera], BEFORE),
                        f"{camera}: candidate path",
                        width=640,
                    ),
                    _panel(
                        _mask_overlay(
                            real,
                            final_masks[camera],
                            AFTER if context["decision"].accepted else TARGET,
                        ),
                        (
                            f"{camera}: committed"
                            if context["decision"].accepted
                            else f"{camera}: rolled back"
                        ),
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
    evidence = context["evidence"]
    cv2.putText(
        canvas,
        "RT3-C candidate, evidence, and final decision",
        (48, 74),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.22,
        TEXT,
        3,
        cv2.LINE_AA,
    )
    detail = (
        f"{context['state_id']}  decision={context['decision'].reason}  "
        f"gain={evidence.visual_gain_ratio:.3f}  grad={evidence.gradient_cosine:.3f}  "
        f"state={evidence.correction_cosine:.3f}  disagree="
        f"{evidence.relative_correction_disagreement:.3f}"
    )
    cv2.putText(
        canvas,
        detail,
        (48, 1034),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.66,
        MUTED,
        2,
        cv2.LINE_AA,
    )
    return canvas


def write_guard_video(
    path: Path,
    *,
    contexts: Sequence[Mapping[str, Any]],
    frame_count: int,
    fps: int,
) -> None:
    if not contexts or frame_count <= 0 or fps <= 0:
        raise ValueError("contexts, frame_count, and fps must be positive")
    selected = list(contexts[:2])
    path.parent.mkdir(parents=True, exist_ok=True)
    writer, fallback = _open_mp4_writer(path, fps)
    try:
        for index in range(frame_count):
            context_index = min(index * len(selected) // frame_count, len(selected) - 1)
            context = selected[context_index]
            segment_start = context_index * frame_count / len(selected)
            segment_end = (context_index + 1) * frame_count / len(selected)
            progress = (index - segment_start) / max(segment_end - segment_start - 1, 1)
            progress = float(np.clip(progress, 0.0, 1.0))
            measured = context["measured"]
            candidate = context["candidate"]
            final = context["final"]
            if progress <= 0.7:
                current = measured + (candidate - measured) * (progress / 0.7)
            else:
                current = candidate + (final - candidate) * ((progress - 0.7) / 0.3)
            canvas = _video_canvas(context, current)
            writer.write(cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    _finalize_mp4(fallback, path)


__all__ = ["write_guard_comparison", "write_guard_video"]
