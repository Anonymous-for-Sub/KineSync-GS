"""Publication-ready media for RT6 synchronized-state evidence.

Each context has one provenance authority under ``frame``: an actual
``SynchronizedStateFrame`` whose integrity is checked before rendering.  The
remaining context fields are paired ``head`` and ``extra`` observations and a
backend with ``render(qpos)``.  IDs, qpos vectors, component decisions and
outcome text are derived only from the frame.  Renderer outputs may expose
``mask``, ``alpha`` or ``values`` and optionally ``rgb``.  When only a
mask/alpha is available, the real RGB image is retained as the background and
the state is shown with a colored overlay and contour.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import torch

from kinesync.sync.schema import SynchronizedStateFrame

from .real_observation import (
    AFTER,
    BACKGROUND,
    BEFORE,
    MUTED,
    TARGET,
    TEXT,
    _finalize_mp4,
    _open_mp4_writer,
)


_WIDTH = 1920
_HEIGHT = 1080
_CAMERAS = ("head", "extra")
_PANEL_WIDTH = 480
_PANEL_HEIGHT = 450
_IMAGE_HEIGHT = 390
_HEADER_HEIGHT = 105
_FOOTER_Y = 1024
_STATE_COLORS = (TARGET, BEFORE, (245, 178, 48), AFTER)
_LEGACY_PROVENANCE_FIELDS = frozenset(
    {
        "accepted_names",
        "candidate_qpos",
        "case_id",
        "decisions",
        "measured_qpos",
        "outcome_labels",
        "state_id",
        "synchronized_qpos",
    }
)


@dataclass(frozen=True)
class SynchronizationPresentation:
    """Semantic labels for a synchronization comparison and diagnostic video."""

    comparison_title: str
    comparison_subtitle: str
    candidate_label: str
    committed_label: str
    video_title: str
    video_subtitle: str
    stack_video_contexts: bool = False


RT6_SYNCHRONIZATION_PRESENTATION = SynchronizationPresentation(
    comparison_title="KineSync-GS RT6 synchronized state evidence",
    comparison_subtitle=(
        "Each context shows paired head/extra cameras; colors identify target, "
        "measured, candidate, and committed state."
    ),
    candidate_label="RT2 CANDIDATE",
    committed_label="RT6 SYNCHRONIZED",
    video_title="KineSync-GS RT6 synchronization diagnostic",
    video_subtitle="Diagnostic linear interpolation, not temporal execution footage",
)


def _value(item: Any, name: str) -> Any:
    if isinstance(item, Mapping):
        return item.get(name)
    return getattr(item, name, None)


def _array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _finite_qpos(value: Any, name: str) -> np.ndarray:
    array = _array(value)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite nonempty qpos vector")
    return array


def _rgb(value: Any, name: str) -> np.ndarray:
    array = _array(value)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"{name} rgb must have shape (H, W, 3)")
    if array.dtype == np.uint8:
        return array.copy()
    if not np.isfinite(array).all():
        raise ValueError(f"{name} rgb must be finite")
    if float(array.max()) <= 1.0:
        array = array * 255.0
    return np.clip(array, 0.0, 255.0).astype(np.uint8)


def _mask(value: Any, name: str, shape: tuple[int, int] | None = None) -> np.ndarray:
    array = _array(value)
    if array.ndim != 2:
        raise ValueError(f"{name} mask must have shape (H, W)")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} mask must be finite")
    if shape is not None and tuple(array.shape) != shape:
        array = cv2.resize(array.astype(np.float32), (shape[1], shape[0]), cv2.INTER_NEAREST)
    return np.clip(array.astype(np.float32), 0.0, 1.0)


def _frame(context: Mapping[str, Any]) -> SynchronizedStateFrame:
    frame = context.get("frame")
    if not isinstance(frame, SynchronizedStateFrame):
        raise ValueError("synchronization context requires a SynchronizedStateFrame")
    return frame


def _context_label(context: Mapping[str, Any]) -> str:
    frame = _frame(context)
    return f"{frame.case_id} / {frame.state_id}"


def _validate_context(context: Mapping[str, Any]) -> None:
    if not isinstance(context, Mapping):
        raise ValueError("synchronization context must be a mapping")
    forbidden = sorted(set(context).intersection(_LEGACY_PROVENANCE_FIELDS))
    if forbidden:
        raise ValueError(
            "legacy provenance fields are forbidden; use frame only: "
            + ", ".join(forbidden)
        )
    frame = _frame(context)
    frame.validate_integrity()
    observations = context.get("observations")
    if not isinstance(observations, Mapping) or not set(_CAMERAS).issubset(observations):
        raise ValueError("synchronization context requires two cameras: head and extra")
    for camera in _CAMERAS:
        observation = observations[camera]
        rgb = _rgb(_value(observation, "rgb"), f"{camera} observation")
        mask = _mask(_value(observation, "mask"), f"{camera} observation")
        if tuple(rgb.shape[:2]) != tuple(mask.shape):
            raise ValueError(f"{camera} observation rgb and mask dimensions must match")
    measured = _finite_qpos(frame.measured_qpos, "frame.measured_qpos")
    candidate = _finite_qpos(frame.candidate_qpos, "frame.candidate_qpos")
    synchronized = _finite_qpos(frame.synchronized_qpos, "frame.synchronized_qpos")
    if candidate.shape != measured.shape or synchronized.shape != measured.shape:
        raise ValueError("frame measured, candidate, and synchronized qpos must match")
    backend = context.get("backend")
    if backend is None or not callable(getattr(backend, "render", None)):
        raise ValueError("synchronization context requires a backend.render(qpos)")
    if not frame.decisions:
        raise ValueError("frame must contain per-joint decisions")


def _select_contexts(contexts: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    if not contexts:
        raise ValueError("at least one synchronization context is required")
    selected = list(contexts)
    for context in selected:
        _validate_context(context)
    fingerprints = [_frame(context).fingerprint for context in selected]
    if len(set(fingerprints)) != len(fingerprints):
        raise ValueError("synchronization contexts require unique frame fingerprints")
    selected.sort(
        key=lambda context: (
            _frame(context).case_id,
            _frame(context).state_id,
            _frame(context).fingerprint,
        )
    )
    return selected[:2]


def _render_views(context: Mapping[str, Any], qpos: Any) -> Mapping[str, Any]:
    with torch.no_grad():
        rendered = context["backend"].render(qpos)
    if not isinstance(rendered, Mapping) or not set(_CAMERAS).issubset(rendered):
        raise ValueError("backend.render(qpos) must return head and extra views")
    return rendered


def _render_mask(rendered: Any, camera: str, shape: tuple[int, int]) -> np.ndarray:
    value = None
    for key in ("mask", "alpha", "values"):
        value = _value(rendered, key)
        if value is not None:
            break
    if value is None:
        raise ValueError(f"backend render output for {camera} needs mask, alpha, or values")
    return _mask(value, f"{camera} rendered", shape)


def _fit_image(image: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def _overlay(rgb: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    image = rgb.copy()
    active = mask >= 0.5
    fill = np.empty_like(image)
    fill[:] = color
    image[active] = (
        0.58 * image[active].astype(np.float32) + 0.42 * fill[active].astype(np.float32)
    ).astype(np.uint8)
    contours, _ = cv2.findContours(
        active.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(image, contours, -1, color, 3, cv2.LINE_AA)
    return image


def _fitted_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    max_width: int,
    color: tuple[int, int, int],
    font_scale: float,
    thickness: int = 2,
) -> None:
    text = str(text)
    scale = font_scale
    while scale >= 0.38:
        size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        if size[0] <= max_width:
            cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)
            return
        scale -= 0.04
    ellipsis = "..."
    while text and cv2.getTextSize(text + ellipsis, cv2.FONT_HERSHEY_SIMPLEX, 0.38, thickness)[0][0] > max_width:
        text = text[:-1]
    cv2.putText(image, text + ellipsis, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, thickness, cv2.LINE_AA)


def _camera_state_image(
    observation: Any,
    rendered: Any,
    color: tuple[int, int, int],
    width: int,
    height: int,
) -> np.ndarray:
    rgb = _rgb(_value(observation, "rgb"), "observation")
    target_mask = _mask(_value(observation, "mask"), "observation")
    rendered_mask = _render_mask(rendered, "camera", target_mask.shape)
    image = _overlay(rgb, rendered_mask, color)
    return _fit_image(image, width, height)


def _camera_target_image(observation: Any, width: int, height: int) -> np.ndarray:
    rgb = _rgb(_value(observation, "rgb"), "observation")
    target_mask = _mask(_value(observation, "mask"), "observation")
    return _fit_image(_overlay(rgb, target_mask, TARGET), width, height)


def _panel(
    context: Mapping[str, Any],
    rendered: Mapping[str, Any] | None,
    label: str,
    color: tuple[int, int, int],
    prefix: str,
) -> np.ndarray:
    observations = context["observations"]
    camera_height = _IMAGE_HEIGHT // 2
    camera_images = []
    for camera in _CAMERAS:
        if rendered is None:
            image = _camera_target_image(observations[camera], _PANEL_WIDTH, camera_height)
        else:
            image = _camera_state_image(
                observations[camera], rendered[camera], color, _PANEL_WIDTH, camera_height
            )
        _fitted_text(image, camera.upper(), (12, 27), 110, TEXT, 0.62, 2)
        camera_images.append(image)
    panel = np.full((_PANEL_HEIGHT, _PANEL_WIDTH, 3), BACKGROUND, dtype=np.uint8)
    panel[: _IMAGE_HEIGHT] = np.concatenate(camera_images, axis=0)
    cv2.rectangle(panel, (0, _IMAGE_HEIGHT), (_PANEL_WIDTH, _PANEL_HEIGHT), BACKGROUND, -1)
    _fitted_text(panel, f"{prefix} {label}", (14, _PANEL_HEIGHT - 17), _PANEL_WIDTH - 28, TEXT, 0.67, 2)
    return panel


def _qpos_interpolate(first: Any, second: Any, progress: float) -> Any:
    if hasattr(first, "detach"):
        return first + (second - first) * float(progress)
    return np.asarray(first) + (np.asarray(second) - np.asarray(first)) * float(progress)


def _accepted_names(frame: SynchronizedStateFrame) -> str:
    return ",".join(frame.accepted_joint_names) if frame.accepted_joint_names else "none"


def _presentation(
    presentation: SynchronizationPresentation | None,
) -> SynchronizationPresentation:
    resolved = RT6_SYNCHRONIZATION_PRESENTATION if presentation is None else presentation
    if not isinstance(resolved, SynchronizationPresentation):
        raise TypeError("presentation must be a SynchronizationPresentation")
    if any(
        not isinstance(value, str) or not value.strip()
        for value in (
            resolved.comparison_title,
            resolved.comparison_subtitle,
            resolved.candidate_label,
            resolved.committed_label,
            resolved.video_title,
            resolved.video_subtitle,
        )
    ):
        raise ValueError("presentation labels must be nonempty strings")
    if not isinstance(resolved.stack_video_contexts, bool):
        raise TypeError("presentation stack_video_contexts must be a bool")
    return resolved


def _outcome_text(frame: SynchronizedStateFrame) -> str:
    rejected = [
        f"{decision.joint_name}:{decision.reason}"
        for decision in frame.decisions
        if not decision.accepted
    ]
    rejected_text = ",".join(rejected) if rejected else "none"
    return (
        f"guard={frame.guard_reason} | accepted={_accepted_names(frame)} | "
        f"rejected={rejected_text}"
    )


def _canvas_for_context(
    context: Mapping[str, Any],
    *,
    candidate_qpos: Any,
    synchronized_qpos: Any,
    candidate_label: str,
    synchronized_label: str,
    prefix: str,
) -> np.ndarray:
    frame = _frame(context)
    measured = _render_views(context, frame.measured_qpos)
    candidate = _render_views(context, candidate_qpos)
    synchronized = _render_views(context, synchronized_qpos)
    panels = [
        _panel(context, None, "REAL TARGET", TARGET, prefix),
        _panel(context, measured, "MEASURED", BEFORE, prefix),
        _panel(context, candidate, candidate_label, _STATE_COLORS[2], prefix),
        _panel(context, synchronized, synchronized_label, AFTER, prefix),
    ]
    return np.concatenate(panels, axis=1)


def _base_canvas(title: str, subtitle: str) -> np.ndarray:
    canvas = np.full((_HEIGHT, _WIDTH, 3), BACKGROUND, dtype=np.uint8)
    _fitted_text(canvas, title, (42, 42), 900, TEXT, 1.05, 2)
    _fitted_text(canvas, subtitle, (42, 82), 1800, MUTED, 0.60, 2)
    return canvas


def _footer(canvas: np.ndarray, contexts: Sequence[Mapping[str, Any]]) -> None:
    parts = []
    for index, context in enumerate(contexts):
        frame = _frame(context)
        parts.append(
            f"{chr(65 + index)} {_context_label(context)} | {_outcome_text(frame)}"
        )
    for index, part in enumerate(parts):
        _fitted_text(canvas, part, (42, _FOOTER_Y + index * 31), 1836, MUTED, 0.47, 2)


def write_synchronization_comparison(
    path: Path,
    contexts: Sequence[Mapping[str, Any]],
    *,
    presentation: SynchronizationPresentation | None = None,
) -> None:
    """Write a deterministic 1920x1080 comparison for up to two contexts."""

    selected = _select_contexts(contexts)
    presentation = _presentation(presentation)
    canvas = _base_canvas(
        presentation.comparison_title,
        presentation.comparison_subtitle,
    )
    for index, context in enumerate(selected):
        frame = _frame(context)
        row = _canvas_for_context(
            context,
            candidate_qpos=frame.candidate_qpos,
            synchronized_qpos=frame.synchronized_qpos,
            candidate_label=presentation.candidate_label,
            synchronized_label=presentation.committed_label,
            prefix=f"{chr(65 + index)}",
        )
        y0 = _HEADER_HEIGHT + index * _PANEL_HEIGHT
        canvas[y0 : y0 + _PANEL_HEIGHT] = row
    _footer(canvas, selected)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 6]):
        raise RuntimeError(f"unable to write synchronization comparison: {path}")


def write_synchronization_video(
    path: Path,
    contexts: Sequence[Mapping[str, Any]],
    frame_count: int,
    fps: int,
    *,
    presentation: SynchronizationPresentation | None = None,
) -> None:
    """Write diagnostic linear interpolation, not temporal execution footage.

    The 1920x1080 MP4 independently interpolates measured-to-candidate and
    measured-to-synchronized qpos solely to expose commit and rollback state.
    """

    if not isinstance(frame_count, int) or frame_count <= 0:
        raise ValueError("frame_count must be a positive integer")
    if not isinstance(fps, int) or fps <= 0:
        raise ValueError("fps must be a positive integer")
    selected = _select_contexts(contexts)
    presentation = _presentation(presentation)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer, fallback = _open_mp4_writer(path, fps)
    try:
        if presentation.stack_video_contexts:
            for frame_index in range(frame_count):
                progress = float(frame_index / max(frame_count - 1, 1))
                percent = int(round(progress * 100.0))
                canvas = _base_canvas(
                    presentation.video_title,
                    f"{presentation.video_subtitle} | A/B measured -> candidate and measured -> "
                    f"{presentation.committed_label.lower()}",
                )
                for context_index, context in enumerate(selected):
                    frame = _frame(context)
                    body = _canvas_for_context(
                        context,
                        candidate_qpos=_qpos_interpolate(
                            frame.measured_qpos, frame.candidate_qpos, progress
                        ),
                        synchronized_qpos=_qpos_interpolate(
                            frame.measured_qpos, frame.synchronized_qpos, progress
                        ),
                        candidate_label=f"{presentation.candidate_label} {percent}%",
                        synchronized_label=f"{presentation.committed_label} {percent}%",
                        prefix=chr(65 + context_index),
                    )
                    y0 = _HEADER_HEIGHT + context_index * _PANEL_HEIGHT
                    canvas[y0 : y0 + _PANEL_HEIGHT] = body
                _footer(canvas, selected)
                writer.write(cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
        else:
            for frame_index in range(frame_count):
                context_index = min(frame_index * len(selected) // frame_count, len(selected) - 1)
                context = selected[context_index]
                frame = _frame(context)
                start = context_index * frame_count / len(selected)
                end = (context_index + 1) * frame_count / len(selected)
                progress = float(np.clip((frame_index - start) / max(end - start - 1.0, 1.0), 0.0, 1.0))
                candidate_qpos = _qpos_interpolate(
                    frame.measured_qpos, frame.candidate_qpos, progress
                )
                synchronized_qpos = _qpos_interpolate(
                    frame.measured_qpos, frame.synchronized_qpos, progress
                )
                percent = int(round(progress * 100.0))
                body = _canvas_for_context(
                    context,
                    candidate_qpos=candidate_qpos,
                    synchronized_qpos=synchronized_qpos,
                    candidate_label=f"{presentation.candidate_label} {percent}%",
                    synchronized_label=f"{presentation.committed_label} {percent}%",
                    prefix=chr(65 + context_index),
                )
                canvas = _base_canvas(
                    presentation.video_title,
                    f"{presentation.video_subtitle} | measured -> candidate and measured -> "
                    f"{presentation.committed_label.lower()} | {_context_label(context)}",
                )
                canvas[_HEADER_HEIGHT : _HEADER_HEIGHT + _PANEL_HEIGHT] = body
                _footer(canvas, [context])
                writer.write(cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    _finalize_mp4(fallback, path)


__all__ = [
    "RT6_SYNCHRONIZATION_PRESENTATION",
    "SynchronizationPresentation",
    "write_synchronization_comparison",
    "write_synchronization_video",
]
