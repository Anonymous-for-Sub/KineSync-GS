"""Clear, flat visual evidence for real PiPER-D455 multitask trajectories."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Protocol, Sequence

import cv2
import numpy as np

from .paper_style import METHOD_COLORS


_HEAD_BGR = (39, 39, 39)
_WRIST_BGR = tuple(
    int(METHOD_COLORS["ours_full"][index : index + 2], 16) for index in (5, 3, 1)
)


class _Trajectory(Protocol):
    task: str
    instruction: str
    episode_id: str
    length: int

    def read_rgb(self, camera: str, indices: Sequence[int] | np.ndarray) -> np.ndarray: ...

    def policy_state(self) -> np.ndarray: ...


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _bgr(rgb: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("PiPER visual frame must be uint8 HWC RGB")
    return cv2.resize(
        cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), size, interpolation=cv2.INTER_LANCZOS4
    )


def _label(image: np.ndarray, text: str, color: tuple[int, int, int]) -> np.ndarray:
    result = image.copy()
    cv2.rectangle(result, (16, 14), (205, 58), (248, 248, 248), -1)
    cv2.rectangle(result, (16, 14), (205, 58), color, 3)
    cv2.putText(
        result,
        text,
        (30, 46),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.78,
        color,
        2,
        cv2.LINE_AA,
    )
    return result


def _phase_indices(length: int) -> tuple[int, int, int]:
    if length < 3:
        raise ValueError("Visual evidence requires at least three frames")
    return (0, length // 2, length - 1)


def _safe_prefix(task: str) -> str:
    if not task or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for character in task.lower()):
        raise ValueError("Task visual prefix contains unsupported characters")
    return f"D-REAL-{task.lower()}"


def _ensure_absent(paths: Sequence[Path]) -> None:
    existing = [path for path in paths if path.exists()]
    if existing:
        raise FileExistsError(existing[0])


def choose_representative_trajectory(
    trajectories: Sequence[_Trajectory],
) -> tuple[_Trajectory, dict[str, float]]:
    """Choose a reproducible demonstration with strong visual and state motion."""

    if not trajectories:
        raise ValueError("At least one trajectory is required")
    scores: dict[str, float] = {}
    for trajectory in trajectories:
        state = np.asarray(trajectory.policy_state(), dtype=np.float64)
        if state.shape != (trajectory.length, 7) or not np.isfinite(state).all():
            raise ValueError(f"Invalid policy state for episode {trajectory.episode_id}")
        endpoint_difference = 0.0
        for camera in ("head", "wrist"):
            endpoints = trajectory.read_rgb(camera, [0, trajectory.length - 1])
            first = cv2.resize(endpoints[0], (160, 120), interpolation=cv2.INTER_AREA)
            last = cv2.resize(endpoints[1], (160, 120), interpolation=cv2.INTER_AREA)
            endpoint_difference += float(
                np.mean(np.abs(first.astype(np.float32) - last.astype(np.float32)))
            )
        joint_travel = float(np.linalg.norm(np.max(state[:, :6], axis=0) - np.min(state[:, :6], axis=0)))
        gripper_travel = float(np.ptp(state[:, 6]))
        scores[str(trajectory.episode_id)] = (
            endpoint_difference
            + 10.0 * joint_travel
            + 5.0 * gripper_travel
            + 0.001 * trajectory.length
        )
    selected = max(
        trajectories,
        key=lambda trajectory: (scores[str(trajectory.episode_id)], str(trajectory.episode_id)),
    )
    return selected, scores


def export_task_evidence(
    trajectory: _Trajectory, output_dir: str | Path, *, fps: float = 10.0
) -> dict[str, object]:
    """Export one complete dual-view demonstration and its process frames."""

    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    prefix = _safe_prefix(trajectory.task)
    phases = _phase_indices(trajectory.length)
    phase_names = ("start", "middle", "end")
    frame_paths = [
        output / f"{prefix}-episode-{trajectory.episode_id}-{camera}-{phase}-1280x960.png"
        for camera in ("head", "wrist")
        for phase in phase_names
    ]
    sheet_path = output / f"{prefix}-episode-{trajectory.episode_id}-process-3x2-1920x960.png"
    video_path = output / f"{prefix}-episode-{trajectory.episode_id}-dual-camera-1920x720.mp4"
    _ensure_absent([*frame_paths, sheet_path, video_path])

    phase_images = {
        camera: trajectory.read_rgb(camera, phases)
        for camera in ("head", "wrist")
    }
    single_names = []
    path_index = 0
    for camera, color in (("head", _HEAD_BGR), ("wrist", _WRIST_BGR)):
        for phase_name, rgb in zip(phase_names, phase_images[camera], strict=True):
            path = frame_paths[path_index]
            path_index += 1
            frame = _label(_bgr(rgb, (1280, 960)), f"{camera} | {phase_name}", color)
            if not cv2.imwrite(str(path), frame):
                raise OSError(f"Unable to write {path}")
            single_names.append(path.name)

    sheet = np.empty((960, 1920, 3), dtype=np.uint8)
    for row, (camera, color) in enumerate((("head", _HEAD_BGR), ("wrist", _WRIST_BGR))):
        for column, (phase_name, rgb) in enumerate(
            zip(phase_names, phase_images[camera], strict=True)
        ):
            panel = _label(_bgr(rgb, (640, 480)), f"{camera} | {phase_name}", color)
            sheet[row * 480 : (row + 1) * 480, column * 640 : (column + 1) * 640] = panel
    if not cv2.imwrite(str(sheet_path), sheet):
        raise OSError(f"Unable to write {sheet_path}")

    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (1920, 720),
    )
    if not writer.isOpened():
        raise RuntimeError("OpenCV cannot initialize the MP4 writer")
    try:
        chunk_size = 32
        for start in range(0, trajectory.length, chunk_size):
            indices = np.arange(start, min(start + chunk_size, trajectory.length))
            head_frames = trajectory.read_rgb("head", indices)
            wrist_frames = trajectory.read_rgb("wrist", indices)
            for head, wrist in zip(head_frames, wrist_frames, strict=True):
                left = _label(_bgr(head, (960, 720)), "head", _HEAD_BGR)
                right = _label(_bgr(wrist, (960, 720)), "wrist", _WRIST_BGR)
                writer.write(np.concatenate((left, right), axis=1))
    finally:
        writer.release()

    capture = cv2.VideoCapture(str(video_path))
    try:
        decoded_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ok, _ = capture.read()
    finally:
        capture.release()
    if not ok or decoded_frames != trajectory.length or (width, height) != (1920, 720):
        raise RuntimeError("Exported task video failed decode validation")

    artifacts = [*frame_paths, sheet_path, video_path]
    return {
        "schema": "kinesync.piper_task_visuals.v1",
        "task": trajectory.task,
        "instruction": trajectory.instruction,
        "episode_id": trajectory.episode_id,
        "source_frame_count": trajectory.length,
        "fps": float(fps),
        "single_frames": single_names,
        "contact_sheet": sheet_path.name,
        "video": video_path.name,
        "video_resolution": [1920, 720],
        "contact_sheet_layout": "3x2",
        "source_semantic": "recorded_real_demonstration",
        "comparison_claim": False,
        "sha256": {path.name: _sha256(path) for path in artifacts},
    }


__all__ = ["choose_representative_trajectory", "export_task_evidence"]
