"""Streaming visual and numeric artifacts for live hardware observations."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .schema import ObservationPacket


_COMPARISON_SIZE = (1920, 1080)
_HEAD_COLOR = (180, 119, 0)
_AUXILIARY_COLOR = (0, 114, 178)


class LiveArtifactRecorder:
    """Write full videos while retaining only selected lossless frames in memory."""

    def __init__(
        self,
        run_path: str | Path,
        *,
        expected_packets: int,
        fps: float,
        sample_count: int = 12,
    ) -> None:
        if expected_packets < 1 or fps <= 0.0 or sample_count < 1:
            raise ValueError("expected_packets, fps, and sample_count must be positive")
        self._root = Path(run_path)
        self._root.mkdir(parents=True, exist_ok=True)
        self._fps = float(fps)
        self._sample_slots = frozenset(
            int(value)
            for value in np.linspace(
                0, expected_packets - 1, min(expected_packets, sample_count), dtype=int
            )
        )
        self._writers: dict[str, cv2.VideoWriter] = {}
        self._shape: tuple[int, int, int] | None = None
        self._sample_head: list[np.ndarray] = []
        self._sample_auxiliary: list[np.ndarray] = []
        self._sample_sequence: list[int] = []
        self._qpos: list[np.ndarray] = []
        self._timestamps: list[tuple[int, int, int, int]] = []
        self._device_timestamps: list[tuple[float, float]] = []
        self._camera_frame_numbers: list[tuple[int, int]] = []
        self._state_sdk_timestamps: list[tuple[float, float]] = []
        self._state_sdk_feedback_frame_timestamps: list[
            tuple[float, float, float, float]
        ] = []
        self._closed = False

    def append(self, packet: ObservationPacket) -> None:
        if self._closed:
            raise RuntimeError("artifact recorder is closed")
        device_timestamps = (
            _required_metadata(packet.head_device_timestamp_ms, "head device timestamp"),
            _required_metadata(
                packet.auxiliary_device_timestamp_ms,
                "auxiliary device timestamp",
            ),
        )
        camera_frame_numbers = (
            int(_required_metadata(packet.head_frame_number, "head frame number")),
            int(
                _required_metadata(
                    packet.auxiliary_frame_number, "auxiliary frame number"
                )
            ),
        )
        state_sdk_timestamps = (
            _required_metadata(
                packet.state_sdk_joint_timestamp_s, "joint SDK timestamp"
            ),
            _required_metadata(
                packet.state_sdk_gripper_timestamp_s, "gripper SDK timestamp"
            ),
        )
        raw_feedback_frame_timestamps = packet.state_sdk_feedback_frame_timestamps_s
        if raw_feedback_frame_timestamps is None:
            raise ValueError("per-frame PiPER SDK timestamps are missing")
        state_sdk_feedback_frame_timestamps = tuple(
            float(value) for value in raw_feedback_frame_timestamps
        )
        if self._shape is None:
            self._shape = packet.head_rgb.shape
            if packet.auxiliary_rgb.shape != self._shape:
                raise ValueError("head and auxiliary frame shapes differ")
            self._open_writers(self._shape)
        elif packet.head_rgb.shape != self._shape or packet.auxiliary_rgb.shape != self._shape:
            raise ValueError("frame shape changed during live capture")

        head_bgr = cv2.cvtColor(packet.head_rgb, cv2.COLOR_RGB2BGR)
        auxiliary_bgr = cv2.cvtColor(packet.auxiliary_rgb, cv2.COLOR_RGB2BGR)
        comparison = _comparison_frame(head_bgr, auxiliary_bgr, packet)
        self._writers["head"].write(head_bgr)
        self._writers["auxiliary"].write(auxiliary_bgr)
        self._writers["comparison"].write(comparison)
        self._qpos.append(np.asarray(packet.qpos).copy())
        self._timestamps.append(
            (
                packet.arrival_monotonic_ns,
                packet.head_capture_ns,
                packet.auxiliary_capture_ns,
                packet.state_capture_ns,
            )
        )
        self._device_timestamps.append(device_timestamps)
        self._camera_frame_numbers.append(camera_frame_numbers)
        self._state_sdk_timestamps.append(state_sdk_timestamps)
        self._state_sdk_feedback_frame_timestamps.append(
            state_sdk_feedback_frame_timestamps
        )
        if packet.sequence_id in self._sample_slots:
            self._sample_head.append(packet.head_rgb.copy())
            self._sample_auxiliary.append(packet.auxiliary_rgb.copy())
            self._sample_sequence.append(packet.sequence_id)

    def close(self, *, allow_empty: bool = False) -> dict[str, Path]:
        if self._closed:
            return self._paths() if self._qpos else {}
        self._closed = True
        for writer in self._writers.values():
            writer.release()
        if not self._qpos:
            if allow_empty:
                return {}
            raise RuntimeError("cannot close artifact recorder with no packets")
        paths = self._paths()
        np.savez_compressed(
            paths["samples"],
            head_rgb=np.stack(self._sample_head),
            auxiliary_rgb=np.stack(self._sample_auxiliary),
            sampled_sequence_id=np.asarray(self._sample_sequence, dtype=np.int64),
            sequence_id=np.arange(len(self._qpos), dtype=np.int64),
            qpos=np.stack(self._qpos),
            timestamps_ns=np.asarray(self._timestamps, dtype=np.int64),
            device_timestamps_ms=np.asarray(self._device_timestamps, dtype=np.float64),
            camera_frame_numbers=np.asarray(self._camera_frame_numbers, dtype=np.int64),
            state_sdk_timestamps_s=np.asarray(self._state_sdk_timestamps, dtype=np.float64),
            state_sdk_feedback_frame_timestamps_s=np.asarray(
                self._state_sdk_feedback_frame_timestamps, dtype=np.float64
            ),
        )
        contact = _contact_sheet(
            self._sample_head,
            self._sample_auxiliary,
            self._sample_sequence,
        )
        if not cv2.imwrite(str(paths["contact_sheet"]), contact):
            raise RuntimeError("failed to write live contact sheet")
        for key in ("head_video", "auxiliary_video", "comparison_video"):
            if not paths[key].is_file() or paths[key].stat().st_size == 0:
                raise RuntimeError(f"failed to write {key}")
        return paths

    def _open_writers(self, shape: tuple[int, int, int]) -> None:
        height, width, _ = shape
        specifications = {
            "head": (self._root / "head.mp4", (width, height)),
            "auxiliary": (self._root / "auxiliary.mp4", (width, height)),
            "comparison": (self._root / "comparison.mp4", _COMPARISON_SIZE),
        }
        for name, (path, size) in specifications.items():
            writer = cv2.VideoWriter(
                str(path), cv2.VideoWriter_fourcc(*"mp4v"), self._fps, size
            )
            if not writer.isOpened():
                for opened in self._writers.values():
                    opened.release()
                raise RuntimeError(f"failed to open video writer: {path}")
            self._writers[name] = writer

    def _paths(self) -> dict[str, Path]:
        return {
            "head_video": self._root / "head.mp4",
            "auxiliary_video": self._root / "auxiliary.mp4",
            "comparison_video": self._root / "comparison.mp4",
            "contact_sheet": self._root / "contact_sheet.png",
            "samples": self._root / "frames.npz",
        }


def _comparison_frame(
    head_bgr: np.ndarray, auxiliary_bgr: np.ndarray, packet: ObservationPacket
) -> np.ndarray:
    canvas = np.full((_COMPARISON_SIZE[1], _COMPARISON_SIZE[0], 3), 22, dtype=np.uint8)
    for column, (image, label, color) in enumerate(
        (
            (head_bgr, "HEAD D455", _HEAD_COLOR),
            (auxiliary_bgr, "AUXILIARY D455", _AUXILIARY_COLOR),
        )
    ):
        x0 = column * 960
        fitted, x_margin, y_margin = _fit(image, 920, 820)
        x = x0 + 20 + x_margin
        y = 130 + y_margin
        canvas[y : y + fitted.shape[0], x : x + fitted.shape[1]] = fitted
        cv2.rectangle(
            canvas,
            (x - 2, y - 2),
            (x + fitted.shape[1] + 1, y + fitted.shape[0] + 1),
            color,
            3,
        )
        cv2.putText(
            canvas,
            label,
            (x0 + 34, 78),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            color,
            3,
            cv2.LINE_AA,
        )
    state = "q [rad/m]: " + " ".join(f"{value:+.3f}" for value in packet.qpos)
    cv2.putText(
        canvas,
        f"frame {packet.sequence_id:05d}   {state}",
        (36, 1040),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (235, 235, 235),
        2,
        cv2.LINE_AA,
    )
    return canvas


def _contact_sheet(
    heads: list[np.ndarray], auxiliaries: list[np.ndarray], sequence_ids: list[int]
) -> np.ndarray:
    cells = []
    for head, auxiliary, sequence_id in zip(heads, auxiliaries, sequence_ids):
        pair = np.concatenate(
            (cv2.cvtColor(head, cv2.COLOR_RGB2BGR), cv2.cvtColor(auxiliary, cv2.COLOR_RGB2BGR)),
            axis=1,
        )
        cell = np.full((360, 480, 3), 22, dtype=np.uint8)
        fitted, x_margin, y_margin = _fit(pair, 456, 300)
        x, y = 12 + x_margin, 44 + y_margin
        cell[y : y + fitted.shape[0], x : x + fitted.shape[1]] = fitted
        cv2.putText(
            cell,
            f"t={sequence_id}",
            (14, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (235, 235, 235),
            2,
            cv2.LINE_AA,
        )
        cells.append(cell)
    while len(cells) < 12:
        cells.append(cells[-1].copy())
    cells = cells[:12]
    return np.concatenate(
        [np.concatenate(cells[row * 4 : (row + 1) * 4], axis=1) for row in range(3)],
        axis=0,
    )


def _fit(image: np.ndarray, max_width: int, max_height: int) -> tuple[np.ndarray, int, int]:
    scale = min(max_width / image.shape[1], max_height / image.shape[0])
    size = (
        max(1, int(round(image.shape[1] * scale))),
        max(1, int(round(image.shape[0] * scale))),
    )
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(image, size, interpolation=interpolation)
    return resized, (max_width - size[0]) // 2, (max_height - size[1]) // 2


def _required_metadata(value: int | float | None, field: str) -> float:
    if value is None:
        raise ValueError(f"{field} is missing from live observation packet")
    return float(value)


__all__ = ["LiveArtifactRecorder"]
