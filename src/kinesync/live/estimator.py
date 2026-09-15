"""Causal, observation-only temporal diagnostics for RT10-P."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

from kinesync.temporal.telemetry import (
    CorruptedState,
    MotionFeatures,
    OffsetEstimate,
    TemporalGuard,
    TemporalGuardDecision,
    decide_temporal_guard,
    search_motion_offset,
)

from .schema import ObservationPacket


@dataclass(frozen=True)
class ShadowEstimate:
    """Guarded diagnostic output for one newest-25-interval causal window."""

    estimate: OffsetEstimate
    decision: TemporalGuardDecision
    window_size: int
    camera_midpoint_ns: np.ndarray
    visual_motion: np.ndarray


class CausalTemporalShadowEstimator:
    """Estimate RT9 temporal offsets using only already-observed packets."""

    def __init__(self, guard: TemporalGuard):
        self._guard = guard
        self._packets: deque[ObservationPacket] = deque(maxlen=26)

    def push(self, packet: ObservationPacket) -> ShadowEstimate | None:
        """Append one packet and return a guarded estimate after 25 intervals."""

        self._packets.append(packet)
        if len(self._packets) < 26:
            return None

        packets = tuple(self._packets)
        head_frames = np.stack([_grayscale(packet.head_rgb) for packet in packets])
        auxiliary_frames = np.stack(
            [_grayscale(packet.auxiliary_rgb) for packet in packets]
        )
        head_motion = _frame_mad(head_frames)
        auxiliary_motion = _frame_mad(auxiliary_frames)
        head_camera_ns = np.fromiter(
            (packet.head_capture_ns for packet in packets), dtype=np.int64, count=26
        )
        auxiliary_camera_ns = np.fromiter(
            (packet.auxiliary_capture_ns for packet in packets),
            dtype=np.int64,
            count=26,
        )
        state_ns = np.fromiter(
            (packet.state_capture_ns for packet in packets), dtype=np.int64, count=26
        )
        qpos = np.stack([packet.qpos for packet in packets])
        state_motion = np.linalg.norm(np.diff(qpos[:, :6], axis=0), axis=1)
        head_midpoint_ns = _interval_midpoints(head_camera_ns)
        auxiliary_midpoint_ns = _interval_midpoints(auxiliary_camera_ns)
        auxiliary_overlap = (head_midpoint_ns >= auxiliary_midpoint_ns[0]) & (
            head_midpoint_ns <= auxiliary_midpoint_ns[-1]
        )
        head_motion = _robust_normalize(head_motion)
        auxiliary_motion = _robust_normalize(auxiliary_motion)
        camera_midpoint_ns = head_midpoint_ns[auxiliary_overlap]
        aligned_auxiliary_motion = np.interp(
            camera_midpoint_ns, auxiliary_midpoint_ns, auxiliary_motion
        )
        visual_motion = (
            head_motion[auxiliary_overlap] + aligned_auxiliary_motion
        ) / 2.0
        features = MotionFeatures(
            camera_midpoint_ns=camera_midpoint_ns,
            visual_motion=visual_motion,
            state_motion=state_motion[auxiliary_overlap],
        )
        corrupted_state = CorruptedState(
            timestamp_ns=state_ns,
            qpos=qpos,
            retained_indices=np.arange(len(packets), dtype=np.int64),
        )
        estimate = search_motion_offset(
            features, corrupted_state, strict_overlap=True
        )
        return ShadowEstimate(
            estimate=estimate,
            decision=decide_temporal_guard(estimate, self._guard),
            window_size=len(visual_motion),
            camera_midpoint_ns=_read_only_copy(features.camera_midpoint_ns),
            visual_motion=_read_only_copy(features.visual_motion),
        )


def _grayscale(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(cv2.resize(rgb, (80, 60)), cv2.COLOR_RGB2GRAY).astype(
        np.float64
    )


def _frame_mad(frames: np.ndarray) -> np.ndarray:
    return np.mean(np.abs(np.diff(frames, axis=0)), axis=(1, 2))


def _robust_normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    if not np.isfinite(mad) or mad == 0.0:
        return np.zeros(values.shape, dtype=np.float64)
    return np.nan_to_num((values - median) / mad, nan=0.0, posinf=0.0, neginf=0.0)


def _interval_midpoints(timestamps: np.ndarray) -> np.ndarray:
    return timestamps[:-1] + (timestamps[1:] - timestamps[:-1]) // 2


def _read_only_copy(values: np.ndarray) -> np.ndarray:
    return np.frombuffer(values.tobytes(order="C"), dtype=values.dtype).reshape(values.shape)


__all__ = ["CausalTemporalShadowEstimator", "ShadowEstimate"]
