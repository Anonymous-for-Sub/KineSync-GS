"""Immutable observation values and frozen limits for live shadowing."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral

import numpy as np


_REJECTION_REASONS = frozenset(
    {
        "timeout",
        "schema",
        "sequence",
        "session",
        "clock_domain",
        "timestamp",
        "camera_skew",
        "state_skew",
        "stale",
        "deadline",
    }
)


class ObservationRejection(RuntimeError):
    """Typed terminal rejection with one stable, auditable reason."""

    def __init__(self, reason: str):
        if reason not in _REJECTION_REASONS:
            raise ValueError(f"Unsupported observation rejection reason: {reason}")
        self.reason = reason
        super().__init__(reason)


def _is_nonnegative_integer(value: object) -> bool:
    return isinstance(value, Integral) and not isinstance(value, bool) and value >= 0


def _copy_rgb(value: object) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise ObservationRejection("schema")
    if value.dtype != np.uint8 or value.ndim != 3 or value.shape[2] != 3:
        raise ObservationRejection("schema")
    if any(dimension == 0 for dimension in value.shape):
        raise ObservationRejection("schema")
    return _read_only_copy(value)


def _copy_qpos(value: object) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise ObservationRejection("schema")
    is_real_numeric = np.issubdtype(value.dtype, np.integer) or np.issubdtype(
        value.dtype, np.floating
    )
    if value.shape != (8,) or not is_real_numeric:
        raise ObservationRejection("schema")
    if not np.isfinite(value).all():
        raise ObservationRejection("schema")
    return _read_only_copy(value)


def _read_only_copy(value: np.ndarray) -> np.ndarray:
    """Copy into immutable bytes so consumers cannot re-enable writes."""

    return np.frombuffer(value.tobytes(order="C"), dtype=value.dtype).reshape(value.shape)


@dataclass(frozen=True)
class ObservationPacket:
    """Validated, defensively copied input to the observation-only path."""

    session_id: str
    clock_domain: str
    sequence_id: int
    arrival_monotonic_ns: int
    head_rgb: np.ndarray
    head_capture_ns: int
    auxiliary_rgb: np.ndarray
    auxiliary_capture_ns: int
    auxiliary_role: str
    qpos: np.ndarray
    state_capture_ns: int
    source_id: str
    frame_id: str
    head_device_timestamp_ms: float | None = None
    auxiliary_device_timestamp_ms: float | None = None
    head_frame_number: int | None = None
    auxiliary_frame_number: int | None = None
    state_sdk_joint_timestamp_s: float | None = None
    state_sdk_gripper_timestamp_s: float | None = None
    state_sdk_feedback_frame_timestamps_s: tuple[float, float, float, float] | None = None

    def __post_init__(self) -> None:
        if not all(
            isinstance(value, str) and value
            for value in (self.session_id, self.clock_domain, self.source_id, self.frame_id)
        ):
            raise ObservationRejection("schema")
        if self.auxiliary_role not in {"wrist", "external"}:
            raise ObservationRejection("schema")
        if not all(
            _is_nonnegative_integer(value)
            for value in (
                self.sequence_id,
                self.arrival_monotonic_ns,
                self.head_capture_ns,
                self.auxiliary_capture_ns,
                self.state_capture_ns,
            )
        ):
            raise ObservationRejection("schema")
        optional_timestamps = (
            self.head_device_timestamp_ms,
            self.auxiliary_device_timestamp_ms,
            self.state_sdk_joint_timestamp_s,
            self.state_sdk_gripper_timestamp_s,
        )
        if any(
            value is not None
            and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0.0
            )
            for value in optional_timestamps
        ):
            raise ObservationRejection("schema")
        frame_timestamps = self.state_sdk_feedback_frame_timestamps_s
        if frame_timestamps is not None and (
            not isinstance(frame_timestamps, tuple)
            or len(frame_timestamps) != 4
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0.0
                for value in frame_timestamps
            )
        ):
            raise ObservationRejection("schema")
        if any(
            value is not None and not _is_nonnegative_integer(value)
            for value in (self.head_frame_number, self.auxiliary_frame_number)
        ):
            raise ObservationRejection("schema")
        object.__setattr__(self, "head_rgb", _copy_rgb(self.head_rgb))
        object.__setattr__(self, "auxiliary_rgb", _copy_rgb(self.auxiliary_rgb))
        object.__setattr__(self, "qpos", _copy_qpos(self.qpos))


@dataclass(frozen=True)
class LiveShadowPolicy:
    """Fixed safety limits for the observation-only preflight."""

    packet_timeout_ns: int = 250_000_000
    max_head_auxiliary_skew_ns: int = 35_000_000
    max_camera_state_skew_ns: int = 35_000_000

    def __post_init__(self) -> None:
        if not all(
            _is_nonnegative_integer(value) and value > 0
            for value in (
                self.packet_timeout_ns,
                self.max_head_auxiliary_skew_ns,
                self.max_camera_state_skew_ns,
            )
        ):
            raise ValueError("Live shadow policy limits must be positive integer nanoseconds")
