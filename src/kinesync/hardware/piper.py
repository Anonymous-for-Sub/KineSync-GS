"""Read-only PiPER feedback adapter."""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from numbers import Real
from typing import Any, Callable

import numpy as np


class PiperFeedbackError(RuntimeError):
    """Raised when a safe, initialized PiPER feedback sample is unavailable."""


class _CachedFeedbackError(PiperFeedbackError):
    pass


@dataclass(frozen=True)
class PiperFeedback:
    qpos: np.ndarray
    capture_monotonic_ns: int
    joint_hz: float
    gripper_hz: float
    sdk_joint_timestamp_s: float
    sdk_gripper_timestamp_s: float
    sdk_feedback_frame_timestamps_s: tuple[float, float, float, float]


class PiperFeedbackReader:
    """Expose only PiPER connection lifecycle and measured state feedback."""

    def __init__(
        self,
        *,
        can_name: str,
        interface_factory: Callable[..., Any] | None = None,
        monotonic_clock_ns: Callable[[], int] = time.monotonic_ns,
        wall_clock_ns: Callable[[], int] = time.time_ns,
        max_gripper_stroke_mm: float = 70.0,
        finger_travel_m: float = 0.05,
        max_clock_mapping_error_ns: int = 5_000_000,
    ) -> None:
        if not isinstance(can_name, str) or not can_name:
            raise ValueError("can_name must be a nonempty string")
        if not callable(monotonic_clock_ns):
            raise TypeError("monotonic_clock_ns must be callable")
        if not callable(wall_clock_ns):
            raise TypeError("wall_clock_ns must be callable")
        if max_gripper_stroke_mm <= 0.0 or finger_travel_m <= 0.0:
            raise ValueError("gripper dimensions must be positive")
        if (
            not isinstance(max_clock_mapping_error_ns, int)
            or isinstance(max_clock_mapping_error_ns, bool)
            or max_clock_mapping_error_ns < 0
        ):
            raise ValueError("max_clock_mapping_error_ns must be nonnegative")
        self._can_name = can_name
        self._interface_factory = interface_factory
        self._clock = monotonic_clock_ns
        self._wall_clock = wall_clock_ns
        self._max_gripper_stroke_mm = float(max_gripper_stroke_mm)
        self._finger_travel_m = float(finger_travel_m)
        self._max_clock_mapping_error_ns = max_clock_mapping_error_ns
        self._interface: Any | None = None
        self._last_feedback_key: tuple[float, float, float, float] | None = None

    def connect(self) -> None:
        if self._interface is not None:
            return
        factory = self._interface_factory or _load_interface_factory()
        interface = factory(
            self._can_name,
            judge_flag=True,
            can_auto_init=True,
            start_sdk_joint_limit=False,
            start_sdk_gripper_limit=False,
        )
        try:
            interface.ConnectPort(piper_init=False)
        except Exception:
            try:
                _stop_sdk_background_workers(interface)
            except PiperFeedbackError:
                pass
            raise
        self._interface = interface
        self._last_feedback_key = None

    def read(self) -> PiperFeedback:
        if self._interface is None:
            raise PiperFeedbackError("PiPER feedback reader is not connected")
        before_ns = _clock_ns(self._clock)
        wall_before_ns = _clock_ns(self._wall_clock)
        try:
            snapshot = self._interface.GetKineSyncSynchronizedFeedback()
        except AttributeError as error:
            raise PiperFeedbackError(
                "PiPER interface lacks per-frame synchronized feedback"
            ) from error
        wall_after_ns = _clock_ns(self._wall_clock)
        after_ns = _clock_ns(self._clock)
        if after_ns < before_ns:
            raise PiperFeedbackError("monotonic clock regressed during feedback read")
        if wall_after_ns < wall_before_ns:
            raise PiperFeedbackError("wall clock regressed during feedback read")
        monotonic_elapsed_ns = after_ns - before_ns
        wall_elapsed_ns = wall_after_ns - wall_before_ns
        if (
            abs(monotonic_elapsed_ns - wall_elapsed_ns)
            > self._max_clock_mapping_error_ns
        ):
            raise PiperFeedbackError("wall-to-monotonic clock mapping is unstable")

        joint_timestamp = _positive_float(
            snapshot["joint_timestamp_s"], "joint feedback is not initialized"
        )
        joint_hz = _positive_float(
            snapshot["joint_hz"], "joint feedback is not initialized"
        )
        gripper_timestamp = _positive_float(
            snapshot["gripper_timestamp_s"], "gripper feedback is not initialized"
        )
        gripper_hz = _positive_float(
            snapshot["gripper_hz"], "gripper feedback is not initialized"
        )
        raw_frame_timestamps = snapshot["feedback_frame_timestamps_s"]
        if not isinstance(raw_frame_timestamps, tuple) or len(raw_frame_timestamps) != 4:
            raise PiperFeedbackError("per-frame feedback timestamps are unavailable")
        feedback_key = tuple(
            _positive_float(value, "per-frame feedback timestamps are unavailable")
            for value in raw_frame_timestamps
        )
        if self._last_feedback_key is not None and (
            any(
                current <= previous
                for current, previous in zip(feedback_key, self._last_feedback_key)
            )
        ):
            raise _CachedFeedbackError("PiPER SDK returned cached or regressed feedback")
        joints = np.asarray(snapshot["joints_millidegree"], dtype=np.float64)
        if joints.shape != (6,):
            raise PiperFeedbackError("joint feedback does not contain six values")
        if not np.isfinite(joints).all():
            raise PiperFeedbackError("joint feedback contains nonfinite values")
        joints *= 1e-3 * math.pi / 180.0

        raw_stroke = float(snapshot["gripper_angle_micrometre"])
        if not math.isfinite(raw_stroke):
            raise PiperFeedbackError("gripper feedback contains a nonfinite value")
        stroke_mm = max(0.0, raw_stroke * 1e-3)
        finger = min(stroke_mm / self._max_gripper_stroke_mm, 1.0) * self._finger_travel_m
        qpos = np.concatenate((joints, np.asarray((finger, -finger), dtype=np.float64)))
        qpos = np.frombuffer(qpos.tobytes(), dtype=qpos.dtype)
        mapped_feedback_ns = tuple(
            _sdk_epoch_to_monotonic_ns(
                timestamp,
                monotonic_before_ns=before_ns,
                monotonic_after_ns=after_ns,
                wall_before_ns=wall_before_ns,
                wall_after_ns=wall_after_ns,
                future_tolerance_ns=self._max_clock_mapping_error_ns,
            )
            for timestamp in feedback_key
        )
        capture_ns = min(mapped_feedback_ns)
        feedback = PiperFeedback(
            qpos=qpos,
            capture_monotonic_ns=capture_ns,
            joint_hz=joint_hz,
            gripper_hz=gripper_hz,
            sdk_joint_timestamp_s=joint_timestamp,
            sdk_gripper_timestamp_s=gripper_timestamp,
            sdk_feedback_frame_timestamps_s=feedback_key,
        )
        self._last_feedback_key = feedback_key
        return feedback

    def read_before(
        self,
        deadline_monotonic_ns: int,
        *,
        poll_interval_s: float = 0.002,
        sleep: Callable[[float], None] = time.sleep,
    ) -> PiperFeedback | None:
        """Wait for a newly arrived joint-and-gripper snapshot before a deadline."""

        if not isinstance(deadline_monotonic_ns, int) or isinstance(
            deadline_monotonic_ns, bool
        ) or deadline_monotonic_ns < 0:
            raise ValueError("deadline_monotonic_ns must be a nonnegative integer")
        if poll_interval_s < 0.0:
            raise ValueError("poll_interval_s must be nonnegative")
        while _clock_ns(self._clock) <= deadline_monotonic_ns:
            try:
                return self.read()
            except _CachedFeedbackError:
                pass
            sleep(poll_interval_s)
        return None

    def wait_until_ready(
        self,
        *,
        timeout_s: float = 5.0,
        poll_interval_s: float = 0.05,
        sleep: Callable[[float], None] = time.sleep,
    ) -> PiperFeedback:
        if timeout_s <= 0.0 or poll_interval_s < 0.0:
            raise ValueError("feedback readiness timing must be nonnegative")
        deadline = time.monotonic() + timeout_s
        last_error: PiperFeedbackError | None = None
        while True:
            try:
                return self.read()
            except PiperFeedbackError as error:
                last_error = error
            if time.monotonic() >= deadline:
                raise PiperFeedbackError(
                    f"PiPER feedback was not ready within {timeout_s:.1f}s"
                ) from last_error
            sleep(poll_interval_s)

    def describe(self) -> dict[str, object]:
        return {
            "can_name": self._can_name,
            "connected": self._interface is not None,
            "max_gripper_stroke_mm": self._max_gripper_stroke_mm,
            "finger_travel_m": self._finger_travel_m,
            "read_only": True,
            "per_feedback_frame_timestamps": True,
            "feedback_ids": ["0x2A5", "0x2A6", "0x2A7", "0x2A8"],
        }

    def close(self) -> None:
        interface, self._interface = self._interface, None
        self._last_feedback_key = None
        if interface is not None:
            _stop_sdk_background_workers(interface)

    def __enter__(self) -> "PiperFeedbackReader":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


def _load_interface_factory() -> Callable[..., Any]:
    try:
        from piper_sdk import C_PiperInterface_V2
    except ImportError as error:
        raise PiperFeedbackError(
            "piper_sdk is unavailable; install the PiPER Python SDK on the hardware host"
        ) from error
    class TimestampedReadOnlyPiperInterface(C_PiperInterface_V2):
        _FEEDBACK_IDS = (0x2A5, 0x2A6, 0x2A7, 0x2A8)

        def __init__(self, *args, **kwargs):
            self._kinesync_feedback_lock = threading.Lock()
            self._kinesync_feedback_timestamps = {
                identifier: 0.0 for identifier in self._FEEDBACK_IDS
            }
            super().__init__(*args, **kwargs)

        def ParseCANFrame(self, rx_message):
            with self._kinesync_feedback_lock:
                result = super().ParseCANFrame(rx_message)
                if rx_message is not None:
                    identifier = int(rx_message.arbitration_id)
                    timestamp = float(rx_message.timestamp)
                    if (
                        identifier in self._kinesync_feedback_timestamps
                        and math.isfinite(timestamp)
                        and _sdk_feedback_frame_was_accepted(
                            self, identifier, timestamp
                        )
                    ):
                        self._kinesync_feedback_timestamps[identifier] = timestamp
                return result

        def GetKineSyncSynchronizedFeedback(self) -> dict[str, object]:
            with self._kinesync_feedback_lock:
                joint_message = super().GetArmJointMsgs()
                gripper_message = super().GetArmGripperMsgs()
                state = joint_message.joint_state
                return {
                    "joint_timestamp_s": joint_message.time_stamp,
                    "joint_hz": joint_message.Hz,
                    "joints_millidegree": tuple(
                        getattr(state, f"joint_{index}") for index in range(1, 7)
                    ),
                    "gripper_timestamp_s": gripper_message.time_stamp,
                    "gripper_hz": gripper_message.Hz,
                    "gripper_angle_micrometre": (
                        gripper_message.gripper_state.grippers_angle
                    ),
                    "feedback_frame_timestamps_s": tuple(
                        self._kinesync_feedback_timestamps[identifier]
                        for identifier in self._FEEDBACK_IDS
                    ),
                }

    return TimestampedReadOnlyPiperInterface


def _sdk_feedback_frame_was_accepted(
    interface: Any, identifier: int, timestamp: float
) -> bool:
    if identifier in (0x2A5, 0x2A6, 0x2A7):
        return float(interface.GetArmJointMsgs().time_stamp) == timestamp
    if identifier == 0x2A8:
        return float(interface.GetArmGripperMsgs().time_stamp) == timestamp
    return False


def _positive_float(value: object, message: str) -> float:
    if not isinstance(value, Real) or isinstance(value, bool):
        raise PiperFeedbackError(message)
    converted = float(value)
    if not math.isfinite(converted) or converted <= 0.0:
        raise PiperFeedbackError(message)
    return converted


def _clock_ns(clock: Callable[[], int]) -> int:
    value = clock()
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise PiperFeedbackError("monotonic clock returned an invalid timestamp")
    return value


def _sdk_epoch_to_monotonic_ns(
    sdk_timestamp_s: float,
    *,
    monotonic_before_ns: int,
    monotonic_after_ns: int,
    wall_before_ns: int,
    wall_after_ns: int,
    future_tolerance_ns: int,
) -> int:
    sdk_epoch_ns = int(round(sdk_timestamp_s * 1_000_000_000.0))
    offset_before_ns = monotonic_before_ns - wall_before_ns
    offset_after_ns = monotonic_after_ns - wall_after_ns
    mapped_ns = sdk_epoch_ns + (offset_before_ns + offset_after_ns) // 2
    if mapped_ns < 0:
        raise PiperFeedbackError("PiPER SDK timestamp cannot map to monotonic time")
    if mapped_ns > monotonic_after_ns + future_tolerance_ns:
        raise PiperFeedbackError("PiPER SDK timestamp is in the future")
    return min(mapped_ns, monotonic_after_ns)


def _stop_sdk_background_workers(interface: Any) -> None:
    """Compensate for incomplete worker cleanup in piper_sdk 0.6.1."""

    errors: list[Exception] = []
    attributes = vars(interface)
    for suffix in ("__read_can_stop_event", "__can_monitor_stop_event"):
        for name, value in attributes.items():
            if name.endswith(suffix) and hasattr(value, "set"):
                value.set()
    try:
        interface.DisconnectPort()
    except Exception as error:
        errors.append(error)
    for suffix in ("__can_deal_th", "__can_monitor_th"):
        for name, worker in attributes.items():
            if name.endswith(suffix) and hasattr(worker, "is_alive") and worker.is_alive():
                worker.join(timeout=0.5)
                if worker.is_alive():
                    errors.append(RuntimeError(f"PiPER SDK worker did not stop: {suffix}"))
    for name, counter in attributes.items():
        if name.endswith("__fps_counter") and hasattr(counter, "stop"):
            try:
                counter.stop()
            except Exception as error:
                errors.append(error)
    if errors:
        raise PiperFeedbackError("failed to stop all PiPER SDK workers") from errors[0]


__all__ = ["PiperFeedback", "PiperFeedbackError", "PiperFeedbackReader"]
