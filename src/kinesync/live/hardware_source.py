"""Synchronized PiPER and D455 observation source."""

from __future__ import annotations

import time
from numbers import Integral
from typing import Any, Callable

from .schema import ObservationPacket


class PiperD455ObservationSource:
    """Compose read-only robot feedback and one or two RGB cameras."""

    def __init__(
        self,
        *,
        robot: Any,
        head_camera: Any,
        auxiliary_camera: Any | None,
        mode: str,
        cap: int,
        session_id: str,
        monotonic_clock_ns: Callable[[], int] = time.monotonic_ns,
        packet_sink: Callable[[ObservationPacket], None] | None = None,
        feedback_ready_timeout_s: float = 5.0,
    ) -> None:
        if mode not in {"dual", "single_preflight"}:
            raise ValueError("mode must be dual or single_preflight")
        if not isinstance(cap, Integral) or isinstance(cap, bool) or cap < 1:
            raise ValueError("cap must be a positive integer")
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("session_id must be a nonempty string")
        if not callable(monotonic_clock_ns):
            raise TypeError("monotonic_clock_ns must be callable")
        if packet_sink is not None and not callable(packet_sink):
            raise TypeError("packet_sink must be callable")
        if feedback_ready_timeout_s <= 0.0:
            raise ValueError("feedback_ready_timeout_s must be positive")
        if mode == "dual":
            if auxiliary_camera is None:
                raise ValueError("dual mode requires an auxiliary camera")
            if head_camera.serial == auxiliary_camera.serial:
                raise ValueError("dual mode requires distinct camera serials")
        elif auxiliary_camera is not None:
            raise ValueError("single_preflight mode accepts exactly one camera")

        self._robot = robot
        self._head = head_camera
        self._auxiliary = auxiliary_camera
        self._mode = mode
        self._cap = int(cap)
        self._session_id = session_id
        self._clock = monotonic_clock_ns
        self._packet_sink = packet_sink
        self._feedback_ready_timeout_s = float(feedback_ready_timeout_s)
        self._sequence_id = 0
        self._started = False
        self._capture_records: list[dict[str, object]] = []
        self.exhausted = False

    def start(self) -> None:
        if self._started:
            return
        started: list[Any] = []
        try:
            self._head.start()
            started.append(self._head)
            if self._auxiliary is not None:
                self._auxiliary.start()
                started.append(self._auxiliary)
            self._robot.connect()
            started.append(self._robot)
            if hasattr(self._robot, "wait_until_ready"):
                self._robot.wait_until_ready(timeout_s=self._feedback_ready_timeout_s)
        except Exception:
            for device in reversed(started):
                device.close()
            raise
        self._started = True

    def next_before(self, deadline_monotonic_ns: int) -> ObservationPacket | None:
        if not self._started:
            raise RuntimeError("hardware observation source is not started")
        if not isinstance(deadline_monotonic_ns, Integral) or isinstance(
            deadline_monotonic_ns, bool
        ):
            raise ValueError("deadline_monotonic_ns must be an integer")
        if self._sequence_id >= self._cap:
            self.exhausted = True
            return None

        head = self._head.read_before(int(deadline_monotonic_ns))
        if head is None:
            return None
        auxiliary = (
            head
            if self._auxiliary is None
            else self._auxiliary.read_before(int(deadline_monotonic_ns))
        )
        if auxiliary is None:
            return None
        if hasattr(self._robot, "read_before"):
            feedback = self._robot.read_before(int(deadline_monotonic_ns))
            if feedback is None:
                return None
        else:
            feedback = self._robot.read()
        arrival_ns = _clock_ns(self._clock)
        if arrival_ns > deadline_monotonic_ns:
            return None

        packet = ObservationPacket(
            session_id=self._session_id,
            clock_domain="monotonic_ns",
            sequence_id=self._sequence_id,
            arrival_monotonic_ns=arrival_ns,
            head_rgb=head.rgb,
            head_capture_ns=head.capture_monotonic_ns,
            auxiliary_rgb=auxiliary.rgb,
            auxiliary_capture_ns=auxiliary.capture_monotonic_ns,
            auxiliary_role="external",
            qpos=feedback.qpos,
            state_capture_ns=feedback.capture_monotonic_ns,
            source_id="piper-d455-live",
            frame_id=(
                f"{head.serial}:{head.frame_number}|"
                f"{auxiliary.serial}:{auxiliary.frame_number}"
            ),
            head_device_timestamp_ms=head.device_timestamp_ms,
            auxiliary_device_timestamp_ms=auxiliary.device_timestamp_ms,
            head_frame_number=head.frame_number,
            auxiliary_frame_number=auxiliary.frame_number,
            state_sdk_joint_timestamp_s=feedback.sdk_joint_timestamp_s,
            state_sdk_gripper_timestamp_s=feedback.sdk_gripper_timestamp_s,
            state_sdk_feedback_frame_timestamps_s=(
                feedback.sdk_feedback_frame_timestamps_s
            ),
        )
        self._capture_records.append(
            {
                "sequence_id": packet.sequence_id,
                "head_device_timestamp_ms": head.device_timestamp_ms,
                "auxiliary_device_timestamp_ms": auxiliary.device_timestamp_ms,
                "head_frame_number": head.frame_number,
                "auxiliary_frame_number": auxiliary.frame_number,
                "state_sdk_joint_timestamp_s": feedback.sdk_joint_timestamp_s,
                "state_sdk_gripper_timestamp_s": feedback.sdk_gripper_timestamp_s,
                "state_sdk_feedback_frame_timestamps_s": list(
                    feedback.sdk_feedback_frame_timestamps_s
                ),
            }
        )
        self._sequence_id += 1
        if self._packet_sink is not None:
            self._packet_sink(packet)
        return packet

    @property
    def capture_records(self) -> tuple[dict[str, object], ...]:
        return tuple(dict(record) for record in self._capture_records)

    def describe(self) -> dict[str, object]:
        return {
            "mode": self._mode,
            "session_id": self._session_id,
            "cap": self._cap,
            "packets_emitted": self._sequence_id,
            "independent_auxiliary_view": self._auxiliary is not None,
            "robot": self._robot.describe(),
            "head_camera": self._head.describe(),
            "auxiliary_camera": (
                None if self._auxiliary is None else self._auxiliary.describe()
            ),
        }

    def close(self) -> None:
        errors: list[Exception] = []
        for device in (self._robot, self._auxiliary, self._head):
            if device is None:
                continue
            try:
                device.close()
            except Exception as error:
                errors.append(error)
        self._started = False
        if errors:
            raise RuntimeError("one or more hardware devices failed to close") from errors[0]

    def __enter__(self) -> "PiperD455ObservationSource":
        self.start()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


def _clock_ns(clock: Callable[[], int]) -> int:
    value = clock()
    if not isinstance(value, Integral) or isinstance(value, bool) or value < 0:
        raise ValueError("monotonic clock returned an invalid timestamp")
    return int(value)


__all__ = ["PiperD455ObservationSource"]
