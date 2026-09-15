"""Fail-closed sequencing, clock, and freshness monitor for observations."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

from .schema import LiveShadowPolicy, ObservationPacket, ObservationRejection


@dataclass(frozen=True)
class LiveShadowReceipt:
    """Immutable safety state with explicit zero-command evidence."""

    accepted_packets: int
    terminal_reason: str | None
    max_head_auxiliary_skew_ns: int
    max_camera_state_skew_ns: int
    robot_command_requests: int = 0
    robot_commands_sent: int = 0


class FailClosedObservationMonitor:
    """Accept contiguous valid packets until the first terminal violation."""

    def __init__(self, policy: LiveShadowPolicy | None = None):
        self._policy = policy or LiveShadowPolicy()
        self._accepted_packets = 0
        self._terminal_reason: str | None = None
        self._session_id: str | None = None
        self._clock_domain: str | None = None
        self._timestamps: tuple[int, int, int] | None = None
        self._arrival_monotonic_ns: int | None = None
        self._max_head_auxiliary_skew_ns = 0
        self._max_camera_state_skew_ns = 0

    @property
    def receipt(self) -> LiveShadowReceipt:
        return LiveShadowReceipt(
            accepted_packets=self._accepted_packets,
            terminal_reason=self._terminal_reason,
            max_head_auxiliary_skew_ns=self._max_head_auxiliary_skew_ns,
            max_camera_state_skew_ns=self._max_camera_state_skew_ns,
        )

    def accept(
        self, packet: ObservationPacket, now_ns: int, decision_deadline_ns: int
    ) -> LiveShadowReceipt:
        """Validate one packet before committing any monitor state."""

        if self._terminal_reason is not None:
            raise ObservationRejection(self._terminal_reason)
        if not _is_nonnegative_integer(now_ns) or not _is_nonnegative_integer(
            decision_deadline_ns
        ):
            self._reject("schema")
        if now_ns > decision_deadline_ns:
            self._reject("deadline")
        if packet.arrival_monotonic_ns > now_ns:
            self._reject("timestamp")
        if now_ns - packet.arrival_monotonic_ns > self._policy.packet_timeout_ns:
            self._reject("stale")
        if packet.sequence_id != self._accepted_packets:
            self._reject("sequence")
        if self._session_id is not None and packet.session_id != self._session_id:
            self._reject("session")
        if self._clock_domain is not None and packet.clock_domain != self._clock_domain:
            self._reject("clock_domain")
        if self._timestamps is not None and any(
            current <= previous
            for current, previous in zip(
                (
                    packet.head_capture_ns,
                    packet.auxiliary_capture_ns,
                    packet.state_capture_ns,
                ),
                self._timestamps,
                strict=True,
            )
        ):
            self._reject("timestamp")
        if (
            self._arrival_monotonic_ns is not None
            and packet.arrival_monotonic_ns < self._arrival_monotonic_ns
        ):
            self._reject("timestamp")

        head_auxiliary_skew_ns = abs(packet.head_capture_ns - packet.auxiliary_capture_ns)
        if head_auxiliary_skew_ns > self._policy.max_head_auxiliary_skew_ns:
            self._reject("camera_skew")
        camera_state_skew_ns = max(
            abs(packet.head_capture_ns - packet.state_capture_ns),
            abs(packet.auxiliary_capture_ns - packet.state_capture_ns),
        )
        if camera_state_skew_ns > self._policy.max_camera_state_skew_ns:
            self._reject("state_skew")

        self._accepted_packets += 1
        self._session_id = packet.session_id
        self._clock_domain = packet.clock_domain
        self._timestamps = (
            packet.head_capture_ns,
            packet.auxiliary_capture_ns,
            packet.state_capture_ns,
        )
        self._arrival_monotonic_ns = packet.arrival_monotonic_ns
        self._max_head_auxiliary_skew_ns = max(
            self._max_head_auxiliary_skew_ns, head_auxiliary_skew_ns
        )
        self._max_camera_state_skew_ns = max(
            self._max_camera_state_skew_ns, camera_state_skew_ns
        )
        return self.receipt

    def _reject(self, reason: str) -> None:
        self._terminal_reason = reason
        raise ObservationRejection(reason)


def _is_nonnegative_integer(value: object) -> bool:
    return isinstance(value, Integral) and not isinstance(value, bool) and value >= 0
