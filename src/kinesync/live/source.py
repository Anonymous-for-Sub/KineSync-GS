"""Recorded, observation-only sources for the RT10-P shadow runner."""

from __future__ import annotations

from numbers import Integral
from typing import Protocol, runtime_checkable

import numpy as np

from kinesync.data.take_pens import TakePensTrajectory

from .schema import ObservationPacket


@runtime_checkable
class ObservationSource(Protocol):
    """Synchronous input boundary for observation-only shadowing."""

    def next_before(self, deadline_monotonic_ns: int) -> ObservationPacket | None:
        """Return one packet, or ``None`` when no packet is available before a deadline."""


class RecordedTakePensSource:
    """Lazily replay one validated, half-open Take-Pens trajectory segment."""

    def __init__(
        self,
        trajectory: TakePensTrajectory,
        *,
        start: int,
        stop: int,
        cap: int,
        session_id: str,
        clock_domain: str,
        monotonic_clock_ns,
    ):
        if not isinstance(trajectory, TakePensTrajectory):
            raise TypeError("trajectory must be a validated TakePensTrajectory")
        if not all(_is_nonnegative_integer(value) for value in (start, stop, cap)):
            raise ValueError("start, stop, and cap must be nonnegative integers")
        if start >= stop or stop > trajectory.length or cap < 1:
            raise ValueError("replay range and cap must describe a nonempty trajectory segment")
        if not all(isinstance(value, str) and value for value in (session_id, clock_domain)):
            raise ValueError("session_id and clock_domain must be nonempty strings")
        if not callable(monotonic_clock_ns):
            raise TypeError("monotonic_clock_ns must be callable")

        self._trajectory = trajectory
        self._stop = min(stop, start + cap)
        self._session_id = session_id
        self._clock_domain = clock_domain
        self._monotonic_clock_ns = monotonic_clock_ns
        self._next_row = start
        self._sequence_id = 0
        self.exhausted = False

        self._head_capture_ns = trajectory.timestamps("head")
        self._auxiliary_capture_ns = trajectory.timestamps("wrist")
        self._state_capture_ns = trajectory.timestamps("slave")
        self._qpos = trajectory.slave_qpos()
        if self._qpos.shape != (trajectory.length, 8):
            raise ValueError("Take-Pens trajectory must provide canonical 8D qpos")

    def next_before(self, deadline_monotonic_ns: int) -> ObservationPacket | None:
        """Read only the next RGB pair after confirming the local deadline remains valid."""

        if not _is_nonnegative_integer(deadline_monotonic_ns):
            raise ValueError("deadline_monotonic_ns must be a nonnegative integer")
        if self._next_row >= self._stop:
            self.exhausted = True
            return None

        if _read_monotonic_ns(self._monotonic_clock_ns) > deadline_monotonic_ns:
            return None

        row = self._next_row
        head_rgb = self._trajectory.read_rgb("head", [row])[0]
        if _read_monotonic_ns(self._monotonic_clock_ns) > deadline_monotonic_ns:
            return None
        auxiliary_rgb = self._trajectory.read_rgb("wrist", [row])[0]
        arrival_ns = _read_monotonic_ns(self._monotonic_clock_ns)
        if arrival_ns > deadline_monotonic_ns:
            return None
        packet = ObservationPacket(
            session_id=self._session_id,
            clock_domain=self._clock_domain,
            sequence_id=self._sequence_id,
            arrival_monotonic_ns=arrival_ns,
            head_rgb=head_rgb,
            head_capture_ns=int(self._head_capture_ns[row]),
            auxiliary_rgb=auxiliary_rgb,
            auxiliary_capture_ns=int(self._auxiliary_capture_ns[row]),
            auxiliary_role="wrist",
            qpos=np.asarray(self._qpos[row]),
            state_capture_ns=int(self._state_capture_ns[row]),
            source_id=self._trajectory.trajectory_id,
            frame_id=f"{self._trajectory.trajectory_id}:{row}",
        )
        self._next_row += 1
        self._sequence_id += 1
        return packet


def _is_nonnegative_integer(value: object) -> bool:
    return isinstance(value, Integral) and not isinstance(value, bool) and value >= 0


def _read_monotonic_ns(clock) -> int:
    value = clock()
    if not _is_nonnegative_integer(value):
        raise ValueError("monotonic_clock_ns must return a nonnegative integer")
    return int(value)


__all__ = ["ObservationSource", "RecordedTakePensSource"]
