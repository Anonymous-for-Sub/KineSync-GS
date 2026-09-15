"""Append-only, command-incapable observation shadow runner."""

from __future__ import annotations

import math
import time
import uuid
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any

from .monitor import FailClosedObservationMonitor
from .safety import canonical_json_bytes, sha256_file, sha256_hex
from .schema import ObservationPacket, ObservationRejection
from .source import ObservationSource


_EVENT_SCHEMA = "kinesync.live_shadow_event.v2"
_SAFETY_SCHEMA = "kinesync.live_shadow_safety.v2"


@dataclass(frozen=True)
class LiveShadowRunSummary:
    """Terminal receipt for one append-only observation-only run."""

    event_path: Path
    safety_path: Path
    accepted_packets: int
    event_count: int
    terminal_reason: str | None
    robot_command_requests: int = 0
    robot_commands_sent: int = 0


class _EventSerializationError(RuntimeError):
    pass


class _EventWriter:
    def __init__(self, handle):
        self._handle = handle
        self.previous_event_sha256: str | None = None
        self.event_count = 0
        self.complete = True

    def append(self, payload: dict[str, object]) -> str:
        event = {
            "schema": _EVENT_SCHEMA,
            **payload,
            "event_index": self.event_count,
            "previous_event_sha256": self.previous_event_sha256,
        }
        try:
            event["event_sha256"] = sha256_hex(canonical_json_bytes(event))
            serialized = canonical_json_bytes(event).decode("ascii") + "\n"
        except (TypeError, ValueError) as error:
            raise _EventSerializationError from error
        try:
            self._handle.write(serialized)
            self._handle.flush()
        except OSError:
            self.complete = False
            raise
        self.previous_event_sha256 = str(event["event_sha256"])
        self.event_count += 1
        return self.previous_event_sha256


def run_observation_only_shadow(
    *,
    source: ObservationSource,
    estimator: Any,
    monitor: FailClosedObservationMonitor,
    event_path: str | Path,
    safety_path: str | Path,
    monotonic_clock_ns=time.monotonic_ns,
    packet_timeout_ns: int = 250_000_000,
) -> LiveShadowRunSummary:
    """Process observations until clean source exhaustion or the first terminal rejection."""

    if not callable(monotonic_clock_ns):
        raise TypeError("monotonic_clock_ns must be callable")
    if not _is_positive_integer(packet_timeout_ns):
        raise ValueError("packet_timeout_ns must be a positive integer")
    event_target = Path(event_path)
    safety_target = Path(safety_path)
    if event_target == safety_target:
        raise ValueError("event and safety paths must be distinct")
    if safety_target.exists():
        raise FileExistsError(safety_target)

    terminal_reason: str | None = None
    accepted_session_id: str | None = None
    accepted_clock_domain: str | None = None
    writer: _EventWriter | None = None
    run_started = False
    try:
        with event_target.open("x", encoding="ascii", newline="") as event_handle:
            run_started = True
            writer = _EventWriter(event_handle)
            try:
                while terminal_reason is None:
                    poll_now_ns = _read_monotonic_ns(monotonic_clock_ns)
                    deadline_ns = poll_now_ns + packet_timeout_ns
                    packet = source.next_before(deadline_ns)
                    if packet is None:
                        if _source_is_exhausted(source):
                            break
                        terminal_reason = "timeout"
                        _append_terminal(writer, terminal_reason, packet=None, monitor_accepted=False)
                        break

                    monitor_accepted = False
                    try:
                        refreshed_now_ns = _read_monotonic_ns(monotonic_clock_ns)
                        if refreshed_now_ns > deadline_ns:
                            raise ObservationRejection("deadline")
                        monitor.accept(
                            packet,
                            now_ns=refreshed_now_ns,
                            decision_deadline_ns=deadline_ns,
                        )
                        monitor_accepted = True
                        if accepted_session_id is None:
                            accepted_session_id = packet.session_id
                            accepted_clock_domain = packet.clock_domain
                        processing_started_ns = _read_monotonic_ns(monotonic_clock_ns)
                        estimate = estimator.push(packet)
                        processing_finished_ns = _read_monotonic_ns(monotonic_clock_ns)
                        if processing_finished_ns < processing_started_ns:
                            raise ObservationRejection("timestamp")
                        if processing_finished_ns > deadline_ns:
                            raise ObservationRejection("deadline")
                        writer.append(
                            _accepted_event(
                                packet,
                                estimate=estimate,
                                processing_latency_ns=processing_finished_ns
                                - processing_started_ns,
                            )
                        )
                    except ObservationRejection as error:
                        terminal_reason = error.reason
                        _append_terminal(
                            writer,
                            terminal_reason,
                            packet=packet,
                            monitor_accepted=monitor_accepted,
                        )
                    except _EventSerializationError:
                        terminal_reason = "serialization"
                        _append_terminal(
                            writer,
                            terminal_reason,
                            packet=packet,
                            monitor_accepted=monitor_accepted,
                        )
                    except Exception:
                        terminal_reason = "internal"
                        _append_terminal(
                            writer,
                            terminal_reason,
                            packet=packet,
                            monitor_accepted=monitor_accepted,
                        )
            except _EventSerializationError:
                terminal_reason = "serialization"
                _append_terminal(writer, terminal_reason, packet=None, monitor_accepted=False)
            except Exception:
                terminal_reason = "internal"
                _append_terminal(writer, terminal_reason, packet=None, monitor_accepted=False)
    finally:
        if run_started and writer is not None:
            receipt = monitor.receipt
            safety_payload = {
                "schema": _SAFETY_SCHEMA,
                "closed": True,
                "events_complete": writer.complete,
                "terminal_reason": terminal_reason,
                "accepted_packets": receipt.accepted_packets,
                "event_count": writer.event_count,
                "session_id": accepted_session_id,
                "clock_domain": accepted_clock_domain,
                "final_event_sha256": writer.previous_event_sha256,
                "events_file_sha256": sha256_file(event_target),
                "max_head_auxiliary_skew_ns": receipt.max_head_auxiliary_skew_ns,
                "max_camera_state_skew_ns": receipt.max_camera_state_skew_ns,
                "robot_command_requests": 0,
                "robot_commands_sent": 0,
            }
            _write_safety_receipt(safety_target, safety_payload)

    assert writer is not None
    receipt = monitor.receipt
    return LiveShadowRunSummary(
        event_path=event_target,
        safety_path=safety_target,
        accepted_packets=receipt.accepted_packets,
        event_count=writer.event_count,
        terminal_reason=terminal_reason,
    )


def _append_terminal(
    writer: _EventWriter,
    reason: str,
    *,
    packet: ObservationPacket | None,
    monitor_accepted: bool,
) -> None:
    try:
        writer.append(
            _terminal_event(
                reason, packet=packet, monitor_accepted=monitor_accepted
            )
        )
    except Exception:
        writer.complete = False


def _accepted_event(
    packet: ObservationPacket, *, estimate: Any, processing_latency_ns: int
) -> dict[str, object]:
    return {
        **_packet_fields(packet),
        "terminal": False,
        "monitor_accepted": True,
        "estimate": _estimate_payload(estimate),
        "processing_latency_ns": processing_latency_ns,
        "robot_command_requests": 0,
        "robot_commands_sent": 0,
    }


def _terminal_event(
    reason: str, *, packet: ObservationPacket | None, monitor_accepted: bool
) -> dict[str, object]:
    fields = {} if packet is None else _packet_fields(packet)
    return {
        **fields,
        "terminal": True,
        "monitor_accepted": monitor_accepted,
        "terminal_reason": reason,
        "robot_command_requests": 0,
        "robot_commands_sent": 0,
    }


def _packet_fields(packet: ObservationPacket) -> dict[str, object]:
    fields = {
        "session_id": packet.session_id,
        "clock_domain": packet.clock_domain,
        "sequence_id": packet.sequence_id,
        "arrival_monotonic_ns": packet.arrival_monotonic_ns,
        "head_capture_ns": packet.head_capture_ns,
        "auxiliary_capture_ns": packet.auxiliary_capture_ns,
        "state_capture_ns": packet.state_capture_ns,
        "auxiliary_role": packet.auxiliary_role,
        "source_id": packet.source_id,
        "frame_id": packet.frame_id,
    }
    metadata = {
        "head_device_timestamp_ms": packet.head_device_timestamp_ms,
        "auxiliary_device_timestamp_ms": packet.auxiliary_device_timestamp_ms,
        "head_frame_number": packet.head_frame_number,
        "auxiliary_frame_number": packet.auxiliary_frame_number,
        "state_sdk_joint_timestamp_s": packet.state_sdk_joint_timestamp_s,
        "state_sdk_gripper_timestamp_s": packet.state_sdk_gripper_timestamp_s,
        "state_sdk_feedback_frame_timestamps_s": (
            packet.state_sdk_feedback_frame_timestamps_s
        ),
    }
    fields.update({key: value for key, value in metadata.items() if value is not None})
    return fields


def _estimate_payload(estimate: Any) -> dict[str, object]:
    if estimate is None:
        return {"available": False}
    return {
        "available": True,
        "absolute_offset_ms": _finite_or_none(estimate.estimate.offset_ms),
        "relative_correction_ms": _finite_or_none(estimate.decision.correction_ms),
        "commit": bool(estimate.decision.accepted),
        "rejection_reason": str(estimate.decision.reason),
        "evidence": _json_safe(estimate.estimate.to_dict()),
        "window_size": int(estimate.window_size),
    }


def _write_safety_receipt(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="ascii", newline="") as handle:
            handle.write(canonical_json_bytes(payload).decode("ascii") + "\n")
            handle.flush()
        path.hardlink_to(temporary)
    finally:
        if temporary.exists():
            temporary.unlink()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float):
        return _finite_or_none(value)
    if hasattr(value, "item"):
        return _json_safe(value.item())
    return value


def _finite_or_none(value: object) -> float | None:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if math.isfinite(converted) else None


def _is_positive_integer(value: object) -> bool:
    return isinstance(value, Integral) and not isinstance(value, bool) and value > 0


def _source_is_exhausted(source: ObservationSource) -> bool:
    try:
        exhausted = source.exhausted
    except AttributeError:
        return False
    return bool(exhausted)


def _read_monotonic_ns(clock) -> int:
    value = clock()
    if not isinstance(value, Integral) or isinstance(value, bool) or value < 0:
        raise ValueError("monotonic_clock_ns must return a nonnegative integer")
    return int(value)


__all__ = ["LiveShadowRunSummary", "run_observation_only_shadow"]
