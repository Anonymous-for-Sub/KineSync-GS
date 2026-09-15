"""Run KineSync RT10 with read-only PiPER feedback and RealSense D455 RGB."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

import yaml

from kinesync.config import load_config
from kinesync.hardware import PiperFeedbackReader, RealSenseColorReader
from kinesync.live import (
    FailClosedObservationMonitor,
    LiveShadowPolicy,
    PiperD455ObservationSource,
    run_observation_only_shadow,
)
from kinesync.live.artifacts import LiveArtifactRecorder
from kinesync.live.estimator import CausalTemporalShadowEstimator
from kinesync.live.safety import sha256_file, verify_shadow_audit
from kinesync.temporal.telemetry import TemporalGuard


SourceFactory = Callable[[Mapping[str, Any], str, Callable[[Any], None]], Any]


def execute_rt10_piper_d455_live(
    config: Mapping[str, Any],
    *,
    run_id: str | None = None,
    source_factory: SourceFactory | None = None,
    estimator: Any | None = None,
) -> Path:
    protocol = _protocol_config(config)
    _validate_guard(protocol["guard_path"], protocol["guard_sha256"])
    effective_estimator = estimator or _temporal_estimator(protocol["guard_path"])
    effective_run_id = run_id or _generated_run_id()
    _validate_run_id(effective_run_id)
    run_path = protocol["run_root"] / effective_run_id
    run_path.mkdir(parents=True, exist_ok=False)
    (run_path / "config.yaml").write_text(
        yaml.safe_dump(dict(config), sort_keys=True), encoding="utf-8"
    )

    session_id = f"rt10h-{effective_run_id}"
    recorder = LiveArtifactRecorder(
        run_path,
        expected_packets=protocol["packet_count"],
        fps=protocol["fps"],
        sample_count=protocol["sample_count"],
    )
    factory = source_factory or _hardware_source
    source = None
    summary = None
    hardware_description = None
    execution_error: Exception | None = None
    try:
        source = factory(protocol, session_id, recorder.append)
        source.start()
        if protocol["hardware_warmup_s"]:
            time.sleep(protocol["hardware_warmup_s"])
        summary = run_observation_only_shadow(
            source=source,
            estimator=effective_estimator,
            monitor=FailClosedObservationMonitor(
                LiveShadowPolicy(
                    packet_timeout_ns=protocol["packet_timeout_ns"],
                    max_head_auxiliary_skew_ns=protocol["max_head_auxiliary_skew_ns"],
                    max_camera_state_skew_ns=protocol["max_camera_state_skew_ns"],
                )
            ),
            event_path=run_path / "events.jsonl",
            safety_path=run_path / "safety.json",
            packet_timeout_ns=protocol["packet_timeout_ns"],
        )
    except Exception as error:
        execution_error = error
    finally:
        if source is not None:
            try:
                hardware_description = source.describe()
            except Exception as error:
                execution_error = execution_error or error
            try:
                source.close()
            except Exception as error:
                execution_error = execution_error or error

    try:
        visual_paths = recorder.close(allow_empty=True)
    except Exception as error:
        visual_paths = {}
        execution_error = execution_error or error

    audit_valid = False
    safety: dict[str, Any] = {}
    if summary is not None:
        audit_valid = verify_shadow_audit(summary.event_path, summary.safety_path).valid
        safety = _read_json(run_path / "safety.json")
    terminal_reason = (
        "runtime_error"
        if execution_error is not None
        else None if summary is None else summary.terminal_reason
    )
    accepted_packets = 0 if summary is None else summary.accepted_packets
    event_count = 0 if summary is None else summary.event_count
    capture_complete = (
        execution_error is None
        and summary is not None
        and summary.terminal_reason is None
        and accepted_packets == protocol["packet_count"]
    )
    metrics = {
        "schema": "kinesync.rt10_piper_d455_metrics.v1",
        "accepted_packets": accepted_packets,
        "expected_packets": protocol["packet_count"],
        "event_count": event_count,
        "terminal_reason": terminal_reason,
        "capture_complete": capture_complete,
        "shadow_audit_valid": audit_valid,
        "max_head_auxiliary_skew_ms": safety.get("max_head_auxiliary_skew_ns", 0) / 1e6,
        "max_camera_state_skew_ms": safety.get("max_camera_state_skew_ns", 0) / 1e6,
        "robot_command_requests": safety.get("robot_command_requests", 0),
        "robot_commands_sent": safety.get("robot_commands_sent", 0),
        "error_type": None if execution_error is None else type(execution_error).__name__,
        "error_message": None if execution_error is None else str(execution_error),
    }
    _write_json(run_path / "metrics.json", metrics)
    output_names = (
        "config.yaml",
        "metrics.json",
        *(path.name for path in visual_paths.values()),
    )
    optional_names = tuple(
        name for name in ("events.jsonl", "safety.json") if (run_path / name).is_file()
    )
    output_hashes = {
        name: {
            "sha256": sha256_file(run_path / name),
            "size_bytes": (run_path / name).stat().st_size,
        }
        for name in sorted(set((*output_names, *optional_names)))
    }
    manifest = {
        "schema": "kinesync.rt10_piper_d455_hardware_manifest.v1",
        "experiment": "rt10_piper_d455_live_observation",
        "run_id": effective_run_id,
        "session_id": session_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "dependency_versions": {
            name: _package_version(name) for name in ("piper_sdk", "pyrealsense2")
        },
        "command_mode": "disabled",
        "hardware": hardware_description,
        "paper_eligible_multi_view": bool(
            hardware_description
            and hardware_description.get("independent_auxiliary_view", False)
            and capture_complete
        ),
        "guard": {
            "path": str(protocol["guard_path"]),
            "semantic_sha256": protocol["guard_sha256"],
            "file_sha256": sha256_file(protocol["guard_path"]),
        },
        "metrics": metrics,
        "output_hashes": output_hashes,
    }
    _write_json(run_path / "hardware_manifest.json", manifest)
    if execution_error is not None:
        raise RuntimeError(
            f"live hardware run failed; inspect {run_path / 'hardware_manifest.json'}"
        ) from execution_error
    if not capture_complete:
        raise RuntimeError(
            f"live capture did not complete; inspect {run_path / 'hardware_manifest.json'}"
        )
    return run_path


def _hardware_source(
    protocol: Mapping[str, Any], session_id: str, packet_sink: Callable[[Any], None]
) -> PiperD455ObservationSource:
    robot = PiperFeedbackReader(
        can_name=protocol["can_name"],
        max_gripper_stroke_mm=protocol["max_gripper_stroke_mm"],
        finger_travel_m=protocol["finger_travel_m"],
    )
    head = RealSenseColorReader(
        serial=protocol["head_serial"],
        width=protocol["width"],
        height=protocol["height"],
        fps=protocol["fps"],
    )
    auxiliary = None
    if protocol["mode"] == "dual":
        auxiliary = RealSenseColorReader(
            serial=protocol["auxiliary_serial"],
            width=protocol["width"],
            height=protocol["height"],
            fps=protocol["fps"],
        )
    return PiperD455ObservationSource(
        robot=robot,
        head_camera=head,
        auxiliary_camera=auxiliary,
        mode=protocol["mode"],
        cap=protocol["packet_count"],
        session_id=session_id,
        packet_sink=packet_sink,
        feedback_ready_timeout_s=protocol["feedback_ready_timeout_s"],
    )


def _protocol_config(config: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(config, Mapping):
        raise ValueError("live hardware configuration must be a mapping")
    if config.get("command_mode") != "disabled":
        raise ValueError("command_mode must be the literal disabled")
    mode = config.get("mode")
    if mode not in {"dual", "single_preflight"}:
        raise ValueError("mode must be dual or single_preflight")
    robot = _mapping(config.get("robot"), "robot")
    cameras = _mapping(config.get("cameras"), "cameras")
    capture = _mapping(config.get("capture"), "capture")
    monitor = _mapping(config.get("monitor"), "monitor")
    guard = _mapping(config.get("guard"), "guard")

    head_serial = _string(cameras.get("head_serial"), "cameras.head_serial")
    auxiliary_serial = cameras.get("auxiliary_serial")
    if mode == "dual":
        auxiliary_serial = _string(auxiliary_serial, "cameras.auxiliary_serial")
        if auxiliary_serial == head_serial:
            raise ValueError("dual mode requires distinct camera serials")
    elif auxiliary_serial not in {None, ""}:
        raise ValueError("single_preflight requires an empty auxiliary_serial")

    packet_count = _integer(capture.get("packet_count"), "capture.packet_count", minimum=26)
    packet_timeout_ms = _number(
        capture.get("packet_timeout_ms"),
        "capture.packet_timeout_ms",
        positive=True,
    )
    return {
        "mode": mode,
        "can_name": _string(robot.get("can_name"), "robot.can_name"),
        "max_gripper_stroke_mm": _number(
            robot.get("max_gripper_stroke_mm"),
            "robot.max_gripper_stroke_mm",
            positive=True,
        ),
        "finger_travel_m": _number(
            robot.get("finger_travel_m"), "robot.finger_travel_m", positive=True
        ),
        "head_serial": head_serial,
        "auxiliary_serial": auxiliary_serial,
        "width": _integer(cameras.get("width"), "cameras.width", minimum=1),
        "height": _integer(cameras.get("height"), "cameras.height", minimum=1),
        "fps": _integer(cameras.get("fps"), "cameras.fps", minimum=1),
        "packet_count": packet_count,
        "packet_timeout_ns": _milliseconds_to_ns(
            packet_timeout_ms, "capture.packet_timeout_ms"
        ),
        "sample_count": _integer(
            capture.get("sample_count"), "capture.sample_count", minimum=1
        ),
        "hardware_warmup_s": _number(
            capture.get("hardware_warmup_s", 2.0),
            "capture.hardware_warmup_s",
            nonnegative=True,
        ),
        "feedback_ready_timeout_s": _number(
            capture.get("feedback_ready_timeout_s", 5.0),
            "capture.feedback_ready_timeout_s",
            positive=True,
        ),
        "max_head_auxiliary_skew_ns": _milliseconds_to_ns(
            _number(
                monitor.get("max_head_auxiliary_skew_ms"),
                "monitor.max_head_auxiliary_skew_ms",
                positive=True,
            ),
            "monitor.max_head_auxiliary_skew_ms",
        ),
        "max_camera_state_skew_ns": _milliseconds_to_ns(
            _number(
                monitor.get("max_camera_state_skew_ms"),
                "monitor.max_camera_state_skew_ms",
                positive=True,
            ),
            "monitor.max_camera_state_skew_ms",
        ),
        "guard_path": Path(_string(guard.get("path"), "guard.path"))
        .expanduser()
        .resolve(),
        "guard_sha256": _string(guard.get("sha256"), "guard.sha256"),
        "run_root": Path(_string(config.get("run_root"), "run_root"))
        .expanduser()
        .resolve(),
    }


def _validate_guard(path: Path, expected_hash: str) -> None:
    payload = _read_json(path)
    embedded = payload.pop("sha256", None)
    semantic = hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")
    ).hexdigest()
    if embedded != expected_hash or semantic != expected_hash:
        raise ValueError("guard hash does not match the frozen temporal guard")
    try:
        TemporalGuard(**payload["GuardCalibrationReport"]["guard"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("guard file has an incompatible schema") from error


def _temporal_estimator(path: Path) -> CausalTemporalShadowEstimator:
    payload = _read_json(path)
    guard = payload["GuardCalibrationReport"]["guard"]
    return CausalTemporalShadowEstimator(TemporalGuard(**guard))


def _mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    return value


def _string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a nonempty string")
    return value


def _integer(value: object, field: str, *, minimum: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}")
    return value


def _number(
    value: object, field: str, *, positive: bool = False, nonnegative: bool = False
) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{field} must be numeric")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{field} must be finite")
    if positive and converted <= 0.0:
        raise ValueError(f"{field} must be positive")
    if nonnegative and converted < 0.0:
        raise ValueError(f"{field} must be nonnegative")
    return converted


def _milliseconds_to_ns(value: float, field: str) -> int:
    converted = value * 1_000_000.0
    if not math.isfinite(converted) or converted < 1.0 or converted > sys.maxsize:
        raise ValueError(f"{field} must fit positive integer nanoseconds")
    return int(converted)


def _validate_run_id(run_id: str) -> None:
    allowed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._-")
    if (
        not run_id
        or not run_id.isascii()
        or not run_id[0].isalnum()
        or not run_id[-1].isalnum()
        or any(character not in allowed for character in run_id)
    ):
        raise ValueError("run_id must be a flat ASCII slug")


def _generated_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f__rt10_piper_d455")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON file must contain an object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n",
        encoding="ascii",
    )


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", default=None)
    arguments = parser.parse_args()
    print(
        execute_rt10_piper_d455_live(
            load_config(arguments.config), run_id=arguments.run_id
        )
    )


if __name__ == "__main__":
    main()
