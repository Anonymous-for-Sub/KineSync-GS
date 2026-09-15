"""Run the immutable, command-disabled RT10-P recorded live-shadow preflight."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from kinesync.config import load_config
from kinesync.data.take_pens import load_take_pens_trajectory
from kinesync.live.estimator import CausalTemporalShadowEstimator
from kinesync.live.monitor import FailClosedObservationMonitor
from kinesync.live.runner import run_observation_only_shadow
from kinesync.live.safety import audit_live_imports, canonical_json_bytes, verify_shadow_audit
from kinesync.live.source import RecordedTakePensSource
from kinesync.temporal.telemetry import TemporalGuard


_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_CANONICAL_FILE = "f1.hdf5"
_CANONICAL_START = 62
_CANONICAL_STOP = 200
_CANONICAL_CAP = 120
_CANONICAL_AUXILIARY_ROLE = "wrist"
_COMMAND_MODE = "disabled"
_FROZEN_GUARD_PATH = (
    _REPOSITORY_ROOT / "runs" / "20260825_rt9_take_pens_temporal_v1" / "guard.json"
)
_FROZEN_GUARD_SHA256 = "398bf0b06f3bdf54fa001e9b2b9edf66476a4b31918ad34649b707843f72b89d"
_EVIDENCE_SCOPE = "recorded_real_observation_only_live_shadow_preflight"


def execute_rt10_live_shadow_preflight(
    config: Mapping[str, Any], *, run_id: str | None = None
) -> Path:
    """Execute one immutable, canonical, command-disabled RT10-P replay."""

    protocol = _protocol_config(config)
    guard, guard_file_sha256 = _load_frozen_guard(protocol["guard_path"])
    trajectory = load_take_pens_trajectory(
        protocol["dataset_root"], _CANONICAL_FILE, split="test"
    )
    if trajectory.length < _CANONICAL_STOP:
        raise ValueError("canonical f1.hdf5 replay requires at least 200 frames")
    _validate_run_id(run_id)
    if run_id is not None and (protocol["run_root"] / run_id).exists():
        raise FileExistsError(protocol["run_root"] / run_id)

    dependency_audit = audit_live_imports(
        [Path(__file__).resolve(), _REPOSITORY_ROOT / "src" / "kinesync" / "live"]
    )
    if not dependency_audit.valid:
        raise ValueError(f"RT10-P dependency audit failed: {dependency_audit.violations}")

    run_path = _create_run(
        root=protocol["run_root"],
        run_id=run_id,
        config=config,
        assets={
            "take_pens_f1.hdf5": protocol["dataset_root"] / _CANONICAL_FILE,
            "rt9_guard.json": protocol["guard_path"],
        },
    )
    session_id = f"rt10p-{run_path.name}"
    clock_domain = "host_monotonic_ns"
    source = RecordedTakePensSource(
        trajectory,
        start=_CANONICAL_START,
        stop=_CANONICAL_STOP,
        cap=_CANONICAL_CAP,
        session_id=session_id,
        clock_domain=clock_domain,
        monotonic_clock_ns=time.monotonic_ns,
    )
    summary = run_observation_only_shadow(
        source=source,
        estimator=CausalTemporalShadowEstimator(guard),
        monitor=FailClosedObservationMonitor(),
        event_path=run_path / "events.jsonl",
        safety_path=run_path / "safety.json",
        monotonic_clock_ns=time.monotonic_ns,
    )
    shadow_audit = verify_shadow_audit(summary.event_path, summary.safety_path)
    safety = _read_json_object(run_path / "safety.json", "safety receipt")
    events = _read_events(run_path / "events.jsonl")
    metrics = _metrics(events, safety, dependency_audit.valid, shadow_audit.valid)
    _write_json(run_path / "metrics.json", metrics)
    output_hashes = _fingerprint_outputs(
        run_path, ("config.yaml", "events.jsonl", "safety.json", "metrics.json")
    )
    output_hashes_match = all(
        record["sha256"] == _sha256_file(run_path / name)
        for name, record in output_hashes.items()
    )
    if not output_hashes_match:
        raise RuntimeError("RT10-P output fingerprinting failed")
    manifest_root = hashlib.sha256(canonical_json_bytes(output_hashes)).hexdigest()
    runtime_gates = _mapping(metrics["predeclared_gates"], "metrics predeclared gates")
    manifest_gates = {
        **{
            name: value
            for name, value in runtime_gates.items()
            if name != "overall_pass"
        },
        "output_hashes_match": output_hashes_match,
        "overall_pass": runtime_gates.get("overall_pass") is True
        and output_hashes_match,
    }
    manifest_path = run_path / "manifest.json"
    manifest = _read_json_object(manifest_path, "run manifest")
    manifest.update(
        {
            "evidence_scope": _EVIDENCE_SCOPE,
            "guard_sha256": _FROZEN_GUARD_SHA256,
            "guard_file_sha256": guard_file_sha256,
            "source_segment": {
                "file": _CANONICAL_FILE,
                "start": _CANONICAL_START,
                "stop": _CANONICAL_STOP,
                "cap": _CANONICAL_CAP,
            },
            "session_id": safety["session_id"],
            "clock_domain": safety["clock_domain"],
            "dependency_audit": {
                "valid": dependency_audit.valid,
                "paths": list(dependency_audit.paths),
                "violations": list(dependency_audit.violations),
            },
            "shadow_audit": {
                "valid": shadow_audit.valid,
                "event_count": shadow_audit.event_count,
                "terminal_reason": shadow_audit.terminal_reason,
            },
            "predeclared_gates": manifest_gates,
            "output_hashes": output_hashes,
            "external_manifest_root_sha256": manifest_root,
        }
    )
    _write_json(manifest_path, manifest)
    return run_path


def _protocol_config(config: Mapping[str, Any]) -> dict[str, Path]:
    if not isinstance(config, Mapping):
        raise ValueError("RT10-P configuration must be a mapping")
    if config.get("command_mode") != _COMMAND_MODE:
        raise ValueError("command_mode must be the literal disabled")
    dataset = _mapping(config.get("dataset"), "dataset")
    if dataset.get("file") != _CANONICAL_FILE:
        raise ValueError("dataset.file is frozen at f1.hdf5")
    dataset_root = _path(dataset.get("root"), "dataset.root")
    replay = _mapping(config.get("replay"), "replay")
    if replay != {"start": _CANONICAL_START, "stop": _CANONICAL_STOP, "cap": _CANONICAL_CAP}:
        raise ValueError("replay must be the canonical f1.hdf5:[62,200) capped at 120 packets")
    if config.get("auxiliary_role") != _CANONICAL_AUXILIARY_ROLE:
        raise ValueError("auxiliary_role is frozen at wrist")
    guard = _mapping(config.get("guard"), "guard")
    guard_path = _path(guard.get("path"), "guard.path")
    if guard_path != _FROZEN_GUARD_PATH:
        raise ValueError("guard.path must reference the frozen RT9 guard")
    if guard.get("sha256") != _FROZEN_GUARD_SHA256:
        raise ValueError("guard.sha256 does not match the frozen RT9 guard")
    run_root = _path(config.get("run_root", "runs"), "run_root")
    return {
        "dataset_root": dataset_root,
        "guard_path": guard_path,
        "run_root": run_root,
    }


def _load_frozen_guard(path: Path) -> tuple[TemporalGuard, str]:
    payload = _read_json_object(path, "frozen RT9 guard")
    embedded_hash = payload.pop("sha256", None)
    semantic_hash = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    if semantic_hash != embedded_hash or semantic_hash != _FROZEN_GUARD_SHA256:
        raise ValueError("frozen RT9 guard content hash does not match")
    if payload.get("schema") != "kinesync.rt9_temporal_guard.v1":
        raise ValueError("frozen RT9 guard schema is invalid")
    report = _mapping(payload.get("GuardCalibrationReport"), "GuardCalibrationReport")
    guard_values = _mapping(report.get("guard"), "GuardCalibrationReport.guard")
    return TemporalGuard(**guard_values), _sha256_file(path)


def _create_run(
    *,
    root: Path,
    run_id: str | None,
    config: Mapping[str, Any],
    assets: Mapping[str, Path],
) -> Path:
    resolved_root = root.expanduser().resolve()
    effective_run_id = run_id or _generated_run_id()
    _validate_run_id(effective_run_id)
    asset_records = _asset_records(
        assets, full_hash_names=frozenset({"take_pens_f1.hdf5"})
    )
    run_path = resolved_root / effective_run_id
    run_path.mkdir(parents=True, exist_ok=False)
    with (run_path / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(config), handle, sort_keys=True)
    _write_json(
        run_path / "manifest.json",
        {
            "schema_version": 1,
            "experiment": "rt10_live_shadow_preflight",
            "run_id": effective_run_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "target_provenance": "real_observation_live_shadow_preflight",
            "observation_backend": "recorded_take_pens_dual_camera_live_shadow",
            "assets": asset_records,
            "environment": {
                "python": sys.version.split()[0],
                "platform": platform.platform(),
            },
        },
    )
    return run_path


def _generated_run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    return f"{stamp}__rt10_live_shadow_preflight"


def _validate_run_id(run_id: str | None) -> None:
    if run_id is None:
        return
    allowed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._-")
    if (
        not isinstance(run_id, str)
        or not run_id
        or not run_id.isascii()
        or not run_id[0].isalnum()
        or not run_id[-1].isalnum()
        or any(character not in allowed for character in run_id)
    ):
        raise ValueError("Run ID must be a nonempty flat ASCII slug")


def _asset_records(
    assets: Mapping[str, Path], *, full_hash_names: frozenset[str]
) -> dict[str, dict[str, object]]:
    records: dict[str, dict[str, object]] = {}
    for name, raw_path in sorted(assets.items()):
        path = raw_path.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Asset does not exist: {path}")
        records[name] = {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            **_fingerprint(path, force_full=name in full_hash_names),
        }
    return records


def _fingerprint_outputs(
    run_path: Path, names: tuple[str, ...]
) -> dict[str, dict[str, object]]:
    return {
        name: {"size_bytes": (run_path / name).stat().st_size, **_fingerprint(run_path / name)}
        for name in sorted(names)
    }


def _fingerprint(
    path: Path,
    full_hash_limit: int = 64 * 1024 * 1024,
    *,
    force_full: bool = False,
) -> dict[str, str]:
    size = path.stat().st_size
    if force_full or size <= full_hash_limit:
        return {"sha256": _sha256_file(path), "sha256_mode": "full"}
    chunk_size = 8 * 1024 * 1024
    offsets = (0, max((size - chunk_size) // 2, 0), max(size - chunk_size, 0))
    digest = hashlib.sha256()
    digest.update(str(size).encode("ascii"))
    with path.open("rb") as handle:
        for offset in offsets:
            handle.seek(offset)
            digest.update(handle.read(chunk_size))
    return {"sha256": digest.hexdigest(), "sha256_mode": "sampled_first_mid_last_v1"}


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _metrics(
    events: list[dict[str, object]],
    safety: Mapping[str, object],
    dependency_audit_valid: bool,
    shadow_audit_valid: bool,
) -> dict[str, object]:
    accepted_packets = safety.get("accepted_packets")
    if (
        not isinstance(accepted_packets, int)
        or isinstance(accepted_packets, bool)
        or accepted_packets < 0
    ):
        raise ValueError("safety receipt accepted_packets must be a nonnegative integer")
    completed = [
        event
        for event in events
        if event.get("monitor_accepted") is True and event.get("terminal") is False
    ]
    if len(completed) > accepted_packets:
        raise ValueError("completed events exceed safety receipt accepted_packets")
    finite_estimates = [event for event in completed if _has_finite_estimate(event)]
    commits = [
        event
        for event in finite_estimates
        if _mapping(event.get("estimate"), "event estimate").get("commit") is True
    ]
    latencies = [int(event["processing_latency_ns"]) for event in completed]
    if len(latencies) != len(completed) or any(latency < 0 for latency in latencies):
        raise ValueError("accepted events have invalid processing latency")
    p50_latency_ns = float(np.percentile(latencies, 50)) if latencies else None
    p95_latency_ns = float(np.percentile(latencies, 95)) if latencies else None
    zero_command_status = all(
        event.get("robot_command_requests") == 0 and event.get("robot_commands_sent") == 0
        for event in events
    ) and safety.get("robot_command_requests") == 0 and safety.get("robot_commands_sent") == 0
    gates = {
        "accepted_packets_at_least_120": accepted_packets >= _CANONICAL_CAP,
        "finite_estimates_at_least_90": len(finite_estimates) >= 90,
        "no_terminal_rejection": safety.get("terminal_reason") is None,
        "p95_processing_latency_below_40ms": p95_latency_ns is not None
        and p95_latency_ns < 40_000_000,
        "zero_command_status": zero_command_status,
        "dependency_audit_valid": dependency_audit_valid,
        "shadow_audit_valid": shadow_audit_valid,
    }
    return {
        "schema": "kinesync.rt10_live_shadow_metrics.v1",
        "packet_count": accepted_packets,
        "completed_event_count": len(completed),
        "event_count": len(events),
        "finite_estimate_count": len(finite_estimates),
        "finite_event_rate": len(finite_estimates) / len(completed) if completed else 0.0,
        "commit_count": len(commits),
        "commit_rate": len(commits) / len(finite_estimates) if finite_estimates else 0.0,
        "p50_processing_latency_ns": p50_latency_ns,
        "p95_processing_latency_ns": p95_latency_ns,
        "max_head_auxiliary_skew_ns": safety["max_head_auxiliary_skew_ns"],
        "max_camera_state_skew_ns": safety["max_camera_state_skew_ns"],
        "zero_command_status": zero_command_status,
        "predeclared_gates": {**gates, "overall_pass": all(gates.values())},
    }


def _has_finite_estimate(event: Mapping[str, object]) -> bool:
    estimate = _mapping(event.get("estimate"), "event estimate")
    if estimate.get("available") is not True:
        return False
    return all(
        isinstance(estimate.get(field), (int, float))
        and not isinstance(estimate.get(field), bool)
        and math.isfinite(float(estimate[field]))
        for field in ("absolute_offset_ms", "relative_correction_ms")
    )


def _read_events(path: Path) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    for line in path.read_text(encoding="ascii").splitlines():
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError("events.jsonl rows must be JSON objects")
        events.append(payload)
    return events


def _read_json_object(path: Path, label: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} contains malformed JSON") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def _mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    return value


def _path(value: object, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a nonempty path")
    return Path(value).expanduser().resolve()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="RT10-P YAML configuration path")
    parser.add_argument("--run-id", default=None, help="immutable flat ASCII run ID")
    arguments = parser.parse_args()
    print(execute_rt10_live_shadow_preflight(load_config(arguments.config), run_id=arguments.run_id))


if __name__ == "__main__":
    main()
