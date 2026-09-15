"""Discover read-only RT10 PiPER and RealSense hardware prerequisites."""

from __future__ import annotations

import argparse
import importlib.util
import importlib.metadata
import json
from pathlib import Path
from typing import Callable

import yaml

from kinesync.hardware.compatibility import probe_vendor_contract
from kinesync.hardware.realsense import discover_realsense_devices


_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_FROZEN_GUARD_PATH = _REPOSITORY_ROOT / "configs/rt10_frozen_temporal_guard.json"
_FROZEN_GUARD_SHA256 = (
    "398bf0b06f3bdf54fa001e9b2b9edf66476a4b31918ad34649b707843f72b89d"
)


def build_hardware_report(
    *,
    can_root: str | Path = "/sys/class/net",
    dependency_probe: Callable[[str], bool] | None = None,
    dependency_version: Callable[[str], str] | None = None,
    contract_probe: Callable[[str], dict[str, object]] | None = None,
    realsense_discover: Callable[[], list[dict[str, str]]] = discover_realsense_devices,
) -> dict[str, object]:
    probe = dependency_probe or _dependency_available
    version = dependency_version or _dependency_version
    dependencies = {name: bool(probe(name)) for name in ("piper_sdk", "pyrealsense2")}
    dependency_versions = {
        name: version(name) if available else "unavailable"
        for name, available in dependencies.items()
    }
    inspect_contract = contract_probe or probe_vendor_contract
    sdk_contracts: dict[str, dict[str, object]] = {}
    for name, available in dependencies.items():
        if not available:
            sdk_contracts[name] = {
                "compatible": False,
                "version": "unavailable",
                "missing": ["module"],
            }
            continue
        try:
            sdk_contracts[name] = inspect_contract(name)
        except Exception as error:
            sdk_contracts[name] = {
                "compatible": False,
                "version": dependency_versions[name],
                "missing": [],
                "probe_error": f"{type(error).__name__}: {error}",
            }
    root = Path(can_root)
    can_interfaces = []
    if root.is_dir():
        for path in sorted(root.iterdir()):
            if not path.name.startswith(("can", "vcan")):
                continue
            state_path = path / "operstate"
            state = state_path.read_text(encoding="ascii").strip() if state_path.is_file() else "unknown"
            can_interfaces.append({"name": path.name, "operstate": state})
    devices: list[dict[str, str]] = []
    discovery_error = None
    if dependencies["pyrealsense2"]:
        try:
            devices = realsense_discover()
        except Exception as error:
            discovery_error = f"{type(error).__name__}: {error}"
    d455_devices = [device for device in devices if "D455" in device.get("name", "").upper()]
    blockers = []
    for name, available in dependencies.items():
        if not available:
            blockers.append(f"missing Python dependency: {name}")
        elif sdk_contracts[name].get("compatible") is not True:
            blockers.append(f"incompatible Python SDK: {name}")
    if not can_interfaces:
        blockers.append("no CAN interface found")
    elif not any(row["operstate"] in {"up", "unknown"} for row in can_interfaces):
        blockers.append("no active CAN interface found")
    if discovery_error:
        blockers.append(f"RealSense discovery failed: {discovery_error}")
    elif not d455_devices:
        blockers.append("no RealSense D455 found")
    return {
        "schema": "kinesync.rt10_hardware_discovery.v1",
        "ready": not blockers,
        "dependencies": dependencies,
        "dependency_versions": dependency_versions,
        "sdk_contracts": sdk_contracts,
        "can_interfaces": can_interfaces,
        "realsense_devices": devices,
        "d455_count": len(d455_devices),
        "blockers": blockers,
    }


def _dependency_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _dependency_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def build_live_config(
    report: dict[str, object],
    *,
    mode: str = "auto",
    head_serial: str | None = None,
    auxiliary_serial: str | None = None,
    packet_count: int = 120,
    can_name: str | None = None,
    run_root: str | Path = "hardware-runs/rt10",
) -> dict[str, object]:
    if report.get("ready") is not True:
        raise ValueError("hardware discovery must be ready before generating a config")
    interfaces = report.get("can_interfaces")
    if not isinstance(interfaces, list):
        raise ValueError("hardware discovery has no CAN interface list")
    active = sorted(
        (
            row
            for row in interfaces
            if isinstance(row, dict) and row.get("operstate") in {"up", "unknown"}
        ),
        key=lambda row: str(row.get("name", "")),
    )
    if not active or not isinstance(active[0].get("name"), str):
        raise ValueError("hardware discovery has no active CAN interface")
    active_names = [row["name"] for row in active]
    selected_can = can_name or active_names[0]
    if selected_can not in active_names:
        raise ValueError(f"active CAN interface was not discovered: {selected_can}")

    devices = report.get("realsense_devices")
    if not isinstance(devices, list):
        raise ValueError("hardware discovery has no RealSense device list")
    d455 = sorted(
        (
            device
            for device in devices
            if isinstance(device, dict)
            and "D455" in str(device.get("name", "")).upper()
            and isinstance(device.get("serial"), str)
            and device["serial"]
        ),
        key=lambda device: device["serial"],
    )
    if not d455:
        raise ValueError("hardware discovery has no identified D455 serial")
    serials = [device["serial"] for device in d455]
    selected_mode = "dual" if mode == "auto" and len(d455) >= 2 else mode
    if selected_mode == "auto":
        selected_mode = "single_preflight"
    if selected_mode not in {"single_preflight", "dual"}:
        raise ValueError("mode must be auto, single_preflight, or dual")
    selected_head = head_serial or serials[0]
    if selected_head not in serials:
        raise ValueError(f"head D455 serial was not discovered: {selected_head}")
    selected_auxiliary = None
    if selected_mode == "dual":
        candidates = [serial for serial in serials if serial != selected_head]
        selected_auxiliary = auxiliary_serial or (candidates[0] if candidates else None)
        if selected_auxiliary not in serials:
            raise ValueError(
                f"auxiliary D455 serial was not discovered: {selected_auxiliary}"
            )
        if selected_auxiliary == selected_head:
            raise ValueError("dual mode requires distinct D455 serials")
    elif auxiliary_serial is not None:
        raise ValueError("single_preflight does not accept an auxiliary D455 serial")
    if isinstance(packet_count, bool) or packet_count < 26:
        raise ValueError("packet_count must be an integer >= 26")
    selected_run_root = str(run_root).strip()
    if not selected_run_root:
        raise ValueError("run_root must be a nonempty path")
    return {
        "command_mode": "disabled",
        "mode": selected_mode,
        "robot": {
            "can_name": selected_can,
            "max_gripper_stroke_mm": 70.0,
            "finger_travel_m": 0.05,
        },
        "cameras": {
            "head_serial": selected_head,
            "auxiliary_serial": selected_auxiliary,
            "width": 1280,
            "height": 720,
            "fps": 30,
        },
        "capture": {
            "packet_count": packet_count,
            "packet_timeout_ms": 500,
            "sample_count": 12,
            "hardware_warmup_s": 2.0,
            "feedback_ready_timeout_s": 5.0,
        },
        "monitor": {
            "max_head_auxiliary_skew_ms": 50,
            "max_camera_state_skew_ms": 50,
        },
        "guard": {
            "path": str(_FROZEN_GUARD_PATH),
            "sha256": _FROZEN_GUARD_SHA256,
        },
        "run_root": selected_run_root,
    }


def write_live_config(path: str | Path, config: dict[str, object]) -> Path:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, sort_keys=False)
    return target


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Discover read-only PiPER CAN and RealSense D455 hardware."
    )
    parser.add_argument(
        "--write-config",
        type=Path,
        help="write a ready-to-review RT10 live YAML configuration",
    )
    parser.add_argument(
        "--mode",
        choices=("auto", "single_preflight", "dual"),
        default="auto",
        help="camera mode for the generated config (default: auto)",
    )
    parser.add_argument(
        "--head-serial",
        help="discovered D455 serial assigned to the primary camera",
    )
    parser.add_argument(
        "--auxiliary-serial",
        help="discovered D455 serial assigned to the auxiliary camera in dual mode",
    )
    parser.add_argument(
        "--packet-count",
        type=int,
        default=120,
        help="number of observation packets to capture (default: 120)",
    )
    parser.add_argument(
        "--can-name",
        help="discovered active CAN interface assigned to PiPER",
    )
    parser.add_argument(
        "--run-root",
        default="hardware-runs/rt10",
        help="target-host directory for captured sessions",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _parser()
    args = parser.parse_args(argv)
    report = build_hardware_report()
    exit_code = 0 if report["ready"] else 2
    if args.write_config is not None and report["ready"]:
        try:
            config = build_live_config(
                report,
                mode=args.mode,
                head_serial=args.head_serial,
                auxiliary_serial=args.auxiliary_serial,
                packet_count=args.packet_count,
                can_name=args.can_name,
                run_root=args.run_root,
            )
            target = write_live_config(args.write_config, config)
        except FileExistsError:
            parser.error(f"configuration already exists: {args.write_config}")
        except ValueError as error:
            parser.error(str(error))
        report = {
            **report,
            "generated_config": str(target),
            "camera_assignment": config["cameras"],
        }
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True))
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
