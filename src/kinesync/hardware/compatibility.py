"""Side-effect-free API contract checks for hardware vendor SDKs."""

from __future__ import annotations

import importlib.metadata
import inspect
from types import ModuleType
from typing import Any


_PIPER_CONSTRUCTOR_PARAMETERS = (
    "can_name",
    "judge_flag",
    "can_auto_init",
    "start_sdk_joint_limit",
    "start_sdk_gripper_limit",
)
_PIPER_METHODS = (
    "ParseCANFrame",
    "GetArmJointMsgs",
    "GetArmGripperMsgs",
    "DisconnectPort",
)
_REALSENSE_MODULE_ATTRIBUTES = ("pipeline", "config", "context")
_REALSENSE_NESTED_ATTRIBUTES = (
    ("stream", "color"),
    ("format", "bgr8"),
    ("camera_info", "serial_number"),
    ("camera_info", "name"),
    ("camera_info", "firmware_version"),
    ("camera_info", "usb_type_descriptor"),
)


def probe_piper_sdk_contract(
    *, module: Any | None = None, version: str | None = None
) -> dict[str, object]:
    """Check the exact read-only PiPER API used by ``PiperFeedbackReader``."""

    sdk = module or _import_piper_sdk()
    interface = getattr(sdk, "C_PiperInterface_V2", None)
    missing: list[str] = []
    if interface is None:
        missing.append("C_PiperInterface_V2")
    else:
        constructor = _signature_parameters(interface)
        missing.extend(
            name
            for name in _PIPER_CONSTRUCTOR_PARAMETERS
            if not _accepts_parameter(constructor, name)
        )
        for name in _PIPER_METHODS:
            if not callable(getattr(interface, name, None)):
                missing.append(name)
        connect_port = getattr(interface, "ConnectPort", None)
        if not callable(connect_port):
            missing.append("ConnectPort")
        elif not _accepts_parameter(_signature_parameters(connect_port), "piper_init"):
            missing.append("ConnectPort.piper_init")
    return {
        "compatible": not missing,
        "version": version or _package_version("piper_sdk"),
        "implementation": "C_PiperInterface_V2",
        "module_path": _module_path(sdk),
        "connect_port_piper_init": "ConnectPort.piper_init" not in missing,
        "missing": sorted(set(missing)),
    }


def probe_realsense_contract(
    *, module: Any | None = None, version: str | None = None
) -> dict[str, object]:
    """Check the pyrealsense2 symbols needed for D455 discovery and RGB capture."""

    sdk = module or _import_realsense()
    missing = [name for name in _REALSENSE_MODULE_ATTRIBUTES if not hasattr(sdk, name)]
    for parent_name, child_name in _REALSENSE_NESTED_ATTRIBUTES:
        parent = getattr(sdk, parent_name, None)
        if parent is None or not hasattr(parent, child_name):
            missing.append(f"{parent_name}.{child_name}")
    return {
        "compatible": not missing,
        "version": version or _package_version("pyrealsense2"),
        "implementation": "pyrealsense2",
        "module_path": _module_path(sdk),
        "missing": sorted(set(missing)),
    }


def probe_vendor_contract(name: str) -> dict[str, object]:
    if name == "piper_sdk":
        return probe_piper_sdk_contract()
    if name == "pyrealsense2":
        return probe_realsense_contract()
    raise ValueError(f"unsupported hardware SDK: {name}")


def _signature_parameters(callable_object: Any) -> dict[str, inspect.Parameter]:
    try:
        return dict(inspect.signature(callable_object).parameters)
    except (TypeError, ValueError):
        return {}


def _accepts_parameter(
    parameters: dict[str, inspect.Parameter], name: str
) -> bool:
    return name in parameters or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _module_path(module: Any) -> str:
    value = getattr(module, "__file__", None)
    return "unknown" if value is None else str(value)


def _import_piper_sdk() -> ModuleType:
    import piper_sdk

    return piper_sdk


def _import_realsense() -> ModuleType:
    import pyrealsense2

    return pyrealsense2


__all__ = [
    "probe_piper_sdk_contract",
    "probe_realsense_contract",
    "probe_vendor_contract",
]
