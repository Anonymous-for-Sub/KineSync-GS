"""RealSense D455 color acquisition with host-monotonic timestamps."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


class RealSenseError(RuntimeError):
    """Raised when RealSense acquisition cannot safely continue."""


@dataclass(frozen=True)
class RealSenseFrame:
    rgb: np.ndarray
    capture_monotonic_ns: int
    device_timestamp_ms: float
    frame_number: int
    serial: str


class RealSenseColorReader:
    def __init__(
        self,
        *,
        serial: str,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        rs_module: Any | None = None,
        monotonic_clock_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        if not isinstance(serial, str) or not serial:
            raise ValueError("serial must be a nonempty string")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in (width, height, fps)
        ):
            raise ValueError("width, height, and fps must be positive integers")
        if not callable(monotonic_clock_ns):
            raise TypeError("monotonic_clock_ns must be callable")
        self.serial = serial
        self.width = width
        self.height = height
        self.fps = fps
        self._rs = rs_module
        self._clock = monotonic_clock_ns
        self._pipeline = None
        self._device_description: dict[str, str] | None = None

    def start(self) -> None:
        if self._pipeline is not None:
            return
        rs = self._rs or _load_realsense_module()
        self._rs = rs
        matching = [
            device
            for device in discover_realsense_devices(rs_module=rs)
            if device["serial"] == self.serial
        ]
        if not matching:
            raise ValueError(f"RealSense serial {self.serial!r} is not connected")
        if "D455" not in matching[0]["name"].upper():
            raise ValueError(
                f"RealSense serial {self.serial!r} is not a D455: {matching[0]['name']}"
            )
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(self.serial)
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        try:
            pipeline.start(config)
        except Exception:
            try:
                pipeline.stop()
            except Exception:
                pass
            raise
        self._pipeline = pipeline
        self._device_description = matching[0]

    def read_before(self, deadline_monotonic_ns: int) -> RealSenseFrame | None:
        if self._pipeline is None:
            raise RealSenseError("RealSense reader is not started")
        now_ns = _clock_ns(self._clock)
        if now_ns > deadline_monotonic_ns:
            return None
        timeout_ms = max(1, math.ceil((deadline_monotonic_ns - now_ns) / 1_000_000))
        try:
            frames = self._pipeline.wait_for_frames(timeout_ms)
        except RuntimeError as error:
            if "frame didn't arrive" in str(error).lower() or "timeout" in str(error).lower():
                return None
            raise RealSenseError(str(error)) from error
        color = frames.get_color_frame()
        if not color:
            return None
        capture_ns = _clock_ns(self._clock)
        if capture_ns > deadline_monotonic_ns:
            return None
        bgr = np.asanyarray(color.get_data())
        if bgr.dtype != np.uint8 or bgr.ndim != 3 or bgr.shape[2] != 3:
            raise RealSenseError("RealSense color frame is not uint8 BGR")
        rgb_copy = np.ascontiguousarray(bgr[:, :, ::-1])
        immutable_rgb = np.frombuffer(rgb_copy.tobytes(), dtype=np.uint8).reshape(rgb_copy.shape)
        return RealSenseFrame(
            rgb=immutable_rgb,
            capture_monotonic_ns=capture_ns,
            device_timestamp_ms=float(color.get_timestamp()),
            frame_number=int(color.get_frame_number()),
            serial=self.serial,
        )

    def describe(self) -> dict[str, object]:
        return {
            **(self._device_description or {"serial": self.serial}),
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "started": self._pipeline is not None,
        }

    def close(self) -> None:
        pipeline, self._pipeline = self._pipeline, None
        if pipeline is not None:
            pipeline.stop()

    def __enter__(self) -> "RealSenseColorReader":
        self.start()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


def discover_realsense_devices(*, rs_module: Any | None = None) -> list[dict[str, str]]:
    rs = rs_module or _load_realsense_module()
    keys = (
        ("serial", rs.camera_info.serial_number),
        ("name", rs.camera_info.name),
        ("firmware_version", rs.camera_info.firmware_version),
        ("usb_type", rs.camera_info.usb_type_descriptor),
    )
    result: list[dict[str, str]] = []
    for device in rs.context().query_devices():
        description = {
            name: str(device.get_info(key)) if device.supports(key) else "unknown"
            for name, key in keys
        }
        result.append(description)
    return result


def _load_realsense_module() -> Any:
    try:
        import pyrealsense2 as rs
    except ImportError as error:
        raise RealSenseError(
            "pyrealsense2 is unavailable; install the RealSense Python SDK on the hardware host"
        ) from error
    return rs


def _clock_ns(clock: Callable[[], int]) -> int:
    value = clock()
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise RealSenseError("monotonic clock returned an invalid timestamp")
    return value


__all__ = [
    "RealSenseColorReader",
    "RealSenseError",
    "RealSenseFrame",
    "discover_realsense_devices",
]
