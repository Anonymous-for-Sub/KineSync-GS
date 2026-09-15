"""Typed access to DROID HDF5 state and timestamp streams.

This module intentionally does not assign a video frame index to any HDF5 row.
That relationship requires the original SVO image timestamps to be verified.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np


@dataclass(frozen=True)
class DroidHdf5Trajectory:
    """Native DROID state and clock streams with their source units preserved."""

    source: Path
    joint_positions: np.ndarray
    gripper_position: np.ndarray
    robot_timestamp_ns: np.ndarray
    camera_timestamps_ms: dict[str, dict[str, np.ndarray]]

    @property
    def robot_timestamp_strictly_increasing(self) -> bool:
        return bool(np.all(np.diff(self.robot_timestamp_ns) > 0))


def _trajectory_path(path: Path) -> Path:
    path = Path(path)
    return path / "trajectory.h5" if path.is_dir() else path


def _read_trajectory_array(handle: h5py.File, name: str, length: int) -> np.ndarray:
    if name not in handle:
        raise KeyError(f"DROID HDF5 field is missing: {name}")
    values = np.asarray(handle[name][()])
    if values.shape[0] != length:
        raise ValueError(f"DROID HDF5 field has unexpected length: {name}")
    return values


def _camera_timestamps(handle: h5py.File, length: int) -> dict[str, dict[str, np.ndarray]]:
    group = handle.get("observation/timestamp/cameras")
    if not isinstance(group, h5py.Group):
        raise KeyError("DROID HDF5 camera timestamp group is missing")
    result: dict[str, dict[str, np.ndarray]] = {}
    for name, dataset in group.items():
        if not isinstance(dataset, h5py.Dataset) or "_" not in name:
            continue
        serial, signal = name.split("_", 1)
        values = np.asarray(dataset[()])
        if values.ndim != 1 or values.shape[0] != length:
            raise ValueError(f"DROID camera timestamp has unexpected shape: {name}")
        result.setdefault(serial, {})[signal] = values
    if not result:
        raise ValueError("DROID HDF5 contains no usable camera timestamp streams")
    return result


def load_droid_hdf5_trajectory(path: Path) -> DroidHdf5Trajectory:
    """Load native 7-DoF state, gripper, robot clock, and per-camera clocks.

    Robot seconds/nanos are composed into an epoch-nanosecond integer. Camera
    timestamps retain their source millisecond unit and are not interpolated.
    """
    source = _trajectory_path(path)
    with h5py.File(source, "r") as handle:
        joints = np.asarray(handle["observation/robot_state/joint_positions"][()])
        if joints.ndim != 2 or joints.shape[1] != 7:
            raise ValueError("DROID joint_positions must have shape (T, 7)")
        length = joints.shape[0]
        gripper = _read_trajectory_array(handle, "observation/robot_state/gripper_position", length)
        if gripper.ndim != 1:
            raise ValueError("DROID gripper_position must have shape (T,)")
        seconds = _read_trajectory_array(
            handle, "observation/timestamp/robot_state/robot_timestamp_seconds", length
        ).astype(np.int64)
        nanos = _read_trajectory_array(
            handle, "observation/timestamp/robot_state/robot_timestamp_nanos", length
        ).astype(np.int64)
        if np.any(nanos < 0) or np.any(nanos >= 1_000_000_000):
            raise ValueError("DROID robot_timestamp_nanos is outside [0, 1e9)")
        robot_timestamp_ns = seconds * 1_000_000_000 + nanos
        if not np.all(np.diff(robot_timestamp_ns) > 0):
            raise ValueError("DROID composed robot timestamps must be strictly increasing")
        cameras = _camera_timestamps(handle, length)
    return DroidHdf5Trajectory(
        source=source,
        joint_positions=joints,
        gripper_position=gripper,
        robot_timestamp_ns=robot_timestamp_ns,
        camera_timestamps_ms=cameras,
    )
