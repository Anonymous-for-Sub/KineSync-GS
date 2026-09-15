"""CPU-only schema audit helpers for native external robot recordings."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Iterator

import h5py
import numpy as np


def _media_binary(name: str) -> str | None:
    configured = os.environ.get(f"KINESYNC_{name.upper()}")
    return configured if configured and Path(configured).is_file() else shutil.which(name.lower())


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest without loading an artifact into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _datasets(group: h5py.Group, prefix: str = "") -> Iterator[tuple[str, h5py.Dataset]]:
    for name, item in group.items():
        path = f"{prefix}/{name}" if prefix else name
        if isinstance(item, h5py.Dataset):
            yield path, item
        elif isinstance(item, h5py.Group):
            yield from _datasets(item, path)


def _json_scalar(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _timestamp_summary(dataset: h5py.Dataset) -> dict[str, Any]:
    values = np.asarray(dataset[()]).reshape(-1)
    numeric = bool(np.issubdtype(values.dtype, np.number))
    strictly_increasing = bool(numeric and values.size > 1 and np.all(np.diff(values) > 0))
    return {
        "shape": list(dataset.shape),
        "dtype": str(dataset.dtype),
        "count": int(values.size),
        "strictly_increasing": strictly_increasing,
        "first": _json_scalar(values[0]) if values.size else None,
        "last": _json_scalar(values[-1]) if values.size else None,
    }


def _matrix_from_intrinsics(values: Any) -> list[list[float]] | None:
    if isinstance(values, dict):
        values = values.get("cameraMatrix")
    if not isinstance(values, (list, tuple)) or len(values) != 4:
        return None
    fx, cx, fy, cy = (float(value) for value in values)
    return [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]


def _calibration_entries(root: Path | None, episode_id: str | None) -> dict[str, Any]:
    if root is None or episode_id is None:
        return {"status": "not_requested"}
    result: dict[str, Any] = {"status": "missing"}
    intrinsics_file = root / "intrinsics.json"
    if intrinsics_file.is_file():
        intrinsics = json.loads(intrinsics_file.read_text(encoding="utf-8"))
        values = intrinsics.get(episode_id)
        if values is not None:
            result["intrinsics"] = {
                str(serial): {"values": raw, "K": _matrix_from_intrinsics(raw)}
                for serial, raw in values.items()
            }
    for filename in ("cam2base_extrinsic_superset.json", "cam2base_extrinsics.json"):
        path = root / filename
        if not path.is_file():
            continue
        entries = json.loads(path.read_text(encoding="utf-8"))
        values = entries.get(episode_id)
        if values is not None:
            result["camera_to_base"] = {
                str(serial): {"values": raw, "dimension": len(raw) if isinstance(raw, list) else None}
                for serial, raw in values.items()
                if isinstance(raw, list)
            }
            result["camera_to_base_source"] = filename
            break
    if "intrinsics" in result or "camera_to_base" in result:
        result["status"] = "matched"
    return result


def _native_video_info(path: Path) -> dict[str, Any]:
    command = [
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
        "stream=width,height,avg_frame_rate,nb_frames,duration", "-of", "json", str(path),
    ]
    ffprobe = _media_binary("ffprobe")
    if ffprobe is None:
        return {"path": str(path), "status": "ffprobe_unavailable"}
    command[0] = ffprobe
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        return {"path": str(path), "status": "ffprobe_failed", "error": completed.stderr.strip()}
    streams = json.loads(completed.stdout).get("streams", [])
    stream = streams[0] if streams else {}
    timestamps = path.with_name(f"{path.stem}_timestamps.json")
    timestamp_count: int | None = None
    if timestamps.is_file():
        try:
            timestamp_count = len(json.loads(timestamps.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            timestamp_count = None
    return {
        "path": str(path),
        "status": "ok",
        "native_width": stream.get("width"),
        "native_height": stream.get("height"),
        "avg_frame_rate": stream.get("avg_frame_rate"),
        "frame_count_container": stream.get("nb_frames"),
        "duration_seconds": stream.get("duration"),
        "timestamp_sidecar": str(timestamps) if timestamps.is_file() else None,
        "timestamp_count": timestamp_count,
        "sha256": sha256_file(path),
    }


def _episode_id(episode_dir: Path) -> str | None:
    for metadata in sorted(episode_dir.glob("metadata_*.json")):
        match = re.fullmatch(r"metadata_(.+)\.json", metadata.name)
        if match:
            return match.group(1)
    return None


def extract_native_frames(episode_dir: Path, output_dir: Path, frame_index: int = 0) -> list[Path]:
    """Decode one frame per native MP4 using CPU ffmpeg, without resizing the source."""
    ffmpeg = _media_binary("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to extract native review frames")
    output_dir.mkdir(parents=True, exist_ok=True)
    frames: list[Path] = []
    for video in sorted((episode_dir / "recordings" / "MP4").glob("*.mp4")):
        target = output_dir / f"{video.stem}_frame{frame_index:06d}.png"
        select = f"select=eq(n\\,{frame_index})"
        completed = subprocess.run(
            [ffmpeg, "-y", "-v", "error", "-i", str(video), "-vf", select, "-vframes", "1", str(target)],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0 or not target.is_file():
            raise RuntimeError(f"ffmpeg failed for {video}: {completed.stderr.strip()}")
        frames.append(target)
    return frames


def audit_droid_episode(
    episode_dir: Path,
    *,
    calibration_root: Path | None = None,
    frame_output_dir: Path | None = None,
    frame_index: int = 0,
) -> dict[str, Any]:
    """Audit a native DROID episode without changing any source artifact."""
    episode_dir = Path(episode_dir)
    trajectory = episode_dir / "trajectory.h5"
    if not trajectory.is_file():
        raise FileNotFoundError(f"missing native DROID trajectory: {trajectory}")

    timestamps: dict[str, Any] = {}
    joint_dimensions: dict[str, int] = {}
    hdf5_camera_extrinsics: dict[str, Any] = {}
    datasets: dict[str, Any] = {}
    with h5py.File(trajectory, "r") as handle:
        for path, dataset in _datasets(handle):
            datasets[path] = {"shape": list(dataset.shape), "dtype": str(dataset.dtype)}
            leaf = path.rsplit("/", 1)[-1]
            if "timestamp" in path.lower():
                timestamps[path] = _timestamp_summary(dataset)
            if leaf in {"joint_position", "joint_positions"} and dataset.ndim >= 1:
                joint_dimensions[path] = int(dataset.shape[-1])
        extrinsics = handle.get("observation/camera_extrinsics")
        if isinstance(extrinsics, h5py.Group):
            for name, dataset in extrinsics.items():
                if isinstance(dataset, h5py.Dataset):
                    hdf5_camera_extrinsics[name] = {
                        "shape": list(dataset.shape),
                        "dimension": int(dataset.shape[-1]) if dataset.ndim else None,
                    }
        attributes = {str(key): _json_scalar(value) for key, value in handle.attrs.items()}

    video_root = episode_dir / "recordings" / "MP4"
    native_video = [_native_video_info(path) for path in sorted(video_root.glob("*.mp4"))] if video_root.is_dir() else []
    frames = []
    if frame_output_dir is not None and native_video:
        frames = [str(path) for path in extract_native_frames(episode_dir, Path(frame_output_dir), frame_index)]

    return {
        "schema": "kinesync.external_raw_droid_audit.v1",
        "dataset": "DROID",
        "episode_path": str(episode_dir),
        "episode_id": _episode_id(episode_dir),
        "hdf5_attributes": attributes,
        "hdf5_datasets": datasets,
        "joint_dimensions": joint_dimensions,
        "timestamps": timestamps,
        "hdf5_camera_extrinsics": hdf5_camera_extrinsics,
        "supplementary_calibration": _calibration_entries(calibration_root, _episode_id(episode_dir)),
        "native_video": native_video,
        "extracted_native_frames": frames,
        "files": {"trajectory.h5": {"path": str(trajectory), "sha256": sha256_file(trajectory)}},
    }
