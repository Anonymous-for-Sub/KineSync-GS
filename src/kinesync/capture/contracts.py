"""Closed, local-only RT10 capture, calibration, and evidence contracts."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta
import hashlib
import json
import math
from pathlib import Path
import re
from types import MappingProxyType
from typing import AbstractSet, Any, Mapping

import numpy as np
import yaml


_SCHEDULE_SCHEMA = "kinesync.rt10_capture_schedule.v1"
_CALIBRATION_SCHEMA = "kinesync.rt10_camera_calibration.v1"
_MANIFEST_SCHEMA = "kinesync.rt10_capture_manifest.v1"
_CAMERAS = ("head", "external")
_SLOT_ROLES = frozenset({"calibration", "formal", "reserve"})
_FORMAL_KINDS = frozenset({"single_joint", "zero_control"})
_UNSET = "UNSET"
_MAX_SKEW_NS = 35_000_000
_CANONICAL_SCHEDULE_FINGERPRINT = "17ff10f638f356e79d3ecbdc1ba31429956de8bf515fb410ad6f0cbf76b98590"
_SESSION_ID = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?")
_MANIFEST_SLOT_FIELDS = frozenset({
    "slot_id", "acquisition_status", "media", "state", "frame_counts",
    "timestamp_ranges_ns", "image_shape", "qpos_shape", "command_mode", "command_counters",
})


@dataclass(frozen=True)
class CaptureSlot:
    slot_id: str
    split: str
    role: str
    camera_order: tuple[str, str]
    pose_region: str
    visibility: str
    hold_duration_s: float
    command_mode: str
    calibration_emphasis: str | None
    formal_trial_ids: tuple[str, ...]


@dataclass(frozen=True)
class FormalTrial:
    trial_id: str
    slot_id: str
    kind: str
    joint_index: int
    sign: int
    magnitude_deg: float
    magnitude_rad: float
    seed: int
    command_mode: str


@dataclass(frozen=True)
class CaptureSchedule:
    source_path: Path
    slots: tuple[CaptureSlot, ...]
    formal_trials: tuple[FormalTrial, ...]
    role_counts: Mapping[str, int]
    fingerprint: str
    source_file_sha256: str


@dataclass(frozen=True)
class CalibrationReceipt:
    source_path: Path
    template_valid: bool
    acquisition_ready: bool
    source_file_sha256: str
    camera_serials: Mapping[str, str | None]


@dataclass(frozen=True)
class CaptureManifestReceipt:
    source_path: Path
    source_file_sha256: str
    session_id: str
    session_root: Path
    slot_ids: tuple[str, ...]
    schedule_sha256: str
    calibration_sha256: str


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_mapping(path: str | Path, label: str) -> tuple[Path, dict[str, Any]]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    with resolved.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    return resolved, value


def _closed_mapping(value: Any, allowed: AbstractSet[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    unknown = set(value) - allowed
    missing = allowed - set(value)
    if unknown:
        ordered_unknown = sorted(
            unknown,
            key=lambda key: (type(key).__module__, type(key).__qualname__, repr(key)),
        )
        raise ValueError(f"{label} has unknown keys: {ordered_unknown}")
    if missing:
        raise ValueError(f"{label} is missing keys: {sorted(missing)}")
    return value


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _acquisition_string(value: Any, label: str) -> str:
    text = _string(value, label)
    if text == _UNSET:
        raise ValueError(f"{label} must not be UNSET")
    return text


def _utc_iso8601(value: Any, label: str) -> str:
    text = _acquisition_string(value, label)
    if text.endswith("Z"):
        normalized = f"{text[:-1]}+00:00"
    elif text.endswith("+00:00"):
        normalized = text
    else:
        raise ValueError(f"{label} must be a timezone-aware UTC ISO-8601 timestamp")
    try:
        timestamp = datetime.fromisoformat(normalized)
    except ValueError as error:
        raise ValueError(
            f"{label} must be a timezone-aware UTC ISO-8601 timestamp"
        ) from error
    if timestamp.tzinfo is None or timestamp.utcoffset() != timedelta(0):
        raise ValueError(f"{label} must be a timezone-aware UTC ISO-8601 timestamp")
    return text


def _integer(value: Any, label: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{label} must be at least {minimum}")
    return value


def _number(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    if positive and number <= 0.0:
        raise ValueError(f"{label} must be positive")
    return number


def _sha256(value: Any, label: str) -> str:
    digest = _string(value, label)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{label} must be a lowercase full SHA-256")
    return digest


def _session_id(value: Any) -> str:
    session_id = _acquisition_string(value, "session_id")
    if not session_id.isascii() or not _SESSION_ID.fullmatch(session_id):
        raise ValueError("session_id must be a nonempty flat ASCII slug")
    return session_id


def _schedule_payload(slots: tuple[CaptureSlot, ...], trials: tuple[FormalTrial, ...]) -> dict[str, Any]:
    return {
        "slots": [
            {
                "slot_id": slot.slot_id,
                "split": slot.split,
                "role": slot.role,
                "camera_order": list(slot.camera_order),
                "pose_region": slot.pose_region,
                "visibility": slot.visibility,
                "hold_duration_s": slot.hold_duration_s,
                "command_mode": slot.command_mode,
                "calibration_emphasis": slot.calibration_emphasis,
                "formal_trial_ids": list(slot.formal_trial_ids),
            }
            for slot in slots
        ],
        "formal_trials": [
            {
                "trial_id": trial.trial_id,
                "slot_id": trial.slot_id,
                "kind": trial.kind,
                "joint_index": trial.joint_index,
                "sign": trial.sign,
                "magnitude_deg": trial.magnitude_deg,
                "magnitude_rad": trial.magnitude_rad,
                "seed": trial.seed,
                "command_mode": trial.command_mode,
            }
            for trial in trials
        ],
    }


def _fingerprint_schedule(slots: tuple[CaptureSlot, ...], trials: tuple[FormalTrial, ...]) -> str:
    encoded = json.dumps(
        _schedule_payload(slots, trials),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _parse_slot(value: Any) -> CaptureSlot:
    raw = _closed_mapping(
        value,
        {
            "slot_id", "split", "role", "camera_order", "pose_region", "visibility",
            "hold_duration_s", "command_mode", "calibration_emphasis", "formal_trial_ids",
        },
        "capture slot",
    )
    slot_id = _string(raw["slot_id"], "slot_id")
    split = _string(raw["split"], f"slot split for {slot_id}")
    role = _string(raw["role"], f"slot role for {slot_id}")
    if role not in _SLOT_ROLES or split != role:
        raise ValueError(f"slot role/split is invalid for {slot_id}")
    camera_order = raw["camera_order"]
    if not isinstance(camera_order, list) or tuple(camera_order) != _CAMERAS:
        raise ValueError(f"slot camera order must be head+external for {slot_id}")
    command_mode = _string(raw["command_mode"], f"slot command_mode for {slot_id}")
    if command_mode != "disabled":
        raise ValueError(f"slot command_mode must be disabled for {slot_id}")
    emphasis = raw["calibration_emphasis"]
    if role == "calibration":
        if emphasis not in {"head", "external"}:
            raise ValueError(f"calibration emphasis is invalid for {slot_id}")
    elif emphasis is not None:
        raise ValueError(f"non-calibration slot cannot have emphasis: {slot_id}")
    trial_ids = raw["formal_trial_ids"]
    if not isinstance(trial_ids, list) or any(not isinstance(item, str) or not item for item in trial_ids):
        raise ValueError(f"formal trial IDs are invalid for {slot_id}")
    if len(set(trial_ids)) != len(trial_ids):
        raise ValueError(f"formal trial IDs are duplicated for {slot_id}")
    if (role == "formal" and len(trial_ids) != 2) or (role != "formal" and trial_ids):
        raise ValueError(f"formal trial membership is invalid for {slot_id}")
    return CaptureSlot(
        slot_id=slot_id,
        split=split,
        role=role,
        camera_order=_CAMERAS,
        pose_region=_string(raw["pose_region"], f"pose_region for {slot_id}"),
        visibility=_string(raw["visibility"], f"visibility for {slot_id}"),
        hold_duration_s=_number(raw["hold_duration_s"], f"hold_duration_s for {slot_id}", positive=True),
        command_mode=command_mode,
        calibration_emphasis=emphasis,
        formal_trial_ids=tuple(trial_ids),
    )


def _parse_trial(value: Any) -> FormalTrial:
    raw = _closed_mapping(
        value,
        {
            "trial_id", "slot_id", "kind", "joint_index", "sign", "magnitude_deg",
            "magnitude_rad", "seed", "command_mode",
        },
        "formal trial",
    )
    trial_id = _string(raw["trial_id"], "trial_id")
    kind = _string(raw["kind"], f"trial kind for {trial_id}")
    if kind not in _FORMAL_KINDS:
        raise ValueError(f"trial kind is invalid for {trial_id}")
    joint_index = _integer(raw["joint_index"], f"joint_index for {trial_id}")
    if joint_index not in range(1, 7):
        raise ValueError(f"joint_index must be in 1..6 for {trial_id}")
    sign = _integer(raw["sign"], f"sign for {trial_id}")
    magnitude_deg = _number(raw["magnitude_deg"], f"magnitude_deg for {trial_id}")
    magnitude_rad = _number(raw["magnitude_rad"], f"magnitude_rad for {trial_id}")
    if kind == "single_joint":
        if sign not in {-1, 1} or magnitude_deg <= 0.0:
            raise ValueError(f"signed single-joint trial is invalid for {trial_id}")
    elif sign != 0 or magnitude_deg != 0.0 or magnitude_rad != 0.0:
        raise ValueError(f"zero control must be exactly zero for {trial_id}")
    if not math.isclose(magnitude_rad, math.radians(sign * magnitude_deg), abs_tol=1e-12):
        raise ValueError(f"degree/radian mismatch for {trial_id}")
    command_mode = _string(raw["command_mode"], f"trial command_mode for {trial_id}")
    if command_mode != "disabled":
        raise ValueError(f"trial command_mode must be disabled for {trial_id}")
    return FormalTrial(
        trial_id=trial_id,
        slot_id=_string(raw["slot_id"], f"slot_id for {trial_id}"),
        kind=kind,
        joint_index=joint_index,
        sign=sign,
        magnitude_deg=magnitude_deg,
        magnitude_rad=magnitude_rad,
        seed=_integer(raw["seed"], f"seed for {trial_id}", minimum=0),
        command_mode=command_mode,
    )


def load_capture_schedule(path: str | Path) -> CaptureSchedule:
    """Load the fully explicit immutable RT10 capture schedule."""

    schedule_path, raw = _load_mapping(path, "capture schedule")
    root = _closed_mapping(raw, {"schema", "command_mode", "slots", "formal_trials"}, "capture schedule")
    if root["schema"] != _SCHEDULE_SCHEMA or root["command_mode"] != "disabled":
        raise ValueError("capture schedule schema or command_mode is invalid")
    if not isinstance(root["slots"], list) or not isinstance(root["formal_trials"], list):
        raise ValueError("capture schedule slots and formal_trials must be lists")
    slots = tuple(_parse_slot(item) for item in root["slots"])
    trials = tuple(_parse_trial(item) for item in root["formal_trials"])
    if len(slots) != 36 or len({slot.slot_id for slot in slots}) != 36:
        raise ValueError("capture schedule requires exactly 36 unique slots")
    role_counts = Counter(slot.role for slot in slots)
    if role_counts != Counter({"calibration": 12, "formal": 16, "reserve": 8}):
        raise ValueError("capture schedule role counts must be 12 calibration, 16 formal, 8 reserve")
    calibration = [slot for slot in slots if slot.role == "calibration"]
    pairs = Counter((slot.pose_region, slot.calibration_emphasis) for slot in calibration)
    regions = {slot.pose_region for slot in calibration}
    if len(regions) != 6 or set(pairs.values()) != {1} or {emphasis for _, emphasis in pairs} != {"head", "external"}:
        raise ValueError("calibration slots must be six regions times head/external emphasis")
    if len(trials) != 32 or len({trial.trial_id for trial in trials}) != 32:
        raise ValueError("capture schedule requires exactly 32 unique formal trials")
    if len({trial.seed for trial in trials}) != 32:
        raise ValueError("formal trial deterministic seeds must be unique")
    formal_slots = {slot.slot_id: slot for slot in slots if slot.role == "formal"}
    if any(trial.slot_id not in formal_slots for trial in trials):
        raise ValueError("formal trials must belong only to formal slots")
    slot_membership = {trial_id for slot in formal_slots.values() for trial_id in slot.formal_trial_ids}
    if slot_membership != {trial.trial_id for trial in trials}:
        raise ValueError("formal slot membership must name every and only formal trial")
    for slot in formal_slots.values():
        if {trial.trial_id for trial in trials if trial.slot_id == slot.slot_id} != set(slot.formal_trial_ids):
            raise ValueError(f"formal slot membership differs for {slot.slot_id}")
    singles = [trial for trial in trials if trial.kind == "single_joint"]
    zeros = [trial for trial in trials if trial.kind == "zero_control"]
    if len(singles) != 26 or len(zeros) != 6:
        raise ValueError("formal trials must contain 26 signed single-joint and 6 zero controls")
    if {trial.joint_index for trial in singles} != set(range(1, 7)):
        raise ValueError("signed single-joint trials must represent joints 1-6")
    if Counter(trial.joint_index for trial in zeros) != Counter(range(1, 7)):
        raise ValueError("zero controls must represent every joint exactly once")
    fingerprint = _fingerprint_schedule(slots, trials)
    if fingerprint != _CANONICAL_SCHEDULE_FINGERPRINT:
        raise ValueError("capture schedule differs from the frozen semantic canonical schedule")
    return CaptureSchedule(
        source_path=schedule_path,
        slots=slots,
        formal_trials=trials,
        role_counts=MappingProxyType(dict(sorted(role_counts.items()))),
        fingerprint=fingerprint,
        source_file_sha256=_sha256_file(schedule_path),
    )


def _is_unset(value: Any) -> bool:
    return value == _UNSET


def _optional_string(value: Any, label: str) -> str | None:
    if _is_unset(value):
        return None
    return _string(value, label)


def _optional_number(value: Any, label: str, *, positive: bool = False) -> float | None:
    if _is_unset(value):
        return None
    return _number(value, label, positive=positive)


def _optional_vector(value: Any, length: int, label: str) -> tuple[float | None, ...] | None:
    if _is_unset(value):
        return None
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{label} must be a length-{length} list")
    return tuple(_optional_number(item, f"{label}[{index}]") for index, item in enumerate(value))


def _optional_matrix(value: Any, label: str) -> tuple[tuple[float | None, ...], ...] | None:
    if _is_unset(value):
        return None
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError(f"{label} must be a 4x4 matrix")
    rows = []
    for row_index, row in enumerate(value):
        if not isinstance(row, list) or len(row) != 4:
            raise ValueError(f"{label} must be a 4x4 matrix")
        rows.append(tuple(_optional_number(item, f"{label}[{row_index}][{column_index}]") for column_index, item in enumerate(row)))
    return tuple(rows)


def _fully_set(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, (list, tuple)):
        return all(_fully_set(item) for item in value)
    return True


def _resolve_local_asset(root: Path, raw_path: Any, label: str) -> Path:
    relative = _string(raw_path, f"{label} path")
    candidate = Path(relative)
    if candidate.is_absolute() or not candidate.parts or any(part in {"", ".", ".."} for part in candidate.parts):
        raise ValueError(f"{label} path must be relative and traversal-free")
    resolved_root = root.resolve()
    resolved = (resolved_root / candidate).resolve(strict=True)
    if not resolved.is_file() or not resolved.is_relative_to(resolved_root):
        raise ValueError(f"{label} path escapes declared root")
    return resolved


def _validate_camera(value: Any, name: str, root: Path) -> tuple[str | None, bool]:
    raw = _closed_mapping(
        value,
        {
            "serial", "image_size", "intrinsics", "distortion", "base_to_camera",
            "camera_to_base", "calibrated_at_utc", "method", "source_images",
        },
        f"{name} camera calibration",
    )
    serial = _optional_string(raw["serial"], f"{name} serial")
    size = _optional_vector(raw["image_size"], 2, f"{name} image_size")
    if size is not None:
        width, height = size
        if width is not None and (not width.is_integer() or width <= 0):
            raise ValueError(f"{name} image_size must be positive integers")
        if height is not None and (not height.is_integer() or height <= 0):
            raise ValueError(f"{name} image_size must be positive integers")
    intrinsics = _closed_mapping(raw["intrinsics"], {"fx", "fy", "cx", "cy"}, f"{name} intrinsics") if isinstance(raw["intrinsics"], Mapping) else None
    if intrinsics is None and not _is_unset(raw["intrinsics"]):
        raise ValueError(f"{name} intrinsics must be a mapping or UNSET")
    values = {key: _optional_number(intrinsics[key], f"{name} {key}", positive=key in {"fx", "fy"}) for key in ("fx", "fy", "cx", "cy")} if intrinsics is not None else {key: None for key in ("fx", "fy", "cx", "cy")}
    distortion = _optional_vector(raw["distortion"], 5, f"{name} distortion")
    base_to_camera = _optional_matrix(raw["base_to_camera"], f"{name} base_to_camera")
    camera_to_base = _optional_matrix(raw["camera_to_base"], f"{name} camera_to_base")
    timestamp = _optional_string(raw["calibrated_at_utc"], f"{name} calibrated_at_utc")
    method = _optional_string(raw["method"], f"{name} method")
    source_images_raw = raw["source_images"]
    source_images: list[tuple[Path | None, str | None]] | None
    if _is_unset(source_images_raw):
        source_images = None
    else:
        if not isinstance(source_images_raw, list):
            raise ValueError(f"{name} source_images must be a list or UNSET")
        source_images = []
        for index, source in enumerate(source_images_raw):
            entry = _closed_mapping(source, {"path", "sha256"}, f"{name} source image {index}")
            if _is_unset(entry["path"]) or _is_unset(entry["sha256"]):
                source_images.append((None, None))
            else:
                path = _resolve_local_asset(root, entry["path"], f"{name} source image {index}")
                source_images.append((path, _sha256(entry["sha256"], f"{name} source image {index} hash")))
    complete = all(
        _fully_set(item)
        for item in (serial, size, distortion, base_to_camera, camera_to_base, timestamp, method, source_images)
    ) and all(value is not None for value in values.values())
    if not complete:
        return serial, False
    assert size is not None and source_images is not None and base_to_camera is not None and camera_to_base is not None
    if not source_images:
        raise ValueError(f"{name} source_images must not be empty when populated")
    width, height = (int(size[0]), int(size[1]))
    if not (0.0 <= values["cx"] < width and 0.0 <= values["cy"] < height):
        raise ValueError(f"{name} principal point must lie inside the image")
    base = np.asarray(base_to_camera, dtype=np.float64)
    camera = np.asarray(camera_to_base, dtype=np.float64)
    if not np.isfinite(base).all() or not np.isfinite(camera).all() or not np.allclose(base @ camera, np.eye(4), atol=1e-8, rtol=0.0) or not np.allclose(camera @ base, np.eye(4), atol=1e-8, rtol=0.0):
        raise ValueError(f"{name} transforms must be finite mutual inverses")
    for path, expected_hash in source_images:
        assert path is not None and expected_hash is not None
        if _sha256_file(path) != expected_hash:
            raise ValueError(f"{name} source image hash does not match")
    return serial, True


def validate_camera_calibration(path: str | Path) -> CalibrationReceipt:
    """Validate an RT10 calibration template without guessing any hardware data."""

    calibration_path, raw = _load_mapping(path, "camera calibration")
    root = _closed_mapping(raw, {"schema", "cameras"}, "camera calibration")
    if root["schema"] != _CALIBRATION_SCHEMA:
        raise ValueError("camera calibration schema is invalid")
    cameras = _closed_mapping(root["cameras"], set(_CAMERAS), "camera calibration cameras")
    serials: dict[str, str | None] = {}
    ready: dict[str, bool] = {}
    for name in _CAMERAS:
        serials[name], ready[name] = _validate_camera(cameras[name], name, calibration_path.parent)
    if all(ready.values()) and serials["head"] == serials["external"]:
        raise ValueError("head and external camera serials must be distinct")
    return CalibrationReceipt(
        source_path=calibration_path,
        template_valid=True,
        acquisition_ready=all(ready.values()),
        source_file_sha256=_sha256_file(calibration_path),
        camera_serials=MappingProxyType(serials),
    )


def _asset(value: Any, label: str, session_root: Path) -> Path:
    raw = _closed_mapping(value, {"path", "sha256"}, label)
    path = _resolve_local_asset(session_root, raw["path"], label)
    if _sha256_file(path) != _sha256(raw["sha256"], f"{label} hash"):
        raise ValueError(f"{label} hash does not match")
    return path


def _load_npy(path: Path, label: str) -> np.ndarray:
    try:
        array = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ValueError(f"{label} is not a readable local NumPy payload") from error
    if not isinstance(array, np.ndarray):
        raise ValueError(f"{label} must be a NumPy array")
    return array


def _rgb_frames(array: np.ndarray, label: str) -> tuple[int, tuple[int, int, int]]:
    if array.dtype != np.uint8 or array.ndim not in {3, 4} or array.shape[-1] != 3 or any(size <= 0 for size in array.shape):
        raise ValueError(f"{label} must be uint8 HxWx3 RGB frames")
    if array.ndim == 3:
        return 1, tuple(int(size) for size in array.shape)
    return int(array.shape[0]), tuple(int(size) for size in array.shape[1:])


def _timestamps(array: np.ndarray, label: str) -> np.ndarray:
    if array.ndim != 1 or array.size == 0 or not np.issubdtype(array.dtype, np.integer) or np.issubdtype(array.dtype, np.bool_):
        raise ValueError(f"{label} must be a nonempty integer timestamp vector")
    if int(array.min()) < 0 or int(array.max()) > np.iinfo(np.int64).max:
        raise ValueError(f"{label} timestamp range must be within 0..int64 max")
    normalized = array.astype(np.int64, copy=False)
    if np.any(np.diff(normalized) <= 0):
        raise ValueError(f"{label} must be strictly increasing")
    return normalized


def _skew_exceeds(left: np.ndarray, right: np.ndarray) -> bool:
    return any(abs(int(left_value) - int(right_value)) > _MAX_SKEW_NS for left_value, right_value in zip(left, right, strict=True))


def _exact_shape(value: Any, expected: tuple[int, ...], label: str) -> None:
    if not isinstance(value, list) or len(value) != len(expected) or any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise ValueError(f"{label} is invalid")
    if tuple(value) != expected:
        raise ValueError(f"{label} does not match payload")


def _validate_manifest_slot(value: Any, schedule_slot: CaptureSlot, session_root: Path) -> None:
    raw = _closed_mapping(value, _MANIFEST_SLOT_FIELDS, "manifest slot")
    if raw["slot_id"] != schedule_slot.slot_id or raw["acquisition_status"] != "acquired":
        raise ValueError("manifest slot identity or acquisition status is invalid")
    if raw["command_mode"] != "disabled":
        raise ValueError("manifest slot command_mode must be disabled")
    counters = _closed_mapping(raw["command_counters"], {"requested", "sent"}, "manifest command counters")
    if _integer(counters["requested"], "requested command counter", minimum=0) != 0 or _integer(counters["sent"], "sent command counter", minimum=0) != 0:
        raise ValueError("manifest command counters must be zero")
    media = _closed_mapping(raw["media"], {"head_rgb", "external_rgb", "head_timestamps", "external_timestamps"}, "manifest media")
    state = _closed_mapping(raw["state"], {"qpos", "timestamps"}, "manifest state")
    head_frames, image_shape = _rgb_frames(_load_npy(_asset(media["head_rgb"], "head RGB", session_root), "head RGB"), "head RGB")
    external_frames, external_shape = _rgb_frames(_load_npy(_asset(media["external_rgb"], "external RGB", session_root), "external RGB"), "external RGB")
    if image_shape != external_shape:
        raise ValueError("head and external RGB image shapes differ")
    head_times = _timestamps(_load_npy(_asset(media["head_timestamps"], "head timestamps", session_root), "head timestamps"), "head timestamps")
    external_times = _timestamps(_load_npy(_asset(media["external_timestamps"], "external timestamps", session_root), "external timestamps"), "external timestamps")
    qpos = _load_npy(_asset(state["qpos"], "qpos", session_root), "qpos")
    qpos_is_real_numeric = np.issubdtype(qpos.dtype, np.integer) or np.issubdtype(qpos.dtype, np.floating)
    if qpos.ndim != 2 or qpos.shape[0] == 0 or qpos.shape[1] != 8 or not qpos_is_real_numeric or not np.isfinite(qpos).all():
        raise ValueError("qpos must be finite real numeric Nx8")
    state_times = _timestamps(_load_npy(_asset(state["timestamps"], "state timestamps", session_root), "state timestamps"), "state timestamps")
    if not (head_frames == external_frames == qpos.shape[0] == len(head_times) == len(external_times) == len(state_times)):
        raise ValueError("manifest frame counts do not match payloads")
    if any(_skew_exceeds(left, right) for left, right in ((head_times, external_times), (head_times, state_times), (external_times, state_times))):
        raise ValueError("per-frame camera/state skew exceeds 35ms")
    counts = _closed_mapping(raw["frame_counts"], {"head", "external", "state"}, "manifest frame_counts")
    if {name: _integer(counts[name], f"frame_count {name}", minimum=1) for name in counts} != {"head": head_frames, "external": external_frames, "state": int(qpos.shape[0])}:
        raise ValueError("manifest frame_counts do not match payload")
    ranges = _closed_mapping(raw["timestamp_ranges_ns"], {"head", "external", "state"}, "manifest timestamp_ranges_ns")
    actual_ranges = {"head": [int(head_times[0]), int(head_times[-1])], "external": [int(external_times[0]), int(external_times[-1])], "state": [int(state_times[0]), int(state_times[-1])]}
    for name, expected in actual_ranges.items():
        declared = ranges[name]
        if not isinstance(declared, list) or len(declared) != 2 or any(isinstance(item, bool) or not isinstance(item, int) for item in declared) or declared[0] > declared[1]:
            raise ValueError(f"timestamp range is invalid for {name}")
        if declared != expected:
            raise ValueError(f"timestamp range does not match payload for {name}")
    _exact_shape(raw["image_shape"], image_shape, "manifest image_shape")
    _exact_shape(raw["qpos_shape"], tuple(int(size) for size in qpos.shape), "manifest qpos_shape")


def _resolve_session_root(manifest_path: Path, value: Any) -> Path:
    relative = _string(value, "session_root")
    candidate = Path(relative)
    if candidate.is_absolute() or not candidate.parts or any(part in {"", ".", ".."} for part in candidate.parts):
        raise ValueError("session_root must be a relative traversal-free path")
    root = (manifest_path.parent / candidate).resolve(strict=True)
    if not root.is_dir() or not root.is_relative_to(manifest_path.parent.resolve()):
        raise ValueError("session_root escapes manifest directory")
    return root


def validate_capture_manifest(
    path: str | Path,
    schedule: CaptureSchedule | str | Path,
    calibration: CalibrationReceipt | str | Path,
) -> CaptureManifestReceipt:
    """Validate one fully local, command-disabled RT10 observation session."""

    active_schedule = load_capture_schedule(schedule) if isinstance(schedule, (str, Path)) else schedule
    active_calibration = validate_camera_calibration(calibration) if isinstance(calibration, (str, Path)) else calibration
    if not isinstance(active_schedule, CaptureSchedule) or not isinstance(active_calibration, CalibrationReceipt):
        raise TypeError("schedule and calibration must be RT10 contract receipts")
    if not active_calibration.acquisition_ready:
        raise ValueError("camera calibration is not acquisition-ready")
    manifest_path, raw = _load_mapping(path, "capture manifest")
    root = _closed_mapping(
        raw,
        {
            "schema", "project", "stage", "operator_acknowledgement", "robot_serial",
            "camera_serials", "host_clocks", "session_id", "session_root", "schedule_sha256",
            "calibration_sha256", "slots",
        },
        "capture manifest",
    )
    if root["schema"] != _MANIFEST_SCHEMA or root["project"] != "kinesync_gs" or root["stage"] != "RT10-P":
        raise ValueError("capture manifest project/stage/schema is invalid")
    acknowledgement = _closed_mapping(root["operator_acknowledgement"], {"operator_id", "acknowledged_at_utc"}, "operator acknowledgement")
    _acquisition_string(acknowledgement["operator_id"], "operator_id")
    _utc_iso8601(acknowledgement["acknowledged_at_utc"], "acknowledged_at_utc")
    _acquisition_string(root["robot_serial"], "robot_serial")
    camera_serials = _closed_mapping(root["camera_serials"], set(_CAMERAS), "camera_serials")
    if any(_string(camera_serials[name], f"{name} camera serial") != active_calibration.camera_serials[name] for name in _CAMERAS):
        raise ValueError("manifest camera serials do not match calibration")
    clocks = _closed_mapping(root["host_clocks"], {"wall_utc", "monotonic_domain"}, "host_clocks")
    _utc_iso8601(clocks["wall_utc"], "wall_utc")
    if _acquisition_string(clocks["monotonic_domain"], "monotonic_domain") != "host_monotonic_ns":
        raise ValueError("monotonic_domain must be the literal host_monotonic_ns")
    session_id = _session_id(root["session_id"])
    if _sha256(root["schedule_sha256"], "schedule_sha256") != active_schedule.source_file_sha256:
        raise ValueError("manifest schedule hash does not match")
    if _sha256(root["calibration_sha256"], "calibration_sha256") != active_calibration.source_file_sha256:
        raise ValueError("manifest calibration hash does not match")
    session_root = _resolve_session_root(manifest_path, root["session_root"])
    if not isinstance(root["slots"], list):
        raise ValueError("manifest slots must be a list")
    slot_rows = tuple(_closed_mapping(item, _MANIFEST_SLOT_FIELDS, "manifest slot") for item in root["slots"])
    actual_slots = tuple(_string(item["slot_id"], "manifest slot_id") for item in slot_rows)
    if len(set(actual_slots)) != len(actual_slots):
        raise ValueError("manifest slot IDs must not contain duplicates")
    schedule_by_id = {slot.slot_id: slot for slot in active_schedule.slots}
    if any(slot_id not in schedule_by_id for slot_id in actual_slots):
        raise ValueError("manifest contains an unknown schedule slot")
    schedule_positions = {slot.slot_id: index for index, slot in enumerate(active_schedule.slots)}
    positions = tuple(schedule_positions[slot_id] for slot_id in actual_slots)
    if positions != tuple(sorted(positions)):
        raise ValueError("manifest slots must retain canonical schedule order")
    required_formal = {slot.slot_id for slot in active_schedule.slots if slot.role == "formal"}
    if not required_formal <= set(actual_slots):
        raise ValueError("manifest must include all 16 mandatory formal slots")
    for slot in slot_rows:
        _validate_manifest_slot(slot, schedule_by_id[slot["slot_id"]], session_root)
    return CaptureManifestReceipt(
        source_path=manifest_path,
        source_file_sha256=_sha256_file(manifest_path),
        session_id=session_id,
        session_root=session_root,
        slot_ids=actual_slots,
        schedule_sha256=active_schedule.source_file_sha256,
        calibration_sha256=active_calibration.source_file_sha256,
    )


def materialize_formal_matrix_receipt(
    schedule: CaptureSchedule,
    calibration: CalibrationReceipt,
    manifest: CaptureManifestReceipt,
) -> dict[str, Any]:
    """Bind the exact frozen 32-trial matrix to validated local evidence."""

    if not isinstance(schedule, CaptureSchedule) or not isinstance(calibration, CalibrationReceipt) or not isinstance(manifest, CaptureManifestReceipt):
        raise TypeError("formal receipt inputs must be RT10 validator receipts")
    fresh_schedule = load_capture_schedule(schedule.source_path)
    fresh_calibration = validate_camera_calibration(calibration.source_path)
    fresh_manifest = validate_capture_manifest(
        manifest.source_path,
        schedule=fresh_schedule,
        calibration=fresh_calibration,
    )
    if schedule != fresh_schedule or calibration != fresh_calibration or manifest != fresh_manifest:
        raise ValueError("formal receipt inputs differ from freshly revalidated sources")
    if not fresh_calibration.acquisition_ready:
        raise ValueError("cannot materialize a formal receipt from incomplete calibration")
    if fresh_manifest.schedule_sha256 != fresh_schedule.source_file_sha256 or fresh_manifest.calibration_sha256 != fresh_calibration.source_file_sha256:
        raise ValueError("formal receipt inputs do not bind to the validated schedule/calibration")
    trials = _schedule_payload((), fresh_schedule.formal_trials)["formal_trials"]
    trial_fingerprint = hashlib.sha256(
        json.dumps(trials, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")
    ).hexdigest()
    return {
        "schema": "kinesync.rt10_formal_matrix_receipt.v1",
        "schedule_sha256": fresh_schedule.source_file_sha256,
        "schedule_fingerprint": fresh_schedule.fingerprint,
        "calibration_sha256": fresh_calibration.source_file_sha256,
        "manifest_sha256": fresh_manifest.source_file_sha256,
        "formal_trial_fingerprint": trial_fingerprint,
        "formal_trials": trials,
    }
