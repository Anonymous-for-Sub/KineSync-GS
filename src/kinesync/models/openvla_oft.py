"""Read-only OpenVLA-OFT checkpoint and action-interface validation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np


_REQUIRED_FILES = (
    "action_head--30000_checkpoint.pt",
    "added_tokens.json",
    "config.json",
    "configuration_prismatic.py",
    "dataset_statistics.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "modeling_prismatic.py",
    "preprocessor_config.json",
    "processing_prismatic.py",
    "processor_config.json",
    "proprio_projector--30000_checkpoint.pt",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
)
_STAT_FIELDS = ("mean", "std", "min", "max", "q01", "q99")


def _fingerprint(path: Path, *, sample_bytes: int = 1024 * 1024) -> dict[str, object]:
    size = path.stat().st_size
    if size <= 64 * 1024 * 1024:
        mode = "full_sha256"
        offsets = (0,)
    else:
        mode = "sampled_sha256_first_middle_last"
        offsets = tuple(sorted({0, max(0, size // 2 - sample_bytes // 2), max(0, size - sample_bytes)}))
    digest = hashlib.sha256()
    digest.update(str(size).encode("ascii"))
    with path.open("rb") as handle:
        for offset in offsets:
            handle.seek(offset)
            block = handle.read(sample_bytes if mode != "full_sha256" else size)
            digest.update(str(offset).encode("ascii"))
            digest.update(block)
    return {
        "name": path.name,
        "byte_size": size,
        "fingerprint_mode": mode,
        "sha256": digest.hexdigest(),
    }


def _seven(values: object, *, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.shape != (7,):
        raise ValueError(f"OpenVLA-OFT {name} must be 7D")
    if not (np.issubdtype(array.dtype, np.integer) or np.issubdtype(array.dtype, np.floating)):
        raise ValueError(f"OpenVLA-OFT {name} must be numeric")
    array = array.astype(np.float64, copy=False)
    if not np.isfinite(array).all():
        raise ValueError(f"OpenVLA-OFT {name} must be finite")
    return array


def _validate_statistics(channel: Mapping[str, object], *, name: str) -> None:
    values = {field: _seven(channel.get(field), name=f"{name}.{field}") for field in _STAT_FIELDS}
    if np.any(values["std"] <= 0):
        raise ValueError(f"OpenVLA-OFT {name}.std must be positive")
    if np.any(values["min"] > values["q01"]) or np.any(values["q01"] > values["q99"]):
        raise ValueError(f"OpenVLA-OFT {name} quantiles are inconsistent")
    if np.any(values["q99"] > values["max"]):
        raise ValueError(f"OpenVLA-OFT {name} quantiles exceed extrema")


def audit_openvla_oft_checkpoint(
    checkpoint: str | Path, unnorm_key: str
) -> dict[str, object]:
    """Validate a local checkpoint without importing Transformers or Torch."""

    root = Path(checkpoint).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    for name in _REQUIRED_FILES:
        path = root / name
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)

    index = json.loads((root / "model.safetensors.index.json").read_text(encoding="utf-8"))
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("OpenVLA-OFT model index has no weight_map")
    shards = tuple(sorted({str(name) for name in weight_map.values()}))
    for name in shards:
        if Path(name).name != name or not name.endswith(".safetensors"):
            raise ValueError(f"Invalid OpenVLA-OFT shard name: {name}")
        path = root / name
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)

    statistics = json.loads((root / "dataset_statistics.json").read_text(encoding="utf-8"))
    try:
        task_statistics = statistics[unnorm_key]
    except KeyError as error:
        raise ValueError(f"Missing OpenVLA-OFT unnorm_key: {unnorm_key}") from error
    if not isinstance(task_statistics, dict):
        raise ValueError("OpenVLA-OFT task statistics must be a mapping")
    for channel_name in ("action", "proprio"):
        channel = task_statistics.get(channel_name)
        if not isinstance(channel, dict):
            raise ValueError(f"Missing OpenVLA-OFT {channel_name} statistics")
        _validate_statistics(channel, name=channel_name)
    mask = task_statistics["action"].get("mask")
    if not isinstance(mask, list) or len(mask) != 7 or not all(isinstance(value, bool) for value in mask):
        raise ValueError("OpenVLA-OFT action.mask must be seven booleans")
    trajectories = int(task_statistics.get("num_trajectories", 0))
    transitions = int(task_statistics.get("num_transitions", 0))
    if trajectories < 1 or transitions < trajectories:
        raise ValueError("OpenVLA-OFT trajectory/transition counts are invalid")

    paths = {root / name for name in _REQUIRED_FILES}
    paths.update(root / name for name in shards)
    files = [_fingerprint(path) for path in sorted(paths, key=lambda value: value.name)]
    return {
        "schema": "kinesync.openvla_oft_checkpoint_audit.v1",
        "valid": True,
        "checkpoint": str(root),
        "unnorm_key": unnorm_key,
        "action_dimension": 7,
        "proprio_dimension": 7,
        "num_trajectories": trajectories,
        "num_transitions": transitions,
        "model_shard_count": len(shards),
        "command_mode": "disabled",
        "files": files,
    }


def validate_action_chunk(
    actions: object,
    action_statistics: Mapping[str, object],
    *,
    envelope: str = "q01_q99",
) -> dict[str, object]:
    """Validate and envelope an offline prediction; never execute it."""

    array = np.asarray(actions)
    if array.ndim not in {1, 2} or array.shape[-1] != 7:
        raise ValueError("OpenVLA-OFT action output must have shape (7,) or (N, 7)")
    if not (np.issubdtype(array.dtype, np.integer) or np.issubdtype(array.dtype, np.floating)):
        raise ValueError("OpenVLA-OFT actions must be real numeric")
    array = array.astype(np.float64, copy=False)
    if not np.isfinite(array).all():
        raise ValueError("OpenVLA-OFT actions must be finite")
    if envelope == "q01_q99":
        lower_name, upper_name = "q01", "q99"
    elif envelope == "min_max":
        lower_name, upper_name = "min", "max"
    else:
        raise ValueError(f"Unknown OpenVLA-OFT action envelope: {envelope}")
    lower = _seven(action_statistics.get(lower_name), name=f"action.{lower_name}")
    upper = _seven(action_statistics.get(upper_name), name=f"action.{upper_name}")
    if np.any(lower > upper):
        raise ValueError("OpenVLA-OFT action envelope is inconsistent")
    clipped = np.clip(array, lower, upper)
    return {
        "schema": "kinesync.openvla_oft_action_contract.v1",
        "valid": True,
        "shape": list(array.shape),
        "envelope": envelope,
        "clipped_value_count": int(np.count_nonzero(clipped != array)),
        "command_mode": "disabled",
        "clipped_actions": clipped.tolist(),
    }


def select_high_motion_indices(
    policy_state: object,
    recorded_action: object,
    *,
    count: int,
    minimum_separation: int,
) -> np.ndarray:
    """Select distinct frames where the recorded next action differs from hold."""

    state = np.asarray(policy_state)
    action = np.asarray(recorded_action)
    if state.ndim != 2 or state.shape != action.shape or state.shape[1] != 7:
        raise ValueError("Policy state and recorded action must be aligned Nx7 arrays")
    if not np.isfinite(state).all() or not np.isfinite(action).all():
        raise ValueError("Policy state and recorded action must be finite")
    if count < 1 or minimum_separation < 0:
        raise ValueError("count must be positive and minimum_separation nonnegative")
    scores = np.mean(np.abs(action.astype(np.float64) - state.astype(np.float64)), axis=1)
    order = np.argsort(-scores, kind="stable")
    selected: list[int] = []
    for raw_index in order:
        index = int(raw_index)
        if all(abs(index - other) >= minimum_separation for other in selected):
            selected.append(index)
            if len(selected) == count:
                break
    if len(selected) != count:
        raise ValueError(
            f"Cannot select {count} frames with minimum separation {minimum_separation}"
        )
    return np.asarray(selected, dtype=np.int64)


__all__ = [
    "audit_openvla_oft_checkpoint",
    "select_high_motion_indices",
    "validate_action_chunk",
]
