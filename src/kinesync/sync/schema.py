"""Immutable schemas for detectability-calibrated synchronization."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Any

import torch


@dataclass(frozen=True)
class JointDetectability:
    joint_name: str
    supported: bool
    minimum_magnitude_rad: float | None
    minimum_candidate_correction_rad: float | None
    controlled_count: int
    supporting_count: int
    zero_count: int

    def __post_init__(self) -> None:
        if not self.joint_name:
            raise ValueError("joint_name must be nonempty")
        if min(self.controlled_count, self.supporting_count, self.zero_count) < 0:
            raise ValueError("detectability counts must be nonnegative")
        if self.supporting_count > self.controlled_count:
            raise ValueError("supporting_count cannot exceed controlled_count")
        thresholds = (
            self.minimum_magnitude_rad,
            self.minimum_candidate_correction_rad,
        )
        if self.supported:
            if any(value is None for value in thresholds):
                raise ValueError("supported joints require finite thresholds")
            if any(not math.isfinite(value) or value <= 0 for value in thresholds):
                raise ValueError("supported thresholds must be finite and positive")
            if self.supporting_count <= 0:
                raise ValueError("supported joints require supporting examples")
        elif any(value is not None for value in thresholds):
            raise ValueError("unsupported joints cannot carry thresholds")


@dataclass(frozen=True)
class DetectabilityProfile:
    joint_names: tuple[str, ...]
    joints: tuple[JointDetectability, ...]
    source_sha256: str
    joint_order_source_sha256: str
    source_split: str
    calibration_state_ids: tuple[str, ...]
    algorithm_version: str
    fingerprint: str

    def __post_init__(self) -> None:
        if not self.joint_names or len(set(self.joint_names)) != len(self.joint_names):
            raise ValueError("joint_names must be nonempty and unique")
        if tuple(item.joint_name for item in self.joints) != self.joint_names:
            raise ValueError("joint detectability order must match joint_names")
        if self.source_split != "train":
            raise ValueError("detectability profiles require source_split=train")
        profile_hashes = (
            self.source_sha256,
            self.joint_order_source_sha256,
            self.fingerprint,
        )
        if any(len(value) != 64 for value in profile_hashes):
            raise ValueError("profile hashes must be SHA-256 hex digests")
        for value in profile_hashes:
            try:
                int(value, 16)
            except ValueError as error:
                raise ValueError("profile hashes must be SHA-256 hex digests") from error
        if not self.calibration_state_ids:
            raise ValueError("calibration_state_ids must be nonempty")
        if tuple(sorted(set(self.calibration_state_ids))) != self.calibration_state_ids:
            raise ValueError("calibration_state_ids must be sorted and unique")
        if not self.algorithm_version:
            raise ValueError("algorithm_version must be nonempty")

    def for_joint(self, joint_name: str) -> JointDetectability:
        try:
            index = self.joint_names.index(joint_name)
        except ValueError as error:
            raise KeyError(f"Unknown joint: {joint_name}") from error
        return self.joints[index]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)



@dataclass(frozen=True)
class ComponentDecision:
    joint_name: str
    accepted: bool
    reason: str
    correction_rad: float
    signal_floor_rad: float | None
    margin_rad: float | None

    def __post_init__(self) -> None:
        valid_reasons = {
            "accepted",
            "guard_rejected",
            "unsupported_joint",
            "below_signal_floor",
            "nonfinite",
            "outside_joint_limit",
        }
        if not self.joint_name:
            raise ValueError("component joint_name must be nonempty")
        if self.reason not in valid_reasons:
            raise ValueError(f"Unknown component decision reason: {self.reason}")
        if self.accepted != (self.reason == "accepted"):
            raise ValueError("accepted flag must match component reason")


def synchronized_state_fingerprint(
    *,
    joint_names: tuple[str, ...],
    measured_qpos: torch.Tensor,
    candidate_qpos: torch.Tensor,
    synchronized_qpos: torch.Tensor,
    decisions: tuple[ComponentDecision, ...],
    timestamp_ns: int,
    state_id: str,
    case_id: str,
    factor_fingerprint: str,
    guard_fingerprint: str,
    profile_fingerprint: str,
    guard_reason: str,
) -> str:
    def values(tensor: torch.Tensor) -> list[float | str]:
        output: list[float | str] = []
        for value in tensor.detach().cpu().tolist():
            number = float(value)
            output.append(number if math.isfinite(number) else str(number))
        return output

    payload = {
        "candidate_qpos": values(candidate_qpos),
        "case_id": case_id,
        "decisions": [
            {
                "accepted": item.accepted,
                "correction_rad": (
                    item.correction_rad
                    if math.isfinite(item.correction_rad)
                    else str(item.correction_rad)
                ),
                "joint_name": item.joint_name,
                "margin_rad": item.margin_rad,
                "reason": item.reason,
                "signal_floor_rad": item.signal_floor_rad,
            }
            for item in decisions
        ],
        "factor_fingerprint": factor_fingerprint,
        "guard_fingerprint": guard_fingerprint,
        "guard_reason": guard_reason,
        "joint_names": list(joint_names),
        "measured_qpos": values(measured_qpos),
        "profile_fingerprint": profile_fingerprint,
        "state_id": state_id,
        "synchronized_qpos": values(synchronized_qpos),
        "timestamp_ns": timestamp_ns,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class SynchronizedStateFrame:
    joint_names: tuple[str, ...]
    measured_qpos: torch.Tensor
    candidate_qpos: torch.Tensor
    synchronized_qpos: torch.Tensor
    decisions: tuple[ComponentDecision, ...]
    timestamp_ns: int
    state_id: str
    case_id: str
    factor_fingerprint: str
    guard_fingerprint: str
    profile_fingerprint: str
    guard_reason: str
    fingerprint: str

    def __post_init__(self) -> None:
        if not self.joint_names or len(set(self.joint_names)) != len(self.joint_names):
            raise ValueError("joint_names must be nonempty and unique")
        shapes = {
            tuple(self.measured_qpos.shape),
            tuple(self.candidate_qpos.shape),
            tuple(self.synchronized_qpos.shape),
        }
        if len(shapes) != 1 or self.measured_qpos.ndim != 1:
            raise ValueError("state qpos tensors must be matching vectors")
        if self.measured_qpos.shape[0] != len(self.joint_names):
            raise ValueError("joint_names must match state qpos width")
        if not torch.isfinite(self.measured_qpos).all():
            raise ValueError("measured_qpos must be finite")
        if not torch.isfinite(self.synchronized_qpos).all():
            raise ValueError("synchronized_qpos must be finite")
        if isinstance(self.timestamp_ns, bool) or not isinstance(self.timestamp_ns, int):
            raise ValueError("timestamp_ns must be an integer")
        if self.timestamp_ns < 0:
            raise ValueError("timestamp_ns must be nonnegative")
        if not self.state_id:
            raise ValueError("state_id must be nonempty")
        if not self.case_id:
            raise ValueError("case_id must be nonempty")
        hashes = (
            self.factor_fingerprint,
            self.guard_fingerprint,
            self.profile_fingerprint,
            self.fingerprint,
        )
        if any(len(value) != 64 for value in hashes):
            raise ValueError("state fingerprints must be SHA-256 digests")
        for value in hashes:
            try:
                int(value, 16)
            except ValueError as error:
                raise ValueError("state fingerprints must be SHA-256 digests") from error
        if not self.guard_reason:
            raise ValueError("guard_reason must be nonempty")
        names = tuple(item.joint_name for item in self.decisions)
        if len(set(names)) != len(names):
            raise ValueError("component decisions must have unique joints")
        if set(names) - set(self.joint_names):
            raise ValueError("component decisions contain unknown joints")
        self.validate_integrity()

    def validate_integrity(self) -> None:
        expected = self.measured_qpos.detach().clone()
        for decision in self.decisions:
            index = self.joint_names.index(decision.joint_name)
            if decision.accepted:
                expected[index] = self.candidate_qpos[index]
        if not torch.equal(self.synchronized_qpos, expected):
            raise ValueError("synchronized_qpos integrity violation")
        expected_fingerprint = synchronized_state_fingerprint(
            joint_names=self.joint_names,
            measured_qpos=self.measured_qpos,
            candidate_qpos=self.candidate_qpos,
            synchronized_qpos=self.synchronized_qpos,
            decisions=self.decisions,
            timestamp_ns=self.timestamp_ns,
            state_id=self.state_id,
            case_id=self.case_id,
            factor_fingerprint=self.factor_fingerprint,
            guard_fingerprint=self.guard_fingerprint,
            profile_fingerprint=self.profile_fingerprint,
            guard_reason=self.guard_reason,
        )
        if self.fingerprint != expected_fingerprint:
            raise ValueError("state fingerprint integrity violation")

    @property
    def accepted_joint_names(self) -> tuple[str, ...]:
        return tuple(item.joint_name for item in self.decisions if item.accepted)
