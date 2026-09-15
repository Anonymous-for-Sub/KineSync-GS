"""External-GS adapters that apply the project's existing guard contracts."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import torch

from kinesync.guard.decision import decide_update
from kinesync.guard.evidence import assemble_update_evidence
from kinesync.guard.schema import GuardDecision, GuardThresholds, UpdateEvidence
from kinesync.sync.calibration import calibration_rows_sha256
from kinesync.sync.schema import DetectabilityProfile, JointDetectability


class ExternalRGBObservationBackend:
    """Add project metrics to a native RGB renderer without changing its loss."""

    def __init__(self, renderer: Any) -> None:
        self.renderer = renderer
        self.cameras = dict(renderer.cameras)

    def render(self, qpos: torch.Tensor):
        return self.renderer.render(qpos)

    def loss(self, qpos: torch.Tensor, target):
        return self.renderer.loss(qpos, target)

    def metrics(self, qpos: torch.Tensor, target) -> dict[str, float]:
        with torch.no_grad():
            loss, terms = self.loss(qpos, target)
        values = {f"render_rgb_mse_{name}": float(value) for name, value in terms.items()}
        values["render_rgb_mse"] = float(loss)
        return values


@dataclass(frozen=True)
class ProjectGlobalGuardResult:
    """Whole-candidate decision from the existing multi-view evidence contract."""

    evidence: UpdateEvidence
    decision: GuardDecision
    committed_qpos: torch.Tensor


def apply_project_global_guard(
    *,
    joint_backend: Any,
    joint_target: Any,
    camera_backends: Mapping[str, Any],
    camera_targets: Mapping[str, Any],
    measured_qpos: torch.Tensor,
    candidate_qpos: torch.Tensor,
    camera_corrections: Mapping[str, torch.Tensor],
    selected_indices: Sequence[int],
    candidate_bound: float,
    thresholds: GuardThresholds,
) -> ProjectGlobalGuardResult:
    """Apply the existing whole-candidate evidence and decision functions.

    The function deliberately takes no reference/true state.  Its result is an
    all-or-nothing commit of the already generated fused candidate.
    """
    evidence = assemble_update_evidence(
        joint_backend=joint_backend,
        joint_target=joint_target,
        camera_backends=camera_backends,
        camera_targets=camera_targets,
        measured_qpos=measured_qpos,
        candidate_qpos=candidate_qpos,
        camera_corrections=camera_corrections,
        selected_indices=selected_indices,
        candidate_bound=candidate_bound,
    )
    decision = decide_update(evidence, thresholds)
    committed = candidate_qpos if decision.accepted else measured_qpos
    return ProjectGlobalGuardResult(
        evidence=evidence,
        decision=decision,
        committed_qpos=committed.detach().clone(),
    )


def build_development_detectability_profile(
    records: Sequence[Mapping[str, Any]],
    *,
    joint_names: Sequence[str],
    minimum_state_reduction: float,
    zero_tolerance_rad: float,
) -> DetectabilityProfile:
    """Freeze per-joint signal floors from development-only evaluation records.

    This calibration boundary is intentionally separate from guard application:
    callers provide evaluation rows only after candidate generation, while the
    deployed global/component gates receive no reference state.
    """
    names = tuple(str(name) for name in joint_names)
    if not names or len(set(names)) != len(names):
        raise ValueError("joint_names must be nonempty and unique")
    if not 0.0 <= minimum_state_reduction <= 1.0:
        raise ValueError("minimum_state_reduction must lie in [0, 1]")
    if not math.isfinite(zero_tolerance_rad) or zero_tolerance_rad < 0.0:
        raise ValueError("zero_tolerance_rad must be finite and nonnegative")
    if not records:
        raise ValueError("development detectability requires records")
    if any(str(row.get("split", "")) != "development" for row in records):
        raise ValueError("detectability calibration accepts development records only")

    state_ids = tuple(sorted({str(row.get("state_id", "")) for row in records}))
    if not state_ids or any(not value for value in state_ids):
        raise ValueError("development detectability records require state_id")
    corrections: dict[str, list[float]] = {name: [] for name in names}
    controlled_counts: dict[str, int] = {name: 0 for name in names}
    zero_counts: dict[str, int] = {name: 0 for name in names}
    supporting_counts: dict[str, int] = {name: 0 for name in names}
    for row in records:
        try:
            initial = tuple(float(value) for value in row["initial_abs_error_rad"])
            candidate = tuple(float(value) for value in row["candidate_abs_error_rad"])
            applied = tuple(float(value) for value in row["candidate_correction_rad"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("development detectability records require finite vectors") from error
        if any(len(values) != len(names) for values in (initial, candidate, applied)):
            raise ValueError("development detectability vectors must match joint_names")
        if not all(math.isfinite(value) for values in (initial, candidate, applied) for value in values):
            raise ValueError("development detectability vectors must be finite")
        for index, name in enumerate(names):
            if initial[index] <= zero_tolerance_rad:
                zero_counts[name] += 1
                continue
            controlled_counts[name] += 1
            reduction = 1.0 - candidate[index] / initial[index]
            if reduction >= minimum_state_reduction and abs(applied[index]) > zero_tolerance_rad:
                supporting_counts[name] += 1
                corrections[name].append(abs(applied[index]))

    source_sha256 = calibration_rows_sha256(records)
    joint_order_sha256 = hashlib.sha256(
        json.dumps(names, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    ).hexdigest()
    joints: list[JointDetectability] = []
    for name in names:
        values = corrections[name]
        supported = bool(values)
        floor = max(min(values), zero_tolerance_rad) if supported else None
        joints.append(
            JointDetectability(
                joint_name=name,
                supported=supported,
                minimum_magnitude_rad=floor,
                minimum_candidate_correction_rad=floor,
                controlled_count=controlled_counts[name],
                supporting_count=supporting_counts[name],
                zero_count=zero_counts[name],
            )
        )
    payload = {
        "algorithm_version": "external-gs-development-detectability-v1",
        "joint_names": names,
        "joint_order_source_sha256": joint_order_sha256,
        "joints": [
            {
                "joint_name": joint.joint_name,
                "supported": joint.supported,
                "minimum_magnitude_rad": joint.minimum_magnitude_rad,
                "minimum_candidate_correction_rad": joint.minimum_candidate_correction_rad,
                "controlled_count": joint.controlled_count,
                "supporting_count": joint.supporting_count,
                "zero_count": joint.zero_count,
            }
            for joint in joints
        ],
        "source_sha256": source_sha256,
        "source_split": "train",
        "calibration_state_ids": state_ids,
    }
    fingerprint = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    ).hexdigest()
    return DetectabilityProfile(
        joint_names=names,
        joints=tuple(joints),
        source_sha256=source_sha256,
        joint_order_source_sha256=joint_order_sha256,
        source_split="train",
        calibration_state_ids=state_ids,
        algorithm_version="external-gs-development-detectability-v1",
        fingerprint=fingerprint,
    )


__all__ = [
    "ExternalRGBObservationBackend",
    "ProjectGlobalGuardResult",
    "apply_project_global_guard",
    "build_development_detectability_profile",
]
