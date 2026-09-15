"""Shared RT7 component-verified policy application for replay consumers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import torch

from kinesync.guard.component_verification import (
    ComponentVerificationRun,
    verify_candidate_components,
)
from kinesync.guard.schema import GuardThresholds
from kinesync.sync.schema import DetectabilityProfile, SynchronizedStateFrame
from kinesync.sync.verified_commit import commit_component_verified_state


RenderReceipt = Callable[[Any, torch.Tensor], Mapping[str, Any]]


def _qpos_values(qpos: torch.Tensor) -> list[float]:
    return [float(value) for value in qpos.detach().cpu().tolist()]


def _validated_receipt(
    receipt: Mapping[str, Any], *, qpos: torch.Tensor, label: str
) -> Mapping[str, Any]:
    if not isinstance(receipt, Mapping) or receipt.get("qpos") != _qpos_values(qpos):
        raise ValueError(f"{label} Gaussian receipt qpos does not match submitted state")
    return dict(receipt)


@dataclass(frozen=True)
class ComponentVerifiedPolicyApplication:
    """Verified RT7 component state plus actual full/component render receipts."""

    frame: SynchronizedStateFrame
    verifications: ComponentVerificationRun
    synchronized_render_receipt: Mapping[str, Any]
    component_render_receipts: Mapping[str, Mapping[str, Any]]


def apply_component_verified_policy(
    *,
    profile: DetectabilityProfile,
    joint_backend: Any,
    joint_target: Any,
    camera_backends: Mapping[str, Any],
    camera_targets: Mapping[str, Any],
    measured_qpos: torch.Tensor,
    candidate_qpos: torch.Tensor,
    camera_corrections: Mapping[str, torch.Tensor],
    joint_names: Sequence[str],
    selected_joints: Sequence[str],
    candidate_bound: float,
    thresholds: GuardThresholds,
    joint_limits: Mapping[str, tuple[float | None, float | None]],
    timestamp_ns: int,
    state_id: str,
    case_id: str,
    factor_fingerprint: str,
    guard_fingerprint: str,
    render_receipt: RenderReceipt,
) -> ComponentVerifiedPolicyApplication:
    """Verify, commit, and receipt a component-wise RT7 candidate state.

    This public boundary is intentionally independent of RT6 replay I/O so
    later replay tasks can apply the same policy to a different frozen source.
    """

    verifications = verify_candidate_components(
        joint_backend=joint_backend,
        joint_target=joint_target,
        camera_backends=camera_backends,
        camera_targets=camera_targets,
        measured_qpos=measured_qpos,
        candidate_qpos=candidate_qpos,
        camera_corrections=camera_corrections,
        joint_names=joint_names,
        selected_joints=selected_joints,
        candidate_bound=candidate_bound,
        thresholds=thresholds,
    )
    frame = commit_component_verified_state(
        profile=profile,
        verifications=verifications,
        measured_qpos=measured_qpos,
        candidate_qpos=candidate_qpos,
        selected_joints=selected_joints,
        joint_limits=joint_limits,
        timestamp_ns=timestamp_ns,
        state_id=state_id,
        case_id=case_id,
        factor_fingerprint=factor_fingerprint,
        guard_fingerprint=guard_fingerprint,
    )
    frame.validate_integrity()
    synchronized_receipt = _validated_receipt(
        render_receipt(joint_backend, frame.synchronized_qpos),
        qpos=frame.synchronized_qpos,
        label="component-verified",
    )
    component_receipts = {
        verification.joint_name: _validated_receipt(
            render_receipt(joint_backend, verification.candidate_qpos),
            qpos=verification.candidate_qpos,
            label=f"component {verification.joint_name}",
        )
        for verification in verifications.verifications
    }
    return ComponentVerifiedPolicyApplication(
        frame=frame,
        verifications=verifications,
        synchronized_render_receipt=synchronized_receipt,
        component_render_receipts=component_receipts,
    )


def inherit_source_result_row(
    source_row: Mapping[str, Any], replay_columns: Mapping[str, Any]
) -> dict[str, Any]:
    """Copy a source CSV row exactly while adding collision-free replay fields."""

    if not isinstance(source_row, Mapping) or not source_row:
        raise ValueError("source result row must be a nonempty mapping")
    if not isinstance(replay_columns, Mapping) or not replay_columns:
        raise ValueError("replay columns must be a nonempty mapping")
    collisions = sorted(set(source_row).intersection(replay_columns))
    if collisions:
        raise ValueError(
            "replay-only columns cannot overwrite source result fields: "
            + ", ".join(collisions)
        )
    return {**dict(source_row), **dict(replay_columns)}


__all__ = [
    "ComponentVerifiedPolicyApplication",
    "apply_component_verified_policy",
    "inherit_source_result_row",
]
