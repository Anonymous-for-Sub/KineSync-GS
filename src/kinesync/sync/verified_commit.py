"""Floor-free commits protected by component-level verification."""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import torch

from kinesync.guard.component_verification import ComponentVerificationRun

from .schema import (
    ComponentDecision,
    DetectabilityProfile,
    SynchronizedStateFrame,
    synchronized_state_fingerprint,
)


def commit_component_verified_state(
    *,
    profile: DetectabilityProfile,
    verifications: ComponentVerificationRun,
    measured_qpos: torch.Tensor,
    candidate_qpos: torch.Tensor,
    selected_joints: Sequence[str],
    joint_limits: Mapping[str, tuple[float | None, float | None]],
    timestamp_ns: int,
    state_id: str,
    case_id: str,
    factor_fingerprint: str,
    guard_fingerprint: str,
) -> SynchronizedStateFrame:
    """Commit supported joints that pass their own finite guard evidence."""

    if not isinstance(verifications, ComponentVerificationRun):
        raise TypeError("verifications must be a ComponentVerificationRun")
    if not isinstance(measured_qpos, torch.Tensor) or not isinstance(
        candidate_qpos, torch.Tensor
    ):
        raise TypeError("measured_qpos and candidate_qpos must be torch tensors")
    expected = (len(profile.joint_names),)
    if measured_qpos.shape != expected or candidate_qpos.shape != expected:
        raise ValueError("measured and candidate qpos must be matching profile vectors")
    if (
        candidate_qpos.dtype != measured_qpos.dtype
        or candidate_qpos.device != measured_qpos.device
    ):
        raise ValueError("candidate_qpos must exactly match measured_qpos dtype and device")
    selected = tuple(str(name) for name in selected_joints)
    if not selected:
        raise ValueError("selected_joints must be nonempty")
    if len(set(selected)) != len(selected):
        raise ValueError("selected_joints must be unique")
    unknown = set(selected) - set(profile.joint_names)
    if unknown:
        raise ValueError(f"Unknown selected joints: {sorted(unknown)}")
    verification_names = {
        verification.joint_name for verification in verifications.verifications
    }
    if verification_names != set(selected):
        raise ValueError("verification joint names must exactly match selected_joints")
    if set(joint_limits) != set(profile.joint_names):
        raise ValueError("joint_limits must exactly match the profile joint names")
    if isinstance(timestamp_ns, bool) or not isinstance(timestamp_ns, int):
        raise ValueError("timestamp_ns must be an integer")
    if timestamp_ns < 0:
        raise ValueError("timestamp_ns must be nonnegative")
    if not str(state_id).strip():
        raise ValueError("state_id must be nonempty")
    if not str(case_id).strip():
        raise ValueError("case_id must be nonempty")
    for field, value in (
        ("factor_fingerprint", factor_fingerprint),
        ("guard_fingerprint", guard_fingerprint),
    ):
        if len(value) != 64:
            raise ValueError(f"{field} must be a SHA-256 digest")
        try:
            int(value, 16)
        except ValueError as error:
            raise ValueError(f"{field} must be a SHA-256 digest") from error
    if not torch.isfinite(measured_qpos).all():
        raise ValueError("measured_qpos must be finite")

    normalized_limits: dict[str, tuple[float | None, float | None]] = {}
    for name in profile.joint_names:
        lower, upper = joint_limits[name]
        if lower is not None:
            lower = float(lower)
        if upper is not None:
            upper = float(upper)
        if lower is not None and not math.isfinite(lower):
            raise ValueError("joint_limits must be finite")
        if upper is not None and not math.isfinite(upper):
            raise ValueError("joint_limits must be finite")
        if lower is not None and upper is not None and lower > upper:
            raise ValueError("joint_limits lower bound exceeds upper bound")
        normalized_limits[name] = (lower, upper)

    measured = measured_qpos.detach().clone()
    candidate = candidate_qpos.detach().clone()
    for joint_name in selected:
        index = profile.joint_names.index(joint_name)
        expected_component_qpos = measured.clone()
        expected_component_qpos[index] = candidate[index]
        verification_candidate_qpos = verifications.for_joint(joint_name).candidate_qpos
        if (
            verification_candidate_qpos.dtype != expected_component_qpos.dtype
            or verification_candidate_qpos.device != expected_component_qpos.device
            or not torch.equal(verification_candidate_qpos, expected_component_qpos)
        ):
            raise ValueError(
                "component verification candidate_qpos must exactly match its "
                "submitted component state"
            )

    synchronized = measured.clone()
    decisions: list[ComponentDecision] = []
    for joint_name in selected:
        verification = verifications.for_joint(joint_name)
        index = profile.joint_names.index(joint_name)
        candidate_value = float(candidate[index])
        correction = candidate_value - float(measured[index])
        detectability = profile.for_joint(joint_name)
        floor = detectability.minimum_candidate_correction_rad
        accepted = False

        if not detectability.supported:
            reason = "unsupported_joint"
        elif (
            not verification.evidence.finite
            or not math.isfinite(candidate_value)
            or not math.isfinite(correction)
        ):
            reason = "nonfinite"
        elif not verification.decision.accepted:
            reason = "guard_rejected"
        else:
            lower, upper = normalized_limits[joint_name]
            outside = (
                (lower is not None and candidate_value < lower)
                or (upper is not None and candidate_value > upper)
            )
            if outside:
                reason = "outside_joint_limit"
            else:
                reason = "accepted"
                accepted = True
                synchronized[index] = candidate[index]

        decisions.append(
            ComponentDecision(
                joint_name=joint_name,
                accepted=accepted,
                reason=reason,
                correction_rad=correction,
                signal_floor_rad=floor,
                margin_rad=None,
            )
        )

    guard_reason = "component_verified"
    fingerprint = synchronized_state_fingerprint(
        joint_names=profile.joint_names,
        measured_qpos=measured,
        candidate_qpos=candidate,
        synchronized_qpos=synchronized,
        decisions=tuple(decisions),
        timestamp_ns=timestamp_ns,
        state_id=str(state_id),
        case_id=str(case_id),
        factor_fingerprint=factor_fingerprint,
        guard_fingerprint=guard_fingerprint,
        profile_fingerprint=profile.fingerprint,
        guard_reason=guard_reason,
    )
    return SynchronizedStateFrame(
        joint_names=profile.joint_names,
        measured_qpos=measured,
        candidate_qpos=candidate,
        synchronized_qpos=synchronized,
        decisions=tuple(decisions),
        timestamp_ns=timestamp_ns,
        state_id=str(state_id),
        case_id=str(case_id),
        factor_fingerprint=factor_fingerprint,
        guard_fingerprint=guard_fingerprint,
        profile_fingerprint=profile.fingerprint,
        guard_reason=guard_reason,
        fingerprint=fingerprint,
    )


__all__ = ["commit_component_verified_state"]
