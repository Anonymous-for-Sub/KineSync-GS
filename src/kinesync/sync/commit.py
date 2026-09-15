"""Component-wise state commit after multi-view guard protection."""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import torch

from kinesync.guard.schema import GuardDecision

from .schema import (
    ComponentDecision,
    DetectabilityProfile,
    SynchronizedStateFrame,
    synchronized_state_fingerprint,
)


def commit_synchronized_state(
    *,
    profile: DetectabilityProfile,
    guard_decision: GuardDecision,
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
    """Commit only selected components with guard and detectability support."""

    if not isinstance(measured_qpos, torch.Tensor) or not isinstance(
        candidate_qpos, torch.Tensor
    ):
        raise TypeError("measured_qpos and candidate_qpos must be torch tensors")
    expected = (len(profile.joint_names),)
    if measured_qpos.shape != expected or candidate_qpos.shape != expected:
        raise ValueError("measured and candidate qpos must be matching profile vectors")
    selected = tuple(str(name) for name in selected_joints)
    if not selected:
        raise ValueError("selected_joints must be nonempty")
    if len(set(selected)) != len(selected):
        raise ValueError("selected_joints must be unique")
    unknown = set(selected) - set(profile.joint_names)
    if unknown:
        raise ValueError(f"Unknown selected joints: {sorted(unknown)}")
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
    candidate = candidate_qpos.detach().to(measured).clone()
    synchronized = measured.clone()
    decisions: list[ComponentDecision] = []
    for joint_name in selected:
        index = profile.joint_names.index(joint_name)
        candidate_value = float(candidate[index])
        measured_value = float(measured[index])
        correction = candidate_value - measured_value
        detectability = profile.for_joint(joint_name)
        floor = detectability.minimum_candidate_correction_rad
        accepted = False
        margin: float | None = None

        if not guard_decision.accepted:
            reason = "guard_rejected"
        elif not math.isfinite(candidate_value) or not math.isfinite(correction):
            reason = "nonfinite"
        elif not detectability.supported:
            reason = "unsupported_joint"
        else:
            assert floor is not None
            margin = abs(correction) - floor
            if margin < 0:
                reason = "below_signal_floor"
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
                margin_rad=margin,
            )
        )

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
        guard_reason=guard_decision.reason,
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
        guard_reason=guard_decision.reason,
        fingerprint=fingerprint,
    )


__all__ = ["commit_synchronized_state"]
