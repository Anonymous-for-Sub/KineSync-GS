"""Per-joint Gaussian evidence for component-verified state updates."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from types import MappingProxyType
from typing import Mapping, Sequence

import torch

from .decision import decide_update
from .evidence import assemble_update_evidence
from .schema import GuardDecision, GuardThresholds, UpdateEvidence


def _canonical_number(value: float) -> float | str:
    number = float(value)
    return number if math.isfinite(number) else str(number)


def _canonical_tensor(tensor: torch.Tensor) -> list[float | str]:
    return [_canonical_number(value) for value in tensor.detach().cpu().tolist()]


def _canonical_mapping(values: Mapping[str, float]) -> dict[str, float | str]:
    return {
        str(name): _canonical_number(value)
        for name, value in sorted(values.items())
    }


def _fingerprint(
    *,
    joint_name: str,
    candidate_qpos: torch.Tensor,
    evidence: UpdateEvidence,
    decision: GuardDecision,
    before_metrics: Mapping[str, float],
    after_metrics: Mapping[str, float],
    per_view_visual_gain: Mapping[str, float],
    thresholds: GuardThresholds,
) -> str:
    payload = {
        "after_metrics": _canonical_mapping(after_metrics),
        "before_metrics": _canonical_mapping(before_metrics),
        "candidate_qpos": _canonical_tensor(candidate_qpos),
        "decision": {
            "accepted": decision.accepted,
            "minimum_margin": _canonical_number(decision.minimum_margin),
            "reason": decision.reason,
        },
        "evidence": {
            key: _canonical_number(value) if isinstance(value, float) else value
            for key, value in asdict(evidence).items()
        },
        "joint_name": joint_name,
        "per_view_visual_gain": _canonical_mapping(per_view_visual_gain),
        "thresholds": {
            key: _canonical_number(value) for key, value in asdict(thresholds).items()
        },
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
            "utf-8"
        )
    ).hexdigest()


def _immutable_metrics(values: Mapping[str, float]) -> Mapping[str, float]:
    return MappingProxyType({str(name): float(value) for name, value in values.items()})


@dataclass(frozen=True, init=False)
class ComponentVerification:
    """Immutable evidence and decision for a single candidate joint."""

    joint_name: str
    _canonical_candidate_qpos: torch.Tensor = field(repr=False)
    evidence: UpdateEvidence
    decision: GuardDecision
    before_metrics: Mapping[str, float]
    after_metrics: Mapping[str, float]
    per_view_visual_gain: Mapping[str, float]
    fingerprint: str

    def __init__(
        self,
        joint_name: str,
        candidate_qpos: torch.Tensor,
        evidence: UpdateEvidence,
        decision: GuardDecision,
        before_metrics: Mapping[str, float],
        after_metrics: Mapping[str, float],
        per_view_visual_gain: Mapping[str, float],
        fingerprint: str,
    ) -> None:
        if not joint_name:
            raise ValueError("joint_name must be nonempty")
        if not isinstance(candidate_qpos, torch.Tensor) or candidate_qpos.ndim != 1:
            raise ValueError("candidate_qpos must be a one-dimensional tensor")
        if len(fingerprint) != 64:
            raise ValueError("fingerprint must be a SHA-256 digest")
        try:
            int(fingerprint, 16)
        except ValueError as error:
            raise ValueError("fingerprint must be a SHA-256 digest") from error
        object.__setattr__(self, "joint_name", joint_name)
        object.__setattr__(
            self, "_canonical_candidate_qpos", candidate_qpos.detach().clone()
        )
        object.__setattr__(self, "evidence", evidence)
        object.__setattr__(self, "decision", decision)
        object.__setattr__(self, "before_metrics", _immutable_metrics(before_metrics))
        object.__setattr__(self, "after_metrics", _immutable_metrics(after_metrics))
        object.__setattr__(
            self, "per_view_visual_gain", _immutable_metrics(per_view_visual_gain)
        )
        object.__setattr__(self, "fingerprint", fingerprint)

    @property
    def candidate_qpos(self) -> torch.Tensor:
        """Return a defensive copy of the canonical component qpos."""

        return self._canonical_candidate_qpos.detach().clone()


@dataclass(frozen=True)
class ComponentVerificationRun:
    """Immutable component verification records for one candidate state."""

    verifications: tuple[ComponentVerification, ...]

    def __post_init__(self) -> None:
        verifications = tuple(self.verifications)
        object.__setattr__(self, "verifications", verifications)
        names = tuple(item.joint_name for item in verifications)
        if not names or len(set(names)) != len(names):
            raise ValueError("component verifications must have unique joint names")

    def for_joint(self, joint_name: str) -> ComponentVerification:
        for verification in self.verifications:
            if verification.joint_name == joint_name:
                return verification
        raise KeyError(f"Unknown verified joint: {joint_name}")


def _validate_inputs(
    *,
    camera_backends: Mapping[str, object],
    camera_targets: Mapping[str, object],
    camera_corrections: Mapping[str, torch.Tensor],
    measured_qpos: torch.Tensor,
    candidate_qpos: torch.Tensor,
    joint_names: Sequence[str],
    selected_joints: Sequence[str],
    candidate_bound: float,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if not isinstance(measured_qpos, torch.Tensor) or not isinstance(
        candidate_qpos, torch.Tensor
    ):
        raise TypeError("measured_qpos and candidate_qpos must be torch tensors")
    if measured_qpos.shape != candidate_qpos.shape or measured_qpos.ndim != 1:
        raise ValueError("Measured and candidate qpos must be matching vectors")
    names = tuple(str(name) for name in joint_names)
    if not names or len(set(names)) != len(names) or len(names) != measured_qpos.numel():
        raise ValueError("joint_names must exactly match the qpos vector")
    selected = tuple(str(name) for name in selected_joints)
    if not selected:
        raise ValueError("selected_joints must be nonempty")
    if len(set(selected)) != len(selected):
        raise ValueError("selected_joints must be unique")
    unknown = set(selected) - set(names)
    if unknown:
        raise ValueError(f"Unknown selected joints: {sorted(unknown)}")
    if set(camera_backends) != set(camera_targets) or set(camera_backends) != set(
        camera_corrections
    ):
        raise ValueError("Camera backends, targets, and corrections must match")
    if not math.isfinite(float(candidate_bound)) or candidate_bound <= 0:
        raise ValueError("candidate_bound must be positive and finite")
    for name, correction in camera_corrections.items():
        if not isinstance(correction, torch.Tensor) or correction.shape != measured_qpos.shape:
            raise ValueError(f"camera correction for {name} must match the qpos vector")
    return names, selected


def _visual_gain(backend, target, before_qpos: torch.Tensor, after_qpos: torch.Tensor) -> float:
    with torch.no_grad():
        before_loss = backend.loss(before_qpos, target)[0]
        after_loss = backend.loss(after_qpos, target)[0]
    return float((before_loss - after_loss) / before_loss.abs().clamp_min(1e-8))


def verify_candidate_components(
    *,
    joint_backend,
    joint_target,
    camera_backends: Mapping[str, object],
    camera_targets: Mapping[str, object],
    measured_qpos: torch.Tensor,
    candidate_qpos: torch.Tensor,
    camera_corrections: Mapping[str, torch.Tensor],
    joint_names: Sequence[str],
    selected_joints: Sequence[str],
    candidate_bound: float,
    thresholds: GuardThresholds,
) -> ComponentVerificationRun:
    """Construct independent Gaussian evidence records for selected joints."""

    names, selected = _validate_inputs(
        camera_backends=camera_backends,
        camera_targets=camera_targets,
        camera_corrections=camera_corrections,
        measured_qpos=measured_qpos,
        candidate_qpos=candidate_qpos,
        joint_names=joint_names,
        selected_joints=selected_joints,
        candidate_bound=candidate_bound,
    )
    if not isinstance(thresholds, GuardThresholds):
        raise TypeError("thresholds must be GuardThresholds")

    measured = measured_qpos.detach().clone()
    candidate = candidate_qpos.detach().to(measured).clone()
    before_metrics = _immutable_metrics(joint_backend.metrics(measured, joint_target))
    verifications: list[ComponentVerification] = []
    for joint_name in selected:
        index = names.index(joint_name)
        component_qpos = measured.clone()
        component_qpos[index] = candidate[index]
        component_corrections = {
            name: torch.zeros_like(measured).index_copy_(
                0,
                torch.tensor([index], device=measured.device),
                correction.detach().to(measured)[index].reshape(1),
            )
            for name, correction in camera_corrections.items()
        }
        evidence = assemble_update_evidence(
            joint_backend=joint_backend,
            joint_target=joint_target,
            camera_backends=camera_backends,
            camera_targets=camera_targets,
            measured_qpos=measured,
            candidate_qpos=component_qpos,
            camera_corrections=component_corrections,
            selected_indices=[index],
            candidate_bound=candidate_bound,
        )
        after_metrics = _immutable_metrics(joint_backend.metrics(component_qpos, joint_target))
        per_view_visual_gain = _immutable_metrics(
            {
                name: _visual_gain(
                    camera_backends[name],
                    camera_targets[name],
                    measured,
                    component_qpos,
                )
                for name in sorted(camera_backends)
            }
        )
        decision = decide_update(evidence, thresholds)
        fingerprint = _fingerprint(
            joint_name=joint_name,
            candidate_qpos=component_qpos,
            evidence=evidence,
            decision=decision,
            before_metrics=before_metrics,
            after_metrics=after_metrics,
            per_view_visual_gain=per_view_visual_gain,
            thresholds=thresholds,
        )
        verifications.append(
            ComponentVerification(
                joint_name=joint_name,
                candidate_qpos=component_qpos,
                evidence=evidence,
                decision=decision,
                before_metrics=before_metrics,
                after_metrics=after_metrics,
                per_view_visual_gain=per_view_visual_gain,
                fingerprint=fingerprint,
            )
        )
    return ComponentVerificationRun(tuple(verifications))


__all__ = [
    "ComponentVerification",
    "ComponentVerificationRun",
    "verify_candidate_components",
]
