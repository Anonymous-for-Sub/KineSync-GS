"""Temporal shared-bias candidates with RT3-compatible evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch

from kinesync.cli.rt1_real_observation import (
    _evaluate_trial,
    _observability_prior_mask,
)
from kinesync.correction.temporal_offset import (
    TemporalJointOffsetRecovery,
    TemporalJointOffsetResult,
)

from .evidence import assemble_update_evidence
from .schema import UpdateEvidence


@dataclass(frozen=True)
class TemporalGuardCandidateRun:
    joint_result: TemporalJointOffsetResult
    camera_results: Mapping[str, TemporalJointOffsetResult]
    temporal_gradient: torch.Tensor
    prior_cauchy_mask: torch.Tensor
    camera_prior_cauchy_masks: Mapping[str, torch.Tensor]
    evidence: UpdateEvidence
    before_metrics: Mapping[str, float]
    candidate_metrics: Mapping[str, float]
    evaluation: Mapping[str, Any]


def _selected_indices(qpos: torch.Tensor, selected_indices: Sequence[int]) -> list[int]:
    indices = [int(value) for value in selected_indices]
    if not indices or len(set(indices)) != len(indices):
        raise ValueError("selected_indices must be nonempty and unique")
    if min(indices) < 0 or max(indices) >= qpos.numel():
        raise ValueError("selected_indices are outside the qpos vector")
    return indices


def temporal_visual_gradient(
    backend,
    state_ids: Sequence[str],
    measured_qpos: Mapping[str, torch.Tensor],
    targets: Mapping[str, Any],
    *,
    selected_indices: Sequence[int],
) -> torch.Tensor:
    """Differentiate the mean window loss through one shared correction."""

    ordered = [str(value) for value in state_ids]
    if not ordered or len(set(ordered)) != len(ordered):
        raise ValueError("Temporal state IDs must be nonempty and unique")
    expected = set(ordered)
    if set(measured_qpos) != expected or set(targets) != expected:
        raise ValueError("Temporal gradient inputs must use the same state IDs")
    template = measured_qpos[ordered[0]]
    indices = _selected_indices(template, selected_indices)
    selection = torch.zeros(
        (template.numel(), len(indices)),
        dtype=template.dtype,
        device=template.device,
    )
    for column, row in enumerate(indices):
        selection[row, column] = 1.0
    probe = torch.zeros(
        len(indices), dtype=template.dtype, device=template.device, requires_grad=True
    )
    correction = selection @ probe
    losses = [
        backend.loss(measured_qpos[state_id] + correction, targets[state_id])[0]
        for state_id in ordered
    ]
    gradient = torch.autograd.grad(torch.stack(losses).mean(), probe)[0]
    full = selection @ gradient
    if not bool(torch.isfinite(full).all()):
        raise FloatingPointError("Temporal visual gradient is nonfinite")
    return full.detach().clone()


def _recover(
    backend,
    state_ids: Sequence[str],
    measured_qpos: Mapping[str, torch.Tensor],
    targets: Mapping[str, Any],
    reference_qpos: Mapping[str, torch.Tensor],
    *,
    center_state_id: str,
    joint_names: Sequence[str],
    selected_joints: Sequence[str],
    steps: int,
    learning_rate: float,
    offset_bound: float,
    seed: int,
    prior_weight: float,
    prior_kind: str,
    prior_delta_rad: float,
    prior_cauchy_mask: torch.Tensor,
) -> TemporalJointOffsetResult:
    return TemporalJointOffsetRecovery(
        backend,
        joint_names=joint_names,
        selected_joints=selected_joints,
        steps=steps,
        learning_rate=learning_rate,
        offset_bound=offset_bound,
        seed=seed,
        prior_weight=prior_weight,
        prior_kind=prior_kind,
        prior_delta_rad=prior_delta_rad,
        prior_cauchy_mask=prior_cauchy_mask,
    ).recover(
        state_ids,
        measured_qpos,
        targets,
        center_state_id=center_state_id,
        reference_qpos=reference_qpos,
    )


def run_temporal_guard_candidate(
    *,
    state_ids: Sequence[str],
    center_state_id: str,
    joint_visual_backend,
    joint_targets: Mapping[str, Any],
    camera_visual_backends: Mapping[str, object],
    camera_targets: Mapping[str, Mapping[str, Any]],
    measured_qpos: Mapping[str, torch.Tensor],
    reference_qpos: Mapping[str, torch.Tensor],
    joint_names: Sequence[str],
    selected_joints: Sequence[str],
    steps: int,
    learning_rate: float,
    offset_bound: float,
    seed: int,
    prior_weight: float,
    prior_kind: str,
    prior_delta_rad: float,
    minimum_gradient: float,
    minimum_relative: float,
    acceptance: Mapping[str, Any],
) -> TemporalGuardCandidateRun:
    """Run joint and leave-one-camera candidates over one causal window."""

    ordered = [str(value) for value in state_ids]
    names = sorted(camera_visual_backends)
    if names != sorted(camera_targets):
        raise ValueError("Temporal camera backends and targets must match")
    if center_state_id not in ordered:
        raise ValueError("Temporal center state is absent from the window")
    selected = [list(joint_names).index(name) for name in selected_joints]
    temporal_gradient = temporal_visual_gradient(
        joint_visual_backend,
        ordered,
        measured_qpos,
        joint_targets,
        selected_indices=selected,
    )
    prior_mask = _observability_prior_mask(
        temporal_gradient,
        selected_indices=selected,
        minimum_gradient=minimum_gradient,
        minimum_relative=minimum_relative,
    )
    joint_result = _recover(
        joint_visual_backend,
        ordered,
        measured_qpos,
        joint_targets,
        reference_qpos,
        center_state_id=center_state_id,
        joint_names=joint_names,
        selected_joints=selected_joints,
        steps=steps,
        learning_rate=learning_rate,
        offset_bound=offset_bound,
        seed=seed,
        prior_weight=prior_weight,
        prior_kind=prior_kind,
        prior_delta_rad=prior_delta_rad,
        prior_cauchy_mask=prior_mask,
    )

    camera_results: dict[str, TemporalJointOffsetResult] = {}
    camera_masks: dict[str, torch.Tensor] = {}
    for name in names:
        gradient = temporal_visual_gradient(
            camera_visual_backends[name],
            ordered,
            measured_qpos,
            camera_targets[name],
            selected_indices=selected,
        )
        camera_mask = _observability_prior_mask(
            gradient,
            selected_indices=selected,
            minimum_gradient=minimum_gradient,
            minimum_relative=minimum_relative,
        )
        camera_masks[name] = camera_mask
        camera_results[name] = _recover(
            camera_visual_backends[name],
            ordered,
            measured_qpos,
            camera_targets[name],
            reference_qpos,
            center_state_id=center_state_id,
            joint_names=joint_names,
            selected_joints=selected_joints,
            steps=steps,
            learning_rate=learning_rate,
            offset_bound=offset_bound,
            seed=seed,
            prior_weight=prior_weight,
            prior_kind=prior_kind,
            prior_delta_rad=prior_delta_rad,
            prior_cauchy_mask=camera_mask,
        )

    measured_center = measured_qpos[center_state_id]
    reference_center = reference_qpos[center_state_id]
    target_center = joint_targets[center_state_id]
    before_metrics = joint_visual_backend.metrics(measured_center, target_center)
    candidate_metrics = joint_visual_backend.metrics(
        joint_result.corrected_qpos, target_center
    )
    initial_mae = float(
        (measured_center[selected] - reference_center[selected]).abs().mean()
    )
    final_mae = float(
        (joint_result.corrected_qpos[selected] - reference_center[selected]).abs().mean()
    )
    applied_mae = float(joint_result.applied_correction[selected].abs().mean())
    evaluation = _evaluate_trial(
        initial_selected_mae=initial_mae,
        final_selected_mae=final_mae,
        applied_correction_mae=applied_mae,
        before_metrics=before_metrics,
        after_metrics=candidate_metrics,
        acceptance=acceptance,
    )
    evidence = assemble_update_evidence(
        joint_backend=joint_visual_backend,
        joint_target=target_center,
        camera_backends=camera_visual_backends,
        camera_targets={
            name: camera_targets[name][center_state_id] for name in names
        },
        measured_qpos=measured_center,
        candidate_qpos=joint_result.corrected_qpos,
        camera_corrections={
            name: result.applied_correction
            for name, result in camera_results.items()
        },
        selected_indices=selected,
        candidate_bound=offset_bound,
    )
    return TemporalGuardCandidateRun(
        joint_result=joint_result,
        camera_results=camera_results,
        temporal_gradient=temporal_gradient,
        prior_cauchy_mask=prior_mask,
        camera_prior_cauchy_masks=camera_masks,
        evidence=evidence,
        before_metrics=before_metrics,
        candidate_metrics=candidate_metrics,
        evaluation=evaluation,
    )


__all__ = [
    "TemporalGuardCandidateRun",
    "run_temporal_guard_candidate",
    "temporal_visual_gradient",
]
