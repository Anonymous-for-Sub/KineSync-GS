"""Matched joint and leave-one-camera candidates for guard evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch

from kinesync.cli.rt1_real_observation import _evaluate_trial
from kinesync.correction.joint_offset import JointOffsetRecovery
from kinesync.correction.result import JointOffsetResult

from .evidence import assemble_update_evidence
from .schema import UpdateEvidence


@dataclass(frozen=True)
class GuardCandidateRun:
    joint_result: JointOffsetResult
    camera_results: Mapping[str, JointOffsetResult]
    evidence: UpdateEvidence
    before_metrics: Mapping[str, float]
    candidate_metrics: Mapping[str, float]
    evaluation: Mapping[str, Any]


def _recover(
    backend,
    target,
    *,
    measured_qpos: torch.Tensor,
    reference_qpos: torch.Tensor,
    joint_names: Sequence[str],
    selected_joints: Sequence[str],
    steps: int,
    learning_rate: float,
    offset_bound: float,
    seed: int,
) -> JointOffsetResult:
    return JointOffsetRecovery(
        backend,
        joint_names=joint_names,
        selected_joints=selected_joints,
        steps=steps,
        learning_rate=learning_rate,
        offset_bound=offset_bound,
        seed=seed,
    ).recover(measured_qpos, target, reference_qpos=reference_qpos)


def run_guard_candidate(
    *,
    joint_backend,
    joint_visual_backend,
    joint_target,
    camera_backends: Mapping[str, object],
    camera_visual_backends: Mapping[str, object],
    camera_targets: Mapping[str, object],
    measured_qpos: torch.Tensor,
    reference_qpos: torch.Tensor,
    joint_names: Sequence[str],
    selected_joints: Sequence[str],
    steps: int,
    learning_rate: float,
    offset_bound: float,
    seed: int,
    acceptance: Mapping[str, Any],
) -> GuardCandidateRun:
    """Run matched candidates and assemble deployment-visible evidence."""

    names = sorted(camera_backends)
    if names != sorted(camera_visual_backends) or names != sorted(camera_targets):
        raise ValueError("Candidate camera backends and targets must match")
    selected_indices = [list(joint_names).index(name) for name in selected_joints]
    joint_result = _recover(
        joint_backend,
        joint_target,
        measured_qpos=measured_qpos,
        reference_qpos=reference_qpos,
        joint_names=joint_names,
        selected_joints=selected_joints,
        steps=steps,
        learning_rate=learning_rate,
        offset_bound=offset_bound,
        seed=seed,
    )
    camera_results = {
        name: _recover(
            camera_backends[name],
            camera_targets[name],
            measured_qpos=measured_qpos,
            reference_qpos=reference_qpos,
            joint_names=joint_names,
            selected_joints=selected_joints,
            steps=steps,
            learning_rate=learning_rate,
            offset_bound=offset_bound,
            seed=seed,
        )
        for name in names
    }
    evidence = assemble_update_evidence(
        joint_backend=joint_visual_backend,
        joint_target=joint_target,
        camera_backends=camera_visual_backends,
        camera_targets=camera_targets,
        measured_qpos=measured_qpos,
        candidate_qpos=joint_result.corrected_qpos,
        camera_corrections={
            name: result.applied_correction for name, result in camera_results.items()
        },
        selected_indices=selected_indices,
        candidate_bound=offset_bound,
    )
    before_metrics = joint_visual_backend.metrics(measured_qpos, joint_target)
    candidate_metrics = joint_visual_backend.metrics(
        joint_result.corrected_qpos, joint_target
    )
    initial_mae = float(
        (measured_qpos[selected_indices] - reference_qpos[selected_indices]).abs().mean()
    )
    final_mae = float(
        (
            joint_result.corrected_qpos[selected_indices]
            - reference_qpos[selected_indices]
        )
        .abs()
        .mean()
    )
    applied_mae = float(
        joint_result.applied_correction[selected_indices].abs().mean()
    )
    evaluation = _evaluate_trial(
        initial_selected_mae=initial_mae,
        final_selected_mae=final_mae,
        applied_correction_mae=applied_mae,
        before_metrics=before_metrics,
        after_metrics=candidate_metrics,
        acceptance=acceptance,
    )
    return GuardCandidateRun(
        joint_result=joint_result,
        camera_results=camera_results,
        evidence=evidence,
        before_metrics=before_metrics,
        candidate_metrics=candidate_metrics,
        evaluation=evaluation,
    )


__all__ = ["GuardCandidateRun", "run_guard_candidate"]
