"""Differentiable evidence extraction for conditional physical updates."""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import torch

from .schema import UpdateEvidence


def _selected_indices(
    qpos: torch.Tensor, selected_indices: Sequence[int]
) -> list[int]:
    indices = [int(value) for value in selected_indices]
    if not indices or len(set(indices)) != len(indices):
        raise ValueError("selected_indices must be nonempty and unique")
    if min(indices) < 0 or max(indices) >= qpos.numel():
        raise ValueError("selected_indices are outside the qpos vector")
    return indices


def _cosine(first: torch.Tensor, second: torch.Tensor, *, epsilon: float) -> float:
    first_norm = torch.linalg.vector_norm(first)
    second_norm = torch.linalg.vector_norm(second)
    denominator = first_norm * second_norm
    if not torch.isfinite(denominator) or float(denominator) <= epsilon:
        return 0.0
    return float(torch.dot(first.reshape(-1), second.reshape(-1)) / denominator)


def selected_visual_gradient(
    backend,
    qpos: torch.Tensor,
    target,
    selected_indices: Sequence[int],
) -> torch.Tensor:
    """Return a detached visual-loss gradient for selected physical joints."""

    indices = _selected_indices(qpos, selected_indices)
    probe = qpos.detach().clone().requires_grad_(True)
    loss, _ = backend.loss(probe, target)
    gradient = torch.autograd.grad(loss, probe)[0]
    return gradient[indices].detach().clone()


def correction_consistency(
    first: torch.Tensor,
    second: torch.Tensor,
    selected_indices: Sequence[int],
    *,
    epsilon: float = 1e-8,
) -> tuple[float, float]:
    """Return directional cosine and normalized correction disagreement."""

    if first.shape != second.shape or first.ndim != 1:
        raise ValueError("Correction vectors must be matching one-dimensional tensors")
    indices = _selected_indices(first, selected_indices)
    first_selected = first[indices].detach()
    second_selected = second[indices].detach().to(first_selected)
    first_norm = torch.linalg.vector_norm(first_selected)
    second_norm = torch.linalg.vector_norm(second_selected)
    denominator = first_norm + second_norm
    if float(denominator) <= epsilon:
        return 0.0, 0.0
    cosine = _cosine(first_selected, second_selected, epsilon=epsilon)
    disagreement = float(
        torch.linalg.vector_norm(first_selected - second_selected)
        / denominator.clamp_min(epsilon)
    )
    return cosine, disagreement


def assemble_update_evidence(
    *,
    joint_backend,
    joint_target,
    camera_backends: Mapping[str, object],
    camera_targets: Mapping[str, object],
    measured_qpos: torch.Tensor,
    candidate_qpos: torch.Tensor,
    camera_corrections: Mapping[str, torch.Tensor],
    selected_indices: Sequence[int],
    candidate_bound: float,
    epsilon: float = 1e-8,
) -> UpdateEvidence:
    """Assemble all deployment-time guard evidence from matched candidates."""

    if measured_qpos.shape != candidate_qpos.shape or measured_qpos.ndim != 1:
        raise ValueError("Measured and candidate qpos must be matching vectors")
    indices = _selected_indices(measured_qpos, selected_indices)
    if candidate_bound <= 0:
        raise ValueError("candidate_bound must be positive")
    if set(camera_backends) != set(camera_targets) or set(camera_backends) != set(
        camera_corrections
    ):
        raise ValueError("Camera backends, targets, and corrections must match")

    with torch.no_grad():
        before_loss = joint_backend.loss(measured_qpos, joint_target)[0]
        after_loss = joint_backend.loss(candidate_qpos, joint_target)[0]
    visual_gain = float(
        (before_loss - after_loss) / before_loss.abs().clamp_min(epsilon)
    )
    names = sorted(camera_backends)
    gradients = [
        selected_visual_gradient(
            camera_backends[name], measured_qpos, camera_targets[name], indices
        )
        for name in names
    ]
    gradient_cosine = (
        _cosine(gradients[0], gradients[1], epsilon=epsilon)
        if len(gradients) >= 2
        else 0.0
    )
    correction_cosine = 0.0
    correction_disagreement = 0.0
    if len(names) >= 2:
        correction_cosine, correction_disagreement = correction_consistency(
            camera_corrections[names[0]],
            camera_corrections[names[1]],
            indices,
            epsilon=epsilon,
        )
    candidate_correction = candidate_qpos - measured_qpos
    bound_fraction = float(candidate_correction[indices].abs().max() / candidate_bound)
    values = (
        visual_gain,
        gradient_cosine,
        correction_cosine,
        correction_disagreement,
        bound_fraction,
    )
    return UpdateEvidence(
        view_count=len(names),
        visual_gain_ratio=visual_gain,
        gradient_cosine=gradient_cosine,
        correction_cosine=correction_cosine,
        relative_correction_disagreement=correction_disagreement,
        candidate_bound_fraction=bound_fraction,
        finite=all(math.isfinite(value) for value in values),
    )


__all__ = [
    "assemble_update_evidence",
    "correction_consistency",
    "selected_visual_gradient",
]
