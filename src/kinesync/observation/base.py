"""Shared contracts and link-bound point posing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol

import torch

from kinesync.geometry.urdf import TorchURDFKinematics


@dataclass(frozen=True)
class RenderedObservation:
    values: torch.Tensor
    valid: torch.Tensor


class StateObservationBackend(Protocol):
    def render(self, qpos: torch.Tensor) -> dict[str, RenderedObservation]: ...

    def loss(
        self, qpos: torch.Tensor, target: Mapping[str, RenderedObservation]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]: ...


def pose_link_points(
    kinematics: TorchURDFKinematics,
    qpos: torch.Tensor,
    local_xyz: torch.Tensor,
    link_index: torch.Tensor,
    link_names: Mapping[int, str],
) -> torch.Tensor:
    """Pose link-local points into the URDF root coordinate system."""

    transforms = kinematics.forward(qpos)
    ordered_indices = sorted(link_names)
    if ordered_indices != list(range(len(ordered_indices))):
        raise ValueError("Link indices must be contiguous from zero")
    matrices = torch.stack([transforms[link_names[index]] for index in ordered_indices], dim=-3)
    indices = link_index.to(device=qpos.device, dtype=torch.long)
    if indices.ndim != 1 or local_xyz.shape != (indices.numel(), 3):
        raise ValueError("local_xyz must be (N, 3) and link_index must be (N,)")
    if int(indices.min()) < 0 or int(indices.max()) >= len(ordered_indices):
        raise ValueError("Point references an unknown link index")
    point_matrices = matrices[..., indices, :, :]
    points = local_xyz.to(device=qpos.device, dtype=qpos.dtype)
    batch_shape = qpos.shape[:-1]
    points = points.expand(batch_shape + points.shape)
    return (
        torch.matmul(point_matrices[..., :3, :3], points[..., None]).squeeze(-1)
        + point_matrices[..., :3, 3]
    )
