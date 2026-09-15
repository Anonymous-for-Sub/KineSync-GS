"""Correspondence-aware projected Gaussian-center observations."""

from __future__ import annotations

from typing import Mapping

import torch

from kinesync.geometry.urdf import TorchURDFKinematics

from .base import RenderedObservation, pose_link_points
from .camera import TorchCamera


class ProjectedCentersBackend:
    """Diagnostic backend measuring calibrated 2D center reprojection."""

    def __init__(
        self,
        kinematics: TorchURDFKinematics,
        local_xyz: torch.Tensor,
        link_index: torch.Tensor,
        link_names: Mapping[int, str],
        cameras: Mapping[str, TorchCamera],
    ):
        if not cameras:
            raise ValueError("At least one camera is required")
        self.kinematics = kinematics
        self.local_xyz = local_xyz
        self.link_index = link_index
        self.link_names = dict(link_names)
        self.cameras = dict(cameras)

    def render(self, qpos: torch.Tensor) -> dict[str, RenderedObservation]:
        world_xyz = pose_link_points(
            self.kinematics,
            qpos,
            self.local_xyz,
            self.link_index,
            self.link_names,
        )
        rendered = {}
        for name, camera in self.cameras.items():
            pixels, _, valid = camera.project(world_xyz)
            rendered[name] = RenderedObservation(values=pixels, valid=valid)
        return rendered

    def loss(
        self,
        qpos: torch.Tensor,
        target: Mapping[str, RenderedObservation],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        prediction = self.render(qpos)
        per_camera: dict[str, torch.Tensor] = {}
        for name, camera in self.cameras.items():
            if name not in target:
                raise KeyError(f"Target lacks camera {name}")
            mask = prediction[name].valid & target[name].valid.to(prediction[name].valid)
            if not bool(mask.any()):
                raise ValueError(f"No mutually visible projected centers for camera {name}")
            scale = torch.tensor(
                [camera.width, camera.height],
                dtype=qpos.dtype,
                device=qpos.device,
            )
            difference = (
                prediction[name].values[mask] - target[name].values.to(qpos)[mask]
            ) / scale
            per_camera[name] = difference.square().mean()
        return torch.stack(list(per_camera.values())).mean(), per_camera
