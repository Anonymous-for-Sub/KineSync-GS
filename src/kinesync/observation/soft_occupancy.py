"""Low-resolution differentiable occupancy splats from Gaussian centers."""

from __future__ import annotations

from typing import Mapping

import torch

from kinesync.geometry.urdf import TorchURDFKinematics

from .base import RenderedObservation, pose_link_points
from .camera import TorchCamera


class SoftOccupancyBackend:
    def __init__(
        self,
        kinematics: TorchURDFKinematics,
        local_xyz: torch.Tensor,
        link_index: torch.Tensor,
        link_names: Mapping[int, str],
        cameras: Mapping[str, TorchCamera],
        *,
        output_size: tuple[int, int],
        sigma_px: float = 1.5,
        chunk_size: int = 256,
    ):
        height, width = output_size
        if height <= 0 or width <= 0 or sigma_px <= 0 or chunk_size <= 0:
            raise ValueError("Output dimensions, sigma_px, and chunk_size must be positive")
        self.kinematics = kinematics
        self.local_xyz = local_xyz
        self.link_index = link_index
        self.link_names = dict(link_names)
        self.cameras = dict(cameras)
        self.output_size = output_size
        self.sigma_px = sigma_px
        self.chunk_size = chunk_size

    def _splat(self, world_xyz: torch.Tensor, camera: TorchCamera) -> torch.Tensor:
        pixels, depth, valid = camera.project(world_xyz)
        height, width = self.output_size
        scale = torch.tensor(
            [width / camera.width, height / camera.height],
            dtype=world_xyz.dtype,
            device=world_xyz.device,
        )
        pixels = pixels * scale
        y, x = torch.meshgrid(
            torch.arange(height, dtype=world_xyz.dtype, device=world_xyz.device),
            torch.arange(width, dtype=world_xyz.dtype, device=world_xyz.device),
            indexing="ij",
        )
        grid = torch.stack((x, y), dim=-1).reshape(-1, 2)
        density = torch.zeros(
            world_xyz.shape[:-2] + (height * width,),
            dtype=world_xyz.dtype,
            device=world_xyz.device,
        )
        point_count = pixels.shape[-2]
        for start in range(0, point_count, self.chunk_size):
            stop = min(start + self.chunk_size, point_count)
            chunk_pixels = pixels[..., start:stop, :]
            chunk_valid = valid[..., start:stop]
            squared_distance = (
                chunk_pixels[..., :, None, :] - grid[..., None, :, :]
            ).square().sum(dim=-1)
            contributions = torch.exp(
                -0.5 * squared_distance / (self.sigma_px * self.sigma_px)
            )
            contributions = contributions * chunk_valid[..., :, None]
            density = density + contributions.sum(dim=-2)
        occupancy = 1.0 - torch.exp(-0.7 * density)
        return occupancy.reshape(world_xyz.shape[:-2] + (height, width))

    def render(self, qpos: torch.Tensor) -> dict[str, RenderedObservation]:
        world_xyz = pose_link_points(
            self.kinematics,
            qpos,
            self.local_xyz,
            self.link_index,
            self.link_names,
        )
        return {
            name: RenderedObservation(
                values=self._splat(world_xyz, camera),
                valid=torch.ones(self.output_size, dtype=torch.bool, device=qpos.device),
            )
            for name, camera in self.cameras.items()
        }

    def loss(
        self,
        qpos: torch.Tensor,
        target: Mapping[str, RenderedObservation],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        prediction = self.render(qpos)
        per_camera = {
            name: (prediction[name].values - target[name].values.to(qpos)).square().mean()
            for name in self.cameras
        }
        return torch.stack(list(per_camera.values())).mean(), per_camera
