"""Calibrated OpenCV pinhole camera in Torch."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from kinesync.data.schema import CameraCalibration


@dataclass(frozen=True)
class TorchCamera:
    name: str
    intrinsic: torch.Tensor
    world_to_camera: torch.Tensor
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.intrinsic.shape != (3, 3):
            raise ValueError("intrinsic must have shape (3, 3)")
        if self.world_to_camera.shape != (4, 4):
            raise ValueError("world_to_camera must have shape (4, 4)")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Camera dimensions must be positive")

    @classmethod
    def from_calibration(
        cls,
        calibration: CameraCalibration,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> "TorchCamera":
        return cls(
            name=calibration.name,
            intrinsic=torch.as_tensor(calibration.intrinsic, device=device, dtype=dtype),
            world_to_camera=torch.as_tensor(
                calibration.world_to_camera, device=device, dtype=dtype
            ),
            width=calibration.width,
            height=calibration.height,
        )

    def project(
        self, world_xyz: torch.Tensor, *, min_depth: float = 1e-4
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return pixel coordinates, depth, and finite in-image visibility."""

        if world_xyz.shape[-1] != 3:
            raise ValueError("world_xyz must end with dimension 3")
        rotation = self.world_to_camera[:3, :3].to(world_xyz)
        translation = self.world_to_camera[:3, 3].to(world_xyz)
        camera_xyz = torch.einsum("ij,...nj->...ni", rotation, world_xyz) + translation
        depth = camera_xyz[..., 2]
        safe_depth = depth.clamp_min(min_depth)
        intrinsic = self.intrinsic.to(world_xyz)
        u = intrinsic[0, 0] * camera_xyz[..., 0] / safe_depth + intrinsic[0, 2]
        v = intrinsic[1, 1] * camera_xyz[..., 1] / safe_depth + intrinsic[1, 2]
        pixels = torch.stack((u, v), dim=-1)
        valid = (
            (depth > min_depth)
            & torch.isfinite(pixels).all(dim=-1)
            & (u >= 0.0)
            & (u <= self.width - 1)
            & (v >= 0.0)
            & (v <= self.height - 1)
        )
        return pixels, depth, valid
