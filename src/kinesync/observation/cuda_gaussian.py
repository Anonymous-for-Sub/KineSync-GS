"""Native Graphdeco CUDA rasterization of state-bound Route-A Gaussians."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import torch
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)

from kinesync.data.schema import CameraCalibration, GaussianAnchorSet
from kinesync.geometry.se3 import se3_exp
from kinesync.geometry.so3 import matrix_to_quaternion
from kinesync.geometry.urdf import TorchURDFKinematics

from .base import RenderedObservation


C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199


def virtual_world_camera_transform(
    world_to_camera: torch.Tensor,
    camera_to_world: torch.Tensor,
    twist: torch.Tensor,
) -> torch.Tensor:
    """Transform world Gaussians so a fixed camera matches a corrected camera."""

    if world_to_camera.shape != (4, 4) or camera_to_world.shape != (4, 4):
        raise ValueError("Camera transforms must be 4x4")
    if twist.shape != (6,):
        raise ValueError("Camera twist must contain six values")
    corrected_world_to_camera = se3_exp(twist) @ world_to_camera.to(twist)
    return camera_to_world.to(twist) @ corrected_world_to_camera


def _projection_matrix(
    *, znear: float, zfar: float, fovx: float, fovy: float, device: torch.device
) -> torch.Tensor:
    tan_x = math.tan(fovx * 0.5)
    tan_y = math.tan(fovy * 0.5)
    matrix = torch.zeros((4, 4), dtype=torch.float32, device=device)
    matrix[0, 0] = 1.0 / tan_x
    matrix[1, 1] = 1.0 / tan_y
    matrix[3, 2] = 1.0
    matrix[2, 2] = zfar / (zfar - znear)
    matrix[2, 3] = -(zfar * znear) / (zfar - znear)
    return matrix


@dataclass(frozen=True)
class CudaRasterCamera:
    name: str
    image_width: int
    image_height: int
    FoVx: float
    FoVy: float
    world_view_transform: torch.Tensor
    full_proj_transform: torch.Tensor
    camera_center: torch.Tensor
    world_to_camera: torch.Tensor
    camera_to_world: torch.Tensor

    @classmethod
    def from_calibration(
        cls,
        calibration: CameraCalibration,
        *,
        output_size: tuple[int, int],
        device: torch.device,
        znear: float = 0.01,
        zfar: float = 10.0,
    ) -> "CudaRasterCamera":
        height, width = output_size
        fx = float(calibration.intrinsic[0, 0])
        fy = float(calibration.intrinsic[1, 1])
        fovx = 2.0 * math.atan(calibration.width / (2.0 * fx))
        fovy = 2.0 * math.atan(calibration.height / (2.0 * fy))
        world_to_camera = torch.as_tensor(
            calibration.world_to_camera, dtype=torch.float32, device=device
        )
        camera_to_world = torch.as_tensor(
            calibration.camera_to_world, dtype=torch.float32, device=device
        )
        view = world_to_camera.transpose(0, 1).contiguous()
        projection = _projection_matrix(
            znear=znear, zfar=zfar, fovx=fovx, fovy=fovy, device=device
        ).transpose(0, 1).contiguous()
        full = view.unsqueeze(0).bmm(projection.unsqueeze(0)).squeeze(0)
        return cls(
            name=calibration.name,
            image_width=width,
            image_height=height,
            FoVx=fovx,
            FoVy=fovy,
            world_view_transform=view,
            full_proj_transform=full,
            camera_center=camera_to_world[:3, 3],
            world_to_camera=world_to_camera,
            camera_to_world=camera_to_world,
        )


@dataclass(frozen=True)
class GaussianRenderBuffers:
    """Native RGB and alpha buffers rendered from one articulated pose."""

    rgb: torch.Tensor
    alpha: torch.Tensor


class CudaGaussianBackend:
    """Differentiable RGB or alpha observations from articulated Gaussians."""

    def __init__(
        self,
        *,
        kinematics: TorchURDFKinematics,
        local_xyz: torch.Tensor,
        canonical_frame: torch.Tensor,
        link_index: torch.Tensor,
        features_dc: torch.Tensor,
        features_l1: torch.Tensor,
        opacity_logit: torch.Tensor,
        log_scale: torch.Tensor,
        link_names: Mapping[int, str],
        cameras: Mapping[str, CudaRasterCamera],
        observation_mode: str,
        background: Sequence[float] = (0.0, 0.0, 0.0),
    ):
        if observation_mode not in {"alpha", "rgb"}:
            raise ValueError("observation_mode must be 'alpha' or 'rgb'")
        if local_xyz.device.type != "cuda":
            raise ValueError("CudaGaussianBackend requires CUDA tensors")
        self.kinematics = kinematics
        self.local_xyz = local_xyz
        self.canonical_frame = canonical_frame
        self.link_index = link_index.long()
        self.features_dc = features_dc
        self.features_l1 = features_l1
        self.opacity_logit = opacity_logit
        self.log_scale = log_scale
        self.link_names = dict(link_names)
        self.cameras = dict(cameras)
        self.observation_mode = observation_mode
        self.background = tuple(float(value) for value in background)

    @classmethod
    def from_route_a(
        cls,
        *,
        kinematics: TorchURDFKinematics,
        anchors: GaussianAnchorSet,
        anchor_indices: np.ndarray,
        link_names: Mapping[int, str],
        calibrations: Mapping[str, CameraCalibration],
        output_size: tuple[int, int],
        observation_mode: str,
        device: torch.device,
    ) -> "CudaGaussianBackend":
        indices = np.asarray(anchor_indices, dtype=np.int64)
        cameras = {
            name: CudaRasterCamera.from_calibration(
                calibration, output_size=output_size, device=device
            )
            for name, calibration in calibrations.items()
        }
        tensor = lambda value: torch.as_tensor(value[indices], dtype=torch.float32, device=device)
        return cls(
            kinematics=kinematics,
            local_xyz=tensor(anchors.xyz),
            canonical_frame=tensor(anchors.surface_frame),
            link_index=torch.as_tensor(
                anchors.link_index[indices], dtype=torch.long, device=device
            ),
            features_dc=tensor(anchors.features_dc),
            features_l1=tensor(anchors.features_rest_degree1),
            opacity_logit=tensor(anchors.opacity_logit),
            log_scale=tensor(anchors.log_scale),
            link_names=link_names,
            cameras=cameras,
            observation_mode=observation_mode,
        )

    def _pose_frames(
        self, qpos: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if qpos.ndim != 1:
            raise ValueError("CUDA Gaussian rasterization currently expects one qpos state")
        transforms = self.kinematics.forward(qpos)
        ordered = sorted(self.link_names)
        matrices = torch.stack(
            [transforms[self.link_names[index]] for index in ordered], dim=0
        )
        point_matrices = matrices[self.link_index]
        rotations = point_matrices[:, :3, :3]
        xyz = torch.bmm(rotations, self.local_xyz.unsqueeze(-1)).squeeze(-1)
        xyz = xyz + point_matrices[:, :3, 3]
        posed_frame = torch.bmm(rotations, self.canonical_frame)
        return xyz, posed_frame, rotations

    def _pose(
        self, qpos: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        xyz, posed_frame, rotations = self._pose_frames(qpos)
        return xyz, matrix_to_quaternion(posed_frame), rotations

    def _local_sh_color(
        self,
        posed_xyz: torch.Tensor,
        per_rotation: torch.Tensor,
        camera: CudaRasterCamera,
    ) -> torch.Tensor:
        direction_base = camera.camera_center[None] - posed_xyz
        direction_local = torch.bmm(
            per_rotation.transpose(1, 2), direction_base.unsqueeze(-1)
        ).squeeze(-1)
        direction_local = torch.nn.functional.normalize(direction_local, dim=-1, eps=1e-8)
        x = direction_local[:, 0:1]
        y = direction_local[:, 1:2]
        z = direction_local[:, 2:3]
        color = C0 * self.features_dc
        color = color - SH_C1 * y * self.features_l1[:, :, 0]
        color = color + SH_C1 * z * self.features_l1[:, :, 1]
        color = color - SH_C1 * x * self.features_l1[:, :, 2]
        return (color + 0.5).clamp_min(0.0)

    def _rasterize(
        self,
        camera: CudaRasterCamera,
        posed_xyz: torch.Tensor,
        quaternion: torch.Tensor,
        colors: torch.Tensor,
        background: Sequence[float],
    ) -> torch.Tensor:
        settings = GaussianRasterizationSettings(
            image_height=camera.image_height,
            image_width=camera.image_width,
            tanfovx=math.tan(camera.FoVx * 0.5),
            tanfovy=math.tan(camera.FoVy * 0.5),
            bg=torch.tensor(background, dtype=torch.float32, device=posed_xyz.device),
            scale_modifier=1.0,
            viewmatrix=camera.world_view_transform,
            projmatrix=camera.full_proj_transform,
            sh_degree=0,
            campos=camera.camera_center,
            prefiltered=False,
            debug=False,
            antialiasing=True,
        )
        rasterizer = GaussianRasterizer(raster_settings=settings)
        image, _, _ = rasterizer(
            means3D=posed_xyz,
            means2D=torch.zeros_like(posed_xyz),
            shs=None,
            colors_precomp=colors,
            opacities=torch.sigmoid(self.opacity_logit),
            scales=torch.exp(self.log_scale),
            rotations=torch.nn.functional.normalize(quaternion, dim=-1),
        )
        return image.clamp(0.0, 1.0).permute(1, 2, 0)

    def render(self, qpos: torch.Tensor) -> dict[str, RenderedObservation]:
        posed_xyz, quaternion, rotations = self._pose(qpos)
        rendered = {}
        for name, camera in self.cameras.items():
            if self.observation_mode == "alpha":
                rgb = self._rasterize(
                    camera,
                    posed_xyz,
                    quaternion,
                    torch.ones_like(posed_xyz),
                    (0.0, 0.0, 0.0),
                )
                values = rgb[..., 0]
            else:
                colors = self._local_sh_color(posed_xyz, rotations, camera)
                values = self._rasterize(
                    camera, posed_xyz, quaternion, colors, self.background
                )
            rendered[name] = RenderedObservation(
                values=values,
                valid=torch.ones(values.shape[:2], dtype=torch.bool, device=qpos.device),
            )
        return rendered

    def render_buffers(self, qpos: torch.Tensor) -> dict[str, GaussianRenderBuffers]:
        """Render RGB and alpha while sharing articulated pose computation."""

        posed_xyz, quaternion, rotations = self._pose(qpos)
        rendered = {}
        for name, camera in self.cameras.items():
            colors = self._local_sh_color(posed_xyz, rotations, camera)
            rgb = self._rasterize(
                camera, posed_xyz, quaternion, colors, self.background
            )
            alpha_rgb = self._rasterize(
                camera,
                posed_xyz,
                quaternion,
                torch.ones_like(posed_xyz),
                (0.0, 0.0, 0.0),
            )
            rendered[name] = GaussianRenderBuffers(rgb=rgb, alpha=alpha_rgb[..., 0])
        return rendered

    def render_alpha_with_camera_deltas(
        self,
        qpos: torch.Tensor,
        camera_twists: Mapping[str, torch.Tensor],
    ) -> dict[str, RenderedObservation]:
        """Render alpha through differentiable left camera-pose perturbations."""

        if set(camera_twists) != set(self.cameras):
            raise ValueError(
                f"Camera twists {sorted(camera_twists)} do not match renderer "
                f"{sorted(self.cameras)}"
            )
        posed_xyz, posed_frames, _ = self._pose_frames(qpos)
        rendered = {}
        for name, camera in self.cameras.items():
            twist = camera_twists[name]
            virtual = virtual_world_camera_transform(
                camera.world_to_camera, camera.camera_to_world, twist
            )
            rotation = virtual[:3, :3]
            translation = virtual[:3, 3]
            virtual_xyz = posed_xyz @ rotation.transpose(0, 1) + translation
            virtual_frames = rotation.unsqueeze(0) @ posed_frames
            quaternion = matrix_to_quaternion(virtual_frames)
            alpha_rgb = self._rasterize(
                camera,
                virtual_xyz,
                quaternion,
                torch.ones_like(virtual_xyz),
                (0.0, 0.0, 0.0),
            )
            alpha = alpha_rgb[..., 0]
            rendered[name] = RenderedObservation(
                values=alpha,
                valid=torch.ones_like(alpha, dtype=torch.bool),
            )
        return rendered

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
