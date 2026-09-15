"""Validation-only adapter for the inherited RoboSplat renderer."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Mapping

import numpy as np
import torch

from kinesync.data.schema import CameraCalibration, GaussianAnchorSet


def render_legacy_rgb(
    *,
    robosplat_root: Path,
    urdf_path: Path,
    anchors: GaussianAnchorSet,
    anchor_indices: np.ndarray,
    link_names: Mapping[int, str],
    qpos: np.ndarray,
    calibrations: Mapping[str, CameraCalibration],
    output_size: tuple[int, int],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Render matched RGB through the old code for one-off regression checks."""

    sys.path.insert(0, str(robosplat_root.resolve()))
    try:
        from data_aug.util.gaussian_util import GaussianRenderer
        from gaussian_process.optimize_mesh_bound_gaussian import (
            TensorCamera,
            build_gaussian,
            evaluate_local_sh_degree1,
            frame_transforms,
            pose_state,
        )
        from gaussian_process.urdf_kinematics import URDFKinematics
    finally:
        sys.path.pop(0)

    indices = np.asarray(anchor_indices, dtype=np.int64)
    rotations, translations = frame_transforms(
        URDFKinematics(urdf_path), qpos, dict(link_names), device
    )
    xyz, quaternion, per_rotation = pose_state(
        torch.as_tensor(anchors.xyz[indices], dtype=torch.float32, device=device),
        torch.as_tensor(
            anchors.surface_frame[indices], dtype=torch.float32, device=device
        ),
        torch.as_tensor(anchors.link_index[indices], dtype=torch.long, device=device),
        rotations,
        translations,
    )
    gaussian = build_gaussian(
        xyz,
        quaternion,
        torch.as_tensor(anchors.log_scale[indices], dtype=torch.float32, device=device),
        torch.as_tensor(
            anchors.opacity_logit[indices], dtype=torch.float32, device=device
        ),
    )
    height, width = output_size
    rendered = {}
    for name, calibration in calibrations.items():
        c2w = torch.as_tensor(
            calibration.camera_to_world, dtype=torch.float32, device=device
        )
        fovx = 2.0 * math.atan(
            calibration.width / (2.0 * calibration.intrinsic[0, 0])
        )
        fovy = 2.0 * math.atan(
            calibration.height / (2.0 * calibration.intrinsic[1, 1])
        )
        camera = TensorCamera(c2w, fovx=fovx, fovy=fovy, width=width, height=height)
        direction_base = camera.camera_center[None] - xyz
        direction_local = torch.bmm(
            per_rotation.transpose(1, 2), direction_base.unsqueeze(-1)
        ).squeeze(-1)
        colors = evaluate_local_sh_degree1(
            torch.as_tensor(
                anchors.features_dc[indices], dtype=torch.float32, device=device
            ),
            torch.as_tensor(
                anchors.features_rest_degree1[indices],
                dtype=torch.float32,
                device=device,
            ),
            direction_local,
        ).clamp_min(0.0)
        rendered[name] = GaussianRenderer(camera).render_with_depth_alpha(
            gaussian, colors_precomp=colors
        ).rgb
    return rendered
