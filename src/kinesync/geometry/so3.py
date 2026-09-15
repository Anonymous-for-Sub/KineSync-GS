"""Small Torch-native rotation helpers."""

from __future__ import annotations

import torch


def skew(vector: torch.Tensor) -> torch.Tensor:
    """Return the skew-symmetric matrix for vectors shaped ``(..., 3)``."""

    if vector.shape[-1] != 3:
        raise ValueError(f"Expected (..., 3), got {tuple(vector.shape)}")
    x, y, z = vector.unbind(dim=-1)
    zero = torch.zeros_like(x)
    return torch.stack(
        (
            zero,
            -z,
            y,
            z,
            zero,
            -x,
            -y,
            x,
            zero,
        ),
        dim=-1,
    ).reshape(vector.shape[:-1] + (3, 3))


def axis_angle_matrix(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Rodrigues rotation for a normalized axis and arbitrary angle batch."""

    axis = axis / torch.linalg.vector_norm(axis).clamp_min(
        torch.finfo(axis.dtype).eps
    )
    matrix = skew(axis)
    identity = torch.eye(3, dtype=axis.dtype, device=axis.device)
    return (
        identity
        + torch.sin(angle)[..., None, None] * matrix
        + (1.0 - torch.cos(angle))[..., None, None] * (matrix @ matrix)
    )


def rpy_matrix(rpy: torch.Tensor) -> torch.Tensor:
    """URDF fixed-axis roll-pitch-yaw rotation matrix."""

    if rpy.shape != (3,):
        raise ValueError(f"Expected (3,) RPY, got {tuple(rpy.shape)}")
    roll, pitch, yaw = rpy.unbind()
    one = torch.ones((), dtype=rpy.dtype, device=rpy.device)
    zero = torch.zeros((), dtype=rpy.dtype, device=rpy.device)
    rx = torch.stack(
        (one, zero, zero, zero, torch.cos(roll), -torch.sin(roll), zero, torch.sin(roll), torch.cos(roll))
    ).reshape(3, 3)
    ry = torch.stack(
        (torch.cos(pitch), zero, torch.sin(pitch), zero, one, zero, -torch.sin(pitch), zero, torch.cos(pitch))
    ).reshape(3, 3)
    rz = torch.stack(
        (torch.cos(yaw), -torch.sin(yaw), zero, torch.sin(yaw), torch.cos(yaw), zero, zero, zero, one)
    ).reshape(3, 3)
    return rz @ ry @ rx


def homogeneous(rotation: torch.Tensor, translation: torch.Tensor) -> torch.Tensor:
    """Compose batched rotation and translation into homogeneous transforms."""

    batch_shape = torch.broadcast_shapes(rotation.shape[:-2], translation.shape[:-1])
    rotation = rotation.expand(batch_shape + (3, 3))
    translation = translation.expand(batch_shape + (3,))
    upper = torch.cat((rotation, translation[..., None]), dim=-1)
    bottom = torch.zeros(batch_shape + (1, 4), dtype=rotation.dtype, device=rotation.device)
    bottom[..., 0, 3] = 1.0
    return torch.cat((upper, bottom), dim=-2)


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrices to real-part-first quaternions."""

    if matrix.shape[-2:] != (3, 3):
        raise ValueError(f"Expected (..., 3, 3), got {tuple(matrix.shape)}")
    m00, m01, m02 = matrix[..., 0, 0], matrix[..., 0, 1], matrix[..., 0, 2]
    m10, m11, m12 = matrix[..., 1, 0], matrix[..., 1, 1], matrix[..., 1, 2]
    m20, m21, m22 = matrix[..., 2, 0], matrix[..., 2, 1], matrix[..., 2, 2]
    radicand = torch.stack(
        (
            1.0 + m00 + m11 + m22,
            1.0 + m00 - m11 - m22,
            1.0 - m00 + m11 - m22,
            1.0 - m00 - m11 + m22,
        ),
        dim=-1,
    )
    q_abs = torch.zeros_like(radicand)
    positive = radicand > 0.0
    q_abs[positive] = torch.sqrt(radicand[positive])
    candidates = torch.stack(
        (
            torch.stack((q_abs[..., 0].square(), m21 - m12, m02 - m20, m10 - m01), dim=-1),
            torch.stack((m21 - m12, q_abs[..., 1].square(), m10 + m01, m02 + m20), dim=-1),
            torch.stack((m02 - m20, m10 + m01, q_abs[..., 2].square(), m12 + m21), dim=-1),
            torch.stack((m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3].square()), dim=-1),
        ),
        dim=-2,
    )
    denominator = 2.0 * q_abs.clamp_min(0.1)[..., :, None]
    candidates = candidates / denominator
    best = q_abs.argmax(dim=-1)
    gather_index = best[..., None, None].expand(best.shape + (1, 4))
    quaternion = candidates.gather(dim=-2, index=gather_index).squeeze(-2)
    return torch.nn.functional.normalize(quaternion, dim=-1)
