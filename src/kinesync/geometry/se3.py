"""Stable Torch-native exponential maps for SO(3) and SE(3)."""

from __future__ import annotations

import torch

from .so3 import homogeneous, skew


def _coefficients(rotvec: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    theta_squared = rotvec.square().sum(dim=-1)
    epsilon = torch.finfo(rotvec.dtype).eps
    safe_theta_squared = theta_squared.clamp_min(epsilon)
    safe_theta = torch.sqrt(safe_theta_squared)
    small = theta_squared < 1e-8
    theta_fourth = theta_squared.square()
    a = torch.where(
        small,
        1.0 - theta_squared / 6.0 + theta_fourth / 120.0,
        torch.sin(safe_theta) / safe_theta,
    )
    b = torch.where(
        small,
        0.5 - theta_squared / 24.0 + theta_fourth / 720.0,
        (1.0 - torch.cos(safe_theta)) / safe_theta_squared,
    )
    c = torch.where(
        small,
        1.0 / 6.0 - theta_squared / 120.0 + theta_fourth / 5040.0,
        (safe_theta - torch.sin(safe_theta)) / (safe_theta_squared * safe_theta),
    )
    return a, b, c


def so3_exp(rotvec: torch.Tensor) -> torch.Tensor:
    """Map rotation vectors shaped ``(..., 3)`` to rotation matrices."""

    if rotvec.shape[-1:] != (3,):
        raise ValueError(f"SO(3) rotation vector requires three values, got {rotvec.shape}")
    if not rotvec.is_floating_point():
        rotvec = rotvec.to(torch.get_default_dtype())
    a, b, _ = _coefficients(rotvec)
    omega = skew(rotvec)
    identity = torch.eye(3, dtype=rotvec.dtype, device=rotvec.device).expand(
        rotvec.shape[:-1] + (3, 3)
    )
    return identity + a[..., None, None] * omega + b[..., None, None] * (omega @ omega)


def se3_exp(twist: torch.Tensor) -> torch.Tensor:
    """Map rotation-then-translation twists shaped ``(..., 6)`` to SE(3)."""

    if twist.shape[-1:] != (6,):
        raise ValueError(f"SE(3) twist requires six values, got {twist.shape}")
    if not twist.is_floating_point():
        twist = twist.to(torch.get_default_dtype())
    rotvec = twist[..., :3]
    velocity = twist[..., 3:]
    a, b, c = _coefficients(rotvec)
    omega = skew(rotvec)
    omega_squared = omega @ omega
    identity = torch.eye(3, dtype=twist.dtype, device=twist.device).expand(
        twist.shape[:-1] + (3, 3)
    )
    rotation = identity + a[..., None, None] * omega + b[..., None, None] * omega_squared
    left_jacobian = identity + b[..., None, None] * omega + c[..., None, None] * omega_squared
    translation = (left_jacobian @ velocity[..., None]).squeeze(-1)
    return homogeneous(rotation, translation)


__all__ = ["se3_exp", "so3_exp", "skew"]
