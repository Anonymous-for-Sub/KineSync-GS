"""Differentiable real-observation losses and thresholded reporting metrics."""

from __future__ import annotations

from typing import Sequence

import torch
from torch.nn import functional as F


def _validate_mask_pair(prediction: torch.Tensor, target: torch.Tensor) -> None:
    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError(
            f"Expected matching HxW masks, got {prediction.shape} and {target.shape}"
        )


def soft_iou_loss(
    prediction: torch.Tensor, target: torch.Tensor, *, epsilon: float = 1e-7
) -> torch.Tensor:
    """One minus soft intersection-over-union for HxW occupancy tensors."""

    _validate_mask_pair(prediction, target)
    target = target.to(prediction)
    intersection = (prediction * target).sum()
    union = prediction.sum() + target.sum() - intersection
    return 1.0 - (intersection + epsilon) / (union + epsilon)


def _sobel_boundary(mask: torch.Tensor) -> torch.Tensor:
    mask_4d = mask[None, None]
    kernel_x = mask.new_tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
    )[None, None]
    kernel_y = kernel_x.transpose(-1, -2)
    gradient_x = F.conv2d(mask_4d, kernel_x, padding=1)
    gradient_y = F.conv2d(mask_4d, kernel_y, padding=1)
    return torch.sqrt(gradient_x.square() + gradient_y.square() + 1e-12)[0, 0]


def boundary_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L1 discrepancy between differentiable Sobel boundary magnitudes."""

    _validate_mask_pair(prediction, target)
    target = target.to(prediction)
    return (_sobel_boundary(prediction) - _sobel_boundary(target)).abs().mean()


def truncated_sdf_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    signed_distance: torch.Tensor,
    *,
    radii: Sequence[float] = (4.0, 12.0),
) -> torch.Tensor:
    """Align a soft silhouette with a fixed multi-scale signed distance field."""

    _validate_mask_pair(prediction, target)
    if signed_distance.shape != target.shape or signed_distance.ndim != 2:
        raise ValueError(
            "signed_distance must have the same HxW shape as the target"
        )
    scales = tuple(float(value) for value in radii)
    if not scales or any(
        value <= 0 or not torch.isfinite(torch.tensor(value)) for value in scales
    ):
        raise ValueError("TSDF radii must be finite and positive")
    target = target.to(prediction)
    distance = signed_distance.to(prediction)
    if not bool(torch.isfinite(distance).all()):
        raise ValueError("signed_distance must be finite")
    residual = prediction - target
    terms = [
        (
            residual
            * distance.clamp(min=-radius, max=radius)
            / radius
        ).mean()
        for radius in scales
    ]
    return torch.stack(terms).mean()


def masked_rgb_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    fit_affine: bool = True,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Robust RGB discrepancy inside a target robot mask."""

    if prediction.shape != target.shape or prediction.ndim != 3 or prediction.shape[-1] != 3:
        raise ValueError(
            f"Expected matching HxWx3 RGB tensors, got {prediction.shape} and {target.shape}"
        )
    if mask.shape != prediction.shape[:2]:
        raise ValueError(f"Mask shape {mask.shape} does not match RGB {prediction.shape[:2]}")
    target = target.to(prediction)
    weights = mask.to(prediction).clamp(0.0, 1.0)[..., None]
    denominator = weights.sum().clamp_min(epsilon)
    aligned = prediction
    if fit_affine:
        prediction_mean = (prediction * weights).sum(dim=(0, 1)) / denominator
        target_mean = (target * weights).sum(dim=(0, 1)) / denominator
        centered_prediction = prediction - prediction_mean
        centered_target = target - target_mean
        covariance = (weights * centered_prediction * centered_target).sum(dim=(0, 1))
        variance = (weights * centered_prediction.square()).sum(dim=(0, 1))
        gain = (covariance / variance.clamp_min(epsilon)).detach()
        bias = (target_mean - gain * prediction_mean).detach()
        aligned = prediction * gain + bias
    residual = torch.sqrt((aligned - target).square() + epsilon**2) - epsilon
    return (residual * weights).sum() / (denominator * prediction.shape[-1])


def binary_iou(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    threshold: float = 0.5,
    epsilon: float = 1e-7,
) -> torch.Tensor:
    """Thresholded IoU used only for reporting."""

    _validate_mask_pair(prediction, target)
    predicted = prediction >= threshold
    expected = target.to(prediction) >= threshold
    intersection = (predicted & expected).sum().to(prediction.dtype)
    union = (predicted | expected).sum().to(prediction.dtype)
    return torch.where(
        union > 0,
        intersection / union.clamp_min(epsilon),
        torch.ones_like(union),
    )


def boundary_f1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    threshold: float = 0.5,
    tolerance_px: int = 1,
    epsilon: float = 1e-7,
) -> torch.Tensor:
    """Tolerance-aware F1 between thresholded Sobel boundaries."""

    _validate_mask_pair(prediction, target)
    if tolerance_px < 0:
        raise ValueError("tolerance_px must be nonnegative")
    predicted_boundary = (_sobel_boundary((prediction >= threshold).to(prediction)) > 0.1).to(
        prediction.dtype
    )
    target_boundary = (
        _sobel_boundary((target.to(prediction) >= threshold).to(prediction)) > 0.1
    ).to(prediction.dtype)
    kernel = 2 * tolerance_px + 1
    predicted_dilated = F.max_pool2d(
        predicted_boundary[None, None], kernel, stride=1, padding=tolerance_px
    )[0, 0]
    target_dilated = F.max_pool2d(
        target_boundary[None, None], kernel, stride=1, padding=tolerance_px
    )[0, 0]
    predicted_count = predicted_boundary.sum()
    target_count = target_boundary.sum()
    if float(predicted_count) == 0.0 and float(target_count) == 0.0:
        return prediction.new_tensor(1.0)
    precision = (predicted_boundary * target_dilated).sum() / predicted_count.clamp_min(epsilon)
    recall = (target_boundary * predicted_dilated).sum() / target_count.clamp_min(epsilon)
    return 2.0 * precision * recall / (precision + recall).clamp_min(epsilon)


__all__ = [
    "binary_iou",
    "boundary_f1",
    "boundary_loss",
    "masked_rgb_loss",
    "soft_iou_loss",
    "truncated_sdf_loss",
]
