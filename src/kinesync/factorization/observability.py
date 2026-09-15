"""Frame-wise gradient observability analysis for physical parameter blocks."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class BlockObservability:
    frame_count: int
    rms_gradient_norm: float
    median_gradient_norm: float
    singular_values: tuple[float, ...]
    retained_rank: int
    median_absolute_gradient: torch.Tensor
    sign_agreement: torch.Tensor
    parameter_mask: torch.Tensor
    accepted: bool


def analyze_block_gradients(
    gradient_matrix: torch.Tensor,
    *,
    rms_min: float,
    relative_parameter_min: float,
    singular_ratio_min: float,
) -> BlockObservability:
    """Summarize whether a frame-by-parameter gradient block can be updated."""

    if gradient_matrix.ndim != 2 or gradient_matrix.shape[0] < 2:
        raise ValueError("Observability requires at least two frames")
    if gradient_matrix.shape[1] == 0:
        raise ValueError("Observability requires at least one parameter")
    if not torch.isfinite(gradient_matrix).all():
        raise ValueError("Observability gradients must be finite")
    if rms_min < 0 or not 0 <= relative_parameter_min <= 1:
        raise ValueError("Invalid observability magnitude thresholds")
    if not 0 <= singular_ratio_min <= 1:
        raise ValueError("singular_ratio_min must be in [0, 1]")

    row_norms = torch.linalg.vector_norm(gradient_matrix, dim=1)
    rms = torch.sqrt(gradient_matrix.square().sum(dim=1).mean())
    median_norm = row_norms.median()
    centered = gradient_matrix - gradient_matrix.mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(centered)
    maximum_singular = singular.max() if singular.numel() else gradient_matrix.new_zeros(())
    if float(maximum_singular) > 0.0:
        retained_rank = int(
            (singular >= maximum_singular * singular_ratio_min).sum().item()
        )
    else:
        retained_rank = 0

    median_absolute = torch.quantile(gradient_matrix.abs(), 0.5, dim=0)
    maximum_parameter = median_absolute.max()
    if float(maximum_parameter) > 0.0:
        parameter_mask = median_absolute >= maximum_parameter * relative_parameter_min
    else:
        parameter_mask = torch.zeros_like(median_absolute, dtype=torch.bool)
    signs = torch.sign(gradient_matrix)
    sign_agreement = signs.mean(dim=0).abs()
    accepted = bool(float(rms) >= rms_min and retained_rank >= 1 and parameter_mask.any())
    if not accepted:
        parameter_mask = torch.zeros_like(parameter_mask)
    return BlockObservability(
        frame_count=int(gradient_matrix.shape[0]),
        rms_gradient_norm=float(rms),
        median_gradient_norm=float(median_norm),
        singular_values=tuple(float(value) for value in singular),
        retained_rank=retained_rank,
        median_absolute_gradient=median_absolute.detach().clone(),
        sign_agreement=sign_agreement.detach().clone(),
        parameter_mask=parameter_mask.detach().clone(),
        accepted=accepted,
    )


__all__ = ["BlockObservability", "analyze_block_gradients"]
