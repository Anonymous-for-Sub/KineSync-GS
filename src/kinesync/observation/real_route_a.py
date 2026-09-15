"""Real Route-A RGB/mask objective over native articulated Gaussian renders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol

import cv2
import torch
from torch.nn import functional as F

from kinesync.data.schema import RealObservation

from .base import RenderedObservation
from .real_losses import (
    binary_iou,
    boundary_f1,
    boundary_loss,
    masked_rgb_loss,
    soft_iou_loss,
)


class _RenderBuffers(Protocol):
    rgb: torch.Tensor
    alpha: torch.Tensor


class _GaussianRenderer(Protocol):
    cameras: Mapping[str, object]

    def render_buffers(self, qpos: torch.Tensor) -> Mapping[str, _RenderBuffers]: ...

    def render(self, qpos: torch.Tensor) -> Mapping[str, RenderedObservation]: ...


@dataclass(frozen=True)
class RealObservationTarget:
    rgb: torch.Tensor
    mask: torch.Tensor
    signed_distance: torch.Tensor | None = None


@dataclass(frozen=True)
class RealObservationLossWeights:
    mask_iou: float = 1.0
    boundary: float = 0.1
    rgb: float = 0.05
    prior: float = 0.01
    prior_kind: str = "quadratic"
    prior_delta_rad: float = 0.02

    def __post_init__(self) -> None:
        values = (self.mask_iou, self.boundary, self.rgb, self.prior)
        if any(value < 0 for value in values):
            raise ValueError("Real-observation loss weights must be nonnegative")
        if not any(value > 0 for value in values):
            raise ValueError("At least one real-observation loss weight must be positive")
        if self.prior_kind not in {"quadratic", "huber", "cauchy", "adaptive"}:
            raise ValueError(
                "prior_kind must be 'quadratic', 'huber', 'cauchy', or 'adaptive'"
            )
        if self.prior_delta_rad <= 0:
            raise ValueError("prior_delta_rad must be positive")


def trust_prior_loss(
    qpos: torch.Tensor,
    measured_qpos: torch.Tensor,
    *,
    kind: str,
    delta_rad: float,
    cauchy_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Penalize visual correction while allowing robust large-offset recovery."""

    if qpos.shape != measured_qpos.shape:
        raise ValueError("measured_qpos must match optimized qpos")
    if delta_rad <= 0:
        raise ValueError("delta_rad must be positive")
    if kind == "quadratic":
        return (qpos - measured_qpos.to(qpos)).square().mean()
    if kind == "huber":
        return F.smooth_l1_loss(
            qpos,
            measured_qpos.to(qpos),
            beta=delta_rad,
            reduction="mean",
        )
    if kind == "cauchy":
        normalized = (qpos - measured_qpos.to(qpos)) / delta_rad
        return delta_rad**2 * torch.log1p(normalized.square()).mean()
    if kind == "adaptive":
        if cauchy_mask is None or cauchy_mask.shape != qpos.shape:
            raise ValueError("adaptive prior requires a qpos-shaped cauchy_mask")
        difference = qpos - measured_qpos.to(qpos)
        quadratic = difference.square()
        cauchy = delta_rad**2 * torch.log1p((difference / delta_rad).square())
        return torch.where(cauchy_mask.to(device=qpos.device), cauchy, quadratic).mean()
    raise ValueError("Unsupported trust prior kind")


def prepare_real_targets(
    observations: Mapping[str, RealObservation],
    *,
    output_size: tuple[int, int],
    device: torch.device,
) -> dict[str, RealObservationTarget]:
    """Resize immutable real observations to renderer resolution."""

    height, width = output_size
    if height <= 0 or width <= 0:
        raise ValueError("output_size dimensions must be positive")
    prepared = {}
    for camera, observation in observations.items():
        if observation.camera != camera:
            raise ValueError(f"Observation camera mismatch for {camera}")
        rgb = cv2.resize(observation.rgb, (width, height), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(
            observation.mask.astype("uint8"),
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        )
        binary = (mask > 0).astype("uint8")
        inside_distance = cv2.distanceTransform(
            binary, cv2.DIST_L2, cv2.DIST_MASK_PRECISE
        )
        outside_distance = cv2.distanceTransform(
            1 - binary, cv2.DIST_L2, cv2.DIST_MASK_PRECISE
        )
        signed_distance = outside_distance - inside_distance
        prepared[camera] = RealObservationTarget(
            rgb=torch.as_tensor(rgb, dtype=torch.float32, device=device) / 255.0,
            mask=torch.as_tensor(binary, dtype=torch.float32, device=device),
            signed_distance=torch.as_tensor(
                signed_distance, dtype=torch.float32, device=device
            ),
        )
    return prepared


class RealRouteAObservationBackend:
    """Compare articulated Gaussian render buffers with real RGB and masks."""

    def __init__(
        self,
        renderer: _GaussianRenderer,
        *,
        loss_weights: RealObservationLossWeights,
        measured_qpos: torch.Tensor | None = None,
        prior_cauchy_mask: torch.Tensor | None = None,
    ):
        self.renderer = renderer
        self.cameras = dict(renderer.cameras)
        self.loss_weights = loss_weights
        self.measured_qpos = (
            None if measured_qpos is None else measured_qpos.detach().clone()
        )
        self.prior_cauchy_mask = (
            None if prior_cauchy_mask is None else prior_cauchy_mask.detach().clone().bool()
        )

    def _validate_target(
        self, target: Mapping[str, RealObservationTarget]
    ) -> None:
        if set(target) != set(self.cameras):
            raise ValueError(
                f"Target cameras {sorted(target)} do not match renderer {sorted(self.cameras)}"
            )

    def render(self, qpos: torch.Tensor) -> dict[str, RenderedObservation]:
        buffers = self.renderer.render_buffers(qpos)
        return {
            camera: RenderedObservation(
                values=value.alpha,
                valid=torch.ones_like(value.alpha, dtype=torch.bool),
            )
            for camera, value in buffers.items()
        }

    def loss(
        self,
        qpos: torch.Tensor,
        target: Mapping[str, RealObservationTarget],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        self._validate_target(target)
        use_rgb = self.loss_weights.rgb > 0.0
        buffers = self.renderer.render_buffers(qpos) if use_rgb else None
        alpha_observations = None if use_rgb else self.renderer.render(qpos)
        terms: dict[str, torch.Tensor] = {}
        mask_iou_terms = []
        boundary_terms = []
        rgb_terms = []
        for camera in self.cameras:
            expected = target[camera]
            predicted_alpha = (
                buffers[camera].alpha
                if buffers is not None
                else alpha_observations[camera].values
            )
            mask_iou = soft_iou_loss(predicted_alpha, expected.mask)
            boundary = boundary_loss(predicted_alpha, expected.mask)
            rgb = (
                masked_rgb_loss(
                    buffers[camera].rgb, expected.rgb, expected.mask, fit_affine=True
                )
                if buffers is not None
                else qpos.new_zeros(())
            )
            terms[f"mask_iou_{camera}"] = mask_iou
            terms[f"boundary_{camera}"] = boundary
            terms[f"rgb_{camera}"] = rgb
            mask_iou_terms.append(mask_iou)
            boundary_terms.append(boundary)
            rgb_terms.append(rgb)
        mean_mask_iou = torch.stack(mask_iou_terms).mean()
        mean_boundary = torch.stack(boundary_terms).mean()
        mean_rgb = torch.stack(rgb_terms).mean()
        if self.measured_qpos is None:
            prior = qpos.new_zeros(())
        else:
            prior = trust_prior_loss(
                qpos,
                self.measured_qpos,
                kind=self.loss_weights.prior_kind,
                delta_rad=self.loss_weights.prior_delta_rad,
                cauchy_mask=self.prior_cauchy_mask,
            )
        terms["prior"] = prior
        weights = self.loss_weights
        total = (
            weights.mask_iou * mean_mask_iou
            + weights.boundary * mean_boundary
            + weights.rgb * mean_rgb
            + weights.prior * prior
        )
        return total, terms

    def metrics(
        self,
        qpos: torch.Tensor,
        target: Mapping[str, RealObservationTarget],
    ) -> dict[str, float]:
        self._validate_target(target)
        with torch.no_grad():
            buffers = self.renderer.render_buffers(qpos)
            metrics: dict[str, float] = {}
            ious = []
            boundary_scores = []
            rgb_losses = []
            for camera in self.cameras:
                expected = target[camera]
                predicted = buffers[camera]
                iou = float(binary_iou(predicted.alpha, expected.mask))
                boundary_score = float(boundary_f1(predicted.alpha, expected.mask))
                rgb_loss = float(
                    masked_rgb_loss(
                        predicted.rgb, expected.rgb, expected.mask, fit_affine=True
                    )
                )
                metrics[f"mask_iou_{camera}"] = iou
                metrics[f"boundary_f1_{camera}"] = boundary_score
                metrics[f"masked_rgb_loss_{camera}"] = rgb_loss
                ious.append(iou)
                boundary_scores.append(boundary_score)
                rgb_losses.append(rgb_loss)
            metrics["mean_mask_iou"] = sum(ious) / len(ious)
            metrics["mean_boundary_f1"] = sum(boundary_scores) / len(boundary_scores)
            metrics["mean_masked_rgb_loss"] = sum(rgb_losses) / len(rgb_losses)
            return metrics


__all__ = [
    "RealObservationLossWeights",
    "RealObservationTarget",
    "RealRouteAObservationBackend",
    "prepare_real_targets",
    "trust_prior_loss",
]
