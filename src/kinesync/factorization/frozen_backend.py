"""Real-mask backend with immutable per-camera physical factors."""

from __future__ import annotations

from typing import Mapping, Sequence
import math


import torch

from kinesync.observation.base import RenderedObservation
from kinesync.observation.real_losses import (
    binary_iou,
    boundary_f1,
    boundary_loss,
    soft_iou_loss,
    truncated_sdf_loss,
)
from kinesync.observation.real_route_a import (
    RealObservationTarget,
    trust_prior_loss,
)


class FrozenCameraObservationBackend:
    """Optimize qpos while applying a fixed camera-twist calibration."""

    def __init__(
        self,
        renderer,
        *,
        camera_twists: Mapping[str, torch.Tensor],
        boundary_weight: float,
        sdf_weight: float = 0.0,
        sdf_radii: Sequence[float] = (4.0, 12.0),
        prior_weight: float = 0.0,
        measured_qpos: torch.Tensor | None = None,
        prior_kind: str = "quadratic",
        prior_delta_rad: float = 0.02,
        prior_cauchy_mask: torch.Tensor | None = None,
    ):
        self.renderer = renderer
        self.cameras = dict(renderer.cameras)
        if set(camera_twists) != set(self.cameras):
            raise ValueError("Frozen camera twists do not match renderer cameras")
        if boundary_weight < 0 or prior_weight < 0 or prior_delta_rad <= 0:
            raise ValueError("Loss weights must be nonnegative and delta positive")
        radii = tuple(float(value) for value in sdf_radii)
        if (
            sdf_weight < 0
            or not radii
            or any(value <= 0 or not math.isfinite(value) for value in radii)
        ):
            raise ValueError("TSDF weight/radii must be nonnegative/positive")
        self.camera_twists = {
            name: value.detach().clone() for name, value in camera_twists.items()
        }
        self.boundary_weight = float(boundary_weight)
        self.sdf_weight = float(sdf_weight)
        self.sdf_radii = radii
        self.prior_weight = float(prior_weight)
        self.measured_qpos = (
            None if measured_qpos is None else measured_qpos.detach().clone()
        )
        self.prior_kind = prior_kind
        self.prior_delta_rad = float(prior_delta_rad)
        self.prior_cauchy_mask = (
            None
            if prior_cauchy_mask is None
            else prior_cauchy_mask.detach().clone().bool()
        )

    def _validate_target(
        self, target: Mapping[str, RealObservationTarget]
    ) -> None:
        if set(target) != set(self.cameras):
            raise ValueError("Real targets do not match frozen-factor cameras")

    def render(self, qpos: torch.Tensor) -> dict[str, RenderedObservation]:
        twists = {name: value.to(qpos) for name, value in self.camera_twists.items()}
        return dict(self.renderer.render_alpha_with_camera_deltas(qpos, twists))

    def loss(
        self,
        qpos: torch.Tensor,
        target: Mapping[str, RealObservationTarget],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        self._validate_target(target)
        predictions = self.render(qpos)
        terms: dict[str, torch.Tensor] = {}
        visual_terms = []
        for camera in self.cameras:
            expected = target[camera].mask.to(qpos)
            prediction = predictions[camera].values
            iou = soft_iou_loss(prediction, expected)
            boundary = boundary_loss(prediction, expected)
            terms[f"mask_iou_{camera}"] = iou
            terms[f"boundary_{camera}"] = boundary
            visual = iou + self.boundary_weight * boundary
            if self.sdf_weight > 0:
                signed_distance = target[camera].signed_distance
                if signed_distance is None:
                    raise ValueError(
                        "Positive sdf_weight requires target signed distance"
                    )
                sdf = truncated_sdf_loss(
                    prediction,
                    expected,
                    signed_distance,
                    radii=self.sdf_radii,
                )
                terms[f"tsdf_{camera}"] = sdf
                visual = visual + self.sdf_weight * sdf
            visual_terms.append(visual)
        if self.measured_qpos is None:
            prior = qpos.new_zeros(())
        else:
            prior = trust_prior_loss(
                qpos,
                self.measured_qpos.to(qpos),
                kind=self.prior_kind,
                delta_rad=self.prior_delta_rad,
                cauchy_mask=(
                    None
                    if self.prior_cauchy_mask is None
                    else self.prior_cauchy_mask.to(qpos.device)
                ),
            )
        terms["prior"] = prior
        return torch.stack(visual_terms).mean() + self.prior_weight * prior, terms

    def metrics(
        self,
        qpos: torch.Tensor,
        target: Mapping[str, RealObservationTarget],
    ) -> dict[str, float]:
        self._validate_target(target)
        metrics: dict[str, float] = {}
        ious = []
        boundaries = []
        with torch.no_grad():
            predictions = self.render(qpos)
            for camera in self.cameras:
                expected = target[camera].mask.to(qpos)
                prediction = predictions[camera].values
                iou = float(binary_iou(prediction, expected))
                boundary = float(boundary_f1(prediction, expected))
                metrics[f"mask_iou_{camera}"] = iou
                metrics[f"boundary_f1_{camera}"] = boundary
                ious.append(iou)
                boundaries.append(boundary)
        metrics["mean_mask_iou"] = sum(ious) / len(ious)
        metrics["mean_boundary_f1"] = sum(boundaries) / len(boundaries)
        return metrics


__all__ = ["FrozenCameraObservationBackend"]
