"""Multi-frame real-mask backend for shared joint and camera factors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence

import numpy as np
import torch

from kinesync.observation.base import RenderedObservation
from kinesync.observation.real_losses import (
    binary_iou,
    boundary_f1,
    boundary_loss,
    soft_iou_loss,
)
from kinesync.observation.real_route_a import RealObservationTarget


class _DynamicRenderer(Protocol):
    cameras: Mapping[str, object]

    def render_alpha_with_camera_deltas(
        self, qpos: torch.Tensor, camera_twists: Mapping[str, torch.Tensor]
    ) -> Mapping[str, RenderedObservation]: ...


@dataclass(frozen=True)
class FactorFrame:
    state_id: str
    split: str
    qpos: torch.Tensor
    targets: Mapping[str, RealObservationTarget]


def select_evenly(values: Sequence[str], count: int) -> list[str]:
    """Select deterministic, endpoint-preserving entries from an ordered sequence."""

    if count <= 0:
        raise ValueError("count must be positive")
    if count > len(values):
        raise ValueError(f"Cannot select {count} values from {len(values)}")
    indices = np.linspace(0, len(values) - 1, count).round().astype(np.int64)
    if len(set(indices.tolist())) != count:
        raise ValueError("Even selection produced duplicate indices")
    return [str(values[index]) for index in indices]


class RouteAFactorizationBackend:
    """Evaluate shared arm-joint and camera factors over real-mask frames."""

    def __init__(
        self,
        renderer: _DynamicRenderer,
        *,
        arm_joint_count: int,
        boundary_weight: float,
    ):
        if arm_joint_count <= 0:
            raise ValueError("arm_joint_count must be positive")
        if boundary_weight < 0:
            raise ValueError("boundary_weight must be nonnegative")
        self.renderer = renderer
        self.cameras = dict(renderer.cameras)
        self.arm_joint_count = arm_joint_count
        self.boundary_weight = float(boundary_weight)

    def _validate_frame(self, frame: FactorFrame) -> None:
        if frame.qpos.ndim != 1 or frame.qpos.numel() < self.arm_joint_count:
            raise ValueError(f"Invalid qpos shape for {frame.state_id}: {frame.qpos.shape}")
        if set(frame.targets) != set(self.cameras):
            raise ValueError(
                f"Target cameras {sorted(frame.targets)} do not match renderer "
                f"{sorted(self.cameras)}"
            )

    def _calibrated_qpos(
        self, frame: FactorFrame, joint_zero: torch.Tensor
    ) -> torch.Tensor:
        if joint_zero.shape != (self.arm_joint_count,):
            raise ValueError(
                f"Expected {self.arm_joint_count} joint-zero values, got {joint_zero.shape}"
            )
        padding = torch.zeros(
            frame.qpos.numel() - self.arm_joint_count,
            dtype=joint_zero.dtype,
            device=joint_zero.device,
        )
        return frame.qpos.to(joint_zero) + torch.cat((joint_zero, padding))

    def frame_loss(
        self,
        frame: FactorFrame,
        joint_zero: torch.Tensor,
        camera_twists: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        self._validate_frame(frame)
        if set(camera_twists) != set(self.cameras):
            raise ValueError("Camera-twist keys do not match factorization cameras")
        qpos = self._calibrated_qpos(frame, joint_zero)
        predictions = self.renderer.render_alpha_with_camera_deltas(
            qpos, camera_twists
        )
        terms: dict[str, torch.Tensor] = {}
        camera_losses = []
        for camera in self.cameras:
            prediction = predictions[camera].values
            target = frame.targets[camera].mask.to(prediction)
            iou = soft_iou_loss(prediction, target)
            boundary = boundary_loss(prediction, target)
            terms[f"{camera}_iou"] = iou
            terms[f"{camera}_boundary"] = boundary
            camera_losses.append(iou + self.boundary_weight * boundary)
        return torch.stack(camera_losses).mean(), terms

    def batch_loss(
        self,
        frames: Sequence[FactorFrame],
        joint_zero: torch.Tensor,
        camera_twists: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if not frames:
            raise ValueError("At least one factor frame is required")
        if any(frame.split != "train" for frame in frames):
            raise ValueError("Factor fitting accepts train frames only")
        losses = []
        terms: dict[str, torch.Tensor] = {}
        for frame in frames:
            loss, frame_terms = self.frame_loss(frame, joint_zero, camera_twists)
            losses.append(loss)
            for name, value in frame_terms.items():
                terms[f"{frame.state_id}/{name}"] = value
        return torch.stack(losses).mean(), terms

    def metrics(
        self,
        frames: Sequence[FactorFrame],
        joint_zero: torch.Tensor,
        camera_twists: Mapping[str, torch.Tensor],
    ) -> dict[str, float]:
        if not frames:
            raise ValueError("At least one factor frame is required")
        ious = []
        boundaries = []
        metrics: dict[str, float] = {}
        with torch.no_grad():
            for frame in frames:
                self._validate_frame(frame)
                qpos = self._calibrated_qpos(frame, joint_zero)
                predictions = self.renderer.render_alpha_with_camera_deltas(
                    qpos, camera_twists
                )
                for camera in self.cameras:
                    prediction = predictions[camera].values
                    target = frame.targets[camera].mask.to(prediction)
                    iou = float(binary_iou(prediction, target))
                    boundary = float(boundary_f1(prediction, target))
                    metrics[f"{frame.state_id}/{camera}_mask_iou"] = iou
                    metrics[f"{frame.state_id}/{camera}_boundary_f1"] = boundary
                    ious.append(iou)
                    boundaries.append(boundary)
        metrics["mean_mask_iou"] = sum(ious) / len(ious)
        metrics["mean_boundary_f1"] = sum(boundaries) / len(boundaries)
        return metrics


__all__ = ["FactorFrame", "RouteAFactorizationBackend", "select_evenly"]
