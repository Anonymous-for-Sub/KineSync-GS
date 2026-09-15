"""Configuration and result schemas for physical factorization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch

from .observability import BlockObservability


@dataclass(frozen=True)
class FactorizationConfig:
    joint_count: int
    camera_names: tuple[str, ...]
    joint_bound: float
    camera_rotation_bound: float
    camera_translation_bound: float
    joint_steps: int = 20
    camera_steps: int = 30
    refinement_steps: int = 20
    joint_learning_rate: float = 0.05
    camera_learning_rate: float = 0.03
    batch_size: int = 4
    seed: int = 0
    rms_gradient_min: float = 1e-4
    relative_parameter_min: float = 0.1
    singular_ratio_min: float = 1e-3
    joint_prior_weight: float = 0.0
    joint_prior_delta: float = 0.02
    camera_rotation_prior_weight: float = 0.0
    camera_translation_prior_weight: float = 0.0

    def __post_init__(self) -> None:
        if self.joint_count <= 0:
            raise ValueError("joint_count must be positive")
        if not self.camera_names or len(set(self.camera_names)) != len(
            self.camera_names
        ):
            raise ValueError("camera_names must be nonempty and unique")
        if min(
            self.joint_bound,
            self.camera_rotation_bound,
            self.camera_translation_bound,
            self.joint_learning_rate,
            self.camera_learning_rate,
            self.joint_prior_delta,
        ) <= 0:
            raise ValueError("factor bounds, learning rates, and prior delta must be positive")
        if min(self.joint_steps, self.camera_steps, self.refinement_steps) < 0:
            raise ValueError("stage step counts must be nonnegative")
        if self.joint_steps + self.camera_steps + self.refinement_steps == 0:
            raise ValueError("at least one optimization stage must have steps")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.rms_gradient_min < 0:
            raise ValueError("rms_gradient_min must be nonnegative")
        if not 0 <= self.relative_parameter_min <= 1:
            raise ValueError("relative_parameter_min must be in [0, 1]")
        if not 0 <= self.singular_ratio_min <= 1:
            raise ValueError("singular_ratio_min must be in [0, 1]")
        if min(
            self.joint_prior_weight,
            self.camera_rotation_prior_weight,
            self.camera_translation_prior_weight,
        ) < 0:
            raise ValueError("prior weights must be nonnegative")


@dataclass(frozen=True)
class FactorizationResult:
    joint_zero: torch.Tensor
    camera_twists: Mapping[str, torch.Tensor]
    initial_loss: float
    final_loss: float
    trace: list[dict[str, Any]]
    observability: Mapping[str, BlockObservability]
    nonfinite_skips: int
    train_state_ids: tuple[str, ...]


__all__ = ["FactorizationConfig", "FactorizationResult"]
