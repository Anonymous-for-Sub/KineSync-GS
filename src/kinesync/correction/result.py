"""Correction result schemas."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class JointOffsetResult:
    corrected_qpos: torch.Tensor
    applied_correction: torch.Tensor
    estimated_measurement_offset: torch.Tensor
    initial_loss: float
    final_loss: float
    initial_state_mae: float | None
    final_state_mae: float | None
    trace: list[dict[str, Any]]
