"""Bounded visual recovery of persistent robot joint-state offsets."""

from __future__ import annotations

from typing import Mapping, Sequence

import torch

from kinesync.observation.base import RenderedObservation, StateObservationBackend

from .result import JointOffsetResult


class JointOffsetRecovery:
    """Estimate measurement bias and apply its inverse to selected joints."""

    def __init__(
        self,
        backend: StateObservationBackend,
        *,
        joint_names: Sequence[str],
        selected_joints: Sequence[str],
        steps: int,
        learning_rate: float,
        offset_bound: float,
        seed: int,
    ):
        if steps <= 0 or learning_rate <= 0 or offset_bound <= 0:
            raise ValueError("steps, learning_rate, and offset_bound must be positive")
        if len(set(joint_names)) != len(joint_names):
            raise ValueError("joint_names contains duplicates")
        unknown = set(selected_joints) - set(joint_names)
        if unknown:
            raise ValueError(f"Unknown selected joints: {sorted(unknown)}")
        if not selected_joints:
            raise ValueError("At least one joint must be selected")
        self.backend = backend
        self.joint_names = list(joint_names)
        self.selected_joints = list(selected_joints)
        self.selected_indices = [self.joint_names.index(name) for name in selected_joints]
        self.steps = steps
        self.learning_rate = learning_rate
        self.offset_bound = offset_bound
        self.seed = seed

    def _selection_matrix(self, reference: torch.Tensor) -> torch.Tensor:
        matrix = torch.zeros(
            (len(self.joint_names), len(self.selected_indices)),
            dtype=reference.dtype,
            device=reference.device,
        )
        for column, row in enumerate(self.selected_indices):
            matrix[row, column] = 1.0
        return matrix

    def recover(
        self,
        measured_qpos: torch.Tensor,
        target: Mapping[str, RenderedObservation],
        *,
        reference_qpos: torch.Tensor | None = None,
    ) -> JointOffsetResult:
        if measured_qpos.shape != (len(self.joint_names),):
            raise ValueError(
                f"Expected measured_qpos shape {(len(self.joint_names),)}, "
                f"got {tuple(measured_qpos.shape)}"
            )
        if not measured_qpos.is_floating_point():
            measured_qpos = measured_qpos.to(torch.get_default_dtype())
        if reference_qpos is not None and reference_qpos.shape != measured_qpos.shape:
            raise ValueError("reference_qpos must match measured_qpos")

        torch.manual_seed(self.seed)
        selection = self._selection_matrix(measured_qpos)
        raw = torch.nn.Parameter(
            torch.zeros(
                len(self.selected_indices),
                dtype=measured_qpos.dtype,
                device=measured_qpos.device,
            )
        )
        optimizer = torch.optim.Adam([raw], lr=self.learning_rate)
        trace: list[dict[str, float | int]] = []

        def state() -> tuple[torch.Tensor, torch.Tensor]:
            selected_correction = self.offset_bound * torch.tanh(raw)
            correction = selection @ selected_correction
            return measured_qpos + correction, correction

        for step in range(self.steps):
            optimizer.zero_grad(set_to_none=True)
            corrected, correction = state()
            loss, per_camera = self.backend.loss(corrected, target)
            loss.backward()
            gradient_norm = float(torch.linalg.vector_norm(raw.grad).detach())
            trace.append(
                self._trace_row(
                    step,
                    loss,
                    per_camera,
                    correction,
                    gradient_norm,
                    corrected,
                    reference_qpos,
                )
            )
            optimizer.step()

        with torch.no_grad():
            corrected, correction = state()
            final_loss, final_per_camera = self.backend.loss(corrected, target)
            trace.append(
                self._trace_row(
                    self.steps,
                    final_loss,
                    final_per_camera,
                    correction,
                    0.0,
                    corrected,
                    reference_qpos,
                )
            )

        initial_state_mae = None
        final_state_mae = None
        if reference_qpos is not None:
            initial_state_mae = float(
                (measured_qpos - reference_qpos.to(measured_qpos)).abs().mean()
            )
            final_state_mae = float(
                (corrected - reference_qpos.to(corrected)).abs().mean()
            )
        return JointOffsetResult(
            corrected_qpos=corrected.detach().clone(),
            applied_correction=correction.detach().clone(),
            estimated_measurement_offset=(-correction).detach().clone(),
            initial_loss=float(trace[0]["loss"]),
            final_loss=float(final_loss),
            initial_state_mae=initial_state_mae,
            final_state_mae=final_state_mae,
            trace=trace,
        )

    def _trace_row(
        self,
        step: int,
        loss: torch.Tensor,
        per_camera: Mapping[str, torch.Tensor],
        correction: torch.Tensor,
        gradient_norm: float,
        corrected_qpos: torch.Tensor,
        reference_qpos: torch.Tensor | None,
    ) -> dict[str, float | int]:
        row: dict[str, float | int] = {
            "step": step,
            "loss": float(loss.detach()),
            "gradient_norm": gradient_norm,
        }
        for name, value in per_camera.items():
            row[f"loss_{name}"] = float(value.detach())
        for index, name in enumerate(self.joint_names):
            row[f"correction_{name}"] = float(correction[index].detach())
        if reference_qpos is not None:
            row["state_mae"] = float(
                (corrected_qpos.detach() - reference_qpos.to(corrected_qpos)).abs().mean()
            )
        return row
