"""Bounded joint-bias recovery shared across an ordered state window."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch

from kinesync.observation.real_route_a import trust_prior_loss


@dataclass(frozen=True)
class TemporalJointOffsetResult:
    center_state_id: str
    corrected_qpos: torch.Tensor
    corrected_qpos_by_state: Mapping[str, torch.Tensor]
    applied_correction: torch.Tensor
    estimated_measurement_offset: torch.Tensor
    initial_loss: float
    final_loss: float
    initial_state_mae: float | None
    final_state_mae: float | None
    trace: list[dict[str, Any]]


class TemporalJointOffsetRecovery:
    """Estimate one persistent correction from multiple visual observations."""

    def __init__(
        self,
        backend,
        *,
        joint_names: Sequence[str],
        selected_joints: Sequence[str],
        steps: int,
        learning_rate: float,
        offset_bound: float,
        seed: int,
        prior_weight: float,
        prior_kind: str = "quadratic",
        prior_delta_rad: float = 0.02,
        prior_cauchy_mask: torch.Tensor | None = None,
    ):
        if steps <= 0 or learning_rate <= 0 or offset_bound <= 0:
            raise ValueError("steps, learning_rate, and offset_bound must be positive")
        if prior_weight < 0 or prior_delta_rad <= 0:
            raise ValueError("Temporal prior parameters are invalid")
        if len(set(joint_names)) != len(joint_names):
            raise ValueError("joint_names contains duplicates")
        unknown = set(selected_joints) - set(joint_names)
        if unknown:
            raise ValueError(f"Unknown selected joints: {sorted(unknown)}")
        if not selected_joints:
            raise ValueError("At least one joint must be selected")
        if prior_kind not in {"quadratic", "huber", "cauchy", "adaptive"}:
            raise ValueError("Unsupported temporal prior kind")
        if prior_kind == "adaptive" and (
            prior_cauchy_mask is None
            or prior_cauchy_mask.shape != (len(joint_names),)
        ):
            raise ValueError("Adaptive temporal prior requires a qpos-shaped cauchy mask")
        self.backend = backend
        self.joint_names = list(joint_names)
        self.selected_joints = list(selected_joints)
        self.selected_indices = [
            self.joint_names.index(name) for name in self.selected_joints
        ]
        self.steps = int(steps)
        self.learning_rate = float(learning_rate)
        self.offset_bound = float(offset_bound)
        self.seed = int(seed)
        self.prior_weight = float(prior_weight)
        self.prior_kind = str(prior_kind)
        self.prior_delta_rad = float(prior_delta_rad)
        self.prior_cauchy_mask = (
            None
            if prior_cauchy_mask is None
            else prior_cauchy_mask.detach().clone().bool()
        )

    def _selection_matrix(self, reference: torch.Tensor) -> torch.Tensor:
        matrix = torch.zeros(
            (len(self.joint_names), len(self.selected_indices)),
            dtype=reference.dtype,
            device=reference.device,
        )
        for column, row in enumerate(self.selected_indices):
            matrix[row, column] = 1.0
        return matrix

    def _validate_inputs(
        self,
        state_ids: Sequence[str],
        measured_qpos: Mapping[str, torch.Tensor],
        targets: Mapping[str, Any],
        center_state_id: str,
        reference_qpos: Mapping[str, torch.Tensor] | None,
    ) -> list[str]:
        ordered = [str(value) for value in state_ids]
        if not ordered or len(set(ordered)) != len(ordered):
            raise ValueError("Temporal state IDs must be nonempty and unique")
        expected = set(ordered)
        if set(measured_qpos) != expected or set(targets) != expected:
            raise ValueError("Temporal inputs must use the same state IDs")
        if reference_qpos is not None and set(reference_qpos) != expected:
            raise ValueError("Temporal references must use the same state IDs")
        if center_state_id not in expected:
            raise ValueError("Temporal center state is absent from the window")
        for state_id in ordered:
            value = measured_qpos[state_id]
            if value.shape != (len(self.joint_names),):
                raise ValueError(f"Invalid qpos shape for {state_id}: {tuple(value.shape)}")
            if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
                raise ValueError(f"Invalid measured qpos for {state_id}")
            if reference_qpos is not None:
                reference = reference_qpos[state_id]
                if reference.shape != value.shape or not bool(torch.isfinite(reference).all()):
                    raise ValueError(f"Invalid reference qpos for {state_id}")
        return ordered

    def recover(
        self,
        state_ids: Sequence[str],
        measured_qpos: Mapping[str, torch.Tensor],
        targets: Mapping[str, Any],
        *,
        center_state_id: str,
        reference_qpos: Mapping[str, torch.Tensor] | None = None,
    ) -> TemporalJointOffsetResult:
        ordered = self._validate_inputs(
            state_ids, measured_qpos, targets, center_state_id, reference_qpos
        )
        template = measured_qpos[ordered[0]]
        selection = self._selection_matrix(template)
        torch.manual_seed(self.seed)
        raw = torch.nn.Parameter(
            torch.zeros(
                len(self.selected_indices),
                dtype=template.dtype,
                device=template.device,
            )
        )
        optimizer = torch.optim.Adam([raw], lr=self.learning_rate)
        trace: list[dict[str, Any]] = []

        def state():
            selected = self.offset_bound * torch.tanh(raw)
            correction = selection @ selected
            corrected = {
                state_id: measured_qpos[state_id] + correction for state_id in ordered
            }
            return corrected, correction

        def objective(corrected, correction):
            state_losses = []
            term_values: dict[str, list[torch.Tensor]] = {}
            for state_id in ordered:
                loss, terms = self.backend.loss(corrected[state_id], targets[state_id])
                state_losses.append(loss)
                for name, value in terms.items():
                    term_values.setdefault(str(name), []).append(value)
            visual = torch.stack(state_losses).mean()
            prior = trust_prior_loss(
                corrected[center_state_id],
                measured_qpos[center_state_id],
                kind=self.prior_kind,
                delta_rad=self.prior_delta_rad,
                cauchy_mask=(
                    None
                    if self.prior_cauchy_mask is None
                    else self.prior_cauchy_mask.to(template.device)
                ),
            )
            means = {
                name: torch.stack(values).mean()
                for name, values in term_values.items()
            }
            return visual + self.prior_weight * prior, visual, prior, means

        for step in range(self.steps):
            optimizer.zero_grad(set_to_none=True)
            corrected, correction = state()
            loss, visual, prior, terms = objective(corrected, correction)
            loss.backward()
            if raw.grad is None or not bool(torch.isfinite(raw.grad).all()):
                raise FloatingPointError("Temporal correction produced a nonfinite gradient")
            trace.append(
                self._trace_row(
                    step,
                    loss,
                    visual,
                    prior,
                    terms,
                    correction,
                    float(torch.linalg.vector_norm(raw.grad).detach()),
                    corrected,
                    center_state_id,
                    reference_qpos,
                )
            )
            optimizer.step()

        with torch.no_grad():
            corrected, correction = state()
            final_loss, final_visual, final_prior, final_terms = objective(
                corrected, correction
            )
            trace.append(
                self._trace_row(
                    self.steps,
                    final_loss,
                    final_visual,
                    final_prior,
                    final_terms,
                    correction,
                    0.0,
                    corrected,
                    center_state_id,
                    reference_qpos,
                )
            )

        initial_mae = None
        final_mae = None
        if reference_qpos is not None:
            reference = reference_qpos[center_state_id].to(template)
            initial_mae = float(
                (measured_qpos[center_state_id] - reference).abs().mean()
            )
            final_mae = float((corrected[center_state_id] - reference).abs().mean())
        detached = {
            name: value.detach().clone() for name, value in corrected.items()
        }
        return TemporalJointOffsetResult(
            center_state_id=center_state_id,
            corrected_qpos=detached[center_state_id],
            corrected_qpos_by_state=detached,
            applied_correction=correction.detach().clone(),
            estimated_measurement_offset=(-correction).detach().clone(),
            initial_loss=float(trace[0]["loss"]),
            final_loss=float(final_loss),
            initial_state_mae=initial_mae,
            final_state_mae=final_mae,
            trace=trace,
        )

    def _trace_row(
        self,
        step: int,
        loss: torch.Tensor,
        visual: torch.Tensor,
        prior: torch.Tensor,
        terms: Mapping[str, torch.Tensor],
        correction: torch.Tensor,
        gradient_norm: float,
        corrected: Mapping[str, torch.Tensor],
        center_state_id: str,
        reference_qpos: Mapping[str, torch.Tensor] | None,
    ) -> dict[str, Any]:
        row: dict[str, Any] = {
            "step": int(step),
            "loss": float(loss.detach()),
            "visual_loss": float(visual.detach()),
            "prior_loss": float(prior.detach()),
            "gradient_norm": float(gradient_norm),
        }
        for name, value in terms.items():
            row[f"term_{name}"] = float(value.detach())
        for index, name in enumerate(self.joint_names):
            row[f"correction_{name}"] = float(correction[index].detach())
        if reference_qpos is not None:
            reference = reference_qpos[center_state_id].to(
                corrected[center_state_id]
            )
            row["center_state_mae"] = float(
                (corrected[center_state_id].detach() - reference).abs().mean()
            )
        return row


__all__ = ["TemporalJointOffsetRecovery", "TemporalJointOffsetResult"]
