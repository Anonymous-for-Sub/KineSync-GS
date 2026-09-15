"""Alternating bounded optimizer for shared joint and camera factors."""

from __future__ import annotations

from typing import Mapping, Sequence

import torch

from kinesync.observation.real_route_a import trust_prior_loss

from .backend import FactorFrame
from .observability import BlockObservability, analyze_block_gradients
from .schema import FactorizationConfig, FactorizationResult


class PhysicalFactorOptimizer:
    """Fit shared joint-zero and per-camera SE(3) factors on train frames."""

    def __init__(self, backend: object, config: FactorizationConfig):
        self.backend = backend
        self.config = config
        backend_cameras = tuple(getattr(backend, "cameras", {}).keys())
        if set(backend_cameras) != set(config.camera_names):
            raise ValueError("Backend cameras do not match factorization config")
        backend_joint_count = getattr(backend, "arm_joint_count", None)
        if backend_joint_count != config.joint_count:
            raise ValueError("Backend joint count does not match factorization config")

    def fit(self, frames: Sequence[FactorFrame]) -> FactorizationResult:
        if len(frames) < 2:
            raise ValueError("At least two train frames are required")
        if any(frame.split != "train" for frame in frames):
            raise ValueError("Physical factor fitting accepts train frames only")
        state_ids = [frame.state_id for frame in frames]
        if len(set(state_ids)) != len(state_ids):
            raise ValueError("Factor frame state IDs must be unique")

        config = self.config
        torch.manual_seed(config.seed)
        reference = frames[0].qpos
        dtype = reference.dtype if reference.is_floating_point() else torch.float32
        device = reference.device
        joint_raw = torch.nn.Parameter(
            torch.zeros(config.joint_count, dtype=dtype, device=device)
        )
        camera_raw = {
            name: torch.nn.Parameter(torch.zeros(6, dtype=dtype, device=device))
            for name in config.camera_names
        }

        initial_joint, initial_camera = self._physical_factors(joint_raw, camera_raw)
        observability: dict[str, BlockObservability] = {}
        initial_reports = self._observability(
            frames, initial_joint.detach(), initial_camera
        )
        observability["initial_joint"] = initial_reports[0]
        observability["initial_camera"] = initial_reports[1]
        initial_loss = self._dataset_loss(frames, joint_raw, camera_raw)
        trace: list[dict[str, object]] = []
        nonfinite_skips = 0
        global_step = 0

        global_step, skipped = self._run_stage(
            "joint_warmup",
            config.joint_steps,
            frames,
            joint_raw,
            camera_raw,
            observability["initial_joint"].parameter_mask,
            global_step,
            trace,
        )
        nonfinite_skips += skipped
        global_step, skipped = self._run_stage(
            "camera_warmup",
            config.camera_steps,
            frames,
            joint_raw,
            camera_raw,
            observability["initial_camera"].parameter_mask,
            global_step,
            trace,
        )
        nonfinite_skips += skipped

        warmed_joint, warmed_camera = self._physical_factors(joint_raw, camera_raw)
        post_reports = self._observability(
            frames, warmed_joint.detach(), warmed_camera
        )
        observability["post_warmup_joint"] = post_reports[0]
        observability["post_warmup_camera"] = post_reports[1]
        global_step, skipped = self._run_stage(
            "joint_refinement",
            config.refinement_steps,
            frames,
            joint_raw,
            camera_raw,
            observability["post_warmup_joint"].parameter_mask,
            global_step,
            trace,
        )
        nonfinite_skips += skipped

        joint_zero, camera_twists = self._physical_factors(joint_raw, camera_raw)
        final_loss = self._dataset_loss(frames, joint_raw, camera_raw)
        if not torch.isfinite(torch.tensor(final_loss)):
            raise RuntimeError("Factorization did not retain a finite final state")
        return FactorizationResult(
            joint_zero=joint_zero.detach().cpu().clone(),
            camera_twists={
                name: value.detach().cpu().clone()
                for name, value in camera_twists.items()
            },
            initial_loss=initial_loss,
            final_loss=final_loss,
            trace=trace,
            observability=observability,
            nonfinite_skips=nonfinite_skips,
            train_state_ids=tuple(state_ids),
        )

    def _physical_factors(
        self,
        joint_raw: torch.Tensor,
        camera_raw: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        config = self.config
        joint = config.joint_bound * torch.tanh(joint_raw)
        cameras = {}
        for name in config.camera_names:
            raw = camera_raw[name]
            cameras[name] = torch.cat(
                (
                    config.camera_rotation_bound * torch.tanh(raw[:3]),
                    config.camera_translation_bound * torch.tanh(raw[3:]),
                )
            )
        return joint, cameras

    def _objective(
        self,
        frames: Sequence[FactorFrame],
        joint_raw: torch.Tensor,
        camera_raw: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        joint, cameras = self._physical_factors(joint_raw, camera_raw)
        visual, _ = self.backend.batch_loss(frames, joint, cameras)
        config = self.config
        joint_prior = trust_prior_loss(
            joint,
            torch.zeros_like(joint),
            kind="huber",
            delta_rad=config.joint_prior_delta,
        )
        rotations = torch.cat([cameras[name][:3] for name in config.camera_names])
        translations = torch.cat([cameras[name][3:] for name in config.camera_names])
        return (
            visual
            + config.joint_prior_weight * joint_prior
            + config.camera_rotation_prior_weight * rotations.square().mean()
            + config.camera_translation_prior_weight * translations.square().mean()
        )

    def _dataset_loss(
        self,
        frames: Sequence[FactorFrame],
        joint_raw: torch.Tensor,
        camera_raw: Mapping[str, torch.Tensor],
    ) -> float:
        joint, cameras = self._physical_factors(joint_raw, camera_raw)
        with torch.no_grad():
            visual_losses = [
                self.backend.frame_loss(frame, joint, cameras)[0] for frame in frames
            ]
            visual = torch.stack(visual_losses).mean()
            config = self.config
            joint_prior = trust_prior_loss(
                joint,
                torch.zeros_like(joint),
                kind="huber",
                delta_rad=config.joint_prior_delta,
            )
            rotations = torch.cat(
                [cameras[name][:3] for name in config.camera_names]
            )
            translations = torch.cat(
                [cameras[name][3:] for name in config.camera_names]
            )
            loss = (
                visual
                + config.joint_prior_weight * joint_prior
                + config.camera_rotation_prior_weight * rotations.square().mean()
                + config.camera_translation_prior_weight
                * translations.square().mean()
            )
        return float(loss)

    def _observability(
        self,
        frames: Sequence[FactorFrame],
        joint_value: torch.Tensor,
        camera_values: Mapping[str, torch.Tensor],
    ) -> tuple[BlockObservability, BlockObservability]:
        config = self.config
        joint_rows = []
        camera_rows = []
        for frame in frames:
            joint = joint_value.detach().clone().requires_grad_(True)
            cameras = {
                name: camera_values[name].detach().clone().requires_grad_(True)
                for name in config.camera_names
            }
            inputs = (joint, *(cameras[name] for name in config.camera_names))
            loss, _ = self.backend.frame_loss(frame, joint, cameras)
            gradients = torch.autograd.grad(loss, inputs, allow_unused=True)
            joint_rows.append(self._gradient_or_zero(gradients[0], joint))
            camera_rows.append(
                torch.cat(
                    [
                        self._gradient_or_zero(gradient, cameras[name])
                        for gradient, name in zip(gradients[1:], config.camera_names)
                    ]
                )
            )
        arguments = dict(
            rms_min=config.rms_gradient_min,
            relative_parameter_min=config.relative_parameter_min,
            singular_ratio_min=config.singular_ratio_min,
        )
        return (
            analyze_block_gradients(torch.stack(joint_rows), **arguments),
            analyze_block_gradients(torch.stack(camera_rows), **arguments),
        )

    @staticmethod
    def _gradient_or_zero(
        gradient: torch.Tensor | None, reference: torch.Tensor
    ) -> torch.Tensor:
        if gradient is None:
            return torch.zeros_like(reference)
        return gradient.detach()

    def _run_stage(
        self,
        stage: str,
        steps: int,
        frames: Sequence[FactorFrame],
        joint_raw: torch.nn.Parameter,
        camera_raw: Mapping[str, torch.nn.Parameter],
        parameter_mask: torch.Tensor,
        global_step: int,
        trace: list[dict[str, object]],
    ) -> tuple[int, int]:
        if steps == 0:
            return global_step, 0
        joint_stage = stage != "camera_warmup"
        parameters = [joint_raw] if joint_stage else list(camera_raw.values())
        learning_rate = (
            self.config.joint_learning_rate
            if joint_stage
            else self.config.camera_learning_rate
        )
        optimizer = torch.optim.Adam(parameters, lr=learning_rate)
        stage_start = self._snapshot(joint_raw, camera_raw)
        best_batch_loss = float("inf")
        best_batch_state = stage_start
        nonfinite_skips = 0

        for stage_step in range(steps):
            batch = self._cyclic_batch(frames, global_step)
            optimizer.zero_grad(set_to_none=True)
            loss = self._objective(batch, joint_raw, camera_raw)
            row: dict[str, object] = {
                "global_step": global_step,
                "stage_step": stage_step,
                "stage": stage,
                "batch_state_ids": [frame.state_id for frame in batch],
                "loss": float(loss.detach()) if torch.isfinite(loss) else float("nan"),
                "joint_gradient_norm": 0.0,
                "camera_gradient_norm": 0.0,
                "skipped_nonfinite": False,
            }
            if not torch.isfinite(loss):
                row["skipped_nonfinite"] = True
                nonfinite_skips += 1
                trace.append(row)
                global_step += 1
                continue

            current_state = self._snapshot(joint_raw, camera_raw)
            current_loss = float(loss.detach())
            if current_loss < best_batch_loss:
                best_batch_loss = current_loss
                best_batch_state = current_state
            loss.backward()
            self._apply_gradient_mask(
                joint_stage, joint_raw, camera_raw, parameter_mask
            )
            joint_norm, camera_norm = self._gradient_norms(joint_raw, camera_raw)
            row["joint_gradient_norm"] = joint_norm
            row["camera_gradient_norm"] = camera_norm
            gradients_finite = all(
                parameter.grad is None or torch.isfinite(parameter.grad).all()
                for parameter in parameters
            )
            if not gradients_finite:
                self._restore(current_state, joint_raw, camera_raw)
                row["skipped_nonfinite"] = True
                nonfinite_skips += 1
            else:
                optimizer.step()
                parameters_finite = all(
                    torch.isfinite(parameter).all() for parameter in parameters
                )
                if not parameters_finite:
                    self._restore(current_state, joint_raw, camera_raw)
                    row["skipped_nonfinite"] = True
                    nonfinite_skips += 1
            trace.append(row)
            global_step += 1

        endpoint = self._snapshot(joint_raw, camera_raw)
        self._select_stage_state(
            (stage_start, best_batch_state, endpoint), frames, joint_raw, camera_raw
        )
        return global_step, nonfinite_skips

    def _cyclic_batch(
        self, frames: Sequence[FactorFrame], global_step: int
    ) -> list[FactorFrame]:
        start = (global_step * self.config.batch_size) % len(frames)
        return [
            frames[(start + offset) % len(frames)]
            for offset in range(min(self.config.batch_size, len(frames)))
        ]

    @staticmethod
    def _snapshot(
        joint_raw: torch.Tensor, camera_raw: Mapping[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return (
            joint_raw.detach().clone(),
            {name: value.detach().clone() for name, value in camera_raw.items()},
        )

    @staticmethod
    def _restore(
        state: tuple[torch.Tensor, Mapping[str, torch.Tensor]],
        joint_raw: torch.Tensor,
        camera_raw: Mapping[str, torch.Tensor],
    ) -> None:
        with torch.no_grad():
            joint_raw.copy_(state[0])
            for name, value in state[1].items():
                camera_raw[name].copy_(value)

    def _select_stage_state(
        self,
        states: Sequence[tuple[torch.Tensor, Mapping[str, torch.Tensor]]],
        frames: Sequence[FactorFrame],
        joint_raw: torch.Tensor,
        camera_raw: Mapping[str, torch.Tensor],
    ) -> None:
        best_loss = float("inf")
        best_state = states[0]
        for state in states:
            self._restore(state, joint_raw, camera_raw)
            loss = self._dataset_loss(frames, joint_raw, camera_raw)
            if torch.isfinite(torch.tensor(loss)) and loss < best_loss:
                best_loss = loss
                best_state = state
        self._restore(best_state, joint_raw, camera_raw)

    def _apply_gradient_mask(
        self,
        joint_stage: bool,
        joint_raw: torch.Tensor,
        camera_raw: Mapping[str, torch.Tensor],
        parameter_mask: torch.Tensor,
    ) -> None:
        if joint_stage:
            if joint_raw.grad is not None:
                joint_raw.grad.mul_(parameter_mask.to(joint_raw.grad))
            return
        expected = 6 * len(self.config.camera_names)
        if parameter_mask.numel() != expected:
            raise ValueError("Camera observability mask has the wrong size")
        for index, name in enumerate(self.config.camera_names):
            gradient = camera_raw[name].grad
            if gradient is not None:
                gradient.mul_(parameter_mask[index * 6 : (index + 1) * 6].to(gradient))

    @staticmethod
    def _gradient_norms(
        joint_raw: torch.Tensor, camera_raw: Mapping[str, torch.Tensor]
    ) -> tuple[float, float]:
        joint_norm = (
            0.0
            if joint_raw.grad is None
            else float(torch.linalg.vector_norm(joint_raw.grad.detach()))
        )
        camera_gradients = [
            value.grad.detach().reshape(-1)
            for value in camera_raw.values()
            if value.grad is not None
        ]
        camera_norm = (
            0.0
            if not camera_gradients
            else float(torch.linalg.vector_norm(torch.cat(camera_gradients)))
        )
        return joint_norm, camera_norm


__all__ = ["PhysicalFactorOptimizer"]
