import math
import unittest

import torch

from kinesync.factorization.backend import FactorFrame
from kinesync.factorization.optimizer import PhysicalFactorOptimizer
from kinesync.factorization.schema import FactorizationConfig


class SyntheticFactorBackend:
    def __init__(self, *, fail_batch_call: int | None = None):
        self.cameras = {"head": object(), "extra": object()}
        self.arm_joint_count = 2
        self.true_joint = torch.tensor([0.16, -0.11], dtype=torch.float64)
        self.true_camera = {
            "head": torch.tensor([0.13, 0.0, 0.0, -0.08, 0.0, 0.0], dtype=torch.float64),
            "extra": torch.tensor([-0.10, 0.0, 0.0, 0.07, 0.0, 0.0], dtype=torch.float64),
        }
        self.fail_batch_call = fail_batch_call
        self.batch_calls = 0

    def _prediction(
        self,
        x: torch.Tensor,
        camera_index: int,
        joint_zero: torch.Tensor,
        camera_twist: torch.Tensor,
    ) -> torch.Tensor:
        camera_feature = x.square() + 0.15 * camera_index
        translation_feature = x.pow(3) - 0.20 * camera_index
        return (
            joint_zero[0]
            + x * joint_zero[1]
            + camera_feature * camera_twist[0]
            + translation_feature * camera_twist[3]
        )

    def frame_loss(self, frame, joint_zero, camera_twists):
        x = frame.qpos[0].to(joint_zero)
        losses = []
        terms = {}
        for camera_index, camera in enumerate(self.cameras):
            prediction = self._prediction(
                x, camera_index, joint_zero, camera_twists[camera]
            )
            target = self._prediction(
                x,
                camera_index,
                self.true_joint.to(joint_zero),
                self.true_camera[camera].to(joint_zero),
            )
            loss = (prediction - target).square()
            losses.append(loss)
            terms[camera] = loss
        return torch.stack(losses).mean(), terms

    def batch_loss(self, frames, joint_zero, camera_twists):
        self.batch_calls += 1
        losses = [self.frame_loss(frame, joint_zero, camera_twists)[0] for frame in frames]
        loss = torch.stack(losses).mean()
        if self.batch_calls == self.fail_batch_call:
            loss = loss * torch.tensor(float("nan"), dtype=loss.dtype)
        return loss, {}


def make_frames(count: int = 12) -> list[FactorFrame]:
    return [
        FactorFrame(
            state_id=f"state_{index:02d}",
            split="train",
            qpos=torch.tensor([-1.0 + 2.0 * index / (count - 1)], dtype=torch.float64),
            targets={},
        )
        for index in range(count)
    ]


def make_config(**overrides) -> FactorizationConfig:
    values = dict(
        joint_count=2,
        camera_names=("head", "extra"),
        joint_bound=0.30,
        camera_rotation_bound=0.25,
        camera_translation_bound=0.20,
        joint_steps=80,
        camera_steps=100,
        refinement_steps=50,
        joint_learning_rate=0.08,
        camera_learning_rate=0.08,
        batch_size=4,
        seed=13,
        rms_gradient_min=1e-8,
        relative_parameter_min=0.02,
        singular_ratio_min=1e-4,
    )
    values.update(overrides)
    return FactorizationConfig(**values)


class FactorOptimizerTest(unittest.TestCase):
    def test_recovers_bounded_factors_with_deterministic_stage_order(self) -> None:
        frames = make_frames()
        first = PhysicalFactorOptimizer(
            SyntheticFactorBackend(), make_config()
        ).fit(frames)
        second = PhysicalFactorOptimizer(
            SyntheticFactorBackend(), make_config()
        ).fit(frames)

        stages = []
        for row in first.trace:
            if not stages or stages[-1] != row["stage"]:
                stages.append(row["stage"])
        self.assertEqual(
            stages, ["joint_warmup", "camera_warmup", "joint_refinement"]
        )
        self.assertEqual(len(first.trace), 230)
        self.assertEqual(
            first.trace[0]["batch_state_ids"],
            ["state_00", "state_01", "state_02", "state_03"],
        )
        self.assertIn("joint_gradient_norm", first.trace[0])
        self.assertIn("camera_gradient_norm", first.trace[0])
        self.assertLess(first.final_loss, first.initial_loss * 0.08)
        torch.testing.assert_close(
            first.joint_zero,
            torch.tensor([0.16, -0.11], dtype=torch.float64),
            atol=0.04,
            rtol=0,
        )
        for camera in ("head", "extra"):
            torch.testing.assert_close(
                first.camera_twists[camera][[0, 3]],
                SyntheticFactorBackend().true_camera[camera][[0, 3]],
                atol=0.055,
                rtol=0,
            )
        self.assertLessEqual(float(first.joint_zero.abs().max()), 0.30)
        self.assertLessEqual(
            max(float(value[:3].abs().max()) for value in first.camera_twists.values()),
            0.25,
        )
        self.assertLessEqual(
            max(float(value[3:].abs().max()) for value in first.camera_twists.values()),
            0.20,
        )
        torch.testing.assert_close(first.joint_zero, second.joint_zero)
        for camera in first.camera_twists:
            torch.testing.assert_close(
                first.camera_twists[camera], second.camera_twists[camera]
            )

        joint_only = PhysicalFactorOptimizer(
            SyntheticFactorBackend(),
            make_config(camera_steps=0, refinement_steps=50),
        ).fit(frames)
        self.assertLess(first.final_loss, joint_only.final_loss * 0.35)
        self.assertTrue(first.observability["initial_joint"].accepted)
        self.assertTrue(first.observability["initial_camera"].accepted)

    def test_skips_nonfinite_update_and_returns_finite_best_state(self) -> None:
        result = PhysicalFactorOptimizer(
            SyntheticFactorBackend(fail_batch_call=3),
            make_config(joint_steps=6, camera_steps=6, refinement_steps=4),
        ).fit(make_frames(8))

        self.assertEqual(result.nonfinite_skips, 1)
        self.assertTrue(any(row["skipped_nonfinite"] for row in result.trace))
        self.assertTrue(torch.isfinite(result.joint_zero).all())
        self.assertTrue(
            all(torch.isfinite(value).all() for value in result.camera_twists.values())
        )
        self.assertTrue(math.isfinite(result.final_loss))


if __name__ == "__main__":
    unittest.main()
