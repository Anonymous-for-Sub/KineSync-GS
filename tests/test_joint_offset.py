import unittest
from pathlib import Path

import torch

from kinesync.correction.joint_offset import JointOffsetRecovery
from kinesync.geometry.urdf import TorchURDFKinematics
from kinesync.observation.camera import TorchCamera
from kinesync.observation.projected_centers import ProjectedCentersBackend


PROJECT = Path(__file__).resolve().parents[1]
FIXTURE = PROJECT / "tests" / "fixtures" / "two_link.urdf"


class JointOffsetRecoveryTest(unittest.TestCase):
    def setUp(self) -> None:
        fk = TorchURDFKinematics(FIXTURE)
        xyz = torch.tensor(
            [
                [0.00, 0.00, 0.00],
                [0.15, 0.03, 0.02],
                [0.00, 0.00, 0.00],
                [0.14, -0.05, 0.04],
                [0.00, 0.00, 0.00],
                [0.12, 0.08, 0.03],
            ],
            dtype=torch.float64,
        )
        links = torch.tensor([1, 1, 2, 2, 3, 3])
        names = {0: "base", 1: "shoulder", 2: "slider", 3: "wrist"}
        intrinsic = torch.tensor(
            [[80.0, 0.0, 32.0], [0.0, 80.0, 24.0], [0.0, 0.0, 1.0]],
            dtype=torch.float64,
        )
        front = torch.eye(4, dtype=torch.float64)
        front[2, 3] = 5.0
        side = front.clone()
        side[0, 3] = -0.5
        cameras = {
            "front": TorchCamera("front", intrinsic, front, 64, 48),
            "side": TorchCamera("side", intrinsic, side, 64, 48),
        }
        self.backend = ProjectedCentersBackend(fk, xyz, links, names, cameras)
        self.true_qpos = torch.tensor([0.25, 0.35, -0.15], dtype=torch.float64)
        self.target = self.backend.render(self.true_qpos)

    def test_recovers_held_out_measurement_offsets(self) -> None:
        injected = torch.tensor([0.12, -0.08, 0.0], dtype=torch.float64)
        measured = self.true_qpos + injected
        recovery = JointOffsetRecovery(
            self.backend,
            joint_names=["shoulder_joint", "slider_joint", "wrist_joint"],
            selected_joints=["shoulder_joint", "slider_joint"],
            steps=220,
            learning_rate=0.07,
            offset_bound=0.25,
            seed=23,
        )
        result = recovery.recover(
            measured, self.target, reference_qpos=self.true_qpos
        )

        self.assertLess(result.final_loss, result.initial_loss * 0.01)
        torch.testing.assert_close(
            result.estimated_measurement_offset[:2], injected[:2], atol=4e-3, rtol=0
        )
        self.assertEqual(float(result.applied_correction[2]), 0.0)
        self.assertLess(result.final_state_mae, 3e-3)
        self.assertEqual(len(result.trace), 221)
        self.assertIn("loss_side", result.trace[-1])
        self.assertIn("gradient_norm", result.trace[0])

    def test_is_deterministic_and_respects_bounds(self) -> None:
        measured = self.true_qpos + torch.tensor([0.30, 0.0, 0.0])

        def run_once():
            return JointOffsetRecovery(
                self.backend,
                joint_names=["shoulder_joint", "slider_joint", "wrist_joint"],
                selected_joints=["shoulder_joint"],
                steps=25,
                learning_rate=0.1,
                offset_bound=0.05,
                seed=5,
            ).recover(measured, self.target)

        first = run_once()
        second = run_once()
        torch.testing.assert_close(first.applied_correction, second.applied_correction)
        self.assertLessEqual(float(first.applied_correction.abs().max()), 0.05 + 1e-10)
        self.assertEqual(float(first.applied_correction[1]), 0.0)
        self.assertEqual(float(first.applied_correction[2]), 0.0)


if __name__ == "__main__":
    unittest.main()
