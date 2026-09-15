import unittest

import torch

from kinesync.correction.temporal_offset import TemporalJointOffsetRecovery


class QuadraticVisualBackend:
    def loss(self, qpos: torch.Tensor, target: torch.Tensor):
        value = (qpos - target.to(qpos)).square().mean()
        return value, {"quadratic": value}


class TemporalJointOffsetRecoveryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.backend = QuadraticVisualBackend()
        self.state_ids = ["state_a", "state_b"]
        self.reference = {
            "state_a": torch.tensor([0.0, 0.1]),
            "state_b": torch.tensor([0.3, -0.2]),
        }
        bias = torch.tensor([0.05, 0.0])
        self.measured = {
            name: value + bias for name, value in self.reference.items()
        }
        self.targets = dict(self.reference)

    def test_recovers_one_bias_shared_by_all_states(self) -> None:
        result = TemporalJointOffsetRecovery(
            self.backend,
            joint_names=["joint1", "joint2"],
            selected_joints=["joint1"],
            steps=120,
            learning_rate=0.08,
            offset_bound=0.1,
            seed=7,
            prior_weight=0.0,
        ).recover(
            self.state_ids,
            self.measured,
            self.targets,
            center_state_id="state_b",
            reference_qpos=self.reference,
        )

        self.assertAlmostEqual(float(result.applied_correction[0]), -0.05, delta=2e-3)
        self.assertAlmostEqual(float(result.applied_correction[1]), 0.0, delta=1e-7)
        self.assertTrue(
            torch.allclose(
                result.corrected_qpos_by_state["state_a"],
                self.reference["state_a"],
                atol=2e-3,
            )
        )
        self.assertTrue(
            torch.allclose(result.corrected_qpos, self.reference["state_b"], atol=2e-3)
        )
        self.assertLess(result.final_loss, result.initial_loss)
        self.assertLess(result.final_state_mae, result.initial_state_mae)
        self.assertEqual(len(result.trace), 121)
        self.assertTrue(
            all(
                torch.isfinite(torch.tensor(float(row["loss"])))
                for row in result.trace
            )
        )

    def test_correction_is_bounded_and_keys_must_match(self) -> None:
        recovery = TemporalJointOffsetRecovery(
            self.backend,
            joint_names=["joint1", "joint2"],
            selected_joints=["joint1"],
            steps=20,
            learning_rate=0.1,
            offset_bound=0.01,
            seed=7,
            prior_weight=0.0,
        )
        result = recovery.recover(
            self.state_ids,
            self.measured,
            self.targets,
            center_state_id="state_a",
            reference_qpos=self.reference,
        )
        self.assertLessEqual(float(result.applied_correction.abs().max()), 0.01)
        with self.assertRaisesRegex(ValueError, "same state IDs"):
            recovery.recover(
                self.state_ids,
                self.measured,
                {"state_a": self.targets["state_a"]},
                center_state_id="state_a",
                reference_qpos=self.reference,
            )

    def test_adaptive_prior_requires_qpos_shaped_mask(self) -> None:
        with self.assertRaisesRegex(ValueError, "cauchy mask"):
            TemporalJointOffsetRecovery(
                self.backend,
                joint_names=["joint1", "joint2"],
                selected_joints=["joint1"],
                steps=2,
                learning_rate=0.1,
                offset_bound=0.1,
                seed=7,
                prior_weight=1.0,
                prior_kind="adaptive",
                prior_cauchy_mask=torch.ones(1, dtype=torch.bool),
            )


if __name__ == "__main__":
    unittest.main()
