import unittest

import torch

from kinesync.guard.temporal_candidate import (
    run_temporal_guard_candidate,
    temporal_visual_gradient,
)


class QuadraticMetricsBackend:
    def loss(self, qpos: torch.Tensor, target: torch.Tensor):
        value = (qpos - target.to(qpos)).square().mean()
        return value, {"quadratic": value}

    def metrics(self, qpos: torch.Tensor, target: torch.Tensor):
        error = float((qpos - target.to(qpos)).square().mean())
        score = 1.0 - error
        return {"mean_mask_iou": score, "mean_boundary_f1": score}


class TemporalCandidateTest(unittest.TestCase):
    def setUp(self) -> None:
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

    def test_temporal_gradient_accumulates_shared_selected_evidence(self) -> None:
        gradient = temporal_visual_gradient(
            QuadraticMetricsBackend(),
            self.state_ids,
            self.measured,
            self.targets,
            selected_indices=[0],
        )
        self.assertEqual(tuple(gradient.shape), (2,))
        self.assertAlmostEqual(float(gradient[0]), 0.05, places=5)
        self.assertAlmostEqual(float(gradient[1]), 0.0, places=7)

    def test_candidate_returns_centered_finite_guard_evidence(self) -> None:
        joint_backend = QuadraticMetricsBackend()
        camera_backends = {
            "extra": QuadraticMetricsBackend(),
            "head": QuadraticMetricsBackend(),
        }
        camera_targets = {
            camera: dict(self.targets) for camera in camera_backends
        }
        result = run_temporal_guard_candidate(
            state_ids=self.state_ids,
            center_state_id="state_b",
            joint_visual_backend=joint_backend,
            joint_targets=self.targets,
            camera_visual_backends=camera_backends,
            camera_targets=camera_targets,
            measured_qpos=self.measured,
            reference_qpos=self.reference,
            joint_names=["joint1", "joint2"],
            selected_joints=["joint1"],
            steps=120,
            learning_rate=0.08,
            offset_bound=0.1,
            seed=7,
            prior_weight=0.0,
            prior_kind="adaptive",
            prior_delta_rad=0.02,
            minimum_gradient=0.0,
            minimum_relative=0.0,
            acceptance={"state_error_reduction_min": 0.6},
        )

        self.assertEqual(result.joint_result.center_state_id, "state_b")
        self.assertEqual(set(result.camera_results), {"extra", "head"})
        self.assertTrue(result.evidence.finite)
        self.assertEqual(result.evidence.view_count, 2)
        self.assertGreater(result.evidence.visual_gain_ratio, 0.0)
        self.assertTrue(result.evaluation["recovery_success"])
        self.assertAlmostEqual(
            float(result.joint_result.applied_correction[0]), -0.05, delta=2e-3
        )
        self.assertTrue(result.prior_cauchy_mask[0])
        self.assertFalse(result.prior_cauchy_mask[1])


if __name__ == "__main__":
    unittest.main()
