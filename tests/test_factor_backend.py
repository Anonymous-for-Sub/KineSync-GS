import unittest

import torch

from kinesync.factorization.backend import (
    FactorFrame,
    RouteAFactorizationBackend,
    select_evenly,
)
from kinesync.observation.base import RenderedObservation
from kinesync.observation.real_route_a import RealObservationTarget


class FakeDynamicRenderer:
    def __init__(self, height: int = 20, width: int = 28):
        self.height = height
        self.width = width
        self.cameras = {"head": object(), "extra": object()}

    def render_alpha_with_camera_deltas(self, qpos, camera_twists):
        y = torch.linspace(0.0, 1.0, self.height, dtype=qpos.dtype, device=qpos.device)
        x = torch.linspace(0.0, 1.0, self.width, dtype=qpos.dtype, device=qpos.device)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        rendered = {}
        for camera_index, camera in enumerate(self.cameras):
            twist = camera_twists[camera]
            center_x = 0.45 + 0.18 * qpos[0] + 0.7 * twist[3]
            center_y = 0.48 + 0.14 * qpos[1] + 0.7 * twist[4]
            center_x = center_x + 0.03 * camera_index
            radius = torch.sqrt(
                (xx - center_x).square() + (yy - center_y).square() + 1e-8
            )
            alpha = torch.sigmoid((0.23 - radius) * 32.0)
            rendered[camera] = RenderedObservation(
                values=alpha, valid=torch.ones_like(alpha, dtype=torch.bool)
            )
        return rendered


class FactorBackendTest(unittest.TestCase):
    def setUp(self) -> None:
        self.renderer = FakeDynamicRenderer()
        self.zero_twists = {
            camera: torch.zeros(6, dtype=torch.float64)
            for camera in self.renderer.cameras
        }

    def _frame(self, state_id: str, q0: float, *, split: str = "train") -> FactorFrame:
        qpos = torch.zeros(8, dtype=torch.float64)
        qpos[0] = q0
        qpos[1] = -0.1
        rendered = self.renderer.render_alpha_with_camera_deltas(qpos, self.zero_twists)
        targets = {
            camera: RealObservationTarget(
                rgb=torch.zeros((*item.values.shape, 3), dtype=torch.float64),
                mask=(item.values.detach() >= 0.5).to(torch.float64),
            )
            for camera, item in rendered.items()
        }
        return FactorFrame(state_id=state_id, split=split, qpos=qpos, targets=targets)

    def test_even_selection_is_deterministic_and_keeps_endpoints(self) -> None:
        values = [f"state_{index:04d}" for index in range(10)]
        self.assertEqual(
            select_evenly(values, 4),
            ["state_0000", "state_0003", "state_0006", "state_0009"],
        )
        self.assertEqual(select_evenly(values, 4), select_evenly(values, 4))

    def test_batch_loss_averages_frames_and_reaches_both_blocks(self) -> None:
        frames = [self._frame("state_0000", 0.0), self._frame("state_0001", 0.25)]
        backend = RouteAFactorizationBackend(
            self.renderer, arm_joint_count=6, boundary_weight=0.05
        )
        joint_zero = torch.zeros(6, dtype=torch.float64, requires_grad=True)
        joint_zero.data[0] = 0.06
        twists = {
            "head": torch.tensor(
                [0.0, 0.0, 0.0, 0.02, 0.0, 0.0],
                dtype=torch.float64,
                requires_grad=True,
            ),
            "extra": torch.tensor(
                [0.0, 0.0, 0.0, -0.01, 0.0, 0.0],
                dtype=torch.float64,
                requires_grad=True,
            ),
        }
        first, _ = backend.frame_loss(frames[0], joint_zero, twists)
        second, _ = backend.frame_loss(frames[1], joint_zero, twists)
        total, terms = backend.batch_loss(frames, joint_zero, twists)
        torch.testing.assert_close(total, (first + second) / 2.0)
        self.assertIn("state_0000/head_iou", terms)
        self.assertIn("state_0001/extra_boundary", terms)
        total.backward()
        self.assertTrue(torch.isfinite(joint_zero.grad).all())
        self.assertGreater(float(joint_zero.grad.abs().sum()), 1e-8)
        for twist in twists.values():
            self.assertTrue(torch.isfinite(twist.grad).all())
            self.assertGreater(float(twist.grad.abs().sum()), 1e-8)

    def test_training_loss_rejects_validation_frame_but_metrics_allow_it(self) -> None:
        frame = self._frame("state_0300", 0.1, split="validation")
        backend = RouteAFactorizationBackend(
            self.renderer, arm_joint_count=6, boundary_weight=0.05
        )
        joint_zero = torch.zeros(6, dtype=torch.float64)
        with self.assertRaisesRegex(ValueError, "train frames only"):
            backend.batch_loss([frame], joint_zero, self.zero_twists)
        metrics = backend.metrics([frame], joint_zero, self.zero_twists)
        self.assertGreater(metrics["mean_mask_iou"], 0.9)
        self.assertGreater(metrics["mean_boundary_f1"], 0.9)

    def test_rejects_mismatched_target_cameras(self) -> None:
        frame = self._frame("state_0000", 0.0)
        bad = FactorFrame(
            state_id=frame.state_id,
            split=frame.split,
            qpos=frame.qpos,
            targets={"head": frame.targets["head"]},
        )
        backend = RouteAFactorizationBackend(
            self.renderer, arm_joint_count=6, boundary_weight=0.05
        )
        with self.assertRaisesRegex(ValueError, "Target cameras"):
            backend.frame_loss(bad, torch.zeros(6), self.zero_twists)


if __name__ == "__main__":
    unittest.main()
