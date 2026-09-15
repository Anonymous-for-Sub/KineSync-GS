import unittest
from pathlib import Path

import torch

from kinesync.geometry.urdf import TorchURDFKinematics
from kinesync.observation.camera import TorchCamera
from kinesync.observation.projected_centers import ProjectedCentersBackend
from kinesync.observation.soft_occupancy import SoftOccupancyBackend


PROJECT = Path(__file__).resolve().parents[1]
FIXTURE = PROJECT / "tests" / "fixtures" / "two_link.urdf"


class ObservationBackendTest(unittest.TestCase):
    def setUp(self) -> None:
        self.fk = TorchURDFKinematics(FIXTURE)
        self.xyz = torch.tensor(
            [
                [0.00, 0.00, 0.00],
                [0.08, 0.02, 0.02],
                [0.00, 0.00, 0.00],
                [0.10, -0.04, 0.03],
                [0.00, 0.00, 0.00],
                [0.08, 0.05, 0.02],
            ],
            dtype=torch.float64,
        )
        self.link_index = torch.tensor([1, 1, 2, 2, 3, 3])
        self.link_names = {0: "base", 1: "shoulder", 2: "slider", 3: "wrist"}
        intrinsic = torch.tensor(
            [[72.0, 0.0, 32.0], [0.0, 72.0, 24.0], [0.0, 0.0, 1.0]],
            dtype=torch.float64,
        )
        w2c = torch.eye(4, dtype=torch.float64)
        w2c[2, 3] = 5.0
        second_w2c = w2c.clone()
        second_w2c[0, 3] = -0.4
        self.cameras = {
            "front": TorchCamera("front", intrinsic, w2c, width=64, height=48),
            "side": TorchCamera("side", intrinsic, second_w2c, width=64, height=48),
        }
        self.true_qpos = torch.tensor([0.25, 0.35, -0.15], dtype=torch.float64)

    def test_projected_centers_have_state_sensitive_gradients(self) -> None:
        backend = ProjectedCentersBackend(
            self.fk, self.xyz, self.link_index, self.link_names, self.cameras
        )
        target = backend.render(self.true_qpos)
        exact_loss, _ = backend.loss(self.true_qpos, target)
        self.assertLess(float(exact_loss), 1e-12)

        perturbed = (self.true_qpos + torch.tensor([0.10, -0.08, 0.05])).requires_grad_()
        loss, per_camera = backend.loss(perturbed, target)
        self.assertGreater(float(loss), 1e-7)
        self.assertEqual(set(per_camera), {"front", "side"})
        loss.backward()
        self.assertGreater(float(perturbed.grad.abs().sum()), 1e-7)

    def test_soft_occupancy_has_image_space_gradient(self) -> None:
        backend = SoftOccupancyBackend(
            self.fk,
            self.xyz,
            self.link_index,
            self.link_names,
            self.cameras,
            output_size=(24, 32),
            sigma_px=1.4,
        )
        target = backend.render(self.true_qpos)
        self.assertEqual(target["front"].values.shape, (24, 32))
        exact_loss, _ = backend.loss(self.true_qpos, target)
        self.assertLess(float(exact_loss), 1e-12)

        perturbed = (self.true_qpos + torch.tensor([-0.12, 0.09, 0.06])).requires_grad_()
        loss, _ = backend.loss(perturbed, target)
        self.assertGreater(float(loss), 1e-7)
        loss.backward()
        self.assertGreater(float(perturbed.grad.abs().sum()), 1e-7)


if __name__ == "__main__":
    unittest.main()
