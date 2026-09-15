import unittest

import torch

from kinesync.factorization import FrozenCameraObservationBackend
from kinesync.observation.base import RenderedObservation
from kinesync.observation.real_route_a import RealObservationTarget


class FakeDynamicRenderer:
    def __init__(self) -> None:
        self.cameras = {"head": object()}

    def render_alpha_with_camera_deltas(self, qpos, camera_twists):
        self.assert_twists(camera_twists)
        y = torch.linspace(0.0, 1.0, 24, dtype=qpos.dtype, device=qpos.device)
        x = torch.linspace(0.0, 1.0, 32, dtype=qpos.dtype, device=qpos.device)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        radius = torch.sqrt(
            (xx - (0.5 + 0.2 * qpos[0])).square()
            + (yy - 0.5).square()
            + 1e-8
        )
        alpha = torch.sigmoid((0.22 - radius) * 35.0)
        return {
            "head": RenderedObservation(
                values=alpha, valid=torch.ones_like(alpha, dtype=torch.bool)
            )
        }

    @staticmethod
    def assert_twists(camera_twists):
        if set(camera_twists) != {"head"}:
            raise ValueError("unexpected camera twists")


class FrozenBackendTsdfTest(unittest.TestCase):
    def setUp(self) -> None:
        self.renderer = FakeDynamicRenderer()
        qpos = torch.zeros(2, dtype=torch.float64)
        alpha = self.renderer.render_alpha_with_camera_deltas(
            qpos, {"head": torch.zeros(6, dtype=torch.float64)}
        )["head"].values
        mask = (alpha.detach() >= 0.5).to(torch.float64)
        signed = torch.where(
            mask > 0,
            torch.full_like(mask, -4.0),
            torch.full_like(mask, 4.0),
        )
        self.target = {
            "head": RealObservationTarget(
                rgb=torch.zeros((24, 32, 3), dtype=torch.float64),
                mask=mask,
                signed_distance=signed,
            )
        }
        self.twists = {"head": torch.zeros(6, dtype=torch.float64)}

    def test_optional_tsdf_term_composes_and_preserves_gradient(self) -> None:
        base = FrozenCameraObservationBackend(
            self.renderer, camera_twists=self.twists, boundary_weight=0.05
        )
        enhanced = FrozenCameraObservationBackend(
            self.renderer,
            camera_twists=self.twists,
            boundary_weight=0.05,
            sdf_weight=0.4,
            sdf_radii=(2.0, 6.0),
        )
        qpos = torch.tensor([0.35, 0.0], dtype=torch.float64, requires_grad=True)
        base_loss, base_terms = base.loss(qpos, self.target)
        enhanced_loss, enhanced_terms = enhanced.loss(qpos, self.target)
        self.assertNotIn("tsdf_head", base_terms)
        self.assertIn("tsdf_head", enhanced_terms)
        torch.testing.assert_close(
            enhanced_loss,
            base_loss + 0.4 * enhanced_terms["tsdf_head"],
        )
        enhanced_loss.backward()
        self.assertTrue(torch.isfinite(qpos.grad).all())
        self.assertGreater(float(qpos.grad.abs().sum()), 1e-7)

    def test_positive_weight_requires_target_distance(self) -> None:
        backend = FrozenCameraObservationBackend(
            self.renderer,
            camera_twists=self.twists,
            boundary_weight=0.05,
            sdf_weight=0.2,
        )
        target = {
            "head": RealObservationTarget(
                rgb=self.target["head"].rgb,
                mask=self.target["head"].mask,
            )
        }
        with self.assertRaises(ValueError):
            backend.loss(torch.zeros(2, dtype=torch.float64), target)


if __name__ == "__main__":
    unittest.main()
