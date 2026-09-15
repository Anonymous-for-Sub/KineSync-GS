import unittest

import torch

from kinesync.observation.real_losses import (
    binary_iou,
    boundary_f1,
    boundary_loss,
    masked_rgb_loss,
    soft_iou_loss,
    truncated_sdf_loss,
)


class RealObservationLossTest(unittest.TestCase):
    def setUp(self) -> None:
        self.target = torch.zeros((24, 32), dtype=torch.float64)
        self.target[6:19, 9:23] = 1.0

    def test_identical_masks_have_zero_loss_and_perfect_metrics(self) -> None:
        self.assertLess(float(soft_iou_loss(self.target, self.target)), 1e-9)
        self.assertLess(float(boundary_loss(self.target, self.target)), 1e-9)
        self.assertAlmostEqual(float(binary_iou(self.target, self.target)), 1.0)
        self.assertAlmostEqual(float(boundary_f1(self.target, self.target)), 1.0)

    def test_shifted_mask_is_worse_and_provides_gradient(self) -> None:
        logits = torch.full((24, 32), -5.0, dtype=torch.float64)
        logits[6:19, 12:26] = 5.0
        logits.requires_grad_()
        prediction = torch.sigmoid(logits)
        loss = soft_iou_loss(prediction, self.target) + boundary_loss(
            prediction, self.target
        )
        self.assertGreater(float(loss), 0.1)
        self.assertLess(float(binary_iou(prediction, self.target)), 0.8)
        self.assertLess(float(boundary_f1(prediction, self.target)), 0.8)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(float(logits.grad.abs().sum()), 1e-6)

    def test_truncated_sdf_orders_alignment_and_provides_gradient(self) -> None:
        signed_distance = torch.where(
            self.target > 0,
            torch.full_like(self.target, -3.0),
            torch.full_like(self.target, 3.0),
        )
        exact = truncated_sdf_loss(
            self.target, self.target, signed_distance, radii=(2.0, 5.0)
        )
        shifted = torch.roll(self.target, shifts=4, dims=1)
        shifted_loss = truncated_sdf_loss(
            shifted, self.target, signed_distance, radii=(2.0, 5.0)
        )
        logits = torch.zeros_like(self.target, requires_grad=True)
        soft_loss = truncated_sdf_loss(
            torch.sigmoid(logits),
            self.target,
            signed_distance,
            radii=(2.0, 5.0),
        )
        self.assertAlmostEqual(float(exact), 0.0, places=10)
        self.assertGreater(float(shifted_loss), float(exact))
        soft_loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(float(logits.grad.abs().sum()), 1e-6)
        with self.assertRaises(ValueError):
            truncated_sdf_loss(self.target, self.target, signed_distance, radii=())
        with self.assertRaises(ValueError):
            truncated_sdf_loss(self.target, self.target, signed_distance, radii=(0.0,))

    def test_masked_rgb_ignores_background(self) -> None:
        target_rgb = torch.zeros((12, 16, 3), dtype=torch.float64)
        target_rgb[3:9, 4:12] = torch.tensor([0.3, 0.5, 0.7])
        prediction = target_rgb.clone()
        prediction[:3] = 1.0
        mask = torch.zeros((12, 16), dtype=torch.float64)
        mask[3:9, 4:12] = 1.0
        self.assertLess(
            float(masked_rgb_loss(prediction, target_rgb, mask, fit_affine=False)),
            1e-6,
        )

    def test_affine_rgb_fit_handles_exposure_shift(self) -> None:
        generator = torch.Generator().manual_seed(17)
        target_rgb = torch.rand((18, 20, 3), generator=generator, dtype=torch.float64)
        prediction = (target_rgb - 0.12) / 0.62
        mask = torch.zeros((18, 20), dtype=torch.float64)
        mask[2:16, 3:18] = 1.0
        raw = masked_rgb_loss(prediction, target_rgb, mask, fit_affine=False)
        aligned = masked_rgb_loss(prediction, target_rgb, mask, fit_affine=True)
        self.assertLess(float(aligned), float(raw) * 0.01)

    def test_metrics_remain_in_unit_interval_for_empty_prediction(self) -> None:
        empty = torch.zeros_like(self.target)
        for value in (binary_iou(empty, self.target), boundary_f1(empty, self.target)):
            self.assertGreaterEqual(float(value), 0.0)
            self.assertLessEqual(float(value), 1.0)


if __name__ == "__main__":
    unittest.main()
