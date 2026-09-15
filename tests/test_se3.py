import math
import unittest

import torch

from kinesync.geometry.se3 import se3_exp, so3_exp


class SE3GeometryTest(unittest.TestCase):
    def test_zero_twist_is_identity(self) -> None:
        transform = se3_exp(torch.zeros(6, dtype=torch.float64))
        torch.testing.assert_close(transform, torch.eye(4, dtype=torch.float64))

    def test_pure_translation_and_rotation_have_known_values(self) -> None:
        translation = se3_exp(
            torch.tensor([0.0, 0.0, 0.0, 0.2, -0.3, 0.4], dtype=torch.float64)
        )
        torch.testing.assert_close(
            translation[:3, 3], torch.tensor([0.2, -0.3, 0.4], dtype=torch.float64)
        )
        rotation = so3_exp(
            torch.tensor([0.0, 0.0, math.pi / 2.0], dtype=torch.float64)
        )
        expected = torch.tensor(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=torch.float64,
        )
        torch.testing.assert_close(rotation, expected, atol=1e-8, rtol=0)

    def test_general_twist_inverse_and_batch_shape(self) -> None:
        twists = torch.tensor(
            [
                [0.2, -0.1, 0.05, 0.03, -0.02, 0.01],
                [-0.05, 0.12, 0.18, -0.04, 0.01, 0.02],
            ],
            dtype=torch.float64,
        )
        transforms = se3_exp(twists)
        inverses = se3_exp(-twists)
        self.assertEqual(transforms.shape, (2, 4, 4))
        torch.testing.assert_close(
            transforms @ inverses,
            torch.eye(4, dtype=torch.float64).expand(2, 4, 4),
            atol=1e-9,
            rtol=1e-9,
        )

    def test_small_angle_has_finite_nonzero_gradients(self) -> None:
        twist = torch.tensor(
            [1e-9, -2e-9, 3e-9, 0.1, -0.2, 0.3],
            dtype=torch.float64,
            requires_grad=True,
        )
        point = torch.tensor([0.4, -0.1, 0.2, 1.0], dtype=torch.float64)
        value = (se3_exp(twist) @ point)[:3].square().sum()
        value.backward()
        self.assertTrue(torch.isfinite(twist.grad).all())
        self.assertGreater(float(twist.grad.abs().sum()), 1e-6)

    def test_exact_zero_rotation_has_finite_gradients(self) -> None:
        twist = torch.zeros(6, dtype=torch.float32, requires_grad=True)
        point = torch.tensor([0.4, -0.1, 0.2, 1.0], dtype=torch.float32)
        value = (se3_exp(twist) @ point)[:3].square().sum()
        value.backward()
        self.assertTrue(torch.isfinite(twist.grad).all())
        self.assertGreater(float(twist.grad.abs().sum()), 1e-6)

    def test_rejects_bad_last_dimension(self) -> None:
        with self.assertRaisesRegex(ValueError, "six values"):
            se3_exp(torch.zeros(5))
        with self.assertRaisesRegex(ValueError, "three values"):
            so3_exp(torch.zeros(4))


if __name__ == "__main__":
    unittest.main()
