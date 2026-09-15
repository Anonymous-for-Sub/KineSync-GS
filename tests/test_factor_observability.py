import unittest

import torch

from kinesync.factorization.observability import analyze_block_gradients


class FactorObservabilityTest(unittest.TestCase):
    def test_full_rank_block_is_accepted_with_all_parameters(self) -> None:
        gradients = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]],
            dtype=torch.float64,
        )
        report = analyze_block_gradients(
            gradients,
            rms_min=1e-4,
            relative_parameter_min=0.1,
            singular_ratio_min=1e-3,
        )
        self.assertTrue(report.accepted)
        self.assertEqual(report.retained_rank, 2)
        self.assertEqual(report.parameter_mask.tolist(), [True, True])
        self.assertEqual(report.frame_count, 4)
        self.assertGreater(report.rms_gradient_norm, 0.9)

    def test_weak_block_is_frozen(self) -> None:
        gradients = torch.tensor(
            [[1e-7, 0.0], [-1e-7, 0.0], [0.0, 1e-7]], dtype=torch.float64
        )
        report = analyze_block_gradients(
            gradients,
            rms_min=1e-4,
            relative_parameter_min=0.1,
            singular_ratio_min=1e-3,
        )
        self.assertFalse(report.accepted)
        self.assertFalse(report.parameter_mask.any())

    def test_rank_deficiency_and_relative_parameter_mask_are_reported(self) -> None:
        gradients = torch.tensor(
            [[1.0, 2.0, 0.01], [-1.0, -2.0, -0.01], [0.5, 1.0, 0.005]],
            dtype=torch.float64,
        )
        report = analyze_block_gradients(
            gradients,
            rms_min=1e-4,
            relative_parameter_min=0.1,
            singular_ratio_min=1e-3,
        )
        self.assertTrue(report.accepted)
        self.assertEqual(report.retained_rank, 1)
        self.assertEqual(report.parameter_mask.tolist(), [True, True, False])
        self.assertEqual(len(report.singular_values), 3)

    def test_single_frame_and_nonfinite_inputs_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least two frames"):
            analyze_block_gradients(
                torch.ones((1, 3)),
                rms_min=1e-4,
                relative_parameter_min=0.1,
                singular_ratio_min=1e-3,
            )
        with self.assertRaisesRegex(ValueError, "finite"):
            analyze_block_gradients(
                torch.tensor([[1.0, float("nan")], [0.0, 1.0]]),
                rms_min=1e-4,
                relative_parameter_min=0.1,
                singular_ratio_min=1e-3,
            )


if __name__ == "__main__":
    unittest.main()
