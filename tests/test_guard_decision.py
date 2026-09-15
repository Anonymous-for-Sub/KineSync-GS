import math
import unittest

from kinesync.guard import (
    GuardThresholds,
    UpdateEvidence,
    decide_update,
)


def evidence(**overrides) -> UpdateEvidence:
    values = dict(
        view_count=2,
        visual_gain_ratio=0.12,
        gradient_cosine=0.70,
        correction_cosine=0.80,
        relative_correction_disagreement=0.15,
        candidate_bound_fraction=0.40,
        finite=True,
    )
    values.update(overrides)
    return UpdateEvidence(**values)


def thresholds() -> GuardThresholds:
    return GuardThresholds(
        min_visual_gain_ratio=0.05,
        min_gradient_cosine=0.20,
        min_correction_cosine=0.30,
        max_relative_correction_disagreement=0.40,
    )


class GuardDecisionTest(unittest.TestCase):
    def test_accepts_paired_evidence_and_reports_minimum_margin(self) -> None:
        decision = decide_update(evidence(), thresholds())
        self.assertTrue(decision.accepted)
        self.assertEqual(decision.reason, "accepted")
        self.assertAlmostEqual(decision.minimum_margin, 0.07)

    def test_rejection_reasons_have_stable_priority(self) -> None:
        cases = (
            (evidence(finite=False, visual_gain_ratio=float("nan")), "nonfinite"),
            (evidence(view_count=1), "insufficient_views"),
            (evidence(visual_gain_ratio=0.01), "visual_gain"),
            (evidence(gradient_cosine=0.10), "gradient_agreement"),
            (evidence(correction_cosine=0.10), "correction_direction"),
            (
                evidence(relative_correction_disagreement=0.50),
                "correction_disagreement",
            ),
        )
        for item, expected in cases:
            with self.subTest(expected):
                decision = decide_update(item, thresholds())
                self.assertFalse(decision.accepted)
                self.assertEqual(decision.reason, expected)
                self.assertLess(decision.minimum_margin, 0.0)

    def test_nonfinite_values_cannot_be_marked_finite(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite=True"):
            evidence(visual_gain_ratio=float("inf"))

    def test_thresholds_validate_cosine_and_nonnegative_ranges(self) -> None:
        with self.assertRaises(ValueError):
            GuardThresholds(0.0, -1.1, 0.0, 1.0)
        with self.assertRaises(ValueError):
            GuardThresholds(-0.1, 0.0, 0.0, 1.0)
        with self.assertRaises(ValueError):
            GuardThresholds(0.0, 0.0, 0.0, -0.1)

    def test_nonfinite_decision_margin_is_negative_infinity(self) -> None:
        decision = decide_update(
            evidence(finite=False, correction_cosine=float("nan")), thresholds()
        )
        self.assertTrue(math.isinf(decision.minimum_margin))
        self.assertLess(decision.minimum_margin, 0)


if __name__ == "__main__":
    unittest.main()
