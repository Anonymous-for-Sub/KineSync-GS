import re
import unittest

from kinesync.guard.calibration import (
    CalibrationRecord,
    calibrate_event_gain_guard,
    calibrate_guard,
)
from kinesync.guard.schema import GuardThresholds, UpdateEvidence


def evidence(
    gain: float,
    gradient: float,
    correction: float,
    disagreement: float,
) -> UpdateEvidence:
    return UpdateEvidence(
        view_count=2,
        visual_gain_ratio=gain,
        gradient_cosine=gradient,
        correction_cosine=correction,
        relative_correction_disagreement=disagreement,
        candidate_bound_fraction=0.4,
        finite=True,
    )


class GuardCalibrationTest(unittest.TestCase):
    def test_event_gain_search_finds_boundary_missing_from_quantiles(self) -> None:
        event_gains = (0.004, 0.008, 0.013, 0.021)
        records = [
            CalibrationRecord(
                evidence(0.013, 0.0, 0.0, 0.0), "controlled", True, "c0"
            ),
            CalibrationRecord(
                evidence(0.013, 0.0, 0.0, 0.0), "controlled", True, "c1"
            ),
            CalibrationRecord(
                evidence(0.013, 0.0, 0.0, 0.0), "controlled", True, "c2"
            ),
            CalibrationRecord(
                evidence(0.021, 0.0, 0.0, 0.0), "controlled", True, "c3"
            ),
            CalibrationRecord(
                evidence(0.021, 0.0, 0.0, 0.0), "controlled", True, "c4"
            ),
            CalibrationRecord(evidence(0.004, 0.0, 0.0, 0.0), "zero", False, "z0"),
            CalibrationRecord(evidence(0.004, 0.0, 0.0, 0.0), "zero", False, "z1"),
            CalibrationRecord(evidence(0.004, 0.0, 0.0, 0.0), "zero", False, "z2"),
            CalibrationRecord(evidence(0.004, 0.0, 0.0, 0.0), "zero", False, "z3"),
            CalibrationRecord(evidence(0.008, 0.0, 0.0, 0.0), "zero", False, "z4"),
        ]

        calibration = calibrate_event_gain_guard(
            records,
            fixed_thresholds=GuardThresholds(0.0, -1.0, -1.0, 1.0),
        )

        self.assertEqual(calibration.thresholds.min_visual_gain_ratio, 0.013)
        self.assertEqual(calibration.threshold_generation, "event_gain_v1")
        self.assertEqual(
            calibration.evaluated_candidate_count, len(set(event_gains)) + 1
        )

    def test_quantile_calibration_preserves_historical_fingerprint(self) -> None:
        records = [
            CalibrationRecord(
                evidence(0.8, 0.8, 0.8, 0.2), "controlled", True, "c0"
            ),
            CalibrationRecord(
                evidence(0.8, 0.8, 0.8, 0.2), "controlled", True, "c1"
            ),
            CalibrationRecord(evidence(0.1, 0.1, 0.1, 0.9), "zero", False, "z0"),
            CalibrationRecord(evidence(0.1, 0.1, 0.1, 0.9), "zero", False, "z1"),
        ]

        calibration = calibrate_guard(records, quantiles=[0.0, 0.5, 1.0])
        reversed_calibration = calibrate_guard(
            list(reversed(records)), quantiles=[1.0, 0.5, 0.0]
        )

        self.assertEqual(
            calibration.thresholds,
            GuardThresholds(0.1, 0.1, 0.1, 0.55),
        )
        self.assertEqual(calibration.threshold_generation, "quantile_grid_v1")
        self.assertEqual(
            calibration.evaluated_candidate_count,
            len(calibration.candidate_table),
        )
        self.assertEqual(
            calibration.fingerprint,
            "f2572470798d8ccf5bb0de23f7c14821513f1377dd0b8d8f6a2237863f6ce33b",
        )
        self.assertEqual(reversed_calibration, calibration)

    def test_selects_feasible_thresholds_deterministically(self) -> None:
        records = [
            CalibrationRecord(
                evidence=evidence(0.8, 0.8, 0.8, 0.2),
                trial_type="controlled",
                candidate_success=True,
                state_id=f"controlled_{index}",
            )
            for index in range(5)
        ]
        records.extend(
            CalibrationRecord(
                evidence=evidence(0.1, 0.1, 0.1, 0.9),
                trial_type="zero",
                candidate_success=False,
                state_id=f"zero_{index}",
            )
            for index in range(10)
        )

        calibration = calibrate_guard(records, quantiles=[0.0, 1.0])
        reversed_calibration = calibrate_guard(
            list(reversed(records)), quantiles=[1.0, 0.0]
        )

        self.assertEqual(
            calibration.thresholds,
            GuardThresholds(
                min_visual_gain_ratio=0.1,
                min_gradient_cosine=0.1,
                min_correction_cosine=0.1,
                max_relative_correction_disagreement=0.2,
            ),
        )
        self.assertEqual(calibration.controlled_coverage, 1.0)
        self.assertEqual(calibration.zero_stability, 1.0)
        self.assertEqual(calibration.guarded_success, 1.0)
        self.assertEqual(len(calibration.candidate_table), 16)
        self.assertEqual(
            calibration.source_state_ids,
            tuple(sorted(record.state_id for record in records)),
        )
        self.assertEqual(calibration, reversed_calibration)
        self.assertRegex(calibration.fingerprint, re.compile(r"^[0-9a-f]{64}$"))

    def test_raises_when_zero_stability_and_coverage_are_incompatible(self) -> None:
        shared = evidence(0.5, 0.5, 0.5, 0.5)
        records = [
            CalibrationRecord(shared, "controlled", True, f"controlled_{index}")
            for index in range(3)
        ]
        records.extend(
            CalibrationRecord(shared, "zero", False, f"zero_{index}")
            for index in range(3)
        )

        with self.assertRaisesRegex(ValueError, "No feasible guard threshold"):
            calibrate_guard(records, quantiles=[0.0, 0.5, 1.0])

    def test_rejects_invalid_trial_types_and_quantiles(self) -> None:
        shared = evidence(0.5, 0.5, 0.5, 0.5)
        with self.assertRaisesRegex(ValueError, "trial_type"):
            CalibrationRecord(shared, "stress", False, "bad")
        records = [
            CalibrationRecord(shared, "controlled", True, "controlled"),
            CalibrationRecord(shared, "zero", True, "zero"),
        ]
        with self.assertRaisesRegex(ValueError, "quantiles"):
            calibrate_guard(records, quantiles=[])
        with self.assertRaisesRegex(ValueError, r"\[0, 1\]"):
            calibrate_guard(records, quantiles=[-0.1, 0.5])
        with self.assertRaisesRegex(ValueError, "quantiles"):
            calibrate_guard(
                records,
                quantiles=[],
                minimum_zero_stability=1.1,
            )


if __name__ == "__main__":
    unittest.main()
