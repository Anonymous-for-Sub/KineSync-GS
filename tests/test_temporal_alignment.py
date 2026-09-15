from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from types import SimpleNamespace
import unittest

import numpy as np

from kinesync.temporal import telemetry
from kinesync.temporal import TemporalGuardDecision, decide_temporal_guard


class MotionOffsetSearchTest(unittest.TestCase):
    def test_recovers_known_offset_and_reports_the_complete_score_profile(self):
        grid_ms = 20
        injected_offset_ms = 80
        sample_index = np.arange(64, dtype=np.float64)
        visual_motion = (
            0.15
            + 1.20 * np.exp(-((sample_index - 9.0) / 2.1) ** 2)
            + 0.75 * np.exp(-((sample_index - 27.0) / 4.3) ** 2)
            + 1.55 * np.exp(-((sample_index - 49.0) / 1.7) ** 2)
            + 0.03 * sample_index
        )
        camera_midpoint_ns = (
            np.arange(len(visual_motion), dtype=np.int64) * grid_ms * 1_000_000
        )
        state_timestamp_ns = (
            np.arange(len(visual_motion) + 1, dtype=np.int64) * grid_ms * 1_000_000
            - 10_000_000
            + injected_offset_ms * 1_000_000
        )
        qpos = np.zeros((len(state_timestamp_ns), 6), dtype=np.float64)
        qpos[:, 0] = np.cumsum(np.concatenate(([0.0], visual_motion)))
        features = telemetry.MotionFeatures(
            camera_midpoint_ns=camera_midpoint_ns,
            visual_motion=visual_motion,
            state_motion=visual_motion.copy(),
        )
        corrupted = telemetry.CorruptedState(
            timestamp_ns=state_timestamp_ns,
            qpos=qpos,
            retained_indices=np.arange(len(state_timestamp_ns), dtype=np.int64),
        )

        search = getattr(telemetry, "search_motion_offset", None)
        self.assertIsNotNone(search, "motion offset search must be available")
        estimate = search(features, corrupted)

        self.assertEqual(estimate.offset_ms, injected_offset_ms)
        self.assertEqual(len(estimate.score_profile), 21)
        self.assertEqual(
            tuple(score.offset_ms for score in estimate.score_profile),
            tuple(range(-200, 201, 20)),
        )
        self.assertGreater(estimate.peak_correlation, 0.99)
        self.assertLessEqual(estimate.peak_correlation, 1.0)
        self.assertGreater(estimate.peak_margin, 0.0)

    def test_strict_overlap_drops_leading_and_trailing_endpoint_candidates(self):
        interval_ns = 40_000_000
        visual_motion = np.array(
            (0.0, 1.0, 4.0, 2.0, 7.0, 3.0, 8.0, 5.0, 9.0, 6.0)
        )
        midpoint_ns = np.arange(len(visual_motion), dtype=np.int64) * interval_ns
        state_timestamp_ns = (
            np.arange(len(visual_motion) + 1, dtype=np.int64) * interval_ns
            - interval_ns // 2
        )
        qpos = np.zeros((len(state_timestamp_ns), 6), dtype=np.float64)
        qpos[:, 0] = np.cumsum(np.concatenate(([0.0], visual_motion)))
        features = telemetry.MotionFeatures(
            camera_midpoint_ns=midpoint_ns,
            visual_motion=visual_motion,
            state_motion=visual_motion.copy(),
        )
        corrupted = telemetry.CorruptedState(
            timestamp_ns=state_timestamp_ns,
            qpos=qpos,
            retained_indices=np.arange(len(state_timestamp_ns), dtype=np.int64),
        )
        candidates = (-120.0, -80.0, 80.0, 120.0)

        legacy = telemetry.search_motion_offset(
            features, corrupted, candidate_offsets_ms=candidates
        )
        strict = telemetry.search_motion_offset(
            features,
            corrupted,
            candidate_offsets_ms=candidates,
            strict_overlap=True,
        )
        legacy_scores = {score.offset_ms: score.correlation for score in legacy.score_profile}
        strict_scores = {score.offset_ms: score.correlation for score in strict.score_profile}

        self.assertIsNotNone(legacy_scores[-120.0])
        self.assertIsNotNone(legacy_scores[120.0])
        self.assertIsNone(strict_scores[-120.0])
        self.assertIsNone(strict_scores[120.0])
        self.assertIsNotNone(strict_scores[-80.0])
        self.assertIsNotNone(strict_scores[80.0])

    def test_strict_overlap_fails_closed_when_all_candidates_lack_eight_samples(self):
        interval_ns = 40_000_000
        visual_motion = np.arange(10, dtype=np.float64)
        midpoint_ns = np.arange(len(visual_motion), dtype=np.int64) * interval_ns
        state_timestamp_ns = (
            np.arange(len(visual_motion) + 1, dtype=np.int64) * interval_ns
            - interval_ns // 2
        )
        qpos = np.zeros((len(state_timestamp_ns), 6), dtype=np.float64)
        qpos[:, 0] = np.cumsum(np.concatenate(([0.0], visual_motion)))
        features = telemetry.MotionFeatures(
            camera_midpoint_ns=midpoint_ns,
            visual_motion=visual_motion,
            state_motion=visual_motion.copy(),
        )
        corrupted = telemetry.CorruptedState(
            timestamp_ns=state_timestamp_ns,
            qpos=qpos,
            retained_indices=np.arange(len(state_timestamp_ns), dtype=np.int64),
        )

        estimate = telemetry.search_motion_offset(
            features,
            corrupted,
            candidate_offsets_ms=(-120.0, 120.0),
            strict_overlap=True,
        )

        self.assertTrue(np.isneginf(estimate.peak_correlation))
        self.assertTrue(np.isneginf(estimate.peak_margin))
        self.assertTrue(all(score.correlation is None for score in estimate.score_profile))


def _informative_trial(
    trial_type,
    *,
    trajectory_id: str,
    injected_offset_ms: float,
    native_delay_ms: float = 40.0,
) -> object:
    grid_ms = 20
    sample_index = np.arange(64, dtype=np.float64)
    visual_motion = (
        0.15
        + 1.20 * np.exp(-((sample_index - 9.0) / 2.1) ** 2)
        + 0.75 * np.exp(-((sample_index - 27.0) / 4.3) ** 2)
        + 1.55 * np.exp(-((sample_index - 49.0) / 1.7) ** 2)
        + 0.03 * sample_index
    )
    camera_midpoint_ns = (
        np.arange(len(visual_motion), dtype=np.int64) * grid_ms * 1_000_000
    )
    uncorrupted_timestamp_ns = (
        np.arange(len(visual_motion) + 1, dtype=np.int64) * grid_ms * 1_000_000
        - 10_000_000
        + int(native_delay_ms * 1_000_000)
    )
    qpos = np.zeros((len(uncorrupted_timestamp_ns), 6), dtype=np.float64)
    qpos[:, 0] = np.cumsum(np.concatenate(([0.0], visual_motion)))
    features = telemetry.MotionFeatures(
        camera_midpoint_ns=camera_midpoint_ns,
        visual_motion=visual_motion,
        state_motion=visual_motion.copy(),
    )
    target_qpos = telemetry.interpolate_qpos(
        camera_midpoint_ns, uncorrupted_timestamp_ns, qpos, mode="linear"
    )
    corrupted = telemetry.CorruptedState(
        timestamp_ns=(
            uncorrupted_timestamp_ns + int(injected_offset_ms * 1_000_000)
        ),
        qpos=qpos,
        retained_indices=np.arange(len(uncorrupted_timestamp_ns), dtype=np.int64),
    )
    return trial_type(
        trajectory_id=trajectory_id,
        split="development",
        condition_id=f"{trajectory_id}-{injected_offset_ms:g}",
        features=features,
        corrupted_state=corrupted,
        target_qpos=target_qpos,
        injected_offset_ms=injected_offset_ms,
        jitter_ms=0.0,
        dropout=0.0,
    )


def _varied_informative_trial(
    trial_type,
    *,
    trajectory_id: str,
    injected_offset_ms: float,
    variation_index: int,
    native_delay_ms: float = 40.0,
) -> object:
    grid_ms = 20
    sample_index = np.arange(64, dtype=np.float64)
    base_motion = (
        0.15
        + 1.20 * np.exp(-((sample_index - 9.0) / 2.1) ** 2)
        + 0.75 * np.exp(-((sample_index - 27.0) / 4.3) ** 2)
        + 1.55 * np.exp(-((sample_index - 49.0) / 1.7) ** 2)
        + 0.03 * sample_index
    )
    center = 4.0 + float(variation_index % 53)
    width = 1.2 + 0.15 * float(variation_index % 7)
    visual_motion = base_motion + (0.01 * (variation_index + 1)) * np.exp(
        -((sample_index - center) / width) ** 2
    )
    camera_midpoint_ns = (
        np.arange(len(visual_motion), dtype=np.int64) * grid_ms * 1_000_000
    )
    uncorrupted_timestamp_ns = (
        np.arange(len(visual_motion) + 1, dtype=np.int64) * grid_ms * 1_000_000
        - 10_000_000
        + int(native_delay_ms * 1_000_000)
    )
    qpos = np.zeros((len(uncorrupted_timestamp_ns), 6), dtype=np.float64)
    qpos[:, 0] = np.cumsum(np.concatenate(([0.0], visual_motion)))
    features = telemetry.MotionFeatures(
        camera_midpoint_ns=camera_midpoint_ns,
        visual_motion=visual_motion,
        state_motion=visual_motion.copy(),
    )
    target_qpos = telemetry.interpolate_qpos(
        camera_midpoint_ns, uncorrupted_timestamp_ns, qpos, mode="linear"
    )
    return trial_type(
        trajectory_id=trajectory_id,
        split="development",
        condition_id=f"{trajectory_id}-{injected_offset_ms:g}",
        features=features,
        corrupted_state=telemetry.CorruptedState(
            timestamp_ns=(
                uncorrupted_timestamp_ns + int(injected_offset_ms * 1_000_000)
            ),
            qpos=qpos,
            retained_indices=np.arange(len(uncorrupted_timestamp_ns), dtype=np.int64),
        ),
        target_qpos=target_qpos,
        injected_offset_ms=injected_offset_ms,
        jitter_ms=0.0,
        dropout=0.0,
    )


def _static_zero_trial(trial_type, *, trajectory_id: str) -> object:
    camera_midpoint_ns = np.arange(16, dtype=np.int64) * 20_000_000
    state_timestamp_ns = np.arange(17, dtype=np.int64) * 20_000_000 - 10_000_000
    qpos = np.zeros((17, 6), dtype=np.float64)
    features = telemetry.MotionFeatures(
        camera_midpoint_ns=camera_midpoint_ns,
        visual_motion=np.zeros(16, dtype=np.float64),
        state_motion=np.zeros(16, dtype=np.float64),
    )
    return trial_type(
        trajectory_id=trajectory_id,
        split="development",
        condition_id=f"{trajectory_id}-static-zero",
        features=features,
        corrupted_state=telemetry.CorruptedState(
            timestamp_ns=state_timestamp_ns,
            qpos=qpos,
            retained_indices=np.arange(17, dtype=np.int64),
        ),
        target_qpos=np.zeros((16, 6), dtype=np.float64),
        injected_offset_ms=0.0,
        jitter_ms=0.0,
        dropout=0.0,
    )


class _UnreadableTestTrial:
    split = "test"

    def __getattr__(self, name: str) -> object:
        raise AssertionError(f"calibration must not read test field {name}")


class TemporalGuardCalibrationTest(unittest.TestCase):
    def test_calibrates_one_development_guard_without_reading_test_rows(self):
        trial_type = getattr(telemetry, "TemporalTrial", None)
        constraints_type = getattr(telemetry, "TemporalGuardConstraints", None)
        calibrate = getattr(telemetry, "calibrate_temporal_guard", None)
        self.assertIsNotNone(trial_type, "typed temporal trials must be available")
        self.assertIsNotNone(constraints_type, "typed guard constraints must be available")
        self.assertIsNotNone(calibrate, "temporal guard calibration must be available")

        development_trials = [
            _informative_trial(
                trial_type, trajectory_id="dev-native", injected_offset_ms=0.0
            ),
            *[
                _static_zero_trial(trial_type, trajectory_id=f"dev-static-{index}")
                for index in range(10)
            ],
            *[
                _informative_trial(
                    trial_type,
                    trajectory_id=f"dev-controlled-{offset_ms}",
                    injected_offset_ms=float(offset_ms),
                )
                for offset_ms in (-80, -40, 40, 80, 120)
            ],
        ]
        constraints = constraints_type(
            minimum_zero_stability=0.90,
            minimum_benefit_precision=0.85,
            minimum_controlled_coverage=0.40,
        )

        guard = calibrate([_UnreadableTestTrial(), *development_trials], constraints)
        reordered_guard = calibrate(
            list(reversed(development_trials)), constraints
        )

        self.assertEqual(guard, reordered_guard)
        self.assertEqual(guard.native_delay_ms, 40.0)
        estimates = [
            telemetry.search_motion_offset(trial.features, trial.corrupted_state)
            for trial in development_trials
        ]
        commits = [
            _guard_accepts(estimate, guard)
            for estimate in estimates
        ]
        zero_commits = [
            committed
            for trial, committed in zip(development_trials, commits, strict=True)
            if trial.injected_offset_ms == 0.0
        ]
        controlled = [
            (trial, estimate, committed)
            for trial, estimate, committed in zip(
                development_trials, estimates, commits, strict=True
            )
            if trial.injected_offset_ms != 0.0
        ]
        committed_controlled = [item for item in controlled if item[2]]
        precision = sum(
            abs((estimate.offset_ms - guard.native_delay_ms) - trial.injected_offset_ms)
            <= 20.0
            for trial, estimate, _ in committed_controlled
        ) / len(committed_controlled)

        self.assertGreaterEqual(1.0 - sum(zero_commits) / len(zero_commits), 0.90)
        self.assertGreaterEqual(precision, 0.85)

    def test_report_serializes_benefit_precision_and_separate_exact_recovery_metrics(self):
        constraints = telemetry.TemporalGuardConstraints(
            minimum_zero_stability=0.90,
            minimum_benefit_precision=0.85,
            minimum_controlled_coverage=0.40,
            max_calibration_candidates=640,
        )
        development_trials = [
            _informative_trial(
                telemetry.TemporalTrial,
                trajectory_id="metrics-native",
                injected_offset_ms=0.0,
            ),
            *[
                _static_zero_trial(
                    telemetry.TemporalTrial,
                    trajectory_id=f"metrics-static-{index}",
                )
                for index in range(10)
            ],
            *[
                _informative_trial(
                    telemetry.TemporalTrial,
                    trajectory_id=f"metrics-controlled-{offset_ms}",
                    injected_offset_ms=float(offset_ms),
                )
                for offset_ms in (-80, -40, 40, 80, 120)
            ],
        ]

        report = telemetry.calibrate_temporal_guard_with_report(
            development_trials, constraints
        )
        payload = report.to_dict()

        self.assertIsInstance(report.selected_metrics, telemetry.GuardCalibrationMetrics)
        self.assertEqual(
            payload["selected_metrics"],
            {
                "controlled_trial_count": 5,
                "committed_controlled_count": 5,
                "successful_controlled_count": 5,
                "successful_committed_count": 5,
                "beneficial_committed_count": 5,
                "zero_trial_count": 11,
                "stable_zero_count": 11,
                "controlled_recovery": 1.0,
                "controlled_qmae_rad": 0.0,
                "benefit_precision": 1.0,
                "exact_recovery_precision": 1.0,
                "controlled_coverage": 1.0,
                "zero_stability": 1.0,
            },
        )
        self.assertEqual(
            payload["constraints"],
            {
                "minimum_zero_stability": 0.90,
                "minimum_benefit_precision": 0.85,
                "minimum_controlled_coverage": 0.40,
                "recovery_offset_tolerance_ms": 20.0,
                "recovery_qmae_tolerance_rad": 0.01,
                "candidate_offsets_ms": list(range(-200, 201, 20)),
                "max_calibration_candidates": 640,
            },
        )

    def test_bounds_event_coordinate_search_and_keeps_diagnostics_deterministic(self):
        report_type = getattr(telemetry, "GuardCalibrationReport", None)
        calibrate_with_report = getattr(
            telemetry, "calibrate_temporal_guard_with_report", None
        )
        self.assertIsNotNone(
            report_type, "typed temporal guard calibration reports must be available"
        )
        self.assertIsNotNone(
            calibrate_with_report,
            "bounded temporal guard calibration must be available",
        )

        constraints = telemetry.TemporalGuardConstraints(
            minimum_zero_stability=0.90,
            minimum_benefit_precision=0.85,
            minimum_controlled_coverage=0.40,
            max_calibration_candidates=640,
        )
        development_trials = [
            _informative_trial(
                telemetry.TemporalTrial,
                trajectory_id="bounded-native",
                injected_offset_ms=0.0,
            ),
            *[
                _varied_informative_trial(
                    telemetry.TemporalTrial,
                    trajectory_id=f"bounded-{index:02d}",
                    injected_offset_ms=float((-120, -80, -40, 40, 80, 120)[index % 6]),
                    variation_index=index,
                )
                for index in range(63)
            ],
        ]

        report = calibrate_with_report(development_trials, constraints)
        reordered = calibrate_with_report(
            list(reversed(development_trials)), constraints
        )

        self.assertIsInstance(report, report_type)
        self.assertEqual(report, reordered)
        self.assertEqual(
            telemetry.calibrate_temporal_guard(development_trials, constraints),
            report.guard,
        )
        self.assertEqual(
            report.strategy,
            "deterministic_bounded_coordinate_event_search_v1",
        )
        self.assertGreater(report.evaluated_candidate_count, 0)
        self.assertGreater(report.evaluated_candidate_count, 9)
        self.assertLessEqual(
            report.evaluated_candidate_count,
            constraints.max_calibration_candidates,
        )
        self.assertGreaterEqual(report.iterations, 1)
        self.assertFalse(report.budget_exhausted)
        self.assertTrue(report.converged)
        json.dumps(report.to_dict(), allow_nan=False)

    def test_advances_through_multiple_infeasible_coordinates_and_excludes_harmful_commits(self):
        constraints = telemetry.TemporalGuardConstraints(
            minimum_zero_stability=0.90,
            minimum_benefit_precision=0.85,
            minimum_controlled_coverage=0.40,
            max_calibration_candidates=96,
        )

        def evidence(
            *,
            injected_offset_ms: float,
            correlation: float,
            margin: float,
            visual_mad: float,
            state_mad: float,
            candidate_qmae_rad: float,
        ):
            return telemetry._CalibrationEvidence(
                trial=SimpleNamespace(injected_offset_ms=injected_offset_ms),
                estimate=telemetry.OffsetEstimate(
                    offset_ms=40.0,
                    peak_correlation=correlation,
                    peak_margin=margin,
                    visual_mad=visual_mad,
                    state_mad=state_mad,
                ),
                correction_ms=40.0,
                raw_qmae_rad=1.0,
                candidate_qmae_rad=candidate_qmae_rad,
            )

        report = telemetry._calibrate_guard_by_coordinate_events(
            (
                evidence(injected_offset_ms=0.0, correlation=0.9, margin=0.9, visual_mad=0.1, state_mad=0.9, candidate_qmae_rad=0.0),
                evidence(injected_offset_ms=0.0, correlation=0.9, margin=0.1, visual_mad=0.9, state_mad=0.9, candidate_qmae_rad=0.0),
                evidence(injected_offset_ms=0.0, correlation=0.1, margin=0.9, visual_mad=0.9, state_mad=0.9, candidate_qmae_rad=0.0),
                *(evidence(injected_offset_ms=40.0, correlation=0.9, margin=0.9, visual_mad=0.9, state_mad=0.9, candidate_qmae_rad=0.5) for _ in range(3)),
                *(evidence(injected_offset_ms=40.0, correlation=0.9, margin=0.9, visual_mad=0.9, state_mad=0.1, candidate_qmae_rad=1.1) for _ in range(3)),
            ),
            native_delay_ms=0.0,
            constraints=constraints,
        )

        self.assertEqual(report.guard.min_peak_correlation, 0.9)
        self.assertEqual(report.guard.min_peak_margin, 0.9)
        self.assertEqual(report.guard.min_visual_mad, 0.9)
        self.assertEqual(report.guard.min_state_mad, 0.9)
        self.assertEqual(report.selected_metrics.controlled_coverage, 0.5)
        self.assertEqual(report.selected_metrics.beneficial_committed_count, 3)
        self.assertEqual(report.selected_metrics.benefit_precision, 1.0)
        self.assertEqual(report.selected_metrics.exact_recovery_precision, 0.0)
        self.assertTrue(report.converged)
        self.assertFalse(report.budget_exhausted)

    def test_searches_correction_magnitude_before_local_evidence_thresholds(self):
        constraints = telemetry.TemporalGuardConstraints(
            minimum_zero_stability=0.90,
            minimum_benefit_precision=0.85,
            minimum_controlled_coverage=0.40,
            max_calibration_candidates=96,
        )

        def evidence(*, injected_offset_ms: float, correlation: float, correction_ms: float):
            return telemetry._CalibrationEvidence(
                trial=SimpleNamespace(injected_offset_ms=injected_offset_ms),
                estimate=telemetry.OffsetEstimate(
                    offset_ms=correction_ms,
                    peak_correlation=correlation,
                    peak_margin=0.5,
                    visual_mad=0.5,
                    state_mad=0.5,
                ),
                correction_ms=correction_ms,
                raw_qmae_rad=1.0,
                candidate_qmae_rad=0.5,
            )

        report = telemetry._calibrate_guard_by_coordinate_events(
            (
                *(evidence(injected_offset_ms=0.0, correlation=0.1, correction_ms=20.0) for _ in range(7)),
                *(evidence(injected_offset_ms=0.0, correlation=0.9, correction_ms=20.0) for _ in range(3)),
                *(evidence(injected_offset_ms=40.0, correlation=0.1, correction_ms=100.0) for _ in range(5)),
                *(evidence(injected_offset_ms=40.0, correlation=0.1, correction_ms=20.0) for _ in range(2)),
                *(evidence(injected_offset_ms=40.0, correlation=0.9, correction_ms=20.0) for _ in range(3)),
            ),
            native_delay_ms=0.0,
            constraints=constraints,
        )

        self.assertEqual(report.guard.min_correction_ms, 100.0)
        self.assertEqual(report.selected_metrics.benefit_precision, 1.0)
        self.assertEqual(report.selected_metrics.controlled_coverage, 0.5)
        self.assertEqual(report.selected_metrics.zero_stability, 1.0)


def _guard_accepts(estimate, guard) -> bool:
    correction_ms = estimate.offset_ms - guard.native_delay_ms
    return (
        np.isfinite(estimate.peak_correlation)
        and np.isfinite(estimate.peak_margin)
        and np.isfinite(estimate.visual_mad)
        and np.isfinite(estimate.state_mad)
        and estimate.peak_correlation >= guard.min_peak_correlation
        and estimate.peak_margin >= guard.min_peak_margin
        and estimate.visual_mad >= guard.min_visual_mad
        and estimate.state_mad >= guard.min_state_mad
        and abs(correction_ms) >= guard.min_correction_ms
        and correction_ms != 0.0
    )


class PublicTemporalGuardDecisionTest(unittest.TestCase):
    def test_decision_reports_rt9_thresholds_in_existing_priority_order(self):
        guard = telemetry.TemporalGuard(
            native_delay_ms=40.0,
            min_peak_correlation=0.80,
            min_peak_margin=0.10,
            min_visual_mad=0.20,
            min_state_mad=0.30,
            min_correction_ms=60.0,
        )
        cases = (
            (
                telemetry.OffsetEstimate(120.0, 0.95, 0.20, 0.40, 0.50),
                True,
                "accepted",
            ),
            (
                telemetry.OffsetEstimate(120.0, float("nan"), 0.20, 0.40, 0.50),
                False,
                "nonfinite",
            ),
            (
                telemetry.OffsetEstimate(120.0, 0.79, 0.20, 0.40, 0.50),
                False,
                "peak_correlation",
            ),
            (
                telemetry.OffsetEstimate(120.0, 0.95, 0.09, 0.40, 0.50),
                False,
                "peak_margin",
            ),
            (
                telemetry.OffsetEstimate(120.0, 0.95, 0.20, 0.19, 0.50),
                False,
                "visual_mad",
            ),
            (
                telemetry.OffsetEstimate(120.0, 0.95, 0.20, 0.40, 0.29),
                False,
                "state_mad",
            ),
            (
                telemetry.OffsetEstimate(80.0, 0.95, 0.20, 0.40, 0.50),
                False,
                "minimum_correction",
            ),
        )

        for estimate, accepted, reason in cases:
            with self.subTest(reason=reason):
                decision = decide_temporal_guard(estimate, guard)
                self.assertIsInstance(decision, TemporalGuardDecision)
                self.assertEqual(decision.accepted, accepted)
                self.assertEqual(decision.correction_ms, estimate.offset_ms - 40.0)
                self.assertEqual(decision.reason, reason)
        with self.assertRaises(FrozenInstanceError):
            decide_temporal_guard(cases[0][0], guard).accepted = False

    def test_decision_is_exactly_parity_with_guarded_rt9_evaluation(self):
        trial = _informative_trial(
            telemetry.TemporalTrial,
            trajectory_id="public-guard-parity",
            injected_offset_ms=80.0,
        )
        guard = telemetry.TemporalGuard(
            native_delay_ms=40.0,
            min_peak_correlation=0.80,
            min_peak_margin=0.10,
            min_visual_mad=0.20,
            min_state_mad=0.30,
            min_correction_ms=60.0,
        )
        estimates = (
            telemetry.OffsetEstimate(120.0, 0.95, 0.20, 0.40, 0.50),
            telemetry.OffsetEstimate(120.0, 0.79, 0.20, 0.40, 0.50),
            telemetry.OffsetEstimate(80.0, 0.95, 0.20, 0.40, 0.50),
            telemetry.OffsetEstimate(120.0, float("nan"), 0.20, 0.40, 0.50),
        )

        for estimate in estimates:
            with self.subTest(estimate=estimate):
                decision = decide_temporal_guard(estimate, guard)
                row = telemetry.evaluate_temporal_method(
                    trial, "kinesync_guarded", guard=guard, estimate=estimate
                )
                self.assertEqual(row.committed, decision.accepted)
                self.assertEqual(row.guard_reason, decision.reason)
                self.assertEqual(
                    row.applied_correction_ms,
                    decision.correction_ms if decision.accepted else 0.0,
                )


class TemporalMethodEvaluationTest(unittest.TestCase):
    def test_emits_matched_methods_with_exact_raw_oracle_and_rollback_semantics(self):
        evaluate = getattr(telemetry, "evaluate_temporal_method", None)
        methods = getattr(telemetry, "TEMPORAL_METHODS", None)
        row_type = getattr(telemetry, "TemporalResultRow", None)
        self.assertIsNotNone(evaluate, "temporal method evaluation must be available")
        self.assertIsNotNone(methods, "the five matched methods must be declared")
        self.assertIsNotNone(row_type, "typed temporal result rows must be available")

        trial = _informative_trial(
            telemetry.TemporalTrial,
            trajectory_id="held-out-a",
            injected_offset_ms=80.0,
        )
        permissive_guard = telemetry.TemporalGuard(
            native_delay_ms=40.0,
            min_peak_correlation=-1.0,
            min_peak_margin=0.0,
            min_visual_mad=0.0,
            min_state_mad=0.0,
            min_correction_ms=0.0,
        )
        rollback_guard = telemetry.TemporalGuard(
            native_delay_ms=40.0,
            min_peak_correlation=1.0,
            min_peak_margin=1.0,
            min_visual_mad=0.0,
            min_state_mad=0.0,
            min_correction_ms=0.0,
        )

        rows = [
            evaluate(trial, method, guard=permissive_guard)
            for method in methods
        ]
        by_method = {row.method: row for row in rows}
        self.assertEqual(
            set(by_method),
            {
                "raw_nearest",
                "raw_linear",
                "motion_unguarded",
                "kinesync_guarded",
                "oracle_offset",
            },
        )
        raw_nearest = by_method["raw_nearest"]
        raw_linear = by_method["raw_linear"]
        oracle = by_method["oracle_offset"]
        np.testing.assert_array_equal(
            raw_nearest.interpolated_qpos,
            telemetry.interpolate_qpos(
                trial.features.camera_midpoint_ns,
                trial.corrupted_state.timestamp_ns,
                trial.corrupted_state.qpos,
                mode="nearest",
            ),
        )
        np.testing.assert_array_equal(
            raw_linear.interpolated_qpos,
            telemetry.interpolate_qpos(
                trial.features.camera_midpoint_ns,
                trial.corrupted_state.timestamp_ns,
                trial.corrupted_state.qpos,
                mode="linear",
            ),
        )
        np.testing.assert_array_equal(
            oracle.interpolated_qpos,
            telemetry.interpolate_qpos(
                trial.features.camera_midpoint_ns,
                trial.corrupted_state.timestamp_ns - 80_000_000,
                trial.corrupted_state.qpos,
                mode="linear",
            ),
        )
        self.assertEqual(raw_nearest.applied_correction_ms, 0.0)
        self.assertEqual(raw_linear.applied_correction_ms, 0.0)
        self.assertEqual(oracle.applied_correction_ms, 80.0)
        self.assertFalse(raw_nearest.committed)
        self.assertFalse(raw_linear.committed)
        self.assertFalse(oracle.committed)
        self.assertTrue(by_method["kinesync_guarded"].committed)

        guarded_rollback = evaluate(
            trial, "kinesync_guarded", guard=rollback_guard
        )
        self.assertFalse(guarded_rollback.committed)
        self.assertEqual(guarded_rollback.applied_correction_ms, 0.0)
        self.assertEqual(
            guarded_rollback.interpolated_qpos.tobytes(),
            raw_linear.interpolated_qpos.tobytes(),
        )

        zero_trial = _informative_trial(
            telemetry.TemporalTrial,
            trajectory_id="held-out-zero",
            injected_offset_ms=0.0,
        )
        zero_guarded = evaluate(
            zero_trial, "kinesync_guarded", guard=permissive_guard
        )
        self.assertFalse(zero_guarded.committed)
        self.assertEqual(zero_guarded.applied_correction_ms, 0.0)

    def test_summarizes_qmae_as_a_trajectory_macro_average(self):
        row_type = getattr(telemetry, "TemporalResultRow", None)
        summarize = getattr(telemetry, "summarize_temporal_results", None)
        self.assertIsNotNone(row_type, "typed temporal result rows must be available")
        self.assertIsNotNone(summarize, "temporal result summarization must be available")

        rows = [
            _result_row(
                row_type,
                trajectory_id="long-trajectory",
                condition_id=f"long-{index}",
                qmae_rad=1.0,
            )
            for index in range(100)
        ]
        rows.append(
            _result_row(
                row_type,
                trajectory_id="short-trajectory",
                condition_id="short-0",
                qmae_rad=3.0,
            )
        )

        summary = summarize(rows)
        metrics = {metric.method: metric for metric in summary.methods}

        self.assertEqual(metrics["raw_linear"].condition_count, 101)
        self.assertEqual(metrics["raw_linear"].trajectory_count, 2)
        self.assertEqual(metrics["raw_linear"].qmae_rad, 2.0)

    def test_summarizes_commit_precision_as_trajectory_macro_benefit_precision(self):
        row_type = telemetry.TemporalResultRow
        rows = [
            _result_row(
                row_type,
                trajectory_id="long-trajectory",
                condition_id=f"long-{index}",
                qmae_rad=0.5,
                method="kinesync_guarded",
                committed=True,
                qmae_reduction_rad=0.5,
                recovery_success=False,
            )
            for index in range(100)
        ]
        rows.append(
            _result_row(
                row_type,
                trajectory_id="short-trajectory",
                condition_id="short-0",
                qmae_rad=2.0,
                method="kinesync_guarded",
                committed=True,
                qmae_reduction_rad=-1.0,
                recovery_success=True,
            )
        )

        summary = telemetry.summarize_temporal_results(rows)

        self.assertEqual(summary.methods[0].commit_precision, 0.5)


def _result_row(
    row_type,
    *,
    trajectory_id: str,
    condition_id: str,
    qmae_rad: float,
    method: str = "raw_linear",
    committed: bool = False,
    qmae_reduction_rad: float = 0.0,
    recovery_success: bool = False,
):
    return row_type(
        trajectory_id=trajectory_id,
        split="test",
        condition_id=condition_id,
        method=method,
        injected_offset_ms=80.0,
        jitter_ms=0.0,
        dropout=0.0,
        native_delay_ms=40.0,
        estimated_absolute_offset_ms=120.0,
        estimated_offset_ms=80.0,
        applied_correction_ms=80.0 if committed else 0.0,
        committed=committed,
        guard_reason="baseline",
        peak_correlation=1.0,
        peak_margin=0.2,
        visual_mad=1.0,
        state_mad=1.0,
        offset_mae_ms=80.0,
        qmae_rad=qmae_rad,
        qmae_reduction_rad=qmae_reduction_rad,
        recovery_success=recovery_success,
        interpolated_qpos=np.zeros((1, 6), dtype=np.float64),
    )


if __name__ == "__main__":
    unittest.main()
