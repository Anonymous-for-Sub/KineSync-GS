from __future__ import annotations

import unittest

import numpy as np

from kinesync.data.take_pens import ContinuousSegment
from kinesync.data.embodiment import (
    ABC_YAM_INTERLEAVED_14,
    FRANKA_7,
    PIPER_6,
    YAM_DUAL_ARM_12,
)
from kinesync.temporal.telemetry import (
    CorruptedState,
    MotionFeatures,
    OffsetEstimate,
    TemporalTrial,
    corrupt_state_stream,
    evaluate_temporal_method,
    extract_motion_features,
    interpolate_qpos,
    search_motion_offset,
)


class SyntheticTrajectory:
    trajectory_id = "f1.hdf5"
    split = "test"

    def __init__(self) -> None:
        self._head_timestamps = np.array([0, 10, 20, 30], dtype=np.int64)
        self._wrist_timestamps = np.array([1, 11, 21, 31], dtype=np.int64)
        self._slave_timestamps = np.array([0, 10, 20, 30], dtype=np.int64)
        self._head_frames = self._constant_frames([0, 10, 30, 60])
        self._wrist_frames = self._constant_frames([0, 30, 40, 60])
        self._joints = np.zeros((4, 6), dtype=np.float64)
        self._joints[:, 0] = [0, 1, 3, 6]
        self.rgb_requests: list[tuple[str, np.ndarray]] = []

    @staticmethod
    def _constant_frames(values: list[int]) -> np.ndarray:
        return np.stack(
            [np.full((3, 4, 3), value, dtype=np.uint8) for value in values]
        )

    def timestamps(self, camera: str) -> np.ndarray:
        timestamps = {
            "head": self._head_timestamps,
            "wrist": self._wrist_timestamps,
            "slave": self._slave_timestamps,
        }[camera]
        return timestamps

    def slave_joints(self) -> np.ndarray:
        return self._joints

    def read_rgb(self, camera: str, indices: np.ndarray) -> np.ndarray:
        self.rgb_requests.append((camera, np.asarray(indices).copy()))
        frames = {"head": self._head_frames, "wrist": self._wrist_frames}[camera]
        return frames[indices]


class TimedTrajectory:
    trajectory_id = "f1.hdf5"
    split = "test"

    def __init__(
        self,
        *,
        head_timestamps: list[int],
        wrist_timestamps: list[int],
        slave_timestamps: list[int],
        head_values: list[int],
        wrist_values: list[int],
        slave_joint_values: list[float],
    ) -> None:
        self._timestamps = {
            "head": np.asarray(head_timestamps, dtype=np.int64),
            "wrist": np.asarray(wrist_timestamps, dtype=np.int64),
            "slave": np.asarray(slave_timestamps, dtype=np.int64),
        }
        self._frames = {
            "head": SyntheticTrajectory._constant_frames(head_values),
            "wrist": SyntheticTrajectory._constant_frames(wrist_values),
        }
        self._joints = np.zeros((len(slave_joint_values), 6), dtype=np.float64)
        self._joints[:, 0] = slave_joint_values

    def timestamps(self, camera: str) -> np.ndarray:
        return self._timestamps[camera]

    def slave_joints(self) -> np.ndarray:
        return self._joints

    def read_rgb(self, camera: str, indices: np.ndarray) -> np.ndarray:
        return self._frames[camera][indices]


class ExtractMotionFeaturesTest(unittest.TestCase):
    def test_fuses_normalized_head_and_wrist_motion_at_head_midpoints(self):
        trajectory = SyntheticTrajectory()
        segment = ContinuousSegment("f1.hdf5", "test", 0, 4)

        features = extract_motion_features(trajectory, segment, image_size=(2, 3))

        np.testing.assert_array_equal(features.camera_midpoint_ns, [5, 15, 25])
        np.testing.assert_allclose(features.visual_motion, [0.0, -0.4, 0.45])
        np.testing.assert_allclose(features.state_motion, [-1.0, 0.0, 1.0])
        self.assertEqual(features.visual_motion.shape, (3,))
        self.assertEqual(features.state_motion.shape, (3,))
        self.assertEqual(
            [(camera, indices.tolist()) for camera, indices in trajectory.rgb_requests],
            [("head", [0, 1, 2, 3]), ("wrist", [0, 1, 2, 3])],
        )

    def test_returns_finite_constant_motion_without_mutating_trajectory_arrays(self):
        trajectory = SyntheticTrajectory()
        trajectory._head_frames.fill(17)
        trajectory._wrist_frames.fill(23)
        trajectory._joints.fill(4.0)
        before = (
            trajectory._head_timestamps.copy(),
            trajectory._wrist_timestamps.copy(),
            trajectory._slave_timestamps.copy(),
            trajectory._head_frames.copy(),
            trajectory._wrist_frames.copy(),
            trajectory._joints.copy(),
        )

        features = extract_motion_features(
            trajectory, ContinuousSegment("f1.hdf5", "test", 0, 4)
        )

        self.assertTrue(np.isfinite(features.visual_motion).all())
        self.assertTrue(np.isfinite(features.state_motion).all())
        np.testing.assert_array_equal(features.visual_motion, np.zeros(3))
        np.testing.assert_array_equal(features.state_motion, np.zeros(3))
        for actual, expected in zip(
            (
                trajectory._head_timestamps,
                trajectory._wrist_timestamps,
                trajectory._slave_timestamps,
                trajectory._head_frames,
                trajectory._wrist_frames,
                trajectory._joints,
            ),
            before,
            strict=True,
        ):
            np.testing.assert_array_equal(actual, expected)

    def test_resamples_slave_joints_onto_head_frame_timestamps_before_differencing(self):
        trajectory = TimedTrajectory(
            head_timestamps=[0, 10, 30, 60, 100],
            wrist_timestamps=[0, 10, 30, 60, 100],
            slave_timestamps=[0, 50, 100, 150, 200],
            head_values=[0, 0, 0, 0, 0],
            wrist_values=[0, 0, 0, 0, 0],
            slave_joint_values=[0, 10, 20, 30, 40],
        )

        features = extract_motion_features(
            trajectory, ContinuousSegment("f1.hdf5", "test", 0, 5)
        )

        np.testing.assert_array_equal(features.camera_midpoint_ns, [5, 20, 45, 80])
        np.testing.assert_allclose(features.state_motion, [-1.5, -0.5, 0.5, 1.5])

    def test_interpolates_normalized_wrist_motion_to_head_interval_midpoints(self):
        trajectory = TimedTrajectory(
            head_timestamps=[0, 10, 40, 100],
            wrist_timestamps=[0, 20, 50, 110],
            slave_timestamps=[0, 10, 40, 100],
            head_values=[7, 7, 7, 7],
            wrist_values=[0, 10, 30, 60],
            slave_joint_values=[0, 0, 0, 0],
        )

        features = extract_motion_features(
            trajectory, ContinuousSegment("f1.hdf5", "test", 0, 4)
        )

        np.testing.assert_array_equal(features.camera_midpoint_ns, [5, 25, 70])
        np.testing.assert_allclose(features.visual_motion, [-0.5, -0.2, 7.0 / 18.0])

    def test_uses_all_explicitly_selected_franka_joints_for_state_motion(self):
        trajectory = TimedTrajectory(
            head_timestamps=[0, 10, 20, 30],
            wrist_timestamps=[0, 10, 20, 30],
            slave_timestamps=[0, 10, 20, 30],
            head_values=[0, 0, 0, 0],
            wrist_values=[0, 0, 0, 0],
            slave_joint_values=[0, 0, 0, 0],
        )
        trajectory._joints = np.zeros((4, 7), dtype=np.float64)
        trajectory._joints[:, 6] = [0.0, 1.0, 3.0, 6.0]

        default_features = extract_motion_features(
            trajectory, ContinuousSegment("f1.hdf5", "test", 0, 4)
        )
        franka_features = extract_motion_features(
            trajectory,
            ContinuousSegment("f1.hdf5", "test", 0, 4),
            joint_selection=FRANKA_7.select_groups("arm"),
        )

        np.testing.assert_array_equal(default_features.state_motion, np.zeros(3))
        np.testing.assert_allclose(franka_features.state_motion, [-1.0, 0.0, 1.0])


class JointSelectionEvaluationTest(unittest.TestCase):
    def test_abc_interleaved_selection_flows_through_features_search_and_qmae(self):
        selection = ABC_YAM_INTERLEAVED_14.select_groups("left_arm", "right_arm")
        trajectory = TimedTrajectory(
            head_timestamps=[0, 10, 20, 30],
            wrist_timestamps=[0, 10, 20, 30],
            slave_timestamps=[0, 10, 20, 30],
            head_values=[0, 10, 30, 60],
            wrist_values=[0, 10, 30, 60],
            slave_joint_values=[0, 0, 0, 0],
        )
        qpos = np.zeros((4, 14), dtype=np.float64)
        qpos[:, 7] = [0.0, 1.0, 3.0, 6.0]
        qpos[:, 6] = [0.0, 100.0, 1100.0, 1200.0]
        qpos[:, 13] = [0.0, 100.0, 1100.0, 1200.0]
        trajectory._joints = qpos
        features = extract_motion_features(
            trajectory,
            ContinuousSegment("f1.hdf5", "test", 0, 4),
            joint_selection=selection,
        )
        state = CorruptedState(
            timestamp_ns=np.array([0, 10, 20, 30], dtype=np.int64),
            qpos=qpos,
            retained_indices=np.array([0, 1, 2, 3], dtype=np.int64),
        )

        self.assertEqual(features.joint_selection, selection)
        np.testing.assert_allclose(features.state_motion, [-1.0, 0.0, 1.0])
        estimate = search_motion_offset(features, state, candidate_offsets_ms=(0.0,))
        self.assertEqual(estimate.offset_ms, 0.0)
        self.assertAlmostEqual(estimate.peak_correlation, 1.0)

        target = np.zeros((3, 14), dtype=np.float64)
        target[:, 7] = [1.5, 3.0, 5.5]
        target[:, 8:13] = [2.0, 3.0, 4.0, 5.0, 6.0]
        target[:, (6, 13)] = 1_000_000.0
        trial = TemporalTrial(
            trajectory_id="abc-yam",
            split="test",
            condition_id="zero",
            features=features,
            corrupted_state=state,
            target_qpos=target,
            injected_offset_ms=0.0,
            jitter_ms=0.0,
            dropout=0.0,
            joint_selection=selection,
        )
        row = evaluate_temporal_method(trial, "raw_linear", estimate=estimate)

        self.assertAlmostEqual(row.qmae_rad, 21.0 / 12.0)
        self.assertEqual(row.to_dict()["joint_names"].split(",")[6:], list(selection.joint_names[6:]))

    def test_rejects_trial_selection_that_conflicts_with_feature_metadata(self):
        trajectory = TimedTrajectory(
            head_timestamps=[0, 10, 20, 30],
            wrist_timestamps=[0, 10, 20, 30],
            slave_timestamps=[0, 10, 20, 30],
            head_values=[0, 0, 0, 0],
            wrist_values=[0, 0, 0, 0],
            slave_joint_values=[0, 0, 0, 0],
        )
        trajectory._joints = np.zeros((4, 14), dtype=np.float64)
        features = extract_motion_features(
            trajectory,
            ContinuousSegment("f1.hdf5", "test", 0, 4),
            joint_selection=ABC_YAM_INTERLEAVED_14.select_groups("left_arm", "right_arm"),
        )
        state = CorruptedState(
            timestamp_ns=np.array([0, 10, 20, 30], dtype=np.int64),
            qpos=np.zeros((4, 14), dtype=np.float64),
            retained_indices=np.array([0, 1, 2, 3], dtype=np.int64),
        )

        with self.assertRaisesRegex(ValueError, "motion feature selection"):
            TemporalTrial(
                trajectory_id="abc-yam-mismatch",
                split="test",
                condition_id="zero",
                features=features,
                corrupted_state=state,
                target_qpos=np.zeros((3, 14), dtype=np.float64),
                injected_offset_ms=0.0,
                jitter_ms=0.0,
                dropout=0.0,
            )

    def test_uses_trial_selection_for_qmae_and_keeps_piper_default_identical(self):
        features = MotionFeatures(
            camera_midpoint_ns=np.array([5, 15], dtype=np.int64),
            visual_motion=np.array([0.0, 1.0]),
            state_motion=np.array([0.0, 1.0]),
        )
        state = CorruptedState(
            timestamp_ns=np.array([0, 10, 20], dtype=np.int64),
            qpos=np.zeros((3, 8), dtype=np.float64),
            retained_indices=np.array([0, 1, 2], dtype=np.int64),
        )
        target = np.zeros((2, 8), dtype=np.float64)
        target[:, 6] = 2.0
        estimate = OffsetEstimate(0.0, 1.0, 1.0, 1.0, 1.0)
        kwargs = dict(
            trajectory_id="selection",
            split="test",
            condition_id="zero",
            features=features,
            corrupted_state=state,
            target_qpos=target,
            injected_offset_ms=0.0,
            jitter_ms=0.0,
            dropout=0.0,
        )

        default_row = evaluate_temporal_method(
            TemporalTrial(**kwargs), "raw_linear", estimate=estimate
        )
        explicit_piper_row = evaluate_temporal_method(
            TemporalTrial(**kwargs, joint_selection=PIPER_6.select_groups("arm")),
            "raw_linear",
            estimate=estimate,
        )
        franka_row = evaluate_temporal_method(
            TemporalTrial(**kwargs, joint_selection=FRANKA_7.select_groups("arm")),
            "raw_linear",
            estimate=estimate,
        )

        self.assertEqual(default_row.qmae_rad, explicit_piper_row.qmae_rad)
        self.assertEqual(default_row.qmae_rad, 0.0)
        self.assertAlmostEqual(franka_row.qmae_rad, 2.0 / 7.0)

    def test_yam_twelve_arm_joint_metric_excludes_separate_gripper_units(self):
        features = MotionFeatures(
            camera_midpoint_ns=np.array([5, 15], dtype=np.int64),
            visual_motion=np.array([0.0, 1.0]),
            state_motion=np.array([0.0, 1.0]),
        )
        state = CorruptedState(
            timestamp_ns=np.array([0, 10, 20], dtype=np.int64),
            qpos=np.zeros((3, 14), dtype=np.float64),
            retained_indices=np.array([0, 1, 2], dtype=np.int64),
        )
        target = np.zeros((2, 14), dtype=np.float64)
        target[:, 10] = 1.0
        target[:, 12] = 100.0
        trial = TemporalTrial(
            trajectory_id="yam",
            split="test",
            condition_id="zero",
            features=features,
            corrupted_state=state,
            target_qpos=target,
            injected_offset_ms=0.0,
            jitter_ms=0.0,
            dropout=0.0,
            joint_selection=YAM_DUAL_ARM_12.select_groups("left_arm", "right_arm"),
        )

        row = evaluate_temporal_method(
            trial, "raw_linear", estimate=OffsetEstimate(0.0, 1.0, 1.0, 1.0, 1.0)
        )

        self.assertAlmostEqual(row.qmae_rad, 1.0 / 12.0)


class CorruptStateStreamTest(unittest.TestCase):
    def setUp(self) -> None:
        self.timestamps = np.arange(5, dtype=np.int64) * 10_000_000
        self.qpos = np.arange(40, dtype=np.float64).reshape(5, 8)

    def test_applies_an_exact_offset_without_jitter_or_dropout(self):
        corrupted = corrupt_state_stream(
            self.timestamps,
            self.qpos,
            offset_ms=-12,
            jitter_ms=0,
            dropout=0,
            seed=7,
        )

        np.testing.assert_array_equal(
            corrupted.timestamp_ns, self.timestamps - 12_000_000
        )
        np.testing.assert_array_equal(corrupted.qpos, self.qpos)
        np.testing.assert_array_equal(corrupted.retained_indices, np.arange(5))

    def test_repeats_byte_identically_for_the_same_seed(self):
        first = corrupt_state_stream(
            self.timestamps, self.qpos, 7, 4, 0.4, seed=1234
        )
        second = corrupt_state_stream(
            self.timestamps, self.qpos, 7, 4, 0.4, seed=1234
        )

        self.assertEqual(first.timestamp_ns.tobytes(), second.timestamp_ns.tobytes())
        self.assertEqual(first.qpos.tobytes(), second.qpos.tobytes())
        self.assertEqual(first.retained_indices.tobytes(), second.retained_indices.tobytes())

    def test_different_seeds_change_a_nonzero_jitter_stream(self):
        first = corrupt_state_stream(
            self.timestamps, self.qpos, 0, 4, 0, seed=17
        )
        second = corrupt_state_stream(
            self.timestamps, self.qpos, 0, 4, 0, seed=18
        )

        self.assertNotEqual(first.timestamp_ns.tobytes(), second.timestamp_ns.tobytes())

    def test_known_seeds_change_a_dropout_only_stream(self):
        timestamps = np.arange(8, dtype=np.int64) * 10_000_000
        qpos = np.arange(64, dtype=np.float64).reshape(8, 8)
        first = corrupt_state_stream(timestamps, qpos, 0, 0, 0.5, seed=1)
        second = corrupt_state_stream(timestamps, qpos, 0, 0, 0.5, seed=2)

        self.assertNotEqual(first.retained_indices.tobytes(), second.retained_indices.tobytes())
        self.assertNotEqual(first.qpos.tobytes(), second.qpos.tobytes())
        self.assertIn(0, first.retained_indices)
        self.assertIn(7, first.retained_indices)
        self.assertIn(0, second.retained_indices)
        self.assertIn(7, second.retained_indices)

    def test_retains_original_endpoints_and_reorders_state_with_timestamps(self):
        original_timestamps = self.timestamps.copy()
        original_qpos = self.qpos.copy()
        corrupted = corrupt_state_stream(
            self.timestamps, self.qpos, 0, 15, 0.6, seed=4
        )

        self.assertIn(0, corrupted.retained_indices)
        self.assertIn(len(self.timestamps) - 1, corrupted.retained_indices)
        self.assertTrue(np.all(np.diff(corrupted.timestamp_ns) > 0))
        np.testing.assert_array_equal(
            corrupted.qpos, original_qpos[corrupted.retained_indices]
        )
        np.testing.assert_array_equal(self.timestamps, original_timestamps)
        np.testing.assert_array_equal(self.qpos, original_qpos)

    def test_stably_repairs_forced_jitter_inversions_and_collisions(self):
        timestamps = np.array([0, 257_835, 300_000, 400_000, 500_000], dtype=np.int64)
        qpos = np.arange(40, dtype=np.float64).reshape(5, 8)

        corrupted = corrupt_state_stream(timestamps, qpos, 0, 1, 0, seed=0)

        np.testing.assert_array_equal(corrupted.retained_indices, [4, 0, 1, 3, 2])
        np.testing.assert_array_equal(
            corrupted.timestamp_ns, [-35_669, 125_730, 125_731, 504_900, 940_423]
        )
        np.testing.assert_array_equal(corrupted.qpos, qpos[[4, 0, 1, 3, 2]])
        self.assertIn(0, corrupted.retained_indices)
        self.assertIn(4, corrupted.retained_indices)


class InterpolateQposTest(unittest.TestCase):
    def setUp(self) -> None:
        self.state_ns = np.array([0, 10, 20], dtype=np.int64)
        self.qpos = np.array([[0.0, 0.0], [10.0, 20.0], [20.0, 40.0]])
        self.query_ns = np.array([-5, 0, 4, 5, 6, 10, 15, 25], dtype=np.int64)

    def test_selects_the_nearest_sample_with_earlier_tie_breaking(self):
        interpolated = interpolate_qpos(
            self.query_ns, self.state_ns, self.qpos, mode="nearest"
        )

        np.testing.assert_array_equal(
            interpolated,
            np.array(
                [
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [10.0, 20.0],
                    [10.0, 20.0],
                    [10.0, 20.0],
                    [20.0, 40.0],
                ]
            ),
        )

    def test_linearly_interpolates_a_known_ramp_and_clamps_endpoints(self):
        interpolated = interpolate_qpos(
            self.query_ns, self.state_ns, self.qpos, mode="linear"
        )

        np.testing.assert_allclose(
            interpolated,
            np.array(
                [
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [4.0, 8.0],
                    [5.0, 10.0],
                    [6.0, 12.0],
                    [10.0, 20.0],
                    [15.0, 30.0],
                    [20.0, 40.0],
                ]
            ),
        )


if __name__ == "__main__":
    unittest.main()
