from __future__ import annotations

import unittest

import numpy as np

from kinesync.data.paired_real_replay import PairedReplayFrame
from kinesync.replay.offline_shadow import (
    GuardConstraints,
    build_replay_trials,
    cluster_bootstrap,
    evaluate_replay,
    select_guard_threshold,
    select_shared_guard_threshold,
)


def _frame(index: int, *, split: str) -> PairedReplayFrame:
    image = np.zeros((24, 32, 3), dtype=np.uint8)
    image[:, index * 5 : index * 5 + 4] = (40 + index * 20, 120, 220)
    qpos = np.asarray([index * 0.04, 0.0, 0.0, 0.0, 0.0, 0.0, 0.01, -0.01])
    return PairedReplayFrame(
        source_pair_id=f"pair-{index}",
        split=split,
        episode_id="episode-0",
        pair_index=index,
        frame_index=index,
        timestamp_ns=index * 100_000_000,
        qpos=qpos,
        real_rgb=image.copy(),
        gaussian_rgb=image.copy(),
        real_path=None,
        gaussian_path=None,
    )


class OfflineShadowReplayTest(unittest.TestCase):
    def test_calibration_guard_recovers_lag_and_preserves_zero_controls(self):
        calibration_episode = [_frame(index, split="calibration") for index in range(8)]
        trials = build_replay_trials(
            {"episode-0": calibration_episode},
            backend="a_p2",
            lags=(-2, 0, 2),
            window_radius=3,
            image_size=(24, 32),
        )
        threshold, calibration_metrics = select_guard_threshold(
            trials,
            constraints=GuardConstraints(
                minimum_commit_precision=0.85,
                minimum_controlled_coverage=0.40,
                minimum_zero_stability=0.95,
                success_qmae_rad=0.01,
            ),
        )
        guarded = evaluate_replay(
            trials,
            threshold=threshold,
            success_qmae_rad=0.01,
        )

        self.assertGreaterEqual(calibration_metrics.commit_precision, 0.85)
        self.assertGreaterEqual(guarded.zero_stability, 0.95)
        self.assertLess(guarded.qmae_rad, guarded.delayed_qmae_rad)
        self.assertGreater(guarded.recovery_success, 0.80)

    def test_event_threshold_is_deterministic_and_calibration_only(self):
        episode = [_frame(index, split="calibration") for index in range(7)]
        trials = build_replay_trials(
            {"episode-0": episode},
            backend="a_p2",
            lags=(-1, 0, 1),
            window_radius=2,
            image_size=(24, 32),
        )
        constraints = GuardConstraints(
            minimum_commit_precision=0.80,
            minimum_controlled_coverage=0.30,
            minimum_zero_stability=0.90,
            success_qmae_rad=0.01,
        )

        first = select_guard_threshold(trials, constraints=constraints)
        second = select_guard_threshold(trials, constraints=constraints)

        self.assertEqual(first, second)
        self.assertTrue(all(trial.split == "calibration" for trial in trials))

    def test_episode_cluster_bootstrap_is_reproducible_and_reports_gain(self):
        episodes = {
            f"episode-{episode}": [
                PairedReplayFrame(
                    **{
                        **_frame(index, split="test").__dict__,
                        "episode_id": f"episode-{episode}",
                        "source_pair_id": f"episode-{episode}-pair-{index}",
                    }
                )
                for index in range(7)
            ]
            for episode in range(3)
        }
        trials = build_replay_trials(
            episodes,
            backend="a_p2",
            lags=(-1, 0, 1),
            window_radius=2,
            image_size=(24, 32),
        )

        first = cluster_bootstrap(
            trials,
            threshold=0.0,
            success_qmae_rad=0.01,
            samples=200,
            seed=17,
        )
        second = cluster_bootstrap(
            trials,
            threshold=0.0,
            success_qmae_rad=0.01,
            samples=200,
            seed=17,
        )

        self.assertEqual(first, second)
        self.assertGreater(first["qmae_reduction_rad"]["estimate"], 0.0)
        self.assertGreaterEqual(first["qmae_reduction_rad"]["ci_low"], 0.0)
        self.assertEqual(first["cluster_count"], 3)

    def test_shared_threshold_satisfies_every_visual_backend(self):
        episode = [_frame(index, split="calibration") for index in range(8)]
        first = build_replay_trials(
            {"episode-0": episode},
            backend="component_gs",
            lags=(-2, 0, 2),
            window_radius=3,
            image_size=(24, 32),
        )
        second = [
            trial.__class__(**{**trial.__dict__, "backend": "hybrid_gs"})
            for trial in first
        ]
        constraints = GuardConstraints(
            minimum_commit_precision=0.85,
            minimum_controlled_coverage=0.40,
            minimum_zero_stability=0.95,
            success_qmae_rad=0.01,
        )

        threshold, metrics = select_shared_guard_threshold(
            {"component_gs": first, "hybrid_gs": second},
            constraints=constraints,
        )

        self.assertGreaterEqual(threshold, 0.0)
        self.assertEqual(set(metrics), {"component_gs", "hybrid_gs"})
        self.assertTrue(
            all(
                value.commit_precision >= constraints.minimum_commit_precision
                and value.controlled_coverage >= constraints.minimum_controlled_coverage
                and value.zero_stability >= constraints.minimum_zero_stability
                for value in metrics.values()
            )
        )


if __name__ == "__main__":
    unittest.main()
