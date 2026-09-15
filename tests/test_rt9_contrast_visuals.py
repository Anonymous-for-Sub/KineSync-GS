"""Contracts for visibly distinct RT9 temporal visual evidence."""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path
import unittest

import cv2
import numpy as np

from kinesync.visualization.rt9_contrast_visuals import (
    choose_visible_temporal_frame,
    select_best_segment_conditions,
)


class RT9ContrastVisualsTest(unittest.TestCase):
    def test_selects_one_strongest_positive_guarded_condition_per_segment(self) -> None:
        fields = ["trajectory_id", "condition_id", "method", "committed", "qmae_rad", "estimated_offset_ms", "applied_correction_ms"]
        rows = [
            ["f1.hdf5", "f1.hdf5:10:80|a", "raw_linear", "False", "0.10", "120", "0"],
            ["f1.hdf5", "f1.hdf5:10:80|a", "kinesync_guarded", "True", "0.02", "120", "120"],
            ["f1.hdf5", "f1.hdf5:10:80|b", "raw_linear", "False", "0.08", "120", "0"],
            ["f1.hdf5", "f1.hdf5:10:80|b", "kinesync_guarded", "True", "0.03", "120", "120"],
            ["f2.hdf5", "f2.hdf5:20:90|a", "raw_linear", "False", "0.09", "-120", "0"],
            ["f2.hdf5", "f2.hdf5:20:90|a", "kinesync_guarded", "True", "0.01", "-120", "-120"],
        ]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "results.csv"
            with path.open("w", newline="") as handle:
                writer = csv.writer(handle); writer.writerow(fields); writer.writerows(rows)
            selected = select_best_segment_conditions(path)
        self.assertEqual(len(selected), 2)
        self.assertEqual(selected[0].condition_id, "f1.hdf5:10:80|a")
        self.assertAlmostEqual(selected[0].qmae_reduction, 0.08)

    def test_visible_frame_uses_motion_and_positive_state_difference(self) -> None:
        images = np.full((12, 48, 64, 3), 180, dtype=np.uint8)
        for index in range(12):
            cv2.rectangle(images[index], (4 + index * 3, 12), (14 + index * 3, 36), (20, 20, 20), -1)
        qpos = np.zeros((12, 6), dtype=np.float64)
        qpos[:, 0] = np.linspace(0.0, 1.1, 12)
        chosen = choose_visible_temporal_frame(
            images,
            qpos,
            start=2,
            stop=10,
            baseline_shift=-2,
            ours_shift=0,
        )
        self.assertEqual(chosen.ours_index, chosen.true_index)
        self.assertEqual(chosen.baseline_index, chosen.true_index - 2)
        self.assertGreater(chosen.baseline_qmae, chosen.ours_qmae)
        self.assertGreater(chosen.pixel_mae, 0.0)


if __name__ == "__main__":
    unittest.main()
