"""Contracts for the high-fidelity Robust V2 visual evidence bundle."""

from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

import cv2
import numpy as np

from kinesync.visualization.robust_v2_visuals import (
    AlignmentCase,
    METHOD_BGR,
    compose_comparison_grid,
    extract_replay_panels,
    retrieve_index,
    select_recommended_cases,
)


class RobustV2VisualsTest(unittest.TestCase):
    def test_extracts_four_native_panels_without_header(self) -> None:
        frame = np.zeros((512, 2560, 3), dtype=np.uint8)
        for index, value in enumerate((30, 70, 110, 150)):
            frame[32:, index * 640 : (index + 1) * 640] = value
        panels = extract_replay_panels(frame)
        self.assertEqual(set(panels), {"real_head", "gs_head", "real_extra", "gs_extra"})
        self.assertTrue(all(image.shape == (480, 640, 3) for image in panels.values()))
        self.assertEqual([int(panels[name][0, 0, 0]) for name in panels], [30, 70, 110, 150])

    def test_recommendation_requires_positive_state_and_visible_pixel_gain(self) -> None:
        cases = [
            AlignmentCase("head", 10, 7, 10, 0.08, 0.01, 12.0, 1.0),
            AlignmentCase("head", 20, 17, 20, 0.08, 0.01, 0.2, 1.0),
            AlignmentCase("head", 30, 27, 26, 0.08, 0.09, 15.0, 1.0),
        ]
        selected = select_recommended_cases(cases, count=3, minimum_pixel_mae=2.0)
        self.assertEqual([case.true_index for case in selected], [10])

    def test_edge_retrieval_selects_the_aligned_articulated_silhouette(self) -> None:
        candidates = []
        for center_x in (180, 260, 340):
            image = np.full((480, 640, 3), 255, dtype=np.uint8)
            cv2.rectangle(image, (center_x - 35, 100), (center_x + 35, 380), (30, 30, 30), -1)
            candidates.append(image)
        real = np.full((480, 640, 3), 170, dtype=np.uint8)
        cv2.rectangle(real, (225, 100), (295, 380), (30, 30, 30), -1)
        selected, scores = retrieve_index(real, candidates, candidate_indices=(7, 8, 9))
        self.assertEqual(selected, 8)
        self.assertLess(scores[8], scores[7])
        self.assertLess(scores[8], scores[9])

    def test_grid_rows_are_semantic_and_borders_touch_image_bounds(self) -> None:
        real = np.full((480, 640, 3), (90, 120, 140), dtype=np.uint8)
        baseline = np.full((480, 640, 3), (20, 40, 60), dtype=np.uint8)
        ours = np.full((480, 640, 3), (180, 160, 140), dtype=np.uint8)
        cases = [
            AlignmentCase("head", index, index - 3, index, 0.08, 0.0, 20.0, 1.0)
            for index in (100, 200, 300)
        ]
        images = {case.true_index: (real, baseline, ours) for case in cases}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "grid.png"
            geometry = compose_comparison_grid(
                cases,
                images,
                path,
                rows=("real", "baseline", "ours"),
                cell_size=(640, 480),
            )
            rendered = cv2.imread(str(path), cv2.IMREAD_COLOR)
        self.assertEqual(geometry.layout, "3x3")
        self.assertEqual((geometry.width, geometry.height), (1944, 1536))
        # First image starts after a two-pixel reference border and no square letterbox.
        self.assertTrue(np.all(rendered[2:482, 2:642] == real))
        self.assertTrue(np.all(rendered[514:994, 2:642] == baseline))
        self.assertTrue(np.all(rendered[1026:1506, 2:642] == ours))
        self.assertTrue(np.all(rendered[512:514, 0:644] == METHOD_BGR["baseline"]))
        self.assertTrue(np.all(rendered[1024:1026, 0:644] == METHOD_BGR["ours"]))

    def test_rejects_unpaired_two_row_grid(self) -> None:
        case = AlignmentCase("head", 10, 7, 10, 0.08, 0.0, 20.0, 1.0)
        image = np.full((480, 640, 3), 127, dtype=np.uint8)
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "column count"):
                compose_comparison_grid(
                    [case],
                    {10: (image, image, image)},
                    Path(temporary) / "bad.png",
                    rows=("baseline", "ours"),
                    cell_size=(640, 480),
                )


if __name__ == "__main__":
    unittest.main()
