from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from kinesync.visualization.offline_shadow import (
    OfflineShadowVisualFrame,
    write_offline_shadow_image,
    write_offline_shadow_video,
)


def _context(backend: str, value: int) -> OfflineShadowVisualFrame:
    base = np.full((120, 160, 3), (value, 100, 180), dtype=np.uint8)
    delayed = base.copy()
    delayed[:, :40] = (20, 40, 220)
    corrected = base.copy()
    corrected[:, 80:] = (30, 190, 90)
    return OfflineShadowVisualFrame(
        backend=backend,
        episode_id="003",
        source_pair_id=f"pair-{backend}",
        lag=-2,
        committed=True,
        delayed_qmae_rad=0.08,
        corrected_qmae_rad=0.003,
        real_rgb=base,
        delayed_gaussian_rgb=delayed,
        corrected_gaussian_rgb=corrected,
    )


class OfflineShadowVisualizationTest(unittest.TestCase):
    def test_image_and_video_are_decodable_1080p(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frames = [_context("A-P2 Component-GS", 60), _context("Route-B Hybrid", 90)]
            image_path = root / "comparison.png"
            video_path = root / "comparison.mp4"

            write_offline_shadow_image(image_path, frames)
            write_offline_shadow_video(video_path, [frames, list(reversed(frames))], fps=12)

            image = cv2.imread(str(image_path))
            self.assertIsNotNone(image)
            self.assertEqual(image.shape[:2], (1080, 1920))
            capture = cv2.VideoCapture(str(video_path))
            self.assertTrue(capture.isOpened())
            self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), 1920)
            self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)), 1080)
            self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 2)
            capture.release()


if __name__ == "__main__":
    unittest.main()
