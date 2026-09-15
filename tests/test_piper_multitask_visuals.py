from __future__ import annotations

import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from kinesync.visualization.piper_multitask_visuals import (
    choose_representative_trajectory,
    export_task_evidence,
)


@dataclass(frozen=True)
class _Trajectory:
    task: str = "test_task"
    instruction: str = "move object"
    episode_id: str = "7"
    length: int = 4

    def read_rgb(self, camera, indices):
        base = 20 if camera == "head" else 90
        frames = []
        for index in indices:
            image = np.full((48, 64, 3), base, dtype=np.uint8)
            cv2.rectangle(image, (4 + int(index) * 6, 12), (18 + int(index) * 6, 36), (220, 40, 40), -1)
            frames.append(image)
        return np.stack(frames)

    def policy_state(self):
        state = np.zeros((self.length, 7), dtype=np.float64)
        state[:, 0] = np.linspace(0.0, 0.3, self.length)
        state[:, 6] = np.linspace(0.0, 0.8, self.length)
        return state


class PiperMultitaskVisualsTest(unittest.TestCase):
    def test_selects_episode_with_stronger_visible_and_state_motion(self):
        quiet = _Trajectory(episode_id="1")
        active = _Trajectory(episode_id="2", length=8)

        selected, scores = choose_representative_trajectory((quiet, active))

        self.assertEqual(selected.episode_id, "2")
        self.assertGreater(scores["2"], scores["1"])

    def test_exports_flat_native_aspect_frames_sheet_and_decodable_video(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)

            manifest = export_task_evidence(_Trajectory(), output, fps=10.0)

            self.assertEqual(manifest["schema"], "kinesync.piper_task_visuals.v1")
            self.assertEqual(manifest["task"], "test_task")
            self.assertEqual(len(manifest["single_frames"]), 6)
            self.assertEqual(len(list(output.iterdir())), 8)
            sheet = cv2.imread(str(output / manifest["contact_sheet"]))
            self.assertEqual(sheet.shape, (960, 1920, 3))
            for name in manifest["single_frames"]:
                self.assertEqual(cv2.imread(str(output / name)).shape, (960, 1280, 3))
            capture = cv2.VideoCapture(str(output / manifest["video"]))
            ok, frame = capture.read()
            frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            capture.release()
            self.assertTrue(ok)
            self.assertEqual(frame.shape, (720, 1920, 3))
            self.assertEqual(frames, 4)

    def test_refuses_to_overwrite_existing_visuals(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            export_task_evidence(_Trajectory(), output, fps=10.0)
            with self.assertRaisesRegex(FileExistsError, "D-REAL-test_task"):
                export_task_evidence(_Trajectory(), output, fps=10.0)


if __name__ == "__main__":
    unittest.main()
