import json
import tempfile
import unittest
from pathlib import Path

import cv2
import yaml

from kinesync.cli.rt0_joint_offset import execute_rt0


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT / "configs" / "rt0_piper_joint_offset.yaml"


@unittest.skipUnless(DEFAULT_CONFIG.is_file(), "RT0 configuration is unavailable")
class RT0CliIntegrationTest(unittest.TestCase):
    def test_real_route_a_assets_produce_complete_review_run(self) -> None:
        config = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
        if not Path(config["assets"]["anchor"]).is_file():
            self.skipTest("Route-A assets are not installed")
        with tempfile.TemporaryDirectory() as tmp:
            config["device"] = "cpu"
            config["runs_root"] = str(Path(tmp) / "runs")
            config["data"].update(
                {"frame_index": 120, "cameras": ["head"], "anchors_per_link": 4}
            )
            config["target"]["injected_joint_offsets_rad"] = {"joint4": -0.04}
            config["optimizer"].update({"steps": 3, "learning_rate": 0.03})
            config["visualization"] = {"video_frames": 3, "video_fps": 3}

            run_path = execute_rt0(config, run_id="cli-integration")

            metrics = json.loads((run_path / "metrics.json").read_text())
            self.assertEqual(metrics["target_provenance"], "synthetic_state")
            self.assertEqual(metrics["camera_names"], ["head"])
            self.assertEqual(metrics["anchor_count"], 36)
            self.assertIn("offset_mae_rad", metrics)
            self.assertTrue((run_path / "images" / "before_target_after.png").is_file())
            self.assertTrue((run_path / "images" / "recovery_curves.png").is_file())
            video_path = run_path / "videos" / "joint_offset_recovery.mp4"
            self.assertTrue(video_path.is_file())
            image = cv2.imread(str(run_path / "images" / "before_target_after.png"))
            self.assertEqual(image.shape[:2], (1080, 1920))
            capture = cv2.VideoCapture(str(video_path))
            self.assertTrue(capture.isOpened())
            self.assertGreaterEqual(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 3)
            capture.release()


if __name__ == "__main__":
    unittest.main()
