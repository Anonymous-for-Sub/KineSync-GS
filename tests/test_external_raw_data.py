from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from kinesync.external_data.acquire import droid_raw_object_selection
from kinesync.external_data.audit import audit_droid_episode, sha256_file
from kinesync.external_data.droid import load_droid_hdf5_trajectory


class ExternalRawDataAuditTest(unittest.TestCase):
    def _write_droid_episode(self, root: Path) -> Path:
        episode = root / "RAIL/success/2023-04-05/Wed_Apr__5_15:08:50_2023"
        episode.mkdir(parents=True)
        (episode / "metadata_episode-123.json").write_text(
            json.dumps({"episode_id": "episode-123", "source": "synthetic-test"}),
            encoding="utf-8",
        )
        with h5py.File(episode / "trajectory.h5", "w") as handle:
            handle.attrs["robot_serial_number"] = "franka-test"
            handle.create_dataset("action/joint_position", data=np.zeros((3, 7)))
            handle.create_dataset("observation/joint_positions", data=np.ones((3, 7)))
            handle.create_dataset("observation/timestamp/control/step_start", data=[1000, 1100, 1200])
            handle.create_dataset("observation/timestamp/cameras/111_frame_received", data=[1002, 1102, 1202])
            handle.create_dataset("observation/camera_type/111", data=[0])
            handle.create_dataset("observation/camera_type/222", data=[1])
            handle.create_dataset("observation/camera_extrinsics/111_left", data=np.zeros((3, 6)))
            handle.create_dataset("observation/camera_extrinsics/222_left", data=np.ones((3, 6)))
        return episode

    def _write_calibration(self, root: Path) -> Path:
        root.mkdir()
        (root / "intrinsics.json").write_text(
            json.dumps(
                {"episode-123": {"111": {"cameraMatrix": [100.0, 10.0, 101.0, 11.0], "width": 20, "height": 10}}}
            ),
            encoding="utf-8",
        )
        (root / "cam2base_extrinsic_superset.json").write_text(
            json.dumps({"episode-123": {"111": [0.1, 0.2, 0.3, 0.0, 0.0, 0.0]}}), encoding="utf-8"
        )
        return root

    def test_droid_audit_reports_native_schema_and_supplementary_calibration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            receipt = audit_droid_episode(
                self._write_droid_episode(root),
                calibration_root=self._write_calibration(root / "calibration"),
            )

            self.assertEqual(receipt["schema"], "kinesync.external_raw_droid_audit.v1")
            self.assertEqual(receipt["episode_id"], "episode-123")
            self.assertEqual(receipt["joint_dimensions"], {"action/joint_position": 7, "observation/joint_positions": 7})
            self.assertTrue(receipt["timestamps"]["observation/timestamp/control/step_start"]["strictly_increasing"])
            self.assertEqual(receipt["hdf5_camera_extrinsics"]["111_left"]["dimension"], 6)
            self.assertEqual(receipt["supplementary_calibration"]["intrinsics"]["111"]["K"], [[100.0, 0.0, 10.0], [0.0, 101.0, 11.0], [0.0, 0.0, 1.0]])
            self.assertEqual(receipt["supplementary_calibration"]["camera_to_base"]["111"]["dimension"], 6)
            self.assertEqual(receipt["native_video"], [])
            self.assertEqual(len(receipt["files"]["trajectory.h5"]["sha256"]), 64)

    def test_file_sha256_is_stable_for_raw_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw.bin"
            path.write_bytes(b"raw-data")
            self.assertEqual(
                sha256_file(path),
                "577b96e7309b761c80704b407c637155a0424b43b5ebf45de348d6e81093433b",
            )

    def test_droid_selection_keeps_native_mp4_and_sidecars_without_stereo_or_svo(self):
        objects = [
            "episode/trajectory.h5",
            "episode/metadata_abc.json",
            "episode/recordings/MP4/111.mp4",
            "episode/recordings/MP4/111_timestamps.json",
            "episode/recordings/MP4/111-stereo.mp4",
            "episode/recordings/SVO/111.svo",
            "episode/unrelated.txt",
        ]

        self.assertEqual(
            droid_raw_object_selection(objects),
            [
                "episode/metadata_abc.json",
                "episode/recordings/MP4/111.mp4",
                "episode/recordings/MP4/111_timestamps.json",
                "episode/trajectory.h5",
            ],
        )
        self.assertEqual(
            droid_raw_object_selection(objects, include_svo_serials={"111"}),
            [
                "episode/metadata_abc.json",
                "episode/recordings/MP4/111.mp4",
                "episode/recordings/MP4/111_timestamps.json",
                "episode/recordings/SVO/111.svo",
                "episode/trajectory.h5",
            ],
        )

    def test_droid_adapter_combines_robot_clock_and_keeps_state_camera_streams_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            episode = self._write_droid_episode(Path(directory))
            with h5py.File(episode / "trajectory.h5", "a") as handle:
                handle.create_dataset(
                    "observation/robot_state/joint_positions",
                    data=np.array([[0.0] * 7, [0.1] * 7, [0.2] * 7]),
                )
                handle.create_dataset("observation/robot_state/gripper_position", data=[0.0, 0.4, 0.8])
                handle.create_dataset("observation/timestamp/robot_state/robot_timestamp_seconds", data=[7, 7, 8])
                handle.create_dataset("observation/timestamp/robot_state/robot_timestamp_nanos", data=[900_000_000, 966_000_000, 32_000_000])
                handle.create_dataset(
                    "observation/timestamp/cameras/111_estimated_capture", data=[950, 1_016, 1_082]
                )

            trajectory = load_droid_hdf5_trajectory(episode)

            self.assertEqual(trajectory.joint_positions.shape, (3, 7))
            np.testing.assert_allclose(trajectory.gripper_position, [0.0, 0.4, 0.8])
            np.testing.assert_array_equal(
                trajectory.robot_timestamp_ns,
                [7_900_000_000, 7_966_000_000, 8_032_000_000],
            )
            self.assertTrue(trajectory.robot_timestamp_strictly_increasing)
            np.testing.assert_array_equal(trajectory.camera_timestamps_ms["111"]["frame_received"], [1002, 1102, 1202])
            np.testing.assert_array_equal(trajectory.camera_timestamps_ms["111"]["estimated_capture"], [950, 1016, 1082])
            self.assertNotIn("video_index", trajectory.__dict__)


if __name__ == "__main__":
    unittest.main()
