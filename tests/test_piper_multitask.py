from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from kinesync.data.piper_multitask import (
    PiperMultitaskDataset,
    gaussian_qpos_to_policy_state,
    policy_state_to_gaussian_qpos,
)


class PiperMultitaskDatasetTest(unittest.TestCase):
    def _write_episode(self, root: Path, task: str, episode: int, frames: int = 4) -> None:
        directory = root / task
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{episode}.hdf5"
        timestamp = np.arange(frames, dtype=np.int64) * 100_000_000 + 1_000_000_000
        joints = np.arange(frames * 6, dtype=np.float64).reshape(frames, 6) / 100.0
        gripper = np.linspace(0.0, 0.8, frames, dtype=np.float64)
        action = np.concatenate((joints, gripper[:, None]), axis=1).astype(np.float32)
        head = np.full((frames, 8, 10, 3), 30 + episode, dtype=np.uint8)
        wrist = np.full((frames, 8, 10, 3), 80 + episode, dtype=np.uint8)
        with h5py.File(path, "w") as handle:
            handle.create_dataset("cam_head/color", data=head)
            handle.create_dataset("cam_head/timestamp", data=timestamp)
            handle.create_dataset("cam_wrist/color", data=wrist)
            handle.create_dataset("cam_wrist/timestamp", data=timestamp + 1_000_000)
            handle.create_dataset("left_arm/action", data=action)
            handle.create_dataset("left_arm/gripper", data=gripper)
            handle.create_dataset("left_arm/joint", data=joints)
            handle.create_dataset("left_arm/timestamp", data=timestamp + 2_000_000)
            handle.create_dataset("master_left_arm/gripper", data=gripper)
            handle.create_dataset("master_left_arm/joint", data=joints)
            handle.create_dataset("master_left_arm/timestamp", data=timestamp + 3_000_000)

    def test_policy_and_gaussian_state_mapping_is_explicit_and_reversible(self):
        policy = np.array(
            [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.0],
             [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]],
            dtype=np.float64,
        )

        gaussian = policy_state_to_gaussian_qpos(policy)

        self.assertEqual(gaussian.shape, (2, 8))
        np.testing.assert_allclose(gaussian[:, -2:], [[0.0, 0.0], [0.04, -0.04]])
        np.testing.assert_allclose(gaussian_qpos_to_policy_state(gaussian), policy)
        with self.assertRaisesRegex(ValueError, "gripper"):
            policy_state_to_gaussian_qpos(np.array([0.0] * 6 + [1.1]))
        with self.assertRaisesRegex(ValueError, "antisymmetric"):
            gaussian_qpos_to_policy_state(np.array([0.0] * 6 + [0.02, -0.01]))

    def test_audits_multiple_tasks_without_loading_full_rgb_and_exposes_trajectory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for task in ("corn_to_plate", "red_cube_to_plate", "red_pen_to_bucket"):
                self._write_episode(root, task, 0)
                self._write_episode(root, task, 1)

            dataset = PiperMultitaskDataset(
                root=root,
                tasks={
                    "corn_to_plate": "put corn on plate",
                    "red_cube_to_plate": "put red cube on plate",
                    "red_pen_to_bucket": "put red pen in bucket",
                },
                expected_episodes_per_task=2,
                expected_image_shape=(8, 10, 3),
            )
            receipt = dataset.audit_receipt()

            self.assertEqual(receipt["schema"], "kinesync.piper_multitask_audit.v1")
            self.assertTrue(receipt["valid"])
            self.assertEqual(receipt["task_count"], 3)
            self.assertEqual(receipt["episode_count"], 6)
            self.assertEqual(receipt["frame_count"], 24)
            self.assertEqual(receipt["state_contract"]["policy_dimension"], 7)
            self.assertEqual(receipt["state_contract"]["gaussian_dimension"], 8)
            self.assertEqual(len(receipt["files"]), 6)
            self.assertEqual(len(receipt["files"][0]["numeric_sha256"]), 64)
            self.assertEqual(len(receipt["files"][0]["rgb_sample_sha256"]), 64)

            trajectory = dataset.trajectory("red_pen_to_bucket", "1")
            self.assertEqual(trajectory.length, 4)
            self.assertEqual(trajectory.policy_state().shape, (4, 7))
            self.assertEqual(trajectory.gaussian_qpos().shape, (4, 8))
            self.assertEqual(trajectory.actions().shape, (4, 7))
            self.assertEqual(trajectory.read_rgb("head", [3, 1]).shape, (2, 8, 10, 3))

    def test_rejects_bad_timestamps_and_wrong_episode_count(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_episode(root, "corn_to_plate", 0)
            path = root / "corn_to_plate/0.hdf5"
            with h5py.File(path, "r+") as handle:
                values = handle["cam_head/timestamp"][()]
                values[2] = values[1]
                del handle["cam_head/timestamp"]
                handle.create_dataset("cam_head/timestamp", data=values)
            with self.assertRaisesRegex(ValueError, "strictly increasing"):
                PiperMultitaskDataset(
                    root=root,
                    tasks={"corn_to_plate": "put corn on plate"},
                    expected_episodes_per_task=1,
                    expected_image_shape=(8, 10, 3),
                )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_episode(root, "corn_to_plate", 0)
            with self.assertRaisesRegex(ValueError, "expected 2 episodes"):
                PiperMultitaskDataset(
                    root=root,
                    tasks={"corn_to_plate": "put corn on plate"},
                    expected_episodes_per_task=2,
                    expected_image_shape=(8, 10, 3),
                )


if __name__ == "__main__":
    unittest.main()
