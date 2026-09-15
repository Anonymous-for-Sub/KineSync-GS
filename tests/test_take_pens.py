from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np

from kinesync.data.take_pens import (
    ContinuousSegment,
    TakePensDataset,
    continuous_segments,
)


class TakePensDatasetTest(unittest.TestCase):
    def _write_take(
        self,
        root: Path,
        name: str,
        *,
        color_dtype: np.dtype = np.dtype(np.uint8),
        timestamps: np.ndarray | None = None,
        action: np.ndarray | None = None,
    ) -> Path:
        count = 4
        timestamp_values = (
            np.array([100, 200, 300, 400], dtype=np.int64)
            if timestamps is None
            else timestamps
        )
        action_values = (
            np.arange(count * 7, dtype=np.float32).reshape(count, 7)
            if action is None
            else action
        )
        color = np.empty((count, 480, 640, 3), dtype=color_dtype)
        for index in range(count):
            color[index].fill(index + 10)
        joints = np.arange(count * 6, dtype=np.float64).reshape(count, 6) / 10.0
        gripper = np.array([-0.2, 0.0, 0.4, 1.0], dtype=np.float64)

        path = root / name
        with h5py.File(path, "w") as handle:
            handle.create_dataset("cam_head/color", data=color)
            handle.create_dataset("cam_head/timestamp", data=timestamp_values)
            handle.create_dataset("cam_wrist/color", data=color + 1)
            handle.create_dataset("cam_wrist/timestamp", data=timestamp_values + 1)
            handle.create_dataset("left_arm/action", data=action_values)
            handle.create_dataset("left_arm/gripper", data=gripper)
            handle.create_dataset("left_arm/joint", data=joints)
            handle.create_dataset("left_arm/timestamp", data=timestamp_values + 2)
            handle.create_dataset("master_left_arm/gripper", data=gripper)
            handle.create_dataset("master_left_arm/joint", data=joints)
            handle.create_dataset("master_left_arm/timestamp", data=timestamp_values + 3)
        return path

    @staticmethod
    def _replace_dataset(path: Path, name: str, value: np.ndarray | np.generic) -> None:
        with h5py.File(path, "r+") as handle:
            del handle[name]
            handle.create_dataset(name, data=value)

    def test_validates_schema_and_split_without_reading_rgb_until_requested(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_take(root, "0.hdf5")
            self._write_take(root, "f1.hdf5")
            original_getitem = h5py.Dataset.__getitem__

            def reject_color_reads(dataset, arguments):
                if dataset.name in {"/cam_head/color", "/cam_wrist/color"}:
                    raise AssertionError("Construction must not read RGB datasets")
                return original_getitem(dataset, arguments)

            with patch.object(h5py.Dataset, "__getitem__", new=reject_color_reads):
                dataset = TakePensDataset(
                    root=root,
                    development_files=("0.hdf5",),
                    test_files=("f1.hdf5",),
                )

            trajectory = dataset.trajectories("development")[0]
            self.assertEqual(trajectory.trajectory_id, "0.hdf5")
            self.assertEqual(trajectory.split, "development")
            self.assertEqual(trajectory.length, 4)
            np.testing.assert_array_equal(
                trajectory.timestamps("head"), np.array([100, 200, 300, 400])
            )
            np.testing.assert_array_equal(
                trajectory.actions(), np.arange(28, dtype=np.float32).reshape(4, 7)
            )
            np.testing.assert_allclose(
                trajectory.slave_qpos()[2],
                [1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 0.02, -0.02],
            )
            np.testing.assert_array_equal(
                trajectory.read_rgb("head", [3, 1, 3]),
                np.stack(
                    [
                        np.full((480, 640, 3), 13, dtype=np.uint8),
                        np.full((480, 640, 3), 11, dtype=np.uint8),
                        np.full((480, 640, 3), 13, dtype=np.uint8),
                    ]
                ),
            )
            np.testing.assert_array_equal(
                trajectory.read_rgb("wrist", [2]),
                np.full((1, 480, 640, 3), 13, dtype=np.uint8),
            )

    def test_rejects_file_assigned_to_the_wrong_declared_split_or_both_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_take(root, "0.hdf5")
            self._write_take(root, "f1.hdf5")

            with self.assertRaisesRegex(ValueError, "split identity"):
                TakePensDataset(root, ("f1.hdf5",), ("0.hdf5",))
            with self.assertRaisesRegex(ValueError, "both"):
                TakePensDataset(root, ("0.hdf5",), ("0.hdf5",))

    def test_rejects_missing_dataset_invalid_dtype_and_invalid_numeric_values(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            valid = self._write_take(root, "0.hdf5")
            self._write_take(root, "f1.hdf5")
            with h5py.File(valid, "r+") as handle:
                del handle["master_left_arm/gripper"]
            with self.assertRaisesRegex(ValueError, "master_left_arm/gripper"):
                TakePensDataset(root, ("0.hdf5",), ("f1.hdf5",))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_take(root, "0.hdf5", color_dtype=np.dtype(np.float32))
            self._write_take(root, "f1.hdf5")
            with self.assertRaisesRegex(ValueError, "cam_head/color"):
                TakePensDataset(root, ("0.hdf5",), ("f1.hdf5",))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            timestamps = np.array([100, 200, 200, 400], dtype=np.int64)
            self._write_take(root, "0.hdf5", timestamps=timestamps)
            self._write_take(root, "f1.hdf5")
            with self.assertRaisesRegex(ValueError, "strictly increasing"):
                TakePensDataset(root, ("0.hdf5",), ("f1.hdf5",))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            action = np.arange(28, dtype=np.float32).reshape(4, 7)
            action[1, 4] = np.nan
            self._write_take(root, "0.hdf5", action=action)
            self._write_take(root, "f1.hdf5")
            with self.assertRaisesRegex(ValueError, "finite"):
                TakePensDataset(root, ("0.hdf5",), ("f1.hdf5",))

    def test_rejects_an_unexpected_nested_dataset_from_the_closed_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            development = self._write_take(root, "0.hdf5")
            self._write_take(root, "f1.hdf5")
            with h5py.File(development, "r+") as handle:
                handle.create_dataset("unexpected/nested_value", data=np.array([1]))

            with self.assertRaisesRegex(ValueError, "exactly"):
                TakePensDataset(root, ("0.hdf5",), ("f1.hdf5",))

    def test_rejects_scalar_rank_shape_and_frame_count_schema_problems_with_value_error(self):
        malformed = (
            ("left_arm/gripper", np.float64(0.0)),
            ("left_arm/joint", np.zeros((4, 6, 1), dtype=np.float64)),
            ("left_arm/action", np.zeros((4, 8), dtype=np.float32)),
            ("master_left_arm/joint", np.zeros((3, 6), dtype=np.float64)),
        )
        for name, value in malformed:
            with self.subTest(dataset=name, shape=np.shape(value)):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    development = self._write_take(root, "0.hdf5")
                    self._write_take(root, "f1.hdf5")
                    self._replace_dataset(development, name, value)

                    with self.assertRaises(ValueError):
                        TakePensDataset(root, ("0.hdf5",), ("f1.hdf5",))

    def test_rejects_absolute_traversing_and_symlinked_hdf5_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            root = workspace / "root"
            outside = workspace / "outside"
            root.mkdir()
            outside.mkdir()
            self._write_take(root, "0.hdf5")
            self._write_take(root, "f1.hdf5")
            self._write_take(outside, "0.hdf5")

            with self.assertRaises(ValueError):
                TakePensDataset(root, ((root / "0.hdf5").resolve(),), ("f1.hdf5",))
            with self.assertRaises(ValueError):
                TakePensDataset(root, ("../outside/0.hdf5",), ("f1.hdf5",))

            (root / "0.hdf5").unlink()
            (root / "0.hdf5").symlink_to(outside / "0.hdf5")
            with self.assertRaises(ValueError):
                TakePensDataset(root, ("0.hdf5",), ("f1.hdf5",))

    def test_mutating_getter_results_cannot_change_hdf5_source_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            development = self._write_take(root, "0.hdf5")
            self._write_take(root, "f1.hdf5")
            trajectory = TakePensDataset(root, ("0.hdf5",), ("f1.hdf5",)).trajectories(
                "development"
            )[0]
            with h5py.File(development, "r") as handle:
                expected = {
                    name: np.asarray(handle[name][()])
                    for name in (
                        "cam_head/timestamp",
                        "left_arm/joint",
                        "left_arm/gripper",
                        "left_arm/action",
                        "cam_head/color",
                    )
                }

            timestamp = trajectory.timestamps("head")
            joints = trajectory.slave_joints()
            qpos = trajectory.slave_qpos()
            actions = trajectory.actions()
            rgb = trajectory.read_rgb("head", [1])
            timestamp[0] = -1
            joints[0, 0] = -1.0
            qpos[0, 0] = -1.0
            qpos[0, 6] = -1.0
            actions[0, 0] = -1.0
            rgb[0, 0, 0, 0] = 0

            with h5py.File(development, "r") as handle:
                for name, original in expected.items():
                    np.testing.assert_array_equal(handle[name][()], original)


class ContinuousSegmentsTest(unittest.TestCase):
    def test_splits_on_pause_uses_half_open_rows_and_excludes_short_segments(self):
        class SyntheticTrajectory:
            trajectory_id = "f1.hdf5"
            split = "test"

            def __init__(self):
                first = np.arange(65, dtype=np.int64) * 10_000_000
                second = first[-1] + 250_000_000 + np.arange(
                    60, dtype=np.int64
                ) * 10_000_000
                third = second[-1] + 250_000_000 + np.arange(
                    59, dtype=np.int64
                ) * 10_000_000
                self._timestamps = np.concatenate((first, second, third))

            def timestamps(self, camera: str) -> np.ndarray:
                if camera != "slave":
                    raise AssertionError("Segmentation must use slave-state timestamps")
                return self._timestamps

        segments = continuous_segments(
            SyntheticTrajectory(), pause_ns=200_000_000, minimum_frames=60
        )

        self.assertEqual(
            segments,
            (
                ContinuousSegment("f1.hdf5", "test", 0, 65),
                ContinuousSegment("f1.hdf5", "test", 65, 125),
            ),
        )
        covered_rows = [
            row for segment in segments for row in range(segment.start, segment.stop)
        ]
        self.assertEqual(covered_rows, list(range(125)))
        self.assertEqual(len(covered_rows), len(set(covered_rows)))


if __name__ == "__main__":
    unittest.main()
