from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from kinesync.data.paired_real_replay import PairedRealGaussianDataset


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PairedRealReplayDatasetTest(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Path, Path]:
        real_path = root / "real.png"
        gaussian_path = root / "gaussian.png"
        self.assertTrue(
            cv2.imwrite(
                str(real_path), np.full((24, 32, 3), (20, 80, 160), np.uint8)
            )
        )
        self.assertTrue(
            cv2.imwrite(
                str(gaussian_path),
                np.full((24, 32, 3), (25, 75, 150), np.uint8),
            )
        )
        pair_id = "site:task:001:0007"
        selection_path = root / "selection.json"
        selection_path.write_text(
            json.dumps(
                {
                    "schema": "fedsplat.piper_canonical_selection.v1",
                    "pairs": [
                        {
                            "source_pair_id": pair_id,
                            "split": "calibration",
                            "episode_id": "1",
                            "pair_index": 2,
                            "frame_index": 7,
                            "timestamp_ns": 123,
                            "state": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.4],
                            "canonical_head": real_path.name,
                            "canonical_head_sha256": _sha256(real_path),
                        }
                    ],
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        render_path = root / "render.json"
        render_path.write_text(
            json.dumps(
                {
                    "schema": "robosplat.piper_render_export.v1",
                    "robot_representation": "articulated_component_gs",
                    "processed_splits": ["calibration"],
                    "historical_renderer_validation_exposure": False,
                    "source_selection_manifest_sha256": _sha256(selection_path),
                    "rows": [
                        {
                            "source_pair_id": pair_id,
                            "split": "calibration",
                            "episode_id": "1",
                            "frame_index": 7,
                            "gaussian_head": gaussian_path.name,
                            "gaussian_head_sha256": _sha256(gaussian_path),
                            "qpos_render": [
                                0.1,
                                0.2,
                                0.3,
                                0.4,
                                0.5,
                                0.6,
                                0.02,
                                -0.02,
                            ],
                        }
                    ],
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return selection_path, render_path

    def test_adapter_loads_hash_verified_real_gaussian_pair_and_8d_qpos(self):
        with tempfile.TemporaryDirectory() as directory:
            selection, render = self._fixture(Path(directory))
            dataset = PairedRealGaussianDataset(
                selection_manifest=selection,
                render_manifests={"a_p2": [render]},
            )

            episode = dataset.load_episode(
                split="calibration", episode_id="1", backend="a_p2"
            )

            self.assertEqual(len(episode), 1)
            frame = episode[0]
            self.assertEqual(frame.source_pair_id, "site:task:001:0007")
            np.testing.assert_allclose(
                frame.qpos,
                [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.02, -0.02],
            )
            self.assertEqual(frame.real_rgb.shape, (24, 32, 3))
            self.assertEqual(frame.gaussian_rgb.shape, (24, 32, 3))
            self.assertEqual(dataset.episode_ids(split="calibration"), ("1",))
            self.assertEqual(dataset.backends, ("a_p2",))

    def test_adapter_rejects_mutated_media_at_load_time(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            selection, render = self._fixture(root)
            dataset = PairedRealGaussianDataset(
                selection_manifest=selection,
                render_manifests={"a_p2": [render]},
            )
            (root / "real.png").write_bytes(b"mutated")

            with self.assertRaisesRegex(ValueError, "SHA-256"):
                dataset.load_episode(
                    split="calibration", episode_id="1", backend="a_p2"
                )

    def test_adapter_rejects_render_qpos_that_breaks_exact_pair_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            selection, render = self._fixture(root)
            payload = json.loads(render.read_text(encoding="utf-8"))
            payload["rows"][0]["qpos_render"][1] += 0.1
            render.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "qpos"):
                PairedRealGaussianDataset(
                    selection_manifest=selection,
                    render_manifests={"a_p2": [render]},
                )


if __name__ == "__main__":
    unittest.main()
