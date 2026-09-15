from __future__ import annotations

import json
import ast
import tempfile
import unittest
from pathlib import Path

import numpy as np

from kinesync.models.openvla_oft import (
    audit_openvla_oft_checkpoint,
    select_high_motion_indices,
    validate_action_chunk,
)


class OpenVlaOftCheckpointTest(unittest.TestCase):
    def test_selects_spaced_high_motion_frames_against_hold_baseline(self):
        state = np.zeros((12, 7), dtype=np.float64)
        action = np.zeros((12, 7), dtype=np.float64)
        action[2] = 0.4
        action[3] = 0.5
        action[8] = 0.3

        selected = select_high_motion_indices(
            state, action, count=2, minimum_separation=3
        )

        self.assertEqual(selected.tolist(), [3, 8])

    def test_offline_acceptance_script_has_no_hardware_or_network_imports(self):
        script = Path(__file__).parents[1] / "scripts/run_openvla_oft_offline_acceptance.py"
        tree = ast.parse(script.read_text(encoding="utf-8"))
        roots = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                roots.add(node.module.split(".", 1)[0])
        self.assertTrue({"h5py", "numpy", "torch"} <= roots)
        self.assertTrue(
            roots.isdisjoint(
                {"piper_sdk", "pyrealsense2", "can", "requests", "socket", "serial"}
            )
        )

    def _checkpoint(self, root: Path) -> Path:
        root.mkdir(parents=True, exist_ok=True)
        required = {
            "config.json": "{}",
            "generation_config.json": "{}",
            "preprocessor_config.json": "{}",
            "processor_config.json": "{}",
            "tokenizer.json": "{}",
            "tokenizer.model": "tokenizer",
            "tokenizer_config.json": "{}",
            "special_tokens_map.json": "{}",
            "added_tokens.json": "{}",
            "configuration_prismatic.py": "# config\n",
            "modeling_prismatic.py": "# model\n",
            "processing_prismatic.py": "# processor\n",
            "action_head--30000_checkpoint.pt": "action",
            "proprio_projector--30000_checkpoint.pt": "proprio",
            "model-00001-of-00002.safetensors": "shard-a",
            "model-00002-of-00002.safetensors": "shard-b",
        }
        for name, value in required.items():
            (root / name).write_text(value, encoding="ascii")
        (root / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "metadata": {"total_size": 14},
                    "weight_map": {
                        "layer.a": "model-00001-of-00002.safetensors",
                        "layer.b": "model-00002-of-00002.safetensors",
                    },
                }
            ),
            encoding="ascii",
        )
        stats = {}
        for channel in ("action", "proprio"):
            stats[channel] = {
                "mean": [0.0] * 7,
                "std": [1.0] * 7,
                "min": [-1.0] * 7,
                "max": [1.0] * 7,
                "q01": [-0.8] * 7,
                "q99": [0.8] * 7,
            }
        stats["action"]["mask"] = [True] * 7
        (root / "dataset_statistics.json").write_text(
            json.dumps(
                {
                    "piper_d455_oft": {
                        **stats,
                        "num_transitions": 22562,
                        "num_trajectories": 120,
                    }
                }
            ),
            encoding="ascii",
        )
        return root

    def test_audits_complete_checkpoint_and_7d_statistics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self._checkpoint(Path(directory) / "checkpoint")

            receipt = audit_openvla_oft_checkpoint(root, "piper_d455_oft")

            self.assertTrue(receipt["valid"])
            self.assertEqual(receipt["schema"], "kinesync.openvla_oft_checkpoint_audit.v1")
            self.assertEqual(receipt["action_dimension"], 7)
            self.assertEqual(receipt["proprio_dimension"], 7)
            self.assertEqual(receipt["num_trajectories"], 120)
            self.assertEqual(receipt["num_transitions"], 22562)
            self.assertEqual(receipt["model_shard_count"], 2)
            self.assertEqual(len(receipt["files"]), 18)

    def test_action_contract_accepts_chunks_and_reports_clipping_without_execution(self):
        statistics = {
            "min": [-1.0] * 7,
            "max": [1.0] * 7,
            "q01": [-0.5] * 7,
            "q99": [0.5] * 7,
        }
        chunk = np.array([[0.0] * 7, [0.8] * 7], dtype=np.float64)

        report = validate_action_chunk(chunk, statistics)

        self.assertEqual(report["shape"], [2, 7])
        self.assertEqual(report["clipped_value_count"], 7)
        self.assertEqual(report["command_mode"], "disabled")
        np.testing.assert_allclose(report["clipped_actions"][1], [0.5] * 7)
        bounds_report = validate_action_chunk(chunk, statistics, envelope="min_max")
        self.assertEqual(bounds_report["clipped_value_count"], 0)
        self.assertEqual(bounds_report["envelope"], "min_max")
        with self.assertRaisesRegex(ValueError, "7"):
            validate_action_chunk(np.zeros((2, 8)), statistics)
        with self.assertRaisesRegex(ValueError, "finite"):
            validate_action_chunk(np.array([0.0] * 6 + [np.nan]), statistics)

    def test_rejects_missing_shard_and_non_7d_statistics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self._checkpoint(Path(directory) / "checkpoint")
            (root / "model-00002-of-00002.safetensors").unlink()
            with self.assertRaisesRegex(FileNotFoundError, "model-00002"):
                audit_openvla_oft_checkpoint(root, "piper_d455_oft")

        with tempfile.TemporaryDirectory() as directory:
            root = self._checkpoint(Path(directory) / "checkpoint")
            path = root / "dataset_statistics.json"
            payload = json.loads(path.read_text(encoding="ascii"))
            payload["piper_d455_oft"]["action"]["mean"] = [0.0] * 6
            path.write_text(json.dumps(payload), encoding="ascii")
            with self.assertRaisesRegex(ValueError, "7D"):
                audit_openvla_oft_checkpoint(root, "piper_d455_oft")


if __name__ == "__main__":
    unittest.main()
