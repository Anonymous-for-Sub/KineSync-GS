from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))

from summarize_abc_diagnostic import IncompleteMatrixError, summarize_matrix  # noqa: E402


SCENE_XML = """<mujoco model="diagnostic">
  <worldbody>
    <body name="robot"><joint name="robot_joint" type="hinge" /></body>
    <body name="bin"><freejoint name="bin_joint" /></body>
    <body name="bottle_1"><freejoint name="bottle_1_joint" /></body>
    <body name="bottle_2"><freejoint name="bottle_2_joint" /></body>
  </worldbody>
</mujoco>
"""


def _case(case_id: str, chunks: int, seed: int = 42000) -> dict[str, object]:
    return {
        "id": case_id,
        "cell": 0,
        "physics": "vanilla",
        "chunks": chunks,
        "seed": seed,
        "policy_seed": 123,
    }


def _write_matrix(root: Path, cases: list[dict[str, object]]) -> None:
    (root / "matrix.json").write_text(
        json.dumps({"policy_rng_scope": "one_independent_episode_per_process", "cases": cases}),
        encoding="utf-8",
    )


def _trace(*, moved: bool) -> np.ndarray:
    qpos = np.zeros((3, 22), dtype=np.float64)
    qpos[:, 1:4] = [0.0, 0.0, 0.0]  # bin free joint
    qpos[:, 8:11] = [0.40, 0.0, 0.0]
    qpos[:, 15:18] = [0.30, 0.0, -0.10]
    if moved:
        qpos[1:, 8:11] = [0.10, 0.0, 0.0]
    return qpos


def _write_complete_case(root: Path, case: dict[str, object], *, action_offset: float = 0.0) -> None:
    run = root / str(case["id"])
    run.mkdir()
    (run / "completion.json").write_text(json.dumps({"status": "complete"}), encoding="utf-8")
    (run / "config.yaml").write_text(
        "scene:\n  eval_bin_radius: 0.155\n  eval_min_rel_z: -0.06\n  eval_max_rel_z: 0.26\n",
        encoding="utf-8",
    )
    (run / "seed_42000__scene.xml").write_text(SCENE_XML, encoding="utf-8")
    final_eval = {
        "success": False,
        "ever_success": False,
        "num_bottles_in_bin": 1,
        "num_active_bottles": 2,
        "max_bottles_in_bin_so_far": 1,
    }
    (run / "summary.json").write_text(
        json.dumps(
            {
                "num_success": 0,
                "num_worlds": 1,
                "worlds": [
                    {
                        "world_seed": 42000,
                        "success": False,
                        "wall_s": 2.5,
                        "chunk_metrics": [{"chunk": 0, "wall_s": 0.9}],
                        "final_task_eval": final_eval,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    qpos = _trace(moved=True)
    np.savez_compressed(
        run / "seed_42000__physical_trace.npz",
        qpos=qpos,
        time_seconds=np.array([0.0, 0.1, 0.2]),
    )
    np.savez_compressed(
        run / "policy_inference.npz",
        state=np.zeros((2, 14)),
        noise=np.ones((2, 3, 14)),
        noise_present=np.array([True, True]),
        actions=np.full((2, 3, 14), action_offset),
        policy_seed=np.array(123),
        provenance=np.array("actual_policy_inference_calls_not_a_reconstructed_rng_sequence"),
    )


class AbcDiagnosticSummaryTest(unittest.TestCase):
    def test_allow_incomplete_keeps_missing_metrics_blank(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_matrix(root, [_case("vanilla_h120_s42000", 120)])
            report = root / "report.md"

            with self.assertRaises(IncompleteMatrixError):
                summarize_matrix(root, report_path=report)

            payload = summarize_matrix(root, report_path=report, allow_incomplete=True)

            self.assertEqual(payload["status_counts"], {"pending": 1})
            row = payload["cases"][0]
            self.assertEqual(row["status"], "pending")
            self.assertIsNone(row["native_success"])
            self.assertIsNone(row["duration_seconds"])
            self.assertIn("pending", report.read_text(encoding="utf-8"))
            with (root / "abc_diagnostic_summary.csv").open(newline="", encoding="utf-8") as handle:
                csv_row = next(csv.DictReader(handle))
            self.assertEqual(csv_row["native_success"], "")

    def test_completed_pair_reports_prefix_differences_and_tail_diagnostics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            short = _case("vanilla_h120_s42000", 120)
            long = _case("vanilla_h240_s42000", 240)
            _write_matrix(root, [short, long])
            _write_complete_case(root, short, action_offset=0.0)
            _write_complete_case(root, long, action_offset=0.25)

            payload = summarize_matrix(root, report_path=root / "report.md")

            self.assertEqual(payload["status_counts"], {"complete": 2})
            row = payload["cases"][0]
            self.assertFalse(row["native_success"])
            self.assertEqual(row["native_num_bottles_in_bin"], 1)
            self.assertAlmostEqual(row["duration_seconds"], 0.2)
            bottles = row["tail_diagnostics"]["bottles"]
            self.assertTrue(bottles[0]["in_bin_by_native_threshold"])
            self.assertFalse(bottles[1]["in_bin_by_native_threshold"])
            self.assertTrue(bottles[1]["below_lower_height_bound"])
            self.assertEqual(bottles[1]["grasp_inference"], "likely_not_grabbed_no_motion")

            comparison = payload["same_seed_horizon_checks"][0]
            self.assertTrue(comparison["noise_exact"])
            self.assertFalse(comparison["actions_prefix_exact"])
            self.assertGreater(comparison["actions_prefix_max_abs_diff"], 0.0)
            self.assertTrue(comparison["qpos_prefix_exact"])
            self.assertTrue(comparison["initial_randomization_qpos_exact"])
            self.assertTrue(comparison["initial_randomization_scene_exact"])

    def test_failed_case_is_never_reported_as_zero_result(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case = _case("vanilla_h120_s42000", 120)
            _write_matrix(root, [case])
            run = root / str(case["id"])
            run.mkdir()
            (run / "completion.json").write_text(
                json.dumps({"status": "failed", "traceback": "intentional failure"}),
                encoding="utf-8",
            )

            payload = summarize_matrix(root, report_path=root / "report.md", allow_incomplete=True)

            row = payload["cases"][0]
            self.assertEqual(row["status"], "failed")
            self.assertIsNone(row["native_num_bottles_in_bin"])
            self.assertIn("intentional failure", row["status_detail"])

    def test_complete_receipt_requires_actual_policy_state_array(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case = _case("vanilla_h120_s42000", 120)
            _write_matrix(root, [case])
            _write_complete_case(root, case)
            np.savez_compressed(
                root / str(case["id"]) / "policy_inference.npz",
                noise=np.ones((2, 3, 14)),
                noise_present=np.array([True, True]),
                actions=np.zeros((2, 3, 14)),
            )

            payload = summarize_matrix(root, report_path=root / "report.md", allow_incomplete=True)

            row = payload["cases"][0]
            self.assertEqual(row["status"], "incomplete_artifacts")
            self.assertIn("state", row["status_detail"])
            self.assertIsNone(row["native_success"])


if __name__ == "__main__":
    unittest.main()
