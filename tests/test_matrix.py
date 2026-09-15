import csv
import json
import tempfile
import unittest
from pathlib import Path

from kinesync.experiments.matrix import aggregate_rt0_runs


class RT0MatrixAggregationTest(unittest.TestCase):
    def test_aggregates_runs_and_writes_machine_and_human_summaries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = []
            for index, success in enumerate((True, True, False)):
                run = root / f"run-{index}"
                run.mkdir()
                (run / "metrics.json").write_text(
                    json.dumps(
                        {
                            "frame_index": 100 + index,
                            "camera_names": ["head", "extra"],
                            "selected_joints": ["joint2", "joint4"],
                            "offset_mae_rad": 0.001 + index * 0.002,
                            "pixel_rmse_before": 12.0,
                            "pixel_rmse_after": 0.1 + index * 0.1,
                            "tip_error_before_mm": 30.0,
                            "tip_error_after_mm": 0.2 + index * 0.1,
                            "recovery_success": success,
                        }
                    ),
                    encoding="utf-8",
                )
                runs.append(run)

            summary = aggregate_rt0_runs(runs, root / "summary")

            self.assertEqual(summary["run_count"], 3)
            self.assertAlmostEqual(summary["success_rate"], 2 / 3)
            self.assertAlmostEqual(summary["offset_mae_rad_mean"], 0.003)
            self.assertTrue((root / "summary" / "results.csv").is_file())
            self.assertTrue((root / "summary" / "summary.json").is_file())
            self.assertTrue((root / "summary" / "RESULTS.md").is_file())
            with (root / "summary" / "results.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 3)
            self.assertIn("run-0", (root / "summary" / "RESULTS.md").read_text())


if __name__ == "__main__":
    unittest.main()
