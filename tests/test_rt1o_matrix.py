import json
import tempfile
import unittest
from pathlib import Path

from kinesync.cli.rt1_real_matrix import (
    aggregate_rt1o_runs,
    build_cartesian_trials,
)


class RT1OMatrixTest(unittest.TestCase):
    def test_cartesian_trials_are_deterministic_unique_and_complete(self) -> None:
        grid = {
            "states": ["state_0300", "state_0310"],
            "camera_modes": [["head"], ["head", "extra"]],
            "offset_cases": [
                {"name": "proximal_1deg", "offsets_rad": {"joint1": 0.0174533}},
                {"name": "distal_5deg", "offsets_rad": {"joint5": -0.0872665}},
            ],
            "seeds": [11, 23],
        }
        first = build_cartesian_trials(grid)
        second = build_cartesian_trials(grid)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 16)
        self.assertEqual(len({case["run_id"] for case in first}), 16)
        self.assertEqual(first[0]["state_id"], "state_0300")
        self.assertEqual(first[0]["cameras"], ["head"])
        self.assertEqual(first[0]["seed"], 11)
        self.assertEqual(first[-1]["offset_name"], "distal_5deg")

    def _write_run(
        self,
        root: Path,
        *,
        index: int,
        camera_names: list[str],
        joint: str,
        success: bool,
        reduction: float,
        trial_type: str = "controlled_offset",
        natural_correction: float | None = None,
    ) -> Path:
        run = root / f"run-{index}"
        (run / "images").mkdir(parents=True)
        (run / "videos").mkdir()
        metrics = {
            "state_id": f"state_{300 + index:04d}",
            "camera_names": camera_names,
            "selected_joints": [joint],
            "injected_measurement_offsets_rad": {joint: 0.0174533 * (index + 1)},
            "offset_mae_rad": 0.002 + index * 0.001,
            "state_error_reduction": reduction,
            "mean_mask_iou_before": 0.42 + index * 0.01,
            "mean_mask_iou_after": 0.48 + index * 0.01,
            "mean_boundary_f1_before": 0.31 + index * 0.01,
            "mean_boundary_f1_after": 0.39 + index * 0.01,
            "recovery_success": success,
            "trial_type": trial_type,
            "natural_state_correction_rad": natural_correction,
            "seed": 11 + index,
        }
        (run / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
        (run / "manifest.json").write_text("{}\n", encoding="utf-8")
        (run / "config.yaml").write_text("seed: 1\n", encoding="utf-8")
        (run / "trace.csv").write_text("step,loss\n0,1.0\n", encoding="utf-8")
        (run / "images" / "real_observation_before_after.png").write_bytes(b"png")
        (run / "images" / "real_observation_recovery_curves.png").write_bytes(b"png")
        (run / "videos" / "real_observation_recovery.mp4").write_bytes(b"mp4")
        return run

    def test_aggregate_reports_bootstrap_ci_and_subgroup_macro_averages(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runs = [
                self._write_run(
                    root,
                    index=index,
                    camera_names=["head"] if index < 2 else ["head", "extra"],
                    joint="joint1" if index % 2 == 0 else "joint5",
                    success=index != 3,
                    reduction=(0.82, 0.71, 0.66, 0.38)[index],
                )
                for index in range(4)
            ]
            summary = aggregate_rt1o_runs(
                runs, root / "summary", bootstrap_seed=37, bootstrap_samples=500
            )
            self.assertEqual(summary["run_count"], 4)
            self.assertEqual(summary["success_count"], 3)
            self.assertAlmostEqual(summary["success_rate"], 0.75)
            self.assertLessEqual(
                summary["success_rate_ci95"][0], summary["success_rate"]
            )
            self.assertGreaterEqual(
                summary["success_rate_ci95"][1], summary["success_rate"]
            )
            self.assertIn("head", summary["by_camera_mode"])
            self.assertIn("head+extra", summary["by_camera_mode"])
            self.assertIn("joint1", summary["by_joint_group"])
            self.assertIn("results.csv", {path.name for path in (root / "summary").iterdir()})
            self.assertTrue((root / "summary" / "RESULTS.md").is_file())

    def test_aggregate_rejects_incomplete_child_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = self._write_run(
                root,
                index=0,
                camera_names=["head"],
                joint="joint2",
                success=True,
                reduction=0.8,
            )
            (run / "trace.csv").unlink()
            with self.assertRaisesRegex(ValueError, "Incomplete RT1-O run"):
                aggregate_rt1o_runs([run], root / "summary")

    def test_zero_controls_are_reported_separately_from_recovery_reduction(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            controlled = [
                self._write_run(
                    root,
                    index=index,
                    camera_names=["head"],
                    joint="joint2",
                    success=True,
                    reduction=reduction,
                )
                for index, reduction in enumerate((0.7, 0.9))
            ]
            zero = self._write_run(
                root,
                index=2,
                camera_names=["head", "extra"],
                joint="joint2",
                success=True,
                reduction=0.0,
                trial_type="zero_injection_control",
                natural_correction=0.004,
            )
            summary = aggregate_rt1o_runs(controlled + [zero], root / "summary")
            self.assertEqual(summary["controlled_run_count"], 2)
            self.assertEqual(summary["zero_control_count"], 1)
            self.assertAlmostEqual(summary["state_error_reduction_median"], 0.8)
            self.assertAlmostEqual(summary["zero_control_correction_rad_median"], 0.004)


if __name__ == "__main__":
    unittest.main()
