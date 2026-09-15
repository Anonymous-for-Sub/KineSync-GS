"""Synthetic CPU-only trajectory aggregation tests, with no GPU operations."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/analyze_risk_atomic_optimizer_trace.py"


class OptimizerTraceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if SCRIPT.exists():
            spec = importlib.util.spec_from_file_location("optimizer_analysis", SCRIPT)
            cls.a = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cls.a)

    def setUp(self):
        self.assertTrue(SCRIPT.exists(), "trace analysis script not implemented")
        self.candidates, self.trace = [], []
        for pose in range(12):
            for condition in self.a.CONDITIONS:
                zero = condition.endswith("_zero")
                stale = condition.startswith("one_view")
                measured = 0. if zero else .05
                candidate = .12 if stale else 0.
                state = f"pose_{pose:02d}__{condition}"
                self.candidates.append(dict(state_id=state, case_id=f"pose_{pose:02d}", condition_id=condition,
                    true_qpos_rad=json.dumps([0.] * 7), measured_qpos_rad=json.dumps([measured] * 7),
                    candidate_qpos_rad=json.dumps([candidate] * 7)))
                for step in range(61):
                    q = measured * (1-step/60) + candidate * step/60
                    if step == 60:
                        q = candidate
                    self.trace.append(dict(state_id=state, case_id=f"pose_{pose:02d}", condition_id=condition,
                        split="heldout", target_provenance="analysis_same_gs", optimizer_source="fused_candidate",
                        step=str(step), loss=str((pose+1)*(.002-.001*step/60)), gradient_norm="DO NOT READ",
                        **{f"qpos_joint{j}": str(q) for j in range(1,8)}))

    def test_full_matrix_and_reference_only_qmae(self):
        rows = self.a.trace_rows(self.trace, self.candidates)
        self.assertEqual(len(rows), 2928)
        row = next(r for r in rows if r["condition_id"] == "one_view_stale_zero" and r["step"] == 60)
        self.assertAlmostEqual(row["qmae_deg"], np.rad2deg(.12))
        self.assertNotIn("gradient_norm", row)

    def test_duplicate_or_missing_steps_rejected(self):
        with self.assertRaisesRegex(ValueError, "unique|duplicate|61|2928"):
            self.a.trace_rows(self.trace + [self.trace[0]], self.candidates)
        with self.assertRaisesRegex(ValueError, "61|2928|matrix"):
            self.a.trace_rows(self.trace[:-1], self.candidates)

    def test_nonfused_camera_traces_not_aggregated(self):
        extras = [dict(r, optimizer_source="view_a_evidence_recovery", loss="9999") for r in self.trace]
        self.assertEqual(self.a.trace_rows(self.trace, self.candidates),
                         self.a.trace_rows(self.trace + extras, self.candidates))

    def test_terminal_candidate_and_initial_measurement_must_match(self):
        changed = copy.deepcopy(self.trace)
        changed[0]["qpos_joint1"] = "999"
        with self.assertRaisesRegex(ValueError, "initial|measurement"):
            self.a.trace_rows(changed, self.candidates)
        changed = copy.deepcopy(self.trace)
        changed[60]["qpos_joint1"] = "999"
        with self.assertRaisesRegex(ValueError, "terminal|candidate"):
            self.a.trace_rows(changed, self.candidates)

    def test_nonfinite_loss_or_state_rejected(self):
        changed = copy.deepcopy(self.trace)
        changed[2]["loss"] = "nan"
        with self.assertRaisesRegex(ValueError, "finite"):
            self.a.trace_rows(changed, self.candidates)
        changed[2]["loss"] = ".001"
        changed[2]["qpos_joint1"] = "inf"
        with self.assertRaisesRegex(ValueError, "finite"):
            self.a.trace_rows(changed, self.candidates)

    def test_aggregate_uses_all_12_poses_and_paired_bootstrap(self):
        rows = self.a.trace_rows(self.trace, self.candidates)
        summary = self.a.aggregate(rows)
        self.assertEqual(len(summary), 244)
        first = summary[0]
        self.assertEqual(first["pose_count"], 12)
        self.assertAlmostEqual(first["loss_mean"], .002 * 6.5)
        draws = np.random.default_rng(20260908).integers(0, 12, (10000, 12))
        expected = np.quantile((.002*np.arange(1,13))[draws].mean(1), [.025,.975])
        np.testing.assert_allclose([first["loss_ci_low"], first["loss_ci_high"]], expected)

    def test_loss_down_qmae_up_is_reported_without_ours_trace(self):
        rows = self.a.trace_rows(self.trace, self.candidates)
        summary = self.a.aggregate(rows)
        endpoints = self.a.endpoint_summary(summary)
        stale = next(r for r in endpoints if r["condition_id"] == "one_view_stale_nonzero")
        self.assertLess(stale["loss_final"], stale["loss_initial"])
        self.assertGreater(stale["qmae_final_deg"], stale["qmae_initial_deg"])
        self.assertAlmostEqual(stale["loss_relative_reduction_percent"], 50.)
        self.assertTrue(all(r["optimizer_source"] == "fused_candidate" for r in rows))

    def test_figures_square_and_same_stem_csv(self):
        summary = self.a.aggregate(self.a.trace_rows(self.trace, self.candidates))
        with tempfile.TemporaryDirectory() as tmp:
            figures = self.a.plot(summary, Path(tmp))
            self.assertEqual(len(figures), 2)
            from PIL import Image
            for figure in figures:
                path = Path(tmp) / (figure["stem"] + ".png")
                with Image.open(path) as image:
                    self.assertEqual(image.size, (2100,2100))
                self.assertTrue(path.with_suffix(".pdf").is_file())
                self.assertTrue(path.with_suffix(".csv").is_file())


if __name__ == "__main__":
    unittest.main()
