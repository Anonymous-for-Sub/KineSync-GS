import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_external_gs_guard as runner


class ExternalGuardRunnerContractTest(unittest.TestCase):
    def test_applied_update_is_distinct_from_gate_acceptance(self):
        measured = torch.zeros(7)
        for method, global_decision, component_decisions in (
            ("gaussian_inverse_unguarded", None, None),
            ("global_guard", True, None),
            ("component_guard", None, {f"joint{i}": True for i in range(1, 8)}),
        ):
            with self.subTest(method=method):
                result, joints = runner._evaluation_rows(
                    split="development", state_id="a__zero", case_id="a",
                    condition_id="clean_zero", method=method, measured=measured,
                    candidate=measured, final=measured, true_q=measured,
                    global_guard_accepted=global_decision,
                    component_guard_accepted=component_decisions,
                    latency_ms=0.0, frame_paths={},
                )
                self.assertEqual(result["update_applied_joint_count"], 0)
                self.assertTrue(all(not row["update_applied"] for row in joints))
                summary = runner._aggregate_rows(joints)[0]
                self.assertEqual(summary["update_applied_count"], 0)
                self.assertEqual(summary["update_applied_coverage"], 0.0)
                if method == "component_guard":
                    self.assertEqual(summary["component_guard_accept_count"], 7)

    def _provenance_fixture(self, root):
        config = {
            "development_cases": [{"id": "a"}],
            "heldout_cases": [{"id": "b"}],
            "conditions": [{"id": "clean"}],
        }
        source = root / "config.yaml"
        source.write_text(yaml.safe_dump(config))
        development = root / "development"
        development.mkdir()
        (development / "config_source.yaml").write_bytes(source.read_bytes())
        sha = hashlib.sha256(source.read_bytes()).hexdigest()
        (development / "manifest.json").write_text(json.dumps({
            "split": "development", "target_provenance": "analysis_same_gs",
            "config_source_sha256": sha,
        }))
        (development / "heldout_manifest.json").write_text(json.dumps({
            "config_sha256": sha, "case_ids": ["b"], "condition_ids": ["clean"],
        }))
        kwargs = {
            "config_path": source, "development_root": development,
            "calibration": SimpleNamespace(source_state_ids=("a__clean",)),
            "profile": SimpleNamespace(calibration_state_ids=("a__clean",)),
        }
        return config, kwargs

    def test_heldout_accepts_matching_development_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            config, kwargs = self._provenance_fixture(Path(directory))
            runner._validate_heldout_calibration_provenance(config, **kwargs)

    def test_heldout_rejects_foreign_calibration_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            config, kwargs = self._provenance_fixture(Path(directory))
            kwargs["calibration"] = SimpleNamespace(source_state_ids=("foreign__clean",))
            with self.assertRaisesRegex(ValueError, "development state IDs"):
                runner._validate_heldout_calibration_provenance(config, **kwargs)

    def test_heldout_rejects_config_change_and_split_overlap(self):
        with tempfile.TemporaryDirectory() as directory:
            config, kwargs = self._provenance_fixture(Path(directory))
            kwargs["config_path"].write_text("changed: true\n")
            with self.assertRaisesRegex(ValueError, "config source hash"):
                runner._validate_heldout_calibration_provenance(config, **kwargs)
            config["heldout_cases"] = [{"id": "a"}]
            with self.assertRaisesRegex(ValueError, "overlap"):
                runner._validate_heldout_calibration_provenance(config, **kwargs)

    def test_candidate_evidence_is_durable_before_calibration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            receipt = runner._persist_candidate_diagnostics(
                root, candidate_rows=[{"state_id": "a", "loss": 0.1}],
                trace_rows=[{"state_id": "a", "step": 0, "loss": 0.1}],
            )
            self.assertEqual(receipt["candidate_row_count"], 1)
            self.assertEqual(receipt["candidate_evidence_sha256"], hashlib.sha256(
                (root / "candidate_evidence.csv").read_bytes()).hexdigest())
            self.assertEqual(json.loads((root / "candidate_flush.json").read_text()), receipt)


if __name__ == "__main__":
    unittest.main()
