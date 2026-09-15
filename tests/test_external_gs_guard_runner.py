import hashlib
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import torch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_external_gs_guard.py"
SPEC = importlib.util.spec_from_file_location("external_gs_guard_runner", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


class ExternalGSGuardRunnerTest(unittest.TestCase):
    def test_heldout_calibration_requires_exact_frozen_development_ids_and_hash(self):
        config = {
            "development_cases": [{"id": "dev_a"}, {"id": "dev_b"}],
            "heldout_cases": [{"id": "test_a"}],
            "conditions": [{"id": "clean_zero"}, {"id": "clean_nonzero"}],
        }
        expected_ids = {
            "dev_a__clean_zero",
            "dev_a__clean_nonzero",
            "dev_b__clean_zero",
            "dev_b__clean_nonzero",
        }
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "config.yaml"
            config_path.write_text("frozen: true\n", encoding="utf-8")
            config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
            development = root / "development"
            development.mkdir()
            (development / "config_source.yaml").write_bytes(config_path.read_bytes())
            (development / "manifest.json").write_text(
                json.dumps({
                    "split": "development",
                    "config_source_sha256": config_hash,
                    "target_provenance": "analysis_same_gs",
                }),
                encoding="utf-8",
            )
            (development / "heldout_manifest.json").write_text(
                json.dumps({
                    "case_ids": ["test_a"],
                    "condition_ids": ["clean_zero", "clean_nonzero"],
                    "config_sha256": config_hash,
                }),
                encoding="utf-8",
            )
            calibration = SimpleNamespace(source_state_ids=tuple(sorted(expected_ids)))
            profile = SimpleNamespace(calibration_state_ids=tuple(sorted(expected_ids)))

            RUNNER._validate_heldout_calibration_provenance(
                config,
                config_path=config_path,
                development_root=development,
                calibration=calibration,
                profile=profile,
            )
            profile.calibration_state_ids = ("test_a__clean_zero",)
            with self.assertRaisesRegex(ValueError, "development state IDs"):
                RUNNER._validate_heldout_calibration_provenance(
                    config,
                    config_path=config_path,
                    development_root=development,
                    calibration=calibration,
                    profile=profile,
                )

    def test_update_applied_is_distinct_from_global_guard_decision(self):
        measured = torch.tensor([0.0, 0.0])
        rows, per_joint = RUNNER._evaluation_rows(
            split="development",
            state_id="dev_a__clean_zero",
            case_id="dev_a",
            condition_id="clean_zero",
            method="global_guard",
            measured=measured,
            candidate=torch.tensor([0.1, 0.0]),
            final=measured,
            true_q=measured,
            global_guard_accepted=True,
            component_guard_accepted=None,
            latency_ms=1.0,
            frame_paths={},
        )

        self.assertEqual(rows["update_applied_joint_count"], 0)
        self.assertTrue(rows["global_guard_accepted"])
        self.assertNotIn("accepted_joint_count", rows)
        self.assertTrue(all(not row["update_applied"] for row in per_joint))
        self.assertTrue(all(row["global_guard_accepted"] for row in per_joint))

    def test_candidate_flush_persists_diagnostics_before_calibration(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            receipt = RUNNER._persist_candidate_diagnostics(
                root,
                candidate_rows=[{"state_id": "dev_a__clean_zero", "loss": 0.1}],
                trace_rows=[{"state_id": "dev_a__clean_zero", "step": 0}],
            )
            self.assertEqual(receipt["candidate_row_count"], 1)
            self.assertEqual(receipt["trace_row_count"], 1)
            self.assertTrue((root / "candidate_evidence.csv").is_file())
            self.assertTrue((root / "optimizer_trace_qpos.csv").is_file())
            self.assertEqual(
                json.loads((root / "candidate_flush.json").read_text(encoding="utf-8")),
                receipt,
            )

    def test_rt7_receipt_persists_hashes_and_native_frame_paths(self):
        class Renderer:
            def render_buffers(self, _qpos):
                return {
                    "view_a": SimpleNamespace(rgb=torch.full((2, 2, 3), 0.5)),
                    "view_b": SimpleNamespace(rgb=torch.full((2, 2, 3), 0.25)),
                }

        with TemporaryDirectory() as directory:
            root = Path(directory)
            frames = root / "native_frames"
            frames.mkdir()
            receipt = RUNNER._receipt_with_native_frames(
                SimpleNamespace(renderer=Renderer()),
                torch.tensor([0.1, -0.2]),
                frames_dir=frames,
                state_id="dev_a__clean_zero",
                receipt_label="rt7_synchronized_receipt",
            )
            self.assertEqual(set(receipt["camera_render_sha256"]), {"view_a", "view_b"})
            self.assertEqual(set(receipt["native_frame_paths"]), {"view_a", "view_b"})
            self.assertTrue(
                all((root / path).is_file() for path in receipt["native_frame_paths"].values())
            )


if __name__ == "__main__":
    unittest.main()
