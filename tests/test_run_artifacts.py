import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from kinesync.runs.artifacts import RunArtifacts, validate_run_id


class RunArtifactsTest(unittest.TestCase):
    def test_run_id_validation_rejects_traversal_before_creating_directories(self) -> None:
        invalid_ids = ("../escape", "/tmp/escape", ".", "..", "nested/run", "nested\\run", "", 7)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "runs"
            for run_id in invalid_ids:
                with self.subTest(run_id=run_id):
                    with self.assertRaises(ValueError):
                        RunArtifacts.create(
                            root=root,
                            experiment="rt0-unit",
                            run_id=run_id,  # type: ignore[arg-type]
                            config={},
                            assets={},
                            target_provenance="synthetic_state",
                            observation_backend="projected_centers",
                        )
                    self.assertFalse(root.exists())
        for run_id in (None, 7):
            with self.subTest(run_id=run_id):
                with self.assertRaises(ValueError):
                    validate_run_id(run_id)  # type: ignore[arg-type]

    def test_run_id_validation_preserves_ascii_slug(self) -> None:
        self.assertEqual(validate_run_id("20260825_rt10_preflight_v1"), "20260825_rt10_preflight_v1")

    def test_run_id_validation_accepts_grammar_edges_and_default_generation(self) -> None:
        self.assertEqual(validate_run_id("a"), "a")
        self.assertEqual(validate_run_id("a.b_c-d"), "a.b_c-d")
        for run_id in (".a", "a.", "_a", "a_", "-a", "a-", "r\u00fcn", "\u6d4b\u8bd5"):
            with self.subTest(run_id=run_id):
                with self.assertRaises(ValueError):
                    validate_run_id(run_id)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "runs"
            run = RunArtifacts.create(
                root=root,
                experiment="rt0-unit",
                run_id=None,
                config={},
                assets={},
                target_provenance="synthetic_state",
                observation_backend="projected_centers",
            )
            self.assertEqual(run.path.parent, root)
            self.assertEqual(validate_run_id(run.path.name), run.path.name)

    def test_writes_reproducible_run_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            asset = root / "anchor.npz"
            asset.write_bytes(b"route-a-anchor")

            run = RunArtifacts.create(
                root=root / "runs",
                experiment="rt0-unit",
                run_id="unit-run",
                config={"seed": 7, "optimizer": {"steps": 3}},
                assets={"anchor": asset},
                target_provenance="synthetic_state",
                observation_backend="projected_centers",
            )
            run.write_trace(
                [
                    {"step": 0, "loss": 1.0},
                    {"step": 1, "loss": 0.25},
                ]
            )
            run.write_metrics({"initial_loss": 1.0, "final_loss": 0.25})

            self.assertTrue((run.path / "config.yaml").is_file())
            self.assertTrue((run.path / "manifest.json").is_file())
            self.assertTrue((run.path / "metrics.json").is_file())
            self.assertTrue((run.path / "trace.csv").is_file())
            self.assertTrue((run.path / "images").is_dir())
            self.assertTrue((run.path / "videos").is_dir())

            manifest = json.loads((run.path / "manifest.json").read_text())
            self.assertEqual(manifest["experiment"], "rt0-unit")
            self.assertEqual(manifest["target_provenance"], "synthetic_state")
            self.assertEqual(manifest["observation_backend"], "projected_centers")
            self.assertEqual(
                manifest["assets"]["anchor"]["sha256"],
                hashlib.sha256(asset.read_bytes()).hexdigest(),
            )
            self.assertEqual(manifest["assets"]["anchor"]["path"], str(asset.resolve()))

            metrics = json.loads((run.path / "metrics.json").read_text())
            self.assertEqual(metrics["final_loss"], 0.25)
            with (run.path / "trace.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["step"] for row in rows], ["0", "1"])


if __name__ == "__main__":
    unittest.main()
