import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import yaml

from kinesync.cli.rt6_formal import (
    _formal_source_hashes,
    _frozen_formal_inputs,
    _sha256,
)
from kinesync.config import load_config
from kinesync.replay.rt6_source import (
    load_frozen_rt6_config_snapshot,
    verify_frozen_rt6_source,
)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


class RT7SourcePreflightTest(unittest.TestCase):
    def test_snapshot_loader_preserves_embedded_canonical_formal_config_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            canonical = root / "rt6_formal.yaml"
            snapshot = root / "source-config.yaml"
            canonical.write_text("formal: {}\n", encoding="utf-8")
            snapshot.write_text(
                yaml.safe_dump(
                    {"_config_path": str(canonical), "formal": {}}, sort_keys=True
                ),
                encoding="utf-8",
            )
            loaded = load_frozen_rt6_config_snapshot(snapshot)
            self.assertEqual(loaded["_config_path"], str(canonical.resolve()))

    def _fixture(self, root: Path):
        source_run = root / "source-rt6"
        source_run.mkdir()
        package_root = root / "package"
        package_root.mkdir()
        assets_root = root / "assets"
        assets_root.mkdir()

        files = {
            "anchor": assets_root / "anchors.npz",
            "metadata": assets_root / "metadata.json",
            "trajectory_h5": assets_root / "trajectory.h5",
            "head_camera": assets_root / "head_camera.json",
            "extra_camera": assets_root / "extra_camera.json",
            "urdf": assets_root / "robot.urdf",
            "mesh": assets_root / "mesh.stl",
            "factors": root / "factors.json",
            "guard": root / "guard.json",
            "profile": root / "profile.json",
            "matrix": root / "matrix.yaml",
        }
        for name, path in files.items():
            if name == "urdf":
                path.write_text(
                    '<robot name="fixture"><link name="base"><visual><geometry>'
                    '<mesh filename="mesh.stl"/></geometry></visual></link></robot>',
                    encoding="utf-8",
                )
            else:
                path.write_bytes(f"{name}-fixture".encode("utf-8"))

        records = {}
        mutation_paths = {"camera": files["head_camera"]}
        for camera in ("head", "extra"):
            image_root = root / f"{camera}-images"
            mask_root = root / f"{camera}-masks"
            image_root.mkdir()
            mask_root.mkdir()
            image = image_root / "frame.png"
            mask = mask_root / f"{camera}-sample.png"
            image.write_bytes(f"{camera}-rgb".encode("utf-8"))
            mask.write_bytes(f"{camera}-mask".encode("utf-8"))
            records_path = root / f"{camera}-records.jsonl"
            records_path.write_text(
                json.dumps(
                    {
                        "camera_id": camera,
                        "image_path": image.name,
                        "sample_id": f"{camera}-sample",
                        "split": "validation",
                        "state_id": "state-a",
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            records[camera] = {
                "records": records_path,
                "image_root": image_root,
                "mask_root": mask_root,
            }
            if camera == "head":
                mutation_paths["rgb"] = image
                mutation_paths["mask"] = mask

        converted = source_run / "converted_piper.urdf"
        conversion_manifest = source_run / "conversion_manifest.json"
        converted.write_bytes(b"converted")
        _write_json(conversion_manifest, {"converted": True})

        source_config_path = source_run / "config.yaml"
        source_config_payload = {
            "assets": {name: str(files[name]) for name in (
                "anchor",
                "extra_camera",
                "head_camera",
                "metadata",
                "trajectory_h5",
                "urdf",
            )},
            "formal": {
                "expected_hashes": {},
                "expected_optimizer_executions_per_candidate": 3,
                "expected_trial_count": 1,
                "expected_unique_state_count": 1,
                "frozen": True,
                "inputs": {
                    "factors": str(files["factors"]),
                    "guard": str(files["guard"]),
                    "matrix": str(files["matrix"]),
                    "profile": str(files["profile"]),
                },
                "max_source_timestamp_delta_ns": 1,
                "package_root": str(package_root),
            },
            "real_observations": {
                "image_roots": {
                    camera: str(records[camera]["image_root"])
                    for camera in records
                },
                "mask_roots": {
                    camera: str(records[camera]["mask_root"])
                    for camera in records
                },
                "records": {
                    camera: str(records[camera]["records"])
                    for camera in records
                },
            },
        }
        source_config_path.write_text(
            yaml.safe_dump(source_config_payload, sort_keys=True), encoding="utf-8"
        )
        source_config = load_config(source_config_path)
        matrix = SimpleNamespace(
            cases=(SimpleNamespace(state_id="state-a"),),
            fingerprint="d" * 64,
            source_file_sha256=_sha256(files["matrix"]),
            trial_count=1,
            unique_state_count=1,
        )
        frozen = _frozen_formal_inputs(source_config)
        formal_hashes, _ = _formal_source_hashes(
            config=source_config,
            matrix=matrix,
            frozen=frozen,
            factor_sha256=_sha256(files["factors"]),
            guard_sha256=_sha256(files["guard"]),
            profile_sha256=_sha256(files["profile"]),
            profile_fingerprint="d" * 64,
        )
        source_config_payload["formal"]["expected_hashes"] = formal_hashes
        source_config_path.write_text(
            yaml.safe_dump(source_config_payload, sort_keys=True), encoding="utf-8"
        )
        source_config = load_config(source_config_path)
        source_hashes = {
            **formal_hashes,
            "formal_config": _sha256(source_config_path),
            "converted_urdf": _sha256(converted),
            "conversion_manifest": _sha256(conversion_manifest),
        }
        source_metrics = {"source_hashes": source_hashes}
        return source_config, source_metrics, source_run, matrix, files, mutation_paths

    def _preflight(self, source_config, source_metrics, source_run, matrix, files):
        profile = SimpleNamespace(fingerprint="d" * 64)
        with (
            mock.patch(
                "kinesync.replay.rt6_source._load_factors",
                return_value=(files["factors"], {"joint_names": []}),
            ),
            mock.patch(
                "kinesync.replay.rt6_source._validate_factor_bounds",
                return_value=None,
            ),
            mock.patch(
                "kinesync.replay.rt6_source._load_guard",
                return_value=(files["guard"], {}),
            ),
            mock.patch(
                "kinesync.replay.rt6_source._load_detectability_profile",
                return_value=(files["profile"], profile),
            ),
            mock.patch(
                "kinesync.replay.rt6_source.load_rt6_matrix",
                return_value=matrix,
            ),
        ):
            return verify_frozen_rt6_source(
                source_config=source_config,
                source_metrics=source_metrics,
                source_run_path=source_run,
            )

    def test_preflight_records_all_frozen_source_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            source_config, metrics, source_run, matrix, files, _ = self._fixture(
                Path(directory)
            )
            preflight = self._preflight(
                source_config, metrics, source_run, matrix, files
            )
            self.assertEqual(preflight.verified_source_hashes, metrics["source_hashes"])
            self.assertIn("observation/head/state-a/rgb", preflight.asset_receipt["files"])
            self.assertIn("observation/head/state-a/mask", preflight.asset_receipt["files"])
            self.assertIn("urdf_mesh/000/mesh.stl", preflight.asset_receipt["files"])

    def test_preflight_rejects_mutated_camera_rgb_or_mask_before_output(self):
        for mutation_name in ("camera", "rgb", "mask"):
            with self.subTest(mutation=mutation_name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source_config, metrics, source_run, matrix, files, mutation_paths = self._fixture(root)
                mutation_paths[mutation_name].write_bytes(b"mutated-after-rt6")
                with self.assertRaisesRegex(ValueError, "frozen RT6 source hashes"):
                    self._preflight(source_config, metrics, source_run, matrix, files)
                self.assertFalse((root / "runs" / "rejected-before-output").exists())


if __name__ == "__main__":
    unittest.main()
