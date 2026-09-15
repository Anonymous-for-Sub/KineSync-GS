import copy
import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import yaml

from kinesync.cli.rt6_sync_bridge import execute_rt6_sync_bridge
from kinesync.sync.calibration import calibration_rows_sha256


JOINTS = tuple(f"joint{index}" for index in range(1, 9))
FACTOR_JOINTS = JOINTS[:6]
FACTOR_ZERO = [0.01, -0.02, 0.03, -0.04, 0.05, -0.06]
HEAD_QPOS = [0.10, -0.11, 0.20, -0.21, 0.30, -0.31, 0.015, -0.015]
EXTRA_QPOS = [0.102, -0.108, 0.202, -0.208, 0.302, -0.308, 0.017, -0.017]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _legacy_row(
    trial_id: str,
    state_id: str,
    offsets: dict[str, float],
    correction: list[float],
    *,
    candidate_success: bool,
) -> dict[str, object]:
    return {
        "trial_id": trial_id,
        "state_id": state_id,
        "split": "train",
        "case_role": "paired_zero" if not any(offsets.values()) else "paired_controlled",
        "offset_name": trial_id.removeprefix(f"{state_id}_"),
        "offsets_rad": json.dumps(offsets, sort_keys=True),
        "candidate_success": candidate_success,
        "state_error_reduction": 0.8 if candidate_success else 0.2,
        "mask_iou_before": 0.60,
        "mask_iou_candidate": 0.65,
        "boundary_f1_before": 0.70,
        "boundary_f1_candidate": 0.75,
        "evidence_finite": True,
        "view_count": 2,
        "visual_gain_ratio": 0.10,
        "gradient_cosine": 1.0,
        "correction_cosine": 1.0,
        "relative_correction_disagreement": 0.01,
        "candidate_bound_fraction": 0.10,
        "joint_correction_rad": json.dumps(correction),
    }


def _write_chain_urdf(path: Path) -> None:
    links = ['  <link name="link0"/>']
    for index in range(1, 9):
        links.extend(
            [
                f'  <link name="link{index}">',
                '    <inertial>',
                '      <origin xyz="0 0 0" rpy="0 0 0"/>',
                '      <mass value="0.1"/>',
                '      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>',
                '    </inertial>',
                '  </link>',
            ]
        )
    joints = []
    for index, name in enumerate(JOINTS, start=1):
        joints.extend(
            [
                f'  <joint name="{name}" type="revolute">',
                f'    <parent link="link{index - 1}"/>',
                f'    <child link="link{index}"/>',
                '    <axis xyz="0 0 1"/>',
                '    <limit lower="-1.0" upper="1.0" effort="1" velocity="1"/>',
                '  </joint>',
            ]
        )
    path.write_text(
        "\n".join(['<?xml version="1.0"?>', '<robot name="test_piper">', *links, *joints, '</robot>', '']),
        encoding="utf-8",
    )


def _write_real_observations(
    root: Path,
    state_ids: list[str],
    *,
    qpos_overrides: dict[str, dict[str, list[float]]] | None = None,
) -> dict[str, object]:
    records: dict[str, str] = {}
    image_roots: dict[str, str] = {}
    mask_roots: dict[str, str] = {}
    for camera, qpos in (("head", HEAD_QPOS), ("extra", EXTRA_QPOS)):
        camera_root = root / camera
        image_root = camera_root / "images"
        mask_root = camera_root / "masks"
        image_root.mkdir(parents=True)
        mask_root.mkdir()
        rows = []
        for index, state_id in enumerate(state_ids):
            sample_id = f"{camera}_{state_id}"
            state_qpos = (qpos_overrides or {}).get(state_id, {}).get(camera, qpos)
            cv2.imwrite(
                str(image_root / f"{sample_id}.jpg"),
                np.full((8, 10, 3), 127 + index, dtype=np.uint8),
            )
            cv2.imwrite(
                str(mask_root / f"{sample_id}.png"),
                np.full((8, 10), 255, dtype=np.uint8),
            )
            rows.append(
                {
                    "sample_id": sample_id,
                    "state_id": state_id,
                    "split": "validation",
                    "camera_id": camera,
                    "frame_index": index,
                    "host_monotonic_timestamp_ns": 1_000 + index,
                    "image_path": f"images/{sample_id}.jpg",
                    "urdf_qpos_rad": state_qpos,
                }
            )
        records_path = camera_root / "records.jsonl"
        records_path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
        records[camera] = str(records_path)
        image_roots[camera] = str(camera_root)
        mask_roots[camera] = str(mask_root)
    return {
        "records": records,
        "image_roots": image_roots,
        "mask_roots": mask_roots,
        "max_pair_qpos_delta_rad": 0.02,
    }


class RT6SynchronizationBridgeIntegrationTest(unittest.TestCase):
    def _fixture(
        self,
        root: Path,
        *,
        audit_case_count: int = 3,
        bridge_exclusion_state_id: str | None = None,
    ) -> dict[str, object]:
        factors_path = root / "factors.json"
        factors_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "frozen": True,
                    "fit_split": "train",
                    "joint_names": list(FACTOR_JOINTS),
                    "joint_zero_rad": FACTOR_ZERO,
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        factor_sha = _sha256(factors_path)

        calibration_path = root / "legacy_calibration.csv"
        _write_csv(
            calibration_path,
            [
                _legacy_row(
                    "train_001_j1",
                    "train_001",
                    {"joint1": 0.04},
                    [-0.036, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    candidate_success=True,
                ),
                _legacy_row(
                    "train_002_j2",
                    "train_002",
                    {"joint2": 0.04},
                    [0.0, -0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    candidate_success=False,
                ),
                _legacy_row(
                    "train_003_zero",
                    "train_003",
                    {"joint1": 0.0, "joint2": 0.0},
                    [0.002, 0.001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    candidate_success=False,
                ),
            ],
        )

        guard_path = root / "guard.json"
        guard_path.write_text(
            json.dumps(
                {
                    "frozen": True,
                    "source_split": "train",
                    "factor_sha256": factor_sha,
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )

        audit_matrix_path = root / "audit_matrix.yaml"
        audit_matrix = {
            "split": "validation",
            "provenance": {"factor_train_state_ids": ["fit_001"]},
            "cases": [
                {
                    "state_id": "audit_001",
                    "case_role": "paired_controlled",
                    "offset_name": "j1",
                    "cameras": ["head", "extra"],
                    "offsets_rad": {"joint1": 0.04},
                },
                {
                    "state_id": "audit_002",
                    "case_role": "paired_controlled",
                    "offset_name": "j2",
                    "cameras": ["head", "extra"],
                    "offsets_rad": {"joint2": 0.04},
                },
                {
                    "state_id": "audit_003",
                    "case_role": "paired_zero",
                    "offset_name": "zero",
                    "cameras": ["head", "extra"],
                    "offsets_rad": {"joint1": 0.0},
                },
            ],
        }
        for index in range(4, audit_case_count + 1):
            audit_matrix["cases"].append(
                {
                    "state_id": f"audit_{index:03d}",
                    "case_role": "paired_zero",
                    "offset_name": "zero",
                    "cameras": ["head", "extra"],
                    "offsets_rad": {"joint1": 0.0},
                }
            )
        audit_matrix_path.write_text(yaml.safe_dump(audit_matrix, sort_keys=False), encoding="utf-8")

        audit_rows = []
        trace_rows = []
        final_corrections = {
            "audit_001_j1": [-0.037, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "audit_002_j2": [0.0, -0.036, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "audit_003_zero": [0.04, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        }
        for case in audit_matrix["cases"]:
            trial_id = f"{case['state_id']}_{case['offset_name']}"
            for method in ("rt3_full", "rt2f_unguarded"):
                audit_rows.append(
                    {
                        "trial_id": trial_id,
                        "state_id": case["state_id"],
                        "split": "validation",
                        "case_role": case["case_role"],
                        "method": method,
                        "candidate_committed": True,
                        "decision_reason": "accepted",
                        "minimum_margin": 0.01,
                        "recovery_success": case["case_role"] != "paired_zero",
                    }
                )
            for step, correction in enumerate(
                ([0.0] * 8, final_corrections.get(trial_id, [0.0] * 8))
            ):
                trace_rows.append(
                    {
                        "trial_id": trial_id,
                        "candidate_source": "rt2_joint",
                        "step": step,
                        **{
                            f"correction_{name}": value
                            for name, value in zip(JOINTS, correction, strict=True)
                        },
                    }
                )
        audit_results_path = root / "audit_results.csv"
        audit_trace_path = root / "audit_trace.csv"
        _write_csv(audit_results_path, audit_rows)
        _write_csv(audit_trace_path, trace_rows)

        urdf_path = root / "piper.urdf"
        _write_chain_urdf(urdf_path)
        qpos_overrides = None
        if bridge_exclusion_state_id is not None:
            head = list(HEAD_QPOS)
            extra = list(EXTRA_QPOS)
            head[2] = 1.10
            extra[2] = 1.10
            qpos_overrides = {
                bridge_exclusion_state_id: {"head": head, "extra": extra}
            }
        real_observations = _write_real_observations(
            root / "real_observations",
            [str(case["state_id"]) for case in audit_matrix["cases"]],
            qpos_overrides=qpos_overrides,
        )
        return {
            "experiment": "rt6_sync_bridge_test",
            "runs_root": str(root / "runs"),
            "full_joint_order": list(JOINTS),
            "inputs": {
                "calibration_rows": str(calibration_path),
                "factors": str(factors_path),
                "guard": str(guard_path),
                "audit_results": str(audit_results_path),
                "audit_trace": str(audit_trace_path),
                "audit_matrix": str(audit_matrix_path),
            },
            "assets": {
                "urdf": str(urdf_path),
                "package_root": str(root),
            },
            "real_observations": real_observations,
            "audit": {
                "method": "rt3_full",
                "candidate_method": "rt2f_unguarded",
                "candidate_source": "rt2_joint",
            },
        }

    def test_migrates_legacy_rows_and_reports_development_only_component_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self._fixture(Path(directory))

            first_run = execute_rt6_sync_bridge(config, run_id="first")
            second_run = execute_rt6_sync_bridge(config, run_id="second")

            for relative in (
                "migrated_calibration_rows.csv",
                "detectability.json",
                "results.csv",
                "metrics.json",
                "manifest.json",
                "parity_report.json",
                "converted_piper.urdf",
                "conversion_manifest.json",
            ):
                self.assertTrue((first_run / relative).is_file(), relative)

            for relative in (
                "migrated_calibration_rows.csv",
                "detectability.json",
                "results.csv",
                "metrics.json",
                "parity_report.json",
                "converted_piper.urdf",
                "conversion_manifest.json",
            ):
                self.assertEqual(
                    (first_run / relative).read_bytes(),
                    (second_run / relative).read_bytes(),
                    relative,
                )

            with (first_run / "migrated_calibration_rows.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                migrated_rows = list(csv.DictReader(handle))
            self.assertEqual(len(migrated_rows), 3)
            self.assertEqual(json.loads(migrated_rows[0]["joint_order"]), list(JOINTS))
            factor_sha = _sha256(Path(config["inputs"]["factors"]))
            self.assertEqual(migrated_rows[0]["joint_order_source_sha256"], factor_sha)
            self.assertIn("joint1", json.loads(migrated_rows[0]["per_joint_recovery"]))

            detectability = json.loads((first_run / "detectability.json").read_text(encoding="utf-8"))
            second_detectability = json.loads((second_run / "detectability.json").read_text(encoding="utf-8"))
            migrated_content_sha = calibration_rows_sha256(migrated_rows)
            self.assertEqual(detectability["source_sha256"], migrated_content_sha)
            self.assertEqual(detectability["joint_order_source_sha256"], factor_sha)
            self.assertEqual(detectability["fingerprint"], second_detectability["fingerprint"])

            metrics = json.loads((first_run / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(metrics["audit_designation"], "development_audit_only")
            self.assertFalse(metrics["formal_evidence"])
            self.assertEqual(metrics["controlled_component_count"], 2)
            self.assertEqual(metrics["committed_component_count"], 2)
            self.assertEqual(metrics["committed_precision"], 1.0)
            self.assertEqual(metrics["controlled_commit_coverage"], 0.5)
            self.assertEqual(metrics["successful_candidate_retention"], 0.5)
            self.assertEqual(metrics["zero_false_update_rate"], 1.0)
            self.assertEqual(metrics["zero_stability"], 0.0)
            self.assertEqual(
                metrics["source_hashes"]["migrated_calibration_rows"],
                migrated_content_sha,
            )
            self.assertTrue(metrics["parity_report"]["passed"])

            with (first_run / "results.csv").open(encoding="utf-8", newline="") as handle:
                result_rows = list(csv.DictReader(handle))
            self.assertEqual(len(result_rows), 3)
            self.assertTrue(all(row["audit_designation"] == "development_audit_only" for row in result_rows))
            self.assertEqual(result_rows[1]["component_decisions"].count("unsupported_joint"), 1)
            expected_measured = [
                0.151,
                -0.129,
                0.231,
                -0.249,
                0.351,
                -0.369,
                0.016,
                -0.016,
            ]
            expected_candidate = [0.114, *expected_measured[1:]]
            for field in ("measured_qpos", "candidate_qpos", "synchronized_qpos"):
                self.assertIn(field, result_rows[0])
            np.testing.assert_allclose(
                json.loads(result_rows[0]["measured_qpos"]), expected_measured, atol=1e-7
            )
            np.testing.assert_allclose(
                json.loads(result_rows[0]["candidate_qpos"]), expected_candidate, atol=1e-7
            )
            np.testing.assert_allclose(
                json.loads(result_rows[0]["synchronized_qpos"]), expected_candidate
            )

            parity = json.loads(
                (first_run / "parity_report.json").read_text(encoding="utf-8")
            )
            self.assertEqual(parity["audit_designation"], "development_audit_only")
            self.assertFalse(parity["formal_evidence"])
            np.testing.assert_allclose(
                parity["states"][0]["measured_qpos"], expected_measured, atol=1e-7
            )
            np.testing.assert_allclose(
                parity["states"][0]["candidate_qpos"], expected_candidate, atol=1e-7
            )
            np.testing.assert_allclose(
                parity["states"][0]["synchronized_qpos"], expected_candidate, atol=1e-7
            )

            manifest = json.loads(
                (first_run / "manifest.json").read_text(encoding="utf-8")
            )
            second_manifest = json.loads(
                (second_run / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                {
                    "derived_artifacts": manifest["derived_artifacts"],
                    "source_hashes": manifest["source_hashes"],
                },
                {
                    "derived_artifacts": second_manifest["derived_artifacts"],
                    "source_hashes": second_manifest["source_hashes"],
                },
            )
            self.assertEqual(
                manifest["derived_artifacts"]["converted_piper_urdf"]["sha256"],
                _sha256(first_run / "converted_piper.urdf"),
            )
            self.assertEqual(
                manifest["derived_artifacts"]["conversion_manifest"]["sha256"],
                _sha256(first_run / "conversion_manifest.json"),
            )

    def test_rejects_calibration_and_audit_state_overlap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self._fixture(root)
            matrix_path = Path(config["inputs"]["audit_matrix"])
            matrix = yaml.safe_load(matrix_path.read_text(encoding="utf-8"))
            matrix["cases"][0]["state_id"] = "train_001"
            matrix_path.write_text(yaml.safe_dump(matrix, sort_keys=False), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "overlap"):
                execute_rt6_sync_bridge(config, run_id="overlap")

    def test_rejects_missing_matrix_cameras_and_nonvalidation_observations(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self._fixture(root)
            matrix_path = Path(config["inputs"]["audit_matrix"])
            matrix = yaml.safe_load(matrix_path.read_text(encoding="utf-8"))
            del matrix["cases"][0]["cameras"]
            matrix_path.write_text(
                yaml.safe_dump(matrix, sort_keys=False), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "cameras"):
                execute_rt6_sync_bridge(config, run_id="missing-cameras")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self._fixture(root)
            for camera in ("head", "extra"):
                records_path = Path(
                    config["real_observations"]["records"][camera]
                )
                records = [
                    {**json.loads(line), "split": "train"}
                    for line in records_path.read_text(encoding="utf-8").splitlines()
                ]
                records_path.write_text(
                    "".join(
                        json.dumps(row, sort_keys=True) + "\n" for row in records
                    ),
                    encoding="utf-8",
                )
            with self.assertRaisesRegex(ValueError, "validation"):
                execute_rt6_sync_bridge(config, run_id="train-observation")

    def test_records_one_bridge_exclusion_and_keeps_all_48_result_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self._fixture(
                root,
                audit_case_count=48,
                bridge_exclusion_state_id="audit_004",
            )
            try:
                run = execute_rt6_sync_bridge(config, run_id="one-exclusion")
            except ValueError as error:
                self.fail(f"bridge exclusion must be recorded, not raised: {error}")

            with (run / "results.csv").open(encoding="utf-8", newline="") as handle:
                results = list(csv.DictReader(handle))
            self.assertEqual(len(results), 48)
            excluded_results = [
                row for row in results if row["bridge_eligible"] == "False"
            ]
            self.assertEqual(len(excluded_results), 1)
            self.assertEqual(excluded_results[0]["trial_id"], "audit_004_zero")
            exclusion = json.loads(excluded_results[0]["bridge_exclusion"])
            self.assertEqual(
                exclusion["reason"], "bridge_ineligible_measured_outside_limit"
            )
            self.assertEqual(exclusion["case_id"], "audit_004_zero")
            self.assertEqual(exclusion["violations"][0]["joint_name"], "joint3")
            self.assertAlmostEqual(
                exclusion["violations"][0]["measured_value"], 1.13, places=6
            )
            self.assertEqual(
                exclusion["violations"][0]["original_limit"], [-1.0, 1.0]
            )
            self.assertGreater(
                json.loads(excluded_results[0]["measured_qpos"])[2], 1.0
            )

            metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
            parity = json.loads(
                (run / "parity_report.json").read_text(encoding="utf-8")
            )
            manifest = json.loads(
                (run / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metrics["bridge_parity_exclusions"], [exclusion])
            self.assertEqual(parity["exclusions"], [exclusion])
            self.assertEqual(manifest["bridge_parity_exclusions"], [exclusion])
            self.assertEqual(parity["state_count"], 47)
            self.assertEqual(len(parity["states"]), 47)


if __name__ == "__main__":
    unittest.main()
