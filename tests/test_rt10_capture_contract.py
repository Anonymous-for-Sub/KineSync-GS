from __future__ import annotations

import copy
import contextlib
import hashlib
import io
import json
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCHEDULE_PATH = REPOSITORY_ROOT / "configs" / "rt10_capture_schedule.yaml"
CALIBRATION_PATH = REPOSITORY_ROOT / "configs" / "rt10_camera_calibration_template.yaml"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class RT10CaptureContractTest(unittest.TestCase):
    def _schedule_payload(self) -> dict[str, object]:
        return yaml.safe_load(SCHEDULE_PATH.read_text(encoding="utf-8"))

    def _write_yaml(self, root: Path, name: str, payload: object) -> Path:
        path = root / name
        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        return path

    def _write_slot_assets(self, root: Path, slot_id: str) -> dict[str, object]:
        values = {
            "head_rgb": np.full((2, 3, 4, 3), 7, dtype=np.uint8),
            "external_rgb": np.full((2, 3, 4, 3), 9, dtype=np.uint8),
            "head_timestamps": np.array([1_000_000_000, 1_010_000_000], dtype=np.int64),
            "external_timestamps": np.array([1_001_000_000, 1_011_000_000], dtype=np.int64),
            "qpos": np.zeros((2, 8), dtype=np.float64),
            "state_timestamps": np.array([1_002_000_000, 1_012_000_000], dtype=np.int64),
        }
        assets: dict[str, dict[str, str]] = {}
        for name, value in values.items():
            path = root / f"{slot_id}_{name}.npy"
            np.save(path, value, allow_pickle=False)
            assets[name] = {"path": path.name, "sha256": _sha256(path)}
        return {
            "media": {
                "head_rgb": assets["head_rgb"],
                "external_rgb": assets["external_rgb"],
                "head_timestamps": assets["head_timestamps"],
                "external_timestamps": assets["external_timestamps"],
            },
            "state": {
                "qpos": assets["qpos"],
                "timestamps": assets["state_timestamps"],
            },
            "frame_counts": {"head": 2, "external": 2, "state": 2},
            "timestamp_ranges_ns": {
                "head": [1_000_000_000, 1_010_000_000],
                "external": [1_001_000_000, 1_011_000_000],
                "state": [1_002_000_000, 1_012_000_000],
            },
            "image_shape": [3, 4, 3],
            "qpos_shape": [2, 8],
        }

    def _write_populated_calibration(self, root: Path) -> Path:
        source = root / "calibration_source.png"
        source.write_bytes(b"recorded calibration source")
        identity = np.eye(4, dtype=float).tolist()
        camera = {
            "serial": "camera-serial",
            "image_size": [640, 480],
            "intrinsics": {"fx": 500.0, "fy": 501.0, "cx": 320.0, "cy": 240.0},
            "distortion": [0.0, 0.0, 0.0, 0.0, 0.0],
            "base_to_camera": identity,
            "camera_to_base": identity,
            "calibrated_at_utc": "2026-08-25T00:00:00Z",
            "method": "offline_checkerboard",
            "source_images": [{"path": source.name, "sha256": _sha256(source)}],
        }
        return self._write_yaml(
            root,
            "calibration.yaml",
            {"schema": "kinesync.rt10_camera_calibration.v1", "cameras": {"head": camera, "external": {**camera, "serial": "external-serial"}}},
        )

    def _write_manifest(self, root: Path, schedule_path: Path, calibration_path: Path) -> Path:
        schedule = yaml.safe_load(schedule_path.read_text(encoding="utf-8"))
        slots = []
        for schedule_slot in schedule["slots"]:
            slot_id = schedule_slot["slot_id"]
            slots.append(
                {
                    "slot_id": slot_id,
                    "acquisition_status": "acquired",
                    **self._write_slot_assets(root / "session", slot_id),
                    "command_mode": "disabled",
                    "command_counters": {"requested": 0, "sent": 0},
                }
            )
        payload = {
            "schema": "kinesync.rt10_capture_manifest.v1",
            "project": "kinesync_gs",
            "stage": "RT10-P",
            "operator_acknowledgement": {"operator_id": "operator-1", "acknowledged_at_utc": "2026-08-25T00:00:00Z"},
            "robot_serial": "robot-serial",
            "camera_serials": {"head": "camera-serial", "external": "external-serial"},
            "host_clocks": {"wall_utc": "2026-08-25T00:00:00Z", "monotonic_domain": "host_monotonic_ns"},
            "session_id": "rt10_capture_v1",
            "session_root": "session",
            "schedule_sha256": _sha256(schedule_path),
            "calibration_sha256": _sha256(calibration_path),
            "slots": slots,
        }
        return self._write_yaml(root, "manifest.yaml", payload)

    def _replace_timestamps(
        self,
        root: Path,
        manifest: dict[str, object],
        *,
        slot_index: int,
        stream: str,
        values: np.ndarray,
        declared_range: list[int],
    ) -> None:
        slot = manifest["slots"][slot_index]
        if stream == "state":
            asset = slot["state"]["timestamps"]
        else:
            asset = slot["media"][f"{stream}_timestamps"]
        path = root / "session" / asset["path"]
        np.save(path, values, allow_pickle=False)
        asset["sha256"] = _sha256(path)
        slot["timestamp_ranges_ns"][stream] = declared_range

    def _replace_qpos(
        self,
        root: Path,
        manifest: dict[str, object],
        *,
        slot_index: int,
        values: np.ndarray,
    ) -> None:
        slot = manifest["slots"][slot_index]
        asset = slot["state"]["qpos"]
        path = root / "session" / asset["path"]
        np.save(path, values, allow_pickle=values.dtype.hasobject)
        asset["sha256"] = _sha256(path)

    def test_schedule_is_explicit_frozen_and_order_sensitive(self) -> None:
        from kinesync.capture.contracts import load_capture_schedule

        schedule = load_capture_schedule(SCHEDULE_PATH)
        self.assertEqual(
            schedule.fingerprint,
            "17ff10f638f356e79d3ecbdc1ba31429956de8bf515fb410ad6f0cbf76b98590",
        )
        self.assertEqual(len(schedule.slots), 36)
        self.assertEqual(schedule.role_counts, {"calibration": 12, "formal": 16, "reserve": 8})
        self.assertEqual(len(schedule.formal_trials), 32)
        self.assertEqual(sum(trial.kind == "single_joint" for trial in schedule.formal_trials), 26)
        self.assertEqual(sum(trial.kind == "zero_control" for trial in schedule.formal_trials), 6)
        self.assertEqual({trial.joint_index for trial in schedule.formal_trials}, set(range(1, 7)))
        self.assertEqual(len({trial.seed for trial in schedule.formal_trials}), 32)
        self.assertTrue(all(slot.camera_order == ("head", "external") for slot in schedule.slots))
        self.assertTrue(all(slot.command_mode == "disabled" for slot in schedule.slots))
        self.assertTrue(all(math.isclose(trial.magnitude_rad, math.radians(trial.sign * trial.magnitude_deg), abs_tol=1e-12) for trial in schedule.formal_trials))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            copied_path = root / "copied-schedule.yaml"
            copied_path.write_bytes(SCHEDULE_PATH.read_bytes())
            copied = load_capture_schedule(copied_path)
            self.assertEqual(copied.fingerprint, schedule.fingerprint)

            payload = self._schedule_payload()
            payload["slots"] = list(reversed(payload["slots"]))
            with self.assertRaisesRegex(ValueError, "canonical"):
                load_capture_schedule(self._write_yaml(root, "reordered.yaml", payload))

            payload = self._schedule_payload()
            payload["formal_trials"][0]["seed"] = 9001
            with self.assertRaisesRegex(ValueError, "canonical"):
                load_capture_schedule(self._write_yaml(root, "mutated.yaml", payload))

    def test_schedule_rejects_unknown_keys_booleans_bad_roles_and_promotions(self) -> None:
        from kinesync.capture.contracts import load_capture_schedule

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for mutation, message in (
                (lambda payload: payload["slots"][0].update({"unknown": 1}), "unknown"),
                (lambda payload: payload["slots"][0].update({"hold_duration_s": True}), "number"),
                (lambda payload: payload["slots"][0].update({"role": "promoted"}), "role"),
                (lambda payload: payload["slots"][12].update({"promotion": "reserve"}), "unknown"),
                (lambda payload: payload["formal_trials"][0].update({"magnitude_rad": float("nan")}), "finite"),
                (lambda payload: payload["formal_trials"][0].update({"seed": True}), "integer"),
            ):
                payload = self._schedule_payload()
                mutation(payload)
                with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                    load_capture_schedule(self._write_yaml(root, f"{message}.yaml", payload))

    def test_calibration_template_is_valid_but_not_ready(self) -> None:
        from kinesync.capture.contracts import validate_camera_calibration

        receipt = validate_camera_calibration(CALIBRATION_PATH)
        self.assertTrue(receipt.template_valid)
        self.assertFalse(receipt.acquisition_ready)

    def test_calibration_requires_complete_finite_inverse_local_evidence(self) -> None:
        from kinesync.capture.contracts import validate_camera_calibration

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calibration = self._write_populated_calibration(root)
            receipt = validate_camera_calibration(calibration)
            self.assertTrue(receipt.template_valid)
            self.assertTrue(receipt.acquisition_ready)

            payload = yaml.safe_load(calibration.read_text(encoding="utf-8"))
            payload["cameras"]["head"]["intrinsics"]["fx"] = True
            invalid = self._write_yaml(root, "bad-calibration.yaml", payload)
            with self.assertRaisesRegex(ValueError, "number"):
                validate_camera_calibration(invalid)

    def test_partial_nested_unset_calibration_stays_template_valid_and_not_ready(self) -> None:
        from kinesync.capture.contracts import validate_camera_calibration

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calibration = self._write_populated_calibration(root)
            original = yaml.safe_load(calibration.read_text(encoding="utf-8"))
            mutations = (
                ("image_size", lambda camera: camera["image_size"].__setitem__(0, "UNSET")),
                ("intrinsics", lambda camera: camera["intrinsics"].update({"cx": "UNSET"})),
                ("distortion", lambda camera: camera["distortion"].__setitem__(0, "UNSET")),
                ("base_to_camera", lambda camera: camera["base_to_camera"][0].__setitem__(0, "UNSET")),
                ("camera_to_base", lambda camera: camera["camera_to_base"][0].__setitem__(0, "UNSET")),
                ("source_path", lambda camera: camera["source_images"][0].update({"path": "UNSET"})),
                ("source_hash", lambda camera: camera["source_images"][0].update({"sha256": "UNSET"})),
            )
            for index, (name, mutate) in enumerate(mutations):
                payload = copy.deepcopy(original)
                mutate(payload["cameras"]["head"])
                candidate = self._write_yaml(root, f"partial-{index}.yaml", payload)
                with self.subTest(name=name):
                    receipt = validate_camera_calibration(candidate)
                    self.assertTrue(receipt.template_valid)
                    self.assertFalse(receipt.acquisition_ready)

    def test_manifest_validates_assets_and_formal_receipt_ignores_outcomes(self) -> None:
        from kinesync.capture.contracts import (
            load_capture_schedule,
            materialize_formal_matrix_receipt,
            validate_camera_calibration,
            validate_capture_manifest,
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "session").mkdir()
            schedule_path = root / "schedule.yaml"
            schedule_path.write_bytes(SCHEDULE_PATH.read_bytes())
            calibration_path = self._write_populated_calibration(root)
            manifest_path = self._write_manifest(root, schedule_path, calibration_path)
            schedule = load_capture_schedule(schedule_path)
            calibration = validate_camera_calibration(calibration_path)
            manifest = validate_capture_manifest(
                manifest_path, schedule=schedule, calibration=calibration
            )
            receipt = materialize_formal_matrix_receipt(schedule, calibration, manifest)
            self.assertEqual(receipt["schedule_sha256"], _sha256(schedule_path))
            self.assertEqual(receipt["calibration_sha256"], _sha256(calibration_path))
            self.assertEqual(receipt["manifest_sha256"], _sha256(manifest_path))
            self.assertEqual(len(receipt["formal_trials"]), 32)
            self.assertNotIn("outcome", receipt)

    def test_manifest_requires_acquisition_ready_provenance(self) -> None:
        from kinesync.capture.contracts import (
            load_capture_schedule,
            validate_camera_calibration,
            validate_capture_manifest,
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "session").mkdir()
            schedule_path = root / "schedule.yaml"
            schedule_path.write_bytes(SCHEDULE_PATH.read_bytes())
            calibration_path = self._write_populated_calibration(root)
            manifest_path = self._write_manifest(root, schedule_path, calibration_path)
            schedule = load_capture_schedule(schedule_path)
            calibration = validate_camera_calibration(calibration_path)
            original = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))

            cases = (
                (
                    "operator_unset",
                    lambda value: value["operator_acknowledgement"].update(
                        {"operator_id": "UNSET"}
                    ),
                    "operator_id",
                ),
                (
                    "robot_unset",
                    lambda value: value.update({"robot_serial": "UNSET"}),
                    "robot_serial",
                ),
                (
                    "session_unset",
                    lambda value: value.update({"session_id": "UNSET"}),
                    "session_id",
                ),
                (
                    "acknowledgement_unset",
                    lambda value: value["operator_acknowledgement"].update(
                        {"acknowledged_at_utc": "UNSET"}
                    ),
                    "acknowledged_at_utc",
                ),
                (
                    "wall_unset",
                    lambda value: value["host_clocks"].update({"wall_utc": "UNSET"}),
                    "wall_utc",
                ),
                (
                    "monotonic_unset",
                    lambda value: value["host_clocks"].update(
                        {"monotonic_domain": "UNSET"}
                    ),
                    "monotonic_domain",
                ),
                (
                    "monotonic_other",
                    lambda value: value["host_clocks"].update(
                        {"monotonic_domain": "monotonic_ns"}
                    ),
                    "monotonic_domain",
                ),
            )
            for field in ("acknowledged_at_utc", "wall_utc"):
                for form, timestamp in (
                    ("naive", "2026-08-25T00:00:00"),
                    ("non_utc", "2026-08-25T01:00:00+01:00"),
                    ("malformed", "not-a-timestamp"),
                ):
                    if field == "acknowledged_at_utc":
                        mutate = lambda value, timestamp=timestamp: value[
                            "operator_acknowledgement"
                        ].update({"acknowledged_at_utc": timestamp})
                    else:
                        mutate = lambda value, timestamp=timestamp: value[
                            "host_clocks"
                        ].update({"wall_utc": timestamp})
                    cases += ((f"{field}_{form}", mutate, field),)

            for name, mutate, message in cases:
                candidate = copy.deepcopy(original)
                mutate(candidate)
                candidate_path = self._write_yaml(root, f"provenance-{name}.yaml", candidate)
                with self.subTest(name=name), self.assertRaisesRegex(ValueError, message):
                    validate_capture_manifest(candidate_path, schedule, calibration)

            offset_utc = copy.deepcopy(original)
            offset_utc["operator_acknowledgement"]["acknowledged_at_utc"] = (
                "2026-08-25T00:00:00+00:00"
            )
            offset_utc["host_clocks"]["wall_utc"] = "2026-08-25T00:00:00+00:00"
            valid_path = self._write_yaml(root, "provenance-offset-utc.yaml", offset_utc)
            self.assertEqual(
                validate_capture_manifest(valid_path, schedule, calibration).session_id,
                "rt10_capture_v1",
            )

    def test_formal_receipt_rejects_mutated_placeholder_and_malformed_provenance(
        self,
    ) -> None:
        from kinesync.capture.contracts import (
            load_capture_schedule,
            materialize_formal_matrix_receipt,
            validate_camera_calibration,
            validate_capture_manifest,
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "session").mkdir()
            schedule_path = root / "schedule.yaml"
            schedule_path.write_bytes(SCHEDULE_PATH.read_bytes())
            calibration_path = self._write_populated_calibration(root)
            manifest_path = self._write_manifest(root, schedule_path, calibration_path)
            schedule = load_capture_schedule(schedule_path)
            calibration = validate_camera_calibration(calibration_path)
            manifest = validate_capture_manifest(manifest_path, schedule, calibration)
            original_bytes = manifest_path.read_bytes()
            original = yaml.safe_load(original_bytes)

            cases = (
                (
                    "placeholder",
                    lambda value: value["operator_acknowledgement"].update(
                        {"operator_id": "UNSET"}
                    ),
                    "operator_id",
                ),
                (
                    "malformed",
                    lambda value: value["host_clocks"].update(
                        {"wall_utc": "2026-08-25T00:00:00"}
                    ),
                    "wall_utc",
                ),
            )
            for name, mutate, message in cases:
                candidate = copy.deepcopy(original)
                mutate(candidate)
                manifest_path.write_text(
                    yaml.safe_dump(candidate, sort_keys=False), encoding="utf-8"
                )
                with self.subTest(name=name), self.assertRaisesRegex(ValueError, message):
                    materialize_formal_matrix_receipt(schedule, calibration, manifest)
                manifest_path.write_bytes(original_bytes)

            self.assertEqual(
                len(
                    materialize_formal_matrix_receipt(
                        schedule, calibration, manifest
                    )["formal_trials"]
                ),
                32,
            )

    def test_formal_receipt_revalidates_objects_sources_and_assets(self) -> None:
        from kinesync.capture.contracts import (
            load_capture_schedule,
            materialize_formal_matrix_receipt,
            validate_camera_calibration,
            validate_capture_manifest,
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "session").mkdir()
            schedule_path = root / "schedule.yaml"
            schedule_path.write_bytes(SCHEDULE_PATH.read_bytes())
            calibration_path = self._write_populated_calibration(root)
            manifest_path = self._write_manifest(root, schedule_path, calibration_path)
            schedule = load_capture_schedule(schedule_path)
            calibration = validate_camera_calibration(calibration_path)
            manifest = validate_capture_manifest(manifest_path, schedule, calibration)

            modified_trial = replace(schedule.formal_trials[0], trial_id="fabricated_trial")
            modified_schedule = replace(
                schedule,
                formal_trials=(modified_trial, *schedule.formal_trials[1:]),
            )
            modified_calibration = replace(
                calibration,
                camera_serials={"head": "fabricated", "external": "external-serial"},
            )
            modified_manifest = replace(manifest, slot_ids=("formal_slot_01",))
            for name, inputs in (
                ("schedule_object", (modified_schedule, calibration, manifest)),
                ("calibration_object", (schedule, modified_calibration, manifest)),
                ("manifest_object", (schedule, calibration, modified_manifest)),
            ):
                with self.subTest(name=name), self.assertRaisesRegex(ValueError, "revalidat"):
                    materialize_formal_matrix_receipt(*inputs)

            source_paths = (schedule_path, calibration_path, manifest_path)
            for source_path in source_paths:
                original = source_path.read_bytes()
                source_path.write_bytes(original + b"\n")
                with self.subTest(source=source_path.name), self.assertRaises(ValueError):
                    materialize_formal_matrix_receipt(schedule, calibration, manifest)
                source_path.write_bytes(original)

            manifest_payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
            qpos_asset = root / "session" / manifest_payload["slots"][0]["state"]["qpos"]["path"]
            original_asset = qpos_asset.read_bytes()
            qpos_asset.write_bytes(original_asset + b"tampered")
            with self.assertRaisesRegex(ValueError, "hash"):
                materialize_formal_matrix_receipt(schedule, calibration, manifest)

    def test_manifest_rejects_path_escape_hash_payload_skew_and_commands(self) -> None:
        from kinesync.capture.contracts import (
            load_capture_schedule,
            validate_camera_calibration,
            validate_capture_manifest,
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "session").mkdir()
            schedule_path = root / "schedule.yaml"
            schedule_path.write_bytes(SCHEDULE_PATH.read_bytes())
            calibration_path = self._write_populated_calibration(root)
            manifest_path = self._write_manifest(root, schedule_path, calibration_path)
            schedule = load_capture_schedule(schedule_path)
            calibration = validate_camera_calibration(calibration_path)
            payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))

            mutations = [
                (lambda value: value.update({"session_id": "../escape"}), "session_id"),
                (lambda value: value["slots"][0]["media"]["head_rgb"].update({"path": "../escape.npy"}), "path"),
                (lambda value: value["slots"][0]["command_counters"].update({"sent": 1}), "command"),
                (lambda value: value["slots"].pop(12), "formal"),
                (lambda value: value["slots"][0]["media"]["head_rgb"].update({"sha256": "0" * 64}), "hash"),
                (lambda value: value["slots"][0]["timestamp_ranges_ns"].update({"head": [9, 1]}), "range"),
            ]
            for index, (mutation, message) in enumerate(mutations):
                candidate = copy.deepcopy(payload)
                mutation(candidate)
                path = self._write_yaml(root, f"invalid-{index}.yaml", candidate)
                with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                    validate_capture_manifest(path, schedule=schedule, calibration=calibration)

    def test_manifest_rejects_negative_out_of_range_and_overflowing_timestamp_skew(self) -> None:
        from kinesync.capture.contracts import (
            load_capture_schedule,
            validate_camera_calibration,
            validate_capture_manifest,
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "session").mkdir()
            schedule_path = root / "schedule.yaml"
            schedule_path.write_bytes(SCHEDULE_PATH.read_bytes())
            calibration_path = self._write_populated_calibration(root)
            manifest_path = self._write_manifest(root, schedule_path, calibration_path)
            schedule = load_capture_schedule(schedule_path)
            calibration = validate_camera_calibration(calibration_path)
            original = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))

            cases = (
                ("negative", np.array([-2, -1], dtype=np.int64), [-2, -1]),
                (
                    "int64_min",
                    np.array([np.iinfo(np.int64).min, np.iinfo(np.int64).min + 1], dtype=np.int64),
                    [-(2**63), -(2**63) + 1],
                ),
                (
                    "uint64_overflow",
                    np.array([2**63, 2**63 + 1], dtype=np.uint64),
                    [-(2**63), -(2**63) + 1],
                ),
            )
            for index, (name, timestamps, declared_range) in enumerate(cases):
                payload = copy.deepcopy(original)
                for stream in ("head", "external", "state"):
                    self._replace_timestamps(
                        root,
                        payload,
                        slot_index=0,
                        stream=stream,
                        values=timestamps,
                        declared_range=declared_range,
                    )
                candidate = self._write_yaml(root, f"timestamp-{index}.yaml", payload)
                with self.subTest(name=name), self.assertRaisesRegex(ValueError, "timestamp.*range"):
                    validate_capture_manifest(candidate, schedule, calibration)

            payload = copy.deepcopy(original)
            self._replace_timestamps(
                root,
                payload,
                slot_index=0,
                stream="head",
                values=np.array([0, 1], dtype=np.int64),
                declared_range=[0, 1],
            )
            extreme = np.array([np.iinfo(np.int64).max - 1, np.iinfo(np.int64).max], dtype=np.int64)
            for stream in ("external", "state"):
                self._replace_timestamps(
                    root,
                    payload,
                    slot_index=0,
                    stream=stream,
                    values=extreme,
                    declared_range=[2**63 - 2, 2**63 - 1],
                )
            candidate = self._write_yaml(root, "timestamp-extreme-skew.yaml", payload)
            with self.assertRaisesRegex(ValueError, "skew"):
                validate_capture_manifest(candidate, schedule, calibration)

    def test_manifest_qpos_requires_finite_real_numeric_nx8_payloads(self) -> None:
        from kinesync.capture.contracts import (
            load_capture_schedule,
            validate_camera_calibration,
            validate_capture_manifest,
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "session").mkdir()
            schedule_path = root / "schedule.yaml"
            schedule_path.write_bytes(SCHEDULE_PATH.read_bytes())
            calibration_path = self._write_populated_calibration(root)
            manifest_path = self._write_manifest(root, schedule_path, calibration_path)
            schedule = load_capture_schedule(schedule_path)
            calibration = validate_camera_calibration(calibration_path)
            original = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))

            for dtype in (np.int8, np.int64, np.uint8, np.uint64, np.float32, np.float64):
                payload = copy.deepcopy(original)
                self._replace_qpos(
                    root,
                    payload,
                    slot_index=0,
                    values=np.arange(16, dtype=dtype).reshape(2, 8),
                )
                candidate = self._write_yaml(root, f"qpos-valid-{np.dtype(dtype).name}.yaml", payload)
                with self.subTest(valid_dtype=np.dtype(dtype).name):
                    validate_capture_manifest(candidate, schedule, calibration)

            invalid_payloads = (
                np.ones((2, 8), dtype=np.complex64) * (1 + 2j),
                np.ones((2, 8), dtype=np.complex128) * (1 + 2j),
                np.ones((2, 8), dtype=np.bool_),
                np.full((2, 8), "1", dtype="U1"),
                np.full((2, 8), object(), dtype=object),
            )
            for index, values in enumerate(invalid_payloads):
                payload = copy.deepcopy(original)
                self._replace_qpos(root, payload, slot_index=0, values=values)
                candidate = self._write_yaml(root, f"qpos-invalid-{index}.yaml", payload)
                with self.subTest(invalid_dtype=values.dtype.name), self.assertRaisesRegex(ValueError, "qpos"):
                    validate_capture_manifest(candidate, schedule, calibration)

    def test_manifest_validates_slot_rows_and_ids_before_membership_operations(self) -> None:
        from kinesync.capture.contracts import (
            load_capture_schedule,
            validate_camera_calibration,
            validate_capture_manifest,
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "session").mkdir()
            schedule_path = root / "schedule.yaml"
            schedule_path.write_bytes(SCHEDULE_PATH.read_bytes())
            calibration_path = self._write_populated_calibration(root)
            manifest_path = self._write_manifest(root, schedule_path, calibration_path)
            schedule = load_capture_schedule(schedule_path)
            calibration = validate_camera_calibration(calibration_path)
            original = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))

            for index, row in enumerate(([], 7, None)):
                payload = copy.deepcopy(original)
                payload["slots"][0] = row
                candidate = self._write_yaml(root, f"manifest-invalid-slot-row-{index}.yaml", payload)
                with self.subTest(slot_row=row), self.assertRaisesRegex(ValueError, "manifest slot.*mapping"):
                    validate_capture_manifest(candidate, schedule, calibration)

            for index, slot_id in enumerate(([], {}, 7, None, "")):
                payload = copy.deepcopy(original)
                payload["slots"][0]["slot_id"] = slot_id
                candidate = self._write_yaml(root, f"manifest-invalid-slot-id-{index}.yaml", payload)
                with self.subTest(slot_id=slot_id), self.assertRaisesRegex(ValueError, "manifest slot_id"):
                    validate_capture_manifest(candidate, schedule, calibration)

    def test_manifest_mixed_unknown_slot_keys_raise_controlled_value_error(self) -> None:
        from kinesync.capture.contracts import (
            load_capture_schedule,
            validate_camera_calibration,
            validate_capture_manifest,
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "session").mkdir()
            schedule_path = root / "schedule.yaml"
            schedule_path.write_bytes(SCHEDULE_PATH.read_bytes())
            calibration_path = self._write_populated_calibration(root)
            manifest_path = self._write_manifest(root, schedule_path, calibration_path)
            schedule = load_capture_schedule(schedule_path)
            calibration = validate_camera_calibration(calibration_path)
            payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
            payload["slots"][0][7] = "integer-key"
            payload["slots"][0]["unknown"] = "string-key"
            candidate = self._write_yaml(root, "manifest-mixed-unknown-slot-keys.yaml", payload)

            with self.assertRaisesRegex(ValueError, "manifest slot has unknown keys"):
                validate_capture_manifest(candidate, schedule, calibration)

    def test_manifest_requires_all_formal_slots_but_allows_no_calibration_or_reserve(self) -> None:
        from kinesync.capture.contracts import (
            load_capture_schedule,
            validate_camera_calibration,
            validate_capture_manifest,
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "session").mkdir()
            schedule_path = root / "schedule.yaml"
            schedule_path.write_bytes(SCHEDULE_PATH.read_bytes())
            calibration_path = self._write_populated_calibration(root)
            manifest_path = self._write_manifest(root, schedule_path, calibration_path)
            schedule = load_capture_schedule(schedule_path)
            calibration = validate_camera_calibration(calibration_path)
            payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))

            formal_ids = {slot.slot_id for slot in schedule.slots if slot.role == "formal"}
            payload["slots"] = [slot for slot in payload["slots"] if slot["slot_id"] in formal_ids]
            formal_only_path = self._write_yaml(root, "manifest-formal-only.yaml", payload)
            receipt = validate_capture_manifest(formal_only_path, schedule, calibration)
            self.assertEqual(receipt.slot_ids, tuple(f"formal_slot_{index:02d}" for index in range(1, 17)))

            payload["slots"] = payload["slots"][1:]
            missing_formal_path = self._write_yaml(root, "manifest-missing-formal.yaml", payload)
            with self.assertRaisesRegex(ValueError, "formal"):
                validate_capture_manifest(missing_formal_path, schedule, calibration)

    def test_optional_manifest_slots_are_a_canonical_role_preserving_subsequence(self) -> None:
        from kinesync.capture.contracts import (
            load_capture_schedule,
            validate_camera_calibration,
            validate_capture_manifest,
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "session").mkdir()
            schedule_path = root / "schedule.yaml"
            schedule_path.write_bytes(SCHEDULE_PATH.read_bytes())
            calibration_path = self._write_populated_calibration(root)
            manifest_path = self._write_manifest(root, schedule_path, calibration_path)
            schedule = load_capture_schedule(schedule_path)
            calibration = validate_camera_calibration(calibration_path)
            original = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))

            payload = copy.deepcopy(original)
            payload["slots"] = [payload["slots"][1], payload["slots"][0], *payload["slots"][12:28]]
            reordered_path = self._write_yaml(root, "manifest-reordered-optional.yaml", payload)
            with self.assertRaisesRegex(ValueError, "canonical"):
                validate_capture_manifest(reordered_path, schedule, calibration)

            payload = copy.deepcopy(original)
            reserve = next(slot for slot in payload["slots"] if slot["slot_id"] == "reserve_slot_01")
            reserve["role"] = "formal"
            promoted_path = self._write_yaml(root, "manifest-promoted-reserve.yaml", payload)
            with self.assertRaisesRegex(ValueError, "unknown"):
                validate_capture_manifest(promoted_path, schedule, calibration)

            payload = copy.deepcopy(original)
            formal = next(slot for slot in payload["slots"] if slot["slot_id"] == "formal_slot_01")
            payload["slots"].insert(13, copy.deepcopy(formal))
            duplicate_path = self._write_yaml(root, "manifest-duplicate-formal.yaml", payload)
            with self.assertRaisesRegex(ValueError, "duplicate"):
                validate_capture_manifest(duplicate_path, schedule, calibration)

    def test_cli_has_no_hardware_imports_and_writes_receipt_only_on_request(self) -> None:
        from kinesync.cli.rt10_capture_validate import main

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "session").mkdir()
            schedule_path = root / "schedule.yaml"
            schedule_path.write_bytes(SCHEDULE_PATH.read_bytes())
            calibration_path = self._write_populated_calibration(root)
            manifest_path = self._write_manifest(root, schedule_path, calibration_path)
            receipt_path = root / "receipt.json"
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                result = main([
                    "--schedule", str(schedule_path),
                    "--calibration", str(calibration_path),
                    "--manifest", str(manifest_path),
                ])
            self.assertEqual(result, 0)
            self.assertEqual(output.getvalue(), "")
            self.assertFalse(receipt_path.exists())
            result = main([
                "--schedule", str(schedule_path),
                "--calibration", str(calibration_path),
                "--manifest", str(manifest_path),
                "--receipt", str(receipt_path),
            ])
            self.assertEqual(result, 0)
            self.assertEqual(len(json.loads(receipt_path.read_text(encoding="utf-8"))["formal_trials"]), 32)

        source = (REPOSITORY_ROOT / "src" / "kinesync" / "cli" / "rt10_capture_validate.py").read_text(encoding="utf-8")
        self.assertFalse(any(name in source for name in ("rclpy", "rospy", "serial", "socket", "subprocess", "can", "mujoco")))


if __name__ == "__main__":
    unittest.main()
