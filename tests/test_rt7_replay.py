import csv
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import cv2
import numpy as np
import torch
import yaml

import kinesync.cli.rt7_replay as replay
import kinesync.visualization.synchronization as synchronization
from kinesync.cli.rt7_replay import execute_rt7_replay
from kinesync.replay.rt6_source import candidate_fingerprint
from kinesync.sync.mujoco_bridge import FKParityReport
from kinesync.sync.schema import DetectabilityProfile, JointDetectability


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


class _RenderBackend:
    def __init__(self, cameras: tuple[str, ...]):
        self.cameras = {name: object() for name in cameras}

    def loss(self, qpos: torch.Tensor, target):
        return 1.0 + qpos.square().sum(), {}

    def metrics(self, qpos: torch.Tensor, target):
        quality = 1.0 - float(qpos.abs().mean())
        return {
            "mean_mask_iou": quality,
            "mean_boundary_f1": quality,
        }

    def render(self, qpos: torch.Tensor):
        value = torch.sigmoid(2.0 - qpos.square().sum()).detach()
        return {
            name: SimpleNamespace(values=value.expand(12, 16))
            for name in self.cameras
        }


class _Bridge:
    def __init__(self, joint_names: tuple[str, ...]):
        self.qpos_addresses = tuple(range(len(joint_names)))
        self.joint_limits = {name: (-1.0, 1.0) for name in joint_names}
        self.data = SimpleNamespace(qpos=np.zeros(len(joint_names), dtype=np.float64))

    def write(self, frame):
        frame.validate_integrity()
        self.data.qpos[:] = frame.synchronized_qpos.detach().cpu().numpy()


class _Runtime:
    def __init__(self, *, factor_sha256: str, source_guard_sha256: str):
        self.factor_sha256 = factor_sha256
        self.source_guard_sha256 = source_guard_sha256
        self.profile = DetectabilityProfile(
            joint_names=("joint1", "joint2"),
            joints=(
                JointDetectability("joint1", True, 0.04, 0.03, 2, 2, 1),
                JointDetectability("joint2", True, 0.04, 0.03, 2, 2, 1),
            ),
            source_sha256="c" * 64,
            joint_order_source_sha256=self.factor_sha256,
            source_split="train",
            calibration_state_ids=("train-1",),
            algorithm_version="test-v1",
            fingerprint="d" * 64,
        )
        self.bridge = _Bridge(self.profile.joint_names)
        self.fk = object()

    def prepare_case(self, source_case):
        measured = torch.tensor(source_case.frame["measured_qpos"], dtype=torch.float32)
        visual = _RenderBackend(("head", "extra"))
        observations = {
            camera: SimpleNamespace(
                camera=camera,
                rgb=np.full((12, 16, 3), 80 + index * 30, dtype=np.uint8),
                mask=np.ones((12, 16), dtype=bool),
            )
            for index, camera in enumerate(("head", "extra"))
        }
        return SimpleNamespace(
            observations=observations,
            targets={"head": object(), "extra": object()},
            visual=visual,
            camera_backends={
                "head": _RenderBackend(("head",)),
                "extra": _RenderBackend(("extra",)),
            },
            camera_targets={"head": {"head": object()}, "extra": {"extra": object()}},
            reference_qpos=torch.zeros_like(measured),
            selected_indices=[
                self.profile.joint_names.index(name)
                for name in source_case.selected_joints
            ],
            candidate_bound=0.5,
            acceptance={"state_error_reduction_min": 0.6},
        )


class RT7ReplayTest(unittest.TestCase):
    def test_main_loads_config_before_executing_replay(self):
        with (
            mock.patch.object(
                replay, "load_config", return_value={"fixture": True}
            ) as load_config,
            mock.patch.object(
                replay, "execute_rt7_replay", return_value=Path("fixture-run")
            ) as execute,
            mock.patch.object(
                sys,
                "argv",
                [
                    "kinesync-rt7r",
                    "--config",
                    "fixture.yaml",
                    "--run-id",
                    "fixture-run-id",
                ],
            ),
            mock.patch("builtins.print"),
        ):
            replay.main()

        load_config.assert_called_once_with(Path("fixture.yaml"))
        execute.assert_called_once_with({"fixture": True}, run_id="fixture-run-id")

    def test_replay_normalizes_immutable_matrix_offset_pairs(self):
        matrix_case = SimpleNamespace(
            offsets_rad=(("joint2", -0.1), ("joint1", 0.2))
        )
        self.assertEqual(
            replay._matrix_offsets(matrix_case),
            {"joint1": 0.2, "joint2": -0.1},
        )

    def _source_fixture(self, root: Path) -> tuple[dict, dict[str, str], Path]:
        source = root / "source-rt6"
        source.mkdir()
        (source / "images").mkdir()
        (source / "videos").mkdir()

        factors = root / "factors.json"
        source_guard = root / "source-guard.json"
        profile = root / "profile.json"
        matrix = root / "matrix.yaml"
        converted = source / "converted_piper.urdf"
        conversion_manifest = source / "conversion_manifest.json"
        _write_json(factors, {"joint_names": ["joint1", "joint2"]})
        _write_json(source_guard, {"frozen": True, "source_split": "train"})
        _write_json(profile, {"fingerprint": "d" * 64})
        matrix.write_text("cases: []\n", encoding="utf-8")
        converted.write_text("<robot/>\n", encoding="utf-8")
        _write_json(conversion_manifest, {"converted": True})

        factor_sha = _sha256(factors)
        source_guard_sha = _sha256(source_guard)
        profile_sha = _sha256(profile)
        source_config = {
            "formal": {
                "frozen": True,
                "inputs": {
                    "factors": str(factors),
                    "guard": str(source_guard),
                    "profile": str(profile),
                    "matrix": str(matrix),
                },
            }
        }
        with (source / "config.yaml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(source_config, handle, sort_keys=True)

        cases = [
            {
                "case_id": "controlled-a",
                "state_id": "state-a",
                "role": "single_joint",
                "magnitude_deg": 5.0,
                "joint_name": "joint1",
                "offset_rad": 0.2,
                "measured": [0.2, 0.0],
                "candidate": [0.0, 0.0],
                "joint_correction": [-0.2, 0.0],
                "camera_corrections": {
                    "head": [-0.2, 0.0],
                    "extra": [-0.18, 0.0],
                },
                "source_guard_accepted": True,
                "candidate_success": True,
            },
            {
                "case_id": "zero-a",
                "state_id": "state-z",
                "role": "zero_control",
                "magnitude_deg": 0.0,
                "joint_name": "joint2",
                "offset_rad": 0.0,
                "measured": [0.0, 0.0],
                "candidate": [0.0, 0.0],
                "joint_correction": [0.0, 0.0],
                "camera_corrections": {
                    "head": [0.0, 0.0],
                    "extra": [0.0, 0.0],
                },
                "source_guard_accepted": False,
                "candidate_success": False,
            },
        ]
        state_rows = []
        result_rows = []
        component_rows = []
        trace_rows = []
        fingerprints = {}
        for index, case in enumerate(cases):
            fingerprint = candidate_fingerprint(
                case_id=case["case_id"],
                state_id=case["state_id"],
                candidate_qpos=torch.tensor(case["candidate"], dtype=torch.float32),
                factor_sha256=factor_sha,
                guard_sha256=source_guard_sha,
            )
            fingerprints[case["case_id"]] = fingerprint
            frame_fingerprint = (str(index + 1) * 64)[:64]
            timestamp_ns = 11 + index * 10
            source_observations = [
                {
                    "camera": "extra",
                    "frame_index": index,
                    "qpos": [0.0, 0.0],
                    "sample_id": f"extra-{index}",
                    "timestamp_ns": timestamp_ns - 1,
                },
                {
                    "camera": "head",
                    "frame_index": index,
                    "qpos": [0.0, 0.0],
                    "sample_id": f"head-{index}",
                    "timestamp_ns": timestamp_ns + 1,
                },
            ]
            source_state = case["candidate"] if case["source_guard_accepted"] else case["measured"]
            state_rows.append(
                {
                    "case_id": case["case_id"],
                    "state_id": case["state_id"],
                    "timestamp_ns": timestamp_ns,
                    "joint_names": ["joint1", "joint2"],
                    "measured_qpos": case["measured"],
                    "candidate_qpos": case["candidate"],
                    "synchronized_qpos": source_state,
                    "mujoco_qpos": source_state,
                    "gaussian_render_qpos": source_state,
                    "gaussian_render_receipt": {
                        "qpos": source_state,
                        "camera_render_sha256": {"head": "e" * 64, "extra": "f" * 64},
                        "camera_shapes": {"head": [12, 16], "extra": [12, 16]},
                    },
                    "source_fusion_policy": "paired_qpos_mean_v1",
                    "source_observations": source_observations,
                    "source_timestamp_span_ns": 2,
                    "decisions": [
                        {
                            "joint_name": case["joint_name"],
                            "accepted": case["source_guard_accepted"],
                            "reason": "accepted" if case["source_guard_accepted"] else "guard_rejected",
                            "correction_rad": case["joint_correction"][0]
                            if case["joint_name"] == "joint1"
                            else case["joint_correction"][1],
                            "signal_floor_rad": None,
                            "margin_rad": 0.1 if case["source_guard_accepted"] else -0.1,
                        }
                    ],
                    "factor_fingerprint": factor_sha,
                    "guard_fingerprint": source_guard_sha,
                    "profile_fingerprint": "d" * 64,
                    "frame_fingerprint": frame_fingerprint,
                }
            )
            for method in ("rt2f_unguarded", "rt3c_guarded", "rt6dm_synchronized"):
                result_rows.append(
                    {
                        "case_id": case["case_id"],
                        "state_id": case["state_id"],
                        "split": "validation",
                        "role": case["role"],
                        "magnitude_deg": case["magnitude_deg"],
                        "method": method,
                        "candidate_fingerprint": fingerprint,
                        "frame_fingerprint": frame_fingerprint,
                        "guard_accepted": str(case["source_guard_accepted"]),
                        "guard_reason": "accepted" if case["source_guard_accepted"] else "visual_gain",
                        "any_component_accepted": str(case["source_guard_accepted"]),
                        "state_error_reduction": 1.0 if case["candidate_success"] else 0.0,
                        "recovery_success": str(case["candidate_success"]),
                        "mask_iou_before": 0.8,
                        "mask_iou_after": 1.0,
                        "mask_iou_gain": 0.2,
                        "boundary_f1_before": 0.8,
                        "boundary_f1_after": 1.0,
                        "boundary_f1_gain": 0.2,
                        "evidence_finite": "True",
                        "visual_gain_ratio": 0.1 if case["candidate_success"] else 0.0,
                        "gradient_cosine": 1.0,
                        "correction_cosine": 1.0,
                        "relative_correction_disagreement": 0.1,
                    }
                )
            component_rows.append(
                {
                    "case_id": case["case_id"],
                    "state_id": case["state_id"],
                    "role": case["role"],
                    "magnitude_deg": case["magnitude_deg"],
                    "joint_name": case["joint_name"],
                    "offset_rad": case["offset_rad"],
                    "candidate_correction_rad": case["joint_correction"][0]
                    if case["joint_name"] == "joint1"
                    else case["joint_correction"][1],
                    "state_error_reduction": 1.0 if case["candidate_success"] else None,
                    "physical_success": str(case["candidate_success"]),
                    "trial_visual_success": str(case["candidate_success"]),
                    "candidate_success": str(case["candidate_success"]),
                    "accepted": str(case["source_guard_accepted"]),
                    "decision_reason": "accepted" if case["source_guard_accepted"] else "guard_rejected",
                    "signal_floor_rad": None,
                    "margin_rad": 0.1 if case["source_guard_accepted"] else -0.1,
                    "frame_fingerprint": frame_fingerprint,
                }
            )
            for source_name, correction in {
                "rt2_joint": case["joint_correction"],
                "rt2_head": case["camera_corrections"]["head"],
                "rt2_extra": case["camera_corrections"]["extra"],
            }.items():
                trace_rows.append(
                    {
                        "case_id": case["case_id"],
                        "state_id": case["state_id"],
                        "candidate_source": source_name,
                        "step": 0,
                        "correction_joint1": correction[0],
                        "correction_joint2": correction[1],
                    }
                )

        def write_csv(path: Path, rows: list[dict]) -> None:
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

        with (source / "state_frames.jsonl").open("w", encoding="utf-8") as handle:
            for row in state_rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        write_csv(source / "results.csv", result_rows)
        write_csv(source / "component_results.csv", component_rows)
        write_csv(source / "trace.csv", trace_rows)
        with (source / "optimizer_executions.jsonl").open("w", encoding="utf-8") as handle:
            for row in trace_rows:
                handle.write(
                    json.dumps(
                        {
                            "case_id": row["case_id"],
                            "candidate_source": row["candidate_source"],
                            "completed": True,
                            "trace_step_count": 1,
                        }
                    )
                    + "\n"
                )
        metrics = {
            "formal_evidence": True,
            "trial_count": len(cases),
            "unique_state_count": len(cases),
            "candidate_fingerprints": fingerprints,
            "source_hashes": {
                "factors_sha256": factor_sha,
                "guard_sha256": source_guard_sha,
                "profile_sha256": profile_sha,
                "profile_fingerprint": "d" * 64,
                "converted_urdf": _sha256(converted),
                "conversion_manifest": _sha256(conversion_manifest),
            },
        }
        _write_json(source / "metrics.json", metrics)
        _write_json(
            source / "manifest.json",
            {
                "schema_version": 1,
                "run_id": "fixture-source-rt6",
                "formal_evidence": True,
                "source_hashes": metrics["source_hashes"],
            },
        )

        rt7_guard = root / "rt7-guard.json"
        _write_json(
            rt7_guard,
            {
                "schema_version": 1,
                "frozen": True,
                "formal_evidence": False,
                "source_split": "train",
                "threshold_generation": "event_gain_v1",
                "frozen_rt6_guard_sha256": source_guard_sha,
                "calibration_fingerprint": "9" * 64,
                "thresholds": {
                    "min_visual_gain_ratio": 0.01,
                    "min_gradient_cosine": -1.0,
                    "min_correction_cosine": -1.0,
                    "max_relative_correction_disagreement": 1.0,
                },
            },
        )
        hashed = {
            "source_rt6_config_sha256": _sha256(source / "config.yaml"),
            "source_rt6_manifest_sha256": _sha256(source / "manifest.json"),
            "source_rt6_metrics_sha256": _sha256(source / "metrics.json"),
            "source_rt6_state_frames_sha256": _sha256(source / "state_frames.jsonl"),
            "source_rt6_results_sha256": _sha256(source / "results.csv"),
            "source_rt6_component_results_sha256": _sha256(source / "component_results.csv"),
            "source_rt6_trace_sha256": _sha256(source / "trace.csv"),
            "source_rt6_optimizer_executions_sha256": _sha256(
                source / "optimizer_executions.jsonl"
            ),
            "rt7_guard_sha256": _sha256(rt7_guard),
        }
        config = {
            "experiment": "rt7-replay-test",
            "runs_root": str(root / "runs"),
            "device": "cpu",
            "inputs": {"source_rt6_run": str(source), "rt7_guard": str(rt7_guard)},
            "provenance": {
                "source_run_id": "fixture-source-rt6",
                "expected_trial_count": 2,
                "expected_unique_state_count": 2,
                "expected_hashes": hashed,
            },
            "media": {"frame_count": 2, "fps": 1},
            "diagnostic_targets": {
                "successful_candidate_retention": {"minimum_numerator": 1, "denominator": 1},
                "committed_precision": 1.0,
                "zero_false_updates": {"maximum_numerator": 0, "denominator": 1},
                "matched_success": {"minimum_numerator": 1, "denominator": 2},
            },
        }
        return config, fingerprints, source

    def _runtime_patch(self, source: Path):
        source_config = yaml.safe_load((source / "config.yaml").read_text(encoding="utf-8"))
        factor_path = Path(source_config["formal"]["inputs"]["factors"])
        guard_path = Path(source_config["formal"]["inputs"]["guard"])
        return mock.patch.object(
            replay,
            "_build_replay_runtime",
            return_value=_Runtime(
                factor_sha256=_sha256(factor_path),
                source_guard_sha256=_sha256(guard_path),
            ),
        )

    def _preflight_patch(self):
        return mock.patch.object(
            replay,
            "verify_frozen_rt6_source",
            return_value=SimpleNamespace(
                asset_paths={},
                asset_receipt={"files": {}, "verified_source_hashes": {}},
                verified_source_hashes={},
            ),
        )

    def _parity_patch(self):
        return mock.patch.object(
            replay,
            "compare_fk_parity",
            return_value=FKParityReport(
                state_count=2,
                compared_body_names=("link1",),
                max_translation_error_m=0.0,
                max_rotation_abs_error=0.0,
                max_qpos_error=0.0,
                passed=True,
            ),
        )

    def test_replay_is_counterfactual_preserves_candidate_identity_and_avoids_optimizer(self):
        with tempfile.TemporaryDirectory() as directory:
            config, fingerprints, source = self._source_fixture(Path(directory))
            with (
                self._preflight_patch(),
                self._runtime_patch(source),
                self._parity_patch(),
                mock.patch(
                    "kinesync.correction.joint_offset.JointOffsetRecovery.recover",
                    side_effect=AssertionError("the optimizer must not run during replay"),
                ),
            ):
                run = execute_rt7_replay(config, run_id="rt7-replay-test")

            metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
            self.assertIs(metrics["formal_evidence"], False)
            self.assertEqual(metrics["evidence_role"], "development_counterfactual")
            self.assertEqual(metrics["formal_decision"], "not_applicable")
            self.assertEqual(metrics["source_candidate_fingerprints"], fingerprints)

            with (run / "results.csv").open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(
                {row["method"] for row in rows},
                {
                    "measured_no_correction",
                    "rt2f_unguarded",
                    "rt3c_guarded",
                    "rt6dm_synchronized",
                    "rt7cv_component_verified",
                },
            )
            self.assertEqual(len(rows), 10)
            for row in rows:
                self.assertEqual(row["candidate_fingerprint"], fingerprints[row["case_id"]])

            state_frames = [
                json.loads(line)
                for line in (run / "state_frames.jsonl").read_text(encoding="utf-8").splitlines()
                if line
            ]
            self.assertEqual(len(state_frames), 2)
            for frame in state_frames:
                self.assertEqual(frame["synchronized_qpos"], frame["mujoco_qpos"])
                self.assertEqual(frame["synchronized_qpos"], frame["gaussian_render_qpos"])
                self.assertEqual(frame["source_candidate_fingerprint"], fingerprints[frame["case_id"]])

            evidence = [
                json.loads(line)
                for line in (run / "component_evidence.jsonl").read_text(encoding="utf-8").splitlines()
                if line
            ]
            receipts = [
                json.loads(line)
                for line in (run / "component_receipts.jsonl").read_text(encoding="utf-8").splitlines()
                if line
            ]
            self.assertEqual(len(evidence), 2)
            self.assertEqual(len(receipts), 2)
            self.assertTrue(all(set(row["render_receipt"]["camera_render_sha256"]) == {"head", "extra"} for row in receipts))

            image = cv2.imread(str(run / "images" / "rt7_component_replay.png"), cv2.IMREAD_COLOR)
            self.assertIsNotNone(image)
            self.assertEqual(image.shape, (1080, 1920, 3))
            capture = cv2.VideoCapture(str(run / "videos" / "rt7_component_replay.mp4"))
            self.addCleanup(capture.release)
            self.assertTrue(capture.isOpened())
            self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), 1920)
            self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)), 1080)

    def test_replay_media_draws_rt7_component_verified_semantics(self):
        with tempfile.TemporaryDirectory() as directory:
            config, _, source = self._source_fixture(Path(directory))
            drawn_text = []
            fitted_text = synchronization._fitted_text

            def capture_text(*args, **kwargs):
                drawn_text.append(str(args[1] if len(args) > 1 else kwargs["text"]))
                return fitted_text(*args, **kwargs)

            with (
                self._preflight_patch(),
                self._runtime_patch(source),
                self._parity_patch(),
                mock.patch.object(
                    synchronization, "_fitted_text", side_effect=capture_text
                ),
            ):
                run = execute_rt7_replay(config, run_id="rt7-media-semantics")

            text = "\n".join(drawn_text)
            self.assertIn("RT7 COMPONENT-VERIFIED", text)
            self.assertNotIn("RT6", text)
            image = cv2.imread(str(run / "images" / "rt7_component_replay.png"), cv2.IMREAD_COLOR)
            self.assertIsNotNone(image)
            capture = cv2.VideoCapture(str(run / "videos" / "rt7_component_replay.mp4"))
            self.addCleanup(capture.release)
            self.assertTrue(capture.isOpened())
            self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), 1920)
            self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)), 1080)
            frames = []
            while True:
                ok, video_frame = capture.read()
                if not ok:
                    break
                frames.append(video_frame)
            self.assertEqual(len(frames), config["media"]["frame_count"])
            lower_start = synchronization._HEADER_HEIGHT + synchronization._PANEL_HEIGHT
            lower_end = lower_start + synchronization._PANEL_HEIGHT
            for index, video_frame in enumerate(frames):
                with self.subTest(frame_index=index):
                    self.assertGreater(float(np.std(video_frame[lower_start:lower_end])), 10.0)

            context_lines = {
                "A controlled-a / state-a | ": [],
                "B zero-a / state-z | ": [],
            }
            for text_line in drawn_text:
                for prefix, lines in context_lines.items():
                    if text_line.startswith(prefix):
                        lines.append(text_line)
            for prefix, lines in context_lines.items():
                with self.subTest(context_prefix=prefix):
                    self.assertEqual(len(lines), config["media"]["frame_count"] + 1)
                    self.assertTrue(all("| guard=" in line for line in lines))
                    self.assertTrue(all("| accepted=" in line for line in lines))
                    self.assertTrue(all("| rejected=" in line for line in lines))

    def test_replay_preserves_every_shared_source_result_cell(self):
        with tempfile.TemporaryDirectory() as directory:
            config, _, source = self._source_fixture(Path(directory))
            with self._preflight_patch(), self._runtime_patch(source), self._parity_patch():
                run = execute_rt7_replay(config, run_id="rt7-inherited-rows")

            with (source / "results.csv").open(encoding="utf-8", newline="") as handle:
                source_rows = {
                    (row["case_id"], row["method"]): row
                    for row in csv.DictReader(handle)
                }
            with (run / "results.csv").open(encoding="utf-8", newline="") as handle:
                replay_rows = {
                    (row["case_id"], row["method"]): row
                    for row in csv.DictReader(handle)
                }

            inherited = [
                key
                for key in source_rows
                if key[1] in {"rt2f_unguarded", "rt3c_guarded", "rt6dm_synchronized"}
            ]
            self.assertEqual(len(inherited), 6)
            for key in inherited:
                with self.subTest(case_id=key[0], method=key[1]):
                    source_row = source_rows[key]
                    replay_row = replay_rows[key]
                    self.assertEqual(
                        {field: replay_row[field] for field in source_row}, source_row
                    )
                    self.assertEqual(replay_row["replay_origin"], "source_rt6_inherited")

            metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(
                metrics["method_summaries"]["rt2f_unguarded"]["trial_success_rate"],
                0.5,
            )

    def test_replay_preflight_rejects_before_run_artifacts_create(self):
        with tempfile.TemporaryDirectory() as directory:
            config, _, source = self._source_fixture(Path(directory))
            run_id = "rejected-before-artifacts"
            with (
                mock.patch.object(
                    replay,
                    "verify_frozen_rt6_source",
                    side_effect=ValueError("frozen RT6 source hashes changed"),
                    create=True,
                ),
                mock.patch.object(
                    replay.RunArtifacts,
                    "create",
                    side_effect=AssertionError("preflight must run before output creation"),
                ),
                self.assertRaisesRegex(ValueError, "frozen RT6 source hashes"),
            ):
                execute_rt7_replay(config, run_id=run_id)
            self.assertFalse((Path(config["runs_root"]) / run_id).exists())

    def test_replay_rejects_malformed_source_provenance_before_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            config, _, source = self._source_fixture(Path(directory))
            path = source / "state_frames.jsonl"
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            rows[0]["mujoco_qpos"][0] = 0.75
            path.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
                encoding="utf-8",
            )
            config["provenance"]["expected_hashes"]["source_rt6_state_frames_sha256"] = _sha256(path)
            with mock.patch.object(
                replay,
                "_build_replay_runtime",
                side_effect=AssertionError("invalid source provenance reached runtime construction"),
            ), self.assertRaisesRegex(ValueError, "source RT6 qpos routing"):
                execute_rt7_replay(config, run_id="must-not-exist")


if __name__ == "__main__":
    unittest.main()
