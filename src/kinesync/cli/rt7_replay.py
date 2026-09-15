"""Replay RT7 component verification on immutable RT6 formal states."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np
import torch

from kinesync.cli.rt1_real_observation import _anchor_indices, _real_dataset
from kinesync.cli.rt3_evaluate import _outcome
from kinesync.config import load_config
from kinesync.data.route_a import RouteAAssets, RouteAPaths
from kinesync.factorization import FrozenCameraObservationBackend
from kinesync.geometry.urdf import TorchURDFKinematics
from kinesync.guard.schema import GuardThresholds
from kinesync.observation.real_route_a import prepare_real_targets
from kinesync.replay.rt6_source import (
    FrozenRT6SourcePreflight,
    candidate_fingerprint as bound_candidate_fingerprint,
    formal_renderer_factory,
    gaussian_render_receipt,
    load_frozen_rt6_config_snapshot,
    sha256_file,
    source_frame_provenance,
    verify_frozen_rt6_source,
)
from kinesync.replay.rt7_policy import (
    apply_component_verified_policy,
    inherit_source_result_row,
)
from kinesync.runs.artifacts import RunArtifacts
from kinesync.sync.mujoco_bridge import MujocoStateBridge, compare_fk_parity
from kinesync.visualization.synchronization import (
    SynchronizationPresentation,
    write_synchronization_comparison,
    write_synchronization_video,
)


_SOURCE_METHODS = (
    "rt2f_unguarded",
    "rt3c_guarded",
    "rt6dm_synchronized",
)
_REPLAY_METHODS = (
    "measured_no_correction",
    "rt2f_unguarded",
    "rt3c_guarded",
    "rt6dm_synchronized",
    "rt7cv_component_verified",
)
_EXPECTED_SOURCE_HASHES = {
    "source_rt6_config_sha256": "config.yaml",
    "source_rt6_manifest_sha256": "manifest.json",
    "source_rt6_metrics_sha256": "metrics.json",
    "source_rt6_state_frames_sha256": "state_frames.jsonl",
    "source_rt6_results_sha256": "results.csv",
    "source_rt6_component_results_sha256": "component_results.csv",
    "source_rt6_trace_sha256": "trace.csv",
    "source_rt6_optimizer_executions_sha256": "optimizer_executions.jsonl",
}
_RT7_COMPONENT_PRESENTATION = SynchronizationPresentation(
    comparison_title="KineSync-GS RT7 component-verified state evidence",
    comparison_subtitle=(
        "Paired head/extra cameras show target, measured, source candidate, and "
        "the RT7 component-verified committed state."
    ),
    candidate_label="RT2 CANDIDATE",
    committed_label="RT7 COMPONENT-VERIFIED",
    video_title="KineSync-GS RT7 component-verification diagnostic",
    video_subtitle="Diagnostic interpolation for RT7 component verification",
    stack_video_contexts=True,
)


@dataclass(frozen=True)
class _SourceCase:
    case_id: str
    state_id: str
    role: str
    magnitude_deg: float
    frame: Mapping[str, Any]
    result_rows: Mapping[str, Mapping[str, str]]
    component_rows: Mapping[str, Mapping[str, str]]
    selected_joints: tuple[str, ...]
    offsets_rad: Mapping[str, float]
    camera_corrections: Mapping[str, tuple[float, ...]]
    candidate_fingerprint: str


@dataclass(frozen=True)
class _SourceReplay:
    run_path: Path
    config: Mapping[str, Any]
    manifest: Mapping[str, Any]
    metrics: Mapping[str, Any]
    cases: tuple[_SourceCase, ...]
    factor_path: Path
    source_guard_path: Path
    profile_path: Path
    matrix_path: Path
    factor_sha256: str
    source_guard_sha256: str
    profile_sha256: str
    profile_fingerprint: str
    file_hashes: Mapping[str, str]
    asset_paths: Mapping[str, Path]


@dataclass(frozen=True)
class _CaseRuntime:
    observations: Mapping[str, Any]
    targets: Mapping[str, Any]
    visual: Any
    camera_backends: Mapping[str, Any]
    camera_targets: Mapping[str, Any]
    reference_qpos: torch.Tensor
    selected_indices: tuple[int, ...]
    candidate_bound: float
    acceptance: Mapping[str, Any]


def _matrix_offsets(matrix_case: Any) -> dict[str, float]:
    """Normalize the RT6 matrix's immutable ``(joint, offset)`` pairs."""

    try:
        offsets = {str(name): float(value) for name, value in matrix_case.offsets_rad}
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError("frozen RT6 matrix has invalid offsets_rad") from error
    if not offsets or not all(math.isfinite(value) for value in offsets.values()):
        raise ValueError("frozen RT6 matrix has invalid offsets_rad")
    return offsets


class _NativeReplayRuntime:
    def __init__(
        self,
        *,
        profile,
        factor_sha256: str,
        source_guard_sha256: str,
        bridge: MujocoStateBridge,
        fk: TorchURDFKinematics,
        dataset,
        renderer_for,
        camera_factors: Mapping[str, torch.Tensor],
        shared_joint: torch.Tensor,
        boundary_weight: float,
        candidate_bound: float,
        acceptance: Mapping[str, Any],
        max_source_timestamp_delta_ns: int,
        output_size: tuple[int, int],
        cases: Mapping[str, Any],
        device: torch.device,
    ) -> None:
        self.profile = profile
        self.factor_sha256 = factor_sha256
        self.source_guard_sha256 = source_guard_sha256
        self.bridge = bridge
        self.fk = fk
        self.dataset = dataset
        self.renderer_for = renderer_for
        self.camera_factors = dict(camera_factors)
        self.shared_joint = shared_joint
        self.boundary_weight = boundary_weight
        self.candidate_bound = candidate_bound
        self.acceptance = dict(acceptance)
        self.max_source_timestamp_delta_ns = max_source_timestamp_delta_ns
        self.output_size = output_size
        self.cases = dict(cases)
        self.device = device

    def prepare_case(self, source_case: _SourceCase) -> _CaseRuntime:
        try:
            matrix_case = self.cases[source_case.case_id]
        except KeyError as error:
            raise ValueError(f"source RT6 matrix lacks case: {source_case.case_id}") from error
        if (
            matrix_case.state_id != source_case.state_id
            or matrix_case.role != source_case.role
            or not math.isclose(
                float(matrix_case.magnitude_deg),
                float(source_case.magnitude_deg),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise ValueError("source RT6 case metadata does not match frozen matrix")
        matrix_offsets = _matrix_offsets(matrix_case)
        if tuple(name for name in self.profile.joint_names if name in matrix_offsets) != source_case.selected_joints:
            raise ValueError("source RT6 selected joints do not match frozen matrix")
        if set(matrix_offsets) != set(source_case.offsets_rad) or any(
            not math.isclose(
                matrix_offsets[name], source_case.offsets_rad[name], rel_tol=0.0, abs_tol=1e-12
            )
            for name in matrix_offsets
        ):
            raise ValueError("source RT6 component offsets do not match frozen matrix")

        cameras = list(matrix_case.cameras)
        if set(cameras) != {"head", "extra"}:
            raise ValueError("RT7 development replay requires paired head/extra cameras")
        observations = self.dataset.load_state(source_case.state_id, cameras=cameras)
        if any(item.split != "validation" for item in observations.values()):
            raise ValueError("RT7 development replay loaded a non-validation observation")
        timestamp_ns, provenance = source_frame_provenance(
            observations,
            max_timestamp_delta_ns=self.max_source_timestamp_delta_ns,
        )
        if (
            timestamp_ns != source_case.frame["timestamp_ns"]
            or provenance != source_case.frame["source_observations"]
        ):
            raise ValueError("source RT6 observation provenance no longer matches frozen state")

        height, width = self.output_size
        targets = prepare_real_targets(
            observations,
            output_size=(height, width),
            device=self.device,
        )
        reference = torch.as_tensor(
            np.mean(np.stack([item.qpos for item in observations.values()]), axis=0),
            dtype=torch.float32,
            device=self.device,
        )
        calibrated_reference = reference + self.shared_joint
        injected = torch.zeros_like(calibrated_reference)
        for name, offset in source_case.offsets_rad.items():
            injected[self.profile.joint_names.index(name)] = offset
        expected_measured = calibrated_reference + injected
        source_measured = torch.as_tensor(
            source_case.frame["measured_qpos"], dtype=torch.float32, device=self.device
        )
        if not torch.equal(expected_measured, source_measured):
            raise ValueError("source RT6 measured qpos does not match frozen Route-A state")

        renderer = self.renderer_for(cameras)
        joint_twists = {name: self.camera_factors[name] for name in cameras}
        visual = FrozenCameraObservationBackend(
            renderer,
            camera_twists=joint_twists,
            boundary_weight=self.boundary_weight,
        )
        camera_targets = {name: {name: targets[name]} for name in cameras}
        camera_backends = {}
        for camera in cameras:
            camera_renderer = self.renderer_for([camera])
            twists = {camera: self.camera_factors[camera]}
            camera_backends[camera] = FrozenCameraObservationBackend(
                camera_renderer,
                camera_twists=twists,
                boundary_weight=self.boundary_weight,
            )
        return _CaseRuntime(
            observations=observations,
            targets=targets,
            visual=visual,
            camera_backends=camera_backends,
            camera_targets=camera_targets,
            reference_qpos=calibrated_reference,
            selected_indices=tuple(
                self.profile.joint_names.index(name) for name in source_case.selected_joints
            ),
            candidate_bound=self.candidate_bound,
            acceptance=self.acceptance,
        )


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write empty replay rows")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write empty replay JSONL")
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
                + "\n"
            )


def _read_json(path: Path, description: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"malformed {description}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"malformed {description}")
    return payload


def _read_jsonl(path: Path, description: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ValueError(f"missing {description}") from error
    for line in lines:
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"malformed {description}") from error
        if not isinstance(row, dict):
            raise ValueError(f"malformed {description}")
        rows.append(row)
    if not rows:
        raise ValueError(f"{description} must be nonempty")
    return rows


def _read_csv(path: Path, description: str) -> list[dict[str, str]]:
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as error:
        raise ValueError(f"missing {description}") from error
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"{description} must be nonempty")
    return rows


def _parse_bool(value: Any, *, field: str) -> bool:
    if value is True or value == "True":
        return True
    if value is False or value == "False":
        return False
    raise ValueError(f"invalid boolean {field}: {value!r}")


def _finite_vector(value: Any, *, field: str) -> list[float]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be a nonempty qpos vector")
    try:
        vector = [float(item) for item in value]
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be a finite qpos vector") from error
    if not all(math.isfinite(item) for item in vector):
        raise ValueError(f"{field} must be a finite qpos vector")
    return vector


def _sha256_digest(value: Any, *, field: str) -> str:
    digest = str(value)
    if len(digest) != 64:
        raise ValueError(f"{field} must be a SHA-256 digest")
    try:
        int(digest, 16)
    except ValueError as error:
        raise ValueError(f"{field} must be a SHA-256 digest") from error
    return digest


def _optional_float(value: Any, *, field: str) -> float | None:
    if value in (None, "", "None"):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid numeric {field}") from error
    if not math.isfinite(number):
        raise ValueError(f"invalid numeric {field}")
    return number


def _required_input_path(inputs: Mapping[str, Any], name: str) -> Path:
    path = Path(str(inputs.get(name, ""))).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"RT7 replay input does not exist: {name}={path}")
    return path


def _source_file_hashes(run_path: Path, rt7_guard_path: Path) -> dict[str, str]:
    hashes = {
        key: sha256_file(run_path / filename)
        for key, filename in _EXPECTED_SOURCE_HASHES.items()
    }
    hashes["rt7_guard_sha256"] = sha256_file(rt7_guard_path)
    return hashes


def _validate_source_frame(
    frame: Mapping[str, Any],
    *,
    factor_sha256: str,
    source_guard_sha256: str,
    profile_fingerprint: str,
) -> None:
    case_id = str(frame.get("case_id", ""))
    state_id = str(frame.get("state_id", ""))
    if not case_id or not state_id:
        raise ValueError("source RT6 state frame requires case and state IDs")
    names = frame.get("joint_names")
    if not isinstance(names, list) or not names or len(set(names)) != len(names):
        raise ValueError("source RT6 state frame has invalid joint names")
    names = [str(name) for name in names]
    measured = _finite_vector(frame.get("measured_qpos"), field="source RT6 measured_qpos")
    candidate = _finite_vector(frame.get("candidate_qpos"), field="source RT6 candidate_qpos")
    synchronized = _finite_vector(
        frame.get("synchronized_qpos"), field="source RT6 synchronized_qpos"
    )
    mujoco = _finite_vector(frame.get("mujoco_qpos"), field="source RT6 mujoco_qpos")
    gaussian = _finite_vector(
        frame.get("gaussian_render_qpos"), field="source RT6 gaussian_render_qpos"
    )
    if not all(len(vector) == len(names) for vector in (measured, candidate, synchronized, mujoco, gaussian)):
        raise ValueError("source RT6 qpos vectors do not match joint names")
    if synchronized != mujoco or synchronized != gaussian:
        raise ValueError("source RT6 qpos routing contract failed")
    receipt = frame.get("gaussian_render_receipt")
    if not isinstance(receipt, Mapping) or receipt.get("qpos") != synchronized:
        raise ValueError("source RT6 Gaussian receipt contract failed")
    camera_hashes = receipt.get("camera_render_sha256")
    if not isinstance(camera_hashes, Mapping) or set(camera_hashes) != {"head", "extra"}:
        raise ValueError("source RT6 Gaussian receipt contract failed")
    for digest in camera_hashes.values():
        _sha256_digest(digest, field="source RT6 Gaussian render digest")
    source = frame.get("source_observations")
    if not isinstance(source, list) or len(source) != 2 or {
        item.get("camera") for item in source if isinstance(item, Mapping)
    } != {"head", "extra"}:
        raise ValueError("source RT6 observation provenance contract failed")
    timestamps: list[int] = []
    for item in source:
        if not isinstance(item, Mapping):
            raise ValueError("source RT6 observation provenance contract failed")
        if not item.get("sample_id") or not isinstance(item.get("frame_index"), int):
            raise ValueError("source RT6 observation provenance contract failed")
        if isinstance(item.get("timestamp_ns"), bool) or not isinstance(item.get("timestamp_ns"), int):
            raise ValueError("source RT6 observation provenance contract failed")
        if len(_finite_vector(item.get("qpos"), field="source observation qpos")) != len(names):
            raise ValueError("source RT6 observation provenance contract failed")
        timestamps.append(int(item["timestamp_ns"]))
    if (
        frame.get("source_fusion_policy") != "paired_qpos_mean_v1"
        or frame.get("timestamp_ns") != sum(timestamps) // len(timestamps)
        or frame.get("source_timestamp_span_ns") != max(timestamps) - min(timestamps)
    ):
        raise ValueError("source RT6 observation provenance contract failed")
    decisions = frame.get("decisions")
    if not isinstance(decisions, list) or not decisions:
        raise ValueError("source RT6 state frame requires component decisions")
    decision_names = [str(item.get("joint_name", "")) for item in decisions if isinstance(item, Mapping)]
    if len(decision_names) != len(decisions) or len(set(decision_names)) != len(decision_names):
        raise ValueError("source RT6 state frame has invalid component decisions")
    if not set(decision_names).issubset(set(names)):
        raise ValueError("source RT6 component decisions reference unknown joints")
    if frame.get("factor_fingerprint") != factor_sha256:
        raise ValueError("source RT6 factor fingerprint mismatch")
    if frame.get("guard_fingerprint") != source_guard_sha256:
        raise ValueError("source RT6 guard fingerprint mismatch")
    if frame.get("profile_fingerprint") != profile_fingerprint:
        raise ValueError("source RT6 profile fingerprint mismatch")
    _sha256_digest(frame.get("frame_fingerprint"), field="source RT6 frame fingerprint")


def _last_trace_corrections(
    trace_rows: Iterable[Mapping[str, str]],
    *,
    joint_names: Sequence[str],
    expected_case_ids: set[str],
) -> dict[str, dict[str, tuple[float, ...]]]:
    grouped: dict[tuple[str, str], list[Mapping[str, str]]] = defaultdict(list)
    for row in trace_rows:
        case_id = str(row.get("case_id", ""))
        source = str(row.get("candidate_source", ""))
        if source in {"rt2_joint", "rt2_head", "rt2_extra"}:
            grouped[(case_id, source)].append(row)
    corrections: dict[str, dict[str, tuple[float, ...]]] = {}
    for case_id in expected_case_ids:
        per_case: dict[str, tuple[float, ...]] = {}
        for source in ("rt2_joint", "rt2_head", "rt2_extra"):
            rows = grouped.get((case_id, source), [])
            if not rows:
                raise ValueError("source RT6 trace lacks an optimizer candidate receipt")
            try:
                maximum = max(int(row["step"]) for row in rows)
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError("source RT6 trace has invalid optimizer steps") from error
            final = [row for row in rows if int(row["step"]) == maximum]
            if len(final) != 1:
                raise ValueError("source RT6 trace has ambiguous final optimizer state")
            values = []
            for joint_name in joint_names:
                try:
                    value = float(final[0][f"correction_{joint_name}"])
                except (KeyError, TypeError, ValueError) as error:
                    raise ValueError("source RT6 trace lacks candidate corrections") from error
                if not math.isfinite(value):
                    raise ValueError("source RT6 trace has nonfinite candidate corrections")
                values.append(value)
            per_case[source] = tuple(values)
        corrections[case_id] = per_case
    return corrections


def _load_source_replay(config: Mapping[str, Any]) -> tuple[_SourceReplay, Path, dict[str, Any], str]:
    try:
        inputs = config["inputs"]
        provenance = config["provenance"]
    except (KeyError, TypeError) as error:
        raise ValueError("RT7 replay config requires inputs and provenance") from error
    if not isinstance(inputs, Mapping) or not isinstance(provenance, Mapping):
        raise ValueError("RT7 replay config requires mapping inputs and provenance")
    run_path = Path(str(inputs.get("source_rt6_run", ""))).expanduser().resolve()
    if not run_path.is_dir():
        raise FileNotFoundError(f"RT7 replay source run does not exist: {run_path}")
    rt7_guard_path = _required_input_path(inputs, "rt7_guard")
    expected_hashes = provenance.get("expected_hashes")
    if not isinstance(expected_hashes, Mapping) or set(expected_hashes) != {
        *_EXPECTED_SOURCE_HASHES,
        "rt7_guard_sha256",
    }:
        raise ValueError("RT7 replay config requires complete immutable source hashes")
    actual_hashes = _source_file_hashes(run_path, rt7_guard_path)
    if {str(name): str(value) for name, value in expected_hashes.items()} != actual_hashes:
        raise ValueError("RT7 replay immutable source hashes do not match")

    config_path = run_path / "config.yaml"
    source_config = load_frozen_rt6_config_snapshot(config_path)
    manifest = _read_json(run_path / "manifest.json", "source RT6 manifest")
    metrics = _read_json(run_path / "metrics.json", "source RT6 metrics")
    if (
        str(manifest.get("run_id", "")) != str(provenance.get("source_run_id", ""))
        or manifest.get("formal_evidence") is not True
        or metrics.get("formal_evidence") is not True
    ):
        raise ValueError("RT7 replay source must be an immutable formal RT6 run")
    expected_trial_count = int(provenance.get("expected_trial_count", -1))
    expected_unique_state_count = int(provenance.get("expected_unique_state_count", -1))
    if expected_trial_count <= 0 or expected_unique_state_count <= 0:
        raise ValueError("RT7 replay provenance requires positive source counts")
    if (
        int(metrics.get("trial_count", -1)) != expected_trial_count
        or int(metrics.get("unique_state_count", -1)) != expected_unique_state_count
    ):
        raise ValueError("RT7 replay source metrics count contract failed")

    try:
        frozen_inputs = source_config["formal"]["inputs"]
    except (KeyError, TypeError) as error:
        raise ValueError("source RT6 config lacks frozen inputs") from error
    if not isinstance(frozen_inputs, Mapping):
        raise ValueError("source RT6 config lacks frozen inputs")
    factor_path = _required_input_path(frozen_inputs, "factors")
    source_guard_path = _required_input_path(frozen_inputs, "guard")
    profile_path = _required_input_path(frozen_inputs, "profile")
    matrix_path = _required_input_path(frozen_inputs, "matrix")
    factor_sha256 = sha256_file(factor_path)
    source_guard_sha256 = sha256_file(source_guard_path)
    profile_sha256 = sha256_file(profile_path)
    source_hashes = metrics.get("source_hashes")
    if not isinstance(source_hashes, Mapping) or manifest.get("source_hashes") != source_hashes:
        raise ValueError("source RT6 manifest and metrics provenance mismatch")
    if (
        source_hashes.get("factors_sha256") != factor_sha256
        or source_hashes.get("guard_sha256") != source_guard_sha256
        or source_hashes.get("profile_sha256") != profile_sha256
    ):
        raise ValueError("source RT6 frozen factor, guard, or profile hash mismatch")
    profile_fingerprint = _sha256_digest(
        source_hashes.get("profile_fingerprint"), field="source RT6 profile fingerprint"
    )
    converted_urdf = run_path / "converted_piper.urdf"
    conversion_manifest = run_path / "conversion_manifest.json"
    if not converted_urdf.is_file() or not conversion_manifest.is_file():
        raise ValueError("source RT6 MuJoCo assets are missing")
    if (
        source_hashes.get("converted_urdf") != sha256_file(converted_urdf)
        or source_hashes.get("conversion_manifest") != sha256_file(conversion_manifest)
    ):
        raise ValueError("source RT6 MuJoCo asset hash mismatch")

    frames = _read_jsonl(run_path / "state_frames.jsonl", "source RT6 state frames")
    if len(frames) != expected_trial_count:
        raise ValueError("source RT6 state frame count contract failed")
    frame_by_case: dict[str, dict[str, Any]] = {}
    for frame in frames:
        _validate_source_frame(
            frame,
            factor_sha256=factor_sha256,
            source_guard_sha256=source_guard_sha256,
            profile_fingerprint=profile_fingerprint,
        )
        case_id = str(frame["case_id"])
        if case_id in frame_by_case:
            raise ValueError("source RT6 state frames have duplicate case IDs")
        frame_by_case[case_id] = frame
    if len({str(frame["state_id"]) for frame in frames}) != expected_unique_state_count:
        raise ValueError("source RT6 state frame unique-state contract failed")

    result_rows = _read_csv(run_path / "results.csv", "source RT6 results")
    results_by_case: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in result_rows:
        case_id = str(row.get("case_id", ""))
        method = str(row.get("method", ""))
        if not case_id or method not in _SOURCE_METHODS or method in results_by_case[case_id]:
            raise ValueError("source RT6 results lack matched method rows")
        results_by_case[case_id][method] = row
    if set(results_by_case) != set(frame_by_case) or any(
        set(rows) != set(_SOURCE_METHODS) for rows in results_by_case.values()
    ):
        raise ValueError("source RT6 results lack matched method rows")

    metric_fingerprints = metrics.get("candidate_fingerprints")
    if not isinstance(metric_fingerprints, Mapping) or set(metric_fingerprints) != set(frame_by_case):
        raise ValueError("source RT6 candidate fingerprint ledger is malformed")
    candidate_fingerprints: dict[str, str] = {}
    for case_id, frame in frame_by_case.items():
        rows = results_by_case[case_id]
        fingerprints = {str(row.get("candidate_fingerprint", "")) for row in rows.values()}
        if len(fingerprints) != 1:
            raise ValueError("source RT6 methods do not share an original candidate fingerprint")
        candidate_fingerprint = _sha256_digest(
            fingerprints.pop(), field="source RT6 candidate fingerprint"
        )
        if candidate_fingerprint != metric_fingerprints.get(case_id):
            raise ValueError("source RT6 candidate fingerprint ledger mismatch")
        candidate = torch.as_tensor(frame["candidate_qpos"], dtype=torch.float32)
        expected_fingerprint = bound_candidate_fingerprint(
            case_id=case_id,
            state_id=str(frame["state_id"]),
            candidate_qpos=candidate,
            factor_sha256=factor_sha256,
            guard_sha256=source_guard_sha256,
        )
        if candidate_fingerprint != expected_fingerprint:
            raise ValueError("source RT6 candidate fingerprint does not bind stored qpos")
        if any(row.get("frame_fingerprint") != frame.get("frame_fingerprint") for row in rows.values()):
            raise ValueError("source RT6 result and state-frame fingerprints differ")
        candidate_fingerprints[case_id] = candidate_fingerprint

    component_rows = _read_csv(run_path / "component_results.csv", "source RT6 component results")
    components_by_case: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in component_rows:
        case_id = str(row.get("case_id", ""))
        joint_name = str(row.get("joint_name", ""))
        if not case_id or not joint_name or joint_name in components_by_case[case_id]:
            raise ValueError("source RT6 component results have duplicate entries")
        components_by_case[case_id][joint_name] = row
    if set(components_by_case) != set(frame_by_case):
        raise ValueError("source RT6 component results do not cover state frames")

    trace_rows = _read_csv(run_path / "trace.csv", "source RT6 optimizer trace")
    trace_by_case = _last_trace_corrections(
        trace_rows,
        joint_names=tuple(str(name) for name in frames[0]["joint_names"]),
        expected_case_ids=set(frame_by_case),
    )
    optimizer_rows = _read_jsonl(
        run_path / "optimizer_executions.jsonl", "source RT6 optimizer receipts"
    )
    optimizer_by_case: dict[str, set[str]] = defaultdict(set)
    for row in optimizer_rows:
        if row.get("completed") is not True or int(row.get("trace_step_count", 0)) <= 0:
            raise ValueError("source RT6 optimizer receipt is incomplete")
        optimizer_by_case[str(row.get("case_id", ""))].add(str(row.get("candidate_source", "")))
    if set(optimizer_by_case) != set(frame_by_case) or any(
        names != {"rt2_joint", "rt2_head", "rt2_extra"}
        for names in optimizer_by_case.values()
    ):
        raise ValueError("source RT6 optimizer receipt contract failed")

    cases: list[_SourceCase] = []
    for case_id, frame in sorted(frame_by_case.items()):
        joint_names = tuple(str(name) for name in frame["joint_names"])
        source_components = components_by_case[case_id]
        selected = tuple(str(item["joint_name"]) for item in frame["decisions"])
        if set(selected) != set(source_components):
            raise ValueError("source RT6 component results do not match frame decisions")
        offsets: dict[str, float] = {}
        measured = torch.as_tensor(frame["measured_qpos"], dtype=torch.float32)
        candidate = torch.as_tensor(frame["candidate_qpos"], dtype=torch.float32)
        for name in selected:
            row = source_components[name]
            offset = _optional_float(row.get("offset_rad"), field="source RT6 offset_rad")
            correction = _optional_float(
                row.get("candidate_correction_rad"), field="source RT6 candidate correction"
            )
            if offset is None or correction is None:
                raise ValueError("source RT6 component result lacks offset or correction")
            index = joint_names.index(name)
            if not math.isclose(
                correction,
                float(candidate[index] - measured[index]),
                rel_tol=0.0,
                abs_tol=1e-7,
            ):
                raise ValueError("source RT6 component correction does not match candidate qpos")
            offsets[name] = offset
        joint_correction = torch.as_tensor(trace_by_case[case_id]["rt2_joint"], dtype=torch.float32)
        if not torch.equal(measured + joint_correction, candidate):
            raise ValueError("source RT6 trace candidate does not match stored qpos")
        rows = results_by_case[case_id]
        reference = rows["rt2f_unguarded"]
        if (
            str(reference.get("state_id", "")) != str(frame["state_id"])
            or str(reference.get("role", "")) not in {"single_joint", "zero_control", "multi_joint_diagnostic"}
        ):
            raise ValueError("source RT6 results have invalid case provenance")
        magnitude = _optional_float(reference.get("magnitude_deg"), field="source RT6 magnitude_deg")
        if magnitude is None:
            raise ValueError("source RT6 results lack a magnitude")
        cases.append(
            _SourceCase(
                case_id=case_id,
                state_id=str(frame["state_id"]),
                role=str(reference["role"]),
                magnitude_deg=magnitude,
                frame=frame,
                result_rows=rows,
                component_rows=source_components,
                selected_joints=selected,
                offsets_rad=offsets,
                camera_corrections={
                    "head": trace_by_case[case_id]["rt2_head"],
                    "extra": trace_by_case[case_id]["rt2_extra"],
                },
                candidate_fingerprint=candidate_fingerprints[case_id],
            )
        )

    asset_paths = {
        "source_rt6_config": config_path,
        "source_rt6_manifest": run_path / "manifest.json",
        "source_rt6_metrics": run_path / "metrics.json",
        "source_rt6_state_frames": run_path / "state_frames.jsonl",
        "source_rt6_results": run_path / "results.csv",
        "source_rt6_component_results": run_path / "component_results.csv",
        "source_rt6_trace": run_path / "trace.csv",
        "source_rt6_optimizer_executions": run_path / "optimizer_executions.jsonl",
        "source_rt6_factors": factor_path,
        "source_rt6_guard": source_guard_path,
        "source_rt6_profile": profile_path,
        "source_rt6_matrix": matrix_path,
        "source_rt6_converted_urdf": converted_urdf,
        "source_rt6_conversion_manifest": conversion_manifest,
        "rt7_guard": rt7_guard_path,
    }
    return (
        _SourceReplay(
            run_path=run_path,
            config=source_config,
            manifest=manifest,
            metrics=metrics,
            cases=tuple(cases),
            factor_path=factor_path,
            source_guard_path=source_guard_path,
            profile_path=profile_path,
            matrix_path=matrix_path,
            factor_sha256=factor_sha256,
            source_guard_sha256=source_guard_sha256,
            profile_sha256=profile_sha256,
            profile_fingerprint=profile_fingerprint,
            file_hashes=actual_hashes,
            asset_paths=asset_paths,
        ),
        rt7_guard_path,
        _read_json(rt7_guard_path, "RT7 event guard"),
        actual_hashes["rt7_guard_sha256"],
    )


def _load_rt7_guard(
    payload: Mapping[str, Any],
    *,
    source_guard_sha256: str,
) -> GuardThresholds:
    try:
        if (
            payload.get("schema_version") != 1
            or payload.get("frozen") is not True
            or payload.get("formal_evidence") is not False
            or payload.get("source_split") != "train"
            or payload.get("threshold_generation") != "event_gain_v1"
            or payload.get("frozen_rt6_guard_sha256") != source_guard_sha256
        ):
            raise ValueError("invalid event guard provenance")
        _sha256_digest(payload["calibration_fingerprint"], field="RT7 guard calibration fingerprint")
        return GuardThresholds(**dict(payload["thresholds"]))
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("malformed RT7 event guard provenance") from error


def _build_replay_runtime(
    source: _SourceReplay,
    preflight: FrozenRT6SourcePreflight,
    device: torch.device,
) -> _NativeReplayRuntime:
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("RT7 development replay requires the native CUDA rasterizer")
    frozen = preflight.frozen
    factor_sha256 = preflight.factor_sha256
    if factor_sha256 != source.factor_sha256:
        raise ValueError("source RT6 factor path changed from its state ledger")
    source_guard_sha256 = preflight.guard_sha256
    if source_guard_sha256 != source.source_guard_sha256:
        raise ValueError("source RT6 guard path changed from its state ledger")
    profile = preflight.profile
    if (
        preflight.profile_sha256 != source.profile_sha256
        or profile.fingerprint != source.profile_fingerprint
    ):
        raise ValueError("source RT6 profile changed from its state ledger")
    matrix = preflight.matrix
    matrix_cases = {case.case_id: case for case in matrix.cases}
    if set(matrix_cases) != {case.case_id for case in source.cases}:
        raise ValueError("source RT6 matrix no longer matches the state ledger")

    paths = RouteAPaths.from_mapping(source.config["assets"])
    assets = RouteAAssets(paths)
    dataset = _real_dataset(source.config)
    fk = TorchURDFKinematics(paths.urdf)
    if tuple(fk.active_joint_names) != profile.joint_names:
        raise ValueError("RT7 replay profile joint order does not match PiPER URDF")
    factor_joint_names = tuple(str(value) for value in preflight.factors["joint_names"])
    if factor_joint_names != profile.joint_names[: len(factor_joint_names)]:
        raise ValueError("RT7 replay factor joint order does not match profile")
    shared_joint = torch.zeros(len(profile.joint_names), dtype=torch.float32, device=device)
    shared_joint[: len(factor_joint_names)] = torch.as_tensor(
        preflight.factors["joint_zero_rad"], dtype=torch.float32, device=device
    )
    camera_factors = {
        name: torch.as_tensor(value, dtype=torch.float32, device=device)
        for name, value in preflight.factors["camera_twists"].items()
    }
    indices = _anchor_indices(
        assets,
        source.config["data"]["anchors_per_link"],
        seed=int(source.config.get("seed", 41)),
    )
    renderer_for = formal_renderer_factory(
        fk=fk,
        assets=assets,
        indices=indices,
        config=source.config,
        device=device,
    )
    bridge = MujocoStateBridge.from_urdf(
        source.run_path / "converted_piper.urdf", profile.joint_names
    )
    return _NativeReplayRuntime(
        profile=profile,
        factor_sha256=factor_sha256,
        source_guard_sha256=source_guard_sha256,
        bridge=bridge,
        fk=fk,
        dataset=dataset,
        renderer_for=renderer_for,
        camera_factors=camera_factors,
        shared_joint=shared_joint,
        boundary_weight=float(source.config["observation"]["boundary_weight"]),
        candidate_bound=float(source.config["evaluation"]["offset_bound_rad"]),
        acceptance=source.config.get("acceptance", {}),
        max_source_timestamp_delta_ns=int(frozen["max_source_timestamp_delta_ns"]),
        output_size=tuple(map(int, source.config["observation"]["image_size"])),
        cases=matrix_cases,
        device=device,
    )


def _tensor_values(values: torch.Tensor) -> list[float]:
    return [float(value) for value in values.detach().cpu().tolist()]


def _source_evidence(row: Mapping[str, str]) -> dict[str, Any]:
    return {
        "evidence_finite": _parse_bool(row.get("evidence_finite"), field="source evidence_finite"),
        "visual_gain_ratio": float(row["visual_gain_ratio"]),
        "gradient_cosine": float(row["gradient_cosine"]),
        "correction_cosine": float(row["correction_cosine"]),
        "relative_correction_disagreement": float(row["relative_correction_disagreement"]),
    }


def _component_record(
    *,
    source_case: _SourceCase,
    source_row: Mapping[str, str],
    verification,
    commit_decision,
    rt7_frame,
    render_receipt: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    candidate_success = _parse_bool(
        source_row.get("candidate_success"), field="source component candidate_success"
    )
    physical_success = _parse_bool(
        source_row.get("physical_success"), field="source component physical_success"
    )
    trial_visual_success = _parse_bool(
        source_row.get("trial_visual_success"), field="source component trial_visual_success"
    )
    source_state_error = _optional_float(
        source_row.get("state_error_reduction"), field="source component state_error_reduction"
    )
    evidence_payload = {
        "case_id": source_case.case_id,
        "state_id": source_case.state_id,
        "joint_name": verification.joint_name,
        "source_candidate_fingerprint": source_case.candidate_fingerprint,
        "rt7_frame_fingerprint": rt7_frame.fingerprint,
        "component_verification_fingerprint": verification.fingerprint,
        "component_candidate_qpos": _tensor_values(verification.candidate_qpos),
        "evidence": asdict(verification.evidence),
        "guard_decision": asdict(verification.decision),
        "before_metrics": dict(verification.before_metrics),
        "after_metrics": dict(verification.after_metrics),
        "per_view_visual_gain": dict(verification.per_view_visual_gain),
    }
    receipt_payload = {
        "case_id": source_case.case_id,
        "state_id": source_case.state_id,
        "joint_name": verification.joint_name,
        "source_candidate_fingerprint": source_case.candidate_fingerprint,
        "component_verification_fingerprint": verification.fingerprint,
        "component_candidate_qpos": _tensor_values(verification.candidate_qpos),
        "camera_trace_corrections": {
            camera: list(values) for camera, values in sorted(source_case.camera_corrections.items())
        },
        "render_receipt": dict(render_receipt),
        "commit_decision": asdict(commit_decision),
    }
    csv_row = {
        "case_id": source_case.case_id,
        "state_id": source_case.state_id,
        "role": source_case.role,
        "magnitude_deg": source_case.magnitude_deg,
        "joint_name": verification.joint_name,
        "offset_rad": source_case.offsets_rad[verification.joint_name],
        "candidate_correction_rad": float(
            verification.candidate_qpos[
                list(rt7_frame.joint_names).index(verification.joint_name)
            ]
            - rt7_frame.measured_qpos[
                list(rt7_frame.joint_names).index(verification.joint_name)
            ]
        ),
        "state_error_reduction": source_state_error,
        "physical_success": physical_success,
        "trial_visual_success": trial_visual_success,
        "candidate_success": candidate_success,
        "accepted": commit_decision.accepted,
        "decision_reason": commit_decision.reason,
        "signal_floor_rad": commit_decision.signal_floor_rad,
        "margin_rad": verification.decision.minimum_margin,
        "component_evidence_finite": verification.evidence.finite,
        "component_visual_gain_ratio": verification.evidence.visual_gain_ratio,
        "component_gradient_cosine": verification.evidence.gradient_cosine,
        "component_correction_cosine": verification.evidence.correction_cosine,
        "component_relative_correction_disagreement": verification.evidence.relative_correction_disagreement,
        "component_verification_fingerprint": verification.fingerprint,
        "source_candidate_fingerprint": source_case.candidate_fingerprint,
        "frame_fingerprint": rt7_frame.fingerprint,
    }
    return csv_row, evidence_payload, receipt_payload


def _metric_bool(value: Any, *, field: str) -> bool:
    return _parse_bool(value, field=field) if isinstance(value, str) else bool(value)


def _rate(values: Iterable[bool]) -> float | None:
    materialized = [_metric_bool(value, field="replay boolean metric") for value in values]
    return mean(materialized) if materialized else None


def _component_group_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    committed = [
        row for row in rows if _metric_bool(row["accepted"], field="component accepted")
    ]
    successful = [
        row
        for row in rows
        if _metric_bool(row["candidate_success"], field="component candidate_success")
    ]
    return {
        "count": len(rows),
        "candidate_success_rate": _rate(row["candidate_success"] for row in rows),
        "commit_coverage": _rate(row["accepted"] for row in rows),
        "committed_precision": _rate(row["candidate_success"] for row in committed),
        "successful_candidate_retention": _rate(row["accepted"] for row in successful),
    }


def _component_subgroups(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    controlled = [row for row in rows if row.get("role") == "single_joint"]
    families: dict[str, dict[str, list[Mapping[str, Any]]]] = {
        "by_joint": defaultdict(list),
        "by_magnitude_deg": defaultdict(list),
        "by_sign": defaultdict(list),
    }
    for row in controlled:
        families["by_joint"][str(row["joint_name"])].append(row)
        families["by_magnitude_deg"][f"{float(row['magnitude_deg']):.1f}"].append(row)
        sign = "positive" if float(row["offset_rad"]) > 0 else "negative"
        families["by_sign"][sign].append(row)
    return {
        family: {
            name: _component_group_summary(group)
            for name, group in sorted(groups.items())
        }
        for family, groups in families.items()
    }


def _summarize_replay(
    *, method_rows: Sequence[Mapping[str, Any]], component_rows: Sequence[Mapping[str, Any]], parity_passed: bool
) -> dict[str, Any]:
    method_summaries: dict[str, dict[str, Any]] = {}
    for method in _REPLAY_METHODS:
        rows = [row for row in method_rows if row.get("method") == method]
        if not rows:
            raise ValueError(f"replay results lack method rows: {method}")
        by_state: dict[str, list[bool]] = defaultdict(list)
        for row in rows:
            by_state[str(row["state_id"])].append(
                _metric_bool(row["recovery_success"], field="method recovery_success")
            )
        method_summaries[method] = {
            "trial_count": len(rows),
            "unique_state_count": len(by_state),
            "trial_success_rate": _rate(row["recovery_success"] for row in rows),
            "state_clustered_success_rate": mean(mean(values) for values in by_state.values()),
            "mean_mask_iou_gain": mean(float(row["mask_iou_gain"]) for row in rows),
            "mean_boundary_f1_gain": mean(float(row["boundary_f1_gain"]) for row in rows),
        }
    controlled = [row for row in component_rows if row.get("role") == "single_joint"]
    zeros = [row for row in component_rows if row.get("role") == "zero_control"]
    if not controlled or not zeros:
        raise ValueError("replay component rows require controlled and zero cases")
    committed = [
        row
        for row in controlled
        if _metric_bool(row["accepted"], field="component accepted")
    ]
    successful = [
        row
        for row in controlled
        if _metric_bool(row["candidate_success"], field="component candidate_success")
    ]
    primary = {
        "committed_precision": _rate(row["candidate_success"] for row in committed),
        "controlled_commit_coverage": _rate(row["accepted"] for row in controlled),
        "successful_candidate_retention": _rate(row["accepted"] for row in successful),
        "zero_false_update_rate": _rate(row["accepted"] for row in zeros),
    }
    primary["zero_stability"] = (
        None
        if primary["zero_false_update_rate"] is None
        else 1.0 - float(primary["zero_false_update_rate"])
    )
    finite = all(bool(row["component_evidence_finite"]) for row in component_rows)
    return {
        "method_summaries": method_summaries,
        "primary_metrics": primary,
        "component_subgroups": _component_subgroups(component_rows),
        "finite_component_evidence_and_fk_parity": bool(finite and parity_passed),
    }


def _development_diagnostics(
    *,
    component_rows: Sequence[Mapping[str, Any]],
    method_rows: Sequence[Mapping[str, Any]],
    targets: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        retention_target = targets["successful_candidate_retention"]
        precision_target = float(targets["committed_precision"])
        zero_target = targets["zero_false_updates"]
        matched_target = targets["matched_success"]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("RT7 replay config requires complete diagnostic targets") from error
    controlled = [row for row in component_rows if row["role"] == "single_joint"]
    successful = [
        row
        for row in controlled
        if _metric_bool(row["candidate_success"], field="component candidate_success")
    ]
    committed = [
        row
        for row in controlled
        if _metric_bool(row["accepted"], field="component accepted")
    ]
    retained = sum(
        _metric_bool(row["accepted"], field="component accepted") for row in successful
    )
    precision_numerator = sum(
        _metric_bool(row["candidate_success"], field="component candidate_success")
        for row in committed
    )
    zeros = [row for row in component_rows if row["role"] == "zero_control"]
    false_updates = sum(
        _metric_bool(row["accepted"], field="component accepted") for row in zeros
    )
    rt7_rows = [row for row in method_rows if row["method"] == "rt7cv_component_verified"]
    matched_success = sum(
        _metric_bool(row["recovery_success"], field="method recovery_success")
        for row in rt7_rows
    )
    precision = None if not committed else precision_numerator / len(committed)
    return {
        "successful_candidate_retention": {
            "observed_numerator": retained,
            "observed_denominator": len(successful),
            "minimum_numerator": int(retention_target["minimum_numerator"]),
            "expected_denominator": int(retention_target["denominator"]),
            "passed": bool(
                len(successful) == int(retention_target["denominator"])
                and retained >= int(retention_target["minimum_numerator"])
            ),
        },
        "committed_precision": {
            "observed": precision,
            "minimum": precision_target,
            "passed": bool(precision is not None and precision >= precision_target),
        },
        "zero_false_updates": {
            "observed_numerator": false_updates,
            "observed_denominator": len(zeros),
            "maximum_numerator": int(zero_target["maximum_numerator"]),
            "expected_denominator": int(zero_target["denominator"]),
            "passed": bool(
                len(zeros) == int(zero_target["denominator"])
                and false_updates <= int(zero_target["maximum_numerator"])
            ),
        },
        "matched_success": {
            "observed_numerator": matched_success,
            "observed_denominator": len(rt7_rows),
            "minimum_numerator": int(matched_target["minimum_numerator"]),
            "expected_denominator": int(matched_target["denominator"]),
            "passed": bool(
                len(rt7_rows) == int(matched_target["denominator"])
                and matched_success >= int(matched_target["minimum_numerator"])
            ),
        },
    }


def _validate_replay_artifacts(
    path: Path,
    *,
    expected_trial_count: int,
    source_candidate_fingerprints: Mapping[str, str],
) -> None:
    metrics = _read_json(path / "metrics.json", "RT7 replay metrics")
    if (
        metrics.get("formal_evidence") is not False
        or metrics.get("evidence_role") != "development_counterfactual"
        or metrics.get("formal_decision") != "not_applicable"
        or metrics.get("trial_count") != expected_trial_count
        or metrics.get("source_candidate_fingerprints") != dict(source_candidate_fingerprints)
    ):
        raise ValueError("RT7 replay development evidence labels are invalid")
    results = _read_csv(path / "results.csv", "RT7 replay results")
    by_case: dict[str, list[Mapping[str, str]]] = defaultdict(list)
    for row in results:
        by_case[str(row["case_id"])].append(row)
    if set(by_case) != set(source_candidate_fingerprints) or any(
        {row["method"] for row in rows} != set(_REPLAY_METHODS) for rows in by_case.values()
    ):
        raise ValueError("RT7 replay does not contain matched method rows")
    for case_id, rows in by_case.items():
        if any(row.get("candidate_fingerprint") != source_candidate_fingerprints[case_id] for row in rows):
            raise ValueError("RT7 replay changed an original source candidate fingerprint")
    frames = _read_jsonl(path / "state_frames.jsonl", "RT7 replay state frames")
    if len(frames) != expected_trial_count:
        raise ValueError("RT7 replay state frame count is invalid")
    for frame in frames:
        if (
            frame.get("synchronized_qpos") != frame.get("mujoco_qpos")
            or frame.get("synchronized_qpos") != frame.get("gaussian_render_qpos")
            or frame.get("source_candidate_fingerprint")
            != source_candidate_fingerprints.get(str(frame.get("case_id")))
        ):
            raise ValueError("RT7 replay synchronized qpos routing is invalid")
    image = cv2.imread(str(path / "images" / "rt7_component_replay.png"), cv2.IMREAD_COLOR)
    if image is None or image.shape != (1080, 1920, 3) or float(image.std()) <= 1.0:
        raise ValueError("RT7 replay image contract failed")
    capture = cv2.VideoCapture(str(path / "videos" / "rt7_component_replay.mp4"))
    try:
        if (
            not capture.isOpened()
            or int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) != 1920
            or int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) != 1080
            or int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) <= 0
        ):
            raise ValueError("RT7 replay video contract failed")
    finally:
        capture.release()


def execute_rt7_replay(config: Mapping[str, Any], *, run_id: str | None = None) -> Path:
    """Execute a development-only RT7 replay without invoking the optimizer."""

    source, rt7_guard_path, rt7_guard, rt7_guard_sha256 = _load_source_replay(config)
    thresholds = _load_rt7_guard(
        rt7_guard, source_guard_sha256=source.source_guard_sha256
    )
    provenance = config["provenance"]
    media = config.get("media", {})
    try:
        media_frame_count = int(media.get("frame_count", 36))
        media_fps = int(media.get("fps", 12))
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError("RT7 replay media config is invalid") from error
    if media_frame_count <= 0 or media_fps <= 0:
        raise ValueError("RT7 replay media config is invalid")
    device = torch.device(str(config.get("device", "cuda")))
    preflight = verify_frozen_rt6_source(
        source_config=source.config,
        source_metrics=source.metrics,
        source_run_path=source.run_path,
    )
    run = RunArtifacts.create(
        root=config.get("runs_root", "runs"),
        experiment=str(config.get("experiment", "rt7_component_development_replay")),
        run_id=run_id,
        config=config,
        assets={**source.asset_paths, **preflight.asset_paths},
        target_provenance="real_observation_guard_evaluation",
        observation_backend="native_cuda_rt7_component_verified_development_replay",
    )
    runtime = _build_replay_runtime(source, preflight, device)
    if (
        runtime.factor_sha256 != source.factor_sha256
        or runtime.source_guard_sha256 != source.source_guard_sha256
        or runtime.profile.fingerprint != source.profile_fingerprint
        or tuple(runtime.profile.joint_names)
        != tuple(str(name) for name in source.cases[0].frame["joint_names"])
    ):
        raise ValueError("RT7 replay runtime provenance does not match source RT6 states")

    method_rows: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []
    evidence_rows: list[dict[str, Any]] = []
    receipt_rows: list[dict[str, Any]] = []
    state_rows: list[dict[str, Any]] = []
    media_contexts: list[dict[str, Any]] = []
    frames = []
    for source_case in source.cases:
        case_runtime = runtime.prepare_case(source_case)
        measured = torch.as_tensor(
            source_case.frame["measured_qpos"], dtype=torch.float32, device=device
        )
        candidate = torch.as_tensor(
            source_case.frame["candidate_qpos"], dtype=torch.float32, device=device
        )
        if not torch.equal(measured.detach().cpu(), torch.as_tensor(source_case.frame["measured_qpos"], dtype=torch.float32)):
            raise ValueError("RT7 replay measured qpos changed while loading source state")
        if not torch.equal(candidate.detach().cpu(), torch.as_tensor(source_case.frame["candidate_qpos"], dtype=torch.float32)):
            raise ValueError("RT7 replay candidate qpos changed while loading source state")
        camera_corrections = {
            name: torch.as_tensor(values, dtype=measured.dtype, device=device)
            for name, values in source_case.camera_corrections.items()
        }
        application = apply_component_verified_policy(
            profile=runtime.profile,
            joint_backend=case_runtime.visual,
            joint_target=case_runtime.targets,
            camera_backends=case_runtime.camera_backends,
            camera_targets=case_runtime.camera_targets,
            measured_qpos=measured,
            candidate_qpos=candidate,
            camera_corrections=camera_corrections,
            joint_names=runtime.profile.joint_names,
            selected_joints=source_case.selected_joints,
            candidate_bound=case_runtime.candidate_bound,
            thresholds=thresholds,
            joint_limits=runtime.bridge.joint_limits,
            timestamp_ns=int(source_case.frame["timestamp_ns"]),
            state_id=source_case.state_id,
            case_id=source_case.case_id,
            factor_fingerprint=source.factor_sha256,
            guard_fingerprint=rt7_guard_sha256,
            render_receipt=gaussian_render_receipt,
        )
        verifications = application.verifications
        rt7_frame = application.frame
        runtime.bridge.write(rt7_frame)
        synchronized_values = _tensor_values(rt7_frame.synchronized_qpos)
        mujoco_values = [
            float(value)
            for value in runtime.bridge.data.qpos[list(runtime.bridge.qpos_addresses)].tolist()
        ]
        if mujoco_values != synchronized_values:
            raise ValueError("MuJoCo qpos differs from RT7 component-verified qpos")
        gaussian_receipt = application.synchronized_render_receipt
        gaussian_values = list(gaussian_receipt["qpos"])
        if gaussian_values != synchronized_values:
            raise ValueError("Gaussian qpos differs from RT7 component-verified qpos")
        frames.append(rt7_frame)

        commit_decisions = {item.joint_name: item for item in rt7_frame.decisions}
        component_fingerprints = {}
        for verification in verifications.verifications:
            component_render_receipt = application.component_render_receipts[
                verification.joint_name
            ]
            csv_row, evidence_row, receipt_row = _component_record(
                source_case=source_case,
                source_row=source_case.component_rows[verification.joint_name],
                verification=verification,
                commit_decision=commit_decisions[verification.joint_name],
                rt7_frame=rt7_frame,
                render_receipt=component_render_receipt,
            )
            component_rows.append(csv_row)
            evidence_rows.append(evidence_row)
            receipt_rows.append(receipt_row)
            component_fingerprints[verification.joint_name] = verification.fingerprint

        source_rt2 = source_case.result_rows["rt2f_unguarded"]
        source_rt3 = source_case.result_rows["rt3c_guarded"]
        source_rt6 = source_case.result_rows["rt6dm_synchronized"]
        source_evidence = _source_evidence(source_rt2)
        source_rt3_accepted = _parse_bool(
            source_rt3.get("guard_accepted"), field="source RT3 guard_accepted"
        )
        source_rt6_state = torch.as_tensor(
            source_case.frame["synchronized_qpos"], dtype=measured.dtype, device=device
        )
        source_method_states = {
            "rt2f_unguarded": candidate,
            "rt3c_guarded": candidate if source_rt3_accepted else measured,
            "rt6dm_synchronized": source_rt6_state,
        }
        method_states = {
            "measured_no_correction": (
                measured,
                False,
                "not_applied",
                source_case.frame["frame_fingerprint"],
                False,
            ),
            "rt7cv_component_verified": (
                rt7_frame.synchronized_qpos,
                bool(rt7_frame.accepted_joint_names),
                rt7_frame.guard_reason,
                rt7_frame.fingerprint,
                bool(rt7_frame.accepted_joint_names),
            ),
        }
        for method in _REPLAY_METHODS:
            if method in source_method_states:
                method_rows.append(
                    inherit_source_result_row(
                        source_case.result_rows[method],
                        {
                            "replay_component_verification_fingerprints": json.dumps(
                                component_fingerprints, sort_keys=True
                            ),
                            "replay_final_qpos": json.dumps(
                                _tensor_values(source_method_states[method])
                            ),
                            "replay_origin": "source_rt6_inherited",
                            "replay_source_frame_fingerprint": source_case.frame[
                                "frame_fingerprint"
                            ],
                        },
                    )
                )
                continue
            final_qpos, guard_accepted, guard_reason, frame_fingerprint, any_component_accepted = method_states[method]
            outcome, before, after = _outcome(
                backend=case_runtime.visual,
                target=case_runtime.targets,
                measured=measured,
                final=final_qpos,
                reference=case_runtime.reference_qpos,
                selected_indices=list(case_runtime.selected_indices),
                acceptance=case_runtime.acceptance,
            )
            method_rows.append(
                {
                    "case_id": source_case.case_id,
                    "state_id": source_case.state_id,
                    "split": "validation",
                    "role": source_case.role,
                    "magnitude_deg": source_case.magnitude_deg,
                    "method": method,
                    "candidate_fingerprint": source_case.candidate_fingerprint,
                    "source_candidate_fingerprint": source_case.candidate_fingerprint,
                    "source_frame_fingerprint": source_case.frame["frame_fingerprint"],
                    "frame_fingerprint": frame_fingerprint,
                    "guard_accepted": guard_accepted,
                    "guard_reason": guard_reason,
                    "any_component_accepted": any_component_accepted,
                    "state_error_reduction": outcome["state_error_reduction"],
                    "recovery_success": bool(outcome["recovery_success"]),
                    "mask_iou_before": before["mean_mask_iou"],
                    "mask_iou_after": after["mean_mask_iou"],
                    "mask_iou_gain": after["mean_mask_iou"] - before["mean_mask_iou"],
                    "boundary_f1_before": before["mean_boundary_f1"],
                    "boundary_f1_after": after["mean_boundary_f1"],
                    "boundary_f1_gain": after["mean_boundary_f1"] - before["mean_boundary_f1"],
                    **source_evidence,
                    "component_verification_fingerprints": json.dumps(
                        component_fingerprints, sort_keys=True
                    ),
                }
            )
        state_rows.append(
            {
                "case_id": source_case.case_id,
                "state_id": source_case.state_id,
                "timestamp_ns": rt7_frame.timestamp_ns,
                "joint_names": list(rt7_frame.joint_names),
                "measured_qpos": _tensor_values(rt7_frame.measured_qpos),
                "candidate_qpos": _tensor_values(rt7_frame.candidate_qpos),
                "synchronized_qpos": synchronized_values,
                "mujoco_qpos": mujoco_values,
                "gaussian_render_qpos": gaussian_values,
                "gaussian_render_receipt": gaussian_receipt,
                "source_rt6_synchronized_qpos": source_case.frame["synchronized_qpos"],
                "source_candidate_fingerprint": source_case.candidate_fingerprint,
                "source_frame_fingerprint": source_case.frame["frame_fingerprint"],
                "source_fusion_policy": source_case.frame["source_fusion_policy"],
                "source_observations": source_case.frame["source_observations"],
                "source_timestamp_span_ns": source_case.frame["source_timestamp_span_ns"],
                "decisions": [asdict(item) for item in rt7_frame.decisions],
                "component_verification_fingerprints": component_fingerprints,
                "factor_fingerprint": rt7_frame.factor_fingerprint,
                "guard_fingerprint": rt7_frame.guard_fingerprint,
                "profile_fingerprint": rt7_frame.profile_fingerprint,
                "frame_fingerprint": rt7_frame.fingerprint,
            }
        )
        media_contexts.append(
            {"frame": rt7_frame, "observations": case_runtime.observations, "backend": case_runtime.visual}
        )

    parity = compare_fk_parity(
        runtime.bridge,
        runtime.fk,
        np.asarray([frame.synchronized_qpos.detach().cpu().numpy() for frame in frames]),
    )
    if not parity.passed:
        raise ValueError("RT7 component-verified state failed MuJoCo/Torch FK parity")
    summary = _summarize_replay(
        method_rows=method_rows,
        component_rows=component_rows,
        parity_passed=parity.passed,
    )
    diagnostics = _development_diagnostics(
        component_rows=component_rows,
        method_rows=method_rows,
        targets=config.get("diagnostic_targets", {}),
    )
    source_candidate_fingerprints = {
        case.case_id: case.candidate_fingerprint for case in source.cases
    }
    source_hashes = {
        **source.file_hashes,
        "source_rt6_factor_sha256": source.factor_sha256,
        "source_rt6_guard_sha256": source.source_guard_sha256,
        "source_rt6_profile_sha256": source.profile_sha256,
        "source_rt6_profile_fingerprint": source.profile_fingerprint,
        "rt7_guard_calibration_fingerprint": str(rt7_guard["calibration_fingerprint"]),
    }
    image_path = run.path / "images" / "rt7_component_replay.png"
    video_path = run.path / "videos" / "rt7_component_replay.mp4"
    write_synchronization_comparison(
        image_path,
        media_contexts,
        presentation=_RT7_COMPONENT_PRESENTATION,
    )
    write_synchronization_video(
        video_path,
        media_contexts,
        frame_count=media_frame_count,
        fps=media_fps,
        presentation=_RT7_COMPONENT_PRESENTATION,
    )
    _write_csv(run.path / "results.csv", method_rows)
    _write_csv(run.path / "component_results.csv", component_rows)
    _write_jsonl(run.path / "component_evidence.jsonl", evidence_rows)
    _write_jsonl(run.path / "component_receipts.jsonl", receipt_rows)
    _write_jsonl(run.path / "state_frames.jsonl", state_rows)
    _write_json(run.path / "parity_report.json", asdict(parity))
    _write_json(run.path / "hashes.json", source_hashes)
    frozen_asset_receipt_path = run.path / "frozen_asset_receipt.json"
    _write_json(frozen_asset_receipt_path, preflight.asset_receipt)
    _write_json(
        run.path / "source_receipt.json",
        {
            "source_rt6_run": str(source.run_path),
            "source_rt6_run_id": source.manifest["run_id"],
            "source_candidate_fingerprints": source_candidate_fingerprints,
            "source_hashes": source_hashes,
            "verified_frozen_asset_hashes": preflight.verified_source_hashes,
            "frozen_asset_receipt": preflight.asset_receipt,
            "rt7_guard": str(rt7_guard_path),
            "rt7_guard_sha256": rt7_guard_sha256,
        },
    )
    metrics = {
        "formal_evidence": False,
        "evidence_role": "development_counterfactual",
        "formal_decision": "not_applicable",
        "source_rt6_run_id": source.manifest["run_id"],
        "trial_count": len(source.cases),
        "unique_state_count": len({case.state_id for case in source.cases}),
        "method_row_count": len(method_rows),
        "component_row_count": len(component_rows),
        "source_candidate_fingerprints": source_candidate_fingerprints,
        "source_hashes": source_hashes,
        "verified_frozen_asset_hashes": preflight.verified_source_hashes,
        "frozen_asset_receipt_sha256": sha256_file(frozen_asset_receipt_path),
        "rt7_guard_sha256": rt7_guard_sha256,
        "rt7_guard_calibration_fingerprint": str(rt7_guard["calibration_fingerprint"]),
        "profile_fingerprint": source.profile_fingerprint,
        "parity_report": asdict(parity),
        "development_diagnostics": diagnostics,
        "all_development_diagnostics_passed": all(
            value["passed"] for value in diagnostics.values()
        ),
        **summary,
    }
    run.write_metrics(metrics)
    manifest = _read_json(run.path / "manifest.json", "RT7 replay manifest")
    derived_paths = {
        "component_evidence": run.path / "component_evidence.jsonl",
        "component_receipts": run.path / "component_receipts.jsonl",
        "component_results": run.path / "component_results.csv",
        "frozen_asset_receipt": frozen_asset_receipt_path,
        "hashes": run.path / "hashes.json",
        "metrics": run.path / "metrics.json",
        "parity_report": run.path / "parity_report.json",
        "results": run.path / "results.csv",
        "source_receipt": run.path / "source_receipt.json",
        "state_frames": run.path / "state_frames.jsonl",
        "synchronization_image": image_path,
        "synchronization_video": video_path,
    }
    manifest.update(
        {
            "formal_evidence": False,
            "evidence_role": "development_counterfactual",
            "formal_decision": "not_applicable",
            "source_hashes": source_hashes,
            "source_rt6_run": str(source.run_path),
            "rt7_guard_sha256": rt7_guard_sha256,
            "derived_artifacts": {
                name: {"path": str(path), "sha256": sha256_file(path)}
                for name, path in sorted(derived_paths.items())
            },
        }
    )
    _write_json(run.path / "manifest.json", manifest)
    _validate_replay_artifacts(
        run.path,
        expected_trial_count=int(provenance["expected_trial_count"]),
        source_candidate_fingerprints=source_candidate_fingerprints,
    )
    return run.path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id")
    arguments = parser.parse_args()
    print(execute_rt7_replay(load_config(arguments.config), run_id=arguments.run_id))


if __name__ == "__main__":
    main()
