"""Run the read-only RT9 Take-Pens temporal observation bridge benchmark."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from kinesync.config import load_config
from kinesync.data.take_pens import ContinuousSegment, TakePensDataset, TakePensTrajectory, continuous_segments
from kinesync.runs.artifacts import RunArtifacts
from kinesync.temporal.telemetry import (
    TEMPORAL_METHODS,
    GuardCalibrationReport,
    MotionFeatures,
    OffsetEstimate,
    OffsetScore,
    TemporalGuard,
    TemporalGuardConstraints,
    TemporalResultRow,
    TemporalTrial,
    calibrate_temporal_guard_with_report,
    corrupt_state_stream,
    evaluate_temporal_method,
    extract_motion_features,
    interpolate_qpos,
    search_motion_offset,
    summarize_temporal_results,
)
from kinesync.visualization.temporal_telemetry import (
    TemporalTelemetryVisualFrame,
    write_temporal_telemetry_image,
    write_temporal_telemetry_video,
)


_DEVELOPMENT_FILES = ("0.hdf5", "1.hdf5", "2.hdf5", "s.hdf5")
_TEST_FILES = ("f1.hdf5", "f2.hdf5")
_OFFSETS_MS = (-120, -80, -40, 0, 40, 80, 120)
_JITTER_MS = (0, 10, 20)
_DROPOUT = (0.0, 0.15, 0.30)
_CANDIDATE_OFFSETS_MS = tuple(range(-200, 201, 20))
_PAUSE_MS = 200
_MINIMUM_FRAMES = 60
_IMAGE_SIZE = (60, 80)
_MAX_CALIBRATION_CANDIDATES = 25_000
_CANONICAL_DATASET_ROOT = Path("data/take_pens")
_EVIDENCE_SCOPE = "offline_recorded_telemetry_zero_robot_commands"
_MEDIA_SELECTION_RULE = "beneficial_committed_qmae_gain_closest_to_trajectory_median_v1"
_MEDIA_SAFETY_LABEL = "Offline recorded telemetry; no robot commands"
_MEDIA_VIDEO_FPS = 12
_MEDIA_MAX_VIDEO_FRAMES = 60


@dataclass(frozen=True)
class _TemporalVisualSelection:
    trajectory: TakePensTrajectory
    segment: ContinuousSegment
    trial: TemporalTrial
    estimate: OffsetEstimate
    raw: TemporalResultRow
    guarded: TemporalResultRow
    receipt: dict[str, Any]


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    return value


def _sequence(value: object, *, field: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{field} must be a sequence")
    return value


def _require_exact_sequence(
    value: object, *, field: str, expected: Sequence[object]
) -> tuple[object, ...]:
    actual = tuple(_sequence(value, field=field))
    if actual != tuple(expected):
        raise ValueError(f"{field} is frozen for the RT9 protocol")
    return actual


def _protocol_config(config: Mapping[str, Any]) -> dict[str, Any]:
    dataset = _mapping(config.get("dataset"), field="dataset")
    segmentation = _mapping(config.get("segmentation"), field="segmentation")
    features = _mapping(config.get("features"), field="features")
    corruption = _mapping(config.get("corruption"), field="corruption")
    search = _mapping(config.get("search"), field="search")
    guard = _mapping(config.get("guard"), field="guard")
    evaluation = _mapping(config.get("evaluation"), field="evaluation")

    development_files = tuple(
        str(value)
        for value in _require_exact_sequence(
            dataset.get("development_files"),
            field="dataset.development_files",
            expected=_DEVELOPMENT_FILES,
        )
    )
    test_files = tuple(
        str(value)
        for value in _require_exact_sequence(
            dataset.get("test_files"),
            field="dataset.test_files",
            expected=_TEST_FILES,
        )
    )
    if int(segmentation.get("pause_ms", -1)) != _PAUSE_MS:
        raise ValueError("segmentation.pause_ms is frozen at 200")
    if int(segmentation.get("minimum_frames", -1)) != _MINIMUM_FRAMES:
        raise ValueError("segmentation.minimum_frames is frozen at 60")
    if tuple(int(value) for value in _sequence(features.get("image_size"), field="features.image_size")) != _IMAGE_SIZE:
        raise ValueError("features.image_size is frozen at [60, 80]")
    if tuple(int(value) for value in _require_exact_sequence(corruption.get("offsets_ms"), field="corruption.offsets_ms", expected=_OFFSETS_MS)) != _OFFSETS_MS:
        raise ValueError("corruption.offsets_ms is frozen")
    if tuple(int(value) for value in _require_exact_sequence(corruption.get("jitter_ms"), field="corruption.jitter_ms", expected=_JITTER_MS)) != _JITTER_MS:
        raise ValueError("corruption.jitter_ms is frozen")
    if tuple(float(value) for value in _require_exact_sequence(corruption.get("dropout"), field="corruption.dropout", expected=_DROPOUT)) != _DROPOUT:
        raise ValueError("corruption.dropout is frozen")
    if tuple(int(value) for value in _require_exact_sequence(search.get("candidate_offsets_ms"), field="search.candidate_offsets_ms", expected=_CANDIDATE_OFFSETS_MS)) != _CANDIDATE_OFFSETS_MS:
        raise ValueError("search.candidate_offsets_ms is frozen")
    if float(guard.get("minimum_zero_stability", -1.0)) != 0.90:
        raise ValueError("guard.minimum_zero_stability is frozen at 0.90")
    if float(guard.get("minimum_benefit_precision", -1.0)) != 0.85:
        raise ValueError("guard.minimum_benefit_precision is frozen at 0.85")
    if float(guard.get("minimum_controlled_coverage", -1.0)) != 0.40:
        raise ValueError("guard.minimum_controlled_coverage is frozen at 0.40")
    if int(guard.get("max_calibration_candidates", -1)) != _MAX_CALIBRATION_CANDIDATES:
        raise ValueError("guard.max_calibration_candidates is frozen at 25000")
    expected_segment_counts = _mapping(
        config.get("expected_segment_counts"), field="expected_segment_counts"
    )
    normalized_segment_counts = {
        split: int(expected_segment_counts.get(split, -1))
        for split in ("development", "test")
    }
    if any(value < 1 for value in normalized_segment_counts.values()):
        raise ValueError("expected_segment_counts must contain positive development/test counts")
    if (
        Path(dataset["root"]).expanduser().resolve() == _CANONICAL_DATASET_ROOT
        and normalized_segment_counts != {"development": 31, "test": 10}
    ):
        raise ValueError("canonical Take-Pens expected_segment_counts must be 31/10")
    expected_fingerprints = _mapping(
        config.get("expected_input_fingerprints"), field="expected_input_fingerprints"
    )

    return {
        "root": str(dataset["root"]),
        "development_files": development_files,
        "test_files": test_files,
        "expected_segment_counts": normalized_segment_counts,
        "expected_input_fingerprints": dict(expected_fingerprints),
        "recovery_offset_tolerance_ms": float(evaluation["recovery_offset_tolerance_ms"]),
        "recovery_qmae_tolerance_rad": float(evaluation["recovery_qmae_tolerance_rad"]),
        "minimum_zero_stability": float(guard["minimum_zero_stability"]),
        "minimum_benefit_precision": float(guard["minimum_benefit_precision"]),
        "minimum_controlled_coverage": float(guard["minimum_controlled_coverage"]),
    }


def _segment_id(segment: ContinuousSegment) -> str:
    return f"{segment.trajectory_id}:{segment.start}:{segment.stop}"


def _array_sha256(values: Sequence[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for value in values:
        array = np.ascontiguousarray(value)
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _feature_receipt(
    trajectory: TakePensTrajectory,
    segment: ContinuousSegment,
    features: MotionFeatures,
) -> dict[str, Any]:
    return {
        "segment_id": _segment_id(segment),
        "trajectory_id": trajectory.trajectory_id,
        "split": trajectory.split,
        "start": segment.start,
        "stop": segment.stop,
        "frame_count": segment.stop - segment.start,
        "feature_sha256": _array_sha256(
            (features.camera_midpoint_ns, features.visual_motion, features.state_motion)
        ),
    }


def _trial_seed(segment: ContinuousSegment, offset_ms: int, jitter_ms: int, dropout: float) -> int:
    material = f"{_segment_id(segment)}|{offset_ms}|{jitter_ms}|{dropout:.2f}".encode("ascii")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def _build_trials(
    entries: Sequence[tuple[TakePensTrajectory, ContinuousSegment, MotionFeatures]],
) -> list[TemporalTrial]:
    trials: list[TemporalTrial] = []
    for trajectory, segment, features in entries:
        timestamp_ns = trajectory.timestamps("slave")[segment.start : segment.stop]
        qpos = trajectory.slave_qpos()[segment.start : segment.stop]
        target_qpos = interpolate_qpos(
            features.camera_midpoint_ns, timestamp_ns, qpos, mode="linear"
        )
        for offset_ms, jitter_ms, dropout in itertools.product(
            _OFFSETS_MS, _JITTER_MS, _DROPOUT
        ):
            condition_id = (
                f"{_segment_id(segment)}|offset={offset_ms}|jitter={jitter_ms}|dropout={dropout:.2f}"
            )
            trials.append(
                TemporalTrial(
                    trajectory_id=trajectory.trajectory_id,
                    split=trajectory.split,
                    condition_id=condition_id,
                    features=features,
                    corrupted_state=corrupt_state_stream(
                        timestamp_ns,
                        qpos,
                        offset_ms=offset_ms,
                        jitter_ms=jitter_ms,
                        dropout=dropout,
                        seed=_trial_seed(segment, offset_ms, jitter_ms, dropout),
                    ),
                    target_qpos=target_qpos,
                    injected_offset_ms=float(offset_ms),
                    jitter_ms=float(jitter_ms),
                    dropout=float(dropout),
                )
            )
    return trials


def _extract_entries(
    dataset: TakePensDataset, *, split: str
) -> tuple[list[tuple[TakePensTrajectory, ContinuousSegment, MotionFeatures]], list[dict[str, Any]]]:
    entries: list[tuple[TakePensTrajectory, ContinuousSegment, MotionFeatures]] = []
    receipts: list[dict[str, Any]] = []
    for trajectory in dataset.trajectories(split):
        for segment in continuous_segments(
            trajectory,
            pause_ns=_PAUSE_MS * 1_000_000,
            minimum_frames=_MINIMUM_FRAMES,
        ):
            features = extract_motion_features(trajectory, segment, image_size=_IMAGE_SIZE)
            entries.append((trajectory, segment, features))
            receipts.append(_feature_receipt(trajectory, segment, features))
    return entries, receipts


def _guard_payload(report: GuardCalibrationReport) -> dict[str, Any]:
    report_payload = report.to_dict()
    payload = {
        "schema": "kinesync.rt9_temporal_guard.v1",
        "source_split": "development",
        "test_split": "test",
        "test_split_untouched_during_calibration": True,
        "constraints": report_payload["constraints"],
        "selected_metrics": report_payload["selected_metrics"],
        "GuardCalibrationReport": report_payload,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return {**payload, "sha256": hashlib.sha256(encoded.encode("ascii")).hexdigest()}


def _write_results(path: Path, rows: Sequence[TemporalResultRow]) -> None:
    serialized = [row.to_dict() for row in rows]
    if not serialized:
        raise ValueError("RT9 test results cannot be empty")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(serialized[0]))
        writer.writeheader()
        writer.writerows(serialized)


def _trajectory_macro_mean(
    rows: Sequence[TemporalResultRow], selector
) -> float | None:
    by_trajectory: dict[str, list[TemporalResultRow]] = {}
    for row in rows:
        by_trajectory.setdefault(row.trajectory_id, []).append(row)
    if not by_trajectory:
        return None
    return float(
        np.mean(
            [
                np.mean([selector(row) for row in trajectory_rows])
                for trajectory_rows in by_trajectory.values()
            ]
        )
    )


def _calibration_validity(report: GuardCalibrationReport) -> dict[str, bool]:
    metrics = report.selected_metrics
    constraints = report.constraints
    converged = bool(report.converged)
    budget_not_exhausted = not report.budget_exhausted
    zero_stability = metrics.zero_stability >= constraints.minimum_zero_stability
    benefit_precision = (
        metrics.benefit_precision >= constraints.minimum_benefit_precision
    )
    controlled_coverage = (
        metrics.controlled_coverage >= constraints.minimum_controlled_coverage
    )
    return {
        "converged": converged,
        "budget_not_exhausted": budget_not_exhausted,
        "development_zero_stability_at_least_90pct": zero_stability,
        "development_benefit_precision_at_least_85pct": benefit_precision,
        "development_controlled_coverage_at_least_40pct": controlled_coverage,
        "formal_eligible": bool(
            converged
            and budget_not_exhausted
            and zero_stability
            and benefit_precision
            and controlled_coverage
        ),
    }


def _predeclared_gates(
    rows: Sequence[TemporalResultRow], report: GuardCalibrationReport
) -> dict[str, bool]:
    by_method = {method: [row for row in rows if row.method == method] for method in TEMPORAL_METHODS}
    guarded = by_method["kinesync_guarded"]
    raw_linear = by_method["raw_linear"]
    nonzero_guarded = [row for row in guarded if row.injected_offset_ms != 0.0]
    nonzero_raw = [row for row in raw_linear if row.injected_offset_ms != 0.0]

    jitter_dropout = [
        row
        for row in guarded
        if row.jitter_ms == 20.0 and row.dropout == 0.30 and row.injected_offset_ms != 0.0
    ]
    matched_raw = {row.condition_id: row for row in raw_linear}
    guarded_nonzero_qmae = _trajectory_macro_mean(
        nonzero_guarded, lambda row: row.qmae_rad
    )
    raw_nonzero_qmae = _trajectory_macro_mean(
        nonzero_raw, lambda row: row.qmae_rad
    )
    guarded_nonzero_recovery = _trajectory_macro_mean(
        nonzero_guarded, lambda row: float(row.recovery_success)
    )
    raw_nonzero_recovery = _trajectory_macro_mean(
        nonzero_raw, lambda row: float(row.recovery_success)
    )
    benefit_precision = _trajectory_macro_mean(
        [row for row in guarded if row.committed],
        lambda row: float(row.qmae_reduction_rad > 1e-12),
    )
    controlled_coverage = _trajectory_macro_mean(
        nonzero_guarded, lambda row: float(row.committed)
    )
    zero_stability = _trajectory_macro_mean(
        [row for row in guarded if row.injected_offset_ms == 0.0],
        lambda row: float(not row.committed),
    )
    jitter_dropout_qmae = _trajectory_macro_mean(
        jitter_dropout, lambda row: row.qmae_rad
    )
    jitter_dropout_raw_qmae = _trajectory_macro_mean(
        [matched_raw[row.condition_id] for row in jitter_dropout],
        lambda row: row.qmae_rad,
    )
    validity = _calibration_validity(report)
    gates = {
        "nonzero_qmae_improves_over_raw_linear": bool(
            guarded_nonzero_qmae is not None
            and raw_nonzero_qmae is not None
            and guarded_nonzero_qmae < raw_nonzero_qmae
        ),
        "nonzero_recovery_improves_over_raw_linear": bool(
            guarded_nonzero_recovery is not None
            and raw_nonzero_recovery is not None
            and guarded_nonzero_recovery > raw_nonzero_recovery
        ),
        "benefit_precision_at_least_85pct": bool(
            benefit_precision is not None and benefit_precision >= 0.85
        ),
        "controlled_coverage_at_least_40pct": bool(
            controlled_coverage is not None and controlled_coverage >= 0.40
        ),
        "zero_offset_stability_at_least_90pct": bool(
            zero_stability is not None and zero_stability >= 0.90
        ),
        "positive_at_20ms_jitter_and_30pct_dropout": bool(
            jitter_dropout_qmae is not None
            and jitter_dropout_raw_qmae is not None
            and jitter_dropout_qmae < jitter_dropout_raw_qmae
        ),
        "calibration_converged": validity["converged"],
        "calibration_budget_not_exhausted": validity["budget_not_exhausted"],
        "development_zero_stability_at_least_90pct": validity[
            "development_zero_stability_at_least_90pct"
        ],
        "development_benefit_precision_at_least_85pct": validity[
            "development_benefit_precision_at_least_85pct"
        ],
        "development_controlled_coverage_at_least_40pct": validity[
            "development_controlled_coverage_at_least_40pct"
        ],
        "overall_formal_eligible": validity["formal_eligible"],
    }
    gates["overall_formal_pass"] = bool(
        gates["overall_formal_eligible"]
        and all(
            gates[name]
            for name in (
                "nonzero_qmae_improves_over_raw_linear",
                "nonzero_recovery_improves_over_raw_linear",
                "benefit_precision_at_least_85pct",
                "controlled_coverage_at_least_40pct",
                "zero_offset_stability_at_least_90pct",
                "positive_at_20ms_jitter_and_30pct_dropout",
            )
        )
    )
    return gates


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _media_artifact(path: Path, *, run_path: Path, **extra: Any) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"RT9 media artifact is missing or empty: {path}")
    return {
        "relative_path": path.relative_to(run_path).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
        **extra,
    }


def _select_temporal_visual_rows(
    rows: Sequence[TemporalResultRow],
) -> list[tuple[TemporalResultRow, TemporalResultRow, dict[str, Any]]]:
    """Choose one held-out condition per trajectory without changing outcomes."""

    raw_by_trajectory: dict[str, dict[str, TemporalResultRow]] = {}
    guarded_by_trajectory: dict[str, dict[str, TemporalResultRow]] = {}
    for row in rows:
        if row.split != "test":
            continue
        if row.method == "raw_linear":
            raw_by_trajectory.setdefault(row.trajectory_id, {})[row.condition_id] = row
        elif row.method == "kinesync_guarded":
            guarded_by_trajectory.setdefault(row.trajectory_id, {})[row.condition_id] = row
    if set(raw_by_trajectory) != set(guarded_by_trajectory):
        raise ValueError("RT9 visual selection requires matched raw and guarded rows")

    selected: list[tuple[TemporalResultRow, TemporalResultRow, dict[str, Any]]] = []
    for trajectory_id in sorted(raw_by_trajectory):
        raw_by_condition = raw_by_trajectory[trajectory_id]
        guarded_by_condition = guarded_by_trajectory[trajectory_id]
        if set(raw_by_condition) != set(guarded_by_condition):
            raise ValueError("RT9 visual selection requires matched condition identifiers")
        beneficial = sorted(
            (
                guarded
                for guarded in guarded_by_condition.values()
                if guarded.committed and guarded.qmae_reduction_rad > 1e-12
            ),
            key=lambda row: row.condition_id,
        )
        if beneficial:
            median_gain = float(np.median([row.qmae_reduction_rad for row in beneficial]))
            guarded = min(
                beneficial,
                key=lambda row: (
                    abs(row.qmae_reduction_rad - median_gain),
                    row.condition_id,
                ),
            )
            selection_kind = "beneficial_committed"
        else:
            guarded = min(
                guarded_by_condition.values(),
                key=lambda row: (-abs(row.injected_offset_ms), row.condition_id),
            )
            median_gain = None
            selection_kind = "raw_fallback"
        raw = raw_by_condition[guarded.condition_id]
        selected.append(
            (
                raw,
                guarded,
                {
                    "trajectory_id": trajectory_id,
                    "selection_rule": _MEDIA_SELECTION_RULE,
                    "selection_kind": selection_kind,
                    "fallback_rule": (
                        None
                        if beneficial
                        else "largest_absolute_injected_offset_then_condition_id_v1"
                    ),
                    "beneficial_committed_condition_count": len(beneficial),
                    "beneficial_committed_condition_ids": [
                        row.condition_id for row in beneficial
                    ],
                    "trajectory_median_qmae_gain_rad": median_gain,
                    "selected_condition_id": guarded.condition_id,
                    "selected_qmae_gain_rad": float(guarded.qmae_reduction_rad),
                    "raw_qmae_rad": float(raw.qmae_rad),
                    "guarded_qmae_rad": float(guarded.qmae_rad),
                    "guarded_committed": bool(guarded.committed),
                    "safety_label": _MEDIA_SAFETY_LABEL,
                },
            )
        )
    if not selected:
        raise ValueError("RT9 visual selection needs at least one held-out trajectory")
    return selected


def _visual_context(
    selection: _TemporalVisualSelection,
    *,
    frame_index: int,
    head_rgb: np.ndarray,
    wrist_rgb: np.ndarray,
) -> TemporalTelemetryVisualFrame:
    raw_error = np.abs(
        selection.raw.interpolated_qpos[frame_index, :6]
        - selection.trial.target_qpos[frame_index, :6]
    )
    guarded_error = np.abs(
        selection.guarded.interpolated_qpos[frame_index, :6]
        - selection.trial.target_qpos[frame_index, :6]
    )
    relative_score_profile = tuple(
        OffsetScore(
            offset_ms=score.offset_ms - selection.guarded.native_delay_ms,
            correlation=score.correlation,
        )
        for score in selection.estimate.score_profile
    )
    return TemporalTelemetryVisualFrame(
        trajectory_id=selection.trajectory.trajectory_id,
        condition_id=selection.trial.condition_id,
        frame_index=frame_index,
        frame_count=len(selection.trial.target_qpos),
        head_rgb=head_rgb,
        wrist_rgb=wrist_rgb,
        score_profile=relative_score_profile,
        raw_joint_abs_error_rad=raw_error,
        guarded_joint_abs_error_rad=guarded_error,
        injected_offset_ms=selection.trial.injected_offset_ms,
        estimated_offset_ms=selection.guarded.estimated_offset_ms,
        committed_offset_ms=selection.guarded.applied_correction_ms,
        guarded_committed=selection.guarded.committed,
    )


def _write_temporal_media(
    *,
    run: RunArtifacts,
    rows: Sequence[TemporalResultRow],
    entries: Sequence[tuple[TakePensTrajectory, ContinuousSegment, MotionFeatures]],
    trials: Sequence[TemporalTrial],
    estimates: Mapping[str, OffsetEstimate],
) -> dict[str, Any]:
    entries_by_segment = {_segment_id(segment): (trajectory, segment) for trajectory, segment, _ in entries}
    trials_by_condition = {trial.condition_id: trial for trial in trials}
    selections: list[_TemporalVisualSelection] = []
    for raw, guarded, receipt in _select_temporal_visual_rows(rows):
        trial = trials_by_condition.get(guarded.condition_id)
        if trial is None:
            raise ValueError("RT9 visual condition was not found in test trials")
        segment_id = guarded.condition_id.split("|offset=", 1)[0]
        entry = entries_by_segment.get(segment_id)
        estimate = estimates.get(guarded.condition_id)
        if entry is None or estimate is None:
            raise ValueError("RT9 visual condition is missing recorded telemetry")
        trajectory, segment = entry
        selections.append(
            _TemporalVisualSelection(
                trajectory=trajectory,
                segment=segment,
                trial=trial,
                estimate=estimate,
                raw=raw,
                guarded=guarded,
                receipt={**receipt, "segment_id": segment_id},
            )
        )
    if len(selections) > 2:
        selections = selections[:2]
    video_frame_count = min(
        _MEDIA_MAX_VIDEO_FRAMES,
        *(len(selection.trial.target_qpos) for selection in selections),
    )
    if video_frame_count < 1:
        raise ValueError("RT9 visual telemetry selection has no frames")

    timelines = [
        np.linspace(0, len(selection.trial.target_qpos) - 1, video_frame_count)
        .round()
        .astype(np.int64)
        for selection in selections
    ]
    recorded_rgb: list[tuple[np.ndarray, np.ndarray]] = []
    for selection, indices in zip(selections, timelines, strict=True):
        source_indices = selection.segment.start + np.minimum(indices + 1, selection.segment.stop - selection.segment.start - 1)
        recorded_rgb.append(
            (
                selection.trajectory.read_rgb("head", source_indices),
                selection.trajectory.read_rgb("wrist", source_indices),
            )
        )
    sequence: list[list[TemporalTelemetryVisualFrame]] = []
    for video_index in range(video_frame_count):
        sequence.append(
            [
                _visual_context(
                    selection,
                    frame_index=int(timelines[selection_index][video_index]),
                    head_rgb=recorded_rgb[selection_index][0][video_index],
                    wrist_rgb=recorded_rgb[selection_index][1][video_index],
                )
                for selection_index, selection in enumerate(selections)
            ]
        )
    image_path = run.path / "images" / "rt9_temporal_telemetry.png"
    video_path = run.path / "videos" / "rt9_temporal_telemetry.mp4"
    write_temporal_telemetry_image(image_path, sequence[video_frame_count // 2])
    write_temporal_telemetry_video(video_path, sequence, fps=_MEDIA_VIDEO_FPS)
    return {
        "selection_rule": _MEDIA_SELECTION_RULE,
        "selection_receipts": [selection.receipt for selection in selections],
        "image": _media_artifact(image_path, run_path=run.path),
        "video": _media_artifact(
            video_path,
            run_path=run.path,
            frame_count=video_frame_count,
            fps=_MEDIA_VIDEO_FPS,
        ),
    }


def execute_rt9_temporal_bridge(
    config: Mapping[str, Any], *, run_id: str | None = None
) -> Path:
    """Execute one immutable, offline-only RT9 temporal benchmark run."""

    protocol = _protocol_config(config)
    dataset = TakePensDataset(
        protocol["root"], protocol["development_files"], protocol["test_files"]
    )
    assets = {
        f"take_pens_{name}": str(Path(protocol["root"]) / name)
        for name in (*protocol["development_files"], *protocol["test_files"])
    }
    run = RunArtifacts.create(
        root=config.get("run_root", "runs"),
        experiment="rt9_take_pens_temporal_bridge",
        config=config,
        assets=assets,
        target_provenance="real_telemetry_injected_time_offset",
        observation_backend="dual_camera_motion_correlation_offline_telemetry",
        run_id=run_id,
    )
    if Path(protocol["root"]).expanduser().resolve() == _CANONICAL_DATASET_ROOT:
        actual_fingerprints = {
            name: {
                "sha256": record["sha256"],
                "sha256_mode": record["sha256_mode"],
            }
            for name, record in json.loads(
                (run.path / "manifest.json").read_text(encoding="utf-8")
            )["assets"].items()
        }
        if actual_fingerprints != protocol["expected_input_fingerprints"]:
            raise ValueError("canonical Take-Pens input fingerprints do not match config")

    development_entries, development_receipts = _extract_entries(dataset, split="development")
    if len(development_entries) != protocol["expected_segment_counts"]["development"]:
        raise ValueError("development segment count does not match the frozen contract")
    development_trials = _build_trials(development_entries)
    constraints = TemporalGuardConstraints(
        minimum_zero_stability=protocol["minimum_zero_stability"],
        minimum_benefit_precision=protocol["minimum_benefit_precision"],
        minimum_controlled_coverage=protocol["minimum_controlled_coverage"],
        recovery_offset_tolerance_ms=protocol["recovery_offset_tolerance_ms"],
        recovery_qmae_tolerance_rad=protocol["recovery_qmae_tolerance_rad"],
        candidate_offsets_ms=tuple(float(value) for value in _CANDIDATE_OFFSETS_MS),
        max_calibration_candidates=_MAX_CALIBRATION_CANDIDATES,
    )
    report = calibrate_temporal_guard_with_report(development_trials, constraints)
    guard_payload = _guard_payload(report)
    run.write_json("guard.json", guard_payload)
    guard: TemporalGuard = report.guard

    test_entries, test_receipts = _extract_entries(dataset, split="test")
    if len(test_entries) != protocol["expected_segment_counts"]["test"]:
        raise ValueError("test segment count does not match the frozen contract")
    test_trials = _build_trials(test_entries)
    result_rows: list[TemporalResultRow] = []
    estimates_by_condition: dict[str, OffsetEstimate] = {}
    for trial in test_trials:
        estimate = search_motion_offset(
            trial.features,
            trial.corrupted_state,
            candidate_offsets_ms=tuple(float(value) for value in _CANDIDATE_OFFSETS_MS),
        )
        estimates_by_condition[trial.condition_id] = estimate
        result_rows.extend(
            evaluate_temporal_method(
                trial,
                method,
                guard=guard,
                estimate=estimate,
                recovery_offset_tolerance_ms=constraints.recovery_offset_tolerance_ms,
                recovery_qmae_tolerance_rad=constraints.recovery_qmae_tolerance_rad,
            )
            for method in TEMPORAL_METHODS
        )

    feature_payload = {
        "schema": "kinesync.rt9_motion_features.v1",
        "feature_parameters": {"image_size": list(_IMAGE_SIZE)},
        "segmentation": {"pause_ms": _PAUSE_MS, "minimum_frames": _MINIMUM_FRAMES},
        "segment_counts": {"development": len(development_entries), "test": len(test_entries)},
        "feature_receipt_count": len(development_receipts) + len(test_receipts),
        "segment_receipts": development_receipts + test_receipts,
    }
    run.write_json("features.json", feature_payload)
    _write_results(run.path / "results.csv", result_rows)
    media = _write_temporal_media(
        run=run,
        rows=result_rows,
        entries=test_entries,
        trials=test_trials,
        estimates=estimates_by_condition,
    )
    media_paths = sorted(
        (media["image"]["relative_path"], media["video"]["relative_path"])
    )
    summary = summarize_temporal_results(result_rows)
    metrics = {
        "schema": "kinesync.rt9_temporal_metrics.v1",
        "test_trial_count": len(test_trials),
        "test_method_row_count": len(result_rows),
        "summary": summary.to_dict(),
        "calibration_validity": _calibration_validity(report),
        "predeclared_gates": _predeclared_gates(result_rows, report),
        "trajectory_clustering": "descriptive_two_held_out_trajectories_not_independent_robot_episodes",
        "media_paths": media_paths,
        "media": media,
    }
    run.write_metrics(metrics)
    output_hashes = run.fingerprint_outputs(
        ("config.yaml", "guard.json", "features.json", "results.csv", "metrics.json")
    )
    output_hashes.update(
        {
            media["image"]["relative_path"]: {
                "size_bytes": media["image"]["size_bytes"],
                "sha256": media["image"]["sha256"],
                "sha256_mode": "full",
            },
            media["video"]["relative_path"]: {
                "size_bytes": media["video"]["size_bytes"],
                "sha256": media["video"]["sha256"],
                "sha256_mode": "full",
            },
        }
    )
    run.append_manifest(
        {
            "evidence_scope": _EVIDENCE_SCOPE,
            "guard_sha256": guard_payload["sha256"],
            "calibration_validity": metrics["calibration_validity"],
            "output_hashes": output_hashes,
            "media_paths": media_paths,
            "media": media,
        }
    )
    return run.path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="RT9 YAML configuration path")
    parser.add_argument("--run-id", default=None, help="Immutable run directory name")
    arguments = parser.parse_args()
    print(execute_rt9_temporal_bridge(load_config(arguments.config), run_id=arguments.run_id))


if __name__ == "__main__":
    main()
