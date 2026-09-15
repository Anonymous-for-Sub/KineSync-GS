"""Visible, result-bound RT9 temporal comparison cases."""

from __future__ import annotations

from dataclasses import dataclass
import csv
from pathlib import Path
from typing import Mapping, Sequence

import cv2
import h5py
import numpy as np

from .robust_v2_visuals import AlignmentCase


@dataclass(frozen=True)
class SegmentCondition:
    trajectory_id: str
    condition_id: str
    start: int
    stop: int
    baseline_residual_ms: float
    ours_residual_ms: float
    baseline_qmae: float
    ours_qmae: float

    @property
    def qmae_reduction(self) -> float:
        return self.baseline_qmae - self.ours_qmae


@dataclass(frozen=True)
class TemporalFrameChoice:
    true_index: int
    baseline_index: int
    ours_index: int
    baseline_qmae: float
    ours_qmae: float
    pixel_mae: float


@dataclass(frozen=True)
class RT9VisualCase:
    condition: SegmentCondition
    camera: str
    frame: TemporalFrameChoice
    real_bgr: np.ndarray
    baseline_bgr: np.ndarray
    ours_bgr: np.ndarray

    @property
    def alias_camera(self) -> str:
        return f"{Path(self.condition.trajectory_id).stem}-{self.camera}"

    def alignment_case(self) -> AlignmentCase:
        return AlignmentCase(
            camera=self.alias_camera,
            true_index=self.frame.true_index,
            baseline_index=self.frame.baseline_index,
            ours_index=self.frame.ours_index,
            baseline_qmae=self.frame.baseline_qmae,
            ours_qmae=self.frame.ours_qmae,
            pixel_mae=self.frame.pixel_mae,
            score_gain=self.frame.pixel_mae,
        )


def _segment(condition_id: str) -> tuple[int, int]:
    head = condition_id.split("|", 1)[0]
    fields = head.rsplit(":", 2)
    if len(fields) != 3:
        raise ValueError(f"Malformed RT9 condition: {condition_id}")
    start, stop = int(fields[1]), int(fields[2])
    if start < 0 or stop <= start:
        raise ValueError(f"Malformed RT9 segment: {condition_id}")
    return start, stop


def select_best_segment_conditions(results_csv: Path) -> tuple[SegmentCondition, ...]:
    with Path(results_csv).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    raw = {
        (row["trajectory_id"], row["condition_id"]): row
        for row in rows
        if row["method"] == "raw_linear"
    }
    best: dict[tuple[str, int, int], SegmentCondition] = {}
    for row in rows:
        if row["method"] != "kinesync_guarded" or str(row.get("committed", "")).lower() not in {"1", "true"}:
            continue
        key = (row["trajectory_id"], row["condition_id"])
        if key not in raw:
            continue
        baseline = raw[key]
        start, stop = _segment(row["condition_id"])
        candidate = SegmentCondition(
            trajectory_id=row["trajectory_id"],
            condition_id=row["condition_id"],
            start=start,
            stop=stop,
            baseline_residual_ms=float(baseline["estimated_offset_ms"]) - float(baseline["applied_correction_ms"]),
            ours_residual_ms=float(row["estimated_offset_ms"]) - float(row["applied_correction_ms"]),
            baseline_qmae=float(baseline["qmae_rad"]),
            ours_qmae=float(row["qmae_rad"]),
        )
        segment_key = (candidate.trajectory_id, start, stop)
        if candidate.qmae_reduction > 0.0 and (
            segment_key not in best or candidate.qmae_reduction > best[segment_key].qmae_reduction
        ):
            best[segment_key] = candidate
    return tuple(sorted(best.values(), key=lambda item: (item.trajectory_id, item.start)))


def choose_visible_temporal_frame(
    images: np.ndarray,
    qpos: np.ndarray,
    *,
    start: int,
    stop: int,
    baseline_shift: int,
    ours_shift: int,
) -> TemporalFrameChoice:
    if images.ndim != 4 or images.shape[-1] != 3 or qpos.ndim != 2 or len(images) != len(qpos):
        raise ValueError("Temporal images and qpos must be aligned frame arrays")
    choices: list[tuple[float, TemporalFrameChoice]] = []
    lower = max(start, -baseline_shift, -ours_shift)
    upper = min(stop, len(images) - baseline_shift, len(images) - ours_shift)
    for true_index in range(lower, upper):
        baseline_index = true_index + baseline_shift
        ours_index = true_index + ours_shift
        baseline_qmae = float(np.mean(np.abs(qpos[baseline_index] - qpos[true_index])))
        ours_qmae = float(np.mean(np.abs(qpos[ours_index] - qpos[true_index])))
        if baseline_qmae <= ours_qmae:
            continue
        first = cv2.resize(images[baseline_index], (160, 120), interpolation=cv2.INTER_AREA).astype(np.float32)
        second = cv2.resize(images[ours_index], (160, 120), interpolation=cv2.INTER_AREA).astype(np.float32)
        pixel_mae = float(np.abs(first - second).mean())
        choice = TemporalFrameChoice(true_index, baseline_index, ours_index, baseline_qmae, ours_qmae, pixel_mae)
        choices.append((pixel_mae + 1000.0 * (baseline_qmae - ours_qmae), choice))
    if not choices:
        raise ValueError("RT9 segment contains no positive visible temporal frame")
    return max(choices, key=lambda item: (item[0], -item[1].true_index))[1]


def load_rt9_visual_cases(
    conditions: Sequence[SegmentCondition], dataset_root: Path
) -> tuple[RT9VisualCase, ...]:
    by_trajectory: dict[str, list[SegmentCondition]] = {}
    for condition in conditions:
        by_trajectory.setdefault(condition.trajectory_id, []).append(condition)
    cases = []
    for trajectory_id, trajectory_conditions in sorted(by_trajectory.items()):
        path = Path(dataset_root) / trajectory_id
        with h5py.File(path, "r") as handle:
            qpos = np.asarray(handle["left_arm/joint"], dtype=np.float64)
            for camera in ("head", "wrist"):
                dataset = handle[f"cam_{camera}/color"]
                images = np.asarray(dataset, dtype=np.uint8)
                timestamps = np.asarray(handle[f"cam_{camera}/timestamp"], dtype=np.int64)
                period_ns = float(np.median(np.diff(timestamps)))
                for condition in trajectory_conditions:
                    baseline_shift = int(round(condition.baseline_residual_ms * 1e6 / period_ns))
                    ours_shift = int(round(condition.ours_residual_ms * 1e6 / period_ns))
                    choice = choose_visible_temporal_frame(
                        images,
                        qpos,
                        start=condition.start,
                        stop=condition.stop,
                        baseline_shift=baseline_shift,
                        ours_shift=ours_shift,
                    )
                    cases.append(
                        RT9VisualCase(
                            condition=condition,
                            camera=camera,
                            frame=choice,
                            real_bgr=cv2.cvtColor(images[choice.true_index], cv2.COLOR_RGB2BGR),
                            baseline_bgr=cv2.cvtColor(images[choice.baseline_index], cv2.COLOR_RGB2BGR),
                            ours_bgr=cv2.cvtColor(images[choice.ours_index], cv2.COLOR_RGB2BGR),
                        )
                    )
    return tuple(cases)


def load_rt9_sequence(
    condition: SegmentCondition,
    dataset_root: Path,
    *,
    camera: str,
    count: int,
) -> tuple[RT9VisualCase, ...]:
    path = Path(dataset_root) / condition.trajectory_id
    with h5py.File(path, "r") as handle:
        images = np.asarray(handle[f"cam_{camera}/color"], dtype=np.uint8)
        timestamps = np.asarray(handle[f"cam_{camera}/timestamp"], dtype=np.int64)
        qpos = np.asarray(handle["left_arm/joint"], dtype=np.float64)
    period_ns = float(np.median(np.diff(timestamps)))
    baseline_shift = int(round(condition.baseline_residual_ms * 1e6 / period_ns))
    ours_shift = int(round(condition.ours_residual_ms * 1e6 / period_ns))
    lower = max(condition.start, -baseline_shift, -ours_shift)
    upper = min(condition.stop, len(images) - baseline_shift, len(images) - ours_shift)
    indices = np.linspace(lower, upper - 1, count).round().astype(int)
    cases = []
    for true_index in indices:
        baseline_index, ours_index = int(true_index + baseline_shift), int(true_index + ours_shift)
        baseline_qmae = float(np.mean(np.abs(qpos[baseline_index] - qpos[true_index])))
        ours_qmae = float(np.mean(np.abs(qpos[ours_index] - qpos[true_index])))
        pixel_mae = float(np.abs(images[baseline_index].astype(np.float32) - images[ours_index].astype(np.float32)).mean())
        choice = TemporalFrameChoice(int(true_index), baseline_index, ours_index, baseline_qmae, ours_qmae, pixel_mae)
        cases.append(
            RT9VisualCase(
                condition, camera, choice,
                cv2.cvtColor(images[true_index], cv2.COLOR_RGB2BGR),
                cv2.cvtColor(images[baseline_index], cv2.COLOR_RGB2BGR),
                cv2.cvtColor(images[ours_index], cv2.COLOR_RGB2BGR),
            )
        )
    return tuple(cases)


def as_alignment_bundle(cases: Sequence[RT9VisualCase]) -> tuple[tuple[AlignmentCase, ...], dict[tuple[str, int], tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    alignments = tuple(case.alignment_case() for case in cases)
    images = {
        (case.alias_camera, case.frame.true_index): (case.real_bgr, case.baseline_bgr, case.ours_bgr)
        for case in cases
    }
    if len(images) != len(cases):
        raise ValueError("RT9 visual case aliases are not unique")
    return alignments, images


__all__ = [
    "RT9VisualCase", "SegmentCondition", "TemporalFrameChoice", "as_alignment_bundle",
    "choose_visible_temporal_frame", "load_rt9_sequence", "load_rt9_visual_cases",
    "select_best_segment_conditions",
]
