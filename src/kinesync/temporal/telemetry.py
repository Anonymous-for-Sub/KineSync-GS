"""Motion telemetry derived from recorded Take-Pens trajectories."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Iterable, Mapping, Sequence

import cv2
import numpy as np

from kinesync.data.take_pens import ContinuousSegment, TakePensTrajectory
from kinesync.data.embodiment import (
    JointSelection,
    default_piper_joint_selection,
    selected_joint_mae,
)


DEFAULT_OFFSET_CANDIDATES_MS = tuple(float(value) for value in range(-200, 201, 20))
GUARD_CALIBRATION_STRATEGY = "deterministic_bounded_coordinate_event_search_v1"
TEMPORAL_METHODS = (
    "raw_nearest",
    "raw_linear",
    "motion_unguarded",
    "kinesync_guarded",
    "oracle_offset",
)


@dataclass(frozen=True)
class MotionFeatures:
    """Aligned visual and joint-motion signals for one continuous segment."""

    camera_midpoint_ns: np.ndarray
    visual_motion: np.ndarray
    state_motion: np.ndarray
    joint_selection: JointSelection | None = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class CorruptedState:
    """State samples after deterministic timestamp-only corruption."""

    timestamp_ns: np.ndarray
    qpos: np.ndarray
    retained_indices: np.ndarray


@dataclass(frozen=True)
class OffsetScore:
    """One candidate correction and its finite correlation score, if available."""

    offset_ms: float
    correlation: float | None

    def to_dict(self) -> dict[str, float | None]:
        return {"offset_ms": float(self.offset_ms), "correlation": self.correlation}


@dataclass(frozen=True)
class OffsetEstimate:
    """Motion-correlation correction estimate with its complete score profile."""

    offset_ms: float
    peak_correlation: float
    peak_margin: float
    visual_mad: float
    state_mad: float
    score_profile: tuple[OffsetScore, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "offset_ms": float(self.offset_ms),
            "peak_correlation": _finite_or_none(self.peak_correlation),
            "peak_margin": _finite_or_none(self.peak_margin),
            "visual_mad": _finite_or_none(self.visual_mad),
            "state_mad": _finite_or_none(self.state_mad),
            "score_profile": [score.to_dict() for score in self.score_profile],
        }


@dataclass(frozen=True)
class TemporalTrial:
    """One synthetic timestamp-corruption condition with its untouched target."""

    trajectory_id: str
    split: str
    condition_id: str
    features: MotionFeatures
    corrupted_state: CorruptedState
    target_qpos: np.ndarray
    injected_offset_ms: float
    jitter_ms: float
    dropout: float
    joint_selection: JointSelection = field(default_factory=default_piper_joint_selection)

    def __post_init__(self) -> None:
        if not self.trajectory_id or not self.condition_id:
            raise ValueError("temporal trial identifiers must be nonempty")
        if self.split not in {"development", "test"}:
            raise ValueError("temporal trial split must be development or test")
        if not np.isfinite(self.injected_offset_ms) or not np.isfinite(self.jitter_ms):
            raise ValueError("temporal trial offsets and jitter must be finite")
        if self.jitter_ms < 0 or not 0 <= self.dropout <= 1:
            raise ValueError("temporal trial jitter/dropout values are invalid")
        midpoint_ns = _validate_motion_features(self.features)
        _, positions = _validate_state_stream(
            self.corrupted_state.timestamp_ns, self.corrupted_state.qpos
        )
        target_qpos = np.asarray(self.target_qpos)
        if (
            target_qpos.ndim != 2
            or target_qpos.shape[0] != len(midpoint_ns)
            or target_qpos.shape[1] > positions.shape[1]
            or not np.isfinite(target_qpos).all()
        ):
            raise ValueError("temporal trial target_qpos must be finite state samples")
        if not isinstance(self.joint_selection, JointSelection):
            raise TypeError("temporal trial joint_selection must be a JointSelection")
        if self.joint_selection.unit != "rad":
            raise ValueError("temporal qMAE requires a radians joint selection")
        if self.features.joint_selection is not None:
            _require_radian_selection(
                self.features.joint_selection, field="motion feature selection"
            )
            if self.features.joint_selection != self.joint_selection:
                raise ValueError("motion feature selection does not match trial selection")
        self.joint_selection.validate_width(positions.shape[1], field="corrupted qpos")
        self.joint_selection.validate_width(target_qpos.shape[1], field="target qpos")

    def to_dict(self) -> dict[str, str | float]:
        return {
            "trajectory_id": self.trajectory_id,
            "split": self.split,
            "condition_id": self.condition_id,
            "injected_offset_ms": float(self.injected_offset_ms),
            "jitter_ms": float(self.jitter_ms),
            "dropout": float(self.dropout),
        }


@dataclass(frozen=True)
class TemporalGuardConstraints:
    """Development-only acceptance requirements for temporal guard selection."""

    minimum_zero_stability: float = 0.90
    minimum_benefit_precision: float = 0.85
    minimum_controlled_coverage: float = 0.40
    recovery_offset_tolerance_ms: float = 20.0
    recovery_qmae_tolerance_rad: float = 0.01
    candidate_offsets_ms: tuple[float, ...] = DEFAULT_OFFSET_CANDIDATES_MS
    max_calibration_candidates: int = 10_000

    def __post_init__(self) -> None:
        if not 0 <= self.minimum_zero_stability <= 1:
            raise ValueError("minimum_zero_stability must lie in [0, 1]")
        if not 0 <= self.minimum_benefit_precision <= 1:
            raise ValueError("minimum_benefit_precision must lie in [0, 1]")
        if not 0 <= self.minimum_controlled_coverage <= 1:
            raise ValueError("minimum_controlled_coverage must lie in [0, 1]")
        if self.recovery_offset_tolerance_ms < 0 or self.recovery_qmae_tolerance_rad < 0:
            raise ValueError("temporal recovery tolerances must be nonnegative")
        if (
            not isinstance(self.max_calibration_candidates, int)
            or isinstance(self.max_calibration_candidates, bool)
            or self.max_calibration_candidates < 1
        ):
            raise ValueError("max_calibration_candidates must be a positive integer")
        _normalize_candidate_offsets(self.candidate_offsets_ms)

    def to_dict(self) -> dict[str, object]:
        return {
            "minimum_zero_stability": float(self.minimum_zero_stability),
            "minimum_benefit_precision": float(self.minimum_benefit_precision),
            "minimum_controlled_coverage": float(self.minimum_controlled_coverage),
            "recovery_offset_tolerance_ms": float(self.recovery_offset_tolerance_ms),
            "recovery_qmae_tolerance_rad": float(self.recovery_qmae_tolerance_rad),
            "candidate_offsets_ms": [float(value) for value in self.candidate_offsets_ms],
            "max_calibration_candidates": int(self.max_calibration_candidates),
        }


@dataclass(frozen=True)
class TemporalGuard:
    """Frozen train-only thresholds for a relative temporal correction."""

    native_delay_ms: float
    min_peak_correlation: float
    min_peak_margin: float
    min_visual_mad: float
    min_state_mad: float
    min_correction_ms: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.native_delay_ms):
            raise ValueError("native_delay_ms must be finite")
        if not -1 <= self.min_peak_correlation <= 1:
            raise ValueError("min_peak_correlation must lie in [-1, 1]")
        if any(
            value < 0 or not np.isfinite(value)
            for value in (
                self.min_peak_margin,
                self.min_visual_mad,
                self.min_state_mad,
                self.min_correction_ms,
            )
        ):
            raise ValueError("temporal guard thresholds must be finite and nonnegative")

    def to_dict(self) -> dict[str, float]:
        return {
            "native_delay_ms": float(self.native_delay_ms),
            "min_peak_correlation": float(self.min_peak_correlation),
            "min_peak_margin": float(self.min_peak_margin),
            "min_visual_mad": float(self.min_visual_mad),
            "min_state_mad": float(self.min_state_mad),
            "min_correction_ms": float(self.min_correction_ms),
        }


@dataclass(frozen=True)
class TemporalGuardDecision:
    """One immutable commit decision from a frozen temporal guard."""

    accepted: bool
    correction_ms: float
    reason: str


@dataclass(frozen=True)
class GuardCalibrationMetrics:
    """Selected guard's auditable development-only calibration outcomes."""

    controlled_trial_count: int
    committed_controlled_count: int
    successful_controlled_count: int
    successful_committed_count: int
    beneficial_committed_count: int
    zero_trial_count: int
    stable_zero_count: int
    controlled_recovery: float
    controlled_qmae_rad: float
    benefit_precision: float
    controlled_coverage: float
    zero_stability: float

    def __post_init__(self) -> None:
        counts = (
            self.controlled_trial_count,
            self.committed_controlled_count,
            self.successful_controlled_count,
            self.successful_committed_count,
            self.beneficial_committed_count,
            self.zero_trial_count,
            self.stable_zero_count,
        )
        if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in counts):
            raise ValueError("guard calibration counts must be nonnegative integers")
        if self.controlled_trial_count == 0 or self.zero_trial_count == 0:
            raise ValueError("guard calibration requires controlled and zero development trials")
        if (
            self.committed_controlled_count > self.controlled_trial_count
            or self.successful_controlled_count > self.controlled_trial_count
            or self.successful_committed_count > self.committed_controlled_count
            or self.beneficial_committed_count > self.committed_controlled_count
            or self.stable_zero_count > self.zero_trial_count
        ):
            raise ValueError("guard calibration counts are inconsistent")
        expected = {
            "controlled_recovery": self.successful_controlled_count / self.controlled_trial_count,
            "benefit_precision": (
                self.beneficial_committed_count / self.committed_controlled_count
                if self.committed_controlled_count
                else 0.0
            ),
            "controlled_coverage": self.committed_controlled_count / self.controlled_trial_count,
            "zero_stability": self.stable_zero_count / self.zero_trial_count,
        }
        actuals = {
            "controlled_recovery": float(self.controlled_recovery),
            "benefit_precision": float(self.benefit_precision),
            "controlled_coverage": float(self.controlled_coverage),
            "zero_stability": float(self.zero_stability),
        }
        for name, value in expected.items():
            actual = actuals[name]
            if not np.isfinite(actual) or not np.isclose(actual, value):
                raise ValueError(f"guard calibration {name} does not match its counts")
        if not np.isfinite(self.controlled_qmae_rad) or self.controlled_qmae_rad < 0:
            raise ValueError("controlled calibration qMAE must be finite and nonnegative")

    def to_dict(self) -> dict[str, int | float]:
        return {
            "controlled_trial_count": int(self.controlled_trial_count),
            "committed_controlled_count": int(self.committed_controlled_count),
            "successful_controlled_count": int(self.successful_controlled_count),
            "successful_committed_count": int(self.successful_committed_count),
            "beneficial_committed_count": int(self.beneficial_committed_count),
            "zero_trial_count": int(self.zero_trial_count),
            "stable_zero_count": int(self.stable_zero_count),
            "controlled_recovery": float(self.controlled_recovery),
            "controlled_qmae_rad": float(self.controlled_qmae_rad),
            "benefit_precision": float(self.benefit_precision),
            "exact_recovery_precision": float(self.exact_recovery_precision),
            "controlled_coverage": float(self.controlled_coverage),
            "zero_stability": float(self.zero_stability),
        }

    @property
    def committed_precision(self) -> float:
        """Backward-compatible name for benefit precision, not exact recovery."""

        return self.benefit_precision

    @property
    def exact_recovery_precision(self) -> float:
        if not self.committed_controlled_count:
            return 0.0
        return self.successful_committed_count / self.committed_controlled_count


@dataclass(frozen=True)
class GuardCalibrationReport:
    """Bounded development-only guard calibration outcome."""

    guard: TemporalGuard
    constraints: TemporalGuardConstraints
    selected_metrics: GuardCalibrationMetrics
    strategy: str
    evaluated_candidate_count: int
    iterations: int
    budget_exhausted: bool
    converged: bool

    def __post_init__(self) -> None:
        if self.strategy != GUARD_CALIBRATION_STRATEGY:
            raise ValueError("unknown temporal guard calibration strategy")
        if self.evaluated_candidate_count < 0 or self.iterations < 0:
            raise ValueError("temporal guard calibration counts must be nonnegative")
        if self.budget_exhausted and self.converged:
            raise ValueError("budget-exhausted temporal calibration cannot converge")

    def to_dict(self) -> dict[str, object]:
        return {
            "guard": self.guard.to_dict(),
            "constraints": self.constraints.to_dict(),
            "selected_metrics": self.selected_metrics.to_dict(),
            "strategy": self.strategy,
            "evaluated_candidate_count": int(self.evaluated_candidate_count),
            "iterations": int(self.iterations),
            "budget_exhausted": bool(self.budget_exhausted),
            "converged": bool(self.converged),
        }


@dataclass(frozen=True)
class TemporalResultRow:
    """One matched temporal-method outcome, with scalar fields ready for Task 4."""

    trajectory_id: str
    split: str
    condition_id: str
    method: str
    injected_offset_ms: float
    jitter_ms: float
    dropout: float
    native_delay_ms: float
    estimated_absolute_offset_ms: float
    estimated_offset_ms: float
    applied_correction_ms: float
    committed: bool
    guard_reason: str
    peak_correlation: float
    peak_margin: float
    visual_mad: float
    state_mad: float
    offset_mae_ms: float
    qmae_rad: float
    qmae_reduction_rad: float
    recovery_success: bool
    interpolated_qpos: np.ndarray = field(repr=False, compare=False)
    joint_selection: JointSelection = field(default_factory=default_piper_joint_selection)

    def __post_init__(self) -> None:
        if not self.trajectory_id or not self.condition_id:
            raise ValueError("temporal result identifiers must be nonempty")
        if self.split not in {"development", "test"}:
            raise ValueError("temporal result split must be development or test")
        if self.method not in TEMPORAL_METHODS:
            raise ValueError(f"unknown temporal method: {self.method}")
        if self.committed and (
            self.method != "kinesync_guarded" or self.applied_correction_ms == 0.0
        ):
            raise ValueError("only nonzero guarded corrections may be committed")
        qpos = np.asarray(self.interpolated_qpos)
        if qpos.ndim != 2 or not np.isfinite(qpos).all():
            raise ValueError("interpolated_qpos must contain finite state samples")
        _require_radian_selection(self.joint_selection, field="temporal result selection")
        self.joint_selection.validate_width(qpos.shape[1], field="interpolated qpos")

    def to_dict(self) -> dict[str, str | float | bool | None]:
        return {
            "trajectory_id": self.trajectory_id,
            "split": self.split,
            "condition_id": self.condition_id,
            "method": self.method,
            "injected_offset_ms": float(self.injected_offset_ms),
            "jitter_ms": float(self.jitter_ms),
            "dropout": float(self.dropout),
            "native_delay_ms": float(self.native_delay_ms),
            "estimated_absolute_offset_ms": _finite_or_none(
                self.estimated_absolute_offset_ms
            ),
            "estimated_offset_ms": _finite_or_none(self.estimated_offset_ms),
            "applied_correction_ms": float(self.applied_correction_ms),
            "committed": bool(self.committed),
            "guard_reason": self.guard_reason,
            "peak_correlation": _finite_or_none(self.peak_correlation),
            "peak_margin": _finite_or_none(self.peak_margin),
            "visual_mad": _finite_or_none(self.visual_mad),
            "state_mad": _finite_or_none(self.state_mad),
            "offset_mae_ms": float(self.offset_mae_ms),
            "qmae_rad": float(self.qmae_rad),
            "qmae_reduction_rad": float(self.qmae_reduction_rad),
            "recovery_success": bool(self.recovery_success),
            "embodiment": self.joint_selection.embodiment_name,
            "joint_groups": ",".join(self.joint_selection.group_names),
            "joint_names": ",".join(self.joint_selection.joint_names),
            "joint_unit": self.joint_selection.unit,
        }


@dataclass(frozen=True)
class TemporalMethodSummary:
    """Trajectory-macro metrics for one method and optional subgroup."""

    method: str
    condition_count: int
    trajectory_count: int
    offset_mae_ms: float
    qmae_rad: float
    qmae_reduction_rad: float
    recovery_success_rate: float
    commit_precision: float | None
    controlled_coverage: float | None
    zero_stability: float | None

    def to_dict(self) -> dict[str, str | int | float | None]:
        return {
            "method": self.method,
            "condition_count": int(self.condition_count),
            "trajectory_count": int(self.trajectory_count),
            "offset_mae_ms": float(self.offset_mae_ms),
            "qmae_rad": float(self.qmae_rad),
            "qmae_reduction_rad": float(self.qmae_reduction_rad),
            "recovery_success_rate": float(self.recovery_success_rate),
            "commit_precision": self.commit_precision,
            "controlled_coverage": self.controlled_coverage,
            "zero_stability": self.zero_stability,
        }


@dataclass(frozen=True)
class TemporalSubgroupSummary:
    """One method's trajectory-macro metrics for a jitter or dropout subgroup."""

    subgroup: str
    value: float
    metrics: TemporalMethodSummary

    def to_dict(self) -> dict[str, object]:
        return {
            "subgroup": self.subgroup,
            "value": float(self.value),
            "metrics": self.metrics.to_dict(),
        }


@dataclass(frozen=True)
class TemporalSummary:
    """Serializable aggregate output for all matched temporal result rows."""

    methods: tuple[TemporalMethodSummary, ...]
    jitter_groups: tuple[TemporalSubgroupSummary, ...]
    dropout_groups: tuple[TemporalSubgroupSummary, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "methods": [metric.to_dict() for metric in self.methods],
            "jitter_groups": [group.to_dict() for group in self.jitter_groups],
            "dropout_groups": [group.to_dict() for group in self.dropout_groups],
        }


@dataclass(frozen=True)
class _CalibrationEvidence:
    trial: TemporalTrial
    estimate: OffsetEstimate
    correction_ms: float
    raw_qmae_rad: float
    candidate_qmae_rad: float


def extract_motion_features(
    trajectory: TakePensTrajectory,
    segment: ContinuousSegment,
    image_size: tuple[int, int] = (60, 80),
    *,
    joint_selection: JointSelection | None = None,
) -> MotionFeatures:
    """Derive normalized dual-camera and selected-radian-joint motion for a segment."""
    height, width = image_size
    if height < 1 or width < 1:
        raise ValueError("image_size dimensions must be positive")
    if segment.stop - segment.start < 2:
        raise ValueError("Motion features require at least two frames")

    indices = np.arange(segment.start, segment.stop, dtype=np.int64)
    head_frames = trajectory.read_rgb("head", indices)
    wrist_frames = trajectory.read_rgb("wrist", indices)
    head_motion = _frame_difference_motion(head_frames, height, width)
    wrist_motion = _frame_difference_motion(wrist_frames, height, width)

    head_timestamps = trajectory.timestamps("head")[segment.start : segment.stop]
    wrist_timestamps = trajectory.timestamps("wrist")[segment.start : segment.stop]
    slave_timestamps = trajectory.timestamps("slave")[segment.start : segment.stop]
    selection = joint_selection or default_piper_joint_selection()
    _require_radian_selection(selection, field="motion feature selection")
    all_slave_joints = trajectory.slave_joints()[segment.start : segment.stop]
    if all_slave_joints.ndim != 2:
        raise ValueError("slave joints must be a two-dimensional array")
    selection.validate_width(all_slave_joints.shape[1], field="slave joints")
    slave_joints = all_slave_joints[:, selection.indices]
    midpoint_ns = _interval_midpoints(head_timestamps)
    wrist_midpoint_ns = _interval_midpoints(wrist_timestamps)
    head_motion = _robust_normalize(head_motion)
    wrist_motion = _robust_normalize(wrist_motion)
    aligned_wrist_motion = interpolate_qpos(
        midpoint_ns, wrist_midpoint_ns, wrist_motion[:, None], mode="linear"
    )[:, 0]
    head_aligned_joints = interpolate_qpos(
        head_timestamps, slave_timestamps, slave_joints, mode="linear"
    )
    state_motion = np.linalg.norm(np.diff(head_aligned_joints, axis=0), axis=1)
    visual_motion = (head_motion + aligned_wrist_motion) / 2.0

    return MotionFeatures(
        camera_midpoint_ns=midpoint_ns.astype(np.int64, copy=False),
        visual_motion=visual_motion,
        state_motion=_robust_normalize(state_motion),
        joint_selection=selection,
    )


def _frame_difference_motion(frames: np.ndarray, height: int, width: int) -> np.ndarray:
    grayscale = np.stack(
        [
            cv2.cvtColor(cv2.resize(frame, (width, height)), cv2.COLOR_RGB2GRAY)
            for frame in frames
        ]
    ).astype(np.float64)
    return np.mean(np.abs(np.diff(grayscale, axis=0)), axis=(1, 2))


def _interval_midpoints(timestamps: np.ndarray) -> np.ndarray:
    return timestamps[:-1] + (timestamps[1:] - timestamps[:-1]) // 2


def _robust_normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    if not np.isfinite(mad) or mad == 0.0:
        return np.zeros(values.shape, dtype=np.float64)
    normalized = (values - median) / mad
    return np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)


def corrupt_state_stream(
    timestamp_ns: np.ndarray,
    qpos: np.ndarray,
    offset_ms: float,
    jitter_ms: float,
    dropout: float,
    seed: int,
) -> CorruptedState:
    """Inject deterministic timestamp offset, jitter, and interior-sample dropout."""
    timestamps, positions = _validate_state_stream(timestamp_ns, qpos)
    if jitter_ms < 0:
        raise ValueError("jitter_ms must be nonnegative")
    if not 0 <= dropout <= 1:
        raise ValueError("dropout must be between zero and one")

    generator = np.random.default_rng(seed)
    jitter_ns = np.rint(
        generator.normal(0.0, jitter_ms * 1_000_000.0, size=len(timestamps))
    ).astype(np.int64)
    corrupted_timestamps = timestamps + int(round(offset_ms * 1_000_000.0)) + jitter_ns
    retained = generator.random(len(timestamps)) >= dropout
    retained[0] = True
    retained[-1] = True
    retained_indices = np.flatnonzero(retained)
    order = np.argsort(corrupted_timestamps[retained_indices], kind="stable")
    retained_indices = retained_indices[order]
    ordered_timestamps = corrupted_timestamps[retained_indices].copy()
    for index in range(1, len(ordered_timestamps)):
        if ordered_timestamps[index] <= ordered_timestamps[index - 1]:
            ordered_timestamps[index] = ordered_timestamps[index - 1] + 1

    return CorruptedState(
        timestamp_ns=ordered_timestamps,
        qpos=positions[retained_indices].copy(),
        retained_indices=retained_indices,
    )


def search_motion_offset(
    features: MotionFeatures,
    corrupted_state: CorruptedState,
    *,
    candidate_offsets_ms: Sequence[float] = DEFAULT_OFFSET_CANDIDATES_MS,
    strict_overlap: bool = False,
    joint_selection: JointSelection | None = None,
) -> OffsetEstimate:
    """Recover a timestamp correction by aligning state motion to visual midpoints.

    ``strict_overlap`` is the RT10-P non-clamping mode. It scores a candidate
    only when at least eight visual midpoints lie inside its corrected observed
    state-motion domain. The default preserves RT9 endpoint-interpolation
    behavior.
    """

    midpoint_ns = _validate_motion_features(features)
    selection = _resolve_feature_selection(features, joint_selection)
    state_midpoint_ns, state_motion = _state_motion_stream(corrupted_state, selection)
    candidates = _normalize_candidate_offsets(candidate_offsets_ms)
    visual_motion = np.asarray(features.visual_motion, dtype=np.float64)
    visual_mad = _median_absolute_deviation(visual_motion)

    scores: list[OffsetScore] = []
    sampled_by_offset: dict[float, np.ndarray] = {}
    for offset_ms in candidates:
        corrected_midpoints = state_midpoint_ns - int(round(offset_ms * 1_000_000.0))
        if strict_overlap:
            overlap = (midpoint_ns >= corrected_midpoints[0]) & (
                midpoint_ns <= corrected_midpoints[-1]
            )
            if np.count_nonzero(overlap) < 8:
                sampled_by_offset[offset_ms] = np.empty(0, dtype=np.float64)
                scores.append(OffsetScore(offset_ms, None))
                continue
            sampled = np.interp(
                midpoint_ns[overlap], corrected_midpoints, state_motion
            )
            score = _finite_pearson(visual_motion[overlap], sampled)
        else:
            sampled = np.interp(midpoint_ns, corrected_midpoints, state_motion)
            score = _finite_pearson(visual_motion, sampled)
        sampled_by_offset[offset_ms] = sampled
        scores.append(OffsetScore(offset_ms, score))

    finite_scores = [score for score in scores if score.correlation is not None]
    if not finite_scores:
        fallback_offset = min(candidates, key=lambda value: (abs(value), value))
        return OffsetEstimate(
            offset_ms=fallback_offset,
            peak_correlation=float("-inf"),
            peak_margin=float("-inf"),
            visual_mad=visual_mad,
            state_mad=_median_absolute_deviation(sampled_by_offset[fallback_offset]),
            score_profile=tuple(scores),
        )

    winner = max(
        finite_scores,
        key=lambda score: (
            score.correlation,
            -abs(score.offset_ms),
            -score.offset_ms,
        ),
    )
    grid_step_ms = _candidate_grid_step(candidates)
    competitors = [
        score.correlation
        for score in finite_scores
        if abs(score.offset_ms - winner.offset_ms) > grid_step_ms
    ]
    peak_margin = (
        winner.correlation - max(competitors) if competitors else 0.0
    )
    return OffsetEstimate(
        offset_ms=winner.offset_ms,
        peak_correlation=winner.correlation,
        peak_margin=float(peak_margin),
        visual_mad=visual_mad,
        state_mad=_median_absolute_deviation(sampled_by_offset[winner.offset_ms]),
        score_profile=tuple(scores),
    )


def calibrate_temporal_guard(
    development_trials: Iterable[TemporalTrial | Mapping[str, object]],
    constraints: TemporalGuardConstraints | Mapping[str, object],
) -> TemporalGuard:
    """Freeze one guard from development evidence without inspecting test rows."""

    return calibrate_temporal_guard_with_report(
        development_trials, constraints
    ).guard


def calibrate_temporal_guard_with_report(
    development_trials: Iterable[TemporalTrial | Mapping[str, object]],
    constraints: TemporalGuardConstraints | Mapping[str, object],
) -> GuardCalibrationReport:
    """Calibrate a guard with deterministic bounded coordinate event search."""

    normalized_constraints = _coerce_guard_constraints(constraints)
    trials = _development_trials_only(development_trials)
    native_estimates: list[float] = []
    estimated_trials: list[tuple[TemporalTrial, OffsetEstimate]] = []
    has_zero_trial = False
    has_controlled_trial = False
    for trial in trials:
        estimate = search_motion_offset(
            trial.features,
            trial.corrupted_state,
            candidate_offsets_ms=normalized_constraints.candidate_offsets_ms,
            joint_selection=trial.joint_selection,
        )
        estimated_trials.append((trial, estimate))
        if trial.injected_offset_ms == 0.0:
            has_zero_trial = True
            if np.isfinite(estimate.peak_correlation):
                native_estimates.append(estimate.offset_ms)
        else:
            has_controlled_trial = True
    if not has_zero_trial or not has_controlled_trial:
        raise ValueError("temporal guard calibration requires zero and controlled development trials")
    if not native_estimates:
        raise ValueError("temporal guard calibration requires an informative zero-offset trial")
    native_delay_ms = float(np.median(native_estimates))
    evidence = tuple(
        _calibration_evidence(trial, estimate, native_delay_ms)
        for trial, estimate in estimated_trials
    )
    return _calibrate_guard_by_coordinate_events(
        evidence,
        native_delay_ms=native_delay_ms,
        constraints=normalized_constraints,
    )


def _calibrate_guard_by_coordinate_events(
    evidence: Sequence[_CalibrationEvidence],
    *,
    native_delay_ms: float,
    constraints: TemporalGuardConstraints,
) -> GuardCalibrationReport:
    coordinates = _guard_event_coordinates(evidence)
    current_guard = TemporalGuard(
        native_delay_ms=native_delay_ms,
        min_peak_correlation=-1.0,
        min_peak_margin=0.0,
        min_visual_mad=0.0,
        min_state_mad=0.0,
        min_correction_ms=0.0,
    )
    evaluated: dict[TemporalGuard, GuardCalibrationMetrics] = {}
    iterations = 0
    budget_exhausted = False

    while not budget_exhausted:
        iterations += 1
        changed = False
        for field_name, values in coordinates:
            candidates: list[tuple[TemporalGuard, GuardCalibrationMetrics]] = []
            for value in values:
                candidate = replace(current_guard, **{field_name: value})
                metrics = evaluated.get(candidate)
                if metrics is None:
                    if len(evaluated) >= constraints.max_calibration_candidates:
                        budget_exhausted = True
                        break
                    metrics = _guard_calibration_metrics(evidence, candidate, constraints)
                    evaluated[candidate] = metrics
                candidates.append((candidate, metrics))
            if candidates:
                best_guard, _ = max(
                    candidates,
                    key=lambda candidate: _guard_coordinate_rank(
                        candidate, constraints
                    ),
                )
                if best_guard != current_guard:
                    current_guard = best_guard
                    changed = True
            if budget_exhausted:
                break
        if not changed:
            break

    feasible = [
        (guard, metrics)
        for guard, metrics in evaluated.items()
        if _guard_meets_calibration_constraints(metrics, constraints)
    ]
    if not feasible:
        raise ValueError("No temporal guard satisfies development calibration constraints")
    guard, selected_metrics = max(feasible, key=_guard_calibration_rank)
    return GuardCalibrationReport(
        guard=guard,
        constraints=constraints,
        selected_metrics=selected_metrics,
        strategy=GUARD_CALIBRATION_STRATEGY,
        evaluated_candidate_count=len(evaluated),
        iterations=iterations,
        budget_exhausted=budget_exhausted,
        converged=not budget_exhausted,
    )


def evaluate_temporal_method(
    trial: TemporalTrial,
    method: str,
    *,
    guard: TemporalGuard | None = None,
    estimate: OffsetEstimate | None = None,
    recovery_offset_tolerance_ms: float = 20.0,
    recovery_qmae_tolerance_rad: float = 0.01,
) -> TemporalResultRow:
    """Evaluate one of the five matched interpolation methods for one condition."""

    if method not in TEMPORAL_METHODS:
        raise ValueError(f"unknown temporal method: {method}")
    if (
        recovery_offset_tolerance_ms < 0
        or recovery_qmae_tolerance_rad < 0
        or not np.isfinite(recovery_offset_tolerance_ms)
        or not np.isfinite(recovery_qmae_tolerance_rad)
    ):
        raise ValueError("temporal recovery tolerances must be finite and nonnegative")
    if method == "kinesync_guarded" and guard is None:
        raise ValueError("kinesync_guarded evaluation requires a TemporalGuard")

    estimate = estimate or search_motion_offset(
        trial.features, trial.corrupted_state, joint_selection=trial.joint_selection
    )
    native_delay_ms = guard.native_delay_ms if guard is not None else 0.0
    estimated_offset_ms = float(estimate.offset_ms - native_delay_ms)
    raw_linear = _interpolate_trial_qpos(trial, correction_ms=0.0, mode="linear")
    raw_linear_qmae_rad = _selected_joint_qmae(
        raw_linear, trial.target_qpos, trial.joint_selection
    )
    committed = False

    if method == "raw_nearest":
        interpolated_qpos = _interpolate_trial_qpos(
            trial, correction_ms=0.0, mode="nearest"
        )
        applied_correction_ms = 0.0
        guard_reason = "raw_nearest"
    elif method == "raw_linear":
        interpolated_qpos = raw_linear
        applied_correction_ms = 0.0
        guard_reason = "raw_linear"
    elif method == "motion_unguarded":
        interpolated_qpos = _interpolate_trial_qpos(
            trial, correction_ms=estimated_offset_ms, mode="linear"
        )
        applied_correction_ms = estimated_offset_ms
        guard_reason = "unguarded"
    elif method == "kinesync_guarded":
        decision = decide_temporal_guard(estimate, guard)
        if decision.accepted:
            interpolated_qpos = _interpolate_trial_qpos(
                trial, correction_ms=decision.correction_ms, mode="linear"
            )
            applied_correction_ms = decision.correction_ms
            committed = True
            guard_reason = decision.reason
        else:
            interpolated_qpos = raw_linear
            applied_correction_ms = 0.0
            guard_reason = decision.reason
    else:
        interpolated_qpos = _interpolate_trial_qpos(
            trial, correction_ms=trial.injected_offset_ms, mode="linear"
        )
        applied_correction_ms = float(trial.injected_offset_ms)
        guard_reason = "oracle_offset"

    qmae_rad = _selected_joint_qmae(
        interpolated_qpos, trial.target_qpos, trial.joint_selection
    )
    offset_mae_ms = abs(applied_correction_ms - trial.injected_offset_ms)
    recovery_success = bool(
        offset_mae_ms <= recovery_offset_tolerance_ms
        and qmae_rad <= recovery_qmae_tolerance_rad
    )
    return TemporalResultRow(
        trajectory_id=trial.trajectory_id,
        split=trial.split,
        condition_id=trial.condition_id,
        method=method,
        injected_offset_ms=float(trial.injected_offset_ms),
        jitter_ms=float(trial.jitter_ms),
        dropout=float(trial.dropout),
        native_delay_ms=float(native_delay_ms),
        estimated_absolute_offset_ms=float(estimate.offset_ms),
        estimated_offset_ms=estimated_offset_ms,
        applied_correction_ms=float(applied_correction_ms),
        committed=committed,
        guard_reason=guard_reason,
        peak_correlation=float(estimate.peak_correlation),
        peak_margin=float(estimate.peak_margin),
        visual_mad=float(estimate.visual_mad),
        state_mad=float(estimate.state_mad),
        offset_mae_ms=float(offset_mae_ms),
        qmae_rad=qmae_rad,
        qmae_reduction_rad=float(raw_linear_qmae_rad - qmae_rad),
        recovery_success=recovery_success,
        interpolated_qpos=interpolated_qpos,
        joint_selection=trial.joint_selection,
    )


def summarize_temporal_results(
    rows: Iterable[TemporalResultRow],
) -> TemporalSummary:
    """Summarize matched rows with equal weight for every trajectory."""

    normalized_rows = tuple(rows)
    if not normalized_rows:
        raise ValueError("temporal results cannot be empty")
    if not all(isinstance(row, TemporalResultRow) for row in normalized_rows):
        raise TypeError("temporal results must use TemporalResultRow rows")
    methods = tuple(
        _summarize_method_rows(
            [row for row in normalized_rows if row.method == method]
        )
        for method in sorted({row.method for row in normalized_rows})
    )
    jitter_groups = _summarize_subgroups(normalized_rows, "jitter_ms")
    dropout_groups = _summarize_subgroups(normalized_rows, "dropout")
    return TemporalSummary(
        methods=methods,
        jitter_groups=jitter_groups,
        dropout_groups=dropout_groups,
    )


def decide_temporal_guard(
    estimate: OffsetEstimate, guard: TemporalGuard
) -> TemporalGuardDecision:
    """Apply the frozen RT9 guard without recalibration or side effects."""

    correction_ms = float(estimate.offset_ms - guard.native_delay_ms)
    reason = _guard_decision_reason(estimate, correction_ms, guard)
    return TemporalGuardDecision(
        accepted=reason == "accepted",
        correction_ms=correction_ms,
        reason=reason,
    )


def _guard_decision_reason(
    estimate: OffsetEstimate,
    correction_ms: float,
    guard: TemporalGuard,
) -> str:
    values = (
        estimate.peak_correlation,
        estimate.peak_margin,
        estimate.visual_mad,
        estimate.state_mad,
        correction_ms,
    )
    if not all(np.isfinite(value) for value in values):
        return "nonfinite"
    if estimate.peak_correlation < guard.min_peak_correlation:
        return "peak_correlation"
    if estimate.peak_margin < guard.min_peak_margin:
        return "peak_margin"
    if estimate.visual_mad < guard.min_visual_mad:
        return "visual_mad"
    if estimate.state_mad < guard.min_state_mad:
        return "state_mad"
    if abs(correction_ms) < guard.min_correction_ms or correction_ms == 0.0:
        return "minimum_correction"
    return "accepted"


def _summarize_method_rows(
    rows: Sequence[TemporalResultRow],
) -> TemporalMethodSummary:
    if not rows:
        raise ValueError("temporal method rows cannot be empty")
    methods = {row.method for row in rows}
    if len(methods) != 1:
        raise ValueError("temporal method summary requires one method")
    by_trajectory: dict[str, list[TemporalResultRow]] = {}
    for row in rows:
        by_trajectory.setdefault(row.trajectory_id, []).append(row)
    trajectory_rows = tuple(by_trajectory.values())
    return TemporalMethodSummary(
        method=rows[0].method,
        condition_count=len(rows),
        trajectory_count=len(trajectory_rows),
        offset_mae_ms=_trajectory_macro_mean(
            trajectory_rows, lambda row: row.offset_mae_ms
        ),
        qmae_rad=_trajectory_macro_mean(
            trajectory_rows, lambda row: row.qmae_rad
        ),
        qmae_reduction_rad=_trajectory_macro_mean(
            trajectory_rows, lambda row: row.qmae_reduction_rad
        ),
        recovery_success_rate=_trajectory_macro_mean(
            trajectory_rows, lambda row: float(row.recovery_success)
        ),
        commit_precision=_trajectory_macro_optional(
            trajectory_rows, _trajectory_commit_precision
        ),
        controlled_coverage=_trajectory_macro_optional(
            trajectory_rows, _trajectory_controlled_coverage
        ),
        zero_stability=_trajectory_macro_optional(
            trajectory_rows, _trajectory_zero_stability
        ),
    )


def _summarize_subgroups(
    rows: Sequence[TemporalResultRow], attribute: str
) -> tuple[TemporalSubgroupSummary, ...]:
    selectors = {
        "jitter_ms": lambda row: row.jitter_ms,
        "dropout": lambda row: row.dropout,
    }
    try:
        selector = selectors[attribute]
    except KeyError as error:
        raise ValueError(f"unsupported temporal subgroup: {attribute}") from error
    methods = tuple(sorted({row.method for row in rows}))
    values = tuple(sorted({float(selector(row)) for row in rows}))
    return tuple(
        TemporalSubgroupSummary(
            subgroup=attribute,
            value=value,
            metrics=_summarize_method_rows(
                [
                    row
                    for row in rows
                    if row.method == method and float(selector(row)) == value
                ]
            ),
        )
        for value in values
        for method in methods
    )


def _trajectory_macro_mean(
    trajectory_rows: Sequence[Sequence[TemporalResultRow]], selector
) -> float:
    return float(
        np.mean(
            [np.mean([selector(row) for row in rows]) for rows in trajectory_rows]
        )
    )


def _trajectory_macro_optional(
    trajectory_rows: Sequence[Sequence[TemporalResultRow]], selector
) -> float | None:
    values = [value for rows in trajectory_rows if (value := selector(rows)) is not None]
    return float(np.mean(values)) if values else None


def _trajectory_commit_precision(rows: Sequence[TemporalResultRow]) -> float | None:
    committed = [row for row in rows if row.committed]
    if not committed:
        return None
    return sum(row.qmae_reduction_rad > 1e-12 for row in committed) / len(committed)


def _trajectory_controlled_coverage(
    rows: Sequence[TemporalResultRow],
) -> float | None:
    controlled = [row for row in rows if row.injected_offset_ms != 0.0]
    if not controlled:
        return None
    return sum(row.committed for row in controlled) / len(controlled)


def _trajectory_zero_stability(rows: Sequence[TemporalResultRow]) -> float | None:
    zero_rows = [row for row in rows if row.injected_offset_ms == 0.0]
    if not zero_rows:
        return None
    return sum(not row.committed for row in zero_rows) / len(zero_rows)


def interpolate_qpos(
    query_ns: np.ndarray,
    state_ns: np.ndarray,
    qpos: np.ndarray,
    mode: str,
) -> np.ndarray:
    """Sample qpos at query timestamps with endpoint clamping."""
    query = np.asarray(query_ns)
    if query.ndim != 1 or not np.issubdtype(query.dtype, np.integer):
        raise ValueError("query_ns must be a one-dimensional integer array")
    timestamps, positions = _validate_state_stream(state_ns, qpos)
    if mode == "nearest":
        upper = np.searchsorted(timestamps, query, side="left")
        lower = np.clip(upper - 1, 0, len(timestamps) - 1)
        upper = np.clip(upper, 0, len(timestamps) - 1)
        choose_lower = np.abs(query - timestamps[lower]) <= np.abs(
            timestamps[upper] - query
        )
        return positions[np.where(choose_lower, lower, upper)].copy()
    if mode == "linear":
        if len(timestamps) == 1:
            return np.repeat(positions, len(query), axis=0)
        upper = np.clip(np.searchsorted(timestamps, query, side="right"), 1, len(timestamps) - 1)
        lower = upper - 1
        clipped_query = np.clip(query, timestamps[0], timestamps[-1])
        fraction = (clipped_query - timestamps[lower]) / (
            timestamps[upper] - timestamps[lower]
        )
        return positions[lower] + fraction[:, None] * (positions[upper] - positions[lower])
    raise ValueError(f"Unknown interpolation mode: {mode}")


def _development_trials_only(
    trials: Iterable[TemporalTrial | Mapping[str, object]],
) -> tuple[TemporalTrial, ...]:
    development: list[TemporalTrial] = []
    for value in trials:
        split = _trial_split(value)
        if split != "development":
            continue
        if not isinstance(value, TemporalTrial):
            raise TypeError("development trials must use TemporalTrial rows")
        development.append(value)
    if not development:
        raise ValueError("temporal guard calibration requires development trials")
    return tuple(
        sorted(
            development,
            key=lambda trial: (
                trial.trajectory_id,
                trial.condition_id,
                trial.injected_offset_ms,
                trial.jitter_ms,
                trial.dropout,
            ),
        )
    )


def _trial_split(value: TemporalTrial | Mapping[str, object]) -> str | None:
    if isinstance(value, Mapping):
        split = value.get("split")
    else:
        try:
            split = value.split
        except AttributeError:
            split = None
    return str(split) if split is not None else None


def _coerce_guard_constraints(
    constraints: TemporalGuardConstraints | Mapping[str, object],
) -> TemporalGuardConstraints:
    if isinstance(constraints, TemporalGuardConstraints):
        return constraints
    if not isinstance(constraints, Mapping):
        raise TypeError("constraints must be TemporalGuardConstraints or a mapping")
    values = dict(constraints)
    aliases = {
        "min_zero_stability": "minimum_zero_stability",
        "min_committed_precision": "minimum_benefit_precision",
        "minimum_committed_precision": "minimum_benefit_precision",
        "offset_tolerance_ms": "recovery_offset_tolerance_ms",
        "qmae_tolerance_rad": "recovery_qmae_tolerance_rad",
    }
    for source, target in aliases.items():
        if source in values and target not in values:
            values[target] = values.pop(source)
    if "candidate_offsets_ms" in values:
        values["candidate_offsets_ms"] = tuple(
            float(value) for value in values["candidate_offsets_ms"]
        )
    return TemporalGuardConstraints(**values)


def _calibration_evidence(
    trial: TemporalTrial,
    estimate: OffsetEstimate,
    native_delay_ms: float,
) -> _CalibrationEvidence:
    correction_ms = float(estimate.offset_ms - native_delay_ms)
    raw_qpos = _interpolate_trial_qpos(trial, correction_ms=0.0, mode="linear")
    candidate_qpos = _interpolate_trial_qpos(
        trial, correction_ms=correction_ms, mode="linear"
    )
    return _CalibrationEvidence(
        trial=trial,
        estimate=estimate,
        correction_ms=correction_ms,
        raw_qmae_rad=_selected_joint_qmae(
            raw_qpos, trial.target_qpos, trial.joint_selection
        ),
        candidate_qmae_rad=_selected_joint_qmae(
            candidate_qpos, trial.target_qpos, trial.joint_selection
        ),
    )


def _guard_event_coordinates(
    evidence: Sequence[_CalibrationEvidence],
) -> tuple[tuple[str, tuple[float, ...]], ...]:
    """Return every observed threshold in the fixed coordinate-search order."""

    return (
        (
            "min_correction_ms",
            tuple(sorted({0.0, *(abs(item.correction_ms) for item in evidence)})),
        ),
        (
            "min_peak_correlation",
            _event_threshold_values(
                evidence, lambda item: item.estimate.peak_correlation, lower=-1.0
            ),
        ),
        (
            "min_peak_margin",
            _event_threshold_values(
                evidence, lambda item: item.estimate.peak_margin, lower=0.0
            ),
        ),
        (
            "min_visual_mad",
            _event_threshold_values(
                evidence, lambda item: item.estimate.visual_mad, lower=0.0
            ),
        ),
        (
            "min_state_mad",
            _event_threshold_values(
                evidence, lambda item: item.estimate.state_mad, lower=0.0
            ),
        ),
    )


def _event_threshold_values(
    evidence: Sequence[_CalibrationEvidence],
    selector,
    *,
    lower: float,
) -> tuple[float, ...]:
    return tuple(
        sorted(
            {
                lower,
                *(
                    float(value)
                    for value in (selector(item) for item in evidence)
                    if np.isfinite(value) and value >= lower
                ),
            }
        )
    )


def _guard_calibration_metrics(
    evidence: Sequence[_CalibrationEvidence],
    guard: TemporalGuard,
    constraints: TemporalGuardConstraints,
) -> GuardCalibrationMetrics:
    controlled_count = 0
    committed_controlled_count = 0
    successful_controlled_count = 0
    successful_committed_count = 0
    beneficial_committed_count = 0
    controlled_qmae_total = 0.0
    zero_count = 0
    stable_zero_count = 0
    for item in evidence:
        committed = _guard_accepts_estimate(item.estimate, item.correction_ms, guard)
        if item.trial.injected_offset_ms == 0.0:
            zero_count += 1
            stable_zero_count += not committed
            continue
        controlled_count += 1
        recovered = _temporal_recovery_success(item, committed, constraints)
        successful_controlled_count += recovered
        controlled_qmae_total += (
            item.candidate_qmae_rad if committed else item.raw_qmae_rad
        )
        if committed:
            committed_controlled_count += 1
            successful_committed_count += recovered
            beneficial_committed_count += (
                item.candidate_qmae_rad + 1e-12 < item.raw_qmae_rad
            )

    return GuardCalibrationMetrics(
        controlled_trial_count=controlled_count,
        committed_controlled_count=committed_controlled_count,
        successful_controlled_count=successful_controlled_count,
        successful_committed_count=successful_committed_count,
        beneficial_committed_count=beneficial_committed_count,
        zero_trial_count=zero_count,
        stable_zero_count=stable_zero_count,
        controlled_recovery=successful_controlled_count / controlled_count,
        controlled_qmae_rad=controlled_qmae_total / controlled_count,
        benefit_precision=(
            beneficial_committed_count / committed_controlled_count
            if committed_controlled_count
            else 0.0
        ),
        controlled_coverage=committed_controlled_count / controlled_count,
        zero_stability=stable_zero_count / zero_count,
    )


def _guard_meets_calibration_constraints(
    metrics: GuardCalibrationMetrics,
    constraints: TemporalGuardConstraints,
) -> bool:
    return bool(
        metrics.zero_stability >= constraints.minimum_zero_stability
        and metrics.benefit_precision >= constraints.minimum_benefit_precision
        and metrics.controlled_coverage >= constraints.minimum_controlled_coverage
    )


def _guard_calibration_rank(
    candidate: tuple[TemporalGuard, GuardCalibrationMetrics],
) -> tuple[float, ...]:
    guard, metrics = candidate
    return (
        metrics.controlled_recovery,
        -metrics.controlled_qmae_rad,
        metrics.benefit_precision,
        metrics.controlled_coverage,
        guard.min_peak_correlation,
        guard.min_peak_margin,
        guard.min_visual_mad,
        guard.min_state_mad,
        guard.min_correction_ms,
    )


def _guard_coordinate_rank(
    candidate: tuple[TemporalGuard, GuardCalibrationMetrics],
    constraints: TemporalGuardConstraints,
) -> tuple[float, ...]:
    """Rank feasible guards first and infeasible guards by constraint progress."""

    _, metrics = candidate
    outcome_rank = _guard_calibration_rank(candidate)
    if _guard_meets_calibration_constraints(metrics, constraints):
        return (1.0, *outcome_rank)
    zero_satisfaction = _normalized_constraint_satisfaction(
        metrics.zero_stability, constraints.minimum_zero_stability
    )
    benefit_satisfaction = _normalized_constraint_satisfaction(
        metrics.benefit_precision, constraints.minimum_benefit_precision
    )
    coverage_satisfaction = _normalized_constraint_satisfaction(
        metrics.controlled_coverage, constraints.minimum_controlled_coverage
    )
    return (
        0.0,
        min(zero_satisfaction, benefit_satisfaction, coverage_satisfaction),
        zero_satisfaction + benefit_satisfaction + coverage_satisfaction,
        *outcome_rank,
    )


def _normalized_constraint_satisfaction(value: float, minimum: float) -> float:
    if minimum == 0.0:
        return 1.0
    return float(np.clip(value / minimum, 0.0, 1.0))


def _guard_accepts_estimate(
    estimate: OffsetEstimate,
    correction_ms: float,
    guard: TemporalGuard,
) -> bool:
    return _guard_decision_reason(estimate, correction_ms, guard) == "accepted"


def _temporal_recovery_success(
    evidence: _CalibrationEvidence,
    committed: bool,
    constraints: TemporalGuardConstraints,
) -> bool:
    applied_correction_ms = evidence.correction_ms if committed else 0.0
    qmae_rad = (
        evidence.candidate_qmae_rad if committed else evidence.raw_qmae_rad
    )
    return bool(
        abs(applied_correction_ms - evidence.trial.injected_offset_ms)
        <= constraints.recovery_offset_tolerance_ms
        and qmae_rad <= constraints.recovery_qmae_tolerance_rad
    )


def _interpolate_trial_qpos(
    trial: TemporalTrial, *, correction_ms: float, mode: str
) -> np.ndarray:
    correction_ns = int(round(correction_ms * 1_000_000.0))
    return interpolate_qpos(
        trial.features.camera_midpoint_ns,
        trial.corrupted_state.timestamp_ns - correction_ns,
        trial.corrupted_state.qpos,
        mode=mode,
    )


def _selected_joint_qmae(
    predicted_qpos: np.ndarray,
    target_qpos: np.ndarray,
    selection: JointSelection,
) -> float:
    _require_radian_selection(selection, field="temporal qMAE selection")
    return selected_joint_mae(predicted_qpos, target_qpos, selection)


def _validate_motion_features(features: MotionFeatures) -> np.ndarray:
    midpoint_ns = np.asarray(features.camera_midpoint_ns)
    visual_motion = np.asarray(features.visual_motion)
    if midpoint_ns.ndim != 1 or not np.issubdtype(midpoint_ns.dtype, np.integer):
        raise ValueError("camera midpoint timestamps must be a one-dimensional integer array")
    if len(midpoint_ns) < 2 or visual_motion.shape != midpoint_ns.shape:
        raise ValueError("motion features require at least two aligned visual samples")
    if np.any(np.diff(midpoint_ns) <= 0):
        raise ValueError("camera midpoint timestamps must be strictly increasing")
    if features.joint_selection is not None:
        _require_radian_selection(features.joint_selection, field="motion feature selection")
    return midpoint_ns.astype(np.int64, copy=False)


def _state_motion_stream(
    corrupted_state: CorruptedState, selection: JointSelection
) -> tuple[np.ndarray, np.ndarray]:
    timestamps, positions = _validate_state_stream(
        corrupted_state.timestamp_ns, corrupted_state.qpos
    )
    if len(timestamps) < 3:
        raise ValueError("motion offset search requires at least three state samples")
    _require_radian_selection(selection, field="state motion selection")
    selection.validate_width(positions.shape[1], field="state qpos")
    state_motion = np.linalg.norm(np.diff(positions[:, selection.indices], axis=0), axis=1)
    return _interval_midpoints(timestamps), state_motion


def _require_radian_selection(selection: JointSelection, *, field: str) -> None:
    if not isinstance(selection, JointSelection):
        raise TypeError(f"{field} must be a JointSelection")
    if selection.unit != "rad":
        raise ValueError(f"{field} must select radians joints")


def _resolve_feature_selection(
    features: MotionFeatures, requested: JointSelection | None
) -> JointSelection:
    metadata = features.joint_selection
    if metadata is None:
        selection = requested if requested is not None else default_piper_joint_selection()
    else:
        _require_radian_selection(metadata, field="motion feature selection")
        if requested is not None and requested != metadata:
            raise ValueError("motion feature selection does not match requested selection")
        selection = metadata
    _require_radian_selection(selection, field="motion offset selection")
    return selection


def _normalize_candidate_offsets(candidate_offsets_ms: Sequence[float]) -> tuple[float, ...]:
    candidates = tuple(float(value) for value in candidate_offsets_ms)
    if not candidates:
        raise ValueError("candidate_offsets_ms must be nonempty")
    if not all(np.isfinite(value) for value in candidates):
        raise ValueError("candidate_offsets_ms must be finite")
    if len(set(candidates)) != len(candidates):
        raise ValueError("candidate_offsets_ms must be unique")
    return tuple(sorted(candidates))


def _candidate_grid_step(candidates: Sequence[float]) -> float:
    if len(candidates) < 2:
        return 0.0
    return min(
        later - earlier for earlier, later in zip(candidates, candidates[1:])
    )


def _finite_pearson(left: np.ndarray, right: np.ndarray) -> float | None:
    finite = np.isfinite(left) & np.isfinite(right)
    if np.count_nonzero(finite) < 2:
        return None
    aligned_left = left[finite]
    aligned_right = right[finite]
    left_centered = aligned_left - np.mean(aligned_left)
    right_centered = aligned_right - np.mean(aligned_right)
    denominator = np.linalg.norm(left_centered) * np.linalg.norm(right_centered)
    if not np.isfinite(denominator) or denominator == 0.0:
        return None
    correlation = float(np.dot(left_centered, right_centered) / denominator)
    return float(np.clip(correlation, -1.0, 1.0)) if np.isfinite(correlation) else None


def _median_absolute_deviation(values: np.ndarray) -> float:
    finite_values = np.asarray(values, dtype=np.float64)
    finite_values = finite_values[np.isfinite(finite_values)]
    if len(finite_values) == 0:
        return float("nan")
    median = np.median(finite_values)
    return float(np.median(np.abs(finite_values - median)))


def _finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _validate_state_stream(
    timestamp_ns: np.ndarray, qpos: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    timestamps = np.asarray(timestamp_ns)
    positions = np.asarray(qpos)
    if timestamps.ndim != 1 or not np.issubdtype(timestamps.dtype, np.integer):
        raise ValueError("state timestamps must be a one-dimensional integer array")
    if len(timestamps) == 0:
        raise ValueError("state timestamps cannot be empty")
    if positions.ndim != 2 or positions.shape[0] != len(timestamps):
        raise ValueError("qpos must have one row per state timestamp")
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError("state timestamps must be strictly increasing")
    return timestamps, positions


__all__ = [
    "CorruptedState",
    "DEFAULT_OFFSET_CANDIDATES_MS",
    "GUARD_CALIBRATION_STRATEGY",
    "GuardCalibrationMetrics",
    "GuardCalibrationReport",
    "MotionFeatures",
    "OffsetEstimate",
    "OffsetScore",
    "TEMPORAL_METHODS",
    "TemporalGuard",
    "TemporalGuardDecision",
    "TemporalGuardConstraints",
    "TemporalMethodSummary",
    "TemporalResultRow",
    "TemporalSubgroupSummary",
    "TemporalSummary",
    "TemporalTrial",
    "calibrate_temporal_guard",
    "calibrate_temporal_guard_with_report",
    "corrupt_state_stream",
    "decide_temporal_guard",
    "evaluate_temporal_method",
    "extract_motion_features",
    "interpolate_qpos",
    "search_motion_offset",
    "summarize_temporal_results",
]
