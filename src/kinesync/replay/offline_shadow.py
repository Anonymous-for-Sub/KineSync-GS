"""Guarded offline Real-to-GS synchronization over exact paired replays."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import cv2
import numpy as np

from kinesync.data.paired_real_replay import PairedReplayFrame


@dataclass(frozen=True)
class GuardConstraints:
    minimum_commit_precision: float
    minimum_controlled_coverage: float
    minimum_zero_stability: float
    success_qmae_rad: float

    def __post_init__(self) -> None:
        for name in (
            "minimum_commit_precision",
            "minimum_controlled_coverage",
            "minimum_zero_stability",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.success_qmae_rad <= 0:
            raise ValueError("success_qmae_rad must be positive")


@dataclass(frozen=True)
class ReplayTrial:
    backend: str
    split: str
    episode_id: str
    source_pair_id: str
    measured_pair_id: str
    selected_pair_id: str
    lag: int
    true_index: int
    measured_index: int
    selected_index: int
    visual_gain: float
    score_separation: float
    delayed_qmae_rad: float
    selected_qmae_rad: float
    correction_qmae_rad: float


@dataclass(frozen=True)
class ReplayMetrics:
    trial_count: int
    controlled_trial_count: int
    zero_trial_count: int
    delayed_qmae_rad: float
    qmae_rad: float
    recovery_success: float
    commit_precision: float
    controlled_coverage: float
    zero_stability: float
    zero_exact_no_commit: float
    commit_count: int

    def as_dict(self) -> dict[str, int | float]:
        return asdict(self)


def _resized_float(image: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    height, width = image_size
    if height <= 0 or width <= 0:
        raise ValueError("image_size must be positive")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected HWC RGB image, got {image.shape}")
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32) / 255.0


def _arm_qmae(first: np.ndarray, second: np.ndarray) -> float:
    if first.shape != (8,) or second.shape != (8,):
        raise ValueError("Replay qpos values must have shape (8,)")
    return float(np.mean(np.abs(first[:6] - second[:6])))


def build_replay_trials(
    episodes: Mapping[str, Sequence[PairedReplayFrame]],
    *,
    backend: str,
    lags: Sequence[int],
    window_radius: int,
    image_size: tuple[int, int],
) -> list[ReplayTrial]:
    """Build matched delay and zero-control trials from immutable paired frames."""

    if not episodes:
        raise ValueError("At least one replay episode is required")
    if window_radius < 0:
        raise ValueError("window_radius must be nonnegative")
    lag_values = tuple(int(lag) for lag in lags)
    if not lag_values or 0 not in lag_values:
        raise ValueError("lags must include a zero control")
    trials: list[ReplayTrial] = []
    for episode_id in sorted(episodes):
        frames = sorted(episodes[episode_id], key=lambda frame: frame.pair_index)
        if not frames:
            raise ValueError(f"Episode {episode_id} is empty")
        split = frames[0].split
        if any(frame.episode_id != episode_id or frame.split != split for frame in frames):
            raise ValueError(f"Episode identity mismatch for {episode_id}")
        real_images = [_resized_float(frame.real_rgb, image_size) for frame in frames]
        gaussian_images = [
            _resized_float(frame.gaussian_rgb, image_size) for frame in frames
        ]
        for true_index, frame in enumerate(frames):
            for lag in lag_values:
                measured_index = int(np.clip(true_index + lag, 0, len(frames) - 1))
                start = max(0, measured_index - window_radius)
                stop = min(len(frames), measured_index + window_radius + 1)
                candidate_indices = list(range(start, stop))
                scores = np.asarray(
                    [
                        np.mean(
                            np.abs(real_images[true_index] - gaussian_images[candidate])
                        )
                        for candidate in candidate_indices
                    ],
                    dtype=np.float64,
                )
                selected_position = int(np.argmin(scores))
                selected_index = candidate_indices[selected_position]
                measured_position = candidate_indices.index(measured_index)
                ordered_scores = np.sort(scores)
                measured_score = float(scores[measured_position])
                selected_score = float(scores[selected_position])
                visual_gain = (measured_score - selected_score) / max(
                    measured_score, 1e-12
                )
                score_separation = (
                    float(ordered_scores[1] - ordered_scores[0])
                    / max(measured_score, 1e-12)
                    if len(ordered_scores) > 1
                    else 0.0
                )
                true_qpos = frames[true_index].qpos
                measured_qpos = frames[measured_index].qpos
                selected_qpos = frames[selected_index].qpos
                trials.append(
                    ReplayTrial(
                        backend=backend,
                        split=split,
                        episode_id=episode_id,
                        source_pair_id=frame.source_pair_id,
                        measured_pair_id=frames[measured_index].source_pair_id,
                        selected_pair_id=frames[selected_index].source_pair_id,
                        lag=lag,
                        true_index=true_index,
                        measured_index=measured_index,
                        selected_index=selected_index,
                        visual_gain=float(visual_gain),
                        score_separation=score_separation,
                        delayed_qmae_rad=_arm_qmae(measured_qpos, true_qpos),
                        selected_qmae_rad=_arm_qmae(selected_qpos, true_qpos),
                        correction_qmae_rad=_arm_qmae(selected_qpos, measured_qpos),
                    )
                )
    return trials


def _committed(trial: ReplayTrial, threshold: float) -> bool:
    return trial.visual_gain >= threshold and trial.correction_qmae_rad > 1e-8


def evaluate_replay(
    trials: Sequence[ReplayTrial], *, threshold: float, success_qmae_rad: float
) -> ReplayMetrics:
    if not trials:
        raise ValueError("At least one replay trial is required")
    if threshold < 0 or success_qmae_rad <= 0:
        raise ValueError("threshold must be nonnegative and success_qmae_rad positive")
    controlled = [
        trial
        for trial in trials
        if trial.lag != 0 and trial.delayed_qmae_rad > 1e-8
    ]
    zero = [trial for trial in trials if trial.lag == 0]
    if not controlled or not zero:
        raise ValueError("Replay evaluation requires controlled and zero trials")
    committed = [trial for trial in controlled if _committed(trial, threshold)]
    committed_improvements = [
        trial
        for trial in committed
        if trial.selected_qmae_rad + 1e-8 < trial.delayed_qmae_rad
    ]
    final_errors = [
        trial.selected_qmae_rad
        if _committed(trial, threshold)
        else trial.delayed_qmae_rad
        for trial in controlled
    ]
    zero_stable = [
        (not _committed(trial, threshold))
        or trial.selected_qmae_rad <= success_qmae_rad
        for trial in zero
    ]
    return ReplayMetrics(
        trial_count=len(trials),
        controlled_trial_count=len(controlled),
        zero_trial_count=len(zero),
        delayed_qmae_rad=float(
            np.mean([trial.delayed_qmae_rad for trial in controlled])
        ),
        qmae_rad=float(np.mean(final_errors)),
        recovery_success=float(
            np.mean([error <= success_qmae_rad for error in final_errors])
        ),
        commit_precision=len(committed_improvements) / max(len(committed), 1),
        controlled_coverage=len(committed) / len(controlled),
        zero_stability=float(np.mean(zero_stable)),
        zero_exact_no_commit=float(
            np.mean([not _committed(trial, threshold) for trial in zero])
        ),
        commit_count=len(committed),
    )


def select_guard_threshold(
    trials: Sequence[ReplayTrial], *, constraints: GuardConstraints
) -> tuple[float, ReplayMetrics]:
    """Freeze an event-complete visual-gain threshold on calibration trials."""

    if not trials or any(trial.split != "calibration" for trial in trials):
        raise ValueError("Guard calibration accepts calibration split trials only")
    thresholds = sorted({0.0, *(max(0.0, trial.visual_gain) for trial in trials)})
    feasible: list[tuple[tuple[float, float, float, float], float, ReplayMetrics]] = []
    for threshold in thresholds:
        metrics = evaluate_replay(
            trials,
            threshold=threshold,
            success_qmae_rad=constraints.success_qmae_rad,
        )
        if (
            metrics.commit_precision >= constraints.minimum_commit_precision
            and metrics.controlled_coverage >= constraints.minimum_controlled_coverage
            and metrics.zero_stability >= constraints.minimum_zero_stability
        ):
            rank = (
                metrics.recovery_success,
                -metrics.qmae_rad,
                metrics.commit_precision,
                threshold,
            )
            feasible.append((rank, threshold, metrics))
    if not feasible:
        raise ValueError("No calibration threshold satisfies the guard constraints")
    _, threshold, metrics = max(feasible, key=lambda item: item[0])
    return threshold, metrics


def select_shared_guard_threshold(
    trials_by_backend: Mapping[str, Sequence[ReplayTrial]],
    *,
    constraints: GuardConstraints,
) -> tuple[float, dict[str, ReplayMetrics]]:
    """Select one normalized visual-gain threshold safe for every backend."""

    if not trials_by_backend:
        raise ValueError("At least one visual backend is required")
    for backend, trials in trials_by_backend.items():
        if not trials or any(trial.split != "calibration" for trial in trials):
            raise ValueError(
                f"Shared guard calibration requires calibration trials: {backend}"
            )
    thresholds = sorted(
        {
            0.0,
            *(
                max(0.0, trial.visual_gain)
                for trials in trials_by_backend.values()
                for trial in trials
            ),
        }
    )
    feasible = []
    for threshold in thresholds:
        metrics_by_backend = {
            backend: evaluate_replay(
                trials,
                threshold=threshold,
                success_qmae_rad=constraints.success_qmae_rad,
            )
            for backend, trials in trials_by_backend.items()
        }
        if not all(
            metrics.commit_precision >= constraints.minimum_commit_precision
            and metrics.controlled_coverage >= constraints.minimum_controlled_coverage
            and metrics.zero_stability >= constraints.minimum_zero_stability
            for metrics in metrics_by_backend.values()
        ):
            continue
        values = tuple(metrics_by_backend.values())
        rank = (
            min(metrics.recovery_success for metrics in values),
            -float(np.mean([metrics.qmae_rad for metrics in values])),
            min(metrics.commit_precision for metrics in values),
            min(metrics.zero_stability for metrics in values),
            threshold,
        )
        feasible.append((rank, threshold, metrics_by_backend))
    if not feasible:
        raise ValueError("No shared threshold satisfies every backend constraint")
    _, threshold, metrics = max(feasible, key=lambda item: item[0])
    return threshold, metrics


def cluster_bootstrap(
    trials: Sequence[ReplayTrial],
    *,
    threshold: float,
    success_qmae_rad: float,
    samples: int,
    seed: int,
) -> dict[str, object]:
    """Bootstrap gains by episode so correlated replay trials stay together."""

    if samples <= 0:
        raise ValueError("samples must be positive")
    by_episode: dict[str, list[ReplayTrial]] = {}
    for trial in trials:
        by_episode.setdefault(trial.episode_id, []).append(trial)
    episode_ids = sorted(by_episode)
    if not episode_ids:
        raise ValueError("At least one episode cluster is required")

    def gains(rows: Sequence[ReplayTrial]) -> tuple[float, float]:
        metrics = evaluate_replay(
            rows, threshold=threshold, success_qmae_rad=success_qmae_rad
        )
        controlled = [
            trial
            for trial in rows
            if trial.lag != 0 and trial.delayed_qmae_rad > 1e-8
        ]
        delayed_success = float(
            np.mean(
                [trial.delayed_qmae_rad <= success_qmae_rad for trial in controlled]
            )
        )
        return (
            metrics.delayed_qmae_rad - metrics.qmae_rad,
            metrics.recovery_success - delayed_success,
        )

    estimate_qmae, estimate_success = gains(trials)
    rng = np.random.default_rng(seed)
    sampled_qmae = []
    sampled_success = []
    for _ in range(samples):
        selected_ids = rng.choice(episode_ids, size=len(episode_ids), replace=True)
        sampled_rows = [
            trial for episode_id in selected_ids for trial in by_episode[str(episode_id)]
        ]
        qmae_gain, success_gain = gains(sampled_rows)
        sampled_qmae.append(qmae_gain)
        sampled_success.append(success_gain)

    def interval(estimate: float, values: Sequence[float]) -> dict[str, float]:
        low, high = np.percentile(np.asarray(values), [2.5, 97.5])
        return {
            "estimate": float(estimate),
            "ci_low": float(low),
            "ci_high": float(high),
        }

    return {
        "cluster_count": len(episode_ids),
        "samples": samples,
        "seed": seed,
        "qmae_reduction_rad": interval(estimate_qmae, sampled_qmae),
        "recovery_success_gain": interval(estimate_success, sampled_success),
    }


__all__ = [
    "GuardConstraints",
    "ReplayMetrics",
    "ReplayTrial",
    "build_replay_trials",
    "cluster_bootstrap",
    "evaluate_replay",
    "select_guard_threshold",
    "select_shared_guard_threshold",
]
