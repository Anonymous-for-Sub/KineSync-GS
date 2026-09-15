"""Run guarded KineSync synchronization on paired offline real/GS trajectories."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from kinesync.config import load_config
from kinesync.data.paired_real_replay import (
    PairedRealGaussianDataset,
    PairedReplayFrame,
)
from kinesync.replay.offline_shadow import (
    GuardConstraints,
    ReplayMetrics,
    ReplayTrial,
    build_replay_trials,
    cluster_bootstrap,
    evaluate_replay,
    select_shared_guard_threshold,
)
from kinesync.runs.artifacts import RunArtifacts
from kinesync.visualization.offline_shadow import (
    OfflineShadowVisualFrame,
    write_offline_shadow_image,
    write_offline_shadow_video,
)


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    return value


def _sequence(value: object, *, field: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{field} must be a sequence")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _episodes(
    dataset: PairedRealGaussianDataset, *, split: str, backend: str
) -> dict[str, list[PairedReplayFrame]]:
    return {
        episode_id: dataset.load_episode(
            split=split, episode_id=episode_id, backend=backend
        )
        for episode_id in dataset.episode_ids(split=split)
    }


def _delayed_method(trials: Sequence[ReplayTrial], success_threshold: float) -> dict[str, Any]:
    controlled = [
        trial
        for trial in trials
        if trial.lag != 0 and trial.delayed_qmae_rad > 1e-8
    ]
    zero = [trial for trial in trials if trial.lag == 0]
    return {
        "qmae_rad": float(np.mean([trial.delayed_qmae_rad for trial in controlled])),
        "recovery_success": float(
            np.mean(
                [trial.delayed_qmae_rad <= success_threshold for trial in controlled]
            )
        ),
        "commit_precision": None,
        "controlled_coverage": 0.0,
        "zero_stability": 1.0,
        "zero_exact_no_commit": 1.0,
        "controlled_trial_count": len(controlled),
        "zero_trial_count": len(zero),
    }


def _method_table(
    trials: Sequence[ReplayTrial], *, threshold: float, success_threshold: float
) -> dict[str, Any]:
    unguarded = evaluate_replay(
        trials, threshold=0.0, success_qmae_rad=success_threshold
    )
    guarded = evaluate_replay(
        trials, threshold=threshold, success_qmae_rad=success_threshold
    )
    controlled_count = guarded.controlled_trial_count
    zero_count = guarded.zero_trial_count
    return {
        "delayed_state": _delayed_method(trials, success_threshold),
        "unguarded_visual_retrieval": unguarded.as_dict(),
        "kinesync_guarded_retrieval": guarded.as_dict(),
        "paired_state_oracle": {
            "qmae_rad": 0.0,
            "recovery_success": 1.0,
            "commit_precision": 1.0,
            "controlled_coverage": 1.0,
            "zero_stability": 1.0,
            "zero_exact_no_commit": 1.0,
            "controlled_trial_count": controlled_count,
            "zero_trial_count": zero_count,
        },
    }


def _trial_row(trial: ReplayTrial, threshold: float) -> dict[str, Any]:
    committed = (
        trial.visual_gain >= threshold and trial.correction_qmae_rad > 1e-8
    )
    final_error = trial.selected_qmae_rad if committed else trial.delayed_qmae_rad
    return {
        **trial.__dict__,
        "guard_threshold": threshold,
        "guard_committed": committed,
        "guarded_qmae_rad": final_error,
        "guard_improved": bool(
            committed and trial.selected_qmae_rad + 1e-8 < trial.delayed_qmae_rad
        ),
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty result table")
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _visual_sequence(
    *,
    backends: Sequence[str],
    episodes_by_backend: Mapping[str, Mapping[str, Sequence[PairedReplayFrame]]],
    trials_by_backend: Mapping[str, Sequence[ReplayTrial]],
    thresholds: Mapping[str, float],
    lag: int,
    backend_labels: Mapping[str, str],
    hold_frames: int,
) -> list[list[OfflineShadowVisualFrame]]:
    common_episodes = set(episodes_by_backend[backends[0]])
    for backend in backends[1:]:
        common_episodes.intersection_update(episodes_by_backend[backend])
    if not common_episodes:
        raise ValueError("Visual backends have no shared test episode")
    episode_id = sorted(common_episodes, key=lambda value: int(value) if value.isdigit() else value)[0]
    trial_index = {
        backend: {
            trial.true_index: trial
            for trial in trials_by_backend[backend]
            if trial.episode_id == episode_id and trial.lag == lag
        }
        for backend in backends
    }
    common_indices = set(trial_index[backends[0]])
    for backend in backends[1:]:
        common_indices.intersection_update(trial_index[backend])
    sequence = []
    for true_index in sorted(common_indices):
        contexts = []
        for backend in backends:
            frames = episodes_by_backend[backend][episode_id]
            trial = trial_index[backend][true_index]
            committed = (
                trial.visual_gain >= thresholds[backend]
                and trial.correction_qmae_rad > 1e-8
            )
            final_index = trial.selected_index if committed else trial.measured_index
            contexts.append(
                OfflineShadowVisualFrame(
                    backend=backend_labels.get(backend, backend),
                    episode_id=episode_id,
                    source_pair_id=trial.source_pair_id,
                    lag=lag,
                    committed=committed,
                    delayed_qmae_rad=trial.delayed_qmae_rad,
                    corrected_qmae_rad=(
                        trial.selected_qmae_rad
                        if committed
                        else trial.delayed_qmae_rad
                    ),
                    real_rgb=frames[true_index].real_rgb,
                    delayed_gaussian_rgb=frames[trial.measured_index].gaussian_rgb,
                    corrected_gaussian_rgb=frames[final_index].gaussian_rgb,
                )
            )
        sequence.extend([contexts] * hold_frames)
    if not sequence:
        raise ValueError("No visual replay states were selected")
    return sequence


def execute_offline_shadow(
    config: Mapping[str, Any], *, run_id: str | None = None
) -> Path:
    source = _mapping(config.get("paired_replay"), field="paired_replay")
    configured_backends = _mapping(
        source.get("render_manifests"), field="paired_replay.render_manifests"
    )
    render_manifests = {
        str(backend): [str(path) for path in _sequence(paths, field=f"render_manifests.{backend}")]
        for backend, paths in configured_backends.items()
    }
    dataset = PairedRealGaussianDataset(
        selection_manifest=str(source["selection_manifest"]),
        render_manifests=render_manifests,
    )
    benchmark = _mapping(config.get("benchmark"), field="benchmark")
    calibration_split = str(benchmark.get("calibration_split", "calibration"))
    test_split = str(benchmark.get("test_split", "test"))
    if calibration_split == test_split:
        raise ValueError("Calibration and test splits must differ")
    lags = tuple(int(value) for value in _sequence(benchmark["lags"], field="benchmark.lags"))
    image_size_values = _sequence(benchmark["image_size"], field="benchmark.image_size")
    if len(image_size_values) != 2:
        raise ValueError("benchmark.image_size must be [height, width]")
    image_size = (int(image_size_values[0]), int(image_size_values[1]))
    constraints = GuardConstraints(
        minimum_commit_precision=float(benchmark["minimum_commit_precision"]),
        minimum_controlled_coverage=float(benchmark["minimum_controlled_coverage"]),
        minimum_zero_stability=float(benchmark["minimum_zero_stability"]),
        success_qmae_rad=float(benchmark["success_qmae_rad"]),
    )
    assets = {"selection_manifest": source["selection_manifest"]}
    for backend, paths in render_manifests.items():
        for index, path in enumerate(paths):
            assets[f"render_manifest_{backend}_{index}"] = path
    run = RunArtifacts.create(
        root=config.get("run_root", "runs"),
        experiment="rt8_paired_real_offline_shadow",
        config=config,
        assets=assets,
        target_provenance="real_paired_offline_replay",
        observation_backend="exact_pair_rgb_visual_gain",
        run_id=run_id,
    )

    calibration_trials_by_backend: dict[str, list[ReplayTrial]] = {}
    test_trials_by_backend: dict[str, list[ReplayTrial]] = {}
    test_episodes_by_backend: dict[str, dict[str, list[PairedReplayFrame]]] = {}
    calibration_episode_counts: dict[str, int] = {}
    metrics_by_backend: dict[str, Any] = {}
    for backend in dataset.backends:
        calibration_episodes = _episodes(
            dataset, split=calibration_split, backend=backend
        )
        test_episodes = _episodes(dataset, split=test_split, backend=backend)
        calibration_trials = build_replay_trials(
            calibration_episodes,
            backend=backend,
            lags=lags,
            window_radius=int(benchmark["window_radius"]),
            image_size=image_size,
        )
        test_trials = build_replay_trials(
            test_episodes,
            backend=backend,
            lags=lags,
            window_radius=int(benchmark["window_radius"]),
            image_size=image_size,
        )
        calibration_trials_by_backend[backend] = calibration_trials
        test_trials_by_backend[backend] = test_trials
        test_episodes_by_backend[backend] = test_episodes
        calibration_episode_counts[backend] = len(calibration_episodes)

    shared_threshold, calibration_metrics_by_backend = select_shared_guard_threshold(
        calibration_trials_by_backend, constraints=constraints
    )
    thresholds = {backend: shared_threshold for backend in dataset.backends}
    guard_payload: dict[str, Any] = {
        "schema": "kinesync.rt8_paired_replay_guard.v2",
        "source_split": calibration_split,
        "test_split_untouched_during_calibration": True,
        "threshold_scope": "shared_normalized_visual_gain_across_backends",
        "shared_visual_gain_threshold": shared_threshold,
        "constraints": constraints.__dict__,
        "backends": {
            backend: {
                "calibration_metrics": calibration_metrics.as_dict(),
                "calibration_trial_count": len(calibration_trials_by_backend[backend]),
            }
            for backend, calibration_metrics in calibration_metrics_by_backend.items()
        },
    }
    result_rows = []
    for backend in dataset.backends:
        threshold = thresholds[backend]
        test_trials = test_trials_by_backend[backend]
        test_episodes = test_episodes_by_backend[backend]
        calibration_metrics = calibration_metrics_by_backend[backend]
        methods = _method_table(
            test_trials,
            threshold=threshold,
            success_threshold=constraints.success_qmae_rad,
        )
        bootstrap = cluster_bootstrap(
            test_trials,
            threshold=threshold,
            success_qmae_rad=constraints.success_qmae_rad,
            samples=int(benchmark.get("bootstrap_samples", 2000)),
            seed=int(benchmark.get("bootstrap_seed", 20260824)),
        )
        metrics_by_backend[backend] = {
            "evidence_status": (
                "kinesync_unseen_historically_renderer_exposed_retrospective_test"
                if dataset.historical_renderer_validation_exposure[backend]
                else "kinesync_unseen_offline_test"
            ),
            "calibration_episode_count": calibration_episode_counts[backend],
            "test_episode_count": len(test_episodes),
            "guard_threshold": threshold,
            "guard_threshold_scope": "shared_across_backends",
            "calibration_metrics": calibration_metrics.as_dict(),
            "test_methods": methods,
            "episode_cluster_bootstrap": bootstrap,
        }
        result_rows.extend(_trial_row(trial, threshold) for trial in test_trials)

    guard_path = run.path / "guard.json"
    guard_path.write_text(
        json.dumps(guard_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_csv(run.path / "results.csv", result_rows)
    run.write_trace(result_rows)
    visual = _mapping(config.get("visualization", {}), field="visualization")
    backend_labels = {
        str(key): str(value)
        for key, value in _mapping(
            visual.get("backend_labels", {}), field="visualization.backend_labels"
        ).items()
    }
    visual_sequence = _visual_sequence(
        backends=list(dataset.backends),
        episodes_by_backend=test_episodes_by_backend,
        trials_by_backend=test_trials_by_backend,
        thresholds=thresholds,
        lag=int(visual.get("lag", -2)),
        backend_labels=backend_labels,
        hold_frames=int(visual.get("hold_frames", 2)),
    )
    image_path = run.path / "images" / "rt8_paired_real_shadow.png"
    video_path = run.path / "videos" / "rt8_paired_real_shadow.mp4"
    write_offline_shadow_image(image_path, visual_sequence[len(visual_sequence) // 2])
    write_offline_shadow_video(
        video_path, visual_sequence, fps=float(visual.get("fps", 12.0))
    )
    metrics = {
        "schema": "kinesync.rt8_paired_real_offline_shadow.v1",
        "evidence_scope": "offline_recorded_real_replay_not_live_robot_control",
        "selection_manifest": str(dataset.selection_manifest),
        "selection_manifest_sha256": _sha256(dataset.selection_manifest),
        "backend_count": len(dataset.backends),
        "backends": metrics_by_backend,
        "result_row_count": len(result_rows),
        "guard_sha256": _sha256(guard_path),
        "visuals": {
            "image": str(image_path),
            "image_sha256": _sha256(image_path),
            "video": str(video_path),
            "video_sha256": _sha256(video_path),
        },
        "formal_evidence": False,
    }
    run.write_metrics(metrics)
    return run.path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    config = load_config(args.config)
    print(execute_offline_shadow(config, run_id=args.run_id))


if __name__ == "__main__":
    main()
