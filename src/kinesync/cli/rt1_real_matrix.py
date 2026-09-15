"""Execute, shard, and aggregate held-out RT1-O real-observation trials."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable, Mapping

import numpy as np
import yaml

from kinesync.cli.rt1_real_observation import execute_rt1o
from kinesync.config import load_config


REQUIRED_CHILD_FILES = (
    "config.yaml",
    "manifest.json",
    "metrics.json",
    "trace.csv",
    "images/real_observation_before_after.png",
    "images/real_observation_recovery_curves.png",
    "videos/real_observation_recovery.mp4",
)


def _slug_cameras(cameras: Iterable[str]) -> str:
    return "-".join(cameras)


def build_cartesian_trials(grid: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Expand a stable state/camera/seed/offset Cartesian product."""

    trials = []
    for state_id in grid["states"]:
        for cameras in grid["camera_modes"]:
            for seed in grid["seeds"]:
                for offset_case in grid["offset_cases"]:
                    offset_name = str(offset_case["name"])
                    run_id = (
                        f"rt1o_{state_id}_{_slug_cameras(cameras)}_"
                        f"{offset_name}_s{int(seed)}"
                    )
                    trials.append(
                        {
                            "run_id": run_id,
                            "state_id": str(state_id),
                            "cameras": list(cameras),
                            "seed": int(seed),
                            "offset_name": offset_name,
                            "offsets_rad": {
                                str(name): float(value)
                                for name, value in offset_case["offsets_rad"].items()
                            },
                        }
                    )
    return trials


def _build_balanced_trials(specification: Mapping[str, Any]) -> list[dict[str, Any]]:
    trials = []
    base_seed = int(specification.get("seed", 41))
    camera_modes = specification["camera_modes"]
    for state_case in specification["state_offset_cases"]:
        for cameras in camera_modes:
            offset_name = str(state_case["offset_name"])
            state_id = str(state_case["state_id"])
            trials.append(
                {
                    "run_id": (
                        f"rt1o_{state_id}_{_slug_cameras(cameras)}_"
                        f"{offset_name}_s{base_seed}"
                    ),
                    "state_id": state_id,
                    "cameras": list(cameras),
                    "seed": base_seed,
                    "offset_name": offset_name,
                    "offsets_rad": {
                        str(name): float(value)
                        for name, value in state_case["offsets_rad"].items()
                    },
                }
            )
    for case in specification.get("replicate_cases", []):
        normalized = dict(case)
        normalized["cameras"] = list(case["cameras"])
        normalized["seed"] = int(case["seed"])
        normalized["offsets_rad"] = {
            str(name): float(value) for name, value in case["offsets_rad"].items()
        }
        normalized.setdefault(
            "run_id",
            f"rt1o_{case['state_id']}_{_slug_cameras(case['cameras'])}_"
            f"{case['offset_name']}_s{int(case['seed'])}",
        )
        trials.append(normalized)
    run_ids = [case["run_id"] for case in trials]
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("RT1-O matrix contains duplicate run IDs")
    return trials


def _validate_complete_run(run: Path) -> None:
    missing = [relative for relative in REQUIRED_CHILD_FILES if not (run / relative).is_file()]
    if missing:
        raise ValueError(f"Incomplete RT1-O run {run}: missing {missing}")


def _bootstrap_interval(
    values: list[float], *, seed: int, samples: int
) -> list[float]:
    if not values:
        raise ValueError("Bootstrap values cannot be empty")
    if samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    array = np.asarray(values, dtype=np.float64)
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(array), size=(samples, len(array)))
    estimates = array[indices].mean(axis=1)
    lower, upper = np.quantile(estimates, [0.025, 0.975])
    return [float(lower), float(upper)]


def _group_summary(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row[key]), []).append(row)
    return {
        name: {
            "run_count": len(group),
            "success_rate": mean(float(row["recovery_success"]) for row in group),
            "state_error_reduction_mean": mean(
                float(row["state_error_reduction"]) for row in group
            ),
            "mask_iou_gain_mean": mean(float(row["mask_iou_gain"]) for row in group),
            "boundary_f1_gain_mean": mean(
                float(row["boundary_f1_gain"]) for row in group
            ),
        }
        for name, group in sorted(grouped.items())
    }


def aggregate_rt1o_runs(
    run_paths: Iterable[str | Path],
    output_dir: str | Path,
    *,
    bootstrap_seed: int = 2027,
    bootstrap_samples: int = 2_000,
) -> dict[str, Any]:
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for raw_path in run_paths:
        run = Path(raw_path).expanduser().resolve()
        _validate_complete_run(run)
        metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
        offsets = {
            str(name): float(value)
            for name, value in metrics["injected_measurement_offsets_rad"].items()
        }
        magnitude_deg = math.degrees(max((abs(value) for value in offsets.values()), default=0.0))
        rows.append(
            {
                "run_id": run.name,
                "state_id": metrics["state_id"],
                "camera_mode": "+".join(metrics["camera_names"]),
                "joint_group": "+".join(metrics["selected_joints"]),
                "magnitude_deg": round(magnitude_deg, 4),
                "seed": int(metrics.get("seed", 0)),
                "trial_type": metrics.get("trial_type", "controlled_offset"),
                "natural_state_correction_rad": metrics.get(
                    "natural_state_correction_rad"
                ),
                "offset_mae_rad": float(metrics["offset_mae_rad"]),
                "state_error_reduction": float(metrics["state_error_reduction"]),
                "mask_iou_before": float(metrics["mean_mask_iou_before"]),
                "mask_iou_after": float(metrics["mean_mask_iou_after"]),
                "mask_iou_gain": float(
                    metrics["mean_mask_iou_after"] - metrics["mean_mask_iou_before"]
                ),
                "boundary_f1_before": float(metrics["mean_boundary_f1_before"]),
                "boundary_f1_after": float(metrics["mean_boundary_f1_after"]),
                "boundary_f1_gain": float(
                    metrics["mean_boundary_f1_after"]
                    - metrics["mean_boundary_f1_before"]
                ),
                "recovery_success": bool(metrics["recovery_success"]),
            }
        )
    if not rows:
        raise ValueError("At least one complete RT1-O run is required")

    fieldnames = list(rows[0])
    with (output / "results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    success_values = [float(row["recovery_success"]) for row in rows]
    controlled_rows = [row for row in rows if row["trial_type"] == "controlled_offset"]
    zero_rows = [row for row in rows if row["trial_type"] == "zero_injection_control"]
    if not controlled_rows:
        raise ValueError("RT1-O aggregate requires at least one controlled-offset run")
    reductions = [float(row["state_error_reduction"]) for row in controlled_rows]
    controlled_success = [float(row["recovery_success"]) for row in controlled_rows]
    zero_corrections = [
        float(row["natural_state_correction_rad"])
        for row in zero_rows
        if row["natural_state_correction_rad"] is not None
    ]
    summary: dict[str, Any] = {
        "run_count": len(rows),
        "success_count": int(sum(success_values)),
        "success_rate": mean(success_values),
        "success_rate_ci95": _bootstrap_interval(
            success_values, seed=bootstrap_seed, samples=bootstrap_samples
        ),
        "state_error_reduction_mean": mean(reductions),
        "state_error_reduction_median": median(reductions),
        "state_error_reduction_mean_ci95": _bootstrap_interval(
            reductions, seed=bootstrap_seed + 1, samples=bootstrap_samples
        ),
        "mask_iou_gain_mean": mean(float(row["mask_iou_gain"]) for row in rows),
        "boundary_f1_gain_mean": mean(
            float(row["boundary_f1_gain"]) for row in rows
        ),
        "offset_mae_rad_mean": mean(float(row["offset_mae_rad"]) for row in rows),
        "controlled_run_count": len(controlled_rows),
        "controlled_success_rate": mean(controlled_success),
        "controlled_success_rate_ci95": _bootstrap_interval(
            controlled_success, seed=bootstrap_seed + 2, samples=bootstrap_samples
        ),
        "zero_control_count": len(zero_rows),
        "zero_control_success_rate": (
            mean(float(row["recovery_success"]) for row in zero_rows)
            if zero_rows
            else None
        ),
        "zero_control_correction_rad_median": (
            median(zero_corrections) if zero_corrections else None
        ),
        "by_camera_mode": _group_summary(rows, "camera_mode"),
        "by_joint_group": _group_summary(rows, "joint_group"),
        "by_magnitude_deg": _group_summary(rows, "magnitude_deg"),
        "bootstrap_seed": bootstrap_seed,
        "bootstrap_samples": bootstrap_samples,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# KineSync-GS RT1-O Held-Out Results",
        "",
        f"- Complete runs: {summary['run_count']}",
        f"- Recovery success: {summary['success_count']}/{summary['run_count']} "
        f"({summary['success_rate']:.1%}, bootstrap 95% CI "
        f"[{summary['success_rate_ci95'][0]:.1%}, {summary['success_rate_ci95'][1]:.1%}])",
        f"- Median state-error reduction: {summary['state_error_reduction_median']:.1%}",
        f"- Controlled-offset success: {summary['controlled_success_rate']:.1%}",
        f"- Zero-injection controls: {summary['zero_control_count']}",
        f"- Mean mask-IoU gain: {summary['mask_iou_gain_mean']:+.4f}",
        f"- Mean boundary-F1 gain: {summary['boundary_f1_gain_mean']:+.4f}",
        "",
        "| Run | State | Cameras | Joints | Offset (deg) | Error reduction | IoU gain | Boundary gain | Success |",
        "|---|---|---|---|---:|---:|---:|---:|:---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['run_id']} | {row['state_id']} | {row['camera_mode']} | "
            f"{row['joint_group']} | {float(row['magnitude_deg']):.2f} | "
            f"{float(row['state_error_reduction']):.1%} | "
            f"{float(row['mask_iou_gain']):+.4f} | "
            f"{float(row['boundary_f1_gain']):+.4f} | "
            f"{'yes' if row['recovery_success'] else 'no'} |"
        )
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def _matrix_trials(matrix: Mapping[str, Any]) -> list[dict[str, Any]]:
    if "cases" in matrix:
        return [dict(case) for case in matrix["cases"]]
    if "grid" in matrix:
        return build_cartesian_trials(matrix["grid"])
    if "balanced" in matrix:
        return _build_balanced_trials(matrix["balanced"])
    raise ValueError("RT1-O matrix requires cases, grid, or balanced specification")


def execute_matrix(
    path: str | Path,
    *,
    pilot: int | None = None,
    shard_index: int = 0,
    num_shards: int = 1,
    aggregate_only: bool = False,
) -> dict[str, Any]:
    matrix_path = Path(path).expanduser().resolve()
    matrix = load_config(matrix_path)
    if num_shards <= 0 or shard_index < 0 or shard_index >= num_shards:
        raise ValueError("Invalid matrix shard selection")
    base_path = Path(matrix["base_config"])
    if not base_path.is_absolute():
        base_path = (matrix_path.parent / base_path).resolve()
    base = load_config(base_path)
    all_trials = _matrix_trials(matrix)
    if pilot is not None:
        if pilot <= 0:
            raise ValueError("pilot must be positive")
        all_trials = all_trials[:pilot]
    trials = all_trials[shard_index::num_shards]
    runs_root = Path(matrix["runs_root"]).expanduser().resolve()
    run_paths = []
    for case in trials:
        run_path = runs_root / str(case["run_id"])
        if run_path.exists():
            _validate_complete_run(run_path)
            run_paths.append(run_path)
            continue
        if aggregate_only:
            raise ValueError(f"Missing RT1-O run for aggregate-only mode: {run_path}")
        config = copy.deepcopy(base)
        config["runs_root"] = str(runs_root)
        config["seed"] = int(case["seed"])
        config["device"] = str(matrix.get("device", config.get("device", "cuda")))
        config["data"]["state_id"] = str(case["state_id"])
        config["data"]["cameras"] = list(case["cameras"])
        config["data"]["anchors_per_link"] = case.get(
            "anchors_per_link", matrix.get("anchors_per_link", config["data"]["anchors_per_link"])
        )
        config["target"]["injected_joint_offsets_rad"] = dict(case["offsets_rad"])
        config["optimizer"]["steps"] = int(case.get("steps", matrix.get("steps", config["optimizer"]["steps"])))
        config["visualization"]["video_frames"] = int(
            case.get("video_frames", matrix.get("video_frames", config["visualization"]["video_frames"]))
        )
        run_paths.append(execute_rt1o(config, run_id=str(case["run_id"])))

    summary_root = Path(matrix["summary_dir"]).expanduser().resolve()
    suffix = "pilot" if pilot is not None else "full"
    if num_shards > 1:
        suffix += f"_shard-{shard_index}-of-{num_shards}"
    output = summary_root / suffix
    summary = aggregate_rt1o_runs(
        run_paths,
        output,
        bootstrap_seed=int(matrix.get("bootstrap_seed", 2027)),
        bootstrap_samples=int(matrix.get("bootstrap_samples", 2_000)),
    )
    (output / "matrix_config.yaml").write_text(
        yaml.safe_dump(matrix, sort_keys=False), encoding="utf-8"
    )
    (output / "resolved_trials.json").write_text(
        json.dumps(trials, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--pilot", type=int)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--aggregate-only", action="store_true")
    arguments = parser.parse_args()
    summary = execute_matrix(
        arguments.config,
        pilot=arguments.pilot,
        shard_index=arguments.shard_index,
        num_shards=arguments.num_shards,
        aggregate_only=arguments.aggregate_only,
    )
    print(yaml.safe_dump(summary, sort_keys=True))


if __name__ == "__main__":
    main()
