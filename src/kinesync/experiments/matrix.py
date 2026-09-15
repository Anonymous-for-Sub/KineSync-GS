"""Aggregate RT0 runs into paper- and machine-readable summaries."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Iterable


RESULT_FIELDS = [
    "run_id",
    "frame_index",
    "camera_names",
    "selected_joints",
    "offset_mae_rad",
    "pixel_rmse_before",
    "pixel_rmse_after",
    "tip_error_before_mm",
    "tip_error_after_mm",
    "recovery_success",
]


def _stats(values: list[float]) -> tuple[float, float]:
    return mean(values), stdev(values) if len(values) > 1 else 0.0


def aggregate_rt0_runs(
    run_paths: Iterable[str | Path], output_dir: str | Path
) -> dict[str, Any]:
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for raw_path in run_paths:
        run_path = Path(raw_path).expanduser().resolve()
        metrics = json.loads((run_path / "metrics.json").read_text(encoding="utf-8"))
        row = {field: metrics.get(field) for field in RESULT_FIELDS}
        row["run_id"] = run_path.name
        row["camera_names"] = "+".join(metrics["camera_names"])
        row["selected_joints"] = "+".join(metrics["selected_joints"])
        rows.append(row)
    if not rows:
        raise ValueError("At least one RT0 run is required")

    with (output / "results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    offset_mean, offset_std = _stats([float(row["offset_mae_rad"]) for row in rows])
    pixel_mean, pixel_std = _stats([float(row["pixel_rmse_after"]) for row in rows])
    tip_mean, tip_std = _stats([float(row["tip_error_after_mm"]) for row in rows])
    summary = {
        "run_count": len(rows),
        "success_count": sum(bool(row["recovery_success"]) for row in rows),
        "success_rate": mean(bool(row["recovery_success"]) for row in rows),
        "offset_mae_rad_mean": offset_mean,
        "offset_mae_rad_std": offset_std,
        "pixel_rmse_after_mean": pixel_mean,
        "pixel_rmse_after_std": pixel_std,
        "tip_error_after_mm_mean": tip_mean,
        "tip_error_after_mm_std": tip_std,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# KineSync-GS RT0 Results",
        "",
        f"- Runs: {summary['run_count']}",
        f"- Recovery success: {summary['success_count']}/{summary['run_count']} "
        f"({summary['success_rate']:.1%})",
        f"- Offset MAE: {offset_mean:.6f} +/- {offset_std:.6f} rad",
        f"- Final pixel RMSE: {pixel_mean:.4f} +/- {pixel_std:.4f} px",
        f"- Final tip error: {tip_mean:.4f} +/- {tip_std:.4f} mm",
        "",
        "| Run | Frame | Cameras | Joints | Offset MAE (rad) | Pixel before/after (px) | Tip before/after (mm) | Success |",
        "|---|---:|---|---|---:|---:|---:|:---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['run_id']} | {row['frame_index']} | {row['camera_names']} | "
            f"{row['selected_joints']} | {float(row['offset_mae_rad']):.6f} | "
            f"{float(row['pixel_rmse_before']):.3f}/{float(row['pixel_rmse_after']):.3f} | "
            f"{float(row['tip_error_before_mm']):.3f}/{float(row['tip_error_after_mm']):.3f} | "
            f"{'yes' if row['recovery_success'] else 'no'} |"
        )
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary
