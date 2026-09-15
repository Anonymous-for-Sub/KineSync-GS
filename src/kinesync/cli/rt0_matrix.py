"""Execute and aggregate a KineSync-GS RT0 experiment matrix."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import yaml

from kinesync.cli.rt0_joint_offset import execute_rt0
from kinesync.config import load_config
from kinesync.experiments.matrix import aggregate_rt0_runs


def execute_matrix(path: str | Path) -> dict:
    matrix_path = Path(path).expanduser().resolve()
    matrix = load_config(matrix_path)
    base_path = Path(matrix["base_config"])
    if not base_path.is_absolute():
        base_path = (matrix_path.parent / base_path).resolve()
    base = load_config(base_path)
    runs_root = Path(matrix["runs_root"]).expanduser().resolve()
    run_paths = []
    for case in matrix["cases"]:
        config = copy.deepcopy(base)
        config["runs_root"] = str(runs_root)
        config["seed"] = int(case["seed"])
        config["device"] = str(matrix.get("device", config.get("device", "cuda")))
        config["data"]["frame_index"] = int(case["frame_index"])
        config["data"]["cameras"] = list(case.get("cameras", ["head", "extra"]))
        config["data"]["anchors_per_link"] = int(
            case.get("anchors_per_link", matrix.get("anchors_per_link", 96))
        )
        config["target"]["injected_joint_offsets_rad"] = dict(case["offsets"])
        config["optimizer"]["steps"] = int(case.get("steps", matrix.get("steps", 140)))
        config["visualization"]["video_frames"] = int(
            case.get("video_frames", matrix.get("video_frames", 24))
        )
        run_id = str(case["run_id"])
        existing = runs_root / run_id
        if (existing / "metrics.json").is_file():
            run_paths.append(existing)
        else:
            run_paths.append(execute_rt0(config, run_id=run_id))
    summary_dir = Path(matrix["summary_dir"]).expanduser().resolve()
    summary = aggregate_rt0_runs(run_paths, summary_dir)
    (summary_dir / "matrix_config.yaml").write_text(
        yaml.safe_dump(matrix, sort_keys=False), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    args = parser.parse_args()
    summary = execute_matrix(args.matrix)
    print(yaml.safe_dump(summary, sort_keys=True))


if __name__ == "__main__":
    main()
