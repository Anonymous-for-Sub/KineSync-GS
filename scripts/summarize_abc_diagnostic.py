#!/usr/bin/env python3
"""Summarize the read-only ABC diagnostic horizon matrix on CPU.

This sidecar never imports or executes the ABC worker.  It only reads artifacts
already written under a matrix root and records the native evaluator's metrics
alongside independently derived trajectory diagnostics.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATRIX_ROOT = ROOT / "runs/external_abc_diagnostic_20260908"
DEFAULT_REPORT = ROOT / "project/ABC_DIAGNOSTIC_ANALYSIS_ZH.md"
SUMMARY_JSON_NAME = "abc_diagnostic_summary.json"
SUMMARY_CSV_NAME = "abc_diagnostic_summary.csv"

_BOTTLE_JOINT_RE = re.compile(r"bottle_(\d+)_joint$")
_QPOS_WIDTH = {"hinge": 1, "slide": 1, "ball": 4, "free": 7, "freejoint": 7}
_MOVEMENT_TOLERANCE_METERS = 0.005


class IncompleteMatrixError(RuntimeError):
    """Raised when a matrix has non-complete cases without --allow-incomplete."""


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _read_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return value


def _scalar(value: np.ndarray) -> Any:
    return value.item() if value.shape == () else value


def _trace_for_seed(run_dir: Path, seed: int) -> Path | None:
    expected = run_dir / f"seed_{seed}__physical_trace.npz"
    if expected.is_file():
        return expected
    traces = sorted(run_dir.glob("*__physical_trace.npz"))
    return traces[0] if len(traces) == 1 else None


def _scene_for_seed(run_dir: Path, seed: int) -> Path | None:
    expected = run_dir / f"seed_{seed}__scene.xml"
    if expected.is_file():
        return expected
    scenes = sorted(run_dir.glob("*__scene.xml"))
    return scenes[0] if len(scenes) == 1 else None


def _max_abs_diff(left: np.ndarray, right: np.ndarray) -> float | None:
    if left.shape != right.shape or left.size == 0:
        return None
    difference = np.abs(left.astype(np.float64) - right.astype(np.float64))
    return float(np.nanmax(difference)) if difference.size else 0.0


def _prefix_comparison(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    prefix_length = min(len(left), len(right))
    left_prefix, right_prefix = left[:prefix_length], right[:prefix_length]
    return {
        "prefix_length": int(prefix_length),
        "prefix_exact": bool(np.array_equal(left_prefix, right_prefix)),
        "prefix_max_abs_diff": _max_abs_diff(left_prefix, right_prefix),
        "full_shape_equal": list(left.shape) == list(right.shape),
        "left_shape": list(left.shape),
        "right_shape": list(right.shape),
    }


def _scene_qpos_addresses(scene_path: Path) -> dict[str, int]:
    """Recover named qpos addresses from an emitted scene XML without MuJoCo."""
    root = ET.parse(scene_path).getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"Scene XML has no worldbody: {scene_path}")
    address = 0
    addresses: dict[str, int] = {}
    for element in worldbody.iter():
        if element.tag not in {"joint", "freejoint"}:
            continue
        joint_type = element.get("type", "free" if element.tag == "freejoint" else "hinge")
        if joint_type not in _QPOS_WIDTH:
            raise ValueError(f"Unsupported joint type {joint_type!r} in {scene_path}")
        name = element.get("name")
        if name:
            if name in addresses:
                raise ValueError(f"Duplicate joint name {name!r} in {scene_path}")
            addresses[name] = address
        address += _QPOS_WIDTH[joint_type]
    return addresses


def _tail_diagnostics(trace_path: Path, scene_path: Path, scene: dict[str, Any]) -> dict[str, Any]:
    required = ("eval_bin_radius", "eval_min_rel_z", "eval_max_rel_z")
    missing = [name for name in required if name not in scene]
    if missing:
        return {"status": "unavailable", "reason": f"config.scene missing: {', '.join(missing)}"}

    addresses = _scene_qpos_addresses(scene_path)
    if "bin_joint" not in addresses:
        return {"status": "unavailable", "reason": "scene XML has no bin_joint"}
    bottle_entries = sorted(
        (int(match.group(1)), name, address)
        for name, address in addresses.items()
        if (match := _BOTTLE_JOINT_RE.fullmatch(name))
    )
    if not bottle_entries:
        return {"status": "unavailable", "reason": "scene XML has no bottle free joints"}

    with np.load(trace_path, allow_pickle=False) as trace:
        if "qpos" not in trace.files:
            return {"status": "unavailable", "reason": "physical trace has no qpos"}
        qpos = np.asarray(trace["qpos"])
    if qpos.ndim != 2 or not len(qpos):
        return {"status": "unavailable", "reason": "physical trace qpos is empty or not rank 2"}

    bin_address = addresses["bin_joint"]
    needed = max([bin_address + 3] + [address + 3 for _, _, address in bottle_entries])
    if qpos.shape[1] < needed:
        return {
            "status": "unavailable",
            "reason": f"qpos width {qpos.shape[1]} cannot address scene requirement {needed}",
        }

    bin_position = qpos[-1, bin_address : bin_address + 3]
    radius_limit = float(scene["eval_bin_radius"])
    lower_limit = float(scene["eval_min_rel_z"])
    upper_limit = float(scene["eval_max_rel_z"])
    bottles = []
    for index, name, address in bottle_entries:
        positions = qpos[:, address : address + 3]
        relative = positions[-1] - bin_position
        radial_distance = float(np.linalg.norm(relative[:2]))
        radial_margin = radius_limit - radial_distance
        lower_margin = float(relative[2] - lower_limit)
        upper_margin = float(upper_limit - relative[2])
        max_displacement = float(np.max(np.linalg.norm(positions - positions[0], axis=1)))
        if max_displacement <= _MOVEMENT_TOLERANCE_METERS:
            grasp_inference = "likely_not_grabbed_no_motion"
        else:
            grasp_inference = "motion_observed_grasp_not_proven"
        bottles.append(
            {
                "bottle_index": index,
                "bottle_joint": name,
                "final_relative_position": [float(value) for value in relative],
                "radial_distance": radial_distance,
                "radial_margin": radial_margin,
                "lower_height_margin": lower_margin,
                "upper_height_margin": upper_margin,
                "in_bin_by_native_threshold": bool(
                    radial_margin >= 0.0 and lower_margin >= 0.0 and upper_margin >= 0.0
                ),
                "outside_bin_radial_bound": bool(radial_margin < 0.0),
                "below_lower_height_bound": bool(lower_margin < 0.0),
                "above_upper_height_bound": bool(upper_margin < 0.0),
                "trajectory_max_displacement_m": max_displacement,
                "grasp_inference": grasp_inference,
                "grasp_inference_note": (
                    "trajectory-only inference; no contact or grasp truth is present in the trace"
                ),
            }
        )
    return {
        "status": "available",
        "method": "qpos positions plus the upstream evaluator thresholds; not evaluator truth",
        "thresholds": {
            "eval_bin_radius": radius_limit,
            "eval_min_rel_z": lower_limit,
            "eval_max_rel_z": upper_limit,
            "no_motion_tolerance_m": _MOVEMENT_TOLERANCE_METERS,
        },
        "bottles": bottles,
    }


def _blank_case(case: dict[str, Any], status: str, detail: str) -> dict[str, Any]:
    return {
        "case_id": case["id"],
        "cell": case.get("cell"),
        "physics": case.get("physics"),
        "chunks": case.get("chunks"),
        "seed": case.get("seed"),
        "policy_seed": case.get("policy_seed"),
        "status": status,
        "status_detail": detail,
        "native_success": None,
        "native_final_success": None,
        "native_num_bottles_in_bin": None,
        "native_max_bottles_in_bin": None,
        "native_num_active_bottles": None,
        "duration_seconds": None,
        "world_wall_seconds": None,
        "chunk_wall_seconds": None,
        "tail_diagnostics": None,
        "artifact_paths": {},
    }


def _summarize_complete_case(case: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    required = {
        "summary": run_dir / "summary.json",
        "config": run_dir / "config.yaml",
        "policy_inference": run_dir / "policy_inference.npz",
    }
    trace_path = _trace_for_seed(run_dir, int(case["seed"]))
    scene_path = _scene_for_seed(run_dir, int(case["seed"]))
    if trace_path is None:
        return _blank_case(case, "incomplete_artifacts", "complete receipt but physical trace is missing or ambiguous")
    if scene_path is None:
        return _blank_case(case, "incomplete_artifacts", "complete receipt but scene XML is missing or ambiguous")
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        return _blank_case(case, "incomplete_artifacts", f"complete receipt but missing: {', '.join(missing)}")

    try:
        summary = _read_json(required["summary"])
        config = _read_config(required["config"])
        worlds = summary.get("worlds")
        if not isinstance(worlds, list) or not worlds or not isinstance(worlds[0], dict):
            raise ValueError("summary.worlds[0] is unavailable")
        world = worlds[0]
        final_eval = world.get("final_task_eval")
        if not isinstance(final_eval, dict):
            raise ValueError("summary.worlds[0].final_task_eval is unavailable")
        with np.load(trace_path, allow_pickle=False) as trace:
            time_seconds = np.asarray(trace["time_seconds"])
            if time_seconds.ndim != 1 or not len(time_seconds):
                raise ValueError("physical trace time_seconds is empty or not rank 1")
        with np.load(required["policy_inference"], allow_pickle=False) as inference:
            required_arrays = ("state", "noise", "noise_present", "actions")
            missing_arrays = [name for name in required_arrays if name not in inference.files]
            if missing_arrays:
                raise ValueError(
                    "policy inference missing actual arrays: " + ", ".join(missing_arrays)
                )
        duration = float(time_seconds[-1] - time_seconds[0])
        scene = config.get("scene")
        if not isinstance(scene, dict):
            raise ValueError("config.scene is unavailable")
        tail = _tail_diagnostics(trace_path, scene_path, scene)
    except (KeyError, OSError, ValueError, yaml.YAMLError) as error:
        return _blank_case(case, "incomplete_artifacts", f"cannot parse complete artifacts: {error}")

    chunk_metrics = world.get("chunk_metrics", [])
    chunk_wall = None
    if isinstance(chunk_metrics, list) and all(isinstance(item, dict) for item in chunk_metrics):
        values = [item.get("wall_s") for item in chunk_metrics]
        if values and all(isinstance(value, (int, float)) for value in values):
            chunk_wall = float(sum(values))
    return {
        "case_id": case["id"],
        "cell": case.get("cell"),
        "physics": case.get("physics"),
        "chunks": case.get("chunks"),
        "seed": case.get("seed"),
        "policy_seed": case.get("policy_seed"),
        "status": "complete",
        "status_detail": "completion.json status=complete",
        "native_success": world.get("success"),
        "native_final_success": final_eval.get("success"),
        "native_num_bottles_in_bin": final_eval.get("num_bottles_in_bin"),
        "native_max_bottles_in_bin": final_eval.get("max_bottles_in_bin_so_far"),
        "native_num_active_bottles": final_eval.get("num_active_bottles"),
        "duration_seconds": duration,
        "world_wall_seconds": world.get("wall_s"),
        "chunk_wall_seconds": chunk_wall,
        "tail_diagnostics": tail,
        "artifact_paths": {
            "run": str(run_dir),
            "summary": str(required["summary"]),
            "config": str(required["config"]),
            "policy_inference": str(required["policy_inference"]),
            "physical_trace": str(trace_path),
            "scene_xml": str(scene_path),
        },
    }


def _load_inference(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        required = ("state", "noise", "noise_present", "actions")
        missing = [name for name in required if name not in source.files]
        if missing:
            raise ValueError(f"policy inference missing: {', '.join(missing)}")
        return {name: np.asarray(source[name]) for name in required}


def _load_qpos(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as source:
        if "qpos" not in source.files:
            raise ValueError("physical trace missing qpos")
        return np.asarray(source["qpos"])


def _same_seed_horizon_checks(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, Any], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["physics"], row["seed"]), []).append(row)
    checks = []
    for (physics, seed), group in sorted(grouped.items()):
        by_horizon = {row["chunks"]: row for row in group}
        short, long = by_horizon.get(120), by_horizon.get(240)
        check: dict[str, Any] = {
            "physics": physics,
            "seed": seed,
            "short_case_id": short["case_id"] if short else None,
            "long_case_id": long["case_id"] if long else None,
        }
        if short is None or long is None:
            check.update({"status": "unpaired", "detail": "120/240 pair not declared"})
            checks.append(check)
            continue
        if short["status"] != "complete" or long["status"] != "complete":
            check.update(
                {
                    "status": "not_ready",
                    "detail": f"short={short['status']}, long={long['status']}",
                }
            )
            checks.append(check)
            continue
        try:
            short_inference = _load_inference(Path(short["artifact_paths"]["policy_inference"]))
            long_inference = _load_inference(Path(long["artifact_paths"]["policy_inference"]))
            short_qpos = _load_qpos(Path(short["artifact_paths"]["physical_trace"]))
            long_qpos = _load_qpos(Path(long["artifact_paths"]["physical_trace"]))
            noise = _prefix_comparison(short_inference["noise"], long_inference["noise"])
            presence = _prefix_comparison(
                short_inference["noise_present"], long_inference["noise_present"]
            )
            actions = _prefix_comparison(short_inference["actions"], long_inference["actions"])
            qpos = _prefix_comparison(short_qpos, long_qpos)
            short_scene = Path(short["artifact_paths"]["scene_xml"])
            long_scene = Path(long["artifact_paths"]["scene_xml"])
            check.update(
                {
                    "status": "available",
                    "noise_exact": bool(noise["prefix_exact"] and presence["prefix_exact"]),
                    "noise_prefix_length": noise["prefix_length"],
                    "noise_prefix_max_abs_diff": noise["prefix_max_abs_diff"],
                    "noise_present_prefix_exact": presence["prefix_exact"],
                    "noise_full_shape_equal": noise["full_shape_equal"],
                    "actions_prefix_exact": actions["prefix_exact"],
                    "actions_prefix_length": actions["prefix_length"],
                    "actions_prefix_max_abs_diff": actions["prefix_max_abs_diff"],
                    "qpos_prefix_exact": qpos["prefix_exact"],
                    "qpos_prefix_length": qpos["prefix_length"],
                    "qpos_prefix_max_abs_diff": qpos["prefix_max_abs_diff"],
                    "initial_randomization_qpos_exact": bool(
                        len(short_qpos) > 0
                        and len(long_qpos) > 0
                        and np.array_equal(short_qpos[0], long_qpos[0])
                    ),
                    "initial_randomization_scene_exact": (
                        hashlib.sha256(short_scene.read_bytes()).hexdigest()
                        == hashlib.sha256(long_scene.read_bytes()).hexdigest()
                    ),
                }
            )
        except (KeyError, OSError, ValueError) as error:
            check.update({"status": "unavailable", "detail": str(error)})
        checks.append(check)
    return checks


def _csv_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flattened = []
    for row in rows:
        flattened.append(
            {
                **{key: value for key, value in row.items() if key not in {"tail_diagnostics", "artifact_paths"}},
                "tail_diagnostics_json": json.dumps(row["tail_diagnostics"], ensure_ascii=False, sort_keys=True)
                if row["tail_diagnostics"] is not None
                else "",
                "artifact_paths_json": json.dumps(row["artifact_paths"], ensure_ascii=False, sort_keys=True),
            }
        )
    return flattened


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "case_id", "cell", "physics", "chunks", "seed", "policy_seed", "status", "status_detail",
        "native_success", "native_final_success", "native_num_bottles_in_bin",
        "native_max_bottles_in_bin", "native_num_active_bottles", "duration_seconds",
        "world_wall_seconds", "chunk_wall_seconds", "tail_diagnostics_json", "artifact_paths_json",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(_csv_rows(rows))


def _display(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _render_report(payload: dict[str, Any]) -> str:
    lines = [
        "# ABC 诊断矩阵分析",
        "",
        f"- 矩阵根目录：`{payload['matrix_root']}`",
        f"- 生成时间（UTC）：`{payload['generated_at_utc']}`",
        f"- 证据阶段：`{payload.get('evidence_stage') or '未声明'}`。本报告不计入方法收益。",
        "- 该 sidecar 只读取已写入的 artifact；不会调用 policy、修改成功判据或影响运行。",
        "- `native_success` 为原生 evaluator 的 ever-success；末态“未抓取”仅由轨迹位移推断，不能替代接触或抓取真值。",
        "",
        "## 运行状态",
        "",
        "| case | physics | chunks | seed | 状态 | native 成功 | 末态入箱 | 最大入箱 | 物理时长(s) |",
        "| --- | --- | ---: | ---: | --- | --- | ---: | ---: | ---: |",
    ]
    for row in payload["cases"]:
        lines.append(
            "| {case_id} | {physics} | {chunks} | {seed} | {status} | {success} | {in_bin} | {max_in_bin} | {duration} |".format(
                case_id=row["case_id"], physics=row["physics"], chunks=row["chunks"], seed=row["seed"],
                status=row["status"], success=_display(row["native_success"]),
                in_bin=_display(row["native_num_bottles_in_bin"]),
                max_in_bin=_display(row["native_max_bottles_in_bin"]), duration=_display(row["duration_seconds"]),
            )
        )
    lines.extend(["", "未完成、失败或 artifact 不完整的 case 保持空值，绝不以 0 代替结果。", "", "## 同 Seed 短长一致性"])
    lines.extend(["", "| physics | seed | 状态 | noise exact | action prefix max diff | qpos prefix max diff | 初始 qpos | 初始 scene |", "| --- | ---: | --- | --- | ---: | ---: | --- | --- |"])
    for check in payload["same_seed_horizon_checks"]:
        lines.append(
            "| {physics} | {seed} | {status} | {noise} | {action} | {qpos} | {initial_qpos} | {initial_scene} |".format(
                physics=check["physics"], seed=check["seed"], status=check["status"],
                noise=_display(check.get("noise_exact")),
                action=_display(check.get("actions_prefix_max_abs_diff")),
                qpos=_display(check.get("qpos_prefix_max_abs_diff")),
                initial_qpos=_display(check.get("initial_randomization_qpos_exact")),
                initial_scene=_display(check.get("initial_randomization_scene_exact")),
            )
        )
    lines.extend([
        "",
        "`noise exact` 比较短 horizon 与长 horizon 的共同推理调用前缀，并同时要求 `noise_present` 一致；action/qpos 同样只比较共同前缀。max diff 始终保留，不以布尔通过掩盖差值。",
        "",
        "## 末态逐瓶诊断",
        "",
        "使用 scene XML 复原 qpos 地址，并按原生 evaluator 的 `eval_bin_radius`、`eval_min_rel_z`、`eval_max_rel_z` 重算末态桶内条件。`桶外`指半径界限外；高度两界分别报告。",
    ])
    for row in payload["cases"]:
        tail = row["tail_diagnostics"]
        if row["status"] != "complete" or not isinstance(tail, dict) or tail.get("status") != "available":
            continue
        lines.extend(["", f"### {row['case_id']}", "", "| bottle | 桶内 | 桶外(半径) | 低于下界 | 高于上界 | 位移(m) | 未抓取推断 |", "| --- | --- | --- | --- | --- | ---: | --- |"])
        for bottle in tail["bottles"]:
            lines.append(
                "| {name} | {inside} | {outside} | {below} | {above} | {displacement:.6g} | {grasp} |".format(
                    name=bottle["bottle_joint"], inside=_display(bottle["in_bin_by_native_threshold"]),
                    outside=_display(bottle["outside_bin_radial_bound"]),
                    below=_display(bottle["below_lower_height_bound"]),
                    above=_display(bottle["above_upper_height_bound"]),
                    displacement=bottle["trajectory_max_displacement_m"], grasp=bottle["grasp_inference"],
                )
            )
    lines.extend(["", "机器可读输出：`abc_diagnostic_summary.json` 与 `abc_diagnostic_summary.csv`（位于矩阵根目录）。", ""])
    return "\n".join(lines)


def summarize_matrix(
    matrix_root: Path,
    *,
    report_path: Path = DEFAULT_REPORT,
    allow_incomplete: bool = False,
) -> dict[str, Any]:
    """Read a declared matrix, write its JSON/CSV/Chinese report, and return it."""
    matrix_root = matrix_root.resolve()
    matrix = _read_json(matrix_root / "matrix.json")
    cases = matrix.get("cases")
    if not isinstance(cases, list):
        raise ValueError("matrix.json cases must be a list")
    rows = []
    for case in cases:
        if not isinstance(case, dict) or "id" not in case:
            raise ValueError("each matrix case must be an object with id")
        run_dir = matrix_root / str(case["id"])
        completion_path = run_dir / "completion.json"
        if not run_dir.is_dir() or not completion_path.is_file():
            rows.append(_blank_case(case, "pending", "case directory or completion.json not yet present"))
            continue
        completion = _read_json(completion_path)
        status = completion.get("status")
        if status == "complete":
            rows.append(_summarize_complete_case(case, run_dir))
        elif status == "failed":
            rows.append(_blank_case(case, "failed", str(completion.get("traceback", "worker reported failed"))))
        else:
            rows.append(_blank_case(case, "pending", f"completion status is {status!r}"))

    status_counts: dict[str, int] = {}
    for row in rows:
        status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1
    incomplete = [row for row in rows if row["status"] != "complete"]
    if incomplete and not allow_incomplete:
        names = ", ".join(str(row["case_id"]) for row in incomplete)
        raise IncompleteMatrixError(f"matrix has incomplete cases; rerun with --allow-incomplete: {names}")

    payload = {
        "format": "kinesync_abc_diagnostic_summary/v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "matrix_root": str(matrix_root),
        "evidence_stage": matrix.get("evidence_stage"),
        "policy_rng_scope": matrix.get("policy_rng_scope"),
        "status_counts": status_counts,
        "cases": rows,
        "same_seed_horizon_checks": _same_seed_horizon_checks(rows),
    }
    (matrix_root / SUMMARY_JSON_NAME).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_csv(matrix_root / SUMMARY_CSV_NAME, rows)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_render_report(payload), encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-root", type=Path, default=DEFAULT_MATRIX_ROOT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="write a partial report with pending/failed cases left blank",
    )
    args = parser.parse_args()
    try:
        payload = summarize_matrix(
            args.matrix_root, report_path=args.report, allow_incomplete=args.allow_incomplete
        )
    except IncompleteMatrixError as error:
        parser.error(str(error))
    print(json.dumps({"status_counts": payload["status_counts"], "matrix_root": payload["matrix_root"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
