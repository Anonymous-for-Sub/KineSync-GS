#!/usr/bin/env python3
"""Aggregate actual fused-candidate optimizer traces on CPU, without decisions."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_risk_atomic_heldout import (
    CONDITIONS, GROUP_LABELS, ROOT, bootstrap_indices, load_validated,
    read_csv, require, sha256, vector, write_csv,
)


STYLE = {
    "clean_nonzero": {"color": "#6B2717", "linestyle": "-", "style_name": "solid"},
    "clean_zero": {"color": "#8B9EA5", "linestyle": "--", "style_name": "dashed"},
    "one_view_stale_nonzero": {"color": "#CC9E4C", "linestyle": "-.", "style_name": "dash-dot"},
    "one_view_stale_zero": {"color": "#442C1B", "linestyle": ":", "style_name": "dotted"},
}


def trace_rows(trace, candidates):
    lookup = {r["state_id"]: r for r in candidates}
    require(len(lookup) == len(candidates) == 48, "expected 48 unique candidate states")
    poses = {r["case_id"] for r in candidates}
    require(len(poses) == 12 and {(r["case_id"], r["condition_id"]) for r in candidates}
            == {(p,c) for p in poses for c in CONDITIONS}, "expected complete 12-pose x four-condition matrix")
    fused = [r for r in trace if r["optimizer_source"] == "fused_candidate"]
    require(len(fused) == 2928, "expected 2928 fused rows = 48 states x 61 steps")
    keys = [(r["state_id"], int(r["step"])) for r in fused]
    require(len(set(keys)) == len(keys), "duplicate fused state/step")
    require(set(keys) == {(state, step) for state in lookup for step in range(61)},
            "each state must have exact 61-step matrix, steps 0..60")
    rows = []
    for row in fused:
        candidate = lookup[row["state_id"]]
        require(row["split"] == "heldout" and row["target_provenance"] == "analysis_same_gs",
                "unexpected trace split/provenance")
        require(all(row[k] == candidate[k] for k in ("case_id", "condition_id")), "trace metadata mismatch")
        step = int(row["step"])
        qpos = np.asarray([float(row[f"qpos_joint{j}"]) for j in range(1,8)])
        loss = float(row["loss"])
        require(np.isfinite(qpos).all() and np.isfinite(loss) and loss >= 0, "nonfinite state or invalid loss")
        if step == 0:
            require(np.array_equal(qpos, vector(candidate["measured_qpos_rad"])), "initial trace differs from measurement")
        if step == 60:
            require(np.array_equal(qpos, vector(candidate["candidate_qpos_rad"])), "terminal trace differs from candidate")
        # Read reference only for this post-hoc qMAE; no update decision is made.
        qmae = float(np.rad2deg(np.abs(qpos - vector(candidate["true_qpos_rad"])).mean()))
        rows.append({"state_id": row["state_id"], "case_id": row["case_id"],
                     "condition_id": row["condition_id"], "optimizer_source": "fused_candidate",
                     "step": step, "loss": loss, "qmae_deg": qmae,
                     "qpos_rad": json.dumps(qpos.tolist())})
    return sorted(rows, key=lambda r: (CONDITIONS.index(r["condition_id"]), r["case_id"], r["step"]))


def aggregate(rows):
    poses = sorted({r["case_id"] for r in rows})
    require(len(poses) == 12 and len(rows) == 2928, "aggregation requires all 12 poses, 61 steps, four conditions")
    lookup = {(r["case_id"], r["condition_id"], r["step"]): r for r in rows}
    require(len(lookup) == len(rows), "duplicate aggregation key")
    draws = bootstrap_indices()
    output = []
    for condition in CONDITIONS:
        metrics = {}
        for metric in ("loss", "qmae_deg"):
            values = np.asarray([[lookup[pose,condition,step][metric] for step in range(61)] for pose in poses])
            sampled = values[draws].mean(axis=1)
            metrics[metric] = (values.mean(axis=0), *np.quantile(sampled, [.025,.975], axis=0))
        for step in range(61):
            row = {"condition_id": condition, "step": step, "pose_count": 12, "bootstrap_replicates": 10000,
                   "optimizer_source": "fused_candidate", **STYLE[condition]}
            for metric, (mean, low, high) in metrics.items():
                row.update({f"{metric}_mean": float(mean[step]), f"{metric}_ci_low": float(low[step]),
                            f"{metric}_ci_high": float(high[step])})
            output.append(row)
    return output


def endpoint_summary(summary):
    endpoints = []
    for condition in CONDITIONS:
        first = next(r for r in summary if r["condition_id"] == condition and r["step"] == 0)
        last = next(r for r in summary if r["condition_id"] == condition and r["step"] == 60)
        initial, final = first["loss_mean"], last["loss_mean"]
        endpoints.append({"condition_id": condition, "loss_initial": initial, "loss_final": final,
                          "loss_relative_reduction_percent": 100*(initial-final)/initial if initial > 0 else None,
                          "qmae_initial_deg": first["qmae_deg_mean"], "qmae_final_deg": last["qmae_deg_mean"],
                          "qmae_change_deg": last["qmae_deg_mean"]-first["qmae_deg_mean"]})
    return endpoints


def plot(summary, visuals):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    visuals.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.linewidth": .7, "xtick.labelsize": 7, "ytick.labelsize": 7})
    figures = []
    for metric, suffix, ylabel, scale in (
        ("loss", "optimizer-loss", r"Candidate visual loss ($\times 10^{-3}$)", 1000.),
        ("qmae_deg", "optimizer-state-error", "Candidate joint qMAE (deg)", 1.),
    ):
        fig, ax = plt.subplots(figsize=(3.5,3.5))
        fig.subplots_adjust(left=.18, right=.97, bottom=.16, top=.78)
        records = []
        for condition in CONDITIONS:
            selected = sorted([r for r in summary if r["condition_id"] == condition], key=lambda r: r["step"])
            steps = [r["step"] for r in selected]
            mean = np.asarray([r[f"{metric}_mean"] for r in selected])
            low = np.asarray([r[f"{metric}_ci_low"] for r in selected])
            high = np.asarray([r[f"{metric}_ci_high"] for r in selected])
            style = STYLE[condition]
            ax.fill_between(steps, scale*low, scale*high, color=style["color"], alpha=.11, linewidth=0)
            ax.plot(steps, scale*mean, color=style["color"], linestyle=style["linestyle"],
                    linewidth=1.5, label=GROUP_LABELS[condition], zorder=3)
            for row in selected:
                records.append({"condition_id": condition, "step": row["step"], "metric": metric,
                                "mean": row[f"{metric}_mean"], "ci_low": row[f"{metric}_ci_low"],
                                "ci_high": row[f"{metric}_ci_high"], "plot_display_scale": scale,
                                "pose_count": 12, "bootstrap_replicates": 10000,
                                "optimizer_source": "fused_candidate", **style})
        ax.set_xlim(0,60)
        ax.set_xticks([0,15,30,45,60])
        ax.set_xlabel("Candidate optimizer step")
        ax.set_ylabel(ylabel)
        ax.grid(color="#EEEEEE", linewidth=.5)
        ax.set_axisbelow(True)
        legend = ax.legend(loc="lower center", bbox_to_anchor=(.46,1.04), ncol=2,
                           frameon=False, fontsize=6, columnspacing=1.3, handlelength=2.7)
        fig.canvas.draw()
        require(not ax.get_title() and fig._suptitle is None, "figures must be titlefree")
        renderer = fig.canvas.get_renderer()
        labels = [legend, ax.xaxis.label, ax.yaxis.label, *ax.get_xticklabels()]
        labels += [label for value,label in zip(ax.get_yticks(),ax.get_yticklabels()) if ax.get_ylim()[0] <= value <= ax.get_ylim()[1]]
        for label in labels:
            box = label.get_window_extent(renderer)
            require(box.x0 >= 0 and box.y0 >= 0 and box.x1 <= fig.bbox.width and box.y1 <= fig.bbox.height,
                    "figure text or legend clipped")
        stem = f"Q4__external-heldout-{suffix}__square-titlefree"
        for extension in ("pdf","png"):
            fig.savefig(visuals / f"{stem}.{extension}", dpi=600, facecolor="white")
        write_csv(visuals / f"{stem}.csv", records)
        with Image.open(visuals / f"{stem}.png") as image:
            require(image.size == (2100,2100), "PNG must be square 2100 pixels")
        figures.append({"stem": stem, "titlefree": True, "size_inches": [3.5,3.5],
                        "files": {ext: sha256(visuals / f"{stem}.{ext}") for ext in ("pdf","png","csv")}})
        plt.close(fig)
    return figures


def report(summary):
    endpoints = summary["endpoints"]
    lines = ["# External Heldout：真实 Candidate 优化轨迹", "",
             "仅分析 optimizer_source=fused_candidate 的实际 qpos/loss 日志。固定全部 12 姿态 × 四条件 × 61 步（0–60），共 2,928 行，不挑案例，不画 ours 中间提交，不使用 gradient_norm。末步 gradient_norm=0 是日志占位，不能解释为梯度收敛。", "",
             "所有 step=0 状态与 measurement 完全一致，step=60 与实际 candidate 完全一致；freeze source hashes、完整 heldout 矩阵及 optimizer_trace_sha256 均已校验。", "",
             "## 数据趋势", "",
             "| 条件 | 初始 loss | 末步 loss | loss 降低 (%) | 初始 qMAE (度) | 末步 qMAE (度) | qMAE 变化 (度) |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    discordant = []
    for row in endpoints:
        percent = "NA" if row["loss_relative_reduction_percent"] is None else f"{row['loss_relative_reduction_percent']:.6f}"
        lines.append(f"| {row['condition_id']} | {row['loss_initial']:.9g} | {row['loss_final']:.9g} | {percent} | {row['qmae_initial_deg']:.6f} | {row['qmae_final_deg']:.6f} | {row['qmae_change_deg']:+.6f} |")
        if row["loss_final"] < row["loss_initial"] and row["qmae_change_deg"] > 0:
            discordant.append(row["condition_id"])
    lines += ["", ("在本次固定 heldout 中，" + "、".join(discordant) + " 的平均 visual loss 下降，但平均 qMAE 上升。这支持不能仅把视觉 loss 下降当作关节状态改善的证据。" if discordant else
                    "本次各条件的端点均值未出现 loss 下降而 qMAE 上升的组合；不据此声称 loss 与状态误差不一致。"),
              "轨迹展示每一步实际均值与波动，不将端点改善表述为逐步单调下降，也不把 qMAE 当 task success。", "",
              "## 图注与颜色映射", "",
              "颜色在本图表示 condition，不表示方法。Clean/nonzero = #6B2717 实线；Clean/zero = #8B9EA5 虚线；Stale/nonzero = #CC9E4C 点划线；Stale/zero = #442C1B 点线。四条曲线全部来自同一 fused-candidate 优化器，不是四种方法。", "",
              "**Loss 图（中文）：** 全部 12 个 heldout 姿态在四种观测条件下的真实候选优化 visual loss。线为按条件的姿态均值，阴影为 10,000 次姿态整簇 bootstrap 的逐步 95% percentile CI。纵轴显示值按 10^-3 标度；CSV 保存未缩放原始 loss。",
              "**Loss caption (English):** Actual fused-candidate visual-loss trajectories across all 12 heldout poses under four conditions. Lines show pose means; bands are pointwise 95% percentile intervals from 10,000 paired pose-cluster bootstrap replicates. The displayed loss axis is in units of 10^-3; CSV values are unscaled. Colors and line styles encode conditions, not methods.", "",
              "**State-error 图（中文）：** 同一批真实候选 qpos 轨迹相对只用于后置评价的 true_qpos 的关节绝对误差均值（qMAE，度）。全部七关节等权，12 姿态按条件聚合；与 loss 图使用相同的姿态 bootstrap 抽样。阴影为逐步 CI，不是同时置信带，不包含任何虚构的 ours 中间状态。",
              "**State-error caption (English):** Post-hoc joint qMAE of the same actual fused-candidate trajectories against evaluation-only reference qpos. Errors average all seven joints and all 12 heldout poses per condition. The same pose bootstrap draws are shared with the loss plot. Bands are pointwise 95% intervals, not simultaneous confidence bands; no intermediate committed states are synthesized.", "",
              "**Mapping for both captions:** Clean/nonzero: #6B2717 solid; Clean/zero: #8B9EA5 dashed; Stale/nonzero: #CC9E4C dash-dot; Stale/zero: #442C1B dotted.", "",
              "## 文件与合并", "",
              "- 独立脚本：`code/scripts/analyze_risk_atomic_optimizer_trace.py`。",
              "- 测试：`code/tests/test_analyze_risk_atomic_optimizer_trace.py`。",
              "- 数据：`code/runs/risk_atomic_heldout_20260908/optimizer_trace_rows.csv`、`optimizer_trace_summary.csv`、`optimizer_trace_summary.json`；handoff 对应 `external_heldout_optimizer_trace_*` 副本。",
              "- Leader 请合并 `paper-visuals/Q4__external-heldout-optimizer-loss__square-titlefree.{pdf,png,csv}` 和 `paper-visuals/Q4__external-heldout-optimizer-state-error__square-titlefree.{pdf,png,csv}`，本脚本不修改任何已有 manifest。",
              "- CPU 复现：`<python> scripts/analyze_risk_atomic_optimizer_trace.py`；无需 GPU。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "runs/external_gs_guard_franka_heldout_20260908")
    parser.add_argument("--freeze", type=Path, default=ROOT / "runs/risk_atomic_freeze_20260908/calibration.json")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/risk_atomic_heldout_20260908")
    parser.add_argument("--handoff", type=Path, default=ROOT.parent / "paper-handoff-20260908")
    parser.add_argument("--visuals", type=Path, default=ROOT.parent / "paper-handoff-20260908/paper-visuals")
    args = parser.parse_args()
    require(args.output.resolve() not in (args.run.resolve(), args.freeze.parent.resolve()), "refusing to overwrite source run/freeze")
    freeze, candidates, _, audit = load_validated(args.freeze, args.run)
    require(freeze["config"]["recovery"]["steps"] == 60, "expected frozen 60 optimizer updates / 61 logged states")
    path = args.run / "optimizer_trace_qpos.csv"
    trace_hash = sha256(path)
    flush = json.loads((args.run / "candidate_flush.json").read_text())
    require(flush["optimizer_trace_sha256"] == trace_hash, "optimizer trace source hash mismatch")
    trace = read_csv(path)
    require(flush["trace_row_count"] == len(trace), "optimizer trace total row count differs from flush receipt")
    rows = trace_rows(trace, candidates)
    grouped = aggregate(rows)
    for directory in (args.output, args.handoff):
        directory.mkdir(parents=True, exist_ok=True)
    figures = plot(grouped, args.visuals)
    result = {"created_utc": datetime.now(timezone.utc).isoformat(), "audit": audit,
              "optimizer_trace_source_sha256": trace_hash, "source_rows_total": len(trace),
              "fused_rows": len(rows), "case_count": 48, "pose_count": 12, "steps": list(range(61)),
              "bootstrap": {"replicates": 10000, "seed": 20260908, "unit": "pose",
                            "interval": "pointwise 95% percentile", "same_draws_all_steps_conditions_metrics": True},
              "gradient_norm_used": False, "synthesized_ours_states": False, "decision_rules_changed": False,
              "palette_encodes": "condition, not method", "style_mapping": STYLE,
              "endpoints": endpoint_summary(grouped), "figures": figures,
              "trace_script_sha256": sha256(__file__)}
    for path, expected in {**audit["input_hashes"], **audit["source_hashes_verified"],
                           str(args.run / "optimizer_trace_qpos.csv"): trace_hash}.items():
        require(sha256(path) == expected, "input source changed during trace analysis")
    write_csv(args.output / "optimizer_trace_rows.csv", rows)
    write_csv(args.output / "optimizer_trace_summary.csv", grouped)
    (args.output / "optimizer_trace_summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    for name in ("optimizer_trace_rows.csv", "optimizer_trace_summary.csv", "optimizer_trace_summary.json"):
        shutil.copyfile(args.output / name, args.handoff / f"external_heldout_{name}")
    text = report(result)
    for directory in (args.output, args.handoff):
        (directory / "EXTERNAL_HELDOUT_OPTIMIZER_TRACE_ZH.md").write_text(text, encoding="utf-8")
    print(json.dumps({"fused_rows": len(rows), "steps_per_state": 61, "aggregated_rows": len(grouped),
                      "figures": [r["stem"] for r in figures], "endpoints": result["endpoints"]}, indent=2))


if __name__ == "__main__":
    main()
