#!/usr/bin/env python3
"""CPU-only, source-verified Gaussian-center diagnostics of recorded states."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch

from kinesync.external_gs.franka import load_franka_external_gs

COLORS = {"ours": "#6B2717", "unguarded": "#CC9E4C",
          "baseline": "#8B9EA5", "reference": "#442C1B"}
LABELS = {"ours": "KineSync-GS", "unguarded": "Unguarded",
          "baseline": "Raw state", "reference": "Reference"}
PAIRS = ((0, 1), (0, 2), (1, 2))


def paired_distances_mm(points, reference):
    points, reference = np.asarray(points), np.asarray(reference)
    if points.shape != reference.shape or points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Expected corresponding N x 3 point arrays")
    if not np.isfinite(points).all() or not np.isfinite(reference).all():
        raise ValueError("Non-finite geometry")
    return np.linalg.norm(points - reference, axis=1) * 1000.0


def world_points(asset, qpos):
    q = torch.as_tensor(qpos, dtype=torch.float64, device="cpu")
    with torch.no_grad():
        poses = asset.kinematics.forward(q)
        transforms = torch.stack([poses[asset.link_names[i]] for i in sorted(asset.link_names)])
        frames = transforms[asset.link_index]
        local = asset.local_xyz.to(dtype=torch.float64, device="cpu")
        result = torch.bmm(frames[:, :3, :3], local.unsqueeze(-1)).squeeze(-1) + frames[:, :3, 3]
    return result.numpy()


def projection(ax, clouds, comparator, dims, limits, compact=False):
    # Identical coordinates and limits for every method; no displacement scaling.
    for role, marker, size, alpha in ((comparator, ".", 2.4, .50),
                                      ("ours", ".", 1.8, .62),
                                      ("reference", "+", 2.0, .40)):
        pts = clouds[role] * 1000
        if role == "reference":
            pts = pts[::5]
        ax.scatter(pts[:, dims[0]], pts[:, dims[1]], s=size, marker=marker,
                   c=COLORS[role], alpha=alpha, linewidths=.25, rasterized=True)
    for axis, dim in zip((ax.xaxis, ax.yaxis), dims):
        axis.set_major_locator(plt.MaxNLocator(4))
    ax.set_xlim(*limits[dims[0]])
    ax.set_ylim(*limits[dims[1]])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"{'XYZ'[dims[0]]} (mm)")
    ax.set_ylabel(f"{'XYZ'[dims[1]]} (mm)")
    handles = [Line2D([], [], color=COLORS[r], marker="+" if r == "reference" else "o",
                      linestyle="none", markersize=3, label=LABELS[r])
               for r in ("reference", comparator, "ours")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, 1.22),
              ncol=1 if compact else 3, fontsize=5.6 if compact else 6.1,
              handletextpad=.3, columnspacing=.8, labelspacing=.2)


def histogram(ax, errors, comparator):
    maximum = max(float(np.max(v)) for v in errors.values())
    bins = np.linspace(0, max(maximum * 1.025, .1), 36)
    for role in (comparator, "ours"):
        ax.hist(errors[role], bins=bins, weights=np.full(len(errors[role]), 100 / len(errors[role])),
                histtype="stepfilled", alpha=.17, color=COLORS[role])
        ax.hist(errors[role], bins=bins, weights=np.full(len(errors[role]), 100 / len(errors[role])),
                histtype="step", linewidth=1.1, color=COLORS[role],
                label=f"{LABELS[role]}: {np.mean(errors[role]):.2f} mm")
    ax.set_xlabel("Center displacement (mm)")
    ax.set_ylabel("Gaussian centers (%)")
    ax.set_xlim(0, bins[-1])
    ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    ax.yaxis.set_major_locator(plt.MaxNLocator(4))
    ax.legend(loc="upper center", bbox_to_anchor=(.5, 1.22), fontsize=5.8,
              handlelength=1.2, labelspacing=.2)


def export(fig, stem, pixels):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax in fig.axes:
        if ax.get_title():
            raise ValueError("Rendered titles are not allowed")
        bbox = ax.get_tightbbox(renderer)
        if bbox.x0 < -1 or bbox.y0 < -1 or bbox.x1 > fig.bbox.width + 1 or bbox.y1 > fig.bbox.height + 1:
            raise ValueError(f"Clipped axes or legend: {stem.name}")
    fig.savefig(stem.with_suffix(".pdf"), dpi=400)
    fig.savefig(stem.with_suffix(".png"), dpi=pixels / fig.get_figwidth())
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    source = json.loads(args.manifest.read_text())
    asset = load_franka_external_gs(args.asset_root)
    if asset.provenance.asset_sha256 != source["source_asset_sha256"]:
        raise ValueError("Asset differs from the evaluated source")
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.linewidth": .65, "xtick.major.width": .65,
                         "ytick.major.width": .65, "legend.frameon": False,
                         "pdf.fonttype": 42, "savefig.facecolor": "white"})
    summary, artifacts = [], []
    for condition, comparator in (("clean_nonzero", "baseline"), ("one_view_stale_nonzero", "unguarded")):
        state = "franka_g__" + condition
        rows = {r["role"]: r for r in source["frames"] if r["state_id"] == state and r["camera"] == "view_a"}
        if len(rows) != 5:
            raise ValueError(f"Incomplete source state: {state}")
        points = {role: world_points(asset, r["qpos_rad"]) for role, r in rows.items()}
        for role, r in rows.items():
            actual = np.rad2deg(np.mean(np.abs(np.asarray(r["qpos_rad"]) - rows["reference"]["qpos_rad"])))
            if not np.isclose(actual, r["qmae_deg"], atol=1e-8):
                raise ValueError("Source joint MAE mismatch")
        np.savez_compressed(output / f"D6__{state}__source-geometry.npz", **points,
                            link_index=asset.link_index.numpy(),
                            **{r + "_qpos": np.asarray(row["qpos_rad"]) for r, row in rows.items()})
        for scope, selected in (("moving-links", asset.link_index.numpy() > 0),
                                 ("distal-links", asset.link_index.numpy() >= 6)):
            clouds = {r: p[selected] for r, p in points.items()}
            errors = {r: paired_distances_mm(p, clouds["reference"]) for r, p in clouds.items() if r != "reference"}
            combined = np.concatenate([clouds[r] for r in ("reference", comparator, "ours")]) * 1000
            low, high = combined.min(axis=0), combined.max(axis=0)
            span = float(np.max(high - low)) * 1.13
            center = (low + high) / 2
            limits = np.column_stack((center - span / 2, center + span / 2))
            for role, values in errors.items():
                summary.append({"state_id": state, "scope": scope, "role": role,
                                "n_gaussians": len(values), "mean_displacement_mm": float(values.mean()),
                                "p95_displacement_mm": float(np.quantile(values, .95)),
                                "qmae_deg": rows[role]["qmae_deg"],
                                "gate_accepted": rows[role]["atomic_gate_accepted"]})
            for panel in range(4):
                fig, ax = plt.subplots(figsize=(89 / 25.4, 89 / 25.4))
                fig.subplots_adjust(left=.19, right=.96, bottom=.17, top=.80)
                if panel < 3:
                    projection(ax, clouds, comparator, PAIRS[panel], limits)
                    name = "projection-" + "".join("xyz"[d] for d in PAIRS[panel])
                else:
                    histogram(ax, errors, comparator)
                    name = "residual-distribution"
                stem = f"D6__{state}__{scope}__{name}__square-titlefree"
                export(fig, output / stem, 2100)
                artifacts.append(stem)
            fig, axes = plt.subplots(1, 4, figsize=(178 / 25.4, 59 / 25.4))
            fig.subplots_adjust(left=.065, right=.988, bottom=.23, top=.72, wspace=.67)
            with plt.rc_context({"font.size": 6, "axes.labelsize": 6, "xtick.labelsize": 5.5, "ytick.labelsize": 5.5}):
                for panel, ax in enumerate(axes):
                    ax.tick_params(labelsize=5.5)
                    if panel < 3:
                        projection(ax, clouds, comparator, PAIRS[panel], limits, compact=True)
                    else:
                        histogram(ax, errors, comparator)
                    ax.xaxis.label.set_size(6)
                    ax.yaxis.label.set_size(6)
            stem = f"D6__{state}__{scope}__geometry-diagnostics__double-column-4x1"
            export(fig, output / stem, 4200)
            artifacts.append(stem)
    with (output / "D6__geometry-metrics.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    receipt = {"source_manifest": str(args.manifest.resolve()),
               "source_manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
               "asset_sha256": asset.provenance.asset_sha256,
               "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "backend": "CPU float64 FK and Python/matplotlib; no new optimization or rasterizer",
               "selection": "First heldout case by ID, franka_g; clean and one-view-stale nonzero conditions",
               "reference": "Known same-asset controlled state; not independent real-world ground truth",
               "displacement": "Euclidean distance between identical Gaussian center IDs after FK, millimeters",
               "scope": "moving-links=1..7; distal-links=6..7; no opacity filtering or point resampling",
               "projection": "Equal metric scales, union limits, true displacements; reference markers shown every fifth center",
               "statistics": "Descriptive point distribution for one pose per condition; points are not independent trials",
               "figures": artifacts, "metrics": summary}
    (output / "D6__geometry-provenance.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"figures": len(artifacts), "metrics": summary}, indent=2))


if __name__ == "__main__":
    main()
