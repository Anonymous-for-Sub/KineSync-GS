#!/usr/bin/env python3
"""Run a compact, explicitly analysis-only external Franka GS recovery batch."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET
from dataclasses import replace

import cv2
# Mesh reference rendering remains CPU-only; Slurm owns CUDA device selection.
os.environ["MUJOCO_GL"] = "osmesa"
import mujoco
import numpy as np
import torch
import yaml

from kinesync.correction.joint_offset import JointOffsetRecovery
from kinesync.external_abc import require_gpu_allocation
from kinesync.external_gs.franka import (
    load_franka_external_gs,
    mujoco_fk_report,
    write_mujoco_mjcf,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    keys = list(rows[0])
    if any(list(row) != keys for row in rows):
        raise ValueError(f"CSV rows have incompatible fields: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _rgb(buffer: object) -> np.ndarray:
    image = buffer.rgb.detach().cpu().numpy()
    return np.clip(np.round(image * 255.0), 0, 255).astype(np.uint8)


def _label(image: np.ndarray, label: str, color: tuple[int, int, int]) -> np.ndarray:
    result = image.copy()
    cv2.rectangle(result, (0, 0), (result.shape[1], 38), color, thickness=-1)
    cv2.putText(result, label, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2, cv2.LINE_AA)
    return result


def _mesh_reference(destination: Path, camera, height: int, width: int) -> None:
    """Save a distinct MuJoCo mesh visual reference; it never enters the loss."""
    tree = ET.parse(destination)
    worldbody = tree.getroot().find("worldbody")
    if worldbody is None:
        raise ValueError("derived MJCF lacks worldbody")
    ET.SubElement(
        worldbody,
        "camera",
        name="framed_analysis",
        pos=" ".join(f"{value:.12g}" for value in camera.camera_to_world[:3, 3]),
        xyaxes=" ".join(f"{value:.12g}" for value in np.concatenate((camera.world_to_camera[0, :3], -camera.world_to_camera[1, :3]))),
        fovy=str(np.rad2deg(2.0 * np.arctan(camera.height / (2.0 * camera.intrinsic[1, 1])))),
    )
    tree.write(destination, encoding="utf-8", xml_declaration=True)
    model = mujoco.MjModel.from_xml_path(str(destination))
    model.vis.global_.offwidth = width
    model.vis.global_.offheight = height
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    with mujoco.Renderer(model, height=height, width=width) as renderer:
        renderer.update_scene(data, camera="framed_analysis")
        image = renderer.render()
    cv2.imwrite(str(destination.with_suffix(".png")), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def _select_cases(cases: list[dict[str, object]], selected_ids: tuple[str, ...]) -> list[dict[str, object]]:
    if not selected_ids:
        return cases
    wanted = set(selected_ids)
    available = {str(case.get("id", "")) for case in cases}
    unknown = wanted - available
    if unknown:
        raise ValueError(f"unknown case IDs: {sorted(unknown)}")
    return [case for case in cases if str(case["id"]) in wanted]


def run(config_path: Path, output: Path, *, case_ids: tuple[str, ...] = ()) -> None:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["cases"] = _select_cases(list(config["cases"]), case_ids)
    if config["experiment"]["target_provenance"] != "analysis_same_gs":
        raise ValueError("this runner only permits an explicit analysis_same_gs target")
    job_id = require_gpu_allocation()
    if not torch.cuda.is_available():
        raise RuntimeError("native external GS rendering requires a Slurm-provided CUDA device")
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(config_path, output / "config_source.yaml")
    (output / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    script_sha256 = _sha256(Path(__file__).resolve())
    (output / "script.sha256").write_text(script_sha256 + "\n", encoding="ascii")
    frames_dir = output / "native_frames"
    comparisons_dir = output / "comparisons"
    frames_dir.mkdir()
    comparisons_dir.mkdir()
    asset = load_franka_external_gs(config["assets"]["root"])
    if asset.provenance.reference_kind != "analysis_same_gs":
        raise ValueError("external GS source provenance is not analysis-only")
    states = np.asarray([case["qpos_rad"] for case in config["cases"]], dtype=np.float64)
    analysis_camera = asset.framed_analysis_camera(states, width=int(config["render"]["width"]), height=int(config["render"]["height"]))
    asset = replace(asset, camera=analysis_camera)
    derived_mjcf = write_mujoco_mjcf(asset, output / "panda_robotiq_224.absolute.xml")
    _mesh_reference(derived_mjcf, analysis_camera, config["render"]["height"], config["render"]["width"])
    parity = mujoco_fk_report(asset, states)
    device = torch.device("cuda")
    backend = asset.cuda_backend(
        output_size=(int(config["render"]["height"]), int(config["render"]["width"])),
        per_link=int(config["render"]["gaussians_per_link"]),
        device=device,
    )
    backend.background = tuple(float(value) for value in config["render"]["background"])
    result_rows: list[dict[str, object]] = []
    trace_rows: list[dict[str, object]] = []
    video_frames: list[np.ndarray] = []
    for index, case in enumerate(config["cases"]):
        true_q = torch.tensor(case["qpos_rad"], dtype=torch.float32, device=device)
        injected = torch.tensor(case["injected_measurement_offset_rad"], dtype=torch.float32, device=device)
        measured = true_q + injected
        with torch.no_grad():
            target = backend.render(true_q)
            baseline = backend.render_buffers(measured)["front"]
            reference = backend.render_buffers(true_q)["front"]
        recovery = JointOffsetRecovery(
            backend,
            joint_names=asset.joint_names,
            selected_joints=config["recovery"]["selected_joints"],
            steps=int(config["recovery"]["steps"]),
            learning_rate=float(config["recovery"]["learning_rate"]),
            offset_bound=float(config["recovery"]["offset_bound_rad"]),
            seed=int(config["experiment"]["seed"]) + index,
        ).recover(measured, target, reference_qpos=true_q)
        with torch.no_grad():
            inverse = backend.render_buffers(recovery.corrected_qpos.to(device))["front"]
        raw_target = _rgb(reference)
        raw_baseline = _rgb(baseline)
        raw_inverse = _rgb(inverse)
        cv2.imwrite(str(frames_dir / f"{case['id']}__target_analysis_same_gs.png"), cv2.cvtColor(raw_target, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(frames_dir / f"{case['id']}__baseline.png"), cv2.cvtColor(raw_baseline, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(frames_dir / f"{case['id']}__gaussian-inverse-unguarded.png"), cv2.cvtColor(raw_inverse, cv2.COLOR_RGB2BGR))
        panel = np.concatenate((
            _label(raw_target, "Analysis target: same GS", (80, 80, 80)),
            _label(raw_baseline, "Baseline: measured state", (139, 158, 165)),
            _label(raw_inverse, "Gaussian inverse: unguarded", (107, 39, 23)),
        ), axis=1)
        comparison_path = comparisons_dir / f"{case['id']}__target-baseline-ours.png"
        cv2.imwrite(str(comparison_path), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
        video_frames.extend([panel] * 8)
        for method, qpos, rgb_mse, optimization_loss, image_path in (
            ("baseline", measured, float((baseline.rgb - reference.rgb).square().mean()), None, frames_dir / f"{case['id']}__baseline.png"),
            ("gaussian_inverse_unguarded", recovery.corrected_qpos.to(device), float((inverse.rgb - reference.rgb).square().mean()), recovery.final_loss, frames_dir / f"{case['id']}__gaussian-inverse-unguarded.png"),
        ):
            result_rows.append({
                "case_id": case["id"], "method": method, "target_provenance": "analysis_same_gs",
                "is_independent_ground_truth": False, "true_qpos_rad": json.dumps(case["qpos_rad"]),
                "measured_qpos_rad": json.dumps(measured.detach().cpu().tolist()),
                "estimated_qpos_rad": json.dumps(qpos.detach().cpu().tolist()),
                "joint_mae_deg": float((qpos - true_q).abs().mean().detach().cpu() * 180.0 / np.pi),
                "render_rgb_mse": rgb_mse, "optimization_loss": optimization_loss,
                "native_frame": str(image_path.relative_to(output)),
            })
        for row in recovery.trace:
            trace_rows.append({"case_id": case["id"], "target_provenance": "analysis_same_gs", **row})
    _write_csv(output / "results.csv", result_rows)
    _write_csv(output / "optimizer_trace.csv", trace_rows)
    coordinate_rows = []
    for link_index, link_name in asset.link_names.items():
        points = asset.local_xyz[asset.link_index == link_index].numpy()
        coordinate_rows.append({
            "link_index": link_index, "link_name": link_name, "coordinate_frame": f"{link_name}_local_m",
            "point_count": len(points), "min_xyz_m": json.dumps(points.min(axis=0).tolist()),
            "max_xyz_m": json.dumps(points.max(axis=0).tolist()), "fk_reference": "derived_mjcf_mujoco",
            "max_fk_translation_error_m": parity.max_translation_error_m,
            "max_fk_rotation_abs_error": parity.max_rotation_abs_error,
        })
    _write_csv(output / "coordinate_audit.csv", coordinate_rows)
    video_path = output / "recovery_comparisons.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 8.0, (video_frames[0].shape[1], video_frames[0].shape[0]))
    for frame in video_frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    manifest = {
        "schema": "kinesync.external_gs.franka.analysis.v1",
        "target_provenance": "analysis_same_gs",
        "claim_boundary": "Targets are rendered from the same external GS asset; results are analysis only, not real ground truth or cross-domain evaluation.",
        "appearance_training": "none; pre-existing external PLY fields are read-only",
        "mesh_reference": "independent MuJoCo mesh render, visual-only, excluded from optimization and numerical comparison",
        "camera_provenance": "framed_analysis_camera from all external PLY points over fixed analysis states; not an external or real-data calibration",
        "method": "gaussian_inverse_unguarded; no KineSync component protection is applied in this batch",
        "source_asset_sha256": asset.provenance.asset_sha256,
        "source_mjcf_sha256": _sha256(asset.provenance.mjcf_path),
        "derived_mjcf_sha256": _sha256(derived_mjcf),
        "config_source_sha256": _sha256(config_path),
        "config_sha256": _sha256(output / "config.yaml"),
        "script_sha256": script_sha256,
        "fk_parity": parity.__dict__,
        "gaussian_count_loaded": int(asset.local_xyz.shape[0]),
        "gaussian_count_rendered": int(len(backend.local_xyz)),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "slurm_job_id": job_id,
        "selected_case_ids": [case["id"] for case in config["cases"]],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"run": str(output), "cases": len(config["cases"]), "fk": parity.__dict__}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case-id", action="append", default=[])
    args = parser.parse_args()
    run(args.config.resolve(), args.output.resolve(), case_ids=tuple(args.case_id))


if __name__ == "__main__":
    main()
