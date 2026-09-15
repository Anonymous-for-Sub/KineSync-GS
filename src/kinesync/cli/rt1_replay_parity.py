"""Render full Route-A RGB and compare KineSync with inherited RoboSplat."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch

from kinesync.config import load_config
from kinesync.data.route_a import RouteAAssets, RouteAPaths
from kinesync.geometry.urdf import TorchURDFKinematics
from kinesync.observation.cuda_gaussian import CudaGaussianBackend
from kinesync.runs.artifacts import RunArtifacts
from kinesync.validation.legacy_robosplat import render_legacy_rgb


def _label(rgb: np.ndarray, text: str) -> np.ndarray:
    image = rgb.copy()
    cv2.putText(
        image,
        text,
        (18, image.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return image


def execute_parity(config: dict, *, run_id: str | None = None) -> Path:
    device = torch.device(str(config.get("device", "cuda")))
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("RT1 replay parity requires CUDA")
    paths = RouteAPaths.from_mapping(config["assets"])
    assets = RouteAAssets(paths)
    anchors = assets.load_anchors()
    indices = np.arange(len(anchors.xyz), dtype=np.int64)
    camera_names = list(config["data"]["cameras"])
    frame_index = int(config["data"]["frame_index"])
    frame = assets.load_frame(camera_names[0], frame_index)
    calibrations = {name: assets.load_camera(name) for name in camera_names}
    height, width = map(int, config["observation"]["image_size"])

    backend = CudaGaussianBackend.from_route_a(
        kinematics=TorchURDFKinematics(paths.urdf),
        anchors=anchors,
        anchor_indices=indices,
        link_names=assets.link_names,
        calibrations=calibrations,
        output_size=(height, width),
        observation_mode="rgb",
        device=device,
    )
    qpos = torch.as_tensor(frame.qpos, dtype=torch.float32, device=device)
    with torch.inference_mode():
        candidate = {name: item.values for name, item in backend.render(qpos).items()}
        inherited = render_legacy_rgb(
            robosplat_root=Path(config["robosplat_root"]),
            urdf_path=paths.urdf,
            anchors=anchors,
            anchor_indices=indices,
            link_names=assets.link_names,
            qpos=frame.qpos,
            calibrations=calibrations,
            output_size=(height, width),
            device=device,
        )

    rows = []
    trace = []
    per_camera = {}
    for name in camera_names:
        new = candidate[name].detach().cpu().numpy()
        old = inherited[name].detach().cpu().numpy()
        difference = np.abs(new - old)
        mse = float(np.mean((new - old) ** 2))
        per_camera[name] = {
            "mean_absolute_difference": float(difference.mean()),
            "max_absolute_difference": float(difference.max()),
            "psnr_db": float(-10.0 * math.log10(max(mse, 1e-12))),
        }
        trace.append({"camera": name, **per_camera[name]})
        real = assets.load_frame(name, frame_index).rgb
        panels = [
            _label(real, f"{name}: real context"),
            _label((old * 255.0).astype(np.uint8), f"{name}: inherited RoboSplat"),
            _label((new * 255.0).astype(np.uint8), f"{name}: KineSync native"),
            _label((np.clip(difference * 25.0, 0.0, 1.0) * 255.0).astype(np.uint8), f"{name}: |difference| x25"),
        ]
        rows.append(np.concatenate(panels, axis=1))
    panel = np.concatenate(rows, axis=0)
    header = np.full((120, panel.shape[1], 3), (20, 24, 29), dtype=np.uint8)
    cv2.putText(
        header,
        "KineSync-GS native RGB replay parity",
        (48, 76),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.35,
        (245, 245, 245),
        3,
        cv2.LINE_AA,
    )
    panel = np.concatenate((header, panel), axis=0)
    metrics = {
        "target_provenance": "recorded_state_replay",
        "frame_index": frame_index,
        "camera_names": camera_names,
        "gaussian_count": int(len(indices)),
        "output_size": [height, width],
        "per_camera": per_camera,
        "mean_absolute_difference": float(
            np.mean([value["mean_absolute_difference"] for value in per_camera.values()])
        ),
        "minimum_psnr_db": float(
            min(value["psnr_db"] for value in per_camera.values())
        ),
    }
    run = RunArtifacts.create(
        root=config.get("runs_root", "runs"),
        experiment=str(config.get("experiment", "rt1_replay_parity")),
        run_id=run_id,
        config=config,
        assets=paths.as_assets(),
        target_provenance="recorded_state_replay",
        observation_backend="native_cuda_rgb_parity",
    )
    run.write_metrics(metrics)
    run.write_trace(trace)
    output = run.path / "images" / "native_vs_inherited_rgb_parity.png"
    cv2.imwrite(str(output), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 6])
    (run.path / "parity_summary.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return run.path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    print(execute_parity(load_config(args.config), run_id=args.run_id))


if __name__ == "__main__":
    main()
