"""Run the matched RT5-O resolution and TSDF observation audit."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any, Mapping

import numpy as np
import torch

from kinesync.cli.rt1_real_observation import _anchor_indices, _real_dataset
from kinesync.cli.rt2_evaluate import _load_factors, _validate_factor_bounds
from kinesync.cli.rt3_calibrate import _prior_backend, _sha256
from kinesync.cli.rt3_evaluate import _exact_mcnemar, _outcome
from kinesync.config import load_config
from kinesync.correction.joint_offset import JointOffsetRecovery
from kinesync.data.route_a import RouteAAssets, RouteAPaths
from kinesync.experiments.rt5_matrix import load_rt5_cases
from kinesync.factorization import FrozenCameraObservationBackend
from kinesync.geometry.urdf import TorchURDFKinematics
from kinesync.observation.cuda_gaussian import CudaGaussianBackend
from kinesync.observation.real_route_a import prepare_real_targets
from kinesync.runs.artifacts import RunArtifacts
from kinesync.visualization.observation_audit import (
    write_observation_audit,
    write_observation_audit_video,
)


_METHODS = ("standard_120", "standard_240", "tsdf_240")


def _source_path(matrix_path: Path) -> Path:
    payload = load_config(matrix_path)
    value = Path(str(payload["source_matrix"]))
    return value.resolve() if value.is_absolute() else (matrix_path.parent / value).resolve()


def _method_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    controlled = [row for row in rows if row["audit_role"] == "paired_one_degree"]
    zeros = [row for row in rows if row["audit_role"] == "paired_zero"]
    return {
        "case_count": len(rows),
        "one_degree_success_rate": (
            mean(bool(row["recovery_success"]) for row in controlled)
            if controlled
            else None
        ),
        "mean_one_degree_state_reduction": (
            mean(float(row["state_error_reduction"]) for row in controlled)
            if controlled
            else None
        ),
        "zero_control_stability": (
            mean(bool(row["recovery_success"]) for row in zeros) if zeros else None
        ),
        "mean_initial_selected_gradient_norm": mean(
            float(row["initial_selected_gradient_norm"]) for row in rows
        ),
        "mean_mask_iou_gain": mean(float(row["mask_iou_gain"]) for row in rows),
        "mean_boundary_f1_gain": mean(
            float(row["boundary_f1_gain"]) for row in rows
        ),
    }


def execute_rt5_observation_audit(
    config: Mapping[str, Any],
    *,
    factors_path: str | Path,
    matrix_path: str | Path,
    run_id: str | None = None,
) -> Path:
    device = torch.device(str(config.get("device", "cuda")))
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("RT5 observation audit requires the native CUDA rasterizer")
    factor_path, factors = _load_factors(factors_path)
    _validate_factor_bounds(factors)
    factor_sha = _sha256(factor_path)
    matrix_path = Path(matrix_path).expanduser().resolve()
    matrix = load_config(matrix_path)
    methods = matrix["methods"]
    if tuple(methods) != _METHODS:
        raise ValueError(f"RT5 methods must be ordered as {_METHODS}")
    source_matrix = _source_path(matrix_path)

    paths = RouteAPaths.from_mapping(config["assets"])
    assets = RouteAAssets(paths)
    dataset = _real_dataset(config)
    cases = load_rt5_cases(matrix_path, dataset, factors)
    fk = TorchURDFKinematics(paths.urdf)
    factor_joint_names = [str(value) for value in factors["joint_names"]]
    if factor_joint_names != fk.active_joint_names[: len(factor_joint_names)]:
        raise ValueError("Frozen factors do not match the PiPER URDF")
    shared_joint = torch.zeros(
        len(fk.active_joint_names), dtype=torch.float32, device=device
    )
    shared_joint[: len(factor_joint_names)] = torch.as_tensor(
        factors["joint_zero_rad"], dtype=torch.float32, device=device
    )
    camera_factors = {
        name: torch.as_tensor(value, dtype=torch.float32, device=device)
        for name, value in factors["camera_twists"].items()
    }
    boundary_weight = float(config["observation"]["boundary_weight"])
    evaluation = config["evaluation"]
    indices = _anchor_indices(
        assets,
        config["data"]["anchors_per_link"],
        seed=int(config.get("seed", 41)),
    )
    renderer_cache: dict[tuple[int, int], CudaGaussianBackend] = {}

    def renderer_for(size: tuple[int, int]) -> CudaGaussianBackend:
        if size not in renderer_cache:
            renderer_cache[size] = CudaGaussianBackend.from_route_a(
                kinematics=fk,
                anchors=assets.load_anchors(),
                anchor_indices=indices,
                link_names=assets.link_names,
                calibrations={
                    name: assets.load_camera(name) for name in ("head", "extra")
                },
                output_size=size,
                observation_mode="alpha",
                device=device,
            )
        return renderer_cache[size]

    rows: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    media_contexts: list[dict[str, Any]] = []
    for case in cases:
        state_id = str(case["state_id"])
        camera_names = [str(value) for value in case["camera_names"]]
        if camera_names != ["head", "extra"]:
            raise ValueError("RT5 observation audit requires paired cameras")
        observations = dataset.load_state(state_id, cameras=camera_names)
        reference = torch.as_tensor(
            np.mean(np.stack([value.qpos for value in observations.values()]), axis=0),
            dtype=torch.float32,
            device=device,
        )
        calibrated_reference = reference + shared_joint
        selected_joints = list(case["offsets_rad"])
        selected_indices = [
            fk.active_joint_names.index(name) for name in selected_joints
        ]
        injected = torch.zeros_like(calibrated_reference)
        for name, value in case["offsets_rad"].items():
            injected[fk.active_joint_names.index(name)] = float(value)
        measured = calibrated_reference + injected
        finals: dict[str, torch.Tensor] = {}
        case_rows: dict[str, dict[str, Any]] = {}
        high_backend = None
        high_targets = None

        for method in _METHODS:
            method_config = methods[method]
            size = tuple(int(value) for value in method_config["image_size"])
            if len(size) != 2 or min(size) <= 0:
                raise ValueError(f"Invalid RT5 image size for {method}")
            renderer = renderer_for(size)
            targets = prepare_real_targets(
                observations, output_size=size, device=device
            )
            sdf_weight = float(method_config.get("sdf_weight", 0.0))
            sdf_radii = tuple(
                float(value)
                for value in method_config.get("sdf_radii_px", (4.0, 12.0))
            )
            visual = FrozenCameraObservationBackend(
                renderer,
                camera_twists=camera_factors,
                boundary_weight=boundary_weight,
                sdf_weight=sdf_weight,
                sdf_radii=sdf_radii,
            )
            probe = measured.detach().clone().requires_grad_(True)
            probe_loss, _ = visual.loss(probe, targets)
            probe_loss.backward()
            if probe.grad is None or not bool(torch.isfinite(probe.grad).all()):
                raise FloatingPointError("RT5 initial visual gradient is nonfinite")
            gradient_norm = float(
                torch.linalg.vector_norm(probe.grad[selected_indices]).detach()
            )
            method_evaluation = dict(evaluation)
            method_evaluation["sdf_weight"] = sdf_weight
            method_evaluation["sdf_radii_px"] = list(sdf_radii)
            backend = _prior_backend(
                visual,
                renderer,
                targets,
                camera_twists=camera_factors,
                measured_qpos=measured,
                selected_indices=selected_indices,
                evaluation=method_evaluation,
                boundary_weight=boundary_weight,
            )
            result = JointOffsetRecovery(
                backend,
                joint_names=fk.active_joint_names,
                selected_joints=selected_joints,
                steps=int(evaluation["steps"]),
                learning_rate=float(evaluation["learning_rate"]),
                offset_bound=float(evaluation["offset_bound_rad"]),
                seed=int(case["seed"]),
            ).recover(
                measured,
                targets,
                reference_qpos=calibrated_reference,
            )
            outcome, before, after = _outcome(
                backend=visual,
                target=targets,
                measured=measured,
                final=result.corrected_qpos,
                reference=calibrated_reference,
                selected_indices=selected_indices,
                acceptance=config.get("acceptance", {}),
            )
            row = {
                "trial_id": case["trial_id"],
                "state_id": state_id,
                "split": "validation",
                "audit_role": case["audit_role"],
                "offset_name": case["offset_name"],
                "selected_joints": "+".join(selected_joints),
                "magnitude_deg": case["magnitude_deg"],
                "method": method,
                "image_height": size[0],
                "image_width": size[1],
                "sdf_weight": sdf_weight,
                "initial_selected_gradient_norm": gradient_norm,
                "trial_type": outcome["trial_type"],
                "state_error_reduction": outcome["state_error_reduction"],
                "natural_state_correction_rad": outcome[
                    "natural_state_correction_rad"
                ],
                "mask_iou_before": before["mean_mask_iou"],
                "mask_iou_after": after["mean_mask_iou"],
                "mask_iou_gain": after["mean_mask_iou"] - before["mean_mask_iou"],
                "boundary_f1_before": before["mean_boundary_f1"],
                "boundary_f1_after": after["mean_boundary_f1"],
                "boundary_f1_gain": (
                    after["mean_boundary_f1"] - before["mean_boundary_f1"]
                ),
                "recovery_success": bool(outcome["recovery_success"]),
            }
            rows.append(row)
            case_rows[method] = row
            finals[method] = result.corrected_qpos
            traces.extend(
                {
                    "trial_id": case["trial_id"],
                    "method": method,
                    **trace,
                }
                for trace in result.trace
            )
            if method == "tsdf_240":
                high_backend = visual
                high_targets = targets

        media_contexts.append(
            {
                "state_id": state_id,
                "audit_role": case["audit_role"],
                "observations": observations,
                "targets": high_targets,
                "backend": high_backend,
                "measured": measured,
                "finals": finals,
                "rows": case_rows,
            }
        )

    summaries = {
        method: _method_summary([row for row in rows if row["method"] == method])
        for method in _METHODS
    }
    by_method = {
        method: {
            row["trial_id"]: row for row in rows if row["method"] == method
        }
        for method in _METHODS
    }
    controlled_ids = [
        row["trial_id"]
        for row in rows
        if row["method"] == "tsdf_240"
        and row["audit_role"] == "paired_one_degree"
    ]
    paired = _exact_mcnemar(
        [
            bool(by_method["tsdf_240"][trial]["recovery_success"])
            for trial in controlled_ids
        ],
        [
            bool(by_method["standard_240"][trial]["recovery_success"])
            for trial in controlled_ids
        ],
    )
    distal_ids = [
        trial
        for trial in controlled_ids
        if set(by_method["tsdf_240"][trial]["selected_joints"].split("+"))
        & {"joint4", "joint5", "joint6"}
    ]
    distal_improvement = [
        float(by_method["tsdf_240"][trial]["state_error_reduction"])
        - float(by_method["standard_120"][trial]["state_error_reduction"])
        for trial in distal_ids
    ]
    primary = summaries["tsdf_240"]
    gates = {
        "all_values_finite": all(
            math.isfinite(float(row["initial_selected_gradient_norm"]))
            and math.isfinite(float(row["mask_iou_gain"]))
            and math.isfinite(float(row["boundary_f1_gain"]))
            for row in rows
        ),
        "tsdf_success_exceeds_both_standards": bool(
            primary["one_degree_success_rate"]
            > max(
                summaries["standard_120"]["one_degree_success_rate"],
                summaries["standard_240"]["one_degree_success_rate"],
            )
        ),
        "tsdf_reduction_improves_15pp_over_standard120": bool(
            primary["mean_one_degree_state_reduction"]
            - summaries["standard_120"]["mean_one_degree_state_reduction"]
            >= 0.15
        ),
        "positive_distal_improvement": bool(
            distal_improvement and mean(distal_improvement) > 0
        ),
        "zero_stability_at_least_80pct": bool(
            primary["zero_control_stability"] is not None
            and primary["zero_control_stability"] >= 0.80
        ),
        "positive_visual_gains": bool(
            primary["mean_mask_iou_gain"] > 0
            and primary["mean_boundary_f1_gain"] > 0
        ),
    }
    metrics = {
        "case_count": len(cases),
        "method_row_count": len(rows),
        "factor_sha256": factor_sha,
        "matrix_sha256": _sha256(matrix_path),
        "source_matrix_sha256": _sha256(source_matrix),
        "method_summaries": summaries,
        "tsdf_vs_standard240_mcnemar": paired,
        "distal_case_count": len(distal_ids),
        "mean_distal_reduction_improvement_vs_standard120": (
            mean(distal_improvement) if distal_improvement else None
        ),
        "acceptance_gates": gates,
        "all_acceptance_gates_passed": all(gates.values()),
    }

    real_source = config["real_observations"]
    run = RunArtifacts.create(
        root=config.get("runs_root", "runs"),
        experiment="rt5_piper_observation_audit",
        run_id=run_id,
        config=config,
        assets={
            **paths.as_assets(),
            "factors": factor_path,
            "rt5_matrix": matrix_path,
            "source_rt3_matrix": source_matrix,
            **{
                f"records_{camera}": real_source["records"][camera]
                for camera in dataset.cameras
            },
        },
        target_provenance="real_observation_factorization_evaluation",
        observation_backend="native_cuda_matched_resolution_tsdf_audit",
    )
    run.write_metrics(metrics)
    run.write_trace(traces)
    with (run.path / "results.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (run.path / "audit_cases.json").write_text(
        json.dumps(cases, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    controlled_media = [
        context
        for context in media_contexts
        if context["audit_role"] == "paired_one_degree"
    ]
    candidates = controlled_media or media_contexts
    selected = max(
        candidates,
        key=lambda context: float(
            context["rows"]["tsdf_240"]["state_error_reduction"]
        )
        - float(context["rows"]["standard_240"]["state_error_reduction"]),
    )
    write_observation_audit(
        run.path / "images" / "observation_audit.png", context=selected
    )
    write_observation_audit_video(
        run.path / "videos" / "observation_audit.mp4",
        context=selected,
        frame_count=int(evaluation.get("video_frames", 36)),
        fps=int(evaluation.get("video_fps", 12)),
    )
    return run.path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--factors", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--run-id")
    arguments = parser.parse_args()
    print(
        execute_rt5_observation_audit(
            load_config(arguments.config),
            factors_path=arguments.factors,
            matrix_path=arguments.matrix,
            run_id=arguments.run_id,
        )
    )


if __name__ == "__main__":
    main()
