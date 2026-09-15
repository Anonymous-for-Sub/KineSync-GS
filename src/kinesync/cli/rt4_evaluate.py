"""Evaluate RT4-W temporal correction on frozen Route-A validation windows."""

from __future__ import annotations

import argparse
from collections import Counter
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
from kinesync.cli.rt3_evaluate import (
    _exact_mcnemar,
    _load_guard,
    _outcome,
    _validate_optimizer,
)
from kinesync.config import load_config
from kinesync.data.route_a import RouteAAssets, RouteAPaths
from kinesync.experiments.rt4_matrix import load_rt4_cases, rt4_case_fingerprint
from kinesync.factorization import FrozenCameraObservationBackend
from kinesync.geometry.urdf import TorchURDFKinematics
from kinesync.guard.candidate import run_guard_candidate
from kinesync.guard.decision import decide_update
from kinesync.guard.schema import GuardDecision, GuardThresholds
from kinesync.guard.temporal_candidate import run_temporal_guard_candidate
from kinesync.observation.cuda_gaussian import CudaGaussianBackend
from kinesync.observation.real_route_a import prepare_real_targets
from kinesync.runs.artifacts import RunArtifacts
from kinesync.visualization.temporal import (
    write_temporal_comparison,
    write_temporal_video,
)


_METHODS = (
    "rt2_single",
    "rt3_single_guarded",
    "rt4_temporal",
    "rt4_temporal_guarded",
)


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    controlled = [row for row in rows if row["trial_type"] == "controlled_offset"]
    zeros = [row for row in rows if row["trial_type"] == "zero_injection_control"]
    one_degree = [
        row
        for row in controlled
        if math.isclose(float(row["magnitude_deg"]), 1.0, abs_tol=0.1)
    ]
    committed = [row for row in rows if row["candidate_committed"]]
    return {
        "case_count": len(rows),
        "controlled_success_rate": (
            mean(bool(row["recovery_success"]) for row in controlled)
            if controlled
            else None
        ),
        "one_degree_success_rate": (
            mean(bool(row["recovery_success"]) for row in one_degree)
            if one_degree
            else None
        ),
        "zero_control_stability": (
            mean(bool(row["recovery_success"]) for row in zeros) if zeros else None
        ),
        "commit_coverage": (
            mean(bool(row["candidate_committed"]) for row in rows) if rows else None
        ),
        "mean_state_error_reduction": (
            mean(float(row["state_error_reduction"]) for row in controlled)
            if controlled
            else None
        ),
        "mean_mask_iou_gain": mean(float(row["mask_iou_gain"]) for row in rows),
        "mean_boundary_f1_gain": mean(
            float(row["boundary_f1_gain"]) for row in rows
        ),
        "committed_mean_mask_iou_gain": (
            mean(float(row["mask_iou_gain"]) for row in committed)
            if committed
            else None
        ),
        "committed_mean_boundary_f1_gain": (
            mean(float(row["boundary_f1_gain"]) for row in committed)
            if committed
            else None
        ),
    }


def _source_matrix_path(matrix_path: Path) -> Path:
    payload = load_config(matrix_path)
    source = Path(str(payload["source_matrix"]))
    return source.resolve() if source.is_absolute() else (matrix_path.parent / source).resolve()


def execute_rt4_evaluate(
    config: Mapping[str, Any],
    *,
    factors_path: str | Path,
    guard_path: str | Path,
    matrix_path: str | Path,
    run_id: str | None = None,
) -> Path:
    device = torch.device(str(config.get("device", "cuda")))
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("RT4 evaluation requires the native CUDA rasterizer")
    factor_path, factors = _load_factors(factors_path)
    _validate_factor_bounds(factors)
    factor_sha = _sha256(factor_path)
    guard_path, guard = _load_guard(guard_path, factor_sha256=factor_sha)
    guard_sha = _sha256(guard_path)
    evaluation = config["evaluation"]
    _validate_optimizer(guard, evaluation)
    thresholds = GuardThresholds(**guard["thresholds"])
    matrix_path = Path(matrix_path).expanduser().resolve()
    source_matrix = _source_matrix_path(matrix_path)

    paths = RouteAPaths.from_mapping(config["assets"])
    assets = RouteAAssets(paths)
    dataset = _real_dataset(config)
    cases = load_rt4_cases(
        matrix_path, dataset, factors, guard["source_state_ids"]
    )
    if not cases:
        raise ValueError("RT4 evaluation matrix produced no cases")
    derived_sha = rt4_case_fingerprint(cases)

    fk = TorchURDFKinematics(paths.urdf)
    factor_joint_names = [str(value) for value in factors["joint_names"]]
    if factor_joint_names != fk.active_joint_names[: len(factor_joint_names)]:
        raise ValueError("Frozen joint factors do not match the PiPER URDF")
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
    height, width = map(int, config["observation"]["image_size"])
    boundary_weight = float(config["observation"]["boundary_weight"])
    seed = int(config.get("seed", 41))
    indices = _anchor_indices(
        assets, config["data"]["anchors_per_link"], seed=seed
    )
    renderer_cache: dict[tuple[str, ...], CudaGaussianBackend] = {}

    def renderer_for(names: list[str]) -> CudaGaussianBackend:
        key = tuple(names)
        if key not in renderer_cache:
            renderer_cache[key] = CudaGaussianBackend.from_route_a(
                kinematics=fk,
                anchors=assets.load_anchors(),
                anchor_indices=indices,
                link_names=assets.link_names,
                calibrations={
                    name: assets.load_camera(name) for name in names
                },
                output_size=(height, width),
                observation_mode="alpha",
                device=device,
            )
        return renderer_cache[key]

    rows: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    media_contexts: list[dict[str, Any]] = []
    for case in cases:
        center = str(case["center_state_id"])
        state_ids = [str(value) for value in case["context_state_ids"]]
        camera_names = [str(value) for value in case["camera_names"]]
        observations = {
            state_id: dataset.load_state(state_id, cameras=camera_names)
            for state_id in state_ids
        }
        targets = {
            state_id: prepare_real_targets(
                observations[state_id],
                output_size=(height, width),
                device=device,
            )
            for state_id in state_ids
        }
        references = {
            state_id: torch.as_tensor(
                np.mean(
                    np.stack(
                        [value.qpos for value in observations[state_id].values()]
                    ),
                    axis=0,
                ),
                dtype=torch.float32,
                device=device,
            )
            + shared_joint
            for state_id in state_ids
        }
        selected_joints = list(case["offsets_rad"])
        selected_indices = [
            fk.active_joint_names.index(name) for name in selected_joints
        ]
        injected = torch.zeros_like(references[center])
        for name, value in case["offsets_rad"].items():
            injected[fk.active_joint_names.index(name)] = float(value)
        measured = {
            state_id: references[state_id] + injected for state_id in state_ids
        }

        renderer = renderer_for(camera_names)
        joint_twists = {name: camera_factors[name] for name in camera_names}
        joint_visual = FrozenCameraObservationBackend(
            renderer,
            camera_twists=joint_twists,
            boundary_weight=boundary_weight,
        )
        camera_visual = {}
        temporal_camera_targets = {}
        single_camera_targets = {}
        single_camera_backends = {}
        for name in camera_names:
            camera_renderer = renderer_for([name])
            twists = {name: camera_factors[name]}
            visual = FrozenCameraObservationBackend(
                camera_renderer,
                camera_twists=twists,
                boundary_weight=boundary_weight,
            )
            camera_visual[name] = visual
            temporal_camera_targets[name] = {
                state_id: {name: targets[state_id][name]}
                for state_id in state_ids
            }
            single_camera_targets[name] = {name: targets[center][name]}
            single_camera_backends[name] = _prior_backend(
                visual,
                camera_renderer,
                single_camera_targets[name],
                camera_twists=twists,
                measured_qpos=measured[center],
                selected_indices=selected_indices,
                evaluation=evaluation,
                boundary_weight=boundary_weight,
            )

        single_backend = _prior_backend(
            joint_visual,
            renderer,
            targets[center],
            camera_twists=joint_twists,
            measured_qpos=measured[center],
            selected_indices=selected_indices,
            evaluation=evaluation,
            boundary_weight=boundary_weight,
        )
        single = run_guard_candidate(
            joint_backend=single_backend,
            joint_visual_backend=joint_visual,
            joint_target=targets[center],
            camera_backends=single_camera_backends,
            camera_visual_backends=camera_visual,
            camera_targets=single_camera_targets,
            measured_qpos=measured[center],
            reference_qpos=references[center],
            joint_names=fk.active_joint_names,
            selected_joints=selected_joints,
            steps=int(evaluation["steps"]),
            learning_rate=float(evaluation["learning_rate"]),
            offset_bound=float(evaluation["offset_bound_rad"]),
            seed=int(case["seed"]),
            acceptance=config.get("acceptance", {}),
        )
        temporal = run_temporal_guard_candidate(
            state_ids=state_ids,
            center_state_id=center,
            joint_visual_backend=joint_visual,
            joint_targets=targets,
            camera_visual_backends=camera_visual,
            camera_targets=temporal_camera_targets,
            measured_qpos=measured,
            reference_qpos=references,
            joint_names=fk.active_joint_names,
            selected_joints=selected_joints,
            steps=int(evaluation["steps"]),
            learning_rate=float(evaluation["learning_rate"]),
            offset_bound=float(evaluation["offset_bound_rad"]),
            seed=int(case["seed"]),
            prior_weight=float(evaluation.get("prior_weight", 5.0)),
            prior_kind=str(evaluation.get("prior_kind", "adaptive")),
            prior_delta_rad=float(evaluation.get("prior_delta_rad", 0.02)),
            minimum_gradient=float(evaluation.get("minimum_gradient", 0.05)),
            minimum_relative=float(evaluation.get("minimum_relative", 0.10)),
            acceptance=config.get("acceptance", {}),
        )
        single_decision = decide_update(single.evidence, thresholds)
        temporal_decision = decide_update(temporal.evidence, thresholds)
        candidates = {
            "rt2_single": (
                single.joint_result.corrected_qpos,
                GuardDecision(True, "unprotected_baseline", 0.0),
                single.evidence,
            ),
            "rt3_single_guarded": (
                single.joint_result.corrected_qpos,
                single_decision,
                single.evidence,
            ),
            "rt4_temporal": (
                temporal.joint_result.corrected_qpos,
                GuardDecision(True, "unprotected_baseline", 0.0),
                temporal.evidence,
            ),
            "rt4_temporal_guarded": (
                temporal.joint_result.corrected_qpos,
                temporal_decision,
                temporal.evidence,
            ),
        }
        case_rows = {}
        for method in _METHODS:
            candidate, decision, evidence = candidates[method]
            final = candidate if decision.accepted else measured[center]
            outcome, before, after = _outcome(
                backend=joint_visual,
                target=targets[center],
                measured=measured[center],
                final=final,
                reference=references[center],
                selected_indices=selected_indices,
                acceptance=config.get("acceptance", {}),
            )
            row = {
                "trial_id": case["source_trial_id"],
                "center_state_id": center,
                "context_state_ids": "+".join(state_ids),
                "window_size": len(state_ids),
                "split": "validation",
                "case_role": case["case_role"],
                "camera_mode": "+".join(camera_names),
                "offset_name": case["offset_name"],
                "selected_joints": "+".join(selected_joints),
                "magnitude_deg": case["magnitude_deg"],
                "method": method,
                "trial_type": outcome["trial_type"],
                "candidate_committed": decision.accepted,
                "decision_reason": decision.reason,
                "minimum_margin": decision.minimum_margin,
                "state_error_reduction": outcome["state_error_reduction"],
                "natural_state_correction_rad": outcome[
                    "natural_state_correction_rad"
                ],
                "mask_iou_before": before["mean_mask_iou"],
                "mask_iou_after": after["mean_mask_iou"],
                "mask_iou_gain": (
                    after["mean_mask_iou"] - before["mean_mask_iou"]
                ),
                "boundary_f1_before": before["mean_boundary_f1"],
                "boundary_f1_after": after["mean_boundary_f1"],
                "boundary_f1_gain": (
                    after["mean_boundary_f1"]
                    - before["mean_boundary_f1"]
                ),
                "recovery_success": bool(outcome["recovery_success"]),
                "evidence_finite": evidence.finite,
                "view_count": evidence.view_count,
                "visual_gain_ratio": evidence.visual_gain_ratio,
                "gradient_cosine": evidence.gradient_cosine,
                "correction_cosine": evidence.correction_cosine,
                "relative_correction_disagreement": (
                    evidence.relative_correction_disagreement
                ),
                "candidate_bound_fraction": evidence.candidate_bound_fraction,
            }
            rows.append(row)
            case_rows[method] = row

        trace_sources = {
            "single_joint": single.joint_result,
            **{
                f"single_{name}": result
                for name, result in single.camera_results.items()
            },
            "temporal_joint": temporal.joint_result,
            **{
                f"temporal_{name}": result
                for name, result in temporal.camera_results.items()
            },
        }
        for source, result in trace_sources.items():
            traces.extend(
                {
                    "trial_id": case["source_trial_id"],
                    "candidate_source": source,
                    **trace,
                }
                for trace in result.trace
            )
        media_contexts.append(
            {
                "case_role": case["case_role"],
                "center_state_id": center,
                "context_state_ids": state_ids,
                "context_observations": observations,
                "context_targets": targets,
                "backend": joint_visual,
                "measured": measured[center],
                "single": single.joint_result.corrected_qpos,
                "temporal": temporal.joint_result.corrected_qpos,
                "final": (
                    temporal.joint_result.corrected_qpos
                    if temporal_decision.accepted
                    else measured[center]
                ),
                "decision": temporal_decision,
                "evidence": temporal.evidence,
                "single_reduction": case_rows["rt2_single"][
                    "state_error_reduction"
                ],
                "temporal_reduction": case_rows["rt4_temporal"][
                    "state_error_reduction"
                ],
            }
        )

    summaries = {
        method: _summary([row for row in rows if row["method"] == method])
        for method in _METHODS
    }
    by_method_trial = {
        method: {
            row["trial_id"]: row for row in rows if row["method"] == method
        }
        for method in _METHODS
    }
    controlled_ids = [
        row["trial_id"]
        for row in rows
        if row["method"] == "rt4_temporal"
        and row["trial_type"] == "controlled_offset"
    ]
    paired_stats = _exact_mcnemar(
        [
            bool(by_method_trial["rt4_temporal"][trial]["recovery_success"])
            for trial in controlled_ids
        ],
        [
            bool(by_method_trial["rt2_single"][trial]["recovery_success"])
            for trial in controlled_ids
        ],
    )
    distal_rows = [
        row
        for row in rows
        if row["method"] == "rt4_temporal"
        and row["trial_type"] == "controlled_offset"
        and set(str(row["selected_joints"]).split("+"))
        & {"joint4", "joint5", "joint6"}
    ]
    distal_improvements = [
        float(row["state_error_reduction"])
        - float(
            by_method_trial["rt2_single"][row["trial_id"]][
                "state_error_reduction"
            ]
        )
        for row in distal_rows
    ]
    temporal_guarded_rows = [
        row for row in rows if row["method"] == "rt4_temporal_guarded"
    ]
    committed_temporal = [
        row for row in temporal_guarded_rows if row["candidate_committed"]
    ]
    controlled_rows = [
        row for row in rows if row["trial_type"] == "controlled_offset"
    ]
    magnitude_values = sorted(
        {float(row["magnitude_deg"]) for row in controlled_rows}
    )
    controlled_by_magnitude = {
        f"{value:g}": {
            method: _summary(
                [
                    row
                    for row in controlled_rows
                    if row["method"] == method
                    and math.isclose(
                        float(row["magnitude_deg"]), value, abs_tol=1e-9
                    )
                ]
            )
            for method in _METHODS
        }
        for value in magnitude_values
    }
    joint_values = sorted(
        {
            joint
            for row in controlled_rows
            for joint in str(row["selected_joints"]).split("+")
        }
    )
    controlled_by_joint = {
        joint: {
            method: _summary(
                [
                    row
                    for row in controlled_rows
                    if row["method"] == method
                    and joint in str(row["selected_joints"]).split("+")
                ]
            )
            for method in _METHODS
        }
        for joint in joint_values
    }
    gates = {
        "temporal_controlled_success_exceeds_single": bool(
            summaries["rt4_temporal"]["controlled_success_rate"]
            > summaries["rt2_single"]["controlled_success_rate"]
        ),
        "temporal_one_degree_success_at_least_60pct": bool(
            summaries["rt4_temporal"]["one_degree_success_rate"] is not None
            and summaries["rt4_temporal"]["one_degree_success_rate"] >= 0.60
        ),
        "guarded_zero_stability_at_least_80pct": bool(
            summaries["rt4_temporal_guarded"]["zero_control_stability"]
            is not None
            and summaries["rt4_temporal_guarded"]["zero_control_stability"]
            >= 0.80
        ),
        "positive_distal_reduction_improvement": bool(
            distal_improvements and mean(distal_improvements) > 0
        ),
        "positive_committed_visual_gains": bool(
            committed_temporal
            and mean(float(row["mask_iou_gain"]) for row in committed_temporal)
            > 0
            and mean(
                float(row["boundary_f1_gain"]) for row in committed_temporal
            )
            > 0
        ),
        "all_evidence_finite": all(
            bool(row["evidence_finite"]) for row in rows
        ),
    }
    metrics = {
        "evaluation_split": "validation",
        "case_count": len(cases),
        "method_row_count": len(rows),
        "window_size": 5,
        "center_state_ids": sorted(
            {str(case["center_state_id"]) for case in cases}
        ),
        "factor_sha256": factor_sha,
        "guard_sha256": guard_sha,
        "test_matrix_sha256": _sha256(matrix_path),
        "source_matrix_sha256": _sha256(source_matrix),
        "derived_matrix_sha256": derived_sha,
        "guard_source_state_ids": guard["source_state_ids"],
        "method_summaries": summaries,
        "controlled_by_magnitude_deg": controlled_by_magnitude,
        "controlled_by_joint": controlled_by_joint,
        "temporal_vs_single_controlled_mcnemar": paired_stats,
        "distal_case_count": len(distal_rows),
        "mean_distal_state_reduction_improvement": (
            mean(distal_improvements) if distal_improvements else None
        ),
        "temporal_guard_rejection_reasons": dict(
            Counter(row["decision_reason"] for row in temporal_guarded_rows)
        ),
        "acceptance_gates": gates,
        "all_acceptance_gates_passed": all(gates.values()),
    }

    real_source = config["real_observations"]
    run = RunArtifacts.create(
        root=config.get("runs_root", "runs"),
        experiment="rt4_piper_temporal_evaluation",
        run_id=run_id,
        config=config,
        assets={
            **paths.as_assets(),
            "factors": factor_path,
            "guard": guard_path,
            "rt4_matrix": matrix_path,
            "source_rt3_matrix": source_matrix,
            **{
                f"records_{camera}": real_source["records"][camera]
                for camera in dataset.cameras
            },
        },
        target_provenance="real_observation_guard_evaluation",
        observation_backend="native_cuda_frozen_factor_temporal_guard_test",
    )
    run.write_metrics(metrics)
    run.write_trace(traces)
    with (run.path / "results.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (run.path / "derived_cases.json").write_text(
        json.dumps(cases, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    media = max(
        media_contexts,
        key=lambda item: float(item["temporal_reduction"])
        - float(item["single_reduction"]),
    )
    write_temporal_comparison(
        run.path / "images" / "temporal_comparison.png", context=media
    )
    write_temporal_video(
        run.path / "videos" / "temporal_correction.mp4",
        context=media,
        frame_count=int(evaluation.get("video_frames", 36)),
        fps=int(evaluation.get("video_fps", 12)),
    )
    zero_contexts = [
        context
        for context in media_contexts
        if context["case_role"] == "paired_zero"
        and not context["decision"].accepted
    ]
    if zero_contexts:
        write_temporal_comparison(
            run.path / "images" / "temporal_zero_rollback.png",
            context=zero_contexts[0],
        )
    return run.path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--factors", type=Path, required=True)
    parser.add_argument("--guard", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--run-id")
    arguments = parser.parse_args()
    print(
        execute_rt4_evaluate(
            load_config(arguments.config),
            factors_path=arguments.factors,
            guard_path=arguments.guard,
            matrix_path=arguments.matrix,
            run_id=arguments.run_id,
        )
    )


if __name__ == "__main__":
    main()
