"""Evaluate a frozen RT3 guard on untouched Route-A validation states."""

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

from kinesync.cli.rt1_real_observation import _anchor_indices, _evaluate_trial, _real_dataset
from kinesync.cli.rt2_evaluate import _load_factors, _validate_factor_bounds
from kinesync.cli.rt3_calibrate import _prior_backend, _sha256
from kinesync.config import load_config
from kinesync.correction.joint_offset import JointOffsetRecovery
from kinesync.data.route_a import RouteAAssets, RouteAPaths
from kinesync.experiments.rt3_matrix import load_rt3_cases
from kinesync.factorization import FrozenCameraObservationBackend
from kinesync.geometry.urdf import TorchURDFKinematics
from kinesync.guard.candidate import run_guard_candidate
from kinesync.guard.decision import decide_update
from kinesync.guard.schema import GuardDecision, GuardThresholds
from kinesync.observation.cuda_gaussian import CudaGaussianBackend
from kinesync.observation.real_route_a import prepare_real_targets
from kinesync.runs.artifacts import RunArtifacts
from kinesync.visualization.guard import write_guard_comparison, write_guard_video


_METHODS = (
    "rt1o_unfactorized",
    "rt2f_unguarded",
    "rt3_gain_only",
    "rt3_gradient_only",
    "rt3_full",
)


def _load_guard(path: str | Path, *, factor_sha256: str) -> tuple[Path, dict[str, Any]]:
    guard_path = Path(path).expanduser().resolve()
    payload = json.loads(guard_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or payload.get("frozen") is not True:
        raise ValueError("RT3 evaluation requires a frozen schema-v1 guard")
    if payload.get("source_split") != "train":
        raise ValueError("RT3 guard must be calibrated on train states")
    if payload.get("factor_sha256") != factor_sha256:
        raise ValueError("RT3 guard factor hash does not match evaluation factors")
    GuardThresholds(**payload["thresholds"])
    return guard_path, payload


def _validate_optimizer(guard: Mapping[str, Any], evaluation: Mapping[str, Any]) -> None:
    expected = guard["optimizer"]
    numeric = {
        "steps": int(evaluation["steps"]),
        "learning_rate": float(evaluation["learning_rate"]),
        "offset_bound_rad": float(evaluation["offset_bound_rad"]),
        "prior_weight": float(evaluation.get("prior_weight", 5.0)),
    }
    for name, value in numeric.items():
        if not math.isclose(float(expected[name]), float(value), rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"RT3 guard optimizer mismatch: {name}")
    if str(expected["prior_kind"]) != str(evaluation.get("prior_kind", "adaptive")):
        raise ValueError("RT3 guard optimizer mismatch: prior_kind")


def _gain_decision(evidence, thresholds: GuardThresholds) -> GuardDecision:
    if not evidence.finite:
        return GuardDecision(False, "nonfinite", float("-inf"))
    margin = evidence.visual_gain_ratio - thresholds.min_visual_gain_ratio
    return GuardDecision(margin >= 0, "accepted" if margin >= 0 else "visual_gain", margin)


def _gradient_decision(evidence, thresholds: GuardThresholds) -> GuardDecision:
    if not evidence.finite:
        return GuardDecision(False, "nonfinite", float("-inf"))
    if evidence.view_count < 2:
        return GuardDecision(False, "insufficient_views", -1.0)
    margin = evidence.gradient_cosine - thresholds.min_gradient_cosine
    return GuardDecision(
        margin >= 0,
        "accepted" if margin >= 0 else "gradient_agreement",
        margin,
    )


def _outcome(
    *,
    backend,
    target,
    measured: torch.Tensor,
    final: torch.Tensor,
    reference: torch.Tensor,
    selected_indices: list[int],
    acceptance: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, float], dict[str, float]]:
    before = backend.metrics(measured, target)
    after = backend.metrics(final, target)
    initial_mae = float((measured[selected_indices] - reference[selected_indices]).abs().mean())
    final_mae = float((final[selected_indices] - reference[selected_indices]).abs().mean())
    applied_mae = float((final[selected_indices] - measured[selected_indices]).abs().mean())
    evaluation = _evaluate_trial(
        initial_selected_mae=initial_mae,
        final_selected_mae=final_mae,
        applied_correction_mae=applied_mae,
        before_metrics=before,
        after_metrics=after,
        acceptance=acceptance,
    )
    return evaluation, before, after


def _exact_mcnemar(first: list[bool], second: list[bool]) -> dict[str, Any]:
    wins = sum(a and not b for a, b in zip(first, second, strict=True))
    losses = sum(b and not a for a, b in zip(first, second, strict=True))
    discordant = wins + losses
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(math.comb(discordant, value) for value in range(min(wins, losses) + 1))
        p_value = min(1.0, 2.0 * tail / (2**discordant))
    return {"wins": wins, "losses": losses, "discordant": discordant, "p_value": p_value}


def execute_rt3_evaluate(
    config: Mapping[str, Any],
    *,
    factors_path: str | Path,
    guard_path: str | Path,
    matrix_path: str | Path,
    run_id: str | None = None,
) -> Path:
    device = torch.device(str(config.get("device", "cuda")))
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("RT3 evaluation requires the native CUDA rasterizer")
    factor_path, factors = _load_factors(factors_path)
    _validate_factor_bounds(factors)
    factor_sha = _sha256(factor_path)
    guard_path, guard = _load_guard(guard_path, factor_sha256=factor_sha)
    guard_sha = _sha256(guard_path)
    evaluation = config["evaluation"]
    _validate_optimizer(guard, evaluation)
    thresholds = GuardThresholds(**guard["thresholds"])
    matrix_path = Path(matrix_path).expanduser().resolve()
    matrix_payload = load_config(matrix_path)
    old_states = matrix_payload["provenance"]["old_validation_state_ids"]

    paths = RouteAPaths.from_mapping(config["assets"])
    assets = RouteAAssets(paths)
    dataset = _real_dataset(config)
    train_states = set(dataset.state_ids(split="train", cameras=["head", "extra"]))
    if not set(guard["source_state_ids"]).issubset(train_states):
        raise ValueError("RT3 guard contains non-train source states")
    cases = load_rt3_cases(matrix_path, dataset, factors, old_states)
    if any(case["split"] != "validation" for case in cases):
        raise ValueError("RT3 evaluation accepts validation cases only")
    if {case["state_id"] for case in cases} & set(guard["source_state_ids"]):
        raise ValueError("RT3 guard calibration and evaluation states overlap")

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
    indices = _anchor_indices(assets, config["data"]["anchors_per_link"], seed=seed)
    renderer_cache: dict[tuple[str, ...], CudaGaussianBackend] = {}

    def renderer_for(names: list[str]) -> CudaGaussianBackend:
        key = tuple(names)
        if key not in renderer_cache:
            renderer_cache[key] = CudaGaussianBackend.from_route_a(
                kinematics=fk,
                anchors=assets.load_anchors(),
                anchor_indices=indices,
                link_names=assets.link_names,
                calibrations={name: assets.load_camera(name) for name in names},
                output_size=(height, width),
                observation_mode="alpha",
                device=device,
            )
        return renderer_cache[key]

    rows: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    media_contexts: list[dict[str, Any]] = []
    for case in cases:
        state_id = case["state_id"]
        camera_names = case["camera_names"]
        observations = dataset.load_state(state_id, cameras=camera_names)
        targets = prepare_real_targets(observations, output_size=(height, width), device=device)
        reference = torch.as_tensor(
            np.mean(np.stack([value.qpos for value in observations.values()]), axis=0),
            dtype=torch.float32,
            device=device,
        )
        calibrated_reference = reference + shared_joint
        selected_joints = list(case["offsets_rad"])
        selected_indices = [fk.active_joint_names.index(name) for name in selected_joints]
        injected = torch.zeros_like(reference)
        for name, value in case["offsets_rad"].items():
            injected[fk.active_joint_names.index(name)] = value
        measured = calibrated_reference + injected
        renderer = renderer_for(camera_names)
        joint_twists = {name: camera_factors[name] for name in camera_names}
        joint_visual = FrozenCameraObservationBackend(
            renderer, camera_twists=joint_twists, boundary_weight=boundary_weight
        )
        joint_backend = _prior_backend(
            joint_visual,
            renderer,
            targets,
            camera_twists=joint_twists,
            measured_qpos=measured,
            selected_indices=selected_indices,
            evaluation=evaluation,
            boundary_weight=boundary_weight,
        )
        camera_targets = {name: {name: targets[name]} for name in camera_names}
        camera_visual = {}
        camera_backends = {}
        for name in camera_names:
            camera_renderer = renderer_for([name])
            twists = {name: camera_factors[name]}
            visual = FrozenCameraObservationBackend(
                camera_renderer, camera_twists=twists, boundary_weight=boundary_weight
            )
            camera_visual[name] = visual
            camera_backends[name] = _prior_backend(
                visual,
                camera_renderer,
                camera_targets[name],
                camera_twists=twists,
                measured_qpos=measured,
                selected_indices=selected_indices,
                evaluation=evaluation,
                boundary_weight=boundary_weight,
            )
        rt2 = run_guard_candidate(
            joint_backend=joint_backend,
            joint_visual_backend=joint_visual,
            joint_target=targets,
            camera_backends=camera_backends,
            camera_visual_backends=camera_visual,
            camera_targets=camera_targets,
            measured_qpos=measured,
            reference_qpos=calibrated_reference,
            joint_names=fk.active_joint_names,
            selected_joints=selected_joints,
            steps=int(evaluation["steps"]),
            learning_rate=float(evaluation["learning_rate"]),
            offset_bound=float(evaluation["offset_bound_rad"]),
            seed=int(case["seed"]),
            acceptance=config.get("acceptance", {}),
        )

        zero_twists = {name: torch.zeros_like(camera_factors[name]) for name in camera_names}
        rt1_measured = reference + injected
        rt1_visual = FrozenCameraObservationBackend(
            renderer, camera_twists=zero_twists, boundary_weight=boundary_weight
        )
        rt1_backend = _prior_backend(
            rt1_visual,
            renderer,
            targets,
            camera_twists=zero_twists,
            measured_qpos=rt1_measured,
            selected_indices=selected_indices,
            evaluation=evaluation,
            boundary_weight=boundary_weight,
        )
        rt1_result = JointOffsetRecovery(
            rt1_backend,
            joint_names=fk.active_joint_names,
            selected_joints=selected_joints,
            steps=int(evaluation["steps"]),
            learning_rate=float(evaluation["learning_rate"]),
            offset_bound=float(evaluation["offset_bound_rad"]),
            seed=int(case["seed"]),
        ).recover(rt1_measured, targets, reference_qpos=reference)

        full_decision = decide_update(rt2.evidence, thresholds)
        decisions = {
            "rt1o_unfactorized": GuardDecision(True, "unprotected_baseline", 0.0),
            "rt2f_unguarded": GuardDecision(True, "unprotected_baseline", 0.0),
            "rt3_gain_only": _gain_decision(rt2.evidence, thresholds),
            "rt3_gradient_only": _gradient_decision(rt2.evidence, thresholds),
            "rt3_full": full_decision,
        }
        for method in _METHODS:
            decision = decisions[method]
            if method == "rt1o_unfactorized":
                method_backend = rt1_visual
                method_target = targets
                method_measured = rt1_measured
                method_reference = reference
                candidate_qpos = rt1_result.corrected_qpos
            else:
                method_backend = joint_visual
                method_target = targets
                method_measured = measured
                method_reference = calibrated_reference
                candidate_qpos = rt2.joint_result.corrected_qpos
            final_qpos = candidate_qpos if decision.accepted else method_measured
            outcome, before, after = _outcome(
                backend=method_backend,
                target=method_target,
                measured=method_measured,
                final=final_qpos,
                reference=method_reference,
                selected_indices=selected_indices,
                acceptance=config.get("acceptance", {}),
            )
            rows.append(
                {
                    "trial_id": f"{state_id}_{case['offset_name']}",
                    "state_id": state_id,
                    "split": "validation",
                    "case_role": case["case_role"],
                    "camera_mode": "+".join(camera_names),
                    "offset_name": case["offset_name"],
                    "magnitude_deg": case["magnitude_deg"],
                    "method": method,
                    "trial_type": outcome["trial_type"],
                    "candidate_committed": decision.accepted,
                    "decision_reason": decision.reason,
                    "minimum_margin": decision.minimum_margin,
                    "state_error_reduction": outcome["state_error_reduction"],
                    "natural_state_correction_rad": outcome["natural_state_correction_rad"],
                    "mask_iou_before": before["mean_mask_iou"],
                    "mask_iou_after": after["mean_mask_iou"],
                    "mask_iou_gain": after["mean_mask_iou"] - before["mean_mask_iou"],
                    "boundary_f1_before": before["mean_boundary_f1"],
                    "boundary_f1_after": after["mean_boundary_f1"],
                    "boundary_f1_gain": after["mean_boundary_f1"] - before["mean_boundary_f1"],
                    "recovery_success": bool(outcome["recovery_success"]),
                    "evidence_finite": rt2.evidence.finite,
                    "view_count": rt2.evidence.view_count,
                    "visual_gain_ratio": rt2.evidence.visual_gain_ratio,
                    "gradient_cosine": rt2.evidence.gradient_cosine,
                    "correction_cosine": rt2.evidence.correction_cosine,
                    "relative_correction_disagreement": rt2.evidence.relative_correction_disagreement,
                    "candidate_bound_fraction": rt2.evidence.candidate_bound_fraction,
                }
            )
        for source, result in {
            "rt1o": rt1_result,
            "rt2_joint": rt2.joint_result,
            **{f"rt2_{name}": value for name, value in rt2.camera_results.items()},
        }.items():
            traces.extend(
                {"trial_id": f"{state_id}_{case['offset_name']}", "candidate_source": source, **trace}
                for trace in result.trace
            )
        media_contexts.append(
            {
                "state_id": state_id,
                "case_role": case["case_role"],
                "observations": observations,
                "targets": targets,
                "backend": joint_visual,
                "measured": measured,
                "candidate": rt2.joint_result.corrected_qpos,
                "final": rt2.joint_result.corrected_qpos if full_decision.accepted else measured,
                "decision": full_decision,
                "evidence": rt2.evidence,
            }
        )

    summaries = {}
    for method in _METHODS:
        method_rows = [row for row in rows if row["method"] == method]
        controlled = [row for row in method_rows if row["trial_type"] == "controlled_offset"]
        zeros = [row for row in method_rows if row["trial_type"] == "zero_injection_control"]
        paired = [row for row in controlled if row["case_role"] == "paired_controlled"]
        one_degree = [row for row in paired if math.isclose(float(row["magnitude_deg"]), 1.0, abs_tol=0.1)]
        summaries[method] = {
            "case_count": len(method_rows),
            "controlled_success_rate": mean(row["recovery_success"] for row in controlled) if controlled else None,
            "paired_controlled_success_rate": mean(row["recovery_success"] for row in paired) if paired else None,
            "paired_one_degree_success_rate": mean(row["recovery_success"] for row in one_degree) if one_degree else None,
            "zero_control_stability": mean(row["recovery_success"] for row in zeros) if zeros else None,
            "paired_controlled_commit_coverage": mean(row["candidate_committed"] for row in paired) if paired else None,
            "mean_mask_iou_gain": mean(row["mask_iou_gain"] for row in method_rows),
            "mean_boundary_f1_gain": mean(row["boundary_f1_gain"] for row in method_rows),
        }
    full_rows = [row for row in rows if row["method"] == "rt3_full"]
    full_paired = [row for row in full_rows if row["case_role"] == "paired_controlled"]
    rt2_by_trial = {
        row["trial_id"]: row for row in rows if row["method"] == "rt2f_unguarded"
    }
    paired_stats = _exact_mcnemar(
        [bool(row["recovery_success"]) for row in full_paired],
        [bool(rt2_by_trial[row["trial_id"]]["recovery_success"]) for row in full_paired],
    )
    committed_full = [row for row in full_rows if row["candidate_committed"]]
    primary = summaries["rt3_full"]
    gates = {
        "paired_controlled_success_at_least_80pct": bool(
            primary["paired_controlled_success_rate"] is not None
            and primary["paired_controlled_success_rate"] >= 0.80
        ),
        "paired_one_degree_success_at_least_60pct": bool(
            primary["paired_one_degree_success_rate"] is not None
            and primary["paired_one_degree_success_rate"] >= 0.60
        ),
        "zero_control_stability_at_least_80pct": bool(
            primary["zero_control_stability"] is not None
            and primary["zero_control_stability"] >= 0.80
        ),
        "paired_controlled_coverage_at_least_60pct": bool(
            primary["paired_controlled_commit_coverage"] is not None
            and primary["paired_controlled_commit_coverage"] >= 0.60
        ),
        "positive_committed_visual_gains": bool(
            committed_full
            and mean(row["mask_iou_gain"] for row in committed_full) > 0
            and mean(row["boundary_f1_gain"] for row in committed_full) > 0
        ),
    }
    metrics = {
        "evaluation_split": "validation",
        "case_count": len(cases),
        "method_row_count": len(rows),
        "validation_state_ids": sorted({case["state_id"] for case in cases}),
        "factor_sha256": factor_sha,
        "guard_sha256": guard_sha,
        "test_matrix_sha256": _sha256(matrix_path),
        "guard_source_state_ids": guard["source_state_ids"],
        "method_summaries": summaries,
        "full_guard_rejection_reasons": dict(Counter(row["decision_reason"] for row in full_rows)),
        "full_vs_rt2f_paired_mcnemar": paired_stats,
        "acceptance_gates": gates,
        "all_acceptance_gates_passed": all(gates.values()),
    }
    real_source = config["real_observations"]
    run = RunArtifacts.create(
        root=config.get("runs_root", "runs"),
        experiment=str(config.get("experiment", "rt3_guard_evaluation")),
        run_id=run_id,
        config=config,
        assets={
            **paths.as_assets(),
            "factors": factor_path,
            "guard": guard_path,
            "guard_test_matrix": matrix_path,
            **{
                f"records_{camera}": real_source["records"][camera]
                for camera in dataset.cameras
            },
        },
        target_provenance="real_observation_guard_evaluation",
        observation_backend="native_cuda_frozen_factor_guard_test",
    )
    run.write_metrics(metrics)
    run.write_trace(traces)
    with (run.path / "results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    accepted = [
        context
        for context in media_contexts
        if context["case_role"] == "paired_controlled" and context["decision"].accepted
    ]
    rolled_zero = [
        context
        for context in media_contexts
        if context["case_role"] == "paired_zero" and not context["decision"].accepted
    ]
    fallback_rejected = [context for context in media_contexts if not context["decision"].accepted]
    selected_media = []
    for candidates in (accepted, rolled_zero, fallback_rejected, media_contexts):
        for context in candidates:
            if context not in selected_media:
                selected_media.append(context)
                break
        if len(selected_media) >= 2:
            break
    write_guard_comparison(
        run.path / "images" / "guard_commit_rollback.png", contexts=selected_media
    )
    write_guard_video(
        run.path / "videos" / "guard_commit_rollback.mp4",
        contexts=selected_media,
        frame_count=int(evaluation.get("video_frames", 36)),
        fps=int(evaluation.get("video_fps", 12)),
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
        execute_rt3_evaluate(
            load_config(arguments.config),
            factors_path=arguments.factors,
            guard_path=arguments.guard,
            matrix_path=arguments.matrix,
            run_id=arguments.run_id,
        )
    )


if __name__ == "__main__":
    main()
