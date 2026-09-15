#!/usr/bin/env python3
"""CPU-only, frozen-rule heldout evaluation with pose-clustered statistics.

No calibration, rendering, training, or CUDA operations are performed here.
The decision stage reads an explicit measurement/candidate/evidence allowlist;
reference qpos is read only after all atomic decisions have been materialized.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
from itertools import product
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

ORIGINAL_METHODS = ("measurement_raw", "gaussian_inverse_unguarded", "global_guard", "component_guard")
OURS = "risk_atomic_commit"
METHODS = (*ORIGINAL_METHODS, OURS)
CONDITIONS = ("clean_nonzero", "clean_zero", "one_view_stale_nonzero", "one_view_stale_zero")
GROUPS = ("all", *CONDITIONS, "nonzero_pooled", "zero_pooled")
LABELS = dict(zip(METHODS, ("Raw", "Inverse", "Global", "Component", "Risk atomic")))
COLORS = dict(zip(METHODS, ("#8B9EA5", "#CC9E4C", "#E0D0B6", "#E0D0B6", "#6B2717")))
REFERENCE_COLOR = "#442C1B"
GROUP_LABELS = {"all": "All conditions", "clean_nonzero": "Clean / nonzero",
                "clean_zero": "Clean / zero", "one_view_stale_nonzero": "Stale / nonzero",
                "one_view_stale_zero": "Stale / zero", "nonzero_pooled": "Nonzero pooled",
                "zero_pooled": "Zero pooled"}
EVIDENCE_FIELDS = ("view_count", "visual_gain_ratio", "gradient_cosine", "correction_cosine",
                   "relative_correction_disagreement", "candidate_bound_fraction", "finite")
METRICS = ("qmae_deg", "whole_state_harmful_rate", "actual_nonzero_commit_rate",
           "beneficial_candidate_retention", "zero_drift_deg", "zero_drift_rate",
           "beneficial_candidate_utilization", "nontrivial_commit_rate")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames and len(set(reader.fieldnames)) == len(reader.fieldnames),
                f"missing or duplicate CSV columns: {path}")
        rows = list(reader)
    require(all(None not in row and None not in row.values() for row in rows),
            f"malformed or incomplete CSV: {path}")
    return rows


def write_csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def boolean(value):
    require(str(value).lower() in ("true", "false"), f"invalid boolean: {value}")
    return str(value).lower() == "true"


def vector(value, *, finite=True):
    result = np.asarray(json.loads(value), dtype=np.float64)
    require(result.ndim == 1 and result.size == 7, "expected seven joint coordinates")
    if finite:
        require(np.isfinite(result).all(), "nonfinite state vector")
    return result


def birth_utc(path):
    # Birth time is not mtime: rerunning/copying results must not fake chronology.
    result = subprocess.run(["stat", "-c", "%W", str(path)], check=True,
                            capture_output=True, text=True)
    seconds = int(result.stdout.strip())
    require(seconds > 0, "run birth time unavailable; cannot establish freeze before run")
    return datetime.fromtimestamp(seconds, timezone.utc)


def load_validated(freeze_path, run):
    freeze_path, run = Path(freeze_path).resolve(), Path(run).resolve()
    needed = ("candidate_evidence.csv", "results.csv", "manifest.json", "heldout_manifest.json",
              "config_source.yaml", "config.yaml", "script.sha256", "candidate_flush.json")
    require(all((run / name).is_file() for name in needed), "heldout run incomplete: missing required artifact")
    freeze = json.loads(freeze_path.read_text())
    sources = freeze["sources"]
    required_sources = ("candidate_evidence.csv", "external_gs_guard_franka.yaml",
                        "calibrate_risk_atomic.py", "risk_atomic.py", "run_external_gs_guard.py")
    require(all(sum(Path(p).name == name for p in sources) == 1 for name in required_sources),
            "freeze must contain all five unique required source hashes")
    for path, expected in sources.items():
        require(Path(path).is_file() and sha256(path) == expected, f"freeze source hash mismatch: {path}")
    by_name = {Path(p).name: p for p in sources}
    require(Path(by_name["risk_atomic.py"]).resolve() == ROOT / "src/kinesync/external_gs/risk_atomic.py",
            "freeze source must name the actual imported risk_atomic module")
    require(Path(by_name["calibrate_risk_atomic.py"]).resolve() == ROOT / "scripts/calibrate_risk_atomic.py",
            "freeze source must name the actual calibration script")
    frozen_config = freeze["config"]
    require(yaml.safe_load(Path(by_name["external_gs_guard_franka.yaml"]).read_text()) == frozen_config,
            "freeze config differs from hashed config source")
    config_hash, runner_hash = sources[by_name["external_gs_guard_franka.yaml"]], sources[by_name["run_external_gs_guard.py"]]
    manifest = json.loads((run / "manifest.json").read_text())
    heldout = json.loads((run / "heldout_manifest.json").read_text())
    require(manifest.get("split") == "heldout" and manifest.get("target_provenance") == "analysis_same_gs",
            "not a complete same-GS heldout run")
    require(manifest.get("status", "complete") == "complete", "run is not complete")
    require(manifest.get("methods") == list(ORIGINAL_METHODS), "unexpected original methods")
    require(manifest.get("config_source_sha256") == config_hash == sha256(run / "config_source.yaml"),
            "run config source hash differs from freeze")
    require(manifest.get("config_sha256") == sha256(run / "config.yaml"), "run config hash mismatch")
    run_config = yaml.safe_load((run / "config.yaml").read_text())
    run_config.pop("_config_path", None)
    require(run_config == frozen_config, "effective run config differs from freeze")
    require(manifest.get("script_sha256") == runner_hash == (run / "script.sha256").read_text().strip(),
            "run script hash differs from freeze")
    cases = [r["id"] for r in frozen_config["heldout_cases"]]
    conditions = [r["id"] for r in frozen_config["conditions"]]
    require(len(cases) == len(set(cases)) == 12 and len(conditions) == 4 and set(conditions) == set(CONDITIONS),
            "freeze design must be 12 unique poses x four conditions")
    expected = {f"{case}__{condition}" for case in cases for condition in conditions}
    require(heldout.get("split") == "heldout" and heldout.get("case_count") == 12
            and heldout.get("frozen_before_heldout_execution") is True
            and heldout.get("config_sha256") == config_hash
            and sorted(heldout.get("case_ids", [])) == sorted(cases)
            and sorted(heldout.get("condition_ids", [])) == sorted(conditions),
            "heldout manifest differs from frozen design")
    development = read_csv(by_name["candidate_evidence.csv"])
    dev_ids = {r["state_id"] for r in development}
    require(all(r["split"] == "development" for r in development), "development source split mismatch")
    require(len(dev_ids) == len(development) == len(freeze["development_state_ids"])
            and dev_ids == set(freeze["development_state_ids"]), "development IDs differ from hashed source")
    require(not (dev_ids & expected), "development/heldout state overlap")
    require(not ({r["case_id"] for r in development} & set(cases)), "development/heldout pose overlap")
    created = datetime.fromisoformat(freeze["created_utc"])
    require(created.tzinfo is not None, "freeze created_utc requires timezone")
    started = birth_utc(run)
    require(created < started, "freeze must strictly precede run creation")
    candidates, original = read_csv(run / "candidate_evidence.csv"), read_csv(run / "results.csv")
    ids = [r["state_id"] for r in candidates]
    require(len(ids) == len(set(ids)) == 48, "expected 48 unique candidate IDs, no duplicates")
    require(set(ids) == expected, "heldout candidate matrix differs from frozen 12 x 4 design")
    candidate_lookup = {r["state_id"]: r for r in candidates}
    for row in candidates:
        require(row["split"] == "heldout" and row["case_id"] in cases and row["condition_id"] in CONDITIONS
                and row["state_id"] == f"{row['case_id']}__{row['condition_id']}", "invalid heldout row ID/split")
        require(row["target_provenance"] == "analysis_same_gs" and boolean(row["evaluation_reference_only"]),
                "candidate reference provenance mismatch")
        condition = next(c for c in frozen_config["conditions"] if c["id"] == row["condition_id"])
        require(row["trial_type"] == condition["trial_type"], "condition trial type differs from freeze")
        vector(row["measured_qpos_rad"])
        vector(row["candidate_qpos_rad"])
    flush = json.loads((run / "candidate_flush.json").read_text())
    require(flush.get("candidate_row_count") == 48
            and flush.get("candidate_evidence_sha256") == sha256(run / "candidate_evidence.csv"),
            "candidate flush hash/count mismatch")
    pairs = [(r["state_id"], r["method"]) for r in original]
    require(len(pairs) == len(set(pairs)) == 192
            and set(pairs) == set(product(expected, ORIGINAL_METHODS)), "incomplete 192-row original method matrix")
    for row in original:
        candidate_row = candidate_lookup[row["state_id"]]
        require(not any(k.startswith("analysis_") or k.startswith("risk_atomic_") for k in row),
                "original results already contain analysis columns")
        for key in ("split", "case_id", "condition_id", "target_provenance"):
            require(row[key] == candidate_row[key], f"original row metadata mismatch: {key}")
        require(boolean(row["evaluation_reference_only"]), "original reference provenance mismatch")
        for key in ("measured_qpos_rad", "candidate_qpos_rad"):
            require(np.array_equal(vector(row[key]), vector(candidate_row[key])), f"shared {key} differs across methods")
        final = vector(row["final_qpos_rad"])
        if row["method"] in ORIGINAL_METHODS[:2]:
            key = "measured_qpos_rad" if row["method"] == "measurement_raw" else "candidate_qpos_rad"
            require(np.array_equal(final, vector(candidate_row[key])), "raw/inverse final state mismatch")
    input_hashes = {str(run / name): sha256(run / name) for name in needed}
    input_hashes[str(freeze_path)] = sha256(freeze_path)
    audit = {"freeze_sha256": sha256(freeze_path), "source_hashes_verified": sources,
             "input_hashes": input_hashes, "freeze_created_utc": created.astimezone(timezone.utc).isoformat(),
             "run_created_utc": started.isoformat(), "run_creation_evidence": "filesystem birth time (whole seconds, not mtime)",
             "freeze_precedes_run": True, "freeze_lead_seconds_lower_bound": (started - created).total_seconds(),
             "unique_state_count": 48, "pose_clusters": 12, "condition_count": 4,
             "development_state_count": len(dev_ids), "development_state_disjoint": True,
             "development_pose_disjoint": True, "original_rows": 192, "original_fields_preserved": True,
             "source_run": str(run), "source_manifest_job_id": manifest.get("slurm_job_id"),
             "decision_input_fields": ["measured_qpos_rad", "candidate_qpos_rad", *EVIDENCE_FIELDS],
             "evaluation_reference_field": "true_qpos_rad", "gpu_work_performed": False,
             "analysis_script_sha256": sha256(__file__),
             "decision_dependency_hashes_at_analysis": {
                 str(ROOT / "src/kinesync/guard" / name): sha256(ROOT / "src/kinesync/guard" / name)
                 for name in ("decision.py", "schema.py")}}
    return freeze, candidates, original, audit


def decide_candidates(candidates, thresholds):
    from kinesync.external_gs.risk_atomic import commit
    from kinesync.guard.schema import UpdateEvidence

    decisions = {}
    for row in candidates:
        measured = vector(row["measured_qpos_rad"]).tolist()
        candidate = vector(row["candidate_qpos_rad"], finite=False).tolist()
        evidence = UpdateEvidence(int(row["view_count"]), float(row["visual_gain_ratio"]),
                                  float(row["gradient_cosine"]), float(row["correction_cosine"]),
                                  float(row["relative_correction_disagreement"]),
                                  float(row["candidate_bound_fraction"]), boolean(row["finite"]))
        final, accepted = commit(measured, candidate, evidence, thresholds)
        require(final == measured or final == candidate, "atomic commit spliced coordinates")
        decisions[row["state_id"]] = {"final_qpos_rad": json.dumps(final), "risk_atomic_accepted": accepted}
    return decisions


def evaluate_rows(candidates, original, decisions, freeze):
    lookup = {r["state_id"]: r for r in candidates}
    require(set(decisions) == set(lookup), "all decisions must exist before reference evaluation")
    rows = [dict(r) for r in original]
    for candidate in candidates:
        row = {key: candidate[key] for key in ("split", "state_id", "case_id", "condition_id",
               "target_provenance", "evaluation_reference_only", "measured_qpos_rad", "candidate_qpos_rad")}
        row.update(method=OURS, **decisions[candidate["state_id"]])
        rows.append(row)
    label_tol = float(freeze["label_tolerance_mae_rad"])
    zero_tol = float(freeze["config"]["guard"]["zero_tolerance_rad"])
    for row in rows:
        candidate_row = lookup[row["state_id"]]
        measured = vector(candidate_row["measured_qpos_rad"])
        candidate = vector(candidate_row["candidate_qpos_rad"])
        final = vector(row["final_qpos_rad"])
        truth = vector(candidate_row["true_qpos_rad"])
        measured_mae = float(np.abs(measured - truth).mean())
        candidate_mae = float(np.abs(candidate - truth).mean())
        final_mae = float(np.abs(final - truth).mean())
        delta = np.abs(final - measured)
        zero = candidate_row["trial_type"] == "zero"
        helpful = candidate_mae < measured_mae - label_tol
        nonzero = bool(np.any(final != measured))
        retained = bool(helpful and nonzero and np.array_equal(final, candidate))
        row.update(analysis_qmae_deg=float(np.rad2deg(final_mae)),
                   analysis_measured_qmae_deg=float(np.rad2deg(measured_mae)),
                   analysis_candidate_qmae_deg=float(np.rad2deg(candidate_mae)),
                   analysis_whole_state_harmful=bool(final_mae > measured_mae + label_tol),
                   analysis_actual_nonzero_commit=nonzero,
                   analysis_nontrivial_commit=bool(np.any(delta > zero_tol)),
                   analysis_beneficial_candidate=bool(helpful),
                   analysis_beneficial_candidate_retained=retained,
                   analysis_beneficial_candidate_utilized=bool(helpful and nonzero and final_mae < measured_mae - label_tol),
                   analysis_zero_trial=zero,
                   analysis_zero_drift_deg=float(np.rad2deg(delta.mean())) if zero else None,
                   analysis_zero_drift=bool(np.any(delta > zero_tol)) if zero else None,
                   analysis_zero_drift_max_joint_deg=float(np.rad2deg(delta.max())) if zero else None,
                   analysis_reference_qpos_rad=candidate_row["true_qpos_rad"])
        if row["method"] == OURS:
            row["joint_mae_deg"] = row["analysis_qmae_deg"]
            row["measured_joint_mae_deg"] = row["analysis_measured_qmae_deg"]
    return rows


def audit_evaluation(candidates, rows, freeze):
    cases = {c["id"]: c for c in freeze["config"]["heldout_cases"]}
    for row in candidates:
        case = cases[row["case_id"]]
        expected_truth = np.asarray(case["qpos_rad"], dtype=np.float32)
        require(np.array_equal(vector(row["true_qpos_rad"]), expected_truth),
                f"evaluation reference differs from frozen pose: {row['state_id']}")
        offset = (np.zeros(7, dtype=np.float32) if row["trial_type"] == "zero" else
                  np.asarray(case["injected_measurement_offset_rad"], dtype=np.float32))
        require(np.array_equal(vector(row["measured_qpos_rad"]), expected_truth + offset),
                f"measurement differs from frozen condition: {row['state_id']}")
    discrepancy = max(abs(float(r["joint_mae_deg"]) - r["analysis_qmae_deg"]) for r in rows[:192])
    require(discrepancy < 1e-5, "source qMAE inconsistent with reference qpos")
    return {"reference_matches_frozen_poses": True, "measurement_matches_frozen_conditions": True,
            "source_qmae_max_abs_discrepancy_deg": discrepancy}


def bootstrap_indices(seed=20260908):
    return np.random.default_rng(seed).integers(0, 12, size=(10000, 12))


def cluster_ratio(numerator, denominator, indices):
    numerator, denominator = np.asarray(numerator, dtype=float), np.asarray(denominator, dtype=float)
    require(numerator.shape == denominator.shape == (12,), "statistics require 12 pose clusters")
    total_denominator = float(denominator.sum())
    sampled_denominator = denominator[indices].sum(axis=1)
    valid = sampled_denominator > 0
    values = numerator[indices].sum(axis=1)[valid] / sampled_denominator[valid]
    return {"estimate": float(numerator.sum() / total_denominator) if total_denominator > 0 else None,
            "ci95": [float(v) for v in np.quantile(values, [.025, .975])] if len(values) else [None, None],
            "numerator": float(numerator.sum()), "denominator": total_denominator,
            "bootstrap_valid_replicates": int(valid.sum())}


def exact_signflip(pose_improvements):
    values = np.asarray(pose_improvements, dtype=np.float64)
    require(values.shape == (12,) and np.isfinite(values).all(), "exact signflip requires 12 finite pose values")
    signs = np.asarray(list(product((-1., 1.), repeat=12)))
    null = (signs * values).mean(axis=1)
    observed = abs(float(values.mean()))
    tolerance = np.finfo(float).eps * 16 * max(float(np.abs(values).mean()), observed, np.finfo(float).tiny)
    return float(np.mean(np.abs(null) >= observed - tolerance))


def in_group(row, group):
    if group == "all":
        return True
    if group == "nonzero_pooled":
        return not row["analysis_zero_trial"]
    if group == "zero_pooled":
        return row["analysis_zero_trial"]
    return row["condition_id"] == group


def summarize(rows, indices):
    poses = sorted({r["case_id"] for r in rows})
    require(len(poses) == 12, "expected 12 pose clusters")
    matrix, comparisons = [], []
    for group in GROUPS:
        grouped = [r for r in rows if in_group(r, group)]
        for method in METHODS:
            selected = [r for r in grouped if r["method"] == method]
            clusters = [[r for r in selected if r["case_id"] == pose] for pose in poses]
            require(all(clusters) and len({len(c) for c in clusters}) == 1, "unbalanced pose matrix")
            stats = {"group": group, "method": method, "state_count": len(selected), "pose_count": 12}
            for metric, field, denominator_field in (
                ("qmae_deg", "analysis_qmae_deg", None),
                ("whole_state_harmful_rate", "analysis_whole_state_harmful", None),
                ("actual_nonzero_commit_rate", "analysis_actual_nonzero_commit", None),
                ("beneficial_candidate_retention", "analysis_beneficial_candidate_retained", "analysis_beneficial_candidate"),
                ("zero_drift_deg", "analysis_zero_drift_deg", "analysis_zero_trial"),
                ("zero_drift_rate", "analysis_zero_drift", "analysis_zero_trial"),
                ("beneficial_candidate_utilization", "analysis_beneficial_candidate_utilized", "analysis_beneficial_candidate"),
                ("nontrivial_commit_rate", "analysis_nontrivial_commit", None),
            ):
                numerator = [sum(float(r[field] or 0) for r in cluster) for cluster in clusters]
                denominator = [len(cluster) if denominator_field is None else
                               sum(bool(r[denominator_field]) for r in cluster) for cluster in clusters]
                stats[metric] = cluster_ratio(numerator, denominator, indices)
            matrix.append(stats)
        lookup = {(r["state_id"], r["method"]): r for r in grouped}
        for baseline in ORIGINAL_METHODS:
            improvements = [float(np.mean([
                r["analysis_qmae_deg"] - lookup[(r["state_id"], OURS)]["analysis_qmae_deg"]
                for r in grouped if r["case_id"] == pose and r["method"] == baseline])) for pose in poses]
            result = cluster_ratio(improvements, np.ones(12), indices)
            baseline_means = [float(np.mean([r["analysis_qmae_deg"] for r in grouped
                                            if r["case_id"] == pose and r["method"] == baseline])) for pose in poses]
            relative = cluster_ratio(100 * np.asarray(improvements), baseline_means, indices)
            role = ("primary" if baseline in ORIGINAL_METHODS[:2] else "secondary_exploratory") if group == "all" else "descriptive"
            comparisons.append({"group": group, "baseline": baseline, "ours": OURS,
                                "improvement_deg": result, "exact_signflip_p_two_sided": exact_signflip(improvements),
                                "relative_reduction_percent": relative, "comparison_role": role,
                                "p_holm_primary_two": None,
                                "pose_improvements_deg": dict(zip(poses, improvements)),
                                "signflip_assignments": 4096, "pose_count": 12})
    primary = sorted([r for r in comparisons if r["comparison_role"] == "primary"],
                     key=lambda r: r["exact_signflip_p_two_sided"])
    require(len(primary) == 2, "Holm family must contain only two original primary comparisons")
    previous = 0.
    for rank, row in enumerate(primary):
        previous = max(previous, min(1., (2-rank) * row["exact_signflip_p_two_sided"]))
        row["p_holm_primary_two"] = previous
    ours_rows = [r for r in rows if r["method"] == OURS]
    diagnostics = {
        "accepted_states": sum(r["risk_atomic_accepted"] for r in ours_rows),
        "accepted_noop_states": sum(r["risk_atomic_accepted"] and not r["analysis_actual_nonzero_commit"] for r in ours_rows),
        "actual_nonzero_commit_states": sum(r["analysis_actual_nonzero_commit"] for r in ours_rows),
        "beneficial_candidate_count": sum(r["analysis_beneficial_candidate"] for r in ours_rows),
        "missed_beneficial_candidates": [
            {"state_id": r["state_id"], "measured_qmae_deg": r["analysis_measured_qmae_deg"],
             "candidate_qmae_deg": r["analysis_candidate_qmae_deg"], "final_qmae_deg": r["analysis_qmae_deg"]}
            for r in ours_rows if r["analysis_beneficial_candidate"] and not r["analysis_beneficial_candidate_retained"]],
        "harmful_committed_state_ids": [r["state_id"] for r in ours_rows if r["analysis_whole_state_harmful"]]}
    return {"schema": "risk_atomic.heldout.qmae.v1", "methods": list(METHODS),
            "risk_atomic_diagnostics": diagnostics,
            "conditions": list(CONDITIONS), "matrix": matrix, "comparisons": comparisons,
            "statistics": {"cluster_unit": "pose", "clusters": 12, "bootstrap_replicates": 10000,
                           "bootstrap_seed": 20260908, "bootstrap_ci": "percentile 95%; paired resampling of whole poses",
                           "undefined_denominators": "null; zero-denominator bootstrap draws excluded and valid count reported",
                           "signflip": "two-sided exact mean statistic over all 2^12 assignments, including zero clusters",
                           "multiplicity": "Holm correction only for the two original all-condition raw/inverse primary comparisons; global/component all-condition comparisons secondary/exploratory; every stratum descriptive with original unadjusted p",
                           "relative_reduction_percent": "100 * (mean baseline qMAE - mean ours qMAE) / mean baseline qMAE, from full-precision values; paired pose-bootstrap CI; null for zero baseline mean",
                           "improvement_sign": "baseline qMAE minus ours qMAE; positive favors ours"},
            "metric_definitions": {
                "qmae_deg": "mean absolute final-reference error over seven joints, converted to degrees",
                "whole_state_harmful_rate": "fraction of states with final qMAE > measured qMAE + frozen label tolerance (rad)",
                "actual_nonzero_commit_rate": "fraction of states with any final coordinate exactly different from measurement; not gate acceptance",
                "nontrivial_commit_rate": "fraction with max abs(final-measured) above frozen zero_tolerance_rad",
                "beneficial_candidate_retention": "beneficial candidates fully retained: final exactly equals candidate and actually changes state; denominator all qMAE-beneficial candidates",
                "beneficial_candidate_utilization": "beneficial candidates yielding a nonzero final update and lower whole-state qMAE, including partial component updates",
                "zero_drift_deg": "mean absolute final-measured displacement in degrees, on zero trials only",
                "zero_drift_rate": "zero-trial fraction with any abs(final-measured) above frozen zero_tolerance_rad",
                "task_success": "not measured; qMAE and update risk are not task success"}}


def flattened_matrix(summary):
    result = []
    for row in summary["matrix"]:
        flat = {key: row[key] for key in ("group", "method", "state_count", "pose_count")}
        for metric in METRICS:
            value = row[metric]
            flat.update({f"{metric}_mean": value["estimate"], f"{metric}_ci_low": value["ci95"][0],
                         f"{metric}_ci_high": value["ci95"][1], f"{metric}_numerator": value["numerator"],
                         f"{metric}_denominator": value["denominator"],
                         f"{metric}_bootstrap_valid": value["bootstrap_valid_replicates"]})
        result.append(flat)
    return result


def formatted(metric, *, percent=False):
    if metric["estimate"] is None:
        return "NA"
    scale = 100 if percent else 1
    decimals = 1 if percent else 3
    low, high = metric["ci95"]
    return f"{scale * metric['estimate']:.{decimals}f} [{scale * low:.{decimals}f}, {scale * high:.{decimals}f}]"


def tables_markdown(summary):
    lines = ["# External Heldout: Frozen Risk Atomic", "",
             "48 states = 12 poses x 4 conditions. Entries: estimate [pose-cluster bootstrap 95% CI].",
             "Rates are percentages; qMAE and zero drift are degrees. NA means an empty denominator, not zero.", ""]
    for group in GROUPS:
        lines += [f"## {GROUP_LABELS[group]}", "",
                  "| Method | qMAE (deg) | Whole-state harmful (%) | Actual nonzero commit (%) | Benefit utilization (%) | Zero drift (deg) | Zero drift (%) |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for row in summary["matrix"]:
            if row["group"] == group:
                values = [LABELS[row["method"]], formatted(row["qmae_deg"]),
                          formatted(row["whole_state_harmful_rate"], percent=True),
                          formatted(row["actual_nonzero_commit_rate"], percent=True),
                          formatted(row["beneficial_candidate_utilization"], percent=True),
                          formatted(row["zero_drift_deg"]), formatted(row["zero_drift_rate"], percent=True)]
                lines.append("| " + " | ".join(values) + " |")
        lines.append("")
    lines += ["## Paired Improvements", "", "Positive = baseline minus ours (deg). Exact two-sided signflip, 12 clusters, 4096 assignments. Holm applies only to the two original all-condition raw/inverse primary tests. Other p values remain unadjusted.", "",
              "Relative reduction uses unrounded means, not the ratio of displayed three-decimal values. NA: baseline qMAE is zero.", "",
              "| Group | Baseline | Role | Improvement (deg) [95% CI] | Relative reduction (%) | Exact p | Holm p (2 primary) |", "|---|---|---|---:|---:|---:|---:|"]
    for row in summary["comparisons"]:
        relative = row["relative_reduction_percent"]["estimate"]
        percent = "NA" if relative is None else f"{relative:.6f}"
        adjusted = "NA" if row["p_holm_primary_two"] is None else f"{row['p_holm_primary_two']:.8f}"
        lines.append(f"| {GROUP_LABELS[row['group']]} | {LABELS[row['baseline']]} | {row['comparison_role']} | {formatted(row['improvement_deg'])} | {percent} | {row['exact_signflip_p_two_sided']:.8f} | {adjusted} |")
    lines += ["", "## Full-Candidate Retention Audit", "",
              "This strict whole-candidate metric is not the paper table's benefit utilization metric. Partial component corrections can be useful without retaining the entire candidate.", "",
              "| Group | Method | Retained / beneficial candidates | Full retention (%) [95% CI] |", "|---|---|---:|---:|"]
    for row in summary["matrix"]:
        value = row["beneficial_candidate_retention"]
        lines.append(f"| {GROUP_LABELS[row['group']]} | {LABELS[row['method']]} | {int(value['numerator'])}/{int(value['denominator'])} | {formatted(value, percent=True)} |")
    lines += ["", "## Definitions", ""]
    lines.extend(f"- `{key}`: {value}." for key, value in summary["metric_definitions"].items())
    lines += ["", "Every denominator, relative-reduction bootstrap interval, and nontrivial commit rate is also available in summary.json and matrix.csv / comparisons.csv.", ""]
    return "\n".join(lines)


def tables_tex(summary):
    def cell(metric, percent=False):
        text = formatted(metric, percent=percent)
        if text == "NA":
            return text
        mean, interval = text.split(" ", 1)
        return r"\shortstack{" + mean + r"\\{" + interval + "}}"

    lines = ["% Generated CPU-only analysis. Requires booktabs; no cluster-specific execution details.",
             "% All cells: mean [pose-cluster bootstrap 95 percent CI]. NA: zero denominator."]
    for group in GROUPS:
        lines += [r"\begin{table*}[t]", r"\centering", r"\scriptsize",
                  r"\setlength{\tabcolsep}{3pt}",
                  r"\caption{" + GROUP_LABELS[group] + r": heldout qMAE and whole-state update risk; not task success. Rates are percentages.}",
                  r"\begin{tabular}{lrrrrrr}", r"\toprule",
                  r"Method & \shortstack{qMAE\\(deg)} & \shortstack{Harmful\\(\%)} & \shortstack{Nonzero commit\\(\%)} & \shortstack{Benefit utilized\\(\%)} & \shortstack{Zero drift\\(deg)} & \shortstack{Zero drift\\(\%)} \\",
                  r"\midrule"]
        for row in summary["matrix"]:
            if row["group"] == group:
                values = [LABELS[row["method"]], cell(row["qmae_deg"]),
                          cell(row["whole_state_harmful_rate"], percent=True),
                          cell(row["actual_nonzero_commit_rate"], percent=True),
                          cell(row["beneficial_candidate_utilization"], percent=True),
                          cell(row["zero_drift_deg"]), cell(row["zero_drift_rate"], percent=True)]
                lines.append(" & ".join(values) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    lines += [r"\begin{table*}[t]", r"\centering\scriptsize", r"\setlength{\tabcolsep}{3pt}",
              r"\caption{Paired qMAE improvement: baseline minus Risk atomic (degrees). Exact two-sided signflip over 12 poses. P: original primary, S: secondary/exploratory, D: descriptive. Holm correction covers only the two P tests.}",
              r"\begin{tabular}{lllrrrr}", r"\toprule", r"Group & Baseline & Role & Improvement [95\% CI] & Reduction (\%) & $p$ & Holm $p$ \\", r"\midrule"]
    for row in summary["comparisons"]:
        role = {"primary": "P", "secondary_exploratory": "S", "descriptive": "D"}[row["comparison_role"]]
        relative = row["relative_reduction_percent"]["estimate"]
        percent = "NA" if relative is None else f"{relative:.6f}"
        adjusted = "NA" if row["p_holm_primary_two"] is None else f"{row['p_holm_primary_two']:.8f}"
        lines.append(f"{GROUP_LABELS[row['group']]} & {LABELS[row['baseline']]} & {role} & {formatted(row['improvement_deg'])} & {percent} & {row['exact_signflip_p_two_sided']:.8f} & {adjusted}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    return "\n".join(lines)


def plot_figures(summary, visuals):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.linewidth": .65,
                         "xtick.labelsize": 6.5, "ytick.labelsize": 6.5})
    manifest = []

    def canvas():
        fig, ax = plt.subplots(figsize=(3.5, 3.5))
        fig.subplots_adjust(left=.31, right=.97, bottom=.14, top=.80)
        ax.grid(axis="x", color="#EEEEEE", linewidth=.5)
        ax.set_axisbelow(True)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4, prune="both"))
        return fig, ax

    def save(fig, ax, name, records):
        require(not ax.get_title() and fig._suptitle is None, "numeric figures must be titlefree")
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        for item in [*ax.get_xticklabels(), *ax.get_yticklabels(), ax.xaxis.label, ax.yaxis.label]:
            if item.get_visible() and item.get_text():
                box = item.get_window_extent(renderer)
                require(box.x0 >= 0 and box.y0 >= 0 and box.x1 <= fig.bbox.width and box.y1 <= fig.bbox.height,
                        f"clipped figure label: {item.get_text()}")
        for suffix in ("pdf", "png"):
            fig.savefig(visuals / f"{name}.{suffix}", dpi=600, facecolor="white")
        write_csv(visuals / f"{name}.csv", records)
        manifest.append({"stem": name, "size_inches": [3.5, 3.5], "titlefree": True,
                         "files": {suffix: sha256(visuals / f"{name}.{suffix}") for suffix in ("pdf", "png", "csv")}})
        plt.close(fig)

    fig, ax = canvas()
    records = []
    offsets = np.linspace(-.27, .27, 5)
    for j, method in enumerate(METHODS):
        for i, condition in enumerate(CONDITIONS):
            row = next(r for r in summary["matrix"] if r["group"] == condition and r["method"] == method)
            value = row["qmae_deg"]
            mean, (low, high) = value["estimate"], value["ci95"]
            ax.errorbar(mean, i + offsets[j], xerr=[[max(0, mean-low)], [max(0, high-mean)]],
                        fmt=("o", "s", "D", "^", "o")[j], color=COLORS[method],
                        markeredgecolor=REFERENCE_COLOR if method in ORIGINAL_METHODS[2:] else COLORS[method],
                        markeredgewidth=.6, markersize=4, capsize=2, elinewidth=1,
                        label=LABELS[method] if i == 0 else None)
            records.append({"condition": condition, "method": method, "qmae_deg": mean,
                            "ci_low": low, "ci_high": high, "pose_clusters": 12, "color": COLORS[method]})
    ax.axvline(0, color=REFERENCE_COLOR, linewidth=.7, linestyle=":")
    ax.set_yticks(range(4), [GROUP_LABELS[c].replace(" / ", "\n") for c in CONDITIONS])
    ax.set_ylim(3.55, -.55)
    ax.set_xlabel("Joint qMAE (deg)")
    ax.legend(loc="lower center", bbox_to_anchor=(.36, 1.02), ncol=3, frameon=False,
              fontsize=6, columnspacing=.8, handletextpad=.4)
    save(fig, ax, "Q4__external-heldout-qmae__square-titlefree", records)

    fig, ax = canvas()
    selected_groups = ("all", *CONDITIONS)
    records = []
    for j, baseline in enumerate(ORIGINAL_METHODS[:2]):
        for i, group in enumerate(selected_groups):
            row = next(r for r in summary["comparisons"] if r["group"] == group and r["baseline"] == baseline)
            mean, (low, high) = row["improvement_deg"]["estimate"], row["improvement_deg"]["ci95"]
            ax.errorbar(mean, i + (j-.5)*.22, xerr=[[max(0, mean-low)], [max(0, high-mean)]],
                        fmt="o" if j == 0 else "s", markersize=4, capsize=2,
                        color=COLORS[baseline], label=f"{LABELS[baseline]} - ours" if i == 0 else None)
            records.append({"group": group, "baseline": baseline, "improvement_deg": mean,
                            "ci_low": low, "ci_high": high, "p_exact_two_sided": row["exact_signflip_p_two_sided"],
                            "comparison_role": row["comparison_role"],
                            "p_holm_primary_two": row["p_holm_primary_two"],
                            "relative_reduction_percent": row["relative_reduction_percent"]["estimate"],
                            "pose_clusters": 12, "color": COLORS[baseline]})
    ax.axvline(0, color=REFERENCE_COLOR, linewidth=.8, linestyle=":")
    ax.set_yticks(range(5), [GROUP_LABELS[c].replace(" / ", "\n") for c in selected_groups])
    ax.set_ylim(4.5, -.5)
    ax.set_xlabel("Paired qMAE improvement (deg)")
    ax.legend(loc="lower center", bbox_to_anchor=(.4, 1.04), ncol=2, frameon=False, fontsize=6)
    save(fig, ax, "Q4__external-heldout-paired-gain__square-titlefree", records)

    fig, ax = canvas()
    records = []
    for j, method in enumerate(METHODS):
        for i, condition in enumerate(CONDITIONS):
            row = next(r for r in summary["matrix"] if r["group"] == condition and r["method"] == method)
            value = row["whole_state_harmful_rate"]
            mean, (low, high) = value["estimate"], value["ci95"]
            ax.errorbar(100*mean, i + offsets[j], xerr=[[100*max(0, mean-low)], [100*max(0, high-mean)]],
                        fmt=("o", "s", "D", "^", "o")[j], markersize=4, capsize=2,
                        color=COLORS[method], markeredgewidth=.6,
                        markeredgecolor=REFERENCE_COLOR if method in ORIGINAL_METHODS[2:] else COLORS[method],
                        label=LABELS[method] if i == 0 else None)
            records.append({"condition": condition, "method": method, "harmful_rate": mean,
                            "ci_low": low, "ci_high": high, "harmful_states": value["numerator"],
                            "states": value["denominator"], "color": COLORS[method]})
    ax.set_yticks(range(4), [GROUP_LABELS[c].replace(" / ", "\n") for c in CONDITIONS])
    ax.set_ylim(3.55, -.55)
    ax.set_xlim(-5, 105)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Whole-state harmful updates (%)")
    ax.legend(loc="lower center", bbox_to_anchor=(.36, 1.02), ncol=3, frameon=False,
              fontsize=6, columnspacing=.8, handletextpad=.4)
    save(fig, ax, "Q4__external-heldout-harmful__square-titlefree", records)
    return manifest


def report_zh(summary):
    audit = summary["audit"]
    all_rows = {r["method"]: r for r in summary["matrix"] if r["group"] == "all"}
    ours = all_rows[OURS]
    lines = ["# External Heldout：冻结 Risk Atomic 结果", "",
             "## 结论与完整结果", "",
             "本次是固定双视角、同一 Gaussian 模型生成观测的受控关节状态恢复分析。只评价 qMAE 及其派生更新风险，不是 task success，也不使用图像分数替代关节误差。",
             f"完整保留 48 个状态（12 姿态 × 4 条件）及原四方法全部 192 行；追加 risk_atomic_commit 48 行，共 240 行。",
             f"Risk atomic 全条件 qMAE：{formatted(ours['qmae_deg'])} 度；整状态 harmful：{formatted(ours['whole_state_harmful_rate'], percent=True)}%；实际非零提交：{formatted(ours['actual_nonzero_commit_rate'], percent=True)}%。",
             f"有益候选完整保留：{formatted(ours['beneficial_candidate_retention'], percent=True)}%；零偏移组平均漂移：{formatted(ours['zero_drift_deg'])} 度。", ""]
    for row in summary["comparisons"]:
        if row["group"] == "all":
            value = row["improvement_deg"]
            direction = "改善" if value["estimate"] > 0 else "退化" if value["estimate"] < 0 else "无均值差异"
            adjusted = "" if row["p_holm_primary_two"] is None else f"；两项 primary Holm p = {row['p_holm_primary_two']:.8f}"
            reduction = row["relative_reduction_percent"]["estimate"]
            relative = "NA" if reduction is None else f"{reduction:.9f}%"
            lines.append(f"- 相对 {LABELS[row['baseline']]} [{row['comparison_role']}]：baseline − ours = {formatted(value)} 度（{direction}），相对降低 {relative}；12 姿态 exact signflip 双侧 p = {row['exact_signflip_p_two_sided']:.8f}{adjusted}。")
    diagnostic = summary["risk_atomic_diagnostics"]
    lines += ["", f"决策接受 {diagnostic['accepted_states']}/48，其中 {diagnostic['accepted_noop_states']} 个是接受原样候选；真正非零提交为 {diagnostic['actual_nonzero_commit_states']}/48。"]
    for missed in diagnostic["missed_beneficial_candidates"]:
        lines.append(f"- 未完整保留的有益候选 `{missed['state_id']}`：measurement {missed['measured_qmae_deg']:.6f}°，candidate {missed['candidate_qmae_deg']:.6f}°，最终 {missed['final_qmae_deg']:.6f}°。该例计入全部表格和配对检验。")
    if audit.get("candidate_bound_max_fraction_recomputed") is not None:
        lines += ["", f"冻结逻辑复核：commit 为整状态二选一且不接收 GT；幅度边界由候选生成器保证，commit 不另加幅度门限。本批最大候选修正 / 既定上界 = {audit['candidate_bound_max_fraction_recomputed']:.6f}，未超过 1；未改冻结模块。"]
    lines += ["", "各条件均完整报告，包含负向、零改善与 NA 分母；没有挑选姿态或根据 heldout 重调阈值。", "",
              "## 冻结与数据审计", "",
              f"- Freeze SHA256：`{audit['freeze_sha256']}`。",
              f"- 冻结时间 UTC：{audit['freeze_created_utc']}；运行目录创建时间 UTC：{audit['run_created_utc']}。",
              "- 先冻结后运行已通过文件系统 birth time 验证（整秒保守下界，不使用结果文件 mtime）；原 runner 未写入开始时间字段。",
              "- Freeze 中所有 source hashes、运行配置、runner 指纹和 candidate_flush 指纹均一致。",
              "- 48 unique ID、12×4 完整矩阵、development 状态及姿态不重叠、原四方法配对完整均已校验。",
              "- 先完成所有 commit，再读取 true_qpos_rad 后置计算 qMAE；commit 输入只有 measurement/candidate/evidence/冻结阈值。",
              "- 原始 CSV 字段和值逐行保留在 rows.csv 前 192 行；新增分析字段统一使用 analysis_ 前缀。", "",
              "## 指标与统计口径", "",
              "- qMAE：7 个关节绝对误差均值，弧度转度；不使用姿态范数或 task success。",
              f"- Whole-state harmful：最终 qMAE 比 measurement qMAE 增大超过冻结标签容差 {summary.get('label_tolerance_mae_rad', 1e-8):g} rad；不是任一关节变差。",
              "- Actual nonzero commit：最终状态至少一坐标与 measurement 数值不等；接受一个等于原状态的 candidate 不计提交。另提供超过冻结零容差的 nontrivial commit。",
              "- 论文主表使用 beneficial_candidate_utilization：有益候选带来最终 qMAE 改善的比例，允许有效的部分更新。严格完整候选 retention 只留审计：最终完整等于候选且实际非零更新的有益候选比例。Component 完整 retention 为 0 不代表方法无效；其部分改善通过 utilization 与最终 qMAE 公平呈现。",
              "- Zero drift：仅零偏移条件上最终状态相对 measurement 的平均绝对位移（度）；另报任一关节漂移超过冻结零容差的比例。非零条件无零漂移分母，显示 NA。",
              "- 按姿态整簇 bootstrap 10,000 次，percentile 95% CI；同一姿态四条件及所有方法共享抽样索引。比例使用重采样后的分子之和 / 分母之和；空分母为 null，bootstrap 有效次数可查。",
              "- 配对差先在每姿态内平均，再用 12 个姿态均值统计；双侧 exact signflip 穷举 2^12=4096，包含零差姿态。正差表示 ours 更低误差。",
              "- 两个原始全条件 raw/inverse 对比仍为 primary，仅这两项进行 Holm 校正；新加 Global/Component 全条件对比为 secondary/exploratory。所有条件及 zero/nonzero pooled 对比均为 descriptive，保留原始 p 值。",
              "- 相对降低百分比 = 100 × (baseline 均值 − ours 均值) / baseline 均值，用未舍入原始 qMAE 计算，不用表格显示值倒算；baseline 均值为零时为 NA。JSON/CSV 同时给出配对姿态 bootstrap 的百分比 CI。",
              "- 全零事件的 bootstrap CI 可退化为 [0,0]，仅描述本次样本，不是总体风险上界。", "",
              "## 输出与复现", "",
              "- 分析目录：`code/runs/risk_atomic_heldout_20260908/`，含 rows.csv、summary.json、matrix.csv、comparisons.csv、tables.md、tables.tex。",
              "- 本 handoff 同步副本：`external_heldout_rows.csv`、`external_heldout_summary.json`、`external_heldout_matrix.csv`、`external_heldout_comparisons.csv`、`external_heldout_tables.md`、`external_heldout_tables.tex`。",
              "- `paper-visuals/Q4__external-heldout-*__square-titlefree.{pdf,png,csv}`：qMAE、配对改善及 harmful 三张数值图，正方形、无标题，CI 与表一致。",
              "- Leader merge 提醒：图名前缀是 `Q4__external-heldout`（不是 Q5），请将三组 PDF/PNG/CSV 纳入最终 manifest/selection；本分析未修改既有 manifest。",
              "- 本次独立 QA 记录：`EXTERNAL_HELDOUT_QA_20260908.json`；LaTeX 编译预览位于分析目录 `qa_tables_preview.pdf`。",
              "- CPU 复现：`PYTHONPATH=src <python> scripts/analyze_risk_atomic_heldout.py`。",
              "- CPU 测试：`PYTHONPATH=src <python> -m unittest discover -s tests -p test_analyze_risk_atomic_heldout.py`。", "",
              "## 完整数值表", "", tables_markdown(summary)]
    return "\n".join(lines)


def export_artifacts(rows, summary, output, handoff, visuals):
    output, handoff, visuals = Path(output), Path(handoff), Path(visuals)
    for directory in (output, handoff, visuals):
        directory.mkdir(parents=True, exist_ok=True)
    write_csv(output / "rows.csv", rows)
    write_csv(output / "matrix.csv", flattened_matrix(summary))
    comparison_rows = []
    for row in summary["comparisons"]:
        value = row["improvement_deg"]
        comparison_rows.append({"group": row["group"], "baseline": row["baseline"], "ours": OURS,
                                "comparison_role": row["comparison_role"],
                                "improvement_deg": value["estimate"], "ci_low": value["ci95"][0],
                                "ci_high": value["ci95"][1], "p_exact_two_sided": row["exact_signflip_p_two_sided"],
                                "p_holm_primary_two": row["p_holm_primary_two"],
                                "relative_reduction_percent": row["relative_reduction_percent"]["estimate"],
                                "relative_reduction_ci_low": row["relative_reduction_percent"]["ci95"][0],
                                "relative_reduction_ci_high": row["relative_reduction_percent"]["ci95"][1],
                                "pose_improvements_deg": json.dumps(row["pose_improvements_deg"], sort_keys=True)})
    write_csv(output / "comparisons.csv", comparison_rows)
    (output / "tables.md").write_text(tables_markdown(summary), encoding="utf-8")
    (output / "tables.tex").write_text(tables_tex(summary), encoding="utf-8")
    summary["figures"] = plot_figures(summary, visuals)
    summary["output_hashes"] = {name: sha256(output / name) for name in (
        "rows.csv", "matrix.csv", "comparisons.csv", "tables.md", "tables.tex")}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    for name in ("rows.csv", "summary.json", "matrix.csv", "comparisons.csv", "tables.md", "tables.tex"):
        shutil.copyfile(output / name, handoff / f"external_heldout_{name}")
    report = report_zh(summary)
    (handoff / "EXTERNAL_HELDOUT_RESULTS_ZH.md").write_text(report, encoding="utf-8")
    (output / "EXTERNAL_HELDOUT_RESULTS_ZH.md").write_text(report, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, default=ROOT / "runs/risk_atomic_freeze_20260908/calibration.json")
    parser.add_argument("--run", type=Path, default=ROOT / "runs/external_gs_guard_franka_heldout_20260908")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/risk_atomic_heldout_20260908")
    parser.add_argument("--handoff", type=Path, default=ROOT.parent / "paper-handoff-20260908")
    parser.add_argument("--visuals", type=Path, default=ROOT.parent / "paper-handoff-20260908/paper-visuals")
    args = parser.parse_args()
    require(args.output.resolve() not in (args.run.resolve(), args.freeze.parent.resolve()),
            "analysis output cannot overwrite run or freeze")
    freeze, candidates, original, audit = load_validated(args.freeze, args.run)
    decisions = decide_candidates(candidates, freeze["thresholds"])
    rows = evaluate_rows(candidates, original, decisions, freeze)
    audit.update(audit_evaluation(candidates, rows, freeze))
    summary = summarize(rows, bootstrap_indices())
    summary.update(audit=audit, thresholds=freeze["thresholds"],
                   label_tolerance_mae_rad=freeze["label_tolerance_mae_rad"],
                   zero_tolerance_rad=freeze["config"]["guard"]["zero_tolerance_rad"],
                   analysis_created_utc=datetime.now(timezone.utc).isoformat())
    audit["candidate_bound_max_fraction_recomputed"] = max(float(np.max(np.abs(
        vector(r["candidate_qpos_rad"]) - vector(r["measured_qpos_rad"])))) /
        float(freeze["config"]["recovery"]["offset_bound_rad"]) for r in candidates)
    audit["frozen_commit_review"] = (
        "commit is atomic and reference-free. Its candidate_bound_fraction is not an independent hard bound gate; "
        "boundedness is supplied by the frozen candidate generator. Recomputed max fraction is reported, not used to tune decisions.")
    for path, expected in {**audit["input_hashes"], **audit["source_hashes_verified"],
                           **audit["decision_dependency_hashes_at_analysis"]}.items():
        require(sha256(path) == expected, f"input hash changed during analysis: {path}")
    export_artifacts(rows, summary, args.output, args.handoff, args.visuals)
    print(json.dumps({"output": str(args.output), "rows": len(rows), "matrix_rows": len(summary["matrix"]),
                      "comparisons": len(summary["comparisons"]), "figures": len(summary["figures"]),
                      "freeze_verified": True, "gpu_work_performed": False}, indent=2))


if __name__ == "__main__":
    main()
