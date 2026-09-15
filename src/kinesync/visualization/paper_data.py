"""Closed adapters for the frozen quantitative paper-evidence runs."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Mapping


@dataclass(frozen=True)
class SeriesPoint:
    """One finite plotted value with its paper-palette semantic."""

    label: str
    value: float
    method: str
    metric_id: str
    ci_low: float | None = None
    ci_high: float | None = None
    x: float | None = None

    def __post_init__(self) -> None:
        if not self.label or not self.method or not self.metric_id or not math.isfinite(self.value):
            raise ValueError("series points require labels, metric IDs, and finite values")
        if (self.ci_low is None) != (self.ci_high is None):
            raise ValueError("confidence intervals require both low and high values")
        if self.ci_low is not None and self.ci_high is not None and (
            not math.isfinite(self.ci_low)
            or not math.isfinite(self.ci_high)
            or self.ci_low > self.value
            or self.value > self.ci_high
        ):
            raise ValueError("confidence intervals must be finite and contain their estimate")
        if self.x is not None and not math.isfinite(self.x):
            raise ValueError("series point x coordinates must be finite")


@dataclass(frozen=True)
class SeriesMetadata:
    """Source-validated observation and cluster counts for one plotted series."""

    observation_count: int
    cluster_count: int
    cluster_unit: str

    def __post_init__(self) -> None:
        if (
            isinstance(self.observation_count, bool)
            or isinstance(self.cluster_count, bool)
            or not isinstance(self.observation_count, int)
            or not isinstance(self.cluster_count, int)
            or self.observation_count <= 0
            or self.cluster_count <= 0
            or not isinstance(self.cluster_unit, str)
            or not self.cluster_unit.strip()
        ):
            raise ValueError("series metadata requires positive counts and a cluster unit")


@dataclass(frozen=True)
class StageEvidence:
    """Immutable, hash-bound values extracted from one allowed frozen run."""

    stage: str
    run_id: str
    run_path: Path
    evidence_semantic: str
    source_hashes: Mapping[str, str]
    cluster_unit: str
    series: Mapping[str, tuple[SeriesPoint, ...]]
    series_metadata: Mapping[str, SeriesMetadata]

    def value(self, series_name: str, label: str, group: str | None = None) -> float:
        """Return one named evidence value; labels are deliberately closed."""

        series_name = {"latency_ms": "latency"}.get(series_name, series_name)
        try:
            points = self.series[series_name]
        except KeyError as error:
            raise ValueError(f"unknown evidence series: {series_name}") from error
        visible_label = f"{group} {label}" if group is not None else label
        matches = [point.value for point in points if point.label == visible_label]
        if len(matches) != 1:
            raise ValueError(f"expected exactly one {series_name!r} value for {visible_label!r}")
        return matches[0]


@dataclass(frozen=True)
class _FrozenStage:
    run_id: str
    evidence_semantic: str
    files: Mapping[str, str]


_FROZEN_STAGES: Mapping[str, _FrozenStage] = MappingProxyType(
    {
        "RT2": _FrozenStage(
            "20260815_rt2f_full24_eval48_v2",
            "formal",
            MappingProxyType(
                {
                    "manifest.json": "5ec3686691be35abe8c3bef2ae369fd60024e8c0932d3f7b12895bb62b2eedc6",
                    "metrics.json": "7003a8078aa16045671b015742a4749c395cab98d186df62f879e3a6e373b666",
                    "evaluation_rows.csv": "3dd03ab03845f67e82823ae8d17d28ae86be05b0cb9d7de3c62d4fe4cc166522",
                }
            ),
        ),
        "RT3": _FrozenStage(
            "20260815_rt3c_fresh48_v1",
            "formal",
            MappingProxyType(
                {
                    "manifest.json": "21a97aef4225657e38421cc4c55587c72fde60ece9b89bf10625eb63ccc3895a",
                    "metrics.json": "189c954e2af29d23acd0a7ca3db68551ae3e5dd6bc1ba0ac4c97e71ba010e6a2",
                    "results.csv": "e16865741988ba06f4ca17ebd27b39ecb06d65d2f72dcd48ee108e46653df198",
                }
            ),
        ),
        "RT6": _FrozenStage(
            "20260823_rt6_formal_fresh32_v1",
            "formal",
            MappingProxyType(
                {
                    "manifest.json": "9cb052332d58bf882a1ca6cbbbbea9a3b007a80089a5610000e522bb3e6b32eb",
                    "metrics.json": "31dc1293a3ac3a372447d3e92b02675b9deb45034705bd7534a2af6bd904b13b",
                    "results.csv": "2e1e6276bd921040ae854fe16bf426e6d1970b7204d7dfc63523250700eee068",
                    "component_results.csv": "32b6f46306d39b58225691499c6b1452b59e8ace38c62b9712431dc03557aa13",
                    "parity_report.json": "56ff910c109ba0136a3412489ebd2988231515dc8b177a357762b090bb8787f2",
                }
            ),
        ),
        "RT7": _FrozenStage(
            "20260824_rt7_component_replay32_v3",
            "development_only",
            MappingProxyType(
                {
                    "manifest.json": "ed06fa1b79d5a7b6f79d30481df94bc083234c9059317e3cd848a64b6be4f238",
                    "metrics.json": "9477f61a5dae86861fe97705f5f4c2e775411e6a0ba63a89a0744c91b8af8fbd",
                    "results.csv": "1f3cf20847f778cdbe050126326ff094ce2e322474cfbf437cc0fca2cb330df8",
                    "component_results.csv": "a693b02f3cc1903ed536cf88a211017edcd8da2351440d6370ede8593dbfdb06",
                    "parity_report.json": "a162b9b026a6c1c27fa256186580227cb4279fa9ecda73020c880a6ea9c59a31",
                }
            ),
        ),
        "RT8": _FrozenStage(
            "20260824_rt8_paired_real_shadow_v2_shared_guard",
            "formal_recorded_real",
            MappingProxyType(
                {
                    "manifest.json": "73c599600d4f2f8b7e208e9f5ac978e1d2841ba96363447ed32ba4d361052779",
                    "metrics.json": "4561c605866cf7dcb7db4885206f2a5ee4c2caf42b40ce32cae3b6dc9e5bdcf3",
                    "results.csv": "2b7737101f20bc223e3ed9876787c4d3f37ad2b245dbc0c06ca01c57e2b0d55a",
                }
            ),
        ),
        "RT9": _FrozenStage(
            "20260825_rt9_take_pens_temporal_v1",
            "held_out_recorded_telemetry",
            MappingProxyType(
                {
                    "manifest.json": "55ef67003b6c78097cdb10bf9e859ad629025477bde078dade47c63409e1598e",
                    "metrics.json": "1c49cf88aa3de57c4183132dd246623701e667cb465099e4218dbcabb0b1a775",
                    "results.csv": "5a5b6bd734865f2940eeae41e25adc7fd569bf53650502a42396474f569bbec1",
                }
            ),
        ),
        "RT10": _FrozenStage(
            "20260825_rt10_live_shadow_preflight_v2",
            "software_preflight",
            MappingProxyType(
                {
                    "manifest.json": "c8d4e8f4952e92a5a245a14d5221e56e56c1f99eee6afe8f39b87d7fd10cd1b2",
                    "metrics.json": "63447b62d09e53ddf1ac63627f293604a2c6c1dc18709220c872254fdb149c8e",
                    "events.jsonl": "dd72a98192645fed29bbaf39fefc9692d8118080999a09d9b7bdd428b7481b9b",
                    "safety.json": "9123c6d63e0ed3e4433e1a29c57e91a5d091a6f4193e1c2317951d00dcc76f7b",
                }
            ),
        ),
    }
)

_RT2_RECEIPT_NAME = "RT2F_FORMAL_RESULTS.md"
_RT2_RECEIPT_SHA256 = "eb51b9dd8711984ddcaba49235df49ed6173b23d10f4d3f446c78b487be3a9c2"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"nonfinite JSON constant {value!r} in {path.name}")

    try:
        return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    except (json.JSONDecodeError, OSError) as error:
        raise ValueError(f"invalid JSON source: {path}") from error


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    return converted


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _read_csv(path: Path) -> tuple[Mapping[str, str], ...]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = tuple(csv.DictReader(handle))
    except OSError as error:
        raise ValueError(f"missing CSV source: {path}") from error
    if not rows or any(None in row for row in rows):
        raise ValueError(f"invalid or empty CSV source: {path}")
    return rows


def _frozen_source_hashes(stage: str, run_path: Path) -> Mapping[str, str]:
    frozen = _FROZEN_STAGES[stage]
    if run_path.name != frozen.run_id:
        raise ValueError(f"{stage} requires the exact frozen run ID {frozen.run_id}")
    hashes: dict[str, str] = {}
    for name, expected in frozen.files.items():
        source = run_path / name
        if not source.is_file():
            raise ValueError(f"missing frozen source: {source}")
        actual = _sha256(source)
        if actual != expected:
            raise ValueError(f"frozen source SHA-256 changed: {source}")
        hashes[name] = actual
    if stage == "RT2":
        receipt = run_path.parents[1] / "project" / _RT2_RECEIPT_NAME
        if not receipt.is_file() or _sha256(receipt) != _RT2_RECEIPT_SHA256:
            raise ValueError("RT2 paired baseline receipt SHA-256 changed")
        hashes[_RT2_RECEIPT_NAME] = _RT2_RECEIPT_SHA256
    return MappingProxyType(dict(sorted(hashes.items())))


def _point(
    label: str,
    value: object,
    method: str,
    metric_id: str,
    ci: Mapping[str, Any] | None = None,
    *,
    x: float | None = None,
) -> SeriesPoint:
    if ci is None:
        return SeriesPoint(label, _finite(value, label), method, metric_id, x=x)
    return SeriesPoint(
        label,
        _finite(value, label),
        method,
        metric_id,
        _finite(ci.get("ci_low"), f"{label} ci_low"),
        _finite(ci.get("ci_high"), f"{label} ci_high"),
        x,
    )


def _rt2_series(run_path: Path, metrics: Mapping[str, Any]) -> Mapping[str, tuple[SeriesPoint, ...]]:
    receipt = (run_path.parents[1] / "project" / _RT2_RECEIPT_NAME).read_text(encoding="utf-8")
    match = re.search(
        r"Paired-camera metric.*?Controlled success \| ([0-9]+)/([0-9]+) \(([^)]+)%\) \| ([0-9]+)/([0-9]+) \(([^)]+)%\)",
        receipt,
        flags=re.DOTALL,
    )
    reduction_match = re.search(r"Median state-error reduction \| ([0-9.]+)% \| ([0-9.]+)%", receipt)
    if match is None or reduction_match is None:
        raise ValueError("RT2 paired baseline receipt has no closed paired-result entries")
    rows = _read_csv(run_path / "evaluation_rows.csv")
    paired = [row for row in rows if row["camera_mode"] == "head+extra" and row["trial_type"] == "controlled_offset"]
    successes = sum(row["recovery_success"] == "True" for row in paired)
    receipt_successes, receipt_count = int(match.group(4)), int(match.group(5))
    if len(paired) != receipt_count or successes != receipt_successes:
        raise ValueError("RT2 paired rows no longer match the frozen paired receipt")
    paired_success = round(100.0 * successes / len(paired), 2)
    if abs(paired_success - float(match.group(6))) > 0.005:
        raise ValueError("RT2 paired CSV and baseline receipt disagree")
    reduction = 100.0 * _finite(metrics.get("paired_camera_median_state_error_reduction"), "RT2 reduction")
    return MappingProxyType(
        {
            "paired_success": (
                _point("RT1-O", float(match.group(3)), "primary_baseline", "recovery_success_rate"),
                _point("RT2-F", paired_success, "ours_full", "recovery_success_rate"),
            ),
            "paired_state_error": (
                _point("RT1-O", float(reduction_match.group(1)), "primary_baseline", "state_error_reduction"),
                _point("RT2-F", reduction, "ours_full", "state_error_reduction"),
            ),
            "visual_gain": (
                _point("RT2-F IoU", 100.0 * _finite(metrics.get("mean_mask_iou_gain"), "RT2 IoU"), "primary_baseline", "mask_iou_gain"),
                _point("RT2-F boundary-F1", 100.0 * _finite(metrics.get("mean_boundary_f1_gain"), "RT2 boundary F1"), "ours_full", "boundary_f1_gain"),
            ),
        }
    )


def _method_series(metrics: Mapping[str, Any], key: str, labels: Mapping[str, tuple[str, str]]) -> tuple[SeriesPoint, ...]:
    summaries = _mapping(metrics.get("method_summaries"), "method_summaries")
    points = []
    for source_name, (label, method) in labels.items():
        summary = _mapping(summaries.get(source_name), source_name)
        points.append(_point(label, summary.get(key), method, key))
    return tuple(points)


def _rt3_series(metrics: Mapping[str, Any]) -> Mapping[str, tuple[SeriesPoint, ...]]:
    labels = {
        "rt1o_unfactorized": ("RT1-O", "primary_baseline"),
        "rt2f_unguarded": ("RT2-F", "secondary_baseline"),
        "rt3_gain_only": ("Gain-only", "ours_weak"),
        "rt3_gradient_only": ("Gradient-only", "ours_ablation"),
        "rt3_full": ("Guarded", "ours_full"),
    }
    return MappingProxyType(
        {
            "zero_stability": _method_series(metrics, "zero_control_stability", labels),
            "coverage": _method_series(metrics, "paired_controlled_commit_coverage", labels),
            "visual_gain": _method_series(metrics, "mean_mask_iou_gain", labels),
        }
    )


def _rt6_series(metrics: Mapping[str, Any]) -> Mapping[str, tuple[SeriesPoint, ...]]:
    labels = {
        "rt2f_unguarded": ("RT2-F", "primary_baseline"),
        "rt3c_guarded": ("RT3-C", "secondary_baseline"),
        "rt6dm_synchronized": ("RT6-DM", "ours_full"),
    }
    primary = _mapping(metrics.get("primary_metrics"), "primary_metrics")
    parity = _mapping(metrics.get("parity_report"), "parity_report")
    return MappingProxyType(
        {
            "matched_success": _method_series(metrics, "trial_success_rate", labels),
            "operating_point": (
                _point("Coverage", primary.get("controlled_commit_coverage"), "primary_baseline", "controlled_commit_coverage"),
                _point("Retention", primary.get("successful_candidate_retention"), "ours_ablation", "successful_candidate_retention"),
                _point("Precision", primary.get("committed_precision"), "ours_full", "committed_precision"),
            ),
            "parity_error": (
                _point("qpos", parity.get("max_qpos_error"), "reference", "max_qpos_error"),
                _point("translation", parity.get("max_translation_error_m"), "primary_baseline", "max_translation_error_m"),
                _point("rotation", parity.get("max_rotation_abs_error"), "ours_full", "max_rotation_abs_error"),
            ),
        }
    )


def _rt7_series(run_path: Path, metrics: Mapping[str, Any]) -> Mapping[str, tuple[SeriesPoint, ...]]:
    labels = {
        "rt6dm_synchronized": ("RT6-DM", "primary_baseline"),
        "rt7cv_component_verified": ("RT7-CV", "ours_full"),
    }
    primary = _mapping(metrics.get("primary_metrics"), "primary_metrics")
    source_hashes = _mapping(metrics.get("source_hashes"), "RT7 source hashes")
    rt6_metrics_path = run_path.parent / "20260823_rt6_formal_fresh32_v1" / "metrics.json"
    expected_rt6_metrics = source_hashes.get("source_rt6_metrics_sha256")
    if not isinstance(expected_rt6_metrics, str) or _sha256(rt6_metrics_path) != expected_rt6_metrics:
        raise ValueError("RT7 frozen RT6 metrics source SHA-256 changed")
    rt6_metrics = _mapping(_load_json(rt6_metrics_path), "RT7 source RT6 metrics")
    rt6_primary = _mapping(rt6_metrics.get("primary_metrics"), "RT7 source RT6 primary metrics")
    return MappingProxyType(
        {
            "retention": (
                _point("RT6-DM", rt6_primary.get("successful_candidate_retention"), "primary_baseline", "successful_candidate_retention"),
                _point("RT7-CV", primary.get("successful_candidate_retention"), "ours_full", "successful_candidate_retention"),
            ),
            "operating_point": (
                _point("Coverage", primary.get("controlled_commit_coverage"), "primary_baseline", "controlled_commit_coverage"),
                _point("Retention", primary.get("successful_candidate_retention"), "ours_ablation", "successful_candidate_retention"),
                _point("Precision", primary.get("committed_precision"), "ours_full", "committed_precision"),
            ),
            "matched_success": _method_series(metrics, "trial_success_rate", labels),
        }
    )


def _rt8_series(metrics: Mapping[str, Any]) -> Mapping[str, tuple[SeriesPoint, ...]]:
    backends = _mapping(metrics.get("backends"), "backends")
    backend_names = {"a_p2_component_gs": "A-P2", "route_b_hybrid": "Route-B"}
    qmae: list[SeriesPoint] = []
    recovery: list[SeriesPoint] = []
    operating: list[SeriesPoint] = []
    qmae_ci: list[SeriesPoint] = []
    recovery_ci: list[SeriesPoint] = []
    for backend_id, backend_label in backend_names.items():
        backend = _mapping(backends.get(backend_id), backend_id)
        methods = _mapping(backend.get("test_methods"), f"{backend_id} test_methods")
        for method_id, label, semantic in (
            ("delayed_state", "Delayed", "primary_baseline"),
            ("unguarded_visual_retrieval", "Unguarded", "ours_ablation"),
            ("kinesync_guarded_retrieval", "Guarded", "ours_full"),
            ("paired_state_oracle", "Oracle", "oracle"),
        ):
            method = _mapping(methods.get(method_id), f"{backend_id} {method_id}")
            qmae.append(_point(f"{backend_label} {label}", method.get("qmae_rad"), semantic, "qmae_rad"))
            recovery.append(_point(f"{backend_label} {label}", method.get("recovery_success"), semantic, "recovery_success"))
        guarded = _mapping(methods.get("kinesync_guarded_retrieval"), f"{backend_id} guarded")
        unguarded = _mapping(methods.get("unguarded_visual_retrieval"), f"{backend_id} unguarded")
        operating.extend(
            (
                _point(f"{backend_label} unguarded precision", unguarded.get("commit_precision"), "primary_baseline", "commit_precision"),
                _point(f"{backend_label} guarded precision", guarded.get("commit_precision"), "ours_full", "commit_precision"),
                _point(f"{backend_label} guarded coverage", guarded.get("controlled_coverage"), "ours_ablation", "controlled_coverage"),
            )
        )
        bootstrap = _mapping(backend.get("episode_cluster_bootstrap"), f"{backend_id} bootstrap")
        qmae_stat = _mapping(bootstrap.get("qmae_reduction_rad"), f"{backend_id} qmae CI")
        recovery_stat = _mapping(bootstrap.get("recovery_success_gain"), f"{backend_id} recovery CI")
        qmae_ci.append(_point(f"{backend_label} reduction", qmae_stat.get("estimate"), "primary_baseline" if backend_id == "a_p2_component_gs" else "ours_full", "qmae_reduction_rad", qmae_stat))
        recovery_ci.append(_point(f"{backend_label} gain", recovery_stat.get("estimate"), "primary_baseline" if backend_id == "a_p2_component_gs" else "ours_full", "recovery_success_gain", recovery_stat))
    return MappingProxyType({"qmae": tuple(qmae), "recovery": tuple(recovery), "operating_point": tuple(operating), "qmae_ci": tuple(qmae_ci), "recovery_ci": tuple(recovery_ci)})


def _rt9_series(metrics: Mapping[str, Any]) -> Mapping[str, tuple[SeriesPoint, ...]]:
    summary = _mapping(metrics.get("summary"), "summary")
    methods = summary.get("methods")
    if not isinstance(methods, list):
        raise ValueError("RT9 requires method summaries")
    semantic = {
        "raw_nearest": ("Raw nearest", "secondary_baseline"),
        "raw_linear": ("Raw linear", "primary_baseline"),
        "motion_unguarded": ("Unguarded", "ours_ablation"),
        "kinesync_guarded": ("Guarded", "ours_full"),
        "oracle_offset": ("Oracle", "oracle"),
    }
    by_method: dict[str, Mapping[str, Any]] = {}
    for entry in methods:
        mapping = _mapping(entry, "RT9 method summary")
        method = mapping.get("method")
        if not isinstance(method, str):
            raise ValueError("RT9 method summary has no method")
        by_method[method] = mapping
    if set(by_method) != set(semantic):
        raise ValueError("RT9 method summaries do not match the closed method allowlist")
    def points(metric: str) -> tuple[SeriesPoint, ...]:
        return tuple(_point(semantic[name][0], by_method[name].get(metric), semantic[name][1], metric) for name in semantic)
    guarded = by_method["kinesync_guarded"]
    raw = by_method["raw_linear"]
    def trends(group_name: str, x_name: str) -> tuple[SeriesPoint, ...]:
        groups = summary.get(group_name)
        if not isinstance(groups, list):
            raise ValueError(f"RT9 requires {group_name}")
        selected = [entry for entry in groups if _mapping(entry, group_name).get("metrics", {}).get("method") in {"raw_linear", "kinesync_guarded"}]
        output = []
        for entry in selected:
            mapping = _mapping(entry, group_name)
            item_metrics = _mapping(mapping.get("metrics"), f"{group_name} metrics")
            source_method = item_metrics.get("method")
            if source_method not in semantic:
                raise ValueError(f"RT9 unknown trend method: {source_method}")
            output.append(
                _point(
                    f"{mapping.get('value')} {x_name} {semantic[source_method][0]}",
                    item_metrics.get("qmae_rad"),
                    semantic[source_method][1],
                    "qmae_rad",
                    x=_finite(mapping.get("value"), f"{group_name} value"),
                )
            )
        return tuple(output)
    return MappingProxyType(
        {
            "qmae": points("qmae_rad"),
            "recovery": points("recovery_success_rate"),
            "operating_point": (
                _point("Raw linear coverage", raw.get("controlled_coverage"), "primary_baseline", "controlled_coverage"),
                _point("Guarded coverage", guarded.get("controlled_coverage"), "ours_full", "controlled_coverage"),
                _point("Guarded precision", guarded.get("commit_precision"), "ours_ablation", "commit_precision"),
            ),
            "jitter_qmae": trends("jitter_groups", "ms"),
            "dropout_qmae": trends("dropout_groups", "dropout"),
        }
    )


def _rt10_series(metrics: Mapping[str, Any], run_path: Path) -> Mapping[str, tuple[SeriesPoint, ...]]:
    events = []
    for line in (run_path / "events.jsonl").read_text(encoding="utf-8").splitlines():
        events.append(_mapping(_load_json_line(line), "RT10 event"))
    if len(events) != int(_finite(metrics.get("event_count"), "RT10 event count")):
        raise ValueError("RT10 event stream count does not match metrics")
    latency = sorted(_finite(event.get("processing_latency_ns"), "processing latency") / 1_000_000.0 for event in events)
    head_aux = sorted(abs(_finite(event.get("head_capture_ns"), "head capture") - _finite(event.get("auxiliary_capture_ns"), "auxiliary capture")) / 1_000_000.0 for event in events)
    camera_state = sorted(max(abs(_finite(event.get("head_capture_ns"), "head capture") - _finite(event.get("state_capture_ns"), "state capture")), abs(_finite(event.get("auxiliary_capture_ns"), "auxiliary capture") - _finite(event.get("state_capture_ns"), "state capture"))) / 1_000_000.0 for event in events)
    median_index = (len(events) - 1) // 2
    return MappingProxyType(
        {
            "latency": (
                _point("p50", _finite(metrics.get("p50_processing_latency_ns"), "RT10 p50") / 1_000_000.0, "primary_baseline", "p50_processing_latency_ms"),
                _point("p95", _finite(metrics.get("p95_processing_latency_ns"), "RT10 p95") / 1_000_000.0, "ours_full", "p95_processing_latency_ms"),
                _point("max", latency[-1], "ours_ablation", "max_processing_latency_ms"),
            ),
            "head_aux_skew": (
                _point("median", head_aux[median_index], "primary_baseline", "median_head_aux_skew_ms"),
                _point("max", head_aux[-1], "ours_full", "max_head_aux_skew_ms"),
            ),
            "camera_state_skew": (
                _point("median", camera_state[median_index], "primary_baseline", "median_camera_state_skew_ms"),
                _point("max", camera_state[-1], "ours_full", "max_camera_state_skew_ms"),
            ),
        }
    )


def _load_json_line(line: str) -> Any:
    return json.loads(line, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"nonfinite JSON constant {value!r}")))


def _count(value: object, name: str) -> int:
    number = _finite(value, name)
    if not number.is_integer() or number <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(number)


def _repeat_metadata(series: Mapping[str, tuple[SeriesPoint, ...]], metadata: SeriesMetadata) -> Mapping[str, SeriesMetadata]:
    return MappingProxyType({name: metadata for name in series})


def _series_metadata(
    stage: str, run_path: Path, metrics: Mapping[str, Any], series: Mapping[str, tuple[SeriesPoint, ...]]
) -> Mapping[str, SeriesMetadata]:
    """Validate source counts before exposing per-series paper metadata."""

    if stage == "RT2":
        controlled = _count(metrics.get("controlled_case_count"), "RT2 controlled_case_count")
        rows = _read_csv(run_path / "evaluation_rows.csv")
        paired = [row for row in rows if row["camera_mode"] == "head+extra" and row["trial_type"] == "controlled_offset"]
        if len(paired) != 21 or controlled != 45:
            raise ValueError("RT2 source counts no longer match frozen evidence")
        return MappingProxyType(
            {
                "paired_success": SeriesMetadata(21, 21, "paired validation case"),
                "paired_state_error": SeriesMetadata(21, 21, "paired validation case"),
                "visual_gain": SeriesMetadata(controlled, controlled, "controlled validation case"),
            }
        )
    if stage == "RT3":
        case_count = _count(metrics.get("case_count"), "RT3 case_count")
        state_ids = metrics.get("validation_state_ids")
        if not isinstance(state_ids, list) or len(state_ids) != case_count or len(set(state_ids)) != case_count:
            raise ValueError("RT3 validation states do not match case_count")
        return _repeat_metadata(series, SeriesMetadata(case_count, case_count, "validation case"))
    if stage in {"RT6", "RT7"}:
        trial_count = _count(metrics.get("trial_count"), f"{stage} trial_count")
        state_count = _count(metrics.get("unique_state_count"), f"{stage} unique_state_count")
        rows = _read_csv(run_path / "results.csv")
        expected_rows = _count(metrics.get("method_row_count"), f"{stage} method_row_count")
        if len(rows) != expected_rows or len({row["state_id"] for row in rows}) != state_count:
            raise ValueError(f"{stage} source rows do not match trial/state metadata")
        return _repeat_metadata(series, SeriesMetadata(trial_count, state_count, "state"))
    if stage == "RT8":
        result_count = _count(metrics.get("result_row_count"), "RT8 result_row_count")
        if len(_read_csv(run_path / "results.csv")) != result_count:
            raise ValueError("RT8 result rows do not match metrics")
        backends = _mapping(metrics.get("backends"), "RT8 backends")
        for backend_name in ("a_p2_component_gs", "route_b_hybrid"):
            backend = _mapping(backends.get(backend_name), backend_name)
            episode_count = _count(backend.get("test_episode_count"), f"{backend_name} test_episode_count")
            bootstrap = _mapping(backend.get("episode_cluster_bootstrap"), f"{backend_name} bootstrap")
            if episode_count != 10 or _count(bootstrap.get("cluster_count"), f"{backend_name} cluster_count") != episode_count:
                raise ValueError("RT8 episode clusters do not match frozen evidence")
        return _repeat_metadata(series, SeriesMetadata(result_count, 10, "test episode"))
    if stage == "RT9":
        trial_count = _count(metrics.get("test_trial_count"), "RT9 test_trial_count")
        row_count = _count(metrics.get("test_method_row_count"), "RT9 test_method_row_count")
        if len(_read_csv(run_path / "results.csv")) != row_count:
            raise ValueError("RT9 result rows do not match metrics")
        summary = _mapping(metrics.get("summary"), "RT9 summary")
        methods = summary.get("methods")
        if not isinstance(methods, list) or not methods:
            raise ValueError("RT9 method summaries are missing")
        for entry in methods:
            method = _mapping(entry, "RT9 method summary")
            if _count(method.get("condition_count"), "RT9 condition_count") != trial_count or _count(method.get("trajectory_count"), "RT9 trajectory_count") != 2:
                raise ValueError("RT9 summary counts do not match frozen evidence")
        return _repeat_metadata(series, SeriesMetadata(trial_count, 2, "trajectory"))
    event_count = _count(metrics.get("event_count"), "RT10 event_count")
    lines = (run_path / "events.jsonl").read_text(encoding="utf-8").splitlines()
    if len(lines) != event_count:
        raise ValueError("RT10 event rows do not match metrics")
    return _repeat_metadata(series, SeriesMetadata(event_count, event_count, "event"))


def load_stage_evidence(stage: str, run_path: Path) -> StageEvidence:
    """Read one approved frozen run and reject every unknown or altered source."""

    if stage not in _FROZEN_STAGES:
        raise ValueError(f"unsupported frozen stage: {stage}")
    path = Path(run_path).resolve()
    frozen = _FROZEN_STAGES[stage]
    source_hashes = _frozen_source_hashes(stage, path)
    manifest = _mapping(_load_json(path / "manifest.json"), "manifest")
    if manifest.get("run_id") != frozen.run_id:
        raise ValueError(f"{stage} manifest run_id is not the frozen run ID")
    metrics = _mapping(_load_json(path / "metrics.json"), "metrics")
    if stage == "RT2":
        series, cluster_unit = _rt2_series(path, metrics), "paired_state"
    elif stage == "RT3":
        series, cluster_unit = _rt3_series(metrics), "validation_case"
    elif stage == "RT6":
        series, cluster_unit = _rt6_series(metrics), "state"
    elif stage == "RT7":
        series, cluster_unit = _rt7_series(path, metrics), "replayed_state"
        source_hashes = MappingProxyType(
            dict(
                sorted(
                    {
                        **source_hashes,
                        "source_rt6_metrics.json": _sha256(
                            path.parent / "20260823_rt6_formal_fresh32_v1" / "metrics.json"
                        ),
                    }.items()
                )
            )
        )
    elif stage == "RT8":
        series, cluster_unit = _rt8_series(metrics), "episode"
    elif stage == "RT9":
        series, cluster_unit = _rt9_series(metrics), "trajectory"
    else:
        series, cluster_unit = _rt10_series(metrics, path), "event"
    metadata = _series_metadata(stage, path, metrics, series)
    if set(metadata) != set(series):
        raise ValueError("every evidence series must have validated metadata")
    return StageEvidence(stage, frozen.run_id, path, frozen.evidence_semantic, source_hashes, cluster_unit, series, metadata)


__all__ = ["SeriesMetadata", "SeriesPoint", "StageEvidence", "load_stage_evidence"]
