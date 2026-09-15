"""Title-free quantitative atoms driven solely by frozen StageEvidence."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import logging
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np

from .paper_contracts import ComparisonMode, EvidenceLevel, PaperArtifactRecord
from .paper_data import SeriesPoint, StageEvidence, load_stage_evidence
from .paper_style import METHOD_COLORS, publication_rc_context, save_square_quantitative


FROZEN_QUANTITATIVE_IDS = (
    "Q-RT2-01", "Q-RT2-02", "Q-RT2-03",
    "Q-RT3-01", "Q-RT3-02", "Q-RT3-03",
    "Q-RT6-01", "Q-RT6-02", "Q-RT6-03",
    "Q-RT7-01", "Q-RT7-02", "Q-RT7-03",
    "Q-RT8-01", "Q-RT8-02", "Q-RT8-03", "Q-RT8-04", "Q-RT8-05",
    "Q-RT9-01", "Q-RT9-02", "Q-RT9-03", "Q-RT9-04", "Q-RT9-05",
    "Q-RT10-01", "Q-RT10-02", "Q-RT10-03",
)
_COMPARISON_DIRECTIONS = frozenset({"higher_is_better", "lower_is_better"})
_PREFLIGHT_OUTPUT_HASH = "0" * 64


@dataclass(frozen=True)
class QuantitativeFigureSpec:
    artifact_id: str
    stage: str
    run_id: str
    run_path: Path
    claim: str
    evidence_semantic: str
    method: str
    baseline: str
    series: str
    y_label: str
    observation_count: int
    n: int
    cluster_unit: str
    ci_declared: bool
    comparison_mode: ComparisonMode | str
    comparison_metric_id: str | None
    comparison_direction: str | None
    primary_baseline_label: str | None
    primary_method_label: str | None
    plot_kind: str = "bar"

    def __post_init__(self) -> None:
        if not self.artifact_id.startswith(f"Q-{self.stage}-"):
            raise ValueError("quantitative artifact ID must be scoped to its stage")
        if not self.claim.strip() or not self.evidence_semantic.strip() or not self.y_label.strip():
            raise ValueError("quantitative specs require nonempty semantic text")
        if self.method not in METHOD_COLORS or self.baseline not in METHOD_COLORS:
            raise ValueError("quantitative spec uses an unknown method semantic")
        if (
            isinstance(self.observation_count, bool)
            or isinstance(self.n, bool)
            or not isinstance(self.observation_count, int)
            or not isinstance(self.n, int)
            or self.observation_count <= 0
            or self.n <= 0
            or not self.cluster_unit.strip()
        ):
            raise ValueError("quantitative specs require source counts and a clustering unit")
        if self.plot_kind not in {"bar", "line"}:
            raise ValueError("quantitative plot kind is closed")
        try:
            mode = ComparisonMode(self.comparison_mode)
        except (TypeError, ValueError) as error:
            raise ValueError(f"unsupported comparison_mode: {self.comparison_mode!r}") from error
        object.__setattr__(self, "comparison_mode", mode)
        if mode is ComparisonMode.METHOD_COMPARISON:
            if not isinstance(self.comparison_metric_id, str) or not self.comparison_metric_id.strip():
                raise ValueError("method_comparison requires comparison_metric_id")
            if self.comparison_direction not in _COMPARISON_DIRECTIONS:
                raise ValueError("method_comparison requires a closed comparison_direction")
            if not isinstance(self.primary_baseline_label, str) or not self.primary_baseline_label.strip():
                raise ValueError("method_comparison requires primary_baseline_label")
            if not isinstance(self.primary_method_label, str) or not self.primary_method_label.strip():
                raise ValueError("method_comparison requires primary_method_label")
        elif any(
            value is not None
            for value in (
                self.comparison_metric_id,
                self.comparison_direction,
                self.primary_baseline_label,
                self.primary_method_label,
            )
        ):
            raise ValueError("descriptive_statistics must not declare primary comparison fields")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _config_mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _resolve_run_path(raw_path: object, config_path: Path, run_root: Path | None) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError("source run_path must be a nonempty repository-relative path")
    relative_path = Path(raw_path)
    if relative_path.is_absolute():
        raise ValueError("source run_path must be repository-relative")
    root = Path(run_root).resolve() if run_root is not None else config_path.resolve().parents[1]
    candidate = (root / relative_path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ValueError("source run_path must remain within the configured repository root") from error
    return candidate


def load_quantitative_specs(
    config_path: Path, *, run_root: Path | None = None
) -> tuple[QuantitativeFigureSpec, ...]:
    """Load JSON-as-YAML configuration without adding a YAML dependency."""

    config_path = Path(config_path)
    try:
        document = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"paper visual config must be JSON-compatible YAML: {config_path}") from error
    root = _config_mapping(document, "paper visual config")
    sources = _config_mapping(root.get("sources"), "sources")
    raw_figures = root.get("figures")
    if set(sources) != {artifact_id.split("-")[1] for artifact_id in FROZEN_QUANTITATIVE_IDS}:
        raise ValueError("paper visual config must declare exactly the frozen evidence stages")
    if not isinstance(raw_figures, list) or len(raw_figures) != len(FROZEN_QUANTITATIVE_IDS):
        raise ValueError("paper visual config must contain exactly 25 quantitative figures")
    specs = []
    for raw in raw_figures:
        figure = _config_mapping(raw, "figure")
        stage = figure.get("stage")
        if not isinstance(stage, str):
            raise ValueError("figure stage must be a string")
        source = _config_mapping(sources.get(stage), f"source for {stage}")
        hashes = _config_mapping(source.get("source_hashes"), f"source hashes for {stage}")
        if not hashes or any(
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in hashes.values()
        ):
            raise ValueError("every configured source must use a full SHA-256")
        ci_declared = figure.get("ci_declared")
        if not isinstance(ci_declared, bool):
            raise ValueError("ci_declared must be a boolean")
        spec = QuantitativeFigureSpec(
            artifact_id=str(figure.get("artifact_id", "")),
            stage=stage,
            run_id=str(source.get("run_id", "")),
            run_path=_resolve_run_path(source.get("run_path"), config_path, run_root),
            claim=str(figure.get("claim", "")),
            evidence_semantic=str(figure.get("evidence_semantic", "")),
            method=str(figure.get("method", "")),
            baseline=str(figure.get("baseline", "")),
            series=str(figure.get("series", "")),
            y_label=str(figure.get("y_label", "")),
            observation_count=int(figure.get("observation_count", 0)),
            n=int(figure.get("n", 0)),
            cluster_unit=str(figure.get("cluster_unit", "")),
            ci_declared=ci_declared,
            comparison_mode=figure.get("comparison_mode"),
            comparison_metric_id=figure.get("comparison_metric_id"),
            comparison_direction=figure.get("comparison_direction"),
            primary_baseline_label=figure.get("primary_baseline_label"),
            primary_method_label=figure.get("primary_method_label"),
            plot_kind=str(figure.get("plot_kind", "bar")),
        )
        if source.get("evidence_semantic") != spec.evidence_semantic:
            raise ValueError("figure evidence semantic must match its frozen source")
        evidence = load_stage_evidence(spec.stage, spec.run_path)
        metadata = evidence.series_metadata.get(spec.series)
        if metadata is None or (
            metadata.observation_count != spec.observation_count
            or metadata.cluster_count != spec.n
            or metadata.cluster_unit != spec.cluster_unit
        ):
            raise ValueError("figure source count metadata must match the verified evidence series")
        if (
            evidence.run_id != spec.run_id
            or evidence.evidence_semantic != spec.evidence_semantic
            or dict(evidence.source_hashes) != dict(hashes)
        ):
            raise ValueError("configured frozen source does not match the verified evidence receipt")
        specs.append(spec)
    if tuple(spec.artifact_id for spec in specs) != FROZEN_QUANTITATIVE_IDS:
        raise ValueError("paper visual figure IDs must match the frozen 25-ID allowlist exactly")
    return tuple(specs)


def _point_by_label(evidence: StageEvidence, series_name: str, label: str) -> SeriesPoint:
    matches = [point for point in evidence.series.get(series_name, ()) if point.label == label]
    if len(matches) != 1:
        raise ValueError(f"primary comparison label {label!r} is unavailable in {series_name!r}")
    return matches[0]


def _primary_points(spec: QuantitativeFigureSpec, evidence: StageEvidence) -> tuple[SeriesPoint, SeriesPoint] | None:
    if spec.comparison_mode is ComparisonMode.DESCRIPTIVE_STATISTICS:
        return None
    baseline = _point_by_label(evidence, spec.series, spec.primary_baseline_label or "")
    method = _point_by_label(evidence, spec.series, spec.primary_method_label or "")
    if baseline.method != spec.baseline:
        raise ValueError("primary baseline point method does not match the quantitative spec baseline")
    if method.method != spec.method:
        raise ValueError("primary method point method does not match the quantitative spec method")
    if baseline.metric_id != spec.comparison_metric_id or method.metric_id != spec.comparison_metric_id:
        raise ValueError("primary points must use the quantitative spec comparison_metric_id")
    if baseline.value == method.value:
        raise ValueError("primary baseline and method values must not be equal")
    return baseline, method


def _render_points(spec: QuantitativeFigureSpec, evidence: StageEvidence):
    points = evidence.series.get(spec.series)
    if not points:
        raise ValueError(f"evidence has no nonempty series {spec.series!r}")
    values = np.asarray([point.value for point in points], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("quantitative coordinates must be finite")
    if any(point.method not in METHOD_COLORS for point in points):
        raise ValueError("evidence point method is not bound to the paper palette")
    primary = _primary_points(spec, evidence)
    if spec.ci_declared != all(point.ci_low is not None for point in points):
        raise ValueError("CI declaration does not match the frozen evidence")
    font_logger = logging.getLogger("matplotlib.font_manager")
    previous_level = font_logger.level
    font_logger.setLevel(logging.ERROR)
    try:
        with publication_rc_context():
            figure, axis = plt.subplots()
            x = np.arange(len(points), dtype=float)
            colors = [METHOD_COLORS[point.method] for point in points]
            if spec.plot_kind == "line":
                grouped: dict[str, list[tuple[float, SeriesPoint]]] = {}
                for position, point in zip(x, points, strict=True):
                    grouped.setdefault(point.method, []).append((position, point))
                for semantic, entries in grouped.items():
                    axis.plot(
                        [position for position, _ in entries],
                        [point.value for _, point in entries],
                        marker="o",
                        color=METHOD_COLORS[semantic],
                        markersize=4,
                        linewidth=1.0,
                    )
            else:
                bars = axis.bar(x, values, color=colors, width=0.7, linewidth=0)
                for index, (bar, point) in enumerate(zip(bars, points, strict=True)):
                    bar.set_gid(f"point-{index}")
                    if point.ci_low is not None:
                        axis.errorbar(
                            bar.get_x() + bar.get_width() / 2,
                            point.value,
                            yerr=[[point.value - point.ci_low], [point.ci_high - point.value]],
                            color=METHOD_COLORS["uncertainty"],
                            capsize=2,
                            linewidth=0.8,
                        )
            axis.set_ylabel(spec.y_label)
            axis.set_xticks(x, [point.label for point in points], rotation=35 if len(points) > 4 else 0, ha="right" if len(points) > 4 else "center")
            axis.tick_params(axis="x", labelsize=7)
            if np.all(values >= 0):
                axis.set_ylim(bottom=0)
            metadata_text = f"observations={spec.observation_count}; clusters={spec.n} {spec.cluster_unit}"
            if spec.ci_declared:
                metadata_text += "; 95% cluster CI"
            axis.text(0.02, 0.98, metadata_text, transform=axis.transAxes, va="top", ha="left", fontsize=7)
            figure.tight_layout(pad=0.6)
    finally:
        font_logger.setLevel(previous_level)
    if primary is None:
        return figure, None, None
    return figure, primary[0], primary[1]


def _artifact_evidence_level(semantic: str) -> EvidenceLevel:
    if semantic == "development_only":
        return EvidenceLevel.DEVELOPMENT_ONLY
    if semantic in {"formal", "formal_recorded_real", "held_out_recorded_telemetry", "software_preflight"}:
        return EvidenceLevel.FORMAL
    raise ValueError(f"unsupported evidence semantic: {semantic}")


def _descriptive_metrics(points: tuple[SeriesPoint, ...]) -> Mapping[str, float]:
    metrics = {f"{point.metric_id}:{point.label}": point.value for point in points}
    if len(metrics) != len(points):
        raise ValueError("descriptive series points must have unique metric/label bindings")
    return MappingProxyType(metrics)


def _comparison_receipt(
    spec: QuantitativeFigureSpec, evidence: StageEvidence
) -> Mapping[str, float]:
    points = evidence.series.get(spec.series)
    if not points:
        raise ValueError(f"evidence has no nonempty series {spec.series!r}")
    if spec.comparison_mode is ComparisonMode.METHOD_COMPARISON:
        primary = _primary_points(spec, evidence)
        if primary is None:
            raise ValueError("method_comparison requires verified primary points")
        baseline, method = primary
        return MappingProxyType(
            {"primary_baseline": baseline.value, "primary_method": method.value}
        )
    return _descriptive_metrics(points)


def _output_filenames(output_stem: Path) -> Mapping[str, str]:
    stem = Path(output_stem)
    return MappingProxyType(
        {
            "pdf": stem.with_suffix(".pdf").name,
            "svg": stem.with_suffix(".svg").name,
            "png": stem.with_suffix(".png").name,
        }
    )


def _caption(spec: QuantitativeFigureSpec) -> str:
    caption = f"{spec.claim} observations={spec.observation_count}; clusters={spec.n} {spec.cluster_unit}."
    if spec.ci_declared:
        caption += " 95% cluster CI reported."
    return caption


def _artifact_record(
    spec: QuantitativeFigureSpec,
    evidence: StageEvidence,
    comparison_metrics: Mapping[str, float],
    output_filenames: Mapping[str, str],
    output_hashes: Mapping[str, str],
) -> PaperArtifactRecord:
    return PaperArtifactRecord(
        artifact_id=spec.artifact_id,
        stage=spec.stage,
        claim=spec.claim,
        source_run=spec.run_id,
        source_hashes=evidence.source_hashes,
        evidence_level=_artifact_evidence_level(spec.evidence_semantic),
        method=spec.method,
        baseline=spec.baseline,
        case_id="aggregate-frozen-result",
        camera="aggregate",
        selection_policy="frozen_quantitative_atom",
        comparison_metrics=comparison_metrics,
        comparison_mode=spec.comparison_mode,
        comparison_metric_id=spec.comparison_metric_id,
        width=2100,
        height=2100,
        recommended_title=spec.claim,
        recommended_subtitle=spec.evidence_semantic,
        caption_draft=_caption(spec),
        output_filenames=output_filenames,
        output_hashes=output_hashes,
    )


def render_quantitative_figure(
    spec: QuantitativeFigureSpec,
    evidence: StageEvidence,
    output_stem: Path,
) -> tuple[PaperArtifactRecord, ...]:
    """Render exactly one title-free atomic quantitative figure."""

    if spec.artifact_id not in FROZEN_QUANTITATIVE_IDS:
        raise ValueError("quantitative artifact ID is outside the frozen 25-ID allowlist")
    if spec.stage != evidence.stage or spec.run_id != evidence.run_id or spec.run_path != evidence.run_path:
        raise ValueError("quantitative spec does not match its frozen evidence run")
    if spec.evidence_semantic != evidence.evidence_semantic:
        raise ValueError("quantitative evidence semantic mixing is forbidden")
    metadata = evidence.series_metadata.get(spec.series)
    if metadata is None or (
        metadata.observation_count != spec.observation_count
        or metadata.cluster_count != spec.n
        or metadata.cluster_unit != spec.cluster_unit
    ):
        raise ValueError("quantitative source count metadata does not match the spec")
    comparison_metrics = _comparison_receipt(spec, evidence)
    filenames = _output_filenames(Path(output_stem))
    _artifact_record(
        spec,
        evidence,
        comparison_metrics,
        filenames,
        MappingProxyType({output_format: _PREFLIGHT_OUTPUT_HASH for output_format in filenames}),
    )
    figure, _, _ = _render_points(spec, evidence)
    try:
        pdf_path, svg_path, png_path = save_square_quantitative(figure, Path(output_stem))
    finally:
        plt.close(figure)
    paths = {"pdf": pdf_path, "svg": svg_path, "png": png_path}
    record = _artifact_record(
        spec,
        evidence,
        comparison_metrics,
        filenames,
        MappingProxyType({key: _sha256(value) for key, value in paths.items()}),
    )
    return (record,)


__all__ = ["FROZEN_QUANTITATIVE_IDS", "QuantitativeFigureSpec", "load_quantitative_specs", "render_quantitative_figure"]
