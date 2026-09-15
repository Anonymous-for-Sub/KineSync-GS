"""Immutable provenance contracts for paper-visualization artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from .paper_style import METHOD_COLORS


class EvidenceLevel(str, Enum):
    FORMAL = "formal"
    DEVELOPMENT_ONLY = "development_only"
    ILLUSTRATIVE = "illustrative"


class ArtifactKind(str, Enum):
    QUANTITATIVE = "quantitative"
    FRAME = "frame"
    GRID = "grid"
    VIDEO = "video"


class ComparisonMode(str, Enum):
    """Closed semantics for quantitative values recorded in an artifact."""

    METHOD_COMPARISON = "method_comparison"
    DESCRIPTIVE_STATISTICS = "descriptive_statistics"


_OUTPUT_FORMATS = {
    ArtifactKind.QUANTITATIVE: frozenset({"pdf", "svg", "png"}),
    ArtifactKind.FRAME: frozenset({"png"}),
    ArtifactKind.GRID: frozenset({"pdf", "svg", "png"}),
    ArtifactKind.VIDEO: frozenset({"mp4"}),
}
_PRIMARY_COMPARISON_KEYS = frozenset({"primary_baseline", "primary_method"})
_GRID_LAYOUTS = frozenset({"3x2", "4x2", "3x3", "4x3", "7x2", "8x2", "9x2", "8x3"})


def _nonempty_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string")
    return value


def _sha256(value: object, field: str) -> str:
    digest = _nonempty_text(value, field)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{field} must be a lowercase full SHA-256")
    return digest


def _hashes(value: Mapping[str, str], field: str) -> Mapping[str, str]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{field} must be a nonempty mapping")
    validated = {
        _nonempty_text(name, f"{field} name"): _sha256(digest, f"{field}[{name!r}]")
        for name, digest in value.items()
    }
    return MappingProxyType(dict(sorted(validated.items())))


def _flat_output_filenames(
    value: Mapping[str, str], *, kind: ArtifactKind, required: frozenset[str]
) -> Mapping[str, str]:
    if not isinstance(value, Mapping) or set(value) != required:
        required_text = ", ".join(sorted(required))
        raise ValueError(
            f"{kind.value} output_filenames must contain exactly {required_text}"
        )
    validated: dict[str, str] = {}
    for output_format, filename in value.items():
        filename = _nonempty_text(filename, f"output filename for {output_format}")
        path = Path(filename)
        if "/" in filename or "\\" in filename or path.name != filename or path.suffix != f".{output_format}":
            raise ValueError("output filenames must be flat and match their format")
        validated[output_format] = filename
    return MappingProxyType(dict(sorted(validated.items())))


def _comparison_metrics(
    value: Mapping[str, float], *, allow_empty: bool
) -> Mapping[str, float]:
    if not isinstance(value, Mapping) or (not allow_empty and not value):
        raise ValueError("comparison_metrics must be a mapping")
    validated: dict[str, float] = {}
    for name, metric in value.items():
        name = _nonempty_text(name, "comparison metric name")
        if isinstance(metric, bool) or not isinstance(metric, (int, float)):
            raise ValueError(f"comparison_metrics[{name!r}] must be finite")
        metric = float(metric)
        if not math.isfinite(metric):
            raise ValueError(f"comparison_metrics[{name!r}] must be finite")
        validated[name] = metric
    return MappingProxyType(dict(sorted(validated.items())))


def _primary_comparison_delta(
    mode: ComparisonMode | None, metrics: Mapping[str, float], comparison_metric_id: str | None
) -> float | None:
    if mode is None:
        if metrics or comparison_metric_id is not None:
            raise ValueError("empty nonquantitative comparison metadata must not declare metrics or metric IDs")
        return None
    present = _PRIMARY_COMPARISON_KEYS & set(metrics)
    if mode is ComparisonMode.METHOD_COMPARISON:
        if comparison_metric_id is None:
            raise ValueError("method_comparison requires comparison_metric_id")
        if present != _PRIMARY_COMPARISON_KEYS:
            raise ValueError("method_comparison must declare primary_baseline and primary_method")
        delta = abs(metrics["primary_baseline"] - metrics["primary_method"])
        if delta == 0.0:
            raise ValueError("primary_baseline and primary_method must not be equal")
        return delta
    if comparison_metric_id is not None:
        raise ValueError("descriptive_statistics must not declare comparison_metric_id")
    if present:
        raise ValueError("descriptive_statistics must not declare primary_baseline or primary_method")
    if len(metrics) < 2 or len(set(metrics.values())) < 2:
        raise ValueError("descriptive_statistics requires two non-identical finite statistics")
    return None


def _comparison_mode(value: object, kind: ArtifactKind) -> ComparisonMode | None:
    if kind is not ArtifactKind.QUANTITATIVE and value in (None, ""):
        return None
    try:
        return ComparisonMode(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Unsupported comparison_mode: {value!r}") from error


def _comparison_metric_id(value: object, mode: ComparisonMode | None) -> str | None:
    if mode is None:
        if value not in (None, ""):
            raise ValueError("empty nonquantitative comparison metadata must not declare comparison_metric_id")
        return None
    if value is None:
        return None
    metric_id = _nonempty_text(value, "comparison_metric_id")
    if mode is ComparisonMode.DESCRIPTIVE_STATISTICS:
        raise ValueError("descriptive_statistics must not declare comparison_metric_id")
    return metric_id


def _infer_nonquantitative_comparison_mode(
    kind: ArtifactKind, mode: ComparisonMode | None, metrics: Mapping[str, float]
) -> ComparisonMode | None:
    if mode is not None:
        return mode
    if kind is ArtifactKind.QUANTITATIVE:
        raise ValueError("quantitative artifacts require an explicit comparison_mode")
    if not metrics:
        return None
    present = _PRIMARY_COMPARISON_KEYS & set(metrics)
    if present:
        if present != _PRIMARY_COMPARISON_KEYS:
            raise ValueError("primary_baseline and primary_method must be declared together")
        return ComparisonMode.METHOD_COMPARISON
    if len(metrics) < 2 or len(set(metrics.values())) < 2:
        raise ValueError("descriptive_statistics requires two non-identical finite statistics")
    return ComparisonMode.DESCRIPTIVE_STATISTICS


def _validate_comparison(
    mode: ComparisonMode | None, metrics: Mapping[str, float], comparison_metric_id: str | None
) -> float | None:
    present = _PRIMARY_COMPARISON_KEYS & set(metrics)
    if mode is ComparisonMode.METHOD_COMPARISON and present and present != _PRIMARY_COMPARISON_KEYS:
        raise ValueError("primary_baseline and primary_method must be declared together")
    return _primary_comparison_delta(mode, metrics, comparison_metric_id)


def _dimensions(kind: ArtifactKind, width: object, height: object) -> tuple[int, int]:
    if (
        isinstance(width, bool)
        or isinstance(height, bool)
        or not isinstance(width, int)
        or not isinstance(height, int)
        or width <= 0
        or height <= 0
    ):
        raise ValueError("artifact dimensions must be positive integers")
    if kind is ArtifactKind.QUANTITATIVE and (width, height) != (2100, 2100):
        raise ValueError("quantitative artifact dimensions must be square 2100x2100")
    if kind is ArtifactKind.FRAME and (width, height) != (1600, 1600):
        raise ValueError("frame artifact dimensions must be square 1600x1600")
    if kind is ArtifactKind.GRID and width < height:
        raise ValueError("grid artifact dimensions must use a supported landscape or square layout")
    if kind is ArtifactKind.VIDEO and (width, height) != (1920, 1080):
        raise ValueError("video artifact dimensions must be 1920x1080")
    return width, height


def _grid_layout(kind: ArtifactKind, value: object) -> str | None:
    if kind is ArtifactKind.GRID:
        if not isinstance(value, str) or value not in _GRID_LAYOUTS:
            raise ValueError("grid_layout must use the closed paper layout allowlist")
        return value
    if value is not None:
        raise ValueError("grid_layout must be None for non-grid artifacts")
    return None


def _semantic_method(value: object, field: str) -> str:
    if not isinstance(value, str) or value not in METHOD_COLORS:
        raise ValueError(f"{field} must be a known METHOD_COLORS entry")
    return value


@dataclass(frozen=True)
class PaperArtifactRecord:
    artifact_id: str
    stage: str
    claim: str
    source_run: str
    source_hashes: Mapping[str, str]
    evidence_level: EvidenceLevel | str
    method: str
    baseline: str
    case_id: str
    camera: str
    selection_policy: str
    comparison_metrics: Mapping[str, float]
    width: int
    height: int
    recommended_title: str
    recommended_subtitle: str
    caption_draft: str
    output_filenames: Mapping[str, str]
    output_hashes: Mapping[str, str]
    rendered_title: str | None = None
    rendered_subtitle: str | None = None
    artifact_kind: ArtifactKind | str = ArtifactKind.QUANTITATIVE
    grid_layout: str | None = None
    comparison_mode: ComparisonMode | str | None = None
    comparison_metric_id: str | None = None

    def __post_init__(self) -> None:
        for field in (
            "artifact_id",
            "stage",
            "claim",
            "source_run",
            "case_id",
            "camera",
            "selection_policy",
            "recommended_title",
            "recommended_subtitle",
            "caption_draft",
        ):
            object.__setattr__(self, field, _nonempty_text(getattr(self, field), field))
        object.__setattr__(self, "method", _semantic_method(self.method, "method"))
        object.__setattr__(self, "baseline", _semantic_method(self.baseline, "baseline"))
        try:
            object.__setattr__(self, "evidence_level", EvidenceLevel(self.evidence_level))
        except (TypeError, ValueError) as error:
            raise ValueError(f"Unsupported evidence level: {self.evidence_level!r}") from error
        if self.rendered_title not in (None, "") or self.rendered_subtitle not in (None, ""):
            raise ValueError("rendered_title and rendered_subtitle are forbidden for paper artifacts")
        try:
            kind = ArtifactKind(self.artifact_kind)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Unsupported artifact kind: {self.artifact_kind!r}") from error
        object.__setattr__(self, "artifact_kind", kind)
        requested_mode = _comparison_mode(self.comparison_mode, kind)
        width, height = _dimensions(kind, self.width, self.height)
        object.__setattr__(self, "width", width)
        object.__setattr__(self, "height", height)
        object.__setattr__(self, "grid_layout", _grid_layout(kind, self.grid_layout))
        filenames = _flat_output_filenames(
            self.output_filenames, kind=kind, required=_OUTPUT_FORMATS[kind]
        )
        hashes = _hashes(self.output_hashes, "output_hashes")
        if set(hashes) != set(filenames):
            raise ValueError("output_hashes must cover every output filename")
        object.__setattr__(self, "source_hashes", _hashes(self.source_hashes, "source_hashes"))
        metrics = _comparison_metrics(self.comparison_metrics, allow_empty=requested_mode is None)
        mode = _infer_nonquantitative_comparison_mode(kind, requested_mode, metrics)
        object.__setattr__(self, "comparison_mode", mode)
        object.__setattr__(self, "comparison_metric_id", _comparison_metric_id(self.comparison_metric_id, mode))
        object.__setattr__(self, "comparison_metrics", metrics)
        _validate_comparison(mode, metrics, self.comparison_metric_id)
        object.__setattr__(self, "output_filenames", filenames)
        object.__setattr__(self, "output_hashes", hashes)

    @property
    def primary_comparison_delta(self) -> float | None:
        """Return the derived, fail-closed primary comparison magnitude."""

        return _validate_comparison(
            self.comparison_mode, self.comparison_metrics, self.comparison_metric_id
        )

    def canonical_json(self) -> str:
        """Return stable JSON suitable for immutable artifact manifests."""

        payload = {
            "artifact_id": self.artifact_id,
            "artifact_kind": self.artifact_kind.value,
            "baseline": self.baseline,
            "camera": self.camera,
            "caption_draft": self.caption_draft,
            "case_id": self.case_id,
            "claim": self.claim,
            "comparison_metric_id": self.comparison_metric_id,
            "comparison_mode": self.comparison_mode.value if self.comparison_mode is not None else None,
            "comparison_metrics": dict(self.comparison_metrics),
            "evidence_level": self.evidence_level.value,
            "height": self.height,
            "method": self.method,
            "output_filenames": dict(self.output_filenames),
            "output_hashes": dict(self.output_hashes),
            "primary_comparison_delta": self.primary_comparison_delta,
            "grid_layout": self.grid_layout,
            "recommended_subtitle": self.recommended_subtitle,
            "recommended_title": self.recommended_title,
            "selection_policy": self.selection_policy,
            "source_hashes": dict(self.source_hashes),
            "source_run": self.source_run,
            "stage": self.stage,
            "width": self.width,
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


__all__ = ["ArtifactKind", "ComparisonMode", "EvidenceLevel", "PaperArtifactRecord"]
