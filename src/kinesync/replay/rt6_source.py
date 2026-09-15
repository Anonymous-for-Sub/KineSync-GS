"""Public frozen-RT6 source contracts reusable by counterfactual replays."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from kinesync.cli.rt2_evaluate import _load_factors, _validate_factor_bounds
from kinesync.cli.rt3_evaluate import _load_guard
from kinesync.cli.rt6_formal import (
    _candidate_fingerprint,
    _formal_renderer_factory,
    _formal_source_hashes,
    _frozen_formal_inputs,
    _gaussian_render_receipt,
    _load_detectability_profile,
    _sha256,
    _source_frame_provenance,
)
from kinesync.config import load_config
from kinesync.data.route_a import RouteAPaths
from kinesync.experiments.rt6_matrix import load_rt6_matrix


sha256_file = _sha256
candidate_fingerprint = _candidate_fingerprint
formal_renderer_factory = _formal_renderer_factory
gaussian_render_receipt = _gaussian_render_receipt
source_frame_provenance = _source_frame_provenance


def load_frozen_rt6_config_snapshot(path: str | Path) -> dict[str, Any]:
    """Load an RT6 run snapshot without losing its canonical formal config path."""

    snapshot_path = Path(path).expanduser().resolve()
    raw = snapshot_path.read_text(encoding="utf-8")
    snapshot = yaml.safe_load(raw)
    if not isinstance(snapshot, Mapping):
        raise ValueError(f"Configuration must be a mapping: {snapshot_path}")
    config = load_config(snapshot_path)
    embedded_path = snapshot.get("_config_path")
    if embedded_path is None:
        return config
    canonical = Path(str(embedded_path)).expanduser()
    if not canonical.is_absolute():
        canonical = snapshot_path.parent / canonical
    canonical = canonical.resolve()
    if not canonical.is_file():
        raise FileNotFoundError(
            f"frozen RT6 canonical formal config does not exist: {canonical}"
        )
    config["_config_path"] = str(canonical)
    return config


@dataclass(frozen=True)
class FrozenRT6SourcePreflight:
    """Validated frozen inputs, current asset hashes, and receipt for a replay."""

    frozen: Mapping[str, Any]
    factor_path: Path
    factors: Mapping[str, Any]
    factor_sha256: str
    guard_path: Path
    guard_sha256: str
    profile_path: Path
    profile: Any
    profile_sha256: str
    matrix: Any
    verified_source_hashes: Mapping[str, str]
    asset_paths: Mapping[str, Path]
    asset_receipt: Mapping[str, Any]


def _file_record(path: Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Frozen RT6 asset does not exist: {resolved}")
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _asset_receipt(
    *,
    paths: RouteAPaths,
    frozen: Mapping[str, Any],
    factor_path: Path,
    guard_path: Path,
    profile_path: Path,
    matrix_path: Path,
    converted_urdf: Path,
    conversion_manifest: Path,
    formal_file_assets: Mapping[str, Path],
    verified_source_hashes: Mapping[str, str],
) -> tuple[dict[str, Path], dict[str, Any]]:
    files: dict[str, Path] = {
        **{f"asset/{name}": path for name, path in paths.as_assets().items()},
        "formal/config": Path(frozen["config_path"]),
        "formal/factors": factor_path,
        "formal/guard": guard_path,
        "formal/matrix": matrix_path,
        "formal/profile": profile_path,
        "source/conversion_manifest": conversion_manifest,
        "source/converted_urdf": converted_urdf,
        **{str(name): Path(path) for name, path in formal_file_assets.items()},
    }
    records = {name: _file_record(path) for name, path in sorted(files.items())}
    artifact_paths = {
        f"frozen_rt6_{name.replace('/', '_')}": Path(record["path"])
        for name, record in records.items()
    }
    return artifact_paths, {
        "files": records,
        "verified_source_hashes": dict(verified_source_hashes),
    }


def verify_frozen_rt6_source(
    *,
    source_config: Mapping[str, Any],
    source_metrics: Mapping[str, Any],
    source_run_path: str | Path,
) -> FrozenRT6SourcePreflight:
    """Re-run RT6's immutable input preflight before a replay creates output.

    The comparison against the RT6 metrics ledger covers the formal config,
    factors, guard, profile, matrix, every selected RGB/mask bundle, cameras,
    URDF/mesh bundle, and trajectory. No optimizer or renderer is invoked.
    """

    run_path = Path(source_run_path).expanduser().resolve()
    if not run_path.is_dir():
        raise FileNotFoundError(f"frozen RT6 source run does not exist: {run_path}")
    if not isinstance(source_metrics, Mapping):
        raise ValueError("frozen RT6 source metrics must be a mapping")

    frozen = _frozen_formal_inputs(source_config)
    factor_path, factors = _load_factors(frozen["factors"])
    _validate_factor_bounds(factors)
    factor_sha256 = sha256_file(factor_path)
    guard_path, _ = _load_guard(frozen["guard"], factor_sha256=factor_sha256)
    guard_sha256 = sha256_file(guard_path)
    profile_path, profile = _load_detectability_profile(
        frozen["profile"], factor_sha256=factor_sha256
    )
    profile_sha256 = sha256_file(profile_path)
    matrix_path = Path(frozen["matrix"])
    matrix = load_rt6_matrix(matrix_path)
    if (
        matrix.trial_count != frozen["expected_trial_count"]
        or matrix.unique_state_count != frozen["expected_unique_state_count"]
    ):
        raise ValueError("frozen RT6 matrix counts do not match executable config")

    current_formal_hashes, formal_file_assets = _formal_source_hashes(
        config=source_config,
        matrix=matrix,
        frozen=frozen,
        factor_sha256=factor_sha256,
        guard_sha256=guard_sha256,
        profile_sha256=profile_sha256,
        profile_fingerprint=profile.fingerprint,
    )
    expected_formal_hashes = frozen["hashes"]
    if expected_formal_hashes != current_formal_hashes:
        raise ValueError("frozen RT6 source hashes do not match formal config")

    converted_urdf = run_path / "converted_piper.urdf"
    conversion_manifest = run_path / "conversion_manifest.json"
    current_source_hashes = {
        **current_formal_hashes,
        "formal_config": sha256_file(frozen["config_path"]),
        "converted_urdf": sha256_file(converted_urdf),
        "conversion_manifest": sha256_file(conversion_manifest),
    }
    metric_source_hashes = source_metrics.get("source_hashes")
    if not isinstance(metric_source_hashes, Mapping) or {
        str(name): str(value) for name, value in metric_source_hashes.items()
    } != current_source_hashes:
        raise ValueError("frozen RT6 source hashes do not match source metrics")

    paths = RouteAPaths.from_mapping(source_config["assets"])
    asset_paths, asset_receipt = _asset_receipt(
        paths=paths,
        frozen=frozen,
        factor_path=factor_path,
        guard_path=guard_path,
        profile_path=profile_path,
        matrix_path=matrix_path,
        converted_urdf=converted_urdf,
        conversion_manifest=conversion_manifest,
        formal_file_assets=formal_file_assets,
        verified_source_hashes=current_source_hashes,
    )
    return FrozenRT6SourcePreflight(
        frozen=frozen,
        factor_path=factor_path,
        factors=factors,
        factor_sha256=factor_sha256,
        guard_path=guard_path,
        guard_sha256=guard_sha256,
        profile_path=profile_path,
        profile=profile,
        profile_sha256=profile_sha256,
        matrix=matrix,
        verified_source_hashes=current_source_hashes,
        asset_paths=asset_paths,
        asset_receipt=asset_receipt,
    )


__all__ = [
    "FrozenRT6SourcePreflight",
    "candidate_fingerprint",
    "formal_renderer_factory",
    "gaussian_render_receipt",
    "load_frozen_rt6_config_snapshot",
    "sha256_file",
    "source_frame_provenance",
    "verify_frozen_rt6_source",
]
