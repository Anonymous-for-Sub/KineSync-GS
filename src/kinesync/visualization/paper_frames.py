"""Frozen-result, semantic paper-frame builder (CPU-only)."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import h5py
import numpy as np
import torch

from kinesync.data.route_a import RouteAAssets, RouteAPaths
from kinesync.geometry.urdf import TorchURDFKinematics
from kinesync.observation.camera import TorchCamera
from kinesync.observation.projected_centers import ProjectedCentersBackend

from .contrast_gate import ComparisonBinding, ContrastReceipt, MetricBinding, RollbackEvidence, SelectionClass, pixel_sha256, validate_informative_comparison
from .paper_contracts import ArtifactKind, EvidenceLevel, PaperArtifactRecord
from .paper_style import METHOD_COLORS

FRAME_SIZE = 1600
_COUNTS = {"RT2": 12, "RT3": 16, "RT6": 16, "RT7": 16, "RT8": 24, "RT9": 24}

def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)

def _sha(value: bytes) -> str: return hashlib.sha256(value).hexdigest()

@lru_cache(maxsize=1024)
def _file_sha(path: str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()

def _row_sha(row: Mapping[str, object]) -> str: return _sha(_canonical(dict(row)).encode("ascii"))
def _bool(value: object) -> bool: return str(value).lower() == "true"
def _float(row: Mapping[str, str], name: str) -> float: return float(row[name])

def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle: return list(csv.DictReader(handle))

def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

@dataclass(frozen=True)
class FrozenRow:
    path: Path
    identifier: str
    payload: Mapping[str, object]
    kind: str = "csv"

    @property
    def file_sha256(self) -> str: return _file_sha(str(self.path))
    @property
    def sha256(self) -> str: return _row_sha(self.payload)

    def verify(self) -> None:
        rows = _read_jsonl(self.path) if self.kind == "jsonl" else _read_csv(self.path)
        if not any(_row_sha(row) == self.sha256 for row in rows):
            raise ValueError(f"frozen result row is absent or mutated: {self.identifier}")

@dataclass(frozen=True)
class SourceRef:
    path: Path
    sha256: str
    kind: str = "image"
    dataset: str | None = None
    frame_index: int | None = None
    timestamp_ns: int | None = None
    manifest_sha256: str | None = None
    image_row_sha256: str | None = None

    def load(self) -> np.ndarray:
        if _file_sha(str(self.path)) != self.sha256: raise ValueError(f"source hash mismatch: {self.path}")
        if self.kind == "image":
            image = cv2.imread(str(self.path), cv2.IMREAD_COLOR)
            if image is None: raise ValueError(f"undecodable image: {self.path}")
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.kind != "hdf5" or self.dataset is None or self.frame_index is None: raise ValueError("invalid source reference")
        with h5py.File(self.path, "r") as handle: image = np.ascontiguousarray(handle[self.dataset][self.frame_index])
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8: raise ValueError("HDF5 source is not uint8 RGB")
        return image

    def hashes(self) -> dict[str, str]:
        hashes = {"container_file": self.sha256}
        if self.manifest_sha256: hashes["render_manifest"] = self.manifest_sha256
        if self.image_row_sha256: hashes["image_row"] = self.image_row_sha256
        return hashes

@dataclass(frozen=True)
class ProjectedSpec:
    manifest_path: Path
    camera: str
    baseline_qpos: tuple[float, ...]
    ours_qpos: tuple[float, ...]
    point_radius: int = 7
    anchors_per_link: int = 12
    focus_links: tuple[int, ...] = ()
    qpos_provenance: Mapping[str, object] | None = None

    def provenance(self) -> dict[str, object]:
        assets = json.loads(self.manifest_path.read_text())["assets"]
        keys = ("urdf", "anchor", "metadata", "head_camera", "extra_camera")
        verified = {}
        for key in keys:
            value = assets[key]
            actual = _file_sha(value["path"])
            if actual != value["sha256"]:
                raise ValueError(f"projected-state asset hash mismatch: {key}")
            verified[key] = actual
        return {"kind": "projected_state_diagnostic_v1", "device": "cpu", "assets": verified, "baseline_qpos": list(self.baseline_qpos), "ours_qpos": list(self.ours_qpos), "camera": self.camera, "anchors_per_link": self.anchors_per_link, "point_radius": self.point_radius, "focus_links": list(self.focus_links), "qpos_provenance": dict(self.qpos_provenance or {})}

@dataclass(frozen=True)
class FrozenFramePair:
    stage: str
    source_run: str
    case_id: str
    camera: str
    timepoint: str
    baseline_role: str
    ours_role: str
    baseline_source: SourceRef
    ours_source: SourceRef
    rows: tuple[FrozenRow, ...]
    metrics: tuple[MetricBinding, ...]
    selection_decision: Mapping[str, object]
    evidence_level: EvidenceLevel
    projected: ProjectedSpec | None = None
    rollback: RollbackEvidence | None = None
    receipt: ContrastReceipt | None = None

    @property
    def pair_id(self) -> str: return f"{self.stage}-{self.case_id}-{self.camera}"
    @property
    def source_row_id(self) -> str: return "|".join(row.identifier for row in self.rows)
    @property
    def result_sha256(self) -> str: return self.rows[-1].file_sha256

    def with_receipt(self) -> "FrozenFramePair":
        for row in self.rows: row.verify()
        baseline, ours, roi, generator = _materialize(self)
        rollback_binding = None
        if self.rollback is not None:
            candidate, final = self.rows[0].payload, self.rows[-1].payload
            if not (_bool(candidate["candidate_committed"]) and not _bool(final["candidate_committed"])):
                raise ValueError("rollback rows do not contain frozen candidate=true/final=false")
            if final.get("decision_reason") != self.rollback.decision_reason:
                raise ValueError("rollback decision reason is not frozen in the final row")
            rollback_binding = {"candidate_role": "candidate", "final_role": "final", "candidate_committed": True, "final_committed": False, "decision_reason": self.rollback.decision_reason}
        binding = ComparisonBinding(case_id=self.case_id, camera=self.camera, timepoint=self.timepoint,
            crop_geometry={"baseline": [0, 0, baseline.shape[1], baseline.shape[0]], "ours": [0, 0, ours.shape[1], ours.shape[0]], "generator": generator},
            source_row_id=self.source_row_id, frozen_result_sha256=self.result_sha256,
            result_row_sha256=self.rows[-1].sha256, related_result_row_sha256s={row.identifier: row.sha256 for row in self.rows}, related_result_file_sha256s={row.identifier: row.file_sha256 for row in self.rows},
            baseline_containers=self.baseline_source.hashes(), ours_containers=self.ours_source.hashes(), selection_decision=self.selection_decision, rollback_binding=rollback_binding)
        receipt = validate_informative_comparison(baseline, ours, roi, binding=binding, metrics=self.metrics, rollback_evidence=self.rollback)
        return FrozenFramePair(**{**self.__dict__, "receipt": receipt})

@lru_cache(maxsize=8)
def _projector(manifest_path: str) -> tuple[RouteAAssets, ProjectedCentersBackend]:
    assets = json.loads(Path(manifest_path).read_text())["assets"]
    paths = RouteAPaths.from_mapping({key: assets[key]["path"] for key in ("urdf", "anchor", "metadata", "trajectory_h5", "head_camera", "extra_camera")})
    route = RouteAAssets(paths); chosen = route.sample_anchor_indices(per_link=12, seed=41); anchors = route.load_anchors()
    cameras = {name: TorchCamera.from_calibration(route.load_camera(name), device="cpu") for name in ("head", "extra")}
    backend = ProjectedCentersBackend(TorchURDFKinematics(paths.urdf), torch.as_tensor(anchors.xyz[chosen], dtype=torch.float32), torch.as_tensor(anchors.link_index[chosen], dtype=torch.long), route.link_names, cameras)
    return route, backend

def _project(spec: ProjectedSpec, qpos: tuple[float, ...], shape: tuple[int, int]) -> np.ndarray:
    _, backend = _projector(str(spec.manifest_path))
    with torch.no_grad(): rendered = backend.render(torch.as_tensor(qpos, dtype=torch.float32))[spec.camera]
    points, valid = rendered.values.cpu().numpy(), rendered.valid.cpu().numpy().astype(bool)
    if spec.focus_links:
        valid &= np.isin(backend.link_index.cpu().numpy(), spec.focus_links)
    mask = np.zeros(shape, np.uint8)
    for x, y in points[valid]: cv2.circle(mask, (int(round(x)), int(round(y))), spec.point_radius, 255, -1)
    return mask > 0

def _draw_projection(rgb: np.ndarray, mask: np.ndarray, role: str) -> np.ndarray:
    color = (182, 67, 66) if role in {"baseline", "candidate", "measured", "rt6", "rt1o"} else (15, 77, 146)
    output = rgb.copy(); contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(output, contours, -1, color, 2, cv2.LINE_AA)
    return output

def _change_roi(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    diff = cv2.cvtColor(cv2.absdiff(a, b), cv2.COLOR_RGB2GRAY)
    threshold = max(6, int(np.percentile(diff, 88)))
    mask = (diff >= threshold).astype(np.uint8)
    mask = cv2.dilate(mask, np.ones((15, 15), np.uint8), iterations=1)
    if not mask.any(): raise ValueError("pair-derived change ROI is empty")
    return mask.astype(bool)

def _materialize(pair: FrozenFramePair) -> tuple[np.ndarray, np.ndarray, np.ndarray, Mapping[str, object]]:
    baseline, ours = pair.baseline_source.load(), pair.ours_source.load()
    if baseline.shape != ours.shape: raise ValueError("paired sources differ in dimensions; stretch is forbidden")
    if pair.projected:
        bmask = _project(pair.projected, pair.projected.baseline_qpos, baseline.shape[:2]); omask = _project(pair.projected, pair.projected.ours_qpos, baseline.shape[:2])
        roi = cv2.dilate((bmask | omask).astype(np.uint8), np.ones((11, 11), np.uint8), iterations=1).astype(bool)
        if np.array_equal(bmask, omask): raise ValueError("projected baseline/ours geometry is identical")
        return _draw_projection(baseline, bmask, pair.baseline_role), _draw_projection(ours, omask, pair.ours_role), roi, pair.projected.provenance()
    roi = _change_roi(baseline, ours)
    return baseline, ours, roi, {"kind": "frozen_distinct_pixels_v1", "baseline": {"path": str(pair.baseline_source.path), "dataset": pair.baseline_source.dataset, "frame_index": pair.baseline_source.frame_index, "timestamp_ns": pair.baseline_source.timestamp_ns}, "ours": {"path": str(pair.ours_source.path), "dataset": pair.ours_source.dataset, "frame_index": pair.ours_source.frame_index, "timestamp_ns": pair.ours_source.timestamp_ns}, "roi": "absdiff_p88_dilate15"}

def _asset(manifest: Mapping[str, Any], key: str) -> SourceRef:
    value = manifest["assets"][key]; path = Path(value["path"]); actual = _file_sha(str(path))
    if value.get("sha256_mode") == "full" and actual != value["sha256"]: raise ValueError(f"asset hash mismatch: {key}")
    return SourceRef(path, actual)

def _obs(manifest: Mapping[str, Any], state: str, camera: str) -> SourceRef:
    assets = manifest["assets"]
    for key in (f"rgb_{state}_{camera}", f"observation/{camera}/{state}/rgb", f"frozen_rt6_observation_{camera}_{state}_rgb"):
        if key in assets: return _asset(manifest, key)
    records = assets.get(f"records_{camera}")
    if not records: raise ValueError(f"no observation for {state}/{camera}")
    record_path = Path(records["path"]); rec = next((json.loads(line) for line in record_path.read_text().splitlines() if json.loads(line).get("state_id") == state), None)
    if not rec: raise ValueError(f"records lack {state}")
    path = record_path.parent / rec["image_path"]
    return SourceRef(path, _file_sha(str(path)))

def _state_rows(path: Path) -> dict[str, dict[str, Any]]: return {row["state_id"]: row for row in _read_jsonl(path)}
def _qpos_error(qpos: Sequence[float], target: Sequence[float]) -> float: return float(np.mean(np.abs(np.asarray(qpos) - np.asarray(target))))

def _row(path: Path, identifier: str, payload: Mapping[str, object], kind: str = "csv") -> FrozenRow: return FrozenRow(path, identifier, dict(payload), kind)

def _classes(rows: Sequence[Mapping[str, Any]], gain: str, stress: str, count: int, key: str, classes: Sequence[SelectionClass] | None = None) -> list[tuple[Mapping[str, Any], SelectionClass, int]]:
    """Select exact median/high/stress across the full eligible population, then stable alternates."""
    if len(rows) < count: raise ValueError("eligible population has fewer rows than requested atoms")
    population = sorted(rows, key=lambda row: str(row[key])); phash = _sha(_canonical(population).encode("ascii"))
    median_value = float(np.median([float(row[gain]) for row in population]))
    chosen: list[tuple[Mapping[str, Any], SelectionClass]] = []
    requested = tuple(classes or (SelectionClass.HIGH_GAIN, SelectionClass.STRESS_POSITIVE, SelectionClass.MEDIAN_POSITIVE))
    orders = {
        SelectionClass.HIGH_GAIN: lambda r: (-float(r[gain]), str(r[key])),
        SelectionClass.STRESS_POSITIVE: lambda r: (-abs(float(r[stress])), str(r[key])),
        SelectionClass.MEDIAN_POSITIVE: lambda r: (abs(float(r[gain]) - median_value), str(r[key])),
    }
    for label in requested:
        if len(chosen) == count:
            break
        order = orders[label]
        candidate = next(row for row in sorted(population, key=order) if row not in [item[0] for item in chosen]); chosen.append((candidate, label))
    for row in sorted(population, key=lambda r: (-float(r[gain]), str(r[key]))):
        if len(chosen) == count: break
        if row not in [item[0] for item in chosen]: chosen.append((row, SelectionClass.HIGH_GAIN))
    if len(chosen) != count: raise ValueError("could not fill selected cases from eligible population")
    return [(row, label, rank) for rank, (row, label) in enumerate(chosen)]

def _decision(label: SelectionClass, row: Mapping[str, Any], rank: int, population: Sequence[Mapping[str, Any]], gain: str, stress: str, key: str) -> dict[str, object]:
    ordered = sorted(population, key=lambda item: str(item[key]))
    return {"algorithm": "complete_population_median_high_stress_v2", "selection_class": label.value, "population_sha256": _sha(_canonical(ordered).encode("ascii")), "population_size": len(ordered), "rank": rank, "criterion": {"gain": gain, "stress": stress}, "selected_row": str(row[key])}

def _make(stage: str, run: str, case: str, camera: str, time: str, brole: str, orole: str, bsrc: SourceRef, osrc: SourceRef, rows: Sequence[FrozenRow], metrics: Sequence[MetricBinding], decision: Mapping[str, object], evidence: EvidenceLevel, projected: ProjectedSpec | None = None, rollback: RollbackEvidence | None = None) -> FrozenFramePair:
    return FrozenFramePair(stage, run, case, camera, time, brole, orole, bsrc, osrc, tuple(rows), tuple(metrics), decision, evidence, projected, rollback).with_receipt()

def _offset_qpos(base: Sequence[float], row: Mapping[str, str], residual: float) -> tuple[float, ...]:
    offsets = json.loads(row["offsets_rad"]); q = np.asarray(base, float).copy(); names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7", "joint8"]
    for name, value in offsets.items(): q[names.index(name)] += float(value) * residual
    return tuple(float(value) for value in q)

def _trace_final_row(path: Path, trial_id: str, candidate_source: str | None = None) -> FrozenRow:
    rows = [row for row in _read_csv(path) if row["trial_id"] == trial_id and (candidate_source is None or row.get("candidate_source") == candidate_source)]
    if not rows:
        raise ValueError(f"missing frozen trace rows for {trial_id}/{candidate_source or 'rt2'}")
    final = max(rows, key=lambda row: int(row["step"]))
    return _row(path, f"{trial_id}:trace:{candidate_source or 'factorized'}:step-{final['step']}", final)

def _trace_initial_mae(path: Path, trial_id: str, candidate_source: str | None = None) -> float:
    rows = [row for row in _read_csv(path) if row["trial_id"] == trial_id and (candidate_source is None or row.get("candidate_source") == candidate_source)]
    if not rows:
        raise ValueError(f"missing frozen trace start for {trial_id}/{candidate_source or 'rt2'}")
    return _float(min(rows, key=lambda row: int(row["step"])), "state_mae")

def _trace_correction(row: FrozenRow) -> tuple[float, ...]:
    return tuple(_float(row.payload, f"correction_joint{joint}") for joint in range(1, 9))

def _verify_state_trace(*, measured: Sequence[float], final: Sequence[float], truth: Sequence[float], initial_trace_mae: float, final_trace_row: FrozenRow, expected_reduction: float, context: str) -> tuple[float, float]:
    before, after = _qpos_error(measured, truth), _qpos_error(final, truth)
    if not np.isclose(before, initial_trace_mae, rtol=0.0, atol=1e-8):
        raise ValueError(f"{context} measured qpos MAE disagrees with frozen trace")
    if not np.isclose(after, _float(final_trace_row.payload, "state_mae"), rtol=0.0, atol=1e-8):
        raise ValueError(f"{context} final qpos MAE disagrees with frozen trace")
    reduction = 1.0 - after / before if before > 0.0 else 0.0
    if not np.isclose(reduction, expected_reduction, rtol=0.0, atol=1e-6):
        raise ValueError(f"{context} qpos reduction disagrees with frozen result")
    return before, after

def _measured_qpos_from_offset_name(truth: Sequence[float], offset_name: str) -> tuple[float, ...]:
    """Reconstruct the recorded controlled offset from the immutable trial id."""
    qpos = np.asarray(truth, dtype=float).copy()
    if offset_name.startswith("zero-"):
        return tuple(float(value) for value in qpos)
    tokens = offset_name.replace("head-", "").replace("extra-", "").split("-")
    magnitude = float(tokens[-1].removesuffix("deg")) * np.pi / 180.0
    direction = -1.0 if "neg" in tokens else 1.0
    joints = tokens[:tokens.index("neg") if "neg" in tokens else tokens.index("pos")]
    for joint in joints:
        if not joint.startswith("j") or not joint[1:].isdigit():
            raise ValueError(f"unparseable frozen RT3 offset name: {offset_name}")
        qpos[int(joint[1:]) - 1] += direction * magnitude
    return tuple(float(value) for value in qpos)

def _rt2(root: Path, cfg: Mapping[str, Any]) -> list[FrozenFramePair]:
    run, result, trace = root / cfg["run_path"], root / cfg["result_file"], root / cfg["trace_file"]; manifest, rows = json.loads((run / "manifest.json").read_text()), _read_csv(result)
    eligible = [row for row in rows if row["camera_mode"] == "head+extra" and _float(row, "state_error_reduction") >= .25]
    selected = _classes(eligible, "state_error_reduction", "magnitude_deg", 3, "trial_id"); output=[]
    for row, label, rank in selected:
        for camera in ("head", "extra"):
            record_path = Path(manifest["assets"][f"records_{camera}"]["path"]); rec = next(json.loads(line) for line in record_path.read_text().splitlines() if json.loads(line)["state_id"] == row["state_id"])
            truth = tuple(rec["urdf_qpos_rad"]); baseline = _offset_qpos(truth, row, 1.0); final_trace = _trace_final_row(trace, row["trial_id"]); ours = tuple(np.asarray(baseline) + np.asarray(_trace_correction(final_trace)))
            before, after = _verify_state_trace(measured=baseline, final=ours, truth=truth, initial_trace_mae=_trace_initial_mae(trace, row["trial_id"]), final_trace_row=final_trace, expected_reduction=_float(row,"state_error_reduction"), context=f"RT2 {row['trial_id']}")
            spec = ProjectedSpec(run / "manifest.json", camera, baseline, ours, qpos_provenance={"algorithm":"observation_qpos_plus_offsets_plus_trace_final_correction","trace_file_sha256":_file_sha(str(trace)),"trace_final_row_sha256":final_trace.sha256,"trace_final_step":final_trace.payload["step"]})
            output.append(_make("RT2", manifest["run_id"], row["trial_id"], camera, str(rec["host_monotonic_timestamp_ns"]), "offset", "factorized", _obs(manifest,row["state_id"],camera), _obs(manifest,row["state_id"],camera), (final_trace,_row(result,row["trial_id"],row)), (MetricBinding("state_error", "lower", before, after),), _decision(label,row,rank,eligible,"state_error_reduction","magnitude_deg","trial_id"), EvidenceLevel.FORMAL, spec))
    return output

def _rt3(root: Path, cfg: Mapping[str, Any]) -> list[FrozenFramePair]:
    run,result,trace=root/cfg["run_path"],root/cfg["result_file"],root/cfg["trace_file"]; rows=_read_csv(result); manifest=json.loads((root/cfg["observation_manifest"]).read_text())
    by={(r["trial_id"],r["method"]):r for r in rows}; accepted=[r for r in rows if r["method"]=="rt3_full" and _bool(r["candidate_committed"]) and r["decision_reason"]=="accepted" and (r["trial_id"],"rt1o_unfactorized") in by]
    rollback=[r for r in rows if r["method"]=="rt3_full" and not _bool(r["candidate_committed"]) and r["decision_reason"] not in {"accepted","unprotected_baseline"} and (r["trial_id"],"rt1o_unfactorized") in by]
    chosen=_classes(accepted,"state_error_reduction","magnitude_deg",2,"trial_id",(SelectionClass.HIGH_GAIN,SelectionClass.STRESS_POSITIVE)) + _classes(rollback,"state_error_reduction","magnitude_deg",2,"trial_id",(SelectionClass.MEDIAN_POSITIVE,SelectionClass.HIGH_GAIN)); output=[]
    for final,label,rank in chosen:
        base=by[(final["trial_id"],"rt1o_unfactorized")]; candidate_result=by[(final["trial_id"],"rt2f_unguarded")]; isrollback=final in rollback
        for camera in ("head","extra"):
            record_path=Path(manifest["assets"][f"records_{camera}"]["path"]); rec=next(json.loads(line) for line in record_path.read_text().splitlines() if json.loads(line)["state_id"]==final["state_id"]); truth=tuple(rec["urdf_qpos_rad"]); measured=_measured_qpos_from_offset_name(truth,base["offset_name"])
            rt1o_trace=_trace_final_row(trace,final["trial_id"],"rt1o"); candidate_trace=_trace_final_row(trace,final["trial_id"],cfg["accepted_candidate_source"]); rt1o=tuple(np.asarray(measured)+np.asarray(_trace_correction(rt1o_trace))); candidate=tuple(np.asarray(measured)+np.asarray(_trace_correction(candidate_trace)))
            _,candidate_after=_verify_state_trace(measured=measured,final=candidate,truth=truth,initial_trace_mae=_trace_initial_mae(trace,final["trial_id"],cfg["accepted_candidate_source"]),final_trace_row=candidate_trace,expected_reduction=_float(candidate_result,"state_error_reduction"),context=f"RT3 candidate {final['trial_id']}")
            if isrollback:
                if _float(final,"state_error_reduction") != 0.0: raise ValueError("RT3 rollback result must retain measured qpos and zero reduction")
                baseline_qpos, final_q, metrics, row_prefix = candidate, measured, (MetricBinding("unsafe_commit","lower",1.,0.),), candidate_result
            else:
                if not np.isclose(_float(final,"state_error_reduction"), _float(candidate_result,"state_error_reduction"), rtol=0.0, atol=1e-7):
                    raise ValueError("RT3 accepted result reduction disagrees with frozen candidate")
                baseline_qpos, final_q, metrics, row_prefix = rt1o, candidate, (MetricBinding("state_error","lower",_qpos_error(rt1o,truth),candidate_after),), base
            rb=RollbackEvidence(True,False,final["decision_reason"]) if isrollback else None
            provenance={"algorithm":"measured_qpos_plus_trace_final_corrections","trace_file_sha256":_file_sha(str(trace)),"rt1o_trace_final_row_sha256":rt1o_trace.sha256,"trace_final_row_sha256":candidate_trace.sha256,"candidate_source":cfg["accepted_candidate_source"],"rollback_final":"measured_qpos" if isrollback else "accepted_candidate"}
            output.append(_make("RT3",json.loads((run/"manifest.json").read_text())["run_id"],final["trial_id"],camera,str(rec["host_monotonic_timestamp_ns"]),"candidate" if isrollback else "rt1o","measured/final" if isrollback else "rt2 candidate",_obs(manifest,final["state_id"],camera),_obs(manifest,final["state_id"],camera),(_row(result,row_prefix["trial_id"]+":candidate",row_prefix),rt1o_trace,candidate_trace,_row(result,final["trial_id"]+":rt3",final)),metrics,_decision(label,final,rank,accepted if not isrollback else rollback,"state_error_reduction","magnitude_deg","trial_id"),EvidenceLevel.FORMAL,ProjectedSpec(root/cfg["observation_manifest"],camera,baseline_qpos,final_q,qpos_provenance=provenance),rb))
    return output

def _state_stage(root: Path,cfg:Mapping[str,Any],stage:str,brole:str,orole:str,evidence:EvidenceLevel,projection_manifest: Path | None = None)->list[FrozenFramePair]:
    run,result=root/cfg["run_path"],root/cfg["result_file"]; manifest=json.loads((run/"manifest.json").read_text()); rows=_read_csv(result); frames=_state_rows(run/"state_frames.jsonl"); method="rt6dm_synchronized" if stage=="RT6" else "rt7cv_component_verified"
    formal={r["case_id"]:r for r in rows if r["method"]==method}; eligible=[]
    component_path = run / "component_results.csv"
    components = _read_csv(component_path) if stage == "RT7" else []
    if stage == "RT7":
        for component in components:
            frame = frames.get(component["state_id"])
            # A positive component outcome is recorded by candidate_success and
            # a positive frozen state-error reduction.  Acceptance remains
            # bound in the row rather than being inferred from the visual.
            if not frame or not _bool(component["candidate_success"]):
                continue
            before_qpos, after_qpos = frame["source_rt6_synchronized_qpos"], frame["synchronized_qpos"]
            if np.allclose(before_qpos, after_qpos) or _float(component, "state_error_reduction") <= 0.0:
                continue
            formal_row = next((row for row in formal.values() if row["state_id"] == component["state_id"]), None)
            eligible.append({"row": formal_row, "component": component, "frame": frame, "gain": _float(component, "state_error_reduction"), "stress": component["magnitude_deg"], "key": f"{component['case_id']}:{component['joint_name']}", "before": 1.0, "after": 1.0 - _float(component, "state_error_reduction")})
    else:
        for state,frame in frames.items():
            row=next((r for c,r in formal.items() if r["state_id"]==state),None)
            if not row: continue
            bq=frame["measured_qpos"]
            oq=frame["synchronized_qpos"]; target=frame["mujoco_qpos"]; before,after=_qpos_error(bq,target),_qpos_error(oq,target)
            if before>0 and before-after>=before*.25 and not np.allclose(bq,oq): eligible.append({"row":row,"frame":frame,"gain":before-after,"stress":row["magnitude_deg"],"key":row["case_id"],"before":before,"after":after})
    selected=_classes(eligible,"gain","stress",4,"key"); output=[]
    for item,label,rank in selected:
        row,frame=item["row"],item["frame"]
        case_id = item["key"]
        frozen_rows = [_row(run/"state_frames.jsonl",frame["state_id"],frame,"jsonl")]
        if row is not None:
            frozen_rows.append(_row(result,row["case_id"],row))
        if stage == "RT7":
            frozen_rows.append(_row(component_path,item["component"]["case_id"],item["component"]))
        for camera in ("head","extra"):
            focus_links = ()
            if stage == "RT7":
                changed = tuple((np.flatnonzero(np.abs(np.asarray(frame["source_rt6_synchronized_qpos"]) - np.asarray(frame["synchronized_qpos"])) > 1e-10) + 1).tolist())
                focus_links = tuple(sorted(set(changed + (int(item["component"]["joint_name"].removeprefix("joint")),))))
            output.append(_make(stage,manifest["run_id"],case_id,camera,str(frame["timestamp_ns"]),brole,orole,_obs(manifest,row["state_id"] if row else frame["state_id"],camera),_obs(manifest,row["state_id"] if row else frame["state_id"],camera),tuple(frozen_rows),(MetricBinding("component_state_error" if stage=="RT7" else "qpos_mae","lower",item["before"],item["after"]),),_decision(label,item,rank,eligible,"gain","stress","key"),evidence,ProjectedSpec(projection_manifest or run/"manifest.json",camera,tuple(frame["measured_qpos"] if stage=="RT6" else frame["source_rt6_synchronized_qpos"]),tuple(frame["synchronized_qpos"]),focus_links=focus_links)))
    return output

def _rt8(root:Path,cfg:Mapping[str,Any])->list[FrozenFramePair]:
    run,result=root/cfg["run_path"],root/cfg["result_file"]; rm=json.loads((run/"manifest.json").read_text()); rows=_read_csv(result); manifest_keys={"a_p2_component_gs":"render_manifest_a_p2_component_gs_1","route_b_hybrid":"render_manifest_route_b_hybrid_1"}; output=[]
    used_image_hashes: set[str] = set()
    for backend,key in manifest_keys.items():
        declared=rm["assets"][key]; mp=Path(declared["path"]); mh=_file_sha(str(mp))
        if mh!=declared["sha256"]: raise ValueError("RT8 declared render manifest hash mismatch")
        renders=json.loads(mp.read_text())["rows"]; byid={r["source_pair_id"]:r for r in renders}
        eligible=[]
        for row in rows:
            if not (row["backend"]==backend and row["split"]=="test" and _bool(row["guard_committed"]) and _bool(row["guard_improved"]) and _float(row,"delayed_qmae_rad")>_float(row,"guarded_qmae_rad") and row["measured_pair_id"] in byid and row["selected_pair_id"] in byid):
                continue
            measured, selected = byid[row["measured_pair_id"]], byid[row["selected_pair_id"]]
            if measured["gaussian_head_sha256"] == selected["gaussian_head_sha256"]:
                continue
            eligible.append(row)
        primary = _classes(eligible,"correction_qmae_rad","lag",3,"source_pair_id")
        used_pair_ids: set[str] = set()
        selected_rows: list[tuple[Mapping[str, str], SelectionClass, int, str]] = []
        for seed, label, rank in primary:
            episode = [row for row in eligible if row["episode_id"] == seed["episode_id"]]
            def available(row: Mapping[str, str]) -> bool:
                measured, guarded = byid[row["measured_pair_id"]], byid[row["selected_pair_id"]]
                return (row["measured_pair_id"] not in used_pair_ids and row["selected_pair_id"] not in used_pair_ids and measured["gaussian_head_sha256"] not in used_image_hashes and guarded["gaussian_head_sha256"] not in used_image_hashes)
            first = next((row for row in [seed] + sorted(episode, key=lambda row: (-_float(row,"correction_qmae_rad"), row["source_pair_id"])) if available(row)), None)
            if first is None:
                raise ValueError("RT8 cannot select a distinct primary render source")
            used_pair_ids.update((first["measured_pair_id"], first["selected_pair_id"]))
            used_image_hashes.update((byid[first["measured_pair_id"]]["gaussian_head_sha256"], byid[first["selected_pair_id"]]["gaussian_head_sha256"]))
            secondary = next((row for row in sorted(episode, key=lambda row: (-abs(int(row["true_index"])-int(first["true_index"])), -_float(row,"correction_qmae_rad"), row["source_pair_id"])) if available(row) and row["true_index"] != first["true_index"]), None)
            if secondary is None:
                raise ValueError("RT8 episode lacks a second distinct frozen timepoint")
            used_pair_ids.update((secondary["measured_pair_id"], secondary["selected_pair_id"]))
            used_image_hashes.update((byid[secondary["measured_pair_id"]]["gaussian_head_sha256"], byid[secondary["selected_pair_id"]]["gaussian_head_sha256"]))
            selected_rows.extend(((first,label,rank,"primary"),(secondary,label,rank,"secondary")))
        for row,label,rank,position in selected_rows:
            def source(pair_id:str)->SourceRef:
                rr=byid[pair_id]; p=mp.parent/rr["gaussian_head"]
                if _file_sha(str(p))!=rr["gaussian_head_sha256"]: raise ValueError("RT8 image row hash mismatch")
                return SourceRef(p,rr["gaussian_head_sha256"],manifest_sha256=mh,image_row_sha256=_row_sha(rr))
            decision=_decision(label,row,rank,eligible,"correction_qmae_rad","lag","source_pair_id")
            decision={**decision,"episode_id":row["episode_id"],"timepoint_position":position,"selection_rule":"full-population-class-then-distinct-episode-timepoint"}
            output.append(_make("RT8",rm["run_id"],f"{backend}-{row['source_pair_id']}","gaussian_head",f"episode-{row['episode_id']}-index-{row['true_index']}-lag-{row['lag']}","delayed GS","guarded GS",source(row["measured_pair_id"]),source(row["selected_pair_id"]),(_row(result,f"{backend}:{row['source_pair_id']}",row),), (MetricBinding("qmae","lower",_float(row,"delayed_qmae_rad"),_float(row,"guarded_qmae_rad")),),decision,EvidenceLevel.FORMAL))
    return output

def _rt9(root:Path,cfg:Mapping[str,Any])->list[FrozenFramePair]:
    run,result=root/cfg["run_path"],root/cfg["result_file"]; manifest=json.loads((run/"manifest.json").read_text()); rows=_read_csv(result); output=[]
    for trajectory in ("f1.hdf5","f2.hdf5"):
        raw={r["condition_id"]:r for r in rows if r["trajectory_id"]==trajectory and r["method"]=="raw_linear"}; eligible=[r for r in rows if r["trajectory_id"]==trajectory and r["method"]=="kinesync_guarded" and _bool(r["committed"]) and r["condition_id"] in raw and _float(raw[r["condition_id"]],"qmae_rad")>_float(r,"qmae_rad")]
        selected=_classes(eligible,"qmae_reduction_rad","jitter_ms",3,"condition_id"); asset=next(v for k,v in manifest["assets"].items() if k.endswith(trajectory)); path=Path(asset["path"]); full=_file_sha(str(path))
        with h5py.File(path,"r") as h: period=float(np.median(np.diff(h["cam_head/timestamp"][:])))
        for guard,label,rank in selected:
            rawrow=raw[guard["condition_id"]]; start,stop=(int(v) for v in guard["condition_id"].split("|")[0].split(":")[1:3]); mid=start+(stop-start)//2
            raw_residual=_float(rawrow,"estimated_offset_ms")- _float(rawrow,"applied_correction_ms"); guard_residual=_float(guard,"estimated_offset_ms")-_float(guard,"applied_correction_ms")
            baseidx=int(np.clip(mid+round(raw_residual*1_000_000/period),start,stop-1)); oursidx=int(np.clip(mid+round(guard_residual*1_000_000/period),start,stop-1))
            if baseidx==oursidx: raise ValueError("RT9 raw/guarded residuals map to same frozen frame")
            for camera,dataset in (("head","cam_head/color"),("wrist","cam_wrist/color")):
                with h5py.File(path,"r") as h: bt,ot=int(h[dataset.replace("/color","/timestamp")][baseidx]),int(h[dataset.replace("/color","/timestamp")][oursidx])
                bs=SourceRef(path,full,"hdf5",dataset,baseidx,bt); os=SourceRef(path,full,"hdf5",dataset,oursidx,ot)
                output.append(_make("RT9",manifest["run_id"],guard["condition_id"],camera,f"{baseidx}->{oursidx}","raw","guarded",bs,os,(_row(result,rawrow["condition_id"]+":raw",rawrow),_row(result,guard["condition_id"]+":guarded",guard)),(MetricBinding("qmae","lower",_float(rawrow,"qmae_rad"),_float(guard,"qmae_rad")),),_decision(label,guard,rank,eligible,"qmae_reduction_rad","jitter_ms","condition_id"),EvidenceLevel.FORMAL))
    return output

def _config(path:Path,run_root:Path|None=None)->tuple[Path,Mapping[str,Any]]:
    payload=json.loads(path.read_text()); root=Path(run_root).resolve() if run_root else path.parent.parent.resolve(); selector=payload["atomic_frame_selector"]
    if selector.get("schema")!="paper-frame-selector-v3": raise ValueError("missing v3 selector")
    for stage,cfg in selector["stages"].items():
        declared=payload["sources"][stage]["source_hashes"]; run=root/cfg["run_path"]
        declared_paths = cfg.get("frozen_hash_files", {})
        if set(declared_paths) != set(declared):
            raise ValueError(f"{stage} frozen hash file inventory does not close the configured source hashes")
        for name, relative in declared_paths.items():
            if _file_sha(str(root / relative))!=declared[name]: raise ValueError(f"{stage} frozen source hash mismatch for {name}")
        if "trace_file" in cfg:
            if _file_sha(str(root / cfg["trace_file"])) != cfg.get("trace_sha256"):
                raise ValueError(f"{stage} trace hash mismatch")
    return root,selector

def select_frozen_frame_pairs(config_path:Path,*,run_root:Path|None=None)->tuple[FrozenFramePair,...]:
    root,selector=_config(Path(config_path),run_root); stages=selector["stages"]
    rt6_manifest = root / stages["RT6"]["run_path"] / "manifest.json"
    pairs=_rt2(root,stages["RT2"])+_rt3(root,stages["RT3"])+_state_stage(root,stages["RT6"],"RT6","measured","synchronized",EvidenceLevel.FORMAL,rt6_manifest)+_state_stage(root,stages["RT7"],"RT7","RT6","RT7",EvidenceLevel.DEVELOPMENT_ONLY,rt6_manifest)+_rt8(root,stages["RT8"])+_rt9(root,stages["RT9"])
    actual={stage:sum(p.stage==stage for p in pairs)*2 for stage in _COUNTS}
    if actual!=_COUNTS or len(pairs)!=54: raise ValueError(f"Task3 must close exactly 108 real atoms: {actual}")
    return tuple(pairs)

def _letterbox(image:np.ndarray,color:tuple[int,int,int],label:str)->np.ndarray:
    available,border=1508,12; scale=min(available/image.shape[1],available/image.shape[0]); resized=cv2.resize(image,(round(image.shape[1]*scale),round(image.shape[0]*scale)),interpolation=cv2.INTER_AREA); canvas=np.full((FRAME_SIZE,FRAME_SIZE,3),255,np.uint8); canvas[:border]=color;canvas[-border:]=color;canvas[:,:border]=color;canvas[:,-border:]=color; top=border+(available-resized.shape[0])//2;left=(FRAME_SIZE-resized.shape[1])//2;canvas[top:top+resized.shape[0],left:left+resized.shape[1]]=resized;cv2.putText(canvas,label,(28,1571),cv2.FONT_HERSHEY_SIMPLEX,.55,color,2,cv2.LINE_AA);return canvas

def _slug(value:str)->str: return "".join(c if c.isalnum() else "-" for c in value).strip("-")[:96]

def export_atomic_frame(pair:FrozenFramePair,source_role:str,output_path:Path,*,manifest_mode:str="paper")->PaperArtifactRecord:
    if source_role not in {"baseline","ours"}: raise ValueError("source role must be baseline or ours")
    checked=pair.with_receipt()
    if pair.receipt is None or checked.receipt.sha256!=pair.receipt.sha256: raise ValueError("frozen source or row binding changed")
    if manifest_mode=="paper" and not checked.receipt.recommended: raise ValueError("paper manifest requires recommended receipt")
    baseline,ours,roi,generator=_materialize(checked); image=baseline if source_role=="baseline" else ours; expected=checked.receipt.baseline_crop_sha256 if source_role=="baseline" else checked.receipt.ours_crop_sha256
    if pixel_sha256(image)!=expected: raise ValueError("source role pixel hash mismatch")
    role=checked.baseline_role if source_role=="baseline" else checked.ours_role; colorname="primary_baseline" if source_role=="baseline" else "ours_full"; hexcolor=METHOD_COLORS[colorname]; color=tuple(int(hexcolor[i:i+2],16) for i in (1,3,5)); diagnostic="projected state | " if checked.projected else ""; label=f"{diagnostic}{role} | {checked.camera} | t={checked.timepoint}"; canvas=_letterbox(image,color,label)
    output_path=Path(output_path);output_path.parent.mkdir(parents=True,exist_ok=True)
    if not cv2.imwrite(str(output_path),cv2.cvtColor(canvas,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_PNG_COMPRESSION,6]):raise RuntimeError("PNG write failed")
    artifact=f"F-{checked.stage}-{_slug(checked.case_id)}-{checked.camera}-{source_role}"; hashes={"contrast_receipt":checked.receipt.sha256,"source_crop":expected,"selected_source":checked.baseline_source.sha256 if source_role=="baseline" else checked.ours_source.sha256,"roi_pixels":checked.receipt.roi_pixel_sha256,"generator":_sha(_canonical(generator).encode("ascii")),"frozen_result":checked.result_sha256,**{f"baseline_source_{name}":value for name,value in checked.receipt.binding.baseline_containers.items()},**{f"ours_source_{name}":value for name,value in checked.receipt.binding.ours_containers.items()},**{f"row_{name}":value for name,value in checked.receipt.binding.related_result_row_sha256s.items()},**{f"row_file_{name}":value for name,value in checked.receipt.binding.related_result_file_sha256s.items()}}
    return PaperArtifactRecord(artifact_id=artifact,artifact_kind=ArtifactKind.FRAME,stage=checked.stage,claim="Frozen source-row-bound comparison atom.",source_run=checked.source_run,source_hashes=hashes,evidence_level=checked.evidence_level,method="ours_full",baseline="primary_baseline",case_id=checked.case_id,camera=checked.camera,selection_policy=checked.receipt.selection_class.value,comparison_metrics={},width=FRAME_SIZE,height=FRAME_SIZE,recommended_title="Source-bound frame",recommended_subtitle="Frozen paired evidence",caption_draft="Source-bound paper frame.",output_filenames={"png":output_path.name},output_hashes={"png":_file_sha(str(output_path))})

def build_frozen_frame_atoms(config_path:Path,output_dir:Path)->tuple[PaperArtifactRecord,...]:
    records=[]
    for pair in select_frozen_frame_pairs(config_path):
        for role in ("baseline","ours"):
            filename=f"F-{pair.stage}-{_slug(pair.case_id)}-{pair.camera}-{role}.png"; records.append(export_atomic_frame(pair,role,Path(output_dir)/filename))
    if len(records)!=108 or len({record.artifact_id for record in records})!=108 or len({record.output_hashes['png'] for record in records})!=108: raise ValueError("Task3 build did not produce 108 distinct atoms")
    return tuple(records)

def load_atomic_frame_specs(config_path:Path)->tuple[FrozenFramePair,...]: return select_frozen_frame_pairs(config_path)
__all__=["FrozenFramePair","SourceRef","build_frozen_frame_atoms","export_atomic_frame","load_atomic_frame_specs","select_frozen_frame_pairs"]
