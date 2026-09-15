"""Illustrative, provenance-bound KineSync workflow visual evidence.

Task 5 intentionally does not produce benchmark measurements.  It renders a
fixed diagnostic trajectory from immutable RT6 state frames and uses frozen
RT6/RT7/RT8/RT9 evidence only to explain the observation-to-twin loop.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Mapping, Sequence

import cv2
import matplotlib.pyplot as plt
import numpy as np

from .paper_contracts import ArtifactKind, EvidenceLevel, PaperArtifactRecord
from kinesync.geometry.urdf import TorchURDFKinematics
from kinesync.sync.mujoco_bridge import MujocoStateBridge

from .paper_frames import FRAME_SIZE, FrozenFramePair, ProjectedSpec, _draw_projection, _materialize, _project, select_frozen_frame_pairs
from .paper_style import METHOD_COLORS
from .paper_video import VideoFrame, VideoSpec, _held_schedule, verify_video_pair, write_method_video


WORKFLOW_STAGE_ORDER = (
    "paired_real_observation",
    "time_aligned_articulated_state",
    "state_conditioned_gs_render",
    "cross_view_residual",
    "bounded_correction_candidate",
    "guard_commit_or_rollback",
    "recorded_qpos_interface_consistency",
    "observation_policy_interface",
)
_SOURCE_STAGE_ORDER = ("RT6", "RT9", "RT8", "RT6", "RT6", "RT7", "RT6", "RT9")
_STATE_IDS = ("state_0359", "state_0359", "state_0359", "state_0378", "state_0378", "state_0379", "state_0379", "state_0379")
_TRAJECTORY_CASE_IDS = (
    "controlled_2p5_joint1_pos", "controlled_2p5_joint1_pos", "controlled_2p5_joint1_pos",
    "controlled_2p5_joint1_neg", "controlled_2p5_joint1_neg", "diagnostic_multi_5",
    "controlled_5_joint4_neg", "diagnostic_multi_5",
)
_MAX_CORRECTION_RAD = 0.15
_GRID_CELL_SIZE, _GRID_FOOTER_HEIGHT, _GRID_GAP = 480, 36, 10


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rgb(method: str) -> tuple[int, int, int]:
    color = METHOD_COLORS[method]
    return tuple(int(color[index : index + 2], 16) for index in (1, 3, 5))


@dataclass(frozen=True)
class IllustrativeTrajectory:
    qpos: np.ndarray
    source_state_ids: tuple[str, ...]
    source_hashes: Mapping[str, str]
    trajectory_sha256: str
    max_correction_rad: float
    joint_limit_contract: str


@dataclass(frozen=True)
class WorkflowAtom:
    artifact_id: str
    stage: str
    visual_recipe: str
    pair: FrozenFramePair | None
    projection_spec: ProjectedSpec | None
    qpos: tuple[float, ...]
    baseline_qpos: tuple[float, ...]
    state_frame: Mapping[str, object]
    trajectory_state_id: str
    trajectory_case_id: str
    source_stage: str
    source_state_id: str | None
    source_case_id: str
    source_hashes: Mapping[str, str]
    trajectory_label: str
    evidence_level: EvidenceLevel | str
    claim: str
    camera: str = "head"
    comparison_metrics: Mapping[str, float] | None = None
    frame_record: PaperArtifactRecord | None = None
    baseline_component: np.ndarray | None = None
    ours_component: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.stage not in WORKFLOW_STAGE_ORDER:
            raise ValueError("workflow atom stage is not in the fixed ordered inventory")
        if not self.artifact_id.startswith("F-LOOP-"):
            raise ValueError("workflow atom must use an F-LOOP identifier")
        if EvidenceLevel(self.evidence_level) is not EvidenceLevel.ILLUSTRATIVE:
            raise ValueError("workflow atoms must remain illustrative")
        object.__setattr__(self, "evidence_level", EvidenceLevel.ILLUSTRATIVE)
        if self.trajectory_label != "illustrative_scripted_trajectory":
            raise ValueError("workflow atom trajectory label must be illustrative_scripted_trajectory")
        values = np.asarray(self.qpos, dtype=np.float64)
        baseline_values = np.asarray(self.baseline_qpos, dtype=np.float64)
        if values.shape != (8,) or baseline_values.shape != (8,) or not np.isfinite(values).all() or not np.isfinite(baseline_values).all():
            raise ValueError("workflow atom qpos must be a finite 8D PiPER vector")
        if float(np.abs(values - baseline_values).max()) > _MAX_CORRECTION_RAD:
            raise ValueError("workflow atom qpos exceeds the fixed correction bound")
        if not self.claim or not self.camera or self.source_stage not in _SOURCE_STAGE_ORDER or not self.source_case_id:
            raise ValueError("workflow atom requires a known source, source camera, and claim")
        if (self.state_frame.get("state_id"), self.state_frame.get("case_id")) != (self.trajectory_state_id, self.trajectory_case_id):
            raise ValueError("workflow trajectory state/case does not match its state frame")
        if self.visual_recipe not in {
            "route_a_paired_real", "rt9_temporal", "rt8_gaussian_render", "cross_view_residual",
            "candidate_vs_measured", "guard_commit_rollback", "recorded_qpos_interface", "rt9_observation_interface",
        }:
            raise ValueError("workflow atom visual recipe is unknown")
        if self.visual_recipe == "route_a_paired_real":
            if self.pair is not None or self.projection_spec is None:
                raise ValueError("Route-A real source must be decoupled from selected frozen pairs")
            if self.source_stage != "RT6":
                raise ValueError("Route-A direct observation source must remain RT6")
        elif self.pair is None or self.pair.stage != self.source_stage:
            raise ValueError("workflow atom pair does not match its declared source stage")
        if self.source_state_id is not None:
            if (self.source_state_id, self.source_case_id) != (self.state_frame.get("state_id"), self.state_frame.get("case_id")):
                raise ValueError("workflow source state/case does not match its state frame")
            if self.pair is not None and not any(
                row.payload.get("state_id") == self.source_state_id and row.payload.get("case_id") == self.source_case_id
                for row in self.pair.rows
            ):
                raise ValueError("workflow pair rows do not contain the declared source state/case")
        if not isinstance(self.source_hashes, Mapping) or len(self.source_hashes) < 5:
            raise ValueError("workflow atom needs source/container/row/generator/trajectory hashes")
        if any(len(value) != 64 for value in self.source_hashes.values()):
            raise ValueError("workflow atom source hashes must be full SHA-256 values")
        if self.comparison_metrics not in (None, {}):
            raise ValueError("illustrative workflow atoms must not carry metrics")
        if "success" in self.claim.lower() or "benchmark" in self.claim.lower():
            raise ValueError("illustrative workflow claims cannot state benchmark outcomes")
        components = (self.baseline_component, self.ours_component)
        if any(item is not None for item in components):
            if any(item is None for item in components):
                raise ValueError("workflow components must be materialized as a matched pair")
            for item in components:
                if item is None or item.shape != (FRAME_SIZE, FRAME_SIZE, 3) or item.dtype != np.uint8:
                    raise ValueError("workflow components must be square uint8 RGB tiles")


@dataclass(frozen=True)
class WorkflowBuild:
    atoms: tuple[WorkflowAtom, ...]
    frame_records: tuple[PaperArtifactRecord, ...]
    grids: tuple[PaperArtifactRecord, ...]
    video: PaperArtifactRecord


def _project_root(config_path: Path) -> Path:
    return config_path.resolve().parents[1]


def _load_config(config_path: Path) -> tuple[Path, Mapping[str, object], Mapping[str, object]]:
    config_path = Path(config_path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    workflow = payload.get("task5_workflow")
    if not isinstance(workflow, Mapping):
        raise ValueError("Task 5 workflow config is missing")
    if workflow.get("evidence_level") != EvidenceLevel.ILLUSTRATIVE.value:
        raise ValueError("Task 5 workflow config must remain illustrative")
    if workflow.get("trajectory_label") != "illustrative_scripted_trajectory":
        raise ValueError("Task 5 workflow config must declare the illustrative trajectory")
    if tuple(workflow.get("state_ids", ())) != ("state_0359", "state_0378", "state_0379"):
        raise ValueError("Task 5 workflow config must freeze the three audited RT6 states")
    return _project_root(config_path), payload, workflow


def _read_state_frames(path: Path) -> dict[tuple[str, str], Mapping[str, object]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    chosen: dict[tuple[str, str], Mapping[str, object]] = {}
    required = {
        ("state_0359", "controlled_2p5_joint1_pos"),
        ("state_0378", "controlled_2p5_joint1_neg"),
        ("state_0379", "diagnostic_multi_5"),
        ("state_0379", "controlled_5_joint4_neg"),
    }
    for row in rows:
        key = (str(row.get("state_id")), str(row.get("case_id")))
        if key in required:
            chosen[key] = row
    if set(chosen) != required:
        raise ValueError("frozen RT6 state-frame inventory is incomplete")
    for row in chosen.values():
        for name in ("measured_qpos", "candidate_qpos", "synchronized_qpos", "gaussian_render_qpos"):
            values = np.asarray(row.get(name), dtype=np.float64)
            if values.shape != (8,) or not np.isfinite(values).all():
                raise ValueError("frozen RT6 state frame lacks finite 8D qpos")
        for decision in row.get("decisions", []):
            if abs(float(decision["correction_rad"])) > _MAX_CORRECTION_RAD:
                raise ValueError("frozen RT6 correction exceeds illustrative trajectory bound")
    return chosen


def _trajectory(state_rows: Mapping[tuple[str, str], Mapping[str, object]], state_file_hash: str) -> IllustrativeTrajectory:
    first = state_rows[("state_0359", "controlled_2p5_joint1_pos")]
    second = state_rows[("state_0378", "controlled_2p5_joint1_neg")]
    third = state_rows[("state_0379", "diagnostic_multi_5")]
    fourth = state_rows[("state_0379", "controlled_5_joint4_neg")]
    measured1, synchronized1 = np.asarray(first["measured_qpos"], float), np.asarray(first["synchronized_qpos"], float)
    measured2, synchronized2 = np.asarray(second["measured_qpos"], float), np.asarray(second["synchronized_qpos"], float)
    measured3, synchronized3 = np.asarray(third["measured_qpos"], float), np.asarray(third["synchronized_qpos"], float)
    measured4, synchronized4 = np.asarray(fourth["measured_qpos"], float), np.asarray(fourth["synchronized_qpos"], float)
    qpos = np.stack((
        measured1, (measured1 + synchronized1) / 2.0, synchronized1,
        measured2, synchronized2, synchronized3, synchronized4, synchronized3,
    ))
    corrections = np.abs(qpos - np.stack((
        measured1, measured1, measured1, measured2, measured2, measured3, measured4, measured3,
    )))
    maximum = float(corrections.max())
    if qpos.shape != (8, 8) or not np.isfinite(qpos).all() or maximum > _MAX_CORRECTION_RAD:
        raise ValueError("illustrative trajectory violates the finite 8D bounded-qpos contract")
    payload = {"state_ids": _STATE_IDS, "qpos": qpos.tolist(), "state_frames": state_file_hash, "label": "illustrative_scripted_trajectory"}
    trajectory_hash = _sha_bytes(_canonical(payload).encode("ascii"))
    return IllustrativeTrajectory(qpos, _STATE_IDS, {"state_frames": state_file_hash}, trajectory_hash, maximum, "unvalidated")


def _validate_urdf_and_state_frame_limits(trajectory: IllustrativeTrajectory, state_rows: Mapping[tuple[str, str], Mapping[str, object]], manifest_path: Path) -> IllustrativeTrajectory:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    urdf = manifest["assets"]["urdf"]
    urdf_path = Path(urdf["path"])
    if _sha_file(urdf_path) != urdf["sha256"]:
        raise ValueError("workflow URDF hash mismatch")
    derived = manifest.get("derived_artifacts", {})
    converted = derived.get("converted_urdf", {})
    conversion = derived.get("conversion_manifest", {})
    converted_path = Path(str(converted.get("path", "")))
    conversion_path = Path(str(conversion.get("path", "")))
    converted_hash = _sha_file(converted_path) if converted_path.is_file() else ""
    conversion_hash = _sha_file(conversion_path) if conversion_path.is_file() else ""
    if converted_hash != converted.get("sha256") or conversion_hash != conversion.get("sha256"):
        raise ValueError("workflow converted MuJoCo artifact hash mismatch")
    conversion_payload = json.loads(conversion_path.read_text(encoding="utf-8"))
    if conversion_payload.get("source_sha256") != urdf["sha256"] or conversion_payload.get("output_sha256") != converted_hash:
        raise ValueError("workflow MuJoCo conversion manifest does not bind source and output URDFs")
    kinematics = TorchURDFKinematics(urdf_path)
    if len(kinematics.active_joint_names) != 8:
        raise ValueError("workflow URDF must expose the canonical 8D PiPER qpos")
    joint_names = tuple(kinematics.active_joint_names)
    bridge = MujocoStateBridge.from_urdf(converted_path, joint_names)
    if (
        bridge.joint_names != joint_names
        or int(bridge.model.nq) != 8
        or int(bridge.model.nv) != 8
        or bridge.qpos_addresses != tuple(range(8))
        or bridge.dof_addresses != tuple(range(8))
    ):
        raise ValueError("workflow MuJoCo model violates the canonical joint order/address contract")
    limits = tuple(bridge.joint_limits[name] for name in bridge.joint_names)
    if any(not np.isfinite((lower, upper)).all() or lower >= upper for lower, upper in limits):
        raise ValueError("workflow MuJoCo model must provide finite ordered joint ranges")
    rows = [trajectory.qpos]
    for row in state_rows.values():
        rows.extend(np.asarray(row[name], dtype=np.float64)[None, :] for name in ("measured_qpos", "candidate_qpos", "synchronized_qpos", "mujoco_qpos", "gaussian_render_qpos"))
    values = np.concatenate(rows, axis=0)
    if not np.isfinite(values).all():
        raise ValueError("workflow qpos contains non-finite values")
    for qpos in values:
        bridge.write_qpos(qpos)
        written = np.asarray(bridge.data.qpos[list(bridge.qpos_addresses)], dtype=np.float64)
        if not np.array_equal(written, qpos):
            raise ValueError("workflow MuJoCo write_qpos contract changed the recorded qpos")
    mujoco_rows = {f"{state_id}|{case_id}": row["mujoco_qpos"] for (state_id, case_id), row in state_rows.items()}
    model_contract = {
        "joint_names": list(bridge.joint_names),
        "nq": int(bridge.model.nq), "nv": int(bridge.model.nv),
        "qpos_addresses": list(bridge.qpos_addresses), "dof_addresses": list(bridge.dof_addresses),
        "joint_limits": {name: list(bridge.joint_limits[name]) for name in bridge.joint_names},
        "converted_urdf": converted_hash, "conversion_manifest": conversion_hash,
        "validation": "MujocoStateBridge.from_urdf+write_qpos",
    }
    source_hashes = {
        **dict(trajectory.source_hashes),
        "urdf": _sha_file(urdf_path),
        "converted_urdf": converted_hash,
        "conversion_manifest": conversion_hash,
        "mujoco_model_contract": _sha_bytes(_canonical(model_contract).encode("ascii")),
        "state_frame_mujoco_contract": _sha_bytes(_canonical(mujoco_rows).encode("ascii")),
    }
    return IllustrativeTrajectory(
        trajectory.qpos, trajectory.source_state_ids, source_hashes, trajectory.trajectory_sha256,
        trajectory.max_correction_rad, "mujoco_bridge_converted_urdf_write_qpos_v1",
    )


def _row_hash(row: Mapping[str, object]) -> str:
    return _sha_bytes(_canonical(dict(row)).encode("ascii"))


def _find_pair(
    pairs: Sequence[FrozenFramePair], stage: str, state_id: str | None = None,
    case_id: str | None = None, ordinal: int = 0,
) -> FrozenFramePair:
    source_camera = "gaussian_head" if stage == "RT8" else "head"
    candidates = [pair for pair in pairs if pair.stage == stage and pair.camera == source_camera]
    if state_id is not None:
        candidates = [pair for pair in candidates if any(str(row.payload.get("state_id", "")) == state_id for row in pair.rows)]
    if case_id is not None:
        candidates = [pair for pair in candidates if pair.case_id == case_id]
    if not candidates:
        raise ValueError(f"no frozen head-camera pair for workflow source {stage}/{state_id}")
    if ordinal < 0 or ordinal >= len(candidates):
        raise ValueError(f"workflow source ordinal is unavailable for {stage}/{source_camera}")
    pair = candidates[ordinal]
    checked = pair.with_receipt()
    if pair.receipt is None or checked.receipt.sha256 != pair.receipt.sha256 or not checked.receipt.recommended:
        raise ValueError("workflow source pair lost its frozen contrast receipt")
    return pair


def frozen_workflow_inventory(config_path: Path) -> tuple[WorkflowAtom, ...]:
    """Return the closed, source-bound eight-stage illustrative inventory."""

    root, _, workflow = _load_config(Path(config_path))
    state_path = root / str(workflow["state_frames"])
    state_hash = _sha_file(state_path)
    state_rows = _read_state_frames(state_path)
    trajectory = _validate_urdf_and_state_frame_limits(_trajectory(state_rows, state_hash), state_rows, state_path.parent / "manifest.json")
    pairs = select_frozen_frame_pairs(Path(config_path), run_root=root)
    selections = (
        None,
        _find_pair(pairs, "RT9"),
        _find_pair(pairs, "RT8"),
        _find_pair(pairs, "RT6", "state_0378", "controlled_2p5_joint1_neg"),
        _find_pair(pairs, "RT6", "state_0378", "controlled_2p5_joint1_neg"),
        _find_pair(pairs, "RT7", "state_0379", "diagnostic_multi_5:joint5"),
        _find_pair(pairs, "RT6", "state_0379", "controlled_5_joint4_neg"),
        _find_pair(pairs, "RT9", ordinal=1),
    )
    recipes = (
        "route_a_paired_real", "rt9_temporal", "rt8_gaussian_render", "cross_view_residual",
        "candidate_vs_measured", "guard_commit_rollback", "recorded_qpos_interface", "rt9_observation_interface",
    )
    cameras = ("head+extra", "head", "gaussian_head", "head", "head", "head", "head", "head")
    claims = (
        "Illustrative paired Route-A real observation.",
        "Illustrative recorded temporal alignment diagnostic.",
        "Illustrative state-conditioned Gaussian render diagnostic.",
        "Illustrative cross-view residual diagnostic from frozen pixels.",
        "Illustrative bounded candidate-versus-measured projection diagnostic.",
        "Illustrative frozen guard commit/rollback diagnostic.",
        "Illustrative recorded qpos interface consistency projected diagnostic.",
        "Illustrative observation/interface handoff from recorded telemetry.",
    )
    rt6_manifest = json.loads((state_path.parent / "manifest.json").read_text(encoding="utf-8"))
    atoms: list[WorkflowAtom] = []
    baseline_schedule = (
        state_rows[("state_0359", "controlled_2p5_joint1_pos")]["measured_qpos"],
        state_rows[("state_0359", "controlled_2p5_joint1_pos")]["measured_qpos"],
        state_rows[("state_0359", "controlled_2p5_joint1_pos")]["measured_qpos"],
        state_rows[("state_0378", "controlled_2p5_joint1_neg")]["measured_qpos"],
        state_rows[("state_0378", "controlled_2p5_joint1_neg")]["measured_qpos"],
        state_rows[("state_0379", "diagnostic_multi_5")]["measured_qpos"],
        state_rows[("state_0379", "controlled_5_joint4_neg")]["measured_qpos"],
        state_rows[("state_0379", "diagnostic_multi_5")]["measured_qpos"],
    )
    source_state_ids = ("state_0359", None, None, "state_0378", "state_0378", "state_0379", "state_0379", None)
    source_case_ids = (
        "controlled_2p5_joint1_pos", selections[1].case_id, selections[2].case_id,
        "controlled_2p5_joint1_neg", "controlled_2p5_joint1_neg", "diagnostic_multi_5",
        "controlled_5_joint4_neg", selections[7].case_id,
    )
    for index, (stage, recipe, camera, claim, pair, qpos, baseline_qpos, trajectory_state_id, trajectory_case_id, source_stage, source_state_id, source_case_id) in enumerate(zip(
        WORKFLOW_STAGE_ORDER, recipes, cameras, claims, selections, trajectory.qpos, baseline_schedule,
        _STATE_IDS, _TRAJECTORY_CASE_IDS, _SOURCE_STAGE_ORDER, source_state_ids, source_case_ids, strict=True,
    ), start=1):
        state_frame = state_rows[(trajectory_state_id, trajectory_case_id)]
        pair_rows = {} if pair is None else {f"row_{row.identifier}": row.sha256 for row in pair.rows}
        hashes: dict[str, str] = {
            "trajectory": trajectory.trajectory_sha256,
            "rt6_state_frames": state_hash,
            "urdf": trajectory.source_hashes["urdf"],
            "converted_urdf": trajectory.source_hashes["converted_urdf"],
            "conversion_manifest": trajectory.source_hashes["conversion_manifest"],
            "mujoco_model_contract": trajectory.source_hashes["mujoco_model_contract"],
            "state_frame_mujoco_contract": trajectory.source_hashes["state_frame_mujoco_contract"],
            "state_frame_row": _row_hash(state_frame),
        }
        if recipe == "route_a_paired_real":
            projection_spec = ProjectedSpec(
                state_path.parent / "manifest.json", "head", tuple(state_frame["measured_qpos"]), tuple(state_frame["synchronized_qpos"]),
                qpos_provenance={"kind": "route_a_state_0359_projection_helper_v1", "state_frame_row": _row_hash(state_frame)},
            )
            hashes["projection_manifest"] = _sha_file(projection_spec.manifest_path)
            for source_camera in ("head", "extra"):
                asset = rt6_manifest["assets"][f"observation/{source_camera}/state_0359/rgb"]
                path = Path(asset["path"])
                if _sha_file(path) != asset["sha256"]:
                    raise ValueError("Route-A workflow observation hash mismatch")
                hashes[f"route_a_{source_camera}_rgb"] = asset["sha256"]
        else:
            if pair is None:
                raise AssertionError("non-Route-A workflow source requires a frozen pair")
            projection_spec = pair.projected
            generator = _materialize(pair)[3]
            hashes.update({
                "frozen_result": pair.result_sha256,
                "contrast_receipt": pair.receipt.sha256 if pair.receipt else "",
                "generator": _sha_bytes(_canonical(generator).encode("ascii")),
                "baseline_container": pair.baseline_source.sha256,
                "ours_container": pair.ours_source.sha256,
                **pair_rows,
            })
        if any(not value for value in hashes.values()):
            raise ValueError("workflow source provenance is incomplete")
        atoms.append(WorkflowAtom(
            artifact_id=f"F-LOOP-{index:02d}", stage=stage, visual_recipe=recipe, pair=pair, projection_spec=projection_spec,
            qpos=tuple(float(value) for value in qpos), baseline_qpos=tuple(float(value) for value in baseline_qpos), state_frame=state_frame,
            trajectory_state_id=trajectory_state_id, trajectory_case_id=trajectory_case_id,
            source_stage=source_stage, source_state_id=source_state_id, source_case_id=source_case_id, source_hashes=hashes,
            trajectory_label="illustrative_scripted_trajectory", evidence_level=EvidenceLevel.ILLUSTRATIVE,
            claim=claim, camera=camera,
        ))
    if len(atoms) != 8 or tuple(atom.stage for atom in atoms) != WORKFLOW_STAGE_ORDER:
        raise AssertionError("Task 5 workflow inventory must contain exactly eight ordered atoms")
    return tuple(atoms)


def validate_illustrative_trajectory(atoms: Sequence[WorkflowAtom], *, evidence_level: EvidenceLevel = EvidenceLevel.ILLUSTRATIVE) -> IllustrativeTrajectory:
    if evidence_level is not EvidenceLevel.ILLUSTRATIVE:
        raise ValueError("only illustrative trajectories may be validated")
    if len(atoms) != 8 or tuple(atom.stage for atom in atoms) != WORKFLOW_STAGE_ORDER:
        raise ValueError("workflow trajectory must use the exact eight-stage order")
    if any(atom.evidence_level is not EvidenceLevel.ILLUSTRATIVE for atom in atoms):
        raise ValueError("workflow trajectory must remain illustrative")
    qpos = np.asarray([atom.qpos for atom in atoms], dtype=np.float64)
    if qpos.shape != (8, 8) or not np.isfinite(qpos).all():
        raise ValueError("workflow trajectory must be finite and 8D")
    state_ids = tuple(atom.trajectory_state_id for atom in atoms)
    expected = _STATE_IDS
    if state_ids != expected:
        raise ValueError("workflow trajectory state schedule is not frozen")
    state_hash = atoms[0].source_hashes["rt6_state_frames"]
    trajectory_hashes = {atom.source_hashes["trajectory"] for atom in atoms}
    if len(trajectory_hashes) != 1:
        raise ValueError("workflow trajectory hash is not shared by all stages")
    maximum = float(np.max(np.abs(qpos - np.asarray([atom.baseline_qpos for atom in atoms], dtype=np.float64))))
    if maximum > _MAX_CORRECTION_RAD:
        raise ValueError("workflow trajectory correction exceeds its fixed bound")
    if any(atom.source_hashes.get("urdf") != atoms[0].source_hashes.get("urdf") for atom in atoms):
        raise ValueError("workflow trajectory must share one validated URDF contract")
    contract_keys = ("urdf", "converted_urdf", "conversion_manifest", "mujoco_model_contract")
    if any(any(atom.source_hashes.get(key) != atoms[0].source_hashes.get(key) for atom in atoms) for key in contract_keys):
        raise ValueError("workflow trajectory must share one validated MuJoCo model contract")
    return IllustrativeTrajectory(
        qpos, state_ids, {"state_frames": state_hash, **{key: atoms[0].source_hashes[key] for key in contract_keys}},
        next(iter(trajectory_hashes)), maximum, "mujoco_bridge_converted_urdf_write_qpos_v1",
    )


def _route_a_image(atom: WorkflowAtom, camera: str) -> np.ndarray:
    if atom.projection_spec is None or atom.source_state_id is None:
        raise ValueError("Route-A workflow image requires the frozen RT6 projection manifest")
    manifest = json.loads(atom.projection_spec.manifest_path.read_text(encoding="utf-8"))
    asset = manifest["assets"][f"observation/{camera}/{atom.source_state_id}/rgb"]
    path = Path(asset["path"])
    if _sha_file(path) != asset["sha256"]:
        raise ValueError("Route-A workflow image hash mismatch")
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Route-A workflow image cannot be decoded")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _roi_for(baseline: np.ndarray, ours: np.ndarray) -> np.ndarray:
    if baseline.shape != ours.shape:
        # The paired real-observation tile has different calibrated views; its
        # complete rectangular panels are the honest diagnostic support.
        return np.ones(baseline.shape[:2], dtype=bool)
    difference = cv2.cvtColor(cv2.absdiff(baseline, ours), cv2.COLOR_RGB2GRAY)
    threshold = max(4, int(np.percentile(difference, 85)))
    roi = cv2.dilate((difference >= threshold).astype(np.uint8), np.ones((9, 9), np.uint8), iterations=1).astype(bool)
    return roi if roi.any() else np.ones(baseline.shape[:2], dtype=bool)


def _projected_pair(atom: WorkflowAtom, first_qpos: Sequence[float], second_qpos: Sequence[float], kind: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, Mapping[str, object]]:
    spec = atom.projection_spec
    if spec is None or atom.pair is None:
        raise ValueError("projected workflow diagnostic requires an RT6 source pair")
    source = atom.pair.baseline_source.load()
    qpos_provenance: dict[str, object] = {"kind": kind, "state_frame_row": _row_hash(atom.state_frame)}
    if kind == "recorded_qpos_interface_projected_diagnostic_v1":
        qpos_provenance.update({
            "mujoco_write_validated": True,
            "native_gaussian_renderer_called": False,
            "visual_backend": "cpu_projected_centers_diagnostic",
        })
    first = ProjectedSpec(spec.manifest_path, "head", tuple(float(value) for value in first_qpos), tuple(float(value) for value in second_qpos), point_radius=spec.point_radius, anchors_per_link=spec.anchors_per_link, focus_links=spec.focus_links, qpos_provenance=qpos_provenance)
    first_mask = _project(first, first.baseline_qpos, source.shape[:2])
    second_mask = _project(first, first.ours_qpos, source.shape[:2])
    baseline = _draw_projection(source, first_mask, "measured")
    ours = _draw_projection(source, second_mask, "ours")
    roi = cv2.dilate((first_mask | second_mask).astype(np.uint8), np.ones((11, 11), np.uint8), iterations=1).astype(bool)
    return baseline, ours, roi, first.provenance()


def _stage_components(atom: WorkflowAtom) -> tuple[np.ndarray, np.ndarray, np.ndarray, Mapping[str, object]]:
    """Produce one source-specific, non-metric diagnostic pair per workflow stage."""

    if atom.visual_recipe == "route_a_paired_real":
        baseline, ours = _route_a_image(atom, "head"), _route_a_image(atom, "extra")
        return baseline, ours, _roi_for(baseline, ours), {"kind": "route_a_paired_real_observation_v1", "camera_pair": ["head", "extra"]}
    baseline, ours, roi, generator = _materialize(atom.pair)
    if atom.visual_recipe in {"rt9_temporal", "rt8_gaussian_render", "guard_commit_rollback", "rt9_observation_interface"}:
        return baseline, ours, roi, {**dict(generator), "kind": f"{atom.visual_recipe}_frozen_source_v1"}
    if atom.visual_recipe == "cross_view_residual":
        residual = cv2.cvtColor(cv2.absdiff(baseline, ours), cv2.COLOR_RGB2GRAY)
        gray = cv2.cvtColor(residual, cv2.COLOR_GRAY2RGB)
        heat = cv2.cvtColor(cv2.applyColorMap(residual, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
        return gray, heat, _roi_for(gray, heat), {"kind": "cross_view_residual_from_frozen_pixels_v1", "source_generator": generator}
    if atom.visual_recipe == "candidate_vs_measured":
        return _projected_pair(atom, atom.state_frame["measured_qpos"], atom.state_frame["candidate_qpos"], "candidate_vs_measured_projected_diagnostic_v1")
    if atom.visual_recipe == "recorded_qpos_interface":
        return _projected_pair(atom, atom.state_frame["mujoco_qpos"], atom.state_frame["gaussian_render_qpos"], "recorded_qpos_interface_projected_diagnostic_v1")
    raise AssertionError(f"unhandled workflow visual recipe: {atom.visual_recipe}")


def _component_tile(image: np.ndarray, *, role: str) -> np.ndarray:
    color = _rgb("primary_baseline" if role == "baseline" else "ours_full")
    available, border = 1508, 12
    scale = min(available / image.shape[1], available / image.shape[0])
    resized = cv2.resize(image, (round(image.shape[1] * scale), round(image.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    canvas = np.full((FRAME_SIZE, FRAME_SIZE, 3), 255, np.uint8)
    canvas[:border] = color; canvas[-border:] = color; canvas[:, :border] = color; canvas[:, -border:] = color
    top, left = border + (available - resized.shape[0]) // 2, (FRAME_SIZE - resized.shape[1]) // 2
    canvas[top : top + resized.shape[0], left : left + resized.shape[1]] = resized
    return canvas


def _comparison_tile(baseline: np.ndarray, ours: np.ndarray) -> np.ndarray:
    """Build one title-free square atom with wide, vertically stacked role panels."""

    canvas = np.full((FRAME_SIZE, FRAME_SIZE, 3), 255, np.uint8)
    margin, gap, border = 20, 20, 10
    panel_width = FRAME_SIZE - 2 * margin
    panel_height = (FRAME_SIZE - 2 * margin - gap) // 2
    for image, color, top in (
        (baseline, _rgb("primary_baseline"), margin),
        (ours, _rgb("ours_full"), margin + panel_height + gap),
    ):
        available_width, available_height = panel_width - 2 * border, panel_height - 2 * border
        scale = min(available_width / image.shape[1], available_height / image.shape[0])
        resized = cv2.resize(image, (round(image.shape[1] * scale), round(image.shape[0] * scale)), interpolation=cv2.INTER_AREA)
        x = margin + border + (available_width - resized.shape[1]) // 2
        y = top + border + (available_height - resized.shape[0]) // 2
        cv2.rectangle(canvas, (margin, top), (margin + panel_width, top + panel_height), color, border)
        canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def _frame_record(atom: WorkflowAtom, image: np.ndarray, baseline_component: np.ndarray, ours_component: np.ndarray, destination: Path, generator: Mapping[str, object]) -> PaperArtifactRecord:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 6]):
        raise RuntimeError("could not write illustrative workflow frame")
    hashes = dict(atom.source_hashes)
    hashes["generator"] = _sha_bytes(_canonical(generator).encode("ascii"))
    hashes["baseline_component_pixels"] = _sha_bytes(np.ascontiguousarray(baseline_component).tobytes())
    hashes["ours_component_pixels"] = _sha_bytes(np.ascontiguousarray(ours_component).tobytes())
    hashes["output_input_pixels"] = _sha_bytes(np.ascontiguousarray(image).tobytes())
    caption = (
        "Recorded qpos interface consistency shown as a CPU projected diagnostic."
        if atom.visual_recipe == "recorded_qpos_interface"
        else "Frozen illustrative workflow state."
    )
    return PaperArtifactRecord(
        artifact_id=atom.artifact_id, artifact_kind=ArtifactKind.FRAME, stage="LOOP",
        claim=atom.claim, source_run="frozen-rt6-rt7-rt8-rt9-illustrative-loop", source_hashes=hashes,
        evidence_level=EvidenceLevel.ILLUSTRATIVE, method="ours_full", baseline="primary_baseline",
        case_id=atom.source_case_id, camera=atom.camera, selection_policy="fixed_eight_stage_illustrative_schedule",
        comparison_metrics={}, width=FRAME_SIZE, height=FRAME_SIZE, recommended_title="Illustrative workflow atom",
        recommended_subtitle=f"{atom.trajectory_label} | {atom.stage}", caption_draft=caption,
        output_filenames={"png": destination.name}, output_hashes={"png": _sha_file(destination)},
    )


def _write_vector_exports(canvas: np.ndarray, destination: Path) -> tuple[Path, Path]:
    figure = plt.figure(figsize=(canvas.shape[1] / 100.0, canvas.shape[0] / 100.0), dpi=100)
    axis = figure.add_axes((0, 0, 1, 1)); axis.imshow(canvas); axis.set_axis_off()
    pdf, svg = destination.with_suffix(".pdf"), destination.with_suffix(".svg")
    figure.savefig(pdf, dpi=100, pad_inches=0); figure.savefig(svg, dpi=100, pad_inches=0); plt.close(figure)
    return pdf, svg


def _constituent_bindings(atoms: Sequence[WorkflowAtom]) -> dict[str, str]:
    """Bind composite evidence to every published atom and its complete source set."""

    if len(atoms) != 8:
        raise ValueError("composite provenance requires the exact eight workflow atoms")
    hashes: dict[str, str] = {}
    set_payload: list[dict[str, str]] = []
    for index, atom in enumerate(atoms, start=1):
        record = atom.frame_record
        if record is None:
            raise ValueError("composite provenance requires materialized frame records")
        canonical_digest = _sha_bytes(record.canonical_json().encode("ascii"))
        source_digest = _sha_bytes(_canonical(dict(record.source_hashes)).encode("ascii"))
        output_hash = record.output_hashes["png"]
        hashes[f"constituent_{index:02d}_canonical"] = canonical_digest
        hashes[f"constituent_{index:02d}_source_digest"] = source_digest
        hashes[f"constituent_{index:02d}_output_png"] = output_hash
        set_payload.append({"artifact_id": record.artifact_id, "canonical": canonical_digest, "source": source_digest, "output_png": output_hash})
    hashes["constituent_set_digest"] = _sha_bytes(_canonical(set_payload).encode("ascii"))
    for key in ("urdf", "converted_urdf", "conversion_manifest", "mujoco_model_contract"):
        hashes[key] = atoms[0].source_hashes[key]
    return hashes


def _compose_workflow_grid(atoms: Sequence[WorkflowAtom], destination: Path, artifact_id: str) -> PaperArtifactRecord:
    if len(atoms) != 8:
        raise ValueError("workflow grids require the exact eight-stage inventory")
    width = 8 * _GRID_CELL_SIZE + 7 * _GRID_GAP
    height = 2 * (_GRID_CELL_SIZE + _GRID_FOOTER_HEIGHT) + _GRID_GAP
    canvas = np.full((height, width, 3), 255, np.uint8)
    source_hashes: dict[str, str] = {"trajectory": atoms[0].source_hashes["trajectory"], **_constituent_bindings(atoms)}
    for row, role in enumerate(("baseline", "ours")):
        for column, atom in enumerate(atoms):
            record = atom.frame_record
            image = atom.baseline_component if role == "baseline" else atom.ours_component
            if record is None or image is None:
                raise ValueError("workflow grid requires bound atom components")
            expected = record.source_hashes[f"{role}_component_pixels"]
            if _sha_bytes(np.ascontiguousarray(image).tobytes()) != expected:
                raise ValueError("workflow component binding changed before grid composition")
            x, y = column * (_GRID_CELL_SIZE + _GRID_GAP), row * (_GRID_CELL_SIZE + _GRID_FOOTER_HEIGHT + _GRID_GAP)
            canvas[y : y + _GRID_CELL_SIZE, x : x + _GRID_CELL_SIZE] = cv2.resize(image, (_GRID_CELL_SIZE, _GRID_CELL_SIZE), interpolation=cv2.INTER_AREA)
            label = f"stage {column + 1:02d} | {role}"
            color = _rgb("primary_baseline" if role == "baseline" else "ours_full")
            cv2.putText(canvas, label, (x + 8, y + _GRID_CELL_SIZE + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 1, cv2.LINE_AA)
            source_hashes[f"atom_{row}_{column}"] = expected
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 6]):
        raise RuntimeError("could not write workflow grid")
    pdf, svg = _write_vector_exports(canvas, destination)
    source_hashes["grid_geometry"] = _sha_bytes(f"8x2;cell={_GRID_CELL_SIZE};footer={_GRID_FOOTER_HEIGHT};gap={_GRID_GAP}".encode("ascii"))
    source_hashes["grid_schedule"] = _sha_bytes("|".join(atom.artifact_id for atom in atoms).encode("ascii"))
    output_paths = {"png": destination, "pdf": pdf, "svg": svg}
    return PaperArtifactRecord(
        artifact_id=artifact_id, artifact_kind=ArtifactKind.GRID, stage="LOOP",
        claim="Illustrative frozen workflow sequence.", source_run="frozen-rt6-rt7-rt8-rt9-illustrative-loop",
        source_hashes=source_hashes, evidence_level=EvidenceLevel.ILLUSTRATIVE, method="ours_full", baseline="primary_baseline",
        case_id="illustrative-eight-stage-loop", camera="heterogeneous-source-cameras", selection_policy="fixed_heterogeneous_source_schedule",
        comparison_metrics={}, width=width, height=height, recommended_title="Illustrative workflow grid",
        recommended_subtitle="illustrative_scripted_trajectory | heterogeneous frozen stage sequence",
        caption_draft="Illustrative frozen workflow progression.", output_filenames={kind: path.name for kind, path in output_paths.items()},
        output_hashes={kind: _sha_file(path) for kind, path in output_paths.items()}, grid_layout="8x2",
    )


def _shared_head_temporal_components(atoms: Sequence[WorkflowAtom]) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray, Mapping[str, str]], ...]:
    """Project every bounded qpos state into one Route-A head source/crop."""

    if len(atoms) != 8 or atoms[0].projection_spec is None:
        raise ValueError("temporal workflow sequence requires the RT6 Route-A head projection contract")
    source = _route_a_image(atoms[0], "head")
    spec = atoms[0].projection_spec
    source_hash = atoms[0].source_hashes["route_a_head_rgb"]
    rows: list[tuple[np.ndarray, np.ndarray, np.ndarray, Mapping[str, str]]] = []
    for index, atom in enumerate(atoms, start=1):
        projected = ProjectedSpec(spec.manifest_path, "head", atom.baseline_qpos, atom.qpos, point_radius=spec.point_radius, anchors_per_link=spec.anchors_per_link, qpos_provenance={"kind": "shared_route_a_head_temporal_projection_v1", "stage": index, "trajectory": atom.source_hashes["trajectory"]})
        baseline_mask = _project(projected, projected.baseline_qpos, source.shape[:2])
        ours_mask = _project(projected, projected.ours_qpos, source.shape[:2])
        baseline = _draw_projection(source, baseline_mask, "measured")
        ours = _draw_projection(source, ours_mask, "ours")
        roi = cv2.dilate((baseline_mask | ours_mask).astype(np.uint8), np.ones((11, 11), np.uint8), iterations=1).astype(bool)
        hashes = {"route_a_head_rgb": source_hash, "trajectory": atom.source_hashes["trajectory"], "generator": _sha_bytes(_canonical(projected.provenance()).encode("ascii")), "state_frame": atom.source_hashes["rt6_state_frames"]}
        rows.append((baseline, ours, roi, hashes))
    return tuple(rows)


def _compose_temporal_grid(atoms: Sequence[WorkflowAtom], destination: Path) -> PaperArtifactRecord:
    components = _shared_head_temporal_components(atoms)
    width = 8 * _GRID_CELL_SIZE + 7 * _GRID_GAP
    height = 2 * (_GRID_CELL_SIZE + _GRID_FOOTER_HEIGHT) + _GRID_GAP
    canvas = np.full((height, width, 3), 255, np.uint8)
    source_hashes: dict[str, str] = {"trajectory": atoms[0].source_hashes["trajectory"], **_constituent_bindings(atoms)}
    for row, role in enumerate(("baseline", "ours")):
        for column, (baseline, ours, _, hashes) in enumerate(components):
            image = baseline if role == "baseline" else ours
            component = _component_tile(image, role=role)
            x, y = column * (_GRID_CELL_SIZE + _GRID_GAP), row * (_GRID_CELL_SIZE + _GRID_FOOTER_HEIGHT + _GRID_GAP)
            canvas[y : y + _GRID_CELL_SIZE, x : x + _GRID_CELL_SIZE] = cv2.resize(component, (_GRID_CELL_SIZE, _GRID_CELL_SIZE), interpolation=cv2.INTER_AREA)
            cv2.putText(canvas, f"t={column + 1:02d} | {role}", (x + 8, y + _GRID_CELL_SIZE + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.50, _rgb("primary_baseline" if role == "baseline" else "ours_full"), 1, cv2.LINE_AA)
            source_hashes[f"component_{row}_{column}"] = _sha_bytes(np.ascontiguousarray(component).tobytes())
            source_hashes[f"generator_{column}"] = hashes["generator"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 6]):
        raise RuntimeError("could not write temporal workflow grid")
    pdf, svg = _write_vector_exports(canvas, destination)
    source_hashes["grid_geometry"] = _sha_bytes(f"8x2;cell={_GRID_CELL_SIZE};footer={_GRID_FOOTER_HEIGHT};gap={_GRID_GAP}".encode("ascii"))
    source_hashes["shared_route_a_head_schedule"] = _sha_bytes("|".join(f"{index:02d}" for index in range(1, 9)).encode("ascii"))
    outputs = {"png": destination, "pdf": pdf, "svg": svg}
    return PaperArtifactRecord(
        artifact_id="G-LOOP-temporal-8x2", artifact_kind=ArtifactKind.GRID, stage="LOOP", claim="Illustrative shared Route-A head temporal diagnostic.", source_run="frozen-rt6-rt7-rt8-rt9-illustrative-loop", source_hashes=source_hashes, evidence_level=EvidenceLevel.ILLUSTRATIVE, method="ours_full", baseline="primary_baseline", case_id="illustrative-shared-route-a-head-trajectory", camera="head", selection_policy="one_route_a_head_source_and_fixed_bounded_qpos_schedule", comparison_metrics={}, width=width, height=height, recommended_title="Illustrative temporal grid", recommended_subtitle="illustrative_scripted_trajectory | shared Route-A head source and crop schedule", caption_draft="Illustrative shared-camera temporal progression.", output_filenames={kind: path.name for kind, path in outputs.items()}, output_hashes={kind: _sha_file(path) for kind, path in outputs.items()}, grid_layout="8x2",
    )


def _loop_video(atoms: Sequence[WorkflowAtom], destination: Path) -> PaperArtifactRecord:
    frames: list[VideoFrame] = []
    for index, (baseline, ours, roi, hashes) in enumerate(_shared_head_temporal_components(atoms), start=1):
        frames.append(VideoFrame(f"F-LOOP-{index:02d}", "head", f"{index:02d}", baseline, ours, roi, hashes))
    frames = list(_held_schedule(tuple(frames), frame_count=36))
    label = "illustrative_scripted_trajectory | shared Route-A head source/crop schedule"
    with tempfile.TemporaryDirectory(prefix="kinesync-loop-video-") as temporary:
        temporary_root = Path(temporary)
        baseline_spec = VideoSpec("V-LOOP-baseline", "LOOP", "frozen-rt6-rt7-rt8-rt9-illustrative-loop", "head", EvidenceLevel.ILLUSTRATIVE, "baseline", tuple(frames), label)
        ours_spec = replace(baseline_spec, artifact_id="V-LOOP-ours", mode="ours")
        side_spec = replace(baseline_spec, artifact_id="V-LOOP-illustrative", mode="side_by_side")
        baseline_path, ours_path = temporary_root / "V-LOOP-baseline.mp4", temporary_root / "V-LOOP-ours.mp4"
        write_method_video(baseline_spec, baseline_path)
        write_method_video(ours_spec, ours_path)
        receipt = verify_video_pair(baseline_path, ours_path)
        if receipt.differing_frame_fraction < 0.20:
            raise ValueError("illustrative loop failed visual contrast receipt")
        record = write_method_video(side_spec, destination)
        record = replace(record, source_hashes={
            **dict(record.source_hashes),
            **_constituent_bindings(atoms),
            "complete_schedule_source_digest": record.source_hashes["shared_schedule"],
            "loop_video_contrast_receipt": receipt.sha256,
        })
    if record.evidence_level is not EvidenceLevel.ILLUSTRATIVE:
        raise ValueError("illustrative loop video lost its evidence level")
    return record


def build_workflow_evidence(config_path: Path, output_dir: Path) -> WorkflowBuild:
    """Materialize exactly eight pairs, two 8x2 grids, and one loop MP4."""

    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    atoms = list(frozen_workflow_inventory(config_path))
    validate_illustrative_trajectory(atoms)
    records: list[PaperArtifactRecord] = []
    materialized: list[WorkflowAtom] = []
    for atom in atoms:
        if atom.visual_recipe != "route_a_paired_real":
            if atom.pair is None:
                raise ValueError("frozen workflow stage lost its pair binding")
            checked = atom.pair.with_receipt()
            if atom.pair.receipt is None or checked.receipt.sha256 != atom.pair.receipt.sha256:
                raise ValueError("workflow atom source provenance changed before materialization")
        baseline, ours, _, generator = _stage_components(atom)
        baseline_component = _component_tile(baseline, role="baseline")
        ours_component = _component_tile(ours, role="ours")
        tile = _comparison_tile(baseline, ours)
        frame_record = _frame_record(atom, tile, baseline_component, ours_component, output_dir / f"{atom.artifact_id}.png", generator)
        records.append(frame_record)
        materialized.append(replace(atom, frame_record=frame_record, baseline_component=baseline_component, ours_component=ours_component))
    atoms = materialized
    grids = (
        _compose_workflow_grid(atoms, output_dir / "G-LOOP-workflow-8x2.png", "G-LOOP-workflow-8x2"),
        _compose_temporal_grid(atoms, output_dir / "G-LOOP-temporal-8x2.png"),
    )
    video = _loop_video(atoms, output_dir / "V-LOOP-illustrative.mp4")
    if len(atoms) != 8 or len(records) != 8 or len(grids) != 2:
        raise AssertionError("Task 5 output inventory must close at 8 frames, 2 grids, and 1 video")
    return WorkflowBuild(tuple(atoms), tuple(records), grids, video)


__all__ = [
    "IllustrativeTrajectory", "WORKFLOW_STAGE_ORDER", "WorkflowAtom", "WorkflowBuild",
    "build_workflow_evidence", "frozen_workflow_inventory", "validate_illustrative_trajectory",
]
