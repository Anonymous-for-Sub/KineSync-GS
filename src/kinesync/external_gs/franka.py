"""Audited GS-Playground Franka asset adapter.

The source PLYs and MJCF remain read-only.  This module explicitly keeps
same-asset synthetic render targets separate from independent mesh references.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
from pathlib import Path
import struct
import tempfile
import xml.etree.ElementTree as ET
from typing import Mapping, Sequence

import numpy as np
import torch

from kinesync.geometry.urdf import axis_angle_matrix, homogeneous


_DEFAULT_XML = "xmls/panda_robotiq_224.xml"
_DEFAULT_PLY_DIRECTORY = "3dgs/franka_224"
_JOINT_NAMES = tuple(f"joint{index}" for index in range(1, 8))
_LINK_NAMES = {index: f"link{index}" for index in range(8)}
_PLY_SCALAR_TYPES = {
    "char": "b", "uchar": "B", "int8": "b", "uint8": "B",
    "short": "h", "ushort": "H", "int16": "h", "uint16": "H",
    "int": "i", "uint": "I", "int32": "i", "uint32": "I",
    "float": "f", "float32": "f", "double": "d", "float64": "d",
}


@dataclass(frozen=True)
class ExternalGSProvenance:
    source_root: Path
    mjcf_path: Path
    asset_sha256: str
    reference_kind: str = "analysis_same_gs"
    is_independent_ground_truth: bool = False


@dataclass(frozen=True)
class ExternalCameraCalibration:
    """Structural camera contract accepted by the project observation backends."""

    name: str
    intrinsic: np.ndarray
    world_to_camera: np.ndarray
    camera_to_world: np.ndarray
    width: int
    height: int


@dataclass(frozen=True)
class ExternalRenderedObservation:
    values: torch.Tensor
    valid: torch.Tensor


class ExternalSoftOccupancyBackend:
    """CPU differentiable diagnostic renderer for audited external GS centers."""

    def __init__(
        self,
        kinematics: "MjcfTorchKinematics",
        local_xyz: torch.Tensor,
        link_index: torch.Tensor,
        link_names: Mapping[int, str],
        camera: ExternalCameraCalibration,
        output_size: tuple[int, int],
    ) -> None:
        self.kinematics = kinematics
        self.local_xyz = local_xyz
        self.link_index = link_index
        self.link_names = dict(link_names)
        self.camera = camera
        self.cameras = {camera.name: camera}
        self.output_size = output_size

    def render(self, qpos: torch.Tensor) -> dict[str, ExternalRenderedObservation]:
        transforms = self.kinematics.forward(qpos)
        matrices = torch.stack([transforms[self.link_names[index]] for index in sorted(self.link_names)])
        frames = matrices[self.link_index.to(qpos.device)]
        points = torch.bmm(
            frames[:, :3, :3], self.local_xyz.to(qpos).unsqueeze(-1)
        ).squeeze(-1) + frames[:, :3, 3]
        camera = self.camera
        world_to_camera = torch.as_tensor(camera.world_to_camera, dtype=qpos.dtype, device=qpos.device)
        intrinsic = torch.as_tensor(camera.intrinsic, dtype=qpos.dtype, device=qpos.device)
        camera_xyz = points @ world_to_camera[:3, :3].transpose(0, 1) + world_to_camera[:3, 3]
        depth = camera_xyz[:, 2]
        pixels = torch.stack((
            intrinsic[0, 0] * camera_xyz[:, 0] / depth.clamp_min(1e-4) + intrinsic[0, 2],
            intrinsic[1, 1] * camera_xyz[:, 1] / depth.clamp_min(1e-4) + intrinsic[1, 2],
        ), dim=-1)
        height, width = self.output_size
        scale = pixels.new_tensor([width / camera.width, height / camera.height])
        pixels = pixels * scale
        valid = (depth > 1e-4) & torch.isfinite(pixels).all(dim=-1)
        y, x = torch.meshgrid(torch.arange(height, dtype=qpos.dtype, device=qpos.device), torch.arange(width, dtype=qpos.dtype, device=qpos.device), indexing="ij")
        grid = torch.stack((x, y), dim=-1).reshape(-1, 2)
        squared_distance = (pixels[:, None, :] - grid[None, :, :]).square().sum(dim=-1)
        density = (torch.exp(-0.5 * squared_distance / (1.35 * 1.35)) * valid[:, None]).sum(dim=0)
        occupancy = (1.0 - torch.exp(-0.7 * density)).reshape(height, width)
        return {camera.name: ExternalRenderedObservation(occupancy, torch.ones_like(occupancy, dtype=torch.bool))}

    def loss(self, qpos: torch.Tensor, target: Mapping[str, ExternalRenderedObservation]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        prediction = self.render(qpos)
        terms = {name: (prediction[name].values - target[name].values.to(qpos)).square().mean() for name in prediction}
        return torch.stack(list(terms.values())).mean(), terms


@dataclass(frozen=True)
class FKParityReport:
    state_count: int
    max_translation_error_m: float
    max_rotation_abs_error: float


@dataclass(frozen=True)
class _Joint:
    name: str
    axis: tuple[float, float, float]
    qpos_index: int


@dataclass(frozen=True)
class _Body:
    name: str
    parent: str | None
    position: tuple[float, float, float]
    quaternion_wxyz: tuple[float, float, float, float]
    joint: _Joint | None


class MjcfTorchKinematics:
    """Differentiable FK for the explicit one-hinge-per-body Panda chain."""

    def __init__(self, mjcf_path: str | Path, *, body_names: Sequence[str]):
        self.mjcf_path = Path(mjcf_path).expanduser().resolve()
        if not self.mjcf_path.is_file():
            raise FileNotFoundError(f"MJCF does not exist: {self.mjcf_path}")
        requested = tuple(body_names)
        root = ET.parse(self.mjcf_path).getroot()
        worldbody = root.find("worldbody")
        if worldbody is None:
            raise ValueError("MJCF has no worldbody")
        bodies: dict[str, _Body] = {}
        active: list[_Joint] = []

        def read_body(element: ET.Element, parent: str | None) -> None:
            name = element.attrib.get("name")
            if not name:
                raise ValueError("MJCF body has no name")
            if name in bodies:
                raise ValueError(f"duplicate MJCF body: {name}")
            position = _numbers(element.attrib.get("pos", "0 0 0"), 3)
            quaternion = _numbers(element.attrib.get("quat", "1 0 0 0"), 4)
            joints = element.findall("joint")
            if len(joints) > 1:
                raise ValueError(f"body {name} has multiple joints; unsupported contract")
            joint = None
            if joints:
                item = joints[0]
                if item.attrib.get("type", "hinge") != "hinge":
                    raise ValueError(f"body {name} joint is not hinge")
                joint_name = item.attrib.get("name")
                if not joint_name:
                    raise ValueError(f"body {name} hinge has no name")
                joint = _Joint(joint_name, _numbers(item.attrib.get("axis", "0 0 1"), 3), len(active))
                active.append(joint)
            bodies[name] = _Body(name, parent, position, quaternion, joint)
            for child in element.findall("body"):
                read_body(child, name)

        for body in worldbody.findall("body"):
            read_body(body, None)
        missing = set(requested) - set(bodies)
        if missing:
            raise ValueError(f"MJCF misses requested bodies: {sorted(missing)}")
        self.bodies = bodies
        self.link_names = requested
        self.active_joint_names = [joint.name for joint in active]
        if tuple(self.active_joint_names[: len(_JOINT_NAMES)]) != _JOINT_NAMES:
            raise ValueError(f"unexpected Panda joint order: {self.active_joint_names}")
        self.active_joint_names = self.active_joint_names[: len(_JOINT_NAMES)]

    def forward(self, qpos: torch.Tensor) -> dict[str, torch.Tensor]:
        if not isinstance(qpos, torch.Tensor):
            qpos = torch.as_tensor(qpos)
        if not qpos.is_floating_point():
            qpos = qpos.to(torch.get_default_dtype())
        if qpos.ndim != 1 or qpos.shape[0] != len(self.active_joint_names):
            raise ValueError(f"Expected {len(self.active_joint_names)} qpos values, got {tuple(qpos.shape)}")
        resolved: dict[str, torch.Tensor] = {}

        def pose(name: str) -> torch.Tensor:
            if name in resolved:
                return resolved[name]
            body = self.bodies[name]
            translation = torch.tensor(body.position, dtype=qpos.dtype, device=qpos.device)
            quaternion = torch.tensor(body.quaternion_wxyz, dtype=qpos.dtype, device=qpos.device)
            local = homogeneous(_quaternion_to_matrix(quaternion), translation)
            if body.joint is not None and body.joint.qpos_index < len(self.active_joint_names):
                axis = torch.tensor(body.joint.axis, dtype=qpos.dtype, device=qpos.device)
                local = local @ homogeneous(axis_angle_matrix(axis, qpos[body.joint.qpos_index]), torch.zeros(3, dtype=qpos.dtype, device=qpos.device))
            parent = torch.eye(4, dtype=qpos.dtype, device=qpos.device) if body.parent is None else pose(body.parent)
            resolved[name] = parent @ local
            return resolved[name]

        return {name: pose(name) for name in self.link_names}


@dataclass(frozen=True)
class FrankaExternalGS:
    root: Path
    provenance: ExternalGSProvenance
    kinematics: MjcfTorchKinematics
    joint_names: tuple[str, ...]
    link_names: Mapping[int, str]
    local_xyz: torch.Tensor
    link_index: torch.Tensor
    colors: torch.Tensor
    opacity_logit: torch.Tensor
    log_scale: torch.Tensor
    rotation_wxyz: torch.Tensor
    camera: ExternalCameraCalibration

    def anchors(self):
        # Importing kinesync.data initializes optional OpenCV dataset adapters.
        # Keep CPU-only asset/FK audits independent of that optional dependency.
        from kinesync.data.schema import GaussianAnchorSet
        count = self.local_xyz.shape[0]
        return GaussianAnchorSet(
            xyz=self.local_xyz.numpy().astype(np.float32, copy=False),
            surface_frame=_quaternion_to_matrix(self.rotation_wxyz).numpy().astype(np.float32, copy=False),
            link_index=self.link_index.numpy().astype(np.int64, copy=False),
            features_dc=((self.colors.numpy() - 0.5) / 0.28209479177387814).astype(np.float32, copy=False),
            features_rest_degree1=np.zeros((count, 3, 3), dtype=np.float32),
            opacity_logit=self.opacity_logit.numpy().astype(np.float32, copy=False),
            log_scale=self.log_scale.numpy().astype(np.float32, copy=False),
            rotation_wxyz=self.rotation_wxyz.numpy().astype(np.float32, copy=False),
        )

    def framed_analysis_camera(
        self,
        qpos_states: torch.Tensor,
        *,
        width: int = 640,
        height: int = 480,
        fovy_deg: float = 45.0,
        margin: float = 1.18,
    ) -> ExternalCameraCalibration:
        """Frame every external GS point across fixed analysis states.

        This is a declared virtual analysis camera, not a recovered external
        calibration or a real-data camera.  The same transform is supplied to
        the independent MuJoCo mesh reference.
        """
        return self.framed_analysis_cameras(
            qpos_states,
            camera_specs={"framed_analysis": (1.0, -1.0, 0.75)},
            width=width,
            height=height,
            fovy_deg=fovy_deg,
            margin=margin,
        )["framed_analysis"]

    def framed_analysis_cameras(
        self,
        qpos_states: torch.Tensor,
        *,
        camera_specs: Mapping[str, Sequence[float]],
        width: int = 640,
        height: int = 480,
        fovy_deg: float = 45.0,
        margin: float = 1.18,
    ) -> dict[str, ExternalCameraCalibration]:
        """Create fixed, distinct virtual analysis cameras around one state set.

        Every camera uses the same all-state external-point envelope.  Only the
        viewing direction differs, so this cannot be used to obtain a per-method
        framing advantage.  These are analysis cameras, not recovered real-world
        calibrations.
        """
        states = torch.as_tensor(qpos_states, dtype=torch.float64)
        if states.ndim != 2 or states.shape[1] != len(self.joint_names):
            raise ValueError(f"expected qpos states shaped (N, {len(self.joint_names)})")
        if width <= 0 or height <= 0 or not (0.0 < fovy_deg < 170.0) or margin <= 1.0:
            raise ValueError("invalid framed-camera dimensions, FoV, or margin")
        if not camera_specs or len(set(camera_specs)) != len(camera_specs):
            raise ValueError("camera_specs must contain unique camera names")
        points = []
        local = self.local_xyz.to(dtype=states.dtype)
        indices = self.link_index
        for qpos in states:
            transforms = self.kinematics.forward(qpos)
            matrices = torch.stack([transforms[self.link_names[index]] for index in sorted(self.link_names)])
            points.append(torch.bmm(matrices[indices, :3, :3], local.unsqueeze(-1)).squeeze(-1) + matrices[indices, :3, 3])
        world = torch.cat(points, dim=0)
        lower, upper = world.amin(dim=0), world.amax(dim=0)
        center = (lower + upper) * 0.5
        radius = torch.linalg.vector_norm(world - center, dim=-1).max().item()
        fovy = np.deg2rad(fovy_deg)
        focal = (height / 2.0) / np.tan(fovy / 2.0)
        fovx = 2.0 * np.arctan(width / (2.0 * focal))
        distance = radius / np.sin(min(fovy, fovx) * 0.5) * margin
        center_np = center.numpy()
        result: dict[str, ExternalCameraCalibration] = {}
        for name, raw_offset in camera_specs.items():
            if not str(name).strip():
                raise ValueError("analysis camera name must be nonempty")
            viewing_offset = np.asarray(raw_offset, dtype=np.float64)
            if viewing_offset.shape != (3,) or not np.isfinite(viewing_offset).all():
                raise ValueError("analysis camera direction must be a finite xyz vector")
            norm = np.linalg.norm(viewing_offset)
            if norm <= 1e-12:
                raise ValueError("analysis camera direction must be nonzero")
            viewing_offset /= norm
            eye = center_np + distance * viewing_offset
            forward = center_np - eye
            forward /= np.linalg.norm(forward)
            right = np.cross(forward, np.asarray([0.0, 0.0, 1.0]))
            right_norm = np.linalg.norm(right)
            if right_norm <= 1e-12:
                raise ValueError("analysis camera direction cannot be parallel to world up")
            right /= right_norm
            down = np.cross(forward, right)
            rotation = np.stack((right, down, forward), axis=0)
            world_to_camera = np.eye(4, dtype=np.float64)
            world_to_camera[:3, :3] = rotation
            world_to_camera[:3, 3] = -rotation @ eye
            result[str(name)] = ExternalCameraCalibration(
                name=str(name),
                intrinsic=np.asarray([[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]], dtype=np.float64),
                world_to_camera=world_to_camera,
                camera_to_world=np.linalg.inv(world_to_camera),
                width=width,
                height=height,
            )
        return result

    def point_visibility_fraction(
        self, qpos: torch.Tensor, camera: ExternalCameraCalibration
    ) -> float:
        """Return the fraction of external GS centers projected into one camera."""
        state = torch.as_tensor(qpos, dtype=torch.float64)
        transforms = self.kinematics.forward(state)
        matrices = torch.stack(
            [transforms[self.link_names[index]] for index in sorted(self.link_names)]
        )
        frames = matrices[self.link_index]
        points = torch.bmm(
            frames[:, :3, :3], self.local_xyz.to(dtype=state.dtype).unsqueeze(-1)
        ).squeeze(-1) + frames[:, :3, 3]
        world_to_camera = torch.as_tensor(camera.world_to_camera, dtype=state.dtype)
        intrinsic = torch.as_tensor(camera.intrinsic, dtype=state.dtype)
        camera_xyz = (
            points @ world_to_camera[:3, :3].transpose(0, 1)
            + world_to_camera[:3, 3]
        )
        depth = camera_xyz[:, 2]
        pixel_x = intrinsic[0, 0] * camera_xyz[:, 0] / depth.clamp_min(1e-8) + intrinsic[0, 2]
        pixel_y = intrinsic[1, 1] * camera_xyz[:, 1] / depth.clamp_min(1e-8) + intrinsic[1, 2]
        visible = (
            (depth > 1e-8)
            & (pixel_x >= 0.0)
            & (pixel_x < camera.width)
            & (pixel_y >= 0.0)
            & (pixel_y < camera.height)
        )
        return float(visible.to(torch.float64).mean())

    def software_backend(self, *, output_size: tuple[int, int], per_link: int = 96) -> ExternalSoftOccupancyBackend:
        selected = _uniform_link_sample(self.link_index.numpy(), per_link=per_link)
        return ExternalSoftOccupancyBackend(
            self.kinematics,
            self.local_xyz[selected],
            self.link_index[selected],
            self.link_names,
            self.camera,
            output_size,
        )

    def cuda_backend(
        self,
        *,
        output_size: tuple[int, int],
        per_link: int,
        device: torch.device,
        cameras: Mapping[str, ExternalCameraCalibration] | None = None,
    ):
        """Create the project's native CUDA rasterizer from external PLY fields."""
        from kinesync.observation.cuda_gaussian import CudaGaussianBackend

        selected = _uniform_link_sample(self.link_index.numpy(), per_link=per_link)
        return CudaGaussianBackend.from_route_a(
            kinematics=self.kinematics,
            anchors=self.anchors(),
            anchor_indices=selected,
            link_names=self.link_names,
            calibrations={"front": self.camera} if cameras is None else cameras,
            output_size=output_size,
            observation_mode="rgb",
            device=device,
        )


def load_franka_external_gs(root: str | Path) -> FrankaExternalGS:
    root_path = Path(root).expanduser().resolve()
    mjcf_path = root_path / _DEFAULT_XML
    ply_directory = root_path / _DEFAULT_PLY_DIRECTORY
    records = []
    digester = hashlib.sha256()
    for index in range(8):
        path = ply_directory / f"link{index}.ply"
        payload = _read_ply(path)
        for field in ("x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"):
            if field not in payload.dtype.names:
                raise ValueError(f"{path} misses required Gaussian field {field}")
        digester.update(path.read_bytes())
        records.append(payload)
    digester.update(mjcf_path.read_bytes())
    xyz = np.concatenate([np.column_stack((record["x"], record["y"], record["z"])) for record in records]).astype(np.float32)
    colors = np.concatenate([np.column_stack((record["f_dc_0"], record["f_dc_1"], record["f_dc_2"])) for record in records]).astype(np.float32)
    colors = np.clip(0.28209479177387814 * colors + 0.5, 0.0, 1.0)
    opacity = np.concatenate([record["opacity"] for record in records]).astype(np.float32)
    scale = np.concatenate([np.column_stack((record["scale_0"], record["scale_1"], record["scale_2"])) for record in records]).astype(np.float32)
    rotation = np.concatenate([np.column_stack((record["rot_0"], record["rot_1"], record["rot_2"], record["rot_3"])) for record in records]).astype(np.float32)
    rotation /= np.linalg.norm(rotation, axis=1, keepdims=True).clip(min=1e-8)
    link_index = np.concatenate([np.full(len(record), index, dtype=np.int64) for index, record in enumerate(records)])
    camera = _front_camera(root_path)
    return FrankaExternalGS(
        root=root_path,
        provenance=ExternalGSProvenance(root_path, mjcf_path, digester.hexdigest()),
        kinematics=MjcfTorchKinematics(mjcf_path, body_names=tuple(_LINK_NAMES.values())),
        joint_names=_JOINT_NAMES,
        link_names=_LINK_NAMES,
        local_xyz=torch.from_numpy(xyz),
        link_index=torch.from_numpy(link_index),
        colors=torch.from_numpy(colors),
        opacity_logit=torch.from_numpy(opacity),
        log_scale=torch.from_numpy(scale),
        rotation_wxyz=torch.from_numpy(rotation),
        camera=camera,
    )


def mujoco_fk_report(asset: FrankaExternalGS, states: Sequence[Sequence[float]] | np.ndarray) -> FKParityReport:
    mujoco = importlib.import_module("mujoco")
    values = np.asarray(states, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(asset.joint_names):
        raise ValueError(f"expected states shaped (N, {len(asset.joint_names)})")
    with tempfile.TemporaryDirectory(prefix="kinesync-franka-mjcf-") as directory:
        derived_mjcf = write_mujoco_mjcf(asset, Path(directory) / "panda_robotiq_224.absolute.xml")
        model = mujoco.MjModel.from_xml_path(str(derived_mjcf))
        return _mujoco_fk_report_from_model(mujoco, model, asset, values)


def write_mujoco_mjcf(asset: FrankaExternalGS, output: str | Path) -> Path:
    """Materialize a derived MJCF with absolute mesh paths for MuJoCo audits."""
    destination = Path(output).expanduser().resolve()
    source = asset.provenance.mjcf_path
    tree = ET.parse(source)
    root = tree.getroot()
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.Element("compiler")
        root.insert(0, compiler)
    compiler.set("meshdir", "")
    for mesh in root.findall(".//mesh"):
        relative = mesh.attrib.get("file")
        if relative is None:
            continue
        resolved = (source.parent / relative).resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"MJCF mesh does not exist: {resolved}")
        mesh.set("file", str(resolved))
    destination.parent.mkdir(parents=True, exist_ok=True)
    tree.write(destination, encoding="utf-8", xml_declaration=True)
    return destination


def _mujoco_fk_report_from_model(mujoco: object, model: object, asset: FrankaExternalGS, values: np.ndarray) -> FKParityReport:
    data = mujoco.MjData(model)
    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in asset.joint_names]
    addresses = [int(model.jnt_qposadr[index]) for index in joint_ids]
    body_ids = {name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in asset.link_names.values()}
    max_translation = 0.0
    max_rotation = 0.0
    for row in values:
        data.qpos[:] = 0.0
        data.qpos[addresses] = row
        mujoco.mj_forward(model, data)
        torch_frames = asset.kinematics.forward(torch.from_numpy(row))
        for name, body_id in body_ids.items():
            expected = torch_frames[name].detach().numpy()
            actual_rotation = data.xmat[body_id].reshape(3, 3)
            max_translation = max(max_translation, float(np.abs(expected[:3, 3] - data.xpos[body_id]).max()))
            max_rotation = max(max_rotation, float(np.abs(expected[:3, :3] - actual_rotation).max()))
    return FKParityReport(len(values), max_translation, max_rotation)


def _front_camera(root: Path) -> ExternalCameraCalibration:
    scene = root / "xmls/table30_02_stack_color_blocks.xml"
    if not scene.is_file():
        raise FileNotFoundError(f"external scene camera definition is unavailable: {scene}")
    camera_element = next(
        (item for item in ET.parse(scene).getroot().iter("camera") if item.attrib.get("name") == "front"),
        None,
    )
    if camera_element is None:
        raise ValueError(f"external scene has no front camera: {scene}")
    width, height = 640, 480
    fovy_deg = float(camera_element.attrib["fovy"])
    focal = (height / 2.0) / np.tan(np.deg2rad(fovy_deg) / 2.0)
    eye = np.asarray(_numbers(camera_element.attrib["pos"], 3), dtype=np.float64)
    axes = np.asarray(_numbers(camera_element.attrib["xyaxes"], 6), dtype=np.float64).reshape(2, 3)
    right, up = axes
    right /= np.linalg.norm(right)
    up /= np.linalg.norm(up)
    forward = -np.cross(right, up)
    forward /= np.linalg.norm(forward)
    rotation = np.stack((right, -up, forward), axis=0)
    world_to_camera = np.eye(4, dtype=np.float64)
    world_to_camera[:3, :3] = rotation
    world_to_camera[:3, 3] = -rotation @ eye
    return ExternalCameraCalibration(
        name="front",
        intrinsic=np.asarray([[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]], dtype=np.float64),
        world_to_camera=world_to_camera,
        camera_to_world=np.linalg.inv(world_to_camera),
        width=width,
        height=height,
    )


def _numbers(text: str, expected: int) -> tuple[float, ...]:
    values = tuple(float(value) for value in text.split())
    if len(values) != expected:
        raise ValueError(f"expected {expected} numeric values, got {text!r}")
    return values


def _quaternion_to_matrix(quaternion_wxyz: torch.Tensor) -> torch.Tensor:
    """Convert normalized-or-not wxyz quaternions without losing gradients."""
    q = torch.nn.functional.normalize(quaternion_wxyz, dim=-1, eps=1e-12)
    w, x, y, z = q.unbind(dim=-1)
    two = q.new_tensor(2.0)
    return torch.stack(
        (
            1 - two * (y.square() + z.square()), two * (x * y - z * w), two * (x * z + y * w),
            two * (x * y + z * w), 1 - two * (x.square() + z.square()), two * (y * z - x * w),
            two * (x * z - y * w), two * (y * z + x * w), 1 - two * (x.square() + y.square()),
        ),
        dim=-1,
    ).reshape(q.shape[:-1] + (3, 3))


def _read_ply(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Gaussian PLY is unavailable: {path}")
    with path.open("rb") as handle:
        header: list[str] = []
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f"truncated PLY header: {path}")
            decoded = line.decode("ascii").strip()
            header.append(decoded)
            if decoded == "end_header":
                break
        if not header or header[0] != "ply" or "format binary_little_endian 1.0" not in header:
            raise ValueError(f"only binary little-endian PLY is supported: {path}")
        vertex_line = next((line for line in header if line.startswith("element vertex ")), None)
        if vertex_line is None:
            raise ValueError(f"PLY has no vertex element: {path}")
        count = int(vertex_line.split()[2])
        properties = [line.split() for line in header if line.startswith("property ")]
        if any(len(parts) != 3 or parts[1] not in _PLY_SCALAR_TYPES for parts in properties):
            raise ValueError(f"PLY uses unsupported properties: {path}")
        dtype = np.dtype([(parts[2], "<" + _PLY_SCALAR_TYPES[parts[1]]) for parts in properties])
        expected_bytes = count * dtype.itemsize
        raw = handle.read(expected_bytes)
        if len(raw) != expected_bytes:
            raise ValueError(f"truncated PLY vertex records: {path}")
    return np.frombuffer(raw, dtype=dtype).copy()


def _uniform_link_sample(link_index: np.ndarray, *, per_link: int) -> np.ndarray:
    selected: list[np.ndarray] = []
    for index in sorted(set(link_index.tolist())):
        candidates = np.flatnonzero(link_index == index)
        count = min(per_link, len(candidates))
        selected.append(candidates[np.linspace(0, len(candidates) - 1, count, dtype=np.int64)])
    return np.concatenate(selected)
