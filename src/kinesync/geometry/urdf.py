"""Differentiable, dependency-light URDF forward kinematics."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import torch

from .so3 import axis_angle_matrix, homogeneous, rpy_matrix


SUPPORTED_JOINTS = {"fixed", "revolute", "continuous", "prismatic"}


def _vector_attribute(
    element: ET.Element | None, name: str, default: tuple[float, float, float]
) -> tuple[float, float, float]:
    if element is None or name not in element.attrib:
        return default
    values = tuple(float(value) for value in element.attrib[name].split())
    if len(values) != 3:
        raise ValueError(f"Expected three values for {name}, got {element.attrib[name]!r}")
    return values


@dataclass(frozen=True)
class JointSpec:
    name: str
    kind: str
    parent: str
    child: str
    xyz: tuple[float, float, float]
    rpy: tuple[float, float, float]
    axis: tuple[float, float, float]
    qpos_index: int | None
    lower: float | None
    upper: float | None


class TorchURDFKinematics:
    """URDF tree FK preserving gradients to batched joint states."""

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser().resolve()
        root = ET.parse(self.path).getroot()
        self.link_names = [element.attrib["name"] for element in root.findall("link")]

        child_links: set[str] = set()
        joints: list[JointSpec] = []
        active_index = 0
        for element in root.findall("joint"):
            kind = element.attrib["type"]
            if kind not in SUPPORTED_JOINTS:
                raise ValueError(
                    f"Unsupported joint type {kind!r} for {element.attrib['name']}"
                )
            parent_element = element.find("parent")
            child_element = element.find("child")
            if parent_element is None or child_element is None:
                raise ValueError(f"Joint {element.attrib['name']} lacks parent or child")
            parent = parent_element.attrib["link"]
            child = child_element.attrib["link"]
            child_links.add(child)
            origin = element.find("origin")
            xyz = _vector_attribute(origin, "xyz", (0.0, 0.0, 0.0))
            rpy = _vector_attribute(origin, "rpy", (0.0, 0.0, 0.0))
            axis = _vector_attribute(element.find("axis"), "xyz", (1.0, 0.0, 0.0))
            axis_norm = sum(value * value for value in axis) ** 0.5
            if kind != "fixed" and axis_norm <= 0.0:
                raise ValueError(f"Joint {element.attrib['name']} has a zero axis")
            if axis_norm > 0.0:
                axis = tuple(value / axis_norm for value in axis)

            qpos_index = None if kind == "fixed" else active_index
            lower = upper = None
            if qpos_index is not None:
                limit = element.find("limit")
                if kind != "continuous":
                    if limit is None:
                        raise ValueError(f"Joint {element.attrib['name']} has no limits")
                    lower = float(limit.attrib["lower"])
                    upper = float(limit.attrib["upper"])
                active_index += 1
            joints.append(
                JointSpec(
                    name=element.attrib["name"],
                    kind=kind,
                    parent=parent,
                    child=child,
                    xyz=xyz,
                    rpy=rpy,
                    axis=axis,
                    qpos_index=qpos_index,
                    lower=lower,
                    upper=upper,
                )
            )

        roots = [name for name in self.link_names if name not in child_links]
        if len(roots) != 1:
            raise ValueError(f"Expected one URDF root link, found {roots}")
        self.root_link = roots[0]
        self.joints = joints
        self.active_joints = [joint for joint in joints if joint.qpos_index is not None]
        self.active_joint_names = [joint.name for joint in self.active_joints]
        self.joint_limits = {
            joint.name: (joint.lower, joint.upper) for joint in self.active_joints
        }
        self.children: dict[str, list[JointSpec]] = {}
        for joint in joints:
            self.children.setdefault(joint.parent, []).append(joint)

    def forward(self, qpos: torch.Tensor) -> dict[str, torch.Tensor]:
        if not isinstance(qpos, torch.Tensor):
            qpos = torch.as_tensor(qpos)
        if not qpos.is_floating_point():
            qpos = qpos.to(dtype=torch.get_default_dtype())
        expected = len(self.active_joints)
        if qpos.ndim == 0 or qpos.shape[-1] != expected:
            raise ValueError(f"Expected {expected} qpos values, got {tuple(qpos.shape)}")

        batch_shape = qpos.shape[:-1]
        identity = torch.eye(4, dtype=qpos.dtype, device=qpos.device).expand(
            batch_shape + (4, 4)
        )
        transforms: dict[str, torch.Tensor] = {self.root_link: identity}
        stack = [self.root_link]
        while stack:
            parent = stack.pop()
            for joint in self.children.get(parent, []):
                xyz = torch.tensor(joint.xyz, dtype=qpos.dtype, device=qpos.device)
                rpy = torch.tensor(joint.rpy, dtype=qpos.dtype, device=qpos.device)
                origin = homogeneous(rpy_matrix(rpy), xyz)

                motion_rotation = torch.eye(3, dtype=qpos.dtype, device=qpos.device)
                motion_translation = torch.zeros(3, dtype=qpos.dtype, device=qpos.device)
                if joint.kind in {"revolute", "continuous"}:
                    axis = torch.tensor(joint.axis, dtype=qpos.dtype, device=qpos.device)
                    motion_rotation = axis_angle_matrix(
                        axis, qpos[..., joint.qpos_index]
                    )
                elif joint.kind == "prismatic":
                    axis = torch.tensor(joint.axis, dtype=qpos.dtype, device=qpos.device)
                    motion_translation = qpos[..., joint.qpos_index, None] * axis
                motion = homogeneous(motion_rotation, motion_translation)
                transforms[joint.child] = transforms[parent] @ origin @ motion
                stack.append(joint.child)

        missing = set(self.link_names) - set(transforms)
        if missing:
            raise ValueError(f"Disconnected URDF links: {sorted(missing)}")
        return transforms
