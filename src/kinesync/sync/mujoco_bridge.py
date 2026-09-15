"""Apply one protected KineSync state to MuJoCo and audit FK parity."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from kinesync.geometry.urdf import TorchURDFKinematics

from .schema import SynchronizedStateFrame


@dataclass(frozen=True)
class FKParityReport:
    state_count: int
    compared_body_names: tuple[str, ...]
    max_translation_error_m: float
    max_rotation_abs_error: float
    max_qpos_error: float
    passed: bool


@dataclass
class MujocoStateBridge:
    model: object
    data: object
    joint_names: tuple[str, ...]
    joint_ids: tuple[int, ...]
    qpos_addresses: tuple[int, ...]
    dof_addresses: tuple[int, ...]
    joint_limits: dict[str, tuple[float, float]]
    _mujoco: object

    @classmethod
    def from_urdf(
        cls, path: str | Path, joint_names: Sequence[str]
    ) -> "MujocoStateBridge":
        mujoco = importlib.import_module("mujoco")
        source = Path(path).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"MuJoCo URDF does not exist: {source}")
        names = tuple(str(name) for name in joint_names)
        if not names or len(set(names)) != len(names):
            raise ValueError("joint_names must be nonempty and unique")

        model = mujoco.MjModel.from_xml_path(str(source))
        model_order = tuple(
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index)
            for index in range(model.njnt)
        )
        if model_order != names:
            raise ValueError(
                f"MuJoCo joint order {model_order} does not match requested joint order {names}"
            )
        if model.nq != len(names) or model.nv != len(names):
            raise ValueError(
                f"Expected one qpos/qvel per joint, got nq={model.nq}, nv={model.nv}"
            )

        joint_ids = tuple(range(model.njnt))
        qpos_addresses = tuple(int(model.jnt_qposadr[index]) for index in joint_ids)
        dof_addresses = tuple(int(model.jnt_dofadr[index]) for index in joint_ids)
        if qpos_addresses != tuple(range(len(names))):
            raise ValueError(f"Unexpected MuJoCo qpos addresses: {qpos_addresses}")
        if dof_addresses != tuple(range(len(names))):
            raise ValueError(f"Unexpected MuJoCo qvel addresses: {dof_addresses}")

        limits: dict[str, tuple[float, float]] = {}
        for name, joint_id in zip(names, joint_ids, strict=True):
            if not bool(model.jnt_limited[joint_id]):
                raise ValueError(f"MuJoCo joint {name} must have finite limits")
            lower, upper = (float(value) for value in model.jnt_range[joint_id])
            if not np.isfinite([lower, upper]).all() or lower >= upper:
                raise ValueError(f"MuJoCo joint {name} has invalid limits")
            limits[name] = (lower, upper)

        return cls(
            model=model,
            data=mujoco.MjData(model),
            joint_names=names,
            joint_ids=joint_ids,
            qpos_addresses=qpos_addresses,
            dof_addresses=dof_addresses,
            joint_limits=limits,
            _mujoco=mujoco,
        )

    def write_qpos(self, qpos: Sequence[float] | np.ndarray) -> None:
        values = np.asarray(qpos, dtype=np.float64)
        if values.shape != (len(self.joint_names),):
            raise ValueError(
                f"Expected {len(self.joint_names)} qpos values, got {values.shape}"
            )
        if not np.isfinite(values).all():
            raise ValueError("MuJoCo qpos values must be finite")
        for index, name in enumerate(self.joint_names):
            lower, upper = self.joint_limits[name]
            if values[index] < lower or values[index] > upper:
                raise ValueError(
                    f"MuJoCo joint {name} value {values[index]} is outside limit [{lower}, {upper}]"
                )
        self.data.qpos[list(self.qpos_addresses)] = values
        self._mujoco.mj_forward(self.model, self.data)

    def write(self, frame: SynchronizedStateFrame) -> None:
        if not isinstance(frame, SynchronizedStateFrame):
            raise TypeError("frame must be a SynchronizedStateFrame")
        if frame.joint_names != self.joint_names:
            raise ValueError("synchronized state joint order does not match MuJoCo")
        frame.validate_integrity()
        self.write_qpos(frame.synchronized_qpos.detach().cpu().numpy())
        accepted = set(frame.accepted_joint_names)
        for name, dof_address in zip(
            self.joint_names, self.dof_addresses, strict=True
        ):
            if name in accepted:
                self.data.qvel[dof_address] = 0.0
        self._mujoco.mj_forward(self.model, self.data)

    def body_transforms(self) -> dict[str, np.ndarray]:
        transforms: dict[str, np.ndarray] = {}
        for body_id in range(1, self.model.nbody):
            name = self._mujoco.mj_id2name(
                self.model, self._mujoco.mjtObj.mjOBJ_BODY, body_id
            )
            if name is None:
                continue
            transform = np.eye(4, dtype=np.float64)
            transform[:3, :3] = self.data.xmat[body_id].reshape(3, 3)
            transform[:3, 3] = self.data.xpos[body_id]
            transforms[name] = transform
        return transforms


def compare_fk_parity(
    bridge: MujocoStateBridge,
    fk: TorchURDFKinematics,
    qpos_states: Sequence[Sequence[float]] | np.ndarray,
    *,
    translation_tolerance_m: float = 1e-6,
    rotation_tolerance: float = 1e-5,
    qpos_tolerance: float = 1e-10,
) -> FKParityReport:
    states = np.asarray(qpos_states, dtype=np.float64)
    if states.ndim != 2 or states.shape[1] != len(bridge.joint_names):
        raise ValueError("parity states must be a matrix matching the joint order")
    if len(states) == 0 or not np.isfinite(states).all():
        raise ValueError("parity states must be nonempty and finite")
    if tuple(fk.active_joint_names) != bridge.joint_names:
        raise ValueError("Torch and MuJoCo joint order must match")

    compared_names: tuple[str, ...] | None = None
    max_translation = 0.0
    max_rotation = 0.0
    max_qpos = 0.0
    for state in states:
        bridge.write_qpos(state)
        torch_transforms = fk.forward(torch.as_tensor(state, dtype=torch.float64))
        mujoco_transforms = bridge.body_transforms()
        shared = tuple(
            name for name in fk.link_names if name in mujoco_transforms
        )
        if not shared:
            raise ValueError("MuJoCo and Torch FK have no shared link/body names")
        if compared_names is None:
            compared_names = shared
        elif compared_names != shared:
            raise ValueError("MuJoCo body set changed during parity evaluation")
        for name in shared:
            torch_transform = torch_transforms[name].detach().cpu().numpy()
            mujoco_transform = mujoco_transforms[name]
            max_translation = max(
                max_translation,
                float(np.linalg.norm(torch_transform[:3, 3] - mujoco_transform[:3, 3])),
            )
            max_rotation = max(
                max_rotation,
                float(np.max(np.abs(torch_transform[:3, :3] - mujoco_transform[:3, :3]))),
            )
        written = bridge.data.qpos[list(bridge.qpos_addresses)]
        max_qpos = max(max_qpos, float(np.max(np.abs(written - state))))

    passed = bool(
        max_translation <= translation_tolerance_m
        and max_rotation <= rotation_tolerance
        and max_qpos <= qpos_tolerance
    )
    return FKParityReport(
        state_count=len(states),
        compared_body_names=compared_names or (),
        max_translation_error_m=max_translation,
        max_rotation_abs_error=max_rotation,
        max_qpos_error=max_qpos,
        passed=passed,
    )


__all__ = ["FKParityReport", "MujocoStateBridge", "compare_fk_parity"]
