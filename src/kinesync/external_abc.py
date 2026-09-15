"""CPU-only recording contracts for externally supplied ABC simulation."""

from dataclasses import dataclass, field
import os
from pathlib import Path

import numpy as np


def diagnostic_cases():
    return [
        {"id": f"{physics}_h{chunks}_s{seed}", "cell": cell,
         "physics": physics, "chunks": chunks, "seed": seed, "policy_seed": 123}
        for cell, (physics, chunks) in enumerate(
            (("vanilla", 120), ("warp", 120), ("vanilla", 240), ("warp", 240)))
        for seed in (42000, 42001, 42002)
    ]


def evaluation_options(*, worlds, seed, chunks, physics, policy_seed):
    if physics not in {"vanilla", "warp"}:
        raise ValueError("Physics must be vanilla or warp")
    if min(worlds, chunks) <= 0 or min(seed, policy_seed) < 0:
        raise ValueError("Counts must be positive and seeds nonnegative")
    return {"num_worlds": worlds, "seed": seed, "num_chunks": chunks,
            "vanilla_physics": physics == "vanilla", "policy_seed": policy_seed}


def make_recording_policy(base_class, calls):
    """Record actual policy inputs/outputs without replacing the RNG or actions."""
    class RecordingPolicy(base_class):
        def infer(self, obs, noise=None, **kwargs):
            record = {
                "state": np.array(obs["state"], copy=True),
                "noise": None if noise is None else np.array(noise, copy=True),
            }
            actions = super().infer(obs, noise=noise, **kwargs)
            record["actions"] = np.array(actions, copy=True)
            calls.append(record)
            return actions

    return RecordingPolicy


def require_gpu_allocation() -> str:
    job_id = os.environ.get("SLURM_JOB_ID", "")
    if not job_id:
        raise RuntimeError("GPU execution requires a Slurm allocation")
    assigned_ids = os.environ.get("SLURM_JOB_GPUS") or os.environ.get("SLURM_STEP_GPUS")
    count = os.environ.get("SLURM_GPUS_ON_NODE", "0")
    if not assigned_ids and (not count.isdigit() or int(count) < 1):
        raise RuntimeError("Slurm allocation has no explicit GPU resources")
    return job_id


@dataclass
class SimulationTrace:
    seed: int
    timestep: float
    rows: list = field(default_factory=list)

    def __post_init__(self):
        if not np.isfinite(self.timestep) or self.timestep <= 0:
            raise ValueError("Control timestep must be positive and finite")

    def append(self, *, qpos, qvel, state, action):
        arrays = [np.array(x, dtype=np.float64, copy=True) for x in (qpos, qvel, state)]
        if any(x.ndim != 1 or not np.isfinite(x).all() for x in arrays):
            raise ValueError("Trace state must contain finite vectors")
        if arrays[2].shape != (14,):
            raise ValueError("ABC policy state must have 14 components")
        if self.rows and any(x.shape != old.shape for x, old in zip(arrays, self.rows[0][:3])):
            raise ValueError("Trace dimensions changed within an episode")
        valid = action is not None
        value = np.zeros(14) if action is None else np.array(action, dtype=np.float64, copy=True)
        if value.shape != (14,) or not np.isfinite(value).all():
            raise ValueError("ABC action must be a finite 14-vector")
        self.rows.append((*arrays, value, valid))

    def save(self, root: Path) -> Path:
        if not self.rows:
            raise ValueError("Cannot save an empty simulation trace")
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"seed_{self.seed}__physical_trace.npz"
        with path.open("xb") as handle:
            np.savez_compressed(
                handle,
                qpos=np.stack([row[0] for row in self.rows]),
                qvel=np.stack([row[1] for row in self.rows]),
                policy_state=np.stack([row[2] for row in self.rows]),
                action=np.stack([row[3] for row in self.rows]),
                action_valid=np.array([row[4] for row in self.rows], dtype=bool),
                action_alignment=np.array("action[i] produces state[i] from state[i-1]; row 0 has no action"),
                transition_from_index=np.array([index - 1 if row[4] else -1 for index, row in enumerate(self.rows)]),
                transition_to_index=np.array([index if row[4] else -1 for index, row in enumerate(self.rows)]),
                time_seconds=np.arange(len(self.rows)) * self.timestep,
                provenance=np.array("simulator_state_from_policy_closed_loop"),
                seed=self.seed,
            )
        return path


def make_recording_env(base_class, output: Path, *, vanilla_physics: bool):
    """Wrap the public environment without changing policy observations or actions."""
    class RecordingEnv(base_class):
        def __init__(self, **kwargs):
            self.trace = None
            super().__init__(**kwargs)

        def _bind(self, xml):
            super()._bind(xml)
            self.trace_xml = xml

        def reset(self, seed):
            self._flush_trace()
            obs = super().reset(seed)
            self.trace = SimulationTrace(seed=seed, timestep=self.scene.timestep * self.control_decimation)
            (output / f"seed_{seed}__scene.xml").write_text(self.trace_xml)
            self._capture(None)
            return obs

        def _capture(self, action):
            if vanilla_physics:
                qpos, qvel, state = self.data.qpos, self.data.qvel, self.get_state_vanilla()
            else:
                qpos, qvel, state = self.sim.qpos(), self.sim.d_warp.qvel.numpy()[0], self.get_state()
            self.trace.append(qpos=qpos, qvel=qvel, state=state, action=action)

        def step_one_vanilla(self, action):
            super().step_one_vanilla(action)
            self._capture(action)

        def step_one(self, action):
            super().step_one(action)
            self._capture(action)

        def _flush_trace(self):
            if self.trace is not None:
                self.trace.save(output)
                self.trace = None

        def close(self):
            self._flush_trace()
            super().close()

    return RecordingEnv
