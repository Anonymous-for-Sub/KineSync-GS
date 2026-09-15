import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import kinesync.external_abc as contracts
from kinesync.external_abc import SimulationTrace, make_recording_env, require_gpu_allocation


class ExternalABCContractTests(unittest.TestCase):
    def test_diagnostic_matrix_has_four_cells_with_paired_seeds(self):
        self.assertTrue(hasattr(contracts, "diagnostic_cases"))
        cases = contracts.diagnostic_cases()
        self.assertEqual(len(cases), 12)
        self.assertEqual(len({case["id"] for case in cases}), 12)
        for cell in range(4):
            group = [case for case in cases if case["cell"] == cell]
            self.assertEqual([case["seed"] for case in group], [42000, 42001, 42002])
            self.assertEqual({case["policy_seed"] for case in group}, {123})
        self.assertEqual({(c["physics"], c["chunks"]) for c in cases},
                         {("vanilla", 120), ("vanilla", 240), ("warp", 120), ("warp", 240)})

    def test_eval_options_preserve_defaults_and_select_physics_explicitly(self):
        self.assertTrue(hasattr(contracts, "evaluation_options"))
        options = contracts.evaluation_options(worlds=1, seed=42000, chunks=240,
                                                physics="warp", policy_seed=123)
        self.assertEqual(options, {"num_worlds": 1, "seed": 42000, "num_chunks": 240,
                                   "vanilla_physics": False, "policy_seed": 123})
        self.assertTrue(contracts.evaluation_options(worlds=3, seed=0, chunks=120,
                        physics="vanilla", policy_seed=5)["vanilla_physics"])
        with self.assertRaises(ValueError):
            contracts.evaluation_options(worlds=0, seed=0, chunks=120,
                                         physics="warp", policy_seed=123)
        with self.assertRaises(ValueError):
            contracts.evaluation_options(worlds=1, seed=0, chunks=120,
                                         physics="unknown", policy_seed=123)

    def test_policy_recording_preserves_output_and_copies_actual_noise(self):
        self.assertTrue(hasattr(contracts, "make_recording_policy"))
        class Policy:
            def infer(self, obs, noise=None, **kwargs):
                return noise + obs["state"][None]

        calls = []
        policy = contracts.make_recording_policy(Policy, calls)()
        state = np.arange(14, dtype=np.float32)
        noise = np.ones((50, 14), dtype=np.float32)
        actions = policy.infer({"state": state}, noise=noise)
        np.testing.assert_array_equal(actions, np.arange(14)[None] + np.ones((50, 14)))
        noise[:] = -99
        state[:] = -88
        actions[:] = -77
        np.testing.assert_array_equal(calls[0]["noise"], np.ones((50, 14)))
        np.testing.assert_array_equal(calls[0]["state"], np.arange(14))
        self.assertEqual(calls[0]["actions"][0, 0], 1)

    def test_requires_gpu_allocation(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "Slurm"):
                require_gpu_allocation()
        with patch.dict(os.environ, {"SLURM_JOB_ID": "123"}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "GPU"):
                require_gpu_allocation()
        with patch.dict(os.environ, {"SLURM_JOB_ID": "123", "SLURM_GPUS_ON_NODE": "0"}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "GPU"):
                require_gpu_allocation()
        with patch.dict(os.environ, {"SLURM_JOB_ID": "123", "SLURM_JOB_GPUS": "2"}, clear=True):
            self.assertEqual(require_gpu_allocation(), "123")

    def test_trace_copies_physical_state_and_marks_sim_provenance(self):
        trace = SimulationTrace(seed=41, timestep=0.034)
        qpos = np.arange(20, dtype=float)
        trace.append(qpos=qpos, qvel=np.zeros(19), state=np.arange(14), action=None)
        qpos[:] = -100
        trace.append(qpos=qpos, qvel=np.ones(19), state=np.arange(14), action=np.ones(14))
        with tempfile.TemporaryDirectory() as tmp:
            path = trace.save(Path(tmp))
            with np.load(path, allow_pickle=False) as data:
                self.assertEqual(data["qpos"].shape, (2, 20))
                self.assertEqual(data["qpos"][0, 0], 0)
                np.testing.assert_allclose(data["time_seconds"], [0, 0.034])
                np.testing.assert_array_equal(data["action_valid"], [False, True])
                self.assertEqual(str(data["provenance"]), "simulator_state_from_policy_closed_loop")
                self.assertEqual(str(data["action_alignment"]), "action[i] produces state[i] from state[i-1]; row 0 has no action")
                np.testing.assert_array_equal(data["transition_from_index"], [-1, 0])
                np.testing.assert_array_equal(data["transition_to_index"], [-1, 1])

    def test_nonfinite_or_changing_state_rejected(self):
        trace = SimulationTrace(seed=41, timestep=0.034)
        with self.assertRaises(ValueError):
            trace.append(qpos=np.array([np.nan]), qvel=np.zeros(1), state=np.zeros(14), action=None)
        trace.append(qpos=np.zeros(20), qvel=np.zeros(19), state=np.zeros(14), action=None)
        with self.assertRaises(ValueError):
            trace.append(qpos=np.zeros(21), qvel=np.zeros(19), state=np.zeros(14), action=None)

    def test_refuses_overwriting_trace(self):
        trace = SimulationTrace(seed=1, timestep=0.034)
        trace.append(qpos=np.zeros(20), qvel=np.zeros(19), state=np.zeros(14), action=None)
        with tempfile.TemporaryDirectory() as tmp:
            trace.save(Path(tmp))
            with self.assertRaises(FileExistsError):
                trace.save(Path(tmp))

    def test_two_worlds_both_survive_one_final_close(self):
        class Env:
            def __init__(self):
                self.scene = SimpleNamespace(timestep=0.002)
                self.control_decimation = 17

            def _bind(self, xml):
                self.close()

            def reset(self, seed):
                self._bind("<mujoco/>")
                self.data = SimpleNamespace(qpos=np.zeros(20), qvel=np.zeros(19))
                return {"state": self.get_state_vanilla()}

            def get_state_vanilla(self):
                return self.data.qpos[:14].copy()

            def step_one_vanilla(self, action):
                self.data.qpos[:14] = action

            def close(self):
                pass

        with tempfile.TemporaryDirectory() as tmp:
            env = make_recording_env(Env, Path(tmp), vanilla_physics=True)()
            env.reset(seed=41)
            env.step_one_vanilla(np.ones(14))
            env.reset(seed=42)
            env.step_one_vanilla(np.ones(14) * 2)
            env.close()
            env.close()
            self.assertEqual(len(list(Path(tmp).glob("*__physical_trace.npz"))), 2)
            for seed, expected in ((41, 1), (42, 2)):
                with np.load(Path(tmp) / f"seed_{seed}__physical_trace.npz", allow_pickle=False) as data:
                    self.assertEqual(data["policy_state"].shape, (2, 14))
                    np.testing.assert_array_equal(data["policy_state"][1], np.full(14, expected))


if __name__ == "__main__":
    unittest.main()
