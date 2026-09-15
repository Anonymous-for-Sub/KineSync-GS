import unittest
from dataclasses import replace

import torch

from kinesync.guard.schema import GuardDecision
from kinesync.sync.commit import commit_synchronized_state
from kinesync.sync.schema import DetectabilityProfile, JointDetectability


def profile():
    joints = (
        JointDetectability("joint1", True, 0.04, 0.03, 4, 2, 2),
        JointDetectability("joint2", False, None, None, 4, 0, 2),
    )
    return DetectabilityProfile(
        joint_names=("joint1", "joint2"),
        joints=joints,
        source_sha256="a" * 64,
        joint_order_source_sha256="f" * 64,
        source_split="train",
        calibration_state_ids=("s1", "s2"),
        algorithm_version="test-v1",
        fingerprint="b" * 64,
    )


LIMITS = {"joint1": (-0.1, 0.1), "joint2": (-0.1, 0.1)}


PROVENANCE = {
    "state_id": "state_0042",
    "case_id": "state_0042_j1-pos-2p5deg",
    "factor_fingerprint": "c" * 64,
    "guard_fingerprint": "d" * 64,
}


class SynchronizedStateCommitTest(unittest.TestCase):
    def test_commits_supported_component_and_preserves_unsupported_component(self):
        frame = commit_synchronized_state(
            profile=profile(),
            guard_decision=GuardDecision(True, "accepted", 0.1),
            measured_qpos=torch.tensor([0.0, 0.0], dtype=torch.float64),
            candidate_qpos=torch.tensor([0.04, 0.08], dtype=torch.float64),
            selected_joints=["joint1", "joint2"],
            joint_limits=LIMITS,
            timestamp_ns=123,
            **PROVENANCE,
        )

        torch.testing.assert_close(
            frame.synchronized_qpos,
            torch.tensor([0.04, 0.0], dtype=torch.float64),
        )
        self.assertEqual(
            [(item.joint_name, item.accepted, item.reason) for item in frame.decisions],
            [
                ("joint1", True, "accepted"),
                ("joint2", False, "unsupported_joint"),
            ],
        )
        self.assertEqual(frame.accepted_joint_names, ("joint1",))
        self.assertEqual(frame.joint_names, ("joint1", "joint2"))
        self.assertEqual(frame.state_id, PROVENANCE["state_id"])
        self.assertEqual(frame.case_id, PROVENANCE["case_id"])
        self.assertEqual(frame.factor_fingerprint, PROVENANCE["factor_fingerprint"])
        self.assertEqual(frame.guard_fingerprint, PROVENANCE["guard_fingerprint"])
        self.assertEqual(len(frame.fingerprint), 64)
        with self.assertRaisesRegex(ValueError, "synchronized_qpos"):
            replace(
                frame,
                synchronized_qpos=torch.tensor([0.05, 0.05], dtype=torch.float64),
                fingerprint="e" * 64,
            )
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            replace(frame, fingerprint="e" * 64)

    def test_guard_rejection_has_priority_and_preserves_measured_state(self):
        frame = commit_synchronized_state(
            profile=profile(),
            guard_decision=GuardDecision(False, "visual_gain", -0.01),
            measured_qpos=torch.tensor([0.01, -0.02]),
            candidate_qpos=torch.tensor([0.08, 0.09]),
            selected_joints=["joint1", "joint2"],
            joint_limits=LIMITS,
            timestamp_ns=0,
            **PROVENANCE,
        )
        torch.testing.assert_close(frame.synchronized_qpos, frame.measured_qpos)
        self.assertTrue(all(item.reason == "guard_rejected" for item in frame.decisions))

    def test_rejects_below_floor_nonfinite_and_outside_limit(self):
        cases = [
            (torch.tensor([0.02, 0.0]), "below_signal_floor"),
            (torch.tensor([float("nan"), 0.0]), "nonfinite"),
            (torch.tensor([0.12, 0.0]), "outside_joint_limit"),
        ]
        for candidate, reason in cases:
            with self.subTest(reason=reason):
                frame = commit_synchronized_state(
                    profile=profile(),
                    guard_decision=GuardDecision(True, "accepted", 0.1),
                    measured_qpos=torch.zeros(2),
                    candidate_qpos=candidate,
                    selected_joints=["joint1"],
                    joint_limits=LIMITS,
                    timestamp_ns=1,
                    **PROVENANCE,
                )
                self.assertEqual(frame.decisions[0].reason, reason)
                self.assertFalse(frame.decisions[0].accepted)
                self.assertEqual(float(frame.synchronized_qpos[0]), 0.0)

    def test_validates_shapes_selection_limits_and_timestamp(self):
        kwargs = dict(
            profile=profile(),
            guard_decision=GuardDecision(True, "accepted", 0.1),
            measured_qpos=torch.zeros(2),
            candidate_qpos=torch.ones(2) * 0.04,
            selected_joints=["joint1"],
            joint_limits=LIMITS,
            timestamp_ns=1,
            **PROVENANCE,
        )
        invalid = [
            (dict(candidate_qpos=torch.zeros(3)), "matching"),
            (dict(selected_joints=[]), "nonempty"),
            (dict(selected_joints=["joint3"]), "Unknown selected"),
            (dict(selected_joints=["joint1", "joint1"]), "unique"),
            (dict(joint_limits={"joint1": (-1, 1)}), "joint_limits"),
            (dict(timestamp_ns=-1), "timestamp"),
            (dict(timestamp_ns=1.5), "integer"),
            (dict(state_id=""), "state_id"),
            (dict(case_id=""), "case_id"),
            (dict(factor_fingerprint="bad"), "factor_fingerprint"),
            (dict(guard_fingerprint="bad"), "guard_fingerprint"),
        ]
        for override, message in invalid:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    commit_synchronized_state(**dict(kwargs, **override))

    def test_fingerprint_is_deterministic_and_changes_with_state(self):
        kwargs = dict(
            profile=profile(),
            guard_decision=GuardDecision(True, "accepted", 0.1),
            measured_qpos=torch.zeros(2),
            candidate_qpos=torch.tensor([0.04, 0.0]),
            selected_joints=["joint1"],
            joint_limits=LIMITS,
            timestamp_ns=9,
            **PROVENANCE,
        )
        first = commit_synchronized_state(**kwargs)
        second = commit_synchronized_state(**kwargs)
        changed = commit_synchronized_state(
            **dict(kwargs, candidate_qpos=torch.tensor([0.05, 0.0]))
        )
        self.assertEqual(first.fingerprint, second.fingerprint)
        self.assertNotEqual(first.fingerprint, changed.fingerprint)
        changed_provenance = commit_synchronized_state(
            **dict(kwargs, case_id="state_0042_j1-neg-2p5deg")
        )
        self.assertNotEqual(first.fingerprint, changed_provenance.fingerprint)


if __name__ == "__main__":
    unittest.main()
