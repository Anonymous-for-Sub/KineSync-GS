import unittest

import torch

import kinesync.sync as sync
from kinesync.guard.component_verification import (
    ComponentVerification,
    ComponentVerificationRun,
)
from kinesync.guard.schema import GuardDecision, UpdateEvidence
from kinesync.sync.schema import DetectabilityProfile, JointDetectability


PROVENANCE = {
    "state_id": "state_0042",
    "case_id": "state_0042_j1-pos-2p5deg",
    "factor_fingerprint": "c" * 64,
    "guard_fingerprint": "d" * 64,
}


def make_component_verification(
    joint_name: str, candidate_qpos: torch.Tensor, *, guard_accepted: bool
) -> ComponentVerification:
    return ComponentVerification(
        joint_name=joint_name,
        candidate_qpos=candidate_qpos,
        evidence=UpdateEvidence(2, 0.1, 0.8, 0.8, 0.1, 0.1, True),
        decision=GuardDecision(guard_accepted, "accepted", 0.1),
        before_metrics={"loss": 1.0},
        after_metrics={"loss": 0.5},
        per_view_visual_gain={"head": 0.1, "extra": 0.1},
        fingerprint="d" * 64,
    )


def make_verified_commit_fixture(
    *, supported: bool, guard_accepted: bool, candidate_value: float = 0.05
) -> dict:
    detectability = JointDetectability(
        "joint1",
        supported,
        0.04 if supported else None,
        0.04 if supported else None,
        4,
        2 if supported else 0,
        2,
    )
    profile = DetectabilityProfile(
        joint_names=("joint1",),
        joints=(detectability,),
        source_sha256="a" * 64,
        joint_order_source_sha256="b" * 64,
        source_split="train",
        calibration_state_ids=("state_0001",),
        algorithm_version="test-v1",
        fingerprint="c" * 64,
    )
    verification = make_component_verification(
        "joint1", torch.tensor([candidate_value]), guard_accepted=guard_accepted
    )
    return {
        "profile": profile,
        "verifications": ComponentVerificationRun((verification,)),
        "measured_qpos": torch.tensor([0.0]),
        "candidate_qpos": torch.tensor([candidate_value]),
        "selected_joints": ["joint1"],
        "joint_limits": {"joint1": (-0.1, 0.1)},
        "timestamp_ns": 123,
        **PROVENANCE,
    }


class VerifiedCommitTest(unittest.TestCase):
    def test_verified_commit_accepts_supported_correction_below_rt6_floor(self):
        kwargs = make_verified_commit_fixture(
            supported=True, guard_accepted=True, candidate_value=0.02
        )
        self.assertTrue(hasattr(sync, "commit_component_verified_state"))
        frame = sync.commit_component_verified_state(**kwargs)

        self.assertEqual(frame.accepted_joint_names, ("joint1",))
        self.assertTrue(torch.equal(frame.synchronized_qpos, torch.tensor([0.02])))
        self.assertEqual(frame.decisions[0].signal_floor_rad, 0.04)
        self.assertEqual(frame.guard_reason, "component_verified")

    def test_verified_commit_rolls_back_unsupported_joint_even_when_guard_accepts(self):
        kwargs = make_verified_commit_fixture(supported=False, guard_accepted=True)
        self.assertTrue(hasattr(sync, "commit_component_verified_state"))
        frame = sync.commit_component_verified_state(**kwargs)

        self.assertEqual(frame.accepted_joint_names, ())
        self.assertEqual(frame.decisions[0].reason, "unsupported_joint")

    def test_rejects_verification_with_different_selected_candidate_value(self):
        kwargs = make_verified_commit_fixture(supported=True, guard_accepted=True)
        mismatched_verification = make_component_verification(
            "joint1", torch.tensor([0.02]), guard_accepted=True
        )

        with self.assertRaisesRegex(ValueError, "candidate_qpos"):
            sync.commit_component_verified_state(
                **dict(
                    kwargs,
                    verifications=ComponentVerificationRun((mismatched_verification,)),
                )
            )

    def test_rejects_verification_that_changes_an_unselected_component(self):
        profile = DetectabilityProfile(
            joint_names=("joint1", "joint2"),
            joints=(
                JointDetectability("joint1", True, 0.04, 0.04, 4, 2, 2),
                JointDetectability("joint2", True, 0.04, 0.04, 4, 2, 2),
            ),
            source_sha256="a" * 64,
            joint_order_source_sha256="b" * 64,
            source_split="train",
            calibration_state_ids=("state_0001",),
            algorithm_version="test-v1",
            fingerprint="c" * 64,
        )
        kwargs = dict(
            profile=profile,
            verifications=ComponentVerificationRun(
                (
                    make_component_verification(
                        "joint1", torch.tensor([0.05, 0.03]), guard_accepted=True
                    ),
                )
            ),
            measured_qpos=torch.zeros(2),
            candidate_qpos=torch.tensor([0.05, 0.0]),
            selected_joints=["joint1"],
            joint_limits={"joint1": (-0.1, 0.1), "joint2": (-0.1, 0.1)},
            timestamp_ns=123,
            **PROVENANCE,
        )

        with self.assertRaisesRegex(ValueError, "candidate_qpos"):
            sync.commit_component_verified_state(**kwargs)

    def test_rejects_extra_unselected_component_verification(self):
        kwargs = make_verified_commit_fixture(supported=True, guard_accepted=True)
        extra_verification = make_component_verification(
            "joint2", torch.tensor([0.0]), guard_accepted=True
        )

        with self.assertRaisesRegex(ValueError, "verification joint names"):
            sync.commit_component_verified_state(
                **dict(
                    kwargs,
                    verifications=ComponentVerificationRun(
                        (*kwargs["verifications"].verifications, extra_verification)
                    ),
                )
            )

    def test_rejects_candidate_dtype_or_device_mismatch(self):
        kwargs = make_verified_commit_fixture(supported=True, guard_accepted=True)
        mismatched_candidates = (
            torch.tensor([0.05], dtype=torch.float64),
            torch.empty(1, device="meta"),
        )

        for candidate in mismatched_candidates:
            with self.subTest(device=candidate.device, dtype=candidate.dtype):
                with self.assertRaisesRegex(ValueError, "dtype and device"):
                    sync.commit_component_verified_state(
                        **dict(kwargs, candidate_qpos=candidate)
                    )


if __name__ == "__main__":
    unittest.main()
