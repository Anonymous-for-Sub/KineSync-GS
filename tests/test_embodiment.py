from __future__ import annotations

import unittest

import numpy as np

from kinesync.data.embodiment import (
    ABC_YAM_INTERLEAVED_14,
    FRANKA_7,
    PIPER_6,
    YAM_DUAL_ARM_12,
    Embodiment,
    JointGroup,
    JointSpec,
    selected_joint_mae,
)


class EmbodimentSchemaTest(unittest.TestCase):
    def test_abc_yam_interleaved_preset_matches_the_source_state_order(self):
        arms = ABC_YAM_INTERLEAVED_14.select_groups("left_arm", "right_arm")
        grippers = ABC_YAM_INTERLEAVED_14.select_groups(
            "left_gripper", "right_gripper"
        )

        self.assertEqual(arms.indices, (0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12))
        self.assertEqual(grippers.indices, (6, 13))
        self.assertEqual(grippers.unit, "normalized")

    def test_builtin_arm_selections_have_expected_joint_indices(self):
        self.assertEqual(PIPER_6.select_groups("arm").indices, (0, 1, 2, 3, 4, 5))
        self.assertEqual(FRANKA_7.select_groups("arm").indices, tuple(range(7)))
        self.assertEqual(
            YAM_DUAL_ARM_12.select_groups("left_arm", "right_arm").indices,
            tuple(range(12)),
        )

    def test_yam_grippers_are_separate_from_rotation_metric_groups(self):
        arm_selection = YAM_DUAL_ARM_12.select_groups("left_arm", "right_arm")
        gripper_selection = YAM_DUAL_ARM_12.select_groups("left_gripper")

        self.assertEqual(arm_selection.unit, "rad")
        self.assertEqual(gripper_selection.unit, "normalized")
        self.assertEqual(gripper_selection.indices, (12,))

        predicted = np.zeros((2, 14), dtype=np.float64)
        target = np.zeros((2, 14), dtype=np.float64)
        target[:, 12] = 0.25
        target[:, 0] = 10.0
        self.assertEqual(selected_joint_mae(predicted, target, gripper_selection), 0.25)

    def test_rejects_group_that_mixes_arm_and_gripper_units(self):
        with self.assertRaisesRegex(ValueError, "same unit"):
            Embodiment(
                name="invalid",
                joints=(
                    JointSpec("arm", 0, "rad"),
                    JointSpec("gripper", 1, "normalized"),
                ),
                groups=(JointGroup("mixed", ("arm", "gripper")),),
            )

    def test_rejects_unknown_group_name_and_duplicate_joint_index(self):
        with self.assertRaisesRegex(ValueError, "unknown joint group"):
            PIPER_6.select_groups("missing")
        with self.assertRaisesRegex(ValueError, "unique"):
            Embodiment(
                name="duplicate-index",
                joints=(JointSpec("a", 0, "rad"), JointSpec("b", 0, "rad")),
                groups=(JointGroup("arm", ("a", "b")),),
            )

    def test_rejects_unknown_joint_name_and_unsupported_unit(self):
        with self.assertRaisesRegex(ValueError, "unknown joint names"):
            Embodiment(
                name="unknown-joint",
                joints=(JointSpec("a", 0, "rad"),),
                groups=(JointGroup("arm", ("missing",)),),
            )
        with self.assertRaisesRegex(ValueError, "unsupported joint unit"):
            JointSpec("a", 0, "degrees")


if __name__ == "__main__":
    unittest.main()
