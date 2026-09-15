import csv
import json
import math
import tempfile
import unittest
from pathlib import Path

from kinesync.sync.calibration import (
    calibration_rows_sha256,
    calibrate_detectability,
    derive_per_joint_recovery,
    migrate_legacy_calibration_rows,
)


JOINTS = [f"joint{index}" for index in range(1, 9)]
ARM_JOINTS = JOINTS[:6]
ORDER_SOURCE_SHA = "6" * 64


def row(
    state_id,
    offsets,
    correction,
    *,
    success,
    reduction,
    split="train",
    finite=True,
    visual_success=True,
):
    before_iou = 0.60
    before_boundary = 0.70
    visual_delta = 0.02 if visual_success else -0.02
    full_correction = list(correction) + [0.0] * (len(JOINTS) - len(correction))
    payload = {
        "state_id": state_id,
        "split": split,
        "offsets_rad": json.dumps(offsets),
        "joint_correction_rad": json.dumps(full_correction),
        "candidate_success": str(bool(success)),
        "state_error_reduction": str(reduction),
        "evidence_finite": str(bool(finite)),
        "mask_iou_before": str(before_iou),
        "mask_iou_candidate": str(before_iou + visual_delta),
        "boundary_f1_before": str(before_boundary),
        "boundary_f1_candidate": str(before_boundary + visual_delta),
    }
    payload["schema_version"] = 2
    payload["joint_order"] = json.dumps(JOINTS)
    payload["joint_order_source_sha256"] = ORDER_SOURCE_SHA
    payload["per_joint_recovery"] = json.dumps(
        derive_per_joint_recovery(payload, joint_names=JOINTS), sort_keys=True
    )
    return payload


class DetectabilityCalibrationTest(unittest.TestCase):
    def test_rejects_legacy_multi_joint_until_deterministically_migrated(self):
        legacy = row(
            "s1",
            {"joint1": 0.04, "joint2": -0.04},
            [-0.035, 0.01],
            success=True,
            reduction=0.7,
        )
        for field in ("schema_version", "joint_order", "per_joint_recovery"):
            legacy.pop(field)
        with self.assertRaisesRegex(ValueError, "schema_version"):
            calibrate_detectability(
                [legacy], joint_names=JOINTS,
                source_sha256=calibration_rows_sha256([legacy]),
                joint_order_source_sha256=ORDER_SOURCE_SHA,
            )

        with tempfile.TemporaryDirectory() as directory:
            factor_path = Path(directory) / "factors.json"
            factor_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "frozen": True,
                        "fit_split": "train",
                        "joint_names": ARM_JOINTS,
                        "joint_zero_rad": [0.0] * 6,
                    }
                ),
                encoding="utf-8",
            )
            migrated = migrate_legacy_calibration_rows(
                [legacy], joint_names=JOINTS, factors_path=factor_path
            )
            self.assertEqual(
                migrated,
                migrate_legacy_calibration_rows(
                    [legacy], joint_names=JOINTS, factors_path=factor_path
                ),
            )
            order_source_sha = migrated[0]["joint_order_source_sha256"]
            csv_path = Path(directory) / "migrated.csv"
            with csv_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(migrated[0]))
                writer.writeheader()
                writer.writerows(migrated)
            with csv_path.open(encoding="utf-8", newline="") as handle:
                round_tripped = list(csv.DictReader(handle))
            self.assertEqual(
                calibration_rows_sha256(migrated),
                calibration_rows_sha256(round_tripped),
            )
            reversed_path = Path(directory) / "reversed.json"
            reversed_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "frozen": True,
                        "fit_split": "train",
                        "joint_names": list(reversed(ARM_JOINTS)),
                        "joint_zero_rad": [0.0] * 6,
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "factor joint order"):
                migrate_legacy_calibration_rows(
                    [legacy], joint_names=JOINTS, factors_path=reversed_path
                )
            bad_schema_path = Path(directory) / "bad-schema.json"
            bad_schema_path.write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "frozen": True,
                        "fit_split": "train",
                        "joint_names": ARM_JOINTS,
                        "joint_zero_rad": [0.0] * 6,
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "schema-v1"):
                migrate_legacy_calibration_rows(
                    [legacy], joint_names=JOINTS, factors_path=bad_schema_path
                )
        profile = calibrate_detectability(
            migrated,
            joint_names=JOINTS,
            source_sha256=calibration_rows_sha256(migrated),
            joint_order_source_sha256=order_source_sha,
        )
        self.assertEqual(profile.algorithm_version, "rt6-detectability-v2")
        self.assertEqual(json.loads(migrated[0]["joint_order"]), list(JOINTS))
        with self.assertRaisesRegex(ValueError, "joint_order"):
            calibrate_detectability(
                [dict(migrated[0], joint_order=json.dumps(list(reversed(JOINTS))))],
                joint_names=JOINTS,
                source_sha256=calibration_rows_sha256(
                    [dict(migrated[0], joint_order=json.dumps(list(reversed(JOINTS))))]
                ),
                joint_order_source_sha256=order_source_sha,
            )
        tampered = [dict(migrated[0], state_id="tampered")]
        with self.assertRaisesRegex(ValueError, "row content"):
            calibrate_detectability(
                tampered,
                joint_names=JOINTS,
                source_sha256=calibration_rows_sha256(migrated),
                joint_order_source_sha256=order_source_sha,
            )

    def test_multi_joint_support_uses_each_joint_recovery(self):
        rows = [
            row(
                "s1",
                {"joint1": math.radians(2.5), "joint2": -math.radians(2.5)},
                [-math.radians(2.25), -math.radians(0.25)],
                success=False,
                reduction=0.5,
            ),
            row(
                "s2",
                {"joint1": -math.radians(2.5), "joint2": math.radians(2.5)},
                [math.radians(2.20), math.radians(0.20)],
                success=False,
                reduction=0.48,
            ),
        ]

        profile = calibrate_detectability(
            rows,
            joint_names=JOINTS,
            source_sha256=calibration_rows_sha256(rows),
            joint_order_source_sha256=ORDER_SOURCE_SHA,
        )

        self.assertTrue(profile.for_joint("joint1").supported)
        self.assertFalse(profile.for_joint("joint2").supported)
        self.assertEqual(profile.algorithm_version, "rt6-detectability-v2")

    def test_rejects_multi_joint_rows_without_visual_evidence(self):
        legacy = row(
            "s1",
            {"joint1": 0.04, "joint2": -0.04},
            [-0.035, 0.01],
            success=True,
            reduction=0.7,
        )
        for field in (
            "mask_iou_before",
            "mask_iou_candidate",
            "boundary_f1_before",
            "boundary_f1_candidate",
        ):
            legacy.pop(field)
        with self.assertRaisesRegex(ValueError, "visual metrics"):
            calibrate_detectability(
                [legacy],
                joint_names=JOINTS,
                source_sha256=calibration_rows_sha256([legacy]),
                joint_order_source_sha256=ORDER_SOURCE_SHA,
            )

    def test_selects_first_fully_successful_group_and_signal_floor(self):
        rows = [
            row("s1", {"joint1": math.radians(1)}, [-0.018, 0], success=True, reduction=0.7),
            row("s2", {"joint1": -math.radians(1)}, [0.015, 0], success=False, reduction=0.2, visual_success=False),
            row("s3", {"joint1": math.radians(2.5)}, [-0.040, 0], success=True, reduction=0.8),
            row("s4", {"joint1": -math.radians(2.5)}, [0.050, 0], success=True, reduction=0.9),
            row("s5", {"joint2": math.radians(5)}, [0, -0.07], success=False, reduction=0.4, visual_success=False),
            row("z1", {"joint1": 0.0, "joint2": 0.0}, [0.010, 0.020], success=False, reduction=0),
            row("z2", {"joint1": 0.0, "joint2": 0.0}, [-0.020, 0.030], success=True, reduction=0),
        ]
        profile = calibrate_detectability(
            rows,
            joint_names=JOINTS,
            source_sha256=calibration_rows_sha256(rows),
            joint_order_source_sha256=ORDER_SOURCE_SHA,
        )

        joint1 = profile.for_joint("joint1")
        self.assertTrue(joint1.supported)
        self.assertAlmostEqual(joint1.minimum_magnitude_rad, math.radians(2.5))
        self.assertAlmostEqual(joint1.minimum_candidate_correction_rad, 0.040)
        self.assertEqual(joint1.controlled_count, 4)
        self.assertEqual(joint1.supporting_count, 2)
        self.assertEqual(joint1.zero_count, 2)

        joint2 = profile.for_joint("joint2")
        self.assertFalse(joint2.supported)
        self.assertIsNone(joint2.minimum_magnitude_rad)
        self.assertIsNone(joint2.minimum_candidate_correction_rad)
        self.assertEqual(profile.source_split, "train")
        self.assertEqual(profile.calibration_state_ids, tuple(sorted({r["state_id"] for r in rows})))
        self.assertEqual(len(profile.fingerprint), 64)

    def test_zero_noise_can_set_the_signal_floor(self):
        rows = [
            row("s1", {"joint1": math.radians(1)}, [-0.016, 0], success=True, reduction=0.7),
            row("z1", {"joint1": 0.0}, [0.025, 0], success=False, reduction=0),
            row("z2", {"joint1": 0.0}, [-0.035, 0], success=False, reduction=0),
        ]
        profile = calibrate_detectability(
            rows,
            joint_names=JOINTS,
            source_sha256=calibration_rows_sha256(rows),
            joint_order_source_sha256=ORDER_SOURCE_SHA,
        )
        self.assertAlmostEqual(
            profile.for_joint("joint1").minimum_candidate_correction_rad,
            0.0345,
        )

    def test_rejects_non_train_nonfinite_unknown_and_duplicate_rows(self):
        base = row(
            "s1",
            {"joint1": math.radians(2.5)},
            [-0.04, 0],
            success=True,
            reduction=0.8,
        )
        invalid = [
            ([dict(base, split="validation")], "split=train"),
            ([dict(base, state_error_reduction="nan")], "finite"),
            ([dict(base, offsets_rad=json.dumps({"joint9": 0.1}))], "Unknown joint"),
            ([base, dict(base)], "Duplicate state_id"),
        ]
        for rows, message in invalid:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    calibrate_detectability(
                        rows,
                        joint_names=JOINTS,
                        source_sha256=calibration_rows_sha256(rows),
                        joint_order_source_sha256=ORDER_SOURCE_SHA,
                    )

    def test_rejects_missing_controlled_examples_and_bad_source_hash(self):
        zero = row(
            "z1",
            {"joint1": 0.0},
            [0.01, 0.0],
            success=False,
            reduction=0,
        )
        with self.assertRaisesRegex(ValueError, "controlled"):
            calibrate_detectability(
                [zero], joint_names=JOINTS,
                source_sha256=calibration_rows_sha256([zero]),
                joint_order_source_sha256=ORDER_SOURCE_SHA,
            )
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            calibrate_detectability(
                [
                    row(
                        "s1",
                        {"joint1": 0.1},
                        [-0.08, 0],
                        success=True,
                        reduction=0.8,
                    )
                ],
                joint_names=JOINTS,
                source_sha256="not-a-hash",
                joint_order_source_sha256=ORDER_SOURCE_SHA,
            )


if __name__ == "__main__":
    unittest.main()
