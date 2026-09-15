import copy
from collections import Counter
import hashlib
import math
import tempfile
import unittest
from pathlib import Path

import yaml

from kinesync.config import load_config
from kinesync.experiments.rt6_matrix import load_rt6_matrix


PROJECT = Path(__file__).resolve().parents[1]
MATRIX = PROJECT / "configs" / "rt6_piper_fresh_matrix.yaml"


class RT6FreshMatrixTest(unittest.TestCase):
    def _load_payload(self) -> dict:
        return copy.deepcopy(load_config(MATRIX))

    def _write_mutation(self, payload: dict) -> Path:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "matrix.yaml"
        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        return path

    def test_real_matrix_is_fresh_balanced_and_immutable(self) -> None:
        matrix = load_rt6_matrix(MATRIX)
        self.assertEqual(matrix.split, "validation")
        self.assertEqual(matrix.trial_count, 32)
        self.assertEqual(matrix.unique_state_count, 16)
        self.assertEqual(len(matrix.cases), 32)
        self.assertEqual(len({case.case_id for case in matrix.cases}), 32)
        self.assertEqual(len({case.state_id for case in matrix.cases}), 16)
        self.assertEqual(
            Counter(case.state_id for case in matrix.cases),
            Counter({state_id: 2 for state_id in {case.state_id for case in matrix.cases}}),
        )
        self.assertNotIn("state_0360", {case.state_id for case in matrix.cases})
        self.assertTrue(all(case.cameras == ("head", "extra") for case in matrix.cases))
        self.assertTrue(all(case.split == "validation" for case in matrix.cases))
        self.assertEqual(
            matrix.summary.role_counts,
            {
                "single_joint": 24,
                "zero_control": 6,
                "multi_joint_diagnostic": 2,
            },
        )
        self.assertEqual(matrix.summary.single_joint_counts_deg, {2.5: 12, 5.0: 12})
        self.assertEqual(matrix.summary.max_trials_per_state, 2)
        self.assertEqual(len(matrix.fingerprint), 64)
        self.assertEqual(
            matrix.source_file_sha256,
            hashlib.sha256(MATRIX.read_bytes()).hexdigest(),
        )
        previous_ids = self._load_payload()["exclusion_provenance"][
            "previously_used_state_ids"
        ]
        self.assertEqual(len(previous_ids), 130)
        self.assertIn("state_0100", previous_ids)
        self.assertIn("state_0199", previous_ids)
        with self.assertRaises((AttributeError, TypeError)):
            matrix.cases[0].state_id = "state_0000"  # type: ignore[misc]
        with self.assertRaises(TypeError):
            matrix.summary.role_counts["single_joint"] = 0

    def test_exact_signed_coverage_and_degree_radian_consistency(self) -> None:
        matrix = load_rt6_matrix(MATRIX)
        coverage: dict[float, dict[str, set[int]]] = {
            2.5: {f"joint{index}": set() for index in range(1, 7)},
            5.0: {f"joint{index}": set() for index in range(1, 7)},
        }
        for case in matrix.cases:
            for joint, degrees in case.offsets_deg:
                radians = dict(case.offsets_rad)[joint]
                self.assertTrue(math.isclose(radians, math.radians(degrees), abs_tol=1e-12))
                if case.role == "single_joint":
                    coverage[case.magnitude_deg][joint].add(1 if degrees > 0 else -1)
        self.assertEqual(
            coverage,
            {
                2.5: {f"joint{index}": {-1, 1} for index in range(1, 7)},
                5.0: {f"joint{index}": {-1, 1} for index in range(1, 7)},
            },
        )

    def test_zero_controls_select_each_arm_joint_once_at_zero(self) -> None:
        matrix = load_rt6_matrix(MATRIX)
        zeros = [case for case in matrix.cases if case.role == "zero_control"]
        self.assertEqual(len(zeros), 6)
        zero_joints = [dict(case.offsets_deg) for case in zeros]
        self.assertEqual(
            {next(iter(offsets)) for offsets in zero_joints},
            {f"joint{index}" for index in range(1, 7)},
        )
        self.assertTrue(all(len(offsets) == 1 for offsets in zero_joints))
        self.assertTrue(all(next(iter(offsets.values())) == 0.0 for offsets in zero_joints))
        self.assertTrue(
            all(dict(case.offsets_rad) == {next(iter(dict(case.offsets_deg))): 0.0} for case in zeros)
        )

    def test_rejects_prior_state_overlap(self) -> None:
        payload = self._load_payload()
        payload["cases"][0]["state_id"] = "state_0000"
        with self.assertRaisesRegex(ValueError, "previously used"):
            load_rt6_matrix(self._write_mutation(payload))

        for state_id in ("state_0360", "state_0100", "state_0199"):
            payload = self._load_payload()
            payload["cases"][0]["state_id"] = state_id
            with self.subTest(state_id=state_id), self.assertRaisesRegex(
                ValueError, "previously used"
            ):
                load_rt6_matrix(self._write_mutation(payload))

    def test_rejects_missing_signed_group(self) -> None:
        payload = self._load_payload()
        payload["cases"][1]["offsets_deg"] = {"joint1": 2.5}
        payload["cases"][1]["offsets_rad"] = {"joint1": math.radians(2.5)}
        with self.assertRaisesRegex(ValueError, "signed coverage"):
            load_rt6_matrix(self._write_mutation(payload))

    def test_rejects_more_than_two_trials_for_one_state(self) -> None:
        payload = self._load_payload()
        payload["cases"][2]["state_id"] = payload["cases"][0]["state_id"]
        with self.assertRaisesRegex(ValueError, "more than two"):
            load_rt6_matrix(self._write_mutation(payload))

    def test_rejects_wrong_split_camera_and_trial_count(self) -> None:
        for field, value, message in [
            ("split", "train", "split"),
            ("cameras", ["head"], "cameras"),
        ]:
            payload = self._load_payload()
            payload["cases"][0][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, message):
                load_rt6_matrix(self._write_mutation(payload))

        payload = self._load_payload()
        payload["cases"].pop()
        with self.assertRaisesRegex(ValueError, "32 trials"):
            load_rt6_matrix(self._write_mutation(payload))

    def test_rejects_mutated_matrix_contract_header(self) -> None:
        mutations = [
            ("trial_count", 31, "32 trials"),
            ("unique_state_count", 15, "16 unique states"),
            ("max_trials_per_state", 3, "maximum state repetitions"),
            ("cameras", ["extra", "head"], "contract cameras"),
        ]
        for field, value, message in mutations:
            payload = self._load_payload()
            payload["matrix_contract"][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, message):
                load_rt6_matrix(self._write_mutation(payload))

    def test_rejects_duplicate_ids_in_each_provenance_list(self) -> None:
        for path in [
            ("untouched_validation_state_ids",),
            ("exclusion_provenance", "previously_used_state_ids"),
        ]:
            payload = self._load_payload()
            values = payload
            for key in path:
                values = values[key]
            values.append(values[0])
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, "duplicates"):
                load_rt6_matrix(self._write_mutation(payload))

    def test_rejects_duplicate_case_unknown_joint_and_nonfinite_offset(self) -> None:
        payload = self._load_payload()
        payload["cases"][1]["case_id"] = payload["cases"][0]["case_id"]
        with self.assertRaisesRegex(ValueError, "case IDs"):
            load_rt6_matrix(self._write_mutation(payload))

        payload = self._load_payload()
        payload["cases"][0]["offsets_deg"] = {"joint9": 2.5}
        payload["cases"][0]["offsets_rad"] = {"joint9": math.radians(2.5)}
        with self.assertRaisesRegex(ValueError, "unknown joint"):
            load_rt6_matrix(self._write_mutation(payload))

        payload = self._load_payload()
        payload["cases"][0]["offsets_deg"] = {"joint1": float("nan")}
        payload["cases"][0]["offsets_rad"] = {"joint1": float("nan")}
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            load_rt6_matrix(self._write_mutation(payload))

    def test_rejects_empty_or_duplicate_zero_joint_coverage(self) -> None:
        payload = self._load_payload()
        zero_index = next(
            index for index, case in enumerate(payload["cases"])
            if case["role"] == "zero_control"
        )
        payload["cases"][zero_index]["offsets_deg"] = {}
        payload["cases"][zero_index]["offsets_rad"] = {}
        with self.assertRaisesRegex(ValueError, "zero control"):
            load_rt6_matrix(self._write_mutation(payload))

        payload = self._load_payload()
        zero_indices = [
            index for index, case in enumerate(payload["cases"])
            if case["role"] == "zero_control"
        ]
        payload["cases"][zero_indices[1]]["offsets_deg"] = {"joint1": 0.0}
        payload["cases"][zero_indices[1]]["offsets_rad"] = {"joint1": 0.0}
        with self.assertRaisesRegex(ValueError, "zero.*coverage"):
            load_rt6_matrix(self._write_mutation(payload))

    def test_fingerprint_is_deterministic_and_declared_order_is_semantic(self) -> None:
        first = load_rt6_matrix(MATRIX)
        second = load_rt6_matrix(MATRIX)
        self.assertEqual(first.fingerprint, second.fingerprint)

        payload = self._load_payload()
        payload["cases"] = list(reversed(payload["cases"]))
        reordered = load_rt6_matrix(self._write_mutation(payload))
        self.assertNotEqual(first.fingerprint, reordered.fingerprint)

        payload = self._load_payload()
        payload["untouched_validation_state_ids"] = list(
            reversed(payload["untouched_validation_state_ids"])
        )
        reordered_provenance = load_rt6_matrix(self._write_mutation(payload))
        self.assertNotEqual(first.fingerprint, reordered_provenance.fingerprint)


if __name__ == "__main__":
    unittest.main()
