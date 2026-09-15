import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_external_gs_franka.py"
SPEC = importlib.util.spec_from_file_location("external_gs_runner", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


class ExternalGSRunnerTest(unittest.TestCase):
    def test_select_cases_preserves_declared_order_and_rejects_unknown_ids(self) -> None:
        cases = [{"id": "franka_a"}, {"id": "franka_b"}, {"id": "franka_c"}]
        selected = RUNNER._select_cases(cases, ("franka_c", "franka_a"))
        self.assertEqual([item["id"] for item in selected], ["franka_a", "franka_c"])
        with self.assertRaisesRegex(ValueError, "unknown case"):
            RUNNER._select_cases(cases, ("franka_z",))


if __name__ == "__main__":
    unittest.main()
