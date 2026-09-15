from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from kinesync.cli.piper_openvla_audit import main, write_integration_receipt


class PiperOpenVlaAuditCliTest(unittest.TestCase):
    def test_writes_receipt_atomically_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run" / "audit.json"
            receipt = {
                "schema": "kinesync.piper_openvla_integration.v1",
                "valid": True,
            }

            write_integration_receipt(output, receipt)

            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), receipt)
            self.assertFalse((output.parent / "audit.json.tmp").exists())
            with self.assertRaisesRegex(FileExistsError, "audit.json"):
                write_integration_receipt(output, receipt)

    def test_cli_binds_canonical_tasks_checkpoint_and_disabled_command_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            checkpoint = root / "checkpoint"
            data.mkdir()
            checkpoint.mkdir()
            output = root / "audit.json"
            expected = {
                "schema": "kinesync.piper_openvla_integration.v1",
                "valid": True,
                "safety": {"command_mode": "disabled"},
            }
            with patch(
                "kinesync.cli.piper_openvla_audit.build_integration_receipt",
                return_value=expected,
            ) as build:
                status = main(
                    [
                        "--data-root",
                        str(data),
                        "--checkpoint",
                        str(checkpoint),
                        "--output",
                        str(output),
                    ]
                )

            self.assertEqual(status, 0)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), expected)
            _, kwargs = build.call_args
            self.assertEqual(kwargs["expected_episodes_per_task"], 40)
            self.assertEqual(
                tuple(kwargs["tasks"]),
                ("corn_to_plate", "red_cube_to_plate", "red_pen_to_bucket"),
            )

    def test_pyproject_exposes_read_only_audit_command(self):
        pyproject = (Path(__file__).parents[1] / "pyproject.toml").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            'kinesync-piper-openvla-audit = "kinesync.cli.piper_openvla_audit:main"',
            pyproject,
        )


if __name__ == "__main__":
    unittest.main()
