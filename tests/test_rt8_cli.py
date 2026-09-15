from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

import kinesync.cli.rt8_offline_shadow as rt8


class RT8OfflineShadowCLITest(unittest.TestCase):
    def test_main_loads_config_and_executes_named_run(self):
        with (
            mock.patch.object(rt8, "load_config", return_value={"fixture": True}) as load,
            mock.patch.object(
                rt8, "execute_offline_shadow", return_value=Path("fixture-run")
            ) as execute,
            mock.patch.object(
                sys,
                "argv",
                [
                    "kinesync-rt8p",
                    "--config",
                    "fixture.yaml",
                    "--run-id",
                    "fixture-run-id",
                ],
            ),
            mock.patch("builtins.print"),
        ):
            rt8.main()

        load.assert_called_once_with(Path("fixture.yaml"))
        execute.assert_called_once_with({"fixture": True}, run_id="fixture-run-id")


if __name__ == "__main__":
    unittest.main()
