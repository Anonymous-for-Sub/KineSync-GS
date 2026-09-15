from pathlib import Path
from types import SimpleNamespace
import unittest

from kinesync.hardware.compatibility import (
    probe_piper_sdk_contract,
    probe_realsense_contract,
)


class _CompatiblePiperInterface:
    def __init__(
        self,
        can_name="can0",
        judge_flag=True,
        can_auto_init=True,
        start_sdk_joint_limit=False,
        start_sdk_gripper_limit=False,
    ):
        pass

    def ConnectPort(self, can_init=False, piper_init=True, start_thread=True):
        pass

    def DisconnectPort(self):
        pass

    def ParseCANFrame(self, message):
        pass

    def GetArmJointMsgs(self):
        pass

    def GetArmGripperMsgs(self):
        pass


class _IncompatiblePiperInterface:
    def __init__(self, can_name="can0"):
        pass

    def ConnectPort(self):
        pass


class HardwareSdkCompatibilityTest(unittest.TestCase):
    def test_accepts_the_piper_sdk_061_api_used_by_the_reader(self):
        module = SimpleNamespace(C_PiperInterface_V2=_CompatiblePiperInterface)

        report = probe_piper_sdk_contract(module=module, version="0.6.1")

        self.assertTrue(report["compatible"])
        self.assertEqual(report["version"], "0.6.1")
        self.assertEqual(report["missing"], [])
        self.assertTrue(report["connect_port_piper_init"])

    def test_rejects_a_piper_sdk_without_read_only_connection_contract(self):
        module = SimpleNamespace(C_PiperInterface_V2=_IncompatiblePiperInterface)

        report = probe_piper_sdk_contract(module=module, version="legacy")

        self.assertFalse(report["compatible"])
        self.assertIn("ConnectPort.piper_init", report["missing"])
        self.assertIn("GetArmJointMsgs", report["missing"])
        self.assertIn("start_sdk_joint_limit", report["missing"])

    def test_accepts_the_pyrealsense2_surface_used_by_the_d455_reader(self):
        module = SimpleNamespace(
            pipeline=object(),
            config=object(),
            context=object(),
            stream=SimpleNamespace(color=object()),
            format=SimpleNamespace(bgr8=object()),
            camera_info=SimpleNamespace(
                serial_number=object(),
                name=object(),
                firmware_version=object(),
                usb_type_descriptor=object(),
            ),
        )

        report = probe_realsense_contract(module=module, version="2.58.4.10922")

        self.assertTrue(report["compatible"])
        self.assertEqual(report["version"], "2.58.4.10922")
        self.assertEqual(report["missing"], [])

    def test_hardware_extra_declares_the_piper_sdk(self):
        root = Path(__file__).parents[1]
        project = (root / "pyproject.toml").read_text(encoding="utf-8")

        self.assertIn('"piper_sdk>=0.6.1,<0.7"', project)


if __name__ == "__main__":
    unittest.main()
