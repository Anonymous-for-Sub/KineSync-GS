import math
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import threading
import unittest

import numpy as np

from kinesync.hardware.piper import (
    PiperFeedbackError,
    PiperFeedbackReader,
    _load_interface_factory,
)


class _FakePiperInterface:
    instances = []

    def __init__(self, can_name, **kwargs):
        self.can_name = can_name
        self.kwargs = kwargs
        self.connect_calls = []
        self.disconnect_calls = 0
        self.joint_message = SimpleNamespace(
            time_stamp=10.0,
            Hz=50.0,
            joint_state=SimpleNamespace(
                joint_1=90_000,
                joint_2=-90_000,
                joint_3=45_000,
                joint_4=0,
                joint_5=180_000,
                joint_6=-45_000,
            ),
        )
        self.gripper_message = SimpleNamespace(
            time_stamp=10.0,
            Hz=50.0,
            gripper_state=SimpleNamespace(grippers_angle=35_000),
        )
        self.feedback_frame_timestamps = (10.0, 10.0, 10.0, 10.0)
        type(self).instances.append(self)

    def ConnectPort(self, **kwargs):
        self.connect_calls.append(kwargs)

    def DisconnectPort(self):
        self.disconnect_calls += 1

    def GetArmJointMsgs(self):
        return self.joint_message

    def GetArmGripperMsgs(self):
        return self.gripper_message

    def GetKineSyncFeedbackFrameTimestamps(self):
        return self.feedback_frame_timestamps

    def GetKineSyncSynchronizedFeedback(self):
        joint_message = self.GetArmJointMsgs()
        gripper_message = self.GetArmGripperMsgs()
        state = joint_message.joint_state
        return {
            "joint_timestamp_s": joint_message.time_stamp,
            "joint_hz": joint_message.Hz,
            "joints_millidegree": tuple(
                getattr(state, f"joint_{index}") for index in range(1, 7)
            ),
            "gripper_timestamp_s": gripper_message.time_stamp,
            "gripper_hz": gripper_message.Hz,
            "gripper_angle_micrometre": gripper_message.gripper_state.grippers_angle,
            "feedback_frame_timestamps_s": self.feedback_frame_timestamps,
        }


class _DelayedFeedbackInterface(_FakePiperInterface):
    def __init__(self, can_name, **kwargs):
        super().__init__(can_name, **kwargs)
        self.read_calls = 0

    def GetArmJointMsgs(self):
        self.read_calls += 1
        self.joint_message.Hz = 0.0 if self.read_calls == 1 else 50.0
        return self.joint_message


class _FailingConnectInterface(_FakePiperInterface):
    def ConnectPort(self, **kwargs):
        super().ConnectPort(**kwargs)
        raise ConnectionError("CAN open failed")


class _AdvancingFeedbackInterface(_FakePiperInterface):
    def __init__(self, can_name, **kwargs):
        super().__init__(can_name, **kwargs)
        self.read_calls = 0

    def GetArmJointMsgs(self):
        self.read_calls += 1
        if self.read_calls >= 3:
            self.joint_message.time_stamp = 10.000001
            self.gripper_message.time_stamp = 10.000001
            self.feedback_frame_timestamps = (10.000001,) * 4
        return self.joint_message


class _SdkWorkerInterface(_FakePiperInterface):
    def __init__(self, can_name, **kwargs):
        super().__init__(can_name, **kwargs)
        self._SdkWorkerInterface__read_can_stop_event = threading.Event()
        self._SdkWorkerInterface__can_monitor_stop_event = threading.Event()
        self._SdkWorkerInterface__can_deal_th = SimpleNamespace(
            is_alive=lambda: False, join=lambda timeout=None: None
        )
        self._SdkWorkerInterface__can_monitor_th = SimpleNamespace(
            is_alive=lambda: False, join=lambda timeout=None: None
        )
        self.fps_stopped = 0
        self._SdkWorkerInterface__fps_counter = SimpleNamespace(
            stop=self._stop_fps
        )

    def _stop_fps(self):
        self.fps_stopped += 1


class PiperHardwareTest(unittest.TestCase):
    def setUp(self):
        _FakePiperInterface.instances.clear()

    def test_reader_connects_without_piper_initialization_and_converts_feedback(self):
        clock_values = iter((1_000_000_000, 1_002_000_000))
        wall_values = iter((9_999_000_000, 10_001_000_000))
        reader = PiperFeedbackReader(
            can_name="can7",
            interface_factory=_FakePiperInterface,
            monotonic_clock_ns=lambda: next(clock_values),
            wall_clock_ns=lambda: next(wall_values),
            max_gripper_stroke_mm=70.0,
            finger_travel_m=0.05,
        )

        reader.connect()
        feedback = reader.read()

        interface = _FakePiperInterface.instances[-1]
        self.assertEqual(interface.can_name, "can7")
        self.assertEqual(
            interface.kwargs,
            {
                "judge_flag": True,
                "can_auto_init": True,
                "start_sdk_joint_limit": False,
                "start_sdk_gripper_limit": False,
            },
        )
        self.assertEqual(interface.connect_calls, [{"piper_init": False}])
        np.testing.assert_allclose(
            feedback.qpos,
            [
                math.pi / 2,
                -math.pi / 2,
                math.pi / 4,
                0.0,
                math.pi,
                -math.pi / 4,
                0.025,
                -0.025,
            ],
        )
        self.assertEqual(feedback.capture_monotonic_ns, 1_001_000_000)
        self.assertEqual(feedback.joint_hz, 50.0)
        self.assertEqual(feedback.gripper_hz, 50.0)
        self.assertEqual(feedback.sdk_joint_timestamp_s, 10.0)
        self.assertEqual(feedback.sdk_gripper_timestamp_s, 10.0)
        self.assertEqual(feedback.sdk_feedback_frame_timestamps_s, (10.0,) * 4)
        self.assertFalse(feedback.qpos.flags.writeable)

        reader.close()
        self.assertEqual(interface.disconnect_calls, 1)

    def test_reader_rejects_uninitialized_joint_feedback(self):
        for field, value in (("time_stamp", 0.0), ("Hz", 0.0)):
            with self.subTest(field=field):
                _FakePiperInterface.instances.clear()
                reader = PiperFeedbackReader(
                    can_name="can0",
                    interface_factory=_FakePiperInterface,
                    monotonic_clock_ns=lambda: 100,
                    wall_clock_ns=lambda: 10_000_000_000,
                )
                reader.connect()
                setattr(_FakePiperInterface.instances[-1].joint_message, field, value)

                with self.assertRaisesRegex(
                    PiperFeedbackError, "joint feedback is not initialized"
                ):
                    reader.read()

    def test_reader_requires_connection_and_clamps_gripper_stroke(self):
        reader = PiperFeedbackReader(
            can_name="can0",
            interface_factory=_FakePiperInterface,
            monotonic_clock_ns=lambda: 100,
            wall_clock_ns=lambda: 10_000_000_000,
        )
        with self.assertRaisesRegex(PiperFeedbackError, "not connected"):
            reader.read()

        reader.connect()
        _FakePiperInterface.instances[-1].gripper_message.gripper_state.grippers_angle = 90_000
        feedback = reader.read()
        np.testing.assert_allclose(feedback.qpos[-2:], [0.05, -0.05])

    def test_wait_until_ready_retries_feedback_without_sending_any_request(self):
        clock = iter((100, 101, 102, 103))
        reader = PiperFeedbackReader(
            can_name="can0",
            interface_factory=_DelayedFeedbackInterface,
            monotonic_clock_ns=lambda: next(clock),
            wall_clock_ns=lambda: 10_000_000_000,
        )
        reader.connect()
        feedback = reader.wait_until_ready(
            timeout_s=1.0, poll_interval_s=0.0, sleep=lambda _: None
        )
        self.assertEqual(feedback.joint_hz, 50.0)
        self.assertEqual(_DelayedFeedbackInterface.instances[-1].read_calls, 2)

    def test_failed_connection_closes_the_partially_initialized_sdk(self):
        reader = PiperFeedbackReader(
            can_name="can0", interface_factory=_FailingConnectInterface
        )
        with self.assertRaisesRegex(ConnectionError, "CAN open failed"):
            reader.connect()
        self.assertEqual(_FailingConnectInterface.instances[-1].disconnect_calls, 1)
        with self.assertRaisesRegex(PiperFeedbackError, "not connected"):
            reader.read()

    def test_rejects_cached_feedback_and_waits_for_new_sdk_timestamps(self):
        now = [1_000]

        def clock():
            now[0] += 1_000
            return now[0]

        reader = PiperFeedbackReader(
            can_name="can0",
            interface_factory=_AdvancingFeedbackInterface,
            monotonic_clock_ns=clock,
            wall_clock_ns=lambda: 10_000_000_000,
        )
        reader.connect()
        first = reader.read()
        with self.assertRaisesRegex(PiperFeedbackError, "cached"):
            reader.read()
        second = reader.read_before(20_000, poll_interval_s=0.0, sleep=lambda _: None)
        self.assertIsNotNone(second)
        assert second is not None
        self.assertGreater(second.capture_monotonic_ns, first.capture_monotonic_ns)
        self.assertEqual(second.sdk_joint_timestamp_s, 10.000001)

        interface = _AdvancingFeedbackInterface.instances[-1]
        interface.joint_message.time_stamp = 10.000002
        interface.gripper_message.time_stamp = 10.000001
        interface.feedback_frame_timestamps = (
            10.000002,
            10.000001,
            10.000001,
            10.000001,
        )
        with self.assertRaisesRegex(PiperFeedbackError, "cached"):
            reader.read()

    def test_close_stops_vendor_background_workers(self):
        reader = PiperFeedbackReader(
            can_name="can0", interface_factory=_SdkWorkerInterface
        )
        reader.connect()
        interface = _SdkWorkerInterface.instances[-1]
        reader.close()
        self.assertTrue(interface._SdkWorkerInterface__read_can_stop_event.is_set())
        self.assertTrue(interface._SdkWorkerInterface__can_monitor_stop_event.is_set())
        self.assertEqual(interface.fps_stopped, 1)

    def test_maps_sdk_epoch_timestamp_to_monotonic_instead_of_retimestamping(self):
        reader = PiperFeedbackReader(
            can_name="can0",
            interface_factory=_FakePiperInterface,
            monotonic_clock_ns=lambda: 5_000_000_000,
            wall_clock_ns=lambda: 10_000_000_000,
        )
        reader.connect()
        interface = _FakePiperInterface.instances[-1]
        interface.joint_message.time_stamp = 9.95
        interface.gripper_message.time_stamp = 9.96
        interface.feedback_frame_timestamps = (9.95, 9.951, 9.952, 9.96)
        feedback = reader.read()
        self.assertEqual(feedback.capture_monotonic_ns, 4_950_000_000)

    def test_rejects_when_any_feedback_frame_timestamp_maps_to_the_future(self):
        reader = PiperFeedbackReader(
            can_name="can0",
            interface_factory=_FakePiperInterface,
            monotonic_clock_ns=lambda: 5_000_000_000,
            wall_clock_ns=lambda: 10_000_000_000,
        )
        reader.connect()
        interface = _FakePiperInterface.instances[-1]
        interface.feedback_frame_timestamps = (10.0, 10.0, 10.0, 10.1)
        with self.assertRaisesRegex(PiperFeedbackError, "future"):
            reader.read()

    @unittest.skipUnless(
        importlib.util.find_spec("piper_sdk") and importlib.util.find_spec("can"),
        "hardware SDK dependencies are unavailable",
    )
    def test_real_sdk_subclass_does_not_advance_timestamp_for_filtered_frame(self):
        import can

        interface = _load_interface_factory()(
            "can0",
            can_auto_init=False,
            start_sdk_joint_limit=False,
            start_sdk_gripper_limit=False,
        )
        abnormal = (4_000_000).to_bytes(4, "big", signed=True) * 2
        interface.ParseCANFrame(
            can.Message(
                arbitration_id=0x2A5,
                data=abnormal,
                is_extended_id=False,
                timestamp=1000.0,
            )
        )
        self.assertEqual(
            interface.GetKineSyncSynchronizedFeedback()[
                "feedback_frame_timestamps_s"
            ][0],
            0.0,
        )
        interface.ParseCANFrame(
            can.Message(
                arbitration_id=0x2A5,
                data=bytes(8),
                is_extended_id=False,
                timestamp=1000.1,
            )
        )
        self.assertEqual(
            interface.GetKineSyncSynchronizedFeedback()[
                "feedback_frame_timestamps_s"
            ][0],
            1000.1,
        )

    def test_hardware_modules_do_not_reference_piper_command_apis(self):
        root = Path(__file__).parents[1] / "src" / "kinesync"
        source = "\n".join(
            path.read_text(encoding="utf-8")
            for path in (
                *sorted((root / "hardware").glob("*.py")),
                root / "live" / "hardware_source.py",
                root / "cli" / "rt10_piper_d455_live.py",
            )
            if path.name != "__init__.py"
        )
        forbidden = (
            "MotionCtrl",
            "JointCtrl",
            "GripperCtrl",
            "EnableArm",
            "DisableArm",
            "EmergencyStop",
        )
        self.assertTrue(all(token not in source for token in forbidden))


if __name__ == "__main__":
    unittest.main()
