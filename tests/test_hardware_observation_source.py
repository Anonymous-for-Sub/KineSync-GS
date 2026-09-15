import unittest

import numpy as np

from kinesync.hardware import PiperFeedback, RealSenseFrame
from kinesync.live.hardware_source import PiperD455ObservationSource


class _Camera:
    def __init__(self, serial, frames):
        self.serial = serial
        self.frames = list(frames)
        self.started = 0
        self.closed = 0

    def start(self):
        self.started += 1

    def read_before(self, deadline_ns):
        return self.frames.pop(0) if self.frames else None

    def describe(self):
        return {"serial": self.serial, "width": 4, "height": 3, "fps": 30}

    def close(self):
        self.closed += 1


class _Robot:
    def __init__(self, captures):
        self.captures = list(captures)
        self.connected = 0
        self.closed = 0
        self.ready_timeouts = []

    def connect(self):
        self.connected += 1

    def read(self):
        capture_ns = self.captures.pop(0)
        return PiperFeedback(
            qpos=np.arange(8, dtype=np.float64),
            capture_monotonic_ns=capture_ns,
            joint_hz=50.0,
            gripper_hz=50.0,
            sdk_joint_timestamp_s=1.0,
            sdk_gripper_timestamp_s=1.0,
            sdk_feedback_frame_timestamps_s=(0.97, 0.98, 0.99, 1.0),
        )

    def read_before(self, deadline_ns):
        return self.read()

    def wait_until_ready(self, *, timeout_s):
        self.ready_timeouts.append(timeout_s)

    def describe(self):
        return {"can_name": "can0", "read_only": True}

    def close(self):
        self.closed += 1


class _TimedOutRobot(_Robot):
    def read_before(self, deadline_ns):
        return None


def _frame(serial, timestamp_ns, value):
    return RealSenseFrame(
        rgb=np.full((3, 4, 3), value, dtype=np.uint8),
        capture_monotonic_ns=timestamp_ns,
        device_timestamp_ms=timestamp_ns / 1e6,
        frame_number=value,
        serial=serial,
    )


class HardwareObservationSourceTest(unittest.TestCase):
    def test_dual_camera_source_emits_capped_monotonic_packets(self):
        head = _Camera("head-1", [_frame("head-1", 100, 10), _frame("head-1", 200, 11)])
        auxiliary = _Camera(
            "aux-2", [_frame("aux-2", 102, 20), _frame("aux-2", 202, 21)]
        )
        robot = _Robot([101, 201])
        clock = iter((103, 203))
        source = PiperD455ObservationSource(
            robot=robot,
            head_camera=head,
            auxiliary_camera=auxiliary,
            mode="dual",
            cap=2,
            session_id="live-session",
            monotonic_clock_ns=lambda: next(clock),
        )

        source.start()
        first = source.next_before(150)
        second = source.next_before(250)
        done = source.next_before(300)

        self.assertEqual((head.started, auxiliary.started, robot.connected), (1, 1, 1))
        self.assertEqual(robot.ready_timeouts, [5.0])
        self.assertEqual(first.sequence_id, 0)
        self.assertEqual(second.sequence_id, 1)
        self.assertEqual(first.clock_domain, "monotonic_ns")
        self.assertEqual(first.auxiliary_role, "external")
        self.assertEqual(first.head_capture_ns, 100)
        self.assertEqual(first.auxiliary_capture_ns, 102)
        self.assertEqual(first.state_capture_ns, 101)
        self.assertEqual(first.arrival_monotonic_ns, 103)
        self.assertEqual(first.head_device_timestamp_ms, 0.0001)
        self.assertEqual(first.auxiliary_device_timestamp_ms, 0.000102)
        self.assertEqual(first.head_frame_number, 10)
        self.assertEqual(first.auxiliary_frame_number, 20)
        self.assertEqual(first.state_sdk_joint_timestamp_s, 1.0)
        self.assertEqual(first.state_sdk_gripper_timestamp_s, 1.0)
        self.assertEqual(
            first.state_sdk_feedback_frame_timestamps_s,
            (0.97, 0.98, 0.99, 1.0),
        )
        np.testing.assert_array_equal(first.qpos, np.arange(8, dtype=np.float64))
        self.assertIsNone(done)
        self.assertTrue(source.exhausted)
        self.assertTrue(source.describe()["independent_auxiliary_view"])

    def test_dual_mode_requires_distinct_camera_serials(self):
        camera = _Camera("same", [])
        with self.assertRaisesRegex(ValueError, "distinct"):
            PiperD455ObservationSource(
                robot=_Robot([]),
                head_camera=camera,
                auxiliary_camera=_Camera("same", []),
                mode="dual",
                cap=1,
                session_id="s",
            )

    def test_single_preflight_mirrors_one_frame_and_marks_it_nonindependent(self):
        head = _Camera("only", [_frame("only", 100, 33)])
        captured = []
        source = PiperD455ObservationSource(
            robot=_Robot([101]),
            head_camera=head,
            auxiliary_camera=None,
            mode="single_preflight",
            cap=1,
            session_id="single",
            monotonic_clock_ns=lambda: 102,
            packet_sink=captured.append,
        )
        source.start()
        packet = source.next_before(200)
        np.testing.assert_array_equal(packet.head_rgb, packet.auxiliary_rgb)
        self.assertEqual(packet.head_capture_ns, packet.auxiliary_capture_ns)
        self.assertFalse(source.describe()["independent_auxiliary_view"])
        self.assertEqual(captured, [packet])

    def test_returns_none_on_camera_timeout_and_context_closes_all_devices(self):
        head = _Camera("head", [])
        auxiliary = _Camera("aux", [])
        robot = _Robot([])
        with PiperD455ObservationSource(
            robot=robot,
            head_camera=head,
            auxiliary_camera=auxiliary,
            mode="dual",
            cap=1,
            session_id="timeout",
        ) as source:
            self.assertIsNone(source.next_before(10**30))
        self.assertEqual((head.closed, auxiliary.closed, robot.closed), (1, 1, 1))

    def test_returns_none_when_piper_feedback_does_not_advance(self):
        source = PiperD455ObservationSource(
            robot=_TimedOutRobot([]),
            head_camera=_Camera("head", [_frame("head", 100, 1)]),
            auxiliary_camera=None,
            mode="single_preflight",
            cap=1,
            session_id="stale-state",
            monotonic_clock_ns=lambda: 101,
        )
        source.start()
        self.assertIsNone(source.next_before(200))
        self.assertEqual(source.capture_records, ())


if __name__ == "__main__":
    unittest.main()
