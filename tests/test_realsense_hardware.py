from types import SimpleNamespace
import unittest

import numpy as np

from kinesync.hardware.realsense import (
    RealSenseColorReader,
    discover_realsense_devices,
)


class _FakeDevice:
    def __init__(self, serial="D455-001", name="Intel RealSense D455"):
        self.values = {
            "serial_number": serial,
            "name": name,
            "firmware_version": "5.16.0.1",
            "usb_type_descriptor": "3.2",
        }

    def supports(self, key):
        return key in self.values

    def get_info(self, key):
        return self.values[key]


class _FakeColorFrame:
    def __init__(self, image, *, timestamp_ms=12.5, frame_number=7):
        self.image = image
        self.timestamp_ms = timestamp_ms
        self.frame_number = frame_number

    def get_data(self):
        return self.image

    def get_timestamp(self):
        return self.timestamp_ms

    def get_frame_number(self):
        return self.frame_number


class _FakeFrames:
    def __init__(self, frame):
        self.frame = frame

    def get_color_frame(self):
        return self.frame


class _FakeConfig:
    def __init__(self):
        self.device_serial = None
        self.stream_args = None

    def enable_device(self, serial):
        self.device_serial = serial

    def enable_stream(self, *args):
        self.stream_args = args


class _FakePipeline:
    def __init__(self, owner):
        self.owner = owner
        self.config = None
        self.stopped = False

    def start(self, config):
        self.config = config
        return SimpleNamespace(get_device=lambda: self.owner.devices[0])

    def wait_for_frames(self, timeout_ms):
        self.owner.timeout_requests.append(timeout_ms)
        if self.owner.wait_error is not None:
            raise RuntimeError(self.owner.wait_error)
        return _FakeFrames(self.owner.frame)

    def stop(self):
        self.stopped = True


class _FakeRealSense:
    stream = SimpleNamespace(color="color")
    format = SimpleNamespace(bgr8="bgr8")
    camera_info = SimpleNamespace(
        serial_number="serial_number",
        name="name",
        firmware_version="firmware_version",
        usb_type_descriptor="usb_type_descriptor",
    )

    def __init__(self):
        self.devices = [_FakeDevice()]
        self.frame = _FakeColorFrame(
            np.asarray([[[1, 2, 3], [10, 20, 30]]], dtype=np.uint8)
        )
        self.wait_error = None
        self.timeout_requests = []
        self.pipelines = []
        self.configs = []

    def pipeline(self):
        pipeline = _FakePipeline(self)
        self.pipelines.append(pipeline)
        return pipeline

    def config(self):
        config = _FakeConfig()
        self.configs.append(config)
        return config

    def context(self):
        return SimpleNamespace(query_devices=lambda: self.devices)


class RealSenseHardwareTest(unittest.TestCase):
    def test_selects_serial_configures_color_and_converts_bgr_to_rgb(self):
        sdk = _FakeRealSense()
        clock = iter((1_000_000_000, 1_005_000_000))
        reader = RealSenseColorReader(
            serial="D455-001",
            width=1280,
            height=720,
            fps=30,
            rs_module=sdk,
            monotonic_clock_ns=lambda: next(clock),
        )

        reader.start()
        frame = reader.read_before(1_100_000_000)

        self.assertIsNotNone(frame)
        assert frame is not None
        self.assertEqual(sdk.configs[0].device_serial, "D455-001")
        self.assertEqual(
            sdk.configs[0].stream_args, ("color", 1280, 720, "bgr8", 30)
        )
        np.testing.assert_array_equal(
            frame.rgb, np.asarray([[[3, 2, 1], [30, 20, 10]]], dtype=np.uint8)
        )
        self.assertFalse(frame.rgb.flags.writeable)
        self.assertEqual(frame.capture_monotonic_ns, 1_005_000_000)
        self.assertEqual(frame.device_timestamp_ms, 12.5)
        self.assertEqual(frame.frame_number, 7)
        self.assertEqual(frame.serial, "D455-001")
        self.assertEqual(sdk.timeout_requests, [100])

        reader.close()
        self.assertTrue(sdk.pipelines[0].stopped)

    def test_returns_none_when_deadline_has_already_passed_or_frame_times_out(self):
        sdk = _FakeRealSense()
        reader = RealSenseColorReader(
            serial="D455-001", rs_module=sdk, monotonic_clock_ns=lambda: 200
        )
        reader.start()
        self.assertIsNone(reader.read_before(199))

        sdk.wait_error = "Frame didn't arrive within 100"
        self.assertIsNone(reader.read_before(1_000_000_200))

    def test_discovery_reports_supported_device_identity_fields(self):
        sdk = _FakeRealSense()
        devices = discover_realsense_devices(rs_module=sdk)
        self.assertEqual(
            devices,
            [
                {
                    "serial": "D455-001",
                    "name": "Intel RealSense D455",
                    "firmware_version": "5.16.0.1",
                    "usb_type": "3.2",
                }
            ],
        )

    def test_rejects_a_serial_that_is_not_connected(self):
        sdk = _FakeRealSense()
        with self.assertRaisesRegex(ValueError, "not connected"):
            RealSenseColorReader(serial="missing", rs_module=sdk).start()

    def test_rejects_matching_serial_when_device_is_not_d455(self):
        sdk = _FakeRealSense()
        sdk.devices = [_FakeDevice(name="Intel RealSense D435")]
        with self.assertRaisesRegex(ValueError, "D455"):
            RealSenseColorReader(serial="D455-001", rs_module=sdk).start()


if __name__ == "__main__":
    unittest.main()
