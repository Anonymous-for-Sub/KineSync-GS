import tempfile
from pathlib import Path
import unittest

import cv2
import numpy as np

from kinesync.live.artifacts import LiveArtifactRecorder
from kinesync.live.schema import ObservationPacket


def _packet(index):
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    image[:, :, 0] = index * 10
    auxiliary = np.zeros_like(image)
    auxiliary[:, :, 1] = index * 10
    return ObservationPacket(
        session_id="artifact-test",
        clock_domain="monotonic_ns",
        sequence_id=index,
        arrival_monotonic_ns=1_000 + index,
        head_rgb=image,
        head_capture_ns=900 + index,
        auxiliary_rgb=auxiliary,
        auxiliary_capture_ns=901 + index,
        auxiliary_role="external",
        qpos=np.full(8, index, dtype=np.float64),
        state_capture_ns=902 + index,
        source_id="fake",
        frame_id=f"frame-{index}",
        head_device_timestamp_ms=10.25 + index,
        auxiliary_device_timestamp_ms=10.75 + index,
        head_frame_number=100 + index,
        auxiliary_frame_number=200 + index,
        state_sdk_joint_timestamp_s=20.25 + index,
        state_sdk_gripper_timestamp_s=20.75 + index,
        state_sdk_feedback_frame_timestamps_s=(
            20.10 + index,
            20.20 + index,
            20.30 + index,
            20.75 + index,
        ),
    )


class LiveHardwareArtifactsTest(unittest.TestCase):
    def test_streams_three_decodable_videos_and_writes_sampled_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recorder = LiveArtifactRecorder(
                root, expected_packets=12, fps=10.0, sample_count=12
            )
            for index in range(12):
                recorder.append(_packet(index))
            outputs = recorder.close()

            self.assertEqual(
                set(outputs),
                {
                    "head_video",
                    "auxiliary_video",
                    "comparison_video",
                    "contact_sheet",
                    "samples",
                },
            )
            self.assertEqual(self._video_shape(outputs["head_video"]), (320, 240, 12))
            self.assertEqual(
                self._video_shape(outputs["auxiliary_video"]), (320, 240, 12)
            )
            self.assertEqual(
                self._video_shape(outputs["comparison_video"]), (1920, 1080, 12)
            )
            contact = cv2.imread(str(outputs["contact_sheet"]), cv2.IMREAD_COLOR)
            self.assertEqual(contact.shape, (1080, 1920, 3))
            with np.load(outputs["samples"]) as payload:
                self.assertEqual(payload["qpos"].shape, (12, 8))
                self.assertEqual(payload["head_rgb"].shape, (12, 240, 320, 3))
                self.assertEqual(payload["auxiliary_rgb"].shape, (12, 240, 320, 3))
                np.testing.assert_array_equal(payload["sequence_id"], np.arange(12))
                self.assertEqual(payload["device_timestamps_ms"].shape, (12, 2))
                self.assertEqual(payload["camera_frame_numbers"].shape, (12, 2))
                self.assertEqual(payload["state_sdk_timestamps_s"].shape, (12, 2))
                self.assertEqual(
                    payload["state_sdk_feedback_frame_timestamps_s"].shape,
                    (12, 4),
                )

    def test_rejects_inconsistent_frame_shapes_and_empty_close(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = LiveArtifactRecorder(Path(directory), expected_packets=2, fps=10.0)
            self.assertEqual(recorder.close(allow_empty=True), {})

        with tempfile.TemporaryDirectory() as directory:
            recorder = LiveArtifactRecorder(Path(directory), expected_packets=2, fps=10.0)
            packet = _packet(0)
            object.__setattr__(packet, "head_device_timestamp_ms", None)
            with self.assertRaisesRegex(ValueError, "head device timestamp"):
                recorder.append(packet)
            self.assertEqual(recorder.close(allow_empty=True), {})
            self.assertFalse((Path(directory) / "head.mp4").exists())

        with tempfile.TemporaryDirectory() as directory:
            recorder = LiveArtifactRecorder(Path(directory), expected_packets=2, fps=10.0)
            recorder.append(_packet(0))
            bad = _packet(1)
            object.__setattr__(bad, "head_rgb", np.zeros((120, 160, 3), dtype=np.uint8))
            with self.assertRaisesRegex(ValueError, "frame shape changed"):
                recorder.append(bad)
            recorder.close()

    @staticmethod
    def _video_shape(path):
        capture = cv2.VideoCapture(str(path))
        try:
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            ok, _ = capture.read()
            if not ok:
                raise AssertionError(f"video is not decodable: {path}")
            return width, height, count
        finally:
            capture.release()


if __name__ == "__main__":
    unittest.main()
