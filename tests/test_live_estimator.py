import unittest
from dataclasses import FrozenInstanceError

import numpy as np

from kinesync.live import ObservationPacket
from kinesync.live.estimator import CausalTemporalShadowEstimator
from kinesync.temporal.telemetry import TemporalGuard


def _packet(
    sequence_id: int,
    *,
    intensity: int,
    auxiliary_intensity: int | None = None,
    qpos0: float,
    head_capture_ns: int | None = None,
    auxiliary_capture_ns: int | None = None,
) -> ObservationPacket:
    timestamp_ns = 1_000_000_000 + sequence_id * 40_000_000
    image = np.full((12, 16, 3), intensity, dtype=np.uint8)
    auxiliary_image = np.full(
        (12, 16, 3),
        intensity if auxiliary_intensity is None else auxiliary_intensity,
        dtype=np.uint8,
    )
    qpos = np.zeros(8, dtype=np.float64)
    qpos[0] = qpos0
    return ObservationPacket(
        session_id="causal-estimator",
        clock_domain="monotonic_ns",
        sequence_id=sequence_id,
        arrival_monotonic_ns=timestamp_ns,
        head_rgb=image,
        head_capture_ns=timestamp_ns if head_capture_ns is None else head_capture_ns,
        auxiliary_rgb=auxiliary_image,
        auxiliary_capture_ns=(
            timestamp_ns if auxiliary_capture_ns is None else auxiliary_capture_ns
        ),
        auxiliary_role="wrist",
        qpos=qpos,
        state_capture_ns=timestamp_ns,
        source_id="unit-test",
        frame_id=f"frame-{sequence_id}",
    )


def _robust_normalize(values: np.ndarray) -> np.ndarray:
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    return np.zeros(values.shape, dtype=np.float64) if mad == 0.0 else (values - median) / mad


class CausalTemporalShadowEstimatorTest(unittest.TestCase):
    def test_aligns_varying_auxiliary_skew_on_head_midpoints_without_future_packets(self):
        guard = TemporalGuard(
            native_delay_ms=40.0,
            min_peak_correlation=-1.0,
            min_peak_margin=0.0,
            min_visual_mad=0.0,
            min_state_mad=0.0,
            min_correction_ms=0.0,
        )
        estimator = CausalTemporalShadowEstimator(guard)
        head_steps = np.array((1, 5, 2, 7, 3, 6, 4, 8) * 4)
        auxiliary_steps = np.array((8, 3, 6, 1, 7, 2, 5, 4) * 4)
        skew_ns = np.array((10, 15, 20, 25, 30) * 6, dtype=np.int64) * 1_000_000
        base_ns = 1_000_000_000
        interval_ns = 40_000_000
        head_intensity = 30
        auxiliary_intensity = 30
        qpos0 = 0.0
        packets = []

        for sequence_id in range(26):
            if sequence_id:
                head_intensity += int(head_steps[sequence_id - 1])
                auxiliary_intensity += int(auxiliary_steps[sequence_id - 1])
                qpos0 += float(head_steps[sequence_id - 1])
            head_capture_ns = base_ns + sequence_id * interval_ns
            packet = _packet(
                sequence_id,
                intensity=head_intensity,
                auxiliary_intensity=auxiliary_intensity,
                qpos0=qpos0,
                head_capture_ns=head_capture_ns,
                auxiliary_capture_ns=head_capture_ns + int(skew_ns[sequence_id]),
            )
            packets.append(packet)
            result = estimator.push(packet)
            if sequence_id < 25:
                self.assertIsNone(result)

        self.assertIsNotNone(result)
        assert result is not None
        head_midpoints = np.array(
            [
                (packets[index].head_capture_ns + packets[index + 1].head_capture_ns)
                // 2
                for index in range(25)
            ],
            dtype=np.int64,
        )
        auxiliary_midpoints = np.array(
            [
                (
                    packets[index].auxiliary_capture_ns
                    + packets[index + 1].auxiliary_capture_ns
                )
                // 2
                for index in range(25)
            ],
            dtype=np.int64,
        )
        in_auxiliary_domain = (head_midpoints >= auxiliary_midpoints[0]) & (
            head_midpoints <= auxiliary_midpoints[-1]
        )
        head_motion = np.abs(np.diff([packet.head_rgb[0, 0, 0] for packet in packets]))
        auxiliary_motion = np.abs(
            np.diff([packet.auxiliary_rgb[0, 0, 0] for packet in packets])
        )
        expected_visual = (
            _robust_normalize(head_motion)[in_auxiliary_domain]
            + np.interp(
                head_midpoints[in_auxiliary_domain],
                auxiliary_midpoints,
                _robust_normalize(auxiliary_motion),
            )
        ) / 2.0

        self.assertFalse(in_auxiliary_domain[0])
        np.testing.assert_array_equal(
            result.camera_midpoint_ns, head_midpoints[in_auxiliary_domain]
        )
        np.testing.assert_allclose(result.visual_motion, expected_visual)
        self.assertEqual(result.camera_midpoint_ns[-1], head_midpoints[-1])

    def test_warms_up_then_uses_only_the_newest_twenty_five_intervals(self):
        guard = TemporalGuard(
            native_delay_ms=40.0,
            min_peak_correlation=-1.0,
            min_peak_margin=0.0,
            min_visual_mad=0.0,
            min_state_mad=0.0,
            min_correction_ms=0.0,
        )
        estimator = CausalTemporalShadowEstimator(guard)
        increments = (1, 5, 2, 7, 3, 6, 4, 8) * 4
        intensity = 30
        qpos0 = 0.0
        packets = []

        first_result = None
        for sequence_id in range(26):
            if sequence_id:
                increment = increments[sequence_id - 1]
                intensity += increment
                qpos0 += float(increment)
            packet = _packet(sequence_id, intensity=intensity, qpos0=qpos0)
            packets.append(packet)
            result = estimator.push(packet)
            if sequence_id < 25:
                self.assertIsNone(result)
            else:
                first_result = result

        self.assertIsNotNone(first_result)
        assert first_result is not None
        self.assertEqual(first_result.window_size, 25)
        self.assertEqual(
            first_result.camera_midpoint_ns[0], packets[1].head_capture_ns - 20_000_000
        )

        result = estimator.push(_packet(26, intensity=intensity + 3, qpos0=qpos0 + 3.0))
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.window_size, 25)
        self.assertEqual(len(result.estimate.score_profile), 21)
        self.assertTrue(
            all(score.correlation is not None for score in result.estimate.score_profile)
        )
        self.assertTrue(result.decision.accepted)
        self.assertEqual(result.decision.reason, "accepted")
        self.assertEqual(result.decision.correction_ms, result.estimate.offset_ms - 40.0)
        self.assertEqual(result.camera_midpoint_ns[0], packets[2].head_capture_ns - 20_000_000)
        self.assertEqual(result.camera_midpoint_ns[-1], 1_000_000_000 + 26 * 40_000_000 - 20_000_000)
        with self.assertRaises(FrozenInstanceError):
            result.window_size = 0

    def test_does_not_mutate_packets_and_rejects_constant_observations(self):
        guard = TemporalGuard(
            native_delay_ms=40.0,
            min_peak_correlation=-1.0,
            min_peak_margin=0.0,
            min_visual_mad=0.0,
            min_state_mad=0.0,
            min_correction_ms=0.0,
        )
        estimator = CausalTemporalShadowEstimator(guard)
        packets = [_packet(index, intensity=42, qpos0=0.0) for index in range(26)]
        before = [
            (packet.head_rgb.tobytes(), packet.auxiliary_rgb.tobytes(), packet.qpos.tobytes())
            for packet in packets
        ]

        result = None
        for packet in packets:
            result = estimator.push(packet)

        self.assertIsNotNone(result)
        assert result is not None
        self.assertFalse(result.decision.accepted)
        self.assertEqual(result.decision.reason, "nonfinite")
        self.assertEqual(result.decision.correction_ms, -40.0)
        self.assertEqual(
            before,
            [
                (packet.head_rgb.tobytes(), packet.auxiliary_rgb.tobytes(), packet.qpos.tobytes())
                for packet in packets
            ],
        )
