import unittest
from dataclasses import FrozenInstanceError

import numpy as np

from kinesync.live import (
    FailClosedObservationMonitor,
    LiveShadowPolicy,
    ObservationPacket,
    ObservationRejection,
)


def packet(**overrides: object) -> ObservationPacket:
    values: dict[str, object] = {
        "session_id": "rt10-session",
        "clock_domain": "monotonic_ns",
        "sequence_id": 0,
        "arrival_monotonic_ns": 1_000_000_000,
        "head_rgb": np.full((4, 6, 3), 17, dtype=np.uint8),
        "head_capture_ns": 980_000_000,
        "auxiliary_rgb": np.full((4, 6, 3), 23, dtype=np.uint8),
        "auxiliary_capture_ns": 985_000_000,
        "auxiliary_role": "wrist",
        "qpos": np.arange(8, dtype=np.float64),
        "state_capture_ns": 990_000_000,
        "source_id": "take-pens-f1",
        "frame_id": "frame-0",
    }
    values.update(overrides)
    return ObservationPacket(**values)  # type: ignore[arg-type]


class ObservationPacketTest(unittest.TestCase):
    def test_packet_defensively_copies_read_only_arrays(self) -> None:
        head = np.full((4, 6, 3), 17, dtype=np.uint8)
        auxiliary = np.full((4, 6, 3), 23, dtype=np.uint8)
        qpos = np.arange(8, dtype=np.float64)

        observed = packet(head_rgb=head, auxiliary_rgb=auxiliary, qpos=qpos)
        head[0, 0, 0] = 99
        auxiliary[0, 0, 0] = 99
        qpos[0] = 99.0

        self.assertEqual(int(observed.head_rgb[0, 0, 0]), 17)
        self.assertEqual(int(observed.auxiliary_rgb[0, 0, 0]), 23)
        self.assertEqual(float(observed.qpos[0]), 0.0)
        self.assertFalse(observed.head_rgb.flags.writeable)
        self.assertFalse(observed.auxiliary_rgb.flags.writeable)
        self.assertFalse(observed.qpos.flags.writeable)
        for value in (observed.head_rgb, observed.auxiliary_rgb, observed.qpos):
            with self.subTest(shape=value.shape):
                with self.assertRaises(ValueError):
                    value.flags.writeable = True
        with self.assertRaises(ValueError):
            observed.head_rgb[0, 0, 0] = 1
        with self.assertRaises(FrozenInstanceError):
            observed.sequence_id = 1

    def test_packet_rejects_invalid_schema(self) -> None:
        invalid_packets = (
            {"session_id": ""},
            {"clock_domain": ""},
            {"auxiliary_role": "head"},
            {"sequence_id": True},
            {"arrival_monotonic_ns": -1},
            {"head_capture_ns": 1.5},
            {"head_rgb": np.zeros((4, 6), dtype=np.uint8)},
            {"auxiliary_rgb": np.zeros((4, 6, 3), dtype=np.float32)},
            {"qpos": np.zeros(7, dtype=np.float64)},
            {"qpos": np.array([0.0] * 7 + [np.nan])},
            {"source_id": ""},
            {"frame_id": ""},
        )

        for overrides in invalid_packets:
            with self.subTest(overrides=overrides):
                with self.assertRaisesRegex(ObservationRejection, "schema") as caught:
                    packet(**overrides)
                self.assertEqual(caught.exception.reason, "schema")

    def test_qpos_requires_finite_real_integer_or_floating_shape_eight(self) -> None:
        for dtype in (np.int64, np.float32, np.float64):
            with self.subTest(valid_dtype=np.dtype(dtype).name):
                observed = packet(qpos=np.arange(8, dtype=dtype))
                self.assertEqual(observed.qpos.dtype, np.dtype(dtype))
                self.assertEqual(observed.qpos.shape, (8,))

        invalid = (
            np.zeros(8, dtype=np.bool_),
            np.zeros(8, dtype=np.complex64),
            np.zeros(8, dtype=np.complex128),
            np.array(["0"] * 8, dtype=np.str_),
            np.array([object()] * 8, dtype=object),
        )
        for qpos in invalid:
            with self.subTest(invalid_dtype=qpos.dtype):
                with self.assertRaisesRegex(ObservationRejection, "schema") as caught:
                    packet(qpos=qpos)
                self.assertEqual(caught.exception.reason, "schema")


class LiveShadowPolicyTest(unittest.TestCase):
    def test_defaults_are_the_frozen_limits(self) -> None:
        policy = LiveShadowPolicy()

        self.assertEqual(policy.packet_timeout_ns, 250_000_000)
        self.assertEqual(policy.max_head_auxiliary_skew_ns, 35_000_000)
        self.assertEqual(policy.max_camera_state_skew_ns, 35_000_000)

    def test_rejects_nonpositive_noninteger_and_boolean_limits(self) -> None:
        for name, value in (
            ("packet_timeout_ns", 0),
            ("max_head_auxiliary_skew_ns", -1),
            ("max_camera_state_skew_ns", 1.0),
            ("packet_timeout_ns", True),
        ):
            with self.subTest(name=name, value=value):
                with self.assertRaises(ValueError):
                    LiveShadowPolicy(**{name: value})  # type: ignore[arg-type]


class FailClosedObservationMonitorTest(unittest.TestCase):
    def test_accepts_contiguous_packets_at_exact_skew_limits(self) -> None:
        monitor = FailClosedObservationMonitor()
        for sequence_id in range(3):
            base = 1_000_000_000 + sequence_id * 100_000_000
            monitor.accept(
                packet(
                    sequence_id=sequence_id,
                    arrival_monotonic_ns=base + 70_000_000,
                    head_capture_ns=base,
                    auxiliary_capture_ns=base + 35_000_000,
                    state_capture_ns=base + 35_000_000,
                    frame_id=f"frame-{sequence_id}",
                ),
                now_ns=base + 70_000_000,
                decision_deadline_ns=base + 80_000_000,
            )

        receipt = monitor.receipt
        self.assertEqual(receipt.accepted_packets, 3)
        self.assertIsNone(receipt.terminal_reason)
        self.assertEqual(receipt.max_head_auxiliary_skew_ns, 35_000_000)
        self.assertEqual(receipt.max_camera_state_skew_ns, 35_000_000)
        self.assertEqual(receipt.robot_command_requests, 0)
        self.assertEqual(receipt.robot_commands_sent, 0)
        with self.assertRaises(FrozenInstanceError):
            receipt.accepted_packets = 0

    def test_terminal_rejection_preserves_state_and_blocks_later_packets(self) -> None:
        monitor = FailClosedObservationMonitor()
        first = packet()
        monitor.accept(first, now_ns=1_000_000_000, decision_deadline_ns=1_000_000_001)

        duplicate = packet(
            sequence_id=0,
            arrival_monotonic_ns=1_100_000_000,
            head_capture_ns=1_100_000_000,
            auxiliary_capture_ns=1_105_000_000,
            state_capture_ns=1_106_000_000,
            frame_id="frame-duplicate",
        )
        with self.assertRaisesRegex(ObservationRejection, "sequence") as caught:
            monitor.accept(
                duplicate, now_ns=1_100_000_000, decision_deadline_ns=1_100_000_001
            )
        self.assertEqual(caught.exception.reason, "sequence")
        self.assertEqual(monitor.receipt.accepted_packets, 1)
        self.assertEqual(monitor.receipt.terminal_reason, "sequence")

        later = packet(
            sequence_id=1,
            arrival_monotonic_ns=1_200_000_000,
            head_capture_ns=1_200_000_000,
            auxiliary_capture_ns=1_205_000_000,
            state_capture_ns=1_206_000_000,
            frame_id="frame-later",
        )
        with self.assertRaisesRegex(ObservationRejection, "sequence"):
            monitor.accept(later, now_ns=1_200_000_000, decision_deadline_ns=1_200_000_001)
        self.assertEqual(monitor.receipt.accepted_packets, 1)

    def test_rejects_sequence_session_clock_and_timestamp_discontinuities(self) -> None:
        cases = (
            ("sequence", {"sequence_id": 2}),
            ("session", {"session_id": "other-session"}),
            ("clock_domain", {"clock_domain": "device_clock"}),
            (
                "timestamp",
                {
                    "head_capture_ns": 979_999_999,
                    "auxiliary_capture_ns": 984_999_999,
                    "state_capture_ns": 985_999_999,
                },
            ),
            (
                "timestamp",
                {
                    "head_capture_ns": 989_999_999,
                    "auxiliary_capture_ns": 984_999_999,
                    "state_capture_ns": 990_999_999,
                },
            ),
            (
                "timestamp",
                {
                    "head_capture_ns": 1_019_000_000,
                    "auxiliary_capture_ns": 1_024_000_000,
                    "state_capture_ns": 989_999_999,
                },
            ),
            ("timestamp", {"arrival_monotonic_ns": 999_999_999}),
        )
        for reason, overrides in cases:
            with self.subTest(reason=reason, overrides=overrides):
                monitor = FailClosedObservationMonitor()
                monitor.accept(
                    packet(), now_ns=1_000_000_000, decision_deadline_ns=1_000_000_001
                )
                values: dict[str, object] = {
                    "sequence_id": 1,
                    "arrival_monotonic_ns": 1_100_000_000,
                    "head_capture_ns": 1_100_000_000,
                    "auxiliary_capture_ns": 1_105_000_000,
                    "state_capture_ns": 1_106_000_000,
                    "frame_id": "frame-1",
                }
                values.update(overrides)
                with self.assertRaisesRegex(ObservationRejection, reason) as caught:
                    monitor.accept(
                        packet(**values),
                        now_ns=1_100_000_000,
                        decision_deadline_ns=1_100_000_001,
                    )
                self.assertEqual(caught.exception.reason, reason)
                self.assertEqual(monitor.receipt.accepted_packets, 1)
                self.assertEqual(monitor.receipt.terminal_reason, reason)

    def test_rejects_skew_stale_and_deadline_violations(self) -> None:
        cases = (
            (
                "camera_skew",
                {"auxiliary_capture_ns": 1_015_000_001},
                1_000_000_000,
                1_000_000_001,
            ),
            (
                "state_skew",
                {"state_capture_ns": 1_015_000_001},
                1_000_000_000,
                1_000_000_001,
            ),
            ("stale", {}, 1_250_000_001, 1_250_000_002),
            ("deadline", {}, 1_000_000_002, 1_000_000_001),
        )
        for reason, overrides, now_ns, deadline_ns in cases:
            with self.subTest(reason=reason):
                monitor = FailClosedObservationMonitor()
                with self.assertRaisesRegex(ObservationRejection, reason) as caught:
                    monitor.accept(
                        packet(**overrides),
                        now_ns=now_ns,
                        decision_deadline_ns=deadline_ns,
                    )
                self.assertEqual(caught.exception.reason, reason)
                self.assertEqual(monitor.receipt.accepted_packets, 0)
                self.assertEqual(monitor.receipt.terminal_reason, reason)

    def test_future_arrival_is_terminal_without_changing_the_receipt(self) -> None:
        monitor = FailClosedObservationMonitor()
        monitor.accept(
            packet(), now_ns=1_000_000_000, decision_deadline_ns=1_000_000_001
        )
        before_rejection = monitor.receipt
        future_arrival = packet(
            sequence_id=1,
            arrival_monotonic_ns=1_100_000_001,
            head_capture_ns=1_100_000_000,
            auxiliary_capture_ns=1_105_000_000,
            state_capture_ns=1_106_000_000,
            frame_id="frame-future-arrival",
        )

        with self.assertRaisesRegex(ObservationRejection, "timestamp") as caught:
            monitor.accept(
                future_arrival,
                now_ns=1_100_000_000,
                decision_deadline_ns=1_100_000_002,
            )
        self.assertEqual(caught.exception.reason, "timestamp")
        self.assertEqual(monitor.receipt.accepted_packets, before_rejection.accepted_packets)
        self.assertEqual(
            monitor.receipt.max_head_auxiliary_skew_ns,
            before_rejection.max_head_auxiliary_skew_ns,
        )
        self.assertEqual(
            monitor.receipt.max_camera_state_skew_ns,
            before_rejection.max_camera_state_skew_ns,
        )
        self.assertEqual(monitor.receipt.terminal_reason, "timestamp")

        later = packet(
            sequence_id=1,
            arrival_monotonic_ns=1_200_000_000,
            head_capture_ns=1_200_000_000,
            auxiliary_capture_ns=1_205_000_000,
            state_capture_ns=1_206_000_000,
            frame_id="frame-after-future-arrival",
        )
        with self.assertRaisesRegex(ObservationRejection, "timestamp"):
            monitor.accept(
                later, now_ns=1_200_000_000, decision_deadline_ns=1_200_000_001
            )
        self.assertEqual(monitor.receipt.accepted_packets, before_rejection.accepted_packets)
        self.assertEqual(
            monitor.receipt.max_head_auxiliary_skew_ns,
            before_rejection.max_head_auxiliary_skew_ns,
        )
        self.assertEqual(
            monitor.receipt.max_camera_state_skew_ns,
            before_rejection.max_camera_state_skew_ns,
        )

    def test_each_monitor_rejection_preserves_acceptance_state(self) -> None:
        cases = (
            ("schema", {}, True, 1_100_000_001),
            ("deadline", {}, 1_100_000_002, 1_100_000_001),
            ("stale", {}, 1_350_000_001, 1_350_000_002),
            ("sequence", {"sequence_id": 2}, 1_100_000_000, 1_100_000_001),
            ("session", {"session_id": "other-session"}, 1_100_000_000, 1_100_000_001),
            (
                "clock_domain",
                {"clock_domain": "other-clock"},
                1_100_000_000,
                1_100_000_001,
            ),
            (
                "timestamp",
                {
                    "head_capture_ns": 979_999_999,
                    "auxiliary_capture_ns": 984_999_999,
                    "state_capture_ns": 985_999_999,
                },
                1_100_000_000,
                1_100_000_001,
            ),
            ("timestamp", {"arrival_monotonic_ns": 1_100_000_001}, 1_100_000_000, 1_100_000_002),
            (
                "camera_skew",
                {"auxiliary_capture_ns": 1_135_000_001},
                1_100_000_000,
                1_100_000_001,
            ),
            (
                "state_skew",
                {"state_capture_ns": 1_135_000_001},
                1_100_000_000,
                1_100_000_001,
            ),
        )
        for reason, overrides, now_ns, deadline_ns in cases:
            with self.subTest(reason=reason, overrides=overrides):
                monitor = FailClosedObservationMonitor()
                monitor.accept(
                    packet(), now_ns=1_000_000_000, decision_deadline_ns=1_000_000_001
                )
                before_rejection = monitor.receipt
                values: dict[str, object] = {
                    "sequence_id": 1,
                    "arrival_monotonic_ns": 1_100_000_000,
                    "head_capture_ns": 1_100_000_000,
                    "auxiliary_capture_ns": 1_105_000_000,
                    "state_capture_ns": 1_106_000_000,
                    "frame_id": f"frame-{reason}",
                }
                values.update(overrides)

                with self.assertRaisesRegex(ObservationRejection, reason) as caught:
                    monitor.accept(
                        packet(**values),  # type: ignore[arg-type]
                        now_ns=now_ns,  # type: ignore[arg-type]
                        decision_deadline_ns=deadline_ns,
                    )
                self.assertEqual(caught.exception.reason, reason)
                rejected_receipt = monitor.receipt
                self.assertEqual(rejected_receipt.accepted_packets, before_rejection.accepted_packets)
                self.assertEqual(
                    rejected_receipt.max_head_auxiliary_skew_ns,
                    before_rejection.max_head_auxiliary_skew_ns,
                )
                self.assertEqual(
                    rejected_receipt.max_camera_state_skew_ns,
                    before_rejection.max_camera_state_skew_ns,
                )
                self.assertEqual(
                    rejected_receipt.robot_command_requests,
                    before_rejection.robot_command_requests,
                )
                self.assertEqual(rejected_receipt.robot_commands_sent, before_rejection.robot_commands_sent)
                self.assertEqual(rejected_receipt.terminal_reason, reason)

                with self.assertRaisesRegex(ObservationRejection, reason):
                    monitor.accept(
                        packet(
                            sequence_id=1,
                            arrival_monotonic_ns=1_200_000_000,
                            head_capture_ns=1_200_000_000,
                            auxiliary_capture_ns=1_205_000_000,
                            state_capture_ns=1_206_000_000,
                            frame_id=f"frame-after-{reason}",
                        ),
                        now_ns=1_200_000_000,
                        decision_deadline_ns=1_200_000_001,
                    )
                self.assertEqual(monitor.receipt, rejected_receipt)


if __name__ == "__main__":
    unittest.main()
