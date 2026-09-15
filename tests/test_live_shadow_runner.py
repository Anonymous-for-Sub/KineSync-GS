from __future__ import annotations

import inspect
import json
import os
import socket
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np

from kinesync.data.take_pens import TakePensDataset
from kinesync.live import FailClosedObservationMonitor, ObservationPacket
from kinesync.live.runner import run_observation_only_shadow
from kinesync.live.safety import audit_live_imports, verify_shadow_audit
from kinesync.live.source import RecordedTakePensSource


def _packet(sequence_id: int, *, arrival_ns: int = 1_000) -> ObservationPacket:
    capture_ns = 100 + sequence_id * 10
    return ObservationPacket(
        session_id="runner-session",
        clock_domain="monotonic_ns",
        sequence_id=sequence_id,
        arrival_monotonic_ns=arrival_ns,
        head_rgb=np.full((4, 6, 3), sequence_id + 1, dtype=np.uint8),
        head_capture_ns=capture_ns,
        auxiliary_rgb=np.full((4, 6, 3), sequence_id + 2, dtype=np.uint8),
        auxiliary_capture_ns=capture_ns + 1,
        auxiliary_role="wrist",
        qpos=np.arange(8, dtype=np.float64) + sequence_id,
        state_capture_ns=capture_ns + 2,
        source_id="runner-fixture",
        frame_id=f"frame-{sequence_id}",
    )


class _FakeSource:
    def __init__(self, packets: list[ObservationPacket], *, exhausted: bool):
        self._packets = list(packets)
        self.exhausted = exhausted
        self.calls = 0

    def next_before(self, deadline_monotonic_ns: int) -> ObservationPacket | None:
        self.calls += 1
        if self._packets:
            return self._packets.pop(0)
        return None


class _FakeEstimator:
    def __init__(self, on_push=None):
        self.packets: list[ObservationPacket] = []
        self._on_push = on_push

    def push(self, packet: ObservationPacket) -> None:
        self.packets.append(packet)
        if self._on_push is not None:
            self._on_push()
        return None


class _Clock:
    def __init__(self, now_ns: int = 1_000):
        self.now_ns = now_ns

    def __call__(self) -> int:
        return self.now_ns


class _SequenceClock:
    def __init__(self, values: list[int]):
        self._values = iter(values)

    def __call__(self) -> int:
        return next(self._values)


class _RaisingSource:
    exhausted = False

    def next_before(self, deadline_monotonic_ns: int) -> ObservationPacket | None:
        raise RuntimeError("read failure")


class _DeadlineCrossingSource(_FakeSource):
    def __init__(self, clock: _Clock):
        super().__init__([_packet(0)], exhausted=True)
        self._clock = clock

    def next_before(self, deadline_monotonic_ns: int) -> ObservationPacket | None:
        packet = super().next_before(deadline_monotonic_ns)
        self._clock.now_ns = deadline_monotonic_ns + 1
        return packet


class _BadEstimate:
    class _Estimate:
        offset_ms = 0.0

        @staticmethod
        def to_dict() -> dict[str, object]:
            return {"unserializable": object()}

    class _Decision:
        correction_ms = 0.0
        accepted = False
        reason = "rejected"

    estimate = _Estimate()
    decision = _Decision()
    window_size = 25


class _BadEstimateEstimator(_FakeEstimator):
    def push(self, packet: ObservationPacket) -> _BadEstimate:
        self.packets.append(packet)
        return _BadEstimate()


class RecordedTakePensSourceTest(unittest.TestCase):
    def _write_fixture(self, root: Path, name: str) -> None:
        frames = 4
        timestamps = np.arange(frames, dtype=np.int64) * 10 + 100
        color = np.empty((frames, 480, 640, 3), dtype=np.uint8)
        for index in range(frames):
            color[index].fill(index + 10)
        with h5py.File(root / name, "w") as handle:
            handle.create_dataset("cam_head/color", data=color)
            handle.create_dataset("cam_head/timestamp", data=timestamps)
            handle.create_dataset("cam_wrist/color", data=color + 1)
            handle.create_dataset("cam_wrist/timestamp", data=timestamps + 1)
            handle.create_dataset("left_arm/action", data=np.zeros((frames, 7), dtype=np.float32))
            handle.create_dataset("left_arm/gripper", data=np.arange(frames, dtype=np.float64))
            handle.create_dataset("left_arm/joint", data=np.arange(frames * 6, dtype=np.float64).reshape(frames, 6))
            handle.create_dataset("left_arm/timestamp", data=timestamps + 2)
            handle.create_dataset("master_left_arm/gripper", data=np.arange(frames, dtype=np.float64))
            handle.create_dataset("master_left_arm/joint", data=np.arange(frames * 6, dtype=np.float64).reshape(frames, 6))
            handle.create_dataset("master_left_arm/timestamp", data=timestamps + 3)

    def test_replays_one_lazy_half_open_segment_with_canonical_packet_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_fixture(root, "0.hdf5")
            self._write_fixture(root, "f1.hdf5")
            trajectory = TakePensDataset(root, ("0.hdf5",), ("f1.hdf5",)).trajectories("test")[0]
            clock = _Clock()
            source = RecordedTakePensSource(
                trajectory,
                start=1,
                stop=4,
                cap=2,
                session_id="replay-session",
                clock_domain="monotonic_ns",
                monotonic_clock_ns=clock,
            )

            original_read_rgb = type(trajectory).read_rgb
            with patch.object(
                type(trajectory), "read_rgb", autospec=True, side_effect=original_read_rgb
            ) as read_rgb:
                first = source.next_before(2_000)
                second = source.next_before(2_000)
                end = source.next_before(2_000)

            assert first is not None
            assert second is not None
            self.assertIsNone(end)
            self.assertTrue(source.exhausted)
            self.assertEqual(first.sequence_id, 0)
            self.assertEqual(second.sequence_id, 1)
            self.assertEqual(first.session_id, second.session_id)
            self.assertEqual(first.clock_domain, second.clock_domain)
            self.assertEqual(first.auxiliary_role, "wrist")
            self.assertEqual(first.frame_id, "f1.hdf5:1")
            np.testing.assert_array_equal(first.qpos, np.array([6, 7, 8, 9, 10, 11, 0.05, -0.05]))
            self.assertEqual(first.arrival_monotonic_ns, 1_000)
            self.assertEqual(second.arrival_monotonic_ns, first.arrival_monotonic_ns)
            self.assertEqual(read_rgb.call_count, 4)
            self.assertEqual(read_rgb.call_args_list[0].args[1:], ("head", [1]))
            self.assertEqual(read_rgb.call_args_list[3].args[1:], ("wrist", [2]))

    def test_never_reads_rgb_when_the_local_deadline_has_already_passed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_fixture(root, "0.hdf5")
            self._write_fixture(root, "f1.hdf5")
            trajectory = TakePensDataset(root, ("0.hdf5",), ("f1.hdf5",)).trajectories("test")[0]
            clock = _Clock(now_ns=2_001)
            source = RecordedTakePensSource(
                trajectory,
                start=0,
                stop=2,
                cap=2,
                session_id="replay-session",
                clock_domain="monotonic_ns",
                monotonic_clock_ns=clock,
            )
            original_read_rgb = type(trajectory).read_rgb
            with patch.object(
                type(trajectory), "read_rgb", autospec=True, side_effect=original_read_rgb
            ) as read_rgb:
                self.assertIsNone(source.next_before(2_000))
            read_rgb.assert_not_called()
            self.assertFalse(source.exhausted)

    def test_advancing_local_clock_records_arrival_after_both_rgb_reads_and_runner_accepts_first_packet(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_fixture(root, "0.hdf5")
            self._write_fixture(root, "f1.hdf5")
            trajectory = TakePensDataset(root, ("0.hdf5",), ("f1.hdf5",)).trajectories("test")[0]
            clock = _SequenceClock([100, 101, 102, 103, 104, 105, 106, 107])
            source = RecordedTakePensSource(
                trajectory,
                start=0,
                stop=1,
                cap=1,
                session_id="replay-session",
                clock_domain="monotonic_ns",
                monotonic_clock_ns=clock,
            )

            summary = run_observation_only_shadow(
                source=source,
                estimator=_FakeEstimator(),
                monitor=FailClosedObservationMonitor(),
                event_path=root / "events.jsonl",
                safety_path=root / "safety.json",
                monotonic_clock_ns=clock,
            )

            self.assertIsNone(summary.terminal_reason)
            event = json.loads((root / "events.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(event["arrival_monotonic_ns"], 103)
            self.assertEqual(event["head_capture_ns"], 100)

    def test_slow_head_or_auxiliary_read_cannot_cross_deadline(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_fixture(root, "0.hdf5")
            self._write_fixture(root, "f1.hdf5")
            trajectory = TakePensDataset(root, ("0.hdf5",), ("f1.hdf5",)).trajectories("test")[0]
            original_read_rgb = type(trajectory).read_rgb

            for delayed_camera, clock_values, expected_calls in (
                ("head", [10, 101], 1),
                ("wrist", [10, 20, 101], 2),
            ):
                with self.subTest(delayed_camera=delayed_camera):
                    source = RecordedTakePensSource(
                        trajectory,
                        start=0,
                        stop=1,
                        cap=1,
                        session_id="replay-session",
                        clock_domain="monotonic_ns",
                        monotonic_clock_ns=_SequenceClock(clock_values),
                    )
                    with patch.object(
                        type(trajectory), "read_rgb", autospec=True, side_effect=original_read_rgb
                    ) as read_rgb:
                        self.assertIsNone(source.next_before(100))
                    self.assertEqual(read_rgb.call_count, expected_calls)
                    self.assertEqual(read_rgb.call_args_list[-1].args[1], delayed_camera)

    def test_source_does_not_repair_a_regressing_arrival_clock_and_monitor_rejects_it(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_fixture(root, "0.hdf5")
            self._write_fixture(root, "f1.hdf5")
            trajectory = TakePensDataset(root, ("0.hdf5",), ("f1.hdf5",)).trajectories("test")[0]
            clock = _SequenceClock([100, 101, 101, 101, 101, 101, 100, 100, 100, 100])
            source = RecordedTakePensSource(
                trajectory,
                start=0,
                stop=2,
                cap=2,
                session_id="replay-session",
                clock_domain="monotonic_ns",
                monotonic_clock_ns=clock,
            )
            summary = run_observation_only_shadow(
                source=source,
                estimator=_FakeEstimator(),
                monitor=FailClosedObservationMonitor(),
                event_path=root / "events.jsonl",
                safety_path=root / "safety.json",
                monotonic_clock_ns=clock,
            )
            self.assertEqual(summary.terminal_reason, "timestamp")


class ObservationOnlyShadowRunnerTest(unittest.TestCase):
    def _run(self, source: _FakeSource, estimator: _FakeEstimator, clock: _Clock):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        return (
            run_observation_only_shadow(
                source=source,
                estimator=estimator,
                monitor=FailClosedObservationMonitor(),
                event_path=root / "events.jsonl",
                safety_path=root / "safety.json",
                monotonic_clock_ns=clock,
            ),
            root,
        )

    def test_clean_close_writes_flushed_contiguous_events_and_zero_counters(self) -> None:
        summary, root = self._run(
            _FakeSource([_packet(0), _packet(1)], exhausted=True), _FakeEstimator(), _Clock()
        )

        self.assertIsNone(summary.terminal_reason)
        self.assertEqual(summary.accepted_packets, 2)
        self.assertEqual(summary.robot_command_requests, 0)
        self.assertEqual(summary.robot_commands_sent, 0)
        events = [json.loads(line) for line in (root / "events.jsonl").read_text().splitlines()]
        self.assertEqual([event["event_index"] for event in events], [0, 1])
        self.assertTrue(all(event["robot_commands_sent"] == 0 for event in events))
        safety = json.loads((root / "safety.json").read_text())
        self.assertTrue(safety["closed"])
        self.assertEqual(safety["event_count"], 2)
        self.assertEqual(safety["accepted_packets"], 2)
        self.assertEqual(safety["session_id"], "runner-session")
        self.assertEqual(safety["clock_domain"], "monotonic_ns")
        self.assertEqual(safety["final_event_sha256"], events[-1]["event_sha256"])
        self.assertRegex(safety["events_file_sha256"], r"^[0-9a-f]{64}$")
        self.assertIsNone(events[0]["previous_event_sha256"])
        self.assertEqual(events[1]["previous_event_sha256"], events[0]["event_sha256"])
        self.assertRegex(events[0]["event_sha256"], r"^[0-9a-f]{64}$")
        receipt = verify_shadow_audit(root / "events.jsonl", root / "safety.json")
        self.assertTrue(receipt.valid)

    def test_source_poll_timeout_is_terminal_and_stops_processing(self) -> None:
        source = _FakeSource([], exhausted=False)
        estimator = _FakeEstimator()
        summary, root = self._run(source, estimator, _Clock())

        self.assertEqual(summary.terminal_reason, "timeout")
        self.assertEqual(source.calls, 1)
        self.assertEqual(estimator.packets, [])
        events = [json.loads(line) for line in (root / "events.jsonl").read_text().splitlines()]
        self.assertEqual(events[-1]["terminal_reason"], "timeout")
        self.assertTrue(events[-1]["terminal"])

    def test_post_estimator_deadline_overrun_is_terminal(self) -> None:
        clock = _Clock()
        estimator = _FakeEstimator(on_push=lambda: setattr(clock, "now_ns", 300_000_001))
        summary, root = self._run(_FakeSource([_packet(0)], exhausted=True), estimator, clock)

        self.assertEqual(summary.terminal_reason, "deadline")
        events = [json.loads(line) for line in (root / "events.jsonl").read_text().splitlines()]
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["terminal_reason"], "deadline")

    def test_refreshes_clock_after_source_before_calling_estimator(self) -> None:
        clock = _Clock()
        source = _DeadlineCrossingSource(clock)
        estimator = _FakeEstimator()
        summary, root = self._run(source, estimator, clock)

        self.assertEqual(summary.terminal_reason, "deadline")
        self.assertEqual(estimator.packets, [])
        event = json.loads((root / "events.jsonl").read_text(encoding="utf-8"))
        self.assertEqual(event["terminal_reason"], "deadline")

    def test_monitor_failure_is_terminal_before_estimation_or_another_poll(self) -> None:
        invalid = _packet(3)
        source = _FakeSource([invalid, _packet(0)], exhausted=True)
        estimator = _FakeEstimator()
        summary, root = self._run(source, estimator, _Clock())

        self.assertEqual(summary.terminal_reason, "sequence")
        self.assertEqual(source.calls, 1)
        self.assertEqual(estimator.packets, [])
        events = [json.loads(line) for line in (root / "events.jsonl").read_text().splitlines()]
        self.assertEqual(events, [events[0]])
        self.assertEqual(events[0]["terminal_reason"], "sequence")

    def test_source_and_event_serialization_exceptions_still_close_with_terminal_receipts(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        for name, source, estimator in (
            ("source", _RaisingSource(), _FakeEstimator()),
            ("serialization", _FakeSource([_packet(0)], exhausted=True), _BadEstimateEstimator()),
        ):
            with self.subTest(name=name):
                event_path = root / f"{name}.jsonl"
                safety_path = root / f"{name}.json"
                summary = run_observation_only_shadow(
                    source=source,
                    estimator=estimator,
                    monitor=FailClosedObservationMonitor(),
                    event_path=event_path,
                    safety_path=safety_path,
                    monotonic_clock_ns=_Clock(),
                )
                expected_reason = "internal" if name == "source" else "serialization"
                self.assertEqual(summary.terminal_reason, expected_reason)
                safety = json.loads(safety_path.read_text(encoding="utf-8"))
                self.assertTrue(safety["closed"])
                self.assertEqual(safety["terminal_reason"], expected_reason)
                self.assertTrue(safety["events_complete"])
                self.assertTrue(verify_shadow_audit(event_path, safety_path).valid)

    def test_verifier_rejects_hash_chain_tampering_even_when_counts_are_coordinated(self) -> None:
        summary, root = self._run(
            _FakeSource([_packet(0), _packet(1)], exhausted=True), _FakeEstimator(), _Clock()
        )
        self.assertIsNone(summary.terminal_reason)
        event_path = root / "events.jsonl"
        safety_path = root / "safety.json"
        events = [json.loads(line) for line in event_path.read_text(encoding="utf-8").splitlines()]
        events[0]["session_id"] = "modified-session"
        event_path.write_text(
            "".join(json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n" for event in events),
            encoding="utf-8",
        )
        with self.assertRaises(ValueError):
            verify_shadow_audit(event_path, safety_path)

    def test_verifier_rejects_receipt_marked_incomplete(self) -> None:
        summary, root = self._run(
            _FakeSource([_packet(0)], exhausted=True), _FakeEstimator(), _Clock()
        )
        self.assertIsNone(summary.terminal_reason)
        safety_path = root / "safety.json"
        safety = json.loads(safety_path.read_text(encoding="utf-8"))
        safety["events_complete"] = False
        safety_path.write_text(json.dumps(safety, sort_keys=True) + "\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            verify_shadow_audit(root / "events.jsonl", safety_path)

    def test_safety_receipt_is_published_without_replacing_an_existing_path(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        with patch.object(Path, "replace", side_effect=AssertionError("replace is not exclusive")):
            summary = run_observation_only_shadow(
                source=_FakeSource([_packet(0)], exhausted=True),
                estimator=_FakeEstimator(),
                monitor=FailClosedObservationMonitor(),
                event_path=root / "events.jsonl",
                safety_path=root / "safety.json",
                monotonic_clock_ns=_Clock(),
            )
        self.assertIsNone(summary.terminal_reason)
        self.assertTrue((root / "safety.json").is_file())

        summary, root = self._run(
            _FakeSource([_packet(0), _packet(1)], exhausted=True), _FakeEstimator(), _Clock()
        )
        self.assertIsNone(summary.terminal_reason)
        event_path = root / "events.jsonl"
        safety_path = root / "safety.json"
        events = [json.loads(line) for line in event_path.read_text(encoding="utf-8").splitlines()][1:]
        safety = json.loads(safety_path.read_text(encoding="utf-8"))
        safety["event_count"] = 1
        safety["accepted_packets"] = 1
        event_path.write_text(
            "".join(
                json.dumps({**event, "event_index": index}, sort_keys=True, separators=(",", ":")) + "\n"
                for index, event in enumerate(events)
            ),
            encoding="utf-8",
        )
        safety_path.write_text(json.dumps(safety, sort_keys=True) + "\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            verify_shadow_audit(event_path, safety_path)

    def test_rejects_existing_event_log_and_invalid_audit_receipts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            events = root / "events.jsonl"
            safety = root / "safety.json"
            events.write_text("already exists\n", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                run_observation_only_shadow(
                    source=_FakeSource([], exhausted=True),
                    estimator=_FakeEstimator(),
                    monitor=FailClosedObservationMonitor(),
                    event_path=events,
                    safety_path=safety,
                    monotonic_clock_ns=_Clock(),
                )

            events.write_text('{"event_index":0,"terminal":false,"robot_command_requests":0,"robot_commands_sent":0}\n', encoding="utf-8")
            with self.assertRaises(FileNotFoundError):
                verify_shadow_audit(events, safety)
            safety.write_text('{"closed":true,"event_count":2,"robot_command_requests":0,"robot_commands_sent":0}\n', encoding="utf-8")
            with self.assertRaises(ValueError):
                verify_shadow_audit(events, safety)
            events.write_text('{"event_index":1,"terminal":false,"robot_command_requests":0,"robot_commands_sent":0}\n', encoding="utf-8")
            safety.write_text('{"closed":true,"event_count":1,"robot_command_requests":0,"robot_commands_sent":0}\n', encoding="utf-8")
            with self.assertRaises(ValueError):
                verify_shadow_audit(events, safety)
            events.write_text('{"event_index":0,"terminal":false,"robot_command_requests":0,"robot_commands_sent":0}', encoding="utf-8")
            with self.assertRaises(ValueError):
                verify_shadow_audit(events, safety)
            events.write_text("not json\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                verify_shadow_audit(events, safety)
            events.write_text('{"event_index":0,"terminal":false,"robot_command_requests":0,"robot_commands_sent":0}\n', encoding="utf-8")
            safety.write_text('{"closed":true,"event_count":1,"robot_command_requests":1,"robot_commands_sent":0}\n', encoding="utf-8")
            with self.assertRaises(ValueError):
                verify_shadow_audit(events, safety)

    def test_runner_has_no_command_capability_or_egress_attempt(self) -> None:
        forbidden_names = {"command", "control", "action"}
        self.assertFalse(forbidden_names & set(inspect.signature(run_observation_only_shadow).parameters))

        def unexpected(*args, **kwargs):
            raise AssertionError("observation-only runner attempted egress")

        with patch.object(subprocess, "run", unexpected), patch.object(os, "system", unexpected), patch.object(socket, "socket", unexpected):
            summary, _ = self._run(
                _FakeSource([_packet(0)], exhausted=True), _FakeEstimator(), _Clock()
            )
        self.assertIsNone(summary.terminal_reason)


class LiveDependencyAuditTest(unittest.TestCase):
    def test_rejects_denied_imports_and_accepts_live_shadow_sources(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        receipt = audit_live_imports(
            [repository / "src/kinesync/live"]
        )
        self.assertTrue(receipt.valid)
        self.assertEqual(receipt.violations, ())

        with tempfile.TemporaryDirectory() as directory:
            forbidden = Path(directory) / "forbidden.py"
            forbidden.write_text("import subprocess\n", encoding="utf-8")
            receipt = audit_live_imports([forbidden])
        self.assertFalse(receipt.valid)
        self.assertEqual(receipt.violations, (f"{forbidden}:subprocess",))

    def test_rejects_transitive_and_dynamic_import_loader_syntax(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / "entry.py"
            sibling = root / "sibling.py"
            entry.write_text("import sibling\n", encoding="utf-8")
            sibling.write_text("import subprocess\n", encoding="utf-8")
            receipt = audit_live_imports([entry])
            self.assertFalse(receipt.valid)
            self.assertIn(f"{sibling}:subprocess", receipt.violations)

            dynamic = root / "dynamic.py"
            dynamic.write_text(
                'import importlib\nimportlib.import_module("socket")\n', encoding="utf-8"
            )
            receipt = audit_live_imports([dynamic])
            self.assertFalse(receipt.valid)
            self.assertIn(f"{dynamic}:dynamic_import_loader", receipt.violations)

            builtin = root / "builtin.py"
            builtin.write_text('__import__("can")\n', encoding="utf-8")
            receipt = audit_live_imports([builtin])
            self.assertFalse(receipt.valid)
            self.assertIn(f"{builtin}:dynamic_import_loader", receipt.violations)

            package = root / "package"
            package.mkdir()
            (package / "__init__.py").write_text("", encoding="utf-8")
            relative_entry = package / "entry.py"
            relative_sibling = package / "sibling.py"
            relative_entry.write_text("from . import sibling\n", encoding="utf-8")
            relative_sibling.write_text("import rospy\n", encoding="utf-8")
            receipt = audit_live_imports([relative_entry])
            self.assertFalse(receipt.valid)
            self.assertIn(f"{relative_sibling}:rospy", receipt.violations)

    def test_rejects_aliased_dynamic_import_loader_syntax(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / "package"
            package.mkdir()
            (package / "__init__.py").write_text("", encoding="utf-8")

            import_from_alias = package / "import_from_alias.py"
            import_from_alias.write_text(
                "from importlib import import_module as load\n"
                "load('.sibling', package=__package__)\n",
                encoding="utf-8",
            )
            receipt = audit_live_imports([import_from_alias])
            self.assertFalse(receipt.valid)
            self.assertIn(f"{import_from_alias}:dynamic_import_loader", receipt.violations)

            builtin_alias = root / "builtin_alias.py"
            builtin_alias.write_text(
                "from builtins import __import__ as load\nload('can')\n",
                encoding="utf-8",
            )
            receipt = audit_live_imports([builtin_alias])
            self.assertFalse(receipt.valid)
            self.assertIn(f"{builtin_alias}:dynamic_import_loader", receipt.violations)

            importlib_assignment = root / "importlib_assignment.py"
            importlib_assignment.write_text(
                "import importlib\nload = importlib.import_module\nload('subprocess')\n",
                encoding="utf-8",
            )
            receipt = audit_live_imports([importlib_assignment])
            self.assertFalse(receipt.valid)
            self.assertIn(f"{importlib_assignment}:dynamic_import_loader", receipt.violations)

            builtin_assignment = root / "builtin_assignment.py"
            builtin_assignment.write_text("load = __import__\nload('rospy')\n", encoding="utf-8")
            receipt = audit_live_imports([builtin_assignment])
            self.assertFalse(receipt.valid)
            self.assertIn(f"{builtin_assignment}:dynamic_import_loader", receipt.violations)

    def test_rejects_any_dynamic_import_loader_syntax(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases = {
                "importlib_import.py": "import importlib\n",
                "importlib_from.py": "from importlib import machinery\n",
                "builtin_from.py": "from builtins import __import__\n",
                "module_alias.py": (
                    "import importlib\nlib = importlib\nlib.import_module('socket')\n"
                ),
                "builtins_attribute.py": "import builtins\nbuiltins.__import__('can')\n",
                "builtins_alias_attribute.py": "import builtins as bi\nbi.__import__('rospy')\n",
                "name_reference.py": "loader = __import__\n",
                "attribute_reference.py": "loader.import_module\n",
                "scope_rebinding.py": (
                    "from importlib import import_module as load\n"
                    "def local():\n"
                    "    load = lambda name: None\n"
                    "    load('subprocess')\n"
                ),
            }
            for filename, contents in cases.items():
                entry = root / filename
                entry.write_text(contents, encoding="utf-8")
                receipt = audit_live_imports([entry])
                self.assertFalse(receipt.valid, filename)
                self.assertIn(f"{entry}:dynamic_import_loader", receipt.violations)

    def test_rejects_direct_reflection_and_execution_primitives(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for primitive in (
                "getattr",
                "setattr",
                "delattr",
                "vars",
                "globals",
                "locals",
                "eval",
                "exec",
                "compile",
            ):
                entry = root / f"direct_{primitive}.py"
                entry.write_text(f"blocked = {primitive}\n", encoding="utf-8")
                receipt = audit_live_imports([entry])
                self.assertFalse(receipt.valid, primitive)
                self.assertIn(f"{entry}:dynamic_import_loader", receipt.violations)

    def test_rejects_getattr_access_to_dynamic_import_loaders(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases = {
                "builtins_getattr.py": (
                    "import builtins\ngetattr(builtins, '__import__')('socket')\n"
                ),
                "dunder_builtins_getattr.py": (
                    "getattr(__builtins__, '__import__')('can')\n"
                ),
            }
            for filename, contents in cases.items():
                entry = root / filename
                entry.write_text(contents, encoding="utf-8")
                receipt = audit_live_imports([entry])
                self.assertFalse(receipt.valid, filename)
                self.assertIn(f"{entry}:dynamic_import_loader", receipt.violations)

    def test_rejects_equivalent_reflective_dynamic_import_loaders(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases = {
                "builtins_getattribute.py": (
                    "import builtins as b\n"
                    "b.__getattribute__('__import__')('subprocess')\n"
                ),
                "object_getattribute.py": (
                    "import builtins as b\n"
                    "object.__getattribute__(b, '__import__')('subprocess')\n"
                ),
                "operator_attrgetter.py": (
                    "import builtins\n"
                    "from operator import attrgetter\n"
                    "attrgetter('__import__')(builtins)('socket')\n"
                ),
            }
            for filename, contents in cases.items():
                entry = root / filename
                entry.write_text(contents, encoding="utf-8")
                receipt = audit_live_imports([entry])
                self.assertFalse(receipt.valid, filename)
                self.assertIn(f"{entry}:dynamic_import_loader", receipt.violations)

    def test_rejects_builtins_namespace_dictionary_access(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases = {
                "builtins_dict.py": (
                    "import builtins\nbuiltins.__dict__['__import__']('rospy')\n"
                ),
                "aliased_builtins_dict.py": (
                    "import builtins as bi\nbi.__dict__['eval']('1 + 1')\n"
                ),
                "dunder_builtins_subscript.py": "__builtins__['__import__']('socket')\n",
            }
            for filename, contents in cases.items():
                entry = root / filename
                entry.write_text(contents, encoding="utf-8")
                receipt = audit_live_imports([entry])
                self.assertFalse(receipt.valid, filename)
                self.assertIn(f"{entry}:dynamic_import_loader", receipt.violations)

    def test_rejects_builtins_mapping_lookup_dynamic_import(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            entry = Path(directory) / "builtins_get.py"
            entry.write_text(
                '__builtins__.get("__import__")("socket")\n', encoding="utf-8"
            )

            receipt = audit_live_imports([entry])

            self.assertFalse(receipt.valid)
            self.assertEqual(
                receipt.violations,
                (f"{entry}:dynamic_import_loader",),
            )

    def test_rejects_indirect_module_builtins_dynamic_imports(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            helper = root / "helper.py"
            helper.write_text("VALUE = 1\n", encoding="utf-8")
            sources = {
                "variable_key.py": (
                    "import helper\n"
                    'key = "__import__"\n'
                    'loaded = helper.__builtins__.get(key)("socket")\n'
                ),
                "operator_getitem.py": (
                    "import helper\n"
                    "from operator import getitem\n"
                    'loaded = getitem(helper.__builtins__, "__import__")("socket")\n'
                ),
            }

            for filename, source in sources.items():
                with self.subTest(filename=filename):
                    entry = root / filename
                    entry.write_text(source, encoding="utf-8")

                    receipt = audit_live_imports([entry])

                    self.assertFalse(receipt.valid)
                    self.assertIn(
                        f"{entry}:dynamic_import_loader",
                        receipt.violations,
                    )

    def test_rejects_relative_imports_by_canonical_package_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / "kinesync"
            live = package / "live"
            visualization = package / "visualization"
            live.mkdir(parents=True)
            visualization.mkdir()
            for init in (
                package / "__init__.py",
                live / "__init__.py",
                visualization / "__init__.py",
            ):
                init.write_text("", encoding="utf-8")
            helper = visualization / "helper.py"
            helper.write_text("VALUE = 1\n", encoding="utf-8")

            package_import = live / "package_import.py"
            package_import.write_text(
                "from .. import visualization\n", encoding="utf-8"
            )
            receipt = audit_live_imports([package_import])
            self.assertFalse(receipt.valid)
            self.assertEqual(
                receipt.violations,
                (f"{package_import}:kinesync.visualization",),
            )
            self.assertIn(str(visualization / "__init__.py"), receipt.paths)

            member_import = live / "member_import.py"
            member_import.write_text(
                "from ..visualization import helper\n", encoding="utf-8"
            )
            receipt = audit_live_imports([member_import])
            self.assertFalse(receipt.valid)
            self.assertEqual(
                receipt.violations,
                (
                    f"{member_import}:kinesync.visualization",
                    f"{member_import}:kinesync.visualization.helper",
                ),
            )
            self.assertIn(str(helper), receipt.paths)

    def test_rejects_aliased_reflection_and_execution_primitives(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases = {
                "import_alias.py": "from builtins import getattr as acquire\n",
                "assignment_alias.py": "run = compile\n",
                "attribute_assignment_alias.py": (
                    "import builtins\nrun = builtins.exec\n"
                ),
            }
            for filename, contents in cases.items():
                entry = root / filename
                entry.write_text(contents, encoding="utf-8")
                receipt = audit_live_imports([entry])
                self.assertFalse(receipt.valid, filename)
                self.assertIn(f"{entry}:dynamic_import_loader", receipt.violations)

    def test_rejects_scope_rebinding_of_reflection_primitives(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases = {
                "local_assignment.py": (
                    "def local():\n"
                    "    getattr = lambda value, name: None\n"
                    "    return getattr(object(), 'value')\n"
                ),
                "parameter_rebinding.py": (
                    "def local(eval):\n"
                    "    return eval('1 + 1')\n"
                ),
            }
            for filename, contents in cases.items():
                entry = root / filename
                entry.write_text(contents, encoding="utf-8")
                receipt = audit_live_imports([entry])
                self.assertFalse(receipt.valid, filename)
                self.assertIn(f"{entry}:dynamic_import_loader", receipt.violations)

    def test_accepts_ordinary_imports_without_dynamic_loader_syntax(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root / "ordinary.py"
            sibling = root / "sibling.py"
            entry.write_text("import sibling\n", encoding="utf-8")
            sibling.write_text("import json\nfrom pathlib import Path\n", encoding="utf-8")
            receipt = audit_live_imports([entry])
            self.assertTrue(receipt.valid)
            self.assertEqual(receipt.violations, ())
            self.assertEqual(set(receipt.paths), {str(entry), str(sibling)})


if __name__ == "__main__":
    unittest.main()
