import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import cv2
import numpy as np
import torch

import kinesync.visualization.synchronization as synchronization
from kinesync.guard.schema import GuardDecision
from kinesync.sync.commit import commit_synchronized_state
from kinesync.sync.schema import (
    DetectabilityProfile,
    JointDetectability,
    SynchronizedStateFrame,
    synchronized_state_fingerprint,
)
from kinesync.visualization.synchronization import (
    SynchronizationPresentation,
    write_synchronization_comparison,
    write_synchronization_video,
)


class FakeBackend:
    def __init__(self, height=24, width=32):
        self.height = height
        self.width = width
        self.calls = []
        self.grad_enabled = []

    def render(self, qpos):
        self.grad_enabled.append(torch.is_grad_enabled())
        if isinstance(qpos, torch.Tensor):
            qpos = qpos.detach().cpu().numpy()
        qpos = np.asarray(qpos, dtype=np.float32)
        self.calls.append(qpos.copy())
        outputs = {}
        for camera_index, camera in enumerate(("head", "extra")):
            yy, xx = np.mgrid[: self.height, : self.width]
            center_x = 8.0 + 5.0 * float(qpos[0]) + camera_index * 3.0
            center_y = 10.0 + 4.0 * float(qpos[1])
            mask = ((xx - center_x) ** 2 + (yy - center_y) ** 2) < 28.0
            outputs[camera] = SimpleNamespace(alpha=mask.astype(np.float32))
        return outputs


class GradientProbeBackend:
    def __init__(self, height=24, width=32):
        self.height = height
        self.width = width
        self.grad_enabled = []
        self.output_requires_grad = []

    def render(self, qpos):
        self.grad_enabled.append(torch.is_grad_enabled())
        yy, xx = torch.meshgrid(
            torch.arange(self.height, dtype=qpos.dtype, device=qpos.device),
            torch.arange(self.width, dtype=qpos.dtype, device=qpos.device),
            indexing="ij",
        )
        outputs = {}
        for camera_index, camera in enumerate(("head", "extra")):
            center_x = 8.0 + 5.0 * qpos[0] + camera_index * 3.0
            center_y = 10.0 + 4.0 * qpos[1]
            radius = torch.sqrt(
                (xx - center_x).square() + (yy - center_y).square() + 1e-6
            )
            alpha = torch.sigmoid((5.3 - radius) * 4.0)
            self.output_requires_grad.append(alpha.requires_grad)
            outputs[camera] = SimpleNamespace(alpha=alpha)
        return outputs


def profile():
    return DetectabilityProfile(
        joint_names=("joint1", "joint2"),
        joints=(
            JointDetectability("joint1", True, 0.04, 0.03, 4, 2, 2),
            JointDetectability("joint2", False, None, None, 4, 0, 2),
        ),
        source_sha256="a" * 64,
        joint_order_source_sha256="f" * 64,
        source_split="train",
        calibration_state_ids=("s1", "s2"),
        algorithm_version="test-v1",
        fingerprint="b" * 64,
    )


def make_frame(
    case_id="case_b",
    state_id="state_0044",
    *,
    timestamp_ns=123,
    candidate=(0.08, -0.05),
):
    return commit_synchronized_state(
        profile=profile(),
        guard_decision=GuardDecision(True, "accepted", 0.1),
        measured_qpos=torch.tensor([0.0, 0.0], dtype=torch.float32),
        candidate_qpos=torch.tensor(candidate, dtype=torch.float32),
        selected_joints=["joint1", "joint2"],
        joint_limits={"joint1": (-1.0, 1.0), "joint2": (-1.0, 1.0)},
        timestamp_ns=timestamp_ns,
        state_id=state_id,
        case_id=case_id,
        factor_fingerprint="c" * 64,
        guard_fingerprint="d" * 64,
    )


def frame_with_gradients():
    base = make_frame()
    measured = base.measured_qpos.clone().requires_grad_()
    candidate = base.candidate_qpos.clone().requires_grad_()
    synchronized = base.synchronized_qpos.clone().requires_grad_()
    fingerprint = synchronized_state_fingerprint(
        joint_names=base.joint_names,
        measured_qpos=measured,
        candidate_qpos=candidate,
        synchronized_qpos=synchronized,
        decisions=base.decisions,
        timestamp_ns=base.timestamp_ns,
        state_id=base.state_id,
        case_id=base.case_id,
        factor_fingerprint=base.factor_fingerprint,
        guard_fingerprint=base.guard_fingerprint,
        profile_fingerprint=base.profile_fingerprint,
        guard_reason=base.guard_reason,
    )
    return SynchronizedStateFrame(
        joint_names=base.joint_names,
        measured_qpos=measured,
        candidate_qpos=candidate,
        synchronized_qpos=synchronized,
        decisions=base.decisions,
        timestamp_ns=base.timestamp_ns,
        state_id=base.state_id,
        case_id=base.case_id,
        factor_fingerprint=base.factor_fingerprint,
        guard_fingerprint=base.guard_fingerprint,
        profile_fingerprint=base.profile_fingerprint,
        guard_reason=base.guard_reason,
        fingerprint=fingerprint,
    )


def make_context(frame=None, backend=None):
    height, width = 24, 32
    observations = {}
    for index, camera in enumerate(("head", "extra")):
        rgb = np.zeros((height, width, 3), dtype=np.uint8)
        rgb[..., :] = (40 + index * 20, 70, 110)
        mask = np.zeros((height, width), dtype=bool)
        mask[7 + index : 18 + index, 9:22] = True
        observations[camera] = SimpleNamespace(rgb=rgb, mask=mask)
    return {
        "frame": make_frame() if frame is None else frame,
        "observations": observations,
        "backend": FakeBackend(height, width) if backend is None else backend,
    }


class SynchronizationVisualizationTest(unittest.TestCase):
    def test_fixed_geometry_reserves_nonoverlapping_panels(self):
        self.assertEqual(4 * synchronization._PANEL_WIDTH, 1920)
        self.assertEqual(2 * synchronization._PANEL_HEIGHT, 900)
        self.assertEqual(
            2 * (synchronization._IMAGE_HEIGHT // 2), synchronization._IMAGE_HEIGHT
        )
        self.assertLess(
            synchronization._HEADER_HEIGHT + 2 * synchronization._PANEL_HEIGHT,
            synchronization._FOOTER_Y,
        )

    def test_static_comparison_is_fixed_size_nonblank_and_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.png"
            second = Path(directory) / "second.png"
            write_synchronization_comparison(first, [make_context()])
            write_synchronization_comparison(second, [make_context()])
            image = cv2.imread(str(first), cv2.IMREAD_COLOR)
            self.assertEqual(image.shape, (1080, 1920, 3))
            self.assertGreater(int(image.std()), 4)
            self.assertEqual(first.read_bytes(), second.read_bytes())

    def test_tied_labels_sort_by_unique_frame_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            first_path = Path(directory) / "first.png"
            second_path = Path(directory) / "second.png"
            first_frame = make_frame(timestamp_ns=1, candidate=(0.07, -0.04))
            second_frame = make_frame(timestamp_ns=2, candidate=(0.09, -0.06))
            write_synchronization_comparison(
                first_path, [make_context(second_frame), make_context(first_frame)]
            )
            write_synchronization_comparison(
                second_path, [make_context(first_frame), make_context(second_frame)]
            )
            self.assertEqual(first_path.read_bytes(), second_path.read_bytes())

    def test_rejects_duplicate_frame_fingerprints_before_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "comparison.png"
            duplicate = make_frame()
            contexts = [
                make_context(make_frame(case_id="case_a", timestamp_ns=1)),
                make_context(make_frame(case_id="case_b", timestamp_ns=2)),
                make_context(duplicate),
                make_context(duplicate),
            ]
            with self.assertRaisesRegex(ValueError, "unique frame fingerprints"):
                write_synchronization_comparison(path, contexts)

    def test_video_is_diagnostic_playable_and_uses_independent_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "comparison.mp4"
            context = make_context()
            write_synchronization_video(path, [context], frame_count=7, fps=7)
            capture = cv2.VideoCapture(str(path))
            self.addCleanup(capture.release)
            self.assertTrue(capture.isOpened())
            self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), 1920)
            self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)), 1080)
            self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 7)
            frames = []
            while True:
                ok, image = capture.read()
                if not ok:
                    break
                frames.append(image)
            self.assertEqual(len(frames), 7)
            self.assertGreater(float(np.std(frames[-1])), 3.0)
            self.assertGreater(float(np.std(frames[0][50:94, 35:1885])), 3.0)
            backend = context["backend"]
            np.testing.assert_allclose(backend.calls[-3], [0.0, 0.0])
            np.testing.assert_allclose(backend.calls[-2], [0.08, -0.05])
            np.testing.assert_allclose(backend.calls[-1], [0.08, 0.0])
            self.assertIn(
                "diagnostic linear interpolation, not temporal execution footage",
                write_synchronization_video.__doc__.lower(),
            )

    def test_stacked_video_contexts_fill_both_rows_with_context_semantics(self):
        presentation = SynchronizationPresentation(
            comparison_title="RT7 comparison",
            comparison_subtitle="RT7 comparison subtitle",
            candidate_label="RT2 CANDIDATE",
            committed_label="RT7 COMPONENT-VERIFIED",
            video_title="RT7 component verification",
            video_subtitle="RT7 stacked context diagnostic",
            stack_video_contexts=True,
        )
        contexts = [
            make_context(
                make_frame(case_id="case-a", state_id="state-a", timestamp_ns=1)
            ),
            make_context(
                make_frame(case_id="case-b", state_id="state-b", timestamp_ns=2)
            ),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rt7-stacked.mp4"
            drawn_text = []
            fitted_text = synchronization._fitted_text

            def capture_text(*args, **kwargs):
                drawn_text.append(str(args[1] if len(args) > 1 else kwargs["text"]))
                return fitted_text(*args, **kwargs)

            with mock.patch.object(
                synchronization, "_fitted_text", side_effect=capture_text
            ):
                write_synchronization_video(
                    path,
                    contexts,
                    frame_count=3,
                    fps=3,
                    presentation=presentation,
                )

            capture = cv2.VideoCapture(str(path))
            self.addCleanup(capture.release)
            self.assertTrue(capture.isOpened())
            self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), 1920)
            self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)), 1080)
            frames = []
            while True:
                ok, image = capture.read()
                if not ok:
                    break
                frames.append(image)
            self.assertEqual(len(frames), 3)
            lower_start = synchronization._HEADER_HEIGHT + synchronization._PANEL_HEIGHT
            lower_end = lower_start + synchronization._PANEL_HEIGHT
            for index, image in enumerate(frames):
                with self.subTest(frame_index=index):
                    self.assertGreater(float(np.std(image[lower_start:lower_end])), 10.0)

            context_a = [text for text in drawn_text if text.startswith("A case-a / state-a | ")]
            context_b = [text for text in drawn_text if text.startswith("B case-b / state-b | ")]
            self.assertEqual(len(context_a), 3)
            self.assertEqual(len(context_b), 3)
            for text in (*context_a, *context_b):
                self.assertIn("| guard=", text)
                self.assertIn("| accepted=", text)
                self.assertIn("| rejected=", text)
            self.assertIn("RT7 COMPONENT-VERIFIED", "\n".join(drawn_text))
            self.assertNotIn("RT6", "\n".join(drawn_text))

    def test_explicit_presentation_draws_rt7_component_verified_semantics(self):
        presentation = SynchronizationPresentation(
            comparison_title="KineSync-GS RT7 component-verified state evidence",
            comparison_subtitle="Paired camera evidence for the RT7 component policy.",
            candidate_label="RT2 CANDIDATE",
            committed_label="RT7 COMPONENT-VERIFIED",
            video_title="KineSync-GS RT7 component-verification diagnostic",
            video_subtitle="Diagnostic interpolation for RT7 component verification.",
        )
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "rt7.png"
            video_path = Path(directory) / "rt7.mp4"
            drawn_text = []
            fitted_text = synchronization._fitted_text

            def capture_text(*args, **kwargs):
                drawn_text.append(str(args[1] if len(args) > 1 else kwargs["text"]))
                return fitted_text(*args, **kwargs)

            with mock.patch.object(
                synchronization, "_fitted_text", side_effect=capture_text
            ):
                write_synchronization_comparison(
                    image_path, [make_context()], presentation=presentation
                )
                write_synchronization_video(
                    video_path,
                    [make_context()],
                    frame_count=2,
                    fps=1,
                    presentation=presentation,
                )

            self.assertTrue(image_path.is_file())
            self.assertTrue(video_path.is_file())
            text = "\n".join(drawn_text)
            self.assertIn("RT7 COMPONENT-VERIFIED", text)
            self.assertNotIn("RT6", text)

    def test_rejects_tampered_frame_tensor_before_rendering(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "comparison.png"
            context = make_context()
            context["frame"].synchronized_qpos[0] = 0.5
            with self.assertRaisesRegex(ValueError, "integrity violation"):
                write_synchronization_comparison(path, [context])
            self.assertEqual(context["backend"].calls, [])

    def test_rejects_legacy_fields_so_provenance_cannot_contradict_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "comparison.png"
            context = make_context()
            context.update(
                {
                    "case_id": "contradictory-case",
                    "candidate_qpos": np.array([0.9, 0.9]),
                    "decisions": [],
                    "outcome_labels": {
                        "candidate": "fabricated",
                        "synchronized": "fabricated",
                    },
                }
            )
            with self.assertRaisesRegex(ValueError, "legacy provenance fields"):
                write_synchronization_comparison(path, [context])

    def test_backend_render_runs_without_autograd_graph(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "comparison.png"
            backend = GradientProbeBackend()
            write_synchronization_comparison(
                path, [make_context(frame_with_gradients(), backend)]
            )
            self.assertTrue(backend.grad_enabled)
            self.assertFalse(any(backend.grad_enabled))
            self.assertFalse(any(backend.output_requires_grad))

    def test_rejects_missing_frame_wrong_frame_and_missing_camera(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "comparison.png"
            missing_frame = make_context()
            missing_frame.pop("frame")
            with self.assertRaisesRegex(ValueError, "SynchronizedStateFrame"):
                write_synchronization_comparison(path, [missing_frame])
            wrong_frame = make_context()
            wrong_frame["frame"] = object()
            with self.assertRaisesRegex(ValueError, "SynchronizedStateFrame"):
                write_synchronization_comparison(path, [wrong_frame])
            missing_camera = make_context()
            missing_camera["observations"].pop("extra")
            with self.assertRaisesRegex(ValueError, "two cameras"):
                write_synchronization_comparison(path, [missing_camera])

    def test_rejects_invalid_video_arguments(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "comparison.mp4"
            with self.assertRaises(ValueError):
                write_synchronization_video(path, [make_context()], frame_count=0, fps=7)
            with self.assertRaises(ValueError):
                write_synchronization_video(path, [], frame_count=3, fps=7)


if __name__ == "__main__":
    unittest.main()
