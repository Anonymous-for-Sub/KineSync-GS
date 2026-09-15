from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from kinesync.cli.rt9_temporal_bridge import _TemporalVisualSelection, _visual_context
from kinesync.data.take_pens import ContinuousSegment, TakePensTrajectory
from kinesync.temporal.telemetry import (
    CorruptedState,
    MotionFeatures,
    OffsetEstimate,
    OffsetScore,
    TemporalResultRow,
    TemporalTrial,
)
from kinesync.visualization.temporal_telemetry import (
    TemporalTelemetryVisualFrame,
    compose_temporal_telemetry_frame,
    write_temporal_telemetry_image,
    write_temporal_telemetry_video,
)


def _frame(*, trajectory_id: str, frame_index: int) -> TemporalTelemetryVisualFrame:
    head = np.full((120, 160, 3), (32, 116, 196), dtype=np.uint8)
    wrist = np.full((90, 120, 3), (210, 124, 35), dtype=np.uint8)
    head[:, : frame_index % 40 + 1] = (242, 244, 248)
    wrist[:, -((frame_index % 35) + 1) :] = (48, 164, 98)
    return TemporalTelemetryVisualFrame(
        trajectory_id=trajectory_id,
        condition_id=f"{trajectory_id}|offset=80|jitter=10|dropout=0.15",
        frame_index=frame_index,
        frame_count=6,
        head_rgb=head,
        wrist_rgb=wrist,
        score_profile=tuple(
            OffsetScore(offset_ms=float(offset), correlation=score)
            for offset, score in ((-120, -0.4), (-80, -0.1), (-40, 0.2), (0, 0.4), (40, 0.72), (80, 0.94), (120, 0.68))
        ),
        raw_joint_abs_error_rad=np.array([0.08, 0.06, 0.04, 0.05, 0.03, 0.02]),
        guarded_joint_abs_error_rad=np.array([0.02, 0.01, 0.015, 0.01, 0.01, 0.005]),
        injected_offset_ms=80.0,
        estimated_offset_ms=80.0,
        committed_offset_ms=80.0,
        guarded_committed=True,
    )


def _relative_offset_context() -> tuple[TemporalTelemetryVisualFrame, OffsetEstimate]:
    features = MotionFeatures(
        camera_midpoint_ns=np.array([10, 20], dtype=np.int64),
        visual_motion=np.array([0.2, 0.7]),
        state_motion=np.array([0.1, 0.6]),
    )
    trial = TemporalTrial(
        trajectory_id="f1.hdf5",
        split="test",
        condition_id="f1.hdf5:0:3|offset=80|jitter=0|dropout=0.00",
        features=features,
        corrupted_state=CorruptedState(
            timestamp_ns=np.array([0, 10, 20], dtype=np.int64),
            qpos=np.zeros((3, 8), dtype=np.float64),
            retained_indices=np.array([0, 1, 2], dtype=np.int64),
        ),
        target_qpos=np.zeros((2, 8), dtype=np.float64),
        injected_offset_ms=80.0,
        jitter_ms=0.0,
        dropout=0.0,
    )
    estimate = OffsetEstimate(
        offset_ms=20.0,
        peak_correlation=0.9,
        peak_margin=0.4,
        visual_mad=0.2,
        state_mad=0.1,
        score_profile=(
            OffsetScore(offset_ms=-40.0, correlation=0.2),
            OffsetScore(offset_ms=20.0, correlation=0.9),
            OffsetScore(offset_ms=80.0, correlation=0.3),
        ),
    )

    def row(*, method: str, qmae_rad: float, committed: bool, applied: float) -> TemporalResultRow:
        return TemporalResultRow(
            trajectory_id=trial.trajectory_id,
            split="test",
            condition_id=trial.condition_id,
            method=method,
            injected_offset_ms=trial.injected_offset_ms,
            jitter_ms=trial.jitter_ms,
            dropout=trial.dropout,
            native_delay_ms=-60.0,
            estimated_absolute_offset_ms=estimate.offset_ms,
            estimated_offset_ms=80.0,
            applied_correction_ms=applied,
            committed=committed,
            guard_reason="accepted" if committed else "raw_linear",
            peak_correlation=estimate.peak_correlation,
            peak_margin=estimate.peak_margin,
            visual_mad=estimate.visual_mad,
            state_mad=estimate.state_mad,
            offset_mae_ms=abs(applied - trial.injected_offset_ms),
            qmae_rad=qmae_rad,
            qmae_reduction_rad=0.4 - qmae_rad,
            recovery_success=committed,
            interpolated_qpos=np.zeros((2, 8), dtype=np.float64),
        )

    selection = _TemporalVisualSelection(
        trajectory=TakePensTrajectory("f1.hdf5", "test", 3, Path("fixture.hdf5")),
        segment=ContinuousSegment("f1.hdf5", "test", 0, 3),
        trial=trial,
        estimate=estimate,
        raw=row(method="raw_linear", qmae_rad=0.4, committed=False, applied=0.0),
        guarded=row(method="kinesync_guarded", qmae_rad=0.1, committed=True, applied=80.0),
        receipt={},
    )
    context = _visual_context(
        selection,
        frame_index=0,
        head_rgb=np.full((120, 160, 3), 40, dtype=np.uint8),
        wrist_rgb=np.full((90, 120, 3), 120, dtype=np.uint8),
    )
    return context, estimate


class TemporalTelemetryVisualizationTest(unittest.TestCase):
    def test_writes_nonblank_decodable_1080p_image_and_video(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            relative_context, estimate = _relative_offset_context()
            winner = max(
                relative_context.score_profile,
                key=lambda score: float(score.correlation),
            )
            self.assertEqual(winner.offset_ms, relative_context.estimated_offset_ms)
            self.assertEqual(
                tuple(score.offset_ms for score in estimate.score_profile),
                (-40.0, 20.0, 80.0),
            )
            contexts = [relative_context, _frame(trajectory_id="f2.hdf5", frame_index=4)]
            image_path = root / "telemetry.png"
            video_path = root / "telemetry.mp4"

            canvas = compose_temporal_telemetry_frame(contexts)
            self.assertEqual(canvas.shape, (1080, 1920, 3))
            self.assertGreater(int(canvas.std()), 0)
            write_temporal_telemetry_image(image_path, contexts)
            write_temporal_telemetry_video(
                video_path,
                [contexts, list(reversed(contexts)), contexts],
                fps=12,
            )

            image = cv2.imread(str(image_path))
            self.assertIsNotNone(image)
            self.assertEqual(image.shape[:2], (1080, 1920))
            self.assertGreater(int(image.std()), 0)
            capture = cv2.VideoCapture(str(video_path))
            self.assertTrue(capture.isOpened())
            self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), 1920)
            self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)), 1080)
            self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 3)
            decoded, decoded_frame = capture.read()
            capture.release()
            self.assertTrue(decoded)
            self.assertGreater(int(decoded_frame.std()), 0)


if __name__ == "__main__":
    unittest.main()
