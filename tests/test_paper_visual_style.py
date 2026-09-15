from __future__ import annotations

from dataclasses import FrozenInstanceError
import json
from pathlib import Path
import re
import tempfile
import unittest
import xml.etree.ElementTree as ElementTree

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from kinesync.visualization.paper_contracts import (
    ComparisonMode,
    EvidenceLevel,
    PaperArtifactRecord,
)
from kinesync.visualization.paper_style import (
    METHOD_COLORS,
    publication_rc_context,
    save_square_quantitative,
)


_SHA_A = "a" * 64
_SHA_B = "b" * 64
_SHA_C = "c" * 64
_SHA_D = "d" * 64


def _record(**overrides: object) -> PaperArtifactRecord:
    values: dict[str, object] = {
        "artifact_id": "Q-RT2-01",
        "stage": "RT2",
        "claim": "Paired-camera state error is lower for KineSync.",
        "source_run": "20260825_rt2_formal",
        "source_hashes": {"metrics.json": _SHA_A},
        "evidence_level": EvidenceLevel.FORMAL,
        "method": "ours_full",
        "baseline": "primary_baseline",
        "case_id": "paired-camera-001",
        "camera": "head",
        "selection_policy": "median-positive",
        "comparison_metrics": {
            "primary_baseline": 10.0,
            "primary_method": 7.0,
            "state_error_reduction": 0.42,
        },
        "comparison_mode": ComparisonMode.METHOD_COMPARISON,
        "comparison_metric_id": "state_error",
        "width": 2100,
        "height": 2100,
        "recommended_title": "Paired-camera state recovery",
        "recommended_subtitle": "Formal evaluation",
        "caption_draft": "KineSync improves paired-camera state recovery.",
        "output_filenames": {
            "pdf": "Q-RT2-01.pdf",
            "svg": "Q-RT2-01.svg",
            "png": "Q-RT2-01.png",
        },
        "output_hashes": {"pdf": _SHA_B, "svg": _SHA_C, "png": _SHA_D},
    }
    values.update(overrides)
    return PaperArtifactRecord(**values)  # type: ignore[arg-type]


class PaperVisualStyleTest(unittest.TestCase):
    def test_method_palette_matches_the_frozen_semantic_mapping(self) -> None:
        self.assertEqual(
            dict(METHOD_COLORS),
            {
                "ours_full": "#0F4D92",
                "ours_ablation": "#3775BA",
                "ours_weak": "#B4C0E4",
                "primary_baseline": "#B64342",
                "secondary_baseline": "#E9A6A1",
                "reference": "#272727",
                "oracle": "#767676",
                "candidate": "#C58A18",
                "positive": "#2E9E44",
                "rollback": "#E53935",
                "uncertainty": "#9A4D8E",
            },
        )

    def test_publication_context_sets_paper_fonts_spines_and_text_output(self) -> None:
        with publication_rc_context():
            self.assertEqual(
                matplotlib.rcParams["font.family"],
                ["Arial", "Helvetica", "DejaVu Sans"],
            )
            self.assertEqual(matplotlib.rcParams["font.size"], 8.0)
            self.assertEqual(matplotlib.rcParams["axes.linewidth"], 0.8)
            self.assertFalse(matplotlib.rcParams["axes.spines.top"])
            self.assertFalse(matplotlib.rcParams["axes.spines.right"])
            self.assertFalse(matplotlib.rcParams["axes.grid"])
            self.assertFalse(matplotlib.rcParams["legend.frameon"])
            self.assertEqual(matplotlib.rcParams["pdf.fonttype"], 42)
            self.assertEqual(matplotlib.rcParams["ps.fonttype"], 42)
            self.assertEqual(matplotlib.rcParams["svg.fonttype"], "none")

    def test_square_export_normalizes_preexisting_artists_and_physical_output_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            stem = Path(directory) / "Q-RT2-01"
            with matplotlib.rc_context(
                {
                    "font.size": 20.0,
                    "axes.linewidth": 3.0,
                    "axes.spines.top": True,
                    "axes.spines.right": True,
                    "axes.grid": True,
                    "legend.frameon": True,
                }
            ):
                figure, axis = plt.subplots(figsize=(8, 3))
            axis.plot([0, 1], [0, 1], color=METHOD_COLORS["ours_full"])
            axis.set_xlabel("State error")
            axis.text(0.5, 0.5, "n=12")
            axis.legend(["KineSync"])
            with self.assertNoLogs("matplotlib.font_manager", level="WARNING"):
                pdf_path, svg_path, png_path = save_square_quantitative(figure, stem)
            self.addCleanup(plt.close, figure)

            self.assertEqual((pdf_path, svg_path, png_path), (stem.with_suffix(".pdf"), stem.with_suffix(".svg"), stem.with_suffix(".png")))
            self.assertTrue(all(path.is_file() and path.stat().st_size > 0 for path in (pdf_path, svg_path, png_path)))
            self.assertEqual(tuple(round(value, 6) for value in figure.get_size_inches()), (round(89 / 25.4, 6), round(89 / 25.4, 6)))
            self.assertEqual(axis.xaxis.label.get_fontsize(), 8.0)
            self.assertEqual(axis.spines["left"].get_linewidth(), 0.8)
            self.assertFalse(axis.spines["top"].get_visible())
            self.assertFalse(axis.spines["right"].get_visible())
            self.assertFalse(any(gridline.get_visible() for gridline in axis.get_xgridlines() + axis.get_ygridlines()))
            self.assertFalse(axis.get_legend().get_frame_on())
            image = matplotlib.image.imread(png_path)
            self.assertEqual(image.shape[:2], (2100, 2100))

            pdf = pdf_path.read_bytes()
            media_box = re.search(rb"/MediaBox \[ 0 0 ([0-9.]+) ([0-9.]+) \]", pdf)
            self.assertIsNotNone(media_box)
            self.assertAlmostEqual(float(media_box.group(1)), 89 / 25.4 * 72, places=4)  # type: ignore[union-attr]
            self.assertAlmostEqual(float(media_box.group(2)), 89 / 25.4 * 72, places=4)  # type: ignore[union-attr]
            self.assertIn(b"/FontFile2", pdf)

            root = ElementTree.parse(svg_path).getroot()
            self.assertAlmostEqual(float(root.attrib["width"].removesuffix("pt")), 89 / 25.4 * 72, places=4)
            self.assertAlmostEqual(float(root.attrib["height"].removesuffix("pt")), 89 / 25.4 * 72, places=4)
            self.assertIn("<text", svg_path.read_text(encoding="utf-8"))

    def test_square_export_rejects_rendered_figure_and_axes_titles(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            figure, axis = plt.subplots()
            figure.suptitle("Narrative title")
            with self.assertRaisesRegex(ValueError, "title"):
                save_square_quantitative(figure, Path(directory) / "Q-RT2-01")
            plt.close(figure)

            figure, axis = plt.subplots()
            axis.set_title("Narrative title")
            with self.assertRaisesRegex(ValueError, "title"):
                save_square_quantitative(figure, Path(directory) / "Q-RT2-01")
            plt.close(figure)

            figure, axis = plt.subplots()
            figure.text(0.5, 0.98, "Narrative subtitle")
            with self.assertRaisesRegex(ValueError, "text"):
                save_square_quantitative(figure, Path(directory) / "Q-RT2-01")
            plt.close(figure)

            figure, _ = plt.subplots()
            second_axis = figure.add_axes((0.1, 0.1, 0.2, 0.2))
            second_axis.set_title("Second narrative title")
            with self.assertRaisesRegex(ValueError, "title"):
                save_square_quantitative(figure, Path(directory) / "Q-RT2-01")
            plt.close(figure)


class PaperArtifactRecordTest(unittest.TestCase):
    def test_record_is_frozen_and_uses_canonical_json(self) -> None:
        source_hashes = {"metrics.json": _SHA_A}
        record = _record(source_hashes=source_hashes)
        source_hashes["metrics.json"] = _SHA_B
        with self.assertRaises(FrozenInstanceError):
            record.stage = "RT8"  # type: ignore[misc]
        with self.assertRaises(TypeError):
            record.source_hashes["metrics.json"] = _SHA_B  # type: ignore[index]
        with self.assertRaises(TypeError):
            record.output_hashes["pdf"] = _SHA_A  # type: ignore[index]
        self.assertEqual(record.source_hashes["metrics.json"], _SHA_A)
        payload = record.canonical_json()
        self.assertEqual(payload, json.dumps(json.loads(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False))
        decoded = json.loads(payload)
        self.assertEqual(decoded["evidence_level"], "formal")
        self.assertEqual(decoded["comparison_mode"], "method_comparison")
        self.assertEqual(decoded["comparison_metric_id"], "state_error")
        self.assertEqual(decoded["primary_comparison_delta"], 3.0)
        self.assertIsNone(decoded["grid_layout"])

    def test_record_rejects_malformed_method_baseline_and_evidence_level(self) -> None:
        for field, value in (("method", "unknown_method"), ("method", []), ("baseline", {})):
            with self.subTest(field=field, value=value):
                with self.assertRaisesRegex(ValueError, "method|baseline"):
                    _record(**{field: value})
        with self.assertRaisesRegex(ValueError, "evidence"):
            _record(evidence_level="not-evidence")

    def test_record_rejects_missing_malformed_and_mismatched_hashes(self) -> None:
        with self.assertRaisesRegex(ValueError, "source_hashes"):
            _record(source_hashes={})
        with self.assertRaisesRegex(ValueError, "output_hashes"):
            _record(output_hashes={"pdf": _SHA_B})
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            _record(source_hashes={"metrics.json": _SHA_A.upper()})
        with self.assertRaisesRegex(ValueError, "source_hashes"):
            _record(source_hashes={"metrics.json": 7})
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            _record(output_hashes={"pdf": "not-a-sha", "svg": _SHA_C, "png": _SHA_D})
        with self.assertRaisesRegex(ValueError, "cover"):
            _record(output_hashes={"pdf": _SHA_B, "svg": _SHA_C, "extra": _SHA_D})

    def test_record_rejects_rendered_title_and_subtitle_text(self) -> None:
        with self.assertRaisesRegex(ValueError, "rendered_title"):
            _record(rendered_title="Narrative title")
        with self.assertRaisesRegex(ValueError, "rendered_title"):
            _record(rendered_subtitle="Narrative subtitle")

    def test_record_rejects_nonflat_output_names_and_non_square_quantitative_dimensions(self) -> None:
        for filename in ("nested/Q-RT2-01.pdf", "nested\\Q-RT2-01.pdf"):
            with self.subTest(filename=filename):
                with self.assertRaisesRegex(ValueError, "flat"):
                    _record(output_filenames={"pdf": filename, "svg": "Q-RT2-01.svg", "png": "Q-RT2-01.png"})
        with self.assertRaisesRegex(ValueError, "square"):
            _record(width=2100, height=1600)

    def test_quantitative_record_derives_primary_delta_and_rejects_equal_values(self) -> None:
        record = _record()
        self.assertEqual(record.primary_comparison_delta, 3.0)
        with self.assertRaisesRegex(ValueError, "primary"):
            _record(
                comparison_metrics={"primary_baseline": 1.0, "primary_method": 1.0},
            )
        with self.assertRaisesRegex(ValueError, "primary"):
            _record(comparison_metrics={"state_error_reduction": 0.42})

    def test_quantitative_record_rejects_a_caller_supplied_pseudo_delta(self) -> None:
        with self.assertRaises(TypeError):
            _record(
                comparison_metrics={"primary_baseline": 1.0, "primary_method": 1.0},
                primary_comparison_delta=0.42,
            )

    def test_comparison_modes_are_closed_and_descriptive_statistics_have_no_primary_delta(self) -> None:
        descriptive = _record(
            comparison_mode=ComparisonMode.DESCRIPTIVE_STATISTICS,
            comparison_metric_id=None,
            comparison_metrics={"median_latency_ms": 1.2, "p95_latency_ms": 4.1},
        )
        self.assertIsNone(descriptive.primary_comparison_delta)
        payload = json.loads(descriptive.canonical_json())
        self.assertEqual(payload["comparison_mode"], "descriptive_statistics")
        self.assertIsNone(payload["comparison_metric_id"])
        with self.assertRaisesRegex(ValueError, "descriptive"):
            _record(
                comparison_mode="descriptive_statistics",
                comparison_metric_id="latency_ms",
                comparison_metrics={"primary_baseline": 1.0, "primary_method": 2.0},
            )
        with self.assertRaisesRegex(ValueError, "descriptive"):
            _record(
                comparison_mode="descriptive_statistics",
                comparison_metric_id=None,
                comparison_metrics={"minimum": 1.0, "maximum": 1.0},
            )
        with self.assertRaisesRegex(ValueError, "comparison_mode"):
            _record(comparison_mode="open_ended")

    def test_method_comparison_requires_a_metric_identifier_and_primary_pair(self) -> None:
        with self.assertRaisesRegex(ValueError, "comparison_metric_id"):
            _record(comparison_metric_id=None)
        with self.assertRaisesRegex(ValueError, "primary"):
            _record(comparison_metrics={"state_error": 0.42})

    def test_nonquantitative_records_allow_no_primary_delta(self) -> None:
        frame = _record(
            artifact_id="F-RT8-01",
            artifact_kind="frame",
            width=1600,
            height=1600,
            output_filenames={"png": "F-RT8-01.png"},
            output_hashes={"png": _SHA_B},
            comparison_mode=None,
            comparison_metric_id=None,
            comparison_metrics={},
        )
        grid = _record(
            artifact_id="G-RT8-01",
            artifact_kind="grid",
            grid_layout="4x2",
            width=3200,
            height=1600,
            output_filenames={"pdf": "G-RT8-01.pdf", "svg": "G-RT8-01.svg", "png": "G-RT8-01.png"},
            output_hashes={"pdf": _SHA_B, "svg": _SHA_C, "png": _SHA_D},
            comparison_mode=None,
            comparison_metric_id=None,
            comparison_metrics={},
        )
        video = _record(
            artifact_id="V-RT8-01",
            artifact_kind="video",
            width=1920,
            height=1080,
            output_filenames={"mp4": "V-RT8-01.mp4"},
            output_hashes={"mp4": _SHA_B},
            comparison_mode=None,
            comparison_metric_id=None,
            comparison_metrics={},
        )
        self.assertEqual((frame.artifact_kind, grid.artifact_kind, video.artifact_kind), ("frame", "grid", "video"))
        self.assertEqual((frame.primary_comparison_delta, grid.primary_comparison_delta, video.primary_comparison_delta), (None, None, None))
        for record in (frame, grid, video):
            with self.subTest(artifact_id=record.artifact_id):
                payload = json.loads(record.canonical_json())
                self.assertIsNone(payload["comparison_mode"])
                self.assertIsNone(payload["comparison_metric_id"])
                self.assertEqual(payload["comparison_metrics"], {})
                self.assertIsNone(payload["primary_comparison_delta"])
        self.assertEqual(json.loads(grid.canonical_json())["grid_layout"], "4x2")

    def test_nonquantitative_primary_values_must_be_complete_and_unequal(self) -> None:
        values = {
            "artifact_kind": "frame",
            "width": 1600,
            "height": 1600,
            "output_filenames": {"png": "F-RT8-01.png"},
            "output_hashes": {"png": _SHA_B},
        }
        with self.assertRaisesRegex(ValueError, "primary"):
            _record(
                **values,
                comparison_metrics={"primary_baseline": 1.0, "primary_method": 1.0},
            )
        with self.assertRaisesRegex(ValueError, "primary"):
            _record(**values, comparison_metrics={"primary_baseline": 1.0})
        with self.assertRaisesRegex(ValueError, "descriptive"):
            _record(
                **values,
                comparison_mode=ComparisonMode.DESCRIPTIVE_STATISTICS,
                comparison_metric_id=None,
                comparison_metrics={"minimum": 1.0, "maximum": 1.0},
            )
        record = _record(
            **values,
            comparison_metrics={"primary_baseline": 1.0, "primary_method": 0.8},
        )
        self.assertAlmostEqual(record.primary_comparison_delta, 0.2)

    def test_nonquantitative_mode_less_metrics_infer_closed_comparison_modes(self) -> None:
        values = {
            "artifact_kind": "frame",
            "width": 1600,
            "height": 1600,
            "output_filenames": {"png": "F-RT8-01.png"},
            "output_hashes": {"png": _SHA_B},
            "comparison_mode": None,
        }
        primary = _record(
            **values,
            comparison_metric_id="state_error",
            comparison_metrics={"primary_baseline": 1.0, "primary_method": 0.8},
        )
        descriptive = _record(
            **values,
            comparison_metric_id=None,
            comparison_metrics={"median_latency_ms": 1.2, "p95_latency_ms": 4.1},
        )
        self.assertEqual(primary.comparison_mode, ComparisonMode.METHOD_COMPARISON)
        self.assertEqual(descriptive.comparison_mode, ComparisonMode.DESCRIPTIVE_STATISTICS)
        self.assertEqual(json.loads(primary.canonical_json())["comparison_mode"], "method_comparison")
        self.assertEqual(json.loads(descriptive.canonical_json())["comparison_mode"], "descriptive_statistics")
        with self.assertRaisesRegex(ValueError, "primary"):
            _record(
                **values,
                comparison_metric_id="state_error",
                comparison_metrics={"primary_baseline": 1.0},
            )
        with self.assertRaisesRegex(ValueError, "comparison_mode"):
            _record(comparison_mode=None)

    def test_record_rejects_noncompliant_closed_kind_rule(self) -> None:
        with self.assertRaisesRegex(ValueError, "frame"):
            _record(
                artifact_kind="frame",
                width=2100,
                height=2100,
                output_filenames={"png": "F-RT8-01.png"},
                output_hashes={"png": _SHA_B},
            )
        with self.assertRaisesRegex(ValueError, "video"):
            _record(
                artifact_kind="video",
                width=1920,
                height=1080,
                output_filenames={"png": "V-RT8-01.png"},
                output_hashes={"png": _SHA_B},
            )

    def test_grid_layout_uses_the_closed_paper_allowlist(self) -> None:
        for layout in ("3x2", "4x2", "3x3", "4x3", "7x2", "8x2", "9x2", "8x3"):
            with self.subTest(layout=layout):
                record = _record(
                    artifact_kind="grid",
                    grid_layout=layout,
                    width=3200,
                    height=1600,
                    output_filenames={"pdf": "G-RT8-01.pdf", "svg": "G-RT8-01.svg", "png": "G-RT8-01.png"},
                    output_hashes={"pdf": _SHA_B, "svg": _SHA_C, "png": _SHA_D},
                )
                self.assertEqual(record.grid_layout, layout)
        for layout in (None, "1x1", "5x1", "2x3", "4 x 2"):
            with self.subTest(layout=layout):
                with self.assertRaisesRegex(ValueError, "grid_layout"):
                    _record(
                        artifact_kind="grid",
                        grid_layout=layout,
                        width=3200,
                        height=1600,
                        output_filenames={"pdf": "G-RT8-01.pdf", "svg": "G-RT8-01.svg", "png": "G-RT8-01.png"},
                        output_hashes={"pdf": _SHA_B, "svg": _SHA_C, "png": _SHA_D},
                    )
        with self.assertRaisesRegex(ValueError, "grid_layout"):
            _record(grid_layout="4x2")


if __name__ == "__main__":
    unittest.main()
