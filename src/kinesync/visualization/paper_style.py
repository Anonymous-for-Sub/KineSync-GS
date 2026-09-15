"""Shared Nature-style settings for title-free quantitative paper figures."""

from __future__ import annotations

from contextlib import AbstractContextManager
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import matplotlib as mpl
from matplotlib import text as mpl_text
from matplotlib.figure import Figure


_SQUARE_SIDE_INCHES = 89.0 / 25.4
_PNG_PIXELS = 2100
_PNG_DPI = _PNG_PIXELS / _SQUARE_SIDE_INCHES
_FONT_FAMILY = ["Arial", "Helvetica", "DejaVu Sans"]
_EXPORT_FONT_FAMILY = ["DejaVu Sans"]
_FONT_SIZE = 8.0
_AXIS_LINEWIDTH = 0.8

METHOD_COLORS: Mapping[str, str] = MappingProxyType(
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
    }
)


def publication_rc_context() -> AbstractContextManager[None]:
    """Provide the fixed typography and line treatment for paper exports."""

    return mpl.rc_context(
        {
            "font.family": _FONT_FAMILY,
            "font.size": _FONT_SIZE,
            "axes.linewidth": _AXIS_LINEWIDTH,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def _require_title_free(figure: Figure) -> None:
    if figure._suptitle is not None and figure._suptitle.get_text().strip():
        raise ValueError("Quantitative paper figures must not contain a rendered title")
    # Figure-level text belongs to the later composition layer, where narrative
    # titles/subtitles are permitted; atomic figures retain axes-local labels and annotations.
    if any(text.get_text().strip() for text in figure.texts):
        raise ValueError("Quantitative paper figures must not contain rendered figure text")
    for axis in figure.get_axes():
        if any(axis.get_title(loc=location).strip() for location in ("left", "center", "right")):
            raise ValueError("Quantitative paper figures must not contain a rendered title")


def _normalize_existing_artists(figure: Figure) -> None:
    """Apply the export contract to artists created before the rc context."""

    figure.set_facecolor("white")
    for axis in figure.get_axes():
        axis.set_facecolor("white")
        axis.grid(False)
        axis.tick_params(axis="both", width=_AXIS_LINEWIDTH, labelsize=_FONT_SIZE)
        for spine_name, spine in axis.spines.items():
            spine.set_linewidth(_AXIS_LINEWIDTH)
            spine.set_visible(spine_name not in {"top", "right"})
        for text in axis.findobj(match=mpl_text.Text):
            text.set_fontfamily(_EXPORT_FONT_FAMILY)
            text.set_fontsize(_FONT_SIZE)
        legend = axis.get_legend()
        if legend is not None:
            legend.set_frame_on(False)


def save_square_quantitative(fig: Figure, output_stem: Path) -> tuple[Path, Path, Path]:
    """Write one title-free 89 mm quantitative atom as PDF, SVG, and PNG."""

    stem = Path(output_stem)
    if stem.suffix:
        raise ValueError("output_stem must not include a file suffix")
    _require_title_free(fig)
    _normalize_existing_artists(fig)
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.set_size_inches(_SQUARE_SIDE_INCHES, _SQUARE_SIDE_INCHES, forward=True)

    pdf_path = stem.with_suffix(".pdf")
    svg_path = stem.with_suffix(".svg")
    png_path = stem.with_suffix(".png")
    with publication_rc_context():
        fig.savefig(pdf_path, format="pdf", facecolor="white")
        fig.savefig(svg_path, format="svg", facecolor="white")
        fig.savefig(png_path, format="png", dpi=_PNG_DPI, facecolor="white")
    return pdf_path, svg_path, png_path


__all__ = ["METHOD_COLORS", "publication_rc_context", "save_square_quantitative"]
