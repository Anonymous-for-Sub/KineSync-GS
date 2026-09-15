"""Manifest-bound, title-free grids for Task 4 paper evidence."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import matplotlib.pyplot as plt
import numpy as np

from .paper_contracts import ArtifactKind, EvidenceLevel, PaperArtifactRecord
from .paper_style import METHOD_COLORS


MAIN_LAYOUTS = frozenset({"3x2", "4x2", "3x3", "4x3", "7x2", "8x2"})
APPENDIX_LAYOUTS = frozenset({"9x2", "8x3"})
_LAYOUTS = MAIN_LAYOUTS | APPENDIX_LAYOUTS
_GRID_ALLOCATION = {
    "RT2": (("4x2", False), ("4x3", False), ("8x2", False)),
    "RT3": (("4x2", False), ("4x3", False), ("8x2", False)),
    "RT6": (("4x2", False), ("8x2", False), ("8x3", True)),
    "RT7": (("4x2", False), ("8x2", False), ("8x3", True)),
    "RT8": (("3x2", False), ("4x2", False), ("8x2", False), ("8x3", True)),
    "RT9": (("4x2", False), ("3x3", False), ("8x2", False), ("9x2", True), ("8x3", True)),
}
GRID_CONTENT_SIZE = 480
GRID_FOOTER_HEIGHT = 36
GRID_GAP = 10


def grid_footer_label(cell: "GridCell") -> str:
    return f"{cell.role} | {cell.record.camera} | t={cell.timepoint}"


def grid_footer_color(role: str) -> tuple[int, int, int]:
    method = {"baseline": "primary_baseline", "ours": "ours_full"}.get(role)
    if method is None:
        raise ValueError("grid footer role must be baseline or ours")
    value = METHOD_COLORS[method]
    return tuple(int(value[index : index + 2], 16) for index in (1, 3, 5))


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _shape(layout: str) -> tuple[int, int]:
    try:
        cols, rows = (int(item) for item in layout.split("x"))
    except (TypeError, ValueError) as error:
        raise ValueError("layout must use CxR notation") from error
    return cols, rows


@dataclass(frozen=True)
class GridCell:
    """One atomic Task 3 frame and its pair-matching metadata."""

    pair_key: str
    role: str
    timepoint: str
    record: PaperArtifactRecord
    image_path: Path


@dataclass(frozen=True)
class GridSpec:
    artifact_id: str
    stage: str
    source_run: str
    layout: str
    appendix: bool
    cells: tuple[GridCell, ...]
    claim: str

    def __post_init__(self) -> None:
        if self.layout not in _LAYOUTS:
            raise ValueError("layout is not in the closed paper grid allowlist")
        if self.appendix != (self.layout in APPENDIX_LAYOUTS):
            raise ValueError("appendix flag must match the selected grid layout")
        cols, rows = _shape(self.layout)
        if len(self.cells) != cols * rows:
            raise ValueError("grid cell count must match its layout")
        if not self.artifact_id or not self.stage or not self.source_run or not self.claim:
            raise ValueError("grid metadata must be nonempty")


def _validate_cell(cell: GridCell, stage: str, source_run: str) -> tuple[np.ndarray, str]:
    if cell.role not in {"baseline", "ours"}:
        raise ValueError("grid cell role must be baseline or ours")
    if not cell.record.output_filenames.get("png", "").endswith(f"-{cell.role}.png"):
        raise ValueError("grid cell role must match its atomic-frame manifest role")
    if cell.record.artifact_kind is not ArtifactKind.FRAME:
        raise ValueError("grids can only consume Task 3 frame artifacts")
    if cell.record.stage != stage:
        raise ValueError("grid cell stage mismatches specification")
    if cell.record.source_run != source_run:
        raise ValueError("grid cell source run mismatches specification")
    if "contrast_receipt" not in cell.record.source_hashes:
        raise ValueError("grid cell is missing its private contrast receipt hash")
    if cell.record.output_filenames.get("png") != cell.image_path.name:
        raise ValueError("grid cell path does not match the atomic artifact manifest")
    if not cell.record.output_filenames["png"].endswith(f"-{cell.role}.png"):
        raise ValueError("grid cell role does not match the atomic artifact filename")
    if not cell.image_path.is_file() or _sha_file(cell.image_path) != cell.record.output_hashes["png"]:
        raise ValueError("grid cell image hash does not match its atomic artifact")
    image = cv2.imread(str(cell.image_path), cv2.IMREAD_COLOR)
    if image is None or image.shape[:2] != (1600, 1600):
        raise ValueError("grid cell must be an undistorted 1600x1600 atomic frame")
    if float(image.std()) <= 1.0:
        raise ValueError("grid cell is blank")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), _sha_file(cell.image_path)


def _validate_pairs(spec: GridSpec) -> None:
    groups: dict[str, list[GridCell]] = {}
    for cell in spec.cells:
        groups.setdefault(cell.pair_key, []).append(cell)
    for pair_key, cells in groups.items():
        if len(cells) == 1 and cells[0].role == "baseline":
            # Odd-cell layouts retain one explicitly baseline-context atom; comparison cells remain paired.
            continue
        if len(cells) != 2 or {cell.role for cell in cells} != {"baseline", "ours"}:
            raise ValueError(f"pair {pair_key} must contain one baseline and one ours cell")
        first, second = cells
        if first.record.case_id != second.record.case_id:
            raise ValueError("paired grid cells must share the same case")
        if first.record.camera != second.record.camera:
            raise ValueError("paired grid cells must share the same camera")
        if first.timepoint != second.timepoint:
            raise ValueError("paired grid cells must share the same timepoint")


def _write_vector_exports(canvas: np.ndarray, destination: Path) -> tuple[Path, Path]:
    rgb = canvas
    figure = plt.figure(figsize=(rgb.shape[1] / 100.0, rgb.shape[0] / 100.0), dpi=100)
    axis = figure.add_axes((0, 0, 1, 1))
    axis.imshow(rgb)
    axis.set_axis_off()
    pdf = destination.with_suffix(".pdf")
    svg = destination.with_suffix(".svg")
    figure.savefig(pdf, dpi=100, pad_inches=0)
    figure.savefig(svg, dpi=100, pad_inches=0)
    plt.close(figure)
    return pdf, svg


def compose_image_grid(spec: GridSpec, destination: Path) -> PaperArtifactRecord:
    """Compose a title-free grid from hash-verified Task 3 frame artifacts."""

    if Path(destination).suffix != ".png":
        raise ValueError("grid destination must be a flat PNG filename")
    destination = Path(destination)
    if "/" in destination.name or "\\" in destination.name:
        raise ValueError("grid destination filename must be flat")
    _validate_pairs(spec)
    images: list[np.ndarray] = []
    source_hashes: dict[str, str] = {}
    for index, cell in enumerate(spec.cells):
        image, image_hash = _validate_cell(cell, spec.stage, spec.source_run)
        images.append(image)
        source_hashes[f"atom_{index:02d}"] = image_hash
        source_hashes[f"receipt_{index:02d}"] = cell.record.source_hashes["contrast_receipt"]

    cols, rows = _shape(spec.layout)
    # 8 columns render at 3910 px. The 480x480 image remains square and the
    # footer occupies its own band, rather than squeezing the atomic frame.
    cell_size, gap, label_height = GRID_CONTENT_SIZE, GRID_GAP, GRID_FOOTER_HEIGHT
    content_width = cols * cell_size + (cols - 1) * gap
    height = rows * (cell_size + label_height) + (rows - 1) * gap
    outer_padding = max(0, (height - content_width + 1) // 2)
    width = content_width + 2 * outer_padding
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    for index, (cell, image) in enumerate(zip(spec.cells, images, strict=True)):
        row, col = divmod(index, cols)
        x, y = outer_padding + col * (cell_size + gap), row * (cell_size + label_height + gap)
        canvas[y : y + cell_size, x : x + cell_size] = cv2.resize(image, (cell_size, cell_size), interpolation=cv2.INTER_AREA)
        label = grid_footer_label(cell)
        scale = 0.55
        while scale > 0.22 and cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0] > cell_size - 12:
            scale -= 0.02
        text_width, text_height = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0]
        if text_width > cell_size - 12 or text_height > label_height - 4:
            raise ValueError("grid footer label cannot fit without clipping")
        cv2.putText(canvas, label, (x + 6, y + cell_size + label_height - 8), cv2.FONT_HERSHEY_SIMPLEX, scale, grid_footer_color(cell.role), 1, cv2.LINE_AA)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 6]):
        raise RuntimeError("could not write grid PNG")
    pdf, svg = _write_vector_exports(canvas, destination)
    source_hashes["grid_cell_population"] = _sha_bytes("|".join(sorted(source_hashes.values())).encode("ascii"))
    source_hashes["deterministic_pair_schedule"] = _sha_bytes("|".join(cell.pair_key for cell in spec.cells).encode("utf-8"))
    source_hashes["grid_geometry"] = _sha_bytes(f"content_width={content_width};height={height};outer_padding={outer_padding};cell={cell_size};footer={label_height};gap={gap}".encode("ascii"))
    output_paths = {"png": destination, "pdf": pdf, "svg": svg}
    return PaperArtifactRecord(
        artifact_id=spec.artifact_id,
        artifact_kind=ArtifactKind.GRID,
        stage=spec.stage,
        claim=spec.claim,
        source_run=spec.source_run,
        source_hashes=source_hashes,
        evidence_level=EvidenceLevel.DEVELOPMENT_ONLY if spec.stage == "RT7" else EvidenceLevel.FORMAL,
        method="ours_full",
        baseline="primary_baseline",
        case_id=f"{spec.stage}-matched-frame-grid",
        camera="matched-per-pair",
        selection_policy="manifest_backed_atomic_pair_grid_deterministic_repeat_if_capacity_exceeds_population",
        comparison_metrics={},
        width=width,
        height=height,
        recommended_title="Matched visual comparison grid",
        recommended_subtitle="Title-free atomic panels",
        caption_draft="Manifest-bound Task 3 atomic frame grid.",
        output_filenames={kind: path.name for kind, path in output_paths.items()},
        output_hashes={kind: _sha_file(path) for kind, path in output_paths.items()},
        grid_layout=spec.layout,
    )


def _slug(value: str) -> str:
    return "".join(character if character.isalnum() else "-" for character in value).strip("-")[:90]


def _record_for(records: Sequence[PaperArtifactRecord], pair: object, role: str) -> PaperArtifactRecord:
    candidates = [
        record for record in records
        if record.stage == pair.stage and record.case_id == pair.case_id and record.camera == pair.camera
        and record.output_filenames["png"].endswith(f"-{role}.png")
    ]
    if len(candidates) != 1:
        raise ValueError(f"cannot resolve one Task 3 atom for {pair.stage}/{pair.case_id}/{pair.camera}/{role}")
    return candidates[0]


def stage_grid_specs(
    pairs: Sequence[object], records: Sequence[PaperArtifactRecord] = (), atoms_dir: Path | None = None
) -> tuple[GridSpec, ...]:
    """Return the fixed 21-grid controller allocation.

    Passing an empty pair sequence exposes the inventory without materializing
    cells, which keeps controller-level count checks independent of Task 3 I/O.
    """

    if not pairs:
        return tuple(
            GridSpec(f"G-{stage}-{index + 1:02d}-{layout}", stage, f"{stage}-run", layout, appendix,
                     tuple(GridCell(f"placeholder-{cell // 2}", "baseline" if cell % 2 == 0 else "ours", "placeholder", _placeholder_record(stage, cell), Path("placeholder.png")) for cell in range(_shape(layout)[0] * _shape(layout)[1])),
                     "Controller inventory placeholder")
            for stage, allocations in _GRID_ALLOCATION.items()
            for index, (layout, appendix) in enumerate(allocations)
        )
    if atoms_dir is None:
        raise ValueError("atoms_dir is required to materialize stage grids")
    specs: list[GridSpec] = []
    for stage, allocations in _GRID_ALLOCATION.items():
        stage_pairs = [pair for pair in pairs if pair.stage == stage]
        if not stage_pairs:
            raise ValueError(f"no Task 3 pairs for {stage}")
        for ordinal, (layout, appendix) in enumerate(allocations, start=1):
            count = _shape(layout)[0] * _shape(layout)[1]
            cells: list[GridCell] = []
            for pair_index in range(count // 2):
                pair = stage_pairs[pair_index % len(stage_pairs)]
                repeat = pair_index // len(stage_pairs)
                for role in ("baseline", "ours"):
                    record = _record_for(records, pair, role)
                    cells.append(GridCell(
                        pair_key=f"{pair.case_id}|{pair.camera}|{pair.timepoint}|repeat-{repeat}",
                        role=role, timepoint=str(pair.timepoint), record=record,
                        image_path=Path(atoms_dir) / record.output_filenames["png"],
                    ))
            if count % 2:
                pair = stage_pairs[(count // 2) % len(stage_pairs)]
                record = _record_for(records, pair, "baseline")
                cells.append(GridCell(
                    pair_key=f"{pair.case_id}|{pair.camera}|{pair.timepoint}|context",
                    role="baseline", timepoint=str(pair.timepoint), record=record,
                    image_path=Path(atoms_dir) / record.output_filenames["png"],
                ))
            specs.append(GridSpec(
                artifact_id=f"G-{stage}-{ordinal:02d}-{layout}", stage=stage, source_run=stage_pairs[0].source_run,
                layout=layout, appendix=appendix, cells=tuple(cells),
                claim="Matched, manifest-backed baseline and synchronized visual states.",
            ))
    if len(specs) != 21:
        raise AssertionError("Task 4 must own exactly 21 stage grids")
    return tuple(specs)


def _placeholder_record(stage: str, index: int) -> PaperArtifactRecord:
    """Private inventory-only record; it is never accepted by the compositor."""
    return PaperArtifactRecord(
        artifact_id=f"inventory-{stage}-{index}", artifact_kind=ArtifactKind.FRAME, stage=stage,
        claim="inventory", source_run=f"{stage}-run", source_hashes={"contrast_receipt": "a" * 64},
        evidence_level=EvidenceLevel.FORMAL, method="ours_full", baseline="primary_baseline", case_id="placeholder",
        camera="placeholder", selection_policy="inventory", comparison_metrics={}, width=1600, height=1600,
        recommended_title="inventory", recommended_subtitle="inventory", caption_draft="inventory",
        output_filenames={"png": "placeholder.png"}, output_hashes={"png": "b" * 64},
    )


__all__ = ["APPENDIX_LAYOUTS", "GRID_CONTENT_SIZE", "GRID_FOOTER_HEIGHT", "GridCell", "GridSpec", "MAIN_LAYOUTS", "compose_image_grid", "grid_footer_color", "grid_footer_label", "stage_grid_specs"]
