"""High-fidelity, recorded-trajectory visual evidence from PiPER Robust V2."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Iterable, Mapping, Sequence

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np

from .paper_style import METHOD_COLORS, publication_rc_context, save_square_quantitative


METHOD_BGR = {
    "real": (39, 39, 39),
    "baseline": tuple(int(METHOD_COLORS["primary_baseline"][index : index + 2], 16) for index in (5, 3, 1)),
    "ours": tuple(int(METHOD_COLORS["ours_full"][index : index + 2], 16) for index in (5, 3, 1)),
}
ALLOWED_LAYOUTS = frozenset({"3x2", "4x2", "3x3", "4x3", "7x2", "8x2", "9x2", "8x3"})
PANEL_NAMES = ("real_head", "gs_head", "real_extra", "gs_extra")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extract_replay_panels(frame_bgr: np.ndarray) -> dict[str, np.ndarray]:
    """Remove the renderer header and return the four native 640x480 panels."""

    if frame_bgr.shape != (512, 2560, 3) or frame_bgr.dtype != np.uint8:
        raise ValueError("Robust V2 replay frame must be uint8 BGR 2560x512")
    content = frame_bgr[32:512]
    return {
        name: content[:, index * 640 : (index + 1) * 640].copy()
        for index, name in enumerate(PANEL_NAMES)
    }


def _signature_real(image: np.ndarray, size: tuple[int, int] = (160, 120)) -> np.ndarray:
    resized = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    return cv2.Canny(cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY), 50, 140)


def _signature_gaussian(
    image: np.ndarray, size: tuple[int, int] = (160, 120)
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    resized = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    mask = np.any(resized < 245, axis=2).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    boundary = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    edge = cv2.Canny(cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY), 40, 120)
    return mask, cv2.bitwise_or(boundary, edge), resized


def _edge_distance_score(real_edge: np.ndarray, gs_mask: np.ndarray, gs_edge: np.ndarray) -> float:
    region = cv2.dilate(gs_mask, np.ones((7, 7), np.uint8))
    observed = cv2.bitwise_and(real_edge, region)
    distance = cv2.distanceTransform((observed == 0).astype(np.uint8), cv2.DIST_L2, 3)
    values = distance[gs_edge > 0]
    if values.size < 20:
        return float("inf")
    return float(np.mean(np.clip(values, 0.0, 12.0)))


def retrieve_index(
    real_bgr: np.ndarray,
    gaussian_candidates_bgr: Sequence[np.ndarray],
    *,
    candidate_indices: Sequence[int],
) -> tuple[int, dict[int, float]]:
    if len(gaussian_candidates_bgr) != len(candidate_indices) or not candidate_indices:
        raise ValueError("candidate images and indices must be nonempty and aligned")
    real_edge = _signature_real(real_bgr)
    scores = {}
    for index, image in zip(candidate_indices, gaussian_candidates_bgr, strict=True):
        mask, edge, _ = _signature_gaussian(image)
        scores[int(index)] = _edge_distance_score(real_edge, mask, edge)
    selected = min(scores, key=lambda index: (scores[index], index))
    return selected, scores


@dataclass(frozen=True)
class AlignmentCase:
    camera: str
    true_index: int
    baseline_index: int
    ours_index: int
    baseline_qmae: float
    ours_qmae: float
    pixel_mae: float
    score_gain: float

    @property
    def qmae_reduction(self) -> float:
        return self.baseline_qmae - self.ours_qmae

    @property
    def exact_recovery(self) -> bool:
        return self.ours_index == self.true_index


@dataclass(frozen=True)
class GridGeometry:
    layout: str
    width: int
    height: int
    image_width: int
    image_height: int
    border: int
    footer: int
    gap: int


@dataclass(frozen=True)
class ReplaySignatures:
    real_edges: Mapping[str, tuple[np.ndarray, ...]]
    gs_masks: Mapping[str, tuple[np.ndarray, ...]]
    gs_edges: Mapping[str, tuple[np.ndarray, ...]]
    gs_thumbnails: Mapping[str, tuple[np.ndarray, ...]]
    frame_count: int


def decode_replay_signatures(video_path: Path) -> ReplaySignatures:
    real_edges = {"head": [], "extra": []}
    gs_masks = {"head": [], "extra": []}
    gs_edges = {"head": [], "extra": []}
    thumbnails = {"head": [], "extra": []}
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Cannot decode Robust V2 replay: {video_path}")
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            panels = extract_replay_panels(frame)
            for camera in ("head", "extra"):
                real_edges[camera].append(_signature_real(panels[f"real_{camera}"]))
                mask, edge, thumb = _signature_gaussian(panels[f"gs_{camera}"])
                gs_masks[camera].append(mask)
                gs_edges[camera].append(edge)
                thumbnails[camera].append(thumb)
    finally:
        capture.release()
    count = len(real_edges["head"])
    if count < 1 or any(len(values) != count for group in (real_edges, gs_masks, gs_edges, thumbnails) for values in group.values()):
        raise ValueError("Robust V2 replay panel inventory is inconsistent")
    return ReplaySignatures(
        real_edges={key: tuple(values) for key, values in real_edges.items()},
        gs_masks={key: tuple(values) for key, values in gs_masks.items()},
        gs_edges={key: tuple(values) for key, values in gs_edges.items()},
        gs_thumbnails={key: tuple(values) for key, values in thumbnails.items()},
        frame_count=count,
    )


def _nearest_indices(reference: np.ndarray, query: np.ndarray) -> np.ndarray:
    positions = np.searchsorted(query, reference)
    right = np.clip(positions, 0, len(query) - 1)
    left = np.clip(positions - 1, 0, len(query) - 1)
    use_left = np.abs(query[left] - reference) <= np.abs(query[right] - reference)
    return np.where(use_left, left, right)


def load_replay_qpos(hdf5_path: Path, expected_count: int, max_pair_gap_ms: float = 10.0) -> np.ndarray:
    with h5py.File(hdf5_path, "r") as handle:
        groups = {camera: handle[f"cam_{camera}"] for camera in ("head", "extra")}
        valid = {
            camera: np.flatnonzero((group["valid"][:] > 0) & (group["robot_state_valid"][:] > 0))
            for camera, group in groups.items()
        }
        head_indices = valid["head"]
        extra_indices = valid["extra"]
        head_time = groups["head"]["host_monotonic_timestamp_ns"][head_indices].astype(np.int64)
        extra_time = groups["extra"]["host_monotonic_timestamp_ns"][extra_indices].astype(np.int64)
        nearest = _nearest_indices(head_time, extra_time)
        pair_gap = np.abs(extra_time[nearest] - head_time) / 1e6
        paired_head = head_indices[pair_gap <= max_pair_gap_ms]
        arm = np.asarray(groups["head"]["measured_joint_interpolated"][paired_head], dtype=np.float64)
        gripper = np.clip(np.asarray(groups["head"]["measured_gripper_interpolated"][paired_head], dtype=np.float64), 0.0, 1.0) * 0.05
    qpos = np.column_stack((arm, gripper, -gripper))
    if qpos.shape != (expected_count, 8) or not np.isfinite(qpos).all():
        raise ValueError(f"Expected {expected_count} finite 8D qpos rows, got {qpos.shape}")
    return qpos


def evaluate_alignment(
    signatures: ReplaySignatures,
    qpos: np.ndarray,
    *,
    lag: int = 3,
    window_radius: int = 4,
) -> tuple[AlignmentCase, ...]:
    if lag <= 0 or window_radius < lag or qpos.shape != (signatures.frame_count, 8):
        raise ValueError("Alignment evaluation needs positive recoverable lag and matching qpos")
    cases = []
    margin = lag + window_radius + 1
    for camera in ("head", "extra"):
        for true_index in range(margin, signatures.frame_count - margin):
            baseline_index = true_index - lag
            candidates = range(baseline_index - window_radius, baseline_index + window_radius + 1)
            scores = {
                index: _edge_distance_score(
                    signatures.real_edges[camera][true_index],
                    signatures.gs_masks[camera][index],
                    signatures.gs_edges[camera][index],
                )
                for index in candidates
            }
            ours_index = min(scores, key=lambda index: (scores[index], index))
            baseline_qmae = float(np.mean(np.abs(qpos[baseline_index] - qpos[true_index])))
            ours_qmae = float(np.mean(np.abs(qpos[ours_index] - qpos[true_index])))
            baseline_thumb = signatures.gs_thumbnails[camera][baseline_index].astype(np.float32)
            ours_thumb = signatures.gs_thumbnails[camera][ours_index].astype(np.float32)
            mask = np.logical_or(
                signatures.gs_masks[camera][baseline_index] > 0,
                signatures.gs_masks[camera][ours_index] > 0,
            )
            pixel_mae = float(np.abs(baseline_thumb - ours_thumb)[mask].mean()) if np.any(mask) else 0.0
            cases.append(
                AlignmentCase(
                    camera=camera,
                    true_index=true_index,
                    baseline_index=baseline_index,
                    ours_index=ours_index,
                    baseline_qmae=baseline_qmae,
                    ours_qmae=ours_qmae,
                    pixel_mae=pixel_mae,
                    score_gain=float(scores[baseline_index] - scores[ours_index]),
                )
            )
    return tuple(cases)


def select_recommended_cases(
    cases: Sequence[AlignmentCase],
    *,
    count: int,
    minimum_pixel_mae: float,
    minimum_separation: int = 8,
) -> tuple[AlignmentCase, ...]:
    eligible = [
        case
        for case in cases
        if case.qmae_reduction > 0.0
        and case.pixel_mae >= minimum_pixel_mae
        and case.ours_index != case.baseline_index
    ]
    eligible.sort(key=lambda case: (-case.qmae_reduction, -case.pixel_mae, case.true_index))
    selected: list[AlignmentCase] = []
    for case in eligible:
        if all(case.camera != prior.camera or abs(case.true_index - prior.true_index) >= minimum_separation for prior in selected):
            selected.append(case)
        if len(selected) == count:
            break
    return tuple(selected)


def load_replay_images(
    video_path: Path, cases: Sequence[AlignmentCase]
) -> dict[tuple[str, int], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    required = {case.true_index for case in cases} | {case.baseline_index for case in cases} | {case.ours_index for case in cases}
    raw: dict[int, dict[str, np.ndarray]] = {}
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Cannot decode Robust V2 replay: {video_path}")
    try:
        index = 0
        while required:
            ok, frame = capture.read()
            if not ok:
                break
            if index in required:
                raw[index] = extract_replay_panels(frame)
                required.remove(index)
            index += 1
    finally:
        capture.release()
    if required:
        raise ValueError(f"Replay is missing requested frames: {sorted(required)}")
    return {
        (case.camera, case.true_index): (
            raw[case.true_index][f"real_{case.camera}"],
            raw[case.baseline_index][f"gs_{case.camera}"],
            raw[case.ours_index][f"gs_{case.camera}"],
        )
        for case in cases
    }


def _fit_image(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    width, height = size
    if image.shape[:2] == (height, width):
        return image
    interpolation = cv2.INTER_AREA if image.shape[1] > width else cv2.INTER_LANCZOS4
    return cv2.resize(image, (width, height), interpolation=interpolation)


def compose_comparison_grid(
    cases: Sequence[AlignmentCase],
    images: Mapping[int | tuple[str, int], tuple[np.ndarray, np.ndarray, np.ndarray]],
    destination: Path,
    *,
    rows: tuple[str, ...],
    cell_size: tuple[int, int] = (640, 480),
) -> GridGeometry:
    if rows not in (("baseline", "ours"), ("real", "baseline", "ours")):
        raise ValueError("Rows must be paired baseline/ours or real/baseline/ours")
    columns = len(cases)
    if columns not in {3, 4, 7, 8, 9}:
        raise ValueError("Grid column count is outside the paper layout contract")
    layout = f"{columns}x{len(rows)}"
    if layout not in ALLOWED_LAYOUTS:
        raise ValueError("Grid layout is outside the paper layout contract")
    image_width, image_height = cell_size
    border, footer, gap = 2, 28, 4
    cell_width = image_width + 2 * border
    row_stride = image_height + 2 * border + footer
    width = columns * (cell_width + gap)
    height = len(rows) * row_stride
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    role_index = {"real": 0, "baseline": 1, "ours": 2}
    for column, case in enumerate(cases):
        key = (case.camera, case.true_index) if (case.camera, case.true_index) in images else case.true_index
        triplet = images[key]
        for row, role in enumerate(rows):
            x = column * (cell_width + gap)
            y = row * row_stride
            color = METHOD_BGR[role]
            canvas[y : y + image_height + 2 * border, x : x + cell_width] = color
            image = _fit_image(triplet[role_index[role]], cell_size)
            canvas[y + border : y + border + image_height, x + border : x + border + image_width] = image
            source_index = {
                "real": case.true_index,
                "baseline": case.baseline_index,
                "ours": case.ours_index,
            }[role]
            label = f"{role} | {case.camera} | frame {source_index}"
            cv2.putText(
                canvas,
                label,
                (x + 4, y + image_height + 2 * border + 19),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                color,
                1,
                cv2.LINE_AA,
            )
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), canvas, [cv2.IMWRITE_PNG_COMPRESSION, 6]):
        raise RuntimeError(f"Could not write {destination}")
    return GridGeometry(layout, width, height, image_width, image_height, border, footer, gap)


def export_grid_vectors(png_path: Path) -> tuple[Path, Path]:
    bgr = cv2.imread(str(png_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Cannot decode grid: {png_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    with publication_rc_context():
        figure = plt.figure(figsize=(rgb.shape[1] / 160.0, rgb.shape[0] / 160.0), dpi=160)
        axis = figure.add_axes((0, 0, 1, 1))
        axis.imshow(rgb)
        axis.set_axis_off()
        pdf_path = Path(png_path).with_suffix(".pdf")
        svg_path = Path(png_path).with_suffix(".svg")
        figure.savefig(pdf_path, format="pdf", dpi=160, pad_inches=0)
        figure.savefig(svg_path, format="svg", dpi=160, pad_inches=0)
        plt.close(figure)
    return pdf_path, svg_path


def write_atomic_frames(
    cases: Sequence[AlignmentCase],
    images: Mapping[tuple[str, int], tuple[np.ndarray, np.ndarray, np.ndarray]],
    output_dir: Path,
) -> list[Path]:
    paths = []
    roles = ("real", "baseline", "ours")
    for case in cases:
        triplet = images[(case.camera, case.true_index)]
        for role, image in zip(roles, triplet, strict=True):
            path = Path(output_dir) / f"F-RV2-{case.camera}-{case.true_index:04d}-{role}.png"
            if not cv2.imwrite(str(path), image, [cv2.IMWRITE_PNG_COMPRESSION, 6]):
                raise RuntimeError(f"Could not write {path}")
            paths.append(path)
    return paths


def _video_canvas(panels: Sequence[tuple[str, str, np.ndarray]], frame_text: str) -> np.ndarray:
    canvas = np.full((1080, 1920, 3), 255, dtype=np.uint8)
    count = len(panels)
    gap = 12
    panel_width = (1920 - (count + 1) * gap) // count
    panel_height = round(panel_width * 3 / 4)
    if panel_height > 840:
        panel_height = 840
        panel_width = round(panel_height * 4 / 3)
    top = (1080 - panel_height) // 2
    start_x = (1920 - count * panel_width - (count - 1) * gap) // 2
    for index, (role, label, image) in enumerate(panels):
        x = start_x + index * (panel_width + gap)
        color = METHOD_BGR[role]
        resized = _fit_image(image, (panel_width, panel_height))
        canvas[top - 2 : top + panel_height + 2, x - 2 : x + panel_width + 2] = color
        canvas[top : top + panel_height, x : x + panel_width] = resized
        cv2.putText(canvas, label, (x, top - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.66, color, 2, cv2.LINE_AA)
    cv2.putText(canvas, frame_text, (18, 1050), cv2.FONT_HERSHEY_SIMPLEX, 0.58, METHOD_BGR["real"], 1, cv2.LINE_AA)
    return canvas


def _transcode_h264(raw_path: Path, output_path: Path) -> None:
    import imageio_ffmpeg

    command = [
        imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(raw_path),
        "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(output_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr}")
    raw_path.unlink()


def write_case_video(
    cases: Sequence[AlignmentCase],
    images: Mapping[tuple[str, int], tuple[np.ndarray, np.ndarray, np.ndarray]],
    output_path: Path,
    *,
    mode: str,
    fps: float = 12.0,
    hold_frames: int = 18,
    baseline_label: str = "Delayed GS",
    ours_label: str = "Synchronized GS",
) -> Path:
    if mode not in {"baseline", "ours", "comparison"}:
        raise ValueError("Video mode must be baseline, ours, or comparison")
    raw_path = Path(output_path).with_suffix(".tmp.mp4")
    writer = cv2.VideoWriter(str(raw_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (1920, 1080))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {raw_path}")
    try:
        for case in cases:
            real, baseline, ours = images[(case.camera, case.true_index)]
            if mode == "baseline":
                panels = (("real", "Real observation", real), ("baseline", baseline_label, baseline))
            elif mode == "ours":
                panels = (("real", "Real observation", real), ("ours", ours_label, ours))
            else:
                panels = (("real", "Real", real), ("baseline", baseline_label, baseline), ("ours", ours_label, ours))
            canvas = _video_canvas(panels, f"{case.camera} frame {case.true_index} | qMAE {case.baseline_qmae:.4f} -> {case.ours_qmae:.4f} rad")
            for _ in range(hold_frames):
                writer.write(canvas)
    finally:
        writer.release()
    _transcode_h264(raw_path, Path(output_path))
    return Path(output_path)


def write_quantitative_summary(cases: Sequence[AlignmentCase], output_stem: Path) -> tuple[Path, Path, Path]:
    cameras = ("head", "extra")
    baseline = [np.mean([case.baseline_qmae for case in cases if case.camera == camera]) for camera in cameras]
    ours = [np.mean([case.ours_qmae for case in cases if case.camera == camera]) for camera in cameras]
    with publication_rc_context():
        figure, axis = plt.subplots()
        figure.set_size_inches(89.0 / 25.4, 89.0 / 25.4, forward=True)
        positions = np.arange(len(cameras), dtype=float)
        width = 0.34
        axis.bar(positions - width / 2, baseline, width, color=METHOD_COLORS["primary_baseline"], label="Delayed GS")
        axis.bar(positions + width / 2, ours, width, color=METHOD_COLORS["ours_full"], label="Synchronized GS")
        axis.set_xticks(positions, ("Head", "Extra"))
        axis.set_ylabel("qMAE (rad)")
        axis.legend(loc="upper right")
        axis.set_ylim(bottom=0.0)
        figure.subplots_adjust(left=0.23, right=0.97, bottom=0.18, top=0.96)
        paths = save_square_quantitative(figure, Path(output_stem))
        plt.close(figure)
    return paths


def write_results(cases: Sequence[AlignmentCase], run_dir: Path, metadata: Mapping[str, object]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(cases[0]).keys()) + ["qmae_reduction", "exact_recovery"])
        writer.writeheader()
        for case in cases:
            writer.writerow({**asdict(case), "qmae_reduction": case.qmae_reduction, "exact_recovery": int(case.exact_recovery)})
    metrics = {}
    for camera in ("head", "extra"):
        rows = [case for case in cases if case.camera == camera]
        metrics[camera] = {
            "count": len(rows),
            "baseline_qmae_rad": float(np.mean([row.baseline_qmae for row in rows])),
            "ours_qmae_rad": float(np.mean([row.ours_qmae for row in rows])),
            "qmae_reduction_rad": float(np.mean([row.qmae_reduction for row in rows])),
            "improved_fraction": float(np.mean([row.qmae_reduction > 0 for row in rows])),
            "exact_index_fraction": float(np.mean([row.exact_recovery for row in rows])),
            "within_one_frame_fraction": float(np.mean([abs(row.ours_index - row.true_index) <= 1 for row in rows])),
        }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    (run_dir / "manifest.json").write_text(json.dumps(dict(metadata), indent=2) + "\n", encoding="utf-8")


__all__ = [
    "ALLOWED_LAYOUTS", "AlignmentCase", "GridGeometry", "METHOD_BGR", "ReplaySignatures",
    "compose_comparison_grid", "decode_replay_signatures", "evaluate_alignment", "export_grid_vectors",
    "extract_replay_panels", "load_replay_images", "load_replay_qpos", "retrieve_index",
    "select_recommended_cases", "sha256_file", "write_atomic_frames", "write_case_video",
    "write_quantitative_summary", "write_results",
]
