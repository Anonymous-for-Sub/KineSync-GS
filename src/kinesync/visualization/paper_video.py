"""Matched baseline/ours H.264 videos for Task 4 evidence."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Mapping, Sequence

import cv2
import imageio_ffmpeg
import numpy as np

from .paper_contracts import ArtifactKind, EvidenceLevel, PaperArtifactRecord
from .paper_style import METHOD_COLORS


VIDEO_WIDTH, VIDEO_HEIGHT = 1920, 1080
_VIDEO_MODES = frozenset({"baseline", "ours", "side_by_side"})
_STAGE_COUNTS = {"RT2": 3, "RT3": 3, "RT6": 3, "RT7": 3, "RT8": 6, "RT9": 6}


def _rgb(method: str) -> tuple[int, int, int]:
    """Return the exact semantic paper color in RGB channel order."""
    value = METHOD_COLORS[method]
    return tuple(int(value[index : index + 2], 16) for index in (1, 3, 5))


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


@dataclass(frozen=True)
class VideoFrame:
    case_id: str
    camera: str
    timepoint: str
    baseline: np.ndarray
    ours: np.ndarray
    roi: np.ndarray
    source_hashes: Mapping[str, str]

    def validate(self, camera: str) -> None:
        if self.camera != camera:
            raise ValueError("video frame camera mismatches the shared schedule")
        if self.baseline.shape != self.ours.shape or self.baseline.ndim != 3 or self.baseline.shape[2] != 3:
            raise ValueError("video frame baseline and ours must have equal RGB geometry")
        if self.roi.shape != self.baseline.shape[:2] or not self.roi.any():
            raise ValueError("video frame ROI must match the frame geometry")
        if self.baseline.dtype != np.uint8 or self.ours.dtype != np.uint8:
            raise ValueError("video frame inputs must be uint8 RGB")
        if np.array_equal(self.baseline, self.ours):
            raise ValueError("video frame baseline and ours are identical")
        if float(self.baseline.std()) <= 1.0 or float(self.ours.std()) <= 1.0:
            raise ValueError("video frame is blank")
        if not self.source_hashes or any(len(value) != 64 for value in self.source_hashes.values()):
            raise ValueError("video frame lacks full source hashes")


@dataclass(frozen=True)
class VideoSpec:
    artifact_id: str
    stage: str
    source_run: str
    camera: str
    evidence_level: EvidenceLevel | str
    mode: str
    frames: tuple[VideoFrame, ...]
    sequence_label: str
    fps: int = 12
    writable: bool = True

    def validate(self) -> None:
        if self.mode not in _VIDEO_MODES:
            raise ValueError("video mode must be baseline, ours, or side_by_side")
        if self.fps not in {12, 24}:
            raise ValueError("video FPS must be 12 or 24")
        if len(self.frames) < 3:
            raise ValueError("video schedule must contain at least three frames")
        if not self.artifact_id or not self.stage or not self.source_run or not self.camera:
            raise ValueError("video metadata must be nonempty")
        if not isinstance(self.writable, bool):
            raise ValueError("video writable flag must be boolean")
        try:
            EvidenceLevel(self.evidence_level)
        except ValueError as error:
            raise ValueError("video evidence level is invalid") from error
        projected = self.stage in {"RT2", "RT3", "RT6", "RT7"}
        if projected and "frozen projected-state case sequence" not in self.sequence_label:
            raise ValueError("projected-state sequences require the frozen projected-state case sequence label")
        if "scripted" in self.sequence_label.lower() and "illustrative_scripted_trajectory" not in self.sequence_label:
            raise ValueError("scripted video labels must declare illustrative_scripted_trajectory")
        for frame in self.frames:
            frame.validate(self.camera)

    @property
    def schedule_hash(self) -> str:
        payload = [
            {"case_id": frame.case_id, "camera": frame.camera, "timepoint": frame.timepoint,
             "baseline": _sha(frame.baseline.tobytes()), "ours": _sha(frame.ours.tobytes()),
             "roi": _sha(np.ascontiguousarray(frame.roi.astype(np.uint8)).tobytes()),
             "sources": dict(sorted(frame.source_hashes.items()))}
            for frame in self.frames
        ]
        return _sha(_canonical(payload).encode("ascii"))


@dataclass(frozen=True)
class VideoContrastReceipt:
    baseline_sha256: str
    ours_sha256: str
    schedule_sha256: str
    differing_frame_fraction: float
    width: int
    height: int
    fps: float
    frame_count: int
    sha256: str


@dataclass(frozen=True)
class VideoInventoryItem:
    """Controller planning metadata only; it cannot be passed to the encoder."""

    artifact_id: str
    stage: str
    camera: str
    mode: str
    writable: bool = False


def _place(
    image: np.ndarray, roi: np.ndarray, width: int, height: int, label: str, color: tuple[int, int, int]
) -> tuple[np.ndarray, np.ndarray]:
    canvas = np.zeros((height, width, 3), np.uint8)
    footer = max(42, height // 18)
    usable_height = height - footer
    scale = min(width / image.shape[1], usable_height / image.shape[0])
    resized = cv2.resize(image, (max(1, int(round(image.shape[1] * scale))), max(1, int(round(image.shape[0] * scale)))), interpolation=cv2.INTER_AREA)
    x = (width - resized.shape[1]) // 2
    y = (usable_height - resized.shape[0]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    mapped_roi = np.zeros((height, width), np.uint8)
    roi_resized = cv2.resize(roi.astype(np.uint8), (resized.shape[1], resized.shape[0]), interpolation=cv2.INTER_NEAREST)
    mapped_roi[y : y + resized.shape[0], x : x + resized.shape[1]] = roi_resized
    cv2.rectangle(canvas, (0, usable_height), (width, height), (250, 250, 250), -1)
    cv2.putText(canvas, label, (18, height - max(14, footer // 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
    return canvas, mapped_roi.astype(bool)


def _render(spec: VideoSpec, frame: VideoFrame) -> tuple[np.ndarray, np.ndarray]:
    if spec.mode == "baseline":
        return _place(frame.baseline, frame.roi, VIDEO_WIDTH, VIDEO_HEIGHT, f"baseline | {frame.camera} | t={frame.timepoint}", _rgb("primary_baseline"))
    if spec.mode == "ours":
        return _place(frame.ours, frame.roi, VIDEO_WIDTH, VIDEO_HEIGHT, f"ours | {frame.camera} | t={frame.timepoint}", _rgb("ours_full"))
    left, left_roi = _place(frame.baseline, frame.roi, VIDEO_WIDTH // 2, VIDEO_HEIGHT, f"baseline | t={frame.timepoint}", _rgb("primary_baseline"))
    right, right_roi = _place(frame.ours, frame.roi, VIDEO_WIDTH // 2, VIDEO_HEIGHT, f"ours | t={frame.timepoint}", _rgb("ours_full"))
    return np.concatenate((left, right), axis=1), np.concatenate((left_roi, right_roi), axis=1)


def _encode(frames: Sequence[np.ndarray], destination: Path, fps: int) -> None:
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    command = [ffmpeg, "-y", "-f", "rawvideo", "-vcodec", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{VIDEO_WIDTH}x{VIDEO_HEIGHT}", "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(destination)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert process.stdin is not None
    try:
        for frame in frames:
            process.stdin.write(np.ascontiguousarray(frame).tobytes())
        process.stdin.close()
        process.stdin = None
        _, error = process.communicate(timeout=180)
    except BaseException:
        process.kill()
        process.wait()
        raise
    if process.returncode:
        raise RuntimeError(f"ffmpeg H.264 export failed: {error.decode('utf-8', errors='replace')[-600:]}")


def _probe(path: Path) -> dict[str, object]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError("video is corrupt or undecodable")
    width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps, count = float(capture.get(cv2.CAP_PROP_FPS)), int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    decoded: list[np.ndarray] = []
    wanted = {0, max(0, count // 2), max(0, count - 1)}
    index = 0
    while index <= max(wanted):
        ok, frame = capture.read()
        if not ok:
            break
        if index in wanted:
            decoded.append(frame)
        index += 1
    capture.release()
    if (width, height) != (VIDEO_WIDTH, VIDEO_HEIGHT) or count < 3 or len(decoded) != 3:
        raise ValueError("video dimensions or decodable first/middle/final frames are invalid")
    if any(float(frame.std()) <= 1.0 for frame in decoded):
        raise ValueError("video contains a blank verification frame")
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    diagnostic = subprocess.run([ffmpeg, "-hide_banner", "-i", str(path)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, text=True).stderr.lower()
    if "h264" not in diagnostic or "yuv420p" not in diagnostic:
        raise ValueError("video must be H.264/yuv420p")
    return {"width": width, "height": height, "fps": fps, "count": count}


def _sidecar(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".json")


def _encode_roi(mask: np.ndarray) -> str:
    ok, payload = cv2.imencode(".png", mask.astype(np.uint8) * 255, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    if not ok:
        raise RuntimeError("could not encode mapped video ROI")
    return payload.tobytes().hex()


def _decode_roi(payload: str) -> np.ndarray:
    data = np.frombuffer(bytes.fromhex(payload), dtype=np.uint8)
    mask = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if mask is None or mask.shape != (VIDEO_HEIGHT, VIDEO_WIDTH):
        raise ValueError("video sidecar has invalid mapped ROI")
    return mask > 0


def write_method_video(spec: VideoSpec, destination: Path) -> PaperArtifactRecord:
    """Encode one mode from a shared baseline/ours frame schedule."""

    if not spec.writable:
        raise ValueError("non-writable inventory metadata cannot be rendered as a video")
    spec.validate()
    destination = Path(destination)
    if destination.suffix != ".mp4" or "/" in destination.name or "\\" in destination.name:
        raise ValueError("video destination must be a flat MP4 filename")
    destination.parent.mkdir(parents=True, exist_ok=True)
    rendered_pairs = tuple(_render(spec, frame) for frame in spec.frames)
    _encode(tuple(item[0] for item in rendered_pairs), destination, spec.fps)
    probe = _probe(destination)
    mp4_sha256 = _sha_file(destination)
    metadata = {"artifact_id": spec.artifact_id, "stage": spec.stage, "camera": spec.camera, "mode": spec.mode,
                "schedule_sha256": spec.schedule_hash, "fps": spec.fps, "frame_count": len(spec.frames),
                "sequence_label": spec.sequence_label, "width": VIDEO_WIDTH, "height": VIDEO_HEIGHT,
                "mp4_sha256": mp4_sha256,
                "mapped_roi_png_hex": [_encode_roi(item[1]) for item in rendered_pairs]}
    _sidecar(destination).write_text(_canonical(metadata), encoding="ascii")
    hashes = {"shared_schedule": spec.schedule_hash}
    for index, frame in enumerate(spec.frames):
        for key, value in sorted(frame.source_hashes.items()):
            hashes[f"frame_{index:02d}_{key}"] = value
    return PaperArtifactRecord(
        artifact_id=spec.artifact_id, artifact_kind=ArtifactKind.VIDEO, stage=spec.stage,
        claim="Matched baseline and synchronized visual sequence.", source_run=spec.source_run,
        source_hashes=hashes, evidence_level=spec.evidence_level, method="ours_full", baseline="primary_baseline",
        case_id=f"{spec.stage}-{spec.camera}-matched-sequence", camera=spec.camera,
        selection_policy="one_shared_schedule_without_method_resampling", comparison_metrics={},
        width=VIDEO_WIDTH, height=VIDEO_HEIGHT, recommended_title="Matched visual sequence",
        recommended_subtitle=spec.sequence_label, caption_draft="Matched baseline/ours evidence video.",
        output_filenames={"mp4": destination.name}, output_hashes={"mp4": mp4_sha256},
    )


def verify_video_pair(baseline: Path, ours: Path) -> VideoContrastReceipt:
    """Verify shared scheduling and decoded visual separation of a video pair."""

    baseline, ours = Path(baseline), Path(ours)
    base_meta = json.loads(_sidecar(baseline).read_text(encoding="ascii"))
    ours_meta = json.loads(_sidecar(ours).read_text(encoding="ascii"))
    for path, metadata in ((baseline, base_meta), (ours, ours_meta)):
        if metadata.get("mp4_sha256") != _sha_file(path):
            raise ValueError("sidecar MP4 hash mismatch")
    for key in ("schedule_sha256", "camera", "fps", "frame_count", "width", "height"):
        if base_meta.get(key) != ours_meta.get(key):
            raise ValueError(f"video pair mismatches shared {key}")
    if base_meta.get("mode") != "baseline" or ours_meta.get("mode") != "ours":
        raise ValueError("video receipt requires baseline and ours mode files")
    base_probe, ours_probe = _probe(baseline), _probe(ours)
    if base_probe != ours_probe:
        raise ValueError("video pair probe metadata differs")
    left, right = cv2.VideoCapture(str(baseline)), cv2.VideoCapture(str(ours))
    if len(base_meta.get("mapped_roi_png_hex", [])) != int(base_probe["count"]):
        raise ValueError("baseline video sidecar lacks one mapped ROI per frame")
    if len(ours_meta.get("mapped_roi_png_hex", [])) != int(ours_probe["count"]):
        raise ValueError("ours video sidecar lacks one mapped ROI per frame")
    differences: list[float] = []
    roi_hashes: list[str] = []
    frame_index = 0
    while True:
        ok_left, frame_left = left.read()
        ok_right, frame_right = right.read()
        if not ok_left and not ok_right:
            break
        if ok_left != ok_right:
            raise ValueError("video pair has different decoded frame counts")
        roi_left = _decode_roi(base_meta["mapped_roi_png_hex"][frame_index])
        roi_right = _decode_roi(ours_meta["mapped_roi_png_hex"][frame_index])
        if not np.array_equal(roi_left, roi_right):
            raise ValueError("video pair mapped ROI differs despite shared scheduling")
        if not roi_left.any():
            raise ValueError("video pair mapped ROI is empty")
        delta = np.abs(frame_left.astype(np.int16) - frame_right.astype(np.int16))
        differences.append(float(delta[roi_left].mean()) / 255.0)
        roi_hashes.append(_sha(roi_left.astype(np.uint8).tobytes()))
        frame_index += 1
    left.release(); right.release()
    fraction = sum(value > 0.01 for value in differences) / max(1, len(differences))
    if fraction < 0.20:
        raise ValueError("video pair has ROI difference in fewer than 20 percent of frames")
    base_hash, ours_hash = _sha_file(baseline), _sha_file(ours)
    payload = {"baseline": base_hash, "ours": ours_hash, "schedule": base_meta["schedule_sha256"], "fraction": fraction, "probe": base_probe, "roi_hashes": roi_hashes}
    return VideoContrastReceipt(base_hash, ours_hash, base_meta["schedule_sha256"], fraction, VIDEO_WIDTH, VIDEO_HEIGHT, float(base_probe["fps"]), int(base_probe["count"]), _sha(_canonical(payload).encode("ascii")))


def _frame_from_pair(pair: object) -> VideoFrame:
    from .paper_frames import _materialize
    baseline, ours, roi, _ = _materialize(pair)
    receipt = pair.receipt
    if receipt is None:
        raise ValueError("Task 4 requires a receipt-validated Task 3 pair")
    hashes = {"contrast_receipt": receipt.sha256, "baseline_crop": receipt.baseline_crop_sha256, "ours_crop": receipt.ours_crop_sha256,
              "frozen_result": pair.result_sha256}
    return VideoFrame(pair.case_id, pair.camera, str(pair.timepoint), baseline, ours, roi, hashes)


def stage_video_specs(pairs: Sequence[object]) -> tuple[VideoSpec, ...]:
    """Build the controller's 24 non-workflow video specs from Task 3 pairs."""

    if not pairs:
        raise ValueError("stage_video_specs requires real Task 3 pairs; use stage_video_inventory for planning metadata")
    result: list[VideoSpec] = []
    for stage in ("RT2", "RT3", "RT6", "RT7"):
        candidates = [pair for pair in pairs if pair.stage == stage and pair.camera == "head"]
        result.extend(_triple_specs(stage, "head", candidates, f"frozen projected-state case sequence"))
    for backend in ("a_p2", "route_b"):
        candidates = [pair for pair in pairs if pair.stage == "RT8" and pair.case_id.startswith(backend)]
        result.extend(_triple_specs("RT8", "gaussian_head", candidates, "native render-manifest episode order", suffix=backend))
    for camera in ("head", "wrist"):
        candidates = [pair for pair in pairs if pair.stage == "RT9" and pair.camera == camera]
        result.extend(_triple_specs("RT9", camera, candidates, "native HDF5 segment order", suffix=camera))
    if len(result) != 24 or {stage: sum(item.stage == stage for item in result) for stage in _STAGE_COUNTS} != _STAGE_COUNTS:
        raise AssertionError("Task 4 must own exactly 24 stage videos")
    return tuple(result)


def _triple_specs(stage: str, camera: str, pairs: Sequence[object], label: str, suffix: str = "") -> tuple[VideoSpec, ...]:
    if not pairs:
        raise ValueError(f"no Task 3 pairs for {stage}/{camera}/{suffix}")
    source_frames = tuple(_frame_from_pair(pair) for pair in pairs)
    # Fixed three-second schedule: contiguous holds preserve frozen-source order
    # and are shared identically by baseline, ours, and side-by-side modes.
    frames = _held_schedule(source_frames, frame_count=36)
    source_run = pairs[0].source_run
    evidence = pairs[0].evidence_level
    tag = f"-{suffix}" if suffix else ""
    return tuple(VideoSpec(f"V-{stage}{tag}-{mode}", stage, source_run, camera, evidence, mode, frames, label) for mode in ("baseline", "ours", "side_by_side"))


def _held_schedule(source_frames: Sequence[VideoFrame], *, frame_count: int) -> tuple[VideoFrame, ...]:
    if not source_frames or frame_count < len(source_frames):
        raise ValueError("held schedule requires source frames and enough output slots")
    count = len(source_frames)
    return tuple(source_frames[(index * count) // frame_count] for index in range(frame_count))


def stage_video_inventory() -> tuple[VideoInventoryItem, ...]:
    """Return the fixed Task 4 planning inventory without synthetic video data."""
    groups = (
        ("RT2", "head", ""), ("RT3", "head", ""), ("RT6", "head", ""), ("RT7", "head", ""),
        ("RT8", "gaussian_head", "a_p2"), ("RT8", "gaussian_head", "route_b"),
        ("RT9", "head", "head"), ("RT9", "wrist", "wrist"),
    )
    items = tuple(
        VideoInventoryItem(f"V-{stage}{('-' + suffix) if suffix else ''}-{mode}", stage, camera, mode)
        for stage, camera, suffix in groups for mode in ("baseline", "ours", "side_by_side")
    )
    if len(items) != 24:
        raise AssertionError("Task 4 inventory must contain exactly 24 stage videos")
    return items


__all__ = ["VIDEO_HEIGHT", "VIDEO_WIDTH", "VideoContrastReceipt", "VideoFrame", "VideoInventoryItem", "VideoSpec", "_held_schedule", "stage_video_inventory", "stage_video_specs", "verify_video_pair", "write_method_video"]
