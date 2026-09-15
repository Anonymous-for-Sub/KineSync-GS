#!/usr/bin/env python3
"""Build compact, paper-aligned videos for the static project page."""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

import cv2
import h5py
import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFont, ImageOps


WIDTH, HEIGHT = 1920, 1080
PAPER = (247, 245, 242)
BISTRE = (68, 44, 27)
SEAL = (107, 39, 23)
GOLD = (204, 158, 76)
DUN = (224, 208, 182)
CADET = (139, 158, 165)


def _font(path: Path, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(path), size=size)


def _cover(image: np.ndarray, width: int, height: int) -> Image.Image:
    source = Image.fromarray(image)
    scale = max(width / source.width, height / source.height)
    resized = source.resize(
        (round(source.width * scale), round(source.height * scale)),
        Image.Resampling.LANCZOS,
    )
    left = (resized.width - width) // 2
    top = (resized.height - height) // 2
    return resized.crop((left, top, left + width, top + height))


def _rounded_paste(
    canvas: Image.Image,
    image: Image.Image,
    box: tuple[int, int, int, int],
    radius: int,
) -> None:
    width, height = box[2] - box[0], box[3] - box[1]
    if image.size != (width, height):
        image = image.resize((width, height), Image.Resampling.LANCZOS)
    mask = Image.new("L", (width, height), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, width, height), radius=radius, fill=255)
    canvas.paste(image, box[:2], mask)


def _trim_white(image: Image.Image, margin: int = 36) -> Image.Image:
    rgb = image.convert("RGB")
    difference = ImageChops.difference(rgb, Image.new("RGB", rgb.size, "white"))
    difference = difference.convert("L").point(lambda value: 255 if value > 10 else 0)
    bounds = difference.getbbox()
    if bounds is None:
        return rgb
    left, top, right, bottom = bounds
    return rgb.crop((max(0, left - margin), max(0, top - margin),
                     min(rgb.width, right + margin), min(rgb.height, bottom + margin)))


def _contain(image: Image.Image, width: int, height: int) -> Image.Image:
    return ImageOps.contain(image.convert("RGB"), (width, height), Image.Resampling.LANCZOS)


class _Writer:
    def __init__(self, destination: Path, fps: int, ffmpeg: Path) -> None:
        self.destination = destination
        self.fps = fps
        self.ffmpeg = ffmpeg
        self.temp = tempfile.TemporaryDirectory(prefix="kinesync-page-video-")
        self.intermediate = Path(self.temp.name) / "frames.avi"
        self.writer = cv2.VideoWriter(
            str(self.intermediate),
            cv2.VideoWriter_fourcc(*"MJPG"),
            fps,
            (WIDTH, HEIGHT),
        )
        if not self.writer.isOpened():
            raise RuntimeError("Could not create intermediate video")

    def write(self, frame: Image.Image) -> None:
        array = np.asarray(frame.convert("RGB"))
        self.writer.write(cv2.cvtColor(array, cv2.COLOR_RGB2BGR))

    def close(self) -> None:
        self.writer.release()
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                str(self.ffmpeg), "-y", "-hide_banner", "-loglevel", "error",
                "-i", str(self.intermediate), "-an", "-c:v", "libx264",
                "-preset", "slow", "-crf", "23", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", str(self.destination),
            ],
            check=True,
        )
        self.temp.cleanup()


def build_temporal_background(
    first_hdf5: Path,
    second_hdf5: Path,
    destination: Path,
    ffmpeg: Path,
    fps: int = 12,
    seconds: int = 9,
) -> None:
    ranges = ((1076, 1199), (644, 801))
    panels = ((20, 20, 950, 530), (970, 20, 1900, 530),
              (20, 550, 950, 1060), (970, 550, 1900, 1060))
    writer = _Writer(destination, fps, ffmpeg)
    with h5py.File(first_hdf5, "r") as first, h5py.File(second_hdf5, "r") as second:
        datasets = (
            first["cam_head/color"], first["cam_wrist/color"],
            second["cam_head/color"], second["cam_wrist/color"],
        )
        for frame_index in range(fps * seconds):
            phase = frame_index / max(1, fps * seconds - 1)
            requested_indices = (
                round(ranges[0][0] + phase * (ranges[0][1] - ranges[0][0])),
                round(ranges[0][0] + phase * (ranges[0][1] - ranges[0][0])),
                round(ranges[1][0] + phase * (ranges[1][1] - ranges[1][0])),
                round(ranges[1][0] + phase * (ranges[1][1] - ranges[1][0])),
            )
            indices = tuple(
                min(requested, len(dataset) - 1)
                for requested, dataset in zip(requested_indices, datasets, strict=True)
            )
            canvas = Image.new("RGB", (WIDTH, HEIGHT), BISTRE)
            for dataset, index, box in zip(datasets, indices, panels, strict=True):
                panel = _cover(dataset[index], box[2] - box[0], box[3] - box[1])
                _rounded_paste(canvas, panel, box, radius=18)
            writer.write(canvas)
    writer.close()


def build_spatial_video(
    source: Path,
    destination: Path,
    ffmpeg: Path,
    regular_font: Path,
    bold_font: Path,
    fps: int = 12,
) -> None:
    capture = cv2.VideoCapture(str(source))
    frames: list[np.ndarray] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    if not frames:
        raise ValueError(f"No frames decoded from {source}")

    labels = (
        ("Reference", CADET),
        ("Measured state", SEAL),
        ("Gaussian proposal", GOLD),
        ("Verified state", BISTRE),
    )
    card_boxes = ((24, 24, 951, 531), (969, 24, 1896, 531),
                  (24, 549, 951, 1056), (969, 549, 1896, 1056))
    title_font = _font(bold_font, 38)
    view_font = _font(regular_font, 27)
    writer = _Writer(destination, fps, ffmpeg)

    output_frames = len(frames) * 2
    for output_index in range(output_frames):
        frame = frames[min(len(frames) - 1, output_index // 2)]
        use_external = output_index >= output_frames // 2
        view_name = "External view" if use_external else "Head view"
        canvas = Image.new("RGB", (WIDTH, HEIGHT), PAPER)
        draw = ImageDraw.Draw(canvas)
        for stage, (card, label) in enumerate(zip(card_boxes, labels, strict=True)):
            x0, y0, x1, y1 = card
            name, accent = label
            draw.rounded_rectangle(card, radius=18, fill=(255, 255, 255), outline=DUN, width=3)
            draw.rounded_rectangle((x0 + 18, y0 + 18, x0 + 66, y0 + 66), radius=8, fill=accent)
            draw.text((x0 + 84, y0 + 19), name, font=title_font, fill=BISTRE)
            view_width = draw.textbbox((0, 0), view_name, font=view_font)[2]
            draw.text((x1 - view_width - 22, y0 + 26), view_name, font=view_font, fill=SEAL)

            source_x = stage * 480
            row_y = slice(300, 492) if use_external else slice(106, 298)
            row = frame[row_y, source_x + 92:source_x + 475]
            panel_box = (x0 + 18, y0 + 82, x1 - 18, y1 - 18)
            panel = _cover(row, panel_box[2] - panel_box[0], panel_box[3] - panel_box[1])
            _rounded_paste(canvas, panel, panel_box, radius=12)
        writer.write(canvas)
    writer.close()


def build_piper_task_video(
    source: Path,
    destination: Path,
    task_name: str,
    ffmpeg: Path,
    regular_font: Path,
    bold_font: Path,
    fps: int = 10,
) -> None:
    capture = cv2.VideoCapture(str(source))
    writer = _Writer(destination, fps, ffmpeg)
    title_font = _font(bold_font, 44)
    view_font = _font(regular_font, 30)
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        head = frame[85:770, 24:936]
        wrist = frame[85:770, 984:1896]
        canvas = _cover(head, WIDTH, HEIGHT)
        draw = ImageDraw.Draw(canvas, "RGBA")
        draw.rounded_rectangle((34, 30, 520, 104), radius=16, fill=(68, 44, 27, 220))
        draw.text((58, 42), task_name, font=title_font, fill=PAPER)
        draw.rounded_rectangle((34, 118, 196, 166), radius=10, fill=(255, 255, 255, 218))
        draw.text((54, 124), "Head view", font=view_font, fill=BISTRE)

        inset_box = (1280, 575, 1882, 1027)
        wrist_panel = _cover(wrist, inset_box[2] - inset_box[0], inset_box[3] - inset_box[1])
        draw.rounded_rectangle(
            (inset_box[0] - 8, inset_box[1] - 8, inset_box[2] + 8, inset_box[3] + 8),
            radius=22, fill=(255, 255, 255, 235),
        )
        _rounded_paste(canvas, wrist_panel, inset_box, radius=16)
        draw.rounded_rectangle((1296, 591, 1474, 639), radius=10, fill=(68, 44, 27, 220))
        draw.text((1316, 597), "Wrist view", font=view_font, fill=PAPER)
        writer.write(canvas)
    capture.release()
    writer.close()


def build_franka_video(
    source_root: Path,
    destination: Path,
    ffmpeg: Path,
    regular_font: Path,
    bold_font: Path,
    fps: int = 12,
) -> None:
    poses = ("g", "h", "i")
    measured = [
        _trim_white(Image.open(source_root / f"F5__franka_{pose}__clean_nonzero__view_a__baseline__matched-crop__1440x1080.png"))
        for pose in poses
    ]
    verified = [
        _trim_white(Image.open(source_root / f"F5__franka_{pose}__clean_nonzero__view_a__ours__matched-crop__1440x1080.png"))
        for pose in poses
    ]
    title_font = _font(bold_font, 50)
    state_font = _font(bold_font, 34)
    body_font = _font(regular_font, 29)
    small_font = _font(regular_font, 24)
    writer = _Writer(destination, fps, ffmpeg)
    seconds_per_pose = 2.4
    total_frames = round(len(poses) * seconds_per_pose * fps)

    for frame_index in range(total_frames):
        pose_index = min(len(poses) - 1, int(frame_index / (seconds_per_pose * fps)))
        local_phase = (frame_index / fps - pose_index * seconds_per_pose) / seconds_per_pose
        blend = min(1.0, max(0.0, (local_phase - 0.28) / 0.44))
        left = _contain(measured[pose_index], 1260, 850)
        right = _contain(verified[pose_index], 1260, 850)
        viewport = Image.new("RGB", (1320, 900), "white")
        measured_layer = Image.new("RGB", viewport.size, "white")
        verified_layer = Image.new("RGB", viewport.size, "white")
        measured_layer.paste(left, ((viewport.width - left.width) // 2, (viewport.height - left.height) // 2))
        verified_layer.paste(right, ((viewport.width - right.width) // 2, (viewport.height - right.height) // 2))
        viewport = Image.blend(measured_layer, verified_layer, blend)

        canvas = Image.new("RGB", (WIDTH, HEIGHT), PAPER)
        draw = ImageDraw.Draw(canvas)
        draw.rounded_rectangle((26, 26, 1400, 1054), radius=20, fill="white", outline=DUN, width=3)
        _rounded_paste(canvas, viewport, (52, 124, 1374, 1024), radius=14)
        draw.text((58, 52), "Franka whole-state recovery", font=title_font, fill=BISTRE)
        stage = "Verified state" if blend >= 0.5 else "Measured state"
        stage_color = BISTRE if blend >= 0.5 else CADET
        draw.rounded_rectangle((1070, 48, 1366, 103), radius=12, fill=stage_color)
        stage_width = draw.textbbox((0, 0), stage, font=small_font)[2]
        draw.text((1218 - stage_width // 2, 61), stage, font=small_font, fill=PAPER)

        draw.rounded_rectangle((1422, 26, 1894, 1054), radius=20, fill="white", outline=DUN, width=3)
        draw.text((1460, 70), f"Pose {poses[pose_index].upper()}", font=state_font, fill=SEAL)
        draw.text((1460, 150), "Aggregate qMAE", font=body_font, fill=BISTRE)
        draw.text((1460, 205), "0.590 deg", font=title_font, fill=CADET)
        draw.text((1460, 280), "to", font=body_font, fill=GOLD)
        draw.text((1460, 326), "0.322 deg", font=title_font, fill=SEAL)
        draw.text((1460, 420), "Cross-view evidence", font=body_font, fill=BISTRE)
        draw.text((1460, 468), "Atomic update", font=body_font, fill=BISTRE)

        line_y = 828
        draw.line((1484, line_y, 1830, line_y), fill=DUN, width=8)
        for index, pose in enumerate(poses):
            x = 1484 + index * 173
            color = GOLD if index == pose_index else CADET
            draw.ellipse((x - 12, line_y - 12, x + 12, line_y + 12), fill=color)
            draw.text((x - 22, line_y + 28), pose.upper(), font=small_font, fill=BISTRE)
        draw.text((1460, 925), "Measured  ->  verified", font=body_font, fill=SEAL)
        writer.write(canvas)
    writer.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--temporal-first", type=Path, required=True)
    parser.add_argument("--temporal-second", type=Path, required=True)
    parser.add_argument("--spatial-source", type=Path, required=True)
    parser.add_argument("--temporal-output", type=Path, required=True)
    parser.add_argument("--spatial-output", type=Path, required=True)
    parser.add_argument("--piper-corn-source", type=Path, required=True)
    parser.add_argument("--piper-corn-output", type=Path, required=True)
    parser.add_argument("--piper-cube-source", type=Path, required=True)
    parser.add_argument("--piper-cube-output", type=Path, required=True)
    parser.add_argument("--franka-source-root", type=Path, required=True)
    parser.add_argument("--franka-output", type=Path, required=True)
    parser.add_argument("--ffmpeg", type=Path, required=True)
    parser.add_argument("--regular-font", type=Path, required=True)
    parser.add_argument("--bold-font", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_temporal_background(
        args.temporal_first, args.temporal_second, args.temporal_output, args.ffmpeg
    )
    build_spatial_video(
        args.spatial_source, args.spatial_output, args.ffmpeg,
        args.regular_font, args.bold_font,
    )
    build_piper_task_video(
        args.piper_corn_source, args.piper_corn_output, "Corn to plate",
        args.ffmpeg, args.regular_font, args.bold_font,
    )
    build_franka_video(
        args.franka_source_root, args.franka_output, args.ffmpeg,
        args.regular_font, args.bold_font,
    )
    build_piper_task_video(
        args.piper_cube_source, args.piper_cube_output, "Red cube to plate",
        args.ffmpeg, args.regular_font, args.bold_font,
    )


if __name__ == "__main__":
    main()
