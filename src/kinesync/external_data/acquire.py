"""Small, explicit download helpers for public DROID and gated ABC metadata."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable
from urllib.error import HTTPError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from .audit import sha256_file

GCS_OBJECTS_API = "https://storage.googleapis.com/storage/v1/b/gresearch/o"
DROID_PUBLIC_URL = "https://storage.googleapis.com/gresearch/"
ABC_DATASET = "XDOF/ABC-130k"


def droid_raw_object_selection(
    objects: Iterable[str], *, include_stereo: bool = False, include_svo_serials: set[str] | None = None
) -> list[str]:
    """Keep only the native files needed to audit one DROID episode."""
    selected = []
    for name in objects:
        is_metadata = name.rsplit("/", 1)[-1].startswith("metadata_") and name.endswith(".json")
        is_native_video = "/recordings/MP4/" in name and name.endswith(".mp4")
        is_video_timestamps = "/recordings/MP4/" in name and name.endswith("_timestamps.json")
        if is_native_video and not include_stereo and "-stereo.mp4" in name:
            continue
        is_requested_svo = (
            "/recordings/SVO/" in name
            and name.endswith(".svo")
            and Path(name).stem in (include_svo_serials or set())
        )
        if name.endswith("trajectory.h5") or is_metadata or is_native_video or is_video_timestamps or is_requested_svo:
            selected.append(name)
    return sorted(selected)


def list_gcs_objects(prefix: str) -> list[dict[str, Any]]:
    """List one public GCS prefix through its JSON API without bucket-wide sync."""
    objects: list[dict[str, Any]] = []
    page_token: str | None = None
    while True:
        params = {"prefix": prefix}
        if page_token:
            params["pageToken"] = page_token
        request = Request(f"{GCS_OBJECTS_API}?{urlencode(params)}")
        with urlopen(request, timeout=60) as response:
            page = json.load(response)
        objects.extend(page.get("items", []))
        page_token = page.get("nextPageToken")
        if not page_token:
            return objects


def _download(url: str, target: Path, headers: dict[str, str] | None = None) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file():
        return
    temporary = target.with_name(f"{target.name}.partial")
    request_headers = dict(headers or {})
    mode = "wb"
    if temporary.is_file() and temporary.stat().st_size:
        request_headers["Range"] = f"bytes={temporary.stat().st_size}-"
        mode = "ab"
    request = Request(url, headers=request_headers)
    try:
        with urlopen(request, timeout=120) as response, temporary.open(mode) as handle:
            if mode == "ab" and response.status != 206:
                raise RuntimeError(f"server refused resumable download for {target}")
            while block := response.read(1024 * 1024):
                handle.write(block)
        temporary.replace(target)
    except BaseException:
        # Preserve a verified byte prefix so a later tmux retry can use HTTP Range.
        raise


def download_droid_raw_episode(
    episode_path: str,
    destination_root: Path,
    *,
    version: str = "1.0.1",
    include_stereo: bool = False,
    include_svo_serials: set[str] | None = None,
) -> dict[str, Any]:
    """Download a selected public DROID episode, never a parent directory tree."""
    episode_path = episode_path.strip("/")
    prefix = f"robotics/droid_raw/{version}/{episode_path}/"
    objects = list_gcs_objects(prefix)
    selected_names = droid_raw_object_selection(
        (item["name"] for item in objects),
        include_stereo=include_stereo,
        include_svo_serials=include_svo_serials,
    )
    if not any(name.endswith("trajectory.h5") for name in selected_names):
        raise ValueError(f"no trajectory.h5 found below gs://gresearch/{prefix}")
    if not any(name.endswith(".mp4") for name in selected_names):
        raise ValueError(f"no non-stereo native MP4 found below gs://gresearch/{prefix}")
    metadata_by_name = {item["name"]: item for item in objects}
    downloaded = []
    for name in selected_names:
        relative = Path(name).relative_to(f"robotics/droid_raw/{version}")
        target = destination_root / relative
        _download(f"{DROID_PUBLIC_URL}{quote(name)}", target)
        downloaded.append(
            {
                "source": f"gs://gresearch/{name}",
                "path": str(target),
                "bytes": target.stat().st_size,
                "sha256": sha256_file(target),
                "gcs_md5": metadata_by_name[name].get("md5Hash"),
            }
        )
    return {"dataset": "DROID", "version": version, "episode_path": episode_path, "files": downloaded}


def download_droid_calibration(destination_root: Path) -> dict[str, Any]:
    """Download the public KarlP/DROID calibration files needed for matching."""
    filenames = ("episode_id_to_path.json", "intrinsics.json", "cam2base_extrinsic_superset.json")
    files = []
    for filename in filenames:
        target = destination_root / filename
        _download(f"https://huggingface.co/KarlP/droid/resolve/main/{filename}", target)
        files.append(
            {
                "source": f"https://huggingface.co/KarlP/droid/resolve/main/{filename}",
                "path": str(target),
                "bytes": target.stat().st_size,
                "sha256": sha256_file(target),
            }
        )
    return {"dataset": "KarlP/droid", "files": files}


def download_abc_raw_mcap(remote_path: str, destination_root: Path, *, dataset: str = ABC_DATASET) -> dict[str, Any]:
    """Download one already-authorized native ABC MCAP without invoking export code."""
    remote_path = remote_path.strip("/")
    if not remote_path.endswith("/episode.mcap") or not remote_path.startswith("data/"):
        raise ValueError("ABC raw path must be a data/<split>/<task>/<episode>/episode.mcap path")
    token = _hf_token()
    if token is None:
        raise PermissionError("no Hugging Face token is available for gated ABC raw data")
    target = destination_root / Path(remote_path).relative_to("data")
    url = f"https://huggingface.co/datasets/{dataset}/resolve/main/{quote(remote_path)}"
    _download(url, target, headers={"Authorization": f"Bearer {token}"})
    return {
        "dataset": dataset,
        "source": url,
        "path": str(target),
        "bytes": target.stat().st_size,
        "sha256": sha256_file(target),
        "native": True,
        "conversion_invoked": False,
    }


def _hf_token() -> str | None:
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        if os.environ.get(name):
            return os.environ[name].strip()
    cache = Path.home() / ".cache" / "huggingface" / "token"
    if cache.is_file():
        return cache.read_text(encoding="utf-8").strip() or None
    return None


def probe_abc_access(dataset: str = ABC_DATASET) -> dict[str, Any]:
    """Check access metadata without accepting gated terms or downloading data."""
    token = _hf_token()
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    request = Request(f"https://huggingface.co/api/datasets/{dataset}", headers=headers)
    try:
        with urlopen(request, timeout=30) as response:
            metadata = json.load(response)
        return {
            "dataset": dataset,
            "token_present": token is not None,
            "http_status": 200,
            "gated": metadata.get("gated"),
            "accessible_metadata": True,
            "terms_accepted_by_script": False,
        }
    except HTTPError as error:
        return {
            "dataset": dataset,
            "token_present": token is not None,
            "http_status": error.code,
            "accessible_metadata": False,
            "terms_accepted_by_script": False,
        }
