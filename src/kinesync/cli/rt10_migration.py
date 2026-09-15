"""Package and validate a portable RT10 PiPER/D455 deployment tree."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping, Sequence


_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_BUNDLE_ROOT = "kinesync-gs"
_MANIFEST_NAME = "RT10_TRANSFER_MANIFEST.json"
_INCLUDED_TOP_LEVEL = (
    ".gitignore",
    "README.md",
    "pyproject.toml",
    "configs",
    "docs",
    "project",
    "scripts",
    "src",
    "tests",
)
_IGNORED_PARTS = {".git", ".mypy_cache", ".pytest_cache", "__pycache__"}
_IGNORED_SUFFIXES = {".log", ".pyc", ".pyo"}
_REQUIRED_FILES = (
    "README.md",
    "pyproject.toml",
    "configs/rt10_frozen_temporal_guard.json",
    "src/kinesync/cli/rt10_piper_d455_live.py",
    "project/RT10_PIPER_D455_LIVE_RUNBOOK_ZH.md",
)
_CAMERA_OWNER_BASENAMES = {
    "collect_piper_d455.py",
    "collect_piper_d455_gamepad.py",
    "collect_piper_d455_master_slave.py",
    "deploy_openpi_piper_d455.py",
    "deploy_openvla_oft_piper_d455.py",
    "deploy_openvla_piper_d455.py",
}
_OPTIONAL_COMMANDS = ("ffplay", "ip", "jq", "rs-enumerate-devices")


def create_transfer_bundle(
    project_root: str | Path, output_path: str | Path
) -> dict[str, object]:
    """Create a compact source bundle with an embedded per-file hash manifest."""

    root = Path(project_root).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    missing = [name for name in _REQUIRED_FILES if not (root / name).is_file()]
    if missing:
        raise ValueError(f"project is missing required RT10 files: {', '.join(missing)}")
    files = _collect_transfer_files(root)
    entries = [_file_entry(root, path) for path in files]
    manifest = {
        "schema": "kinesync.rt10_transfer_bundle.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "bundle_root": _BUNDLE_ROOT,
        "file_count": len(entries),
        "files": entries,
    }
    manifest_bytes = (
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    ).encode("ascii")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    with tarfile.open(output, "x:gz") as archive:
        for path in files:
            archive.add(path, arcname=f"{_BUNDLE_ROOT}/{path.relative_to(root).as_posix()}")
        info = tarfile.TarInfo(f"{_BUNDLE_ROOT}/{_MANIFEST_NAME}")
        info.size = len(manifest_bytes)
        info.mode = 0o644
        info.mtime = 0
        archive.addfile(info, io.BytesIO(manifest_bytes))
    archive_hash = _sha256(output)
    sidecar = output.with_name(f"{output.name}.sha256")
    sidecar.write_text(f"{archive_hash}  {output.name}\n", encoding="ascii")
    return {
        "schema": manifest["schema"],
        "created_at_utc": manifest["created_at_utc"],
        "bundle_root": manifest["bundle_root"],
        "file_count": manifest["file_count"],
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "archive": str(output),
        "archive_sha256": archive_hash,
        "sha256_sidecar": str(sidecar),
        "size_bytes": output.stat().st_size,
    }


def verify_transfer_tree(project_root: str | Path) -> dict[str, object]:
    """Verify an extracted transfer tree against its embedded manifest."""

    root = Path(project_root).expanduser().resolve()
    manifest_path = root / _MANIFEST_NAME
    if not manifest_path.is_file():
        return {
            "schema": "kinesync.rt10_transfer_verification.v1",
            "valid": False,
            "manifest": str(manifest_path),
            "missing": [_MANIFEST_NAME],
            "modified": [],
        }
    manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    entries = manifest.get("files")
    if manifest.get("schema") != "kinesync.rt10_transfer_bundle.v1" or not isinstance(
        entries, list
    ):
        raise ValueError("unsupported or malformed RT10 transfer manifest")
    missing: list[str] = []
    modified: list[str] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError("malformed file entry in RT10 transfer manifest")
        relative = entry.get("path")
        if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
            raise ValueError("unsafe file path in RT10 transfer manifest")
        path = (root / relative).resolve()
        if root not in path.parents:
            raise ValueError("unsafe file path in RT10 transfer manifest")
        if not path.is_file():
            missing.append(relative)
        elif path.stat().st_size != entry.get("size_bytes") or _sha256(path) != entry.get(
            "sha256"
        ):
            modified.append(relative)
    return {
        "schema": "kinesync.rt10_transfer_verification.v1",
        "valid": not missing and not modified,
        "manifest": str(manifest_path),
        "expected_file_count": len(entries),
        "missing": sorted(missing),
        "modified": sorted(modified),
    }


def build_target_report(
    *,
    project_root: str | Path,
    run_root: str | Path,
    control_root: str | Path | None = None,
    hardware_report: Mapping[str, object] | None = None,
    min_free_gb: float = 20.0,
    command_resolver: Callable[[str], str | None] = shutil.which,
    require_hardware: bool = True,
) -> dict[str, object]:
    """Build one target-host readiness report without commanding the robot."""

    root = Path(project_root).expanduser().resolve()
    required_missing = [name for name in _REQUIRED_FILES if not (root / name).is_file()]
    transfer = None
    if (root / _MANIFEST_NAME).is_file():
        transfer = verify_transfer_tree(root)
    project_ready = not required_missing and (
        transfer is None or transfer.get("valid") is True
    )
    storage = _storage_report(Path(run_root).expanduser().resolve(), min_free_gb)
    hardware = dict(hardware_report) if hardware_report is not None else _hardware_report()
    commands = {name: command_resolver(name) for name in _OPTIONAL_COMMANDS}
    optional_warnings = [name for name, path in commands.items() if path is None]
    control = _control_report(control_root)
    blockers: list[str] = []
    if not project_ready:
        blockers.append("project transfer is incomplete")
    if not storage["writable"]:
        blockers.append("run root is not writable")
    if not storage["enough_free_space"]:
        blockers.append(f"run root has less than {min_free_gb:g} GiB free")
    if require_hardware and hardware.get("ready") is not True:
        reasons = hardware.get("blockers")
        if isinstance(reasons, list) and reasons:
            blockers.extend(f"hardware: {reason}" for reason in reasons)
        else:
            blockers.append("hardware discovery is not ready")
    return {
        "schema": "kinesync.rt10_target_doctor.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "ready": not blockers,
        "project": {
            "root": str(root),
            "required_files_present": not required_missing,
            "required_files_missing": required_missing,
            "transfer_verification": transfer,
        },
        "storage": storage,
        "hardware": hardware,
        "control_integration": control,
        "optional_commands": commands,
        "optional_command_warnings": optional_warnings,
        "command_mode": "disabled",
        "blockers": blockers,
    }


def _collect_transfer_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for name in _INCLUDED_TOP_LEVEL:
        candidate = root / name
        if candidate.is_file() and not candidate.is_symlink():
            files.append(candidate)
        elif candidate.is_dir() and not candidate.is_symlink():
            for path in candidate.rglob("*"):
                relative = path.relative_to(root)
                if (
                    path.is_file()
                    and not path.is_symlink()
                    and not (_IGNORED_PARTS & set(relative.parts))
                    and path.suffix.lower() not in _IGNORED_SUFFIXES
                ):
                    files.append(path)
    return sorted(set(files), key=lambda path: path.relative_to(root).as_posix())


def _file_entry(root: Path, path: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _storage_report(path: Path, min_free_gb: float) -> dict[str, object]:
    writable = False
    error = None
    try:
        path.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(prefix=".kinesync-write-", dir=path, delete=True):
            writable = True
        free_bytes = shutil.disk_usage(path).free
    except OSError as exc:
        free_bytes = 0
        error = f"{type(exc).__name__}: {exc}"
    return {
        "path": str(path),
        "writable": writable,
        "free_bytes": free_bytes,
        "free_gib": round(free_bytes / (1024**3), 2),
        "minimum_free_gib": min_free_gb,
        "enough_free_space": free_bytes >= min_free_gb * (1024**3),
        "error": error,
    }


def _control_report(control_root: str | Path | None) -> dict[str, object]:
    if control_root is None:
        return {"status": "not_configured", "root": None, "camera_owner_scripts": []}
    root = Path(control_root).expanduser().resolve()
    if not root.is_dir():
        return {"status": "missing", "root": str(root), "camera_owner_scripts": []}
    scripts = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*.py")
        if path.name in _CAMERA_OWNER_BASENAMES
    )
    return {"status": "available", "root": str(root), "camera_owner_scripts": scripts}


def _hardware_report() -> dict[str, object]:
    from kinesync.cli.rt10_hardware_discover import build_hardware_report

    return build_hardware_report()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_report(report: Mapping[str, object], output: Path | None) -> None:
    payload = json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    if output is not None:
        target = output.expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(payload, encoding="ascii")
    print(payload, end="")


def bundle_main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build a portable KineSync RT10 archive.")
    parser.add_argument("--project-root", type=Path, default=_REPOSITORY_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    _write_report(create_transfer_bundle(args.project_root, args.output), args.report)


def doctor_main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Audit a KineSync RT10 target host.")
    parser.add_argument("--project-root", type=Path, default=_REPOSITORY_ROOT)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--control-root", type=Path)
    parser.add_argument("--min-free-gb", type=float, default=20.0)
    parser.add_argument("--software-only", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = build_target_report(
        project_root=args.project_root,
        run_root=args.run_root,
        control_root=args.control_root,
        min_free_gb=args.min_free_gb,
        require_hardware=not args.software_only,
    )
    _write_report(report, args.output)
    raise SystemExit(0 if report["ready"] else 2)


def verify_main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Verify an extracted KineSync RT10 tree.")
    parser.add_argument("--project-root", type=Path, default=_REPOSITORY_ROOT)
    args = parser.parse_args(argv)
    report = verify_transfer_tree(args.project_root)
    _write_report(report, None)
    raise SystemExit(0 if report["valid"] else 2)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="KineSync RT10 migration utilities.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("bundle", "doctor", "verify"):
        subparsers.add_parser(name)
    args, remainder = parser.parse_known_args(argv)
    {"bundle": bundle_main, "doctor": doctor_main, "verify": verify_main}[args.command](
        remainder
    )


if __name__ == "__main__":
    main()
