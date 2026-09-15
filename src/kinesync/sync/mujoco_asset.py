"""Deterministic conversion of inherited PiPER URDF assets for MuJoCo."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET


@dataclass(frozen=True)
class ConversionManifest:
    """Fingerprints and intentional removals made during one URDF conversion."""

    path: Path
    source_sha256: str
    output_sha256: str
    removed_dae_collisions: tuple[dict[str, str], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "output_sha256": self.output_sha256,
            "removed_dae_collisions": list(self.removed_dae_collisions),
            "source_sha256": self.source_sha256,
        }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _paths_alias(first: Path, second: Path) -> bool:
    if first.resolve(strict=False) == second.resolve(strict=False):
        return True
    return first.exists() and second.exists() and first.samefile(second)


def _require_within_root(path: Path, root: Path, *, filename: str) -> Path:
    resolved_root = root.resolve()
    resolved_path = path.resolve()
    if not resolved_path.is_relative_to(resolved_root):
        raise ValueError(f"URDF mesh path escapes permitted root: {filename!r}")
    return resolved_path


def _resolved_mesh_path(
    filename: str,
    *,
    source_directory: Path,
    package_root: Path,
) -> Path:
    if not filename:
        raise ValueError("URDF mesh filename must be nonempty")
    if filename.startswith("package://"):
        package_reference = filename.removeprefix("package://")
        package_name, separator, resource = package_reference.partition("/")
        if not package_name or not separator or not resource:
            raise ValueError(f"Invalid package URI: {filename!r}")
        if any(part == ".." for part in Path(resource).parts):
            raise ValueError(f"Package URI escapes its package: {filename!r}")
        package_directory = package_root
        if package_root.name != package_name:
            package_directory = package_root / package_name
        resolved = _require_within_root(
            package_directory / resource,
            package_directory,
            filename=filename,
        )
    else:
        path = Path(filename).expanduser()
        resolved = _require_within_root(
            path if path.is_absolute() else source_directory / path,
            source_directory,
            filename=filename,
        )
    if not resolved.is_file():
        raise FileNotFoundError(f"URDF mesh does not exist: {resolved}")
    return resolved


def _configure_mujoco_compiler(root: ET.Element) -> None:
    mujoco = root.find("mujoco")
    if mujoco is None:
        mujoco = ET.Element("mujoco")
        root.insert(0, mujoco)
    compiler = mujoco.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(mujoco, "compiler")
    compiler.attrib.update(
        {
            "strippath": "false",
            "discardvisual": "true",
            "balanceinertia": "true",
        }
    )


def _remove_dae_collisions(root: ET.Element) -> tuple[dict[str, str], ...]:
    removals: list[dict[str, str]] = []
    for link in root.findall("link"):
        link_name = link.attrib.get("name")
        if not link_name:
            raise ValueError("URDF link lacks a name")
        for collision in list(link.findall("collision")):
            mesh = collision.find("./geometry/mesh")
            filename = "" if mesh is None else mesh.attrib.get("filename", "")
            if Path(filename).suffix.lower() == ".dae":
                link.remove(collision)
                removals.append(
                    {"link_name": link_name, "source_filename": filename}
                )
    return tuple(removals)


def convert_urdf_for_mujoco(
    source: str | Path,
    output: str | Path,
    package_root: str | Path,
) -> ConversionManifest:
    """Derive a MuJoCo-loadable URDF without editing the source asset."""

    source_path = Path(source).expanduser().resolve()
    output_path = Path(output).expanduser().resolve()
    manifest_path = output_path.with_name("conversion_manifest.json")
    package_root_path = Path(package_root).expanduser().resolve()
    if _paths_alias(source_path, output_path):
        raise ValueError("output must not be the same file as the source URDF")
    if not source_path.is_file():
        raise FileNotFoundError(f"URDF source does not exist: {source_path}")
    if output_path.name == "conversion_manifest.json":
        raise ValueError("output filename is reserved for the conversion manifest")
    if _paths_alias(manifest_path, source_path):
        raise ValueError("manifest must not be the same file as the source URDF")
    if _paths_alias(manifest_path, output_path):
        raise ValueError("manifest must not be the same file as the output URDF")
    if not package_root_path.is_dir():
        raise NotADirectoryError(
            f"package_root is not a directory: {package_root_path}"
        )

    source_sha256 = _sha256(source_path)
    tree = ET.parse(source_path)
    root = tree.getroot()
    if root.tag != "robot":
        raise ValueError(f"Expected URDF robot root, got {root.tag!r}")

    removed_dae_collisions = _remove_dae_collisions(root)
    for mesh in root.findall(".//mesh"):
        source_filename = mesh.attrib.get("filename")
        if source_filename is None:
            raise ValueError("URDF mesh lacks a filename")
        mesh.set(
            "filename",
            str(
                _resolved_mesh_path(
                    source_filename,
                    source_directory=source_path.parent,
                    package_root=package_root_path,
                )
            ),
        )
    _configure_mujoco_compiler(root)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    output_sha256 = _sha256(output_path)
    manifest = ConversionManifest(
        path=manifest_path,
        source_sha256=source_sha256,
        output_sha256=output_sha256,
        removed_dae_collisions=removed_dae_collisions,
    )
    manifest_path.write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


__all__ = ["ConversionManifest", "convert_urdf_for_mujoco"]
