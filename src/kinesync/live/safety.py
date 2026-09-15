"""Independent receipts for the command-incapable live-shadow path."""

from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Iterable


_EVENT_SCHEMA = "kinesync.live_shadow_event.v2"
_SAFETY_SCHEMA = "kinesync.live_shadow_safety.v2"
_PACKET_FIELDS = frozenset(
    {
        "session_id",
        "clock_domain",
        "sequence_id",
        "arrival_monotonic_ns",
        "head_capture_ns",
        "auxiliary_capture_ns",
        "state_capture_ns",
        "auxiliary_role",
        "source_id",
        "frame_id",
    }
)
_DYNAMIC_SYNTAX_VIOLATION = "dynamic_import_loader"
_DYNAMIC_LOADER_NAMES = frozenset({"__import__", "import_module"})
_REFLECTION_EXECUTION_NAMES = frozenset(
    {
        "getattr",
        "__getattr__",
        "__getattribute__",
        "attrgetter",
        "setattr",
        "delattr",
        "vars",
        "globals",
        "locals",
        "eval",
        "exec",
        "compile",
    }
)
_DENIED_DYNAMIC_NAMES = _DYNAMIC_LOADER_NAMES | _REFLECTION_EXECUTION_NAMES
_BUILTINS_NAMESPACE_NAMES = frozenset({"builtins", "__builtins__"})
_DENIED_IMPORT_PREFIXES = (
    _DYNAMIC_SYNTAX_VIOLATION,
    "subprocess",
    "socket",
    "serial",
    "rclpy",
    "rospy",
    "can",
    "control_your_robot",
    "kinesync.visualization",
    "kinesync.policy",
    "mujoco",
    "requests",
    "urllib",
    "http",
    "zmq",
    "websocket",
)


@dataclass(frozen=True)
class DependencyAuditReceipt:
    """AST-level local import-closure receipt for the observation-only path."""

    paths: tuple[str, ...]
    violations: tuple[str, ...]

    @property
    def valid(self) -> bool:
        return not self.violations


@dataclass(frozen=True)
class ShadowAuditReceipt:
    """Validated relationship between append-only events and the close receipt."""

    event_count: int
    terminal_reason: str | None
    valid: bool = True


def canonical_json_bytes(payload: dict[str, object]) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("ascii")


def sha256_hex(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def audit_live_imports(paths: Iterable[str | Path]) -> DependencyAuditReceipt:
    """Reject denylisted direct, dynamic, and transitive local imports from any entries."""

    pending = list(_python_files(paths))
    audited: set[Path] = set()
    violations: set[str] = set()
    while pending:
        path = pending.pop()
        resolved = path.resolve()
        if resolved in audited:
            continue
        audited.add(resolved)
        tree = ast.parse(resolved.read_text(encoding="utf-8"), filename=str(resolved))
        for imported, node in _imports(tree):
            if _is_denied(imported):
                violations.add(f"{resolved}:{imported}")
            local_targets = _local_import_targets(resolved, imported, node)
            for target in local_targets:
                canonical_name = _canonical_module_name(target)
                if _is_denied(canonical_name):
                    violations.add(f"{resolved}:{canonical_name}")
            pending.extend(local_targets)
    return DependencyAuditReceipt(
        paths=tuple(str(path) for path in sorted(audited)),
        violations=tuple(sorted(violations)),
    )


def verify_shadow_audit(
    event_path: str | Path, safety_path: str | Path
) -> ShadowAuditReceipt:
    """Reject tampered, incomplete, or nonzero-command shadow run evidence."""

    event_target = Path(event_path)
    raw_events = _read_event_lines(event_target)
    safety = _read_json_object(Path(safety_path), "safety receipt")
    if safety.get("schema") != _SAFETY_SCHEMA or safety.get("closed") is not True:
        raise ValueError("safety receipt is not a closed RT10-P receipt")
    if safety.get("events_complete") is not True:
        raise ValueError("event log is incomplete")
    _require_zero_counters(safety, "safety receipt")

    previous_hash: str | None = None
    accepted_packets = 0
    accepted_identity: tuple[str, str] | None = None
    terminal_rows: list[dict[str, object]] = []
    for expected_index, event in enumerate(raw_events):
        _validate_event(event, expected_index, previous_hash)
        previous_hash = str(event["event_sha256"])
        if event["monitor_accepted"] is True:
            accepted_packets += 1
            identity = (str(event["session_id"]), str(event["clock_domain"]))
            if accepted_identity is None:
                accepted_identity = identity
            elif identity != accepted_identity:
                raise ValueError("accepted event session or clock changed")
        if event["terminal"] is True:
            terminal_rows.append(event)

    if len(terminal_rows) > 1 or (terminal_rows and terminal_rows[0] is not raw_events[-1]):
        raise ValueError("only the final event may be terminal")
    terminal_reason = terminal_rows[0].get("terminal_reason") if terminal_rows else None
    if safety.get("event_count") != len(raw_events):
        raise ValueError("event count does not match safety receipt")
    if safety.get("accepted_packets") != accepted_packets:
        raise ValueError("accepted packet count does not match safety receipt")
    session_id, clock_domain = accepted_identity or (None, None)
    if safety.get("session_id") != session_id or safety.get("clock_domain") != clock_domain:
        raise ValueError("session or clock does not match accepted events")
    if safety.get("terminal_reason") != terminal_reason:
        raise ValueError("terminal reason does not match safety receipt")
    if safety.get("final_event_sha256") != previous_hash:
        raise ValueError("final event hash does not match safety receipt")
    if safety.get("events_file_sha256") != sha256_file(event_target):
        raise ValueError("event file hash does not match safety receipt")
    return ShadowAuditReceipt(event_count=len(raw_events), terminal_reason=terminal_reason)


def _validate_event(
    event: dict[str, object], expected_index: int, previous_hash: str | None
) -> None:
    if event.get("schema") != _EVENT_SCHEMA:
        raise ValueError("event schema is invalid")
    if event.get("event_index") != expected_index:
        raise ValueError("event log indices must be contiguous")
    if not isinstance(event.get("terminal"), bool) or not isinstance(
        event.get("monitor_accepted"), bool
    ):
        raise ValueError("event terminal flags are invalid")
    _require_zero_counters(event, "event")
    if event.get("previous_event_sha256") != previous_hash:
        raise ValueError("event hash chain is discontinuous")
    event_hash = event.get("event_sha256")
    if not _is_sha256(event_hash):
        raise ValueError("event hash is invalid")
    unhashed = dict(event)
    del unhashed["event_sha256"]
    if sha256_hex(canonical_json_bytes(unhashed)) != event_hash:
        raise ValueError("event content does not match its hash")

    has_packet = bool(_PACKET_FIELDS & set(event))
    if has_packet and not _PACKET_FIELDS <= set(event):
        raise ValueError("event packet fields are incomplete")
    if event["terminal"] is False:
        if not has_packet or event["monitor_accepted"] is not True:
            raise ValueError("accepted event lacks accepted packet data")
        if "estimate" not in event or not _is_nonnegative_integer(event.get("processing_latency_ns")):
            raise ValueError("accepted event estimate fields are invalid")
    else:
        if not isinstance(event.get("terminal_reason"), str):
            raise ValueError("terminal event must contain a string reason")
        if event["monitor_accepted"] is True and not has_packet:
            raise ValueError("accepted terminal event lacks packet data")


def _read_event_lines(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise ValueError("event log contains an unflushed partial line")
    events: list[dict[str, object]] = []
    for line in raw.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError("event log contains malformed JSON") from error
        if not isinstance(event, dict):
            raise ValueError("event log rows must be JSON objects")
        events.append(event)
    return events


def _python_files(paths: Iterable[str | Path]) -> tuple[Path, ...]:
    files: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_dir():
            files.extend(sorted(child for child in path.rglob("*.py") if child.is_file()))
        elif path.is_file():
            files.append(path)
        else:
            raise FileNotFoundError(path)
    return tuple(sorted({path.resolve() for path in files}))


def _imports(tree: ast.AST) -> tuple[tuple[str, ast.AST], ...]:
    """Collect static imports and fail closed on dynamic execution syntax."""

    imports: list[tuple[str, ast.AST]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend((alias.name, node) for alias in node.names)
            if any(
                _is_importlib_module(alias.name)
                or _is_builtins_module(alias.name)
                or _alias_uses_denied_name(alias)
                for alias in node.names
            ):
                imports.append((_DYNAMIC_SYNTAX_VIOLATION, node))
        elif isinstance(node, ast.ImportFrom):
            if node.module is None:
                imports.extend((alias.name, node) for alias in node.names)
            else:
                imports.append((node.module, node))
                imports.extend((f"{node.module}.{alias.name}", node) for alias in node.names)
            if _import_from_exposes_dynamic_syntax(node):
                imports.append((_DYNAMIC_SYNTAX_VIOLATION, node))
        elif isinstance(node, ast.Name) and node.id in (
            _DENIED_DYNAMIC_NAMES | _BUILTINS_NAMESPACE_NAMES
        ):
            imports.append((_DYNAMIC_SYNTAX_VIOLATION, node))
        elif isinstance(node, ast.Attribute) and (
            node.attr in (_DENIED_DYNAMIC_NAMES | _BUILTINS_NAMESPACE_NAMES)
            or node.attr == "__dict__"
        ):
            imports.append((_DYNAMIC_SYNTAX_VIOLATION, node))
        elif isinstance(node, ast.Subscript) and _is_dynamic_subscript(node):
            imports.append((_DYNAMIC_SYNTAX_VIOLATION, node))
        elif isinstance(node, ast.Call) and (
            _is_dynamic_reference(node.func) or _is_dynamic_mapping_lookup(node)
        ):
            imports.append((_DYNAMIC_SYNTAX_VIOLATION, node))
    return tuple(imports)


def _alias_uses_denied_name(alias: ast.alias) -> bool:
    imported_name = alias.name.rsplit(".", 1)[-1]
    return imported_name in _DENIED_DYNAMIC_NAMES or alias.asname in _DENIED_DYNAMIC_NAMES


def _import_from_exposes_dynamic_syntax(node: ast.ImportFrom) -> bool:
    if _is_importlib_module(node.module):
        return True
    if any(_alias_uses_denied_name(alias) for alias in node.names):
        return True
    return _is_builtins_module(node.module)


def _is_dynamic_reference(node: ast.AST) -> bool:
    if isinstance(node, ast.Name):
        return node.id in _DENIED_DYNAMIC_NAMES
    if isinstance(node, ast.Attribute):
        return node.attr in _DENIED_DYNAMIC_NAMES or node.attr == "__dict__"
    return isinstance(node, ast.Subscript) and _is_dynamic_subscript(node)


def _is_dynamic_subscript(node: ast.Subscript) -> bool:
    if isinstance(node.value, ast.Name) and node.value.id in _BUILTINS_NAMESPACE_NAMES:
        return True
    return isinstance(node.slice, ast.Constant) and node.slice.value in (
        _DENIED_DYNAMIC_NAMES | {"__dict__"}
    )


def _is_dynamic_mapping_lookup(node: ast.Call) -> bool:
    return (
        isinstance(node.func, ast.Attribute)
        and node.func.attr == "get"
        and bool(node.args)
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value in (_DENIED_DYNAMIC_NAMES | {"__dict__"})
    )


def _is_importlib_module(name: str | None) -> bool:
    return name == "importlib" or (isinstance(name, str) and name.startswith("importlib."))


def _is_builtins_module(name: str | None) -> bool:
    return name == "builtins" or (isinstance(name, str) and name.startswith("builtins."))


def _local_import_targets(path: Path, imported: str, node: ast.AST) -> tuple[Path, ...]:
    targets: set[Path] = set()
    parts = tuple(part for part in imported.split(".") if part)
    if not parts:
        return ()
    if isinstance(node, ast.ImportFrom) and node.level:
        base = path.parent
        for _ in range(node.level - 1):
            base = base.parent
        targets.update(_module_candidates(base, parts))
    for base in (path.parent, *path.parents):
        targets.update(_module_candidates(base, parts))
    return tuple(sorted(target for target in targets if target.is_file()))


def _module_candidates(base: Path, parts: tuple[str, ...]) -> tuple[Path, ...]:
    target = base.joinpath(*parts)
    return (target.with_suffix(".py"), target / "__init__.py")


def _canonical_module_name(path: Path) -> str:
    parts: list[str] = [] if path.name == "__init__.py" else [path.stem]
    package = path.parent
    while (package / "__init__.py").is_file():
        parts.append(package.name)
        package = package.parent
    return ".".join(reversed(parts))


def _is_denied(name: str) -> bool:
    return any(name == prefix or name.startswith(prefix + ".") for prefix in _DENIED_IMPORT_PREFIXES)


def _read_json_object(path: Path, label: str) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} contains malformed JSON") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def _require_zero_counters(payload: dict[str, object], label: str) -> None:
    for field in ("robot_command_requests", "robot_commands_sent"):
        value = payload.get(field)
        if not isinstance(value, Integral) or isinstance(value, bool) or value != 0:
            raise ValueError(f"{label} has a nonzero or invalid {field}")


def _is_nonnegative_integer(value: object) -> bool:
    return isinstance(value, Integral) and not isinstance(value, bool) and value >= 0


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


__all__ = [
    "DependencyAuditReceipt",
    "ShadowAuditReceipt",
    "audit_live_imports",
    "canonical_json_bytes",
    "sha256_file",
    "sha256_hex",
    "verify_shadow_audit",
]
