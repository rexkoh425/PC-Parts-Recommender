"""Small shared helpers for deterministic training commands."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_MEBIBYTE = 1024 * 1024
_GIBIBYTE = 1024 * _MEBIBYTE


@dataclass(frozen=True, slots=True)
class HostMemorySnapshot:
    """A point-in-time view of physical host memory used by a training preflight."""

    total_bytes: int
    available_bytes: int
    source: str

    def __post_init__(self) -> None:
        if self.total_bytes <= 0:
            raise ValueError("host memory total must be positive")
        if not 0 <= self.available_bytes <= self.total_bytes:
            raise ValueError("host available memory must be between zero and total memory")
        if not self.source:
            raise ValueError("host memory source must not be empty")

    @property
    def used_bytes(self) -> int:
        return self.total_bytes - self.available_bytes

    def to_dict(self) -> dict[str, int | str | float]:
        return {
            "source": self.source,
            "total_bytes": self.total_bytes,
            "available_bytes": self.available_bytes,
            "used_bytes": self.used_bytes,
            "total_gib": round(self.total_bytes / _GIBIBYTE, 3),
            "available_gib": round(self.available_bytes / _GIBIBYTE, 3),
            "used_gib": round(self.used_bytes / _GIBIBYTE, 3),
        }


@dataclass(frozen=True, slots=True)
class HostMemoryPreflight:
    """Immutable admission decision for one bounded training invocation."""

    snapshot: HostMemorySnapshot
    max_used_bytes: int
    estimated_additional_bytes: int
    minimum_free_bytes: int

    @property
    def projected_used_bytes(self) -> int:
        return self.snapshot.used_bytes + self.estimated_additional_bytes

    @property
    def projected_available_bytes(self) -> int:
        return self.snapshot.available_bytes - self.estimated_additional_bytes

    def to_dict(self) -> dict[str, int | float | str]:
        return {
            **self.snapshot.to_dict(),
            "max_used_gib": round(self.max_used_bytes / _GIBIBYTE, 3),
            "estimated_additional_mib": round(self.estimated_additional_bytes / _MEBIBYTE, 3),
            "minimum_free_mib": round(self.minimum_free_bytes / _MEBIBYTE, 3),
            "projected_used_gib": round(self.projected_used_bytes / _GIBIBYTE, 3),
            "projected_available_gib": round(self.projected_available_bytes / _GIBIBYTE, 3),
        }


def host_memory_snapshot() -> HostMemorySnapshot:
    """Return available physical RAM without adding an optional runtime dependency."""

    if sys.platform == "win32":
        import ctypes

        class _MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = _MemoryStatusEx()
        status.dwLength = ctypes.sizeof(_MemoryStatusEx)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        if not kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise OSError(ctypes.get_last_error(), "GlobalMemoryStatusEx failed")
        return HostMemorySnapshot(
            total_bytes=int(status.ullTotalPhys),
            available_bytes=int(status.ullAvailPhys),
            source="windows_global_memory_status_ex",
        )

    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        total_pages = int(os.sysconf("SC_PHYS_PAGES"))
        available_pages = int(os.sysconf("SC_AVPHYS_PAGES"))
    except (AttributeError, OSError, ValueError) as error:
        raise OSError("unable to determine host physical-memory availability") from error
    return HostMemorySnapshot(
        total_bytes=page_size * total_pages,
        available_bytes=page_size * available_pages,
        source="posix_sysconf",
    )


def require_host_memory_headroom(
    *,
    max_used_gib: float,
    estimated_additional_mib: float,
    minimum_free_mib: float,
    snapshot: HostMemorySnapshot | None = None,
) -> HostMemoryPreflight:
    """Fail before a run would breach the host cap or consume required free RAM.

    ``estimated_additional_mib`` is deliberately conservative.  It should include
    the model's documented peak allocation, even when some of that allocation is
    already resident, because the purpose is safe admission rather than an exact
    prediction of process RSS.
    """

    if max_used_gib <= 0:
        raise ValueError("max_used_gib must be positive")
    if estimated_additional_mib < 0:
        raise ValueError("estimated_additional_mib must be non-negative")
    if minimum_free_mib < 0:
        raise ValueError("minimum_free_mib must be non-negative")
    preflight = HostMemoryPreflight(
        snapshot=snapshot or host_memory_snapshot(),
        max_used_bytes=round(max_used_gib * _GIBIBYTE),
        estimated_additional_bytes=round(estimated_additional_mib * _MEBIBYTE),
        minimum_free_bytes=round(minimum_free_mib * _MEBIBYTE),
    )
    if preflight.projected_used_bytes >= preflight.max_used_bytes:
        raise MemoryError(
            "host memory preflight refused training: projected used memory "
            f"{preflight.projected_used_bytes / _GIBIBYTE:.2f} GiB is at or above "
            f"the {max_used_gib:.2f} GiB cap"
        )
    if preflight.projected_available_bytes < preflight.minimum_free_bytes:
        raise MemoryError(
            "host memory preflight refused training: projected available memory "
            f"{preflight.projected_available_bytes / _MEBIBYTE:.0f} MiB is below "
            f"the required {minimum_free_mib:.0f} MiB headroom"
        )
    return preflight


def estimate_materialized_file_memory_mib(
    paths: Sequence[Path],
    *,
    expansion_factor: float,
    runtime_allowance_mib: float,
) -> float:
    """Conservatively bound full structured-file materialisation before opening files.

    The current ranking and entity-resolution trainers intentionally retain
    parsed records for leakage-safe splitting and model fitting.  Their peak
    includes raw JSON objects, typed contracts, feature arrays, and learner
    workspaces; use a documented multiplier plus a fixed allowance rather than
    treating the on-disk byte count as an in-memory bound.
    """

    if not paths:
        raise ValueError("at least one materialized input file is required")
    if expansion_factor < 1.0:
        raise ValueError("expansion_factor must be at least one")
    if runtime_allowance_mib < 0:
        raise ValueError("runtime_allowance_mib must be non-negative")
    total_bytes = 0
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        total_bytes += path.stat().st_size
    return (total_bytes * expansion_factor / _MEBIBYTE) + runtime_allowance_mib


def comma_separated(value: str) -> tuple[str, ...]:
    """Parse a comma-separated option while rejecting empty results."""

    columns = tuple(part.strip() for part in value.split(",") if part.strip())
    if not columns:
        raise ValueError("at least one column is required")
    if len(columns) != len(set(columns)):
        raise ValueError("column names must not be repeated")
    return columns


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def portable_path_reference(
    path: Path,
    *,
    workspace_root: Path | None = None,
) -> str:
    """Render an evidence path without embedding a host-specific absolute path.

    The content digest carried alongside the reference is the authoritative
    identity.  A repository-relative reference makes checked-in evidence
    portable, while an external input is deliberately reduced to its basename
    rather than leaking an operator's directory layout.
    """

    resolved = path.resolve()
    root = (workspace_root or Path.cwd()).resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError:
        return f"<external>/{resolved.name}" if resolved.name else "<external>"


def write_json(path: Path, payload: Mapping[str, Any] | Sequence[Any]) -> None:
    """Atomically persist deterministic, human-readable JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def print_json(payload: Mapping[str, Any] | Sequence[Any]) -> None:
    json.dump(payload, sys.stdout, indent=2, sort_keys=True, ensure_ascii=False)
    sys.stdout.write("\n")


def read_json_lines(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(value)
    if not rows:
        raise ValueError(f"{path} contains no records")
    return rows


def write_json_lines(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp-{os.getpid()}")
    count = 0
    with temporary.open("w", encoding="utf-8", newline="\n") as destination:
        for row in rows:
            destination.write(json.dumps(row, sort_keys=True, ensure_ascii=False))
            destination.write("\n")
            count += 1
    temporary.replace(path)
    return count


def synthetic_policy(flags: Sequence[bool], *, include_synthetic: bool) -> dict[str, object]:
    """Return an explicit reporting gate for row-level synthetic provenance."""

    total = len(flags)
    synthetic = sum(bool(flag) for flag in flags)
    included = synthetic if include_synthetic else 0
    return {
        "total_rows": total,
        "synthetic_rows": synthetic,
        "synthetic_rows_in_metrics": included,
        "synthetic_rows_excluded": not include_synthetic,
        "promotion_eligible": synthetic == 0 or (not include_synthetic and total > synthetic),
        "reporting_block_reason": (
            "synthetic rows are smoke-test data and cannot support promotion metrics"
            if include_synthetic and synthetic
            else None
        ),
    }
