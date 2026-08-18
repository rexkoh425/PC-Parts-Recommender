"""Bounded maintenance for crash-orphaned ranker publication stages."""

from __future__ import annotations

import ctypes
import importlib
import json
import os
import re
import secrets
import stat as stat_module
import sys
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

RANKER_STAGE_ACTIVITY_LOCK = ".publisher-active.lock"
MAX_RANKER_STAGE_FILES = 32
DEFAULT_MAXIMUM_PARENT_ENTRIES = 10_000
_BUNDLE_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_STAGE_TOKEN_PATTERN = r"[A-Za-z0-9_-]{8,64}"


class RankerPublicationMaintenanceError(RuntimeError):
    """Raised when a maintenance scope or destructive target is unsafe."""


@dataclass(frozen=True, slots=True)
class RankerStageMaintenanceItem:
    stage_name: str
    status: str
    age_seconds: float | None
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            "stage_name": self.stage_name,
            "status": self.status,
            "age_seconds": self.age_seconds,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class RankerStageMaintenanceReport:
    parent: Path
    bundle_name: str
    evaluated_at: datetime
    minimum_age_seconds: float
    dry_run: bool
    scanned_entries: int
    matching_stages: int
    items: tuple[RankerStageMaintenanceItem, ...]

    @property
    def removed_count(self) -> int:
        return sum(item.status == "removed" for item in self.items)

    @property
    def would_remove_count(self) -> int:
        return sum(item.status == "would_remove" for item in self.items)

    @property
    def blocked_count(self) -> int:
        return sum(item.status == "blocked" for item in self.items)

    def to_dict(self) -> dict[str, object]:
        return {
            "parent": str(self.parent),
            "bundle_name": self.bundle_name,
            "evaluated_at": self.evaluated_at.isoformat(),
            "minimum_age_seconds": self.minimum_age_seconds,
            "dry_run": self.dry_run,
            "scanned_entries": self.scanned_entries,
            "matching_stages": self.matching_stages,
            "removed_count": self.removed_count,
            "would_remove_count": self.would_remove_count,
            "blocked_count": self.blocked_count,
            "items": [item.to_dict() for item in self.items],
        }


@dataclass(frozen=True, slots=True)
class _FileIdentity:
    name: str
    device: int
    inode: int
    mode: int
    size: int
    modified_ns: int


@dataclass(frozen=True, slots=True)
class _StageSnapshot:
    device: int
    inode: int
    mode: int
    modified_ns: int
    files: tuple[_FileIdentity, ...]


def _is_linklike(path: Path) -> bool:
    is_junction = getattr(os.path, "isjunction", None)
    return path.is_symlink() or bool(is_junction is not None and is_junction(path))


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _rename_directory_noreplace(source: Path, destination: Path) -> None:
    if os.name == "nt":
        os.rename(source, destination)
        return
    if not sys.platform.startswith("linux"):
        raise RankerPublicationMaintenanceError(
            "atomic no-replace directory rename is unsupported on this platform"
        )
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise RankerPublicationMaintenanceError(
            "renameat2 is unavailable; ranker maintenance fails closed"
        )
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,
        os.fsencode(source),
        -100,
        os.fsencode(destination),
        1,
    )
    if result != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number), destination)


def _lock_descriptor(descriptor: int, *, blocking: bool) -> bool:
    if sys.platform == "win32":
        import msvcrt

        os.lseek(descriptor, 0, os.SEEK_SET)
        mode = msvcrt.LK_LOCK if blocking else msvcrt.LK_NBLCK
        try:
            msvcrt.locking(descriptor, mode, 1)
        except OSError:
            return False
        return True
    fcntl: Any = importlib.import_module("fcntl")

    operation = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
    try:
        fcntl.flock(descriptor, operation)
    except BlockingIOError:
        return False
    return True


def _unlock_descriptor(descriptor: int) -> None:
    if sys.platform == "win32":
        import msvcrt

        os.lseek(descriptor, 0, os.SEEK_SET)
        msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
        return
    fcntl: Any = importlib.import_module("fcntl")

    fcntl.flock(descriptor, fcntl.LOCK_UN)


class RankerStageActivityLock:
    """A process-owned advisory lock stored inside one private stage."""

    def __init__(self, path: Path, descriptor: int) -> None:
        self.path = path
        self._descriptor: int | None = descriptor

    def release(self, *, remove: bool) -> None:
        descriptor = self._descriptor
        if descriptor is None:
            return
        self._descriptor = None
        try:
            _unlock_descriptor(descriptor)
        finally:
            os.close(descriptor)
        if remove:
            with suppress(FileNotFoundError):
                self.path.unlink()


def acquire_ranker_stage_activity_lock(stage: Path) -> RankerStageActivityLock:
    """Create and hold the activity lock for a newly allocated stage."""

    stage_path = Path(stage)
    if _is_linklike(stage_path) or not stage_path.is_dir():
        raise RankerPublicationMaintenanceError("ranker stage is not a direct regular directory")
    lock_path = stage_path / RANKER_STAGE_ACTIVITY_LOCK
    descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
    try:
        os.write(descriptor, b"1")
        os.fsync(descriptor)
        if not _lock_descriptor(descriptor, blocking=False):
            raise RankerPublicationMaintenanceError("could not acquire new stage activity lock")
    except BaseException:
        os.close(descriptor)
        lock_path.unlink(missing_ok=True)
        raise
    return RankerStageActivityLock(lock_path, descriptor)


def _validate_bundle_name(bundle_name: str) -> str:
    if _BUNDLE_NAME_PATTERN.fullmatch(bundle_name) is None:
        raise ValueError("bundle_name must be one conservative direct filename")
    return bundle_name


def _validate_parent(parent: str | Path) -> Path:
    supplied = Path(parent)
    if not supplied.is_absolute():
        raise ValueError("ranker maintenance parent must be an explicit absolute path")
    if ".." in supplied.parts:
        raise ValueError("ranker maintenance parent cannot contain parent traversal")
    current = Path(supplied.anchor)
    for component in supplied.parts[1:]:
        current /= component
        if _is_linklike(current):
            raise RankerPublicationMaintenanceError(
                "ranker maintenance parent cannot traverse a symlink or junction"
            )
    if not supplied.is_dir():
        raise FileNotFoundError(supplied)
    resolved = supplied.resolve(strict=True)
    if resolved.parent == resolved or len(resolved.parts) < 3:
        raise ValueError("ranker maintenance refuses filesystem roots or broad parent paths")
    try:
        home = Path.home().resolve(strict=True)
    except OSError:
        home = None
    if home is not None and resolved == home:
        raise ValueError("ranker maintenance refuses the user home directory")
    return resolved


def _aware_utc(now: datetime | None) -> datetime:
    value = now or datetime.now(UTC)
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("maintenance timestamp must be timezone aware")
    return value.astimezone(UTC)


def _snapshot_flat_stage(stage: Path) -> _StageSnapshot:
    if _is_linklike(stage):
        raise RankerPublicationMaintenanceError("matching stage is a symlink or junction")
    root = os.lstat(stage)
    if not stat_module.S_ISDIR(root.st_mode):
        raise RankerPublicationMaintenanceError("matching stage is not a directory")
    files: list[_FileIdentity] = []
    with os.scandir(stage) as entries:
        for entry in entries:
            if len(files) >= MAX_RANKER_STAGE_FILES:
                raise RankerPublicationMaintenanceError("matching stage exceeds its file bound")
            child = Path(entry.path)
            if entry.is_symlink() or _is_linklike(child):
                raise RankerPublicationMaintenanceError(
                    "matching stage contains a symlink or junction"
                )
            # Use the same lstat contract before and after the atomic claim. On
            # Windows, DirEntry.stat() may report zero device/inode values while
            # os.lstat() exposes the stable file identity.
            child_stat = os.lstat(child)
            if not stat_module.S_ISREG(child_stat.st_mode):
                raise RankerPublicationMaintenanceError(
                    "matching stage contains a nested or non-regular entry"
                )
            files.append(
                _FileIdentity(
                    name=entry.name,
                    device=child_stat.st_dev,
                    inode=child_stat.st_ino,
                    mode=child_stat.st_mode,
                    size=child_stat.st_size,
                    modified_ns=child_stat.st_mtime_ns,
                )
            )
    return _StageSnapshot(
        device=root.st_dev,
        inode=root.st_ino,
        mode=root.st_mode,
        modified_ns=root.st_mtime_ns,
        files=tuple(sorted(files, key=lambda item: item.name)),
    )


def _stage_is_active(stage: Path) -> bool:
    lock_path = stage / RANKER_STAGE_ACTIVITY_LOCK
    if not lock_path.exists() and not _is_linklike(lock_path):
        return False
    if _is_linklike(lock_path) or not lock_path.is_file():
        raise RankerPublicationMaintenanceError("stage activity lock is not a regular file")
    descriptor = os.open(lock_path, os.O_RDWR)
    try:
        metadata = os.fstat(descriptor)
        if not stat_module.S_ISREG(metadata.st_mode) or metadata.st_size < 1:
            raise RankerPublicationMaintenanceError("stage activity lock is malformed")
        acquired = _lock_descriptor(descriptor, blocking=False)
        if not acquired:
            return True
        _unlock_descriptor(descriptor)
        return False
    finally:
        os.close(descriptor)


def _same_stage(left: _StageSnapshot, right: _StageSnapshot) -> bool:
    return left == right


def _claim_stage(stage: Path, *, parent: Path, bundle_name: str) -> Path:
    for _attempt in range(8):
        tombstone = parent / f".{bundle_name}.reap-{secrets.token_hex(12)}"
        try:
            _rename_directory_noreplace(stage, tombstone)
        except FileExistsError:
            continue
        return tombstone
    raise RankerPublicationMaintenanceError("could not allocate a unique maintenance tombstone")


def _delete_claimed_flat_stage(tombstone: Path, snapshot: _StageSnapshot) -> None:
    current = _snapshot_flat_stage(tombstone)
    if not _same_stage(snapshot, current):
        raise RankerPublicationMaintenanceError("claimed stage changed before deletion")
    for expected in snapshot.files:
        child = tombstone / expected.name
        child_stat = os.lstat(child)
        actual = _FileIdentity(
            name=expected.name,
            device=child_stat.st_dev,
            inode=child_stat.st_ino,
            mode=child_stat.st_mode,
            size=child_stat.st_size,
            modified_ns=child_stat.st_mtime_ns,
        )
        if actual != expected or not stat_module.S_ISREG(child_stat.st_mode) or _is_linklike(child):
            raise RankerPublicationMaintenanceError("claimed stage file changed before unlink")
        child.unlink()
    tombstone.rmdir()


def maintain_ranker_publication_stages(
    parent: str | Path,
    *,
    bundle_name: str,
    minimum_age: timedelta,
    dry_run: bool = True,
    now: datetime | None = None,
    maximum_entries: int = DEFAULT_MAXIMUM_PARENT_ENTRIES,
) -> RankerStageMaintenanceReport:
    """Report or remove only old, inactive, flat stages for one exact bundle."""

    selected_parent = _validate_parent(parent)
    selected_bundle = _validate_bundle_name(bundle_name)
    if minimum_age <= timedelta(0):
        raise ValueError("minimum_age must be positive")
    if maximum_entries < 1 or maximum_entries > 1_000_000:
        raise ValueError("maximum_entries must be between 1 and 1,000,000")
    evaluated_at = _aware_utc(now)
    stage_pattern = re.compile(rf"^\.{re.escape(selected_bundle)}\.publish-{_STAGE_TOKEN_PATTERN}$")

    direct_entries: list[os.DirEntry[str]] = []
    with os.scandir(selected_parent) as entries:
        for entry in entries:
            if len(direct_entries) >= maximum_entries:
                raise RankerPublicationMaintenanceError(
                    "ranker maintenance parent exceeds the bounded entry limit"
                )
            direct_entries.append(entry)

    items: list[RankerStageMaintenanceItem] = []
    for entry in sorted(direct_entries, key=lambda item: item.name):
        if stage_pattern.fullmatch(entry.name) is None:
            continue
        stage = selected_parent / entry.name
        if stage == selected_parent / selected_bundle:
            raise AssertionError("stage pattern must never select the final bundle")
        try:
            snapshot = _snapshot_flat_stage(stage)
            modified_at = datetime.fromtimestamp(snapshot.modified_ns / 1_000_000_000, tz=UTC)
            age_seconds = (evaluated_at - modified_at).total_seconds()
            if age_seconds < minimum_age.total_seconds():
                items.append(
                    RankerStageMaintenanceItem(
                        stage_name=entry.name,
                        status="preserved_new",
                        age_seconds=age_seconds,
                        reason="stage is younger than the minimum age",
                    )
                )
                continue
            if _stage_is_active(stage):
                items.append(
                    RankerStageMaintenanceItem(
                        stage_name=entry.name,
                        status="preserved_active",
                        age_seconds=age_seconds,
                        reason="publisher still holds the stage activity lock",
                    )
                )
                continue
            current = _snapshot_flat_stage(stage)
            if not _same_stage(snapshot, current):
                raise RankerPublicationMaintenanceError(
                    "matching stage changed during maintenance validation"
                )
            if dry_run:
                items.append(
                    RankerStageMaintenanceItem(
                        stage_name=entry.name,
                        status="would_remove",
                        age_seconds=age_seconds,
                        reason="old inactive stage passed bounded safety checks",
                    )
                )
                continue
            tombstone = _claim_stage(
                stage,
                parent=selected_parent,
                bundle_name=selected_bundle,
            )
            _delete_claimed_flat_stage(tombstone, current)
            _fsync_directory(selected_parent)
            items.append(
                RankerStageMaintenanceItem(
                    stage_name=entry.name,
                    status="removed",
                    age_seconds=age_seconds,
                    reason="old inactive stage was atomically claimed and removed",
                )
            )
        except (FileNotFoundError, OSError, RankerPublicationMaintenanceError) as error:
            items.append(
                RankerStageMaintenanceItem(
                    stage_name=entry.name,
                    status="blocked",
                    age_seconds=None,
                    reason=f"{type(error).__name__}: {error}",
                )
            )

    return RankerStageMaintenanceReport(
        parent=selected_parent,
        bundle_name=selected_bundle,
        evaluated_at=evaluated_at,
        minimum_age_seconds=minimum_age.total_seconds(),
        dry_run=dry_run,
        scanned_entries=len(direct_entries),
        matching_stages=len(items),
        items=tuple(items),
    )


def report_json(report: RankerStageMaintenanceReport) -> str:
    """Return deterministic JSON for the standalone maintenance CLI."""

    return json.dumps(report.to_dict(), allow_nan=False, indent=2, sort_keys=True)


__all__ = [
    "DEFAULT_MAXIMUM_PARENT_ENTRIES",
    "RANKER_STAGE_ACTIVITY_LOCK",
    "RankerPublicationMaintenanceError",
    "RankerStageActivityLock",
    "RankerStageMaintenanceItem",
    "RankerStageMaintenanceReport",
    "acquire_ranker_stage_activity_lock",
    "maintain_ranker_publication_stages",
    "report_json",
]
