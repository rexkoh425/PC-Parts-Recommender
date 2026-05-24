"""Crash-safe staged publication for governed-web processed runs.

The generic parsed-batch writer intentionally remains unchanged.  A governed-web run is
written beneath a private workspace that has the same ``<source>/<snapshot>`` shape as a
normal processed root.  Only after the manifest, quality report, and retention receipt have
been validated and sealed is the complete run renamed into the catalogue-visible location.

Stale-operation deletion is deliberately outside this module.  Incomplete operations remain
under :data:`WEB_PUBLICATION_CONTROL_DIRECTORY` for the independent retention janitor.
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import os
import re
import secrets
import stat as stat_module
import sys
from contextlib import suppress
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Final, Protocol

from pipelines.retention.web import (
    _RUN_DIRECTORY_PATTERN,
    WEB_PROCESSED_RETENTION_RECEIPT,
    WebRetentionError,
    _direct_source_root,
    _is_linklike,
    _validate_processed_receipt,
    _validated_source_name,
)

WEB_PUBLICATION_CONTROL_DIRECTORY: Final = ".wp"
WEB_PUBLICATION_INTENT_SCHEMA_VERSION: Final = (
    "pc-build-recommender.web-processed-publication-intent.v1"
)
WEB_PUBLICATION_READY_SCHEMA_VERSION: Final = (
    "pc-build-recommender.web-processed-publication-ready.v1"
)

_OPERATION_ID_LENGTH = 24
_OPERATION_ID_PATTERN = re.compile(r"^[0-9a-f]{24}$")
_MAX_CONTROL_RECEIPT_BYTES = 256 * 1024
_MAX_ARTIFACT_BYTES = 512 * 1024 * 1024
_MAX_RUN_BYTES = 1024 * 1024 * 1024
_ALLOWED_RUN_FILES = {
    "records.jsonl",
    "rejections.jsonl",
    "records.parquet",
    "manifest.json",
    "data-quality.json",
    WEB_PROCESSED_RETENTION_RECEIPT,
}
_REQUIRED_RUN_FILES = _ALLOWED_RUN_FILES - {"records.parquet"}
_INTENT_FIELDS = {
    "schema_version",
    "operation_id",
    "source_name",
    "run_sha256",
    "created_at",
    "content_sha256",
}
_READY_FIELDS = {
    "schema_version",
    "operation_id",
    "source_name",
    "run_sha256",
    "intent_sha256",
    "sealed_at",
    "files",
    "content_sha256",
}
_FILE_METADATA_FIELDS = {"sha256", "byte_count"}


class _RecoveryWorkBudget(Protocol):
    """Minimal shared-budget contract supplied by governed-web retention."""

    def consume(self, *, label: str, amount: int = 1) -> None: ...


class WebPublicationError(WebRetentionError):
    """Raised when a staged run cannot be safely sealed or published."""


@dataclass(frozen=True, slots=True)
class WebProcessedPublication:
    """Exact paths and identity for one unpublished governed-web run."""

    processed_root: Path
    source_name: str
    run_sha256: str
    operation_id: str
    source_root: Path
    control_root: Path
    operation_directory: Path
    workspace_processed_root: Path
    staged_run_directory: Path
    final_directory: Path

    @property
    def intent_path(self) -> Path:
        return self.operation_directory / "intent.json"

    @property
    def ready_path(self) -> Path:
        return self.operation_directory / "ready.json"


@dataclass(frozen=True, slots=True)
class PublicationRecoveryReport:
    """Per-source result for stale private publication workspaces."""

    source_name: str
    operations_scanned: int
    operations_eligible: int
    operations_removed: int
    operations_in_grace: int
    published_residues_detected: int
    published_residues_removed: int
    action_sample: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _RecoveryEntry:
    relative_path: tuple[str, ...]
    kind: str
    size: int
    mtime_ns: int


@dataclass(frozen=True, slots=True)
class _PublicationOperationPlan:
    operation_directory: Path
    operation_id: str
    source_name: str | None
    run_sha256: str | None
    created_at: datetime | None
    latest_mtime_ns: int
    entries: tuple[_RecoveryEntry, ...]
    published_residue: bool


@dataclass(frozen=True, slots=True)
class WebPublicationRecoveryPlan:
    """Fully validated work-directory cleanup plan, before any removal starts."""

    processed_root: Path
    control_root: Path | None
    operations: tuple[_PublicationOperationPlan, ...]
    reports: tuple[PublicationRecoveryReport, ...]


def _canonical_json(payload: object) -> bytes:
    try:
        return json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise WebPublicationError("publication receipt contains invalid JSON values") from exc


def _with_content_sha256(payload: dict[str, Any]) -> dict[str, Any]:
    complete = dict(payload)
    complete["content_sha256"] = hashlib.sha256(_canonical_json(payload)).hexdigest()
    return complete


def _write_json_exclusive(path: Path, payload: dict[str, Any], *, label: str) -> str:
    raw = (
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )
    if len(raw) > _MAX_CONTROL_RECEIPT_BYTES:
        raise WebPublicationError(f"{label} exceeds its bounded receipt size")
    if _is_linklike(path):
        raise WebPublicationError(f"{label} target is a symlink or junction")
    try:
        with path.open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise WebPublicationError(f"{label} already exists and is immutable") from exc
    return hashlib.sha256(raw).hexdigest()


def _read_json_receipt(
    path: Path,
    *,
    label: str,
    fields: set[str],
    schema_version: str,
) -> tuple[dict[str, Any], str]:
    if _is_linklike(path) or not path.is_file():
        raise WebPublicationError(f"{label} is not a regular file")
    try:
        with path.open("rb") as handle:
            before = os.fstat(handle.fileno())
            if not stat_module.S_ISREG(before.st_mode):
                raise WebPublicationError(f"{label} is not a regular file")
            if before.st_size > _MAX_CONTROL_RECEIPT_BYTES:
                raise WebPublicationError(f"{label} exceeds its bounded receipt size")
            raw = handle.read(_MAX_CONTROL_RECEIPT_BYTES + 1)
            after = os.fstat(handle.fileno())
    except WebPublicationError:
        raise
    except OSError as exc:
        raise WebPublicationError(f"{label} cannot be read") from exc
    if len(raw) > _MAX_CONTROL_RECEIPT_BYTES:
        raise WebPublicationError(f"{label} exceeds its bounded receipt size")
    if (
        len(raw) != before.st_size
        or after.st_size != before.st_size
        or after.st_mtime_ns != before.st_mtime_ns
    ):
        raise WebPublicationError(f"{label} changed while being read")
    try:
        payload = json.loads(
            raw,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON value {value!r}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise WebPublicationError(f"{label} is unreadable") from exc
    if not isinstance(payload, dict) or set(payload) != fields:
        raise WebPublicationError(f"{label} fields are incomplete or unknown")
    if payload.get("schema_version") != schema_version:
        raise WebPublicationError(f"unsupported {label} schema")
    content_sha = payload.get("content_sha256")
    if not isinstance(content_sha, str) or _RUN_DIRECTORY_PATTERN.fullmatch(content_sha) is None:
        raise WebPublicationError(f"{label} has an invalid content hash")
    semantic = dict(payload)
    semantic.pop("content_sha256")
    if hashlib.sha256(_canonical_json(semantic)).hexdigest() != content_sha:
        raise WebPublicationError(f"{label} content hash does not match its fields")
    return payload, hashlib.sha256(raw).hexdigest()


def _aware_utc(value: datetime | None, *, label: str) -> datetime:
    resolved = value or datetime.now(UTC)
    if resolved.tzinfo is None or resolved.utcoffset() is None:
        raise ValueError(f"{label} must be timezone aware")
    return resolved.astimezone(UTC)


def _fsync_directory(path: Path) -> None:
    """Persist a directory entry where Python exposes a directory file descriptor."""

    if os.name == "nt":
        # CPython does not expose a supported way to open and FlushFileBuffers on directories.
        return
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _validate_publication_geometry(publication: WebProcessedPublication) -> None:
    if _validated_source_name(publication.source_name) != publication.source_name:
        raise WebPublicationError("publication source name is invalid")
    if _RUN_DIRECTORY_PATTERN.fullmatch(publication.run_sha256) is None:
        raise WebPublicationError("publication run must be an exact SHA-256")
    if _OPERATION_ID_PATTERN.fullmatch(publication.operation_id) is None:
        raise WebPublicationError("publication operation ID is invalid")
    processed_root = publication.processed_root.resolve(strict=True)
    source_root = _direct_source_root(
        processed_root,
        publication.source_name,
        label="processed root",
    )
    control_root = processed_root / WEB_PUBLICATION_CONTROL_DIRECTORY
    operation_directory = control_root / publication.operation_id
    workspace_root = operation_directory
    staged_run = workspace_root / publication.source_name / publication.run_sha256
    final_directory = source_root / publication.run_sha256
    expected = (
        (publication.source_root, source_root),
        (publication.control_root, control_root),
        (publication.operation_directory, operation_directory),
        (publication.workspace_processed_root, workspace_root),
        (publication.staged_run_directory, staged_run),
        (publication.final_directory, final_directory),
    )
    for supplied, calculated in expected:
        if Path(supplied).absolute() != calculated.absolute():
            raise WebPublicationError("publication paths do not match their immutable identity")
    for path, label in (
        (processed_root, "processed root"),
        (source_root, "processed source root"),
        (control_root, "publication control root"),
        (operation_directory, "publication operation directory"),
    ):
        if _is_linklike(path) or not path.is_dir():
            raise WebPublicationError(f"{label} is not a direct regular directory")
    if operation_directory.resolve(strict=True).parent != control_root.resolve(strict=True):
        raise WebPublicationError("publication operation escaped its control root")


def _validate_control_identity(
    publication: WebProcessedPublication,
    payload: dict[str, Any],
    *,
    label: str,
) -> None:
    if (
        payload.get("operation_id") != publication.operation_id
        or payload.get("source_name") != publication.source_name
        or payload.get("run_sha256") != publication.run_sha256
    ):
        raise WebPublicationError(f"{label} does not bind this exact publication")
    timestamp_field = "created_at" if label == "publication intent" else "sealed_at"
    timestamp = payload.get(timestamp_field)
    if not isinstance(timestamp, str):
        raise WebPublicationError(f"{label} has an invalid timestamp")
    try:
        parsed = datetime.fromisoformat(timestamp)
    except ValueError as exc:
        raise WebPublicationError(f"{label} has an invalid timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise WebPublicationError(f"{label} timestamp must be timezone aware")


def _validate_direct_run_directory(
    path: Path,
    *,
    expected_parent: Path,
    label: str,
) -> Path:
    if _is_linklike(path) or not path.is_dir():
        raise WebPublicationError(f"{label} is missing, link-like, or not a directory")
    try:
        resolved = path.resolve(strict=True)
        resolved_parent = expected_parent.resolve(strict=True)
    except OSError as exc:
        raise WebPublicationError(f"{label} cannot be resolved safely") from exc
    if resolved.parent != resolved_parent:
        raise WebPublicationError(f"{label} escaped its exact parent")
    return resolved


def _hash_regular_file(path: Path, *, maximum_bytes: int) -> tuple[str, int]:
    if _is_linklike(path) or not path.is_file():
        raise WebPublicationError(f"staged artifact is not a regular file: {path.name}")
    digest = hashlib.sha256()
    byte_count = 0
    try:
        # Windows' CRT ``_commit`` (used by ``os.fsync``) rejects a read-only descriptor.
        # Opening without truncation keeps the bytes immutable while making the flush portable.
        mode = "r+b" if os.name == "nt" else "rb"
        with path.open(mode) as handle:
            before = os.fstat(handle.fileno())
            if not stat_module.S_ISREG(before.st_mode) or before.st_size > maximum_bytes:
                raise WebPublicationError(f"staged artifact exceeds its limit: {path.name}")
            while chunk := handle.read(64 * 1024):
                byte_count += len(chunk)
                if byte_count > maximum_bytes:
                    raise WebPublicationError(f"staged artifact exceeds its limit: {path.name}")
                digest.update(chunk)
            after = os.fstat(handle.fileno())
            os.fsync(handle.fileno())
    except WebPublicationError:
        raise
    except OSError as exc:
        raise WebPublicationError(f"staged artifact cannot be read: {path.name}") from exc
    if (
        byte_count != before.st_size
        or after.st_size != before.st_size
        or after.st_mtime_ns != before.st_mtime_ns
    ):
        raise WebPublicationError(f"staged artifact changed while being sealed: {path.name}")
    return digest.hexdigest(), byte_count


def _sealed_file_manifest(run_directory: Path) -> dict[str, dict[str, object]]:
    names: set[str] = set()
    try:
        entries = os.scandir(run_directory)
    except OSError as exc:
        raise WebPublicationError("staged run cannot be inspected") from exc
    with entries:
        for entry in entries:
            child = Path(entry.path)
            if (
                entry.is_symlink()
                or _is_linklike(child)
                or not entry.is_file(follow_symlinks=False)
            ):
                raise WebPublicationError("staged run contains a link or non-file entry")
            if entry.name not in _ALLOWED_RUN_FILES:
                raise WebPublicationError("staged run contains an undeclared file")
            names.add(entry.name)
    if not names >= _REQUIRED_RUN_FILES or not names <= _ALLOWED_RUN_FILES:
        raise WebPublicationError("staged run is incomplete")
    files: dict[str, dict[str, object]] = {}
    total_bytes = 0
    for name in sorted(names):
        digest, byte_count = _hash_regular_file(
            run_directory / name,
            maximum_bytes=_MAX_ARTIFACT_BYTES,
        )
        total_bytes += byte_count
        if total_bytes > _MAX_RUN_BYTES:
            raise WebPublicationError("staged run exceeds its total byte limit")
        files[name] = {"sha256": digest, "byte_count": byte_count}
    _fsync_directory(run_directory)
    return files


def _validate_ready_files(payload: dict[str, Any]) -> dict[str, dict[str, object]]:
    files = payload.get("files")
    if not isinstance(files, dict):
        raise WebPublicationError("publication ready receipt has invalid files")
    names = set(files)
    if not names >= _REQUIRED_RUN_FILES or not names <= _ALLOWED_RUN_FILES:
        raise WebPublicationError("publication ready receipt has invalid files")
    total_bytes = 0
    for name, metadata in files.items():
        if not isinstance(name, str) or not isinstance(metadata, dict):
            raise WebPublicationError("publication ready receipt has invalid file metadata")
        if set(metadata) != _FILE_METADATA_FIELDS:
            raise WebPublicationError("publication ready receipt has invalid file metadata")
        digest = metadata.get("sha256")
        byte_count = metadata.get("byte_count")
        if not isinstance(digest, str) or _RUN_DIRECTORY_PATTERN.fullmatch(digest) is None:
            raise WebPublicationError("publication ready receipt has invalid file digest")
        if type(byte_count) is not int or not 0 <= byte_count <= _MAX_ARTIFACT_BYTES:
            raise WebPublicationError("publication ready receipt has invalid file size")
        total_bytes += byte_count
        if total_bytes > _MAX_RUN_BYTES:
            raise WebPublicationError("publication ready receipt exceeds the run byte limit")
    return files


def _validate_staged_receipt(run_directory: Path, *, source_name: str, run_sha256: str) -> None:
    try:
        _validate_processed_receipt(
            run_directory / WEB_PROCESSED_RETENTION_RECEIPT,
            source_name,
            run_sha256=run_sha256,
        )
    except WebPublicationError:
        raise
    except WebRetentionError as exc:
        raise WebPublicationError(str(exc)) from exc


def begin_web_processed_publication(
    *,
    processed_root: Path,
    source_name: str,
    run_sha256: str,
    created_at: datetime | None = None,
) -> WebProcessedPublication:
    """Claim a private same-filesystem workspace without creating the visible final run."""

    validated_source = _validated_source_name(source_name)
    if _RUN_DIRECTORY_PATTERN.fullmatch(run_sha256) is None:
        raise WebPublicationError("publication run must be an exact SHA-256")
    root_candidate = Path(processed_root)
    root_candidate.mkdir(parents=True, exist_ok=True)
    if _is_linklike(root_candidate) or not root_candidate.is_dir():
        raise WebPublicationError("processed root is not a direct regular directory")
    resolved_root = root_candidate.resolve(strict=True)
    source_candidate = resolved_root / validated_source
    source_candidate.mkdir(mode=0o700, exist_ok=True)
    source_root = _direct_source_root(
        resolved_root,
        validated_source,
        label="processed root",
    )
    final_directory = source_root / run_sha256
    if final_directory.exists() or _is_linklike(final_directory):
        raise WebPublicationError("governed-web processed run already exists and is immutable")
    control_root = resolved_root / WEB_PUBLICATION_CONTROL_DIRECTORY
    control_root.mkdir(mode=0o700, exist_ok=True)
    if _is_linklike(control_root) or control_root.resolve(strict=True).parent != resolved_root:
        raise WebPublicationError("publication control root escaped its processed root")

    operation_directory: Path | None = None
    operation_id = ""
    for _attempt in range(8):
        operation_id = secrets.token_hex(_OPERATION_ID_LENGTH // 2)
        candidate = control_root / operation_id
        try:
            candidate.mkdir(mode=0o700, exist_ok=False)
        except FileExistsError:
            continue
        operation_directory = candidate
        break
    if operation_directory is None:
        raise WebPublicationError("could not allocate a unique publication operation")

    workspace_root = operation_directory
    publication = WebProcessedPublication(
        processed_root=resolved_root,
        source_name=validated_source,
        run_sha256=run_sha256,
        operation_id=operation_id,
        source_root=source_root,
        control_root=control_root,
        operation_directory=operation_directory,
        workspace_processed_root=workspace_root,
        staged_run_directory=workspace_root / validated_source / run_sha256,
        final_directory=final_directory,
    )
    intent = _with_content_sha256(
        {
            "schema_version": WEB_PUBLICATION_INTENT_SCHEMA_VERSION,
            "operation_id": operation_id,
            "source_name": validated_source,
            "run_sha256": run_sha256,
            "created_at": _aware_utc(created_at, label="publication creation time").isoformat(),
        }
    )
    _write_json_exclusive(publication.intent_path, intent, label="publication intent")
    _fsync_directory(operation_directory)
    _fsync_directory(control_root)
    _validate_publication_geometry(publication)
    return publication


def seal_web_processed_publication(
    publication: WebProcessedPublication,
    *,
    sealed_at: datetime | None = None,
) -> Path:
    """Validate and durably seal a complete staged run before it can be published."""

    _validate_publication_geometry(publication)
    intent, intent_sha = _read_json_receipt(
        publication.intent_path,
        label="publication intent",
        fields=_INTENT_FIELDS,
        schema_version=WEB_PUBLICATION_INTENT_SCHEMA_VERSION,
    )
    _validate_control_identity(publication, intent, label="publication intent")
    if publication.final_directory.exists() or _is_linklike(publication.final_directory):
        raise WebPublicationError("governed-web processed run already exists and is immutable")
    _validate_direct_run_directory(
        publication.staged_run_directory,
        expected_parent=publication.workspace_processed_root / publication.source_name,
        label="staged run",
    )
    _validate_staged_receipt(
        publication.staged_run_directory,
        source_name=publication.source_name,
        run_sha256=publication.run_sha256,
    )
    files = _sealed_file_manifest(publication.staged_run_directory)
    payload = _with_content_sha256(
        {
            "schema_version": WEB_PUBLICATION_READY_SCHEMA_VERSION,
            "operation_id": publication.operation_id,
            "source_name": publication.source_name,
            "run_sha256": publication.run_sha256,
            "intent_sha256": intent_sha,
            "sealed_at": _aware_utc(sealed_at, label="publication seal time").isoformat(),
            "files": files,
        }
    )
    if publication.ready_path.exists() or _is_linklike(publication.ready_path):
        existing, _ready_sha = _read_json_receipt(
            publication.ready_path,
            label="publication ready receipt",
            fields=_READY_FIELDS,
            schema_version=WEB_PUBLICATION_READY_SCHEMA_VERSION,
        )
        _validate_control_identity(publication, existing, label="publication ready receipt")
        if existing.get("intent_sha256") != intent_sha or existing.get("files") != files:
            raise WebPublicationError("publication ready receipt conflicts with staged artifacts")
        return publication.ready_path
    _write_json_exclusive(
        publication.ready_path,
        payload,
        label="publication ready receipt",
    )
    _fsync_directory(publication.operation_directory)
    return publication.ready_path


def _rename_directory_noreplace(source: Path, destination: Path) -> None:
    """Atomically rename a directory without replacing an existing destination."""

    if os.name == "nt":
        os.rename(source, destination)
        return
    if not sys.platform.startswith("linux"):
        raise WebPublicationError("atomic no-replace publication is unsupported on this platform")
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise WebPublicationError("renameat2 is unavailable; publication fails closed")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    at_fdcwd = -100
    rename_noreplace = 1
    result = renameat2(
        at_fdcwd,
        os.fsencode(source),
        at_fdcwd,
        os.fsencode(destination),
        rename_noreplace,
    )
    if result != 0:
        error_number = ctypes.get_errno()
        if error_number == errno.EEXIST:
            raise FileExistsError(error_number, os.strerror(error_number), destination)
        raise OSError(error_number, os.strerror(error_number), destination)


def _clean_published_operation(publication: WebProcessedPublication) -> None:
    """Remove only the known, now-empty successful-operation control entries."""

    staged_source_root = publication.workspace_processed_root / publication.source_name
    staged_source_root.rmdir()
    publication.ready_path.unlink()
    publication.intent_path.unlink()
    publication.operation_directory.rmdir()
    _fsync_directory(publication.control_root)
    try:
        publication.control_root.rmdir()
    except OSError:
        # Other in-flight publishers legitimately keep the shared control root non-empty.
        return
    _fsync_directory(publication.processed_root)


def publish_web_processed_publication(publication: WebProcessedPublication) -> Path:
    """Atomically expose one sealed run without ever replacing an existing run."""

    _validate_publication_geometry(publication)
    intent, intent_sha = _read_json_receipt(
        publication.intent_path,
        label="publication intent",
        fields=_INTENT_FIELDS,
        schema_version=WEB_PUBLICATION_INTENT_SCHEMA_VERSION,
    )
    _validate_control_identity(publication, intent, label="publication intent")
    ready, _ready_sha = _read_json_receipt(
        publication.ready_path,
        label="publication ready receipt",
        fields=_READY_FIELDS,
        schema_version=WEB_PUBLICATION_READY_SCHEMA_VERSION,
    )
    _validate_control_identity(publication, ready, label="publication ready receipt")
    if ready.get("intent_sha256") != intent_sha:
        raise WebPublicationError("publication ready receipt does not bind its intent")
    expected_files = _validate_ready_files(ready)
    _validate_direct_run_directory(
        publication.staged_run_directory,
        expected_parent=publication.workspace_processed_root / publication.source_name,
        label="staged run",
    )
    actual_files = _sealed_file_manifest(publication.staged_run_directory)
    if actual_files != expected_files:
        raise WebPublicationError("staged run changed after it was sealed")
    _validate_staged_receipt(
        publication.staged_run_directory,
        source_name=publication.source_name,
        run_sha256=publication.run_sha256,
    )
    if publication.final_directory.exists() or _is_linklike(publication.final_directory):
        raise WebPublicationError("governed-web processed run already exists and is immutable")
    _validate_direct_run_directory(
        publication.staged_run_directory,
        expected_parent=publication.workspace_processed_root / publication.source_name,
        label="staged run",
    )
    if publication.staged_run_directory.stat().st_dev != publication.source_root.stat().st_dev:
        raise WebPublicationError("staged and final runs must be on the same filesystem")
    try:
        _rename_directory_noreplace(
            publication.staged_run_directory,
            publication.final_directory,
        )
    except FileExistsError as exc:
        raise WebPublicationError(
            "governed-web processed run already exists and was not overwritten"
        ) from exc
    except OSError as exc:
        raise WebPublicationError("atomic processed-run publication failed") from exc
    _validate_direct_run_directory(
        publication.final_directory,
        expected_parent=publication.source_root,
        label="published run",
    )
    _validate_staged_receipt(
        publication.final_directory,
        source_name=publication.source_name,
        run_sha256=publication.run_sha256,
    )
    _fsync_directory(publication.final_directory)
    _fsync_directory(publication.source_root)
    with suppress(OSError):
        # Publication has already committed and revalidated.  A later janitor pass may remove
        # the strictly named, non-catalogue-visible control residue.
        _clean_published_operation(publication)
    return publication.final_directory


def _consume_recovery_budget(
    work_budget: _RecoveryWorkBudget | None,
    *,
    label: str,
    amount: int = 1,
) -> None:
    if work_budget is not None:
        work_budget.consume(label=label, amount=amount)


def _recovery_directory(
    path: Path,
    *,
    expected_parent: Path,
    label: str,
) -> Path:
    """Resolve one direct non-link work-directory entry without following it."""

    if _is_linklike(path) or os.path.ismount(path) or not path.is_dir():
        raise WebPublicationError(f"{label} is not a direct regular directory")
    try:
        resolved = path.resolve(strict=True)
        parent = expected_parent.resolve(strict=True)
    except OSError as exc:
        raise WebPublicationError(f"{label} cannot be resolved safely") from exc
    if resolved.parent != parent:
        raise WebPublicationError(f"{label} escaped its exact parent")
    return resolved


def _recovery_entry(path: Path, *, relative_path: tuple[str, ...]) -> _RecoveryEntry:
    try:
        state = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise WebPublicationError(
            f"publication recovery entry cannot be inspected: {'/'.join(relative_path)}"
        ) from exc
    if _is_linklike(path):
        raise WebPublicationError(
            f"publication recovery entry is link-like: {'/'.join(relative_path)}"
        )
    if stat_module.S_ISREG(state.st_mode):
        kind = "file"
    elif stat_module.S_ISDIR(state.st_mode):
        kind = "directory"
    else:
        raise WebPublicationError(
            f"publication recovery entry is not regular: {'/'.join(relative_path)}"
        )
    return _RecoveryEntry(
        relative_path=relative_path,
        kind=kind,
        size=state.st_size,
        mtime_ns=state.st_mtime_ns,
    )


def _recovery_timestamp(value: object, *, label: str) -> datetime:
    if not isinstance(value, str):
        raise WebPublicationError(f"{label} has an invalid timestamp")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise WebPublicationError(f"{label} has an invalid timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise WebPublicationError(f"{label} timestamp must be timezone aware")
    return parsed.astimezone(UTC)


def _recoverable_stage_file_name(name: str) -> bool:
    if name in _ALLOWED_RUN_FILES:
        return True
    if not name.startswith(".") or not name.endswith(".tmp"):
        return False
    target, separator, token = name[1:-4].rpartition(".")
    return bool(
        separator
        and target in _ALLOWED_RUN_FILES
        and re.fullmatch(r"[A-Za-z0-9_-]{4,128}", token) is not None
    )


def _scan_stage_directory(
    stage_directory: Path,
    *,
    source_name: str,
    run_sha256: str,
    operation_directory: Path,
    work_budget: _RecoveryWorkBudget | None,
) -> tuple[_RecoveryEntry, ...]:
    resolved = _recovery_directory(
        stage_directory,
        expected_parent=operation_directory / source_name,
        label="publication staged run",
    )
    if resolved.name != run_sha256:
        raise WebPublicationError("publication staged run does not match its intent")
    entries: list[_RecoveryEntry] = [
        _recovery_entry(resolved, relative_path=(source_name, run_sha256))
    ]
    total_bytes = 0
    try:
        children = os.scandir(resolved)
    except OSError as exc:
        raise WebPublicationError("publication staged run cannot be inspected") from exc
    with children:
        for child in children:
            _consume_recovery_budget(
                work_budget,
                label=f"publication staged run {run_sha256!r}",
            )
            path = Path(child.path)
            if child.is_symlink() or _is_linklike(path) or not child.is_file(follow_symlinks=False):
                raise WebPublicationError(
                    "publication staged run contains a link or non-file entry"
                )
            if not _recoverable_stage_file_name(child.name):
                raise WebPublicationError(
                    f"publication staged run contains an unknown entry: {child.name!r}"
                )
            state = child.stat(follow_symlinks=False)
            if not stat_module.S_ISREG(state.st_mode) or state.st_size > _MAX_ARTIFACT_BYTES:
                raise WebPublicationError(
                    f"publication staged artifact exceeds its limit: {child.name}"
                )
            total_bytes += state.st_size
            if total_bytes > _MAX_RUN_BYTES:
                raise WebPublicationError("publication staged run exceeds its total byte limit")
            entries.append(
                _RecoveryEntry(
                    relative_path=(source_name, run_sha256, child.name),
                    kind="file",
                    size=state.st_size,
                    mtime_ns=state.st_mtime_ns,
                )
            )
    return tuple(entries)


def _scan_publication_operation(
    operation_directory: Path,
    *,
    processed_root: Path,
    control_root: Path,
    allowed_sources: frozenset[str],
    work_budget: _RecoveryWorkBudget | None,
) -> _PublicationOperationPlan:
    """Validate the bounded shapes left by every crash point in publication."""

    operation_id = operation_directory.name
    if _OPERATION_ID_PATTERN.fullmatch(operation_id) is None:
        raise WebPublicationError(f"publication control operation ID is unsafe: {operation_id!r}")
    resolved_operation = _recovery_directory(
        operation_directory,
        expected_parent=control_root,
        label="publication operation directory",
    )
    root_entry = _recovery_entry(resolved_operation, relative_path=())
    try:
        children = os.scandir(resolved_operation)
    except OSError as exc:
        raise WebPublicationError("publication operation directory cannot be inspected") from exc
    root_children: dict[str, Path] = {}
    with children:
        for child in children:
            _consume_recovery_budget(
                work_budget,
                label=f"publication operation {operation_id!r}",
            )
            path = Path(child.path)
            if child.is_symlink() or _is_linklike(path):
                raise WebPublicationError("publication operation contains a symlink or junction")
            root_children[child.name] = path

    if not root_children:
        return _PublicationOperationPlan(
            operation_directory=resolved_operation,
            operation_id=operation_id,
            source_name=None,
            run_sha256=None,
            created_at=None,
            latest_mtime_ns=root_entry.mtime_ns,
            entries=(),
            published_residue=False,
        )
    intent_path = root_children.get("intent.json")
    if intent_path is None:
        raise WebPublicationError("non-empty publication operation lacks its immutable intent")
    intent, intent_sha256 = _read_json_receipt(
        intent_path,
        label="publication intent",
        fields=_INTENT_FIELDS,
        schema_version=WEB_PUBLICATION_INTENT_SCHEMA_VERSION,
    )
    source_name = intent.get("source_name")
    run_sha256 = intent.get("run_sha256")
    if not isinstance(source_name, str) or _validated_source_name(source_name) != source_name:
        raise WebPublicationError("publication intent has an invalid source name")
    if source_name not in allowed_sources:
        raise WebPublicationError(
            f"publication operation source is not selected for maintenance: {source_name!r}"
        )
    if not isinstance(run_sha256, str) or _RUN_DIRECTORY_PATTERN.fullmatch(run_sha256) is None:
        raise WebPublicationError("publication intent has an invalid run digest")
    if intent.get("operation_id") != operation_id:
        raise WebPublicationError("publication intent does not bind its operation directory")
    created_at = _recovery_timestamp(intent.get("created_at"), label="publication intent")

    allowed_root_names = {"intent.json", "ready.json", source_name}
    unknown_root_names = sorted(set(root_children) - allowed_root_names)
    if unknown_root_names:
        raise WebPublicationError(
            "publication operation contains unknown root entries: "
            + ", ".join(repr(name) for name in unknown_root_names)
        )

    entries: list[_RecoveryEntry] = [root_entry]
    intent_entry = _recovery_entry(intent_path, relative_path=("intent.json",))
    entries.append(intent_entry)
    ready_path = root_children.get("ready.json")
    if ready_path is not None:
        ready, _ready_sha256 = _read_json_receipt(
            ready_path,
            label="publication ready receipt",
            fields=_READY_FIELDS,
            schema_version=WEB_PUBLICATION_READY_SCHEMA_VERSION,
        )
        if (
            ready.get("operation_id") != operation_id
            or ready.get("source_name") != source_name
            or ready.get("run_sha256") != run_sha256
            or ready.get("intent_sha256") != intent_sha256
        ):
            raise WebPublicationError("publication ready receipt does not bind its intent")
        _recovery_timestamp(ready.get("sealed_at"), label="publication ready receipt")
        _validate_ready_files(ready)
        entries.append(_recovery_entry(ready_path, relative_path=("ready.json",)))

    source_directory = root_children.get(source_name)
    staged_run_exists = False
    if source_directory is not None:
        resolved_source = _recovery_directory(
            source_directory,
            expected_parent=resolved_operation,
            label="publication staged source directory",
        )
        entries.append(_recovery_entry(resolved_source, relative_path=(source_name,)))
        try:
            source_children = os.scandir(resolved_source)
        except OSError as exc:
            raise WebPublicationError("publication staged source cannot be inspected") from exc
        children_by_name: dict[str, Path] = {}
        with source_children:
            for child in source_children:
                _consume_recovery_budget(
                    work_budget,
                    label=f"publication staged source {source_name!r}",
                )
                path = Path(child.path)
                if (
                    child.is_symlink()
                    or _is_linklike(path)
                    or not child.is_dir(follow_symlinks=False)
                ):
                    raise WebPublicationError(
                        "publication staged source contains a link or non-directory entry"
                    )
                children_by_name[child.name] = path
        if set(children_by_name) - {run_sha256}:
            raise WebPublicationError("publication staged source contains an unknown run directory")
        staged = children_by_name.get(run_sha256)
        if staged is not None:
            staged_run_exists = True
            entries.extend(
                _scan_stage_directory(
                    staged,
                    source_name=source_name,
                    run_sha256=run_sha256,
                    operation_directory=resolved_operation,
                    work_budget=work_budget,
                )
            )

    source_root = _direct_source_root(processed_root, source_name, label="processed root")
    if not source_root.exists():
        raise WebPublicationError("publication source root is missing")
    final_directory = source_root / run_sha256
    if final_directory.exists() or _is_linklike(final_directory):
        _recovery_directory(
            final_directory,
            expected_parent=source_root,
            label="published run directory",
        )
    latest_mtime_ns = max(entry.mtime_ns for entry in entries)
    return _PublicationOperationPlan(
        operation_directory=resolved_operation,
        operation_id=operation_id,
        source_name=source_name,
        run_sha256=run_sha256,
        created_at=created_at,
        latest_mtime_ns=latest_mtime_ns,
        entries=tuple(sorted(entries, key=lambda entry: entry.relative_path)),
        published_residue=ready_path is not None
        and final_directory.exists()
        and not staged_run_exists,
    )


def plan_web_publication_recovery(
    *,
    processed_root: Path,
    source_names: tuple[str, ...],
    now: datetime,
    orphan_grace: timedelta,
    work_budget: _RecoveryWorkBudget | None = None,
) -> WebPublicationRecoveryPlan:
    """Preflight every stale ``.wp`` operation before any retention deletion begins.

    A work directory is eligible only when both its content-integrity-checked intent timestamp and
    its most recently modified entry are older than the shared grace period. Unknown entries,
    link-like paths, and operations for unselected sources fail the entire retention run closed.
    """

    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("publication recovery time must be timezone aware")
    if orphan_grace < timedelta(hours=1) or orphan_grace > timedelta(days=30):
        raise ValueError("publication recovery grace must be between one hour and 30 days")
    validated_sources = tuple(dict.fromkeys(_validated_source_name(name) for name in source_names))
    source_counts = {
        source_name: {
            "operations_scanned": 0,
            "operations_eligible": 0,
            "operations_in_grace": 0,
            "published_residues_detected": 0,
        }
        for source_name in validated_sources
    }
    root = Path(processed_root)
    if not root.exists():
        return WebPublicationRecoveryPlan(
            processed_root=root.resolve(),
            control_root=None,
            operations=(),
            reports=tuple(
                PublicationRecoveryReport(
                    source_name=source_name,
                    **counts,
                    operations_removed=0,
                    published_residues_removed=0,
                    action_sample=(),
                )
                for source_name, counts in source_counts.items()
            ),
        )
    if _is_linklike(root) or not root.is_dir():
        raise WebPublicationError("processed root is not a direct regular directory")
    resolved_root = root.resolve(strict=True)
    control_root = resolved_root / WEB_PUBLICATION_CONTROL_DIRECTORY
    if not control_root.exists():
        return WebPublicationRecoveryPlan(
            processed_root=resolved_root,
            control_root=None,
            operations=(),
            reports=tuple(
                PublicationRecoveryReport(
                    source_name=source_name,
                    **counts,
                    operations_removed=0,
                    published_residues_removed=0,
                    action_sample=(),
                )
                for source_name, counts in source_counts.items()
            ),
        )
    resolved_control = _recovery_directory(
        control_root,
        expected_parent=resolved_root,
        label="publication control root",
    )
    cutoff = now.astimezone(UTC) - orphan_grace
    cutoff_ns = int(cutoff.timestamp() * 1_000_000_000)
    removals: list[_PublicationOperationPlan] = []
    actions: dict[str, list[str]] = {source_name: [] for source_name in validated_sources}
    try:
        operations = os.scandir(resolved_control)
    except OSError as exc:
        raise WebPublicationError("publication control root cannot be inspected") from exc
    with operations:
        for operation_entry in operations:
            _consume_recovery_budget(
                work_budget,
                label="publication control root",
            )
            operation_path = Path(operation_entry.path)
            if (
                operation_entry.is_symlink()
                or _is_linklike(operation_path)
                or not operation_entry.is_dir(follow_symlinks=False)
            ):
                raise WebPublicationError(
                    "publication control root contains a link or non-directory"
                )
            snapshot = _scan_publication_operation(
                operation_path,
                processed_root=resolved_root,
                control_root=resolved_control,
                allowed_sources=frozenset(validated_sources),
                work_budget=work_budget,
            )
            if snapshot.source_name is None:
                if snapshot.latest_mtime_ns <= cutoff_ns:
                    removals.append(snapshot)
                continue
            counts = source_counts[snapshot.source_name]
            counts["operations_scanned"] += 1
            if snapshot.published_residue:
                counts["published_residues_detected"] += 1
            stale = (
                snapshot.created_at is not None
                and snapshot.created_at <= cutoff
                and snapshot.latest_mtime_ns <= cutoff_ns
            )
            if stale:
                counts["operations_eligible"] += 1
                removals.append(snapshot)
                if len(actions[snapshot.source_name]) < 20:
                    actions[snapshot.source_name].append(
                        f"publication-operation:{snapshot.operation_id}"
                    )
            else:
                counts["operations_in_grace"] += 1
    for operation in removals:
        _consume_recovery_budget(
            work_budget,
            label=f"publication operation revalidation {operation.operation_id!r}",
            amount=max(1, len(operation.entries) + 1),
        )
    reports = tuple(
        PublicationRecoveryReport(
            source_name=source_name,
            operations_scanned=counts["operations_scanned"],
            operations_eligible=counts["operations_eligible"],
            operations_removed=0,
            operations_in_grace=counts["operations_in_grace"],
            published_residues_detected=counts["published_residues_detected"],
            published_residues_removed=0,
            action_sample=tuple(actions[source_name]),
        )
        for source_name, counts in source_counts.items()
    )
    return WebPublicationRecoveryPlan(
        processed_root=resolved_root,
        control_root=resolved_control,
        operations=tuple(removals),
        reports=reports,
    )


def _remove_recovered_publication_operation(
    operation: _PublicationOperationPlan,
    *,
    plan: WebPublicationRecoveryPlan,
) -> None:
    control_root = plan.control_root
    if control_root is None:
        raise WebPublicationError("publication recovery plan has no control root")
    expected_sources = (
        frozenset({operation.source_name}) if operation.source_name is not None else frozenset()
    )
    current = _scan_publication_operation(
        operation.operation_directory,
        processed_root=plan.processed_root,
        control_root=control_root,
        allowed_sources=expected_sources,
        work_budget=None,
    )
    if (
        current.source_name != operation.source_name
        or current.run_sha256 != operation.run_sha256
        or current.created_at != operation.created_at
        or current.entries != operation.entries
    ):
        raise WebPublicationError(
            f"publication operation changed before recovery deletion: {operation.operation_id}"
        )
    operation_directory = operation.operation_directory
    for entry in sorted(
        (
            item
            for item in operation.entries
            if item.kind == "file" and item.relative_path != ("intent.json",)
        ),
        key=lambda item: item.relative_path,
        reverse=True,
    ):
        path = operation_directory.joinpath(*entry.relative_path)
        if _is_linklike(path) or not path.is_file():
            raise WebPublicationError("publication operation changed before recovery deletion")
        path.unlink()
    for entry in sorted(
        (item for item in operation.entries if item.kind == "directory" and item.relative_path),
        key=lambda item: len(item.relative_path),
        reverse=True,
    ):
        path = operation_directory.joinpath(*entry.relative_path)
        if _is_linklike(path) or os.path.ismount(path) or not path.is_dir():
            raise WebPublicationError("publication operation changed before recovery deletion")
        path.rmdir()
    intent_path = operation_directory / "intent.json"
    if intent_path.exists():
        if _is_linklike(intent_path) or not intent_path.is_file():
            raise WebPublicationError("publication intent changed before recovery deletion")
        intent_path.unlink()
    if _is_linklike(operation_directory) or os.path.ismount(operation_directory):
        raise WebPublicationError("publication operation changed before recovery deletion")
    operation_directory.rmdir()


def execute_web_publication_recovery(
    plan: WebPublicationRecoveryPlan,
    *,
    dry_run: bool,
) -> tuple[PublicationRecoveryReport, ...]:
    """Apply a preflighted cleanup plan without ever traversing arbitrary directories."""

    removed_by_source: dict[str, int] = {}
    residue_removed_by_source: dict[str, int] = {}
    if not dry_run:
        for operation in plan.operations:
            _remove_recovered_publication_operation(operation, plan=plan)
            if operation.source_name is not None:
                removed_by_source[operation.source_name] = (
                    removed_by_source.get(operation.source_name, 0) + 1
                )
                if operation.published_residue:
                    residue_removed_by_source[operation.source_name] = (
                        residue_removed_by_source.get(operation.source_name, 0) + 1
                    )
        if plan.control_root is not None:
            try:
                plan.control_root.rmdir()
            except OSError:
                pass
            else:
                _fsync_directory(plan.processed_root)
    return tuple(
        replace(
            report,
            operations_removed=removed_by_source.get(report.source_name, 0),
            published_residues_removed=residue_removed_by_source.get(report.source_name, 0),
        )
        for report in plan.reports
    )


__all__ = [
    "WEB_PUBLICATION_CONTROL_DIRECTORY",
    "WEB_PUBLICATION_INTENT_SCHEMA_VERSION",
    "WEB_PUBLICATION_READY_SCHEMA_VERSION",
    "PublicationRecoveryReport",
    "WebProcessedPublication",
    "WebPublicationRecoveryPlan",
    "WebPublicationError",
    "begin_web_processed_publication",
    "execute_web_publication_recovery",
    "plan_web_publication_recovery",
    "publish_web_processed_publication",
    "seal_web_processed_publication",
]
