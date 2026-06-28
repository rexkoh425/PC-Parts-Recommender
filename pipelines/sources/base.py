"""Content-addressed raw snapshots shared by every source adapter."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

RAW_SNAPSHOT_SCHEMA_VERSION = "pc-build-recommender.raw-snapshot.v1"
_SOURCE_NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


class SnapshotError(RuntimeError):
    """Raised when a raw source cannot be snapshotted safely."""


class SnapshotTooLargeError(SnapshotError):
    """Raised before a source can exhaust the configured download budget."""


def utc_now() -> datetime:
    return datetime.now(UTC)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_source_name(source_name: str) -> None:
    if not _SOURCE_NAME_PATTERN.fullmatch(source_name):
        raise ValueError(
            "source_name must contain only lowercase letters, digits, underscores, or hyphens"
        )


def _normalise_suffix(suffix: str) -> str:
    if not suffix:
        return ".bin"
    value = suffix if suffix.startswith(".") else f".{suffix}"
    if not re.fullmatch(r"\.[A-Za-z0-9._-]+", value):
        raise ValueError(f"unsafe snapshot suffix: {suffix!r}")
    return value.lower()


@dataclass(frozen=True, slots=True)
class RawSnapshot:
    """Immutable metadata for one content-addressed raw source response."""

    source_name: str
    source_url: str
    source_type: str
    retrieved_at: datetime
    content_sha256: str
    byte_count: int
    media_type: str
    parser_version: str
    licence_or_access_note: str
    path: Path
    metadata_path: Path
    reused: bool = False

    def __post_init__(self) -> None:
        _validate_source_name(self.source_name)
        if len(self.content_sha256) != 64:
            raise ValueError("content_sha256 must be a SHA-256 digest")
        if self.byte_count < 0:
            raise ValueError("byte_count must not be negative")
        if self.retrieved_at.tzinfo is None:
            raise ValueError("retrieved_at must be timezone aware")

    def metadata(self) -> dict[str, object]:
        return {
            "schema_version": RAW_SNAPSHOT_SCHEMA_VERSION,
            "source_name": self.source_name,
            "source_url": self.source_url,
            "source_type": self.source_type,
            "retrieved_at": self.retrieved_at.isoformat(),
            "content_sha256": self.content_sha256,
            "byte_count": self.byte_count,
            "media_type": self.media_type,
            "parser_version": self.parser_version,
            "licence_or_access_note": self.licence_or_access_note,
            "raw_file": self.path.name,
        }


@dataclass(slots=True)
class ParsedBatch:
    """Accepted records and auditable rejections from a parser run."""

    source_name: str
    snapshot_sha256: str
    records: list[dict[str, Any]] = field(default_factory=list)
    rejected: list[dict[str, Any]] = field(default_factory=list)
    statistics: dict[str, Any] = field(default_factory=dict)

    @property
    def accepted_count(self) -> int:
        return len(self.records)

    @property
    def rejected_count(self) -> int:
        return len(self.rejected)


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            temporary_path = Path(handle.name)
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _snapshot_from_metadata(*, metadata_path: Path, raw_path: Path, reused: bool) -> RawSnapshot:
    with metadata_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("schema_version") != RAW_SNAPSHOT_SCHEMA_VERSION:
        raise SnapshotError(f"unsupported raw snapshot metadata: {metadata_path}")
    return RawSnapshot(
        source_name=str(payload["source_name"]),
        source_url=str(payload["source_url"]),
        source_type=str(payload["source_type"]),
        retrieved_at=datetime.fromisoformat(str(payload["retrieved_at"])),
        content_sha256=str(payload["content_sha256"]),
        byte_count=int(payload["byte_count"]),
        media_type=str(payload["media_type"]),
        parser_version=str(payload["parser_version"]),
        licence_or_access_note=str(payload["licence_or_access_note"]),
        path=raw_path,
        metadata_path=metadata_path,
        reused=reused,
    )


def _finish_snapshot(
    *,
    temporary_path: Path,
    raw_root: Path,
    source_name: str,
    source_url: str,
    source_type: str,
    content_sha256: str,
    byte_count: int,
    media_type: str,
    parser_version: str,
    licence_or_access_note: str,
    suffix: str,
    retrieved_at: datetime,
) -> RawSnapshot:
    source_root = raw_root / source_name
    raw_path = source_root / f"{content_sha256}{suffix}"
    metadata_path = source_root / f"{content_sha256}{suffix}.metadata.json"
    reused = raw_path.exists()
    if reused:
        if raw_path.stat().st_size != byte_count or sha256_file(raw_path) != content_sha256:
            raise SnapshotError(f"existing content-addressed snapshot is corrupt: {raw_path}")
        temporary_path.unlink()
    else:
        os.replace(temporary_path, raw_path)

    snapshot = RawSnapshot(
        source_name=source_name,
        source_url=source_url,
        source_type=source_type,
        retrieved_at=retrieved_at,
        content_sha256=content_sha256,
        byte_count=byte_count,
        media_type=media_type,
        parser_version=parser_version,
        licence_or_access_note=licence_or_access_note,
        path=raw_path,
        metadata_path=metadata_path,
        reused=reused,
    )
    if metadata_path.exists():
        persisted = _snapshot_from_metadata(
            metadata_path=metadata_path, raw_path=raw_path, reused=True
        )
        if persisted.content_sha256 != content_sha256:
            raise SnapshotError(f"snapshot metadata hash mismatch: {metadata_path}")
        return persisted
    _write_json_atomic(metadata_path, snapshot.metadata())
    return snapshot


def fetch_http_snapshot(
    *,
    source_name: str,
    source_url: str,
    source_type: str,
    raw_root: str | Path,
    parser_version: str,
    licence_or_access_note: str,
    suffix: str,
    expected_sha256: str | None = None,
    maximum_bytes: int | None = None,
    timeout_seconds: float = 180.0,
    headers: Mapping[str, str] | None = None,
) -> RawSnapshot:
    """Stream an HTTP response into immutable content-addressed storage."""

    _validate_source_name(source_name)
    snapshot_suffix = _normalise_suffix(suffix)
    destination = Path(raw_root) / source_name
    destination.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    retrieved_at = utc_now()
    digest = hashlib.sha256()
    byte_count = 0
    media_type = "application/octet-stream"
    request_headers = {
        "User-Agent": "pc-build-recommender-ingestion/0.1 (+data provenance)",
        **dict(headers or {}),
    }

    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination,
            prefix=".download.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            with (
                httpx.Client(
                    follow_redirects=True,
                    timeout=httpx.Timeout(timeout_seconds),
                    headers=request_headers,
                ) as client,
                client.stream("GET", source_url) as response,
            ):
                response.raise_for_status()
                media_type = response.headers.get("content-type", media_type).split(";")[0]
                declared_length = response.headers.get("content-length")
                if (
                    maximum_bytes is not None
                    and declared_length is not None
                    and int(declared_length) > maximum_bytes
                ):
                    raise SnapshotTooLargeError(
                        f"{source_name} declared {declared_length} bytes; limit is {maximum_bytes}"
                    )
                for chunk in response.iter_bytes(chunk_size=1024 * 1024):
                    byte_count += len(chunk)
                    if maximum_bytes is not None and byte_count > maximum_bytes:
                        raise SnapshotTooLargeError(f"{source_name} exceeded {maximum_bytes} bytes")
                    digest.update(chunk)
                    handle.write(chunk)

        content_sha256 = digest.hexdigest()
        if expected_sha256 is not None and content_sha256 != expected_sha256.lower():
            raise SnapshotError(
                f"{source_name} SHA-256 mismatch: expected {expected_sha256}, "
                f"received {content_sha256}"
            )
        assert temporary_path is not None
        completed = _finish_snapshot(
            temporary_path=temporary_path,
            raw_root=Path(raw_root),
            source_name=source_name,
            source_url=source_url,
            source_type=source_type,
            content_sha256=content_sha256,
            byte_count=byte_count,
            media_type=media_type,
            parser_version=parser_version,
            licence_or_access_note=licence_or_access_note,
            suffix=snapshot_suffix,
            retrieved_at=retrieved_at,
        )
        temporary_path = None
        return completed
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def snapshot_local_file(
    *,
    source_name: str,
    source_url: str,
    source_type: str,
    source_path: str | Path,
    raw_root: str | Path,
    parser_version: str,
    licence_or_access_note: str,
    suffix: str | None = None,
    media_type: str = "application/octet-stream",
    expected_sha256: str | None = None,
    maximum_bytes: int | None = None,
) -> RawSnapshot:
    """Copy a controlled local input into the same immutable raw-snapshot contract."""

    _validate_source_name(source_name)
    source_candidate = Path(source_path)
    is_junction = getattr(os.path, "isjunction", None)
    if source_candidate.is_symlink() or bool(
        is_junction is not None and is_junction(source_candidate)
    ):
        raise SnapshotError("local snapshot source must not be a symlink or junction")
    input_path = source_candidate.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if maximum_bytes is not None and maximum_bytes < 1:
        raise ValueError("maximum_bytes must be positive when provided")
    snapshot_suffix = _normalise_suffix(suffix or input_path.suffix)
    destination = Path(raw_root) / source_name
    destination.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    digest = hashlib.sha256()
    byte_count = 0
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination,
            prefix=".copy.",
            suffix=".tmp",
            delete=False,
        ) as output:
            temporary_path = Path(output.name)
            with input_path.open("rb") as source:
                before = os.fstat(source.fileno())
                if not stat.S_ISREG(before.st_mode):
                    raise SnapshotError("local snapshot source must be a regular file")
                if maximum_bytes is not None and before.st_size > maximum_bytes:
                    raise SnapshotTooLargeError(
                        f"{source_name} declared {before.st_size} bytes; limit is {maximum_bytes}"
                    )
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    byte_count += len(chunk)
                    if maximum_bytes is not None and byte_count > maximum_bytes:
                        raise SnapshotTooLargeError(f"{source_name} exceeded {maximum_bytes} bytes")
                    digest.update(chunk)
                    output.write(chunk)
                after = os.fstat(source.fileno())
                if before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns:
                    raise SnapshotError("local snapshot source changed while it was copied")
        content_sha256 = digest.hexdigest()
        if expected_sha256 is not None and content_sha256 != expected_sha256.lower():
            raise SnapshotError(
                f"{source_name} SHA-256 mismatch: expected {expected_sha256}, "
                f"received {content_sha256}"
            )
        assert temporary_path is not None
        completed = _finish_snapshot(
            temporary_path=temporary_path,
            raw_root=Path(raw_root),
            source_name=source_name,
            source_url=source_url,
            source_type=source_type,
            content_sha256=content_sha256,
            byte_count=byte_count,
            media_type=media_type,
            parser_version=parser_version,
            licence_or_access_note=licence_or_access_note,
            suffix=snapshot_suffix,
            retrieved_at=utc_now(),
        )
        temporary_path = None
        return completed
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def rejected_record(record_id: str, reason: str, **details: object) -> dict[str, object]:
    return {"record_id": record_id, "reason": reason, "details": details}


def count_by(records: Iterable[Mapping[str, Any]], field_name: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        value = str(record.get(field_name, "unknown"))
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))
