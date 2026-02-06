"""Crash-safe, bounded JSONL publication for streaming source adapters."""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import os
import re
import secrets
import shutil
import sys
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pipelines.sources.base import sha256_file

STREAMING_MANIFEST_SCHEMA_VERSION = "pc-build-recommender.streaming-processed-batch.v1"
_SOURCE_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CONTROL_DIRECTORY = ".streaming-publication"
_MAX_MANIFEST_BYTES = 4 * 1024 * 1024
_MANIFEST_FIELDS = {
    "schema_version",
    "source_name",
    "run_sha256",
    "accepted_count",
    "rejected_count",
    "files",
    "metadata",
    "content_sha256",
}


class StreamingPublicationError(RuntimeError):
    """Raised when a streaming artifact cannot be published safely."""


@dataclass(frozen=True, slots=True)
class StreamingProcessedArtifacts:
    output_directory: Path
    records_jsonl: Path
    rejections_jsonl: Path
    manifest_json: Path
    quality_json: Path
    accepted_count: int
    rejected_count: int
    reused: bool = False


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise StreamingPublicationError("streaming artifact contains invalid JSON") from exc


def _write_json_exclusive(path: Path, value: object) -> None:
    payload = (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _safe_root(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or not path.is_dir():
        raise StreamingPublicationError("processed root must be a regular directory")
    return path.resolve(strict=True)


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
    """Atomically expose a directory without replacing a competing run."""

    if os.name == "nt":
        os.rename(source, destination)
        return
    if not sys.platform.startswith("linux"):
        raise StreamingPublicationError(
            "atomic no-replace publication is unsupported on this platform"
        )
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise StreamingPublicationError(
            "renameat2 is unavailable; streaming publication fails closed"
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
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise FileExistsError(error_number, os.strerror(error_number), destination)
    raise OSError(error_number, os.strerror(error_number), destination)


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise StreamingPublicationError(
                f"duplicate streaming manifest key is forbidden: {key!r}"
            )
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise StreamingPublicationError(f"non-finite streaming manifest value: {value}")


class AtomicStreamingJSONLWriter:
    """Write records one at a time and expose the run only after it is sealed."""

    def __init__(
        self,
        *,
        processed_root: str | Path,
        source_name: str,
        run_sha256: str,
        maximum_output_bytes: int,
        maximum_record_bytes: int = 2 * 1024 * 1024,
    ) -> None:
        if _SOURCE_NAME.fullmatch(source_name) is None:
            raise ValueError("source_name must be a lowercase slug")
        if _SHA256.fullmatch(run_sha256) is None:
            raise ValueError("run_sha256 must be an exact lowercase SHA-256")
        if type(maximum_output_bytes) is not int or maximum_output_bytes < 1:
            raise ValueError("maximum_output_bytes must be a positive integer")
        if type(maximum_record_bytes) is not int or not 1 <= maximum_record_bytes <= (
            maximum_output_bytes
        ):
            raise ValueError("maximum_record_bytes must fit within maximum_output_bytes")
        self.processed_root = _safe_root(Path(processed_root))
        self.source_name = source_name
        self.run_sha256 = run_sha256
        self.maximum_output_bytes = maximum_output_bytes
        self.maximum_record_bytes = maximum_record_bytes
        self.source_root = self.processed_root / source_name
        self.source_root.mkdir(mode=0o700, exist_ok=True)
        if self.source_root.is_symlink() or self.source_root.resolve(strict=True).parent != (
            self.processed_root
        ):
            raise StreamingPublicationError("processed source root escaped its parent")
        self.final_directory = self.source_root / run_sha256
        self.control_root = self.processed_root / _CONTROL_DIRECTORY
        self.operation_directory: Path | None = None
        self.staged_directory: Path | None = None
        self._records_handle: Any = None
        self._rejections_handle: Any = None
        self._records_digest = hashlib.sha256()
        self._rejections_digest = hashlib.sha256()
        self._output_bytes = 0
        self.accepted_count = 0
        self.rejected_count = 0
        self.artifacts: StreamingProcessedArtifacts | None = None
        self._sealed = False

    def __enter__(self) -> AtomicStreamingJSONLWriter:
        if self.final_directory.exists():
            raise FileExistsError(self.final_directory)
        self.control_root.mkdir(mode=0o700, exist_ok=True)
        if self.control_root.is_symlink() or self.control_root.resolve(strict=True).parent != (
            self.processed_root
        ):
            raise StreamingPublicationError("streaming control root escaped its parent")
        operation = self.control_root / secrets.token_hex(12)
        operation.mkdir(mode=0o700, exist_ok=False)
        staged = operation / self.source_name / self.run_sha256
        staged.mkdir(parents=True, mode=0o700)
        self.operation_directory = operation
        self.staged_directory = staged
        self._records_handle = (staged / "records.jsonl.part").open("xb")
        self._rejections_handle = (staged / "rejections.jsonl.part").open("xb")
        return self

    def _write(self, value: Mapping[str, Any], *, rejection: bool) -> None:
        if self.staged_directory is None or self._sealed:
            raise StreamingPublicationError("streaming writer is not open")
        payload = _canonical_json(value) + b"\n"
        if len(payload) > self.maximum_record_bytes:
            raise StreamingPublicationError(
                f"streaming record exceeds {self.maximum_record_bytes} bytes"
            )
        self._output_bytes += len(payload)
        if self._output_bytes > self.maximum_output_bytes:
            raise StreamingPublicationError(
                f"streaming output exceeds {self.maximum_output_bytes} bytes"
            )
        if rejection:
            self._rejections_handle.write(payload)
            self._rejections_digest.update(payload)
            self.rejected_count += 1
        else:
            self._records_handle.write(payload)
            self._records_digest.update(payload)
            self.accepted_count += 1

    def write_record(self, record: Mapping[str, Any]) -> None:
        self._write(record, rejection=False)

    def write_rejection(self, rejection: Mapping[str, Any]) -> None:
        self._write(rejection, rejection=True)

    def seal(
        self,
        *,
        quality_report: Mapping[str, Any],
        manifest_metadata: Mapping[str, Any],
    ) -> StreamingProcessedArtifacts:
        staged = self.staged_directory
        operation = self.operation_directory
        if staged is None or operation is None or self._sealed:
            raise StreamingPublicationError("streaming writer cannot be sealed")
        for handle in (self._records_handle, self._rejections_handle):
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
        records_path = staged / "records.jsonl"
        rejections_path = staged / "rejections.jsonl"
        os.replace(staged / "records.jsonl.part", records_path)
        os.replace(staged / "rejections.jsonl.part", rejections_path)
        quality_path = staged / "data-quality.json"
        _write_json_exclusive(quality_path, quality_report)
        files = {
            "records.jsonl": {
                "sha256": self._records_digest.hexdigest(),
                "byte_count": records_path.stat().st_size,
            },
            "rejections.jsonl": {
                "sha256": self._rejections_digest.hexdigest(),
                "byte_count": rejections_path.stat().st_size,
            },
            "data-quality.json": {
                "sha256": sha256_file(quality_path),
                "byte_count": quality_path.stat().st_size,
            },
        }
        semantic_manifest: dict[str, Any] = {
            "schema_version": STREAMING_MANIFEST_SCHEMA_VERSION,
            "source_name": self.source_name,
            "run_sha256": self.run_sha256,
            "accepted_count": self.accepted_count,
            "rejected_count": self.rejected_count,
            "files": files,
            "metadata": dict(manifest_metadata),
        }
        semantic_manifest["content_sha256"] = hashlib.sha256(
            _canonical_json(semantic_manifest)
        ).hexdigest()
        manifest_path = staged / "manifest.json"
        _write_json_exclusive(manifest_path, semantic_manifest)
        _fsync_directory(staged)
        if self.final_directory.exists():
            raise FileExistsError(self.final_directory)
        try:
            _rename_directory_noreplace(staged, self.final_directory)
        except OSError as exc:
            raise StreamingPublicationError("atomic streaming publication failed") from exc
        _fsync_directory(self.source_root)
        self._sealed = True
        staged_source = operation / self.source_name
        staged_source.rmdir()
        operation.rmdir()
        with suppress(OSError):
            self.control_root.rmdir()
        _fsync_directory(self.processed_root)
        self.artifacts = StreamingProcessedArtifacts(
            output_directory=self.final_directory,
            records_jsonl=self.final_directory / "records.jsonl",
            rejections_jsonl=self.final_directory / "rejections.jsonl",
            manifest_json=self.final_directory / "manifest.json",
            quality_json=self.final_directory / "data-quality.json",
            accepted_count=self.accepted_count,
            rejected_count=self.rejected_count,
        )
        return self.artifacts

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        for handle in (self._records_handle, self._rejections_handle):
            if handle is not None and not handle.closed:
                handle.close()
        if not self._sealed and self.operation_directory is not None:
            operation = self.operation_directory
            try:
                if operation.resolve(strict=False).parent == self.control_root.resolve(strict=True):
                    shutil.rmtree(operation)
            except OSError:
                pass
            with suppress(OSError):
                self.control_root.rmdir()


def load_existing_streaming_artifacts(
    *,
    processed_root: str | Path,
    source_name: str,
    run_sha256: str,
    expected_metadata: Mapping[str, Any],
) -> StreamingProcessedArtifacts | None:
    """Return a fully revalidated immutable run or ``None`` when it does not exist."""

    root = Path(processed_root)
    run = root / source_name / run_sha256
    if not run.exists():
        return None
    if run.is_symlink() or not run.is_dir():
        raise StreamingPublicationError("existing streaming run is not a regular directory")
    manifest_path = run / "manifest.json"
    try:
        if manifest_path.is_symlink() or not manifest_path.is_file():
            raise StreamingPublicationError("existing streaming manifest is not a regular file")
        if manifest_path.stat().st_size > _MAX_MANIFEST_BYTES:
            raise StreamingPublicationError("existing streaming manifest is too large")
        payload = json.loads(
            manifest_path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except StreamingPublicationError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StreamingPublicationError("existing streaming manifest is unreadable") from exc
    if not isinstance(payload, dict) or set(payload) != _MANIFEST_FIELDS:
        raise StreamingPublicationError("existing streaming manifest schema is invalid")
    content_sha256 = payload.pop("content_sha256", None)
    if content_sha256 != hashlib.sha256(_canonical_json(payload)).hexdigest():
        raise StreamingPublicationError("existing streaming manifest content hash mismatch")
    metadata_payload = payload.get("metadata")
    if (
        payload.get("schema_version") != STREAMING_MANIFEST_SCHEMA_VERSION
        or payload.get("source_name") != source_name
        or payload.get("run_sha256") != run_sha256
        or not isinstance(metadata_payload, dict)
        or any(metadata_payload.get(key) != value for key, value in expected_metadata.items())
    ):
        raise StreamingPublicationError("existing streaming manifest identity mismatch")
    files = payload.get("files")
    if not isinstance(files, dict) or set(files) != {
        "records.jsonl",
        "rejections.jsonl",
        "data-quality.json",
    }:
        raise StreamingPublicationError("existing streaming file manifest is incomplete")
    for name, metadata in files.items():
        path = run / name
        if not isinstance(metadata, dict) or path.is_symlink() or not path.is_file():
            raise StreamingPublicationError("existing streaming artifact is invalid")
        if metadata != {"sha256": sha256_file(path), "byte_count": path.stat().st_size}:
            raise StreamingPublicationError("existing streaming artifact hash mismatch")
    accepted = payload.get("accepted_count")
    rejected = payload.get("rejected_count")
    if type(accepted) is not int or type(rejected) is not int or accepted < 0 or rejected < 0:
        raise StreamingPublicationError("existing streaming counts are invalid")
    return StreamingProcessedArtifacts(
        output_directory=run,
        records_jsonl=run / "records.jsonl",
        rejections_jsonl=run / "rejections.jsonl",
        manifest_json=run / "manifest.json",
        quality_json=run / "data-quality.json",
        accepted_count=accepted,
        rejected_count=rejected,
        reused=True,
    )


__all__ = [
    "AtomicStreamingJSONLWriter",
    "STREAMING_MANIFEST_SCHEMA_VERSION",
    "StreamingProcessedArtifacts",
    "StreamingPublicationError",
    "load_existing_streaming_artifacts",
]
