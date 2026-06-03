"""Aggregate-safe receipts for instrumented pipeline operations.

The serving API must not receive Dagster's database credentials or a mount of raw
source data merely to answer whether an ingestion operation failed.  Pipelines
therefore write small, separately named receipts to a dedicated directory. The API reads
only bounded, schema-validated aggregate counts from a read-only mount of it.

These are *instrumented operation* events, not a substitute for Dagster's run
storage.  Scheduler, queue, and worker failures that happen before user-code
starts remain visible only in Dagster's authenticated control plane.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal
from uuid import uuid4

PIPELINE_OPERATION_EVENT_SCHEMA_VERSION = "pc-build-recommender.pipeline-operation-event.v1"
_EVENT_DIRECTORY = "events"
_MAX_EVENT_BYTES = 4 * 1024
_MAX_EVENTS_TO_READ = 1_000
_OPERATION_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,99}$")
_FAILURE_CLASS_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_.]{0,119}$")

OperationStatus = Literal["succeeded", "failed"]


@dataclass(frozen=True, slots=True)
class PipelineOperationEvent:
    """One safe-to-mount outcome from code that ran inside a pipeline process."""

    event_id: str
    operation_name: str
    status: OperationStatus
    finished_at: datetime
    failure_class: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        return {
            "schema_version": PIPELINE_OPERATION_EVENT_SCHEMA_VERSION,
            "event_id": self.event_id,
            "operation_name": self.operation_name,
            "status": self.status,
            "finished_at": self.finished_at.isoformat(),
            "failure_class": self.failure_class,
        }


@dataclass(frozen=True, slots=True)
class PipelineOperationSummary:
    """Bounded aggregate view suitable for an authenticated serving response."""

    available: bool
    event_count: int = 0
    succeeded_count: int = 0
    failed_count: int = 0
    latest_event_at: datetime | None = None
    latest_failure_at: datetime | None = None
    invalid_receipt_count: int = 0
    truncated: bool = False


def _is_linklike(path: Path) -> bool:
    return path.is_symlink() or path.is_junction()


def _ensure_regular_directory(path: Path, *, create: bool) -> Path:
    if path.exists():
        if _is_linklike(path) or not path.is_dir():
            raise ValueError("pipeline operation path must be a non-link directory")
        return path
    if not create:
        raise FileNotFoundError(path)
    path.mkdir(parents=True, exist_ok=False)
    return path


def _event_directory(root: Path, *, create: bool) -> Path:
    safe_root = _ensure_regular_directory(root, create=create)
    return _ensure_regular_directory(safe_root / _EVENT_DIRECTORY, create=create)


def _validate_operation_name(operation_name: str) -> str:
    if not _OPERATION_NAME_PATTERN.fullmatch(operation_name):
        raise ValueError("operation_name must be a lowercase snake_case identifier")
    return operation_name


def _failure_class(error: BaseException) -> str:
    value = f"{type(error).__module__}.{type(error).__qualname__}"
    if not _FAILURE_CLASS_PATTERN.fullmatch(value):
        return "builtins.Exception"
    return value


def _write_json_no_replace(path: Path, payload: dict[str, str | None]) -> None:
    temporary_name: str | None = None
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
            json.dump(payload, handle, allow_nan=False, separators=(",", ":"), sort_keys=True)
            handle.write("\n")
            temporary_name = handle.name
        if path.exists():
            raise FileExistsError(path)
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def write_pipeline_operation_event(
    root: Path,
    *,
    operation_name: str,
    status: OperationStatus,
    finished_at: datetime | None = None,
    failure_class: str | None = None,
) -> Path:
    """Write one distinct receipt without recording source URLs or exception text."""

    _validate_operation_name(operation_name)
    if status not in {"succeeded", "failed"}:
        raise ValueError("status must be succeeded or failed")
    if status == "succeeded" and failure_class is not None:
        raise ValueError("successful operation receipts cannot contain failure_class")
    if status == "failed" and (
        failure_class is None or not _FAILURE_CLASS_PATTERN.fullmatch(failure_class)
    ):
        raise ValueError("failed operation receipts require a safe failure_class")

    timestamp = (finished_at or datetime.now(UTC)).astimezone(UTC)
    event = PipelineOperationEvent(
        event_id=uuid4().hex,
        operation_name=operation_name,
        status=status,
        finished_at=timestamp,
        failure_class=failure_class,
    )
    timestamp_prefix = event.finished_at.strftime("%Y%m%dT%H%M%S%fZ")
    destination = _event_directory(Path(root), create=True) / (
        f"{timestamp_prefix}-{event.event_id}.json"
    )
    _write_json_no_replace(destination, event.to_dict())
    return destination


@contextmanager
def record_pipeline_operation(root: Path | None, operation_name: str) -> Iterator[None]:
    """Record the final outcome of a pipeline operation when observation is configured."""

    _validate_operation_name(operation_name)
    try:
        yield
    except BaseException as error:
        if root is not None:
            write_pipeline_operation_event(
                root,
                operation_name=operation_name,
                status="failed",
                failure_class=_failure_class(error),
            )
        raise
    else:
        if root is not None:
            write_pipeline_operation_event(
                root,
                operation_name=operation_name,
                status="succeeded",
            )


def _parse_event(path: Path) -> PipelineOperationEvent:
    if _is_linklike(path) or not path.is_file() or path.stat().st_size > _MAX_EVENT_BYTES:
        raise ValueError("invalid pipeline operation receipt file")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version",
        "event_id",
        "operation_name",
        "status",
        "finished_at",
        "failure_class",
    }:
        raise ValueError("pipeline operation receipt schema is invalid")
    if payload["schema_version"] != PIPELINE_OPERATION_EVENT_SCHEMA_VERSION:
        raise ValueError("unsupported pipeline operation receipt schema")
    event_id = payload["event_id"]
    operation_name = payload["operation_name"]
    status = payload["status"]
    failure_class = payload["failure_class"]
    finished_at = payload["finished_at"]
    if (
        not isinstance(event_id, str)
        or len(event_id) != 32
        or any(character not in "0123456789abcdef" for character in event_id)
        or not isinstance(operation_name, str)
        or not _OPERATION_NAME_PATTERN.fullmatch(operation_name)
        or status not in {"succeeded", "failed"}
        or not isinstance(finished_at, str)
    ):
        raise ValueError("pipeline operation receipt values are invalid")
    if status == "succeeded" and failure_class is not None:
        raise ValueError("successful pipeline operation receipt contains failure_class")
    if status == "failed" and (
        not isinstance(failure_class, str) or not _FAILURE_CLASS_PATTERN.fullmatch(failure_class)
    ):
        raise ValueError("failed pipeline operation receipt has invalid failure_class")
    try:
        parsed_time = datetime.fromisoformat(finished_at.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("pipeline operation receipt has invalid finished_at") from error
    if parsed_time.tzinfo is None:
        raise ValueError("pipeline operation receipt finished_at must include timezone")
    return PipelineOperationEvent(
        event_id=event_id,
        operation_name=operation_name,
        status=status,
        finished_at=parsed_time.astimezone(UTC),
        failure_class=failure_class,
    )


def summarize_pipeline_operations(
    root: Path | None,
    *,
    now: datetime | None = None,
    window: timedelta = timedelta(days=7),
) -> PipelineOperationSummary | None:
    """Read at most 1,000 immutable receipts and return aggregate recent outcomes.

    ``None`` means no receipt mount was configured.  A configured but absent or malformed
    directory returns an unavailable summary; callers must not infer that as a healthy pipeline.
    """

    if root is None:
        return None
    if window <= timedelta(0):
        raise ValueError("window must be positive")
    try:
        events = _event_directory(Path(root), create=False)
    except (FileNotFoundError, OSError, ValueError):
        return PipelineOperationSummary(available=False)

    try:
        candidates = sorted(
            (path for path in events.iterdir() if path.suffix == ".json"),
            key=lambda path: path.name,
            reverse=True,
        )
    except OSError:
        return PipelineOperationSummary(available=False)
    truncated = len(candidates) > _MAX_EVENTS_TO_READ
    selected = candidates[:_MAX_EVENTS_TO_READ]
    cutoff = (now or datetime.now(UTC)).astimezone(UTC) - window
    valid: list[PipelineOperationEvent] = []
    invalid_count = 0
    for path in selected:
        try:
            event = _parse_event(path)
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            invalid_count += 1
            continue
        if event.finished_at >= cutoff:
            valid.append(event)
    succeeded = sum(event.status == "succeeded" for event in valid)
    failures = [event for event in valid if event.status == "failed"]
    return PipelineOperationSummary(
        available=True,
        event_count=len(valid),
        succeeded_count=succeeded,
        failed_count=len(failures),
        latest_event_at=max((event.finished_at for event in valid), default=None),
        latest_failure_at=max((event.finished_at for event in failures), default=None),
        invalid_receipt_count=invalid_count,
        truncated=truncated,
    )
