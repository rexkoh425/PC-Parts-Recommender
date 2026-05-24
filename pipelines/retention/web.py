"""Policy-independent retention maintenance for governed-web artifacts.

Acquisition policies deliberately cannot be constructed after their authority expires.  This
module therefore operates only from immutable per-artifact receipts and an explicitly named
source root.  It validates every destructive target before applying a narrowly scoped plan.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import secrets
import stat as stat_module
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final
from urllib.parse import urlsplit

if TYPE_CHECKING:
    from pipelines.sources.web_product import WebSourcePolicy

WEB_PROCESSED_RETENTION_SCHEMA_VERSION: Final = "pc-build-recommender.web-processed-retention.v2"
WEB_PROCESSED_RETENTION_RECEIPT: Final = "retention-receipt.json"
WEB_RAW_METADATA_SCHEMA_VERSION: Final = "pc-build-recommender.web-raw-page.v2"
WEB_CRAWL_CACHE_FILE: Final = "http-cache.json"
WEB_CRAWL_CACHE_SCHEMA_VERSION: Final = "pc-build-recommender.web-crawl-cache.v1"

_SOURCE_NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_RUN_DIRECTORY_PATTERN = _SHA256_PATTERN
_DELETING_RUN_PATTERN = re.compile(r"^\.(?P<run>[0-9a-f]{64})\.(?P<token>[0-9a-f]{24})\.deleting$")
_RAW_BODY_PATTERN = re.compile(
    r"^(?P<url>[0-9a-f]{32})-(?P<content>[0-9a-f]{64})\.(?:html|terms|txt)$"
)
_RAW_RECEIPT_PATTERN = re.compile(
    r"^(?P<url>[0-9a-f]{32})-(?P<content>[0-9a-f]{64})-"
    r"(?P<policy>[0-9a-f]{16})-(?P<nonce>[0-9a-f]{12})\.json$"
)
_LEGACY_RAW_RECEIPT_PATTERN = re.compile(
    r"^(?P<url>[0-9a-f]{32})-(?P<content>[0-9a-f]{64})-"
    r"(?P<policy>[0-9a-f]{16})\.json$"
)
_TEMP_TOKEN_PATTERN = re.compile(r"^[A-Za-z0-9_-]{4,128}$")
_RAW_RECEIPT_FIELDS = {
    "schema_version",
    "source_name",
    "source_url",
    "source_url_sha256",
    "final_url",
    "source_type",
    "retrieved_at",
    "retention_expires_at",
    "content_sha256",
    "byte_count",
    "media_type",
    "parser_version",
    "licence_or_access_note",
    "policy_fingerprint",
    "usage_scope",
    "acquisition_authority",
    "data_use_rights",
    "etag",
    "last_modified",
    "raw_file",
}
_AUTHORITY_FIELDS = {
    "authority_reference",
    "reviewed_on",
    "expires_on",
    "permits_automated_retrieval",
    "permits_raw_snapshot_storage",
    "permits_internal_analysis",
    "retention_days",
    "deletion_required",
}
_PROCESSED_RECEIPT_FIELDS = {
    "schema_version",
    "source_name",
    "run_sha256",
    "manifest_sha256",
    "data_quality_sha256",
    "policy_fingerprint",
    "usage_scope",
    "created_at",
    "retrieval_started_at",
    "retrieval_completed_at",
    "retention_days",
    "retention_expires_at",
    "deletion_required",
    "acquisition_authority",
    "data_use_rights",
}
_RIGHTS_FIELDS = {
    "contract_reference",
    "contract_version_url",
    "consent_effective_on",
    "consent_expires_on",
    "retention_days",
    "deletion_required_on_termination",
    "deletion_sla_days",
    "territories",
    "may_display",
    "may_cache",
    "may_store_history",
    "may_redistribute",
    "may_embed",
    "may_train",
    "may_derive",
}
_USAGE_SCOPES = {"internal_research", "production_catalog"}
_MAX_RECEIPT_BYTES = 256 * 1024
_MAX_PROCESSED_ARTIFACT_BYTES = 512 * 1024 * 1024
_MAX_PROCESSED_RUN_BYTES = 1024 * 1024 * 1024
_REPORT_SAMPLE_LIMIT = 20
_MAXIMUM_SOURCES = 100
_PROCESSED_MANIFEST_SCHEMA_VERSION = "pc-build-recommender.processed-batch.v1"
_DATA_QUALITY_SCHEMA_VERSION = "pc-build-recommender.data-quality.v1"
_PROCESSED_MANIFEST_FIELDS = {
    "schema_version",
    "source_name",
    "source_snapshot_sha256",
    "accepted_count",
    "rejected_count",
    "statistics",
    "files",
    "content_sha256",
}
_PROCESSED_MANIFEST_FILE_FIELDS = {"sha256", "byte_count"}
_REQUIRED_PROCESSED_DATA_FILES = {"records.jsonl", "rejections.jsonl"}
_ALLOWED_PROCESSED_DATA_FILES = _REQUIRED_PROCESSED_DATA_FILES | {"records.parquet"}
_DATA_QUALITY_FIELDS = {
    "schema_version",
    "source_name",
    "snapshot_sha256",
    "status",
    "accepted_count",
    "rejected_count",
    "rejection_rate",
    "checks",
    "record_type_counts",
    "category_counts",
    "eligibility_counts",
    "source_statistics",
}
_DATA_QUALITY_CHECK_FIELDS = {"name", "severity", "count", "message"}


class WebRetentionError(RuntimeError):
    """Raised when retention cannot validate or confine every destructive target."""


@dataclass(frozen=True, slots=True)
class RawRetentionReport:
    receipts_scanned: int
    active_receipts: int
    expired_receipts_eligible: int
    expired_receipts_removed: int
    bodies_scanned: int
    expired_bodies_eligible: int
    expired_bodies_removed: int
    shared_bodies_preserved: int
    orphan_bodies_detected: int
    orphan_bodies_eligible: int
    orphan_bodies_removed: int
    orphan_bodies_in_grace: int
    crash_leftovers_detected: int
    crash_leftovers_eligible: int
    crash_leftovers_removed: int
    crash_leftovers_in_grace: int
    cache_files_eligible: int
    cache_files_removed: int
    unrelated_files_preserved: int


@dataclass(frozen=True, slots=True)
class ProcessedRetentionReport:
    runs_scanned: int
    active_runs: int
    expired_runs_eligible: int
    expired_runs_removed: int
    deletion_tombstones_resumed: int
    unrelated_files_preserved: int
    publication_operations_scanned: int = 0
    publication_operations_eligible: int = 0
    publication_operations_removed: int = 0
    publication_operations_in_grace: int = 0
    published_residues_detected: int = 0
    published_residues_removed: int = 0


@dataclass(frozen=True, slots=True)
class SourceRetentionReport:
    source_name: str
    dry_run: bool
    evaluated_at: str
    raw: RawRetentionReport
    processed: ProcessedRetentionReport
    action_sample: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class _PlannedFile:
    path: Path
    size: int
    mtime_ns: int
    sha256: str | None = None


@dataclass(frozen=True, slots=True)
class _RawPlan:
    source_name: str
    source_root: Path | None
    page_root: Path | None
    cache_removal: _PlannedFile | None
    receipt_removals: tuple[_PlannedFile, ...]
    expired_body_removals: tuple[_PlannedFile, ...]
    orphan_body_removals: tuple[_PlannedFile, ...]
    temp_removals: tuple[_PlannedFile, ...]
    counts: dict[str, int]


@dataclass(frozen=True, slots=True)
class _ProcessedRunPlan:
    path: Path
    run_sha256: str
    manifest: tuple[tuple[str, str, int, int], ...]
    receipt_sha256: str | None
    already_quarantined: bool = False


@dataclass(frozen=True, slots=True)
class _ProcessedPlan:
    source_name: str
    source_root: Path | None
    run_removals: tuple[_ProcessedRunPlan, ...]
    counts: dict[str, int]


@dataclass(frozen=True, slots=True)
class _SourcePlan:
    source_name: str
    raw: _RawPlan
    processed: _ProcessedPlan


@dataclass(slots=True)
class _WorkBudget:
    """One cumulative entry budget for a complete maintenance invocation."""

    limit: int
    consumed: int = 0

    def consume(self, *, label: str, amount: int = 1) -> None:
        if amount < 1:
            raise ValueError("retention work-budget amount must be positive")
        if self.consumed + amount > self.limit:
            raise WebRetentionError(
                f"governed-web maintenance exceeds its global {self.limit}-entry work limit "
                f"while scanning {label}"
            )
        self.consumed += amount


def _is_linklike(path: Path) -> bool:
    is_junction = getattr(os.path, "isjunction", None)
    return path.is_symlink() or bool(is_junction is not None and is_junction(path))


def _validated_source_name(source_name: str) -> str:
    if _SOURCE_NAME_PATTERN.fullmatch(source_name) is None:
        raise WebRetentionError(f"unsafe governed-web source name: {source_name!r}")
    return source_name


def _resolved_root(root: Path, *, label: str) -> Path:
    candidate = Path(root)
    if candidate.exists():
        if _is_linklike(candidate):
            raise WebRetentionError(f"{label} is a symlink or junction; operator action required")
        if not candidate.is_dir():
            raise WebRetentionError(f"{label} is not a directory; operator action required")
    return candidate.resolve()


def _direct_source_root(root: Path, source_name: str, *, label: str) -> Path:
    resolved_root = _resolved_root(root, label=label)
    source_candidate = Path(root) / _validated_source_name(source_name)
    if source_candidate.exists() and _is_linklike(source_candidate):
        raise WebRetentionError(
            f"{label} source {source_name!r} is a symlink or junction; operator action required"
        )
    source_resolved = source_candidate.resolve()
    if source_resolved.parent != resolved_root:
        raise WebRetentionError(f"{label} source escaped its exact configured root")
    if source_candidate.exists() and not source_resolved.is_dir():
        raise WebRetentionError(f"{label} source is not a directory; operator action required")
    return source_resolved


def _direct_pages_root(source_root: Path) -> Path:
    pages_candidate = source_root / "pages"
    if pages_candidate.exists() and _is_linklike(pages_candidate):
        raise WebRetentionError("raw page store is a symlink or junction; operator action required")
    pages_resolved = pages_candidate.resolve()
    if pages_resolved.parent != source_root:
        raise WebRetentionError("raw page store escaped its exact source root")
    if pages_candidate.exists() and not pages_resolved.is_dir():
        raise WebRetentionError("raw page store is not a directory; operator action required")
    return pages_resolved


def _read_json_object(path: Path, *, label: str) -> tuple[dict[str, Any], str]:
    if _is_linklike(path) or not path.is_file():
        raise WebRetentionError(f"{label} is not a regular file: {path}")
    try:
        with path.open("rb") as handle:
            before = os.fstat(handle.fileno())
            if not stat_module.S_ISREG(before.st_mode):
                raise WebRetentionError(f"{label} is not a regular file: {path}")
            if before.st_size > _MAX_RECEIPT_BYTES:
                raise WebRetentionError(
                    f"{label} exceeds the {_MAX_RECEIPT_BYTES}-byte limit: {path}"
                )
            raw = handle.read(_MAX_RECEIPT_BYTES + 1)
            after = os.fstat(handle.fileno())
    except WebRetentionError:
        raise
    except OSError as exc:
        raise WebRetentionError(f"{label} cannot be inspected: {path}") from exc
    if len(raw) > _MAX_RECEIPT_BYTES:
        raise WebRetentionError(f"{label} exceeds the {_MAX_RECEIPT_BYTES}-byte limit: {path}")
    if (
        len(raw) != before.st_size
        or after.st_size != before.st_size
        or after.st_mtime_ns != before.st_mtime_ns
    ):
        raise WebRetentionError(f"{label} changed while being read: {path}")
    try:
        payload = json.loads(
            raw,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON value {value!r}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise WebRetentionError(f"{label} is unreadable: {path}") from exc
    if not isinstance(payload, dict):
        raise WebRetentionError(f"{label} must be a JSON object: {path}")
    return payload, hashlib.sha256(raw).hexdigest()


def _bounded_file_sha256(path: Path, *, label: str, maximum_bytes: int) -> tuple[str, int]:
    """Hash one direct regular file without allocating its contents in memory."""

    if _is_linklike(path) or not path.is_file():
        raise WebRetentionError(f"{label} is not a regular file: {path}")
    digest = hashlib.sha256()
    byte_count = 0
    try:
        with path.open("rb") as handle:
            before = os.fstat(handle.fileno())
            if not stat_module.S_ISREG(before.st_mode):
                raise WebRetentionError(f"{label} is not a regular file: {path}")
            if before.st_size > maximum_bytes:
                raise WebRetentionError(f"{label} exceeds its {maximum_bytes}-byte limit: {path}")
            while chunk := handle.read(64 * 1024):
                byte_count += len(chunk)
                if byte_count > maximum_bytes:
                    raise WebRetentionError(
                        f"{label} exceeds its {maximum_bytes}-byte limit: {path}"
                    )
                digest.update(chunk)
            after = os.fstat(handle.fileno())
    except WebRetentionError:
        raise
    except OSError as exc:
        raise WebRetentionError(f"{label} cannot be inspected: {path}") from exc
    if (
        byte_count != before.st_size
        or after.st_size != before.st_size
        or after.st_mtime_ns != before.st_mtime_ns
    ):
        raise WebRetentionError(f"{label} changed while being read: {path}")
    return digest.hexdigest(), byte_count


def _strict_nonnegative_integer(value: object, *, label: str) -> int:
    if type(value) is not int or value < 0:
        raise WebRetentionError(f"{label} must be a non-negative integer")
    return value


def _validate_count_mapping(value: object, *, label: str) -> None:
    if not isinstance(value, dict) or any(
        not isinstance(key, str) or not key or type(count) is not int or count < 0
        for key, count in value.items()
    ):
        raise WebRetentionError(f"{label} must map non-empty strings to non-negative integers")


def _validate_processed_artifacts(
    run_directory: Path,
    *,
    source_name: str,
    run_sha256: str,
    receipt_required: bool,
) -> tuple[dict[str, Any], str, str]:
    """Validate a complete immutable processed run and return its bound JSON hashes."""

    manifest, manifest_sha = _read_json_object(
        run_directory / "manifest.json", label="processed manifest"
    )
    if set(manifest) != _PROCESSED_MANIFEST_FIELDS:
        raise WebRetentionError("processed manifest fields are incomplete or unknown")
    if manifest.get("schema_version") != _PROCESSED_MANIFEST_SCHEMA_VERSION:
        raise WebRetentionError("unsupported processed manifest schema")
    if manifest.get("source_name") != source_name:
        raise WebRetentionError("processed manifest source does not match its exact source root")
    if manifest.get("source_snapshot_sha256") != run_sha256:
        raise WebRetentionError("processed manifest snapshot does not match its exact run root")
    accepted_count = _strict_nonnegative_integer(
        manifest.get("accepted_count"), label="processed manifest accepted_count"
    )
    rejected_count = _strict_nonnegative_integer(
        manifest.get("rejected_count"), label="processed manifest rejected_count"
    )
    statistics = manifest.get("statistics")
    if not isinstance(statistics, dict):
        raise WebRetentionError("processed manifest statistics must be an object")
    content_sha = manifest.get("content_sha256")
    if not isinstance(content_sha, str) or _SHA256_PATTERN.fullmatch(content_sha) is None:
        raise WebRetentionError("processed manifest has an invalid content_sha256")
    semantic_manifest = dict(manifest)
    semantic_manifest.pop("content_sha256")
    try:
        canonical_manifest = json.dumps(
            semantic_manifest,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise WebRetentionError("processed manifest contains invalid JSON values") from exc
    if hashlib.sha256(canonical_manifest).hexdigest() != content_sha:
        raise WebRetentionError("processed manifest content_sha256 does not match its content")

    files = manifest.get("files")
    if not isinstance(files, dict):
        raise WebRetentionError("processed manifest files must be an object")
    file_names = set(files)
    if not file_names >= _REQUIRED_PROCESSED_DATA_FILES or not file_names <= (
        _ALLOWED_PROCESSED_DATA_FILES
    ):
        raise WebRetentionError("processed manifest declares missing or unsupported data files")
    total_bytes = 0
    for file_name in sorted(file_names):
        metadata = files[file_name]
        if not isinstance(metadata, dict) or set(metadata) != _PROCESSED_MANIFEST_FILE_FIELDS:
            raise WebRetentionError(f"processed manifest has invalid metadata for {file_name!r}")
        expected_sha = metadata.get("sha256")
        if not isinstance(expected_sha, str) or _SHA256_PATTERN.fullmatch(expected_sha) is None:
            raise WebRetentionError(f"processed manifest has invalid SHA-256 for {file_name!r}")
        expected_bytes = _strict_nonnegative_integer(
            metadata.get("byte_count"),
            label=f"processed manifest byte_count for {file_name!r}",
        )
        if expected_bytes > _MAX_PROCESSED_ARTIFACT_BYTES:
            raise WebRetentionError(f"processed manifest file {file_name!r} exceeds its byte limit")
        total_bytes += expected_bytes
        if total_bytes > _MAX_PROCESSED_RUN_BYTES:
            raise WebRetentionError("processed manifest exceeds the total run byte limit")
        actual_sha, actual_bytes = _bounded_file_sha256(
            run_directory / file_name,
            label=f"processed data file {file_name!r}",
            maximum_bytes=_MAX_PROCESSED_ARTIFACT_BYTES,
        )
        if actual_bytes != expected_bytes or actual_sha != expected_sha:
            raise WebRetentionError(
                f"processed data file {file_name!r} does not match its manifest"
            )

    quality, quality_sha = _read_json_object(
        run_directory / "data-quality.json", label="processed data-quality report"
    )
    if set(quality) != _DATA_QUALITY_FIELDS:
        raise WebRetentionError("processed data-quality fields are incomplete or unknown")
    if quality.get("schema_version") != _DATA_QUALITY_SCHEMA_VERSION:
        raise WebRetentionError("unsupported processed data-quality schema")
    if quality.get("source_name") != source_name:
        raise WebRetentionError("processed data-quality source does not match its exact root")
    if quality.get("snapshot_sha256") != run_sha256:
        raise WebRetentionError("processed data-quality snapshot does not match its exact run root")
    if quality.get("accepted_count") != accepted_count:
        raise WebRetentionError("processed data-quality accepted_count does not match its manifest")
    if quality.get("rejected_count") != rejected_count:
        raise WebRetentionError("processed data-quality rejected_count does not match its manifest")
    rejection_rate = quality.get("rejection_rate")
    if (
        isinstance(rejection_rate, bool)
        or not isinstance(rejection_rate, (int, float))
        or not math.isfinite(rejection_rate)
        or not 0 <= rejection_rate <= 1
    ):
        raise WebRetentionError("processed data-quality rejection_rate is invalid")
    checks = quality.get("checks")
    if not isinstance(checks, list):
        raise WebRetentionError("processed data-quality checks must be a list")
    for check in checks:
        if not isinstance(check, dict) or set(check) != _DATA_QUALITY_CHECK_FIELDS:
            raise WebRetentionError("processed data-quality contains an invalid check")
        if (
            not isinstance(check.get("name"), str)
            or not check["name"]
            or check.get("severity") not in {"error", "warning"}
            or type(check.get("count")) is not int
            or check["count"] < 0
            or not isinstance(check.get("message"), str)
            or not check["message"]
        ):
            raise WebRetentionError("processed data-quality contains an invalid check")
    for field_name in ("record_type_counts", "category_counts", "eligibility_counts"):
        _validate_count_mapping(
            quality.get(field_name), label=f"processed data-quality {field_name}"
        )
    if quality.get("source_statistics") != statistics:
        raise WebRetentionError("processed data-quality statistics do not match its manifest")
    error_count = sum(check["count"] for check in checks if check["severity"] == "error")
    warning_count = sum(check["count"] for check in checks if check["severity"] == "warning")
    expected_status = "fail" if error_count else "warning" if warning_count else "pass"
    if quality.get("status") != expected_status:
        raise WebRetentionError("processed data-quality status does not match its checks")

    expected_entries = file_names | {"manifest.json", "data-quality.json"}
    if receipt_required:
        expected_entries.add(WEB_PROCESSED_RETENTION_RECEIPT)
    actual_entries: set[str] = set()
    try:
        entries = os.scandir(run_directory)
    except OSError as exc:
        raise WebRetentionError("processed run cannot be inspected safely") from exc
    with entries:
        for entry in entries:
            child = Path(entry.path)
            if entry.is_symlink() or _is_linklike(child):
                raise WebRetentionError("processed run contains a symlink or junction")
            if not entry.is_file(follow_symlinks=False):
                raise WebRetentionError("processed run contains an undeclared non-file entry")
            if entry.name not in expected_entries:
                raise WebRetentionError("processed run contains missing or undeclared files")
            actual_entries.add(entry.name)
    if actual_entries != expected_entries:
        raise WebRetentionError("processed run contains missing or undeclared files")
    return manifest, manifest_sha, quality_sha


def _aware_datetime(payload: dict[str, Any], field_name: str, *, label: str) -> datetime:
    value = payload.get(field_name)
    if not isinstance(value, str):
        raise WebRetentionError(f"{label} has invalid {field_name!r}")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise WebRetentionError(f"{label} has invalid {field_name!r}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise WebRetentionError(f"{label} {field_name!r} must be timezone aware")
    return parsed.astimezone(UTC)


def _iso_date(value: object, *, field_name: str, allow_none: bool = False) -> date | None:
    if value is None and allow_none:
        return None
    if not isinstance(value, str):
        raise WebRetentionError(f"raw receipt has invalid authority {field_name!r}")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise WebRetentionError(f"raw receipt has invalid authority {field_name!r}") from exc


def _receipt_date(
    value: object,
    *,
    field_name: str,
    label: str,
    allow_none: bool = False,
) -> date | None:
    if value is None and allow_none:
        return None
    if not isinstance(value, str):
        raise WebRetentionError(f"{label} has invalid {field_name!r}")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise WebRetentionError(f"{label} has invalid {field_name!r}") from exc


def _date_expiry_deadline(value: date | None) -> datetime | None:
    """Return the first UTC instant after an inclusive contract/authority expiry date."""

    if value is None:
        return None
    return datetime.combine(value + timedelta(days=1), datetime.min.time(), tzinfo=UTC)


def _effective_expiry(
    *,
    retention_expires_at: datetime,
    acquisition_authority_expires_on: date | None,
    rights_delete_by: datetime | None,
) -> datetime:
    deadlines = [retention_expires_at]
    authority_deadline = _date_expiry_deadline(acquisition_authority_expires_on)
    if authority_deadline is not None:
        deadlines.append(authority_deadline)
    if rights_delete_by is not None:
        deadlines.append(rights_delete_by)
    return min(deadlines)


def _required_nonempty_string(payload: dict[str, Any], field_name: str, *, label: str) -> str:
    value = payload.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise WebRetentionError(f"{label} has invalid {field_name!r}")
    return value


def _validated_https_url(payload: dict[str, Any], field_name: str) -> str:
    value = _required_nonempty_string(payload, field_name, label="raw receipt")
    parsed = urlsplit(value)
    if (
        parsed.scheme.lower() != "https"
        or parsed.hostname is None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
    ):
        raise WebRetentionError(f"raw receipt has unsafe {field_name!r}")
    try:
        port = parsed.port
    except ValueError as exc:
        raise WebRetentionError(f"raw receipt has unsafe {field_name!r}") from exc
    if port not in (None, 443):
        raise WebRetentionError(f"raw receipt has unsafe {field_name!r}")
    return value


def _raw_receipt_match(name: str) -> re.Match[str] | None:
    """Accept the original and nonce-hardened immutable receipt filename shapes."""

    return _RAW_RECEIPT_PATTERN.fullmatch(name) or _LEGACY_RAW_RECEIPT_PATTERN.fullmatch(name)


def _validate_authority(
    authority: object,
    *,
    retrieved_at: datetime,
    retention_expires_at: datetime,
) -> date | None:
    if not isinstance(authority, dict) or set(authority) != _AUTHORITY_FIELDS:
        raise WebRetentionError(
            "raw receipt acquisition authority fields are incomplete or unknown"
        )
    reference = authority.get("authority_reference")
    if not isinstance(reference, str) or not reference.strip():
        raise WebRetentionError("raw receipt has invalid acquisition authority reference")
    reviewed_on = _iso_date(authority.get("reviewed_on"), field_name="reviewed_on")
    expires_on = _iso_date(authority.get("expires_on"), field_name="expires_on", allow_none=True)
    assert reviewed_on is not None
    if expires_on is not None and expires_on < reviewed_on:
        raise WebRetentionError("raw receipt authority expiry precedes review")
    if expires_on is not None and retrieved_at.date() > expires_on:
        raise WebRetentionError("raw receipt was retrieved after acquisition authority expiry")
    for field_name in (
        "permits_automated_retrieval",
        "permits_raw_snapshot_storage",
        "permits_internal_analysis",
        "deletion_required",
    ):
        if type(authority.get(field_name)) is not bool:
            raise WebRetentionError(f"raw receipt authority has invalid {field_name!r}")
    if not authority["permits_automated_retrieval"]:
        raise WebRetentionError("raw receipt authority did not permit automated retrieval")
    if not authority["permits_raw_snapshot_storage"]:
        raise WebRetentionError("raw receipt authority did not permit raw storage")
    if authority["deletion_required"] is not True:
        raise WebRetentionError("governed raw retention must require deletion")
    retention_days = authority.get("retention_days")
    if type(retention_days) is not int or not 1 <= retention_days <= 3650:
        raise WebRetentionError("raw receipt authority has invalid retention_days")
    if retention_expires_at != retrieved_at + timedelta(days=retention_days):
        raise WebRetentionError("raw receipt expiry does not match its authority retention_days")
    return expires_on


def _validate_rights(
    rights: object,
    *,
    retrieved_at: datetime,
    retention_expires_at: datetime,
) -> datetime | None:
    if not isinstance(rights, dict) or set(rights) != _RIGHTS_FIELDS:
        raise WebRetentionError("raw receipt data-use rights fields are incomplete or unknown")
    for field_name in ("contract_reference", "contract_version_url"):
        value = rights.get(field_name)
        if not isinstance(value, str) or not value.strip():
            raise WebRetentionError(f"raw receipt data-use rights has invalid {field_name!r}")
    effective_on = _receipt_date(
        rights.get("consent_effective_on"),
        field_name="consent_effective_on",
        label="raw receipt data-use rights",
    )
    expires_on = _receipt_date(
        rights.get("consent_expires_on"),
        field_name="consent_expires_on",
        label="raw receipt data-use rights",
        allow_none=True,
    )
    assert effective_on is not None
    if expires_on is not None and expires_on < effective_on:
        raise WebRetentionError("raw receipt rights expiry precedes its effective date")
    if retrieved_at.date() < effective_on:
        raise WebRetentionError("raw receipt was retrieved before source rights became effective")
    if expires_on is not None and retrieved_at.date() > expires_on:
        raise WebRetentionError("raw receipt was retrieved after source rights expiry")
    retention_days = rights.get("retention_days")
    if retention_days is not None and (
        type(retention_days) is not int or not 1 <= retention_days <= 3650
    ):
        raise WebRetentionError("raw receipt data-use rights has invalid retention_days")
    if retention_days is not None and retention_expires_at > retrieved_at + timedelta(
        days=retention_days
    ):
        raise WebRetentionError("raw receipt retention exceeds its data-use rights")
    deletion_required = rights.get("deletion_required_on_termination")
    if type(deletion_required) is not bool:
        raise WebRetentionError(
            "raw receipt data-use rights has invalid deletion_required_on_termination"
        )
    deletion_sla = rights.get("deletion_sla_days")
    if deletion_required:
        if type(deletion_sla) is not int or deletion_sla < 1:
            raise WebRetentionError("raw receipt data-use rights has invalid deletion_sla_days")
    elif deletion_sla is not None:
        raise WebRetentionError(
            "raw receipt data-use rights deletion_sla_days requires termination deletion"
        )
    territories = rights.get("territories")
    if (
        not isinstance(territories, list)
        or not territories
        or any(not isinstance(item, str) or not item.strip() for item in territories)
    ):
        raise WebRetentionError("raw receipt data-use rights has invalid territories")
    for field_name in (
        "may_display",
        "may_cache",
        "may_store_history",
        "may_redistribute",
        "may_embed",
        "may_train",
        "may_derive",
    ):
        if type(rights.get(field_name)) is not bool:
            raise WebRetentionError(f"raw receipt data-use rights has invalid {field_name!r}")
    if not deletion_required or expires_on is None:
        return None
    expiry_deadline = _date_expiry_deadline(expires_on)
    assert expiry_deadline is not None
    assert isinstance(deletion_sla, int)
    return expiry_deadline + timedelta(days=deletion_sla)


def _validate_usage_scope(
    *,
    usage_scope: object,
    authority: object,
    rights: object,
    label: str,
) -> None:
    if usage_scope not in _USAGE_SCOPES:
        raise WebRetentionError(f"{label} has an invalid usage_scope")
    assert isinstance(authority, dict)
    assert isinstance(rights, dict)
    grants = (
        "may_display",
        "may_cache",
        "may_store_history",
        "may_redistribute",
        "may_embed",
        "may_train",
        "may_derive",
    )
    if usage_scope == "internal_research":
        if authority.get("permits_internal_analysis") is not True:
            raise WebRetentionError(f"{label} research scope lacks internal-analysis authority")
        if any(rights.get(field_name) is not False for field_name in grants):
            raise WebRetentionError(f"{label} research scope contains downstream rights grants")
        return
    territories = rights.get("territories")
    if not isinstance(territories, list) or "SG" not in {
        str(item).strip().upper() for item in territories
    }:
        raise WebRetentionError(f"{label} production rights do not permit SG")
    for field_name in ("may_display", "may_cache", "may_store_history", "may_derive"):
        if rights.get(field_name) is not True:
            raise WebRetentionError(
                f"{label} production rights do not grant {field_name.removeprefix('may_')}"
            )


def _validate_raw_receipt(
    *,
    path: Path,
    source_name: str,
) -> tuple[dict[str, Any], datetime, str]:
    match = _raw_receipt_match(path.name)
    if match is None:
        raise WebRetentionError(f"unsafe raw receipt filename: {path.name!r}")
    payload, receipt_sha = _read_json_object(path, label="raw receipt")
    if set(payload) != _RAW_RECEIPT_FIELDS:
        raise WebRetentionError("raw receipt fields are incomplete or unknown")
    if payload.get("schema_version") != WEB_RAW_METADATA_SCHEMA_VERSION:
        raise WebRetentionError("unsupported raw receipt schema")
    if payload.get("source_name") != source_name:
        raise WebRetentionError("raw receipt source does not match its exact source root")
    source_url = _validated_https_url(payload, "source_url")
    _validated_https_url(payload, "final_url")
    source_url_sha = hashlib.sha256(source_url.encode("utf-8")).hexdigest()
    if payload.get("source_url_sha256") != source_url_sha:
        raise WebRetentionError("raw receipt source URL digest is invalid")
    content_sha = payload.get("content_sha256")
    policy_fingerprint = payload.get("policy_fingerprint")
    if not isinstance(content_sha, str) or _SHA256_PATTERN.fullmatch(content_sha) is None:
        raise WebRetentionError("raw receipt content digest is invalid")
    if (
        not isinstance(policy_fingerprint, str)
        or _SHA256_PATTERN.fullmatch(policy_fingerprint) is None
    ):
        raise WebRetentionError("raw receipt policy fingerprint is invalid")
    if (
        match.group("url") != source_url_sha[:32]
        or match.group("content") != content_sha
        or match.group("policy") != policy_fingerprint[:16]
    ):
        raise WebRetentionError("raw receipt filename does not match its content")
    raw_file = payload.get("raw_file")
    if not isinstance(raw_file, str) or _RAW_BODY_PATTERN.fullmatch(raw_file) is None:
        raise WebRetentionError("raw receipt contains an unsafe raw_file")
    raw_match = _RAW_BODY_PATTERN.fullmatch(raw_file)
    assert raw_match is not None
    if raw_match.group("url") != source_url_sha[:32] or raw_match.group("content") != content_sha:
        raise WebRetentionError("raw receipt body filename does not match its digests")
    if payload.get("source_type") != "retailer":
        raise WebRetentionError("raw receipt has an invalid source_type")
    byte_count = payload.get("byte_count")
    if type(byte_count) is not int or byte_count < 0:
        raise WebRetentionError("raw receipt has an invalid byte_count")
    for field_name in ("media_type", "parser_version", "licence_or_access_note"):
        _required_nonempty_string(payload, field_name, label="raw receipt")
    for field_name in ("etag", "last_modified"):
        value = payload.get(field_name)
        if value is not None and not isinstance(value, str):
            raise WebRetentionError(f"raw receipt has invalid {field_name!r}")
    retrieved_at = _aware_datetime(payload, "retrieved_at", label="raw receipt")
    retention_expires_at = _aware_datetime(payload, "retention_expires_at", label="raw receipt")
    if retention_expires_at <= retrieved_at:
        raise WebRetentionError("raw receipt expiry must follow retrieval")
    authority = payload.get("acquisition_authority")
    authority_expires_on = _validate_authority(
        authority,
        retrieved_at=retrieved_at,
        retention_expires_at=retention_expires_at,
    )
    rights = payload.get("data_use_rights")
    rights_delete_by = _validate_rights(
        rights,
        retrieved_at=retrieved_at,
        retention_expires_at=retention_expires_at,
    )
    _validate_usage_scope(
        usage_scope=payload.get("usage_scope"),
        authority=authority,
        rights=rights,
        label="raw receipt",
    )
    return (
        payload,
        _effective_expiry(
            retention_expires_at=retention_expires_at,
            acquisition_authority_expires_on=authority_expires_on,
            rights_delete_by=rights_delete_by,
        ),
        receipt_sha,
    )


def _validate_processed_receipt(
    path: Path,
    source_name: str,
    *,
    run_sha256: str,
    artifacts_required: bool = True,
) -> tuple[datetime, str]:
    payload, receipt_sha = _read_json_object(path, label="processed retention receipt")
    if set(payload) != _PROCESSED_RECEIPT_FIELDS:
        raise WebRetentionError("processed retention receipt fields are incomplete or unknown")
    if payload.get("schema_version") != WEB_PROCESSED_RETENTION_SCHEMA_VERSION:
        raise WebRetentionError("unsupported processed retention receipt schema")
    if payload.get("source_name") != source_name:
        raise WebRetentionError("processed retention receipt source does not match its root")
    if payload.get("run_sha256") != run_sha256:
        raise WebRetentionError(
            "processed retention receipt run_sha256 does not match its exact run root"
        )
    if artifacts_required:
        _manifest, manifest_sha, quality_sha = _validate_processed_artifacts(
            path.parent,
            source_name=source_name,
            run_sha256=run_sha256,
            receipt_required=True,
        )
        if payload.get("manifest_sha256") != manifest_sha:
            raise WebRetentionError("processed retention receipt does not bind its manifest")
        if payload.get("data_quality_sha256") != quality_sha:
            raise WebRetentionError(
                "processed retention receipt does not bind its data-quality report"
            )
    else:
        for field_name in ("manifest_sha256", "data_quality_sha256"):
            digest = payload.get(field_name)
            if not isinstance(digest, str) or _SHA256_PATTERN.fullmatch(digest) is None:
                raise WebRetentionError(f"processed retention receipt has an invalid {field_name}")
    fingerprint = payload.get("policy_fingerprint")
    if not isinstance(fingerprint, str) or _SHA256_PATTERN.fullmatch(fingerprint) is None:
        raise WebRetentionError("processed retention receipt has an invalid policy fingerprint")
    if payload.get("deletion_required") is not True:
        raise WebRetentionError("governed processed retention must require deletion")
    created_at = _aware_datetime(payload, "created_at", label="processed retention receipt")
    retrieval_started_at = _aware_datetime(
        payload, "retrieval_started_at", label="processed retention receipt"
    )
    retrieval_completed_at = _aware_datetime(
        payload, "retrieval_completed_at", label="processed retention receipt"
    )
    retention_expires_at = _aware_datetime(
        payload, "retention_expires_at", label="processed retention receipt"
    )
    if not retrieval_started_at <= retrieval_completed_at <= created_at:
        raise WebRetentionError("processed retention receipt has an invalid retrieval interval")
    if retention_expires_at <= retrieval_started_at:
        raise WebRetentionError("processed retention receipt expiry must follow retrieval")
    retention_days = payload.get("retention_days")
    if type(retention_days) is not int or not 1 <= retention_days <= 3650:
        raise WebRetentionError("processed retention receipt has invalid retention_days")
    if retention_expires_at != retrieval_started_at + timedelta(days=retention_days):
        raise WebRetentionError("processed retention receipt expiry does not match retention_days")
    authority = payload.get("acquisition_authority")
    authority_expires_on = _validate_authority(
        authority,
        retrieved_at=retrieval_started_at,
        retention_expires_at=retention_expires_at,
    )
    assert isinstance(authority, dict)
    if authority.get("retention_days") != retention_days:
        raise WebRetentionError(
            "processed retention receipt retention_days does not match its authority"
        )
    if authority_expires_on is not None and retrieval_completed_at.date() > authority_expires_on:
        raise WebRetentionError("processed retrieval completed after acquisition authority expiry")
    rights = payload.get("data_use_rights")
    rights_delete_by = _validate_rights(
        rights,
        retrieved_at=retrieval_started_at,
        retention_expires_at=retention_expires_at,
    )
    assert isinstance(rights, dict)
    rights_expires_on = _receipt_date(
        rights.get("consent_expires_on"),
        field_name="consent_expires_on",
        label="processed retention receipt data-use rights",
        allow_none=True,
    )
    if rights_expires_on is not None and retrieval_completed_at.date() > rights_expires_on:
        raise WebRetentionError("processed retrieval completed after source rights expiry")
    _validate_usage_scope(
        usage_scope=payload.get("usage_scope"),
        authority=authority,
        rights=rights,
        label="processed retention receipt",
    )
    return (
        _effective_expiry(
            retention_expires_at=retention_expires_at,
            acquisition_authority_expires_on=authority_expires_on,
            rights_delete_by=rights_delete_by,
        ),
        receipt_sha,
    )


def write_web_processed_retention_receipt(
    *,
    processed_root: Path,
    output_directory: Path,
    policy: WebSourcePolicy,
    retrieval_started_at: datetime,
    retrieval_completed_at: datetime,
    created_at: datetime | None = None,
) -> Path:
    """Create or idempotently reuse one immutable, run-bound v2 receipt."""

    for label, value in (
        ("retrieval_started_at", retrieval_started_at),
        ("retrieval_completed_at", retrieval_completed_at),
    ):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(f"web processed {label} must be timezone aware")
    created = created_at or datetime.now(UTC)
    if created.tzinfo is None or created.utcoffset() is None:
        raise ValueError("web processed receipt creation time must be timezone aware")
    started_utc = retrieval_started_at.astimezone(UTC)
    completed_utc = retrieval_completed_at.astimezone(UTC)
    created_utc = created.astimezone(UTC)
    if not started_utc <= completed_utc <= created_utc:
        raise WebRetentionError("web processed receipt has an invalid retrieval interval")

    source_root = _direct_source_root(
        processed_root,
        policy.source_name,
        label="processed root",
    )
    run_candidate = Path(output_directory)
    if _is_linklike(run_candidate):
        raise WebRetentionError("processed run is a symlink or junction; operator action required")
    try:
        run_directory = run_candidate.resolve(strict=True)
    except OSError as exc:
        raise WebRetentionError("processed run cannot be resolved safely") from exc
    if run_directory.parent != source_root or not run_directory.is_dir():
        raise WebRetentionError("processed run escaped its exact source root")
    run_sha256 = run_directory.name
    if _RUN_DIRECTORY_PATTERN.fullmatch(run_sha256) is None:
        raise WebRetentionError("processed run directory must be an exact SHA-256")
    receipt_path = run_directory / WEB_PROCESSED_RETENTION_RECEIPT
    receipt_present = receipt_path.exists() or _is_linklike(receipt_path)
    _manifest, manifest_sha, quality_sha = _validate_processed_artifacts(
        run_directory,
        source_name=policy.source_name,
        run_sha256=run_sha256,
        receipt_required=receipt_present,
    )

    retention_days = policy.acquisition_authority.retention_days
    payload: dict[str, Any] = {
        "schema_version": WEB_PROCESSED_RETENTION_SCHEMA_VERSION,
        "source_name": policy.source_name,
        "run_sha256": run_sha256,
        "manifest_sha256": manifest_sha,
        "data_quality_sha256": quality_sha,
        "policy_fingerprint": policy.fingerprint,
        "usage_scope": policy.usage_scope.value,
        "created_at": created_utc.isoformat(),
        "retrieval_started_at": started_utc.isoformat(),
        "retrieval_completed_at": completed_utc.isoformat(),
        "retention_days": retention_days,
        "retention_expires_at": (started_utc + timedelta(days=retention_days)).isoformat(),
        "deletion_required": policy.acquisition_authority.deletion_required,
        "acquisition_authority": policy.acquisition_authority.to_dict(),
        "data_use_rights": policy.rights.to_dict(),
    }
    if receipt_path.exists():
        existing, _sha = _read_json_object(receipt_path, label="processed retention receipt")
        _validate_processed_receipt(
            receipt_path,
            policy.source_name,
            run_sha256=run_sha256,
        )
        expected_immutable = dict(payload)
        existing_immutable = dict(existing)
        expected_immutable.pop("created_at")
        existing_immutable.pop("created_at", None)
        if existing_immutable != expected_immutable:
            raise WebRetentionError(
                "existing processed retention receipt conflicts with this immutable run"
            )
        return receipt_path
    if _is_linklike(receipt_path):
        raise WebRetentionError("processed retention receipt target is a symlink or junction")
    try:
        with receipt_path.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(
                payload,
                handle,
                allow_nan=False,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        return write_web_processed_retention_receipt(
            processed_root=processed_root,
            output_directory=output_directory,
            policy=policy,
            retrieval_started_at=retrieval_started_at,
            retrieval_completed_at=retrieval_completed_at,
            created_at=created_at,
        )
    _validate_processed_receipt(
        receipt_path,
        policy.source_name,
        run_sha256=run_sha256,
    )
    return receipt_path


def _planned_file(path: Path, *, sha256: str | None = None) -> _PlannedFile:
    try:
        stat = path.stat()
    except OSError as exc:
        raise WebRetentionError(f"retention target cannot be inspected: {path}") from exc
    if _is_linklike(path) or not path.is_file():
        raise WebRetentionError(f"retention target is not a regular file: {path}")
    return _PlannedFile(path=path, size=stat.st_size, mtime_ns=stat.st_mtime_ns, sha256=sha256)


def _recognised_temp_target(name: str) -> bool:
    if not name.startswith(".") or not name.endswith(".tmp"):
        return False
    inner = name[1:-4]
    target, separator, token = inner.rpartition(".")
    if not separator or _TEMP_TOKEN_PATTERN.fullmatch(token) is None:
        return False
    return bool(
        _RAW_BODY_PATTERN.fullmatch(target)
        or _raw_receipt_match(target)
        or target == WEB_CRAWL_CACHE_FILE
    )


def _recognised_cache_temp(name: str) -> bool:
    return name.startswith(f".{WEB_CRAWL_CACHE_FILE}.") and _recognised_temp_target(name)


def _scan_raw(
    *,
    raw_root: Path,
    source_name: str,
    now: datetime,
    orphan_grace: timedelta,
    work_budget: _WorkBudget,
) -> _RawPlan:
    source_root = _direct_source_root(raw_root, source_name, label="raw root")
    page_root = _direct_pages_root(source_root)
    counts = {
        "receipts_scanned": 0,
        "active_receipts": 0,
        "expired_receipts_eligible": 0,
        "bodies_scanned": 0,
        "expired_bodies_eligible": 0,
        "shared_bodies_preserved": 0,
        "orphan_bodies_detected": 0,
        "orphan_bodies_eligible": 0,
        "orphan_bodies_in_grace": 0,
        "crash_leftovers_detected": 0,
        "crash_leftovers_eligible": 0,
        "crash_leftovers_in_grace": 0,
        "cache_files_eligible": 0,
        "unrelated_files_preserved": 0,
    }
    if not source_root.exists():
        return _RawPlan(source_name, None, None, None, (), (), (), (), counts)

    active_bodies: dict[str, tuple[str, int]] = {}
    expired_bodies: set[str] = set()
    receipt_removals: list[_PlannedFile] = []
    bodies: list[Path] = []
    temps: list[Path] = []
    cutoff = now - orphan_grace
    cache_path: Path | None = None
    try:
        source_entries = os.scandir(source_root)
    except OSError as exc:
        raise WebRetentionError("raw source root cannot be inspected") from exc
    with source_entries:
        for entry in source_entries:
            work_budget.consume(label=f"raw source root {source_name!r}")
            path = Path(entry.path)
            if entry.is_symlink() or _is_linklike(path):
                raise WebRetentionError(
                    f"raw source root contains a symlink or junction: {entry.name}"
                )
            if entry.name == "pages" and entry.is_dir(follow_symlinks=False):
                continue
            if entry.name == WEB_CRAWL_CACHE_FILE and entry.is_file(follow_symlinks=False):
                cache_path = path
                continue
            if _recognised_cache_temp(entry.name) and entry.is_file(follow_symlinks=False):
                counts["crash_leftovers_detected"] += 1
                temps.append(path)
                continue
            counts["unrelated_files_preserved"] += 1

    if not page_root.exists():
        temp_removals: list[_PlannedFile] = []
        for path in temps:
            planned = _planned_file(path)
            modified_at = datetime.fromtimestamp(planned.mtime_ns / 1_000_000_000, tz=UTC)
            if modified_at <= cutoff:
                counts["crash_leftovers_eligible"] += 1
                temp_removals.append(planned)
            else:
                counts["crash_leftovers_in_grace"] += 1
        return _RawPlan(
            source_name, source_root, None, None, (), (), (), tuple(temp_removals), counts
        )

    try:
        entries = os.scandir(page_root)
    except OSError as exc:
        raise WebRetentionError("raw page store cannot be inspected") from exc
    with entries:
        for entry in entries:
            work_budget.consume(label=f"raw page store {source_name!r}")
            path = Path(entry.path)
            if entry.is_symlink() or _is_linklike(path):
                raise WebRetentionError(
                    f"raw page store contains a symlink or junction: {entry.name}"
                )
            if not entry.is_file(follow_symlinks=False):
                raise WebRetentionError(
                    f"raw page store contains a non-regular entry: {entry.name}"
                )
            if _raw_receipt_match(entry.name):
                payload, expires_at, receipt_sha = _validate_raw_receipt(
                    path=path, source_name=source_name
                )
                counts["receipts_scanned"] += 1
                raw_file = str(payload["raw_file"])
                expected = (str(payload["content_sha256"]), int(payload["byte_count"]))
                if expires_at <= now:
                    counts["expired_receipts_eligible"] += 1
                    expired_bodies.add(raw_file)
                    receipt_removals.append(_planned_file(path, sha256=receipt_sha))
                else:
                    counts["active_receipts"] += 1
                    prior = active_bodies.get(raw_file)
                    if prior is not None and prior != expected:
                        raise WebRetentionError(
                            f"active raw receipts disagree about shared body {raw_file!r}"
                        )
                    active_bodies[raw_file] = expected
                continue
            if _RAW_BODY_PATTERN.fullmatch(entry.name):
                counts["bodies_scanned"] += 1
                bodies.append(path)
                continue
            if _recognised_temp_target(entry.name):
                counts["crash_leftovers_detected"] += 1
                temps.append(path)
                continue
            counts["unrelated_files_preserved"] += 1

    body_names = {path.name for path in bodies}
    missing_active = sorted(set(active_bodies) - body_names)
    if missing_active:
        raise WebRetentionError(f"active raw receipt body is missing: {missing_active[0]!r}")

    expired_body_removals: list[_PlannedFile] = []
    orphan_body_removals: list[_PlannedFile] = []
    for path in bodies:
        name = path.name
        if name in active_bodies:
            if name in expired_bodies:
                counts["shared_bodies_preserved"] += 1
            expected_digest, expected_size = active_bodies[name]
            stat = path.stat()
            if stat.st_size != expected_size or _sha256_file(path) != expected_digest:
                raise WebRetentionError(f"active raw body failed integrity validation: {name}")
            continue
        if name in expired_bodies:
            counts["expired_bodies_eligible"] += 1
            expired_body_removals.append(_planned_file(path))
            continue
        counts["orphan_bodies_detected"] += 1
        planned = _planned_file(path)
        modified_at = datetime.fromtimestamp(planned.mtime_ns / 1_000_000_000, tz=UTC)
        if modified_at <= cutoff:
            counts["orphan_bodies_eligible"] += 1
            orphan_body_removals.append(planned)
        else:
            counts["orphan_bodies_in_grace"] += 1

    temp_removals = []
    for path in temps:
        planned = _planned_file(path)
        modified_at = datetime.fromtimestamp(planned.mtime_ns / 1_000_000_000, tz=UTC)
        if modified_at <= cutoff:
            counts["crash_leftovers_eligible"] += 1
            temp_removals.append(planned)
        else:
            counts["crash_leftovers_in_grace"] += 1
    cache_removal: _PlannedFile | None = None
    if receipt_removals and cache_path is not None:
        cache_payload, cache_sha = _read_json_object(cache_path, label="web crawl cache")
        if set(cache_payload) != {
            "schema_version",
            "source_name",
            "policy_fingerprint",
            "entries",
        }:
            raise WebRetentionError("web crawl cache fields are incomplete or unknown")
        if cache_payload.get("schema_version") != WEB_CRAWL_CACHE_SCHEMA_VERSION:
            raise WebRetentionError("unsupported web crawl cache schema")
        if cache_payload.get("source_name") != source_name:
            raise WebRetentionError("web crawl cache source does not match its exact root")
        cache_fingerprint = cache_payload.get("policy_fingerprint")
        if (
            not isinstance(cache_fingerprint, str)
            or _SHA256_PATTERN.fullmatch(cache_fingerprint) is None
        ):
            raise WebRetentionError("web crawl cache has an invalid policy fingerprint")
        if not isinstance(cache_payload.get("entries"), dict):
            raise WebRetentionError("web crawl cache entries must be an object")
        counts["cache_files_eligible"] = 1
        cache_removal = _planned_file(cache_path, sha256=cache_sha)
    return _RawPlan(
        source_name=source_name,
        source_root=source_root,
        page_root=page_root,
        cache_removal=cache_removal,
        receipt_removals=tuple(receipt_removals),
        expired_body_removals=tuple(expired_body_removals),
        orphan_body_removals=tuple(orphan_body_removals),
        temp_removals=tuple(temp_removals),
        counts=counts,
    )


def _tree_manifest(
    run_directory: Path,
    *,
    work_budget: _WorkBudget,
    reserve_revalidation: bool = False,
) -> tuple[tuple[str, str, int, int], ...]:
    if _is_linklike(run_directory) or os.path.ismount(run_directory) or not run_directory.is_dir():
        raise WebRetentionError(f"processed run is not a regular directory: {run_directory.name}")
    manifest: list[tuple[str, str, int, int]] = []
    try:
        entries = os.scandir(run_directory)
    except OSError as exc:
        raise WebRetentionError(
            f"processed run cannot be inspected safely: {run_directory.name}"
        ) from exc
    with entries:
        for entry in entries:
            work_budget.consume(
                label=f"processed run {run_directory.name!r}",
                amount=2 if reserve_revalidation else 1,
            )
            child = Path(entry.path)
            if entry.is_symlink() or _is_linklike(child):
                raise WebRetentionError(
                    f"processed run contains a symlink or junction: {run_directory.name}"
                )
            if not entry.is_file(follow_symlinks=False):
                raise WebRetentionError(
                    f"processed run contains a non-regular entry: {run_directory.name}"
                )
            stat = entry.stat(follow_symlinks=False)
            manifest.append((entry.name, "file", stat.st_size, stat.st_mtime_ns))
    return tuple(sorted(manifest))


def _validate_tombstone_manifest(
    manifest: tuple[tuple[str, str, int, int], ...],
    *,
    run_name: str,
) -> None:
    allowed = _ALLOWED_PROCESSED_DATA_FILES | {
        "manifest.json",
        "data-quality.json",
        WEB_PROCESSED_RETENTION_RECEIPT,
    }
    if any(kind != "file" or relative not in allowed for relative, kind, _size, _mtime in manifest):
        raise WebRetentionError(
            f"processed deletion tombstone contains an unknown entry: {run_name}"
        )


def _scan_processed(
    *,
    processed_root: Path,
    source_name: str,
    now: datetime,
    work_budget: _WorkBudget,
) -> _ProcessedPlan:
    source_root = _direct_source_root(processed_root, source_name, label="processed root")
    counts = {
        "runs_scanned": 0,
        "active_runs": 0,
        "expired_runs_eligible": 0,
        "deletion_tombstones_resumed": 0,
        "unrelated_files_preserved": 0,
    }
    if not source_root.exists():
        return _ProcessedPlan(source_name, None, (), counts)
    removals: list[_ProcessedRunPlan] = []
    try:
        entries = os.scandir(source_root)
    except OSError as exc:
        raise WebRetentionError("processed source root cannot be inspected") from exc
    with entries:
        for entry in entries:
            work_budget.consume(label=f"processed source root {source_name!r}")
            candidate = Path(entry.path)
            if entry.is_symlink() or _is_linklike(candidate):
                raise WebRetentionError(
                    f"processed source root contains a symlink or junction: {entry.name}"
                )
            if entry.is_file(follow_symlinks=False):
                counts["unrelated_files_preserved"] += 1
                continue
            if not entry.is_dir(follow_symlinks=False):
                raise WebRetentionError(
                    f"processed source root contains a non-regular entry: {entry.name}"
                )
            tombstone_match = _DELETING_RUN_PATTERN.fullmatch(entry.name)
            if tombstone_match is not None:
                resolved = candidate.resolve(strict=True)
                if resolved.parent != source_root or os.path.ismount(candidate):
                    raise WebRetentionError("processed deletion tombstone escaped its source root")
                manifest = _tree_manifest(
                    resolved,
                    work_budget=work_budget,
                    reserve_revalidation=True,
                )
                _validate_tombstone_manifest(manifest, run_name=entry.name)
                receipt_sha: str | None = None
                if manifest:
                    receipt_path = resolved / WEB_PROCESSED_RETENTION_RECEIPT
                    if not receipt_path.is_file() or _is_linklike(receipt_path):
                        raise WebRetentionError(
                            "non-empty processed deletion tombstone lacks its retention receipt"
                        )
                    expires_at, receipt_sha = _validate_processed_receipt(
                        receipt_path,
                        source_name,
                        run_sha256=tombstone_match.group("run"),
                        artifacts_required=False,
                    )
                    if expires_at > now:
                        raise WebRetentionError(
                            "processed deletion tombstone is not authorized for deletion"
                        )
                counts["runs_scanned"] += 1
                counts["expired_runs_eligible"] += 1
                counts["deletion_tombstones_resumed"] += 1
                removals.append(
                    _ProcessedRunPlan(
                        path=resolved,
                        run_sha256=tombstone_match.group("run"),
                        manifest=manifest,
                        receipt_sha256=receipt_sha,
                        already_quarantined=True,
                    )
                )
                continue
            if _RUN_DIRECTORY_PATTERN.fullmatch(entry.name) is None:
                raise WebRetentionError(
                    f"unsafe processed run directory {entry.name!r}; operator action required"
                )
            resolved = candidate.resolve(strict=True)
            if resolved.parent != source_root or os.path.ismount(candidate):
                raise WebRetentionError("processed run escaped its exact source root")
            counts["runs_scanned"] += 1
            receipt_path = resolved / WEB_PROCESSED_RETENTION_RECEIPT
            if not receipt_path.exists():
                raise WebRetentionError(
                    f"governed-web run {entry.name} has no retention receipt; "
                    "operator action required"
                )
            expires_at, receipt_sha = _validate_processed_receipt(
                receipt_path,
                source_name,
                run_sha256=entry.name,
            )
            expired = expires_at <= now
            manifest = _tree_manifest(
                resolved,
                work_budget=work_budget,
                reserve_revalidation=expired,
            )
            if expired:
                counts["expired_runs_eligible"] += 1
                removals.append(
                    _ProcessedRunPlan(
                        path=resolved,
                        run_sha256=entry.name,
                        manifest=manifest,
                        receipt_sha256=receipt_sha,
                    )
                )
            else:
                counts["active_runs"] += 1
    return _ProcessedPlan(source_name, source_root, tuple(removals), counts)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(64 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _remove_planned_file(target: _PlannedFile, *, exact_parent: Path) -> None:
    path = target.path
    if path.resolve().parent != exact_parent:
        raise WebRetentionError("raw retention target escaped its exact artifact root")
    if _is_linklike(path) or not path.is_file():
        raise WebRetentionError(f"raw retention target changed before deletion: {path.name}")
    stat = path.stat()
    if stat.st_size != target.size or stat.st_mtime_ns != target.mtime_ns:
        raise WebRetentionError(f"raw retention target changed before deletion: {path.name}")
    if target.sha256 is not None and _sha256_file(path) != target.sha256:
        raise WebRetentionError(f"raw retention receipt changed before deletion: {path.name}")
    path.unlink()


def _remove_run(target: _ProcessedRunPlan, *, source_root: Path) -> None:
    run_directory = target.path
    if run_directory.resolve().parent != source_root:
        raise WebRetentionError("processed retention target escaped its exact source root")
    if _is_linklike(run_directory) or os.path.ismount(run_directory):
        raise WebRetentionError(f"processed run changed before deletion: {run_directory.name}")
    current_manifest = _tree_manifest(
        run_directory,
        work_budget=_WorkBudget(limit=max(1, len(target.manifest))),
    )
    if current_manifest != target.manifest:
        raise WebRetentionError(f"processed run changed before deletion: {run_directory.name}")
    if target.already_quarantined:
        match = _DELETING_RUN_PATTERN.fullmatch(run_directory.name)
        if match is None or match.group("run") != target.run_sha256:
            raise WebRetentionError("processed deletion tombstone identity is invalid")
        _validate_tombstone_manifest(current_manifest, run_name=run_directory.name)
        if target.receipt_sha256 is not None:
            receipt_path = run_directory / WEB_PROCESSED_RETENTION_RECEIPT
            if _sha256_file(receipt_path) != target.receipt_sha256:
                raise WebRetentionError("processed deletion receipt changed before resumption")
    else:
        receipt_path = run_directory / WEB_PROCESSED_RETENTION_RECEIPT
        assert target.receipt_sha256 is not None
        if _sha256_file(receipt_path) != target.receipt_sha256:
            raise WebRetentionError(
                f"processed run receipt changed before deletion: {run_directory.name}"
            )
        before = run_directory.stat(follow_symlinks=False)
        if not stat_module.S_ISDIR(before.st_mode):
            raise WebRetentionError("processed retention target is no longer a directory")
        tombstone = source_root / (f".{target.run_sha256}.{secrets.token_hex(12)}.deleting")
        if tombstone.exists() or _is_linklike(tombstone):
            raise WebRetentionError("processed deletion tombstone unexpectedly exists")
        try:
            os.rename(run_directory, tombstone)
        except OSError as exc:
            raise WebRetentionError("processed run could not be quarantined for deletion") from exc
        after = tombstone.stat(follow_symlinks=False)
        if (
            _is_linklike(tombstone)
            or os.path.ismount(tombstone)
            or not stat_module.S_ISDIR(after.st_mode)
            or (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino)
        ):
            raise WebRetentionError("processed run identity changed during quarantine")
        run_directory = tombstone

    files = [run_directory / relative for relative, _kind, _size, _mtime in current_manifest]
    files.sort(key=lambda path: path.name == WEB_PROCESSED_RETENTION_RECEIPT)
    for path in files:
        if path.parent != run_directory:
            raise WebRetentionError("processed deletion target escaped its quarantine")
        if _is_linklike(path) or not path.is_file():
            raise WebRetentionError(f"processed run changed before deletion: {run_directory.name}")
        path.unlink()
    if _is_linklike(run_directory) or os.path.ismount(run_directory):
        raise WebRetentionError(f"processed run changed before deletion: {run_directory.name}")
    run_directory.rmdir()


def _bounded_action_sample(
    plan: _SourcePlan,
    *,
    publication_actions: tuple[str, ...] = (),
) -> tuple[str, ...]:
    sample: list[str] = []

    def add(value: str) -> None:
        if len(sample) < _REPORT_SAMPLE_LIMIT:
            sample.append(value)

    if plan.raw.cache_removal is not None:
        add("raw-cache:http-cache.json")
    for prefix, targets in (
        ("raw-receipt", plan.raw.receipt_removals),
        ("raw-expired-body", plan.raw.expired_body_removals),
        ("raw-orphan-body", plan.raw.orphan_body_removals),
        ("raw-temp", plan.raw.temp_removals),
    ):
        for raw_target in targets:
            add(f"{prefix}:{raw_target.path.name}")
    for run_target in plan.processed.run_removals:
        add(f"processed-run:{run_target.path.name}")
    for action in publication_actions:
        add(action)
    return tuple(sample)


def maintain_web_retention(
    *,
    raw_root: Path,
    processed_root: Path,
    source_names: tuple[str, ...],
    now: datetime | None = None,
    orphan_grace: timedelta = timedelta(hours=24),
    maximum_entries: int = 100_000,
    dry_run: bool = False,
) -> tuple[SourceRetentionReport, ...]:
    """Validate and maintain governed-web raw and processed artifacts.

    Every source is fully planned before any deletion begins, so an invalid receipt or path in
    one source fails the whole invocation closed.  The function needs source names, not live
    acquisition policies, and remains usable after acquisition authority has expired.
    """

    evaluated_at = now or datetime.now(UTC)
    if evaluated_at.tzinfo is None or evaluated_at.utcoffset() is None:
        raise ValueError("retention maintenance time must be timezone aware")
    evaluated_at = evaluated_at.astimezone(UTC)
    if orphan_grace < timedelta(hours=1) or orphan_grace > timedelta(days=30):
        raise ValueError("orphan grace must be between one hour and 30 days")
    if type(maximum_entries) is not int or not 1 <= maximum_entries <= 1_000_000:
        raise ValueError("maximum_entries must be between 1 and 1,000,000")
    if not source_names:
        raise ValueError("at least one governed-web source name is required")
    if len(source_names) > _MAXIMUM_SOURCES:
        raise ValueError(f"at most {_MAXIMUM_SOURCES} governed-web sources are allowed")
    unique_names = tuple(dict.fromkeys(_validated_source_name(name) for name in source_names))

    plans: list[_SourcePlan] = []
    work_budget = _WorkBudget(limit=maximum_entries)
    for source_name in unique_names:
        raw_plan = _scan_raw(
            raw_root=raw_root,
            source_name=source_name,
            now=evaluated_at,
            orphan_grace=orphan_grace,
            work_budget=work_budget,
        )
        processed_plan = _scan_processed(
            processed_root=processed_root,
            source_name=source_name,
            now=evaluated_at,
            work_budget=work_budget,
        )
        plans.append(_SourcePlan(source_name, raw_plan, processed_plan))

    # ``publication`` imports this module for receipt validation. Importing it only after this
    # module is fully initialized avoids a cycle while keeping the shared work budget global
    # across raw stores, visible runs, and private `.wp` operation trees.
    from .publication import execute_web_publication_recovery, plan_web_publication_recovery

    publication_plan = plan_web_publication_recovery(
        processed_root=processed_root,
        source_names=unique_names,
        now=evaluated_at,
        orphan_grace=orphan_grace,
        work_budget=work_budget,
    )
    publication_reports = {
        report.source_name: report
        for report in execute_web_publication_recovery(publication_plan, dry_run=dry_run)
    }

    reports: list[SourceRetentionReport] = []
    for plan in plans:
        publication_report = publication_reports[plan.source_name]
        action_sample = _bounded_action_sample(
            plan,
            publication_actions=publication_report.action_sample,
        )
        removed_receipts = 0
        removed_expired_bodies = 0
        removed_orphan_bodies = 0
        removed_temps = 0
        removed_cache_files = 0
        removed_runs = 0
        if not dry_run:
            if plan.raw.cache_removal is not None:
                assert plan.raw.source_root is not None
                _remove_planned_file(
                    plan.raw.cache_removal,
                    exact_parent=plan.raw.source_root,
                )
                removed_cache_files = 1
            if plan.raw.page_root is not None:
                for raw_target in plan.raw.expired_body_removals:
                    _remove_planned_file(raw_target, exact_parent=plan.raw.page_root)
                    removed_expired_bodies += 1
                for raw_target in plan.raw.orphan_body_removals:
                    _remove_planned_file(raw_target, exact_parent=plan.raw.page_root)
                    removed_orphan_bodies += 1
                for raw_target in plan.raw.temp_removals:
                    exact_parent = plan.raw.page_root
                    if raw_target.path.parent.resolve() != plan.raw.page_root:
                        assert plan.raw.source_root is not None
                        exact_parent = plan.raw.source_root
                    _remove_planned_file(raw_target, exact_parent=exact_parent)
                    removed_temps += 1
                for raw_target in plan.raw.receipt_removals:
                    _remove_planned_file(raw_target, exact_parent=plan.raw.page_root)
                    removed_receipts += 1
            elif plan.raw.source_root is not None:
                for raw_target in plan.raw.temp_removals:
                    _remove_planned_file(raw_target, exact_parent=plan.raw.source_root)
                    removed_temps += 1
            if plan.processed.source_root is not None:
                for run_target in plan.processed.run_removals:
                    _remove_run(
                        run_target,
                        source_root=plan.processed.source_root,
                    )
                    removed_runs += 1
        raw_counts = plan.raw.counts
        processed_counts = plan.processed.counts
        reports.append(
            SourceRetentionReport(
                source_name=plan.source_name,
                dry_run=dry_run,
                evaluated_at=evaluated_at.isoformat(),
                raw=RawRetentionReport(
                    receipts_scanned=raw_counts["receipts_scanned"],
                    active_receipts=raw_counts["active_receipts"],
                    expired_receipts_eligible=raw_counts["expired_receipts_eligible"],
                    expired_receipts_removed=removed_receipts,
                    bodies_scanned=raw_counts["bodies_scanned"],
                    expired_bodies_eligible=raw_counts["expired_bodies_eligible"],
                    expired_bodies_removed=removed_expired_bodies,
                    shared_bodies_preserved=raw_counts["shared_bodies_preserved"],
                    orphan_bodies_detected=raw_counts["orphan_bodies_detected"],
                    orphan_bodies_eligible=raw_counts["orphan_bodies_eligible"],
                    orphan_bodies_removed=removed_orphan_bodies,
                    orphan_bodies_in_grace=raw_counts["orphan_bodies_in_grace"],
                    crash_leftovers_detected=raw_counts["crash_leftovers_detected"],
                    crash_leftovers_eligible=raw_counts["crash_leftovers_eligible"],
                    crash_leftovers_removed=removed_temps,
                    crash_leftovers_in_grace=raw_counts["crash_leftovers_in_grace"],
                    cache_files_eligible=raw_counts["cache_files_eligible"],
                    cache_files_removed=removed_cache_files,
                    unrelated_files_preserved=raw_counts["unrelated_files_preserved"],
                ),
                processed=ProcessedRetentionReport(
                    runs_scanned=processed_counts["runs_scanned"],
                    active_runs=processed_counts["active_runs"],
                    expired_runs_eligible=processed_counts["expired_runs_eligible"],
                    expired_runs_removed=removed_runs,
                    deletion_tombstones_resumed=processed_counts["deletion_tombstones_resumed"],
                    unrelated_files_preserved=processed_counts["unrelated_files_preserved"],
                    publication_operations_scanned=publication_report.operations_scanned,
                    publication_operations_eligible=publication_report.operations_eligible,
                    publication_operations_removed=publication_report.operations_removed,
                    publication_operations_in_grace=publication_report.operations_in_grace,
                    published_residues_detected=(publication_report.published_residues_detected),
                    published_residues_removed=(publication_report.published_residues_removed),
                ),
                action_sample=action_sample,
            )
        )
    return tuple(reports)
