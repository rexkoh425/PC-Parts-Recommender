"""Read-only validation and migration planning for governed-web v1 receipts.

The destructive retention engine intentionally accepts only v2 receipts.  This module provides
an operator-safe bridge for legacy stores: it validates the evidence that still exists, records
content hashes for an auditable migration review, and names evidence that cannot be reconstructed.
It never writes, renames, quarantines, or deletes an artifact.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Final

from .web import (
    WEB_PROCESSED_RETENTION_RECEIPT,
    WEB_PROCESSED_RETENTION_SCHEMA_VERSION,
    WEB_RAW_METADATA_SCHEMA_VERSION,
    WebRetentionError,
    _aware_datetime,
    _bounded_file_sha256,
    _direct_pages_root,
    _direct_source_root,
    _is_linklike,
    _raw_receipt_match,
    _read_json_object,
    _required_nonempty_string,
    _validate_authority,
    _validate_processed_artifacts,
    _validate_processed_receipt,
    _validate_raw_receipt,
    _validate_rights,
    _validate_usage_scope,
    _validated_https_url,
    _WorkBudget,
)

LEGACY_WEB_RAW_METADATA_SCHEMA_VERSION: Final = "pc-build-recommender.web-raw-page.v1"
LEGACY_WEB_PROCESSED_RETENTION_SCHEMA_VERSION: Final = (
    "pc-build-recommender.web-processed-retention.v1"
)

_SHA256_LENGTH = 64
_MAXIMUM_SOURCES = 100
_MAX_RAW_BODY_BYTES = 64 * 1024 * 1024
_LEGACY_RAW_RECEIPT_FIELDS = {
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
    "etag",
    "last_modified",
    "raw_file",
}
_LEGACY_PROCESSED_RECEIPT_FIELDS = {
    "schema_version",
    "source_name",
    "policy_fingerprint",
    "usage_scope",
    "created_at",
    "retrieved_at",
    "retention_expires_at",
    "deletion_required",
}
_USAGE_SCOPES = {"internal_research", "production_catalog"}


@dataclass(frozen=True, slots=True)
class LegacyPolicyMigrationPlan:
    """Evidence and blockers for one immutable legacy policy fingerprint."""

    policy_fingerprint: str
    raw_receipt_count: int
    processed_run_count: int
    raw_authority_evidence_sha256: str | None
    processed_authority_evidence_sha256: str | None
    processed_data_use_rights_evidence_sha256: str | None
    processed_retrieval_interval_evidence_sha256: str | None
    artifact_evidence_sha256: str
    missing_evidence: tuple[str, ...]
    migration_ready: bool


@dataclass(frozen=True, slots=True)
class LegacyWebMigrationReport:
    """A zero-write migration assessment for one exact governed-web source."""

    source_name: str
    evaluated_at: str
    legacy_raw_receipts_scanned: int
    legacy_processed_runs_scanned: int
    current_raw_receipts_validated: int
    current_processed_runs_validated: int
    expired_legacy_raw_receipts: int
    expired_legacy_processed_runs: int
    migration_required: bool
    migration_ready: bool
    write_actions_planned: int
    evidence_sha256: str
    policy_plans: tuple[LegacyPolicyMigrationPlan, ...]
    blockers: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class _LegacyRawEvidence:
    policy_fingerprint: str
    receipt_name: str
    receipt_sha256: str
    body_name: str
    body_sha256: str
    authority_sha256: str
    retention_days: int
    retrieved_at: datetime
    retention_expires_at: datetime


@dataclass(frozen=True, slots=True)
class _LegacyProcessedEvidence:
    policy_fingerprint: str
    run_sha256: str
    receipt_sha256: str
    manifest_sha256: str
    data_quality_sha256: str
    authority_sha256: str
    data_use_rights_sha256: str
    expected_observation_count: int
    retention_days: int
    retrieved_at: datetime
    created_at: datetime
    retention_expires_at: datetime


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _SHA256_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _semantic_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validated_legacy_raw_receipt(
    *,
    path: Path,
    source_name: str,
    pages_root: Path,
) -> _LegacyRawEvidence:
    match = _raw_receipt_match(path.name)
    if match is None:
        raise WebRetentionError(f"unsafe legacy raw receipt filename: {path.name!r}")
    payload, receipt_sha = _read_json_object(path, label="legacy raw receipt")
    if set(payload) != _LEGACY_RAW_RECEIPT_FIELDS:
        raise WebRetentionError("legacy raw receipt fields are incomplete or unknown")
    if payload.get("schema_version") != LEGACY_WEB_RAW_METADATA_SCHEMA_VERSION:
        raise WebRetentionError("unsupported legacy raw receipt schema")
    if payload.get("source_name") != source_name:
        raise WebRetentionError("legacy raw receipt source does not match its exact source root")

    source_url = _validated_https_url(payload, "source_url")
    _validated_https_url(payload, "final_url")
    source_url_sha = hashlib.sha256(source_url.encode("utf-8")).hexdigest()
    if payload.get("source_url_sha256") != source_url_sha:
        raise WebRetentionError("legacy raw receipt source URL digest is invalid")
    content_sha = payload.get("content_sha256")
    policy_fingerprint = payload.get("policy_fingerprint")
    if not _is_sha256(content_sha):
        raise WebRetentionError("legacy raw receipt content digest is invalid")
    if not _is_sha256(policy_fingerprint):
        raise WebRetentionError("legacy raw receipt policy fingerprint is invalid")
    assert isinstance(content_sha, str)
    assert isinstance(policy_fingerprint, str)
    if (
        match.group("url") != source_url_sha[:32]
        or match.group("content") != content_sha
        or match.group("policy") != policy_fingerprint[:16]
    ):
        raise WebRetentionError("legacy raw receipt filename does not match its content")

    raw_file = payload.get("raw_file")
    expected_body_prefix = f"{source_url_sha[:32]}-{content_sha}."
    if (
        not isinstance(raw_file, str)
        or not raw_file.startswith(expected_body_prefix)
        or raw_file.rsplit(".", maxsplit=1)[-1] not in {"html", "terms", "txt"}
        or Path(raw_file).name != raw_file
    ):
        raise WebRetentionError("legacy raw receipt contains an unsafe raw_file")
    body_path = (pages_root / raw_file).resolve()
    if body_path.parent != pages_root:
        raise WebRetentionError("legacy raw receipt body escaped its exact page root")
    byte_count = payload.get("byte_count")
    if type(byte_count) is not int or not 0 <= byte_count <= _MAX_RAW_BODY_BYTES:
        raise WebRetentionError("legacy raw receipt has an invalid byte_count")
    body_sha, body_bytes = _bounded_file_sha256(
        body_path,
        label="legacy raw body",
        maximum_bytes=_MAX_RAW_BODY_BYTES,
    )
    if body_sha != content_sha or body_bytes != byte_count:
        raise WebRetentionError("legacy raw body does not match its receipt")

    if payload.get("source_type") != "retailer":
        raise WebRetentionError("legacy raw receipt has an invalid source_type")
    for field_name in ("media_type", "parser_version", "licence_or_access_note"):
        _required_nonempty_string(payload, field_name, label="legacy raw receipt")
    for field_name in ("etag", "last_modified"):
        value = payload.get(field_name)
        if value is not None and not isinstance(value, str):
            raise WebRetentionError(f"legacy raw receipt has invalid {field_name!r}")
    if payload.get("usage_scope") not in _USAGE_SCOPES:
        raise WebRetentionError("legacy raw receipt has an invalid usage_scope")

    retrieved_at = _aware_datetime(payload, "retrieved_at", label="legacy raw receipt")
    retention_expires_at = _aware_datetime(
        payload,
        "retention_expires_at",
        label="legacy raw receipt",
    )
    if retention_expires_at <= retrieved_at:
        raise WebRetentionError("legacy raw receipt expiry must follow retrieval")
    authority = payload.get("acquisition_authority")
    _validate_authority(
        authority,
        retrieved_at=retrieved_at,
        retention_expires_at=retention_expires_at,
    )
    assert isinstance(authority, dict)
    if (
        payload.get("usage_scope") == "internal_research"
        and authority.get("permits_internal_analysis") is not True
    ):
        raise WebRetentionError(
            "legacy raw research receipt lacks internal-analysis authority"
        )
    retention_days = authority.get("retention_days")
    assert isinstance(retention_days, int)
    return _LegacyRawEvidence(
        policy_fingerprint=policy_fingerprint,
        receipt_name=path.name,
        receipt_sha256=receipt_sha,
        body_name=raw_file,
        body_sha256=body_sha,
        authority_sha256=_semantic_sha256(authority),
        retention_days=retention_days,
        retrieved_at=retrieved_at,
        retention_expires_at=retention_expires_at,
    )


def _validated_legacy_processed_receipt(
    *,
    run_directory: Path,
    source_name: str,
) -> _LegacyProcessedEvidence:
    run_sha = run_directory.name
    if not _is_sha256(run_sha):
        raise WebRetentionError(
            f"unsafe legacy processed run directory {run_sha!r}; operator action required"
        )
    receipt_path = run_directory / WEB_PROCESSED_RETENTION_RECEIPT
    payload, receipt_sha = _read_json_object(
        receipt_path,
        label="legacy processed retention receipt",
    )
    if set(payload) != _LEGACY_PROCESSED_RECEIPT_FIELDS:
        raise WebRetentionError(
            "legacy processed retention receipt fields are incomplete or unknown"
        )
    if payload.get("schema_version") != LEGACY_WEB_PROCESSED_RETENTION_SCHEMA_VERSION:
        raise WebRetentionError("unsupported legacy processed retention receipt schema")
    if payload.get("source_name") != source_name:
        raise WebRetentionError(
            "legacy processed retention receipt source does not match its root"
        )
    policy_fingerprint = payload.get("policy_fingerprint")
    if not _is_sha256(policy_fingerprint):
        raise WebRetentionError(
            "legacy processed retention receipt has an invalid policy fingerprint"
        )
    assert isinstance(policy_fingerprint, str)
    if payload.get("usage_scope") not in _USAGE_SCOPES:
        raise WebRetentionError(
            "legacy processed retention receipt has an invalid usage_scope"
        )
    if payload.get("deletion_required") is not True:
        raise WebRetentionError("legacy governed processed retention must require deletion")

    retrieved_at = _aware_datetime(
        payload,
        "retrieved_at",
        label="legacy processed retention receipt",
    )
    created_at = _aware_datetime(
        payload,
        "created_at",
        label="legacy processed retention receipt",
    )
    retention_expires_at = _aware_datetime(
        payload,
        "retention_expires_at",
        label="legacy processed retention receipt",
    )
    if created_at < retrieved_at:
        raise WebRetentionError(
            "legacy processed retention receipt predates its retrieval"
        )
    retention_delta = retention_expires_at - retrieved_at
    if retention_delta <= timedelta(0) or retention_delta.seconds or retention_delta.microseconds:
        raise WebRetentionError(
            "legacy processed retention expiry is not a positive whole-day interval"
        )
    retention_days = retention_delta.days
    if not 1 <= retention_days <= 3650:
        raise WebRetentionError(
            "legacy processed retention interval is outside the supported range"
        )

    manifest, manifest_sha, quality_sha = _validate_processed_artifacts(
        run_directory,
        source_name=source_name,
        run_sha256=run_sha,
        receipt_required=True,
    )
    statistics = manifest.get("statistics")
    assert isinstance(statistics, dict)
    if statistics.get("policy_fingerprint") != policy_fingerprint:
        raise WebRetentionError(
            "legacy processed manifest statistics do not match the receipt policy fingerprint"
        )
    if statistics.get("usage_scope") != payload.get("usage_scope"):
        raise WebRetentionError(
            "legacy processed manifest statistics do not match the receipt usage_scope"
        )
    authority = statistics.get("acquisition_authority")
    _validate_authority(
        authority,
        retrieved_at=retrieved_at,
        retention_expires_at=retention_expires_at,
    )
    rights = statistics.get("data_use_rights")
    _validate_rights(
        rights,
        retrieved_at=retrieved_at,
        retention_expires_at=retention_expires_at,
    )
    _validate_usage_scope(
        usage_scope=payload.get("usage_scope"),
        authority=authority,
        rights=rights,
        label="legacy processed manifest statistics",
    )
    assert isinstance(authority, dict)
    assert isinstance(rights, dict)
    if authority.get("retention_days") != retention_days:
        raise WebRetentionError(
            "legacy processed authority retention does not match its receipt"
        )
    pages_requested = statistics.get("pages_requested")
    robots_hashes = statistics.get("robots_sha256_by_host")
    if (
        type(pages_requested) is not int
        or pages_requested < 0
        or not isinstance(robots_hashes, dict)
        or any(
            not isinstance(host, str)
            or not host
            or not _is_sha256(digest)
            for host, digest in robots_hashes.items()
        )
    ):
        raise WebRetentionError(
            "legacy processed manifest has invalid observation-count evidence"
        )
    control_observations = len(robots_hashes) + (
        2 if "terms_post_receipt_sha256" in statistics else 1
    )
    return _LegacyProcessedEvidence(
        policy_fingerprint=policy_fingerprint,
        run_sha256=run_sha,
        receipt_sha256=receipt_sha,
        manifest_sha256=manifest_sha,
        data_quality_sha256=quality_sha,
        authority_sha256=_semantic_sha256(authority),
        data_use_rights_sha256=_semantic_sha256(rights),
        expected_observation_count=pages_requested + control_observations,
        retention_days=retention_days,
        retrieved_at=retrieved_at,
        created_at=created_at,
        retention_expires_at=retention_expires_at,
    )


def _scan_raw(
    *,
    raw_root: Path,
    source_name: str,
    work_budget: _WorkBudget,
) -> tuple[list[_LegacyRawEvidence], int]:
    source_root = _direct_source_root(raw_root, source_name, label="raw root")
    if not source_root.exists():
        return [], 0
    pages_root = _direct_pages_root(source_root)
    if not pages_root.exists():
        return [], 0
    legacy: list[_LegacyRawEvidence] = []
    current = 0
    with os.scandir(pages_root) as entries:
        for entry in entries:
            work_budget.consume(label=f"legacy raw source {source_name!r}")
            path = Path(entry.path)
            if entry.is_symlink() or _is_linklike(path):
                raise WebRetentionError(
                    f"legacy raw page store contains a symlink or junction: {entry.name}"
                )
            if not entry.is_file(follow_symlinks=False):
                raise WebRetentionError(
                    f"legacy raw page store contains a non-regular entry: {entry.name}"
                )
            if _raw_receipt_match(entry.name) is None:
                continue
            payload, _receipt_sha = _read_json_object(path, label="raw receipt")
            schema = payload.get("schema_version")
            if schema == LEGACY_WEB_RAW_METADATA_SCHEMA_VERSION:
                legacy.append(
                    _validated_legacy_raw_receipt(
                        path=path,
                        source_name=source_name,
                        pages_root=pages_root,
                    )
                )
            elif schema == WEB_RAW_METADATA_SCHEMA_VERSION:
                _validate_raw_receipt(path=path, source_name=source_name)
                current += 1
            else:
                raise WebRetentionError("unsupported raw receipt schema")
    return legacy, current


def _scan_processed(
    *,
    processed_root: Path,
    source_name: str,
    work_budget: _WorkBudget,
) -> tuple[list[_LegacyProcessedEvidence], int]:
    source_root = _direct_source_root(processed_root, source_name, label="processed root")
    if not source_root.exists():
        return [], 0
    legacy: list[_LegacyProcessedEvidence] = []
    current = 0
    with os.scandir(source_root) as entries:
        for entry in entries:
            work_budget.consume(label=f"legacy processed source {source_name!r}")
            run_directory = Path(entry.path)
            if (
                entry.is_symlink()
                or _is_linklike(run_directory)
                or not entry.is_dir(follow_symlinks=False)
            ):
                raise WebRetentionError(
                    f"legacy processed source contains an unsafe entry: {entry.name}"
                )
            if not _is_sha256(entry.name):
                raise WebRetentionError(
                    f"unsafe legacy processed run directory {entry.name!r}; "
                    "operator action required"
                )
            receipt_path = run_directory / WEB_PROCESSED_RETENTION_RECEIPT
            if not receipt_path.exists():
                raise WebRetentionError(
                    f"governed-web run {entry.name} has no retention receipt; "
                    "operator action required"
                )
            payload, _receipt_sha = _read_json_object(
                receipt_path,
                label="processed retention receipt",
            )
            schema = payload.get("schema_version")
            if schema == LEGACY_WEB_PROCESSED_RETENTION_SCHEMA_VERSION:
                legacy.append(
                    _validated_legacy_processed_receipt(
                        run_directory=run_directory,
                        source_name=source_name,
                    )
                )
            elif schema == WEB_PROCESSED_RETENTION_SCHEMA_VERSION:
                _validate_processed_receipt(
                    receipt_path,
                    source_name,
                    run_sha256=entry.name,
                )
                current += 1
            else:
                raise WebRetentionError(
                    "unsupported processed retention receipt schema"
                )
    return legacy, current


def _policy_plan(
    *,
    policy_fingerprint: str,
    raw: list[_LegacyRawEvidence],
    processed: list[_LegacyProcessedEvidence],
) -> tuple[LegacyPolicyMigrationPlan, tuple[str, ...]]:
    authority_hashes = {item.authority_sha256 for item in raw}
    if len(authority_hashes) > 1:
        raise WebRetentionError(
            f"legacy raw receipts disagree about authority for policy {policy_fingerprint}"
        )
    processed_authority_hashes = {item.authority_sha256 for item in processed}
    if len(processed_authority_hashes) > 1:
        raise WebRetentionError(
            f"legacy processed manifests disagree about authority for "
            f"policy {policy_fingerprint}"
        )
    if (
        authority_hashes
        and processed_authority_hashes
        and authority_hashes != processed_authority_hashes
    ):
        raise WebRetentionError(
            f"legacy raw receipts and processed manifests disagree about authority for "
            f"policy {policy_fingerprint}"
        )
    rights_hashes = {item.data_use_rights_sha256 for item in processed}
    if len(rights_hashes) > 1:
        raise WebRetentionError(
            f"legacy processed manifests disagree about data-use rights for "
            f"policy {policy_fingerprint}"
        )
    raw_retention_days = {item.retention_days for item in raw}
    processed_retention_days = {item.retention_days for item in processed}
    if (
        raw_retention_days
        and processed_retention_days
        and raw_retention_days != processed_retention_days
    ):
        raise WebRetentionError(
            f"legacy raw and processed receipts disagree about retention for "
            f"policy {policy_fingerprint}"
        )

    missing: list[str] = []
    blockers: list[str] = []
    interval_evidence: list[dict[str, Any]] = []
    if raw and not rights_hashes:
        missing.append("data_use_rights")
        blockers.append(
            f"policy {policy_fingerprint}: raw v1 receipts omit data_use_rights and no "
            "matching processed manifest evidence remains"
        )
    if processed:
        unresolved_runs: list[str] = []
        retention_anchor_mismatches: list[str] = []
        runs_with_observations: list[
            tuple[_LegacyProcessedEvidence, list[_LegacyRawEvidence]]
        ] = []
        if len(processed) == 1:
            runs_with_observations.append((processed[0], raw))
        else:
            for run in processed:
                runs_with_observations.append((run, []))
        for run, candidate_observations in sorted(
            runs_with_observations,
            key=lambda item: item[0].run_sha256,
        ):
            observations = sorted(
                candidate_observations,
                key=lambda item: (item.retrieved_at, item.receipt_name),
            )
            if (
                len(observations) != run.expected_observation_count
                or not observations
                or observations[-1].retrieved_at > run.created_at
            ):
                unresolved_runs.append(run.run_sha256)
                continue
            interval_evidence.append(
                {
                    "run_sha256": run.run_sha256,
                    "retrieval_started_at": observations[0].retrieved_at.isoformat(),
                    "retrieval_completed_at": observations[-1].retrieved_at.isoformat(),
                    "observation_count": len(observations),
                    "receipt_sha256s": sorted(
                        item.receipt_sha256 for item in observations
                    ),
                }
            )
            if (
                run.retention_expires_at
                != observations[0].retrieved_at + timedelta(days=run.retention_days)
            ):
                retention_anchor_mismatches.append(run.run_sha256)
        if unresolved_runs:
            missing.append("processed_retrieval_interval")
            blockers.append(
                f"policy {policy_fingerprint}: processed v1 run(s) "
                f"{','.join(unresolved_runs)} lack the complete raw observation receipts "
                "needed to recover retrieval_started_at and retrieval_completed_at"
            )
        if retention_anchor_mismatches:
            missing.append("processed_retention_anchor")
            blockers.append(
                f"policy {policy_fingerprint}: processed v1 run(s) "
                f"{','.join(retention_anchor_mismatches)} anchor retention to the first "
                "product retrieval rather than the recovered full-observation start; "
                "a reviewed migration decision is required"
            )

    evidence = {
        "policy_fingerprint": policy_fingerprint,
        "raw": [
            {
                "receipt_name": item.receipt_name,
                "receipt_sha256": item.receipt_sha256,
                "body_name": item.body_name,
                "body_sha256": item.body_sha256,
                "authority_sha256": item.authority_sha256,
                "retention_days": item.retention_days,
                "retrieved_at": item.retrieved_at.isoformat(),
                "retention_expires_at": item.retention_expires_at.isoformat(),
            }
            for item in sorted(raw, key=lambda item: item.receipt_name)
        ],
        "processed": [
            {
                "run_sha256": item.run_sha256,
                "receipt_sha256": item.receipt_sha256,
                "manifest_sha256": item.manifest_sha256,
                "data_quality_sha256": item.data_quality_sha256,
                "authority_sha256": item.authority_sha256,
                "data_use_rights_sha256": item.data_use_rights_sha256,
                "expected_observation_count": item.expected_observation_count,
                "retention_days": item.retention_days,
                "retrieved_at": item.retrieved_at.isoformat(),
                "created_at": item.created_at.isoformat(),
                "retention_expires_at": item.retention_expires_at.isoformat(),
            }
            for item in sorted(processed, key=lambda item: item.run_sha256)
        ],
        "recovered_intervals": interval_evidence,
    }
    plan = LegacyPolicyMigrationPlan(
        policy_fingerprint=policy_fingerprint,
        raw_receipt_count=len(raw),
        processed_run_count=len(processed),
        raw_authority_evidence_sha256=next(iter(authority_hashes), None),
        processed_authority_evidence_sha256=next(
            iter(processed_authority_hashes),
            None,
        ),
        processed_data_use_rights_evidence_sha256=next(iter(rights_hashes), None),
        processed_retrieval_interval_evidence_sha256=(
            _semantic_sha256(interval_evidence)
            if len(interval_evidence) == len(processed) and processed
            else None
        ),
        artifact_evidence_sha256=_semantic_sha256(evidence),
        missing_evidence=tuple(missing),
        migration_ready=not missing,
    )
    return plan, tuple(blockers)


def plan_legacy_web_retention_migration(
    *,
    raw_root: Path,
    processed_root: Path,
    source_names: tuple[str, ...],
    now: datetime | None = None,
    maximum_entries: int = 100_000,
) -> tuple[LegacyWebMigrationReport, ...]:
    """Validate legacy v1 evidence and return a deterministic zero-write migration plan."""

    evaluated_at = now or datetime.now(UTC)
    if evaluated_at.tzinfo is None or evaluated_at.utcoffset() is None:
        raise ValueError("legacy migration planning time must be timezone aware")
    evaluated_at = evaluated_at.astimezone(UTC)
    if type(maximum_entries) is not int or not 1 <= maximum_entries <= 1_000_000:
        raise ValueError("maximum_entries must be between 1 and 1,000,000")
    if not source_names:
        raise ValueError("at least one governed-web source name is required")
    unique_names = tuple(dict.fromkeys(source_names))
    if len(unique_names) > _MAXIMUM_SOURCES:
        raise ValueError(f"at most {_MAXIMUM_SOURCES} governed-web sources are allowed")

    work_budget = _WorkBudget(limit=maximum_entries)
    reports: list[LegacyWebMigrationReport] = []
    for source_name in unique_names:
        raw, current_raw = _scan_raw(
            raw_root=raw_root,
            source_name=source_name,
            work_budget=work_budget,
        )
        processed, current_processed = _scan_processed(
            processed_root=processed_root,
            source_name=source_name,
            work_budget=work_budget,
        )
        fingerprints = sorted(
            {item.policy_fingerprint for item in raw}
            | {item.policy_fingerprint for item in processed}
        )
        policy_plans: list[LegacyPolicyMigrationPlan] = []
        blockers: list[str] = []
        for fingerprint in fingerprints:
            plan, policy_blockers = _policy_plan(
                policy_fingerprint=fingerprint,
                raw=[item for item in raw if item.policy_fingerprint == fingerprint],
                processed=[
                    item for item in processed if item.policy_fingerprint == fingerprint
                ],
            )
            policy_plans.append(plan)
            blockers.extend(policy_blockers)

        evidence = {
            "source_name": source_name,
            "current_raw_receipts_validated": current_raw,
            "current_processed_runs_validated": current_processed,
            "policy_plans": [asdict(plan) for plan in policy_plans],
        }
        migration_required = bool(raw or processed)
        reports.append(
            LegacyWebMigrationReport(
                source_name=source_name,
                evaluated_at=evaluated_at.isoformat(),
                legacy_raw_receipts_scanned=len(raw),
                legacy_processed_runs_scanned=len(processed),
                current_raw_receipts_validated=current_raw,
                current_processed_runs_validated=current_processed,
                expired_legacy_raw_receipts=sum(
                    item.retention_expires_at <= evaluated_at for item in raw
                ),
                expired_legacy_processed_runs=sum(
                    item.retention_expires_at <= evaluated_at for item in processed
                ),
                migration_required=migration_required,
                migration_ready=not blockers,
                write_actions_planned=0,
                evidence_sha256=_semantic_sha256(evidence),
                policy_plans=tuple(policy_plans),
                blockers=tuple(blockers),
            )
        )
    return tuple(reports)


__all__ = [
    "LEGACY_WEB_PROCESSED_RETENTION_SCHEMA_VERSION",
    "LEGACY_WEB_RAW_METADATA_SCHEMA_VERSION",
    "LegacyPolicyMigrationPlan",
    "LegacyWebMigrationReport",
    "plan_legacy_web_retention_migration",
]
