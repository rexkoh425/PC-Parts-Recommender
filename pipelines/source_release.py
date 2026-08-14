"""Fail-closed release contract for signed production retailer batches.

The ingestion adapters deliberately keep raw data, processed data, and policy
documents as separate artifacts.  Production admission must not trust one of
those artifacts in isolation.  This module creates a small, content-addressed
control bundle and independently revalidates the complete chain:

* the current source registry and its signed-feed template;
* an externally pinned Ed25519 trust root and signed per-feed policy;
* the immutable authorization receipt and raw snapshot metadata;
* the streaming parser manifest, every normalized record's authority, and the
  passing data-quality report; and
* the exact raw, accepted-record, and rejection bytes.

Only the bounded control documents are copied into the release bundle.  Large
data files remain content-addressed external artifacts and are streamed during
verification, keeping memory use independent of catalogue size.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import re
import shutil
import stat
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, Final
from urllib.parse import urlsplit

from pc_build_recommender.domain.models import PriceSnapshot, RetailerListing
from pipelines.checks.quality import DATA_QUALITY_SCHEMA_VERSION
from pipelines.parsing.normalizers import NORMALISED_RECORD_SCHEMA_VERSION, stable_identifier
from pipelines.parsing.streaming_writer import STREAMING_MANIFEST_SCHEMA_VERSION
from pipelines.retention.registry import SOURCE_REGISTRY_SCHEMA_VERSION, _load_strict_yaml
from pipelines.sources.awin_feed import (
    AWIN_AUTHORIZATION_RECEIPT_SCHEMA_VERSION,
    AWIN_PARSER_VERSION,
    AwinFeedPolicy,
    AwinFeedSnapshot,
    AwinLocalFeedAdapter,
)
from pipelines.sources.base import RAW_SNAPSHOT_SCHEMA_VERSION, RawSnapshot
from pipelines.sources.signed_policy import VerifiedSignedPolicy, verify_signed_policy

AUTHORIZED_BATCH_RELEASE_SCHEMA_VERSION: Final = (
    "pc-build-recommender.authorized-source-batch-release.v1"
)
AWIN_SOURCE_TEMPLATE: Final = "awin_authorized_local_feed"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SOURCE_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
_MAX_CONTROL_BYTES = 4 * 1024 * 1024
_MAX_REGISTRY_BYTES = 1024 * 1024
_MAX_EXTERNAL_BYTES = 8 * 1024 * 1024 * 1024
_MAX_JSONL_LINE_BYTES = 8 * 1024 * 1024
_MAX_REGISTRY_AGE_DAYS = 30

_BUNDLE_NAMES: Final = {
    "authorization_receipt": "authorization-receipt.json",
    "data_quality": "data-quality.json",
    "policy": "source-policy.json",
    "policy_signature": "source-policy.sig.json",
    "processed_manifest": "processed-manifest.json",
    "raw_metadata": "raw-snapshot.metadata.json",
    "source_registry": "source-registry.yaml",
    "trust_root": "policy-trust-root.json",
}
_RELEASE_FIELDS: Final = frozenset(
    {
        "schema_version",
        "source_name",
        "source_uri",
        "authority",
        "source_registry",
        "raw_snapshot",
        "processed_batch",
        "bundle_files",
        "external_files",
        "content_sha256",
    }
)
_RAW_METADATA_FIELDS: Final = frozenset(
    {
        "schema_version",
        "source_name",
        "source_url",
        "source_type",
        "retrieved_at",
        "content_sha256",
        "byte_count",
        "media_type",
        "parser_version",
        "licence_or_access_note",
        "raw_file",
    }
)
_AUTHORIZATION_RECEIPT_FIELDS: Final = frozenset(
    {
        "schema_version",
        "source_name",
        "source_uri",
        "raw_snapshot_sha256",
        "raw_byte_count",
        "parser_version",
        "policy_id",
        "policy_issued_at",
        "policy_expires_at",
        "policy_sha256",
        "signature_sha256",
        "trust_root_sha256",
        "signer_key_id",
        "data_use_rights",
        "production_catalog_eligible",
        "training_eligible",
        "published_claims_eligible",
        "published_claims_grant_reference",
        "content_sha256",
    }
)
_STREAMING_MANIFEST_FIELDS: Final = frozenset(
    {
        "schema_version",
        "source_name",
        "run_sha256",
        "accepted_count",
        "rejected_count",
        "files",
        "metadata",
        "content_sha256",
    }
)
_QUALITY_FIELDS: Final = frozenset(
    {
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
)
_REGISTRY_ROOT_FIELDS: Final = frozenset(
    {
        "schema_version",
        "verified_on",
        "sources",
        "source_templates",
        "auxiliary_sources",
        "blocked_or_restricted_sources",
    }
)
_REQUIRED_TEMPLATE_REQUIREMENTS: Final = frozenset(
    {
        "already_downloaded_authorized_local_csv_or_gzip",
        "detached_ed25519_policy_signature",
        "independently_pinned_trust_root_sha256",
        "exact_advertiser_feed_merchant_host_and_category_mapping",
        "exact_sgd_currency_and_explicit_shipping",
        "bounded_input_decompression_records_rejections_and_output",
        "separate_signed_contract_reference_for_published_claims",
    }
)
_AWIN_TEMPLATE_FIELDS: Final = frozenset(
    {
        "kind",
        "parser_version",
        "access",
        "scheduled_fetch",
        "source_url",
        "version_policy",
        "licence",
        "attribution_required",
        "production_catalog_eligible",
        "training_eligible",
        "published_claims_eligible",
        "redistribution_eligible",
        "requirements",
        "data_use_rights",
        "access_note",
    }
)
_AWIN_ADMISSION_FIELDS: Final = frozenset(
    {
        "kind",
        "template",
        "status",
        "source_url",
        "advertiser_id",
        "feed_id",
        "retailer",
        "parser_version",
        "policy_id",
        "policy_sha256",
        "admitted_on",
        "admission_expires_on",
        "revoked_on",
        "revocation_reason",
        "production_catalog_eligible",
        "access_note",
    }
)
_AWIN_REPLAY_BASE_CHECKS: Final = frozenset(
    {
        "accepted_records_present",
        "signed_policy_verified",
        "rejection_rate_within_threshold",
        "production_catalog_rights",
    }
)
_AWIN_REPLAY_REGRESSION_CHECKS: Final = frozenset(
    {
        "accepted_count_regression",
        "retailer_listing_count_regression",
        "record_type_regression",
        "category_count_regression",
        "rejection_rate_regression",
    }
)
_AWIN_TEMPLATE_RIGHTS: Final = {
    "rights_status": "requires_per_feed_signed_contract",
    "contract_reference": "supplied-per-policy",
    "contract_version_url": "supplied-per-policy",
    "consent_effective_on": "supplied-per-policy",
    "consent_expires_on": "supplied-per-policy",
    "retention_days": "supplied-per-policy",
    "deletion_required_on_termination": "supplied-per-policy",
    "deletion_sla_days": "supplied-per-policy",
    "territories": "supplied-per-policy",
    "may_display": False,
    "may_cache": False,
    "may_store_history": False,
    "may_redistribute": False,
    "may_embed": False,
    "may_train": False,
    "may_derive": False,
}
_QUALITY_CHECK_FIELDS: Final = frozenset({"name", "severity", "count", "message"})
_QUALITY_STATISTICS_FIELDS: Final = frozenset(
    {
        "advertiser_id",
        "feed_id",
        "decompressed_bytes",
        "scanned_records",
        "accepted_records",
        "rejected_records",
        "rejection_reasons",
        "category_counts",
        "unique_source_listing_ids",
        "production_catalog_eligible",
        "training_eligible",
        "published_claims_eligible",
        "published_claims_grant_reference",
        "data_use_rights",
    }
)
_ACCEPTED_RECORD_FIELDS: Final = frozenset(
    {
        "schema_version",
        "record_type",
        "source_record_id",
        "archive_snapshot_sha256",
        "raw_record_sha256",
        "development_only",
        "training_eligible",
        "published_claims_eligible",
        "data_use_rights",
        "rights_authority",
        "provenance",
        "normalisation_metadata",
        "data",
    }
)
_RECORD_AUTHORITY_FIELDS: Final = frozenset(
    {
        "policy_id",
        "policy_issued_at",
        "policy_expires_at",
        "policy_sha256",
        "signature_sha256",
        "trust_root_sha256",
        "signer_key_id",
        "published_claims_grant_reference",
    }
)
_RECORD_PROVENANCE_FIELDS: Final = frozenset(
    {
        "source_name",
        "source_url",
        "source_type",
        "retrieved_at",
        "parser_version",
        "licence_or_access_note",
        "extraction_confidence",
    }
)
_RECORD_NORMALISATION_FIELDS: Final = frozenset(
    {
        "row_number",
        "category",
        "canonical_mapping_status",
        "advertiser_id",
        "feed_id",
        "brand",
        "manufacturer_part_number",
        "gtin",
        "colour",
        "price_field",
        "stock_basis",
        "source_last_updated",
    }
)
_RECORD_DATA_FIELDS: Final = frozenset({"listing", "price_snapshot"})
_RETAILER_LISTING_FIELDS: Final = frozenset(RetailerListing.model_fields)
_PRICE_SNAPSHOT_FIELDS: Final = frozenset(PriceSnapshot.model_fields)


class AuthorizedBatchReleaseError(RuntimeError):
    """Raised when a source batch cannot be admitted as a production release."""


@dataclass(frozen=True, slots=True)
class AuthorizedBatchReleaseArtifacts:
    """Large and small ingestion artifacts that one release must bind."""

    raw_snapshot: Path
    raw_metadata: Path
    authorization_receipt: Path
    records: Path
    rejections: Path
    processed_manifest: Path
    quality_report: Path


@dataclass(frozen=True, slots=True)
class AuthorizedBatchAuthorityArtifacts:
    """Exact authority inputs; no self-asserted policy shape is accepted."""

    policy: Path
    policy_signature: Path
    trust_root: Path
    source_registry: Path


@dataclass(frozen=True, slots=True)
class VerifiedAuthorizedBatchRelease:
    """Identity returned only after the complete release has been revalidated."""

    manifest_path: Path
    manifest_sha256: str
    content_sha256: str
    source_name: str
    raw_snapshot_sha256: str
    processed_run_sha256: str
    accepted_count: int
    rejected_count: int
    policy_sha256: str
    source_registry_sha256: str
    authority_expires_at: datetime
    reused: bool = False


@dataclass(frozen=True, slots=True)
class _FileMaterial:
    sha256: str
    byte_count: int
    payload: bytes | None = None
    line_count: int | None = None

    def reference(self) -> dict[str, int | str]:
        return {"sha256": self.sha256, "byte_count": self.byte_count}


@dataclass(frozen=True, slots=True)
class _CollectedRelease:
    payload: dict[str, Any]
    bundle_bytes: dict[str, bytes]


def _json_default(value: object) -> str:
    """Serialize YAML date scalars deterministically in release evidence."""

    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            default=_json_default,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise AuthorizedBatchReleaseError("release evidence contains invalid JSON") from error


def _pretty_json(value: object) -> bytes:
    try:
        return (
            json.dumps(
                value,
                allow_nan=False,
                default=_json_default,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ).encode("utf-8")
            + b"\n"
        )
    except (TypeError, ValueError) as error:
        raise AuthorizedBatchReleaseError("release evidence contains invalid JSON") from error


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise AuthorizedBatchReleaseError(f"duplicate JSON key is forbidden: {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise AuthorizedBatchReleaseError(f"non-finite JSON number is forbidden: {value}")


def _finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise AuthorizedBatchReleaseError(f"non-finite JSON number is forbidden: {value}")
    return parsed


def _json_object(raw: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
            parse_float=_finite_float,
        )
    except AuthorizedBatchReleaseError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError) as error:
        raise AuthorizedBatchReleaseError(f"{label} is not strict UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise AuthorizedBatchReleaseError(f"{label} must be a JSON object")
    return value


def _exact_fields(
    value: Mapping[str, Any],
    expected: frozenset[str],
    *,
    label: str,
) -> None:
    actual = frozenset(value)
    if actual != expected:
        raise AuthorizedBatchReleaseError(
            f"{label} fields do not match the contract; "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )


def _sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise AuthorizedBatchReleaseError(f"{label} must be a lowercase SHA-256")
    return value


def _aware_datetime(value: object, *, label: str) -> datetime:
    if not isinstance(value, str):
        raise AuthorizedBatchReleaseError(f"{label} must be a timezone-aware timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise AuthorizedBatchReleaseError(f"{label} must be a valid timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise AuthorizedBatchReleaseError(f"{label} must be timezone-aware")
    return parsed.astimezone(UTC)


def _calendar_date(value: object, *, label: str) -> date:
    if type(value) is date:
        return value
    if not isinstance(value, str):
        raise AuthorizedBatchReleaseError(f"{label} must be an ISO calendar date")
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise AuthorizedBatchReleaseError(f"{label} must be an ISO calendar date") from error


def _verification_time(now: datetime | None) -> datetime:
    result = datetime.now(UTC) if now is None else now
    if result.tzinfo is None or result.utcoffset() is None:
        raise AuthorizedBatchReleaseError("verification time must be timezone-aware")
    return result.astimezone(UTC)


def _signer_key_valid_until(material: _FileMaterial, *, key_id: str) -> datetime:
    payload = _json_object(
        _captured_payload(material, label="policy trust root"),
        label="policy trust root",
    )
    keys = payload.get("keys")
    if not isinstance(keys, list):
        raise AuthorizedBatchReleaseError("policy trust root keys must be an array")
    matches = [item for item in keys if isinstance(item, dict) and item.get("key_id") == key_id]
    if len(matches) != 1:
        raise AuthorizedBatchReleaseError("verified signer key is not unique in the trust root")
    return _aware_datetime(
        matches[0].get("valid_until"),
        label="verified signer key valid_until",
    )


def _consent_expires_at(consent_expires_on: date | None) -> datetime | None:
    if consent_expires_on is None:
        return None
    if consent_expires_on == date.max:
        return datetime.max.replace(tzinfo=UTC)
    return datetime.combine(consent_expires_on + timedelta(days=1), datetime.min.time(), UTC)


def _is_linklike(path: Path, result: os.stat_result | None = None) -> bool:
    if path.is_symlink():
        return True
    is_junction = getattr(os.path, "isjunction", None)
    if is_junction is not None and is_junction(path):
        return True
    if result is None:
        result = os.lstat(path)
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    attributes = getattr(result, "st_file_attributes", 0)
    return stat.S_ISLNK(result.st_mode) or bool(reparse_flag and attributes & reparse_flag)


def _path_components(path: Path) -> tuple[Path, ...]:
    current = Path(os.path.abspath(path))
    result: list[Path] = []
    while True:
        result.append(current)
        if current.parent == current:
            break
        current = current.parent
    result.reverse()
    return tuple(result)


def _assert_regular_file(path: Path, *, label: str) -> os.stat_result:
    components = _path_components(path)
    target: os.stat_result | None = None
    for index, component in enumerate(components):
        try:
            result = os.lstat(component)
        except OSError as error:
            raise AuthorizedBatchReleaseError(f"{label} is unavailable") from error
        if _is_linklike(component, result):
            raise AuthorizedBatchReleaseError(f"{label} path must not contain links or junctions")
        if index < len(components) - 1 and not stat.S_ISDIR(result.st_mode):
            raise AuthorizedBatchReleaseError(f"{label} parent is not a directory")
        target = result
    if target is None or not stat.S_ISREG(target.st_mode):
        raise AuthorizedBatchReleaseError(f"{label} must be a regular file")
    return target


def _identity(result: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        result.st_dev,
        result.st_ino,
        stat.S_IFMT(result.st_mode),
        result.st_size,
        result.st_mtime_ns,
    )


def _inspect_file(
    path: Path,
    *,
    label: str,
    maximum_bytes: int,
    capture: bool,
    line_validator: Callable[[dict[str, Any], int], None] | None = None,
) -> _FileMaterial:
    """Hash one stable file with bounded memory and optional JSONL validation."""

    absolute = Path(os.path.abspath(path))
    before_path = _assert_regular_file(absolute, label=label)
    if before_path.st_size > maximum_bytes:
        raise AuthorizedBatchReleaseError(f"{label} exceeds its {maximum_bytes}-byte limit")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(absolute, flags)
    except OSError as error:
        raise AuthorizedBatchReleaseError(f"{label} cannot be opened") from error

    digest = hashlib.sha256()
    chunks: list[bytes] = []
    byte_count = 0
    line_count: int | None = 0 if line_validator is not None else None
    pending = b""
    try:
        opened = os.fstat(descriptor)
        if _identity(opened) != _identity(before_path):
            raise AuthorizedBatchReleaseError(f"{label} changed while being opened")
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            byte_count += len(chunk)
            if byte_count > maximum_bytes:
                raise AuthorizedBatchReleaseError(f"{label} exceeds its {maximum_bytes}-byte limit")
            digest.update(chunk)
            if capture:
                chunks.append(chunk)
            if line_validator is not None:
                pending += chunk
                while b"\n" in pending:
                    raw_line, pending = pending.split(b"\n", 1)
                    assert line_count is not None
                    line_count += 1
                    if len(raw_line) > _MAX_JSONL_LINE_BYTES:
                        raise AuthorizedBatchReleaseError(f"{label} contains an oversized line")
                    if not raw_line:
                        raise AuthorizedBatchReleaseError(f"{label} contains an empty JSONL line")
                    line_validator(
                        _json_object(raw_line, label=f"{label} line {line_count}"),
                        line_count,
                    )
                if len(pending) > _MAX_JSONL_LINE_BYTES:
                    raise AuthorizedBatchReleaseError(f"{label} contains an oversized line")
        if line_validator is not None and pending:
            raise AuthorizedBatchReleaseError(f"{label} must end every JSONL record with newline")
        after_handle = os.fstat(descriptor)
    except OSError as error:
        raise AuthorizedBatchReleaseError(f"{label} cannot be read") from error
    finally:
        os.close(descriptor)
    after_path = _assert_regular_file(absolute, label=label)
    expected = _identity(before_path)
    if _identity(opened) != expected or _identity(after_handle) != expected:
        raise AuthorizedBatchReleaseError(f"{label} changed while being read")
    if _identity(after_path) != expected or byte_count != before_path.st_size:
        raise AuthorizedBatchReleaseError(f"{label} changed while being read")
    return _FileMaterial(
        sha256=digest.hexdigest(),
        byte_count=byte_count,
        payload=b"".join(chunks) if capture else None,
        line_count=line_count,
    )


def _control(path: Path, *, label: str, maximum_bytes: int = _MAX_CONTROL_BYTES) -> _FileMaterial:
    return _inspect_file(
        path,
        label=label,
        maximum_bytes=maximum_bytes,
        capture=True,
    )


def _captured_payload(material: _FileMaterial, *, label: str) -> bytes:
    if material.payload is None:
        raise AuthorizedBatchReleaseError(f"{label} was not captured")
    return material.payload


def _load_registry(
    raw: bytes,
    *,
    policy: AwinFeedPolicy,
    verified: VerifiedSignedPolicy,
    verification_time: datetime,
) -> tuple[dict[str, Any], str, date, str, date]:
    try:
        payload = _load_strict_yaml(raw.decode("utf-8"))
    except UnicodeDecodeError as error:
        raise AuthorizedBatchReleaseError("source registry is not UTF-8") from error
    except RuntimeError as error:
        raise AuthorizedBatchReleaseError("source registry is invalid") from error
    if not isinstance(payload, dict):
        raise AuthorizedBatchReleaseError("source registry root must be an object")
    _exact_fields(payload, _REGISTRY_ROOT_FIELDS, label="source registry root")
    if payload.get("schema_version") != SOURCE_REGISTRY_SCHEMA_VERSION:
        raise AuthorizedBatchReleaseError("source registry schema is unsupported")
    for field_name in (
        "sources",
        "source_templates",
        "auxiliary_sources",
        "blocked_or_restricted_sources",
    ):
        if not isinstance(payload.get(field_name), dict):
            raise AuthorizedBatchReleaseError(f"source registry {field_name} must be an object")
    verified_on = _calendar_date(payload.get("verified_on"), label="source registry verified_on")
    if verified_on > verification_time.date():
        raise AuthorizedBatchReleaseError("source registry verification date is in the future")
    registry_age = (verification_time.date() - verified_on).days
    if registry_age > _MAX_REGISTRY_AGE_DAYS:
        raise AuthorizedBatchReleaseError(
            f"source registry is older than {_MAX_REGISTRY_AGE_DAYS} days"
        )
    templates = payload.get("source_templates")
    assert isinstance(templates, dict)
    template = templates.get(AWIN_SOURCE_TEMPLATE)
    if not isinstance(template, dict):
        raise AuthorizedBatchReleaseError("signed Awin source template is absent")
    _exact_fields(template, _AWIN_TEMPLATE_FIELDS, label="signed Awin source template")
    expected_scalars = {
        "kind": "authorized_retailer_product_feed",
        "parser_version": AWIN_PARSER_VERSION,
        "access": "controlled_local_import_only",
        "scheduled_fetch": False,
        "source_url": "awin://advertisers/supplied-per-policy/feeds/supplied-per-policy",
        "version_policy": "content_sha256_and_signed_policy",
        "licence": "supplied-per-policy",
        "attribution_required": "supplied-per-policy",
        # The registry template is not itself a grant.  Per-feed production
        # authority must come from the independently signed policy.
        "production_catalog_eligible": False,
        "training_eligible": False,
        "published_claims_eligible": False,
        "redistribution_eligible": False,
    }
    if any(template.get(key) != value for key, value in expected_scalars.items()):
        raise AuthorizedBatchReleaseError("signed Awin source template is not fail closed")
    requirements = template.get("requirements")
    if not isinstance(requirements, list) or any(
        not isinstance(value, str) for value in requirements
    ):
        raise AuthorizedBatchReleaseError("signed Awin source requirements are invalid")
    if len(requirements) != len(set(requirements)):
        raise AuthorizedBatchReleaseError("signed Awin source requirements contain duplicates")
    if frozenset(requirements) != _REQUIRED_TEMPLATE_REQUIREMENTS:
        raise AuthorizedBatchReleaseError("signed Awin source requirements are incomplete")
    rights = template.get("data_use_rights")
    if not isinstance(rights, dict) or rights != _AWIN_TEMPLATE_RIGHTS:
        raise AuthorizedBatchReleaseError("signed Awin template rights are invalid")
    access_note = template.get("access_note")
    if not isinstance(access_note, str) or not access_note.strip():
        raise AuthorizedBatchReleaseError("signed Awin source access note is required")

    blocked = payload.get("blocked_or_restricted_sources")
    assert isinstance(blocked, dict)
    if policy.source_name in blocked or policy.source_uri in blocked:
        raise AuthorizedBatchReleaseError("signed Awin source is explicitly blocked or restricted")
    sources = payload.get("sources")
    assert isinstance(sources, dict)
    admission = sources.get(policy.source_name)
    if not isinstance(admission, dict):
        raise AuthorizedBatchReleaseError("signed Awin feed has no explicit source admission")
    _exact_fields(admission, _AWIN_ADMISSION_FIELDS, label="signed Awin source admission")
    if admission.get("status") != "active":
        raise AuthorizedBatchReleaseError("signed Awin source admission is revoked or inactive")
    expected_admission = {
        "kind": "authorized_retailer_product_feed",
        "template": AWIN_SOURCE_TEMPLATE,
        "source_url": policy.source_uri,
        "advertiser_id": policy.advertiser_id,
        "feed_id": policy.feed_id,
        "retailer": policy.retailer,
        "parser_version": AWIN_PARSER_VERSION,
        "policy_id": verified.policy_id,
        "policy_sha256": verified.policy_sha256,
        "production_catalog_eligible": True,
        "revoked_on": None,
        "revocation_reason": None,
    }
    if any(admission.get(key) != value for key, value in expected_admission.items()):
        raise AuthorizedBatchReleaseError(
            "signed Awin source admission does not bind the exact feed and policy"
        )
    admitted_on = _calendar_date(
        admission.get("admitted_on"),
        label="signed Awin source admitted_on",
    )
    admission_expires_on = _calendar_date(
        admission.get("admission_expires_on"),
        label="signed Awin source admission_expires_on",
    )
    if admitted_on > verified_on or admitted_on > verification_time.date():
        raise AuthorizedBatchReleaseError("signed Awin source admission starts in the future")
    if admission_expires_on < admitted_on:
        raise AuthorizedBatchReleaseError("signed Awin source admission dates are inconsistent")
    if verification_time.date() > admission_expires_on:
        raise AuthorizedBatchReleaseError("signed Awin source admission has expired")
    admission_note = admission.get("access_note")
    if not isinstance(admission_note, str) or not admission_note.strip():
        raise AuthorizedBatchReleaseError("signed Awin source admission note is required")
    return (
        template,
        hashlib.sha256(_canonical_json(template)).hexdigest(),
        verified_on,
        hashlib.sha256(_canonical_json(admission)).hexdigest(),
        admission_expires_on,
    )


def _validate_raw_metadata(
    payload: dict[str, Any],
    *,
    raw: _FileMaterial,
    policy: AwinFeedPolicy,
    verified: VerifiedSignedPolicy,
    verification_time: datetime,
    raw_filename: str,
) -> datetime:
    _exact_fields(payload, _RAW_METADATA_FIELDS, label="raw snapshot metadata")
    if payload.get("schema_version") != RAW_SNAPSHOT_SCHEMA_VERSION:
        raise AuthorizedBatchReleaseError("raw snapshot metadata schema is unsupported")
    if payload.get("source_name") != policy.source_name:
        raise AuthorizedBatchReleaseError("raw snapshot source name does not match policy")
    if payload.get("source_url") != policy.source_uri:
        raise AuthorizedBatchReleaseError("raw snapshot source URI does not match policy")
    if payload.get("source_type") != "authorized_retailer_feed":
        raise AuthorizedBatchReleaseError("raw snapshot type is not an authorized retailer feed")
    if payload.get("parser_version") != AWIN_PARSER_VERSION:
        raise AuthorizedBatchReleaseError("raw snapshot parser version is unsupported")
    expected_media_type = "application/gzip" if policy.compression == "gzip" else "text/csv"
    if payload.get("media_type") != expected_media_type:
        raise AuthorizedBatchReleaseError("raw snapshot media type does not match signed policy")
    if payload.get("licence_or_access_note") != policy.licence_or_access_note:
        raise AuthorizedBatchReleaseError("raw snapshot access note does not match signed policy")
    if payload.get("content_sha256") != raw.sha256 or payload.get("byte_count") != raw.byte_count:
        raise AuthorizedBatchReleaseError("raw snapshot metadata does not bind the raw bytes")
    if payload.get("raw_file") != raw_filename:
        raise AuthorizedBatchReleaseError("raw snapshot metadata names a different raw file")
    retrieved_at = _aware_datetime(payload.get("retrieved_at"), label="raw retrieved_at")
    if retrieved_at > verification_time:
        raise AuthorizedBatchReleaseError("raw snapshot retrieval time is in the future")
    if not verified.issued_at <= retrieved_at < verified.expires_at:
        raise AuthorizedBatchReleaseError("raw snapshot was acquired outside policy validity")
    if policy.expected_input_sha256 is not None and policy.expected_input_sha256 != raw.sha256:
        raise AuthorizedBatchReleaseError("raw snapshot does not match signed input hash")
    return retrieved_at


def _validate_authorization_receipt(
    payload: dict[str, Any],
    *,
    policy: AwinFeedPolicy,
    verified: VerifiedSignedPolicy,
    raw: _FileMaterial,
) -> None:
    _exact_fields(payload, _AUTHORIZATION_RECEIPT_FIELDS, label="authorization receipt")
    content_sha = _sha256(payload.get("content_sha256"), label="authorization content_sha256")
    semantic = dict(payload)
    semantic.pop("content_sha256")
    if not hmac.compare_digest(content_sha, hashlib.sha256(_canonical_json(semantic)).hexdigest()):
        raise AuthorizedBatchReleaseError("authorization receipt content hash mismatch")
    expected: dict[str, Any] = {
        "schema_version": AWIN_AUTHORIZATION_RECEIPT_SCHEMA_VERSION,
        "source_name": policy.source_name,
        "source_uri": policy.source_uri,
        "raw_snapshot_sha256": raw.sha256,
        "raw_byte_count": raw.byte_count,
        "parser_version": AWIN_PARSER_VERSION,
        "policy_id": verified.policy_id,
        "policy_issued_at": verified.issued_at.isoformat(),
        "policy_expires_at": verified.expires_at.isoformat(),
        "policy_sha256": verified.policy_sha256,
        "signature_sha256": verified.signature_sha256,
        "trust_root_sha256": verified.trust_root_sha256,
        "signer_key_id": verified.key_id,
        "data_use_rights": policy.rights.to_dict(),
        "production_catalog_eligible": True,
        "training_eligible": policy.training_eligible,
        "published_claims_eligible": policy.published_claims_eligible,
        "published_claims_grant_reference": policy.published_claims_grant_reference,
    }
    if semantic != expected:
        raise AuthorizedBatchReleaseError(
            "authorization receipt is not the independently verified production authority"
        )


def _strict_count_mapping(value: object, *, label: str) -> dict[str, int]:
    if not isinstance(value, dict):
        raise AuthorizedBatchReleaseError(f"{label} must be an object")
    result: dict[str, int] = {}
    for key, count in value.items():
        if not isinstance(key, str) or not key:
            raise AuthorizedBatchReleaseError(f"{label} keys must be non-empty strings")
        if type(count) is not int or count < 0:
            raise AuthorizedBatchReleaseError(f"{label} counts must be non-negative integers")
        result[key] = count
    return result


def _validate_quality(
    payload: dict[str, Any],
    *,
    policy: AwinFeedPolicy,
    raw_sha256: str,
    accepted_count: int,
    rejected_count: int,
) -> None:
    _exact_fields(payload, _QUALITY_FIELDS, label="data-quality report")
    if payload.get("schema_version") != DATA_QUALITY_SCHEMA_VERSION:
        raise AuthorizedBatchReleaseError("data-quality schema is unsupported")
    expected = {
        "source_name": policy.source_name,
        "snapshot_sha256": raw_sha256,
        "status": "pass",
        "accepted_count": accepted_count,
        "rejected_count": rejected_count,
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise AuthorizedBatchReleaseError("data-quality report is not a passing batch report")
    checks = payload.get("checks")
    if not isinstance(checks, list):
        raise AuthorizedBatchReleaseError("data-quality checks must be an array")
    by_name: dict[str, dict[str, Any]] = {}
    errors = 0
    warnings = 0
    for index, item in enumerate(checks):
        label = f"data-quality check {index}"
        if not isinstance(item, dict):
            raise AuthorizedBatchReleaseError(f"{label} must be an object")
        _exact_fields(item, _QUALITY_CHECK_FIELDS, label=label)
        name = item.get("name")
        severity = item.get("severity")
        count = item.get("count")
        message = item.get("message")
        if not isinstance(name, str) or not name.strip():
            raise AuthorizedBatchReleaseError(f"{label} name is invalid")
        if name in by_name:
            raise AuthorizedBatchReleaseError(f"data-quality check name is duplicated: {name!r}")
        if severity not in {"error", "warning"}:
            raise AuthorizedBatchReleaseError(f"{label} severity is invalid")
        if type(count) is not int or count < 0:
            raise AuthorizedBatchReleaseError(f"{label} count must be a non-negative integer")
        if not isinstance(message, str) or not message.strip():
            raise AuthorizedBatchReleaseError(f"{label} message is invalid")
        by_name[name] = item
        if severity == "error":
            errors += count
        else:
            warnings += count
    derived_status = "fail" if errors else "warning" if warnings else "pass"
    if payload.get("status") != derived_status:
        raise AuthorizedBatchReleaseError("data-quality status does not match its checks")
    for required in ("signed_policy_verified", "production_catalog_rights"):
        check = by_name.get(required)
        if not isinstance(check, dict) or type(check.get("count")) is not int:
            raise AuthorizedBatchReleaseError(
                f"data-quality report has no strict required check {required!r}"
            )
        if check["count"] != 0:
            raise AuthorizedBatchReleaseError(
                f"data-quality report did not pass required check {required!r}"
            )

    total_count = accepted_count + rejected_count
    rejection_rate = payload.get("rejection_rate")
    expected_rate = rejected_count / total_count if total_count else 0.0
    if (
        isinstance(rejection_rate, bool)
        or not isinstance(rejection_rate, int | float)
        or not math.isfinite(float(rejection_rate))
        or not math.isclose(float(rejection_rate), expected_rate, abs_tol=1e-12)
    ):
        raise AuthorizedBatchReleaseError("data-quality rejection rate is inconsistent")

    record_type_counts = _strict_count_mapping(
        payload.get("record_type_counts"),
        label="data-quality record_type_counts",
    )
    if record_type_counts != {"retailer_listing": accepted_count}:
        raise AuthorizedBatchReleaseError("data-quality record-type counts are inconsistent")
    category_counts = _strict_count_mapping(
        payload.get("category_counts"),
        label="data-quality category_counts",
    )
    allowed_categories = {category.value for category in policy.category_mappings.values()}
    if set(category_counts) - allowed_categories or sum(category_counts.values()) != accepted_count:
        raise AuthorizedBatchReleaseError("data-quality category counts are inconsistent")
    eligibility_counts = _strict_count_mapping(
        payload.get("eligibility_counts"),
        label="data-quality eligibility_counts",
    )
    expected_eligibility = {
        "training_eligible": accepted_count if policy.training_eligible else 0,
        "training_ineligible": 0 if policy.training_eligible else accepted_count,
        "published_claims_eligible": (accepted_count if policy.published_claims_eligible else 0),
        "published_claims_ineligible": (0 if policy.published_claims_eligible else accepted_count),
    }
    if eligibility_counts != expected_eligibility:
        raise AuthorizedBatchReleaseError("data-quality eligibility counts are inconsistent")

    statistics = payload.get("source_statistics")
    if not isinstance(statistics, dict):
        raise AuthorizedBatchReleaseError("data-quality source statistics must be an object")
    _exact_fields(statistics, _QUALITY_STATISTICS_FIELDS, label="data-quality source statistics")
    expected_statistics = {
        "advertiser_id": policy.advertiser_id,
        "feed_id": policy.feed_id,
        "scanned_records": total_count,
        "accepted_records": accepted_count,
        "rejected_records": rejected_count,
        "unique_source_listing_ids": accepted_count,
        "production_catalog_eligible": True,
        "training_eligible": policy.training_eligible,
        "published_claims_eligible": policy.published_claims_eligible,
        "published_claims_grant_reference": policy.published_claims_grant_reference,
        "data_use_rights": policy.rights.to_dict(),
        "category_counts": category_counts,
    }
    if any(statistics.get(key) != value for key, value in expected_statistics.items()):
        raise AuthorizedBatchReleaseError("data-quality source statistics are inconsistent")
    decompressed_bytes = statistics.get("decompressed_bytes")
    if type(decompressed_bytes) is not int or decompressed_bytes < 1:
        raise AuthorizedBatchReleaseError("data-quality decompressed byte count is invalid")
    rejection_reasons = _strict_count_mapping(
        statistics.get("rejection_reasons"),
        label="data-quality rejection_reasons",
    )
    if sum(rejection_reasons.values()) != rejected_count:
        raise AuthorizedBatchReleaseError("data-quality rejection reasons are inconsistent")


def _checks_by_name(payload: dict[str, Any], *, label: str) -> dict[str, dict[str, Any]]:
    checks = payload.get("checks")
    if not isinstance(checks, list):
        raise AuthorizedBatchReleaseError(f"{label} checks must be an array")
    result: dict[str, dict[str, Any]] = {}
    for item in checks:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            raise AuthorizedBatchReleaseError(f"{label} contains an invalid check")
        name = str(item["name"])
        if name in result:
            raise AuthorizedBatchReleaseError(f"{label} contains duplicate checks")
        result[name] = item
    return result


def _reparse_and_compare(
    *,
    artifacts: AuthorizedBatchReleaseArtifacts,
    policy: AwinFeedPolicy,
    verified: VerifiedSignedPolicy,
    raw: _FileMaterial,
    raw_metadata_payload: dict[str, Any],
    retrieved_at: datetime,
    records: _FileMaterial,
    rejections: _FileMaterial,
    quality: _FileMaterial,
    quality_payload: dict[str, Any],
) -> None:
    """Replay the signed parser and bind released outputs to the raw bytes.

    Regression checks can depend on a prior independently published batch, so
    replay compares every raw-derived quality field and all intrinsic checks.
    Any additional checks are limited to the known regression-check vocabulary.
    """

    replay_raw = RawSnapshot(
        source_name=policy.source_name,
        source_url=policy.source_uri,
        source_type="authorized_retailer_feed",
        retrieved_at=retrieved_at,
        content_sha256=raw.sha256,
        byte_count=raw.byte_count,
        media_type=str(raw_metadata_payload["media_type"]),
        parser_version=AWIN_PARSER_VERSION,
        licence_or_access_note=policy.licence_or_access_note,
        path=artifacts.raw_snapshot,
        metadata_path=artifacts.raw_metadata,
    )
    snapshot = AwinFeedSnapshot(
        raw=replay_raw,
        policy=policy,
        verified_policy=verified,
        authorization_receipt_path=artifacts.authorization_receipt,
    )
    adapter = AwinLocalFeedAdapter(raw_root=artifacts.raw_snapshot.parent, verified_policy=verified)
    try:
        with tempfile.TemporaryDirectory(prefix="pcbr-awin-release-replay-") as temporary_root:
            replayed = adapter.materialize(
                snapshot,
                processed_root=Path(temporary_root) / "processed",
            )
            replayed_records = _inspect_file(
                replayed.records_jsonl,
                label="replayed accepted records",
                maximum_bytes=policy.limits.maximum_output_bytes,
                capture=False,
                line_validator=lambda _record, _line: None,
            )
            replayed_rejections = _inspect_file(
                replayed.rejections_jsonl,
                label="replayed rejections",
                maximum_bytes=policy.limits.maximum_output_bytes,
                capture=False,
                line_validator=lambda _record, _line: None,
            )
            replayed_quality = _control(replayed.quality_json, label="replayed data-quality")
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise AuthorizedBatchReleaseError(
            "raw snapshot cannot reproduce released outputs"
        ) from error

    if (
        replayed_records.reference() != records.reference()
        or replayed_records.line_count != records.line_count
    ):
        raise AuthorizedBatchReleaseError("accepted records do not reproduce from raw snapshot")
    if (
        replayed_rejections.reference() != rejections.reference()
        or replayed_rejections.line_count != rejections.line_count
    ):
        raise AuthorizedBatchReleaseError("rejections do not reproduce from raw snapshot")

    replayed_quality_payload = _json_object(
        _captured_payload(replayed_quality, label="replayed data-quality"),
        label="replayed data-quality",
    )
    for field_name in _QUALITY_FIELDS - {"checks"}:
        if replayed_quality_payload.get(field_name) != quality_payload.get(field_name):
            raise AuthorizedBatchReleaseError(
                f"data-quality field {field_name!r} does not reproduce from raw snapshot"
            )
    replayed_checks = _checks_by_name(replayed_quality_payload, label="replayed data-quality")
    released_checks = _checks_by_name(quality_payload, label="released data-quality")
    if set(replayed_checks) != _AWIN_REPLAY_BASE_CHECKS:
        raise AuthorizedBatchReleaseError("replayed data-quality intrinsic checks are incomplete")
    if any(released_checks.get(name) != check for name, check in replayed_checks.items()):
        raise AuthorizedBatchReleaseError(
            "data-quality intrinsic checks do not reproduce from raw snapshot"
        )
    additional_checks = set(released_checks) - set(replayed_checks)
    if not additional_checks <= _AWIN_REPLAY_REGRESSION_CHECKS:
        raise AuthorizedBatchReleaseError("data-quality contains an unsupported external check")
    if not additional_checks and replayed_quality.reference() != quality.reference():
        raise AuthorizedBatchReleaseError("data-quality bytes do not reproduce from raw snapshot")


def _record_validator(
    *,
    policy: AwinFeedPolicy,
    verified: VerifiedSignedPolicy,
    raw_sha256: str,
    retrieved_at: datetime,
) -> Callable[[dict[str, Any], int], None]:
    expected_authority = {
        "policy_id": verified.policy_id,
        "policy_issued_at": verified.issued_at.isoformat(),
        "policy_expires_at": verified.expires_at.isoformat(),
        "policy_sha256": verified.policy_sha256,
        "signature_sha256": verified.signature_sha256,
        "trust_root_sha256": verified.trust_root_sha256,
        "signer_key_id": verified.key_id,
        "published_claims_grant_reference": policy.published_claims_grant_reference,
    }

    def validate(record: dict[str, Any], line_number: int) -> None:
        prefix = f"accepted record line {line_number}"
        _exact_fields(record, _ACCEPTED_RECORD_FIELDS, label=prefix)
        expected_values = {
            "schema_version": NORMALISED_RECORD_SCHEMA_VERSION,
            "record_type": "retailer_listing",
            "archive_snapshot_sha256": raw_sha256,
            "development_only": False,
            "training_eligible": policy.training_eligible,
            "published_claims_eligible": policy.published_claims_eligible,
            "data_use_rights": policy.rights.to_dict(),
            "rights_authority": expected_authority,
        }
        if any(record.get(key) != value for key, value in expected_values.items()):
            raise AuthorizedBatchReleaseError(f"{prefix} is not bound to signed authority")
        source_record_id = record.get("source_record_id")
        if (
            not isinstance(source_record_id, str)
            or not source_record_id.startswith(f"{policy.advertiser_id}:")
            or len(source_record_id) > 320
        ):
            raise AuthorizedBatchReleaseError(f"{prefix} has an invalid source record ID")
        raw_product_id = source_record_id.removeprefix(f"{policy.advertiser_id}:")
        if not raw_product_id or len(raw_product_id) > 256:
            raise AuthorizedBatchReleaseError(f"{prefix} has an invalid source product ID")
        _sha256(record.get("raw_record_sha256"), label=f"{prefix} raw_record_sha256")

        authority = record.get("rights_authority")
        if not isinstance(authority, dict):
            raise AuthorizedBatchReleaseError(f"{prefix} has no signed rights authority")
        _exact_fields(authority, _RECORD_AUTHORITY_FIELDS, label=f"{prefix} authority")
        if authority != expected_authority:
            raise AuthorizedBatchReleaseError(f"{prefix} is not bound to signed authority")

        provenance = record.get("provenance")
        if not isinstance(provenance, dict):
            raise AuthorizedBatchReleaseError(f"{prefix} has inconsistent provenance")
        _exact_fields(provenance, _RECORD_PROVENANCE_FIELDS, label=f"{prefix} provenance")
        expected_provenance = {
            "source_name": policy.source_name,
            "source_url": policy.source_uri,
            "source_type": "authorized_retailer_feed",
            "retrieved_at": retrieved_at.isoformat(),
            "parser_version": AWIN_PARSER_VERSION,
            "licence_or_access_note": policy.licence_or_access_note,
            "extraction_confidence": 1.0,
        }
        if (
            provenance != expected_provenance
            or type(provenance["extraction_confidence"]) is not float
        ):
            raise AuthorizedBatchReleaseError(f"{prefix} has inconsistent provenance")

        metadata = record.get("normalisation_metadata")
        if not isinstance(metadata, dict):
            raise AuthorizedBatchReleaseError(f"{prefix} has no normalisation metadata")
        _exact_fields(
            metadata,
            _RECORD_NORMALISATION_FIELDS,
            label=f"{prefix} normalisation metadata",
        )
        expected_metadata = {
            "canonical_mapping_status": "unmatched",
            "advertiser_id": policy.advertiser_id,
            "feed_id": policy.feed_id,
        }
        if any(metadata.get(key) != value for key, value in expected_metadata.items()):
            raise AuthorizedBatchReleaseError(f"{prefix} has inconsistent normalisation metadata")
        row_number = metadata.get("row_number")
        if type(row_number) is not int or row_number < 2:
            raise AuthorizedBatchReleaseError(f"{prefix} has an invalid source row number")
        allowed_categories = {category.value for category in policy.category_mappings.values()}
        if metadata.get("category") not in allowed_categories:
            raise AuthorizedBatchReleaseError(f"{prefix} has an unauthorized category")
        if metadata.get("price_field") not in {"search_price", "price", "store_price"}:
            raise AuthorizedBatchReleaseError(f"{prefix} has an invalid price basis")
        stock_basis = metadata.get("stock_basis")
        if (
            type(stock_basis) is not list
            or len(stock_basis) != len(set(stock_basis))
            or any(
                type(value) is not str
                or value not in {"stock_status", "in_stock", "is_for_sale", "pre_order"}
                for value in stock_basis
            )
        ):
            raise AuthorizedBatchReleaseError(f"{prefix} has invalid stock evidence")
        for field_name, maximum in {
            "brand": 256,
            "manufacturer_part_number": 256,
            "gtin": 64,
            "colour": 128,
            "source_last_updated": 128,
        }.items():
            value = metadata.get(field_name)
            if value is not None and (
                not isinstance(value, str)
                or not value.strip()
                or len(value) > maximum
                or "\n" in value
                or "\r" in value
            ):
                raise AuthorizedBatchReleaseError(f"{prefix} has invalid {field_name} metadata")

        data = record.get("data")
        if not isinstance(data, dict):
            raise AuthorizedBatchReleaseError(f"{prefix} has no retailer listing")
        _exact_fields(data, _RECORD_DATA_FIELDS, label=f"{prefix} data")
        listing_payload = data.get("listing")
        snapshot_payload = data.get("price_snapshot")
        if not isinstance(listing_payload, dict) or not isinstance(snapshot_payload, dict):
            raise AuthorizedBatchReleaseError(f"{prefix} has incomplete listing data")
        _exact_fields(
            listing_payload,
            _RETAILER_LISTING_FIELDS,
            label=f"{prefix} retailer listing",
        )
        _exact_fields(
            snapshot_payload,
            _PRICE_SNAPSHOT_FIELDS,
            label=f"{prefix} price snapshot",
        )
        try:
            listing = RetailerListing.model_validate(listing_payload)
            price_snapshot = PriceSnapshot.model_validate(snapshot_payload)
        except (TypeError, ValueError) as error:
            raise AuthorizedBatchReleaseError(
                f"{prefix} violates the accepted listing schema"
            ) from error
        if (
            listing.model_dump(mode="json") != listing_payload
            or price_snapshot.model_dump(mode="json") != snapshot_payload
        ):
            raise AuthorizedBatchReleaseError(f"{prefix} is not canonically serialized")
        if listing.retailer != policy.retailer or listing.currency != "SGD":
            raise AuthorizedBatchReleaseError(f"{prefix} violates signed retailer constraints")
        if listing.seller_name != policy.retailer or listing.source_listing_id != source_record_id:
            raise AuthorizedBatchReleaseError(f"{prefix} has inconsistent retailer identity")
        if not policy.allow_non_new and listing.condition.value != "new":
            raise AuthorizedBatchReleaseError(f"{prefix} violates the signed condition constraint")
        if (
            listing.base_price <= Decimal("0")
            or listing.base_price > policy.limits.maximum_price_sgd
        ):
            raise AuthorizedBatchReleaseError(f"{prefix} violates signed price constraints")
        try:
            parsed = urlsplit(listing.listing_url)
            port = parsed.port
        except ValueError as error:
            raise AuthorizedBatchReleaseError(f"{prefix} has an invalid listing URL") from error
        host = (parsed.hostname or "").rstrip(".").lower()
        if (
            parsed.scheme != "https"
            or host not in policy.allowed_link_hosts
            or parsed.username is not None
            or parsed.password is not None
            or port not in {None, 443}
        ):
            raise AuthorizedBatchReleaseError(f"{prefix} violates signed link-host constraints")
        expected_listing_id = stable_identifier(
            "listing",
            "awin",
            policy.advertiser_id,
            raw_product_id,
            length=32,
        )
        expected_product_id = stable_identifier(
            "unmatched_product",
            "awin",
            policy.advertiser_id,
            raw_product_id,
        )
        if (
            listing.listing_id != expected_listing_id
            or listing.product_id != expected_product_id
            or listing.first_seen_at != retrieved_at
            or listing.last_seen_at != retrieved_at
        ):
            raise AuthorizedBatchReleaseError(f"{prefix} has inconsistent stable listing fields")
        expected_snapshot_id = stable_identifier(
            "price",
            listing.listing_id,
            retrieved_at.isoformat(),
            listing.base_price,
            listing.shipping_price,
            listing.stock_status.value,
            length=32,
        )
        if (
            price_snapshot.snapshot_id != expected_snapshot_id
            or price_snapshot.listing_id != listing.listing_id
            or price_snapshot.observed_at != retrieved_at
            or price_snapshot.base_price != listing.base_price
            or price_snapshot.shipping_price != listing.shipping_price
            or price_snapshot.stock_status != listing.stock_status
        ):
            raise AuthorizedBatchReleaseError(f"{prefix} has inconsistent price snapshot fields")

    return validate


def _validate_processed_manifest(
    payload: dict[str, Any],
    *,
    policy: AwinFeedPolicy,
    verified: VerifiedSignedPolicy,
    authorization_receipt: _FileMaterial,
    raw: _FileMaterial,
    records: _FileMaterial,
    rejections: _FileMaterial,
    quality: _FileMaterial,
) -> tuple[str, int, int]:
    _exact_fields(payload, _STREAMING_MANIFEST_FIELDS, label="processed manifest")
    content_sha = _sha256(payload.get("content_sha256"), label="processed content_sha256")
    semantic = dict(payload)
    semantic.pop("content_sha256")
    if not hmac.compare_digest(content_sha, hashlib.sha256(_canonical_json(semantic)).hexdigest()):
        raise AuthorizedBatchReleaseError("processed manifest content hash mismatch")
    run_sha = _sha256(payload.get("run_sha256"), label="processed run_sha256")
    accepted = payload.get("accepted_count")
    rejected = payload.get("rejected_count")
    if type(accepted) is not int or accepted < 1 or type(rejected) is not int or rejected < 0:
        raise AuthorizedBatchReleaseError("processed batch counts are invalid")
    if records.line_count != accepted or rejections.line_count != rejected:
        raise AuthorizedBatchReleaseError("processed counts do not match JSONL record counts")
    expected_files = {
        "records.jsonl": records.reference(),
        "rejections.jsonl": rejections.reference(),
        "data-quality.json": quality.reference(),
    }
    metadata = payload.get("metadata")
    if (
        payload.get("schema_version") != STREAMING_MANIFEST_SCHEMA_VERSION
        or payload.get("source_name") != policy.source_name
        or payload.get("files") != expected_files
        or not isinstance(metadata, dict)
    ):
        raise AuthorizedBatchReleaseError("processed manifest does not bind its batch files")
    expected_metadata = {
        "raw_snapshot_sha256": raw.sha256,
        "authorization_receipt_sha256": authorization_receipt.sha256,
        "parser_version": AWIN_PARSER_VERSION,
        "policy_id": verified.policy_id,
        "policy_issued_at": verified.issued_at.isoformat(),
        "policy_expires_at": verified.expires_at.isoformat(),
        "policy_sha256": verified.policy_sha256,
        "signature_sha256": verified.signature_sha256,
        "trust_root_sha256": verified.trust_root_sha256,
        "signer_key_id": verified.key_id,
        "source_uri": policy.source_uri,
        "production_catalog_eligible": True,
        "training_eligible": policy.training_eligible,
        "published_claims_eligible": policy.published_claims_eligible,
        "published_claims_grant_reference": policy.published_claims_grant_reference,
        "limits": policy.limits.to_dict(),
    }
    if any(metadata.get(key) != value for key, value in expected_metadata.items()):
        raise AuthorizedBatchReleaseError("processed manifest authority metadata is inconsistent")
    return run_sha, accepted, rejected


def _collect_release(
    *,
    artifacts: AuthorizedBatchReleaseArtifacts,
    authority: AuthorizedBatchAuthorityArtifacts,
    current_source_registry: Path,
    expected_trust_root_sha256: str,
    now: datetime | None,
) -> _CollectedRelease:
    verification_time = _verification_time(now)
    trust_pin = _sha256(expected_trust_root_sha256, label="expected trust-root SHA-256")
    try:
        verified = verify_signed_policy(
            policy_path=authority.policy,
            signature_path=authority.policy_signature,
            trust_root_path=authority.trust_root,
            expected_trust_root_sha256=trust_pin,
            now=verification_time,
        )
        policy = AwinFeedPolicy.from_verified(verified)
        policy.rights.assert_catalog_serving_allowed(
            territory="SG",
            on_date=verification_time.date(),
        )
    except (OSError, PermissionError, TypeError, ValueError) as error:
        raise AuthorizedBatchReleaseError(
            "production source authority must be a valid, active signed policy"
        ) from error
    if not policy.production_catalog_eligible:
        raise AuthorizedBatchReleaseError(
            "signed policy does not grant production catalogue eligibility"
        )

    policy_file = _control(authority.policy, label="signed policy")
    signature_file = _control(authority.policy_signature, label="policy signature")
    trust_root_file = _control(authority.trust_root, label="policy trust root")
    if (
        policy_file.sha256 != verified.policy_sha256
        or signature_file.sha256 != verified.signature_sha256
        or trust_root_file.sha256 != verified.trust_root_sha256
    ):
        raise AuthorizedBatchReleaseError("signed authority changed after verification")
    key_valid_until = _signer_key_valid_until(trust_root_file, key_id=verified.key_id)

    registry_file = _control(
        authority.source_registry,
        label="release source registry",
        maximum_bytes=_MAX_REGISTRY_BYTES,
    )
    current_registry_file = _control(
        current_source_registry,
        label="current source registry",
        maximum_bytes=_MAX_REGISTRY_BYTES,
    )
    if registry_file.sha256 != current_registry_file.sha256:
        raise AuthorizedBatchReleaseError("release source registry is not the current registry")
    assert registry_file.payload is not None
    (
        _template,
        template_sha256,
        registry_verified_on,
        source_admission_sha256,
        source_admission_expires_on,
    ) = _load_registry(
        registry_file.payload,
        policy=policy,
        verified=verified,
        verification_time=verification_time,
    )

    raw = _inspect_file(
        artifacts.raw_snapshot,
        label="raw snapshot",
        maximum_bytes=_MAX_EXTERNAL_BYTES,
        capture=False,
    )
    raw_metadata = _control(artifacts.raw_metadata, label="raw snapshot metadata")
    receipt = _control(artifacts.authorization_receipt, label="authorization receipt")
    processed_manifest = _control(artifacts.processed_manifest, label="processed manifest")
    quality = _control(artifacts.quality_report, label="data-quality report")
    raw_metadata_payload = _json_object(
        _captured_payload(raw_metadata, label="raw snapshot metadata"),
        label="raw snapshot metadata",
    )
    retrieved_at = _validate_raw_metadata(
        raw_metadata_payload,
        raw=raw,
        policy=policy,
        verified=verified,
        verification_time=verification_time,
        raw_filename=artifacts.raw_snapshot.name,
    )
    retention_expires_at = (
        retrieved_at + timedelta(days=policy.rights.retention_days)
        if policy.rights.retention_days is not None
        else None
    )
    if retention_expires_at is not None and verification_time >= retention_expires_at:
        raise AuthorizedBatchReleaseError("raw snapshot exceeds its signed retention period")
    consent_expires_at = _consent_expires_at(policy.rights.consent_expires_on)
    authority_deadlines = [verified.expires_at, key_valid_until]
    if consent_expires_at is not None:
        authority_deadlines.append(consent_expires_at)
    if retention_expires_at is not None:
        authority_deadlines.append(retention_expires_at)
    authority_expires_at = min(authority_deadlines)
    if verification_time >= authority_expires_at:
        raise AuthorizedBatchReleaseError("source authority is no longer active")
    receipt_payload = _json_object(
        _captured_payload(receipt, label="authorization receipt"),
        label="authorization receipt",
    )
    _validate_authorization_receipt(
        receipt_payload,
        policy=policy,
        verified=verified,
        raw=raw,
    )

    records = _inspect_file(
        artifacts.records,
        label="accepted records",
        maximum_bytes=_MAX_EXTERNAL_BYTES,
        capture=False,
        line_validator=_record_validator(
            policy=policy,
            verified=verified,
            raw_sha256=raw.sha256,
            retrieved_at=retrieved_at,
        ),
    )
    rejections = _inspect_file(
        artifacts.rejections,
        label="rejections",
        maximum_bytes=_MAX_EXTERNAL_BYTES,
        capture=False,
        line_validator=lambda _record, _line: None,
    )
    processed_payload = _json_object(
        _captured_payload(processed_manifest, label="processed manifest"),
        label="processed manifest",
    )
    run_sha, accepted, rejected = _validate_processed_manifest(
        processed_payload,
        policy=policy,
        verified=verified,
        authorization_receipt=receipt,
        raw=raw,
        records=records,
        rejections=rejections,
        quality=quality,
    )
    quality_payload = _json_object(
        _captured_payload(quality, label="data-quality report"),
        label="data-quality report",
    )
    _validate_quality(
        quality_payload,
        policy=policy,
        raw_sha256=raw.sha256,
        accepted_count=accepted,
        rejected_count=rejected,
    )
    _reparse_and_compare(
        artifacts=artifacts,
        policy=policy,
        verified=verified,
        raw=raw,
        raw_metadata_payload=raw_metadata_payload,
        retrieved_at=retrieved_at,
        records=records,
        rejections=rejections,
        quality=quality,
        quality_payload=quality_payload,
    )

    bundle_material = {
        "authorization_receipt": receipt,
        "data_quality": quality,
        "policy": policy_file,
        "policy_signature": signature_file,
        "processed_manifest": processed_manifest,
        "raw_metadata": raw_metadata,
        "source_registry": registry_file,
        "trust_root": trust_root_file,
    }
    bundle_files = {
        _BUNDLE_NAMES[key]: material.reference()
        for key, material in sorted(bundle_material.items())
    }
    payload: dict[str, Any] = {
        "schema_version": AUTHORIZED_BATCH_RELEASE_SCHEMA_VERSION,
        "source_name": policy.source_name,
        "source_uri": policy.source_uri,
        "authority": {
            "authority_type": "ed25519_signed_policy",
            "policy_id": verified.policy_id,
            "policy_sha256": verified.policy_sha256,
            "signature_sha256": verified.signature_sha256,
            "trust_root_sha256": verified.trust_root_sha256,
            "signer_key_id": verified.key_id,
            "policy_issued_at": verified.issued_at.isoformat(),
            "policy_expires_at": verified.expires_at.isoformat(),
            "signer_key_valid_until": key_valid_until.isoformat(),
            "consent_effective_on": policy.rights.consent_effective_on.isoformat(),
            "consent_expires_on": (
                policy.rights.consent_expires_on.isoformat()
                if policy.rights.consent_expires_on is not None
                else None
            ),
            "retention_days": policy.rights.retention_days,
            "authority_expires_at": authority_expires_at.isoformat(),
            "production_catalog_eligible": True,
        },
        "source_registry": {
            "schema_version": SOURCE_REGISTRY_SCHEMA_VERSION,
            "sha256": registry_file.sha256,
            "byte_count": registry_file.byte_count,
            "template_name": AWIN_SOURCE_TEMPLATE,
            "template_sha256": template_sha256,
            "verified_on": registry_verified_on.isoformat(),
            "source_admission_sha256": source_admission_sha256,
            "source_admission_expires_on": source_admission_expires_on.isoformat(),
        },
        "raw_snapshot": {
            "sha256": raw.sha256,
            "byte_count": raw.byte_count,
            "metadata_sha256": raw_metadata.sha256,
            "retrieved_at": retrieved_at.isoformat(),
            "retention_expires_at": (
                retention_expires_at.isoformat() if retention_expires_at is not None else None
            ),
            "parser_version": AWIN_PARSER_VERSION,
        },
        "processed_batch": {
            "run_sha256": run_sha,
            "manifest_sha256": processed_manifest.sha256,
            "quality_sha256": quality.sha256,
            "quality_status": "pass",
            "accepted_count": accepted,
            "rejected_count": rejected,
            "parser_version": AWIN_PARSER_VERSION,
            "authorization_receipt_sha256": receipt.sha256,
        },
        "bundle_files": bundle_files,
        "external_files": {
            "raw_snapshot": raw.reference(),
            "records.jsonl": records.reference(),
            "rejections.jsonl": rejections.reference(),
        },
    }
    payload["content_sha256"] = hashlib.sha256(_canonical_json(payload)).hexdigest()
    bundle_bytes = {
        _BUNDLE_NAMES[key]: material.payload
        for key, material in bundle_material.items()
        if material.payload is not None
    }
    return _CollectedRelease(payload=payload, bundle_bytes=bundle_bytes)


def _ensure_directory(path: Path, *, label: str) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    if _is_linklike(path) or not path.is_dir():
        raise AuthorizedBatchReleaseError(f"{label} must be a regular directory")
    return path.resolve(strict=True)


def _assert_existing_bundle(directory: Path, expected: Mapping[str, bytes]) -> None:
    try:
        names = {path.name for path in directory.iterdir()}
    except OSError as error:
        raise AuthorizedBatchReleaseError("existing source release is unreadable") from error
    if names != set(expected):
        raise AuthorizedBatchReleaseError("existing source release contains unexpected files")
    for name, expected_bytes in expected.items():
        material = _control(directory / name, label=f"existing release {name}")
        if material.payload != expected_bytes:
            raise AuthorizedBatchReleaseError("existing source release conflicts with evidence")


def publish_awin_production_batch_release(
    *,
    release_root: str | Path,
    artifacts: AuthorizedBatchReleaseArtifacts,
    authority: AuthorizedBatchAuthorityArtifacts,
    expected_trust_root_sha256: str,
    now: datetime | None = None,
) -> VerifiedAuthorizedBatchRelease:
    """Publish and revalidate one content-addressed signed Awin batch release."""

    collected = _collect_release(
        artifacts=artifacts,
        authority=authority,
        current_source_registry=authority.source_registry,
        expected_trust_root_sha256=expected_trust_root_sha256,
        now=now,
    )
    manifest_bytes = _pretty_json(collected.payload)
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    source_name = str(collected.payload["source_name"])
    if _SOURCE_NAME.fullmatch(source_name) is None:
        raise AuthorizedBatchReleaseError("release source_name is unsafe")
    root = _ensure_directory(Path(release_root), label="source release root")
    source_root = _ensure_directory(root / source_name, label="source release source root")
    final_directory = source_root / manifest_sha256
    expected_files = {**collected.bundle_bytes, "manifest.json": manifest_bytes}
    reused = False
    if final_directory.exists():
        if _is_linklike(final_directory) or not final_directory.is_dir():
            raise AuthorizedBatchReleaseError("existing source release is not a directory")
        _assert_existing_bundle(final_directory, expected_files)
        reused = True
    else:
        staged = Path(tempfile.mkdtemp(prefix=".authorized-batch.", dir=source_root))
        try:
            for name, payload in expected_files.items():
                with (staged / name).open("xb") as handle:
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
            try:
                os.rename(staged, final_directory)
            except OSError as error:
                if not final_directory.exists():
                    raise AuthorizedBatchReleaseError(
                        "atomic source release publication failed"
                    ) from error
                _assert_existing_bundle(final_directory, expected_files)
        finally:
            if staged.exists():
                shutil.rmtree(staged)

    verified = verify_awin_production_batch_release(
        manifest_path=final_directory / "manifest.json",
        expected_manifest_sha256=manifest_sha256,
        expected_trust_root_sha256=expected_trust_root_sha256,
        current_source_registry=authority.source_registry,
        raw_snapshot=artifacts.raw_snapshot,
        records=artifacts.records,
        rejections=artifacts.rejections,
        now=now,
    )
    return VerifiedAuthorizedBatchRelease(
        manifest_path=verified.manifest_path,
        manifest_sha256=verified.manifest_sha256,
        content_sha256=verified.content_sha256,
        source_name=verified.source_name,
        raw_snapshot_sha256=verified.raw_snapshot_sha256,
        processed_run_sha256=verified.processed_run_sha256,
        accepted_count=verified.accepted_count,
        rejected_count=verified.rejected_count,
        policy_sha256=verified.policy_sha256,
        source_registry_sha256=verified.source_registry_sha256,
        authority_expires_at=verified.authority_expires_at,
        reused=reused,
    )


def verify_awin_production_batch_release(
    *,
    manifest_path: str | Path,
    expected_manifest_sha256: str,
    expected_trust_root_sha256: str,
    current_source_registry: str | Path,
    raw_snapshot: str | Path,
    records: str | Path,
    rejections: str | Path,
    now: datetime | None = None,
) -> VerifiedAuthorizedBatchRelease:
    """Independently verify one exact production source-batch release."""

    expected_manifest = _sha256(
        expected_manifest_sha256,
        label="expected release manifest SHA-256",
    )
    path = Path(manifest_path)
    manifest = _control(path, label="authorized batch release manifest")
    if manifest.sha256 != expected_manifest:
        raise AuthorizedBatchReleaseError("release manifest does not match deployment pin")
    if path.name != "manifest.json" or path.parent.name != expected_manifest:
        raise AuthorizedBatchReleaseError("release manifest is not content-addressed")
    assert manifest.payload is not None
    payload = _json_object(manifest.payload, label="authorized batch release manifest")
    _exact_fields(payload, _RELEASE_FIELDS, label="authorized batch release manifest")
    if payload.get("schema_version") != AUTHORIZED_BATCH_RELEASE_SCHEMA_VERSION:
        raise AuthorizedBatchReleaseError("authorized batch release schema is unsupported")
    content_sha = _sha256(payload.get("content_sha256"), label="release content_sha256")
    semantic = dict(payload)
    semantic.pop("content_sha256")
    if not hmac.compare_digest(content_sha, hashlib.sha256(_canonical_json(semantic)).hexdigest()):
        raise AuthorizedBatchReleaseError("release content hash mismatch")

    expected_names = {"manifest.json", *_BUNDLE_NAMES.values()}
    try:
        actual_names = {item.name for item in path.parent.iterdir()}
    except OSError as error:
        raise AuthorizedBatchReleaseError("release bundle is unreadable") from error
    if actual_names != expected_names:
        raise AuthorizedBatchReleaseError("release bundle contains unexpected files")
    bundle = path.parent
    artifacts = AuthorizedBatchReleaseArtifacts(
        raw_snapshot=Path(raw_snapshot),
        raw_metadata=bundle / _BUNDLE_NAMES["raw_metadata"],
        authorization_receipt=bundle / _BUNDLE_NAMES["authorization_receipt"],
        records=Path(records),
        rejections=Path(rejections),
        processed_manifest=bundle / _BUNDLE_NAMES["processed_manifest"],
        quality_report=bundle / _BUNDLE_NAMES["data_quality"],
    )
    authority = AuthorizedBatchAuthorityArtifacts(
        policy=bundle / _BUNDLE_NAMES["policy"],
        policy_signature=bundle / _BUNDLE_NAMES["policy_signature"],
        trust_root=bundle / _BUNDLE_NAMES["trust_root"],
        source_registry=bundle / _BUNDLE_NAMES["source_registry"],
    )
    recomputed = _collect_release(
        artifacts=artifacts,
        authority=authority,
        current_source_registry=Path(current_source_registry),
        expected_trust_root_sha256=expected_trust_root_sha256,
        now=now,
    )
    if recomputed.payload != payload:
        raise AuthorizedBatchReleaseError("release evidence does not reproduce its manifest")
    authority_payload = payload["authority"]
    registry_payload = payload["source_registry"]
    raw_payload = payload["raw_snapshot"]
    batch_payload = payload["processed_batch"]
    assert isinstance(authority_payload, dict)
    assert isinstance(registry_payload, dict)
    assert isinstance(raw_payload, dict)
    assert isinstance(batch_payload, dict)
    return VerifiedAuthorizedBatchRelease(
        manifest_path=path.resolve(),
        manifest_sha256=manifest.sha256,
        content_sha256=content_sha,
        source_name=str(payload["source_name"]),
        raw_snapshot_sha256=str(raw_payload["sha256"]),
        processed_run_sha256=str(batch_payload["run_sha256"]),
        accepted_count=int(batch_payload["accepted_count"]),
        rejected_count=int(batch_payload["rejected_count"]),
        policy_sha256=str(authority_payload["policy_sha256"]),
        source_registry_sha256=str(registry_payload["sha256"]),
        authority_expires_at=_aware_datetime(
            authority_payload.get("authority_expires_at"),
            label="release authority_expires_at",
        ),
    )


__all__ = [
    "AUTHORIZED_BATCH_RELEASE_SCHEMA_VERSION",
    "AWIN_SOURCE_TEMPLATE",
    "AuthorizedBatchAuthorityArtifacts",
    "AuthorizedBatchReleaseArtifacts",
    "AuthorizedBatchReleaseError",
    "VerifiedAuthorizedBatchRelease",
    "publish_awin_production_batch_release",
    "verify_awin_production_batch_release",
]
