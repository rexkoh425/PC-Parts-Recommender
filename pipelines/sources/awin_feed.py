"""Rights-gated, local-file-only Awin product-feed ingestion.

The adapter intentionally has no network or credential input. Awin feed download
URLs contain API keys, so provenance uses a non-secret ``awin://`` identity and
only an operator-supplied local file is accepted.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import math
import os
import re
import sqlite3
import stat
import tempfile
from collections import Counter
from collections.abc import Iterator, Mapping
from contextlib import closing
from dataclasses import dataclass, field
from datetime import UTC
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from pathlib import Path
from typing import Any, BinaryIO, Final, Protocol
from urllib.parse import parse_qsl, unquote, urlsplit

from pc_build_recommender.data_rights import DataUse, DataUseRights
from pc_build_recommender.domain.enums import (
    ComponentKind,
    ListingCondition,
    StockStatus,
)
from pc_build_recommender.domain.models import PriceSample, RetailerListing
from pipelines.checks.quality import (
    DATA_QUALITY_SCHEMA_VERSION,
    DataQualityReport,
    QualityBaseline,
    QualityRegressionPolicy,
    load_previous_quality_baseline,
    quality_regression_checks,
)
from pipelines.parsing.normalizers import NORMALISED_RECORD_SCHEMA_VERSION, stable_identifier
from pipelines.parsing.streaming_writer import (
    AtomicStreamingJSONLWriter,
    StreamingProcessedArtifacts,
    load_existing_streaming_artifacts,
)
from pipelines.sources.base import RawSnapshot, sha256_bytes, snapshot_local_file
from pipelines.sources.signed_policy import VerifiedSignedPolicy

AWIN_POLICY_PAYLOAD_SCHEMA_VERSION: Final = "pc-build-recommender.awin-feed-policy.v1"
AWIN_PARSER_VERSION: Final = "awin-local-csv-stream-v1"
AWIN_AUTHORIZATION_RECEIPT_SCHEMA_VERSION: Final = (
    "pc-build-recommender.awin-feed-authorization-receipt.v1"
)

_AWIN_ID = re.compile(r"^[0-9]{1,20}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CREDENTIAL_QUERY_KEYS = {
    "access_token",
    "api_key",
    "apikey",
    "authorization",
    "key",
    "token",
}
_CREDENTIAL_TEXT = re.compile(
    r"(?i)(?:/apikey/|%2fapikey%2f|(?:[?&]|&amp;|%3f|%26)"
    r"(?:access_token|api_key|apikey|authorization|key|token)(?:=|%3d))"
)
_CREDENTIAL_BYTES = re.compile(
    rb"(?i)(?:/apikey/|%2fapikey%2f|(?:[?&]|&amp;|%3f|%26)"
    rb"(?:access_token|api_key|apikey|authorization|key|token)(?:=|%3d))"
)
_MONEY = re.compile(
    r"^((?:[0-9]+|[0-9]{1,3}(?:,[0-9]{3})+)(?:\.[0-9]{1,4})?)"
    r"(?:\s+([A-Za-z]{3}))?$"
)

_DEFAULT_LIMITS: Final[dict[str, int | float]] = {
    "maximum_input_bytes": 512 * 1024 * 1024,
    "maximum_decompressed_bytes": 2 * 1024 * 1024 * 1024,
    "maximum_records": 250_000,
    "maximum_rejections": 250_000,
    "maximum_field_characters": 65_536,
    "maximum_columns": 256,
    "maximum_output_bytes": 2 * 1024 * 1024 * 1024,
    "maximum_record_bytes": 2 * 1024 * 1024,
    "maximum_price_sgd": 250_000,
    "maximum_rejection_rate": 0.25,
}
_HARD_LIMITS: Final[dict[str, int | float]] = {
    "maximum_input_bytes": 2 * 1024 * 1024 * 1024,
    "maximum_decompressed_bytes": 8 * 1024 * 1024 * 1024,
    "maximum_records": 1_000_000,
    "maximum_rejections": 1_000_000,
    "maximum_field_characters": 262_144,
    "maximum_columns": 512,
    "maximum_output_bytes": 8 * 1024 * 1024 * 1024,
    "maximum_record_bytes": 8 * 1024 * 1024,
    "maximum_price_sgd": 1_000_000,
    "maximum_rejection_rate": 1.0,
}


class AwinFeedError(RuntimeError):
    """Raised when an Awin feed or its authority fails closed."""


class AwinFeedLimitError(AwinFeedError):
    """Raised before a feed can exceed its signed and hard resource limits."""


class _BinaryReadable(Protocol):
    def read(self, size: int = -1, /) -> bytes: ...


@dataclass(frozen=True, slots=True)
class AwinFeedLimits:
    maximum_input_bytes: int
    maximum_decompressed_bytes: int
    maximum_records: int
    maximum_rejections: int
    maximum_field_characters: int
    maximum_columns: int
    maximum_output_bytes: int
    maximum_record_bytes: int
    maximum_price_sgd: Decimal
    maximum_rejection_rate: float

    @classmethod
    def from_mapping(cls, payload: object) -> AwinFeedLimits:
        if payload is None:
            values = dict(_DEFAULT_LIMITS)
        else:
            if not isinstance(payload, Mapping):
                raise TypeError("limits must be an object")
            unknown = sorted(set(payload) - set(_DEFAULT_LIMITS))
            if unknown:
                raise ValueError(f"limits contain unknown fields: {unknown}")
            values = {**_DEFAULT_LIMITS, **dict(payload)}
        for name in (
            "maximum_input_bytes",
            "maximum_decompressed_bytes",
            "maximum_records",
            "maximum_rejections",
            "maximum_field_characters",
            "maximum_columns",
            "maximum_output_bytes",
            "maximum_record_bytes",
        ):
            value = values[name]
            if type(value) is not int or value < 1 or value > int(_HARD_LIMITS[name]):
                raise ValueError(f"{name} must be a positive integer within its hard ceiling")
        raw_price = values["maximum_price_sgd"]
        if isinstance(raw_price, bool) or not isinstance(raw_price, int | float):
            raise TypeError("maximum_price_sgd must be a number")
        if not math.isfinite(float(raw_price)) or not 0 < float(raw_price) <= float(
            _HARD_LIMITS["maximum_price_sgd"]
        ):
            raise ValueError("maximum_price_sgd is outside its hard ceiling")
        raw_rate = values["maximum_rejection_rate"]
        if isinstance(raw_rate, bool) or not isinstance(raw_rate, int | float):
            raise TypeError("maximum_rejection_rate must be a number")
        rate = float(raw_rate)
        if not math.isfinite(rate) or not 0 <= rate <= 1:
            raise ValueError("maximum_rejection_rate must be between zero and one")
        if values["maximum_record_bytes"] > values["maximum_output_bytes"]:
            raise ValueError("maximum_record_bytes cannot exceed maximum_output_bytes")
        return cls(
            maximum_input_bytes=int(values["maximum_input_bytes"]),
            maximum_decompressed_bytes=int(values["maximum_decompressed_bytes"]),
            maximum_records=int(values["maximum_records"]),
            maximum_rejections=int(values["maximum_rejections"]),
            maximum_field_characters=int(values["maximum_field_characters"]),
            maximum_columns=int(values["maximum_columns"]),
            maximum_output_bytes=int(values["maximum_output_bytes"]),
            maximum_record_bytes=int(values["maximum_record_bytes"]),
            maximum_price_sgd=Decimal(str(raw_price)),
            maximum_rejection_rate=rate,
        )

    def to_dict(self) -> dict[str, int | float | str]:
        return {
            "maximum_input_bytes": self.maximum_input_bytes,
            "maximum_decompressed_bytes": self.maximum_decompressed_bytes,
            "maximum_records": self.maximum_records,
            "maximum_rejections": self.maximum_rejections,
            "maximum_field_characters": self.maximum_field_characters,
            "maximum_columns": self.maximum_columns,
            "maximum_output_bytes": self.maximum_output_bytes,
            "maximum_record_bytes": self.maximum_record_bytes,
            "maximum_price_sgd": str(self.maximum_price_sgd),
            "maximum_rejection_rate": self.maximum_rejection_rate,
        }


@dataclass(frozen=True, slots=True)
class AwinFeedPolicy:
    advertiser_id: str
    feed_id: str
    retailer: str
    licence_or_access_note: str
    rights: DataUseRights
    allowed_currencies: tuple[str, ...]
    allowed_link_hosts: tuple[str, ...]
    category_mappings: Mapping[str, ComponentKind]
    compression: str
    delimiter: str
    default_condition: ListingCondition
    production_catalog_eligible: bool
    training_eligible: bool
    published_claims_eligible: bool
    published_claims_grant_reference: str | None
    allow_non_new: bool
    limits: AwinFeedLimits
    expected_input_sha256: str | None = None

    @property
    def source_name(self) -> str:
        advertiser = self.advertiser_id.lower().replace(".", "_")
        feed = self.feed_id.lower().replace(".", "_")
        return f"awin_{advertiser}_{feed}"

    @property
    def source_uri(self) -> str:
        return f"awin://advertisers/{self.advertiser_id}/feeds/{self.feed_id}"

    @classmethod
    def from_verified(cls, verified: VerifiedSignedPolicy) -> AwinFeedPolicy:
        payload = verified.payload
        if not isinstance(payload, Mapping):
            raise TypeError("signed Awin policy payload must be an object")
        required = {
            "schema_version",
            "advertiser_id",
            "feed_id",
            "retailer",
            "licence_or_access_note",
            "rights",
            "allowed_currencies",
            "allowed_link_hosts",
            "category_mappings",
            "feed",
            "default_condition",
        }
        optional = {
            "production_catalog_eligible",
            "training_eligible",
            "published_claims_eligible",
            "published_claims_grant_reference",
            "allow_non_new",
            "expected_input_sha256",
            "limits",
        }
        missing = sorted(required - set(payload))
        unknown = sorted(set(payload) - required - optional)
        if missing:
            raise ValueError(f"Awin policy payload missing fields: {missing}")
        if unknown:
            raise ValueError(f"Awin policy payload contains unknown fields: {unknown}")
        if payload.get("schema_version") != AWIN_POLICY_PAYLOAD_SCHEMA_VERSION:
            raise ValueError("unsupported Awin policy payload schema")
        advertiser_id = _source_identifier(payload["advertiser_id"], "advertiser_id")
        feed_id = _source_identifier(payload["feed_id"], "feed_id")
        retailer = _required_text(payload["retailer"], "retailer", maximum=256)
        access_note = _required_text(
            payload["licence_or_access_note"], "licence_or_access_note", maximum=2_048
        )
        _reject_credential_text(access_note, label="licence_or_access_note")
        raw_rights = payload["rights"]
        if not isinstance(raw_rights, Mapping):
            raise TypeError("rights must be an object")
        rights = _rights_from_grants(raw_rights)
        currencies = _string_tuple(payload["allowed_currencies"], "allowed_currencies")
        normalized_currencies = tuple(sorted({value.upper() for value in currencies}))
        if normalized_currencies != ("SGD",):
            raise ValueError("Awin Singapore ingestion permits exactly the SGD currency")
        hosts = _string_tuple(payload["allowed_link_hosts"], "allowed_link_hosts")
        normalized_hosts = tuple(sorted({_normalized_host(value) for value in hosts}))
        raw_categories = payload["category_mappings"]
        if not isinstance(raw_categories, Mapping) or not raw_categories:
            raise TypeError("category_mappings must be a non-empty object")
        category_mappings: dict[str, ComponentKind] = {}
        for raw_key, raw_category in raw_categories.items():
            key = _category_key(str(raw_key))
            if key in category_mappings:
                raise ValueError(f"duplicate normalized category mapping: {key}")
            category_mappings[key] = ComponentKind(str(raw_category))
        raw_feed = payload["feed"]
        if not isinstance(raw_feed, Mapping) or set(raw_feed) != {
            "format",
            "compression",
            "delimiter",
        }:
            raise ValueError("feed must contain exactly format, compression, and delimiter")
        if raw_feed.get("format") != "csv":
            raise ValueError("this bounded Awin adapter currently supports CSV only")
        compression = str(raw_feed.get("compression"))
        if compression not in {"none", "gzip"}:
            raise ValueError("Awin feed compression must be none or gzip")
        delimiter = str(raw_feed.get("delimiter"))
        if delimiter not in {",", ";", "|", "\t"}:
            raise ValueError("Awin CSV delimiter must be comma, semicolon, pipe, or tab")
        default_condition = ListingCondition(str(payload["default_condition"]))
        flags: dict[str, bool] = {}
        for name in (
            "production_catalog_eligible",
            "training_eligible",
            "published_claims_eligible",
            "allow_non_new",
        ):
            value = payload.get(name, False)
            if type(value) is not bool:
                raise TypeError(f"{name} must be a boolean")
            flags[name] = value
        if flags["training_eligible"]:
            rights.assert_allowed(DataUse.TRAIN)
        if flags["published_claims_eligible"]:
            rights.assert_allowed(DataUse.DISPLAY)
            rights.assert_allowed(DataUse.DERIVE)
        raw_claim_grant = payload.get("published_claims_grant_reference")
        claim_grant = (
            _required_text(
                raw_claim_grant,
                "published_claims_grant_reference",
                maximum=512,
            )
            if raw_claim_grant is not None
            else None
        )
        if claim_grant is not None:
            _reject_credential_text(
                claim_grant,
                label="published_claims_grant_reference",
            )
        if flags["published_claims_eligible"] and claim_grant is None:
            raise PermissionError(
                "published claims require an explicit signed contractual grant reference"
            )
        if not flags["published_claims_eligible"] and claim_grant is not None:
            raise ValueError(
                "published_claims_grant_reference requires published_claims_eligible=true"
            )
        if flags["production_catalog_eligible"]:
            rights.assert_catalog_serving_allowed(territory="SG")
        expected_sha = payload.get("expected_input_sha256")
        if expected_sha is not None and (
            not isinstance(expected_sha, str) or _SHA256.fullmatch(expected_sha) is None
        ):
            raise ValueError("expected_input_sha256 must be a lowercase SHA-256 or null")
        policy = cls(
            advertiser_id=advertiser_id,
            feed_id=feed_id,
            retailer=retailer,
            licence_or_access_note=access_note,
            rights=rights,
            allowed_currencies=normalized_currencies,
            allowed_link_hosts=normalized_hosts,
            category_mappings=dict(sorted(category_mappings.items())),
            compression=compression,
            delimiter=delimiter,
            default_condition=default_condition,
            production_catalog_eligible=flags["production_catalog_eligible"],
            training_eligible=flags["training_eligible"],
            published_claims_eligible=flags["published_claims_eligible"],
            published_claims_grant_reference=claim_grant,
            allow_non_new=flags["allow_non_new"],
            limits=AwinFeedLimits.from_mapping(payload.get("limits")),
            expected_input_sha256=expected_sha,
        )
        policy.assert_ingestion_allowed()
        return policy

    def assert_ingestion_allowed(self) -> None:
        self.rights.assert_consent_active()
        if "SG" not in self.rights.territories:
            raise PermissionError("Awin feed rights do not permit territory SG")
        self.rights.assert_allowed(DataUse.CACHE)
        self.rights.assert_allowed(DataUse.DERIVE)
        if not self.allow_non_new and self.default_condition is not ListingCondition.NEW:
            raise PermissionError("new-only ingestion requires a signed default_condition of new")


@dataclass(frozen=True, slots=True)
class AwinFeedSnapshot:
    raw: RawSnapshot
    policy: AwinFeedPolicy
    verified_policy: VerifiedSignedPolicy
    authorization_receipt_path: Path


@dataclass(slots=True)
class _ParseStatistics:
    decompressed_bytes: int = 0
    scanned_records: int = 0
    accepted_records: int = 0
    rejected_records: int = 0
    rejection_reasons: Counter[str] = field(default_factory=Counter)
    category_counts: Counter[str] = field(default_factory=Counter)
    unique_source_listing_ids: int = 0


class _CountingRawReader(io.RawIOBase):
    def __init__(self, source: BinaryIO, maximum_bytes: int) -> None:
        self.source = source
        self.maximum_bytes = maximum_bytes
        self.byte_count = 0

    def readable(self) -> bool:
        return True

    def readinto(self, buffer: Any) -> int:
        chunk = self.source.read(len(buffer))
        if not chunk:
            return 0
        self.byte_count += len(chunk)
        if self.byte_count > self.maximum_bytes:
            raise AwinFeedLimitError(f"Awin feed exceeds {self.maximum_bytes} decompressed bytes")
        buffer[: len(chunk)] = chunk
        return len(chunk)


class _HashingRawReader(io.RawIOBase):
    """Bound and hash compressed input while a decoder consumes it."""

    def __init__(self, source: BinaryIO, maximum_bytes: int) -> None:
        self.source = source
        self.maximum_bytes = maximum_bytes
        self.byte_count = 0
        self.digest = hashlib.sha256()
        self._credential_tail = b""

    def readable(self) -> bool:
        return True

    def readinto(self, buffer: Any) -> int:
        chunk = self.source.read(len(buffer))
        if not chunk:
            return 0
        self.byte_count += len(chunk)
        if self.byte_count > self.maximum_bytes:
            raise AwinFeedLimitError(f"Awin feed exceeds {self.maximum_bytes} compressed bytes")
        credential_window = self._credential_tail + chunk
        if _CREDENTIAL_BYTES.search(credential_window) is not None:
            raise AwinFeedError("Awin feed contains credential-bearing URL material")
        self._credential_tail = credential_window[-64:]
        self.digest.update(chunk)
        buffer[: len(chunk)] = chunk
        return len(chunk)


class AwinLocalFeedAdapter:
    """Parse one signed, local Awin CSV or gzip-wrapped CSV as a bounded stream."""

    def __init__(
        self,
        *,
        raw_root: str | Path,
        verified_policy: VerifiedSignedPolicy,
    ) -> None:
        self.raw_root = Path(raw_root)
        self.verified_policy = verified_policy
        self.policy = AwinFeedPolicy.from_verified(verified_policy)

    def fetch(self, *, feed_path: str | Path) -> AwinFeedSnapshot:
        self.policy.assert_ingestion_allowed()
        source_path = Path(feed_path)
        expected_gzip = self.policy.compression == "gzip"
        inspected_sha256 = _inspect_local_feed(
            source_path,
            gzip_encoded=expected_gzip,
            maximum_input_bytes=self.policy.limits.maximum_input_bytes,
            maximum_decompressed_bytes=self.policy.limits.maximum_decompressed_bytes,
        )
        if (
            self.policy.expected_input_sha256 is not None
            and inspected_sha256 != self.policy.expected_input_sha256
        ):
            raise AwinFeedError("Awin feed SHA-256 does not match the signed policy")
        gzip_encoded = expected_gzip
        suffix = ".csv.gz" if gzip_encoded else ".csv"
        snapshot = snapshot_local_file(
            source_name=self.policy.source_name,
            source_url=self.policy.source_uri,
            source_type="authorized_retailer_feed",
            source_path=source_path,
            raw_root=self.raw_root,
            parser_version=AWIN_PARSER_VERSION,
            licence_or_access_note=self.policy.licence_or_access_note,
            suffix=suffix,
            media_type="application/gzip" if gzip_encoded else "text/csv",
            # Bind the copied snapshot to the exact bytes inspected for secrets.
            expected_sha256=inspected_sha256,
            maximum_bytes=self.policy.limits.maximum_input_bytes,
        )
        receipt_path = self._write_authorization_receipt(snapshot)
        return AwinFeedSnapshot(
            raw=snapshot,
            policy=self.policy,
            verified_policy=self.verified_policy,
            authorization_receipt_path=receipt_path,
        )

    def _write_authorization_receipt(self, snapshot: RawSnapshot) -> Path:
        verified = self.verified_policy
        semantic: dict[str, Any] = {
            "schema_version": AWIN_AUTHORIZATION_RECEIPT_SCHEMA_VERSION,
            "source_name": self.policy.source_name,
            "source_uri": self.policy.source_uri,
            "raw_snapshot_sha256": snapshot.content_sha256,
            "raw_byte_count": snapshot.byte_count,
            "parser_version": AWIN_PARSER_VERSION,
            "policy_id": verified.policy_id,
            "policy_issued_at": verified.issued_at.isoformat(),
            "policy_expires_at": verified.expires_at.isoformat(),
            "policy_sha256": verified.policy_sha256,
            "signature_sha256": verified.signature_sha256,
            "trust_root_sha256": verified.trust_root_sha256,
            "signer_key_id": verified.key_id,
            "data_use_rights": self.policy.rights.to_dict(),
            "production_catalog_eligible": self.policy.production_catalog_eligible,
            "training_eligible": self.policy.training_eligible,
            "published_claims_eligible": self.policy.published_claims_eligible,
            "published_claims_grant_reference": (self.policy.published_claims_grant_reference),
        }
        semantic["content_sha256"] = hashlib.sha256(_canonical_json(semantic)).hexdigest()
        receipt_identity = _canonical_json(
            {
                "policy_sha256": verified.policy_sha256,
                "raw_snapshot_sha256": snapshot.content_sha256,
            }
        )
        receipt_id = hashlib.sha256(receipt_identity).hexdigest()
        path = snapshot.path.with_name(f"auth-{receipt_id}.json")
        raw = json.dumps(semantic, allow_nan=False, indent=2, sort_keys=True).encode() + b"\n"
        if path.exists():
            if path.read_bytes() != raw:
                raise AwinFeedError("existing Awin authorization receipt conflicts")
            return path
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=path.parent,
                prefix=".auth.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                handle.write(raw)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            temporary = None
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return path

    def materialize(
        self,
        snapshot: AwinFeedSnapshot,
        *,
        processed_root: str | Path,
    ) -> StreamingProcessedArtifacts:
        if snapshot.policy != self.policy or snapshot.verified_policy != self.verified_policy:
            raise AwinFeedError("Awin snapshot authority does not match this adapter")
        manifest_metadata = self._manifest_metadata(snapshot)
        run_sha256 = hashlib.sha256(_canonical_json(manifest_metadata)).hexdigest()
        existing = load_existing_streaming_artifacts(
            processed_root=processed_root,
            source_name=self.policy.source_name,
            run_sha256=run_sha256,
            expected_metadata=manifest_metadata,
        )
        if existing is not None:
            return existing
        statistics = _ParseStatistics()
        processed_path = Path(processed_root)
        processed_path.mkdir(parents=True, exist_ok=True)
        baseline = load_previous_quality_baseline(
            processed_root=processed_path,
            source_name=self.policy.source_name,
            current_snapshot_sha256=snapshot.raw.content_sha256,
            variant=None,
        )
        with (
            tempfile.TemporaryDirectory(prefix=".awin-dedupe.", dir=processed_path) as dedupe_root,
            closing(sqlite3.connect(Path(dedupe_root) / "source-identities.sqlite3")) as identities,
            AtomicStreamingJSONLWriter(
                processed_root=processed_path,
                source_name=self.policy.source_name,
                run_sha256=run_sha256,
                maximum_output_bytes=self.policy.limits.maximum_output_bytes,
                maximum_record_bytes=self.policy.limits.maximum_record_bytes,
            ) as writer,
        ):
            identities.execute("PRAGMA temp_store=FILE")
            identities.execute("PRAGMA cache_size=-8192")
            identities.execute("CREATE TABLE source_identity (source_listing_id TEXT PRIMARY KEY)")
            for accepted, rejected in self._iter_normalized(
                snapshot,
                statistics,
                identities=identities,
            ):
                if accepted is not None:
                    writer.write_record(accepted)
                else:
                    assert rejected is not None
                    writer.write_rejection(rejected)
            identities.commit()
            quality = self._quality_report(snapshot, statistics, baseline=baseline)
            return writer.seal(
                quality_report=quality.to_dict(),
                manifest_metadata={
                    **manifest_metadata,
                    "statistics": self._statistics_payload(statistics),
                },
            )

    def _manifest_metadata(self, snapshot: AwinFeedSnapshot) -> dict[str, Any]:
        verified = self.verified_policy
        return {
            "raw_snapshot_sha256": snapshot.raw.content_sha256,
            "authorization_receipt_sha256": _sha256_file(snapshot.authorization_receipt_path),
            "parser_version": AWIN_PARSER_VERSION,
            "policy_id": verified.policy_id,
            "policy_issued_at": verified.issued_at.isoformat(),
            "policy_expires_at": verified.expires_at.isoformat(),
            "policy_sha256": verified.policy_sha256,
            "signature_sha256": verified.signature_sha256,
            "trust_root_sha256": verified.trust_root_sha256,
            "signer_key_id": verified.key_id,
            "source_uri": self.policy.source_uri,
            "production_catalog_eligible": self.policy.production_catalog_eligible,
            "training_eligible": self.policy.training_eligible,
            "published_claims_eligible": self.policy.published_claims_eligible,
            "published_claims_grant_reference": (self.policy.published_claims_grant_reference),
            "limits": self.policy.limits.to_dict(),
        }

    def _iter_normalized(
        self,
        snapshot: AwinFeedSnapshot,
        statistics: _ParseStatistics,
        *,
        identities: sqlite3.Connection,
    ) -> Iterator[tuple[dict[str, Any] | None, dict[str, Any] | None]]:
        previous_field_limit = csv.field_size_limit()
        csv.field_size_limit(self.policy.limits.maximum_field_characters)
        try:
            with snapshot.raw.path.open("rb") as raw_handle:
                binary_source: Any
                if self.policy.compression == "gzip":
                    binary_source = gzip.GzipFile(fileobj=raw_handle, mode="rb")
                else:
                    binary_source = raw_handle
                counter = _CountingRawReader(
                    binary_source,
                    self.policy.limits.maximum_decompressed_bytes,
                )
                with (
                    io.BufferedReader(counter, buffer_size=64 * 1024) as buffered,
                    io.TextIOWrapper(
                        buffered, encoding="utf-8-sig", errors="strict", newline=""
                    ) as text,
                ):
                    reader = csv.DictReader(
                        text,
                        delimiter=self.policy.delimiter,
                        strict=True,
                    )
                    raw_headers = reader.fieldnames
                    if raw_headers is None:
                        raise AwinFeedError("Awin CSV has no header row")
                    headers = [_header_name(value) for value in raw_headers]
                    if len(headers) > self.policy.limits.maximum_columns:
                        raise AwinFeedLimitError("Awin CSV exceeds its signed column limit")
                    if len(headers) != len(set(headers)):
                        raise AwinFeedError("Awin CSV contains duplicate normalized headers")
                    for row_number, raw_row in enumerate(reader, start=2):
                        statistics.scanned_records += 1
                        if statistics.scanned_records > self.policy.limits.maximum_records:
                            raise AwinFeedLimitError("Awin CSV exceeds its signed record limit")
                        row = {
                            _header_name(str(key)): str(value or "").strip()
                            for key, value in raw_row.items()
                            if key is not None
                        }
                        if _row_contains_credential(row):
                            raise AwinFeedError(
                                "Awin feed contains credential-bearing URL material"
                            )
                        if any(
                            len(value) > self.policy.limits.maximum_field_characters
                            for value in row.values()
                        ):
                            raise AwinFeedLimitError("Awin CSV field exceeds its signed limit")
                        if sum(len(value) for value in row.values()) > (
                            self.policy.limits.maximum_record_bytes
                        ):
                            raise AwinFeedLimitError("Awin CSV row exceeds its signed record limit")
                        record_hint = _first(
                            row, "merchant_product_id", "aw_product_id", "product_id"
                        )
                        try:
                            record = self._normalize_row(
                                row=row,
                                row_number=row_number,
                                snapshot=snapshot,
                            )
                            source_id = str(record["source_record_id"])
                            try:
                                identities.execute(
                                    "INSERT INTO source_identity(source_listing_id) VALUES (?)",
                                    (source_id,),
                                )
                            except sqlite3.IntegrityError:
                                raise ValueError("duplicate_source_listing_id") from None
                            statistics.unique_source_listing_ids += 1
                            if statistics.unique_source_listing_ids % 10_000 == 0:
                                identities.commit()
                        except (InvalidOperation, TypeError, ValueError) as exc:
                            reason = _safe_rejection_reason(exc)
                            statistics.rejection_reasons[reason] += 1
                            statistics.rejected_records += 1
                            if statistics.rejected_records > self.policy.limits.maximum_rejections:
                                raise AwinFeedLimitError(
                                    "Awin CSV exceeds its signed rejection limit"
                                ) from exc
                            yield (
                                None,
                                {
                                    "record_id": record_hint or f"row-{row_number}",
                                    "reason": reason,
                                    "details": {"row": row_number},
                                },
                            )
                            continue
                        category = str(record["normalisation_metadata"]["category"])
                        statistics.category_counts[category] += 1
                        statistics.accepted_records += 1
                        yield record, None
                    statistics.decompressed_bytes = counter.byte_count
        except (UnicodeDecodeError, csv.Error, gzip.BadGzipFile, EOFError) as exc:
            raise AwinFeedError(f"Awin CSV is malformed: {type(exc).__name__}") from exc
        finally:
            csv.field_size_limit(previous_field_limit)

    def _normalize_row(
        self,
        *,
        row: Mapping[str, str],
        row_number: int,
        snapshot: AwinFeedSnapshot,
    ) -> dict[str, Any]:
        merchant_id = _required_first(row, "merchant_id")
        if merchant_id != self.policy.advertiser_id:
            raise ValueError("merchant_id_mismatch")
        merchant_name = _required_first(row, "merchant_name")
        if merchant_name.casefold() != self.policy.retailer.casefold():
            raise ValueError("merchant_name_mismatch")
        raw_product_id = _required_first(
            row,
            "merchant_product_id",
            "aw_product_id",
            "product_id",
        )
        if len(raw_product_id) > 256:
            raise ValueError("source_product_id_too_long")
        source_listing_id = f"{self.policy.advertiser_id}:{raw_product_id}"
        title = _required_first(row, "product_name", "name")
        if len(title) > 1_000:
            raise ValueError("title_too_long")
        currency = _required_first(row, "currency").upper()
        if currency not in self.policy.allowed_currencies:
            raise ValueError("currency_not_allowed")
        price_field, raw_price = _first_named(row, "search_price", "price", "store_price")
        if raw_price is None:
            raise ValueError("price_missing")
        base_price = _money(raw_price, currency=currency, positive=True)
        if base_price > self.policy.limits.maximum_price_sgd:
            raise ValueError("price_above_signed_limit")
        raw_shipping = _first(row, "delivery_cost", "delcost")
        if not raw_shipping:
            raise ValueError("shipping_price_missing")
        shipping_price = _shipping_money(raw_shipping, currency=currency)
        listing_url = _required_first(
            row,
            "aw_deep_link",
            "merchant_deep_link",
            "deep_link",
            "purl",
        )
        _validate_listing_url(listing_url, self.policy.allowed_link_hosts)
        category = self._category(row)
        stock_status, stock_basis = _stock_status(row)
        condition = _condition(row.get("condition"), self.policy.default_condition)
        if not self.policy.allow_non_new and condition is not ListingCondition.NEW:
            raise ValueError("non_new_condition")
        observed_at = snapshot.raw.retrieved_at.astimezone(UTC)
        listing_id = stable_identifier(
            "listing",
            "awin",
            self.policy.advertiser_id,
            raw_product_id,
            length=32,
        )
        placeholder_product_id = stable_identifier(
            "unmatched_product",
            "awin",
            self.policy.advertiser_id,
            raw_product_id,
        )
        listing = RetailerListing(
            listing_id=listing_id,
            product_id=placeholder_product_id,
            retailer=self.policy.retailer,
            source_listing_id=source_listing_id,
            title=title,
            condition=condition,
            currency=currency,
            base_price=base_price,
            shipping_price=shipping_price,
            stock_status=stock_status,
            seller_name=merchant_name,
            listing_url=listing_url,
            first_seen_at=observed_at,
            last_seen_at=observed_at,
        )
        price_snapshot = PriceSample(
            snapshot_id=stable_identifier(
                "price",
                listing_id,
                observed_at.isoformat(),
                base_price,
                shipping_price,
                stock_status.value,
                length=32,
            ),
            listing_id=listing_id,
            observed_at=observed_at,
            base_price=base_price,
            shipping_price=shipping_price,
            stock_status=stock_status,
            promotion_text=_optional(row.get("promotional_text"), maximum=2_000),
        )
        raw_record = json.dumps(
            dict(sorted(row.items())),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        verified = self.verified_policy
        return {
            "schema_version": NORMALISED_RECORD_SCHEMA_VERSION,
            "record_type": "retailer_listing",
            "source_record_id": source_listing_id,
            "archive_snapshot_sha256": snapshot.raw.content_sha256,
            "raw_record_sha256": sha256_bytes(raw_record),
            "development_only": not self.policy.production_catalog_eligible,
            "training_eligible": self.policy.training_eligible,
            "published_claims_eligible": self.policy.published_claims_eligible,
            "data_use_rights": self.policy.rights.to_dict(),
            "rights_authority": {
                "policy_id": verified.policy_id,
                "policy_issued_at": verified.issued_at.isoformat(),
                "policy_expires_at": verified.expires_at.isoformat(),
                "policy_sha256": verified.policy_sha256,
                "signature_sha256": verified.signature_sha256,
                "trust_root_sha256": verified.trust_root_sha256,
                "signer_key_id": verified.key_id,
                "published_claims_grant_reference": (self.policy.published_claims_grant_reference),
            },
            "provenance": {
                "source_name": self.policy.source_name,
                "source_url": self.policy.source_uri,
                "source_type": "authorized_retailer_feed",
                "retrieved_at": observed_at.isoformat(),
                "parser_version": AWIN_PARSER_VERSION,
                "licence_or_access_note": self.policy.licence_or_access_note,
                "extraction_confidence": 1.0,
            },
            "normalisation_metadata": {
                "row_number": row_number,
                "category": category.value,
                "canonical_mapping_status": "unmatched",
                "advertiser_id": self.policy.advertiser_id,
                "feed_id": self.policy.feed_id,
                "brand": _optional(_first(row, "brand_name", "brand"), maximum=256),
                "manufacturer_part_number": _optional(
                    _first(row, "mpn", "model_number", "modelno"), maximum=256
                ),
                "gtin": _optional(_first(row, "product_gtin", "ean", "upc"), maximum=64),
                "colour": _optional(_first(row, "colour", "color"), maximum=128),
                "price_field": price_field,
                "stock_basis": stock_basis,
                "source_last_updated": _optional(row.get("last_updated"), maximum=128),
            },
            "data": {
                "listing": listing.model_dump(mode="json"),
                "price_snapshot": price_snapshot.model_dump(mode="json"),
            },
        }

    def _category(self, row: Mapping[str, str]) -> ComponentKind:
        candidate_keys: list[str] = []
        for prefix, field_name in (
            ("id", "category_id"),
            ("path", "merchant_product_category_path"),
            ("merchant", "merchant_category"),
            ("name", "category_name"),
        ):
            value = row.get(field_name, "").strip()
            if value:
                candidate_keys.append(_category_key(f"{prefix}:{value}"))
        matched = {
            self.policy.category_mappings[key]
            for key in candidate_keys
            if key in self.policy.category_mappings
        }
        if not matched:
            raise ValueError("category_unmapped")
        if len(matched) != 1:
            raise ValueError("category_mapping_conflict")
        return next(iter(matched))

    def _quality_report(
        self,
        snapshot: AwinFeedSnapshot,
        statistics: _ParseStatistics,
        *,
        baseline: QualityBaseline | None,
    ) -> DataQualityReport:
        total = statistics.accepted_records + statistics.rejected_records
        rejection_rate = statistics.rejected_records / total if total else 0.0
        checks: list[dict[str, Any]] = [
            {
                "name": "accepted_records_present",
                "severity": "error",
                "count": int(statistics.accepted_records == 0),
                "message": "At least one authorized PC-component offer is required.",
            },
            {
                "name": "signed_policy_verified",
                "severity": "error",
                "count": 0,
                "message": "The exact policy and pinned trust root were verified before fetch.",
            },
            {
                "name": "rejection_rate_within_threshold",
                "severity": "error",
                "count": int(rejection_rate > self.policy.limits.maximum_rejection_rate),
                "message": (
                    f"Observed rejection rate {rejection_rate:.4f}; signed threshold is "
                    f"{self.policy.limits.maximum_rejection_rate:.4f}."
                ),
            },
            {
                "name": "production_catalog_rights",
                "severity": "warning",
                "count": int(not self.policy.production_catalog_eligible),
                "message": "Only an explicitly signed production grant permits serving.",
            },
        ]
        checks.extend(
            quality_regression_checks(
                baseline=baseline,
                accepted_count=statistics.accepted_records,
                rejection_rate=rejection_rate,
                record_type_counts={"retailer_listing": statistics.accepted_records},
                category_counts=dict(statistics.category_counts),
                policy=QualityRegressionPolicy(),
            )
        )
        errors = sum(int(item["count"]) for item in checks if item["severity"] == "error")
        warnings = sum(int(item["count"]) for item in checks if item["severity"] == "warning")
        status = "fail" if errors else "warning" if warnings else "pass"
        return DataQualityReport(
            schema_version=DATA_QUALITY_SCHEMA_VERSION,
            source_name=self.policy.source_name,
            snapshot_sha256=snapshot.raw.content_sha256,
            status=status,
            accepted_count=statistics.accepted_records,
            rejected_count=statistics.rejected_records,
            rejection_rate=rejection_rate,
            checks=tuple(checks),
            record_type_counts={"retailer_listing": statistics.accepted_records},
            category_counts=dict(sorted(statistics.category_counts.items())),
            eligibility_counts={
                "training_eligible": (
                    statistics.accepted_records if self.policy.training_eligible else 0
                ),
                "training_ineligible": (
                    0 if self.policy.training_eligible else statistics.accepted_records
                ),
                "published_claims_eligible": (
                    statistics.accepted_records if self.policy.published_claims_eligible else 0
                ),
                "published_claims_ineligible": (
                    0 if self.policy.published_claims_eligible else statistics.accepted_records
                ),
            },
            source_statistics=self._statistics_payload(statistics),
        )

    def _statistics_payload(self, statistics: _ParseStatistics) -> dict[str, Any]:
        return {
            "advertiser_id": self.policy.advertiser_id,
            "feed_id": self.policy.feed_id,
            "decompressed_bytes": statistics.decompressed_bytes,
            "scanned_records": statistics.scanned_records,
            "accepted_records": statistics.accepted_records,
            "rejected_records": statistics.rejected_records,
            "rejection_reasons": dict(sorted(statistics.rejection_reasons.items())),
            "category_counts": dict(sorted(statistics.category_counts.items())),
            "unique_source_listing_ids": statistics.unique_source_listing_ids,
            "production_catalog_eligible": self.policy.production_catalog_eligible,
            "training_eligible": self.policy.training_eligible,
            "published_claims_eligible": self.policy.published_claims_eligible,
            "published_claims_grant_reference": (self.policy.published_claims_grant_reference),
            "data_use_rights": self.policy.rights.to_dict(),
        }


def _rights_from_grants(payload: Mapping[str, Any]) -> DataUseRights:
    required = {
        "contract_reference",
        "contract_version_url",
        "consent_effective_on",
        "consent_expires_on",
        "retention_days",
        "deletion_required_on_termination",
        "deletion_sla_days",
        "territories",
        "grants",
    }
    missing = sorted(required - set(payload))
    unknown = sorted(set(payload) - required)
    if missing:
        raise ValueError(f"Awin rights missing fields: {missing}")
    if unknown:
        raise ValueError(f"Awin rights contain unknown fields: {unknown}")
    _reject_credential_text(payload["contract_reference"], label="contract_reference")
    _reject_credential_text(payload["contract_version_url"], label="contract_version_url")
    grants = payload["grants"]
    if not isinstance(grants, list | tuple):
        raise TypeError("rights grants must be an array")
    normalized_grants: set[DataUse] = set()
    for raw_grant in grants:
        grant = DataUse(str(raw_grant))
        if grant in normalized_grants:
            raise ValueError(f"duplicate Awin rights grant: {grant.value}")
        normalized_grants.add(grant)
    rights_payload = {key: value for key, value in payload.items() if key != "grants"}
    rights_payload.update({use.field_name: use in normalized_grants for use in DataUse})
    return DataUseRights.from_mapping(rights_payload)


def _source_identifier(value: object, label: str) -> str:
    text = str(value).strip()
    if _AWIN_ID.fullmatch(text) is None:
        raise ValueError(f"{label} must be a numeric Awin identifier")
    return text


def _reject_credential_text(value: object, *, label: str) -> None:
    text = str(value)
    if _CREDENTIAL_TEXT.search(text):
        raise ValueError(f"{label} must not contain credential-bearing URL material")


def _row_contains_credential(row: Mapping[str, str]) -> bool:
    return any(_CREDENTIAL_TEXT.search(value) is not None for value in row.values())


def _inspect_local_feed(
    path: Path,
    *,
    gzip_encoded: bool,
    maximum_input_bytes: int,
    maximum_decompressed_bytes: int,
) -> str:
    """Reject credential material and bind inspection to the copied raw bytes."""

    is_junction = getattr(os.path, "isjunction", None)
    if path.is_symlink() or bool(is_junction is not None and is_junction(path)):
        raise AwinFeedError("Awin feed must not be a symlink or junction")
    candidate = path.resolve()
    try:
        with candidate.open("rb") as source:
            before = os.fstat(source.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise AwinFeedError("Awin feed must be a regular local file")
            if before.st_size > maximum_input_bytes:
                raise AwinFeedLimitError(
                    f"Awin feed exceeds {maximum_input_bytes} compressed bytes"
                )
            actual_gzip = source.read(2) == b"\x1f\x8b"
            source.seek(0)
            if actual_gzip is not gzip_encoded:
                raise AwinFeedError("Awin feed compression does not match the signed policy")
            tracker = _HashingRawReader(source, maximum_input_bytes)
            with io.BufferedReader(tracker, buffer_size=64 * 1024) as buffered:
                try:
                    if gzip_encoded:
                        with gzip.GzipFile(fileobj=buffered, mode="rb") as decoded:
                            _scan_for_credentials(
                                decoded,
                                maximum_bytes=maximum_decompressed_bytes,
                            )
                        # Hash any bytes the gzip decoder legitimately left buffered.
                        while buffered.read(64 * 1024):
                            pass
                    else:
                        _scan_for_credentials(
                            buffered,
                            maximum_bytes=maximum_decompressed_bytes,
                        )
                except (gzip.BadGzipFile, EOFError) as exc:
                    raise AwinFeedError(
                        f"Awin gzip input is malformed: {type(exc).__name__}"
                    ) from exc
            after = os.fstat(source.fileno())
    except AwinFeedError:
        raise
    except OSError as exc:
        raise AwinFeedError(f"Awin feed cannot be inspected: {exc}") from exc
    if (
        before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or tracker.byte_count != before.st_size
    ):
        raise AwinFeedError("Awin feed changed while it was inspected")
    return tracker.digest.hexdigest()


def _scan_for_credentials(source: _BinaryReadable, *, maximum_bytes: int) -> None:
    byte_count = 0
    tail = b""
    while chunk := source.read(64 * 1024):
        byte_count += len(chunk)
        if byte_count > maximum_bytes:
            raise AwinFeedLimitError(f"Awin feed exceeds {maximum_bytes} decompressed bytes")
        window = tail + chunk
        if _CREDENTIAL_BYTES.search(window) is not None:
            raise AwinFeedError("Awin feed contains credential-bearing URL material")
        tail = window[-64:]


def _required_text(value: object, label: str, *, maximum: int) -> str:
    text = str(value).strip()
    if not text or len(text) > maximum or any(character in text for character in "\r\n"):
        raise ValueError(f"{label} must be non-empty, bounded, single-line text")
    return text


def _string_tuple(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list | tuple) or not value:
        raise TypeError(f"{label} must be a non-empty array")
    result = tuple(str(item).strip() for item in value)
    if any(not item for item in result):
        raise ValueError(f"{label} contains an empty value")
    return result


def _normalized_host(value: str) -> str:
    host = value.strip().casefold().rstrip(".")
    if not host or "/" in host or ":" in host or host == "localhost" or host.endswith(".localhost"):
        raise ValueError(f"invalid allowed link host: {value!r}")
    return host


def _header_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().casefold()).strip("_")


def _category_key(value: str) -> str:
    prefix, separator, raw_value = value.partition(":")
    if not separator or prefix not in {"id", "path", "merchant", "name"}:
        raise ValueError(f"unsupported Awin category mapping key: {value!r}")
    normalized = re.sub(r"\s+", " ", raw_value.strip().casefold())
    if not normalized or len(normalized) > 512:
        raise ValueError("Awin category mapping value is empty or too long")
    return f"{prefix}:{normalized}"


def _first(row: Mapping[str, str], *names: str) -> str:
    for name in names:
        value = row.get(name, "").strip()
        if value:
            return value
    return ""


def _first_named(row: Mapping[str, str], *names: str) -> tuple[str, str | None]:
    for name in names:
        value = row.get(name, "").strip()
        if value:
            return name, value
    return "", None


def _required_first(row: Mapping[str, str], *names: str) -> str:
    value = _first(row, *names)
    if not value:
        raise ValueError(f"missing_{names[0]}")
    return value


def _money(value: str, *, currency: str, positive: bool) -> Decimal:
    match = _MONEY.fullmatch(value.strip())
    if match is None:
        raise ValueError("invalid_money")
    embedded_currency = match.group(2)
    if embedded_currency is not None and embedded_currency.upper() != currency:
        raise ValueError("money_currency_conflict")
    amount = Decimal(match.group(1).replace(",", "")).quantize(
        Decimal("0.01"),
        rounding=ROUND_HALF_UP,
    )
    if amount < 0 or (positive and amount == 0):
        raise ValueError("invalid_money")
    return amount


def _shipping_money(value: str, *, currency: str) -> Decimal:
    normalized = value.strip().casefold()
    if normalized in {"free", "free delivery", "free shipping"}:
        return Decimal("0.00")
    return _money(value, currency=currency, positive=False)


def _validate_listing_url(url: str, allowed_hosts: tuple[str, ...]) -> None:
    try:
        parsed = urlsplit(url)
        port = parsed.port
    except ValueError as exc:
        raise ValueError("invalid_listing_url") from exc
    host = (parsed.hostname or "").casefold().rstrip(".")
    if (
        parsed.scheme != "https"
        or not host
        or parsed.username
        or parsed.password
        or port
        not in {
            None,
            443,
        }
    ):
        raise ValueError("invalid_listing_url")
    if host not in allowed_hosts:
        raise ValueError("listing_host_not_allowed")
    decoded_url = unquote(url)
    if _CREDENTIAL_TEXT.search(decoded_url) is not None:
        raise ValueError("credential_bearing_listing_url")
    query_keys = {key.casefold() for key, _value in parse_qsl(parsed.query, keep_blank_values=True)}
    if query_keys.intersection(_CREDENTIAL_QUERY_KEYS):
        raise ValueError("credential_bearing_listing_url")


def _optional(value: object, *, maximum: int) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    return text[:maximum]


def _condition(value: object, default: ListingCondition) -> ListingCondition:
    text = _header_name(str(value or ""))
    if not text:
        return default
    aliases = {
        "new": ListingCondition.NEW,
        "open_box": ListingCondition.OPEN_BOX,
        "refurbished": ListingCondition.REFURBISHED,
        "used": ListingCondition.USED,
        "unknown": ListingCondition.UNKNOWN,
    }
    if text not in aliases:
        raise ValueError("condition_unrecognized")
    return aliases[text]


def _boolean(value: str, label: str) -> bool | None:
    text = _header_name(value)
    if not text:
        return None
    if text in {"1", "true", "yes", "y", "available", "in_stock"}:
        return True
    if text in {"0", "false", "no", "n", "unavailable", "out_of_stock"}:
        return False
    raise ValueError(f"{label}_unrecognized")


def _stock_status(row: Mapping[str, str]) -> tuple[StockStatus, list[str]]:
    signals: list[StockStatus] = []
    basis: list[str] = []
    raw_status = _header_name(row.get("stock_status", ""))
    status_aliases = {
        "in_stock": StockStatus.IN_STOCK,
        "instock": StockStatus.IN_STOCK,
        "available": StockStatus.IN_STOCK,
        "out_of_stock": StockStatus.OUT_OF_STOCK,
        "outofstock": StockStatus.OUT_OF_STOCK,
        "sold_out": StockStatus.OUT_OF_STOCK,
        "backorder": StockStatus.BACKORDER,
        "back_order": StockStatus.BACKORDER,
        "preorder": StockStatus.PREORDER,
        "pre_order": StockStatus.PREORDER,
        "unknown": StockStatus.UNKNOWN,
    }
    if raw_status:
        if raw_status not in status_aliases:
            raise ValueError("stock_status_unrecognized")
        signals.append(status_aliases[raw_status])
        basis.append("stock_status")
    in_stock = _boolean(row.get("in_stock", ""), "in_stock")
    if in_stock is not None:
        signals.append(StockStatus.IN_STOCK if in_stock else StockStatus.OUT_OF_STOCK)
        basis.append("in_stock")
    for_sale = _boolean(row.get("is_for_sale", ""), "is_for_sale")
    if for_sale is False:
        signals.append(StockStatus.OUT_OF_STOCK)
        basis.append("is_for_sale")
    preorder = _boolean(row.get("pre_order", ""), "pre_order")
    if preorder is True:
        signals.append(StockStatus.PREORDER)
        basis.append("pre_order")
    meaningful = {signal for signal in signals if signal is not StockStatus.UNKNOWN}
    if len(meaningful) > 1:
        raise ValueError("stock_signal_conflict")
    return (next(iter(meaningful)) if meaningful else StockStatus.UNKNOWN), basis


def _safe_rejection_reason(exc: Exception) -> str:
    value = str(exc).strip()
    if re.fullmatch(r"[a-z0-9_]{1,80}", value):
        return value
    return f"invalid_awin_record_{type(exc).__name__.lower()}"


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "AWIN_AUTHORIZATION_RECEIPT_SCHEMA_VERSION",
    "AWIN_PARSER_VERSION",
    "AWIN_POLICY_PAYLOAD_SCHEMA_VERSION",
    "AwinFeedError",
    "AwinFeedLimitError",
    "AwinFeedLimits",
    "AwinFeedPolicy",
    "AwinFeedSnapshot",
    "AwinLocalFeedAdapter",
]
