"""Provenance-preserving processed catalog loading for online serving.

Retailer-to-canonical mapping is precision first: automatic matches require an exact
manufacturer-part-number occurrence, category agreement, a title-leading brand match,
and no known numeric or colour-variant conflict. Other rows remain unmatched unless an
operator provides a separate, explicit reviewed-mapping manifest.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any
from urllib.parse import urlsplit

from sqlalchemy.orm import Session

from pc_build_recommender.data_rights import DataUse, DataUseRights
from pc_build_recommender.domain import (
    BenchmarkResult,
    MasterProduct,
    ComponentKind,
    PriceSample,
    ProductStatus,
    RetailerListing,
    ReviewNote,
    SourceProvenance,
    StockStatus,
)
from pc_build_recommender.entity_resolution import (
    CanonicalProductRecord,
    EntityResolutionPolicy,
    EntityResolutionRuntime,
    ListingRecord,
    entity_resolution_release_sha256,
    load_entity_resolution_runtime,
)
from pc_build_recommender.entity_resolution.conflicts import find_numeric_conflicts
from pc_build_recommender.entity_resolution.normalization import normalize_identifier, tokenize

from .canonical_identity import audit_canonical_product_identities
from .entity_matcher import CatalogEntityMatcher, listing_record_from_offer
from .er_gate import load_entity_resolution_evaluation
from .mapping_review import (
    MappingDecision,
    MappingOutcome,
    MappingReview,
    ReviewStatus,
    load_mapping_reviews,
)
from .readiness import (
    CatalogReadinessAccumulator,
    CatalogReadinessReport,
    ProductionCatalogPolicy,
)
from .repository import CatalogRepository
from .seed import deterministic_id

PROCESSED_SCHEMA_VERSION = "pc-build-recommender.normalised-record.v1"
DEFAULT_MAX_JSONL_LINE_BYTES = 8 * 1024 * 1024
REVIEW_EVIDENCE_SCHEMA_VERSION = "pc-build-recommender.review-evidence.v1"
REVIEW_EVIDENCE_RECORD_TYPE = "review_evidence"
REVIEW_EVIDENCE_MAX_TEXT_CHARACTERS = 500
_REVIEW_EVIDENCE_ASPECTS = frozenset(
    {
        "performance",
        "thermals",
        "noise",
        "reliability",
        "driver_software",
        "installation",
        "build_quality",
        "warranty_support",
        "power_consumption",
        "value",
    }
)
_DATA_USE_RIGHT_FIELDS = (
    "may_display",
    "may_cache",
    "may_store_history",
    "may_redistribute",
    "may_embed",
    "may_train",
    "may_derive",
)
_CONTROLLED_DYNACORE_SOURCE = "dynacore_controlled_pdf"
_AWARE_ISO_TIMESTAMP = re.compile(
    r"\A\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?(?:Z|[+-]\d{2}:\d{2})\Z"
)
_SHA256 = re.compile(r"\A[0-9a-f]{64}\Z")

_CATEGORY_ALIASES = {
    "psu": ComponentKind.POWER_SUPPLY,
    "power_supply": ComponentKind.POWER_SUPPLY,
    "cpu_cooler": ComponentKind.COOLER,
}
_COLOUR_TOKENS = frozenset(
    {
        "black",
        "white",
        "silver",
        "gray",
        "grey",
        "red",
        "blue",
        "green",
        "orange",
        "pink",
        "purple",
        "brown",
    }
)


def _category(value: object) -> ComponentKind:
    normalised = str(value).strip().casefold().replace("-", "_").replace(" ", "_")
    return _CATEGORY_ALIASES.get(normalised, ComponentKind(normalised))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_jsonl_objects(
    path: str | Path,
    *,
    max_line_bytes: int = DEFAULT_MAX_JSONL_LINE_BYTES,
) -> Iterable[dict[str, Any]]:
    """Stream JSONL objects with a hard per-record memory bound."""

    if max_line_bytes < 1:
        raise ValueError("max_line_bytes must be positive")
    jsonl_path = Path(path)
    with jsonl_path.open("rb") as source:
        line_number = 0
        while raw_line := source.readline(max_line_bytes + 1):
            line_number += 1
            if len(raw_line) > max_line_bytes:
                raise ValueError(
                    f"JSON line exceeds {max_line_bytes} bytes at {jsonl_path}:{line_number}"
                )
            if not raw_line.strip():
                continue
            try:
                value = json.loads(raw_line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError(f"invalid JSON at {jsonl_path}:{line_number}") from error
            if not isinstance(value, dict):
                raise ValueError(
                    f"processed record must be an object at {jsonl_path}:{line_number}"
                )
            yield value


def _jsonl(path: Path) -> Iterable[dict[str, Any]]:
    return iter_jsonl_objects(path)


def _strict_aware_timestamp(value: object, *, label: str) -> datetime:
    """Parse a canonical UTC-aware timestamp without accepting ambiguous forms."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a strict aware ISO-8601 timestamp")
    if _AWARE_ISO_TIMESTAMP.fullmatch(value) is None:
        try:
            loosely_parsed = datetime.fromisoformat(value)
        except ValueError:
            loosely_parsed = None
        if loosely_parsed is not None and (
            loosely_parsed.tzinfo is None or loosely_parsed.utcoffset() is None
        ):
            raise ValueError(f"{label} must be timezone-aware")
        raise ValueError(f"{label} must be a strict aware ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{label} must be a strict aware ISO-8601 timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{label} must be a strict aware ISO-8601 timestamp")
    return parsed.astimezone(UTC)


def _require_https_review_url(value: object, *, label: str) -> str:
    """Return a safe outbound evidence URL suitable for a public product page."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"review evidence {label} must be a non-empty HTTPS URL")
    result = value.strip()
    parsed = urlsplit(result)
    if (
        parsed.scheme != "https"
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise ValueError(f"review evidence {label} must be a credential-free HTTPS URL")
    return result


def _load_review_evidence_record(
    envelope: Mapping[str, Any],
    *,
    path: Path,
    known_product_ids: frozenset[str],
) -> ReviewNote:
    """Validate one explicitly permitted review-evidence envelope.

    The artifact intentionally accepts no generic review or crawler shape.  An
    operator must provide a short, cited statement with the exact active data
    rights needed to retain, derive from, and display it in Singapore.
    """

    expected_fields = frozenset(
        {
            "schema_version",
            "record_type",
            "data",
            "data_use_rights",
            "provenance",
        }
    )
    actual_fields = frozenset(envelope)
    if actual_fields != expected_fields:
        missing = sorted(expected_fields - actual_fields)
        extra = sorted(actual_fields - expected_fields)
        raise ValueError(
            "review evidence envelope fields do not match the contract; "
            f"missing={missing}, extra={extra} in {path}"
        )
    if envelope.get("schema_version") != REVIEW_EVIDENCE_SCHEMA_VERSION:
        raise ValueError(f"unsupported review evidence schema in {path}")
    if envelope.get("record_type") != REVIEW_EVIDENCE_RECORD_TYPE:
        raise ValueError(f"expected review_evidence record in {path}")
    raw_data = envelope.get("data")
    if not isinstance(raw_data, Mapping):
        raise ValueError(f"review evidence record is missing a data object in {path}")
    try:
        evidence = ReviewNote.model_validate(raw_data)
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid review evidence in {path}: {error}") from error
    if evidence.product_id not in known_product_ids:
        raise ValueError(
            f"review evidence references an unknown canonical product: {evidence.product_id}"
        )
    if evidence.aspect not in _REVIEW_EVIDENCE_ASPECTS:
        raise ValueError(
            "review evidence aspect must be one of "
            f"{sorted(_REVIEW_EVIDENCE_ASPECTS)}: {evidence.aspect}"
        )
    if len(evidence.evidence_text) > REVIEW_EVIDENCE_MAX_TEXT_CHARACTERS:
        raise ValueError(
            f"review evidence text exceeds {REVIEW_EVIDENCE_MAX_TEXT_CHARACTERS} characters"
        )
    source_url = _require_https_review_url(evidence.source_url, label="data.source_url")

    raw_rights = envelope.get("data_use_rights")
    if not isinstance(raw_rights, Mapping):
        raise ValueError("review evidence requires complete machine-readable data-use rights")
    try:
        rights = DataUseRights.from_mapping(raw_rights)
        rights.assert_catalog_serving_allowed(territory="SG")
    except (PermissionError, TypeError, ValueError) as error:
        raise ValueError(
            f"review evidence lacks active SG display and derived-data rights: {error}"
        ) from error

    provenance = envelope.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("review evidence provenance must be an object")
    required_provenance = {
        "source_name",
        "source_url",
        "retrieved_at",
        "raw_content_hash",
        "parser_version",
        "licence_or_access_note",
    }
    actual_provenance = set(provenance)
    if actual_provenance != required_provenance:
        missing = sorted(required_provenance - actual_provenance)
        extra = sorted(actual_provenance - required_provenance)
        raise ValueError(
            "review evidence provenance fields do not match the contract; "
            f"missing={missing}, extra={extra}"
        )
    for provenance_field in ("source_name", "parser_version", "licence_or_access_note"):
        if (
            not isinstance(provenance[provenance_field], str)
            or not provenance[provenance_field].strip()
        ):
            raise ValueError(
                f"review evidence provenance.{provenance_field} must be non-empty text"
            )
    provenance_url = _require_https_review_url(
        provenance["source_url"], label="provenance.source_url"
    )
    if provenance_url != source_url:
        raise ValueError("review evidence source URL must match its provenance source URL")
    raw_content_hash = provenance["raw_content_hash"]
    if not isinstance(raw_content_hash, str) or _SHA256.fullmatch(raw_content_hash) is None:
        raise ValueError("review evidence provenance.raw_content_hash must be a SHA-256")
    retrieved_at = _strict_aware_timestamp(
        provenance["retrieved_at"], label="review evidence provenance.retrieved_at"
    )
    now = datetime.now(UTC)
    if retrieved_at > now:
        raise ValueError("review evidence provenance.retrieved_at cannot be in the future")
    consent_effective_at = datetime.combine(
        rights.consent_effective_on,
        datetime.min.time(),
        tzinfo=UTC,
    )
    if retrieved_at < consent_effective_at:
        raise ValueError(
            "review evidence provenance.retrieved_at predates "
            f"consent_effective_on={rights.consent_effective_on.isoformat()}"
        )
    if rights.retention_days is not None:
        age_days = (now - retrieved_at).total_seconds() / 86_400
        if age_days > rights.retention_days:
            raise ValueError(
                "review evidence is stale: provenance.retrieved_at exceeds "
                f"retention_days={rights.retention_days}"
            )
    if evidence.published_at is not None:
        published_at = evidence.published_at
        if published_at.tzinfo is None or published_at.utcoffset() is None:
            raise ValueError("review evidence data.published_at must be timezone-aware")
        if published_at.astimezone(UTC) > retrieved_at:
            raise ValueError(
                "review evidence data.published_at cannot be later than provenance.retrieved_at"
            )
    return evidence.model_copy(update={"source_url": source_url})


def load_review_evidence(
    review_evidence_path: str | Path | None,
    *,
    known_product_ids: Iterable[str],
    max_line_bytes: int = DEFAULT_MAX_JSONL_LINE_BYTES,
) -> tuple[ReviewNote, ...]:
    """Load a bounded, permitted review-evidence release artifact.

    ``None`` denotes an intentionally review-free catalogue release.  In a
    non-development deployment the serving manifest and settings require an
    explicit (possibly empty) artifact instead.
    """

    if review_evidence_path is None:
        return ()
    path = Path(review_evidence_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"review evidence artifact not found: {path}")
    products = frozenset(known_product_ids)
    evidence_by_id: dict[str, ReviewNote] = {}
    for envelope in iter_jsonl_objects(path, max_line_bytes=max_line_bytes):
        evidence = _load_review_evidence_record(
            envelope,
            path=path,
            known_product_ids=products,
        )
        if evidence.evidence_id in evidence_by_id:
            raise ValueError(f"duplicate review evidence ID: {evidence.evidence_id}")
        evidence_by_id[evidence.evidence_id] = evidence
    return tuple(
        sorted(
            evidence_by_id.values(),
            key=lambda item: (item.product_id, item.aspect, item.evidence_id),
        )
    )


def _require_envelope(
    envelope: Mapping[str, Any],
    *,
    record_type: str,
    path: Path,
) -> Mapping[str, Any]:
    if envelope.get("schema_version") != PROCESSED_SCHEMA_VERSION:
        raise ValueError(f"unsupported processed schema in {path}")
    if envelope.get("record_type") != record_type:
        raise ValueError(f"expected {record_type} record in {path}")
    data = envelope.get("data")
    if not isinstance(data, Mapping):
        raise ValueError(f"processed record is missing a data object in {path}")
    return data


def _validate_controlled_offer_rights(envelope: Mapping[str, Any]) -> None:
    rights = envelope.get("data_use_rights")
    if rights is None:
        # Legacy fingerprinted controlled artifacts predate the explicit rights object.
        return
    if not isinstance(rights, Mapping):
        raise ValueError("controlled retailer data_use_rights must be an object")
    if any(rights.get(field) is not False for field in _DATA_USE_RIGHT_FIELDS):
        raise ValueError("controlled Dynacore rows must remain barred for every data use")


def _offer_source_name(envelope: Mapping[str, Any]) -> str:
    provenance = envelope.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("retailer offer provenance must be an object")
    source_name = str(provenance.get("source_name", "")).strip()
    if not source_name:
        raise ValueError("retailer offer provenance requires source_name")
    return source_name


def _validate_offer_freshness(
    envelope: Mapping[str, Any],
    rights: DataUseRights,
) -> None:
    """Reject incoherent, future, or contract-expired offers before serving."""

    def timestamp(container: Mapping[str, Any], field: str, label: str) -> datetime:
        raw_value = container.get(field)
        if not isinstance(raw_value, str) or not raw_value.strip():
            raise ValueError(f"retailer offer {label} must be a strict aware ISO-8601 timestamp")
        if _AWARE_ISO_TIMESTAMP.fullmatch(raw_value) is None:
            try:
                loosely_parsed = datetime.fromisoformat(raw_value)
            except ValueError:
                loosely_parsed = None
            if loosely_parsed is not None and (
                loosely_parsed.tzinfo is None or loosely_parsed.utcoffset() is None
            ):
                raise ValueError(f"retailer offer {label} must be timezone-aware")
            raise ValueError(f"retailer offer {label} must be a strict aware ISO-8601 timestamp")
        try:
            value = datetime.fromisoformat(raw_value.replace("Z", "+00:00"))
        except ValueError as error:
            raise ValueError(
                f"retailer offer {label} must be a strict aware ISO-8601 timestamp"
            ) from error
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(f"retailer offer {label} must be a strict aware ISO-8601 timestamp")
        return value.astimezone(UTC)

    provenance = envelope.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("retailer offer provenance must be an object")
    data = envelope.get("data")
    if not isinstance(data, Mapping):
        raise ValueError("retailer offer data must be an object")
    listing = data.get("listing")
    snapshot = data.get("price_snapshot")
    if not isinstance(listing, Mapping) or not isinstance(snapshot, Mapping):
        raise ValueError("retailer record requires listing and price_snapshot objects")

    retrieved_at = timestamp(provenance, "retrieved_at", "provenance.retrieved_at")
    first_seen_at = timestamp(listing, "first_seen_at", "listing.first_seen_at")
    last_seen_at = timestamp(listing, "last_seen_at", "listing.last_seen_at")
    observed_at = timestamp(snapshot, "observed_at", "price_snapshot.observed_at")

    now = datetime.now(UTC)
    relevant_timestamps = (
        ("provenance.retrieved_at", retrieved_at),
        ("listing.first_seen_at", first_seen_at),
        ("listing.last_seen_at", last_seen_at),
        ("price_snapshot.observed_at", observed_at),
    )
    for label, value in relevant_timestamps:
        if value > now:
            raise ValueError(f"retailer offer {label} cannot be in the future")

    consent_effective_at = datetime.combine(
        rights.consent_effective_on,
        datetime.min.time(),
        tzinfo=UTC,
    )
    for label, value in relevant_timestamps:
        if value < consent_effective_at:
            raise ValueError(
                f"retailer offer {label} predates "
                f"consent_effective_on={rights.consent_effective_on.isoformat()}"
            )

    if rights.retention_days is not None:
        retention_timestamps = (
            ("provenance.retrieved_at", retrieved_at),
            ("price_snapshot.observed_at", observed_at),
        )
        for label, value in retention_timestamps:
            age_days = (now - value).total_seconds() / 86_400
            if age_days > rights.retention_days:
                raise ValueError(
                    f"retailer offer is stale: {label} exceeds "
                    f"retention_days={rights.retention_days}"
                )

    if first_seen_at > last_seen_at:
        raise ValueError(
            "retailer offer listing.first_seen_at cannot be later than listing.last_seen_at"
        )
    if observed_at != last_seen_at:
        raise ValueError(
            "retailer offer price_snapshot.observed_at must equal listing.last_seen_at"
        )
    if retrieved_at < observed_at:
        raise ValueError(
            "retailer offer provenance.retrieved_at cannot be earlier than "
            "price_snapshot.observed_at"
        )


def _validate_offer_governance(
    envelope: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> None:
    """Fail closed on offer rights while preserving the pinned Dynacore quarantine."""

    source_name = _offer_source_name(envelope)
    training_eligible = envelope.get("training_eligible")
    published_claims_eligible = envelope.get("published_claims_eligible")
    if type(training_eligible) is not bool or type(published_claims_eligible) is not bool:
        raise ValueError("retailer offer eligibility flags must be explicit booleans")

    envelope_development_only = envelope.get("development_only")
    metadata_development_only = metadata.get("development_only")
    for marker in (envelope_development_only, metadata_development_only):
        if marker is not None and type(marker) is not bool:
            raise ValueError("retailer offer development_only must be a boolean when provided")
    if (
        envelope_development_only is not None
        and metadata_development_only is not None
        and envelope_development_only is not metadata_development_only
    ):
        raise ValueError("retailer offer development_only markers conflict")
    development_only = bool(
        envelope_development_only
        if envelope_development_only is not None
        else metadata_development_only
    )

    if source_name == _CONTROLLED_DYNACORE_SOURCE:
        if training_eligible or published_claims_eligible:
            raise ValueError("controlled Dynacore rows must remain training/claims ineligible")
        _validate_controlled_offer_rights(envelope)
        if development_only is not True:
            raise ValueError("controlled Dynacore record must be marked development_only")
        return

    raw_rights = envelope.get("data_use_rights")
    if not isinstance(raw_rights, Mapping):
        raise ValueError("retailer offer requires complete machine-readable data-use rights")
    try:
        rights = DataUseRights.from_mapping(raw_rights)
        rights.assert_consent_active()
    except (PermissionError, TypeError, ValueError) as error:
        raise ValueError(f"retailer offer has invalid data-use rights: {error}") from error
    _validate_offer_freshness(envelope, rights)

    if development_only:
        if any(getattr(rights, field) is not False for field in _DATA_USE_RIGHT_FIELDS):
            raise ValueError("development-only retailer offers must be barred for every data use")
        if training_eligible or published_claims_eligible:
            raise ValueError(
                "development-only retailer offers must remain training/claims ineligible"
            )
        raise ValueError(
            "development-only retailer offer is quarantined and cannot enter the serving catalogue"
        )

    try:
        rights.assert_catalog_serving_allowed(territory="SG")
        if training_eligible:
            rights.assert_allowed(DataUse.TRAIN)
        if published_claims_eligible:
            rights.assert_allowed(DataUse.DISPLAY)
            rights.assert_allowed(DataUse.DERIVE)
    except PermissionError as error:
        raise ValueError(
            f"retailer offer lacks active SG production catalogue rights: {error}"
        ) from error


def _resolve_offer_artifact_path(
    offer_path: str | Path | None,
    *,
    dynacore_path: str | Path | None,
) -> Path:
    """Resolve the generic offer artifact with a deprecated Dynacore keyword alias."""

    if offer_path is None and dynacore_path is None:
        raise ValueError("a processed retailer offer artifact path is required")
    if offer_path is not None and dynacore_path is not None:
        generic = Path(offer_path).resolve()
        legacy = Path(dynacore_path).resolve()
        if generic != legacy:
            raise ValueError("offer_path and dynacore_path refer to different artifacts")
        return generic
    selected_path = offer_path if offer_path is not None else dynacore_path
    assert selected_path is not None
    return Path(selected_path).resolve()


def _brand_prefix_matches(title: str, brand: str) -> bool:
    title_tokens = tokenize(title)
    brand_tokens = tokenize(brand)
    return bool(brand_tokens and title_tokens[: len(brand_tokens)] == brand_tokens)


def _colour_conflict(title: str, product: MasterProduct) -> bool:
    title_colours = _COLOUR_TOKENS.intersection(tokenize(title))
    if len(title_colours) > 1:
        # A combined retailer row cannot identify one unique colour SKU.
        return True
    product_colour = product.common_attributes.colour
    if not title_colours or not product_colour:
        return False
    product_colours = _COLOUR_TOKENS.intersection(tokenize(product_colour))
    return bool(product_colours and title_colours.isdisjoint(product_colours))


def _numeric_conflict(title: str, product: MasterProduct) -> bool:
    listing = ListingRecord(
        listing_id="candidate",
        title=title,
        category=product.category.value,
        brand=product.brand,
    )
    canonical = CanonicalProductRecord(
        product_id=product.product_id,
        category=product.category.value,
        brand=product.brand,
        model=product.model,
        canonical_name=product.canonical_name,
        manufacturer_part_number=product.manufacturer_part_number,
        gtin=product.gtin,
        attributes=product.category_attributes.model_dump(mode="json"),
    )
    return bool(find_numeric_conflicts(listing, canonical))


ReviewedMapping = MappingReview


def load_reviewed_mappings(path: str | Path | None) -> dict[str, MappingReview]:
    """Compatibility wrapper for the durable approved/rejected review manifest."""

    return load_mapping_reviews(path)


@dataclass(frozen=True, slots=True)
class ProcessedCatalogStats:
    product_count: int
    offer_count: int
    matched_listing_count: int
    auto_matched_count: int
    reviewed_matched_count: int
    unmatched_offer_count: int
    rejected_conflict_count: int
    ambiguous_exact_match_count: int
    products_by_category: Mapping[str, int]
    matched_listings_by_category: Mapping[str, int]
    in_stock_listings_by_category: Mapping[str, int]
    known_in_stock_listing_count: int
    data_version: str
    review_rejected_count: int = 0
    manual_review_count: int = 0
    model_rejected_count: int = 0

    def __post_init__(self) -> None:
        if sum(self.products_by_category.values()) != self.product_count:
            raise ValueError("products_by_category must sum to product_count")
        if self.matched_listing_count != (self.auto_matched_count + self.reviewed_matched_count):
            raise ValueError("matched listing count must equal automatic plus reviewed matches")
        accounted = self.matched_listing_count + self.unmatched_offer_count
        accounted += self.rejected_conflict_count + self.ambiguous_exact_match_count
        accounted += self.review_rejected_count
        accounted += self.manual_review_count + self.model_rejected_count
        if accounted != self.offer_count:
            raise ValueError("mapping outcome counts must sum to offer_count")
        if sum(self.matched_listings_by_category.values()) != self.matched_listing_count:
            raise ValueError("matched listings by category must sum to matched listing count")
        if sum(self.in_stock_listings_by_category.values()) != self.known_in_stock_listing_count:
            raise ValueError("in-stock listings by category must sum to known in-stock count")
        object.__setattr__(
            self,
            "products_by_category",
            MappingProxyType(dict(self.products_by_category)),
        )
        object.__setattr__(
            self,
            "matched_listings_by_category",
            MappingProxyType(dict(self.matched_listings_by_category)),
        )
        object.__setattr__(
            self,
            "in_stock_listings_by_category",
            MappingProxyType(dict(self.in_stock_listings_by_category)),
        )

    @property
    def priced_categories(self) -> frozenset[str]:
        return frozenset(
            category for category, count in self.matched_listings_by_category.items() if count > 0
        )

    @property
    def has_complete_priced_coverage(self) -> bool:
        return self.priced_categories == frozenset(category.value for category in ComponentKind)

    @property
    def has_complete_in_stock_coverage(self) -> bool:
        return frozenset(
            category for category, count in self.in_stock_listings_by_category.items() if count > 0
        ) == frozenset(category.value for category in ComponentKind)

    def to_dict(self) -> dict[str, Any]:
        return {
            "product_count": self.product_count,
            "offer_count": self.offer_count,
            "matched_listing_count": self.matched_listing_count,
            "auto_matched_count": self.auto_matched_count,
            "reviewed_matched_count": self.reviewed_matched_count,
            "review_rejected_count": self.review_rejected_count,
            "manual_review_count": self.manual_review_count,
            "model_rejected_count": self.model_rejected_count,
            "unmatched_offer_count": self.unmatched_offer_count,
            "rejected_conflict_count": self.rejected_conflict_count,
            "ambiguous_exact_match_count": self.ambiguous_exact_match_count,
            "products_by_category": dict(self.products_by_category),
            "matched_listings_by_category": dict(self.matched_listings_by_category),
            "in_stock_listings_by_category": dict(self.in_stock_listings_by_category),
            "priced_categories": sorted(self.priced_categories),
            "known_in_stock_listing_count": self.known_in_stock_listing_count,
            "has_complete_priced_coverage": self.has_complete_priced_coverage,
            "has_complete_in_stock_coverage": self.has_complete_in_stock_coverage,
            "data_version": self.data_version,
        }


@dataclass(frozen=True, slots=True)
class ProcessedCatalogData:
    products: tuple[MasterProduct, ...]
    listings: tuple[RetailerListing, ...]
    price_snapshots: tuple[PriceSample, ...]
    stats: ProcessedCatalogStats
    review_evidence: tuple[ReviewNote, ...] = ()
    match_method_by_listing: Mapping[str, str] = field(default_factory=dict)
    listing_provenance: tuple[SourceProvenance, ...] = ()
    mapping_decisions: tuple[MappingDecision, ...] = ()
    readiness: CatalogReadinessReport | None = None

    def __post_init__(self) -> None:
        product_ids = {product.product_id for product in self.products}
        evidence_ids: set[str] = set()
        for evidence in self.review_evidence:
            if evidence.product_id not in product_ids:
                raise ValueError(
                    "review evidence product_id does not match the processed catalogue: "
                    f"{evidence.product_id}"
                )
            if evidence.evidence_id in evidence_ids:
                raise ValueError(
                    "processed catalogue contains duplicate review evidence ID: "
                    f"{evidence.evidence_id}"
                )
            evidence_ids.add(evidence.evidence_id)
        object.__setattr__(
            self,
            "review_evidence",
            tuple(
                sorted(
                    self.review_evidence,
                    key=lambda item: (item.product_id, item.aspect, item.evidence_id),
                )
            ),
        )
        object.__setattr__(
            self,
            "match_method_by_listing",
            MappingProxyType(dict(self.match_method_by_listing)),
        )


def _listing_source_provenance(
    envelope: Mapping[str, Any],
    listing: RetailerListing,
) -> SourceProvenance:
    raw = envelope.get("provenance")
    if not isinstance(raw, Mapping):
        raise ValueError(f"retailer listing lacks provenance: {listing.listing_id}")
    raw_hash = str(envelope.get("raw_record_sha256", "")).strip()
    if not raw_hash:
        raise ValueError(f"retailer listing lacks raw_record_sha256: {listing.listing_id}")
    payload = dict(raw)
    payload.update(
        provenance_id=deterministic_id("src", "processed-retailer", listing.listing_id, raw_hash),
        product_id=None,
        listing_id=listing.listing_id,
        raw_content_hash=raw_hash,
    )
    return SourceProvenance.model_validate(payload)


def _automatic_candidates(
    title: str,
    category: ComponentKind,
    products: Sequence[MasterProduct],
) -> tuple[MasterProduct, ...]:
    title_identifier = normalize_identifier(title)
    result: list[MasterProduct] = []
    for product in products:
        if product.category != category or not _brand_prefix_matches(title, product.brand):
            continue
        mpn = normalize_identifier(product.manufacturer_part_number)
        if len(mpn) < 6 or mpn not in title_identifier:
            continue
        if _numeric_conflict(title, product) or _colour_conflict(title, product):
            continue
        result.append(product)
    return tuple(sorted(result, key=lambda item: item.product_id))


def load_processed_catalog(
    buildcores_path: str | Path,
    offer_path: str | Path | None = None,
    *,
    dynacore_path: str | Path | None = None,
    reviewed_mapping_path: str | Path | None = None,
    review_evidence_path: str | Path | None = None,
    entity_resolution_evaluation_path: str | Path | None = None,
    entity_resolution_model_path: str | Path | None = None,
    entity_resolution_runtime: EntityResolutionRuntime | None = None,
    entity_resolution_policy: EntityResolutionPolicy | None = None,
    entity_resolution_binding_sha256: str | None = None,
    allow_unpromoted_entity_resolution: bool = False,
    require_production_entity_resolution: bool = False,
    production_policy: ProductionCatalogPolicy | None = None,
    max_line_bytes: int = DEFAULT_MAX_JSONL_LINE_BYTES,
) -> ProcessedCatalogData:
    """Load an immutable serving snapshot from product and governed-offer JSONL."""

    product_path = Path(buildcores_path).resolve()
    resolved_offer_path = _resolve_offer_artifact_path(
        offer_path,
        dynacore_path=dynacore_path,
    )
    if not product_path.is_file():
        raise FileNotFoundError(f"BuildCores processed catalog not found: {product_path}")
    if not resolved_offer_path.is_file():
        raise FileNotFoundError(f"processed retailer offers not found: {resolved_offer_path}")

    readiness_accumulator = CatalogReadinessAccumulator()
    entity_resolution_evaluation = load_entity_resolution_evaluation(
        entity_resolution_evaluation_path
    )
    if entity_resolution_model_path is not None and entity_resolution_runtime is not None:
        raise ValueError(
            "provide either entity_resolution_model_path or entity_resolution_runtime, not both"
        )
    runtime = entity_resolution_runtime
    if entity_resolution_model_path is not None:
        runtime = load_entity_resolution_runtime(
            entity_resolution_model_path,
            allow_unpromoted_human_diagnostic=allow_unpromoted_entity_resolution,
        )
    if require_production_entity_resolution:
        if runtime is None:
            raise ValueError("production entity resolution requires a serving model")
        if entity_resolution_policy is None:
            raise ValueError("production entity resolution requires an exact serving policy")
        if (
            entity_resolution_binding_sha256 is None
            or len(entity_resolution_binding_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in entity_resolution_binding_sha256
            )
        ):
            raise ValueError("production entity resolution requires an exact release binding")
        active_policy = production_policy or ProductionCatalogPolicy()
        runtime = runtime.authorize_for_production(
            entity_resolution_evaluation,
            minimum_precision=active_policy.minimum_er_precision,
            minimum_labelled_pairs=active_policy.minimum_er_labelled_pairs,
            minimum_auto_matches=entity_resolution_policy.minimum_auto_matches,
            minimum_recall=entity_resolution_policy.minimum_recall,
            minimum_f1=entity_resolution_policy.minimum_f1,
        )
    products: list[MasterProduct] = []
    for envelope in iter_jsonl_objects(product_path, max_line_bytes=max_line_bytes):
        data = _require_envelope(envelope, record_type="canonical_product", path=product_path)
        product = MasterProduct.model_validate(data)
        products.append(product)
        readiness_accumulator.observe_product(product)
    product_ids = [product.product_id for product in products]
    if len(product_ids) != len(set(product_ids)):
        raise ValueError("processed BuildCores catalog contains duplicate product IDs")
    readiness_accumulator.observe_canonical_identity_preflight(
        audit_canonical_product_identities(products)
    )
    products_by_id = {product.product_id: product for product in products}
    review_evidence = load_review_evidence(
        review_evidence_path,
        known_product_ids=products_by_id,
        max_line_bytes=max_line_bytes,
    )
    matcher_products = tuple(
        CanonicalProductRecord(
            product_id=product.product_id,
            category=product.category.value,
            brand=product.brand,
            model=product.model,
            canonical_name=product.canonical_name,
            manufacturer_part_number=product.manufacturer_part_number,
            gtin=product.gtin,
            attributes={
                **product.category_attributes.model_dump(mode="json"),
                "colour": product.common_attributes.colour,
            },
        )
        for product in products
    )
    if entity_resolution_policy is None:
        matcher = CatalogEntityMatcher(matcher_products, runtime=runtime)
    else:
        matcher = CatalogEntityMatcher(
            matcher_products,
            runtime=runtime,
            max_candidates=entity_resolution_policy.max_candidates,
            minimum_text_score=entity_resolution_policy.minimum_text_score,
            minimum_auto_margin=entity_resolution_policy.minimum_auto_margin,
            evidence_candidate_limit=entity_resolution_policy.evidence_candidate_limit,
        )

    reviewed = load_reviewed_mappings(reviewed_mapping_path)
    seen_reviewed: set[str] = set()
    seen_listing_ids: set[str] = set()
    seen_snapshot_ids: set[str] = set()
    listings: list[RetailerListing] = []
    snapshots: list[PriceSample] = []
    listing_provenance: list[SourceProvenance] = []
    decisions: list[MappingDecision] = []
    methods: dict[str, str] = {}
    matched_by_category: Counter[str] = Counter()
    in_stock_by_category: Counter[str] = Counter()
    offer_count = auto_count = reviewed_count = 0
    unmatched_count = conflict_count = ambiguous_count = review_rejected_count = 0
    manual_review_count = model_rejected_count = 0

    for envelope in iter_jsonl_objects(resolved_offer_path, max_line_bytes=max_line_bytes):
        data = _require_envelope(
            envelope,
            record_type="retailer_listing",
            path=resolved_offer_path,
        )
        raw_listing = data.get("listing")
        raw_snapshot = data.get("price_snapshot")
        metadata = envelope.get("normalisation_metadata")
        if not isinstance(raw_listing, Mapping) or not isinstance(raw_snapshot, Mapping):
            raise ValueError("retailer record requires listing and price_snapshot objects")
        if not isinstance(metadata, Mapping):
            raise ValueError("retailer record requires normalisation metadata")
        _validate_offer_governance(envelope, metadata)
        readiness_accumulator.observe_offer_rights(envelope.get("data_use_rights"))
        offer_count += 1
        source_listing = RetailerListing.model_validate(raw_listing)
        snapshot = PriceSample.model_validate(raw_snapshot)
        if source_listing.listing_id in seen_listing_ids:
            raise ValueError(f"duplicate retailer listing ID: {source_listing.listing_id}")
        if snapshot.snapshot_id in seen_snapshot_ids:
            raise ValueError(f"duplicate retailer price snapshot ID: {snapshot.snapshot_id}")
        if snapshot.listing_id != source_listing.listing_id:
            raise ValueError(f"price snapshot listing mismatch: {source_listing.listing_id}")
        seen_listing_ids.add(source_listing.listing_id)
        seen_snapshot_ids.add(snapshot.snapshot_id)
        provenance = _listing_source_provenance(envelope, source_listing)
        readiness_accumulator.observe_offer_provenance(
            provenance,
            listing_id=source_listing.listing_id,
        )
        listing_id = source_listing.listing_id
        category = _category(metadata.get("category"))
        title = source_listing.title
        source_listing_id = source_listing.source_listing_id
        selected: MasterProduct | None = None
        method: str | None = None
        outcome: MappingOutcome
        outcome_candidates: tuple[str, ...] = ()
        reason: str | None = None
        probability: float | None = None
        model_version: str | None = None
        decision_evidence: Mapping[str, Any] = {}

        override = reviewed.get(listing_id)
        if override is not None:
            seen_reviewed.add(listing_id)
            if override.review_status is ReviewStatus.REJECTED:
                review_rejected_count += 1
                outcome = MappingOutcome.REVIEW_REJECTED
                reason = override.evidence
                decisions.append(
                    MappingDecision(
                        listing_id=listing_id,
                        source_listing_id=source_listing_id,
                        title=title,
                        category=category.value,
                        outcome=outcome,
                        reason=reason,
                        evidence={
                            "reviewed_by": override.reviewed_by,
                            "reviewed_at": override.reviewed_at,
                            "review_status": override.review_status.value,
                        },
                    )
                )
                continue
            if override.product_id is None:
                raise AssertionError("approved mapping is missing product_id")
            candidate = products_by_id.get(override.product_id)
            if candidate is None:
                raise ValueError(
                    f"reviewed mapping references unknown product: {override.product_id}"
                )
            if candidate.category != category:
                raise ValueError(f"reviewed mapping category conflict: {listing_id}")
            if _numeric_conflict(title, candidate) or _colour_conflict(title, candidate):
                raise ValueError(f"reviewed mapping has a hard variant conflict: {listing_id}")
            selected = candidate
            method = "reviewed"
            outcome = MappingOutcome.REVIEWED_MATCHED
            reviewed_count += 1
            decision_evidence = {
                "reviewed_by": override.reviewed_by,
                "reviewed_at": override.reviewed_at,
                "review_status": override.review_status.value,
                "review_evidence": override.evidence,
            }
        else:
            match = matcher.match(
                listing_record_from_offer(
                    listing_id=listing_id,
                    title=title,
                    category=category.value,
                    retailer=source_listing.retailer,
                    current_price_sgd=float(source_listing.total_price),
                    metadata=metadata,
                )
            )
            outcome = match.outcome
            outcome_candidates = match.candidate_product_ids
            method = match.method
            reason = match.reason
            probability = match.probability
            model_version = match.model_version
            decision_evidence = match.evidence
            if match.matched_product_id is not None:
                selected = products_by_id[match.matched_product_id]
            if outcome is MappingOutcome.AUTO_MATCHED:
                auto_count += 1
            elif outcome is MappingOutcome.AMBIGUOUS:
                ambiguous_count += 1
            elif outcome is MappingOutcome.HARD_CONFLICT:
                conflict_count += 1
            elif outcome is MappingOutcome.MANUAL_REVIEW:
                manual_review_count += 1
            elif outcome is MappingOutcome.MODEL_REJECTED:
                model_rejected_count += 1
            elif outcome is MappingOutcome.UNMATCHED:
                unmatched_count += 1
        if selected is None or method is None:
            decisions.append(
                MappingDecision(
                    listing_id=listing_id,
                    source_listing_id=source_listing_id,
                    title=title,
                    category=category.value,
                    outcome=outcome,
                    candidate_product_ids=outcome_candidates,
                    method=method,
                    reason=reason,
                    probability=probability,
                    model_version=model_version,
                    evidence=decision_evidence,
                )
            )
            continue

        listing = source_listing.model_copy(update={"product_id": selected.product_id})
        listings.append(listing)
        snapshots.append(snapshot)
        listing_provenance.append(provenance)
        methods[listing.listing_id] = method
        matched_by_category[selected.category.value] += 1
        if listing.stock_status == StockStatus.IN_STOCK:
            in_stock_by_category[selected.category.value] += 1
        readiness_accumulator.observe_listing(
            listing,
            category=selected.category,
            provenance=provenance,
        )
        decisions.append(
            MappingDecision(
                listing_id=listing_id,
                source_listing_id=source_listing_id,
                title=title,
                category=category.value,
                outcome=outcome,
                matched_product_id=selected.product_id,
                candidate_product_ids=(selected.product_id,),
                method=method,
                probability=probability,
                model_version=model_version,
                evidence=decision_evidence,
            )
        )

    missing_reviewed = sorted(set(reviewed) - seen_reviewed)
    if missing_reviewed:
        raise ValueError(f"reviewed mappings reference unknown listings: {missing_reviewed}")
    accounted_offers = auto_count + reviewed_count + review_rejected_count
    accounted_offers += unmatched_count + conflict_count + ambiguous_count
    accounted_offers += manual_review_count + model_rejected_count
    if offer_count != accounted_offers:
        raise AssertionError("every retailer offer must have exactly one mapping outcome")

    version_material = "|".join(
        (
            _sha256(product_path),
            _sha256(resolved_offer_path),
            _sha256(Path(reviewed_mapping_path).resolve())
            if reviewed_mapping_path is not None
            else "no-reviewed-mappings",
            _sha256(Path(review_evidence_path).resolve())
            if review_evidence_path is not None
            else "no-review-evidence",
            (
                entity_resolution_binding_sha256
                if entity_resolution_binding_sha256 is not None
                else entity_resolution_release_sha256(runtime.artifact_path)
                if runtime is not None
                else "no-entity-resolution-model"
            ),
            (
                _sha256(Path(entity_resolution_evaluation_path).resolve())
                if entity_resolution_evaluation_path is not None
                else "no-entity-resolution-evaluation"
            ),
        )
    )
    data_version = "processed-" + hashlib.sha256(version_material.encode()).hexdigest()[:16]
    product_counts = Counter(product.category.value for product in products)
    stats = ProcessedCatalogStats(
        product_count=len(products),
        offer_count=offer_count,
        matched_listing_count=len(listings),
        auto_matched_count=auto_count,
        reviewed_matched_count=reviewed_count,
        unmatched_offer_count=unmatched_count,
        rejected_conflict_count=conflict_count,
        ambiguous_exact_match_count=ambiguous_count,
        products_by_category=dict(sorted(product_counts.items())),
        matched_listings_by_category=dict(sorted(matched_by_category.items())),
        in_stock_listings_by_category=dict(sorted(in_stock_by_category.items())),
        known_in_stock_listing_count=sum(
            listing.stock_status == StockStatus.IN_STOCK for listing in listings
        ),
        data_version=data_version,
        review_rejected_count=review_rejected_count,
        manual_review_count=manual_review_count,
        model_rejected_count=model_rejected_count,
    )
    outcome_counts = Counter(decision.outcome.value for decision in decisions)
    readiness = readiness_accumulator.finish(
        data_version=data_version,
        offer_count=offer_count,
        mapping_outcomes=outcome_counts,
        entity_resolution_evaluation=entity_resolution_evaluation,
        entity_resolution_model_version=matcher.model_version,
        entity_resolution_model_production_authorized=bool(
            runtime is not None and runtime.production_authorized
        ),
    )
    return ProcessedCatalogData(
        products=tuple(sorted(products, key=lambda item: item.product_id)),
        listings=tuple(sorted(listings, key=lambda item: item.listing_id)),
        price_snapshots=tuple(sorted(snapshots, key=lambda item: item.snapshot_id)),
        stats=stats,
        review_evidence=review_evidence,
        match_method_by_listing=methods,
        listing_provenance=tuple(sorted(listing_provenance, key=lambda item: item.provenance_id)),
        mapping_decisions=tuple(sorted(decisions, key=lambda item: item.listing_id)),
        readiness=readiness,
    )


class InMemoryCatalogReader:
    """Read-only CatalogReader implementation backed by a processed snapshot."""

    def __init__(
        self,
        data: ProcessedCatalogData,
        *,
        benchmarks: Sequence[BenchmarkResult] = (),
    ) -> None:
        self.data = data
        self._products = {item.product_id: item for item in data.products}
        listings: dict[str, list[RetailerListing]] = defaultdict(list)
        for listing in data.listings:
            listings[listing.product_id].append(listing)
        self._listings = {
            key: tuple(sorted(value, key=lambda item: (item.total_price, item.listing_id)))
            for key, value in listings.items()
        }
        benchmark_groups: dict[str, list[BenchmarkResult]] = defaultdict(list)
        for benchmark in benchmarks:
            benchmark_groups[benchmark.product_id].append(benchmark)
        self._benchmarks = {
            key: tuple(sorted(value, key=lambda item: item.benchmark_id))
            for key, value in benchmark_groups.items()
        }
        review_groups: dict[str, list[ReviewNote]] = defaultdict(list)
        for evidence in data.review_evidence:
            review_groups[evidence.product_id].append(evidence)
        self._review_evidence = {
            key: tuple(sorted(value, key=lambda item: (item.aspect, item.evidence_id)))
            for key, value in review_groups.items()
        }

    def list_products(
        self,
        *,
        category: ComponentKind | None = None,
        brand: str | None = None,
        status: ProductStatus | None = ProductStatus.ACTIVE,
        offset: int = 0,
        limit: int = 100,
    ) -> list[MasterProduct]:
        if offset < 0 or not 1 <= limit <= 1000:
            raise ValueError("offset must be nonnegative and limit must be between 1 and 1000")
        items = [
            item
            for item in self.data.products
            if (category is None or item.category == category)
            and (brand is None or item.brand.casefold() == brand.casefold())
            and (status is None or item.status == status)
        ]
        return list(items[offset : offset + limit])

    def get_product(self, product_id: str) -> MasterProduct | None:
        return self._products.get(product_id)

    def list_listings(
        self,
        *,
        product_id: str | None = None,
        retailer: str | None = None,
        stock_status: StockStatus | None = None,
        limit: int = 100,
    ) -> list[RetailerListing]:
        if not 1 <= limit <= 1000:
            raise ValueError("limit must be between 1 and 1000")
        items = (
            list(self._listings.get(product_id, ()))
            if product_id is not None
            else list(self.data.listings)
        )
        return [
            item
            for item in items
            if (retailer is None or item.retailer.casefold() == retailer.casefold())
            and (stock_status is None or item.stock_status == stock_status)
        ][:limit]

    def list_benchmarks(
        self, product_id: str, *, workload: str | None = None
    ) -> list[BenchmarkResult]:
        return [
            item
            for item in self._benchmarks.get(product_id, ())
            if workload is None or item.workload.value == workload
        ]

    def list_review_evidence(self, product_id: str) -> list[ReviewNote]:
        return list(self._review_evidence.get(product_id, ()))

    def list_price_snapshots(self, listing_id: str) -> list[PriceSample]:
        return [item for item in self.data.price_snapshots if item.listing_id == listing_id]


def seed_processed_catalog(session: Session, data: ProcessedCatalogData) -> None:
    """Idempotently persist a processed snapshot using the canonical repository."""

    repository = CatalogRepository(session)
    for product in data.products:
        repository.upsert_product(product)
    for listing in data.listings:
        repository.upsert_listing(listing)
    for snapshot in data.price_snapshots:
        repository.upsert_price_snapshot(snapshot)
    for evidence in data.review_evidence:
        repository.upsert_review_evidence(evidence)
    for provenance in data.listing_provenance:
        repository.upsert_provenance(provenance)
