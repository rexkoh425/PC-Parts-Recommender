"""Memory-bounded processed-catalog analysis and transactional database import."""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from pc_build_recommender.domain import (
    MasterProduct,
    ComponentKind,
    PriceSample,
    RetailerListing,
    ReviewNote,
    StockState,
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

from .entity_matcher import CatalogEntityMatcher, listing_record_from_offer
from .er_gate import load_entity_resolution_evaluation
from .mapping_review import (
    MappingDecision,
    MappingOutcome,
    ReviewStatus,
    load_mapping_reviews,
)
from .processed import (
    DEFAULT_MAX_JSONL_LINE_BYTES,
    ProcessedCatalogStats,
    _category,
    _listing_source_provenance,
    _require_envelope,
    _resolve_offer_artifact_path,
    _sha256,
    _validate_offer_governance,
    iter_jsonl_objects,
    load_review_evidence,
)
from .readiness import (
    CatalogReadinessAccumulator,
    CatalogReadinessReport,
    ProductionCatalogPolicy,
    validate_production_readiness,
)
from .repository import CatalogRepository

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


@dataclass(frozen=True, slots=True)
class _ProductIdentity:
    product_id: str
    category: ComponentKind
    brand: str
    model: str
    canonical_name: str
    manufacturer_part_number: str | None
    gtin: str | None
    colour: str | None
    category_attributes: Mapping[str, Any]

    @classmethod
    def from_product(cls, product: MasterProduct) -> _ProductIdentity:
        return cls(
            product_id=product.product_id,
            category=product.category,
            brand=product.brand,
            model=product.model,
            canonical_name=product.canonical_name,
            manufacturer_part_number=product.manufacturer_part_number,
            gtin=product.gtin,
            colour=product.common_attributes.colour,
            category_attributes=product.category_attributes.model_dump(mode="json"),
        )


@dataclass(frozen=True, slots=True)
class StreamedCatalogImportResult:
    stats: ProcessedCatalogStats
    readiness: CatalogReadinessReport
    mapping_decisions: tuple[MappingDecision, ...]
    database_upserted: bool
    product_ids: tuple[str, ...]
    listing_ids: tuple[str, ...]
    review_evidence_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.stats.to_dict(),
            "database_upserted": self.database_upserted,
            "review_evidence_count": self.review_evidence_count,
            "readiness": self.readiness.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class ReviewTarget:
    listing_id: str
    source_listing_id: str
    title: str
    category: ComponentKind
    product_id: str | None = None


def _brand_prefix_matches(title: str, brand: str) -> bool:
    title_tokens = tokenize(title)
    brand_tokens = tokenize(brand)
    return bool(brand_tokens and title_tokens[: len(brand_tokens)] == brand_tokens)


def _numeric_conflict(title: str, product: _ProductIdentity) -> bool:
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
        attributes=product.category_attributes,
    )
    return bool(find_numeric_conflicts(listing, canonical))


def _colour_conflict(title: str, product: _ProductIdentity) -> bool:
    title_colours = _COLOUR_TOKENS.intersection(tokenize(title))
    if len(title_colours) > 1:
        return True
    if not title_colours or not product.colour:
        return False
    product_colours = _COLOUR_TOKENS.intersection(tokenize(product.colour))
    return bool(product_colours and title_colours.isdisjoint(product_colours))


def _identity_candidates(
    title: str,
    category: ComponentKind,
    products: Sequence[_ProductIdentity],
) -> tuple[_ProductIdentity, ...]:
    title_identifier = normalize_identifier(title)
    candidates = [
        product
        for product in products
        if product.category is category
        and _brand_prefix_matches(title, product.brand)
        and len(normalize_identifier(product.manufacturer_part_number)) >= 6
        and normalize_identifier(product.manufacturer_part_number) in title_identifier
    ]
    return tuple(sorted(candidates, key=lambda item: item.product_id))


def _data_version(
    product_path: Path,
    offer_path: Path,
    reviewed_mapping_path: str | Path | None,
    review_evidence_path: str | Path | None,
    entity_resolution_evaluation_path: str | Path | None,
    runtime: EntityResolutionRuntime | None,
    entity_resolution_binding_sha256: str | None,
) -> str:
    version_material = "|".join(
        (
            _sha256(product_path),
            _sha256(offer_path),
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
    return "processed-" + hashlib.sha256(version_material.encode()).hexdigest()[:16]


def stream_processed_catalog(
    buildcores_path: str | Path,
    offer_path: str | Path | None = None,
    *,
    dynacore_path: str | Path | None = None,
    session: Session | None = None,
    reviewed_mapping_path: str | Path | None = None,
    review_evidence_path: str | Path | None = None,
    entity_resolution_evaluation_path: str | Path | None = None,
    entity_resolution_model_path: str | Path | None = None,
    entity_resolution_runtime: EntityResolutionRuntime | None = None,
    entity_resolution_policy: EntityResolutionPolicy | None = None,
    entity_resolution_binding_sha256: str | None = None,
    allow_unpromoted_entity_resolution: bool = False,
    batch_size: int = 250,
    max_line_bytes: int = DEFAULT_MAX_JSONL_LINE_BYTES,
    require_production_ready: bool = False,
    production_policy: ProductionCatalogPolicy | None = None,
) -> StreamedCatalogImportResult:
    """Analyze, and optionally upsert, products plus governed retailer offers.

    The caller owns the transaction. If the production gate fails, its exception
    propagates so :func:`session_scope` rolls back every upsert in this import.
    """

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    product_path = Path(buildcores_path).resolve()
    resolved_offer_path = _resolve_offer_artifact_path(
        offer_path,
        dynacore_path=dynacore_path,
    )
    if not product_path.is_file():
        raise FileNotFoundError(f"BuildCores processed catalog not found: {product_path}")
    if not resolved_offer_path.is_file():
        raise FileNotFoundError(f"processed retailer offers not found: {resolved_offer_path}")

    repository = CatalogRepository(session) if session is not None else None
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
    active_policy = production_policy or ProductionCatalogPolicy()
    require_er_release = require_production_ready and (
        active_policy.require_promoted_entity_resolution_model
    )
    if require_er_release and runtime is not None and entity_resolution_policy is None:
        raise ValueError("production entity resolution requires an exact serving policy")
    if (
        require_er_release
        and runtime is not None
        and (
            entity_resolution_binding_sha256 is None
            or len(entity_resolution_binding_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in entity_resolution_binding_sha256
            )
        )
    ):
        raise ValueError("production entity resolution requires an exact release binding")
    if require_er_release and runtime is not None:
        assert entity_resolution_policy is not None
        runtime = runtime.authorize_for_production(
            entity_resolution_evaluation,
            minimum_precision=active_policy.minimum_er_precision,
            minimum_labelled_pairs=active_policy.minimum_er_labelled_pairs,
            minimum_auto_matches=entity_resolution_policy.minimum_auto_matches,
            minimum_recall=entity_resolution_policy.minimum_recall,
            minimum_f1=entity_resolution_policy.minimum_f1,
        )
    readiness_accumulator = CatalogReadinessAccumulator()
    products_by_id: dict[str, _ProductIdentity] = {}
    product_counts: Counter[str] = Counter()

    for row_number, envelope in enumerate(
        iter_jsonl_objects(product_path, max_line_bytes=max_line_bytes), start=1
    ):
        data = _require_envelope(
            envelope,
            record_type="canonical_product",
            path=product_path,
        )
        product = MasterProduct.model_validate(data)
        if product.product_id in products_by_id:
            raise ValueError(
                f"processed BuildCores catalog contains duplicate product ID: {product.product_id}"
            )
        identity = _ProductIdentity.from_product(product)
        products_by_id[identity.product_id] = identity
        product_counts[identity.category.value] += 1
        readiness_accumulator.observe_product(product)
        if repository is not None and session is not None:
            repository.upsert_product(product)
            if row_number % batch_size == 0:
                session.flush()
                session.expunge_all()

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
                **product.category_attributes,
                "colour": product.colour,
            },
        )
        for product in products_by_id.values()
    )
    review_evidence: tuple[ReviewNote, ...] = load_review_evidence(
        review_evidence_path,
        known_product_ids=products_by_id,
        max_line_bytes=max_line_bytes,
    )
    if repository is not None and session is not None:
        for evidence_index, evidence in enumerate(review_evidence, start=1):
            repository.upsert_review_evidence(evidence)
            if evidence_index % batch_size == 0:
                session.flush()
                session.expunge_all()
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

    reviewed = load_mapping_reviews(reviewed_mapping_path)
    seen_reviewed: set[str] = set()
    seen_listing_ids: set[str] = set()
    matched_listing_ids: set[str] = set()
    seen_snapshot_ids: set[str] = set()
    decisions: list[MappingDecision] = []
    matched_by_category: Counter[str] = Counter()
    in_stock_by_category: Counter[str] = Counter()
    outcome_counts: Counter[str] = Counter()
    offer_count = 0

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
        category = _category(metadata.get("category"))
        title = source_listing.title
        selected: _ProductIdentity | None = None
        method: str | None = None
        candidates: tuple[_ProductIdentity, ...] = ()
        reason: str | None = None
        probability: float | None = None
        model_version: str | None = None
        decision_evidence: Mapping[str, Any] = {}

        override = reviewed.get(source_listing.listing_id)
        if override is not None:
            seen_reviewed.add(source_listing.listing_id)
            if override.review_status is ReviewStatus.REJECTED:
                outcome = MappingOutcome.REVIEW_REJECTED
                reason = override.evidence
                decision_evidence = {
                    "reviewed_by": override.reviewed_by,
                    "reviewed_at": override.reviewed_at,
                    "review_status": override.review_status.value,
                }
            else:
                if override.product_id is None:
                    raise AssertionError("approved mapping is missing product_id")
                selected = products_by_id.get(override.product_id)
                if selected is None:
                    raise ValueError(
                        f"reviewed mapping references unknown product: {override.product_id}"
                    )
                if selected.category is not category:
                    raise ValueError(
                        f"reviewed mapping category conflict: {source_listing.listing_id}"
                    )
                if _numeric_conflict(title, selected) or _colour_conflict(title, selected):
                    raise ValueError(
                        f"reviewed mapping has a hard variant conflict: {source_listing.listing_id}"
                    )
                outcome = MappingOutcome.REVIEWED_MATCHED
                method = "reviewed"
                candidates = (selected,)
                decision_evidence = {
                    "reviewed_by": override.reviewed_by,
                    "reviewed_at": override.reviewed_at,
                    "review_status": override.review_status.value,
                    "review_evidence": override.evidence,
                }
        else:
            match = matcher.match(
                listing_record_from_offer(
                    listing_id=source_listing.listing_id,
                    title=title,
                    category=category.value,
                    retailer=source_listing.retailer,
                    current_price_sgd=float(source_listing.total_price),
                    metadata=metadata,
                )
            )
            outcome = match.outcome
            method = match.method
            reason = match.reason
            probability = match.probability
            model_version = match.model_version
            decision_evidence = match.evidence
            candidates = tuple(
                products_by_id[product_id]
                for product_id in match.candidate_product_ids
                if product_id in products_by_id
            )
            if match.matched_product_id is not None:
                selected = products_by_id[match.matched_product_id]

        listing: RetailerListing | None = None
        if selected is not None and method is not None:
            listing = source_listing.model_copy(update={"product_id": selected.product_id})
            matched_listing_ids.add(listing.listing_id)
            provenance = provenance.model_copy(update={"listing_id": listing.listing_id})
            matched_by_category[selected.category.value] += 1
            if listing.stock_status is StockState.IN_STOCK:
                in_stock_by_category[selected.category.value] += 1
            readiness_accumulator.observe_listing(
                listing,
                category=selected.category,
                provenance=provenance,
            )
            if repository is not None and session is not None:
                repository.upsert_listing(listing)
                repository.upsert_price_snapshot(snapshot)
                repository.upsert_provenance(provenance)
                if offer_count % batch_size == 0:
                    session.flush()
                    session.expunge_all()

        decision = MappingDecision(
            listing_id=source_listing.listing_id,
            source_listing_id=source_listing.source_listing_id,
            title=title,
            category=category.value,
            outcome=outcome,
            matched_product_id=selected.product_id if selected is not None else None,
            candidate_product_ids=tuple(item.product_id for item in candidates),
            method=method,
            reason=reason,
            probability=probability,
            model_version=model_version,
            evidence=decision_evidence,
        )
        decisions.append(decision)
        outcome_counts[outcome.value] += 1

    missing_reviewed = sorted(set(reviewed) - seen_reviewed)
    if missing_reviewed:
        raise ValueError(f"reviewed mappings reference unknown listings: {missing_reviewed}")
    if sum(outcome_counts.values()) != offer_count:
        raise AssertionError("every retailer offer must have exactly one mapping outcome")
    if repository is not None and session is not None:
        session.flush()
        session.expunge_all()

    data_version = _data_version(
        product_path,
        resolved_offer_path,
        reviewed_mapping_path,
        review_evidence_path,
        entity_resolution_evaluation_path,
        runtime,
        entity_resolution_binding_sha256,
    )
    auto_count = outcome_counts[MappingOutcome.AUTO_MATCHED.value]
    reviewed_count = outcome_counts[MappingOutcome.REVIEWED_MATCHED.value]
    stats = ProcessedCatalogStats(
        product_count=len(products_by_id),
        offer_count=offer_count,
        matched_listing_count=auto_count + reviewed_count,
        auto_matched_count=auto_count,
        reviewed_matched_count=reviewed_count,
        unmatched_offer_count=outcome_counts[MappingOutcome.UNMATCHED.value],
        rejected_conflict_count=outcome_counts[MappingOutcome.HARD_CONFLICT.value],
        ambiguous_exact_match_count=outcome_counts[MappingOutcome.AMBIGUOUS.value],
        products_by_category=dict(sorted(product_counts.items())),
        matched_listings_by_category=dict(sorted(matched_by_category.items())),
        in_stock_listings_by_category=dict(sorted(in_stock_by_category.items())),
        known_in_stock_listing_count=sum(in_stock_by_category.values()),
        data_version=data_version,
        review_rejected_count=outcome_counts[MappingOutcome.REVIEW_REJECTED.value],
        manual_review_count=outcome_counts[MappingOutcome.MANUAL_REVIEW.value],
        model_rejected_count=outcome_counts[MappingOutcome.MODEL_REJECTED.value],
    )
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
    if require_production_ready:
        validate_production_readiness(readiness, production_policy)
    return StreamedCatalogImportResult(
        stats=stats,
        readiness=readiness,
        mapping_decisions=tuple(sorted(decisions, key=lambda item: item.listing_id)),
        database_upserted=session is not None,
        product_ids=tuple(sorted(products_by_id)),
        listing_ids=tuple(sorted(matched_listing_ids)),
        review_evidence_count=len(review_evidence),
    )


def validate_review_target(
    offer_path: str | Path | None = None,
    *,
    dynacore_path: str | Path | None = None,
    listing_id: str,
    buildcores_path: str | Path | None = None,
    product_id: str | None = None,
    max_line_bytes: int = DEFAULT_MAX_JSONL_LINE_BYTES,
) -> ReviewTarget:
    """Verify an offer review target; approvals recheck category and variant conflicts."""

    clean_listing_id = listing_id.strip()
    if not clean_listing_id:
        raise ValueError("listing_id is required")
    if (buildcores_path is None) != (product_id is None):
        raise ValueError("approval validation requires both buildcores_path and product_id")

    selected: _ProductIdentity | None = None
    clean_product_id = product_id.strip() if product_id else None
    if buildcores_path is not None and clean_product_id is not None:
        product_path = Path(buildcores_path).resolve()
        for envelope in iter_jsonl_objects(product_path, max_line_bytes=max_line_bytes):
            data = _require_envelope(
                envelope,
                record_type="canonical_product",
                path=product_path,
            )
            product = MasterProduct.model_validate(data)
            if product.product_id == clean_product_id:
                selected = _ProductIdentity.from_product(product)
                break
        if selected is None:
            raise ValueError(f"reviewed mapping references unknown product: {clean_product_id}")

    resolved_offer_path = _resolve_offer_artifact_path(
        offer_path,
        dynacore_path=dynacore_path,
    )
    for envelope in iter_jsonl_objects(resolved_offer_path, max_line_bytes=max_line_bytes):
        data = _require_envelope(
            envelope,
            record_type="retailer_listing",
            path=resolved_offer_path,
        )
        raw_listing = data.get("listing")
        metadata = envelope.get("normalisation_metadata")
        if not isinstance(raw_listing, Mapping) or not isinstance(metadata, Mapping):
            raise ValueError("retailer record requires listing and normalisation metadata")
        if str(raw_listing.get("listing_id", "")).strip() != clean_listing_id:
            continue
        listing = RetailerListing.model_validate(raw_listing)
        _listing_source_provenance(envelope, listing)
        category = _category(metadata.get("category"))
        if selected is not None:
            if selected.category is not category:
                raise ValueError(f"reviewed mapping category conflict: {clean_listing_id}")
            if _numeric_conflict(listing.title, selected) or _colour_conflict(
                listing.title, selected
            ):
                raise ValueError(
                    f"reviewed mapping has a hard variant conflict: {clean_listing_id}"
                )
        return ReviewTarget(
            listing_id=listing.listing_id,
            source_listing_id=listing.source_listing_id,
            title=listing.title,
            category=category,
            product_id=selected.product_id if selected is not None else None,
        )
    raise ValueError(f"reviewed mapping references unknown listing: {clean_listing_id}")
