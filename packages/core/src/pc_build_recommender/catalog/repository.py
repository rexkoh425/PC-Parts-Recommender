"""Catalog repository that keeps SQLAlchemy records behind domain contracts."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import Select, or_, select
from sqlalchemy.orm import Session

from pc_build_recommender.domain import (
    BenchmarkResult,
    BuildComponentSelection,
    BuildRecommendation,
    CanonicalProduct,
    CompatibilityRule,
    ComponentCategory,
    InteractionRecord,
    PriceSample,
    ProductStatus,
    RetailerListing,
    ReviewNote,
    SearchQuery,
    SourceProvenance,
    StockStatus,
)

from .orm import (
    BenchmarkResultRecord,
    BuildComponentRecord,
    CanonicalProductRecord,
    CompatibilityRuleRecord,
    GeneratedBuildRecord,
    InteractionEventRecord,
    PriceSnapshotRecord,
    RetailerListingRecord,
    ReviewEvidenceRecord,
    SearchQueryRecord,
    SourceProvenanceRecord,
)


def _json(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return value


def _as_utc(value: datetime | None) -> datetime | None:
    """Restore UTC on SQLite timestamps, whose driver drops timezone metadata."""

    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value


class CatalogRepository:
    """Synchronous repository used by ingestion jobs and FastAPI session adapters.

    Methods flush but do not commit. A request or pipeline owns the transaction,
    normally through :func:`pc_build_recommender.catalog.session_scope`.
    """

    def __init__(self, session: Session) -> None:
        self.session = session

    @staticmethod
    def _product_values(product: CanonicalProduct) -> dict[str, Any]:
        return {
            "category": product.category.value,
            "brand": product.brand,
            "model": product.model,
            "manufacturer_part_number": product.manufacturer_part_number,
            "gtin": product.gtin,
            "canonical_name": product.canonical_name,
            "release_date": product.release_date,
            "status": product.status.value,
            "common_attributes": product.common_attributes.model_dump(mode="json"),
            "category_attributes": product.category_attributes.model_dump(mode="json"),
            "source_confidence": product.source_confidence,
            "search_document": " ".join(
                part
                for part in (
                    product.category.value,
                    product.brand,
                    product.model,
                    product.manufacturer_part_number,
                    product.canonical_name,
                )
                if part
            ),
            # A general catalogue mutation is not an embedding-index import.
            # Invalidate the release guard until the versioned vector importer
            # writes the matching search document and content hash atomically.
            "search_document_hash": None,
            "created_at": product.created_at,
            "updated_at": product.updated_at,
        }

    @staticmethod
    def _to_provenance(record: SourceProvenanceRecord) -> SourceProvenance:
        return SourceProvenance.model_validate(
            {
                "provenance_id": record.provenance_id,
                "product_id": record.product_id,
                "listing_id": record.listing_id,
                "source_name": record.source_name,
                "source_url": record.source_url,
                "source_type": record.source_type,
                "retrieved_at": _as_utc(record.retrieved_at),
                "raw_content_hash": record.raw_content_hash,
                "parser_version": record.parser_version,
                "licence_or_access_note": record.licence_or_access_note,
                "last_verified_at": _as_utc(record.last_verified_at),
                "extraction_confidence": record.extraction_confidence,
            }
        )

    @classmethod
    def _to_product(cls, record: CanonicalProductRecord) -> CanonicalProduct:
        return CanonicalProduct.model_validate(
            {
                "product_id": record.product_id,
                "category": record.category,
                "brand": record.brand,
                "model": record.model,
                "manufacturer_part_number": record.manufacturer_part_number,
                "gtin": record.gtin,
                "canonical_name": record.canonical_name,
                "release_date": record.release_date,
                "status": record.status,
                "common_attributes": record.common_attributes or {},
                "category_attributes": record.category_attributes or {},
                "source_confidence": record.source_confidence,
                "provenance": [cls._to_provenance(item) for item in record.provenance],
                "created_at": _as_utc(record.created_at),
                "updated_at": _as_utc(record.updated_at),
            }
        )

    def add_product(self, product: CanonicalProduct) -> CanonicalProduct:
        if self.session.get(CanonicalProductRecord, product.product_id) is not None:
            raise ValueError(f"product already exists: {product.product_id}")
        record = CanonicalProductRecord(
            product_id=product.product_id,
            **self._product_values(product),
        )
        self.session.add(record)
        self.session.flush()
        for provenance in product.provenance:
            self.upsert_provenance(
                provenance.model_copy(update={"product_id": product.product_id})
            )
        self.session.flush()
        return self._to_product(record)

    def upsert_product(self, product: CanonicalProduct) -> CanonicalProduct:
        record = self.session.get(CanonicalProductRecord, product.product_id)
        if record is None:
            return self.add_product(product)
        for key, value in self._product_values(product).items():
            setattr(record, key, value)
        for provenance in product.provenance:
            self.upsert_provenance(
                provenance.model_copy(update={"product_id": product.product_id})
            )
        self.session.flush()
        return self._to_product(record)

    def get_product(self, product_id: str) -> CanonicalProduct | None:
        record = self.session.get(CanonicalProductRecord, product_id)
        return None if record is None else self._to_product(record)

    def require_product(self, product_id: str) -> CanonicalProduct:
        product = self.get_product(product_id)
        if product is None:
            raise KeyError(f"product not found: {product_id}")
        return product

    def delete_product(self, product_id: str) -> bool:
        record = self.session.get(CanonicalProductRecord, product_id)
        if record is None:
            return False
        self.session.delete(record)
        self.session.flush()
        return True

    @staticmethod
    def _apply_product_filters(
        statement: Select[tuple[CanonicalProductRecord]],
        *,
        category: ComponentCategory | None,
        brand: str | None,
        status: ProductStatus | None,
    ) -> Select[tuple[CanonicalProductRecord]]:
        if category is not None:
            statement = statement.where(CanonicalProductRecord.category == category.value)
        if brand is not None:
            statement = statement.where(CanonicalProductRecord.brand.ilike(brand))
        if status is not None:
            statement = statement.where(CanonicalProductRecord.status == status.value)
        return statement

    def list_products(
        self,
        *,
        category: ComponentCategory | None = None,
        brand: str | None = None,
        status: ProductStatus | None = ProductStatus.ACTIVE,
        offset: int = 0,
        limit: int = 100,
    ) -> list[CanonicalProduct]:
        if offset < 0 or not 1 <= limit <= 1000:
            raise ValueError("offset must be nonnegative and limit must be between 1 and 1000")
        statement = self._apply_product_filters(
            select(CanonicalProductRecord), category=category, brand=brand, status=status
        )
        statement = statement.order_by(
            CanonicalProductRecord.canonical_name, CanonicalProductRecord.product_id
        ).offset(offset).limit(limit)
        return [self._to_product(record) for record in self.session.scalars(statement)]

    def search_products(
        self,
        query: str,
        *,
        category: ComponentCategory | None = None,
        brand: str | None = None,
        status: ProductStatus | None = ProductStatus.ACTIVE,
        in_stock_only: bool = False,
        max_total_price: Decimal | None = None,
        offset: int = 0,
        limit: int = 50,
    ) -> list[CanonicalProduct]:
        """Deterministic lexical catalog search used before the BM25 index is built."""

        if offset < 0 or not 1 <= limit <= 1000:
            raise ValueError("offset must be nonnegative and limit must be between 1 and 1000")
        if max_total_price is not None and max_total_price < 0:
            raise ValueError("max_total_price cannot be negative")

        needs_listing = bool(query.strip()) or in_stock_only or max_total_price is not None
        statement = select(CanonicalProductRecord)
        if needs_listing:
            statement = statement.join(
                RetailerListingRecord,
                RetailerListingRecord.product_id == CanonicalProductRecord.product_id,
                isouter=not (in_stock_only or max_total_price is not None),
            )

        terms = [term for term in query.split() if term]
        for term in terms:
            pattern = f"%{term}%"
            statement = statement.where(
                or_(
                    CanonicalProductRecord.canonical_name.ilike(pattern),
                    CanonicalProductRecord.brand.ilike(pattern),
                    CanonicalProductRecord.model.ilike(pattern),
                    CanonicalProductRecord.manufacturer_part_number.ilike(pattern),
                    CanonicalProductRecord.search_document.ilike(pattern),
                    RetailerListingRecord.title.ilike(pattern),
                )
            )
        statement = self._apply_product_filters(
            statement, category=category, brand=brand, status=status
        )
        if in_stock_only:
            statement = statement.where(
                RetailerListingRecord.stock_status == StockStatus.IN_STOCK.value
            )
        if max_total_price is not None:
            statement = statement.where(
                RetailerListingRecord.base_price + RetailerListingRecord.shipping_price
                <= max_total_price
            )
        statement = statement.distinct().order_by(
            CanonicalProductRecord.canonical_name, CanonicalProductRecord.product_id
        ).offset(offset).limit(limit)
        return [self._to_product(record) for record in self.session.scalars(statement).unique()]

    @staticmethod
    def _listing_values(listing: RetailerListing) -> dict[str, Any]:
        return {
            "product_id": listing.product_id,
            "retailer": listing.retailer,
            "source_listing_id": listing.source_listing_id,
            "title": listing.title,
            "condition": listing.condition.value,
            "currency": listing.currency,
            "base_price": listing.base_price,
            "shipping_price": listing.shipping_price,
            "stock_status": listing.stock_status.value,
            "seller_name": listing.seller_name,
            "listing_url": listing.listing_url,
            "first_seen_at": listing.first_seen_at,
            "last_seen_at": listing.last_seen_at,
        }

    @staticmethod
    def _to_listing(record: RetailerListingRecord) -> RetailerListing:
        return RetailerListing.model_validate(
            {
                "listing_id": record.listing_id,
                "product_id": record.product_id,
                "retailer": record.retailer,
                "source_listing_id": record.source_listing_id,
                "title": record.title,
                "condition": record.condition,
                "currency": record.currency,
                "base_price": record.base_price,
                "shipping_price": record.shipping_price,
                "stock_status": record.stock_status,
                "seller_name": record.seller_name,
                "listing_url": record.listing_url,
                "first_seen_at": _as_utc(record.first_seen_at),
                "last_seen_at": _as_utc(record.last_seen_at),
            }
        )

    def add_listing(self, listing: RetailerListing) -> RetailerListing:
        if self.session.get(RetailerListingRecord, listing.listing_id) is not None:
            raise ValueError(f"listing already exists: {listing.listing_id}")
        record = RetailerListingRecord(
            listing_id=listing.listing_id, **self._listing_values(listing)
        )
        self.session.add(record)
        self.session.flush()
        return self._to_listing(record)

    def upsert_listing(self, listing: RetailerListing) -> RetailerListing:
        record = self.session.get(RetailerListingRecord, listing.listing_id)
        if record is None:
            record = self.session.scalar(
                select(RetailerListingRecord).where(
                    RetailerListingRecord.retailer == listing.retailer,
                    RetailerListingRecord.source_listing_id == listing.source_listing_id,
                )
            )
        if record is None:
            return self.add_listing(listing)
        for key, value in self._listing_values(listing).items():
            setattr(record, key, value)
        self.session.flush()
        return self._to_listing(record)

    def get_listing(self, listing_id: str) -> RetailerListing | None:
        record = self.session.get(RetailerListingRecord, listing_id)
        return None if record is None else self._to_listing(record)

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
        statement = select(RetailerListingRecord)
        if product_id is not None:
            statement = statement.where(RetailerListingRecord.product_id == product_id)
        if retailer is not None:
            statement = statement.where(RetailerListingRecord.retailer.ilike(retailer))
        if stock_status is not None:
            statement = statement.where(RetailerListingRecord.stock_status == stock_status.value)
        statement = statement.order_by(
            RetailerListingRecord.base_price + RetailerListingRecord.shipping_price,
            RetailerListingRecord.listing_id,
        ).limit(limit)
        return [self._to_listing(record) for record in self.session.scalars(statement)]

    def cheapest_in_stock_listing(self, product_id: str) -> RetailerListing | None:
        listings = self.list_listings(
            product_id=product_id, stock_status=StockStatus.IN_STOCK, limit=1
        )
        return listings[0] if listings else None

    def delete_listing(self, listing_id: str) -> bool:
        record = self.session.get(RetailerListingRecord, listing_id)
        if record is None:
            return False
        self.session.delete(record)
        self.session.flush()
        return True

    @staticmethod
    def _to_price(record: PriceSnapshotRecord) -> PriceSample:
        return PriceSample.model_validate(
            {
                "snapshot_id": record.snapshot_id,
                "listing_id": record.listing_id,
                "observed_at": _as_utc(record.observed_at),
                "base_price": record.base_price,
                "shipping_price": record.shipping_price,
                "stock_status": record.stock_status,
                "promotion_text": record.promotion_text,
            }
        )

    def upsert_price_snapshot(self, snapshot: PriceSample) -> PriceSample:
        record = self.session.get(PriceSnapshotRecord, snapshot.snapshot_id)
        if record is None:
            record = self.session.scalar(
                select(PriceSnapshotRecord).where(
                    PriceSnapshotRecord.listing_id == snapshot.listing_id,
                    PriceSnapshotRecord.observed_at == snapshot.observed_at,
                )
            )
        values = snapshot.model_dump(exclude={"snapshot_id"})
        values["stock_status"] = snapshot.stock_status.value
        if record is None:
            record = PriceSnapshotRecord(snapshot_id=snapshot.snapshot_id, **values)
            self.session.add(record)
        else:
            for key, value in values.items():
                setattr(record, key, value)
        self.session.flush()
        return self._to_price(record)

    add_price_snapshot = upsert_price_snapshot

    def list_price_snapshots(self, listing_id: str, *, limit: int = 1000) -> list[PriceSample]:
        statement = (
            select(PriceSnapshotRecord)
            .where(PriceSnapshotRecord.listing_id == listing_id)
            .order_by(PriceSnapshotRecord.observed_at, PriceSnapshotRecord.snapshot_id)
            .limit(limit)
        )
        return [self._to_price(record) for record in self.session.scalars(statement)]

    @staticmethod
    def _to_benchmark(record: BenchmarkResultRecord) -> BenchmarkResult:
        values = {
            column: getattr(record, column)
            for column in (
                "benchmark_id",
                "product_id",
                "workload",
                "benchmark_name",
                "benchmark_version",
                "score",
                "unit",
                "higher_is_better",
                "resolution",
                "preset",
                "operating_system",
                "driver_version",
                "source_url",
                "observed_at",
            )
        }
        values["observed_at"] = _as_utc(record.observed_at)
        return BenchmarkResult.model_validate(values)

    def upsert_benchmark(self, benchmark: BenchmarkResult) -> BenchmarkResult:
        record = self.session.get(BenchmarkResultRecord, benchmark.benchmark_id)
        values = benchmark.model_dump(exclude={"benchmark_id"})
        values["workload"] = benchmark.workload.value
        if record is None:
            record = BenchmarkResultRecord(benchmark_id=benchmark.benchmark_id, **values)
            self.session.add(record)
        else:
            for key, value in values.items():
                setattr(record, key, value)
        self.session.flush()
        return self._to_benchmark(record)

    add_benchmark = upsert_benchmark

    def list_benchmarks(
        self, product_id: str, *, workload: str | None = None
    ) -> list[BenchmarkResult]:
        statement = select(BenchmarkResultRecord).where(
            BenchmarkResultRecord.product_id == product_id
        )
        if workload is not None:
            statement = statement.where(BenchmarkResultRecord.workload == workload)
        statement = statement.order_by(
            BenchmarkResultRecord.workload,
            BenchmarkResultRecord.benchmark_name,
            BenchmarkResultRecord.observed_at,
        )
        return [self._to_benchmark(record) for record in self.session.scalars(statement)]

    def upsert_provenance(self, provenance: SourceProvenance) -> SourceProvenance:
        record = self.session.get(SourceProvenanceRecord, provenance.provenance_id)
        values = provenance.model_dump(exclude={"provenance_id"})
        values["source_type"] = provenance.source_type.value
        if record is None:
            record = SourceProvenanceRecord(
                provenance_id=provenance.provenance_id, **values
            )
            self.session.add(record)
        else:
            for key, value in values.items():
                setattr(record, key, value)
        self.session.flush()
        return self._to_provenance(record)

    add_provenance = upsert_provenance

    def list_provenance(
        self, *, product_id: str | None = None, listing_id: str | None = None
    ) -> list[SourceProvenance]:
        if product_id is None and listing_id is None:
            raise ValueError("product_id or listing_id is required")
        statement = select(SourceProvenanceRecord)
        if product_id is not None:
            statement = statement.where(SourceProvenanceRecord.product_id == product_id)
        if listing_id is not None:
            statement = statement.where(SourceProvenanceRecord.listing_id == listing_id)
        statement = statement.order_by(
            SourceProvenanceRecord.source_name, SourceProvenanceRecord.provenance_id
        )
        return [self._to_provenance(record) for record in self.session.scalars(statement)]

    def upsert_compatibility_rule(self, rule: CompatibilityRule) -> CompatibilityRule:
        record = self.session.get(CompatibilityRuleRecord, rule.rule_id)
        values = rule.model_dump(exclude={"rule_id"})
        values.update(
            left_category=rule.left_category.value,
            right_category=rule.right_category.value,
            severity=rule.severity.value,
        )
        if record is None:
            record = CompatibilityRuleRecord(rule_id=rule.rule_id, **values)
            self.session.add(record)
        else:
            for key, value in values.items():
                setattr(record, key, value)
        self.session.flush()
        return CompatibilityRule.model_validate(
            {"rule_id": record.rule_id, **values}
        )

    def list_compatibility_rules(
        self, *, rule_version: str | None = None
    ) -> list[CompatibilityRule]:
        statement = select(CompatibilityRuleRecord)
        if rule_version is not None:
            statement = statement.where(CompatibilityRuleRecord.rule_version == rule_version)
        statement = statement.order_by(
            CompatibilityRuleRecord.rule_version, CompatibilityRuleRecord.rule_id
        )
        return [
            CompatibilityRule.model_validate(
                {
                    column: getattr(record, column)
                    for column in (
                        "rule_id",
                        "rule_version",
                        "left_category",
                        "right_category",
                        "rule_type",
                        "severity",
                        "required_fields",
                        "message_template",
                        "evidence_source",
                        "effective_from",
                    )
                }
            )
            for record in self.session.scalars(statement)
        ]

    def upsert_review_evidence(self, evidence: ReviewNote) -> ReviewNote:
        record = self.session.get(ReviewEvidenceRecord, evidence.evidence_id)
        values = evidence.model_dump(exclude={"evidence_id"})
        if record is None:
            record = ReviewEvidenceRecord(evidence_id=evidence.evidence_id, **values)
            self.session.add(record)
        else:
            for key, value in values.items():
                setattr(record, key, value)
        self.session.flush()
        return ReviewNote.model_validate({"evidence_id": record.evidence_id, **values})

    def list_review_evidence(self, product_id: str) -> list[ReviewNote]:
        statement = (
            select(ReviewEvidenceRecord)
            .where(ReviewEvidenceRecord.product_id == product_id)
            .order_by(ReviewEvidenceRecord.aspect, ReviewEvidenceRecord.evidence_id)
        )
        return [
            ReviewNote.model_validate(
                {
                    column: getattr(record, column)
                    for column in (
                        "evidence_id",
                        "product_id",
                        "aspect",
                        "sentiment",
                        "evidence_text",
                        "source_url",
                        "published_at",
                        "confidence",
                    )
                }
            )
            for record in self.session.scalars(statement)
        ]

    def save_search_query(self, query: SearchQuery) -> SearchQuery:
        record = self.session.get(SearchQueryRecord, query.query_id)
        values = query.model_dump(exclude={"query_id"})
        if record is None:
            record = SearchQueryRecord(query_id=query.query_id, **values)
            self.session.add(record)
        else:
            for key, value in values.items():
                setattr(record, key, value)
        self.session.flush()
        return query

    def save_build(
        self,
        *,
        query_id: str,
        build: BuildRecommendation,
        optimizer_status: str,
        rule_version: str,
        model_version: str,
        data_version: str,
    ) -> BuildRecommendation:
        record = self.session.get(GeneratedBuildRecord, build.build_id)
        values = {
            "query_id": query_id,
            "profile": build.profile.value,
            "total_price": build.total_price_sgd,
            "overall_score": build.overall_score,
            "workload_scores": {key.value: value for key, value in build.workload_scores.items()},
            "compatibility_status": build.compatibility_status.value,
            "compatibility_checks": [_json(item) for item in build.compatibility_checks],
            "estimated_power_watts": build.estimated_power_watts,
            "warnings": build.warnings,
            "explanation": build.explanation,
            "alternatives": [_json(item) for item in build.alternatives],
            "optimizer_status": optimizer_status,
            "rule_version": rule_version,
            "model_version": model_version,
            "data_version": data_version,
        }
        if record is None:
            record = GeneratedBuildRecord(build_id=build.build_id, **values)
            self.session.add(record)
        else:
            for key, value in values.items():
                setattr(record, key, value)
            record.components.clear()
        self.session.flush()
        for component in build.components:
            record.components.append(
                BuildComponentRecord(
                    build_id=build.build_id,
                    category=component.category.value,
                    product_id=component.product_id,
                    listing_id=component.listing_id,
                    canonical_name=component.canonical_name,
                    price_sgd=component.price_sgd,
                    component_score=component.component_score,
                    selection_reason=component.selection_reason,
                )
            )
        self.session.flush()
        return self.get_build(build.build_id) or build

    def get_build(self, build_id: str) -> BuildRecommendation | None:
        record = self.session.get(GeneratedBuildRecord, build_id)
        if record is None:
            return None
        components: Sequence[BuildComponentRecord] = sorted(
            record.components, key=lambda item: item.category
        )
        return BuildRecommendation.model_validate(
            {
                "build_id": record.build_id,
                "profile": record.profile,
                "total_price_sgd": record.total_price,
                "overall_score": record.overall_score,
                "components": [
                    BuildComponentSelection.model_validate(
                        {
                            "category": component.category,
                            "product_id": component.product_id,
                            "listing_id": component.listing_id,
                            "canonical_name": component.canonical_name,
                            "price_sgd": component.price_sgd,
                            "component_score": component.component_score,
                            "selection_reason": component.selection_reason,
                        }
                    )
                    for component in components
                ],
                "workload_scores": record.workload_scores or {},
                "compatibility_status": record.compatibility_status,
                "compatibility_checks": record.compatibility_checks or [],
                "estimated_power_watts": record.estimated_power_watts,
                "warnings": record.warnings or [],
                "explanation": record.explanation or [],
                "alternatives": record.alternatives or [],
            }
        )

    def add_interaction(self, event: InteractionRecord) -> InteractionRecord:
        if self.session.get(InteractionEventRecord, event.event_id) is not None:
            raise ValueError(f"interaction already exists: {event.event_id}")
        values = event.model_dump(exclude={"event_id", "metadata"})
        values["event_type"] = event.event_type.value
        record = InteractionEventRecord(
            event_id=event.event_id,
            event_metadata=event.metadata,
            **values,
        )
        self.session.add(record)
        self.session.flush()
        return event


# Concise alias for callers that treat this as the product repository.
ProductRepository = CatalogRepository
