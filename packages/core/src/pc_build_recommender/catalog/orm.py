"""SQLAlchemy 2.0 persistence model for catalog and recommendation evidence."""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Any
from uuid import uuid4

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

POSTGRES_JSON = JSON().with_variant(JSONB(), "postgresql")


def _id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


def _utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class CanonicalProductRecord(Base):
    __tablename__ = "canonical_products"
    __table_args__ = (
        UniqueConstraint(
            "brand", "manufacturer_part_number", name="uq_product_brand_mpn"
        ),
        Index("ix_product_category_status", "category", "status"),
    )

    product_id: Mapped[str] = mapped_column(String(80), primary_key=True)
    category: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    brand: Mapped[str] = mapped_column(String(120), nullable=False, index=True)
    model: Mapped[str] = mapped_column(String(240), nullable=False)
    manufacturer_part_number: Mapped[str | None] = mapped_column(String(160))
    gtin: Mapped[str | None] = mapped_column(String(32), unique=True)
    canonical_name: Mapped[str] = mapped_column(String(400), nullable=False, index=True)
    release_date: Mapped[date | None] = mapped_column(Date)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="active")
    common_attributes: Mapped[dict[str, Any]] = mapped_column(POSTGRES_JSON, default=dict)
    category_attributes: Mapped[dict[str, Any]] = mapped_column(POSTGRES_JSON, default=dict)
    source_confidence: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    search_document: Mapped[str] = mapped_column(Text, nullable=False, default="")
    search_document_hash: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow, onupdate=_utcnow
    )

    listings: Mapped[list[RetailerListingRecord]] = relationship(
        back_populates="product", cascade="all, delete-orphan"
    )
    benchmarks: Mapped[list[BenchmarkResultRecord]] = relationship(
        back_populates="product", cascade="all, delete-orphan"
    )
    review_evidence: Mapped[list[ReviewEvidenceRecord]] = relationship(
        back_populates="product", cascade="all, delete-orphan"
    )
    provenance: Mapped[list[SourceProvenanceRecord]] = relationship(
        back_populates="product", cascade="all, delete-orphan"
    )


class ProductEmbeddingRecord(Base):
    """Versioned pgvector row tied to one canonical product.

    The database currently serves the 384-dimensional MiniLM index.  Artifact
    and dataset hashes are stored alongside each row so a deployment can prove
    which immutable files produced the live index instead of relying on a
    mutable model name alone.
    """

    __tablename__ = "product_embeddings"
    __table_args__ = (
        Index(
            "ix_product_embeddings_serving_version",
            "embedding_model",
            "data_version",
            "index_version",
            "encoder_fingerprint",
            "dataset_content_hash",
        ),
    )

    product_id: Mapped[str] = mapped_column(
        ForeignKey("canonical_products.product_id", ondelete="CASCADE"), primary_key=True
    )
    embedding_model: Mapped[str] = mapped_column(String(240), primary_key=True)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector(384), nullable=False)
    data_version: Mapped[str] = mapped_column(String(160), primary_key=True)
    index_version: Mapped[str] = mapped_column(String(160), primary_key=True)
    encoder_fingerprint: Mapped[str] = mapped_column(String(64), primary_key=True)
    dataset_content_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    embeddings_artifact_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    id_map_artifact_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow, onupdate=_utcnow
    )


class RetailerListingRecord(Base):
    __tablename__ = "retailer_listings"
    __table_args__ = (
        UniqueConstraint("retailer", "source_listing_id", name="uq_retailer_source_listing"),
        CheckConstraint("base_price >= 0", name="ck_listing_base_price_nonnegative"),
        CheckConstraint("shipping_price >= 0", name="ck_listing_shipping_nonnegative"),
        Index("ix_listing_product_stock", "product_id", "stock_status"),
    )

    listing_id: Mapped[str] = mapped_column(String(80), primary_key=True)
    product_id: Mapped[str] = mapped_column(
        ForeignKey("canonical_products.product_id", ondelete="CASCADE"), nullable=False
    )
    retailer: Mapped[str] = mapped_column(String(160), nullable=False, index=True)
    source_listing_id: Mapped[str] = mapped_column(String(240), nullable=False)
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    condition: Mapped[str] = mapped_column(String(32), nullable=False, default="new")
    currency: Mapped[str] = mapped_column(String(3), nullable=False, default="SGD")
    base_price: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)
    shipping_price: Mapped[Decimal] = mapped_column(
        Numeric(12, 2), nullable=False, default=Decimal("0")
    )
    stock_status: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    seller_name: Mapped[str | None] = mapped_column(String(200))
    listing_url: Mapped[str] = mapped_column(Text, nullable=False)
    first_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    last_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    product: Mapped[CanonicalProductRecord] = relationship(back_populates="listings")
    price_snapshots: Mapped[list[PriceSnapshotRecord]] = relationship(
        back_populates="listing", cascade="all, delete-orphan"
    )
    provenance: Mapped[list[SourceProvenanceRecord]] = relationship(
        back_populates="listing", cascade="all, delete-orphan"
    )


class PriceSnapshotRecord(Base):
    __tablename__ = "price_snapshots"
    __table_args__ = (
        UniqueConstraint("listing_id", "observed_at", name="uq_price_listing_observed"),
        CheckConstraint("base_price >= 0", name="ck_price_base_nonnegative"),
        CheckConstraint("shipping_price >= 0", name="ck_price_shipping_nonnegative"),
        Index("ix_price_listing_observed", "listing_id", "observed_at"),
    )

    snapshot_id: Mapped[str] = mapped_column(String(80), primary_key=True)
    listing_id: Mapped[str] = mapped_column(
        ForeignKey("retailer_listings.listing_id", ondelete="CASCADE"), nullable=False
    )
    observed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    base_price: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)
    shipping_price: Mapped[Decimal] = mapped_column(
        Numeric(12, 2), nullable=False, default=Decimal("0")
    )
    stock_status: Mapped[str] = mapped_column(String(32), nullable=False)
    promotion_text: Mapped[str | None] = mapped_column(Text)

    listing: Mapped[RetailerListingRecord] = relationship(back_populates="price_snapshots")


class BenchmarkResultRecord(Base):
    __tablename__ = "benchmark_results"
    __table_args__ = (
        Index("ix_benchmark_product_workload", "product_id", "workload"),
    )

    benchmark_id: Mapped[str] = mapped_column(String(80), primary_key=True)
    product_id: Mapped[str] = mapped_column(
        ForeignKey("canonical_products.product_id", ondelete="CASCADE"), nullable=False
    )
    workload: Mapped[str] = mapped_column(String(64), nullable=False)
    benchmark_name: Mapped[str] = mapped_column(String(240), nullable=False)
    benchmark_version: Mapped[str] = mapped_column(String(120), nullable=False)
    score: Mapped[float] = mapped_column(Float, nullable=False)
    unit: Mapped[str] = mapped_column(String(80), nullable=False)
    higher_is_better: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    resolution: Mapped[str | None] = mapped_column(String(64))
    preset: Mapped[str | None] = mapped_column(String(64))
    operating_system: Mapped[str | None] = mapped_column(String(160))
    driver_version: Mapped[str | None] = mapped_column(String(120))
    source_url: Mapped[str] = mapped_column(Text, nullable=False)
    observed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    product: Mapped[CanonicalProductRecord] = relationship(back_populates="benchmarks")


class SourceProvenanceRecord(Base):
    __tablename__ = "source_provenance"
    __table_args__ = (
        Index("ix_provenance_product", "product_id"),
        Index("ix_provenance_listing", "listing_id"),
    )

    provenance_id: Mapped[str] = mapped_column(String(80), primary_key=True)
    product_id: Mapped[str | None] = mapped_column(
        ForeignKey("canonical_products.product_id", ondelete="CASCADE")
    )
    listing_id: Mapped[str | None] = mapped_column(
        ForeignKey("retailer_listings.listing_id", ondelete="CASCADE")
    )
    source_name: Mapped[str] = mapped_column(String(200), nullable=False)
    source_url: Mapped[str] = mapped_column(Text, nullable=False)
    source_type: Mapped[str] = mapped_column(String(32), nullable=False)
    retrieved_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    raw_content_hash: Mapped[str] = mapped_column(String(160), nullable=False)
    parser_version: Mapped[str] = mapped_column(String(80), nullable=False)
    licence_or_access_note: Mapped[str] = mapped_column(Text, nullable=False)
    last_verified_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    extraction_confidence: Mapped[float] = mapped_column(Float, nullable=False)

    product: Mapped[CanonicalProductRecord | None] = relationship(back_populates="provenance")
    listing: Mapped[RetailerListingRecord | None] = relationship(back_populates="provenance")


class CompatibilityRuleRecord(Base):
    __tablename__ = "compatibility_rules"
    __table_args__ = (
        Index("ix_rule_version_categories", "rule_version", "left_category", "right_category"),
    )

    rule_id: Mapped[str] = mapped_column(String(80), primary_key=True)
    rule_version: Mapped[str] = mapped_column(String(80), nullable=False)
    left_category: Mapped[str] = mapped_column(String(32), nullable=False)
    right_category: Mapped[str] = mapped_column(String(32), nullable=False)
    rule_type: Mapped[str] = mapped_column(String(120), nullable=False)
    severity: Mapped[str] = mapped_column(String(32), nullable=False)
    required_fields: Mapped[list[str]] = mapped_column(JSON, default=list)
    message_template: Mapped[str] = mapped_column(Text, nullable=False)
    evidence_source: Mapped[str] = mapped_column(Text, nullable=False)
    effective_from: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class ReviewEvidenceRecord(Base):
    __tablename__ = "review_evidence"
    __table_args__ = (Index("ix_review_product_aspect", "product_id", "aspect"),)

    evidence_id: Mapped[str] = mapped_column(String(80), primary_key=True)
    product_id: Mapped[str] = mapped_column(
        ForeignKey("canonical_products.product_id", ondelete="CASCADE"), nullable=False
    )
    aspect: Mapped[str] = mapped_column(String(80), nullable=False)
    sentiment: Mapped[float] = mapped_column(Float, nullable=False)
    evidence_text: Mapped[str] = mapped_column(Text, nullable=False)
    source_url: Mapped[str] = mapped_column(Text, nullable=False)
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    confidence: Mapped[float] = mapped_column(Float, nullable=False)

    product: Mapped[CanonicalProductRecord] = relationship(back_populates="review_evidence")


class SearchQueryRecord(Base):
    __tablename__ = "search_queries"

    query_id: Mapped[str] = mapped_column(String(80), primary_key=True)
    raw_query: Mapped[str | None] = mapped_column(Text)
    structured_constraints: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    builds: Mapped[list[GeneratedBuildRecord]] = relationship(back_populates="query")


class GeneratedBuildRecord(Base):
    __tablename__ = "generated_builds"
    __table_args__ = (Index("ix_build_query_profile", "query_id", "profile"),)

    build_id: Mapped[str] = mapped_column(String(80), primary_key=True)
    query_id: Mapped[str] = mapped_column(
        ForeignKey("search_queries.query_id", ondelete="CASCADE"), nullable=False
    )
    profile: Mapped[str] = mapped_column(String(64), nullable=False)
    total_price: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)
    overall_score: Mapped[float] = mapped_column(Float, nullable=False)
    workload_scores: Mapped[dict[str, float]] = mapped_column(JSON, default=dict)
    compatibility_status: Mapped[str] = mapped_column(String(32), nullable=False)
    compatibility_checks: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    estimated_power_watts: Mapped[float | None] = mapped_column(Float)
    warnings: Mapped[list[str]] = mapped_column(JSON, default=list)
    explanation: Mapped[list[str]] = mapped_column(JSON, default=list)
    alternatives: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    optimizer_status: Mapped[str] = mapped_column(String(64), nullable=False)
    rule_version: Mapped[str] = mapped_column(String(80), nullable=False)
    model_version: Mapped[str] = mapped_column(String(80), nullable=False)
    data_version: Mapped[str] = mapped_column(String(80), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )

    query: Mapped[SearchQueryRecord] = relationship(back_populates="builds")
    components: Mapped[list[BuildComponentRecord]] = relationship(
        back_populates="build", cascade="all, delete-orphan"
    )
    shares: Mapped[list[BuildShareRecord]] = relationship(
        back_populates="build", cascade="all, delete-orphan"
    )


class BuildComponentRecord(Base):
    __tablename__ = "build_components"
    __table_args__ = (
        UniqueConstraint("build_id", "category", name="uq_build_component_category"),
    )

    build_id: Mapped[str] = mapped_column(
        ForeignKey("generated_builds.build_id", ondelete="CASCADE"), primary_key=True
    )
    category: Mapped[str] = mapped_column(String(32), primary_key=True)
    product_id: Mapped[str] = mapped_column(
        ForeignKey("canonical_products.product_id"), nullable=False
    )
    listing_id: Mapped[str | None] = mapped_column(ForeignKey("retailer_listings.listing_id"))
    canonical_name: Mapped[str] = mapped_column(String(400), nullable=False)
    price_sgd: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)
    component_score: Mapped[float] = mapped_column(Float, nullable=False)
    selection_reason: Mapped[str] = mapped_column(Text, nullable=False)

    build: Mapped[GeneratedBuildRecord] = relationship(back_populates="components")


class BuildShareRecord(Base):
    """Revocable, immutable public projection of a generated build."""

    __tablename__ = "build_shares"
    __table_args__ = (
        Index("ix_build_share_build", "build_id"),
        Index("ix_build_share_expires", "expires_at"),
    )

    share_id: Mapped[str] = mapped_column(String(80), primary_key=True)
    build_id: Mapped[str] = mapped_column(
        ForeignKey("generated_builds.build_id", ondelete="CASCADE"), nullable=False
    )
    snapshot: Mapped[dict[str, Any]] = mapped_column(POSTGRES_JSON, nullable=False)
    revocation_token_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    build: Mapped[GeneratedBuildRecord] = relationship(back_populates="shares")


class InteractionEventRecord(Base):
    __tablename__ = "interaction_events"
    __table_args__ = (
        Index("ix_event_created_type", "created_at", "event_type"),
        Index("ix_event_query_rank", "query_id", "rank_position"),
        Index("ix_interaction_impression_id", "impression_id"),
        UniqueConstraint(
            "session_id",
            "idempotency_key_sha256",
            name="uq_interaction_session_idempotency",
        ),
        UniqueConstraint(
            "impression_id",
            "event_type",
            name="uq_interaction_impression_event",
        ),
        CheckConstraint(
            "trust_level IN ('verified_impression', 'legacy_untrusted')",
            name="ck_interaction_trust_level",
        ),
        CheckConstraint(
            "((idempotency_key_sha256 IS NULL AND idempotency_payload_sha256 IS NULL) "
            "OR (idempotency_key_sha256 IS NOT NULL "
            "AND idempotency_payload_sha256 IS NOT NULL))",
            name="ck_interaction_idempotency_pair",
        ),
        CheckConstraint(
            "idempotency_key_sha256 IS NULL OR length(idempotency_key_sha256) = 64",
            name="ck_interaction_idempotency_key_length",
        ),
        CheckConstraint(
            "idempotency_payload_sha256 IS NULL OR length(idempotency_payload_sha256) = 64",
            name="ck_interaction_idempotency_payload_length",
        ),
        CheckConstraint(
            "trust_level != 'verified_impression' OR "
            "(impression_id IS NOT NULL AND idempotency_key_sha256 IS NOT NULL)",
            name="ck_interaction_verified_evidence",
        ),
    )

    event_id: Mapped[str] = mapped_column(String(80), primary_key=True)
    session_id: Mapped[str] = mapped_column(String(160), nullable=False, index=True)
    user_id: Mapped[str | None] = mapped_column(String(160), index=True)
    query_id: Mapped[str | None] = mapped_column(ForeignKey("search_queries.query_id"))
    product_id: Mapped[str | None] = mapped_column(ForeignKey("canonical_products.product_id"))
    build_id: Mapped[str | None] = mapped_column(ForeignKey("generated_builds.build_id"))
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)
    rank_position: Mapped[int | None] = mapped_column(Integer)
    model_version: Mapped[str | None] = mapped_column(String(80))
    data_version: Mapped[str | None] = mapped_column(String(80))
    rule_version: Mapped[str | None] = mapped_column(String(80))
    impression_id: Mapped[str | None] = mapped_column(String(80))
    trust_level: Mapped[str] = mapped_column(
        String(40), nullable=False, default="legacy_untrusted", server_default="legacy_untrusted"
    )
    idempotency_key_sha256: Mapped[str | None] = mapped_column(String(64))
    idempotency_payload_sha256: Mapped[str | None] = mapped_column(String(64))
    event_metadata: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


# Explicit aliases keep record naming discoverable for different integration styles.
CanonicalProductORM = CanonicalProductRecord
RetailerListingORM = RetailerListingRecord
PriceSnapshotORM = PriceSnapshotRecord
BenchmarkResultORM = BenchmarkResultRecord
SourceProvenanceORM = SourceProvenanceRecord
CompatibilityRuleORM = CompatibilityRuleRecord
ReviewEvidenceORM = ReviewEvidenceRecord
SearchQueryORM = SearchQueryRecord
GeneratedBuildORM = GeneratedBuildRecord
BuildComponentORM = BuildComponentRecord
BuildShareORM = BuildShareRecord
InteractionEventORM = InteractionEventRecord
ProductEmbeddingORM = ProductEmbeddingRecord
