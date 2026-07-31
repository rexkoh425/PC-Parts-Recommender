"""Create the initial catalog, recommendation, evidence, and vector schema.

Revision ID: 20260722_0001
Revises: None
Create Date: 2026-07-22
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

revision: str = "20260722_0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create the version-one relational and retrieval schema."""

    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "canonical_products",
        sa.Column("product_id", sa.String(length=80), nullable=False),
        sa.Column("category", sa.String(length=32), nullable=False),
        sa.Column("brand", sa.String(length=120), nullable=False),
        sa.Column("model", sa.String(length=240), nullable=False),
        sa.Column("manufacturer_part_number", sa.String(length=160), nullable=True),
        sa.Column("gtin", sa.String(length=32), nullable=True),
        sa.Column("canonical_name", sa.String(length=400), nullable=False),
        sa.Column("release_date", sa.Date(), nullable=True),
        sa.Column("status", sa.String(length=32), server_default="active", nullable=False),
        sa.Column(
            "common_attributes", sa.JSON(), server_default=sa.text("'{}'::json"), nullable=False
        ),
        sa.Column(
            "category_attributes",
            sa.JSON(),
            server_default=sa.text("'{}'::json"),
            nullable=False,
        ),
        sa.Column("source_confidence", sa.Float(), server_default="1", nullable=False),
        sa.Column("search_document", sa.Text(), server_default="", nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.CheckConstraint(
            "source_confidence >= 0 AND source_confidence <= 1",
            name="ck_product_source_confidence_range",
        ),
        sa.PrimaryKeyConstraint("product_id"),
        sa.UniqueConstraint("brand", "manufacturer_part_number", name="uq_product_brand_mpn"),
        sa.UniqueConstraint("gtin", name="uq_canonical_products_gtin"),
    )
    op.create_index("ix_canonical_products_brand", "canonical_products", ["brand"])
    op.create_index(
        "ix_canonical_products_canonical_name", "canonical_products", ["canonical_name"]
    )
    op.create_index("ix_canonical_products_category", "canonical_products", ["category"])
    op.create_index(
        "ix_product_category_status", "canonical_products", ["category", "status"]
    )
    op.execute(
        "CREATE INDEX ix_product_search_document_fts ON canonical_products "
        "USING gin (to_tsvector('english', search_document))"
    )

    op.create_table(
        "retailer_listings",
        sa.Column("listing_id", sa.String(length=80), nullable=False),
        sa.Column("product_id", sa.String(length=80), nullable=False),
        sa.Column("retailer", sa.String(length=160), nullable=False),
        sa.Column("source_listing_id", sa.String(length=240), nullable=False),
        sa.Column("title", sa.String(length=500), nullable=False),
        sa.Column("condition", sa.String(length=32), server_default="new", nullable=False),
        sa.Column("currency", sa.String(length=3), server_default="SGD", nullable=False),
        sa.Column("base_price", sa.Numeric(precision=12, scale=2), nullable=False),
        sa.Column(
            "shipping_price",
            sa.Numeric(precision=12, scale=2),
            server_default="0",
            nullable=False,
        ),
        sa.Column("stock_status", sa.String(length=32), nullable=False),
        sa.Column("seller_name", sa.String(length=200), nullable=True),
        sa.Column("listing_url", sa.Text(), nullable=False),
        sa.Column("first_seen_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_seen_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("base_price >= 0", name="ck_listing_base_price_nonnegative"),
        sa.CheckConstraint("shipping_price >= 0", name="ck_listing_shipping_nonnegative"),
        sa.ForeignKeyConstraint(
            ["product_id"], ["canonical_products.product_id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("listing_id"),
        sa.UniqueConstraint("retailer", "source_listing_id", name="uq_retailer_source_listing"),
    )
    op.create_index("ix_retailer_listings_retailer", "retailer_listings", ["retailer"])
    op.create_index("ix_retailer_listings_stock_status", "retailer_listings", ["stock_status"])
    op.create_index(
        "ix_listing_product_stock", "retailer_listings", ["product_id", "stock_status"]
    )

    op.create_table(
        "price_snapshots",
        sa.Column("snapshot_id", sa.String(length=80), nullable=False),
        sa.Column("listing_id", sa.String(length=80), nullable=False),
        sa.Column("observed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("base_price", sa.Numeric(precision=12, scale=2), nullable=False),
        sa.Column(
            "shipping_price",
            sa.Numeric(precision=12, scale=2),
            server_default="0",
            nullable=False,
        ),
        sa.Column("stock_status", sa.String(length=32), nullable=False),
        sa.Column("promotion_text", sa.Text(), nullable=True),
        sa.CheckConstraint("base_price >= 0", name="ck_price_base_nonnegative"),
        sa.CheckConstraint("shipping_price >= 0", name="ck_price_shipping_nonnegative"),
        sa.ForeignKeyConstraint(
            ["listing_id"], ["retailer_listings.listing_id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("snapshot_id"),
        sa.UniqueConstraint("listing_id", "observed_at", name="uq_price_listing_observed"),
    )
    op.create_index(
        "ix_price_listing_observed", "price_snapshots", ["listing_id", "observed_at"]
    )

    op.create_table(
        "benchmark_results",
        sa.Column("benchmark_id", sa.String(length=80), nullable=False),
        sa.Column("product_id", sa.String(length=80), nullable=False),
        sa.Column("workload", sa.String(length=64), nullable=False),
        sa.Column("benchmark_name", sa.String(length=240), nullable=False),
        sa.Column("benchmark_version", sa.String(length=120), nullable=False),
        sa.Column("score", sa.Float(), nullable=False),
        sa.Column("unit", sa.String(length=80), nullable=False),
        sa.Column("higher_is_better", sa.Boolean(), server_default=sa.true(), nullable=False),
        sa.Column("resolution", sa.String(length=64), nullable=True),
        sa.Column("preset", sa.String(length=64), nullable=True),
        sa.Column("operating_system", sa.String(length=160), nullable=True),
        sa.Column("driver_version", sa.String(length=120), nullable=True),
        sa.Column("source_url", sa.Text(), nullable=False),
        sa.Column("observed_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["product_id"], ["canonical_products.product_id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("benchmark_id"),
    )
    op.create_index(
        "ix_benchmark_product_workload", "benchmark_results", ["product_id", "workload"]
    )

    op.create_table(
        "source_provenance",
        sa.Column("provenance_id", sa.String(length=80), nullable=False),
        sa.Column("product_id", sa.String(length=80), nullable=True),
        sa.Column("listing_id", sa.String(length=80), nullable=True),
        sa.Column("source_name", sa.String(length=200), nullable=False),
        sa.Column("source_url", sa.Text(), nullable=False),
        sa.Column("source_type", sa.String(length=32), nullable=False),
        sa.Column("retrieved_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("raw_content_hash", sa.String(length=160), nullable=False),
        sa.Column("parser_version", sa.String(length=80), nullable=False),
        sa.Column("licence_or_access_note", sa.Text(), nullable=False),
        sa.Column("last_verified_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("extraction_confidence", sa.Float(), nullable=False),
        sa.CheckConstraint(
            "extraction_confidence >= 0 AND extraction_confidence <= 1",
            name="ck_provenance_extraction_confidence_range",
        ),
        sa.CheckConstraint(
            "product_id IS NOT NULL OR listing_id IS NOT NULL", name="ck_provenance_has_owner"
        ),
        sa.ForeignKeyConstraint(
            ["listing_id"], ["retailer_listings.listing_id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["product_id"], ["canonical_products.product_id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("provenance_id"),
    )
    op.create_index("ix_provenance_listing", "source_provenance", ["listing_id"])
    op.create_index("ix_provenance_product", "source_provenance", ["product_id"])

    op.create_table(
        "compatibility_rules",
        sa.Column("rule_id", sa.String(length=80), nullable=False),
        sa.Column("rule_version", sa.String(length=80), nullable=False),
        sa.Column("left_category", sa.String(length=32), nullable=False),
        sa.Column("right_category", sa.String(length=32), nullable=False),
        sa.Column("rule_type", sa.String(length=120), nullable=False),
        sa.Column("severity", sa.String(length=32), nullable=False),
        sa.Column(
            "required_fields", sa.JSON(), server_default=sa.text("'[]'::json"), nullable=False
        ),
        sa.Column("message_template", sa.Text(), nullable=False),
        sa.Column("evidence_source", sa.Text(), nullable=False),
        sa.Column("effective_from", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("rule_id"),
    )
    op.create_index(
        "ix_rule_version_categories",
        "compatibility_rules",
        ["rule_version", "left_category", "right_category"],
    )

    op.create_table(
        "review_evidence",
        sa.Column("evidence_id", sa.String(length=80), nullable=False),
        sa.Column("product_id", sa.String(length=80), nullable=False),
        sa.Column("aspect", sa.String(length=80), nullable=False),
        sa.Column("sentiment", sa.Float(), nullable=False),
        sa.Column("evidence_text", sa.Text(), nullable=False),
        sa.Column("source_url", sa.Text(), nullable=False),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.CheckConstraint("sentiment >= -1 AND sentiment <= 1", name="ck_review_sentiment_range"),
        sa.CheckConstraint(
            "confidence >= 0 AND confidence <= 1", name="ck_review_confidence_range"
        ),
        sa.ForeignKeyConstraint(
            ["product_id"], ["canonical_products.product_id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("evidence_id"),
    )
    op.create_index("ix_review_product_aspect", "review_evidence", ["product_id", "aspect"])

    op.create_table(
        "search_queries",
        sa.Column("query_id", sa.String(length=80), nullable=False),
        sa.Column("raw_query", sa.Text(), nullable=True),
        sa.Column(
            "structured_constraints",
            sa.JSON(),
            server_default=sa.text("'{}'::json"),
            nullable=False,
        ),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.PrimaryKeyConstraint("query_id"),
    )

    op.create_table(
        "generated_builds",
        sa.Column("build_id", sa.String(length=80), nullable=False),
        sa.Column("query_id", sa.String(length=80), nullable=False),
        sa.Column("profile", sa.String(length=64), nullable=False),
        sa.Column("total_price", sa.Numeric(precision=12, scale=2), nullable=False),
        sa.Column("overall_score", sa.Float(), nullable=False),
        sa.Column(
            "workload_scores", sa.JSON(), server_default=sa.text("'{}'::json"), nullable=False
        ),
        sa.Column("compatibility_status", sa.String(length=32), nullable=False),
        sa.Column(
            "compatibility_checks", sa.JSON(), server_default=sa.text("'[]'::json"), nullable=False
        ),
        sa.Column("estimated_power_watts", sa.Float(), nullable=True),
        sa.Column("warnings", sa.JSON(), server_default=sa.text("'[]'::json"), nullable=False),
        sa.Column("explanation", sa.JSON(), server_default=sa.text("'[]'::json"), nullable=False),
        sa.Column("alternatives", sa.JSON(), server_default=sa.text("'[]'::json"), nullable=False),
        sa.Column("optimizer_status", sa.String(length=64), nullable=False),
        sa.Column("rule_version", sa.String(length=80), nullable=False),
        sa.Column("model_version", sa.String(length=80), nullable=False),
        sa.Column("data_version", sa.String(length=80), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.CheckConstraint("total_price >= 0", name="ck_build_total_price_nonnegative"),
        sa.CheckConstraint(
            "overall_score >= 0 AND overall_score <= 100", name="ck_build_overall_score_range"
        ),
        sa.ForeignKeyConstraint(["query_id"], ["search_queries.query_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("build_id"),
    )
    op.create_index("ix_build_query_profile", "generated_builds", ["query_id", "profile"])

    op.create_table(
        "build_components",
        sa.Column("build_id", sa.String(length=80), nullable=False),
        sa.Column("category", sa.String(length=32), nullable=False),
        sa.Column("product_id", sa.String(length=80), nullable=False),
        sa.Column("listing_id", sa.String(length=80), nullable=True),
        sa.Column("canonical_name", sa.String(length=400), nullable=False),
        sa.Column("price_sgd", sa.Numeric(precision=12, scale=2), nullable=False),
        sa.Column("component_score", sa.Float(), nullable=False),
        sa.Column("selection_reason", sa.Text(), nullable=False),
        sa.CheckConstraint("price_sgd >= 0", name="ck_component_price_nonnegative"),
        sa.CheckConstraint(
            "component_score >= 0 AND component_score <= 100",
            name="ck_component_score_range",
        ),
        sa.ForeignKeyConstraint(
            ["build_id"], ["generated_builds.build_id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(["listing_id"], ["retailer_listings.listing_id"]),
        sa.ForeignKeyConstraint(["product_id"], ["canonical_products.product_id"]),
        sa.PrimaryKeyConstraint("build_id", "category"),
        sa.UniqueConstraint("build_id", "category", name="uq_build_component_category"),
    )

    op.create_table(
        "interaction_events",
        sa.Column("event_id", sa.String(length=80), nullable=False),
        sa.Column("session_id", sa.String(length=160), nullable=False),
        sa.Column("user_id", sa.String(length=160), nullable=True),
        sa.Column("query_id", sa.String(length=80), nullable=True),
        sa.Column("product_id", sa.String(length=80), nullable=True),
        sa.Column("build_id", sa.String(length=80), nullable=True),
        sa.Column("event_type", sa.String(length=64), nullable=False),
        sa.Column("rank_position", sa.Integer(), nullable=True),
        sa.Column("model_version", sa.String(length=80), nullable=True),
        sa.Column("data_version", sa.String(length=80), nullable=True),
        sa.Column("rule_version", sa.String(length=80), nullable=True),
        sa.Column("metadata", sa.JSON(), server_default=sa.text("'{}'::json"), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.CheckConstraint(
            "rank_position IS NULL OR rank_position >= 1", name="ck_event_rank_positive"
        ),
        sa.ForeignKeyConstraint(["build_id"], ["generated_builds.build_id"]),
        sa.ForeignKeyConstraint(["product_id"], ["canonical_products.product_id"]),
        sa.ForeignKeyConstraint(["query_id"], ["search_queries.query_id"]),
        sa.PrimaryKeyConstraint("event_id"),
    )
    op.create_index("ix_interaction_events_session_id", "interaction_events", ["session_id"])
    op.create_index("ix_interaction_events_user_id", "interaction_events", ["user_id"])
    op.create_index("ix_event_created_type", "interaction_events", ["created_at", "event_type"])
    op.create_index("ix_event_query_rank", "interaction_events", ["query_id", "rank_position"])

    op.create_table(
        "product_embeddings",
        sa.Column("product_id", sa.String(length=80), nullable=False),
        sa.Column("embedding_model", sa.String(length=240), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("embedding", Vector(dim=384), nullable=False),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.ForeignKeyConstraint(
            ["product_id"], ["canonical_products.product_id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("product_id", "embedding_model"),
    )
    op.execute(
        "CREATE INDEX ix_product_embeddings_cosine ON product_embeddings "
        "USING hnsw (embedding vector_cosine_ops)"
    )


def downgrade() -> None:
    """Remove application tables while leaving the shared vector extension installed."""

    op.drop_index("ix_product_embeddings_cosine", table_name="product_embeddings")
    op.drop_table("product_embeddings")
    op.drop_index("ix_event_query_rank", table_name="interaction_events")
    op.drop_index("ix_event_created_type", table_name="interaction_events")
    op.drop_index("ix_interaction_events_user_id", table_name="interaction_events")
    op.drop_index("ix_interaction_events_session_id", table_name="interaction_events")
    op.drop_table("interaction_events")
    op.drop_table("build_components")
    op.drop_index("ix_build_query_profile", table_name="generated_builds")
    op.drop_table("generated_builds")
    op.drop_table("search_queries")
    op.drop_index("ix_review_product_aspect", table_name="review_evidence")
    op.drop_table("review_evidence")
    op.drop_index("ix_rule_version_categories", table_name="compatibility_rules")
    op.drop_table("compatibility_rules")
    op.drop_index("ix_provenance_product", table_name="source_provenance")
    op.drop_index("ix_provenance_listing", table_name="source_provenance")
    op.drop_table("source_provenance")
    op.drop_index("ix_benchmark_product_workload", table_name="benchmark_results")
    op.drop_table("benchmark_results")
    op.drop_index("ix_price_listing_observed", table_name="price_snapshots")
    op.drop_table("price_snapshots")
    op.drop_index("ix_listing_product_stock", table_name="retailer_listings")
    op.drop_index("ix_retailer_listings_stock_status", table_name="retailer_listings")
    op.drop_index("ix_retailer_listings_retailer", table_name="retailer_listings")
    op.drop_table("retailer_listings")
    op.execute("DROP INDEX IF EXISTS ix_product_search_document_fts")
    op.drop_index("ix_product_category_status", table_name="canonical_products")
    op.drop_index("ix_canonical_products_category", table_name="canonical_products")
    op.drop_index("ix_canonical_products_canonical_name", table_name="canonical_products")
    op.drop_index("ix_canonical_products_brand", table_name="canonical_products")
    op.drop_table("canonical_products")
