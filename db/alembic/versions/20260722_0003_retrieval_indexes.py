"""Harden PostgreSQL structured and hybrid retrieval indexes.

Revision ID: 20260722_0003
Revises: 20260722_0002
Create Date: 2026-07-22
"""

from collections.abc import Sequence

from alembic import op

revision: str = "20260722_0003"
down_revision: str | None = "20260722_0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Use JSONB for filter fields and add the measured-query candidate indexes."""

    op.execute(
        "ALTER TABLE canonical_products "
        "ALTER COLUMN common_attributes TYPE jsonb USING common_attributes::jsonb"
    )
    op.execute(
        "ALTER TABLE canonical_products "
        "ALTER COLUMN category_attributes TYPE jsonb USING category_attributes::jsonb"
    )
    op.execute(
        "CREATE INDEX ix_listing_product_stock_total_price ON retailer_listings "
        "(product_id, stock_status, ((base_price + shipping_price))) "
        "WHERE condition = 'new' AND currency = 'SGD'"
    )
    op.create_index(
        "ix_product_embeddings_serving_version",
        "product_embeddings",
        [
            "embedding_model",
            "data_version",
            "index_version",
            "encoder_fingerprint",
            "dataset_content_hash",
        ],
    )
    op.drop_index("ix_product_embeddings_model_data", table_name="product_embeddings")


def downgrade() -> None:
    """Remove retrieval indexes and return JSONB columns to the initial JSON type."""

    op.drop_index("ix_product_embeddings_serving_version", table_name="product_embeddings")
    op.create_index(
        "ix_product_embeddings_model_data",
        "product_embeddings",
        ["embedding_model", "data_version", "index_version"],
    )
    op.drop_index("ix_listing_product_stock_total_price", table_name="retailer_listings")
    op.execute(
        "ALTER TABLE canonical_products "
        "ALTER COLUMN category_attributes TYPE json USING category_attributes::json"
    )
    op.execute(
        "ALTER TABLE canonical_products "
        "ALTER COLUMN common_attributes TYPE json USING common_attributes::json"
    )
