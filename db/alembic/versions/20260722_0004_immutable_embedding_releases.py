"""Make embedding releases immutable and rollback-safe.

Revision ID: 20260722_0004
Revises: 20260722_0003
Create Date: 2026-07-22
"""

from collections.abc import Sequence

from alembic import op

revision: str = "20260722_0004"
down_revision: str | None = "20260722_0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_RELEASE_KEY = [
    "product_id",
    "embedding_model",
    "data_version",
    "index_version",
    "encoder_fingerprint",
    "dataset_content_hash",
]


def upgrade() -> None:
    """Allow old and new immutable vector releases to coexist for instant rollback."""

    op.drop_constraint("product_embeddings_pkey", "product_embeddings", type_="primary")
    op.create_primary_key("product_embeddings_pkey", "product_embeddings", _RELEASE_KEY)


def downgrade() -> None:
    """Retain the newest row per old key before restoring the legacy primary key."""

    op.execute(
        "DELETE FROM product_embeddings WHERE ctid IN ("
        "SELECT ctid FROM ("
        "SELECT ctid, row_number() OVER ("
        "PARTITION BY product_id, embedding_model "
        "ORDER BY updated_at DESC, data_version DESC, index_version DESC"
        ") AS release_rank FROM product_embeddings"
        ") ranked WHERE release_rank > 1)"
    )
    op.drop_constraint("product_embeddings_pkey", "product_embeddings", type_="primary")
    op.create_primary_key(
        "product_embeddings_pkey",
        "product_embeddings",
        ["product_id", "embedding_model"],
    )
