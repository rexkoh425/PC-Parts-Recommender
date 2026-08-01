"""Add immutable artifact metadata to product embeddings.

Revision ID: 20260722_0002
Revises: 20260722_0001
Create Date: 2026-07-22
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260722_0002"
down_revision: str | None = "20260722_0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Store the exact data, index, encoder, and file hashes for every vector."""

    legacy_hash = "0" * 64
    op.add_column(
        "product_embeddings",
        sa.Column(
            "data_version", sa.String(length=160), server_default="legacy", nullable=False
        ),
    )
    op.add_column(
        "product_embeddings",
        sa.Column(
            "index_version", sa.String(length=160), server_default="legacy", nullable=False
        ),
    )
    for name in (
        "encoder_fingerprint",
        "dataset_content_hash",
        "embeddings_artifact_sha256",
        "id_map_artifact_sha256",
    ):
        op.add_column(
            "product_embeddings",
            sa.Column(name, sa.String(length=64), server_default=legacy_hash, nullable=False),
        )
    op.create_index(
        "ix_product_embeddings_model_data",
        "product_embeddings",
        ["embedding_model", "data_version", "index_version"],
    )
    for name in (
        "data_version",
        "index_version",
        "encoder_fingerprint",
        "dataset_content_hash",
        "embeddings_artifact_sha256",
        "id_map_artifact_sha256",
    ):
        op.alter_column("product_embeddings", name, server_default=None)


def downgrade() -> None:
    """Remove artifact metadata while preserving the original vector rows."""

    op.drop_index("ix_product_embeddings_model_data", table_name="product_embeddings")
    for name in (
        "id_map_artifact_sha256",
        "embeddings_artifact_sha256",
        "dataset_content_hash",
        "encoder_fingerprint",
        "index_version",
        "data_version",
    ):
        op.drop_column("product_embeddings", name)
