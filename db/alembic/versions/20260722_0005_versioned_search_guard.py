"""Guard lexical retrieval against mutable-catalog/vector release mixing.

Revision ID: 20260722_0005
Revises: 20260722_0004
Create Date: 2026-07-22
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260722_0005"
down_revision: str | None = "20260722_0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Store the imported search-document hash used by exact-release joins."""

    op.add_column(
        "canonical_products",
        sa.Column("search_document_hash", sa.String(length=64), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("canonical_products", "search_document_hash")
