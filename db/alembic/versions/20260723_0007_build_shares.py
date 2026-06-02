"""Add revocable immutable public build-share snapshots.

Revision ID: 20260723_0007
Revises: 20260723_0006
Create Date: 2026-07-23
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260723_0007"
down_revision: str | None = "20260723_0006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "build_shares",
        sa.Column("share_id", sa.String(length=80), nullable=False),
        sa.Column("build_id", sa.String(length=80), nullable=False),
        sa.Column("snapshot", sa.JSON(), nullable=False),
        sa.Column("revocation_token_sha256", sa.String(length=64), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["build_id"], ["generated_builds.build_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("share_id"),
    )
    op.create_index("ix_build_share_build", "build_shares", ["build_id"])
    op.create_index("ix_build_share_expires", "build_shares", ["expires_at"])


def downgrade() -> None:
    op.drop_index("ix_build_share_expires", table_name="build_shares")
    op.drop_index("ix_build_share_build", table_name="build_shares")
    op.drop_table("build_shares")
