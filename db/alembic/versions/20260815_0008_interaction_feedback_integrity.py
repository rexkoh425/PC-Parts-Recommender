"""Add trusted-impression and idempotency evidence to interactions.

Revision ID: 20260815_0008
Revises: 20260723_0007
Create Date: 2026-08-15
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260815_0008"
down_revision: str | None = "20260723_0007"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "interaction_events", sa.Column("impression_id", sa.String(length=80), nullable=True)
    )
    op.add_column(
        "interaction_events",
        sa.Column(
            "trust_level",
            sa.String(length=40),
            server_default=sa.text("'legacy_untrusted'"),
            nullable=False,
        ),
    )
    op.add_column(
        "interaction_events",
        sa.Column("idempotency_key_sha256", sa.String(length=64), nullable=True),
    )
    op.add_column(
        "interaction_events",
        sa.Column("idempotency_payload_sha256", sa.String(length=64), nullable=True),
    )
    op.create_index("ix_interaction_impression_id", "interaction_events", ["impression_id"])
    op.create_unique_constraint(
        "uq_interaction_session_idempotency",
        "interaction_events",
        ["session_id", "idempotency_key_sha256"],
    )
    op.create_unique_constraint(
        "uq_interaction_impression_event",
        "interaction_events",
        ["impression_id", "event_type"],
    )
    op.create_check_constraint(
        "ck_interaction_trust_level",
        "interaction_events",
        "trust_level IN ('verified_impression', 'legacy_untrusted')",
    )
    op.create_check_constraint(
        "ck_interaction_idempotency_pair",
        "interaction_events",
        "((idempotency_key_sha256 IS NULL AND idempotency_payload_sha256 IS NULL) "
        "OR (idempotency_key_sha256 IS NOT NULL "
        "AND idempotency_payload_sha256 IS NOT NULL))",
    )
    op.create_check_constraint(
        "ck_interaction_idempotency_key_length",
        "interaction_events",
        "idempotency_key_sha256 IS NULL OR length(idempotency_key_sha256) = 64",
    )
    op.create_check_constraint(
        "ck_interaction_idempotency_payload_length",
        "interaction_events",
        "idempotency_payload_sha256 IS NULL OR length(idempotency_payload_sha256) = 64",
    )
    op.create_check_constraint(
        "ck_interaction_verified_evidence",
        "interaction_events",
        "trust_level != 'verified_impression' OR "
        "(impression_id IS NOT NULL AND idempotency_key_sha256 IS NOT NULL)",
    )


def downgrade() -> None:
    op.drop_constraint("ck_interaction_verified_evidence", "interaction_events", type_="check")
    op.drop_constraint(
        "ck_interaction_idempotency_payload_length", "interaction_events", type_="check"
    )
    op.drop_constraint("ck_interaction_idempotency_key_length", "interaction_events", type_="check")
    op.drop_constraint("ck_interaction_idempotency_pair", "interaction_events", type_="check")
    op.drop_constraint("ck_interaction_trust_level", "interaction_events", type_="check")
    op.drop_constraint("uq_interaction_impression_event", "interaction_events", type_="unique")
    op.drop_constraint("uq_interaction_session_idempotency", "interaction_events", type_="unique")
    op.drop_index("ix_interaction_impression_id", table_name="interaction_events")
    op.drop_column("interaction_events", "idempotency_payload_sha256")
    op.drop_column("interaction_events", "idempotency_key_sha256")
    op.drop_column("interaction_events", "trust_level")
    op.drop_column("interaction_events", "impression_id")
