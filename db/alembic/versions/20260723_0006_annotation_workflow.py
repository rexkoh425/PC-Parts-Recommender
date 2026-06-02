"""Create the durable annotation workflow and immutable decision ledger.

Revision ID: 20260723_0006
Revises: 20260722_0005
Create Date: 2026-07-23
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260723_0006"
down_revision: str | None = "20260722_0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_APPEND_ONLY_TABLES = (
    "annotation_judgments",
    "annotation_adjudications",
    "annotation_audit_events",
)
_APPEND_ONLY_FUNCTION = "reject_annotation_append_only_mutation"


def _create_append_only_triggers() -> None:
    """Reject UPDATE, DELETE, and TRUNCATE of immutable rows in PostgreSQL."""

    if op.get_bind().dialect.name != "postgresql":
        return

    op.execute(
        f"""
        CREATE FUNCTION {_APPEND_ONLY_FUNCTION}()
        RETURNS trigger
        LANGUAGE plpgsql
        AS $$
        BEGIN
            RAISE EXCEPTION '% is append-only; UPDATE, DELETE, and TRUNCATE are forbidden',
                TG_TABLE_NAME;
        END;
        $$
        """
    )
    for table_name in _APPEND_ONLY_TABLES:
        op.execute(
            f"""
            CREATE TRIGGER trg_{table_name}_append_only
            BEFORE UPDATE OR DELETE ON {table_name}
            FOR EACH ROW
            EXECUTE FUNCTION {_APPEND_ONLY_FUNCTION}()
            """
        )
        op.execute(
            f"""
            CREATE TRIGGER trg_{table_name}_append_only_truncate
            BEFORE TRUNCATE ON {table_name}
            FOR EACH STATEMENT
            EXECUTE FUNCTION {_APPEND_ONLY_FUNCTION}()
            """
        )


def _drop_append_only_triggers() -> None:
    """Remove PostgreSQL append-only triggers before dropping their tables."""

    if op.get_bind().dialect.name != "postgresql":
        return

    for table_name in _APPEND_ONLY_TABLES:
        op.execute(f"DROP TRIGGER IF EXISTS trg_{table_name}_append_only_truncate ON {table_name}")
        op.execute(f"DROP TRIGGER IF EXISTS trg_{table_name}_append_only ON {table_name}")
    op.execute(f"DROP FUNCTION IF EXISTS {_APPEND_ONLY_FUNCTION}()")


def upgrade() -> None:
    """Create reviewer identity, annotation workflow, export, and audit tables."""

    op.create_table(
        "annotation_reviewers",
        sa.Column("reviewer_id", sa.String(length=80), nullable=False),
        sa.Column("oidc_issuer", sa.String(length=500), nullable=False),
        sa.Column("oidc_subject", sa.String(length=500), nullable=False),
        sa.Column("display_name", sa.String(length=240), nullable=False),
        sa.Column("roles", sa.JSON(), nullable=False),
        sa.Column("active", sa.Boolean(), nullable=False),
        sa.Column("verified_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "length(oidc_issuer) > 0",
            name="ck_annotation_reviewer_issuer",
        ),
        sa.CheckConstraint(
            "length(oidc_subject) > 0",
            name="ck_annotation_reviewer_subject",
        ),
        sa.PrimaryKeyConstraint("reviewer_id"),
        sa.UniqueConstraint(
            "oidc_issuer",
            "oidc_subject",
            name="uq_annotation_reviewer_oidc",
        ),
    )

    op.create_table(
        "annotation_projects",
        sa.Column("project_id", sa.String(length=80), nullable=False),
        sa.Column("task_type", sa.String(length=32), nullable=False),
        sa.Column("dataset_name", sa.String(length=240), nullable=False),
        sa.Column("dataset_version", sa.String(length=240), nullable=False),
        sa.Column("rubric_version", sa.String(length=160), nullable=False),
        sa.Column("data_version", sa.String(length=240), nullable=False),
        sa.Column("source_policy", sa.JSON(), nullable=False),
        sa.Column("source_policy_sha256", sa.String(length=64), nullable=False),
        sa.Column("training_eligible", sa.Boolean(), nullable=False),
        sa.Column("published_metrics_eligible", sa.Boolean(), nullable=False),
        sa.Column("model_serving_eligible", sa.Boolean(), nullable=False),
        sa.Column("required_reviews", sa.Integer(), nullable=False),
        sa.Column("split_names", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(length=24), nullable=False),
        sa.Column("created_by_reviewer_id", sa.String(length=80), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("opened_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("frozen_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "task_type IN ('entity_resolution', 'relevance')",
            name="ck_annotation_project_task_type",
        ),
        sa.CheckConstraint(
            "status IN ('draft', 'open', 'frozen')",
            name="ck_annotation_project_status",
        ),
        sa.CheckConstraint(
            "required_reviews = 2",
            name="ck_annotation_project_dual_review",
        ),
        sa.ForeignKeyConstraint(
            ["created_by_reviewer_id"],
            ["annotation_reviewers.reviewer_id"],
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("project_id"),
        sa.UniqueConstraint(
            "task_type",
            "dataset_version",
            name="uq_annotation_dataset_version",
        ),
    )
    op.create_index(
        "ix_annotation_project_status_type",
        "annotation_projects",
        ["status", "task_type"],
    )

    op.create_table(
        "annotation_groups",
        sa.Column("group_id", sa.String(length=80), nullable=False),
        sa.Column("project_id", sa.String(length=80), nullable=False),
        sa.Column("group_key", sa.String(length=240), nullable=False),
        sa.Column("leakage_group_id", sa.String(length=240), nullable=False),
        sa.Column("category", sa.String(length=32), nullable=False),
        sa.Column("split_name", sa.String(length=32), nullable=False),
        sa.Column("context_payload", sa.JSON(), nullable=False),
        sa.Column("context_sha256", sa.String(length=64), nullable=False),
        sa.Column("is_synthetic", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["project_id"],
            ["annotation_projects.project_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("group_id"),
        sa.UniqueConstraint(
            "project_id",
            "group_key",
            name="uq_annotation_project_group",
        ),
    )
    op.create_index(
        "ix_annotation_group_project_split",
        "annotation_groups",
        ["project_id", "split_name"],
    )

    op.create_table(
        "annotation_items",
        sa.Column("item_id", sa.String(length=80), nullable=False),
        sa.Column("group_id", sa.String(length=80), nullable=False),
        sa.Column("target_id", sa.String(length=240), nullable=False),
        sa.Column("evidence_payload", sa.JSON(), nullable=False),
        sa.Column("evidence_sha256", sa.String(length=64), nullable=False),
        sa.Column("priority", sa.Integer(), nullable=False),
        sa.Column("state", sa.String(length=32), nullable=False),
        sa.Column("is_synthetic", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "state IN ('pending', 'in_review', 'needs_adjudication', 'resolved')",
            name="ck_annotation_item_state",
        ),
        sa.ForeignKeyConstraint(
            ["group_id"],
            ["annotation_groups.group_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("item_id"),
        sa.UniqueConstraint(
            "group_id",
            "target_id",
            name="uq_annotation_group_target",
        ),
    )
    op.create_index(
        "ix_annotation_item_group_state_priority",
        "annotation_items",
        ["group_id", "state", "priority"],
    )

    op.create_table(
        "annotation_assignments",
        sa.Column("assignment_id", sa.String(length=80), nullable=False),
        sa.Column("item_id", sa.String(length=80), nullable=False),
        sa.Column("reviewer_id", sa.String(length=80), nullable=False),
        sa.Column("phase", sa.String(length=24), nullable=False),
        sa.Column("status", sa.String(length=24), nullable=False),
        sa.Column("lease_token_sha256", sa.String(length=64), nullable=False),
        sa.Column("submission_idempotency_sha256", sa.String(length=64), nullable=True),
        sa.Column("submission_payload_sha256", sa.String(length=64), nullable=True),
        sa.Column("leased_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("submitted_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "phase IN ('review', 'adjudication')",
            name="ck_annotation_assignment_phase",
        ),
        sa.CheckConstraint(
            "status IN ('leased', 'submitted', 'expired')",
            name="ck_annotation_assignment_status",
        ),
        sa.ForeignKeyConstraint(
            ["item_id"],
            ["annotation_items.item_id"],
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["reviewer_id"],
            ["annotation_reviewers.reviewer_id"],
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("assignment_id"),
        sa.UniqueConstraint(
            "item_id",
            "reviewer_id",
            "phase",
            name="uq_annotation_assignment_item_reviewer_phase",
        ),
        sa.UniqueConstraint(
            "lease_token_sha256",
            name="uq_annotation_assignment_lease_token",
        ),
        sa.UniqueConstraint(
            "reviewer_id",
            "submission_idempotency_sha256",
            name="uq_annotation_reviewer_idempotency_key",
        ),
    )
    op.create_index(
        "uq_annotation_active_adjudication",
        "annotation_assignments",
        ["item_id"],
        unique=True,
        sqlite_where=sa.text("phase = 'adjudication' AND status = 'leased'"),
        postgresql_where=sa.text("phase = 'adjudication' AND status = 'leased'"),
    )
    op.create_index(
        "ix_annotation_assignment_item_phase_status",
        "annotation_assignments",
        ["item_id", "phase", "status"],
    )
    op.create_index(
        "ix_annotation_assignment_reviewer_status",
        "annotation_assignments",
        ["reviewer_id", "status"],
    )

    op.create_table(
        "annotation_judgments",
        sa.Column("judgment_id", sa.String(length=80), nullable=False),
        sa.Column("assignment_id", sa.String(length=80), nullable=False),
        sa.Column("item_id", sa.String(length=80), nullable=False),
        sa.Column("reviewer_id", sa.String(length=80), nullable=False),
        sa.Column("label_value", sa.String(length=32), nullable=False),
        sa.Column("rationale", sa.Text(), nullable=False),
        sa.Column("evidence_sha256", sa.String(length=64), nullable=False),
        sa.Column("hard_failure_codes", sa.JSON(), nullable=False),
        sa.Column("submitted_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["assignment_id"],
            ["annotation_assignments.assignment_id"],
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["item_id"],
            ["annotation_items.item_id"],
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["reviewer_id"],
            ["annotation_reviewers.reviewer_id"],
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("judgment_id"),
        sa.UniqueConstraint(
            "assignment_id",
            name="uq_annotation_judgment_assignment",
        ),
        sa.UniqueConstraint(
            "item_id",
            "reviewer_id",
            name="uq_annotation_item_reviewer_judgment",
        ),
    )
    op.create_index(
        "ix_annotation_judgment_item",
        "annotation_judgments",
        ["item_id"],
    )

    op.create_table(
        "annotation_adjudications",
        sa.Column("adjudication_id", sa.String(length=80), nullable=False),
        sa.Column("assignment_id", sa.String(length=80), nullable=False),
        sa.Column("item_id", sa.String(length=80), nullable=False),
        sa.Column("adjudicator_reviewer_id", sa.String(length=80), nullable=False),
        sa.Column("final_label_value", sa.String(length=32), nullable=False),
        sa.Column("rationale", sa.Text(), nullable=False),
        sa.Column("evidence_sha256", sa.String(length=64), nullable=False),
        sa.Column("final_hard_failure_codes", sa.JSON(), nullable=False),
        sa.Column("submitted_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["assignment_id"],
            ["annotation_assignments.assignment_id"],
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["item_id"],
            ["annotation_items.item_id"],
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["adjudicator_reviewer_id"],
            ["annotation_reviewers.reviewer_id"],
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("adjudication_id"),
        sa.UniqueConstraint(
            "assignment_id",
            name="uq_annotation_adjudication_assignment",
        ),
        sa.UniqueConstraint(
            "item_id",
            name="uq_annotation_item_adjudication",
        ),
    )

    op.create_table(
        "annotation_exports",
        sa.Column("export_id", sa.String(length=80), nullable=False),
        sa.Column("project_id", sa.String(length=80), nullable=False),
        sa.Column("release_sha256", sa.String(length=64), nullable=False),
        sa.Column("manifest_sha256", sa.String(length=64), nullable=False),
        sa.Column("artifact_directory", sa.Text(), nullable=False),
        sa.Column("created_by_reviewer_id", sa.String(length=80), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["project_id"],
            ["annotation_projects.project_id"],
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["created_by_reviewer_id"],
            ["annotation_reviewers.reviewer_id"],
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("export_id"),
        sa.UniqueConstraint("project_id", name="uq_annotation_project_export"),
        sa.UniqueConstraint("release_sha256"),
    )

    op.create_table(
        "annotation_audit_events",
        sa.Column("event_id", sa.String(length=80), nullable=False),
        sa.Column("project_id", sa.String(length=80), nullable=True),
        sa.Column("item_id", sa.String(length=80), nullable=True),
        sa.Column("actor_reviewer_id", sa.String(length=80), nullable=True),
        sa.Column("event_type", sa.String(length=80), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("payload_sha256", sa.String(length=64), nullable=False),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("event_id"),
    )
    op.create_index(
        "ix_annotation_audit_project_occurred",
        "annotation_audit_events",
        ["project_id", "occurred_at"],
    )
    op.create_index(
        "ix_annotation_audit_item_occurred",
        "annotation_audit_events",
        ["item_id", "occurred_at"],
    )

    _create_append_only_triggers()


def downgrade() -> None:
    """Remove annotation workflow tables and PostgreSQL append-only enforcement."""

    _drop_append_only_triggers()

    op.drop_index(
        "ix_annotation_audit_item_occurred",
        table_name="annotation_audit_events",
    )
    op.drop_index(
        "ix_annotation_audit_project_occurred",
        table_name="annotation_audit_events",
    )
    op.drop_table("annotation_audit_events")
    op.drop_table("annotation_exports")
    op.drop_table("annotation_adjudications")
    op.drop_index("ix_annotation_judgment_item", table_name="annotation_judgments")
    op.drop_table("annotation_judgments")
    op.drop_index(
        "ix_annotation_assignment_reviewer_status",
        table_name="annotation_assignments",
    )
    op.drop_index(
        "ix_annotation_assignment_item_phase_status",
        table_name="annotation_assignments",
    )
    op.drop_index(
        "uq_annotation_active_adjudication",
        table_name="annotation_assignments",
    )
    op.drop_table("annotation_assignments")
    op.drop_index(
        "ix_annotation_item_group_state_priority",
        table_name="annotation_items",
    )
    op.drop_table("annotation_items")
    op.drop_index(
        "ix_annotation_group_project_split",
        table_name="annotation_groups",
    )
    op.drop_table("annotation_groups")
    op.drop_index(
        "ix_annotation_project_status_type",
        table_name="annotation_projects",
    )
    op.drop_table("annotation_projects")
    op.drop_table("annotation_reviewers")
