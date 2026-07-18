"""SQLAlchemy records for durable, independently reviewed annotations."""

from __future__ import annotations
from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    event,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column

from pc_build_recommender.catalog.orm import Base


def _utcnow() -> datetime:
    return datetime.now(UTC)


class AnnotationReviewerRecord(Base):
    __tablename__ = "annotation_reviewers"
    __table_args__ = (
        UniqueConstraint("oidc_issuer", "oidc_subject", name="uq_annotation_reviewer_oidc"),
        CheckConstraint("length(oidc_issuer) > 0", name="ck_annotation_reviewer_issuer"),
        CheckConstraint("length(oidc_subject) > 0", name="ck_annotation_reviewer_subject"),
    )

    reviewer_id: Mapped[str] = mapped_column(String(80), primary_key=True)
    oidc_issuer: Mapped[str] = mapped_column(String(500), nullable=False)
    oidc_subject: Mapped[str] = mapped_column(String(500), nullable=False)
    display_name: Mapped[str] = mapped_column(String(240), nullable=False)
    roles: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    verified_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )


class AnnotationProjectRecord(Base):
    __tablename__ = "annotation_projects"
    __table_args__ = (
        UniqueConstraint("task_type", "dataset_version", name="uq_annotation_dataset_version"),
        CheckConstraint(
            "task_type IN ('entity_resolution', 'relevance')",
            name="ck_annotation_project_task_type",
        ),
        CheckConstraint(
            "status IN ('draft', 'open', 'frozen')",
            name="ck_annotation_project_status",
        ),
        CheckConstraint("required_reviews = 2", name="ck_annotation_project_dual_review"),
        Index("ix_annotation_project_status_type", "status", "task_type"),
    )

    project_id: Mapped[str] = mapped_column(String(80), primary_key=True)
    task_type: Mapped[str] = mapped_column(String(32), nullable=False)
    dataset_name: Mapped[str] = mapped_column(String(240), nullable=False)
    dataset_version: Mapped[str] = mapped_column(String(240), nullable=False)
    rubric_version: Mapped[str] = mapped_column(String(160), nullable=False)
    data_version: Mapped[str] = mapped_column(String(240), nullable=False)
    source_policy: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    source_policy_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    training_eligible: Mapped[bool] = mapped_column(Boolean, nullable=False)
    published_metrics_eligible: Mapped[bool] = mapped_column(Boolean, nullable=False)
    model_serving_eligible: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    required_reviews: Mapped[int] = mapped_column(Integer, nullable=False, default=2)
    split_names: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    status: Mapped[str] = mapped_column(String(24), nullable=False, default="draft")
    created_by_reviewer_id: Mapped[str] = mapped_column(
        ForeignKey("annotation_reviewers.reviewer_id", ondelete="RESTRICT"), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )
    opened_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    frozen_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class AnnotationGroupRecord(Base):
    __tablename__ = "annotation_groups"
    __table_args__ = (
        UniqueConstraint("project_id", "group_key", name="uq_annotation_project_group"),
        Index("ix_annotation_group_project_split", "project_id", "split_name"),
    )

    group_id: Mapped[str] = mapped_column(String(80), primary_key=True)
    project_id: Mapped[str] = mapped_column(
        ForeignKey("annotation_projects.project_id", ondelete="CASCADE"), nullable=False
    )
    group_key: Mapped[str] = mapped_column(String(240), nullable=False)
    leakage_group_id: Mapped[str] = mapped_column(String(240), nullable=False)
    category: Mapped[str] = mapped_column(String(32), nullable=False)
    split_name: Mapped[str] = mapped_column(String(32), nullable=False)
    context_payload: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    context_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    is_synthetic: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )


class AnnotationItemRecord(Base):
    __tablename__ = "annotation_items"
    __table_args__ = (
        UniqueConstraint("group_id", "target_id", name="uq_annotation_group_target"),
        CheckConstraint(
            "state IN ('pending', 'in_review', 'needs_adjudication', 'resolved')",
            name="ck_annotation_item_state",
        ),
        Index("ix_annotation_item_group_state_priority", "group_id", "state", "priority"),
    )

    item_id: Mapped[str] = mapped_column(String(80), primary_key=True)
    group_id: Mapped[str] = mapped_column(
        ForeignKey("annotation_groups.group_id", ondelete="CASCADE"), nullable=False
    )
    target_id: Mapped[str] = mapped_column(String(240), nullable=False)
    evidence_payload: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    evidence_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    priority: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    state: Mapped[str] = mapped_column(String(32), nullable=False, default="pending")
    is_synthetic: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )


class AnnotationAssignmentRecord(Base):
    __tablename__ = "annotation_assignments"
    __table_args__ = (
        CheckConstraint(
            "phase IN ('review', 'adjudication')", name="ck_annotation_assignment_phase"
        ),
        CheckConstraint(
            "status IN ('leased', 'submitted', 'expired')",
            name="ck_annotation_assignment_status",
        ),
        UniqueConstraint(
            "item_id",
            "reviewer_id",
            "phase",
            name="uq_annotation_assignment_item_reviewer_phase",
        ),
        UniqueConstraint("lease_token_sha256", name="uq_annotation_assignment_lease_token"),
        UniqueConstraint(
            "reviewer_id",
            "submission_idempotency_sha256",
            name="uq_annotation_reviewer_idempotency_key",
        ),
        Index(
            "uq_annotation_active_adjudication",
            "item_id",
            unique=True,
            sqlite_where=text("phase = 'adjudication' AND status = 'leased'"),
            postgresql_where=text("phase = 'adjudication' AND status = 'leased'"),
        ),
        Index("ix_annotation_assignment_item_phase_status", "item_id", "phase", "status"),
        Index("ix_annotation_assignment_reviewer_status", "reviewer_id", "status"),
    )

    assignment_id: Mapped[str] = mapped_column(String(80), primary_key=True)
    item_id: Mapped[str] = mapped_column(
        ForeignKey("annotation_items.item_id", ondelete="RESTRICT"), nullable=False
    )
    reviewer_id: Mapped[str] = mapped_column(
        ForeignKey("annotation_reviewers.reviewer_id", ondelete="RESTRICT"), nullable=False
    )
    phase: Mapped[str] = mapped_column(String(24), nullable=False)
    status: Mapped[str] = mapped_column(String(24), nullable=False, default="leased")
    lease_token_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    submission_idempotency_sha256: Mapped[str | None] = mapped_column(String(64))
    submission_payload_sha256: Mapped[str | None] = mapped_column(String(64))
    leased_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    lease_expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    submitted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class AnnotationJudgmentRecord(Base):
    __tablename__ = "annotation_judgments"
    __table_args__ = (
        UniqueConstraint("assignment_id", name="uq_annotation_judgment_assignment"),
        UniqueConstraint("item_id", "reviewer_id", name="uq_annotation_item_reviewer_judgment"),
        Index("ix_annotation_judgment_item", "item_id"),
    )

    judgment_id: Mapped[str] = mapped_column(String(80), primary_key=True)
    assignment_id: Mapped[str] = mapped_column(
        ForeignKey("annotation_assignments.assignment_id", ondelete="RESTRICT"), nullable=False
    )
    item_id: Mapped[str] = mapped_column(
        ForeignKey("annotation_items.item_id", ondelete="RESTRICT"), nullable=False
    )
    reviewer_id: Mapped[str] = mapped_column(
        ForeignKey("annotation_reviewers.reviewer_id", ondelete="RESTRICT"), nullable=False
    )
    label_value: Mapped[str] = mapped_column(String(32), nullable=False)
    rationale: Mapped[str] = mapped_column(Text, nullable=False)
    evidence_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    hard_failure_codes: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    submitted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class AnnotationAdjudicationRecord(Base):
    __tablename__ = "annotation_adjudications"
    __table_args__ = (
        UniqueConstraint("assignment_id", name="uq_annotation_adjudication_assignment"),
        UniqueConstraint("item_id", name="uq_annotation_item_adjudication"),
    )

    adjudication_id: Mapped[str] = mapped_column(String(80), primary_key=True)
    assignment_id: Mapped[str] = mapped_column(
        ForeignKey("annotation_assignments.assignment_id", ondelete="RESTRICT"), nullable=False
    )
    item_id: Mapped[str] = mapped_column(
        ForeignKey("annotation_items.item_id", ondelete="RESTRICT"), nullable=False
    )
    adjudicator_reviewer_id: Mapped[str] = mapped_column(
        ForeignKey("annotation_reviewers.reviewer_id", ondelete="RESTRICT"), nullable=False
    )
    final_label_value: Mapped[str] = mapped_column(String(32), nullable=False)
    rationale: Mapped[str] = mapped_column(Text, nullable=False)
    evidence_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    final_hard_failure_codes: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    submitted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class AnnotationExportRecord(Base):
    __tablename__ = "annotation_exports"
    __table_args__ = (UniqueConstraint("project_id", name="uq_annotation_project_export"),)

    export_id: Mapped[str] = mapped_column(String(80), primary_key=True)
    project_id: Mapped[str] = mapped_column(
        ForeignKey("annotation_projects.project_id", ondelete="RESTRICT"), nullable=False
    )
    release_sha256: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    manifest_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    artifact_directory: Mapped[str] = mapped_column(Text, nullable=False)
    created_by_reviewer_id: Mapped[str] = mapped_column(
        ForeignKey("annotation_reviewers.reviewer_id", ondelete="RESTRICT"), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class AnnotationAuditEventRecord(Base):
    __tablename__ = "annotation_audit_events"
    __table_args__ = (
        Index("ix_annotation_audit_project_occurred", "project_id", "occurred_at"),
        Index("ix_annotation_audit_item_occurred", "item_id", "occurred_at"),
    )

    event_id: Mapped[str] = mapped_column(String(80), primary_key=True)
    project_id: Mapped[str | None] = mapped_column(String(80))
    item_id: Mapped[str | None] = mapped_column(String(80))
    actor_reviewer_id: Mapped[str | None] = mapped_column(String(80))
    event_type: Mapped[str] = mapped_column(String(80), nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    payload_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


def _immutable_record(*_: object, **__: object) -> None:
    raise RuntimeError("human judgments, adjudications, and audit events are append-only")


for _record_type in (
    AnnotationJudgmentRecord,
    AnnotationAdjudicationRecord,
    AnnotationAuditEventRecord,
):
    event.listen(_record_type, "before_update", _immutable_record)
    event.listen(_record_type, "before_delete", _immutable_record)
