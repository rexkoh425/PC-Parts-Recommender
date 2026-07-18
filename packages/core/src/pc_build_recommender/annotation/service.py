"""Transactional annotation service with fail-closed, content-addressed exports."""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import tempfile
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

from sqlalchemy import Select, exists, func, select
from sqlalchemy.orm import Session, sessionmaker

from pc_build_recommender.entity_resolution import (
    CanonicalProductRecord,
    ListingRecord,
    PairExample,
)
from pc_build_recommender.evaluation.manifest import sha256_json
from pc_build_recommender.retrieval import (
    AdjudicationDecision,
    FrozenQueryGroupSplit,
    HumanJudgmentSet,
    LabelingQuery,
    ReviewerJudgment,
)
from pc_build_recommender.retrieval.benchmark import QUERY_SPLIT_SCHEMA_VERSION

from .blinding import validate_blinded_annotation_payload
from .models import (
    AnnotationAuthorizationError,
    AnnotationConflictError,
    AnnotationFreezeBlockedError,
    AnnotationItemState,
    AnnotationProjectProgress,
    AnnotationProjectStatus,
    AnnotationRelease,
    AnnotationRole,
    AnnotationTaskType,
    AssignmentPhase,
    AssignmentStatus,
    ClaimedAnnotationTask,
    EntityResolutionLabel,
    VerifiedOIDCIdentity,
)
from .orm import (
    AnnotationAdjudicationRecord,
    AnnotationAssignmentRecord,
    AnnotationAuditEventRecord,
    AnnotationExportRecord,
    AnnotationGroupRecord,
    AnnotationItemRecord,
    AnnotationJudgmentRecord,
    AnnotationProjectRecord,
    AnnotationReviewerRecord,
)

_ER_SPLITS = ("train", "calibration", "threshold", "test")
_RELEVANCE_SPLITS = ("train", "validation", "test")


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


def _now(value: datetime | None = None) -> datetime:
    result = value or datetime.now(UTC)
    if result.tzinfo is None:
        raise ValueError("workflow timestamps must be timezone-aware")
    return result.astimezone(UTC)


def _aware(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _iso(value: datetime) -> str:
    return _aware(value).isoformat().replace("+00:00", "Z")


def _json_bytes(value: object, *, pretty: bool = False) -> bytes:
    options: dict[str, Any] = {
        "allow_nan": False,
        "ensure_ascii": False,
        "sort_keys": True,
    }
    if pretty:
        options["indent"] = 2
    else:
        options["separators"] = (",", ":")
    return (json.dumps(value, **options) + "\n").encode("utf-8")


def _canonical_mapping(value: Mapping[str, Any], *, field_name: str) -> dict[str, Any]:
    try:
        encoded = _json_bytes(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field_name} must be finite JSON data") from error
    decoded = json.loads(encoded)
    if not isinstance(decoded, dict):
        raise TypeError(f"{field_name} must be a JSON object")
    return decoded


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_payload(value: Mapping[str, Any]) -> str:
    return _sha256_bytes(_json_bytes(value))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _contains_synthetic_marker(value: object) -> bool:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if str(key).strip().casefold() == "is_synthetic" and nested is True:
                return True
            if _contains_synthetic_marker(nested):
                return True
    elif isinstance(value, list | tuple):
        return any(_contains_synthetic_marker(item) for item in value)
    return False


def _validate_roles(roles: Sequence[AnnotationRole | str]) -> list[str]:
    values = sorted({AnnotationRole(role).value for role in roles})
    if not values:
        raise ValueError("at least one annotation role is required")
    return values


def _group_snapshot_payload(
    project: AnnotationProjectRecord,
    *,
    group_key: str,
    leakage_group_id: str,
    category: str,
    split_name: str,
    context_payload: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "pc-build-recommender.annotation-group-snapshot.v1",
        "task_type": project.task_type,
        "dataset_version": project.dataset_version,
        "group_key": group_key,
        "leakage_group_id": leakage_group_id,
        "category": category,
        "split_name": split_name,
        "context_payload": dict(context_payload),
    }


def _item_snapshot_payload(
    group: AnnotationGroupRecord,
    *,
    target_id: str,
    evidence_payload: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "pc-build-recommender.annotation-item-snapshot.v1",
        "context_sha256": group.context_sha256,
        "group_key": group.group_key,
        "target_id": target_id,
        "evidence_payload": dict(evidence_payload),
    }


def _with_skip_locked(
    statement: Select[tuple[AnnotationItemRecord]],
    session: Session,
) -> Select[tuple[AnnotationItemRecord]]:
    bind = session.get_bind()
    if bind.dialect.name == "postgresql":
        return statement.with_for_update(skip_locked=True, of=AnnotationItemRecord)
    return statement


class AnnotationService:
    """Own annotation state transitions and never infer a human label."""

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def bootstrap_administrator(
        self,
        identity: VerifiedOIDCIdentity,
        *,
        display_name: str,
        now: datetime | None = None,
    ) -> str:
        """Create the first administrator from already-verified upstream OIDC claims.

        This is deliberately a one-time operation. PostgreSQL callers serialize on a
        transaction-scoped advisory lock so two empty-database bootstraps cannot both
        succeed.
        """

        timestamp = _now(now)
        if not display_name.strip():
            raise ValueError("reviewer display_name must not be empty")
        with self._session_factory() as session, session.begin():
            if session.get_bind().dialect.name == "postgresql":
                session.execute(select(func.pg_advisory_xact_lock(1_690_764_622)))
            reviewer_count = int(
                session.scalar(select(func.count(AnnotationReviewerRecord.reviewer_id))) or 0
            )
            if reviewer_count:
                raise AnnotationConflictError(
                    "administrator bootstrap is disabled after the first reviewer is created"
                )
            record = AnnotationReviewerRecord(
                reviewer_id=_new_id("ann_reviewer"),
                oidc_issuer=identity.issuer,
                oidc_subject=identity.subject,
                display_name=display_name.strip(),
                roles=[AnnotationRole.ADMIN.value],
                active=True,
                verified_at=timestamp,
                created_at=timestamp,
            )
            session.add(record)
            self._audit(
                session,
                event_type="administrator_bootstrapped",
                actor=record,
                payload={
                    "reviewer_id": record.reviewer_id,
                    "oidc_issuer": record.oidc_issuer,
                },
                occurred_at=timestamp,
            )
            return record.reviewer_id

    def provision_reviewer(
        self,
        actor: VerifiedOIDCIdentity,
        *,
        identity: VerifiedOIDCIdentity,
        display_name: str,
        roles: Sequence[AnnotationRole | str],
        now: datetime | None = None,
    ) -> str:
        timestamp = _now(now)
        if not display_name.strip():
            raise ValueError("reviewer display_name must not be empty")
        with self._session_factory() as session, session.begin():
            administrator = self._require_actor(session, actor, AnnotationRole.ADMIN)
            existing = session.scalar(
                select(AnnotationReviewerRecord).where(
                    AnnotationReviewerRecord.oidc_issuer == identity.issuer,
                    AnnotationReviewerRecord.oidc_subject == identity.subject,
                )
            )
            if existing is not None:
                raise AnnotationConflictError("OIDC identity is already provisioned")
            record = AnnotationReviewerRecord(
                reviewer_id=_new_id("ann_reviewer"),
                oidc_issuer=identity.issuer,
                oidc_subject=identity.subject,
                display_name=display_name.strip(),
                roles=_validate_roles(roles),
                active=True,
                verified_at=timestamp,
                created_at=timestamp,
            )
            session.add(record)
            self._audit(
                session,
                event_type="reviewer_provisioned",
                actor=administrator,
                payload={"reviewer_id": record.reviewer_id, "roles": record.roles},
                occurred_at=timestamp,
            )
            return record.reviewer_id

    def create_project(
        self,
        actor: VerifiedOIDCIdentity,
        *,
        task_type: AnnotationTaskType | str,
        dataset_name: str,
        dataset_version: str,
        rubric_version: str,
        data_version: str,
        source_policy: Mapping[str, Any],
        now: datetime | None = None,
    ) -> str:
        timestamp = _now(now)
        task = AnnotationTaskType(task_type)
        for name, value in (
            ("dataset_name", dataset_name),
            ("dataset_version", dataset_version),
            ("rubric_version", rubric_version),
            ("data_version", data_version),
        ):
            if not value.strip():
                raise ValueError(f"{name} must not be empty")
        policy = _canonical_mapping(source_policy, field_name="source_policy")
        for name in ("training_eligible", "published_metrics_eligible"):
            if not isinstance(policy.get(name), bool):
                raise TypeError(f"source_policy.{name} must be a JSON boolean")
        serving = policy.get("model_serving_eligible", False)
        if not isinstance(serving, bool):
            raise TypeError("source_policy.model_serving_eligible must be a JSON boolean")
        if not isinstance(policy.get("scope_note"), str) or not policy["scope_note"].strip():
            raise ValueError("source_policy.scope_note must be a non-empty string")

        with self._session_factory() as session, session.begin():
            administrator = self._require_actor(session, actor, AnnotationRole.ADMIN)
            project = AnnotationProjectRecord(
                project_id=_new_id("ann_project"),
                task_type=task.value,
                dataset_name=dataset_name.strip(),
                dataset_version=dataset_version.strip(),
                rubric_version=rubric_version.strip(),
                data_version=data_version.strip(),
                source_policy=policy,
                source_policy_sha256=_sha256_payload(policy),
                training_eligible=policy["training_eligible"],
                published_metrics_eligible=policy["published_metrics_eligible"],
                model_serving_eligible=serving,
                required_reviews=2,
                split_names=list(
                    _ER_SPLITS
                    if task is AnnotationTaskType.ENTITY_RESOLUTION
                    else _RELEVANCE_SPLITS
                ),
                status=AnnotationProjectStatus.DRAFT.value,
                created_by_reviewer_id=administrator.reviewer_id,
                created_at=timestamp,
            )
            session.add(project)
            self._audit(
                session,
                event_type="project_created",
                actor=administrator,
                project_id=project.project_id,
                payload={
                    "task_type": task.value,
                    "dataset_version": project.dataset_version,
                    "source_policy_sha256": project.source_policy_sha256,
                },
                occurred_at=timestamp,
            )
            return project.project_id

    def add_group(
        self,
        actor: VerifiedOIDCIdentity,
        project_id: str,
        *,
        group_key: str,
        leakage_group_id: str,
        category: str,
        split_name: str,
        context_payload: Mapping[str, Any],
        is_synthetic: bool = False,
        now: datetime | None = None,
    ) -> str:
        timestamp = _now(now)
        with self._session_factory() as session, session.begin():
            administrator = self._require_actor(session, actor, AnnotationRole.ADMIN)
            project = self._project(session, project_id)
            self._require_project_status(project, AnnotationProjectStatus.DRAFT)
            group = self._make_group_record(
                project,
                group_key=group_key,
                leakage_group_id=leakage_group_id,
                category=category,
                split_name=split_name,
                context_payload=context_payload,
                is_synthetic=is_synthetic,
                timestamp=timestamp,
            )
            session.add(group)
            self._audit(
                session,
                event_type="group_added",
                actor=administrator,
                project_id=project.project_id,
                payload={
                    "group_id": group.group_id,
                    "context_sha256": group.context_sha256,
                    "split_name": split_name,
                },
                occurred_at=timestamp,
            )
            return group.group_id

    def add_item(
        self,
        actor: VerifiedOIDCIdentity,
        project_id: str,
        group_id: str,
        *,
        target_id: str,
        evidence_payload: Mapping[str, Any],
        priority: int = 0,
        is_synthetic: bool = False,
        now: datetime | None = None,
    ) -> str:
        timestamp = _now(now)
        with self._session_factory() as session, session.begin():
            administrator = self._require_actor(session, actor, AnnotationRole.ADMIN)
            project = self._project(session, project_id)
            self._require_project_status(project, AnnotationProjectStatus.DRAFT)
            group = session.get(AnnotationGroupRecord, group_id)
            if group is None or group.project_id != project.project_id:
                raise KeyError(f"unknown annotation group: {group_id}")
            item = self._make_item_record(
                project,
                group,
                target_id=target_id,
                evidence_payload=evidence_payload,
                priority=priority,
                is_synthetic=is_synthetic,
                timestamp=timestamp,
            )
            session.add(item)
            self._audit(
                session,
                event_type="item_added",
                actor=administrator,
                project_id=project.project_id,
                item_id=item.item_id,
                payload={
                    "group_id": group.group_id,
                    "target_id": item.target_id,
                    "evidence_sha256": item.evidence_sha256,
                },
                occurred_at=timestamp,
            )
            return item.item_id

    def import_batch(
        self,
        actor: VerifiedOIDCIdentity,
        project_id: str,
        *,
        groups: Iterable[Mapping[str, Any]],
        now: datetime | None = None,
    ) -> tuple[int, int]:
        """Atomically import a fully validated group/item batch into a draft project."""

        if isinstance(groups, str | bytes | bytearray):
            raise TypeError("annotation batch groups must be an iterable of objects")
        timestamp = _now(now)
        with self._session_factory() as session, session.begin():
            administrator = self._require_actor(session, actor, AnnotationRole.ADMIN)
            project = self._project(session, project_id, for_update=True)
            self._require_project_status(project, AnnotationProjectStatus.DRAFT)
            group_count = 0
            item_count = 0
            for group_index, raw_group in enumerate(groups):
                group_count += 1
                group_payload = _canonical_mapping(
                    raw_group,
                    field_name=f"groups[{group_index}]",
                )
                raw_items = group_payload.pop("items", None)
                if not isinstance(raw_items, list) or not raw_items:
                    raise ValueError(f"groups[{group_index}].items must be a non-empty list")
                group = self._make_group_record(
                    project,
                    group_key=group_payload.get("group_key"),
                    leakage_group_id=group_payload.get("leakage_group_id"),
                    category=group_payload.get("category"),
                    split_name=group_payload.get("split_name"),
                    context_payload=group_payload.get("context_payload"),
                    is_synthetic=group_payload.get("is_synthetic", False),
                    timestamp=timestamp,
                )
                session.add(group)
                session.flush()
                self._audit(
                    session,
                    event_type="group_added",
                    actor=administrator,
                    project_id=project.project_id,
                    payload={
                        "group_id": group.group_id,
                        "context_sha256": group.context_sha256,
                        "split_name": group.split_name,
                    },
                    occurred_at=timestamp,
                )
                for item_index, raw_item in enumerate(raw_items):
                    if not isinstance(raw_item, Mapping):
                        raise TypeError(
                            f"groups[{group_index}].items[{item_index}] must be an object"
                        )
                    item_payload = _canonical_mapping(
                        raw_item,
                        field_name=f"groups[{group_index}].items[{item_index}]",
                    )
                    item = self._make_item_record(
                        project,
                        group,
                        target_id=item_payload.get("target_id"),
                        evidence_payload=item_payload.get("evidence_payload"),
                        priority=item_payload.get("priority", 0),
                        is_synthetic=item_payload.get("is_synthetic", False),
                        timestamp=timestamp,
                    )
                    session.add(item)
                    self._audit(
                        session,
                        event_type="item_added",
                        actor=administrator,
                        project_id=project.project_id,
                        item_id=item.item_id,
                        payload={
                            "group_id": group.group_id,
                            "target_id": item.target_id,
                            "evidence_sha256": item.evidence_sha256,
                        },
                        occurred_at=timestamp,
                    )
                    item_count += 1
            session.flush()
            self._audit(
                session,
                event_type="batch_imported",
                actor=administrator,
                project_id=project.project_id,
                payload={"group_count": group_count, "item_count": item_count},
                occurred_at=timestamp,
            )
            if group_count == 0:
                raise ValueError("annotation batch must contain at least one group")
            return group_count, item_count

    def open_project(
        self,
        actor: VerifiedOIDCIdentity,
        project_id: str,
        *,
        now: datetime | None = None,
    ) -> None:
        timestamp = _now(now)
        with self._session_factory() as session, session.begin():
            administrator = self._require_actor(session, actor, AnnotationRole.ADMIN)
            project = self._project(session, project_id, for_update=True)
            self._require_project_status(project, AnnotationProjectStatus.DRAFT)
            groups = tuple(
                session.scalars(
                    select(AnnotationGroupRecord).where(
                        AnnotationGroupRecord.project_id == project.project_id
                    )
                )
            )
            if not groups:
                raise AnnotationConflictError("cannot open an annotation project without groups")
            group_ids = [group.group_id for group in groups]
            item_count = int(
                session.scalar(
                    select(func.count(AnnotationItemRecord.item_id)).where(
                        AnnotationItemRecord.group_id.in_(group_ids)
                    )
                )
                or 0
            )
            if item_count < len(groups):
                raise AnnotationConflictError("every annotation group needs at least one item")
            project.status = AnnotationProjectStatus.OPEN.value
            project.opened_at = timestamp
            self._audit(
                session,
                event_type="project_opened",
                actor=administrator,
                project_id=project.project_id,
                payload={"group_count": len(groups), "item_count": item_count},
                occurred_at=timestamp,
            )

    def project_progress(
        self,
        actor: VerifiedOIDCIdentity,
        project_id: str,
        *,
        now: datetime | None = None,
    ) -> AnnotationProjectProgress:
        """Return an admin-only, aggregate-only view of collection and freeze readiness.

        The report intentionally uses grouped SQL counts rather than loading blinded evidence,
        submitted labels, or reviewer information into memory.  Its preflight is deliberately
        coarser than ``freeze_project``; the latter remains the only strict release gate.
        """

        timestamp = _now(now)
        with self._session_factory() as session:
            self._require_actor(session, actor, AnnotationRole.ADMIN)
            project = self._project(session, project_id)

            group_count = int(
                session.scalar(
                    select(func.count(AnnotationGroupRecord.group_id)).where(
                        AnnotationGroupRecord.project_id == project.project_id
                    )
                )
                or 0
            )
            synthetic_group_count = int(
                session.scalar(
                    select(func.count(AnnotationGroupRecord.group_id)).where(
                        AnnotationGroupRecord.project_id == project.project_id,
                        AnnotationGroupRecord.is_synthetic.is_(True),
                    )
                )
                or 0
            )
            observed_splits = set(
                session.scalars(
                    select(AnnotationGroupRecord.split_name)
                    .where(AnnotationGroupRecord.project_id == project.project_id)
                    .distinct()
                )
            )
            empty_group_count = int(
                session.scalar(
                    select(func.count()).select_from(
                        select(AnnotationGroupRecord.group_id)
                        .outerjoin(
                            AnnotationItemRecord,
                            AnnotationGroupRecord.group_id == AnnotationItemRecord.group_id,
                        )
                        .where(AnnotationGroupRecord.project_id == project.project_id)
                        .group_by(AnnotationGroupRecord.group_id)
                        .having(func.count(AnnotationItemRecord.item_id) == 0)
                        .subquery()
                    )
                )
                or 0
            )
            item_scope = (
                select(AnnotationItemRecord.item_id)
                .join(
                    AnnotationGroupRecord,
                    AnnotationItemRecord.group_id == AnnotationGroupRecord.group_id,
                )
                .where(AnnotationGroupRecord.project_id == project.project_id)
            )
            item_count = int(
                session.scalar(select(func.count()).select_from(item_scope.subquery())) or 0
            )
            synthetic_item_count = int(
                session.scalar(
                    select(func.count(AnnotationItemRecord.item_id))
                    .join(
                        AnnotationGroupRecord,
                        AnnotationItemRecord.group_id == AnnotationGroupRecord.group_id,
                    )
                    .where(
                        AnnotationGroupRecord.project_id == project.project_id,
                        AnnotationItemRecord.is_synthetic.is_(True),
                    )
                )
                or 0
            )

            item_state_counts = {state.value: 0 for state in AnnotationItemState}
            for state, count in session.execute(
                select(AnnotationItemRecord.state, func.count(AnnotationItemRecord.item_id))
                .join(
                    AnnotationGroupRecord,
                    AnnotationItemRecord.group_id == AnnotationGroupRecord.group_id,
                )
                .where(AnnotationGroupRecord.project_id == project.project_id)
                .group_by(AnnotationItemRecord.state)
            ):
                item_state_counts[str(state)] = int(count)

            judgment_counts = (
                select(
                    AnnotationJudgmentRecord.item_id.label("item_id"),
                    func.count(AnnotationJudgmentRecord.judgment_id).label("judgment_count"),
                )
                .join(
                    AnnotationItemRecord,
                    AnnotationJudgmentRecord.item_id == AnnotationItemRecord.item_id,
                )
                .join(
                    AnnotationGroupRecord,
                    AnnotationItemRecord.group_id == AnnotationGroupRecord.group_id,
                )
                .where(AnnotationGroupRecord.project_id == project.project_id)
                .group_by(AnnotationJudgmentRecord.item_id)
                .subquery()
            )
            coverage_by_count = {
                int(judgment_count): int(item_count_for_coverage)
                for judgment_count, item_count_for_coverage in session.execute(
                    select(
                        judgment_counts.c.judgment_count,
                        func.count(judgment_counts.c.item_id),
                    ).group_by(judgment_counts.c.judgment_count)
                )
            }
            nonzero_judgment_items = sum(coverage_by_count.values())
            judgment_coverage = {
                "zero_judgments": item_count - nonzero_judgment_items,
                "one_judgment": coverage_by_count.get(1, 0),
                "two_or_more_judgments": sum(
                    count for count_key, count in coverage_by_count.items() if count_key >= 2
                ),
            }

            assignment_rows = session.execute(
                select(
                    AnnotationAssignmentRecord.phase,
                    AnnotationAssignmentRecord.status,
                    func.count(AnnotationAssignmentRecord.assignment_id),
                )
                .join(
                    AnnotationItemRecord,
                    AnnotationAssignmentRecord.item_id == AnnotationItemRecord.item_id,
                )
                .join(
                    AnnotationGroupRecord,
                    AnnotationItemRecord.group_id == AnnotationGroupRecord.group_id,
                )
                .where(AnnotationGroupRecord.project_id == project.project_id)
                .group_by(
                    AnnotationAssignmentRecord.phase,
                    AnnotationAssignmentRecord.status,
                )
            )
            assignment_by_phase_status = {
                (str(phase), str(status)): int(count) for phase, status, count in assignment_rows
            }
            active_lease_rows = session.execute(
                select(
                    AnnotationAssignmentRecord.phase,
                    func.count(AnnotationAssignmentRecord.assignment_id),
                )
                .join(
                    AnnotationItemRecord,
                    AnnotationAssignmentRecord.item_id == AnnotationItemRecord.item_id,
                )
                .join(
                    AnnotationGroupRecord,
                    AnnotationItemRecord.group_id == AnnotationGroupRecord.group_id,
                )
                .where(
                    AnnotationGroupRecord.project_id == project.project_id,
                    AnnotationAssignmentRecord.status == AssignmentStatus.LEASED.value,
                    AnnotationAssignmentRecord.lease_expires_at > timestamp,
                )
                .group_by(AnnotationAssignmentRecord.phase)
            )
            active_leases = {str(phase): int(count) for phase, count in active_lease_rows}

            def assignment_counts(phase: AssignmentPhase) -> dict[str, int]:
                leased = assignment_by_phase_status.get(
                    (phase.value, AssignmentStatus.LEASED.value), 0
                )
                active = active_leases.get(phase.value, 0)
                return {
                    "active_leased": active,
                    "elapsed_leased": leased - active,
                    "submitted": assignment_by_phase_status.get(
                        (phase.value, AssignmentStatus.SUBMITTED.value), 0
                    ),
                    "expired_record": assignment_by_phase_status.get(
                        (phase.value, AssignmentStatus.EXPIRED.value), 0
                    ),
                }

            review_assignment_counts = assignment_counts(AssignmentPhase.REVIEW)
            adjudication_assignment_counts = assignment_counts(AssignmentPhase.ADJUDICATION)
            adjudication_required_count = item_state_counts[
                AnnotationItemState.NEEDS_ADJUDICATION.value
            ]
            adjudication_completed_count = int(
                session.scalar(
                    select(func.count(AnnotationAdjudicationRecord.adjudication_id))
                    .join(
                        AnnotationItemRecord,
                        AnnotationAdjudicationRecord.item_id == AnnotationItemRecord.item_id,
                    )
                    .join(
                        AnnotationGroupRecord,
                        AnnotationItemRecord.group_id == AnnotationGroupRecord.group_id,
                    )
                    .where(AnnotationGroupRecord.project_id == project.project_id)
                )
                or 0
            )
            release_record_present = (
                session.scalar(
                    select(AnnotationExportRecord.export_id).where(
                        AnnotationExportRecord.project_id == project.project_id
                    )
                )
                is not None
            )

            blockers: list[str] = []
            if project.status != AnnotationProjectStatus.OPEN.value:
                blockers.append(f"project status is {project.status!r}, not 'open'")
            if project.training_eligible is not True:
                blockers.append("source rights do not permit model training")
            if project.published_metrics_eligible is not True:
                blockers.append("source rights do not permit published model metrics")
            if project.required_reviews != 2:
                blockers.append("project is not configured for exactly two independent reviews")
            try:
                policy_hash_matches = (
                    _sha256_payload(
                        _canonical_mapping(project.source_policy, field_name="stored source_policy")
                    )
                    == project.source_policy_sha256
                )
            except (TypeError, ValueError):
                policy_hash_matches = False
            if not policy_hash_matches:
                blockers.append("source policy hash does not match the stored policy")
            if group_count == 0:
                blockers.append("project has no annotation groups")
            if item_count == 0:
                blockers.append("project has no annotation items")
            if empty_group_count:
                blockers.append(f"{empty_group_count} annotation group(s) lack candidate items")
            expected_splits = set(project.split_names)
            if observed_splits != expected_splits:
                blockers.append(
                    "frozen split coverage mismatch: "
                    f"expected={sorted(expected_splits)!r}, observed={sorted(observed_splits)!r}"
                )
            if synthetic_group_count or synthetic_item_count:
                blockers.append("synthetic annotation evidence is present")
            incomplete_review_count = (
                judgment_coverage["zero_judgments"] + judgment_coverage["one_judgment"]
            )
            if incomplete_review_count:
                blockers.append(f"{incomplete_review_count} item(s) lack two independent judgments")
            unresolved_count = item_count - item_state_counts[AnnotationItemState.RESOLVED.value]
            if unresolved_count:
                blockers.append(f"{unresolved_count} item(s) are not resolved")
            if adjudication_required_count:
                blockers.append(f"{adjudication_required_count} item(s) await adjudication")

            return AnnotationProjectProgress(
                project_id=project.project_id,
                project_status=AnnotationProjectStatus(project.status),
                task_type=AnnotationTaskType(project.task_type),
                observed_at=timestamp,
                group_count=group_count,
                item_count=item_count,
                item_state_counts=item_state_counts,
                judgment_coverage=judgment_coverage,
                review_assignment_counts=review_assignment_counts,
                adjudication_assignment_counts=adjudication_assignment_counts,
                adjudication_required_count=adjudication_required_count,
                adjudication_completed_count=adjudication_completed_count,
                synthetic_group_count=synthetic_group_count,
                synthetic_item_count=synthetic_item_count,
                preflight_blockers=tuple(blockers),
                coarse_freeze_preflight_passes=not blockers,
                release_record_present=release_record_present,
            )

    def claim_review(
        self,
        actor: VerifiedOIDCIdentity,
        project_id: str,
        *,
        lease_seconds: int = 900,
        now: datetime | None = None,
    ) -> ClaimedAnnotationTask | None:
        return self._claim(
            actor,
            project_id,
            phase=AssignmentPhase.REVIEW,
            lease_seconds=lease_seconds,
            now=now,
        )

    def claim_adjudication(
        self,
        actor: VerifiedOIDCIdentity,
        project_id: str,
        *,
        lease_seconds: int = 900,
        now: datetime | None = None,
    ) -> ClaimedAnnotationTask | None:
        return self._claim(
            actor,
            project_id,
            phase=AssignmentPhase.ADJUDICATION,
            lease_seconds=lease_seconds,
            now=now,
        )

    def submit_judgment(
        self,
        actor: VerifiedOIDCIdentity,
        assignment_id: str,
        *,
        lease_token: str,
        idempotency_key: str,
        evidence_sha256: str,
        label: str | int,
        rationale: str,
        hard_failure_codes: Sequence[str] = (),
        now: datetime | None = None,
    ) -> str:
        timestamp = _now(now)
        if not rationale.strip():
            raise ValueError("human judgment rationale must not be empty")
        with self._session_factory() as session, session.begin():
            reviewer = self._require_actor(session, actor, AnnotationRole.REVIEWER)
            assignment = self._assignment(session, assignment_id, for_update=True)
            self._validate_assignment_actor(
                assignment,
                reviewer,
                AssignmentPhase.REVIEW,
            )
            item, _, project = self._item_context(session, assignment.item_id)
            if evidence_sha256 != item.evidence_sha256:
                raise AnnotationConflictError("annotation evidence snapshot hash mismatch")
            label_value = self._normalise_label(project, label, final=False)
            failure_codes = self._normalise_hard_failure_codes(
                project,
                hard_failure_codes,
                label_value=label_value,
            )
            submission_payload_sha256 = _sha256_payload(
                {
                    "assignment_id": assignment.assignment_id,
                    "evidence_sha256": evidence_sha256,
                    "label_value": label_value,
                    "rationale": rationale.strip(),
                    "hard_failure_codes": failure_codes,
                }
            )
            repeated = self._idempotent_submission(
                session,
                assignment,
                lease_token=lease_token,
                idempotency_key=idempotency_key,
                submission_payload_sha256=submission_payload_sha256,
                decision_type="judgment",
            )
            if repeated is not None:
                return repeated
            self._validate_live_lease(assignment, lease_token, timestamp)
            if (
                session.scalar(
                    select(AnnotationJudgmentRecord.judgment_id).where(
                        AnnotationJudgmentRecord.item_id == item.item_id,
                        AnnotationJudgmentRecord.reviewer_id == reviewer.reviewer_id,
                    )
                )
                is not None
            ):
                raise AnnotationConflictError("reviewer already submitted an immutable judgment")
            judgment = AnnotationJudgmentRecord(
                judgment_id=_new_id("ann_judgment"),
                assignment_id=assignment.assignment_id,
                item_id=item.item_id,
                reviewer_id=reviewer.reviewer_id,
                label_value=label_value,
                rationale=rationale.strip(),
                evidence_sha256=item.evidence_sha256,
                hard_failure_codes=failure_codes,
                submitted_at=timestamp,
            )
            session.add(judgment)
            assignment.status = AssignmentStatus.SUBMITTED.value
            assignment.submission_idempotency_sha256 = _sha256_text(idempotency_key.strip())
            assignment.submission_payload_sha256 = submission_payload_sha256
            assignment.submitted_at = timestamp
            session.flush()
            self._refresh_item_state(session, item, timestamp)
            self._audit(
                session,
                event_type="judgment_submitted",
                actor=reviewer,
                project_id=project.project_id,
                item_id=item.item_id,
                payload={
                    "judgment_id": judgment.judgment_id,
                    "assignment_id": assignment.assignment_id,
                    "label_value": label_value,
                    "hard_failure_codes": failure_codes,
                    "evidence_sha256": item.evidence_sha256,
                },
                occurred_at=timestamp,
            )
            return judgment.judgment_id

    def submit_adjudication(
        self,
        actor: VerifiedOIDCIdentity,
        assignment_id: str,
        *,
        lease_token: str,
        idempotency_key: str,
        evidence_sha256: str,
        final_label: str | int,
        rationale: str,
        final_hard_failure_codes: Sequence[str] = (),
        now: datetime | None = None,
    ) -> str:
        timestamp = _now(now)
        if not rationale.strip():
            raise ValueError("adjudication rationale must not be empty")
        with self._session_factory() as session, session.begin():
            adjudicator = self._require_actor(session, actor, AnnotationRole.ADJUDICATOR)
            assignment = self._assignment(session, assignment_id, for_update=True)
            self._validate_assignment_actor(
                assignment,
                adjudicator,
                AssignmentPhase.ADJUDICATION,
            )
            item, _, project = self._item_context(session, assignment.item_id)
            if evidence_sha256 != item.evidence_sha256:
                raise AnnotationConflictError("annotation evidence snapshot hash mismatch")
            reviewer_ids = set(
                session.scalars(
                    select(AnnotationJudgmentRecord.reviewer_id).where(
                        AnnotationJudgmentRecord.item_id == item.item_id
                    )
                )
            )
            if adjudicator.reviewer_id in reviewer_ids:
                raise AnnotationAuthorizationError(
                    "an item reviewer cannot independently adjudicate the same item"
                )
            label_value = self._normalise_label(project, final_label, final=True)
            failure_codes = self._normalise_hard_failure_codes(
                project,
                final_hard_failure_codes,
                label_value=label_value,
            )
            submission_payload_sha256 = _sha256_payload(
                {
                    "assignment_id": assignment.assignment_id,
                    "evidence_sha256": evidence_sha256,
                    "final_label_value": label_value,
                    "rationale": rationale.strip(),
                    "final_hard_failure_codes": failure_codes,
                }
            )
            repeated = self._idempotent_submission(
                session,
                assignment,
                lease_token=lease_token,
                idempotency_key=idempotency_key,
                submission_payload_sha256=submission_payload_sha256,
                decision_type="adjudication",
            )
            if repeated is not None:
                return repeated
            if item.state != AnnotationItemState.NEEDS_ADJUDICATION.value:
                raise AnnotationConflictError("item no longer requires adjudication")
            self._validate_live_lease(assignment, lease_token, timestamp)
            if (
                session.scalar(
                    select(AnnotationAdjudicationRecord.adjudication_id).where(
                        AnnotationAdjudicationRecord.item_id == item.item_id
                    )
                )
                is not None
            ):
                raise AnnotationConflictError("item already has an immutable adjudication")
            decision = AnnotationAdjudicationRecord(
                adjudication_id=_new_id("ann_adjudication"),
                assignment_id=assignment.assignment_id,
                item_id=item.item_id,
                adjudicator_reviewer_id=adjudicator.reviewer_id,
                final_label_value=label_value,
                rationale=rationale.strip(),
                evidence_sha256=item.evidence_sha256,
                final_hard_failure_codes=failure_codes,
                submitted_at=timestamp,
            )
            session.add(decision)
            assignment.status = AssignmentStatus.SUBMITTED.value
            assignment.submission_idempotency_sha256 = _sha256_text(idempotency_key.strip())
            assignment.submission_payload_sha256 = submission_payload_sha256
            assignment.submitted_at = timestamp
            item.state = AnnotationItemState.RESOLVED.value
            self._audit(
                session,
                event_type="adjudication_submitted",
                actor=adjudicator,
                project_id=project.project_id,
                item_id=item.item_id,
                payload={
                    "adjudication_id": decision.adjudication_id,
                    "assignment_id": assignment.assignment_id,
                    "final_label_value": label_value,
                    "final_hard_failure_codes": failure_codes,
                    "evidence_sha256": item.evidence_sha256,
                },
                occurred_at=timestamp,
            )
            return decision.adjudication_id

    def freeze_project(
        self,
        actor: VerifiedOIDCIdentity,
        project_id: str,
        *,
        output_root: str | Path,
        now: datetime | None = None,
    ) -> AnnotationRelease:
        timestamp = _now(now)
        root = Path(output_root).resolve()
        with self._session_factory() as session, session.begin():
            administrator = self._require_actor(session, actor, AnnotationRole.ADMIN)
            project = self._project(session, project_id, for_update=True)
            existing = session.scalar(
                select(AnnotationExportRecord).where(
                    AnnotationExportRecord.project_id == project.project_id
                )
            )
            if project.status == AnnotationProjectStatus.FROZEN.value:
                if existing is None:
                    raise AnnotationConflictError("frozen project has no export record")
                return self._load_existing_release(existing)
            self._require_project_status(project, AnnotationProjectStatus.OPEN)
            groups = tuple(
                session.scalars(
                    select(AnnotationGroupRecord)
                    .where(AnnotationGroupRecord.project_id == project.project_id)
                    .order_by(AnnotationGroupRecord.group_key, AnnotationGroupRecord.group_id)
                )
            )
            group_ids = [group.group_id for group in groups]
            items = (
                tuple(
                    session.scalars(
                        select(AnnotationItemRecord)
                        .where(AnnotationItemRecord.group_id.in_(group_ids))
                        .order_by(
                            AnnotationItemRecord.group_id,
                            AnnotationItemRecord.target_id,
                            AnnotationItemRecord.item_id,
                        )
                    )
                )
                if group_ids
                else ()
            )
            judgments = (
                tuple(
                    session.scalars(
                        select(AnnotationJudgmentRecord)
                        .where(
                            AnnotationJudgmentRecord.item_id.in_([item.item_id for item in items])
                        )
                        .order_by(
                            AnnotationJudgmentRecord.item_id,
                            AnnotationJudgmentRecord.reviewer_id,
                        )
                    )
                )
                if items
                else ()
            )
            adjudications = (
                tuple(
                    session.scalars(
                        select(AnnotationAdjudicationRecord)
                        .where(
                            AnnotationAdjudicationRecord.item_id.in_(
                                [item.item_id for item in items]
                            )
                        )
                        .order_by(AnnotationAdjudicationRecord.item_id)
                    )
                )
                if items
                else ()
            )
            final_labels = self._freeze_gate(
                project,
                groups,
                items,
                judgments,
                adjudications,
            )
            files = self._export_files(
                project,
                groups,
                items,
                judgments,
                adjudications,
                final_labels,
            )
            release = self._write_release(root, project, files)
            export_record = AnnotationExportRecord(
                export_id=_new_id("ann_export"),
                project_id=project.project_id,
                release_sha256=release.release_sha256,
                manifest_sha256=release.manifest_sha256,
                artifact_directory=str(release.artifact_directory),
                created_by_reviewer_id=administrator.reviewer_id,
                created_at=timestamp,
            )
            session.add(export_record)
            project.status = AnnotationProjectStatus.FROZEN.value
            project.frozen_at = timestamp
            self._audit(
                session,
                event_type="project_frozen",
                actor=administrator,
                project_id=project.project_id,
                payload={
                    "release_sha256": release.release_sha256,
                    "manifest_sha256": release.manifest_sha256,
                    "files": dict(release.files),
                },
                occurred_at=timestamp,
            )
            return release

    @staticmethod
    def _make_group_record(
        project: AnnotationProjectRecord,
        *,
        group_key: object,
        leakage_group_id: object,
        category: object,
        split_name: object,
        context_payload: object,
        is_synthetic: object,
        timestamp: datetime,
    ) -> AnnotationGroupRecord:
        identifiers: list[str] = []
        for name, value in (
            ("group_key", group_key),
            ("leakage_group_id", leakage_group_id),
            ("category", category),
            ("split_name", split_name),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
            identifiers.append(value.strip())
        clean_group_key, clean_leakage_id, clean_category, clean_split = identifiers
        clean_category = clean_category.casefold()
        if not isinstance(is_synthetic, bool):
            raise TypeError("is_synthetic must be a boolean")
        if not isinstance(context_payload, Mapping):
            raise TypeError("context_payload must be an object")
        context = _canonical_mapping(context_payload, field_name="context_payload")
        validate_blinded_annotation_payload(context, path="context_payload")
        if clean_split not in project.split_names:
            raise ValueError(f"split_name must be one of the project splits: {project.split_names}")
        task_type = AnnotationTaskType(project.task_type)
        if task_type is AnnotationTaskType.RELEVANCE:
            for field in ("query_text", "structured_constraints"):
                if field not in context:
                    raise ValueError(f"relevance context requires {field}")
            if not isinstance(context["query_text"], str) or not context["query_text"].strip():
                raise ValueError("relevance query_text must be a non-empty string")
            if not isinstance(context["structured_constraints"], Mapping):
                raise TypeError("relevance structured_constraints must be an object")
        else:
            if not isinstance(context.get("listing"), Mapping):
                raise TypeError("entity-resolution context requires a listing object")
            listing = ListingRecord.from_dict(context["listing"])
            if listing.listing_id != clean_group_key:
                raise ValueError("entity-resolution group_key must equal listing.listing_id")
            if listing.category.casefold() != clean_category:
                raise ValueError("entity-resolution group category must equal listing.category")
        snapshot = _group_snapshot_payload(
            project,
            group_key=clean_group_key,
            leakage_group_id=clean_leakage_id,
            category=clean_category,
            split_name=clean_split,
            context_payload=context,
        )
        return AnnotationGroupRecord(
            group_id=_new_id("ann_group"),
            project_id=project.project_id,
            group_key=clean_group_key,
            leakage_group_id=clean_leakage_id,
            category=clean_category,
            split_name=clean_split,
            context_payload=context,
            context_sha256=_sha256_payload(snapshot),
            is_synthetic=is_synthetic or _contains_synthetic_marker(context),
            created_at=timestamp,
        )

    @staticmethod
    def _make_item_record(
        project: AnnotationProjectRecord,
        group: AnnotationGroupRecord,
        *,
        target_id: object,
        evidence_payload: object,
        priority: object,
        is_synthetic: object,
        timestamp: datetime,
    ) -> AnnotationItemRecord:
        if not isinstance(target_id, str) or not target_id.strip():
            raise ValueError("target_id must be a non-empty string")
        clean_target_id = target_id.strip()
        if not isinstance(priority, int) or isinstance(priority, bool):
            raise TypeError("priority must be an integer")
        if not isinstance(is_synthetic, bool):
            raise TypeError("is_synthetic must be a boolean")
        if not isinstance(evidence_payload, Mapping):
            raise TypeError("evidence_payload must be an object")
        evidence = _canonical_mapping(evidence_payload, field_name="evidence_payload")
        validate_blinded_annotation_payload(evidence, path="evidence_payload")
        if AnnotationTaskType(project.task_type) is AnnotationTaskType.ENTITY_RESOLUTION:
            if not isinstance(evidence.get("product"), Mapping):
                raise TypeError("entity-resolution evidence requires a product object")
            product = CanonicalProductRecord.from_dict(evidence["product"])
            if product.product_id != clean_target_id:
                raise ValueError("entity-resolution target_id must equal product.product_id")
            if product.category.casefold() != group.category:
                raise ValueError("entity-resolution product category must equal group category")
        snapshot = _item_snapshot_payload(
            group,
            target_id=clean_target_id,
            evidence_payload=evidence,
        )
        return AnnotationItemRecord(
            item_id=_new_id("ann_item"),
            group_id=group.group_id,
            target_id=clean_target_id,
            evidence_payload=evidence,
            evidence_sha256=_sha256_payload(snapshot),
            priority=priority,
            state=AnnotationItemState.PENDING.value,
            is_synthetic=is_synthetic or _contains_synthetic_marker(evidence),
            created_at=timestamp,
        )

    def _claim(
        self,
        actor: VerifiedOIDCIdentity,
        project_id: str,
        *,
        phase: AssignmentPhase,
        lease_seconds: int,
        now: datetime | None,
    ) -> ClaimedAnnotationTask | None:
        timestamp = _now(now)
        if not isinstance(lease_seconds, int) or isinstance(lease_seconds, bool):
            raise TypeError("lease_seconds must be an integer")
        if not 30 <= lease_seconds <= 3600:
            raise ValueError("lease_seconds must be between 30 and 3600")
        required_role = (
            AnnotationRole.REVIEWER
            if phase is AssignmentPhase.REVIEW
            else AnnotationRole.ADJUDICATOR
        )
        with self._session_factory() as session, session.begin():
            reviewer = self._require_actor(session, actor, required_role)
            project = self._project(session, project_id)
            self._require_project_status(project, AnnotationProjectStatus.OPEN)
            self._expire_leases(session, timestamp)
            statement = self._claim_statement(
                project,
                reviewer,
                phase=phase,
                now=timestamp,
            )
            item = session.scalars(_with_skip_locked(statement, session)).first()
            if item is None:
                return None
            group = session.get(AnnotationGroupRecord, item.group_id)
            assert group is not None
            lease_token = secrets.token_urlsafe(32)
            assignment = AnnotationAssignmentRecord(
                assignment_id=_new_id("ann_assignment"),
                item_id=item.item_id,
                reviewer_id=reviewer.reviewer_id,
                phase=phase.value,
                status=AssignmentStatus.LEASED.value,
                lease_token_sha256=_sha256_text(lease_token),
                leased_at=timestamp,
                lease_expires_at=timestamp + timedelta(seconds=lease_seconds),
            )
            session.add(assignment)
            if phase is AssignmentPhase.REVIEW:
                item.state = AnnotationItemState.IN_REVIEW.value
            self._audit(
                session,
                event_type=f"{phase.value}_claimed",
                actor=reviewer,
                project_id=project.project_id,
                item_id=item.item_id,
                payload={
                    "assignment_id": assignment.assignment_id,
                    "lease_expires_at": _iso(assignment.lease_expires_at),
                },
                occurred_at=timestamp,
            )
            return ClaimedAnnotationTask(
                assignment_id=assignment.assignment_id,
                lease_token=lease_token,
                project_id=project.project_id,
                item_id=item.item_id,
                task_type=AnnotationTaskType(project.task_type),
                group_key=group.group_key,
                target_id=item.target_id,
                category=group.category,
                context_payload=_canonical_mapping(
                    group.context_payload, field_name="stored context_payload"
                ),
                evidence_payload=_canonical_mapping(
                    item.evidence_payload, field_name="stored evidence_payload"
                ),
                context_sha256=group.context_sha256,
                evidence_sha256=item.evidence_sha256,
                lease_expires_at=assignment.lease_expires_at,
            )

    def _claim_statement(
        self,
        project: AnnotationProjectRecord,
        reviewer: AnnotationReviewerRecord,
        *,
        phase: AssignmentPhase,
        now: datetime,
    ) -> Select[tuple[AnnotationItemRecord]]:
        statement = (
            select(AnnotationItemRecord)
            .join(AnnotationGroupRecord)
            .where(AnnotationGroupRecord.project_id == project.project_id)
        )
        prior_assignment = exists().where(
            AnnotationAssignmentRecord.item_id == AnnotationItemRecord.item_id,
            AnnotationAssignmentRecord.phase == phase.value,
            AnnotationAssignmentRecord.reviewer_id == reviewer.reviewer_id,
        )
        if phase is AssignmentPhase.REVIEW:
            judgment_count = (
                select(func.count(AnnotationJudgmentRecord.judgment_id))
                .where(AnnotationJudgmentRecord.item_id == AnnotationItemRecord.item_id)
                .correlate(AnnotationItemRecord)
                .scalar_subquery()
            )
            active_count = (
                select(func.count(AnnotationAssignmentRecord.assignment_id))
                .where(
                    AnnotationAssignmentRecord.item_id == AnnotationItemRecord.item_id,
                    AnnotationAssignmentRecord.phase == AssignmentPhase.REVIEW.value,
                    AnnotationAssignmentRecord.status == AssignmentStatus.LEASED.value,
                    AnnotationAssignmentRecord.lease_expires_at > now,
                )
                .correlate(AnnotationItemRecord)
                .scalar_subquery()
            )
            reviewer_judged = exists().where(
                AnnotationJudgmentRecord.item_id == AnnotationItemRecord.item_id,
                AnnotationJudgmentRecord.reviewer_id == reviewer.reviewer_id,
            )
            statement = statement.where(
                AnnotationItemRecord.state.in_(
                    (AnnotationItemState.PENDING.value, AnnotationItemState.IN_REVIEW.value)
                ),
                judgment_count < project.required_reviews,
                active_count + judgment_count < project.required_reviews,
                ~reviewer_judged,
                ~prior_assignment,
            )
        else:
            reviewer_judged = exists().where(
                AnnotationJudgmentRecord.item_id == AnnotationItemRecord.item_id,
                AnnotationJudgmentRecord.reviewer_id == reviewer.reviewer_id,
            )
            already_adjudicated = exists().where(
                AnnotationAdjudicationRecord.item_id == AnnotationItemRecord.item_id
            )
            statement = statement.where(
                AnnotationItemRecord.state == AnnotationItemState.NEEDS_ADJUDICATION.value,
                ~reviewer_judged,
                ~already_adjudicated,
                ~prior_assignment,
            )
        return statement.order_by(
            AnnotationItemRecord.priority.desc(), AnnotationItemRecord.item_id
        ).limit(1)

    def _expire_leases(self, session: Session, now: datetime) -> None:
        expired = tuple(
            session.scalars(
                select(AnnotationAssignmentRecord).where(
                    AnnotationAssignmentRecord.status == AssignmentStatus.LEASED.value,
                    AnnotationAssignmentRecord.lease_expires_at <= now,
                )
            )
        )
        for assignment in expired:
            assignment.status = AssignmentStatus.EXPIRED.value
            item, _, project = self._item_context(session, assignment.item_id)
            if assignment.phase == AssignmentPhase.REVIEW.value:
                self._refresh_item_state(session, item, now)
            self._audit(
                session,
                event_type="assignment_expired",
                project_id=project.project_id,
                item_id=item.item_id,
                payload={"assignment_id": assignment.assignment_id},
                occurred_at=now,
            )

    def _refresh_item_state(
        self,
        session: Session,
        item: AnnotationItemRecord,
        now: datetime,
    ) -> None:
        judgments = tuple(
            session.scalars(
                select(AnnotationJudgmentRecord).where(
                    AnnotationJudgmentRecord.item_id == item.item_id
                )
            )
        )
        if len(judgments) >= 2:
            labels = {judgment.label_value for judgment in judgments}
            if labels == {EntityResolutionLabel.UNCERTAIN.value} or len(labels) != 1:
                item.state = AnnotationItemState.NEEDS_ADJUDICATION.value
            else:
                item.state = AnnotationItemState.RESOLVED.value
            return
        active = int(
            session.scalar(
                select(func.count(AnnotationAssignmentRecord.assignment_id)).where(
                    AnnotationAssignmentRecord.item_id == item.item_id,
                    AnnotationAssignmentRecord.phase == AssignmentPhase.REVIEW.value,
                    AnnotationAssignmentRecord.status == AssignmentStatus.LEASED.value,
                    AnnotationAssignmentRecord.lease_expires_at > now,
                )
            )
            or 0
        )
        item.state = (
            AnnotationItemState.IN_REVIEW.value
            if judgments or active
            else AnnotationItemState.PENDING.value
        )

    def _freeze_gate(
        self,
        project: AnnotationProjectRecord,
        groups: Sequence[AnnotationGroupRecord],
        items: Sequence[AnnotationItemRecord],
        judgments: Sequence[AnnotationJudgmentRecord],
        adjudications: Sequence[AnnotationAdjudicationRecord],
    ) -> dict[str, str]:
        reasons: list[str] = []
        policy = _canonical_mapping(project.source_policy, field_name="stored source_policy")
        if _sha256_payload(policy) != project.source_policy_sha256:
            reasons.append("source policy hash does not match the stored policy")
        if policy.get("training_eligible") is not True or not project.training_eligible:
            reasons.append("source rights do not permit model training")
        if (
            policy.get("published_metrics_eligible") is not True
            or not project.published_metrics_eligible
        ):
            reasons.append("source rights do not permit published model metrics")
        if project.required_reviews != 2:
            reasons.append("project is not configured for exactly two independent reviews")
        if not groups:
            reasons.append("project has no annotation groups")
        if not items:
            reasons.append("project has no annotation items")
        if any(group.is_synthetic for group in groups) or any(item.is_synthetic for item in items):
            reasons.append("synthetic annotation evidence is present")

        groups_by_id = {group.group_id: group for group in groups}
        items_by_group: dict[str, list[AnnotationItemRecord]] = defaultdict(list)
        for item in items:
            items_by_group[item.group_id].append(item)
        for group in groups:
            if not items_by_group[group.group_id]:
                reasons.append(f"group {group.group_key!r} has no candidate items")
            expected = _sha256_payload(
                _group_snapshot_payload(
                    project,
                    group_key=group.group_key,
                    leakage_group_id=group.leakage_group_id,
                    category=group.category,
                    split_name=group.split_name,
                    context_payload=group.context_payload,
                )
            )
            if expected != group.context_sha256:
                reasons.append(f"group {group.group_key!r} context snapshot hash mismatch")
        for item in items:
            group = groups_by_id[item.group_id]
            expected = _sha256_payload(
                _item_snapshot_payload(
                    group,
                    target_id=item.target_id,
                    evidence_payload=item.evidence_payload,
                )
            )
            if expected != item.evidence_sha256:
                reasons.append(f"item {item.item_id!r} evidence snapshot hash mismatch")

        required_splits = set(project.split_names)
        observed_splits = {group.split_name for group in groups}
        if observed_splits != required_splits:
            reasons.append(
                "frozen split coverage mismatch: "
                f"expected={sorted(required_splits)!r}, observed={sorted(observed_splits)!r}"
            )
        split_by_leakage_group: dict[str, str] = {}
        for group in groups:
            previous = split_by_leakage_group.setdefault(group.leakage_group_id, group.split_name)
            if previous != group.split_name:
                reasons.append(f"leakage group {group.leakage_group_id!r} crosses frozen splits")

        judgments_by_item: dict[str, list[AnnotationJudgmentRecord]] = defaultdict(list)
        for judgment in judgments:
            judgments_by_item[judgment.item_id].append(judgment)
        adjudication_by_item = {item.item_id: item for item in adjudications}
        for judgment in judgments:
            label_error = self._stored_label_error(
                project,
                judgment.label_value,
                final=False,
            )
            if label_error is not None:
                reasons.append(
                    f"judgment {judgment.judgment_id!r} has an invalid label: {label_error}"
                )
            failure_error = self._stored_hard_failure_codes_error(
                project,
                judgment.hard_failure_codes,
                label_value=judgment.label_value,
            )
            if failure_error is not None:
                reasons.append(
                    f"judgment {judgment.judgment_id!r} has invalid hard-failure codes: "
                    f"{failure_error}"
                )
        for stored_decision in adjudications:
            label_error = self._stored_label_error(
                project,
                stored_decision.final_label_value,
                final=True,
            )
            if label_error is not None:
                reasons.append(
                    f"adjudication {stored_decision.adjudication_id!r} "
                    f"has an invalid label: {label_error}"
                )
            failure_error = self._stored_hard_failure_codes_error(
                project,
                stored_decision.final_hard_failure_codes,
                label_value=stored_decision.final_label_value,
            )
            if failure_error is not None:
                reasons.append(
                    f"adjudication {stored_decision.adjudication_id!r} "
                    "has invalid hard-failure codes: "
                    f"{failure_error}"
                )
        final_labels: dict[str, str] = {}
        for item in items:
            item_judgments = judgments_by_item[item.item_id]
            reviewer_ids = {judgment.reviewer_id for judgment in item_judgments}
            if len(item_judgments) != 2 or len(reviewer_ids) != 2:
                reasons.append(f"item {item.item_id!r} lacks two independent judgments")
                continue
            if any(judgment.evidence_sha256 != item.evidence_sha256 for judgment in item_judgments):
                reasons.append(f"item {item.item_id!r} has a stale judgment snapshot")
                continue
            labels = {judgment.label_value for judgment in item_judgments}
            decision = adjudication_by_item.get(item.item_id)
            uncertain = EntityResolutionLabel.UNCERTAIN.value in labels
            if len(labels) == 1 and not uncertain:
                if decision is not None:
                    reasons.append(f"unanimous item {item.item_id!r} has an adjudication")
                    continue
                final_label = next(iter(labels))
            else:
                if decision is None:
                    reasons.append(f"disagreed item {item.item_id!r} lacks adjudication")
                    continue
                if decision.adjudicator_reviewer_id in reviewer_ids:
                    reasons.append(f"item {item.item_id!r} adjudicator is not independent")
                    continue
                if decision.evidence_sha256 != item.evidence_sha256:
                    reasons.append(f"item {item.item_id!r} adjudication snapshot is stale")
                    continue
                final_label = decision.final_label_value
            label_error = self._stored_label_error(project, final_label, final=True)
            if label_error is not None:
                reasons.append(f"item {item.item_id!r} final label is invalid: {label_error}")
                continue
            if item.state != AnnotationItemState.RESOLVED.value:
                reasons.append(f"item {item.item_id!r} is not resolved")
                continue
            final_labels[item.item_id] = final_label

        extra_adjudications = set(adjudication_by_item) - {item.item_id for item in items}
        if extra_adjudications:
            reasons.append("adjudications reference items outside the project")
        if AnnotationTaskType(project.task_type) is AnnotationTaskType.RELEVANCE:
            for group in groups:
                grades = [
                    int(final_labels[item.item_id])
                    for item in items_by_group[group.group_id]
                    if item.item_id in final_labels
                ]
                if grades and not any(grade > 0 for grade in grades):
                    reasons.append(f"query {group.group_key!r} has no relevant candidate")

        if reasons:
            raise AnnotationFreezeBlockedError(tuple(sorted(set(reasons))))
        return final_labels

    def _export_files(
        self,
        project: AnnotationProjectRecord,
        groups: Sequence[AnnotationGroupRecord],
        items: Sequence[AnnotationItemRecord],
        judgments: Sequence[AnnotationJudgmentRecord],
        adjudications: Sequence[AnnotationAdjudicationRecord],
        final_labels: Mapping[str, str],
    ) -> dict[str, bytes]:
        groups_by_id = {group.group_id: group for group in groups}
        items_by_group: dict[str, list[AnnotationItemRecord]] = defaultdict(list)
        for item in items:
            items_by_group[item.group_id].append(item)
        judgments_by_item: dict[str, list[AnnotationJudgmentRecord]] = defaultdict(list)
        for judgment in judgments:
            judgments_by_item[judgment.item_id].append(judgment)
        adjudication_by_item = {decision.item_id: decision for decision in adjudications}
        evidence = {
            "schema_version": "pc-build-recommender.annotation-evidence-snapshots.v1",
            "task_type": project.task_type,
            "dataset_name": project.dataset_name,
            "dataset_version": project.dataset_version,
            "rubric_version": project.rubric_version,
            "data_version": project.data_version,
            "source_policy": project.source_policy,
            "source_policy_sha256": project.source_policy_sha256,
            "groups": [
                {
                    "group_key": group.group_key,
                    "leakage_group_id": group.leakage_group_id,
                    "category": group.category,
                    "split_name": group.split_name,
                    "context_payload": group.context_payload,
                    "context_sha256": group.context_sha256,
                    "items": [
                        {
                            "target_id": item.target_id,
                            "evidence_payload": item.evidence_payload,
                            "evidence_sha256": item.evidence_sha256,
                            "final_label": final_labels[item.item_id],
                            "judgments": [
                                {
                                    "reviewer_id": judgment.reviewer_id,
                                    "label": judgment.label_value,
                                    "rationale": judgment.rationale,
                                    "hard_failure_codes": list(judgment.hard_failure_codes),
                                    "reviewed_at_utc": _iso(judgment.submitted_at),
                                }
                                for judgment in sorted(
                                    judgments_by_item[item.item_id],
                                    key=lambda row: row.reviewer_id,
                                )
                            ],
                            "adjudication": (
                                {
                                    "adjudicator_id": decision.adjudicator_reviewer_id,
                                    "label": decision.final_label_value,
                                    "rationale": decision.rationale,
                                    "hard_failure_codes": list(decision.final_hard_failure_codes),
                                    "adjudicated_at_utc": _iso(decision.submitted_at),
                                }
                                if (decision := adjudication_by_item.get(item.item_id)) is not None
                                else None
                            ),
                        }
                        for item in sorted(
                            items_by_group[group.group_id], key=lambda row: row.target_id
                        )
                    ],
                }
                for group in sorted(groups, key=lambda row: row.group_key)
            ],
        }
        files = {"evidence-snapshots.json": _json_bytes(evidence, pretty=True)}
        task_type = AnnotationTaskType(project.task_type)
        if task_type is AnnotationTaskType.RELEVANCE:
            labeling_queries = tuple(
                LabelingQuery(
                    query_id=group.group_key,
                    query_group_id=group.leakage_group_id,
                    query_text=str(group.context_payload["query_text"]),
                    category=group.category,
                    candidate_ids=tuple(
                        item.target_id
                        for item in sorted(
                            items_by_group[group.group_id], key=lambda row: row.target_id
                        )
                    ),
                )
                for group in sorted(groups, key=lambda row: row.group_key)
            )
            reviewer_judgments = tuple(
                ReviewerJudgment(
                    query_id=groups_by_id[item.group_id].group_key,
                    product_id=item.target_id,
                    reviewer_id=judgment.reviewer_id,
                    grade=int(judgment.label_value),
                    rationale=judgment.rationale,
                    reviewed_at_utc=_iso(judgment.submitted_at),
                    is_synthetic=False,
                )
                for item in sorted(
                    items,
                    key=lambda row: (groups_by_id[row.group_id].group_key, row.target_id),
                )
                for judgment in sorted(
                    judgments_by_item[item.item_id], key=lambda row: row.reviewer_id
                )
            )
            relevance_adjudications = tuple(
                AdjudicationDecision(
                    query_id=groups_by_id[item.group_id].group_key,
                    product_id=item.target_id,
                    adjudicator_id=decision.adjudicator_reviewer_id,
                    grade=int(decision.final_label_value),
                    rationale=decision.rationale,
                    adjudicated_at_utc=_iso(decision.submitted_at),
                )
                for item in sorted(
                    items,
                    key=lambda row: (groups_by_id[row.group_id].group_key, row.target_id),
                )
                if (decision := adjudication_by_item.get(item.item_id)) is not None
            )
            human = HumanJudgmentSet(
                dataset_name=project.dataset_name,
                dataset_version=project.dataset_version,
                queries=labeling_queries,
                judgments=reviewer_judgments,
                adjudications=relevance_adjudications,
            )
            adjudicated = human.adjudicate()
            frozen = adjudicated.frozen_candidates
            group_by_query = {group.group_key: group for group in groups}
            query_groups = {
                query.query_id: group_by_query[query.query_id].leakage_group_id
                for query in labeling_queries
            }
            assignments = {
                query.query_id: group_by_query[query.query_id].split_name
                for query in labeling_queries
            }
            weights = {name: 1.0 for name in project.split_names}
            split_payload: dict[str, object] = {
                "schema_version": QUERY_SPLIT_SCHEMA_VERSION,
                "version": f"{project.dataset_version}:split-v1",
                "dataset_checksum": frozen.checksum,
                "dataset_evidence_checksum": frozen.evidence_checksum,
                "label_source": frozen.label_source.value,
                "adjudication_complete": frozen.adjudication_complete,
                "contains_synthetic_labels": frozen.contains_synthetic_labels,
                "judgment_manifest_sha256": frozen.judgment_manifest_sha256,
                "query_group_ids": dict(sorted(query_groups.items())),
                "assignments": dict(sorted(assignments.items())),
                "weights": dict(sorted(weights.items())),
                "seed": 0,
            }
            split = FrozenQueryGroupSplit(
                version=str(split_payload["version"]),
                dataset_checksum=frozen.checksum,
                dataset_evidence_checksum=frozen.evidence_checksum,
                label_source=frozen.label_source.value,
                adjudication_complete=frozen.adjudication_complete,
                contains_synthetic_labels=frozen.contains_synthetic_labels,
                judgment_manifest_sha256=frozen.judgment_manifest_sha256,
                query_group_ids=query_groups,
                assignments=assignments,
                weights=weights,
                seed=0,
                checksum=sha256_json(split_payload),
            )
            frozen_payload = {
                "version": frozen.version,
                "checksum": frozen.checksum,
                "evidence_checksum": frozen.evidence_checksum,
                "evidence": {
                    "label_source": frozen.label_source.value,
                    "adjudication_complete": frozen.adjudication_complete,
                    "contains_synthetic_labels": frozen.contains_synthetic_labels,
                    "judgment_manifest_sha256": frozen.judgment_manifest_sha256,
                    "eligible_for_promotion": frozen.eligible_for_promotion,
                    "promotion_block_reasons": list(frozen.promotion_block_reasons),
                },
                "queries": [query.to_dict() for query in frozen.queries],
            }
            files.update(
                {
                    "human-judgments.json": _json_bytes(human.content_payload(), pretty=True),
                    "qrels.json": _json_bytes(frozen_payload, pretty=True),
                    "query-split.json": _json_bytes(split.to_dict(), pretty=True),
                }
            )
            return files

        pair_rows: list[dict[str, Any]] = []
        human_groups: list[dict[str, Any]] = []
        for group in sorted(groups, key=lambda row: row.group_key):
            listing = ListingRecord.from_dict(group.context_payload["listing"])
            pair_payloads: list[dict[str, Any]] = []
            for item in sorted(items_by_group[group.group_id], key=lambda row: row.target_id):
                product = CanonicalProductRecord.from_dict(item.evidence_payload["product"])
                final_label = final_labels[item.item_id]
                pair = PairExample(
                    pair_id=f"human-{item.item_id}",
                    listing=listing,
                    product=product,
                    label=int(final_label == EntityResolutionLabel.MATCH.value),
                    is_synthetic=False,
                )
                pair_rows.append(pair.to_dict())
                decision = adjudication_by_item.get(item.item_id)
                pair_payloads.append(
                    {
                        "pair_id": pair.pair_id,
                        "target_id": item.target_id,
                        "evidence_sha256": item.evidence_sha256,
                        "judgments": [
                            {
                                "reviewer_id": judgment.reviewer_id,
                                "label": judgment.label_value,
                                "rationale": judgment.rationale,
                                "hard_failure_codes": list(judgment.hard_failure_codes),
                                "reviewed_at_utc": _iso(judgment.submitted_at),
                            }
                            for judgment in sorted(
                                judgments_by_item[item.item_id], key=lambda row: row.reviewer_id
                            )
                        ],
                        "adjudication": (
                            {
                                "adjudicator_id": decision.adjudicator_reviewer_id,
                                "label": decision.final_label_value,
                                "rationale": decision.rationale,
                                "hard_failure_codes": list(decision.final_hard_failure_codes),
                                "adjudicated_at_utc": _iso(decision.submitted_at),
                            }
                            if decision is not None
                            else None
                        ),
                        "final_label": final_label,
                    }
                )
            human_groups.append(
                {
                    "listing_id": group.group_key,
                    "leakage_group_id": group.leakage_group_id,
                    "split_name": group.split_name,
                    "context_sha256": group.context_sha256,
                    "pairs": pair_payloads,
                }
            )
        er_labels = {
            "schema_version": "pc-build-recommender.er-adjudicated-human-labels.v2",
            "dataset_name": project.dataset_name,
            "dataset_version": project.dataset_version,
            "data_version": project.data_version,
            "rubric_version": project.rubric_version,
            "source_policy": project.source_policy,
            "source_policy_sha256": project.source_policy_sha256,
            "required_independent_reviews": 2,
            "adjudication_complete": True,
            "contains_synthetic_labels": False,
            "groups": human_groups,
        }
        pair_lines = b"".join(_json_bytes(row) for row in pair_rows)
        split_payload = {
            "schema_version": "pc-build-recommender.er-frozen-listing-split.v1",
            "dataset_version": project.dataset_version,
            "assignments": {
                group.group_key: group.split_name
                for group in sorted(groups, key=lambda row: row.group_key)
            },
            "leakage_group_ids": {
                group.group_key: group.leakage_group_id
                for group in sorted(groups, key=lambda row: row.group_key)
            },
        }
        split_payload["checksum"] = sha256_json(split_payload)
        files.update(
            {
                "human-labels.json": _json_bytes(er_labels, pretty=True),
                "pairs.jsonl": pair_lines,
                "listing-split.json": _json_bytes(split_payload, pretty=True),
            }
        )
        return files

    def _write_release(
        self,
        root: Path,
        project: AnnotationProjectRecord,
        files: Mapping[str, bytes],
    ) -> AnnotationRelease:
        root.mkdir(parents=True, exist_ok=True)
        file_evidence = {
            name: {"sha256": _sha256_bytes(payload), "size_bytes": len(payload)}
            for name, payload in sorted(files.items())
        }
        identity_payload = {
            "schema_version": "pc-build-recommender.annotation-release.v1",
            "task_type": project.task_type,
            "dataset_name": project.dataset_name,
            "dataset_version": project.dataset_version,
            "rubric_version": project.rubric_version,
            "data_version": project.data_version,
            "source_policy_sha256": project.source_policy_sha256,
            "required_independent_reviews": project.required_reviews,
            "files": file_evidence,
        }
        release_sha256 = _sha256_payload(identity_payload)
        manifest = {**identity_payload, "release_sha256": release_sha256}
        manifest_bytes = _json_bytes(manifest, pretty=True)
        manifest_sha256 = _sha256_bytes(manifest_bytes)
        target = root / f"annotation-{release_sha256}"
        expected_files = {**files, "manifest.json": manifest_bytes}
        if target.exists():
            self._verify_release_directory(target, expected_files)
        else:
            with tempfile.TemporaryDirectory(prefix=".annotation-release-", dir=root) as temporary:
                temporary_path = Path(temporary).resolve()
                if not temporary_path.is_relative_to(root):
                    raise RuntimeError("temporary annotation release escaped its output root")
                for name, payload in expected_files.items():
                    destination = temporary_path / name
                    with destination.open("xb") as handle:
                        handle.write(payload)
                        handle.flush()
                        os.fsync(handle.fileno())
                os.replace(temporary_path, target)
            self._verify_release_directory(target, expected_files)
        return AnnotationRelease(
            project_id=project.project_id,
            release_sha256=release_sha256,
            manifest_sha256=manifest_sha256,
            artifact_directory=target,
            files={name: _sha256_bytes(payload) for name, payload in sorted(files.items())},
        )

    @staticmethod
    def _verify_release_directory(directory: Path, expected: Mapping[str, bytes]) -> None:
        if not directory.is_dir():
            raise AnnotationConflictError("annotation release path is not a directory")
        actual_names = {path.name for path in directory.iterdir() if path.is_file()}
        if actual_names != set(expected):
            raise AnnotationConflictError("annotation release directory contents changed")
        for name, payload in expected.items():
            if (directory / name).read_bytes() != payload:
                raise AnnotationConflictError(f"annotation release artifact changed: {name}")

    def _load_existing_release(self, record: AnnotationExportRecord) -> AnnotationRelease:
        directory = Path(record.artifact_directory).resolve()
        manifest_path = directory / "manifest.json"
        if not manifest_path.is_file():
            raise AnnotationConflictError("frozen annotation manifest is missing")
        manifest_bytes = manifest_path.read_bytes()
        if _sha256_bytes(manifest_bytes) != record.manifest_sha256:
            raise AnnotationConflictError("frozen annotation manifest hash mismatch")
        payload = json.loads(manifest_bytes)
        if (
            not isinstance(payload, Mapping)
            or payload.get("release_sha256") != record.release_sha256
        ):
            raise AnnotationConflictError("frozen annotation release identity mismatch")
        raw_files = payload.get("files")
        if not isinstance(raw_files, Mapping):
            raise AnnotationConflictError("frozen annotation manifest lacks file evidence")
        files: dict[str, str] = {}
        for name, raw_evidence in raw_files.items():
            if not isinstance(name, str) or not isinstance(raw_evidence, Mapping):
                raise AnnotationConflictError("invalid frozen annotation file evidence")
            digest = raw_evidence.get("sha256")
            path = directory / name
            if not isinstance(digest, str) or not path.is_file():
                raise AnnotationConflictError(f"frozen annotation artifact is missing: {name}")
            if _sha256_bytes(path.read_bytes()) != digest:
                raise AnnotationConflictError(f"frozen annotation artifact hash mismatch: {name}")
            files[name] = digest
        return AnnotationRelease(
            project_id=record.project_id,
            release_sha256=record.release_sha256,
            manifest_sha256=record.manifest_sha256,
            artifact_directory=directory,
            files=files,
        )

    @staticmethod
    def _normalise_label(
        project: AnnotationProjectRecord,
        label: str | int,
        *,
        final: bool,
    ) -> str:
        task_type = AnnotationTaskType(project.task_type)
        if task_type is AnnotationTaskType.RELEVANCE:
            if isinstance(label, bool) or not isinstance(label, int) or not 0 <= label <= 4:
                raise ValueError("relevance labels must be integers from 0 to 4")
            return str(label)
        if not isinstance(label, str):
            raise TypeError("entity-resolution labels must be strings")
        value = EntityResolutionLabel(label).value
        if final and value == EntityResolutionLabel.UNCERTAIN.value:
            raise ValueError("an adjudicated entity-resolution label cannot remain uncertain")
        return value

    @staticmethod
    def _stored_label_error(
        project: AnnotationProjectRecord,
        value: object,
        *,
        final: bool,
    ) -> str | None:
        task_type = AnnotationTaskType(project.task_type)
        if task_type is AnnotationTaskType.RELEVANCE:
            if not isinstance(value, str) or value not in {"0", "1", "2", "3", "4"}:
                return "stored relevance labels must be canonical strings from '0' to '4'"
            submission_value: str | int = int(value)
        else:
            submission_value = value if isinstance(value, str | int) else ""
        try:
            normalised = AnnotationService._normalise_label(
                project,
                submission_value,
                final=final,
            )
        except (TypeError, ValueError) as error:
            return str(error)
        if normalised != value:
            return "stored label is not canonical"
        return None

    @staticmethod
    def _normalise_hard_failure_codes(
        project: AnnotationProjectRecord,
        values: Sequence[str],
        *,
        label_value: str,
    ) -> list[str]:
        if isinstance(values, str | bytes | bytearray):
            raise TypeError("hard_failure_codes must be a sequence of codes")
        result: list[str] = []
        for raw_value in values:
            if not isinstance(raw_value, str) or not raw_value.strip():
                raise ValueError("hard-failure codes must be non-empty strings")
            code = raw_value.strip()
            if len(code) > 120:
                raise ValueError("hard-failure codes must not exceed 120 characters")
            result.append(code)
        result = sorted(set(result))
        if AnnotationTaskType(project.task_type) is AnnotationTaskType.ENTITY_RESOLUTION:
            if result:
                raise ValueError("hard_failure_codes apply only to relevance judgments")
        elif result and label_value != "0":
            raise ValueError("a product with a hard requirement failure must receive grade 0")
        return result

    @staticmethod
    def _stored_hard_failure_codes_error(
        project: AnnotationProjectRecord,
        values: object,
        *,
        label_value: str,
    ) -> str | None:
        if not isinstance(values, list):
            return "value is not a JSON list"
        try:
            normalised = AnnotationService._normalise_hard_failure_codes(
                project,
                values,
                label_value=label_value,
            )
        except (TypeError, ValueError) as error:
            return str(error)
        if values != normalised:
            return "codes are not unique and lexically ordered"
        return None

    @staticmethod
    def _validate_assignment_actor(
        assignment: AnnotationAssignmentRecord,
        reviewer: AnnotationReviewerRecord,
        phase: AssignmentPhase,
    ) -> None:
        if assignment.reviewer_id != reviewer.reviewer_id:
            raise AnnotationAuthorizationError("assignment belongs to a different OIDC identity")
        if assignment.phase != phase.value:
            raise AnnotationConflictError(f"assignment is not a {phase.value} assignment")

    @staticmethod
    def _validate_live_lease(
        assignment: AnnotationAssignmentRecord,
        lease_token: str,
        now: datetime,
    ) -> None:
        if assignment.status != AssignmentStatus.LEASED.value:
            raise AnnotationConflictError("assignment is no longer leased")
        if not lease_token.strip() or _sha256_text(lease_token) != assignment.lease_token_sha256:
            raise AnnotationAuthorizationError("annotation lease token is invalid")
        if _aware(assignment.lease_expires_at) <= now:
            raise AnnotationConflictError("assignment lease expired before submission")

    @staticmethod
    def _idempotent_submission(
        session: Session,
        assignment: AnnotationAssignmentRecord,
        *,
        lease_token: str,
        idempotency_key: str,
        submission_payload_sha256: str,
        decision_type: str,
    ) -> str | None:
        if not idempotency_key.strip():
            raise ValueError("idempotency_key must not be empty")
        if len(idempotency_key) > 240:
            raise ValueError("idempotency_key must not exceed 240 characters")
        if not lease_token.strip() or _sha256_text(lease_token) != assignment.lease_token_sha256:
            raise AnnotationAuthorizationError("annotation lease token is invalid")
        if assignment.status != AssignmentStatus.SUBMITTED.value:
            return None
        key_sha256 = _sha256_text(idempotency_key.strip())
        if (
            assignment.submission_idempotency_sha256 != key_sha256
            or assignment.submission_payload_sha256 != submission_payload_sha256
        ):
            raise AnnotationConflictError(
                "submitted assignment cannot be replayed with different idempotency evidence"
            )
        if decision_type == "judgment":
            decision_id = session.scalar(
                select(AnnotationJudgmentRecord.judgment_id).where(
                    AnnotationJudgmentRecord.assignment_id == assignment.assignment_id
                )
            )
        else:
            decision_id = session.scalar(
                select(AnnotationAdjudicationRecord.adjudication_id).where(
                    AnnotationAdjudicationRecord.assignment_id == assignment.assignment_id
                )
            )
        if decision_id is None:
            raise AnnotationConflictError("submitted assignment has no immutable decision")
        return decision_id

    @staticmethod
    def _require_project_status(
        project: AnnotationProjectRecord,
        expected: AnnotationProjectStatus,
    ) -> None:
        if project.status != expected.value:
            raise AnnotationConflictError(
                f"annotation project status is {project.status!r}, expected {expected.value!r}"
            )

    def _require_actor(
        self,
        session: Session,
        identity: VerifiedOIDCIdentity,
        role: AnnotationRole,
    ) -> AnnotationReviewerRecord:
        reviewer = session.scalar(
            select(AnnotationReviewerRecord).where(
                AnnotationReviewerRecord.oidc_issuer == identity.issuer,
                AnnotationReviewerRecord.oidc_subject == identity.subject,
            )
        )
        if reviewer is None or not reviewer.active:
            raise AnnotationAuthorizationError("OIDC identity is not an active annotation reviewer")
        try:
            roles = {AnnotationRole(value) for value in reviewer.roles}
        except (TypeError, ValueError) as error:
            raise AnnotationAuthorizationError("stored reviewer roles are invalid") from error
        if role not in roles:
            raise AnnotationAuthorizationError(
                f"OIDC identity lacks annotation role {role.value!r}"
            )
        return reviewer

    @staticmethod
    def _project(
        session: Session,
        project_id: str,
        *,
        for_update: bool = False,
    ) -> AnnotationProjectRecord:
        statement = select(AnnotationProjectRecord).where(
            AnnotationProjectRecord.project_id == project_id
        )
        if for_update and session.get_bind().dialect.name == "postgresql":
            statement = statement.with_for_update()
        project = session.scalar(statement)
        if project is None:
            raise KeyError(f"unknown annotation project: {project_id}")
        return project

    @staticmethod
    def _assignment(
        session: Session,
        assignment_id: str,
        *,
        for_update: bool = False,
    ) -> AnnotationAssignmentRecord:
        statement = select(AnnotationAssignmentRecord).where(
            AnnotationAssignmentRecord.assignment_id == assignment_id
        )
        if for_update and session.get_bind().dialect.name == "postgresql":
            statement = statement.with_for_update()
        assignment = session.scalar(statement)
        if assignment is None:
            raise KeyError(f"unknown annotation assignment: {assignment_id}")
        return assignment

    @staticmethod
    def _item_context(
        session: Session,
        item_id: str,
    ) -> tuple[AnnotationItemRecord, AnnotationGroupRecord, AnnotationProjectRecord]:
        item = session.get(AnnotationItemRecord, item_id)
        if item is None:
            raise KeyError(f"unknown annotation item: {item_id}")
        group = session.get(AnnotationGroupRecord, item.group_id)
        assert group is not None
        project = session.get(AnnotationProjectRecord, group.project_id)
        assert project is not None
        return item, group, project

    @staticmethod
    def _audit(
        session: Session,
        *,
        event_type: str,
        payload: Mapping[str, Any],
        occurred_at: datetime,
        actor: AnnotationReviewerRecord | None = None,
        project_id: str | None = None,
        item_id: str | None = None,
    ) -> None:
        canonical = _canonical_mapping(payload, field_name="audit payload")
        session.add(
            AnnotationAuditEventRecord(
                event_id=_new_id("ann_audit"),
                project_id=project_id,
                item_id=item_id,
                actor_reviewer_id=actor.reviewer_id if actor is not None else None,
                event_type=event_type,
                payload=canonical,
                payload_sha256=_sha256_payload(canonical),
                occurred_at=occurred_at,
            )
        )
