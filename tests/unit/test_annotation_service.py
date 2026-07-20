from __future__ import annotations
from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock

import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.dialects.postgresql.base import PGDialect
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from pc_build_recommender.annotation import (
    AnnotationAuthorizationError,
    AnnotationConflictError,
    AnnotationFreezeBlockedError,
    AnnotationRole,
    AnnotationService,
    AnnotationTaskType,
    EntityResolutionLabel,
    VerifiedOIDCIdentity,
)
from pc_build_recommender.annotation.orm import (
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
from pc_build_recommender.annotation.service import _with_skip_locked
from pc_build_recommender.catalog.orm import Base
from pc_build_recommender.entity_resolution import (
    CanonicalProductRecord,
    ListingRecord,
    PairExample,
)
from pc_build_recommender.evaluation.manifest import sha256_json
from pc_build_recommender.retrieval import FrozenCandidateSet, FrozenQueryGroupSplit
from pc_build_recommender.retrieval.judgments import load_human_judgment_set

_NOW = datetime(2026, 7, 23, 4, 0, tzinfo=UTC)
_GOOD_POLICY = {
    "training_eligible": True,
    "published_metrics_eligible": True,
    "model_serving_eligible": True,
    "scope_note": "Rights-cleared test fixture evidence.",
}


@dataclass(frozen=True, slots=True)
class _AnnotationContext:
    service: AnnotationService
    sessions: sessionmaker[Session]
    admin: VerifiedOIDCIdentity
    reviewer_a: VerifiedOIDCIdentity
    reviewer_b: VerifiedOIDCIdentity
    reviewer_c: VerifiedOIDCIdentity
    adjudicator: VerifiedOIDCIdentity


@pytest.fixture
def annotation_context() -> _AnnotationContext:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    sessions = sessionmaker(engine, class_=Session, expire_on_commit=False)
    service = AnnotationService(sessions)
    admin = VerifiedOIDCIdentity("https://identity.invalid", "admin")
    reviewer_a = VerifiedOIDCIdentity("https://identity.invalid", "reviewer-a")
    reviewer_b = VerifiedOIDCIdentity("https://identity.invalid", "reviewer-b")
    reviewer_c = VerifiedOIDCIdentity("https://identity.invalid", "reviewer-c")
    adjudicator = VerifiedOIDCIdentity("https://identity.invalid", "adjudicator")

    with sessions.begin() as session:
        session.add(
            AnnotationReviewerRecord(
                reviewer_id="ann_reviewer_admin",
                oidc_issuer=admin.issuer,
                oidc_subject=admin.subject,
                display_name="Admin",
                roles=[AnnotationRole.ADMIN.value],
                active=True,
                verified_at=_NOW,
                created_at=_NOW,
            )
        )
    for identity, display_name, roles in (
        (reviewer_a, "Reviewer A", [AnnotationRole.REVIEWER]),
        (reviewer_b, "Reviewer B", [AnnotationRole.REVIEWER]),
        (reviewer_c, "Reviewer C", [AnnotationRole.REVIEWER]),
        (adjudicator, "Adjudicator", [AnnotationRole.ADJUDICATOR]),
    ):
        service.provision_reviewer(
            admin,
            identity=identity,
            display_name=display_name,
            roles=roles,
            now=_NOW,
        )

    return _AnnotationContext(
        service=service,
        sessions=sessions,
        admin=admin,
        reviewer_a=reviewer_a,
        reviewer_b=reviewer_b,
        reviewer_c=reviewer_c,
        adjudicator=adjudicator,
    )


def _create_project(
    context: _AnnotationContext,
    *,
    task_type: AnnotationTaskType,
    suffix: str,
    source_policy: dict[str, object] | None = None,
) -> str:
    return context.service.create_project(
        context.admin,
        task_type=task_type,
        dataset_name=f"annotation-{suffix}",
        dataset_version=f"2026-07-23-{suffix}",
        rubric_version="rubric-v1",
        data_version="catalog-v1",
        source_policy=source_policy or _GOOD_POLICY,
        now=_NOW,
    )


def _add_relevance_candidate(
    context: _AnnotationContext,
    project_id: str,
    *,
    query_id: str,
    split_name: str,
    target_id: str,
    synthetic: bool = False,
) -> str:
    group_id = context.service.add_group(
        context.admin,
        project_id,
        group_key=query_id,
        leakage_group_id=f"intent-{query_id}",
        category="gpu",
        split_name=split_name,
        context_payload={
            "query_text": f"GPU candidate for {query_id}",
            "structured_constraints": {"minimum_gpu_vram_gb": 16},
        },
        is_synthetic=synthetic,
        now=_NOW,
    )
    return context.service.add_item(
        context.admin,
        project_id,
        group_id,
        target_id=target_id,
        evidence_payload={
            "canonical_name": f"Evidence for {target_id}",
            "vram_gb": 16,
            "source_url": f"https://manufacturer.invalid/{target_id}",
        },
        is_synthetic=synthetic,
        now=_NOW,
    )


def _relevance_import_group(group_key: str, split_name: str) -> dict[str, object]:
    return {
        "group_key": group_key,
        "leakage_group_id": f"intent-{group_key}",
        "category": "gpu",
        "split_name": split_name,
        "context_payload": {
            "query_text": f"GPU candidate for {group_key}",
            "structured_constraints": {"minimum_gpu_vram_gb": 16},
        },
        "is_synthetic": False,
        "items": [
            {
                "target_id": f"gpu-{group_key}",
                "evidence_payload": {"canonical_name": f"GPU {group_key}"},
                "priority": 1,
                "is_synthetic": False,
            }
        ],
    }


def _add_relevance_group(
    context: _AnnotationContext,
    project_id: str,
    *,
    query_id: str,
    split_name: str,
    target_ids: tuple[str, ...],
) -> None:
    group_id = context.service.add_group(
        context.admin,
        project_id,
        group_key=query_id,
        leakage_group_id=f"intent-{query_id}",
        category="gpu",
        split_name=split_name,
        context_payload={
            "query_text": f"GPU candidate for {query_id}",
            "structured_constraints": {"minimum_gpu_vram_gb": 16},
        },
        now=_NOW,
    )
    for target_id in target_ids:
        context.service.add_item(
            context.admin,
            project_id,
            group_id,
            target_id=target_id,
            evidence_payload={
                "canonical_name": f"Evidence for {target_id}",
                "vram_gb": 16,
                "source_url": f"https://manufacturer.invalid/{target_id}",
            },
            now=_NOW,
        )


def _submit_relevance_plan(
    context: _AnnotationContext,
    project_id: str,
    actor: VerifiedOIDCIdentity,
    plan: dict[str, tuple[int, tuple[str, ...]]],
    *,
    key_prefix: str,
) -> list[str]:
    submitted: list[str] = []
    while (task := context.service.claim_review(actor, project_id, now=_NOW)) is not None:
        label, hard_failures = plan[task.target_id]
        submitted.append(
            context.service.submit_judgment(
                actor,
                task.assignment_id,
                lease_token=task.lease_token,
                idempotency_key=f"{key_prefix}-{task.target_id}",
                evidence_sha256=task.evidence_sha256,
                label=label,
                rationale=f"Human review of {task.target_id}",
                hard_failure_codes=hard_failures,
                now=_NOW,
            )
        )
    return submitted


def _listing(listing_id: str) -> ListingRecord:
    return ListingRecord(
        listing_id=listing_id,
        title=f"Example RTX 4070 Super {listing_id}",
        category="gpu",
        brand="Example",
        manufacturer_part_number=f"MPN-{listing_id}",
        attributes={"vram_gb": 12},
        current_price_sgd=799.0,
        retailer="Rights-cleared retailer feed",
    )


def _product(product_id: str) -> CanonicalProductRecord:
    return CanonicalProductRecord(
        product_id=product_id,
        category="gpu",
        brand="Example",
        model=f"RTX 4070 Super {product_id}",
        canonical_name=f"Example RTX 4070 Super {product_id}",
        manufacturer_part_number=f"MPN-{product_id}",
        attributes={"vram_gb": 12},
        price_sgd=789.0,
    )


def _add_er_pair(
    context: _AnnotationContext,
    project_id: str,
    *,
    listing_id: str,
    product_id: str,
    split_name: str,
) -> None:
    listing_payload = _listing(listing_id).to_dict()
    product_payload = _product(product_id).to_dict()
    listing_payload.pop("embedding")
    product_payload.pop("embedding")
    group_id = context.service.add_group(
        context.admin,
        project_id,
        group_key=listing_id,
        leakage_group_id=f"family-{listing_id}",
        category="gpu",
        split_name=split_name,
        context_payload={"listing": listing_payload},
        now=_NOW,
    )
    context.service.add_item(
        context.admin,
        project_id,
        group_id,
        target_id=product_id,
        evidence_payload={"product": product_payload},
        now=_NOW,
    )


def _submit_er_plan(
    context: _AnnotationContext,
    project_id: str,
    actor: VerifiedOIDCIdentity,
    labels: dict[str, EntityResolutionLabel],
    *,
    key_prefix: str,
) -> None:
    while (task := context.service.claim_review(actor, project_id, now=_NOW)) is not None:
        context.service.submit_judgment(
            actor,
            task.assignment_id,
            lease_token=task.lease_token,
            idempotency_key=f"{key_prefix}-{task.target_id}",
            evidence_sha256=task.evidence_sha256,
            label=labels[task.target_id].value,
            rationale=f"Compared stable identifiers for {task.target_id}",
            now=_NOW,
        )


def test_atomic_batch_import_rolls_back_midstream_fault_and_retries(
    annotation_context: _AnnotationContext,
) -> None:
    context = annotation_context
    project_id = _create_project(
        context,
        task_type=AnnotationTaskType.RELEVANCE,
        suffix="atomic-import",
    )

    def broken_batch() -> Iterator[dict[str, object]]:
        yield _relevance_import_group("train", "train")
        raise RuntimeError("simulated input stream failure")

    with pytest.raises(RuntimeError, match="simulated input stream failure"):
        context.service.import_batch(
            context.admin,
            project_id,
            groups=broken_batch(),
            now=_NOW,
        )

    with context.sessions() as session:
        assert session.scalar(select(func.count(AnnotationGroupRecord.group_id))) == 0
        assert session.scalar(select(func.count(AnnotationItemRecord.item_id))) == 0

    group_count, item_count = context.service.import_batch(
        context.admin,
        project_id,
        groups=(
            _relevance_import_group("train", "train"),
            _relevance_import_group("validation", "validation"),
        ),
        now=_NOW,
    )
    assert (group_count, item_count) == (2, 2)
    with context.sessions() as session:
        assert session.scalar(select(func.count(AnnotationGroupRecord.group_id))) == 2
        assert session.scalar(select(func.count(AnnotationItemRecord.item_id))) == 2


def test_oidc_roles_lease_capacity_and_reviewer_cannot_reclaim(
    annotation_context: _AnnotationContext,
) -> None:
    context = annotation_context
    unknown = VerifiedOIDCIdentity("https://identity.invalid", "unknown")
    with pytest.raises(AnnotationAuthorizationError, match="not an active"):
        context.service.create_project(
            unknown,
            task_type=AnnotationTaskType.RELEVANCE,
            dataset_name="unauthorized",
            dataset_version="unauthorized-v1",
            rubric_version="rubric-v1",
            data_version="catalog-v1",
            source_policy=_GOOD_POLICY,
            now=_NOW,
        )
    with pytest.raises(AnnotationAuthorizationError, match="admin"):
        context.service.provision_reviewer(
            context.reviewer_a,
            identity=unknown,
            display_name="Unauthorized",
            roles=[AnnotationRole.REVIEWER],
            now=_NOW,
        )

    project_id = _create_project(
        context,
        task_type=AnnotationTaskType.RELEVANCE,
        suffix="lease-capacity",
    )
    with pytest.raises(ValueError, match="model-derived reviewer bias"):
        context.service.add_group(
            context.admin,
            project_id,
            group_key="q-unblinded",
            leakage_group_id="intent-unblinded",
            category="gpu",
            split_name="train",
            context_payload={
                "query_text": "GPU candidate",
                "structured_constraints": {},
                "model_score": 0.99,
            },
            now=_NOW,
        )
    item_id = _add_relevance_candidate(
        context,
        project_id,
        query_id="q-lease",
        split_name="train",
        target_id="gpu-lease",
    )
    context.service.open_project(context.admin, project_id, now=_NOW)

    with pytest.raises(AnnotationAuthorizationError, match="reviewer"):
        context.service.claim_review(context.adjudicator, project_id, now=_NOW)
    with pytest.raises(AnnotationAuthorizationError, match="not an active"):
        context.service.claim_review(unknown, project_id, now=_NOW)

    first = context.service.claim_review(context.reviewer_a, project_id, lease_seconds=30, now=_NOW)
    second = context.service.claim_review(
        context.reviewer_b, project_id, lease_seconds=30, now=_NOW
    )
    assert first is not None and second is not None
    assert first.item_id == second.item_id == item_id
    assert context.service.claim_review(context.reviewer_c, project_id, now=_NOW) is None
    assert context.service.claim_review(context.reviewer_a, project_id, now=_NOW) is None

    with context.sessions() as session:
        assignments = tuple(
            session.scalars(
                select(AnnotationAssignmentRecord).order_by(
                    AnnotationAssignmentRecord.assignment_id
                )
            )
        )
    assert len(assignments) == 2
    raw_tokens = {first.lease_token, second.lease_token}
    assert all(record.lease_token_sha256 not in raw_tokens for record in assignments)
    assert {record.lease_token_sha256 for record in assignments} == {
        hashlib.sha256(first.lease_token.encode()).hexdigest(),
        hashlib.sha256(second.lease_token.encode()).hexdigest(),
    }
    first_record = next(
        record for record in assignments if record.assignment_id == first.assignment_id
    )
    with pytest.raises(IntegrityError), context.sessions.begin() as session:
        session.add(
            AnnotationAssignmentRecord(
                assignment_id="ann_assignment_duplicate_reviewer_race",
                item_id=item_id,
                reviewer_id=first_record.reviewer_id,
                phase="review",
                status="leased",
                lease_token_sha256="f" * 64,
                leased_at=_NOW,
                lease_expires_at=_NOW + timedelta(seconds=30),
            )
        )

    after_expiry = _NOW + timedelta(seconds=31)
    assert context.service.claim_review(context.reviewer_a, project_id, now=after_expiry) is None
    replacement = context.service.claim_review(
        context.reviewer_c, project_id, lease_seconds=30, now=after_expiry
    )
    assert replacement is not None
    assert replacement.item_id == item_id


def test_project_progress_is_admin_only_aggregate_and_tracks_adjudication(
    annotation_context: _AnnotationContext,
) -> None:
    context = annotation_context
    project_id = _create_project(
        context,
        task_type=AnnotationTaskType.RELEVANCE,
        suffix="progress",
    )
    _add_relevance_candidate(
        context,
        project_id,
        query_id="q-progress",
        split_name="train",
        target_id="gpu-progress",
    )
    context.service.open_project(context.admin, project_id, now=_NOW)

    with pytest.raises(AnnotationAuthorizationError, match="admin"):
        context.service.project_progress(context.reviewer_a, project_id, now=_NOW)

    initial = context.service.project_progress(context.admin, project_id, now=_NOW)
    assert initial.group_count == 1
    assert initial.item_count == 1
    assert initial.item_state_counts == {
        "pending": 1,
        "in_review": 0,
        "needs_adjudication": 0,
        "resolved": 0,
    }
    assert initial.judgment_coverage == {
        "zero_judgments": 1,
        "one_judgment": 0,
        "two_or_more_judgments": 0,
    }
    assert initial.review_assignment_counts == {
        "active_leased": 0,
        "elapsed_leased": 0,
        "submitted": 0,
        "expired_record": 0,
    }
    assert initial.adjudication_required_count == 0
    assert initial.adjudication_completed_count == 0
    assert initial.coarse_freeze_preflight_passes is False
    assert any("lack two independent judgments" in reason for reason in initial.preflight_blockers)
    initial_payload = initial.to_dict()
    rendered = json.dumps(initial_payload, sort_keys=True)
    assert "Evidence for gpu-progress" not in rendered
    assert "Reviewer A" not in rendered
    assert "lease_token" not in rendered

    first = context.service.claim_review(context.reviewer_a, project_id, now=_NOW)
    assert first is not None
    leased = context.service.project_progress(context.admin, project_id, now=_NOW)
    assert leased.item_state_counts["in_review"] == 1
    assert leased.review_assignment_counts["active_leased"] == 1
    context.service.submit_judgment(
        context.reviewer_a,
        first.assignment_id,
        lease_token=first.lease_token,
        idempotency_key="progress-review-a",
        evidence_sha256=first.evidence_sha256,
        label=4,
        rationale="Strong workload fit.",
        now=_NOW,
    )

    second = context.service.claim_review(context.reviewer_b, project_id, now=_NOW)
    assert second is not None
    context.service.submit_judgment(
        context.reviewer_b,
        second.assignment_id,
        lease_token=second.lease_token,
        idempotency_key="progress-review-b",
        evidence_sha256=second.evidence_sha256,
        label=3,
        rationale="Acceptable but not best fit.",
        now=_NOW,
    )
    disputed = context.service.project_progress(context.admin, project_id, now=_NOW)
    assert disputed.item_state_counts["needs_adjudication"] == 1
    assert disputed.judgment_coverage["two_or_more_judgments"] == 1
    assert disputed.review_assignment_counts["submitted"] == 2
    assert disputed.adjudication_required_count == 1
    assert disputed.adjudication_completed_count == 0

    adjudication = context.service.claim_adjudication(context.adjudicator, project_id, now=_NOW)
    assert adjudication is not None
    awaiting_decision = context.service.project_progress(context.admin, project_id, now=_NOW)
    assert awaiting_decision.adjudication_assignment_counts["active_leased"] == 1
    context.service.submit_adjudication(
        context.adjudicator,
        adjudication.assignment_id,
        lease_token=adjudication.lease_token,
        idempotency_key="progress-adjudication",
        evidence_sha256=adjudication.evidence_sha256,
        final_label=4,
        rationale="Independent final assessment.",
        now=_NOW,
    )
    resolved = context.service.project_progress(context.admin, project_id, now=_NOW)
    assert resolved.item_state_counts["resolved"] == 1
    assert resolved.adjudication_required_count == 0
    assert resolved.adjudication_completed_count == 1
    assert resolved.adjudication_assignment_counts["submitted"] == 1
    assert resolved.release_record_present is False


def test_relevance_dual_review_adjudication_idempotency_and_contract_exports(
    annotation_context: _AnnotationContext,
    tmp_path: Path,
) -> None:
    context = annotation_context
    project_id = _create_project(
        context,
        task_type=AnnotationTaskType.RELEVANCE,
        suffix="relevance-release",
    )
    _add_relevance_group(
        context,
        project_id,
        query_id="q-train",
        split_name="train",
        target_ids=("gpu-hard-fail", "gpu-train-good"),
    )
    _add_relevance_group(
        context,
        project_id,
        query_id="q-validation",
        split_name="validation",
        target_ids=("gpu-validation",),
    )
    _add_relevance_group(
        context,
        project_id,
        query_id="q-test",
        split_name="test",
        target_ids=("gpu-test",),
    )
    context.service.open_project(context.admin, project_id, now=_NOW)

    plan_a = {
        "gpu-hard-fail": (0, ("MINIMUM_GPU_VRAM",)),
        "gpu-train-good": (4, ()),
        "gpu-validation": (3, ()),
        "gpu-test": (2, ()),
    }
    first_task = context.service.claim_review(context.reviewer_a, project_id, now=_NOW)
    assert first_task is not None
    first_label, first_codes = plan_a[first_task.target_id]
    with pytest.raises(AnnotationAuthorizationError, match="lease token"):
        context.service.submit_judgment(
            context.reviewer_a,
            first_task.assignment_id,
            lease_token="wrong-token",
            idempotency_key="review-a-first",
            evidence_sha256=first_task.evidence_sha256,
            label=first_label,
            rationale="Human review",
            hard_failure_codes=first_codes,
            now=_NOW,
        )
    first_judgment = context.service.submit_judgment(
        context.reviewer_a,
        first_task.assignment_id,
        lease_token=first_task.lease_token,
        idempotency_key="review-a-first",
        evidence_sha256=first_task.evidence_sha256,
        label=first_label,
        rationale="Human review",
        hard_failure_codes=first_codes,
        now=_NOW,
    )
    assert (
        context.service.submit_judgment(
            context.reviewer_a,
            first_task.assignment_id,
            lease_token=first_task.lease_token,
            idempotency_key="review-a-first",
            evidence_sha256=first_task.evidence_sha256,
            label=first_label,
            rationale="Human review",
            hard_failure_codes=first_codes,
            now=_NOW,
        )
        == first_judgment
    )
    with pytest.raises(AnnotationConflictError, match="idempotency evidence"):
        context.service.submit_judgment(
            context.reviewer_a,
            first_task.assignment_id,
            lease_token=first_task.lease_token,
            idempotency_key="different-key",
            evidence_sha256=first_task.evidence_sha256,
            label=first_label,
            rationale="Human review",
            hard_failure_codes=first_codes,
            now=_NOW,
        )

    remaining_a = dict(plan_a)
    remaining_a.pop(first_task.target_id)
    _submit_relevance_plan(
        context,
        project_id,
        context.reviewer_a,
        remaining_a,
        key_prefix="review-a",
    )
    assert context.service.claim_review(context.reviewer_a, project_id, now=_NOW) is None
    plan_b: dict[str, tuple[int, tuple[str, ...]]] = {
        "gpu-hard-fail": (1, ()),
        "gpu-train-good": (4, ()),
        "gpu-validation": (3, ()),
        "gpu-test": (2, ()),
    }
    _submit_relevance_plan(
        context,
        project_id,
        context.reviewer_b,
        plan_b,
        key_prefix="review-b",
    )

    adjudication = context.service.claim_adjudication(context.adjudicator, project_id, now=_NOW)
    assert adjudication is not None
    assert adjudication.target_id == "gpu-hard-fail"
    with pytest.raises(ValueError, match="grade 0"):
        context.service.submit_adjudication(
            context.adjudicator,
            adjudication.assignment_id,
            lease_token=adjudication.lease_token,
            idempotency_key="adjudication-hard-fail",
            evidence_sha256=adjudication.evidence_sha256,
            final_label=2,
            rationale="Final independent decision",
            final_hard_failure_codes=("MINIMUM_GPU_VRAM",),
            now=_NOW,
        )
    adjudication_id = context.service.submit_adjudication(
        context.adjudicator,
        adjudication.assignment_id,
        lease_token=adjudication.lease_token,
        idempotency_key="adjudication-hard-fail",
        evidence_sha256=adjudication.evidence_sha256,
        final_label=0,
        rationale="Final independent decision",
        final_hard_failure_codes=("MINIMUM_GPU_VRAM",),
        now=_NOW,
    )
    assert (
        context.service.submit_adjudication(
            context.adjudicator,
            adjudication.assignment_id,
            lease_token=adjudication.lease_token,
            idempotency_key="adjudication-hard-fail",
            evidence_sha256=adjudication.evidence_sha256,
            final_label=0,
            rationale="Final independent decision",
            final_hard_failure_codes=("MINIMUM_GPU_VRAM",),
            now=_NOW,
        )
        == adjudication_id
    )

    with context.sessions() as session:
        assignment = session.get(AnnotationAssignmentRecord, first_task.assignment_id)
        assert assignment is not None
        assert (
            assignment.lease_token_sha256
            == hashlib.sha256(first_task.lease_token.encode()).hexdigest()
        )
        assert (
            assignment.submission_idempotency_sha256
            == hashlib.sha256(b"review-a-first").hexdigest()
        )
        assert first_task.lease_token not in assignment.lease_token_sha256
        assert "review-a-first" not in assignment.submission_idempotency_sha256
        assert session.scalar(select(func.count()).select_from(AnnotationJudgmentRecord)) == 8
        assert session.scalar(select(func.count()).select_from(AnnotationAdjudicationRecord)) == 1

    progress = context.service.project_progress(context.admin, project_id, now=_NOW)
    assert progress.item_state_counts["resolved"] == 4
    assert progress.judgment_coverage["two_or_more_judgments"] == 4
    assert progress.adjudication_completed_count == 1
    assert progress.coarse_freeze_preflight_passes is True
    assert progress.preflight_blockers == ()

    release = context.service.freeze_project(
        context.admin,
        project_id,
        output_root=tmp_path / "releases",
        now=_NOW,
    )
    repeated = context.service.freeze_project(
        context.admin,
        project_id,
        output_root=tmp_path / "releases",
        now=_NOW + timedelta(minutes=1),
    )
    assert repeated.release_sha256 == release.release_sha256
    assert repeated.manifest_sha256 == release.manifest_sha256
    assert repeated.files == release.files
    assert repeated.artifact_directory == release.artifact_directory
    assert set(release.files) == {
        "evidence-snapshots.json",
        "human-judgments.json",
        "qrels.json",
        "query-split.json",
    }

    artifact = release.artifact_directory
    human = load_human_judgment_set(artifact / "human-judgments.json")
    assert len(human.judgments) == 8
    assert len(human.adjudications) == 1
    frozen = FrozenCandidateSet.load(artifact / "qrels.json")
    split = FrozenQueryGroupSplit.load(artifact / "query-split.json")
    split.validate_dataset(frozen)
    assert frozen.eligible_for_promotion
    assert set(split.assignments.values()) == {"train", "validation", "test"}
    evidence = json.loads((artifact / "evidence-snapshots.json").read_text("utf-8"))
    hard_fail_item = next(
        item
        for group in evidence["groups"]
        for item in group["items"]
        if item["target_id"] == "gpu-hard-fail"
    )
    assert any(
        judgment["hard_failure_codes"] == ["MINIMUM_GPU_VRAM"]
        for judgment in hard_fail_item["judgments"]
    )
    assert hard_fail_item["adjudication"]["hard_failure_codes"] == ["MINIMUM_GPU_VRAM"]
    assert hard_fail_item["final_label"] == "0"

    with context.sessions() as session:
        judgment = session.get(AnnotationJudgmentRecord, first_judgment)
        assert judgment is not None
        judgment.rationale = "Mutation must fail"
        with pytest.raises(RuntimeError, match="append-only"):
            session.commit()
        session.rollback()

        decision = session.get(AnnotationAdjudicationRecord, adjudication_id)
        assert decision is not None
        decision.rationale = "Mutation must also fail"
        with pytest.raises(RuntimeError, match="append-only"):
            session.commit()
        session.rollback()

        event = session.scalar(select(AnnotationAuditEventRecord))
        assert event is not None
        session.delete(event)
        with pytest.raises(RuntimeError, match="append-only"):
            session.commit()
        session.rollback()
        assert session.scalar(select(func.count()).select_from(AnnotationExportRecord)) == 1


def test_entity_resolution_release_preserves_raw_reviews_and_contract_rows(
    annotation_context: _AnnotationContext,
    tmp_path: Path,
) -> None:
    context = annotation_context
    project_id = _create_project(
        context,
        task_type=AnnotationTaskType.ENTITY_RESOLUTION,
        suffix="er-release",
    )
    pairs = {
        "product-train": "train",
        "product-calibration": "calibration",
        "product-threshold": "threshold",
        "product-test": "test",
    }
    for product_id, split_name in pairs.items():
        _add_er_pair(
            context,
            project_id,
            listing_id=f"listing-{split_name}",
            product_id=product_id,
            split_name=split_name,
        )
    context.service.open_project(context.admin, project_id, now=_NOW)

    labels_a = {product_id: EntityResolutionLabel.NON_MATCH for product_id in pairs}
    labels_a["product-train"] = EntityResolutionLabel.MATCH
    labels_b = dict(labels_a)
    labels_b["product-train"] = EntityResolutionLabel.NON_MATCH
    _submit_er_plan(
        context,
        project_id,
        context.reviewer_a,
        labels_a,
        key_prefix="er-a",
    )

    first_b = context.service.claim_review(context.reviewer_b, project_id, now=_NOW)
    assert first_b is not None
    with pytest.raises(ValueError, match="only to relevance"):
        context.service.submit_judgment(
            context.reviewer_b,
            first_b.assignment_id,
            lease_token=first_b.lease_token,
            idempotency_key="er-b-invalid-hard-code",
            evidence_sha256=first_b.evidence_sha256,
            label=labels_b[first_b.target_id].value,
            rationale="Human entity-resolution review",
            hard_failure_codes=("NOT_AN_ER_FIELD",),
            now=_NOW,
        )
    context.service.submit_judgment(
        context.reviewer_b,
        first_b.assignment_id,
        lease_token=first_b.lease_token,
        idempotency_key=f"er-b-{first_b.target_id}",
        evidence_sha256=first_b.evidence_sha256,
        label=labels_b[first_b.target_id].value,
        rationale="Human entity-resolution review",
        now=_NOW,
    )
    remaining_b = dict(labels_b)
    remaining_b.pop(first_b.target_id)
    _submit_er_plan(
        context,
        project_id,
        context.reviewer_b,
        remaining_b,
        key_prefix="er-b",
    )

    adjudication = context.service.claim_adjudication(context.adjudicator, project_id, now=_NOW)
    assert adjudication is not None
    assert adjudication.target_id == "product-train"
    context.service.submit_adjudication(
        context.adjudicator,
        adjudication.assignment_id,
        lease_token=adjudication.lease_token,
        idempotency_key="er-adjudication-product-train",
        evidence_sha256=adjudication.evidence_sha256,
        final_label=EntityResolutionLabel.MATCH.value,
        rationale="MPN and model tokens identify the same variant",
        now=_NOW,
    )

    release = context.service.freeze_project(
        context.admin,
        project_id,
        output_root=tmp_path / "releases",
        now=_NOW,
    )
    repeated = context.service.freeze_project(
        context.admin,
        project_id,
        output_root=tmp_path / "releases",
        now=_NOW + timedelta(minutes=1),
    )
    assert repeated.release_sha256 == release.release_sha256
    artifact = release.artifact_directory
    pair_examples = [
        PairExample.from_dict(json.loads(line))
        for line in (artifact / "pairs.jsonl").read_text("utf-8").splitlines()
    ]
    assert len(pair_examples) == 4
    assert all(not pair.is_synthetic for pair in pair_examples)
    assert {pair.product.product_id: pair.label for pair in pair_examples}["product-train"] == 1

    labels = json.loads((artifact / "human-labels.json").read_text("utf-8"))
    train_group = next(
        group for group in labels["groups"] if group["listing_id"] == "listing-train"
    )
    train_pair = train_group["pairs"][0]
    assert {row["label"] for row in train_pair["judgments"]} == {"MATCH", "NON_MATCH"}
    assert len({row["reviewer_id"] for row in train_pair["judgments"]}) == 2
    assert train_pair["adjudication"]["label"] == "MATCH"
    assert train_pair["final_label"] == "MATCH"

    split_payload = json.loads((artifact / "listing-split.json").read_text("utf-8"))
    split_checksum = split_payload.pop("checksum")
    assert split_checksum == sha256_json(split_payload)
    assert set(split_payload["assignments"].values()) == {
        "train",
        "calibration",
        "threshold",
        "test",
    }


def test_freeze_fails_closed_on_rights_synthetic_completeness_split_and_hash(
    annotation_context: _AnnotationContext,
    tmp_path: Path,
) -> None:
    context = annotation_context
    project_id = _create_project(
        context,
        task_type=AnnotationTaskType.RELEVANCE,
        suffix="blocked",
        source_policy={
            "training_eligible": False,
            "published_metrics_eligible": False,
            "model_serving_eligible": False,
            "scope_note": "Evaluation only; training rights denied.",
        },
    )
    item_id = _add_relevance_candidate(
        context,
        project_id,
        query_id="q-blocked",
        split_name="train",
        target_id="gpu-blocked",
        synthetic=True,
    )
    context.service.open_project(context.admin, project_id, now=_NOW)
    with context.sessions.begin() as session:
        item = session.get(AnnotationItemRecord, item_id)
        assert item is not None
        item.evidence_payload = {**item.evidence_payload, "tampered": True}

    with pytest.raises(AnnotationFreezeBlockedError) as captured:
        context.service.freeze_project(
            context.admin,
            project_id,
            output_root=tmp_path / "blocked-release",
            now=_NOW,
        )
    reasons = captured.value.reasons
    assert reasons == tuple(sorted(set(reasons)))
    assert "source rights do not permit model training" in reasons
    assert "source rights do not permit published model metrics" in reasons
    assert "synthetic annotation evidence is present" in reasons
    assert any(reason.startswith("frozen split coverage mismatch") for reason in reasons)
    assert any("lacks two independent judgments" in reason for reason in reasons)
    assert any("evidence snapshot hash mismatch" in reason for reason in reasons)
    with context.sessions() as session:
        project = session.get(AnnotationProjectRecord, project_id)
        assert project is not None
        assert project.status == "open"
        assert session.scalar(select(func.count()).select_from(AnnotationExportRecord)) == 0


def test_postgres_claim_lock_compiles_with_skip_locked() -> None:
    session = Mock(spec=Session)
    session.get_bind.return_value = SimpleNamespace(dialect=SimpleNamespace(name="postgresql"))
    statement = _with_skip_locked(
        select(AnnotationItemRecord),
        cast(Session, session),
    )
    sql = str(statement.compile(dialect=PGDialect()))  # type: ignore[no-untyped-call]
    assert "FOR UPDATE OF annotation_items SKIP LOCKED" in sql
