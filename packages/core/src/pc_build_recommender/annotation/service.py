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

# TODO: rest of this module still to come.
