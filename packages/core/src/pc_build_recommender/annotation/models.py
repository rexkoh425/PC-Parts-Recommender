"""Typed contracts for the production human-annotation workflow.

The API authentication boundary is responsible for verifying the OIDC token.  This
module deliberately carries only the verified issuer/subject pair into the service;
reviewer IDs and roles are always resolved from PostgreSQL and are never accepted
from a judgment payload.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any


class AnnotationTaskType(StrEnum):
    ENTITY_RESOLUTION = "entity_resolution"
    RELEVANCE = "relevance"


class AnnotationRole(StrEnum):
    REVIEWER = "reviewer"
    ADJUDICATOR = "adjudicator"
    ADMIN = "admin"


class AnnotationProjectStatus(StrEnum):
    DRAFT = "draft"
    OPEN = "open"
    FROZEN = "frozen"


class AnnotationItemState(StrEnum):
    PENDING = "pending"
    IN_REVIEW = "in_review"
    NEEDS_ADJUDICATION = "needs_adjudication"
    RESOLVED = "resolved"


class AssignmentPhase(StrEnum):
    REVIEW = "review"
    ADJUDICATION = "adjudication"


class AssignmentStatus(StrEnum):
    LEASED = "leased"
    SUBMITTED = "submitted"
    EXPIRED = "expired"


class EntityResolutionLabel(StrEnum):
    MATCH = "MATCH"
    NON_MATCH = "NON_MATCH"
    UNCERTAIN = "UNCERTAIN"


class AnnotationAuthorizationError(PermissionError):
    """Raised when a verified identity lacks the required active role."""


class AnnotationConflictError(RuntimeError):
    """Raised for an invalid or stale workflow transition."""


class AnnotationFreezeBlockedError(RuntimeError):
    """Raised with every deterministic reason that prevents a frozen release."""

    def __init__(self, reasons: tuple[str, ...]) -> None:
        self.reasons = reasons
        super().__init__("annotation freeze blocked: " + "; ".join(reasons))


@dataclass(frozen=True, slots=True)
class VerifiedOIDCIdentity:
    """Issuer and subject extracted from an already-verified OIDC token."""

    issuer: str
    subject: str

    def __post_init__(self) -> None:
        if not self.issuer.strip() or not self.subject.strip():
            raise ValueError("verified OIDC issuer and subject must not be empty")


@dataclass(frozen=True, slots=True)
class ClaimedAnnotationTask:
    assignment_id: str
    lease_token: str
    project_id: str
    item_id: str
    task_type: AnnotationTaskType
    group_key: str
    target_id: str
    category: str
    context_payload: Mapping[str, Any]
    evidence_payload: Mapping[str, Any]
    context_sha256: str
    evidence_sha256: str
    lease_expires_at: datetime

    def __post_init__(self) -> None:
        object.__setattr__(self, "context_payload", MappingProxyType(dict(self.context_payload)))
        object.__setattr__(self, "evidence_payload", MappingProxyType(dict(self.evidence_payload)))


@dataclass(frozen=True, slots=True)
class AnnotationProjectProgress:
    """Read-only, aggregate-only progress report for a human-label project.

    This deliberately excludes evidence payloads, labels, rationales, reviewer identities, and
    lease secrets.  It is an operational preflight, not a substitute for the strict integrity
    checks that :meth:`AnnotationService.freeze_project` runs before publishing a release.
    """

    project_id: str
    project_status: AnnotationProjectStatus
    task_type: AnnotationTaskType
    observed_at: datetime
    group_count: int
    item_count: int
    item_state_counts: Mapping[str, int]
    judgment_coverage: Mapping[str, int]
    review_assignment_counts: Mapping[str, int]
    adjudication_assignment_counts: Mapping[str, int]
    adjudication_required_count: int
    adjudication_completed_count: int
    synthetic_group_count: int
    synthetic_item_count: int
    preflight_blockers: tuple[str, ...]
    coarse_freeze_preflight_passes: bool
    release_record_present: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "item_state_counts", MappingProxyType(dict(self.item_state_counts))
        )
        object.__setattr__(
            self, "judgment_coverage", MappingProxyType(dict(self.judgment_coverage))
        )
        object.__setattr__(
            self,
            "review_assignment_counts",
            MappingProxyType(dict(self.review_assignment_counts)),
        )
        object.__setattr__(
            self,
            "adjudication_assignment_counts",
            MappingProxyType(dict(self.adjudication_assignment_counts)),
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe, non-sensitive representation for the trusted CLI."""

        return {
            "project_id": self.project_id,
            "project_status": self.project_status.value,
            "task_type": self.task_type.value,
            "observed_at": self.observed_at.isoformat(),
            "groups": self.group_count,
            "items": {
                "total": self.item_count,
                "states": dict(self.item_state_counts),
                "judgment_coverage": dict(self.judgment_coverage),
                "synthetic": self.synthetic_item_count,
            },
            "assignments": {
                "review": dict(self.review_assignment_counts),
                "adjudication": dict(self.adjudication_assignment_counts),
            },
            "adjudication": {
                "required": self.adjudication_required_count,
                "completed": self.adjudication_completed_count,
            },
            "synthetic_groups": self.synthetic_group_count,
            "freeze_preflight": {
                "coarse_gates_pass": self.coarse_freeze_preflight_passes,
                "blockers": list(self.preflight_blockers),
                "strict_freeze_required": True,
                "release_record_present": self.release_record_present,
            },
        }


@dataclass(frozen=True, slots=True)
class AnnotationRelease:
    project_id: str
    release_sha256: str
    manifest_sha256: str
    artifact_directory: Path
    files: Mapping[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "artifact_directory", self.artifact_directory.resolve())
        object.__setattr__(self, "files", MappingProxyType(dict(self.files)))
