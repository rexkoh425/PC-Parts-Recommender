"""Production-grade human annotation and frozen evidence exports."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

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

if TYPE_CHECKING:
    from .service import AnnotationService


def __getattr__(name: str) -> Any:
    """Keep metadata-only imports independent from the ML and export stack."""

    if name == "AnnotationService":
        from .service import AnnotationService

        return AnnotationService
    raise AttributeError(name)


__all__ = [
    "AnnotationAuthorizationError",
    "AnnotationConflictError",
    "AnnotationFreezeBlockedError",
    "AnnotationItemState",
    "AnnotationProjectProgress",
    "AnnotationProjectStatus",
    "AnnotationRelease",
    "AnnotationRole",
    "AnnotationService",
    "AnnotationTaskType",
    "AssignmentPhase",
    "AssignmentStatus",
    "ClaimedAnnotationTask",
    "EntityResolutionLabel",
    "VerifiedOIDCIdentity",
    "validate_blinded_annotation_payload",
]
