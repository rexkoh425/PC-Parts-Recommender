"""Non-blocking recommendation-feedback ingestion endpoint."""

from __future__ import annotations

from fastapi import APIRouter, status

from services.api.dependencies import ApplicationDependency, SettingsDependency
from services.api.metrics import DOMAIN_METRICS
from services.api.models import InteractionAccepted, InteractionEvent
from services.api.routers.openapi import PAYLOAD_TOO_LARGE_ERROR, VALIDATION_ERROR

router = APIRouter(prefix="/v1/interactions", tags=["interactions"])


@router.post(
    "",
    response_model=InteractionAccepted,
    status_code=status.HTTP_202_ACCEPTED,
    responses={**VALIDATION_ERROR, **PAYLOAD_TOO_LARGE_ERROR},
)
async def record_interaction(
    event: InteractionEvent,
    application: ApplicationDependency,
    settings: SettingsDependency,
) -> InteractionAccepted:
    # Client-supplied version fields remain accepted for API compatibility, but they are never
    # trusted as evidence. The active server release is the sole authority for stored versions.
    canonical_event = event.model_copy(
        update={
            "model_version": settings.ranking_model_version,
            "data_version": settings.data_version,
            "rule_version": settings.compatibility_rule_version,
        },
        deep=True,
    )
    response = await application.record_interaction(canonical_event)
    DOMAIN_METRICS.record_interaction(event_type=canonical_event.event_type)
    return response
