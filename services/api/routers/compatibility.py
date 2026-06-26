"""Auditable compatibility-check endpoint."""

from __future__ import annotations

from fastapi import APIRouter

from services.api.dependencies import ApplicationDependency
from services.api.metrics import DOMAIN_METRICS
from services.api.models import CompatibilityCheckRequest, CompatibilityCheckResponse
from services.api.routers.openapi import PAYLOAD_TOO_LARGE_ERROR, VALIDATION_ERROR
from services.api.routers.response_contracts import validate_service_response

router = APIRouter(prefix="/v1/compatibility", tags=["compatibility"])


@router.post(
    "/check",
    response_model=CompatibilityCheckResponse,
    responses={**VALIDATION_ERROR, **PAYLOAD_TOO_LARGE_ERROR},
)
async def check_compatibility(
    request: CompatibilityCheckRequest, application: ApplicationDependency
) -> CompatibilityCheckResponse:
    response = validate_service_response(
        await application.check_compatibility(request), CompatibilityCheckResponse
    )
    DOMAIN_METRICS.record_compatibility(
        status=response.status.value,
        is_feasible=response.is_feasible,
        check_statuses=tuple(check.status.value for check in response.checks),
    )
    return response
