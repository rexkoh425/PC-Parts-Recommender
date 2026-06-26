"""Read-only, administrator-authenticated operational status."""

from __future__ import annotations

from fastapi import APIRouter, Response

from services.api.dependencies import AdminDependency, ApplicationDependency
from services.api.metrics import record_admin_operations_response
from services.api.models import AdminOperationsResponse
from services.api.routers.openapi import NOT_FOUND_ERROR, SERVICE_ERROR
from services.api.routers.response_contracts import validate_service_response

router = APIRouter(prefix="/v1/admin", tags=["admin"])


@router.get(
    "/operations",
    response_model=AdminOperationsResponse,
    responses={
        **NOT_FOUND_ERROR,
        401: {"description": "Administrator token required."},
        **SERVICE_ERROR,
    },
)
async def operations(
    response: Response,
    _administrator: AdminDependency,
    application: ApplicationDependency,
) -> AdminOperationsResponse:
    response.headers["Cache-Control"] = "no-store"
    response.headers["Vary"] = "Origin, X-PCBR-Admin-Token"
    operations_response = validate_service_response(
        await application.admin_operations(), AdminOperationsResponse
    )
    record_admin_operations_response(operations_response)
    return operations_response
