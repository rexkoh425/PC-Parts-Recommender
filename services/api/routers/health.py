"""Liveness, readiness, and data-freshness endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Response, status
from fastapi.responses import PlainTextResponse

from services.api.dependencies import ApplicationDependency, SettingsDependency
from services.api.metrics import DOMAIN_METRICS, REQUEST_METRICS, record_admin_operations_response
from services.api.models import (
    AdminOperationsResponse,
    FreshnessResponse,
    HealthResponse,
    ReadinessResponse,
)
from services.api.routers.openapi import SERVICE_ERROR
from services.api.routers.response_contracts import validate_service_response

router = APIRouter(tags=["system"])


def _record_freshness_response(response: FreshnessResponse) -> None:
    DOMAIN_METRICS.record_freshness(
        status=response.status,
        catalogue_status=response.catalogue_status,
        price_status=response.price_status,
        last_catalog_update=response.last_catalog_update,
        prices_updated_at=response.prices_updated_at,
        catalogue_stale_after_hours=response.catalogue_stale_after_hours,
        price_stale_after_hours=response.price_stale_after_hours,
        production_ready=response.production_ready,
        product_count=response.product_count,
        listing_count=response.listing_count,
        release_blocker_count=len(response.readiness_blockers),
        release_artifact_verification=response.release_artifact_verification,
    )


async def refresh_freshness_metrics(
    application: ApplicationDependency,
) -> FreshnessResponse | None:
    """Refresh scrape-visible freshness state without requiring the public freshness route."""

    try:
        response = validate_service_response(await application.freshness(), FreshnessResponse)
        _record_freshness_response(response)
    except Exception:
        DOMAIN_METRICS.record_freshness_probe_failure()
        return None
    return response


@router.get("/metrics", response_class=PlainTextResponse, include_in_schema=False)
async def metrics(application: ApplicationDependency) -> PlainTextResponse:
    """Expose bounded process metrics; production ingress must keep this path private."""

    await refresh_freshness_metrics(application)
    operations = validate_service_response(
        await application.admin_operations(), AdminOperationsResponse
    )
    record_admin_operations_response(operations)
    return PlainTextResponse(
        REQUEST_METRICS.render() + DOMAIN_METRICS.render(),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )


@router.get("/health", response_model=HealthResponse, include_in_schema=False)
@router.get("/health/live", response_model=HealthResponse)
async def health(settings: SettingsDependency) -> HealthResponse:
    return HealthResponse(status="ok", service=settings.app_name, environment=settings.environment)


@router.get(
    "/ready",
    response_model=ReadinessResponse,
    responses={503: {"model": ReadinessResponse, "description": "A readiness check failed."}},
    include_in_schema=False,
)
@router.get(
    "/health/ready",
    response_model=ReadinessResponse,
    responses={503: {"model": ReadinessResponse, "description": "A readiness check failed."}},
)
async def readiness(
    response: Response,
    application: ApplicationDependency,
    settings: SettingsDependency,
) -> ReadinessResponse:
    checks = await application.readiness_checks()
    ready = bool(checks) and all(value == "ready" for value in checks.values())
    if not ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return ReadinessResponse(
        status="ready" if ready else "not_ready",
        checks=checks,
        data_version=settings.data_version,
        ranking_model=settings.ranking_model_version,
        rule_version=settings.compatibility_rule_version,
        solver_version=settings.solver_version,
    )


@router.get(
    "/v1/system/freshness",
    response_model=FreshnessResponse,
    responses=SERVICE_ERROR,
)
async def freshness(application: ApplicationDependency) -> FreshnessResponse:
    response = validate_service_response(await application.freshness(), FreshnessResponse)
    _record_freshness_response(response)
    return response
