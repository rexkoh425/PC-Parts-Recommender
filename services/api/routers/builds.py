"""Complete-build generation, retrieval, and replacement endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Response, status

from services.api.dependencies import ApplicationDependency
from services.api.metrics import DOMAIN_METRICS
from services.api.models import (
    BuildShareCreated,
    BuildShareRevoked,
    BuildResult,
    GenerateBuildsRequest,
    GenerateBuildsResponse,
    PublicBuildShare,
    ReplacementRequest,
    ReplacementResponse,
    RevokeBuildShareRequest,
)
from services.api.routers.openapi import (
    CONFLICT_ERROR,
    GENERATION_RATE_LIMIT_ERROR,
    NOT_FOUND_ERROR,
    PAYLOAD_TOO_LARGE_ERROR,
    SERVICE_ERROR,
    VALIDATION_ERROR,
)
from services.api.routers.response_contracts import validate_service_response

router = APIRouter(prefix="/v1", tags=["builds"])


def _record_build_generation(response: GenerateBuildsResponse) -> None:
    """Record bounded optimizer and performance outcomes after validation."""

    DOMAIN_METRICS.record_build_generation(
        outcome=response.status.value,
        solver_status=response.solver_status.value,
        solver_ran=response.solver_ran,
        build_count=len(response.builds),
        validator_rejections=response.solver_validator_rejections,
        profile_statuses=tuple(
            (outcome.profile.value, outcome.status.value)
            for outcome in response.solver_profile_statuses
        ),
    )
    DOMAIN_METRICS.record_performance_signals(
        signals=tuple(
            (signal.basis, signal.confidence, signal.decision)
            for build in response.builds
            for component in build.components
            for signal in component.performance_signals
        )
    )


@router.post(
    "/builds/generate",
    response_model=GenerateBuildsResponse,
    status_code=status.HTTP_200_OK,
    responses={
        **VALIDATION_ERROR,
        **CONFLICT_ERROR,
        **PAYLOAD_TOO_LARGE_ERROR,
        **GENERATION_RATE_LIMIT_ERROR,
        **SERVICE_ERROR,
    },
)
async def generate_builds(
    request: GenerateBuildsRequest, application: ApplicationDependency
) -> GenerateBuildsResponse:
    response = validate_service_response(
        await application.generate_builds(request), GenerateBuildsResponse
    )
    _record_build_generation(response)
    return response


@router.get(
    "/requests/{request_id}/builds",
    response_model=GenerateBuildsResponse,
    responses={**NOT_FOUND_ERROR, **SERVICE_ERROR},
)
async def get_request_builds(
    request_id: str, application: ApplicationDependency
) -> GenerateBuildsResponse:
    return await application.get_request_builds(request_id)


@router.get(
    "/builds/{build_id}",
    response_model=BuildResult,
    responses={**NOT_FOUND_ERROR, **SERVICE_ERROR},
)
async def get_build(build_id: str, application: ApplicationDependency) -> BuildResult:
    return await application.get_build(build_id)


@router.post(
    "/builds/{build_id}/shares",
    response_model=BuildShareCreated,
    status_code=status.HTTP_201_CREATED,
    responses={**NOT_FOUND_ERROR, **SERVICE_ERROR},
)
async def create_build_share(
    build_id: str,
    response: Response,
    application: ApplicationDependency,
) -> BuildShareCreated:
    response.headers["Cache-Control"] = "no-store"
    response.headers["Vary"] = "Origin"
    return await application.create_build_share(build_id)


@router.get(
    "/build-shares/{share_id}",
    response_model=PublicBuildShare,
    responses={**NOT_FOUND_ERROR, **SERVICE_ERROR},
)
async def get_build_share(share_id: str, application: ApplicationDependency) -> PublicBuildShare:
    return await application.get_build_share(share_id)


@router.post(
    "/build-shares/{share_id}/revoke",
    response_model=BuildShareRevoked,
    responses={**NOT_FOUND_ERROR, **SERVICE_ERROR},
)
async def revoke_build_share(
    share_id: str,
    request: RevokeBuildShareRequest,
    application: ApplicationDependency,
) -> BuildShareRevoked:
    return await application.revoke_build_share(share_id, request)


@router.post(
    "/builds/{build_id}/replace",
    response_model=ReplacementResponse,
    responses={
        **NOT_FOUND_ERROR,
        **VALIDATION_ERROR,
        **CONFLICT_ERROR,
        **PAYLOAD_TOO_LARGE_ERROR,
        **SERVICE_ERROR,
    },
)
async def replace_component(
    build_id: str,
    request: ReplacementRequest,
    application: ApplicationDependency,
) -> ReplacementResponse:
    response = validate_service_response(
        await application.replace_component(build_id, request), ReplacementResponse
    )
    DOMAIN_METRICS.record_component_replacement(
        solver_status=response.solver_status.value,
        solver_ran=response.solver_ran,
    )
    DOMAIN_METRICS.record_performance_signals(
        signals=tuple(
            (signal.basis, signal.confidence, signal.decision)
            for component in response.build.components
            for signal in component.performance_signals
        )
    )
    return response
