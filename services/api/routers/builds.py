"""Complete-build generation, retrieval, and replacement endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Request, Response, status

from services.api.dependencies import (
    ApplicationDependency,
    ImpressionSignerDependency,
    SettingsDependency,
)
from services.api.impressions import ImpressionSigner, prepare_impression_response
from services.api.metrics import DOMAIN_METRICS
from services.api.models import (
    BuildShareCreated,
    BuildShareRevoked,
    BuildSummary,
    GenerateBuildsRequest,
    GenerateBuildsResponse,
    PublicBuildShare,
    ReplacementRequest,
    ReplacementResponse,
    RevokeBuildShareRequest,
)
from services.api.routers.openapi import (
    CONFLICT_ERROR,
    NOT_FOUND_ERROR,
    OPTIMIZER_RATE_LIMIT_ERROR,
    PAYLOAD_TOO_LARGE_ERROR,
    SERVICE_ERROR,
    VALIDATION_ERROR,
)
from services.api.routers.response_contracts import validate_service_response

router = APIRouter(prefix="/v1", tags=["builds"])


def _with_build_impressions(
    response: GenerateBuildsResponse,
    *,
    signer: ImpressionSigner,
    actor_id: str,
) -> GenerateBuildsResponse:
    builds: list[BuildSummary] = []
    for rank_position, build in enumerate(response.builds, start=1):
        components = [
            component.model_copy(
                update={
                    "impression_token": signer.issue(
                        actor_id=actor_id,
                        query_id=response.request_id,
                        kind="build_component_result",
                        rank_position=rank_position,
                        build_id=build.build_id,
                        product_id=component.product_id,
                        model_version=response.ranking_model,
                        data_version=response.data_version,
                        rule_version=response.rule_version,
                    )
                },
                deep=True,
            )
            for component in build.components
        ]
        builds.append(
            build.model_copy(
                update={
                    "components": components,
                    "impression_token": signer.issue(
                        actor_id=actor_id,
                        query_id=response.request_id,
                        kind="build_result",
                        rank_position=rank_position,
                        build_id=build.build_id,
                        model_version=response.ranking_model,
                        data_version=response.data_version,
                        rule_version=response.rule_version,
                    ),
                },
                deep=True,
            )
        )
    return response.model_copy(update={"builds": builds}, deep=True)


def _with_build_summary_impressions(
    build: BuildSummary,
    *,
    signer: ImpressionSigner,
    actor_id: str,
    rank_position: int = 1,
) -> BuildSummary:
    components = [
        component.model_copy(
            update={
                "impression_token": signer.issue(
                    actor_id=actor_id,
                    query_id=build.request_id,
                    kind="build_component_result",
                    rank_position=rank_position,
                    build_id=build.build_id,
                    product_id=component.product_id,
                    model_version=build.ranking_model,
                    data_version=build.data_version,
                    rule_version=build.rule_version,
                )
            },
            deep=True,
        )
        for component in build.components
    ]
    return build.model_copy(
        update={
            "components": components,
            "impression_token": signer.issue(
                actor_id=actor_id,
                query_id=build.request_id,
                kind="build_result",
                rank_position=rank_position,
                build_id=build.build_id,
                model_version=build.ranking_model,
                data_version=build.data_version,
                rule_version=build.rule_version,
            ),
        },
        deep=True,
    )


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
        **OPTIMIZER_RATE_LIMIT_ERROR,
        **SERVICE_ERROR,
    },
)
async def generate_builds(
    request: GenerateBuildsRequest,
    http_request: Request,
    http_response: Response,
    application: ApplicationDependency,
    signer: ImpressionSignerDependency,
    settings: SettingsDependency,
) -> GenerateBuildsResponse:
    response = validate_service_response(
        await application.generate_builds(request), GenerateBuildsResponse
    )
    actor_id = prepare_impression_response(
        http_request,
        http_response,
        signer=signer,
        secure_cookie=not settings.is_development_environment,
    )
    response = _with_build_impressions(response, signer=signer, actor_id=actor_id)
    _record_build_generation(response)
    return response


@router.get(
    "/requests/{request_id}/builds",
    response_model=GenerateBuildsResponse,
    responses={**NOT_FOUND_ERROR, **SERVICE_ERROR},
)
async def get_request_builds(
    request_id: str,
    http_request: Request,
    http_response: Response,
    application: ApplicationDependency,
    signer: ImpressionSignerDependency,
    settings: SettingsDependency,
) -> GenerateBuildsResponse:
    response = await application.get_request_builds(request_id)
    actor_id = prepare_impression_response(
        http_request,
        http_response,
        signer=signer,
        secure_cookie=not settings.is_development_environment,
    )
    return _with_build_impressions(response, signer=signer, actor_id=actor_id)


@router.get(
    "/builds/{build_id}",
    response_model=BuildSummary,
    responses={**NOT_FOUND_ERROR, **SERVICE_ERROR},
)
async def get_build(
    build_id: str,
    http_request: Request,
    http_response: Response,
    application: ApplicationDependency,
    signer: ImpressionSignerDependency,
    settings: SettingsDependency,
) -> BuildSummary:
    build = await application.get_build(build_id)
    actor_id = prepare_impression_response(
        http_request,
        http_response,
        signer=signer,
        secure_cookie=not settings.is_development_environment,
    )
    return _with_build_summary_impressions(build, signer=signer, actor_id=actor_id)


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
        **OPTIMIZER_RATE_LIMIT_ERROR,
        **SERVICE_ERROR,
    },
)
async def replace_component(
    build_id: str,
    request: ReplacementRequest,
    http_request: Request,
    http_response: Response,
    application: ApplicationDependency,
    signer: ImpressionSignerDependency,
    settings: SettingsDependency,
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
    actor_id = prepare_impression_response(
        http_request,
        http_response,
        signer=signer,
        secure_cookie=not settings.is_development_environment,
    )
    return response.model_copy(
        update={
            "build": _with_build_summary_impressions(
                response.build,
                signer=signer,
                actor_id=actor_id,
            )
        },
        deep=True,
    )
