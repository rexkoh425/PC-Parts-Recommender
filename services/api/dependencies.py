"""FastAPI dependencies that expose application interfaces, not implementations."""

from __future__ import annotations

import hmac
from typing import Annotated, cast

from fastapi import Depends, Request

from services.api.errors import ApiError
from services.api.impressions import ImpressionSigner
from services.api.service import RecommendationApplication
from services.api.settings import ApiSettings


def get_application(request: Request) -> RecommendationApplication:
    return cast(RecommendationApplication, request.app.state.application_service)


def get_settings(request: Request) -> ApiSettings:
    return cast(ApiSettings, request.app.state.settings)


def get_impression_signer(request: Request) -> ImpressionSigner:
    return cast(ImpressionSigner, request.app.state.impression_signer)


ApplicationDependency = Annotated[RecommendationApplication, Depends(get_application)]
SettingsDependency = Annotated[ApiSettings, Depends(get_settings)]
ImpressionSignerDependency = Annotated[ImpressionSigner, Depends(get_impression_signer)]


def require_admin(request: Request, settings: SettingsDependency) -> None:
    """Require an operator token without advertising a disabled admin surface."""

    configured = settings.admin_token
    if configured is None:
        raise ApiError(
            status_code=404,
            code="admin_surface_disabled",
            message="The admin surface is not enabled for this deployment.",
        )
    supplied = request.headers.get("X-PCBR-Admin-Token", "")
    if not hmac.compare_digest(supplied, configured.get_secret_value()):
        raise ApiError(
            status_code=401,
            code="admin_authentication_required",
            message="A valid administrator token is required.",
        )


AdminDependency = Annotated[None, Depends(require_admin)]
