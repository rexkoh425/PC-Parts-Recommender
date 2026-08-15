"""Non-blocking recommendation-feedback ingestion endpoint."""

from __future__ import annotations

import hashlib
import json
from typing import Annotated, Any

from fastapi import APIRouter, Header, Request, status

from services.api.dependencies import (
    ApplicationDependency,
    ImpressionSignerDependency,
    SettingsDependency,
)
from services.api.errors import ApiError
from services.api.impressions import ACTOR_COOKIE_NAME, ImpressionClaims, InvalidImpressionToken
from services.api.metrics import DOMAIN_METRICS
from services.api.models import CanonicalInteractionEvent, InteractionAccepted, InteractionEvent
from services.api.routers.openapi import (
    CONFLICT_ERROR,
    PAYLOAD_TOO_LARGE_ERROR,
    SERVICE_ERROR,
    UNAUTHORIZED_ERROR,
    VALIDATION_ERROR,
)

router = APIRouter(prefix="/v1/interactions", tags=["interactions"])

_TRUSTED_EVENTS_BY_IMPRESSION_KIND = {
    "product_search_result": frozenset(
        {"component_viewed", "recommendation_dismissed", "feedback_submitted"}
    ),
    "build_result": frozenset(
        {
            "build_viewed",
            "build_saved",
            "build_shared",
            "component_replaced",
            "recommendation_dismissed",
            "feedback_submitted",
        }
    ),
    "build_component_result": frozenset({"component_viewed", "retailer_clicked"}),
}
_RESERVED_ATTRIBUTION_METADATA = frozenset(
    {
        "build_id",
        "actor_id",
        "data_version",
        "idempotency_key_sha256",
        "idempotency_payload_sha256",
        "impression_id",
        "model_version",
        "product_id",
        "query_id",
        "rank_position",
        "rule_version",
        "trust_level",
    }
)

IdempotencyKeyHeader = Annotated[
    str | None,
    Header(alias="Idempotency-Key", min_length=8, max_length=160),
]


def _reject_claim_mismatch(event: InteractionEvent, claims: ImpressionClaims) -> None:
    expected = {
        "query_id": claims.query_id,
        "product_id": claims.product_id,
        "build_id": claims.build_id,
        "rank_position": claims.rank_position,
        "model_version": claims.model_version,
        "data_version": claims.data_version,
        "rule_version": claims.rule_version,
    }
    mismatches = [
        field
        for field, value in expected.items()
        if getattr(event, field) is not None and getattr(event, field) != value
    ]
    if mismatches:
        raise ApiError(
            status_code=409,
            code="impression_claim_mismatch",
            message="The interaction does not match the displayed recommendation.",
            details={"fields": mismatches},
        )


def _payload_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonicalize_event(
    event: InteractionEvent,
    *,
    idempotency_key: str | None,
    actor_token: str | None,
    signer: ImpressionSignerDependency,
    settings: SettingsDependency,
) -> CanonicalInteractionEvent:
    reserved_metadata = sorted(_RESERVED_ATTRIBUTION_METADATA.intersection(event.metadata))
    if reserved_metadata:
        raise ApiError(
            status_code=422,
            code="reserved_interaction_metadata",
            message="Interaction attribution fields are server-controlled.",
            details={"fields": reserved_metadata},
        )
    base = event.model_dump(mode="json", exclude={"impression_token"})
    token = event.impression_token
    if token is not None:
        if idempotency_key is None:
            raise ApiError(
                status_code=422,
                code="interaction_idempotency_required",
                message="Signed interactions require an Idempotency-Key header.",
            )
        try:
            claims = signer.verify(token, actor_token=actor_token)
        except InvalidImpressionToken as error:
            raise ApiError(
                status_code=401,
                code="invalid_impression",
                message="The recommendation impression is invalid or expired.",
            ) from error
        if event.event_type not in _TRUSTED_EVENTS_BY_IMPRESSION_KIND[claims.kind]:
            raise ApiError(
                status_code=422,
                code="impression_event_mismatch",
                message="This interaction type is not valid for the displayed result.",
            )
        _reject_claim_mismatch(event, claims)
        base.update(
            {
                "build_id": claims.build_id,
                "data_version": claims.data_version,
                "impression_id": claims.impression_id,
                "model_version": claims.model_version,
                "product_id": claims.product_id,
                "query_id": claims.query_id,
                "rank_position": claims.rank_position,
                "rule_version": claims.rule_version,
                "session_id": claims.actor_id,
                "trust_level": "verified_impression",
            }
        )
    else:
        if not settings.is_development_environment and any(
            value is not None
            for value in (event.query_id, event.product_id, event.build_id, event.rank_position)
        ):
            raise ApiError(
                status_code=422,
                code="verified_impression_required",
                message="Ranked-result interactions require a server-issued impression.",
            )
        base.update(
            {
                "data_version": settings.data_version,
                "impression_id": None,
                "model_version": settings.ranking_model_version,
                "rule_version": settings.compatibility_rule_version,
                "trust_level": "legacy_untrusted",
            }
        )
    canonical_session_id = str(base["session_id"])
    key_sha256 = (
        signer.idempotency_key_sha256(session_id=canonical_session_id, key=idempotency_key)
        if idempotency_key is not None
        else None
    )
    base["idempotency_key_sha256"] = key_sha256
    base["idempotency_payload_sha256"] = _payload_sha256(base) if key_sha256 is not None else None
    return CanonicalInteractionEvent.model_validate(base)


@router.post(
    "",
    response_model=InteractionAccepted,
    status_code=status.HTTP_202_ACCEPTED,
    responses={
        **UNAUTHORIZED_ERROR,
        **VALIDATION_ERROR,
        **CONFLICT_ERROR,
        **PAYLOAD_TOO_LARGE_ERROR,
        **SERVICE_ERROR,
    },
)
async def record_interaction(
    event: InteractionEvent,
    http_request: Request,
    application: ApplicationDependency,
    settings: SettingsDependency,
    signer: ImpressionSignerDependency,
    idempotency_key: IdempotencyKeyHeader = None,
) -> InteractionAccepted:
    canonical_event = _canonicalize_event(
        event,
        idempotency_key=idempotency_key,
        actor_token=http_request.cookies.get(ACTOR_COOKIE_NAME),
        signer=signer,
        settings=settings,
    )
    response = await application.record_interaction(canonical_event)
    if not response.replayed:
        DOMAIN_METRICS.record_interaction(event_type=canonical_event.event_type)
    return response
