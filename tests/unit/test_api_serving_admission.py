"""Fail-closed serving and optimizer admission contracts."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any, cast

import pytest
from pydantic import ValidationError
from services.api.core_service import CoreRecommendationService
from services.api.errors import ApiError
from services.api.metrics import DomainMetrics
from services.api.middleware import (
    OptimizerAdmissionController,
    OptimizerAdmissionMiddleware,
    RequestBodyLimitMiddleware,
)
from services.api.models import (
    ComponentCategory,
    FreshnessResponse,
    GenerateBuildsRequest,
    ProductSearchRequest,
    ReplacementRequest,
)
from services.api.settings import ApiSettings
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from pc_build_recommender.domain import ComponentCategory as DomainCategory


def _http_scope(path: str = "/v1/builds/generate") -> Scope:
    return cast(
        Scope,
        {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": path,
            "raw_path": path.encode(),
            "query_string": b"",
            "root_path": "",
            "headers": [],
            "client": ("127.0.0.1", 12345),
            "server": ("testserver", 80),
            "state": {"request_id": "serving-admission-test"},
        },
    )


@pytest.mark.asyncio
async def test_optimizer_slot_is_acquired_only_after_the_bounded_body_is_received() -> None:
    upload_started = asyncio.Event()
    finish_upload = asyncio.Event()
    downstream_started = asyncio.Event()
    receive_calls = 0

    async def receive() -> Message:
        nonlocal receive_calls
        receive_calls += 1
        if receive_calls == 1:
            upload_started.set()
            return {"type": "http.request", "body": b"{", "more_body": True}
        await finish_upload.wait()
        return {"type": "http.request", "body": b"}", "more_body": False}

    async def downstream(scope: Scope, receive: Receive, send: Send) -> None:
        downstream_started.set()
        while True:
            message = await receive()
            if not message.get("more_body", False):
                break
        await send({"type": "http.response.start", "status": 204, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    controller = OptimizerAdmissionController(
        max_concurrency=1,
        max_queue_size=1,
        queue_timeout_seconds=1,
    )
    application: ASGIApp = RequestBodyLimitMiddleware(
        OptimizerAdmissionMiddleware(downstream, controller=controller),
        max_body_bytes=1024,
    )
    responses: list[Message] = []

    async def send(message: Message) -> None:
        responses.append(message)

    request = asyncio.create_task(application(_http_scope(), receive, send))
    await asyncio.wait_for(upload_started.wait(), timeout=1)

    assert await controller.snapshot() == (0, 0)
    assert downstream_started.is_set() is False

    finish_upload.set()
    await asyncio.wait_for(request, timeout=1)

    assert downstream_started.is_set() is True
    assert await controller.snapshot() == (0, 0)
    response_status = next(
        item["status"] for item in responses if item["type"] == "http.response.start"
    )
    assert response_status == 204


def _production_service(
    *,
    blockers: tuple[str, ...] = (),
    observed_at: datetime | None = None,
    source_authority_expires_at: datetime | None = None,
    release_artifact_verification: str = "verified",
) -> CoreRecommendationService:
    now = datetime.now(UTC)
    timestamp = observed_at or now
    service = object.__new__(CoreRecommendationService)
    service.settings = ApiSettings(environment="test").model_copy(
        update={"environment": "production", "service_mode": "processed_catalog"}
    )
    service.services = cast(
        Any,
        SimpleNamespace(
            versions=SimpleNamespace(
                data_version="catalog-v1",
                ranking_model="ranker-v1",
                retrieval_model="retriever-v1",
                rule_version="compat-v1",
                optimizer_version="optimizer-v1",
            ),
            catalog=SimpleNamespace(compatibility_evidence_policy="measured"),
        ),
    )
    service.processed_data = cast(
        Any,
        SimpleNamespace(
            products=tuple(
                SimpleNamespace(category=category, updated_at=timestamp, provenance=())
                for category in DomainCategory
            ),
            listings=(SimpleNamespace(listing_id="listing-1"),),
            price_snapshots=(
                SimpleNamespace(listing_id="listing-1", observed_at=timestamp),
            ),
            listing_provenance=(),
            readiness=SimpleNamespace(blockers=lambda: blockers),
            stats=SimpleNamespace(
                product_count=len(DomainCategory),
                matched_listing_count=1,
            ),
        ),
    )
    service._release_artifact_verification = release_artifact_verification
    service._source_authority_expires_at = (
        source_authority_expires_at or now + timedelta(hours=1)
    )
    return service


async def _invoke_decision_operation(
    service: CoreRecommendationService,
    operation: str,
) -> None:
    if operation == "generate":
        await service.generate_builds(
            GenerateBuildsRequest(
                budget_sgd=2500,
                workloads=[{"name": "local_ai", "weight": 1}],
                max_builds=1,
            )
        )
        return
    if operation == "search":
        await service.search_products(ProductSearchRequest(query="gpu"))
        return
    await service.replace_component(
        "build-1",
        ReplacementRequest(
            category=ComponentCategory.GPU,
            replacement_product_id="gpu-1",
        ),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ("generate", "search", "replace"))
@pytest.mark.parametrize("failure", ("legal", "stale", "source_authority", "release"))
async def test_production_decision_operations_recheck_live_serving_eligibility(
    operation: str,
    failure: str,
) -> None:
    now = datetime.now(UTC)
    service = _production_service(
        blockers=("Retailer offer rights are not production-valid.",)
        if failure == "legal"
        else (),
        observed_at=(now - timedelta(days=8)) if failure == "stale" else now,
        source_authority_expires_at=(
            now - timedelta(seconds=1)
            if failure == "source_authority"
            else now + timedelta(hours=1)
        ),
        release_artifact_verification="not_verified" if failure == "release" else "verified",
    )

    with pytest.raises(ApiError) as captured:
        await _invoke_decision_operation(service, operation)

    assert captured.value.status_code == 503
    assert captured.value.code == "production_serving_not_ready"
    assert captured.value.details is not None
    assert captured.value.details["operation"] == operation
    assert captured.value.details["readiness_blockers"]


def test_production_ready_contract_requires_verified_immutable_release() -> None:
    with pytest.raises(
        ValidationError,
        match="production_ready requires a verified immutable serving release",
    ):
        FreshnessResponse(
            data_version="catalog-v1",
            status="fresh",
            catalogue_status="fresh",
            price_status="fresh",
            last_catalog_update=datetime.now(UTC),
            prices_updated_at=datetime.now(UTC),
            stale_after_hours=24,
            catalogue_stale_after_hours=168,
            price_stale_after_hours=24,
            source_count=2,
            product_count=3000,
            listing_count=10000,
            production_ready=True,
            release_artifact_verification="not_verified",
        )


def test_production_ready_metric_rejects_unverified_release_state() -> None:
    now = datetime.now(UTC)
    with pytest.raises(ValueError, match="require verified serving artifacts"):
        DomainMetrics().record_freshness(
            status="fresh",
            catalogue_status="fresh",
            price_status="fresh",
            last_catalog_update=now,
            prices_updated_at=now,
            catalogue_stale_after_hours=168,
            price_stale_after_hours=24,
            production_ready=True,
            product_count=3000,
            listing_count=10000,
            release_blocker_count=0,
            release_artifact_verification="not_verified",
        )


def test_mutation_lock_capacity_is_not_configurable_above_one() -> None:
    with pytest.raises(ValidationError):
        ApiSettings(build_generation_max_concurrency=2)


def test_admin_price_freshness_uses_only_the_latest_observation_per_listing() -> None:
    now = datetime.now(UTC)
    service = object.__new__(CoreRecommendationService)
    service.settings = ApiSettings(environment="test", price_stale_after_hours=24)
    service.processed_data = cast(
        Any,
        SimpleNamespace(
            stats=SimpleNamespace(
                data_version="catalog-v1",
                offer_count=2,
                matched_listing_count=2,
                unmatched_offer_count=0,
                manual_review_count=0,
                rejected_conflict_count=0,
                model_rejected_count=0,
            ),
            price_snapshots=(
                SimpleNamespace(
                    listing_id="listing-a",
                    observed_at=now - timedelta(hours=48),
                ),
                SimpleNamespace(
                    listing_id="listing-a",
                    observed_at=now - timedelta(hours=1),
                ),
                SimpleNamespace(
                    listing_id="listing-b",
                    observed_at=now - timedelta(hours=48),
                ),
            ),
            readiness=SimpleNamespace(
                blockers=lambda: (),
                critical_field_present_by_category={},
                products_by_category={},
            ),
        ),
    )

    response = asyncio.run(service.admin_operations())

    assert response.price_freshness is not None
    assert response.price_freshness.snapshot_count == 2
    assert response.price_freshness.stale_snapshot_count == 1
    assert response.price_freshness.newest_observed_at == now - timedelta(hours=1)
    assert response.notes[0] == (
        "Price freshness evaluates the latest observation for each distinct listing."
    )
