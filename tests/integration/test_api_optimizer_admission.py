"""Integration contracts for shared optimizer admission at the HTTP boundary."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient
from services.api.main import create_app
from services.api.settings import ApiSettings


def _build_request() -> dict[str, Any]:
    return {
        "budget_sgd": 2500,
        "workloads": [{"name": "local_ai", "weight": 1}],
        "max_builds": 1,
    }


@pytest.mark.integration
@pytest.mark.parametrize(
    ("max_queue_size", "expected_status", "expected_code", "outcome"),
    (
        (0, 429, "component_replacement_queue_full", "queue_full"),
        (1, 503, "component_replacement_queue_timeout", "queue_timeout"),
    ),
)
def test_replacement_uses_generation_capacity_and_returns_controlled_overload_errors(
    max_queue_size: int,
    expected_status: int,
    expected_code: str,
    outcome: str,
) -> None:
    settings = ApiSettings(
        environment="test",
        data_version="optimizer-admission-data-v1",
        build_generation_max_concurrency=1,
        build_generation_max_queue_size=max_queue_size,
        build_generation_queue_timeout_seconds=0.01,
    )
    app = create_app(settings)

    with TestClient(app) as client:
        portal = client.portal
        assert portal is not None
        controller = app.state.optimizer_admission
        portal.call(controller.acquire, "generate")
        try:
            response = client.post(
                "/v1/builds/build-not-reached/replace",
                json={
                    "category": "gpu",
                    "replacement_product_id": "gpu-not-reached",
                },
                headers={"X-Request-ID": "replacement-admission-contract"},
            )
        finally:
            portal.call(controller.release, "generate")
        metrics = client.get("/metrics").text

    assert response.status_code == expected_status
    assert response.json()["error"]["code"] == expected_code
    assert response.json()["error"]["request_id"] == "replacement-admission-contract"
    assert response.json()["error"]["details"]["operation"] == "replace"
    assert response.json()["error"]["details"]["retryable"] is True
    assert response.headers["retry-after"] == "1"
    assert response.headers["x-data-version"] == "optimizer-admission-data-v1"
    assert (
        'pcbr_optimizer_admission_outcomes_total{operation="replace",'
        f'outcome="{outcome}"}}' in metrics
    )
    assert 'pcbr_optimizer_admission_active{operation="generate"} 0' in metrics
    assert 'pcbr_optimizer_admission_queued{operation="replace"} 0' in metrics
    assert (
        f'route="/v1/builds/{{build_id}}/replace",status="{expected_status}"' in metrics
    )


@pytest.mark.integration
def test_generation_and_replacement_share_the_same_capacity_pool_in_both_directions() -> None:
    settings = ApiSettings(
        environment="test",
        build_generation_max_concurrency=1,
        build_generation_max_queue_size=0,
        build_generation_queue_timeout_seconds=0.01,
    )
    app = create_app(settings)

    with TestClient(app) as client:
        portal = client.portal
        assert portal is not None
        controller = app.state.optimizer_admission
        portal.call(controller.acquire, "replace")
        try:
            response = client.post("/v1/builds/generate", json=_build_request())
        finally:
            portal.call(controller.release, "replace")

    assert response.status_code == 429
    assert response.json()["error"]["code"] == "build_generation_queue_full"
    assert response.json()["error"]["details"]["operation"] == "generate"
