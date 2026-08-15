"""Focused contracts for bounded API request and generation resources."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from typing import Any, cast

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from services.api.main import create_app
from services.api.metrics import RequestMetrics
from services.api.middleware import (
    OptimizerAdmissionController,
    OptimizerAdmissionMiddleware,
    RequestBodyLimitMiddleware,
)
from services.api.service import InMemoryRecommendationService
from services.api.settings import ApiSettings
from starlette.types import ASGIApp, Message, Receive, Scope, Send


def _http_scope(
    *,
    path: str = "/v1/builds/generate",
    headers: Sequence[tuple[bytes, bytes]] = (),
) -> Scope:
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
            "headers": list(headers),
            "client": ("127.0.0.1", 12345),
            "server": ("testserver", 80),
            "state": {"request_id": "resource-test-request"},
        },
    )


async def _empty_receive() -> Message:
    return {"type": "http.request", "body": b"", "more_body": False}


async def _invoke(app: ASGIApp, *, scope: Scope | None = None) -> list[Message]:
    messages: list[Message] = []

    async def send(message: Message) -> None:
        messages.append(message)

    await app(scope or _http_scope(), _empty_receive, send)
    return messages


def _response_status(messages: Sequence[Message]) -> int:
    start = next(message for message in messages if message["type"] == "http.response.start")
    return int(start["status"])


def _response_headers(messages: Sequence[Message]) -> dict[bytes, bytes]:
    start = next(message for message in messages if message["type"] == "http.response.start")
    return dict(start["headers"])


def _response_json(messages: Sequence[Message]) -> dict[str, Any]:
    payload = b"".join(
        message.get("body", b"") for message in messages if message["type"] == "http.response.body"
    )
    return cast(dict[str, Any], json.loads(payload))


@pytest.mark.asyncio
async def test_streamed_body_limit_counts_chunks_without_content_length() -> None:
    downstream_called = False

    async def downstream(scope: Scope, receive: Receive, send: Send) -> None:
        nonlocal downstream_called
        while True:
            message = await receive()
            if not message.get("more_body", False):
                break
        downstream_called = True
        await send({"type": "http.response.start", "status": 204, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    chunks = iter(
        (
            {"type": "http.request", "body": b"123", "more_body": True},
            {"type": "http.request", "body": b"456", "more_body": False},
        )
    )

    async def receive() -> Message:
        return cast(Message, next(chunks))

    messages: list[Message] = []

    async def send(message: Message) -> None:
        messages.append(message)

    middleware = RequestBodyLimitMiddleware(downstream, max_body_bytes=5)
    await middleware(_http_scope(), receive, send)

    assert downstream_called is False
    assert _response_status(messages) == 413
    payload = _response_json(messages)
    assert payload["error"]["code"] == "request_body_too_large"
    assert payload["error"]["request_id"] == "resource-test-request"
    assert payload["error"]["details"]["max_request_body_bytes"] == 5


@pytest.mark.asyncio
async def test_body_limit_rejects_conflicting_content_lengths() -> None:
    async def downstream(scope: Scope, receive: Receive, send: Send) -> None:
        raise AssertionError("invalid framing must not reach the downstream application")

    middleware = RequestBodyLimitMiddleware(downstream, max_body_bytes=1024)
    messages = await _invoke(
        middleware,
        scope=_http_scope(
            headers=((b"content-length", b"10"), (b"content-length", b"11")),
        ),
    )

    assert _response_status(messages) == 400
    assert _response_json(messages)["error"]["code"] == "invalid_content_length"


@pytest.mark.asyncio
async def test_optimizer_admission_is_shared_and_distinguishes_full_queue_from_timeout() -> None:
    release_generation = asyncio.Event()
    first_started = asyncio.Event()
    downstream_calls = 0

    async def downstream(scope: Scope, receive: Receive, send: Send) -> None:
        nonlocal downstream_calls
        downstream_calls += 1
        first_started.set()
        await release_generation.wait()
        await send({"type": "http.response.start", "status": 204, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    metrics = RequestMetrics()
    controller = OptimizerAdmissionController(
        max_concurrency=1,
        max_queue_size=1,
        queue_timeout_seconds=0.05,
        metrics=metrics,
    )
    middleware = OptimizerAdmissionMiddleware(downstream, controller=controller)

    first = asyncio.create_task(_invoke(middleware))
    await asyncio.wait_for(first_started.wait(), timeout=1)
    replacement_scope = _http_scope(path="/v1/builds/build-resource-test/replace")
    second = asyncio.create_task(_invoke(middleware, scope=replacement_scope))
    for _ in range(100):
        if await controller.snapshot() == (1, 1):
            break
        await asyncio.sleep(0.001)
    assert await controller.snapshot() == (1, 1)
    assert await controller.operation_snapshot() == {
        "generate": (1, 0),
        "replace": (0, 1),
    }

    queue_full = await _invoke(middleware, scope=replacement_scope)
    queue_timeout = await asyncio.wait_for(second, timeout=1)
    release_generation.set()
    completed = await asyncio.wait_for(first, timeout=1)

    assert _response_status(queue_full) == 429
    assert _response_json(queue_full)["error"]["code"] == "component_replacement_queue_full"
    assert _response_json(queue_full)["error"]["details"]["operation"] == "replace"
    assert _response_headers(queue_full)[b"retry-after"] == b"1"
    assert _response_status(queue_timeout) == 503
    assert _response_json(queue_timeout)["error"]["code"] == (
        "component_replacement_queue_timeout"
    )
    assert _response_json(queue_timeout)["error"]["details"]["operation"] == "replace"
    assert _response_headers(queue_timeout)[b"retry-after"] == b"1"
    assert _response_status(completed) == 204
    assert downstream_calls == 1
    assert await controller.snapshot() == (0, 0)
    assert await controller.operation_snapshot() == {
        "generate": (0, 0),
        "replace": (0, 0),
    }
    rendered = metrics.render()
    assert (
        'pcbr_optimizer_admission_outcomes_total{operation="generate",outcome="admitted"} 1'
        in rendered
    )
    assert (
        'pcbr_optimizer_admission_outcomes_total{operation="replace",outcome="queue_full"} 1'
        in rendered
    )
    assert (
        'pcbr_optimizer_admission_outcomes_total{operation="replace",outcome="queue_timeout"} 1'
        in rendered
    )
    assert 'pcbr_optimizer_admission_active{operation="generate"} 0' in rendered
    assert 'pcbr_optimizer_admission_queued{operation="replace"} 0' in rendered


@pytest.mark.asyncio
async def test_optimizer_admission_cancellation_does_not_leak_queue_or_capacity() -> None:
    metrics = RequestMetrics()
    controller = OptimizerAdmissionController(
        max_concurrency=1,
        max_queue_size=1,
        queue_timeout_seconds=1,
        metrics=metrics,
    )
    await controller.acquire("generate")
    queued = asyncio.create_task(controller.acquire("replace"))
    for _ in range(100):
        if await controller.snapshot() == (1, 1):
            break
        await asyncio.sleep(0.001)
    assert await controller.snapshot() == (1, 1)

    queued.cancel()
    with pytest.raises(asyncio.CancelledError):
        await queued
    assert await controller.snapshot() == (1, 0)
    await controller.release("generate")

    assert await controller.snapshot() == (0, 0)
    rendered = metrics.render()
    assert (
        'pcbr_optimizer_admission_outcomes_total{operation="replace",outcome="cancelled"} 1'
        in rendered
    )
    assert 'pcbr_optimizer_admission_queued{operation="replace"} 0' in rendered


def test_optimizer_admission_metrics_reject_unbounded_labels_and_invalid_state() -> None:
    metrics = RequestMetrics()

    with pytest.raises(ValueError, match="unsupported optimizer operation"):
        metrics.record_optimizer_admission_transition(
            operation="caller-controlled",
            outcome="admitted",
            wait_seconds=0,
        )
    with pytest.raises(ValueError, match="unsupported optimizer admission outcome"):
        metrics.record_optimizer_admission_transition(
            operation="generate",
            outcome="caller-controlled",
            wait_seconds=0,
        )
    with pytest.raises(ValueError, match="cannot become negative"):
        metrics.record_optimizer_admission_transition(
            operation="replace",
            queued_delta=-1,
        )


def test_create_app_rejects_oversized_body_before_endpoint() -> None:
    settings = ApiSettings(
        environment="test",
        max_request_body_bytes=1024,
        cors_origins=["https://web.example.test"],
        data_version="resource-data-v1",
    )
    service = InMemoryRecommendationService(settings)
    with TestClient(create_app(settings=settings, service=service)) as client:
        response = client.post(
            "/v1/interactions",
            json={
                "event_type": "feedback_submitted",
                "session_id": "session-resource-test",
                "metadata": {"comment": "x" * 2048},
            },
            headers={
                "Origin": "https://web.example.test",
                "X-Request-ID": "oversized-body-request",
            },
        )
        metrics = client.get("/metrics").text

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "request_body_too_large"
    assert response.json()["error"]["request_id"] == "oversized-body-request"
    assert response.headers["x-data-version"] == "resource-data-v1"
    assert response.headers["access-control-allow-origin"] == "https://web.example.test"
    assert service._interactions == []
    assert 'route="/v1/interactions",status="413"' in metrics


def test_openapi_documents_resource_control_responses() -> None:
    with TestClient(create_app(ApiSettings(environment="test"))) as client:
        schema = client.get("/openapi.json").json()

    generation_responses = schema["paths"]["/v1/builds/generate"]["post"]["responses"]
    assert {"413", "429", "503"}.issubset(generation_responses)
    replacement_responses = schema["paths"]["/v1/builds/{build_id}/replace"]["post"][
        "responses"
    ]
    assert {"413", "429", "503"}.issubset(replacement_responses)
    for path in (
        "/v1/products/search",
        "/v1/compatibility/check",
        "/v1/interactions",
    ):
        assert "413" in schema["paths"][path]["post"]["responses"]


@pytest.mark.parametrize(
    ("settings", "message"),
    (
        (
            {
                "environment": "production",
                "docs_enabled": True,
                "cors_origins": ["https://pcbr.example.test"],
            },
            "API documentation must be disabled",
        ),
        (
            {
                "environment": "production",
                "docs_enabled": False,
                "cors_origins": ["*"],
            },
            "wildcard CORS origins are forbidden",
        ),
    ),
)
def test_non_development_http_exposure_settings_fail_closed(
    settings: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        ApiSettings(**settings)


def test_development_http_exposure_defaults_remain_available() -> None:
    settings = ApiSettings(
        environment="development",
        docs_enabled=True,
        cors_origins=["*"],
    )

    with TestClient(create_app(settings)) as client:
        response = client.get("/openapi.json")

    assert response.status_code == 200


def test_create_app_rechecks_mutated_http_exposure_settings() -> None:
    settings = ApiSettings(
        environment="development",
        docs_enabled=True,
        cors_origins=["*"],
    )
    settings.environment = "production"

    with pytest.raises(ValueError, match="API documentation must be disabled"):
        create_app(settings)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("max_request_body_bytes", 1023),
        ("max_request_body_bytes", 16 * 1024 * 1024 + 1),
        ("build_generation_max_concurrency", 0),
        ("build_generation_max_concurrency", 17),
        ("build_generation_max_queue_size", -1),
        ("build_generation_max_queue_size", 257),
        ("build_generation_queue_timeout_seconds", 0),
        ("build_generation_queue_timeout_seconds", 61),
    ),
)
def test_resource_control_settings_reject_unsafe_values(field: str, value: int) -> None:
    with pytest.raises(ValidationError):
        ApiSettings(**{field: value})
