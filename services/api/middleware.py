"""Request correlation, version headers, and compact structured logging."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import time
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from services.api.errors import error_payload
from services.api.metrics import REQUEST_METRICS
from services.api.settings import ApiSettings

_ROUTE_TEMPLATE_HINT_STATE_KEY = "_pcbr_route_template"


class BuildGenerationQueueFullError(RuntimeError):
    """Raised when a generation request cannot enter the bounded wait queue."""


class BuildGenerationQueueTimeoutError(RuntimeError):
    """Raised when a queued generation request does not obtain capacity in time."""


class BuildGenerationAdmissionController:
    """Bound concurrent build generation and the number of callers waiting for it."""

    def __init__(
        self,
        *,
        max_concurrency: int,
        max_queue_size: int,
        queue_timeout_seconds: float,
    ) -> None:
        self.max_concurrency = max_concurrency
        self.max_queue_size = max_queue_size
        self.queue_timeout_seconds = queue_timeout_seconds
        self._condition = asyncio.Condition()
        self._active = 0
        self._queued = 0

    async def acquire(self) -> None:
        """Reserve one execution slot or raise a bounded-admission error."""

        async with self._condition:
            if self._active < self.max_concurrency:
                self._active += 1
                return
            if self._queued >= self.max_queue_size:
                raise BuildGenerationQueueFullError

            self._queued += 1
            try:
                async with asyncio.timeout(self.queue_timeout_seconds):
                    while self._active >= self.max_concurrency:
                        await self._condition.wait()
                    self._active += 1
            except TimeoutError as error:
                raise BuildGenerationQueueTimeoutError from error
            finally:
                self._queued -= 1

    async def release(self) -> None:
        """Release one execution slot and wake one queued caller."""

        async with self._condition:
            if self._active <= 0:
                raise RuntimeError("build-generation admission slot released without acquire")
            self._active -= 1
            self._condition.notify(1)

    async def snapshot(self) -> tuple[int, int]:
        """Return active and queued counts from one consistent event-loop snapshot."""

        async with self._condition:
            return self._active, self._queued


def _scope_request_id(scope: Scope) -> str | None:
    state = scope.get("state")
    if not isinstance(state, dict):
        return None
    request_id = state.get("request_id")
    return str(request_id) if request_id is not None else None


def _set_route_template_hint(scope: Scope, route_template: str) -> None:
    state = scope.get("state")
    if isinstance(state, dict):
        state.setdefault(_ROUTE_TEMPLATE_HINT_STATE_KEY, route_template)


async def _send_resource_error(
    *,
    scope: Scope,
    receive: Receive,
    send: Send,
    status_code: int,
    code: str,
    message: str,
    details: dict[str, Any],
    headers: dict[str, str] | None = None,
) -> None:
    response = JSONResponse(
        status_code=status_code,
        content=error_payload(
            code=code,
            message=message,
            request_id=_scope_request_id(scope),
            details=details,
        ),
        headers=headers,
    )
    await response(scope, receive, send)


class BuildGenerationAdmissionMiddleware:
    """Apply bounded admission only to the expensive build-generation endpoint.

    A request that arrives after the wait queue is full receives ``429`` immediately. A
    request that entered the queue but could not obtain execution capacity before the queue
    deadline receives ``503``. Both responses are retryable and include ``Retry-After``.
    """

    _PATH = "/v1/builds/generate"

    def __init__(
        self,
        app: ASGIApp,
        controller: BuildGenerationAdmissionController,
    ) -> None:
        self.app = app
        self.controller = controller

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if (
            scope["type"] != "http"
            or scope.get("method") != "POST"
            or scope.get("path") != self._PATH
        ):
            await self.app(scope, receive, send)
            return

        retry_after = str(max(1, math.ceil(self.controller.queue_timeout_seconds)))
        _set_route_template_hint(scope, self._PATH)
        try:
            await self.controller.acquire()
        except BuildGenerationQueueFullError:
            await _send_resource_error(
                scope=scope,
                receive=receive,
                send=send,
                status_code=429,
                code="build_generation_queue_full",
                message="Build-generation capacity is full; retry after the current queue drains.",
                details={
                    "max_concurrency": self.controller.max_concurrency,
                    "max_queue_size": self.controller.max_queue_size,
                    "retryable": True,
                },
                headers={"Retry-After": retry_after},
            )
            return
        except BuildGenerationQueueTimeoutError:
            await _send_resource_error(
                scope=scope,
                receive=receive,
                send=send,
                status_code=503,
                code="build_generation_queue_timeout",
                message=(
                    "Build-generation capacity did not become available before the queue deadline."
                ),
                details={
                    "queue_timeout_seconds": self.controller.queue_timeout_seconds,
                    "retryable": True,
                },
                headers={"Retry-After": retry_after},
            )
            return

        try:
            await self.app(scope, receive, send)
        finally:
            await self.controller.release()


class _RequestBodyTooLargeError(RuntimeError):
    """Internal signal raised before oversized body bytes reach request parsing."""


class RequestBodyLimitMiddleware:
    """Reject declared or streamed HTTP request bodies above a fixed byte limit."""

    _STATIC_BODY_ROUTES = frozenset(
        {
            "/v1/builds/generate",
            "/v1/compatibility/check",
            "/v1/interactions",
            "/v1/products/search",
        }
    )

    def __init__(self, app: ASGIApp, max_body_bytes: int) -> None:
        self.app = app
        self.max_body_bytes = max_body_bytes

    @staticmethod
    def _content_length(scope: Scope) -> int | None:
        values: list[str] = []
        for name, raw_value in scope.get("headers", []):
            if name.lower() != b"content-length":
                continue
            values.extend(part.strip() for part in raw_value.decode("latin-1").split(","))
        if not values:
            return None
        if any(not value.isdecimal() for value in values):
            raise ValueError("content-length must be a non-negative integer")
        lengths = {int(value) for value in values}
        if len(lengths) != 1:
            raise ValueError("conflicting content-length values")
        return lengths.pop()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path")
        if path in self._STATIC_BODY_ROUTES:
            _set_route_template_hint(scope, str(path))

        try:
            declared_length = self._content_length(scope)
        except ValueError as error:
            await _send_resource_error(
                scope=scope,
                receive=receive,
                send=send,
                status_code=400,
                code="invalid_content_length",
                message="The Content-Length header is invalid.",
                details={"reason": str(error)},
            )
            return

        if declared_length is not None and declared_length > self.max_body_bytes:
            await self._reject(scope=scope, receive=receive, send=send)
            return

        observed_bytes = 0
        response_started = False

        async def limited_receive() -> Message:
            nonlocal observed_bytes
            message = await receive()
            if message["type"] == "http.request":
                observed_bytes += len(message.get("body", b""))
                if observed_bytes > self.max_body_bytes:
                    raise _RequestBodyTooLargeError
            return message

        async def tracked_send(message: Message) -> None:
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, tracked_send)
        except _RequestBodyTooLargeError:
            if response_started:
                raise
            await self._reject(scope=scope, receive=receive, send=send)

    async def _reject(self, *, scope: Scope, receive: Receive, send: Send) -> None:
        await _send_resource_error(
            scope=scope,
            receive=receive,
            send=send,
            status_code=413,
            code="request_body_too_large",
            message="The request body exceeds the configured byte limit.",
            details={"max_request_body_bytes": self.max_body_bytes},
        )


class JsonFormatter(logging.Formatter):
    """Small JSON formatter suitable for local containers and log collectors."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key in (
            "request_id",
            "method",
            "path",
            "status_code",
            "duration_ms",
            "data_version",
            "ranking_model",
            "rule_version",
            "solver_version",
            "error_type",
        ):
            if hasattr(record, key):
                payload[key] = getattr(record, key)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, separators=(",", ":"))


def configure_logging(settings: ApiSettings) -> None:
    logger = logging.getLogger("pc_build_recommender.api")
    logger.setLevel(settings.log_level.upper())
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
    logger.propagate = False


class RequestContextMiddleware:
    """Attach a request ID and immutable runtime versions to every HTTP response."""

    def __init__(self, app: ASGIApp, settings: ApiSettings) -> None:
        self.app = app
        self.settings = settings
        self.logger = logging.getLogger("pc_build_recommender.api.requests")

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = {key.lower(): value for key, value in scope.get("headers", [])}
        configured_header = self.settings.request_id_header.lower().encode("latin-1")
        raw_request_id = headers.get(configured_header)
        request_id = (
            raw_request_id.decode("latin-1").strip()[:128]
            if raw_request_id and raw_request_id.strip()
            else str(uuid4())
        )
        scope.setdefault("state", {})["request_id"] = request_id
        started = time.perf_counter()
        status_code = 500
        record_metrics = scope.get("path") != "/metrics"
        if record_metrics:
            REQUEST_METRICS.request_started()

        async def send_with_context(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = int(message["status"])
                response_headers = list(message.get("headers", []))
                response_headers.extend(
                    [
                        (self.settings.request_id_header.encode("latin-1"), request_id.encode()),
                        (b"X-Data-Version", self.settings.data_version.encode()),
                        (b"X-Ranking-Model", self.settings.ranking_model_version.encode()),
                        (
                            b"X-Compatibility-Rule-Version",
                            self.settings.compatibility_rule_version.encode(),
                        ),
                        (b"X-Solver-Version", self.settings.solver_version.encode()),
                    ]
                )
                message["headers"] = response_headers
            await send(message)

        try:
            await self.app(scope, receive, send_with_context)
        finally:
            duration_seconds = time.perf_counter() - started
            duration_ms = round(duration_seconds * 1000, 3)
            route_object = scope.get("route")
            state = scope.get("state")
            route_hint = (
                state.get(_ROUTE_TEMPLATE_HINT_STATE_KEY) if isinstance(state, dict) else None
            )
            route_template = getattr(route_object, "path", None) or route_hint or "unmatched"
            if record_metrics:
                REQUEST_METRICS.request_finished(
                    method=str(scope.get("method", "UNKNOWN")),
                    route=str(route_template),
                    status_code=status_code,
                    duration_seconds=duration_seconds,
                )
            self.logger.info(
                "request_completed",
                extra={
                    "request_id": request_id,
                    "method": scope.get("method"),
                    "path": scope.get("path"),
                    "status_code": status_code,
                    "duration_ms": duration_ms,
                    "data_version": self.settings.data_version,
                    "ranking_model": self.settings.ranking_model_version,
                    "rule_version": self.settings.compatibility_rule_version,
                    "solver_version": self.settings.solver_version,
                },
            )
