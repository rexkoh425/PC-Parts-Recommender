"""Request correlation, version headers, and compact structured logging."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import re
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal
from uuid import uuid4

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from services.api.errors import error_payload
from services.api.metrics import REQUEST_METRICS, RequestMetrics
from services.api.settings import ApiSettings

_ROUTE_TEMPLATE_HINT_STATE_KEY = "_pcbr_route_template"


OptimizerOperation = Literal["generate", "replace"]

_OPTIMIZER_OPERATION_ORDER: tuple[OptimizerOperation, ...] = ("generate", "replace")
_OPTIMIZER_OPERATIONS = frozenset(_OPTIMIZER_OPERATION_ORDER)


class OptimizerQueueFullError(RuntimeError):
    """Raised when an optimizer request cannot enter the bounded wait queue."""


class OptimizerQueueTimeoutError(RuntimeError):
    """Raised when a queued optimizer request does not obtain capacity in time."""


class OptimizerAdmissionController:
    """Bound optimizer execution and waiting callers across every optimizer route.

    Build generation and component replacement share one capacity pool because both can
    invoke CP-SAT.  A shared controller prevents one route from bypassing the process-level
    concurrency and queue limits applied to the other.
    """

    def __init__(
        self,
        *,
        max_concurrency: int,
        max_queue_size: int,
        queue_timeout_seconds: float,
        metrics: RequestMetrics | None = None,
    ) -> None:
        self.max_concurrency = max_concurrency
        self.max_queue_size = max_queue_size
        self.queue_timeout_seconds = queue_timeout_seconds
        self.metrics = metrics
        self._condition = asyncio.Condition()
        self._active = 0
        self._queued = 0
        self._active_by_operation: dict[OptimizerOperation, int] = {
            "generate": 0,
            "replace": 0,
        }
        self._queued_by_operation: dict[OptimizerOperation, int] = {
            "generate": 0,
            "replace": 0,
        }

    @staticmethod
    def _operation(value: str) -> OptimizerOperation:
        if value not in _OPTIMIZER_OPERATIONS:
            raise ValueError(f"unsupported optimizer admission operation: {value!r}")
        return value

    def _record_transition(
        self,
        *,
        operation: OptimizerOperation,
        active_delta: int = 0,
        queued_delta: int = 0,
        outcome: str | None = None,
        wait_seconds: float | None = None,
    ) -> None:
        if self.metrics is None:
            return
        self.metrics.record_optimizer_admission_transition(
            operation=operation,
            active_delta=active_delta,
            queued_delta=queued_delta,
            outcome=outcome,
            wait_seconds=wait_seconds,
        )

    async def acquire(self, operation: str = "generate") -> None:
        """Reserve one execution slot or raise a bounded-admission error."""

        admitted_operation = self._operation(operation)
        started = time.perf_counter()
        async with self._condition:
            if self._active < self.max_concurrency:
                self._active += 1
                self._active_by_operation[admitted_operation] += 1
                self._record_transition(
                    operation=admitted_operation,
                    active_delta=1,
                    outcome="admitted",
                    wait_seconds=time.perf_counter() - started,
                )
                return
            if self._queued >= self.max_queue_size:
                self._record_transition(
                    operation=admitted_operation,
                    outcome="queue_full",
                    wait_seconds=time.perf_counter() - started,
                )
                raise OptimizerQueueFullError

            self._queued += 1
            self._queued_by_operation[admitted_operation] += 1
            self._record_transition(operation=admitted_operation, queued_delta=1)
            outcome: str | None = None
            active_delta = 0
            try:
                async with asyncio.timeout(self.queue_timeout_seconds):
                    while self._active >= self.max_concurrency:
                        await self._condition.wait()
                    self._active += 1
                    self._active_by_operation[admitted_operation] += 1
                    active_delta = 1
                    outcome = "admitted"
            except TimeoutError as error:
                outcome = "queue_timeout"
                raise OptimizerQueueTimeoutError from error
            except asyncio.CancelledError:
                outcome = "cancelled"
                raise
            finally:
                self._queued -= 1
                self._queued_by_operation[admitted_operation] -= 1
                self._record_transition(
                    operation=admitted_operation,
                    active_delta=active_delta,
                    queued_delta=-1,
                    outcome=outcome,
                    wait_seconds=(
                        time.perf_counter() - started if outcome is not None else None
                    ),
                )
                if active_delta == 0 and self._active < self.max_concurrency and self._queued > 0:
                    # A timeout/cancellation can race with a release that notified this waiter.
                    # Pass the now-free slot to another waiter instead of leaving it stranded.
                    self._condition.notify(1)

    async def release(self, operation: str = "generate") -> None:
        """Release one execution slot and wake one queued caller."""

        admitted_operation = self._operation(operation)
        async with self._condition:
            if self._active_by_operation[admitted_operation] <= 0:
                raise RuntimeError(
                    f"optimizer {admitted_operation} admission slot released without acquire"
                )
            self._active -= 1
            self._active_by_operation[admitted_operation] -= 1
            self._record_transition(operation=admitted_operation, active_delta=-1)
            self._condition.notify(1)

    async def snapshot(self) -> tuple[int, int]:
        """Return active and queued counts from one consistent event-loop snapshot."""

        async with self._condition:
            return self._active, self._queued

    async def operation_snapshot(self) -> dict[str, tuple[int, int]]:
        """Return bounded per-operation active and queued counts for verification."""

        async with self._condition:
            return {
                operation: (
                    self._active_by_operation[operation],
                    self._queued_by_operation[operation],
                )
                for operation in _OPTIMIZER_OPERATION_ORDER
            }


# Compatibility aliases keep existing imports working while the implementation and new
# application wiring use optimizer-wide terminology.
BuildGenerationQueueFullError = OptimizerQueueFullError
BuildGenerationQueueTimeoutError = OptimizerQueueTimeoutError
BuildGenerationAdmissionController = OptimizerAdmissionController


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


@dataclass(frozen=True, slots=True)
class _OptimizerRoute:
    operation: OptimizerOperation
    route_template: str
    display_name: str
    error_prefix: str


_GENERATION_ROUTE = _OptimizerRoute(
    operation="generate",
    route_template="/v1/builds/generate",
    display_name="Build-generation",
    error_prefix="build_generation",
)
_REPLACEMENT_ROUTE = _OptimizerRoute(
    operation="replace",
    route_template="/v1/builds/{build_id}/replace",
    display_name="Component-replacement",
    error_prefix="component_replacement",
)
_REPLACEMENT_PATH = re.compile(r"^/v1/builds/[^/]+/replace$")


def _optimizer_route(scope: Scope) -> _OptimizerRoute | None:
    if scope.get("method") != "POST":
        return None
    path = scope.get("path")
    if path == _GENERATION_ROUTE.route_template:
        return _GENERATION_ROUTE
    if isinstance(path, str) and _REPLACEMENT_PATH.fullmatch(path) is not None:
        return _REPLACEMENT_ROUTE
    return None


class OptimizerAdmissionMiddleware:
    """Apply shared bounded admission to every HTTP route that can invoke CP-SAT.

    A request that arrives after the shared wait queue is full receives ``429`` immediately.
    A queued request that cannot obtain execution capacity before the deadline receives
    ``503``. Both responses are retryable and include ``Retry-After``.
    """

    _PATH = _GENERATION_ROUTE.route_template

    def __init__(
        self,
        app: ASGIApp,
        controller: OptimizerAdmissionController,
    ) -> None:
        self.app = app
        self.controller = controller

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        route = _optimizer_route(scope)
        if route is None:
            await self.app(scope, receive, send)
            return

        retry_after = str(max(1, math.ceil(self.controller.queue_timeout_seconds)))
        _set_route_template_hint(scope, route.route_template)
        try:
            await self.controller.acquire(route.operation)
        except OptimizerQueueFullError:
            await _send_resource_error(
                scope=scope,
                receive=receive,
                send=send,
                status_code=429,
                code=f"{route.error_prefix}_queue_full",
                message=(
                    f"{route.display_name} capacity is full; retry after the shared optimizer "
                    "queue drains."
                ),
                details={
                    "operation": route.operation,
                    "max_concurrency": self.controller.max_concurrency,
                    "max_queue_size": self.controller.max_queue_size,
                    "retryable": True,
                },
                headers={"Retry-After": retry_after},
            )
            return
        except OptimizerQueueTimeoutError:
            await _send_resource_error(
                scope=scope,
                receive=receive,
                send=send,
                status_code=503,
                code=f"{route.error_prefix}_queue_timeout",
                message=(
                    f"{route.display_name} capacity did not become available before the shared "
                    "optimizer queue deadline."
                ),
                details={
                    "operation": route.operation,
                    "queue_timeout_seconds": self.controller.queue_timeout_seconds,
                    "retryable": True,
                },
                headers={"Retry-After": retry_after},
            )
            return

        try:
            await self.app(scope, receive, send)
        finally:
            await self.controller.release(route.operation)


BuildGenerationAdmissionMiddleware = OptimizerAdmissionMiddleware


class _RequestBodyTooLargeError(RuntimeError):
    """Internal signal raised before oversized body bytes reach request parsing."""


class RequestBodyLimitMiddleware:
    """Reject request bodies above a fixed limit before scarce optimizer admission.

    Optimizer routes are fully buffered up to the configured bound and then replayed to the
    inner application.  This keeps slow uploads and oversized streamed bodies from holding a
    CP-SAT execution slot while preserving streaming enforcement on all other routes.
    """

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
        optimizer_route = _optimizer_route(scope)
        if optimizer_route is not None:
            _set_route_template_hint(scope, optimizer_route.route_template)
        elif path in self._STATIC_BODY_ROUTES:
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

        if optimizer_route is not None:
            await self._buffer_optimizer_body(scope=scope, receive=receive, send=send)
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

    async def _buffer_optimizer_body(
        self,
        *,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        """Read a complete bounded optimizer body before entering admission middleware."""

        buffered: list[Message] = []
        observed_bytes = 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                # The caller is gone; do not consume optimizer capacity for abandoned work.
                return
            buffered.append(message)
            if message["type"] != "http.request":
                continue
            observed_bytes += len(message.get("body", b""))
            if observed_bytes > self.max_body_bytes:
                await self._reject(scope=scope, receive=receive, send=send)
                return
            if not message.get("more_body", False):
                break

        cursor = 0

        async def replay_receive() -> Message:
            nonlocal cursor
            if cursor < len(buffered):
                message = buffered[cursor]
                cursor += 1
                return message
            return {"type": "http.request", "body": b"", "more_body": False}

        await self.app(scope, replay_receive, send)

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
