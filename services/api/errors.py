"""Stable API error envelopes and exception handlers."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError, ResponseValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger("pc_build_recommender.api.errors")


class ApiError(Exception):
    """An expected application error suitable for returning to an API caller."""

    def __init__(
        self,
        *,
        status_code: int,
        code: str,
        message: str,
        details: Mapping[str, Any] | Sequence[Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details = details


def _request_id(request: Request) -> str | None:
    value = getattr(request.state, "request_id", None)
    return str(value) if value is not None else None


def error_payload(
    *,
    code: str,
    message: str,
    request_id: str | None,
    details: Mapping[str, Any] | Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Return one stable envelope while retaining a top-level web-client message."""

    error: dict[str, Any] = {
        "code": code,
        "message": message,
        "request_id": request_id,
    }
    if details is not None:
        error["details"] = jsonable_encoder(details)
    return {"message": message, "error": error}


def install_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def handle_api_error(request: Request, exc: ApiError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=error_payload(
                code=exc.code,
                message=exc.message,
                request_id=_request_id(request),
                details=exc.details,
            ),
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content=error_payload(
                code="validation_error",
                message="The request did not satisfy the API contract.",
                request_id=_request_id(request),
                details=exc.errors(),
            ),
        )

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        message = str(exc.detail)
        return JSONResponse(
            status_code=exc.status_code,
            content=error_payload(
                code="not_found" if exc.status_code == 404 else "http_error",
                message=message,
                request_id=_request_id(request),
            ),
            headers=exc.headers,
        )

    @app.exception_handler(ResponseValidationError)
    async def handle_response_validation_error(
        request: Request, exc: ResponseValidationError
    ) -> JSONResponse:
        logger.error(
            "API response violated its declared contract",
            extra={
                "request_id": _request_id(request),
                "error_type": type(exc).__name__,
            },
        )
        return JSONResponse(
            status_code=500,
            content=error_payload(
                code="response_contract_error",
                message="The service produced an invalid response and withheld it.",
                request_id=_request_id(request),
            ),
        )

    @app.exception_handler(Exception)
    async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        logger.exception(
            "Unhandled API error",
            extra={"request_id": _request_id(request), "error_type": type(exc).__name__},
        )
        return JSONResponse(
            status_code=500,
            content=error_payload(
                code="internal_error",
                message="The recommendation service could not complete the request.",
                request_id=_request_id(request),
            ),
        )
