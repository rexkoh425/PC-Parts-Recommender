"""Reusable OpenAPI descriptions for the service's structured error envelope."""

from __future__ import annotations

from typing import Any

from services.api.models import ErrorResponse

VALIDATION_ERROR: dict[int | str, dict[str, Any]] = {
    422: {"model": ErrorResponse, "description": "Request contract or domain validation failed."}
}
NOT_FOUND_ERROR: dict[int | str, dict[str, Any]] = {
    404: {"model": ErrorResponse, "description": "The requested resource was not found."}
}
CONFLICT_ERROR: dict[int | str, dict[str, Any]] = {
    409: {"model": ErrorResponse, "description": "The requested change conflicts with hard rules."}
}
UNAUTHORIZED_ERROR: dict[int | str, dict[str, Any]] = {
    401: {
        "model": ErrorResponse,
        "description": "The supplied server-issued capability is invalid or expired.",
    }
}
PAYLOAD_TOO_LARGE_ERROR: dict[int | str, dict[str, Any]] = {
    413: {
        "model": ErrorResponse,
        "description": "The request body exceeds the configured API byte limit.",
    }
}
OPTIMIZER_RATE_LIMIT_ERROR: dict[int | str, dict[str, Any]] = {
    429: {
        "model": ErrorResponse,
        "description": "The shared bounded optimizer wait queue is full; retry later.",
    }
}
GENERATION_RATE_LIMIT_ERROR = OPTIMIZER_RATE_LIMIT_ERROR
SERVICE_ERROR: dict[int | str, dict[str, Any]] = {
    503: {
        "model": ErrorResponse,
        "description": (
            "Required catalogue data, optimizer admission capacity, or a conclusive optimizer "
            "result is unavailable."
        ),
    }
}
