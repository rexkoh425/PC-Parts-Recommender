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
PAYLOAD_TOO_LARGE_ERROR: dict[int | str, dict[str, Any]] = {
    413: {
        "model": ErrorResponse,
        "description": "The request body exceeds the configured API byte limit.",
    }
}
GENERATION_RATE_LIMIT_ERROR: dict[int | str, dict[str, Any]] = {
    429: {
        "model": ErrorResponse,
        "description": "The bounded build-generation wait queue is full; retry later.",
    }
}
SERVICE_ERROR: dict[int | str, dict[str, Any]] = {
    503: {
        "model": ErrorResponse,
        "description": (
            "Required catalogue data, build-generation capacity, or a conclusive optimizer "
            "result is unavailable."
        ),
    }
}
