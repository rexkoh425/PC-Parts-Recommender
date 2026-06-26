"""Explicit service-response validation before HTTP side effects occur."""

from __future__ import annotations
from __future__ import annotations

from fastapi.exceptions import ResponseValidationError
from pydantic import BaseModel, ValidationError


def validate_service_response[ResponseModelT: BaseModel](
    value: object, response_model: type[ResponseModelT]
) -> ResponseModelT:
    """Validate an application result before a router records metrics from it.

    FastAPI normally validates a declared ``response_model`` only after the route
    function returns. Routers that emit metrics therefore need this explicit
    boundary first: an invalid service value must become the standard
    ``ResponseValidationError`` instead of causing an attribute error while
    metrics are being recorded.
    """

    try:
        return response_model.model_validate(value)
    except ValidationError as exc:
        raise ResponseValidationError(errors=exc.errors()) from exc
