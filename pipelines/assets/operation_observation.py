"""Shared, optional operation receipt instrumentation for Dagster user-code."""

from __future__ import annotations

import os
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import ParamSpec, TypeVar

from pc_build_recommender.pipeline_operations import record_pipeline_operation

_Parameters = ParamSpec("_Parameters")
_Result = TypeVar("_Result")


def instrument_pipeline_operation(
    operation_name: str,
) -> Callable[[Callable[_Parameters, _Result]], Callable[_Parameters, _Result]]:
    """Record success/failure only when ``PIPELINE_OPERATIONS_DIR`` is configured."""

    def decorate(function: Callable[_Parameters, _Result]) -> Callable[_Parameters, _Result]:
        @wraps(function)
        def observed(*args: _Parameters.args, **kwargs: _Parameters.kwargs) -> _Result:
            root_value = os.getenv("PIPELINE_OPERATIONS_DIR")
            root = Path(root_value) if root_value else None
            with record_pipeline_operation(root, operation_name):
                return function(*args, **kwargs)

        return observed

    return decorate


__all__ = ["instrument_pipeline_operation"]
