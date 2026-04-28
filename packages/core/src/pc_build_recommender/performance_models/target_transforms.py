"""Explicit target-scale transformations for performance regressors.

Metrics, prediction intervals, and public inference always remain on the native
benchmark scale.  These helpers only change the learner's internal target
space, and the selected transform is sealed into ``PerformanceModelConfig``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .contracts import TargetTransform


def transform_targets(
    values: NDArray[np.float64] | np.ndarray,
    *,
    transform: TargetTransform,
) -> NDArray[np.float64]:
    """Convert positive native target values to the configured learner scale."""

    target = np.asarray(values, dtype=np.float64).reshape(-1)
    if not np.isfinite(target).all() or (target <= 0).any():
        raise ValueError("target values must be finite and positive")
    if transform == "identity":
        return target
    if transform == "log1p":
        return np.log1p(target)
    raise ValueError(f"unsupported target transform: {transform!r}")


def inverse_target_predictions(
    values: NDArray[np.float64] | np.ndarray,
    *,
    transform: TargetTransform,
) -> NDArray[np.float64]:
    """Convert learner outputs back to the native benchmark scale."""

    prediction = np.asarray(values, dtype=np.float64).reshape(-1)
    if not np.isfinite(prediction).all():
        raise RuntimeError("performance model returned a non-finite transformed prediction")
    if transform == "identity":
        result = prediction
    elif transform == "log1p":
        with np.errstate(over="raise", invalid="raise"):
            try:
                result = np.expm1(prediction)
            except FloatingPointError as exc:
                raise RuntimeError(
                    "performance model returned an overflowing log prediction"
                ) from exc
    else:
        raise ValueError(f"unsupported target transform: {transform!r}")
    if not np.isfinite(result).all():
        raise RuntimeError("performance model returned a non-finite native prediction")
    return np.asarray(result, dtype=np.float64)
