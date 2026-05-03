"""Group-aware uncertainty, calibration, and OOD diagnostics for regressors."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from pc_build_recommender.evaluation.metrics import evaluate_regression

from .contracts import (
    FeatureProfile,
    GroupedTestDiagnostics,
    PredictionIntervalCalibration,
    RegressionUncertainty,
)


def _finite_vectors(
    observed: Sequence[float] | np.ndarray,
    predicted: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    actual = np.asarray(observed, dtype=float).reshape(-1)
    estimate = np.asarray(predicted, dtype=float).reshape(-1)
    if actual.shape != estimate.shape:
        raise ValueError("observed and predicted arrays must have the same shape")
    if actual.size < 2:
        raise ValueError("at least two observations are required")
    if not (np.isfinite(actual).all() and np.isfinite(estimate).all()):
        raise ValueError("observed and predicted values must be finite")
    return actual, estimate


def _wilson_lower_bound(covered: int, total: int, *, z_score: float = 1.959963984540054) -> float:
    if total < 1 or not 0 <= covered <= total:
        raise ValueError("coverage counts are invalid")
    probability = covered / total
    denominator = 1.0 + z_score**2 / total
    centre = probability + z_score**2 / (2.0 * total)
    adjustment = z_score * math.sqrt(
        probability * (1.0 - probability) / total + z_score**2 / (4.0 * total**2)
    )
    return max(0.0, (centre - adjustment) / denominator)


def calibrate_prediction_intervals(
    calibration_observed: Sequence[float] | np.ndarray,
    calibration_predicted: Sequence[float] | np.ndarray,
    calibration_groups: Sequence[str],
    test_observed: Sequence[float] | np.ndarray,
    test_predicted: Sequence[float] | np.ndarray,
    *,
    alpha: float,
) -> PredictionIntervalCalibration:
    """Fit an independent split-conformal absolute-residual interval."""

    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between zero and one")
    calibration_actual, calibration_estimate = _finite_vectors(
        calibration_observed, calibration_predicted
    )
    test_actual, test_estimate = _finite_vectors(test_observed, test_predicted)
    if len(calibration_groups) != calibration_actual.size:
        raise ValueError("calibration_groups must contain one value per calibration row")
    group_count = len(set(calibration_groups))
    if group_count < 1:
        raise ValueError("calibration requires at least one group")

    residuals = np.abs(calibration_actual - calibration_estimate)
    quantile_level = min(1.0, math.ceil((len(residuals) + 1) * (1.0 - alpha)) / len(residuals))
    residual_quantile = float(np.quantile(residuals, quantile_level, method="higher"))
    lower = np.maximum(0.0, test_estimate - residual_quantile)
    upper = np.maximum(0.0, test_estimate + residual_quantile)
    covered = int(np.count_nonzero((test_actual >= lower) & (test_actual <= upper)))
    return PredictionIntervalCalibration(
        method="split_conformal_absolute_residual",
        alpha=alpha,
        absolute_error_quantile=residual_quantile,
        calibration_sample_count=int(calibration_actual.size),
        calibration_group_count=group_count,
        test_sample_count=int(test_actual.size),
        test_covered_count=covered,
        test_coverage_lower_95=_wilson_lower_bound(covered, int(test_actual.size)),
    )


def grouped_bootstrap_uncertainty(
    observed: Sequence[float] | np.ndarray,
    predicted: Sequence[float] | np.ndarray,
    groups: Sequence[str],
    *,
    confidence_level: float,
    n_resamples: int,
    seed: int,
) -> RegressionUncertainty:
    """Return family-cluster bootstrap intervals for untouched-test metrics."""

    actual, estimate = _finite_vectors(observed, predicted)
    if len(groups) != actual.size:
        raise ValueError("groups must contain one value per test row")
    group_count = len(set(groups))
    if group_count < 2:
        raise ValueError("grouped bootstrap requires at least two distinct groups")
    evaluation = evaluate_regression(
        actual.tolist(),
        estimate.tolist(),
        is_synthetic=[False] * len(actual),
        groups=list(groups),
        confidence_level=confidence_level,
        n_resamples=n_resamples,
        seed=seed,
    )
    r2 = evaluation.metric("regression.r_squared")
    mae = evaluation.metric("regression.mae")
    mape = evaluation.metric("regression.mape")
    interval_metrics = (r2, mae, mape)
    if any(metric.ci_lower is None or metric.ci_upper is None for metric in interval_metrics):
        raise RuntimeError("grouped bootstrap did not return confidence intervals")
    return RegressionUncertainty(
        confidence_level=confidence_level,
        resamples=n_resamples,
        group_count=group_count,
        r2_lower=float(r2.ci_lower),  # type: ignore[arg-type]
        r2_upper=float(r2.ci_upper),  # type: ignore[arg-type]
        mae_lower=float(mae.ci_lower),  # type: ignore[arg-type]
        mae_upper=float(mae.ci_upper),  # type: ignore[arg-type]
        mape_percent_lower=float(mape.ci_lower) * 100.0,  # type: ignore[arg-type]
        mape_percent_upper=float(mape.ci_upper) * 100.0,  # type: ignore[arg-type]
    )


def grouped_test_diagnostics(
    *,
    test_frame: pd.DataFrame,
    observed: Sequence[float] | np.ndarray,
    predicted: Sequence[float] | np.ndarray,
    group_column: str,
    development_groups: set[str],
    feature_columns: Sequence[str],
    feature_profiles: dict[str, FeatureProfile],
) -> GroupedTestDiagnostics:
    """Summarise family OOD, worst evaluable family, and feature-envelope drift."""

    actual, estimate = _finite_vectors(observed, predicted)
    if len(test_frame) != actual.size:
        raise ValueError("test_frame must contain one row per observed value")
    if group_column not in test_frame:
        raise ValueError(f"test frame is missing group column {group_column!r}")
    groups = test_frame[group_column].astype(str)
    unique_groups = set(groups)
    overlap_count = len(unique_groups & development_groups)

    group_mapes: list[float] = []
    for group in sorted(unique_groups):
        indices = np.flatnonzero(groups.to_numpy() == group)
        if len(indices) < 2:
            continue
        group_actual = actual[indices]
        group_estimate = estimate[indices]
        if (group_actual <= 0).any():
            continue
        group_mapes.append(float(np.mean(np.abs((group_actual - group_estimate) / group_actual))))

    outside = np.zeros(len(test_frame), dtype=bool)
    for feature in feature_columns:
        profile = feature_profiles[feature]
        values = test_frame[feature].to_numpy(dtype=float)
        finite = np.isfinite(values)
        outside |= finite & ((values < profile.minimum) | (values > profile.maximum))
    return GroupedTestDiagnostics(
        group_column=group_column,
        test_group_count=len(unique_groups),
        test_row_count=len(test_frame),
        development_group_overlap_count=overlap_count,
        evaluable_group_count=len(group_mapes),
        worst_group_mape_percent=(max(group_mapes) * 100.0 if group_mapes else None),
        outside_training_envelope_row_count=int(outside.sum()),
    )
