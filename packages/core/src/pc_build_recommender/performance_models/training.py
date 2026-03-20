"""Leakage-safe baseline and LightGBM training for workload performance."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from typing import Any, Literal

import lightgbm as lgb
import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from lightgbm.basic import LightGBMError
from sklearn.impute import SimpleImputer  # type: ignore[import-untyped]
from sklearn.linear_model import Ridge  # type: ignore[import-untyped]
from sklearn.metrics import mean_absolute_error, r2_score  # type: ignore[import-untyped]
from sklearn.pipeline import make_pipeline  # type: ignore[import-untyped]
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

from pc_build_recommender.evaluation.contracts import DataUseDeclaration

from .contracts import (
    DatasetEvidence,
    FeatureProfile,
    GroupedTestDiagnostics,
    ModelEvaluation,
    PerformanceModelArtifact,
    PerformanceModelConfig,
    PerformanceTrainingResult,
    PredictionIntervalCalibration,
    RegressionMetrics,
    RegressionUncertainty,
)
from .data import (
    estimate_peak_training_memory_mb,
    performance_frame_sha256,
    split_performance_frame,
)
from .evaluation import (
    calibrate_prediction_intervals,
    grouped_bootstrap_uncertainty,
    grouped_test_diagnostics,
)
from .target_transforms import inverse_target_predictions, transform_targets


def calculate_regression_metrics(
    observed: np.ndarray,
    predicted: np.ndarray,
) -> RegressionMetrics:
    """Calculate R-squared, MAE, and finite percentage-point MAPE."""

    actual = np.asarray(observed, dtype=float).reshape(-1)
    estimate = np.asarray(predicted, dtype=float).reshape(-1)
    if actual.shape != estimate.shape:
        raise ValueError("observed and predicted arrays must have the same shape")
    if actual.size < 2:
        raise ValueError("at least two observations are required")
    if not (np.isfinite(actual).all() and np.isfinite(estimate).all()):
        raise ValueError("metric inputs must be finite")
    if (actual <= 0).any():
        raise ValueError("observed targets must be positive for MAPE")
    mape_percent = float(np.mean(np.abs((actual - estimate) / actual)) * 100.0)
    return RegressionMetrics(
        r2=float(r2_score(actual, estimate)),
        mae=float(mean_absolute_error(actual, estimate)),
        mape_percent=mape_percent,
        sample_count=int(actual.size),
    )


def _evaluate_candidate(
    *,
    name: str,
    predict: Callable[[pd.DataFrame], np.ndarray],
    validation_features: pd.DataFrame,
    validation_target: np.ndarray,
    test_features: pd.DataFrame,
    test_target: np.ndarray,
) -> ModelEvaluation:
    return ModelEvaluation(
        model_name=name,
        validation=calculate_regression_metrics(
            validation_target,
            predict(validation_features),
        ),
        test=calculate_regression_metrics(test_target, predict(test_features)),
    )


def _base_lightgbm_parameters(config: PerformanceModelConfig, *, device: str) -> dict[str, Any]:
    parameters: dict[str, Any] = {
        "objective": "regression",
        "n_estimators": 700,
        "learning_rate": 0.025,
        "num_leaves": 24,
        "min_child_samples": 5,
        "max_depth": -1,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "reg_alpha": 0.05,
        "reg_lambda": 0.2,
        "random_state": config.split_seed,
        "deterministic": True,
        "force_col_wise": True,
        "n_jobs": config.max_cpu_threads,
        "verbosity": -1,
        "device_type": device,
    }
    overrides = dict(config.lightgbm_params)
    overrides.pop("device_type", None)
    managed_parameters = {
        "random_state",
        "deterministic",
        "force_col_wise",
        "n_jobs",
        "data_random_seed",
        "feature_fraction_seed",
        "bagging_seed",
        "drop_seed",
        "extra_seed",
    }
    managed_overrides = sorted(managed_parameters.intersection(overrides))
    if managed_overrides:
        raise ValueError(
            "LightGBM reproducibility/resource parameters are managed by the trainer: "
            f"{managed_overrides}"
        )
    parameters.update(overrides)
    # Device selection owns these settings so a generic parameter bundle cannot
    # silently report one device while training on another.
    parameters["device_type"] = device
    parameters.update(
        {
            "random_state": config.split_seed,
            "data_random_seed": config.split_seed,
            "feature_fraction_seed": config.split_seed,
            "bagging_seed": config.split_seed,
            "drop_seed": config.split_seed,
            "extra_seed": config.split_seed,
            "deterministic": True,
            "force_col_wise": True,
            "n_jobs": config.max_cpu_threads,
        }
    )
    if device in {"gpu", "cuda"}:
        requested_max_bin = int(parameters.get("max_bin", config.gpu_max_bin))
        parameters["max_bin"] = min(requested_max_bin, config.gpu_max_bin)
    return parameters


def _requested_device(config: PerformanceModelConfig) -> str:
    parameter_device = config.lightgbm_params.get("device_type")
    return str(parameter_device) if parameter_device is not None else config.requested_device


def _device_attempts(config: PerformanceModelConfig) -> tuple[str, list[str]]:
    requested = _requested_device(config)
    if requested == "auto":
        return requested, ["cuda", "gpu", "cpu"]
    if requested == "cpu" or not config.allow_device_fallback:
        return requested, [requested]
    return requested, [requested, "cpu"]


def _fit_lightgbm(
    *,
    config: PerformanceModelConfig,
    train_features: pd.DataFrame,
    train_target: np.ndarray,
    validation_features: pd.DataFrame,
    validation_target: np.ndarray,
) -> tuple[lgb.LGBMRegressor, str, str, str | None]:
    requested, attempts = _device_attempts(config)
    failures: list[str] = []
    for device in attempts:
        estimator = lgb.LGBMRegressor(**_base_lightgbm_parameters(config, device=device))
        try:
            estimator.fit(
                train_features,
                train_target,
                eval_X=validation_features,
                eval_y=validation_target,
                eval_metric="l1",
                callbacks=[
                    lgb.early_stopping(stopping_rounds=50, first_metric_only=True, verbose=False),
                    lgb.log_evaluation(period=0),
                ],
            )
        except LightGBMError as exc:
            failures.append(f"{device}: {str(exc).strip()}")
            continue
        fallback_reason = "; ".join(failures) or None
        return estimator, requested, device, fallback_reason
    joined_failures = "; ".join(failures)
    raise RuntimeError(
        f"LightGBM could not train on requested device {requested!r}: {joined_failures}"
    )

# TODO: rest of this module still to come.
