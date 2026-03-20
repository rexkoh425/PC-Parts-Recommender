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


def _promotion_decision(
    *,
    config: PerformanceModelConfig,
    data_use: DataUseDeclaration,
    dataset_evidence: DatasetEvidence,
    evaluations: dict[str, ModelEvaluation],
    calibration: PredictionIntervalCalibration,
    grouped_test: GroupedTestDiagnostics,
    uncertainty: RegressionUncertainty,
) -> tuple[bool, tuple[str, ...], Literal["high", "medium", "low"]]:
    metrics = evaluations["lightgbm"].test
    blockers: list[str] = []
    if not data_use.eligible_for_reported_metrics:
        blockers.append(data_use.reporting_block_reason or "training provenance is not reportable")
    if metrics.sample_count < config.min_confident_test_rows:
        blockers.append(
            f"test sample count {metrics.sample_count} is below {config.min_confident_test_rows}"
        )
    if metrics.r2 < config.min_confident_r2:
        blockers.append(f"test R2 {metrics.r2:.4f} is below {config.min_confident_r2:.4f}")
    if metrics.mape_percent > config.max_confident_mape_percent:
        blockers.append(
            "test MAPE "
            f"{metrics.mape_percent:.4f}% exceeds {config.max_confident_mape_percent:.4f}%"
        )
    blockers.extend(dataset_evidence.block_reasons)
    if grouped_test.test_group_count < config.min_confident_test_groups:
        blockers.append(
            f"test group count {grouped_test.test_group_count} is below "
            f"{config.min_confident_test_groups}"
        )
    if grouped_test.development_group_overlap_count:
        blockers.append(
            "test leakage units overlap development splits: "
            f"{grouped_test.development_group_overlap_count}"
        )
    if grouped_test.outside_training_envelope_fraction > config.max_test_ood_fraction:
        blockers.append(
            "test feature-envelope OOD fraction "
            f"{grouped_test.outside_training_envelope_fraction:.4f} exceeds "
            f"{config.max_test_ood_fraction:.4f}"
        )
    if calibration.calibration_sample_count < config.min_calibration_rows:
        blockers.append(
            f"calibration sample count {calibration.calibration_sample_count} is below "
            f"{config.min_calibration_rows}"
        )
    if calibration.calibration_group_count < config.min_calibration_groups:
        blockers.append(
            f"calibration group count {calibration.calibration_group_count} is below "
            f"{config.min_calibration_groups}"
        )
    minimum_coverage = calibration.nominal_coverage - config.max_interval_coverage_shortfall
    if calibration.test_coverage < minimum_coverage:
        blockers.append(
            f"test prediction-interval coverage {calibration.test_coverage:.4f} is below "
            f"{minimum_coverage:.4f}"
        )
    if uncertainty.r2_lower < config.min_confident_r2:
        blockers.append(
            f"grouped-bootstrap R2 lower bound {uncertainty.r2_lower:.4f} is below "
            f"{config.min_confident_r2:.4f}"
        )
    if uncertainty.mape_percent_upper > config.max_confident_mape_percent:
        blockers.append(
            "grouped-bootstrap MAPE upper bound "
            f"{uncertainty.mape_percent_upper:.4f}% exceeds "
            f"{config.max_confident_mape_percent:.4f}%"
        )
    if config.require_baseline_improvement:
        required_fraction = 1.0 - config.minimum_baseline_mape_improvement_percent / 100.0
        for split_name in ("validation", "test"):
            lightgbm_mape = getattr(evaluations["lightgbm"], split_name).mape_percent
            best_baseline_mape = min(
                getattr(evaluations[name], split_name).mape_percent
                for name in ("train_median", "ridge")
            )
            if lightgbm_mape >= best_baseline_mape * required_fraction:
                blockers.append(
                    f"LightGBM {split_name} MAPE {lightgbm_mape:.4f}% does not improve "
                    f"on the best baseline {best_baseline_mape:.4f}% by the required "
                    f"{config.minimum_baseline_mape_improvement_percent:.2f}%"
                )
    promotable = not blockers
    confidence: Literal["high", "medium", "low"]
    if promotable:
        confidence = "high"
    elif (
        data_use.eligible_for_reported_metrics
        and metrics.sample_count >= config.min_confident_test_rows
        and (
            metrics.r2 >= min(0.70, config.min_confident_r2)
            or metrics.mape_percent <= max(20.0, config.max_confident_mape_percent)
        )
    ):
        confidence = "medium"
    else:
        confidence = "low"
    return promotable, tuple(dict.fromkeys(blockers)), confidence


def _row_level_dataset_evidence(
    frame: pd.DataFrame,
    evidence: DatasetEvidence,
) -> DatasetEvidence:
    blockers = list(evidence.block_reasons)
    column = "eligible_for_external_claims"
    if column not in frame:
        blockers.append("row-level external-claim eligibility was not declared")
    elif not pd.api.types.is_bool_dtype(frame[column].dtype):
        blockers.append("row-level external-claim eligibility was not explicitly boolean")
    elif not bool(frame[column].all()):
        blockers.append("one or more training rows are ineligible for external claims")
    if not blockers:
        return evidence
    return DatasetEvidence(
        verified=evidence.verified,
        eligible_for_promotion=False,
        manifest_sha256=evidence.manifest_sha256,
        block_reasons=tuple(dict.fromkeys(blockers)),
    )


def _model_version(
    *,
    booster: lgb.Booster,
    config: PerformanceModelConfig,
    training_data_sha256: str,
    best_iteration: int,
) -> str:
    payload = json.dumps(
        {
            "config": config.to_dict(),
            "training_data_sha256": training_data_sha256,
        },
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    digest = hashlib.sha256(payload)
    digest.update(booster.model_to_string(num_iteration=best_iteration).encode("utf-8"))
    return digest.hexdigest()


def train_performance_model(
    frame: pd.DataFrame,
    config: PerformanceModelConfig,
    *,
    dataset_evidence: DatasetEvidence | None = None,
) -> PerformanceTrainingResult:
    """Train baselines and a LightGBM regressor on deterministic grouped splits."""

    prepared = split_performance_frame(frame, config)
    peak_memory_mb = estimate_peak_training_memory_mb(prepared, config)
    if peak_memory_mb > config.max_training_memory_mb:
        raise MemoryError(
            f"estimated peak training memory {peak_memory_mb:.2f} MiB exceeds configured "
            f"budget {config.max_training_memory_mb} MiB"
        )
    evidence = _row_level_dataset_evidence(
        prepared,
        dataset_evidence or DatasetEvidence.unverified(),
    )
    features = list(config.feature_columns)
    split_frames = {
        name: prepared.loc[prepared[config.split_column] == name].copy()
        for name in ("train", "validation", "calibration", "test")
    }
    train = split_frames["train"]
    validation = split_frames["validation"]
    calibration_frame = split_frames["calibration"]
    test = split_frames["test"]
    x_train = train.loc[:, features]
    x_validation = validation.loc[:, features]
    x_calibration = calibration_frame.loc[:, features]
    x_test = test.loc[:, features]
    y_train = train[config.target_column].to_numpy(dtype=float)
    y_validation = validation[config.target_column].to_numpy(dtype=float)
    y_calibration = calibration_frame[config.target_column].to_numpy(dtype=float)
    y_test = test[config.target_column].to_numpy(dtype=float)
    learner_y_train = transform_targets(y_train, transform=config.target_transform)
    learner_y_validation = transform_targets(y_validation, transform=config.target_transform)

    median_value = float(np.median(y_train))
    median_evaluation = _evaluate_candidate(
        name="train_median",
        predict=lambda candidate: np.full(len(candidate), median_value, dtype=float),
        validation_features=x_validation,
        validation_target=y_validation,
        test_features=x_test,
        test_target=y_test,
    )

    ridge = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        Ridge(alpha=config.ridge_alpha),
    )
    ridge.fit(x_train, y_train)
    ridge_evaluation = _evaluate_candidate(
        name="ridge",
        predict=lambda candidate: np.asarray(ridge.predict(candidate), dtype=float),
        validation_features=x_validation,
        validation_target=y_validation,
        test_features=x_test,
        test_target=y_test,
    )

    estimator, requested_device, actual_device, fallback_reason = _fit_lightgbm(
        config=config,
        train_features=x_train,
        train_target=learner_y_train,
        validation_features=x_validation,
        validation_target=learner_y_validation,
    )
    best_iteration = max(1, int(estimator.best_iteration_ or estimator.n_estimators))

    def native_predictions(candidate: pd.DataFrame) -> np.ndarray:
        return inverse_target_predictions(
            np.asarray(
                estimator.predict(candidate, num_iteration=best_iteration),
                dtype=float,
            ),
            transform=config.target_transform,
        )

    lightgbm_evaluation = _evaluate_candidate(
        name="lightgbm",
        predict=native_predictions,
        validation_features=x_validation,
        validation_target=y_validation,
        test_features=x_test,
        test_target=y_test,
    )
    evaluations = {
        "train_median": median_evaluation,
        "ridge": ridge_evaluation,
        "lightgbm": lightgbm_evaluation,
    }

    feature_profiles: dict[str, FeatureProfile] = {}
    for feature in config.feature_columns:
        values = train[feature].to_numpy(dtype=float)
        finite_values = values[np.isfinite(values)]
        if finite_values.size == 0:
            raise ValueError(f"training split feature {feature!r} has no finite values")
        feature_profiles[feature] = FeatureProfile(
            minimum=float(np.min(finite_values)),
            maximum=float(np.max(finite_values)),
            missing_fraction=float(np.isnan(values).mean()),
        )

    calibration_predictions = native_predictions(x_calibration)
    test_predictions = native_predictions(x_test)
    calibration = calibrate_prediction_intervals(
        y_calibration,
        calibration_predictions,
        calibration_frame[config.family_column].astype(str).tolist(),
        y_test,
        test_predictions,
        alpha=config.prediction_interval_alpha,
    )
    uncertainty = grouped_bootstrap_uncertainty(
        y_test,
        test_predictions,
        test[config.family_column].astype(str).tolist(),
        confidence_level=config.bootstrap_confidence_level,
        n_resamples=config.bootstrap_resamples,
        seed=config.split_seed,
    )
    development_groups = set(
        prepared.loc[
            prepared[config.split_column].isin(("train", "validation", "calibration")),
            config.family_column,
        ].astype(str)
    )
    grouped_test = grouped_test_diagnostics(
        test_frame=test,
        observed=y_test,
        predicted=test_predictions,
        group_column=config.family_column,
        development_groups=development_groups,
        feature_columns=config.feature_columns,
        feature_profiles=feature_profiles,
    )

    synthetic_flags = prepared[config.synthetic_column].astype(bool).tolist()
    data_use = DataUseDeclaration.from_flags(synthetic_flags, include_synthetic=True)
    promotable, blockers, confidence = _promotion_decision(
        config=config,
        data_use=data_use,
        dataset_evidence=evidence,
        evaluations=evaluations,
        calibration=calibration,
        grouped_test=grouped_test,
        uncertainty=uncertainty,
    )
    training_data_sha256 = performance_frame_sha256(prepared, config)
    model_version = _model_version(
        booster=estimator.booster_,
        config=config,
        training_data_sha256=training_data_sha256,
        best_iteration=best_iteration,
    )
    group_columns = [config.family_column, config.generation_column]
    split_group_counts = {
        name: int(split_frame.loc[:, group_columns].drop_duplicates().shape[0])
        for name, split_frame in split_frames.items()
    }
    split_row_counts = {name: int(len(split_frame)) for name, split_frame in split_frames.items()}
    split_group_hashes = {
        name: tuple(
            sorted(
                hashlib.sha256(str(value).encode("utf-8")).hexdigest()
                for value in split_frame[config.family_column].astype(str).unique()
            )
        )
        for name, split_frame in split_frames.items()
    }
    development_group_hashes = tuple(
        sorted({value for hashes in split_group_hashes.values() for value in hashes})
    )
    artifact = PerformanceModelArtifact(
        config=config,
        booster=estimator.booster_,
        evaluations=evaluations,
        data_use=data_use,
        training_data_sha256=training_data_sha256,
        model_version=model_version,
        split_group_counts=split_group_counts,
        split_row_counts=split_row_counts,
        split_group_hashes=split_group_hashes,
        development_group_hashes=development_group_hashes,
        feature_profiles=feature_profiles,
        dataset_evidence=evidence,
        calibration=calibration,
        grouped_test=grouped_test,
        test_uncertainty=uncertainty,
        estimated_peak_training_memory_mb=peak_memory_mb,
        allowed_missing_fraction=config.max_prediction_missing_fraction,
        best_iteration=best_iteration,
        confidence_level=confidence,
        precise_predictions_enabled=promotable,
        promotable=promotable,
        promotion_block_reasons=blockers,
        requested_device=requested_device,
        actual_device=actual_device,
        device_fallback_reason=fallback_reason,
    )
    assignments = dict(
        zip(
            prepared[config.product_id_column].astype(str),
            prepared[config.split_column].astype(str),
            strict=True,
        )
    )
    return PerformanceTrainingResult(artifact=artifact, split_assignments=assignments)
