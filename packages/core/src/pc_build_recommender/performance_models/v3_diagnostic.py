"""Isolated CPU feature/CV experiments that cannot become production artifacts.

This module deliberately does not extend :mod:`performance_models.artifacts`.
Its output schema is diagnostic-only, so a promising adaptive experiment cannot
be loaded by the production v2 inference path or accidentally enable precise
predictions.
"""

from __future__ import annotations

import hashlib
import json
import os
import statistics
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import lightgbm as lgb
import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from pc_build_recommender.evaluation.splits import (
    assert_group_disjoint,
    deterministic_group_split,
)

from .contracts import DatasetEvidence, FeatureProfile, PerformanceModelConfig, RegressionMetrics
from .data import validate_performance_frame
from .evaluation import (
    calibrate_prediction_intervals,
    grouped_bootstrap_uncertainty,
    grouped_test_diagnostics,
)
from .training import _base_lightgbm_parameters, calculate_regression_metrics

V3_DIAGNOSTIC_SCHEMA_VERSION = "pc-build-recommender.performance-v3-diagnostic.v1"
V3_SPLIT_SEED = 20260723
V3_CV_SEED = 20260724
V3_FOLD_COUNT = 5

CPU_BASE_FEATURES: tuple[str, ...] = (
    "core_count",
    "thread_count",
    "base_clock_ghz",
    "boost_clock_ghz",
    "tdp_watts",
)
CPU_INTERACTION_FEATURES: tuple[str, ...] = (
    "core_base_clock_capacity",
    "core_boost_clock_capacity",
    "thread_core_ratio",
    "tdp_watts_per_core",
    "boost_capacity_per_watt",
)


def _group_digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _generation_feature_name(value: str) -> str:
    slug = "".join(character if character.isalnum() else "_" for character in value.casefold())
    slug = "_".join(part for part in slug.split("_") if part)[:36] or "unknown"
    return f"generation__{slug}__{_group_digest(value)[:8]}"


@dataclass(frozen=True, slots=True)
class CPUFeatureContractV3:
    """Target-independent fitted vocabulary for CPU feature construction."""

    generation_levels: tuple[str, ...]
    schema_version: str = "pc-build-recommender.cpu-feature-contract.v3"

    def __post_init__(self) -> None:
        object.__setattr__(self, "generation_levels", tuple(sorted(set(self.generation_levels))))
        if not self.generation_levels:
            raise ValueError("CPU feature contract requires at least one generation")

    @property
    def generation_columns(self) -> tuple[str, ...]:
        return tuple(_generation_feature_name(level) for level in self.generation_levels)

    @property
    def engineered_columns(self) -> tuple[str, ...]:
        return (
            *CPU_BASE_FEATURES,
            *CPU_INTERACTION_FEATURES,
            *self.generation_columns,
            "generation__unknown",
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "generation_levels": list(self.generation_levels),
            "base_features": list(CPU_BASE_FEATURES),
            "interaction_features": {
                "core_base_clock_capacity": "core_count * base_clock_ghz",
                "core_boost_clock_capacity": "core_count * boost_clock_ghz",
                "thread_core_ratio": "thread_count / core_count",
                "tdp_watts_per_core": "tdp_watts / core_count",
                "boost_capacity_per_watt": (
                    "core_count * boost_clock_ghz / tdp_watts"
                ),
            },
            "generation_columns": list(self.generation_columns),
            "unknown_generation_column": "generation__unknown",
            "target_columns_used": [],
        }


def fit_cpu_feature_contract(frame: pd.DataFrame) -> CPUFeatureContractV3:
    if "hardware_generation" not in frame:
        raise ValueError("CPU feature fitting requires hardware_generation")
    if frame["hardware_generation"].isna().any():
        raise ValueError("hardware_generation cannot be missing")
    return CPUFeatureContractV3(
        generation_levels=tuple(frame["hardware_generation"].astype(str).unique())
    )


def transform_cpu_features(
    frame: pd.DataFrame,
    contract: CPUFeatureContractV3,
    *,
    engineered: bool,
) -> pd.DataFrame:
    """Build deterministic numeric features without reading any target column."""

    required = {*CPU_BASE_FEATURES, "hardware_generation"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"CPU feature input is missing columns: {missing}")
    base = frame.loc[:, CPU_BASE_FEATURES].astype(float).copy()
    values = base.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("CPU v3 features must be finite")
    if (base[["core_count", "thread_count", "tdp_watts"]] <= 0).any().any():
        raise ValueError("CPU counts and power must be positive")
    if not engineered:
        return base

    result = base.copy()
    result["core_base_clock_capacity"] = base["core_count"] * base["base_clock_ghz"]
    result["core_boost_clock_capacity"] = base["core_count"] * base["boost_clock_ghz"]
    result["thread_core_ratio"] = base["thread_count"] / base["core_count"]
    result["tdp_watts_per_core"] = base["tdp_watts"] / base["core_count"]
    result["boost_capacity_per_watt"] = (
        result["core_boost_clock_capacity"] / base["tdp_watts"]
    )
    generation = frame["hardware_generation"].astype(str)
    known = pd.Series(False, index=frame.index)
    for level, column in zip(
        contract.generation_levels,
        contract.generation_columns,
        strict=True,
    ):
        indicator = generation.eq(level)
        result[column] = indicator.astype(float)
        known |= indicator
    result["generation__unknown"] = (~known).astype(float)
    return result.loc[:, contract.engineered_columns]


@dataclass(frozen=True, slots=True)
class V3Candidate:
    candidate_id: str
    engineered_features: bool
    target_transform: Literal["identity", "log"]
    num_leaves: int
    min_child_samples: int
    learning_rate: float
    n_estimators: int = 500

    def lightgbm_params(self) -> dict[str, Any]:
        return {
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "min_child_samples": self.min_child_samples,
        }


@dataclass(frozen=True, slots=True)
class V3CrossValidationResult:
    candidate: V3Candidate
    metrics: RegressionMetrics
    fold_metrics: tuple[RegressionMetrics, ...]
    best_iterations: tuple[int, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate": asdict(self.candidate),
            "metrics": self.metrics.to_dict(),
            "fold_metrics": [metrics.to_dict() for metrics in self.fold_metrics],
            "best_iterations": list(self.best_iterations),
        }


@dataclass(slots=True)
class V3DiagnosticResult:
    booster: lgb.Booster
    feature_contract: CPUFeatureContractV3
    feature_columns: tuple[str, ...]
    report: dict[str, Any]


def v3_candidate_grid() -> tuple[V3Candidate, ...]:
    candidates = [
        V3Candidate(
            candidate_id="v2_like_base_identity",
            engineered_features=False,
            target_transform="identity",
            num_leaves=24,
            min_child_samples=5,
            learning_rate=0.025,
            n_estimators=700,
        )
    ]
    for target_transform in ("identity", "log"):
        for num_leaves in (8, 16):
            for min_child_samples in (5, 10):
                candidates.append(
                    V3Candidate(
                        candidate_id=(
                            f"engineered_{target_transform}_leaves{num_leaves}_"
                            f"child{min_child_samples}"
                        ),
                        engineered_features=True,
                        target_transform=target_transform,
                        num_leaves=num_leaves,
                        min_child_samples=min_child_samples,
                        learning_rate=0.03,
                    )
                )
    return tuple(candidates)


def _target_for_fit(values: np.ndarray, transform: Literal["identity", "log"]) -> np.ndarray:
    if transform == "identity":
        return np.asarray(values, dtype=float)
    if (values <= 0).any():
        raise ValueError("log target transform requires positive targets")
    return np.asarray(np.log(values), dtype=float)


def _target_from_prediction(
    values: Any,
    transform: Literal["identity", "log"],
) -> np.ndarray:
    restored = values if transform == "identity" else np.exp(values)
    return np.asarray(np.maximum(0.0, np.asarray(restored, dtype=float)), dtype=float)


def _model_config(
    *,
    workload: str,
    feature_columns: tuple[str, ...],
    candidate: V3Candidate,
) -> PerformanceModelConfig:
    return PerformanceModelConfig(
        category="cpu",
        workload=workload,
        feature_columns=feature_columns,
        max_cpu_threads=1,
        bootstrap_resamples=100,
        lightgbm_params=candidate.lightgbm_params(),
    )


def _fit_candidate_fold(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    workload: str,
    target_column: str,
    candidate: V3Candidate,
) -> tuple[np.ndarray, int]:
    contract = fit_cpu_feature_contract(train)
    x_train = transform_cpu_features(
        train,
        contract,
        engineered=candidate.engineered_features,
    )
    x_validation = transform_cpu_features(
        validation,
        contract,
        engineered=candidate.engineered_features,
    )
    config = _model_config(
        workload=workload,
        feature_columns=tuple(x_train.columns),
        candidate=candidate,
    )
    estimator = lgb.LGBMRegressor(**_base_lightgbm_parameters(config, device="cpu"))
    y_train = train[target_column].to_numpy(dtype=float)
    y_validation = validation[target_column].to_numpy(dtype=float)
    estimator.fit(
        x_train,
        _target_for_fit(y_train, candidate.target_transform),
        eval_X=x_validation,
        eval_y=_target_for_fit(y_validation, candidate.target_transform),
        eval_metric="l1",
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, first_metric_only=True, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    best_iteration = max(1, int(estimator.best_iteration_ or estimator.n_estimators))
    predicted = estimator.predict(x_validation, num_iteration=best_iteration)
    return _target_from_prediction(predicted, candidate.target_transform), best_iteration


def _fold_assignments(development: pd.DataFrame) -> pd.Series:
    families = development["product_family"].astype(str).tolist()
    family_generation = dict(
        development.loc[:, ["product_family", "hardware_generation"]]
        .drop_duplicates()
        .astype(str)
        .itertuples(index=False, name=None)
    )
    weights = {f"fold_{index}": 1.0 for index in range(V3_FOLD_COUNT)}
    split = deterministic_group_split(
        families,
        weights=weights,
        seed=V3_CV_SEED,
        strata=family_generation,
    )
    assignments = pd.Series(split.row_assignments(families), index=development.index)
    if assignments.value_counts().min() < 2:
        raise ValueError("every v3 CV fold requires at least two rows")
    assert_group_disjoint(families, assignments.tolist())
    return assignments


def cross_validate_v3_candidates(
    development: pd.DataFrame,
    *,
    workload: str,
    target_column: str,
    candidates: tuple[V3Candidate, ...] | None = None,
) -> tuple[V3CrossValidationResult, ...]:
    """Selectable evidence computed only from the supplied development frame."""

    candidate_grid = candidates or v3_candidate_grid()
    fold_assignment = _fold_assignments(development)
    results: list[V3CrossValidationResult] = []
    for candidate in candidate_grid:
        oof = pd.Series(index=development.index, dtype=float)
        fold_metrics: list[RegressionMetrics] = []
        best_iterations: list[int] = []
        for fold_name in sorted(fold_assignment.unique()):
            validation_mask = fold_assignment.eq(fold_name)
            train = development.loc[~validation_mask]
            validation = development.loc[validation_mask]
            predicted, best_iteration = _fit_candidate_fold(
                train,
                validation,
                workload=workload,
                target_column=target_column,
                candidate=candidate,
            )
            oof.loc[validation.index] = predicted
            best_iterations.append(best_iteration)
            fold_metrics.append(
                calculate_regression_metrics(
                    validation[target_column].to_numpy(dtype=float),
                    predicted,
                )
            )
        if oof.isna().any():
            raise RuntimeError("v3 cross-validation did not predict every development row")
        results.append(
            V3CrossValidationResult(
                candidate=candidate,
                metrics=calculate_regression_metrics(
                    development[target_column].to_numpy(dtype=float),
                    oof.to_numpy(dtype=float),
                ),
                fold_metrics=tuple(fold_metrics),
                best_iterations=tuple(best_iterations),
            )
        )
    return tuple(results)


def _outer_split(frame: pd.DataFrame, config: PerformanceModelConfig) -> pd.DataFrame:
    prepared = validate_performance_frame(frame, config)
    families = prepared[config.family_column].astype(str).tolist()
    family_generation = dict(
        prepared.loc[:, [config.family_column, config.generation_column]]
        .drop_duplicates()
        .astype(str)
        .itertuples(index=False, name=None)
    )
    weights = {"development": 0.70, "calibration": 0.15, "holdout": 0.15}
    split = deterministic_group_split(
        families,
        weights=weights,
        seed=V3_SPLIT_SEED,
        strata=family_generation,
    )
    prepared["v3_split"] = split.row_assignments(families)
    assert_group_disjoint(families, prepared["v3_split"].tolist())
    if prepared["v3_split"].value_counts().min() < 2:
        raise ValueError("every v3 outer split requires at least two rows")
    return prepared


def _fit_final_model(
    development: pd.DataFrame,
    *,
    workload: str,
    target_column: str,
    candidate: V3Candidate,
    n_estimators: int,
) -> tuple[lgb.LGBMRegressor, CPUFeatureContractV3, pd.DataFrame]:
    contract = fit_cpu_feature_contract(development)
    features = transform_cpu_features(
        development,
        contract,
        engineered=candidate.engineered_features,
    )
    final_candidate = V3Candidate(
        candidate_id=candidate.candidate_id,
        engineered_features=candidate.engineered_features,
        target_transform=candidate.target_transform,
        num_leaves=candidate.num_leaves,
        min_child_samples=candidate.min_child_samples,
        learning_rate=candidate.learning_rate,
        n_estimators=n_estimators,
    )
    config = _model_config(
        workload=workload,
        feature_columns=tuple(features.columns),
        candidate=final_candidate,
    )
    estimator = lgb.LGBMRegressor(**_base_lightgbm_parameters(config, device="cpu"))
    target = development[target_column].to_numpy(dtype=float)
    estimator.fit(features, _target_for_fit(target, candidate.target_transform))
    return estimator, contract, features


def run_v3_diagnostic(
    frame: pd.DataFrame,
    config: PerformanceModelConfig,
    *,
    dataset_evidence: DatasetEvidence,
    input_csv_sha256: str,
    dataset_manifest_sha256: str,
    bootstrap_resamples: int = 2000,
) -> V3DiagnosticResult:
    """Run one isolated experiment; holdout is accessed only after CV selection."""

    if config.category != "cpu" or tuple(config.feature_columns) != CPU_BASE_FEATURES:
        raise ValueError("v3 diagnostic currently supports the exact CPU base feature contract")
    prepared = _outer_split(frame, config)
    development = prepared.loc[prepared["v3_split"] == "development"].copy()
    calibration_frame = prepared.loc[prepared["v3_split"] == "calibration"].copy()

    cv_results = cross_validate_v3_candidates(
        development,
        workload=config.workload,
        target_column=config.target_column,
    )
    selected = min(
        cv_results,
        key=lambda result: (
            result.metrics.mape_percent,
            -result.metrics.r2,
            result.metrics.mae,
            result.candidate.candidate_id,
        ),
    )
    baseline = next(
        result for result in cv_results if result.candidate.candidate_id == "v2_like_base_identity"
    )
    final_iterations = max(1, round(statistics.median(selected.best_iterations)))
    estimator, feature_contract, development_features = _fit_final_model(
        development,
        workload=config.workload,
        target_column=config.target_column,
        candidate=selected.candidate,
        n_estimators=final_iterations,
    )

    # The holdout is intentionally materialised only after selection and final fit.
    holdout = prepared.loc[prepared["v3_split"] == "holdout"].copy()
    calibration_features = transform_cpu_features(
        calibration_frame,
        feature_contract,
        engineered=selected.candidate.engineered_features,
    )
    holdout_features = transform_cpu_features(
        holdout,
        feature_contract,
        engineered=selected.candidate.engineered_features,
    )
    calibration_prediction = _target_from_prediction(
        estimator.predict(calibration_features),
        selected.candidate.target_transform,
    )
    holdout_prediction = _target_from_prediction(
        estimator.predict(holdout_features),
        selected.candidate.target_transform,
    )
    holdout_actual = holdout[config.target_column].to_numpy(dtype=float)
    holdout_metrics = calculate_regression_metrics(holdout_actual, holdout_prediction)
    calibration = calibrate_prediction_intervals(
        calibration_frame[config.target_column].to_numpy(dtype=float),
        calibration_prediction,
        calibration_frame[config.family_column].astype(str).tolist(),
        holdout_actual,
        holdout_prediction,
        alpha=config.prediction_interval_alpha,
    )
    uncertainty = grouped_bootstrap_uncertainty(
        holdout_actual,
        holdout_prediction,
        holdout[config.family_column].astype(str).tolist(),
        confidence_level=config.bootstrap_confidence_level,
        n_resamples=bootstrap_resamples,
        seed=V3_SPLIT_SEED,
    )
    profiles = {
        column: FeatureProfile(
            minimum=float(development_features[column].min()),
            maximum=float(development_features[column].max()),
            missing_fraction=0.0,
        )
        for column in development_features.columns
    }
    diagnostics = grouped_test_diagnostics(
        test_frame=holdout_features.assign(
            **{config.family_column: holdout[config.family_column].astype(str).to_numpy()}
        ),
        observed=holdout_actual,
        predicted=holdout_prediction,
        group_column=config.family_column,
        development_groups=set(development[config.family_column].astype(str)),
        feature_columns=tuple(development_features.columns),
        feature_profiles=profiles,
    )

    split_group_hashes = {
        split_name: sorted(
            _group_digest(value)
            for value in split_frame[config.family_column].astype(str).unique()
        )
        for split_name, split_frame in prepared.groupby("v3_split", sort=True)
    }
    relative_cv_mape_improvement = (
        (baseline.metrics.mape_percent - selected.metrics.mape_percent)
        / baseline.metrics.mape_percent
        * 100.0
    )
    cv_r2_delta = selected.metrics.r2 - baseline.metrics.r2
    materially_improves_cv = relative_cv_mape_improvement >= 5.0 and cv_r2_delta >= 0.0
    minimum_coverage = 1.0 - config.prediction_interval_alpha - (
        config.max_interval_coverage_shortfall
    )
    metric_gate_blockers: list[str] = []
    if holdout_metrics.sample_count < config.min_confident_test_rows:
        metric_gate_blockers.append("holdout row count is below the unchanged v2 gate")
    if diagnostics.test_group_count < config.min_confident_test_groups:
        metric_gate_blockers.append("holdout group count is below the unchanged v2 gate")
    if holdout_metrics.r2 < config.min_confident_r2:
        metric_gate_blockers.append("holdout R2 is below the unchanged v2 gate")
    if holdout_metrics.mape_percent > config.max_confident_mape_percent:
        metric_gate_blockers.append("holdout MAPE exceeds the unchanged v2 gate")
    if uncertainty.r2_lower < config.min_confident_r2:
        metric_gate_blockers.append("grouped-bootstrap R2 lower bound is below the v2 gate")
    if uncertainty.mape_percent_upper > config.max_confident_mape_percent:
        metric_gate_blockers.append("grouped-bootstrap MAPE upper bound exceeds the v2 gate")
    if calibration.test_coverage < minimum_coverage:
        metric_gate_blockers.append("holdout interval coverage is below the unchanged v2 gate")
    if diagnostics.outside_training_envelope_fraction > config.max_test_ood_fraction:
        metric_gate_blockers.append("holdout OOD fraction exceeds the unchanged v2 gate")

    model_payload = {
        "schema_version": V3_DIAGNOSTIC_SCHEMA_VERSION,
        "candidate": asdict(selected.candidate),
        "final_iterations": final_iterations,
        "feature_contract": feature_contract.to_dict(),
        "input_csv_sha256": input_csv_sha256,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "split_group_hashes": split_group_hashes,
    }
    model_digest = hashlib.sha256(
        json.dumps(
            model_payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    )
    model_digest.update(estimator.booster_.model_to_string().encode("utf-8"))
    model_version = model_digest.hexdigest()
    report: dict[str, Any] = {
        "schema_version": V3_DIAGNOSTIC_SCHEMA_VERSION,
        "model_version": model_version,
        "task": "cpu_performance_v3_development_diagnostic",
        "category": config.category,
        "workload": config.workload,
        "input": {
            "csv_sha256": input_csv_sha256,
            "dataset_manifest_sha256": dataset_manifest_sha256,
            "rows": len(prepared),
            "groups": int(prepared[config.family_column].nunique()),
        },
        "data_evidence": dataset_evidence.to_dict(),
        "protocol": {
            "split_seed": V3_SPLIT_SEED,
            "cv_seed": V3_CV_SEED,
            "cv_folds": V3_FOLD_COUNT,
            "selection_data": "development groups only",
            "calibration_excluded_from_selection": True,
            "holdout_excluded_from_selection": True,
            "v2_test_metrics_used_for_selection": False,
            "split_group_hashes": split_group_hashes,
            "split_row_counts": prepared["v3_split"].value_counts().sort_index().to_dict(),
            "split_group_counts": prepared.groupby("v3_split")[config.family_column]
            .nunique()
            .sort_index()
            .to_dict(),
        },
        "feature_contract": feature_contract.to_dict(),
        "cross_validation": {
            "selection_metric": "minimum OOF MAPE; then higher OOF R2",
            "candidates": [result.to_dict() for result in cv_results],
            "baseline_candidate_id": baseline.candidate.candidate_id,
            "selected_candidate_id": selected.candidate.candidate_id,
            "selected_metrics": selected.metrics.to_dict(),
            "baseline_metrics": baseline.metrics.to_dict(),
            "relative_mape_improvement_percent": relative_cv_mape_improvement,
            "r2_delta": cv_r2_delta,
            "material_improvement_policy": "MAPE improves by >=5% and R2 does not decline",
            "materially_improves": materially_improves_cv,
        },
        "final_fit": {
            "iterations": final_iterations,
            "device": "cpu",
            "threads": 1,
            "target_transform": selected.candidate.target_transform,
        },
        "calibration": calibration.to_dict(),
        "holdout": {
            "metrics": holdout_metrics.to_dict(),
            "grouped_uncertainty": uncertainty.to_dict(),
            "grouped_test": diagnostics.to_dict(),
        },
        "promotion": {
            "eligible": False,
            "metric_gates_passed": not metric_gate_blockers,
            "metric_gate_blockers": metric_gate_blockers,
            "block_reasons": [
                "v3 output is a development-only schema that production inference cannot load",
                "the source dataset was reused after aggregate v2 test metrics were observed",
                "adaptive feature and hyperparameter experiments require a new external frozen set",
            ],
        },
    }
    return V3DiagnosticResult(
        booster=estimator.booster_,
        feature_contract=feature_contract,
        feature_columns=tuple(development_features.columns),
        report=report,
    )


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            json.dump(payload, handle, allow_nan=False, indent=2, sort_keys=True)
            handle.write("\n")
            temporary_name = handle.name
        os.replace(temporary_name, path)
    finally:
        if temporary_name is not None and Path(temporary_name).exists():
            Path(temporary_name).unlink()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_v3_diagnostic(result: V3DiagnosticResult, path: str | Path) -> Path:
    destination = Path(path)
    if destination.exists() and not destination.is_dir():
        raise NotADirectoryError(destination)
    destination.mkdir(parents=True, exist_ok=True)
    model_path = destination / "model.txt"
    temporary_model = destination / f".model.txt.{os.getpid()}.tmp"
    try:
        result.booster.save_model(str(temporary_model))
        os.replace(temporary_model, model_path)
    finally:
        if temporary_model.exists():
            temporary_model.unlink()
    feature_path = destination / "feature_contract.json"
    report_path = destination / "diagnostic_report.json"
    _write_json_atomic(feature_path, result.feature_contract.to_dict())
    _write_json_atomic(report_path, result.report)
    files = {
        file_path.name: {
            "sha256": _sha256_file(file_path),
            "size_bytes": file_path.stat().st_size,
        }
        for file_path in (model_path, feature_path, report_path)
    }
    _write_json_atomic(
        destination / "artifact_manifest.json",
        {
            "schema_version": "pc-build-recommender.performance-v3-manifest.v1",
            "diagnostic_schema_version": V3_DIAGNOSTIC_SCHEMA_VERSION,
            "model_version": result.report["model_version"],
            "production_loadable": False,
            "files": files,
        },
    )
    return destination
