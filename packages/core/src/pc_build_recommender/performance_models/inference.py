"""Observed-first inference semantics for workload performance."""

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite
from typing import cast

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from .contracts import (
    PerformanceDecision,
    PerformanceEstimate,
    PerformanceModelArtifact,
    TargetTransform,
)
from .target_transforms import inverse_target_predictions


def _feature_frame(
    artifact: PerformanceModelArtifact,
    features: Mapping[str, float | int | None] | pd.Series,
) -> pd.DataFrame:
    missing = [column for column in artifact.config.feature_columns if column not in features]
    if missing:
        raise ValueError(f"prediction features are missing required columns: {missing}")
    if artifact.config.strict_inference_features:
        unexpected = sorted(set(features.keys()).difference(artifact.config.feature_columns))
        if unexpected:
            raise ValueError(f"prediction features contain unexpected columns: {unexpected}")
    values: dict[str, list[float]] = {}
    for column in artifact.config.feature_columns:
        raw_value = features[column]
        if raw_value is None or raw_value is pd.NA:
            values[column] = [float("nan")]
            continue
        if isinstance(raw_value, (bool, np.bool_)) or not isinstance(
            raw_value, (int, float, np.number)
        ):
            raise TypeError(f"prediction feature {column!r} must be numeric or missing")
        value = float(raw_value)
        if np.isnan(value):
            values[column] = [float("nan")]
            continue
        if not isfinite(value):
            raise ValueError(f"prediction feature {column!r} cannot be infinite")
        values[column] = [value]
    return pd.DataFrame(values, columns=artifact.config.feature_columns)


def _request_confidence_issues(
    artifact: PerformanceModelArtifact,
    feature_frame: pd.DataFrame,
) -> tuple[str, ...]:
    issues: list[str] = []
    row = feature_frame.iloc[0]
    missing_fraction = float(row.isna().mean())
    if missing_fraction > artifact.allowed_missing_fraction:
        issues.append(
            f"input missing fraction {missing_fraction:.4f} exceeds allowed "
            f"{artifact.allowed_missing_fraction:.4f}"
        )
    for feature, profile in artifact.feature_profiles.items():
        raw_value = row[feature]
        if pd.isna(raw_value):
            continue
        value = float(raw_value)
        if value < profile.minimum or value > profile.maximum:
            issues.append(
                f"feature {feature!r}={value:.6g} is outside training range "
                f"[{profile.minimum:.6g}, {profile.maximum:.6g}]"
            )
    return tuple(issues)


def _relative_only_decision(
    artifact: PerformanceModelArtifact,
    request_issues: tuple[str, ...],
) -> PerformanceDecision:
    """Return a closed fallback code without exposing raw input values downstream."""

    if request_issues:
        if artifact.precise_predictions_enabled:
            return "input_outside_training_contract"
        if artifact.promotable:
            return "precise_predictions_disabled_and_input_outside_training_contract"
        return "model_not_promotion_eligible_and_input_outside_training_contract"
    if artifact.promotable:
        return "precise_predictions_disabled"
    return "model_not_promotion_eligible"


def estimate_performance(
    artifact: PerformanceModelArtifact,
    features: Mapping[str, float | int | None] | pd.Series | None = None,
    *,
    observed_score: float | None = None,
    observed_source: str | None = None,
) -> PerformanceEstimate:
    """Return an observed value before considering any model prediction.

    Low-confidence or non-promotable models expose the raw model output only as
    ``relative_score``.  Their precise ``score`` remains ``None`` so downstream
    presentation cannot accidentally describe development output as a measured
    benchmark estimate.
    """

    if observed_score is not None:
        return observed_performance_estimate(observed_score, observed_source)

    if observed_source is not None:
        raise ValueError("observed_source cannot be supplied without observed_score")
    if features is None:
        raise ValueError("features are required when no observed score is available")
    feature_frame = _feature_frame(artifact, features)
    request_issues = _request_confidence_issues(artifact, feature_frame)
    # Artifacts sealed before target transforms were introduced contain native
    # scores, so their absence has the unambiguous historical meaning
    # ``identity``.  Keep that schema evolution safe for durable artifacts and
    # lightweight integration fixtures alike.
    target_transform = cast(
        TargetTransform,
        getattr(artifact.config, "target_transform", "identity"),
    )
    prediction = inverse_target_predictions(
        np.asarray(
            artifact.booster.predict(
                feature_frame,
                num_iteration=artifact.best_iteration,
            ),
            dtype=float,
        ),
        transform=target_transform,
    ).reshape(-1)
    if prediction.size != 1 or not np.isfinite(prediction[0]):
        raise RuntimeError("performance model returned an invalid prediction")
    relative_score = max(0.0, float(prediction[0]))

    if artifact.precise_predictions_enabled and not request_issues:
        lower_score, upper_score = artifact.calibration.interval(relative_score)
        return PerformanceEstimate(
            score=relative_score,
            relative_score=relative_score,
            basis="predicted",
            confidence=artifact.confidence_level,
            decision="precise_model_prediction",
            model_version=artifact.model_version,
            lower_score=lower_score,
            upper_score=upper_score,
        )
    reasons = (*artifact.promotion_block_reasons, *request_issues)
    reason = "; ".join(reasons) or ("model confidence is insufficient for a precise estimate")
    return PerformanceEstimate(
        score=None,
        relative_score=relative_score,
        basis="relative_only",
        confidence="low" if request_issues else artifact.confidence_level,
        decision=_relative_only_decision(artifact, request_issues),
        model_version=artifact.model_version,
        reason=reason,
    )


def observed_performance_estimate(
    observed_score: float,
    observed_source: str | None,
) -> PerformanceEstimate:
    """Construct a measured estimate without requiring a model artifact."""

    measured = float(observed_score)
    if not isfinite(measured) or measured <= 0:
        raise ValueError("observed_score must be finite and positive")
    if observed_source is None or not observed_source.strip():
        raise ValueError("observed_source is required with an observed score")
    return PerformanceEstimate(
        score=measured,
        relative_score=measured,
        basis="observed",
        confidence="observed",
        decision="observed_benchmark",
        model_version=None,
        supporting_sources=(observed_source.strip(),),
    )
