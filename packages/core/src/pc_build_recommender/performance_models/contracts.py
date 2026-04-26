"""Contracts for workload-performance regression models.

The package deliberately keeps exact benchmark observations separate from model
estimates.  A caller must never have to infer whether a displayed number was
measured or predicted from an incidental field.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import isfinite
from typing import Any, Literal

from pc_build_recommender.evaluation.contracts import DataUseDeclaration

ARTIFACT_SCHEMA_VERSION = "pc-build-recommender.performance-model.v2"

PredictionBasis = Literal["observed", "predicted", "relative_only"]
ConfidenceLevel = Literal["observed", "high", "medium", "low"]
PerformanceDecision = Literal[
    "observed_benchmark",
    "precise_model_prediction",
    "model_not_promotion_eligible",
    "input_outside_training_contract",
    "model_not_promotion_eligible_and_input_outside_training_contract",
    "precise_predictions_disabled",
    "precise_predictions_disabled_and_input_outside_training_contract",
]
TargetTransform = Literal["identity", "log1p"]


@dataclass(frozen=True, slots=True)
class PerformanceModelConfig:
    """Configuration for one category-and-workload-specific regressor."""

    category: str
    workload: str
    feature_columns: tuple[str, ...]
    target_column: str = "target_score"
    product_id_column: str = "product_id"
    family_column: str = "product_family"
    generation_column: str = "hardware_generation"
    synthetic_column: str = "is_synthetic"
    split_column: str = "split"
    split_seed: int = 20260722
    split_weights: dict[str, float] = field(
        default_factory=lambda: {
            "train": 0.55,
            "validation": 0.15,
            "calibration": 0.15,
            "test": 0.15,
        }
    )
    ridge_alpha: float = 1.0
    min_confident_r2: float = 0.85
    max_confident_mape_percent: float = 12.0
    min_confident_test_rows: int = 20
    min_confident_test_groups: int = 10
    prediction_interval_alpha: float = 0.10
    min_calibration_rows: int = 20
    min_calibration_groups: int = 10
    max_interval_coverage_shortfall: float = 0.10
    max_test_ood_fraction: float = 0.20
    bootstrap_resamples: int = 2000
    bootstrap_confidence_level: float = 0.95
    minimum_baseline_mape_improvement_percent: float = 0.0
    max_prediction_missing_fraction: float = 0.0
    max_training_missing_fraction: float = 0.25
    min_feature_unique_values: int = 2
    strict_inference_features: bool = True
    require_baseline_improvement: bool = True
    target_transform: TargetTransform = "identity"
    requested_device: Literal["auto", "cpu", "gpu", "cuda"] = "cpu"
    allow_device_fallback: bool = True
    max_training_memory_mb: int = 2048
    max_cpu_threads: int = 4
    gpu_max_bin: int = 63
    lightgbm_params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "feature_columns", tuple(self.feature_columns))
        object.__setattr__(self, "split_weights", dict(self.split_weights))
        object.__setattr__(self, "lightgbm_params", dict(self.lightgbm_params))
        if not self.category or not self.workload:
            raise ValueError("category and workload must not be empty")
        if not self.feature_columns:
            raise ValueError("at least one feature column is required")
        if len(self.feature_columns) != len(set(self.feature_columns)):
            raise ValueError("feature columns must be unique")
        reserved = {
            self.target_column,
            self.product_id_column,
            self.family_column,
            self.generation_column,
            self.synthetic_column,
            self.split_column,
        }
        overlap = reserved.intersection(self.feature_columns)
        if overlap:
            raise ValueError(f"reserved columns cannot be model features: {sorted(overlap)}")
        required_splits = {"train", "validation", "calibration", "test"}
        if set(self.split_weights) != required_splits:
            raise ValueError(
                "split_weights must contain exactly train, validation, calibration, and test"
            )
        if any(not isfinite(weight) or weight <= 0 for weight in self.split_weights.values()):
            raise ValueError("split weights must be finite and positive")
        if not isfinite(self.ridge_alpha) or self.ridge_alpha < 0:
            raise ValueError("ridge_alpha must be finite and non-negative")
        if not isfinite(self.min_confident_r2) or self.min_confident_r2 > 1:
            raise ValueError("min_confident_r2 must be finite and at most one")
        if not isfinite(self.max_confident_mape_percent) or self.max_confident_mape_percent <= 0:
            raise ValueError("max_confident_mape_percent must be finite and positive")
        if self.min_confident_test_rows < 2:
            raise ValueError("min_confident_test_rows must be at least two")
        if self.min_confident_test_groups < 2:
            raise ValueError("min_confident_test_groups must be at least two")
        if not 0.0 < self.prediction_interval_alpha < 1.0:
            raise ValueError("prediction_interval_alpha must be between zero and one")
        if self.min_calibration_rows < 2:
            raise ValueError("min_calibration_rows must be at least two")
        if self.min_calibration_groups < 2:
            raise ValueError("min_calibration_groups must be at least two")
        if not 0.0 <= self.max_interval_coverage_shortfall < 1.0:
            raise ValueError("max_interval_coverage_shortfall must be in [0, 1)")
        if not 0.0 <= self.max_test_ood_fraction <= 1.0:
            raise ValueError("max_test_ood_fraction must be between zero and one")
        if self.bootstrap_resamples < 100:
            raise ValueError("bootstrap_resamples must be at least 100")
        if not 0.0 < self.bootstrap_confidence_level < 1.0:
            raise ValueError("bootstrap_confidence_level must be between zero and one")
        if (
            not isfinite(self.minimum_baseline_mape_improvement_percent)
            or self.minimum_baseline_mape_improvement_percent < 0
        ):
            raise ValueError(
                "minimum_baseline_mape_improvement_percent must be finite and non-negative"
            )
        if not 0.0 <= self.max_prediction_missing_fraction <= 1.0:
            raise ValueError("max_prediction_missing_fraction must be between zero and one")
        if not 0.0 <= self.max_training_missing_fraction < 1.0:
            raise ValueError("max_training_missing_fraction must be in [0, 1)")
        if self.min_feature_unique_values < 2:
            raise ValueError("min_feature_unique_values must be at least two")
        if self.target_transform not in {"identity", "log1p"}:
            raise ValueError("target_transform must be identity or log1p")
        if self.requested_device not in {"auto", "cpu", "gpu", "cuda"}:
            raise ValueError("requested_device must be auto, cpu, gpu, or cuda")
        if self.max_training_memory_mb < 64:
            raise ValueError("max_training_memory_mb must be at least 64")
        if self.max_cpu_threads < 1:
            raise ValueError("max_cpu_threads must be positive")
        if self.gpu_max_bin < 16:
            raise ValueError("gpu_max_bin must be at least 16")
        parameter_device = self.lightgbm_params.get("device_type")
        if parameter_device is not None and str(parameter_device) not in {
            "auto",
            "cpu",
            "gpu",
            "cuda",
        }:
            raise ValueError("lightgbm_params.device_type must be auto, cpu, gpu, or cuda")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PerformanceModelConfig:
        values = dict(payload)
        values["feature_columns"] = tuple(values["feature_columns"])
        # JSON metadata is written with sorted keys, but split allocation intentionally
        # follows policy order. Restore the canonical order so save/load cannot change
        # group assignments or the semantic training-data digest.
        split_weights = dict(values["split_weights"])
        values["split_weights"] = {
            name: float(split_weights[name])
            for name in ("train", "validation", "calibration", "test")
        }
        return cls(**values)


@dataclass(frozen=True, slots=True)
class RegressionMetrics:
    """Held-out regression metrics; MAPE is represented as percentage points."""

    r2: float
    mae: float
    mape_percent: float
    sample_count: int

    def __post_init__(self) -> None:
        if not all(isfinite(value) for value in (self.r2, self.mae, self.mape_percent)):
            raise ValueError("regression metrics must be finite")
        if self.mae < 0 or self.mape_percent < 0:
            raise ValueError("error metrics must be non-negative")
        if self.sample_count < 2:
            raise ValueError("regression metrics require at least two observations")

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RegressionMetrics:
        return cls(
            r2=float(payload["r2"]),
            mae=float(payload["mae"]),
            mape_percent=float(payload["mape_percent"]),
            sample_count=int(payload["sample_count"]),
        )


@dataclass(frozen=True, slots=True)
class ModelEvaluation:
    """Validation and untouched-test metrics for one candidate estimator."""

    model_name: str
    validation: RegressionMetrics
    test: RegressionMetrics

    def __post_init__(self) -> None:
        if not self.model_name:
            raise ValueError("model_name must not be empty")

    def to_dict(self) -> dict[str, object]:
        return {
            "model_name": self.model_name,
            "validation": self.validation.to_dict(),
            "test": self.test.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ModelEvaluation:
        return cls(
            model_name=str(payload["model_name"]),
            validation=RegressionMetrics.from_dict(dict(payload["validation"])),
            test=RegressionMetrics.from_dict(dict(payload["test"])),
        )


@dataclass(frozen=True, slots=True)
class FeatureProfile:
    """Training envelope used to reject unsupported inference inputs."""

    minimum: float
    maximum: float
    missing_fraction: float

    def __post_init__(self) -> None:
        if not isfinite(self.minimum) or not isfinite(self.maximum):
            raise ValueError("feature bounds must be finite")
        if self.minimum > self.maximum:
            raise ValueError("feature minimum cannot exceed maximum")
        if not 0.0 <= self.missing_fraction <= 1.0:
            raise ValueError("feature missing_fraction must be between zero and one")

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> FeatureProfile:
        return cls(
            minimum=float(payload["minimum"]),
            maximum=float(payload["maximum"]),
            missing_fraction=float(payload["missing_fraction"]),
        )


@dataclass(frozen=True, slots=True)
class DatasetEvidence:
    """Verified upstream dataset eligibility carried into every model artifact."""

    verified: bool
    eligible_for_promotion: bool
    manifest_sha256: str | None
    block_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "block_reasons", tuple(self.block_reasons))
        if self.manifest_sha256 is not None and (
            len(self.manifest_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.manifest_sha256)
        ):
            raise ValueError("manifest_sha256 must be a lowercase SHA-256 digest")
        if self.verified and self.manifest_sha256 is None:
            raise ValueError("verified dataset evidence requires a manifest digest")
        if self.eligible_for_promotion and not self.verified:
            raise ValueError("unverified dataset evidence cannot be promotion eligible")
        if self.eligible_for_promotion and self.block_reasons:
            raise ValueError("eligible dataset evidence cannot have promotion blockers")
        if not self.eligible_for_promotion and not self.block_reasons:
            raise ValueError("ineligible dataset evidence requires at least one blocker")

    @classmethod
    def unverified(cls) -> DatasetEvidence:
        return cls(
            verified=False,
            eligible_for_promotion=False,
            manifest_sha256=None,
            block_reasons=("dataset manifest was not verified",),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "verified": self.verified,
            "eligible_for_promotion": self.eligible_for_promotion,
            "manifest_sha256": self.manifest_sha256,
            "block_reasons": list(self.block_reasons),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> DatasetEvidence:
        return cls(
            verified=bool(payload["verified"]),
            eligible_for_promotion=bool(payload["eligible_for_promotion"]),
            manifest_sha256=(
                str(payload["manifest_sha256"])
                if payload.get("manifest_sha256") is not None
                else None
            ),
            block_reasons=tuple(str(reason) for reason in payload["block_reasons"]),
        )


@dataclass(frozen=True, slots=True)
class PredictionIntervalCalibration:
    """Independent split-conformal calibration and untouched-test coverage evidence."""

    method: Literal["split_conformal_absolute_residual"]
    alpha: float
    absolute_error_quantile: float
    calibration_sample_count: int
    calibration_group_count: int
    test_sample_count: int
    test_covered_count: int
    test_coverage_lower_95: float

    def __post_init__(self) -> None:
        if self.method != "split_conformal_absolute_residual":
            raise ValueError(f"unsupported calibration method: {self.method!r}")
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("calibration alpha must be between zero and one")
        if not isfinite(self.absolute_error_quantile) or self.absolute_error_quantile < 0:
            raise ValueError("absolute_error_quantile must be finite and non-negative")
        if self.calibration_sample_count < 2 or self.calibration_group_count < 1:
            raise ValueError("calibration evidence is too small")
        if self.test_sample_count < 2:
            raise ValueError("test coverage requires at least two observations")
        if not 0 <= self.test_covered_count <= self.test_sample_count:
            raise ValueError("test_covered_count must be within the test sample count")
        if not 0.0 <= self.test_coverage_lower_95 <= 1.0:
            raise ValueError("test_coverage_lower_95 must be between zero and one")
        if self.test_coverage_lower_95 > self.test_coverage:
            raise ValueError("coverage lower bound cannot exceed observed coverage")

    @property
    def nominal_coverage(self) -> float:
        return 1.0 - self.alpha

    @property
    def test_coverage(self) -> float:
        return self.test_covered_count / self.test_sample_count

    def interval(self, prediction: float) -> tuple[float, float]:
        if not isfinite(prediction):
            raise ValueError("prediction must be finite")
        return (
            max(0.0, prediction - self.absolute_error_quantile),
            max(0.0, prediction + self.absolute_error_quantile),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "method": self.method,
            "alpha": self.alpha,
            "nominal_coverage": self.nominal_coverage,
            "absolute_error_quantile": self.absolute_error_quantile,
            "calibration_sample_count": self.calibration_sample_count,
            "calibration_group_count": self.calibration_group_count,
            "test_sample_count": self.test_sample_count,
            "test_covered_count": self.test_covered_count,
            "test_coverage": self.test_coverage,
            "test_coverage_lower_95": self.test_coverage_lower_95,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PredictionIntervalCalibration:
        return cls(
            method=str(payload["method"]),  # type: ignore[arg-type]
            alpha=float(payload["alpha"]),
            absolute_error_quantile=float(payload["absolute_error_quantile"]),
            calibration_sample_count=int(payload["calibration_sample_count"]),
            calibration_group_count=int(payload["calibration_group_count"]),
            test_sample_count=int(payload["test_sample_count"]),
            test_covered_count=int(payload["test_covered_count"]),
            test_coverage_lower_95=float(payload["test_coverage_lower_95"]),
        )


@dataclass(frozen=True, slots=True)
class GroupedTestDiagnostics:
    """Family-disjoint test evidence and feature-envelope exposure."""

    group_column: str
    test_group_count: int
    test_row_count: int
    development_group_overlap_count: int
    evaluable_group_count: int
    worst_group_mape_percent: float | None
    outside_training_envelope_row_count: int

    def __post_init__(self) -> None:
        if not self.group_column:
            raise ValueError("group_column must not be empty")
        if self.test_group_count < 1 or self.test_row_count < 2:
            raise ValueError("grouped test evidence is too small")
        if not 0 <= self.development_group_overlap_count <= self.test_group_count:
            raise ValueError("development group overlap count is invalid")
        if not 0 <= self.evaluable_group_count <= self.test_group_count:
            raise ValueError("evaluable group count is invalid")
        if self.worst_group_mape_percent is not None and (
            not isfinite(self.worst_group_mape_percent) or self.worst_group_mape_percent < 0
        ):
            raise ValueError("worst_group_mape_percent must be finite and non-negative")
        if not 0 <= self.outside_training_envelope_row_count <= self.test_row_count:
            raise ValueError("outside-training-envelope row count is invalid")

    @property
    def outside_training_envelope_fraction(self) -> float:
        return self.outside_training_envelope_row_count / self.test_row_count

    def to_dict(self) -> dict[str, object]:
        return {
            "group_column": self.group_column,
            "test_group_count": self.test_group_count,
            "test_row_count": self.test_row_count,
            "development_group_overlap_count": self.development_group_overlap_count,
            "evaluable_group_count": self.evaluable_group_count,
            "worst_group_mape_percent": self.worst_group_mape_percent,
            "outside_training_envelope_row_count": self.outside_training_envelope_row_count,
            "outside_training_envelope_fraction": self.outside_training_envelope_fraction,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> GroupedTestDiagnostics:
        return cls(
            group_column=str(payload["group_column"]),
            test_group_count=int(payload["test_group_count"]),
            test_row_count=int(payload["test_row_count"]),
            development_group_overlap_count=int(payload["development_group_overlap_count"]),
            evaluable_group_count=int(payload["evaluable_group_count"]),
            worst_group_mape_percent=(
                float(payload["worst_group_mape_percent"])
                if payload.get("worst_group_mape_percent") is not None
                else None
            ),
            outside_training_envelope_row_count=int(payload["outside_training_envelope_row_count"]),
        )


@dataclass(frozen=True, slots=True)
class RegressionUncertainty:
    """Grouped bootstrap confidence intervals for untouched-test metrics."""

    confidence_level: float
    resamples: int
    group_count: int
    r2_lower: float
    r2_upper: float
    mae_lower: float
    mae_upper: float
    mape_percent_lower: float
    mape_percent_upper: float

    def __post_init__(self) -> None:
        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError("confidence_level must be between zero and one")
        if self.resamples < 100:
            raise ValueError("resamples must be at least 100")
        if self.group_count < 2:
            raise ValueError("grouped bootstrap requires at least two groups")
        values = (
            self.r2_lower,
            self.r2_upper,
            self.mae_lower,
            self.mae_upper,
            self.mape_percent_lower,
            self.mape_percent_upper,
        )
        if not all(isfinite(value) for value in values):
            raise ValueError("bootstrap interval bounds must be finite")
        if self.r2_lower > self.r2_upper:
            raise ValueError("R2 interval bounds are invalid")
        if self.mae_lower < 0 or self.mae_lower > self.mae_upper:
            raise ValueError("MAE interval bounds are invalid")
        if self.mape_percent_lower < 0 or self.mape_percent_lower > self.mape_percent_upper:
            raise ValueError("MAPE interval bounds are invalid")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RegressionUncertainty:
        return cls(
            confidence_level=float(payload["confidence_level"]),
            resamples=int(payload["resamples"]),
            group_count=int(payload["group_count"]),
            r2_lower=float(payload["r2_lower"]),
            r2_upper=float(payload["r2_upper"]),
            mae_lower=float(payload["mae_lower"]),
            mae_upper=float(payload["mae_upper"]),
            mape_percent_lower=float(payload["mape_percent_lower"]),
            mape_percent_upper=float(payload["mape_percent_upper"]),
        )


@dataclass(slots=True)
class PerformanceModelArtifact:
    """In-memory LightGBM artifact plus its complete evidence contract."""

    config: PerformanceModelConfig
    booster: Any
    evaluations: dict[str, ModelEvaluation]
    data_use: DataUseDeclaration
    training_data_sha256: str
    model_version: str
    split_group_counts: dict[str, int]
    split_row_counts: dict[str, int]
    split_group_hashes: dict[str, tuple[str, ...]]
    development_group_hashes: tuple[str, ...]
    feature_profiles: dict[str, FeatureProfile]
    dataset_evidence: DatasetEvidence
    calibration: PredictionIntervalCalibration
    grouped_test: GroupedTestDiagnostics
    test_uncertainty: RegressionUncertainty
    estimated_peak_training_memory_mb: float
    allowed_missing_fraction: float
    best_iteration: int
    confidence_level: Literal["high", "medium", "low"]
    precise_predictions_enabled: bool
    promotable: bool
    promotion_block_reasons: tuple[str, ...]
    requested_device: str
    actual_device: str
    device_fallback_reason: str | None
    schema_version: str = ARTIFACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        self.split_group_hashes = {
            str(name): tuple(hashes) for name, hashes in self.split_group_hashes.items()
        }
        self.development_group_hashes = tuple(self.development_group_hashes)
        if self.schema_version != ARTIFACT_SCHEMA_VERSION:
            raise ValueError(f"unsupported artifact schema: {self.schema_version!r}")
        for value_name, value in (
            ("training_data_sha256", self.training_data_sha256),
            ("model_version", self.model_version),
        ):
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise ValueError(f"{value_name} must be a lowercase SHA-256 digest")
        if "lightgbm" not in self.evaluations:
            raise ValueError("artifact evaluations must contain the LightGBM model")
        if self.best_iteration < 1:
            raise ValueError("best_iteration must be positive")
        if set(self.feature_profiles) != set(self.config.feature_columns):
            raise ValueError("feature profiles must exactly match configured features")
        if set(self.split_group_hashes) != set(self.config.split_weights):
            raise ValueError("split group hashes must cover every configured split")
        for split_name, hashes in self.split_group_hashes.items():
            if len(hashes) != self.split_group_counts.get(split_name) or len(hashes) != len(
                set(hashes)
            ):
                raise ValueError(f"split group hashes are inconsistent for {split_name!r}")
            if any(
                len(value) != 64 or any(character not in "0123456789abcdef" for character in value)
                for value in hashes
            ):
                raise ValueError("split group hashes must be lowercase SHA-256 digests")
        expected_development_hashes = tuple(
            sorted({value for hashes in self.split_group_hashes.values() for value in hashes})
        )
        if self.development_group_hashes != expected_development_hashes:
            raise ValueError("development group hashes must equal the union of all split groups")
        if (
            not isfinite(self.estimated_peak_training_memory_mb)
            or self.estimated_peak_training_memory_mb <= 0
        ):
            raise ValueError("estimated_peak_training_memory_mb must be finite and positive")
        if self.estimated_peak_training_memory_mb > self.config.max_training_memory_mb:
            raise ValueError("artifact exceeds its configured training memory budget")
        if not 0.0 <= self.allowed_missing_fraction <= 1.0:
            raise ValueError("allowed_missing_fraction must be between zero and one")
        if self.precise_predictions_enabled and not self.promotable:
            raise ValueError("a non-promotable artifact cannot enable precise predictions")
        if self.promotable and self.promotion_block_reasons:
            raise ValueError("a promotable artifact cannot have promotion blockers")
        if self.promotable and not self.dataset_evidence.eligible_for_promotion:
            raise ValueError("a promotable artifact requires eligible dataset evidence")
        if self.promotable and (
            self.calibration.calibration_sample_count < self.config.min_calibration_rows
            or self.calibration.calibration_group_count < self.config.min_calibration_groups
            or self.calibration.test_coverage
            < self.calibration.nominal_coverage - self.config.max_interval_coverage_shortfall
        ):
            raise ValueError("a promotable artifact must pass calibration gates")
        if self.promotable and (
            self.grouped_test.development_group_overlap_count
            or self.grouped_test.test_group_count < self.config.min_confident_test_groups
            or self.grouped_test.outside_training_envelope_fraction
            > self.config.max_test_ood_fraction
        ):
            raise ValueError("a promotable artifact must pass grouped OOD gates")
        if self.promotable and (
            self.test_uncertainty.r2_lower < self.config.min_confident_r2
            or self.test_uncertainty.mape_percent_upper > self.config.max_confident_mape_percent
        ):
            raise ValueError("a promotable artifact must pass grouped confidence-interval gates")
        if self.actual_device not in {"cpu", "gpu", "cuda"}:
            raise ValueError("actual_device must be cpu, gpu, or cuda")
        if self.requested_device not in {"auto", "cpu", "gpu", "cuda"}:
            raise ValueError("requested_device must be auto, cpu, gpu, or cuda")
        if (
            self.actual_device != self.requested_device
            and self.requested_device != "auto"
            and not self.device_fallback_reason
        ):
            raise ValueError("device fallback must include a reason")
        if (
            self.requested_device == "auto"
            and self.actual_device != "cuda"
            and not self.device_fallback_reason
        ):
            raise ValueError("auto device fallback must include a reason")

    @property
    def test_metrics(self) -> RegressionMetrics:
        return self.evaluations["lightgbm"].test


@dataclass(frozen=True, slots=True)
class PerformanceTrainingResult:
    """Training output including the persisted model contract and split assignments."""

    artifact: PerformanceModelArtifact
    split_assignments: dict[str, str]

    def __post_init__(self) -> None:
        if not self.split_assignments:
            raise ValueError("split_assignments must not be empty")


@dataclass(frozen=True, slots=True)
class PerformanceEstimate:
    """A measured score, a precise prediction, or a deliberately relative-only score."""

    score: float | None
    relative_score: float
    basis: PredictionBasis
    confidence: ConfidenceLevel
    decision: PerformanceDecision
    model_version: str | None
    supporting_sources: tuple[str, ...] = ()
    reason: str | None = None
    lower_score: float | None = None
    upper_score: float | None = None

    def __post_init__(self) -> None:
        if self.score is not None and not isfinite(self.score):
            raise ValueError("score must be finite when present")
        if not isfinite(self.relative_score):
            raise ValueError("relative_score must be finite")
        if self.score is not None and self.score < 0 or self.relative_score < 0:
            raise ValueError("performance scores must be non-negative")
        if self.basis == "relative_only" and self.score is not None:
            raise ValueError("relative-only estimates cannot expose a precise score")
        if self.basis in {"observed", "predicted"} and self.score is None:
            raise ValueError(f"{self.basis} estimates must expose a score")
        if self.basis == "observed" and not self.supporting_sources:
            raise ValueError("observed estimates require at least one supporting source")
        if self.basis == "observed" and self.model_version is not None:
            raise ValueError("observed estimates must not claim a model version")
        if self.basis != "observed" and self.model_version is None:
            raise ValueError("model-derived estimates require a model version")
        if (self.lower_score is None) != (self.upper_score is None):
            raise ValueError("prediction interval bounds must be supplied together")
        if self.lower_score is not None and self.upper_score is not None:
            if not (isfinite(self.lower_score) and isfinite(self.upper_score)):
                raise ValueError("prediction interval bounds must be finite")
            if self.lower_score < 0 or self.lower_score > self.upper_score:
                raise ValueError("prediction interval bounds are invalid")
            if self.score is None or not self.lower_score <= self.score <= self.upper_score:
                raise ValueError("prediction interval must contain the precise score")
        if self.basis == "predicted" and self.lower_score is None:
            raise ValueError("predicted estimates require calibrated prediction intervals")
        if self.basis != "predicted" and self.lower_score is not None:
            raise ValueError("only predicted estimates may expose prediction intervals")
        if self.basis == "observed" and self.decision != "observed_benchmark":
            raise ValueError("observed estimates require the observed_benchmark decision")
        if self.basis == "predicted" and self.decision != "precise_model_prediction":
            raise ValueError("precise predictions require the precise_model_prediction decision")
        if self.basis == "relative_only" and self.decision not in {
            "model_not_promotion_eligible",
            "input_outside_training_contract",
            "model_not_promotion_eligible_and_input_outside_training_contract",
            "precise_predictions_disabled",
            "precise_predictions_disabled_and_input_outside_training_contract",
        }:
            raise ValueError("relative-only estimates require a bounded fallback decision")
