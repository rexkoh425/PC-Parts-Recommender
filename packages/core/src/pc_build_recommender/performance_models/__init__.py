"""Workload-performance training, evaluation, artifacts, and inference."""

from .artifacts import (
    DATASET_MANIFEST_FILENAME,
    TRAINING_EVIDENCE_FILENAME,
    TRAINING_REPORT_FILENAME,
    load_performance_artifact,
    save_performance_artifact,
    seal_performance_artifact,
)
from .contracts import (
    ARTIFACT_SCHEMA_VERSION,
    DatasetEvidence,
    FeatureProfile,
    GroupedTestDiagnostics,
    ModelEvaluation,
    PerformanceDecision,
    PerformanceEstimate,
    PerformanceModelArtifact,
    PerformanceModelConfig,
    PerformanceTrainingResult,
    PredictionIntervalCalibration,
    RegressionMetrics,
    RegressionUncertainty,
    TargetTransform,
)
from .data import (
    SYNTHETIC_DATASET_PROVENANCE,
    SYNTHETIC_GPU_FEATURE_COLUMNS,
    estimate_peak_training_memory_mb,
    make_synthetic_performance_dataset,
    performance_frame_sha256,
    split_performance_frame,
    validate_performance_frame,
)
from .evaluation import (
    calibrate_prediction_intervals,
    grouped_bootstrap_uncertainty,
    grouped_test_diagnostics,
)
from .inference import estimate_performance, observed_performance_estimate
from .registry import (
    ObservedPerformanceObservation,
    PerformanceModelRegistry,
    WorkloadModelSpec,
)
from .training import calculate_regression_metrics, train_performance_model

__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "DATASET_MANIFEST_FILENAME",
    "SYNTHETIC_DATASET_PROVENANCE",
    "SYNTHETIC_GPU_FEATURE_COLUMNS",
    "TRAINING_EVIDENCE_FILENAME",
    "TRAINING_REPORT_FILENAME",
    "DatasetEvidence",
    "FeatureProfile",
    "GroupedTestDiagnostics",
    "ModelEvaluation",
    "PerformanceDecision",
    "PerformanceEstimate",
    "PerformanceModelArtifact",
    "PerformanceModelConfig",
    "PerformanceTrainingResult",
    "PredictionIntervalCalibration",
    "RegressionMetrics",
    "RegressionUncertainty",
    "TargetTransform",
    "ObservedPerformanceObservation",
    "PerformanceModelRegistry",
    "WorkloadModelSpec",
    "calibrate_prediction_intervals",
    "calculate_regression_metrics",
    "estimate_performance",
    "estimate_peak_training_memory_mb",
    "grouped_bootstrap_uncertainty",
    "grouped_test_diagnostics",
    "load_performance_artifact",
    "make_synthetic_performance_dataset",
    "observed_performance_estimate",
    "performance_frame_sha256",
    "save_performance_artifact",
    "seal_performance_artifact",
    "split_performance_frame",
    "train_performance_model",
    "validate_performance_frame",
]
