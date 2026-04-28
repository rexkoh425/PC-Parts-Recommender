"""Reproducible evaluation contracts for recommendation and ML models."""

from .artifacts import (
    ARTIFACT_SCHEMA_VERSION,
    build_evaluation_artifact,
    load_evaluation_artifact,
    verify_evaluation_artifact,
    write_evaluation_artifact,
)
from .contracts import (
    DataUseDeclaration,
    EvaluationResult,
    MetricEstimate,
    SyntheticDataError,
)
from .manifest import (
    MANIFEST_SCHEMA_VERSION,
    DatasetManifest,
    FileDigest,
    build_dataset_manifest,
    canonical_json_bytes,
    load_dataset_manifest,
    sha256_file,
    sha256_json,
    verify_dataset_manifest,
    write_dataset_manifest,
)
from .metrics import (
    bootstrap_confidence_interval,
    evaluate_entity_resolution,
    evaluate_ranker_lift,
    evaluate_regression,
    evaluate_retrieval,
    wilson_confidence_interval,
)
from .splits import (
    DEFAULT_SPLIT_WEIGHTS,
    GroupSplit,
    assert_group_disjoint,
    deterministic_group_split,
    split_indices,
)

__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "DEFAULT_SPLIT_WEIGHTS",
    "MANIFEST_SCHEMA_VERSION",
    "DataUseDeclaration",
    "DatasetManifest",
    "EvaluationResult",
    "FileDigest",
    "GroupSplit",
    "MetricEstimate",
    "SyntheticDataError",
    "assert_group_disjoint",
    "bootstrap_confidence_interval",
    "build_dataset_manifest",
    "build_evaluation_artifact",
    "canonical_json_bytes",
    "deterministic_group_split",
    "evaluate_entity_resolution",
    "evaluate_ranker_lift",
    "evaluate_regression",
    "evaluate_retrieval",
    "load_dataset_manifest",
    "load_evaluation_artifact",
    "sha256_file",
    "sha256_json",
    "split_indices",
    "verify_dataset_manifest",
    "verify_evaluation_artifact",
    "wilson_confidence_interval",
    "write_dataset_manifest",
    "write_evaluation_artifact",
]
