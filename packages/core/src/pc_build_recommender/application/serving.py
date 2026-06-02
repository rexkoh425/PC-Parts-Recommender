"""Fail-closed evidence gates for production ML and retrieval composition."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from pc_build_recommender.performance_models import (
    ARTIFACT_SCHEMA_VERSION,
    PerformanceModelArtifact,
)
from pc_build_recommender.ranking import (
    ProductRanker,
    RankerArtifactIdentity,
    RankerPromotionDecision,
    RankerPromotionPolicy,
    load_ranker_promotion_decision,
)
from pc_build_recommender.retrieval import (
    COMPARISON_REPORT_SCHEMA_VERSION,
    PostgresHybridRetriever,
    ValidatedEmbeddingArtifact,
    load_ranking_comparison_report,
)
from pc_build_recommender.retrieval.embedding_index import MANIFEST_SCHEMA_VERSION


class ServingConfigurationError(RuntimeError):
    """Raised when production serving evidence is absent, stale, or unpromoted."""


def _require_sha256(value: str, name: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ServingConfigurationError(f"{name} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True, slots=True)
class EmbeddingReleaseExpectation:
    """Operator-reviewed exact identity for one pgvector serving release."""

    data_version: str
    index_version: str
    embedding_model: str
    encoder_revision: str
    encoder_fingerprint: str
    dataset_content_hash: str
    manifest_schema_version: str = MANIFEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in (
            "data_version",
            "index_version",
            "embedding_model",
            "encoder_revision",
            "manifest_schema_version",
        ):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must not be empty")
        _require_sha256(self.encoder_fingerprint, "encoder_fingerprint")
        _require_sha256(self.dataset_content_hash, "dataset_content_hash")


@dataclass(frozen=True, slots=True)
class ActiveServingModels:
    """Truthful, evidence-backed versions safe to expose in responses and logs."""

    catalog_data_version: str
    retrieval_model: str
    ranking_model: str
    performance_models: Mapping[str, str]
    embedding_index_version: str
    retrieval_report_sha256: str
    ranker_promotion_decision_sha256: str
    ranker_model_sha256: str
    ranker_metadata_sha256: str
    ranker_manifest_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "catalog_data_version",
            "retrieval_model",
            "ranking_model",
            "embedding_index_version",
        ):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must not be empty")
        versions = {str(key): str(value) for key, value in self.performance_models.items()}
        if not versions or any(not key or not value for key, value in versions.items()):
            raise ValueError("performance_models must contain non-empty route/version pairs")
        object.__setattr__(
            self,
            "performance_models",
            MappingProxyType(dict(sorted(versions.items()))),
        )
        _require_sha256(self.retrieval_report_sha256, "retrieval_report_sha256")
        _require_sha256(
            self.ranker_promotion_decision_sha256,
            "ranker_promotion_decision_sha256",
        )
        _require_sha256(self.ranker_model_sha256, "ranker_model_sha256")
        _require_sha256(self.ranker_metadata_sha256, "ranker_metadata_sha256")
        _require_sha256(self.ranker_manifest_sha256, "ranker_manifest_sha256")

    @property
    def performance_model_label(self) -> str:
        return (
            "promoted["
            + ",".join(f"{route}={version}" for route, version in self.performance_models.items())
            + "]"
        )


def _validate_embedding_release(
    artifact: ValidatedEmbeddingArtifact,
    retriever: PostgresHybridRetriever,
    expected: EmbeddingReleaseExpectation,
) -> None:
    manifest = artifact.manifest
    encoder = manifest.get("encoder")
    if not isinstance(encoder, Mapping):
        raise ServingConfigurationError("embedding manifest encoder must be an object")
    actual = {
        "manifest_schema_version": str(manifest.get("schema_version", "")),
        "data_version": artifact.data_version,
        "index_version": artifact.index_version,
        "embedding_model": artifact.embedding_model,
        "encoder_revision": str(encoder.get("model_revision", "")),
        "encoder_fingerprint": artifact.encoder_fingerprint,
        "dataset_content_hash": artifact.dataset_content_hash,
    }
    for name, expected_value in (
        ("manifest_schema_version", expected.manifest_schema_version),
        ("data_version", expected.data_version),
        ("index_version", expected.index_version),
        ("embedding_model", expected.embedding_model),
        ("encoder_revision", expected.encoder_revision),
        ("encoder_fingerprint", expected.encoder_fingerprint),
        ("dataset_content_hash", expected.dataset_content_hash),
    ):
        if actual[name] != expected_value:
            raise ServingConfigurationError(
                f"embedding {name} mismatch: expected {expected_value!r}, got {actual[name]!r}"
            )

    release = retriever.release
    for name in (
        "data_version",
        "index_version",
        "embedding_model",
        "encoder_fingerprint",
        "dataset_content_hash",
    ):
        if getattr(release, name) != actual[name]:
            raise ServingConfigurationError(
                f"PostgreSQL retriever {name} does not match the validated embedding artifact"
            )


def _load_passing_ranker_decision(path: str | Path) -> RankerPromotionDecision:
    try:
        decision = load_ranker_promotion_decision(path)
    except (FileNotFoundError, OSError, TypeError, ValueError) as error:
        raise ServingConfigurationError(f"invalid ranker promotion decision: {error}") from error
    if not decision.passed:
        raise ServingConfigurationError("ranker promotion decision did not pass")
    if decision.failures:
        raise ServingConfigurationError("passing ranker decision contains failures")
    return decision


def validate_promoted_serving_models(
    *,
    catalog_data_version: str,
    expected_catalog_data_version: str,
    retriever: PostgresHybridRetriever,
    embedding_artifact: ValidatedEmbeddingArtifact,
    embedding_expectation: EmbeddingReleaseExpectation,
    retrieval_comparison_report_path: str | Path,
    retrieval_evaluation_model: str,
    ranker: ProductRanker,
    expected_ranker_version: str,
    ranker_promotion_decision_path: str | Path,
    expected_ranker_promotion_decision_sha256: str,
    expected_ranker_promotion_policy: RankerPromotionPolicy,
    performance_artifacts: Sequence[PerformanceModelArtifact],
    expected_performance_versions: Mapping[str, str],
) -> ActiveServingModels:
    """Validate every serving component against reviewed immutable evidence."""

    if catalog_data_version != expected_catalog_data_version:
        raise ServingConfigurationError(
            "catalog data version does not match the reviewed production configuration"
        )
    _validate_embedding_release(embedding_artifact, retriever, embedding_expectation)

    report = load_ranking_comparison_report(retrieval_comparison_report_path)
    if report.get("schema_version") != COMPARISON_REPORT_SCHEMA_VERSION:
        raise ServingConfigurationError("unsupported retrieval comparison report schema")
    dataset = report.get("dataset")
    if not isinstance(dataset, Mapping):
        raise ServingConfigurationError("retrieval report dataset evidence is missing")
    if (
        report.get("eligible_for_promotion") is not True
        or dataset.get("label_source") != "human"
        or dataset.get("split_name") != "test"
    ):
        raise ServingConfigurationError(
            "retrieval evidence is not an adjudicated human frozen-test promotion result"
        )
    models = report.get("models")
    if not isinstance(models, Mapping) or retrieval_evaluation_model not in models:
        raise ServingConfigurationError(
            "configured retrieval model is absent from the promotion report"
        )
    selected_evaluation = models[retrieval_evaluation_model]
    if (
        not isinstance(selected_evaluation, Mapping)
        or selected_evaluation.get("eligible_for_promotion") is not True
    ):
        raise ServingConfigurationError("configured retrieval model is not promotion-eligible")

    metadata = ranker.metadata
    if metadata.ranker_version != expected_ranker_version:
        raise ServingConfigurationError("ranker version does not match production configuration")
    if not metadata.promotion_eligible or metadata.ranking_basis != "lightgbm_lambdamart":
        raise ServingConfigurationError("configured ranker is not a promoted LambdaMART model")
    decision = _load_passing_ranker_decision(ranker_promotion_decision_path)
    expected_decision_sha256 = _require_sha256(
        expected_ranker_promotion_decision_sha256,
        "expected_ranker_promotion_decision_sha256",
    )
    if decision.decision_sha256 != expected_decision_sha256:
        raise ServingConfigurationError(
            "ranker promotion decision does not match the operator-pinned digest"
        )
    if decision.policy != expected_ranker_promotion_policy:
        raise ServingConfigurationError(
            "ranker promotion policy does not match the operator-reviewed policy"
        )
    if decision.ranker_version != metadata.ranker_version:
        raise ServingConfigurationError("ranker decision refers to a different model version")
    if decision.comparison_report_sha256 != report.get("report_sha256"):
        raise ServingConfigurationError(
            "ranker decision and retrieval comparison report hashes do not match"
        )
    try:
        artifact_identity = ranker.artifact_identity
    except RuntimeError as error:
        raise ServingConfigurationError("ranker has no verified artifact identity") from error
    if not isinstance(artifact_identity, RankerArtifactIdentity):
        raise ServingConfigurationError("ranker has no verified artifact identity")
    if metadata.model_sha256 != artifact_identity.model_sha256:
        raise ServingConfigurationError("ranker metadata model hash does not match its artifact")
    if (
        decision.ranker_model_sha256 != artifact_identity.model_sha256
        or decision.ranker_metadata_sha256 != artifact_identity.metadata_sha256
        or decision.ranker_manifest_sha256 != artifact_identity.manifest_sha256
    ):
        raise ServingConfigurationError("ranker decision refers to different artifact bytes")
    artifact_bindings = report.get("artifact_bound_rankings")
    selected_binding = (
        artifact_bindings.get(decision.challenger_model)
        if isinstance(artifact_bindings, Mapping)
        else None
    )
    if not isinstance(selected_binding, Mapping):
        raise ServingConfigurationError("retrieval report lacks challenger artifact binding")
    binding_identity = selected_binding.get("artifact_identity")
    if not isinstance(binding_identity, Mapping) or dict(binding_identity) != {
        "model_sha256": artifact_identity.model_sha256,
        "metadata_sha256": artifact_identity.metadata_sha256,
        "manifest_sha256": artifact_identity.manifest_sha256,
    }:
        raise ServingConfigurationError("retrieval report is bound to different artifact bytes")
    if selected_binding.get("ranker_metadata") != metadata.to_dict():
        raise ServingConfigurationError("retrieval report ranker metadata does not match artifact")
    feature_contract = selected_binding.get("feature_contract")
    if not isinstance(feature_contract, Mapping):
        raise ServingConfigurationError("retrieval report feature binding is missing")
    binding_measured_values = {
        "artifact_binding_sha256": selected_binding.get("evidence_sha256"),
        "metadata_payload_sha256": selected_binding.get("metadata_payload_sha256"),
        "feature_snapshot_sha256": feature_contract.get("snapshot_sha256"),
        "candidate_snapshot_sha256": selected_binding.get("candidate_snapshot_sha256"),
        "score_snapshot_sha256": selected_binding.get("score_snapshot_sha256"),
        "ranking_sha256": selected_binding.get("ranking_sha256"),
    }
    if any(
        decision.measured_values.get(name) != value
        for name, value in binding_measured_values.items()
    ):
        raise ServingConfigurationError(
            "ranker decision does not match the report artifact evaluation binding"
        )

    actual_performance: dict[str, str] = {}
    for artifact in performance_artifacts:
        if artifact.schema_version != ARTIFACT_SCHEMA_VERSION:
            raise ServingConfigurationError("unsupported performance artifact schema")
        if not artifact.promotable:
            reasons = "; ".join(artifact.promotion_block_reasons) or "not promotable"
            raise ServingConfigurationError(
                f"performance artifact is not promotion-eligible: {reasons}"
            )
        route = f"{artifact.config.category}/{artifact.config.workload}"
        if route in actual_performance:
            raise ServingConfigurationError(f"duplicate performance model route: {route}")
        actual_performance[route] = artifact.model_version
    expected_performance = {
        str(route): str(version) for route, version in expected_performance_versions.items()
    }
    if not actual_performance or actual_performance != expected_performance:
        raise ServingConfigurationError(
            "loaded performance model routes/versions do not exactly match production config"
        )

    report_hash = str(report["report_sha256"])
    decision_hash = decision.decision_sha256
    return ActiveServingModels(
        catalog_data_version=catalog_data_version,
        retrieval_model=retriever.retrieval_model_version,
        ranking_model=metadata.ranker_version,
        performance_models=actual_performance,
        embedding_index_version=retriever.release.index_version,
        retrieval_report_sha256=_require_sha256(report_hash, "retrieval_report_sha256"),
        ranker_promotion_decision_sha256=_require_sha256(
            decision_hash, "ranker_promotion_decision_sha256"
        ),
        ranker_model_sha256=artifact_identity.model_sha256,
        ranker_metadata_sha256=artifact_identity.metadata_sha256,
        ranker_manifest_sha256=artifact_identity.manifest_sha256,
    )
