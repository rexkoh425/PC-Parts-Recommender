"""Load one immutable, content-addressed production serving release."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session, sessionmaker

from pc_build_recommender.application import (
    ActiveServingModels,
    EmbeddingReleaseExpectation,
    ServingConfigurationError,
    validate_promoted_serving_models,
)
from pc_build_recommender.catalog import ProductionCatalogPolicy
from pc_build_recommender.entity_resolution import (
    EntityResolutionPolicy,
    EntityResolutionRelease,
    EntityResolutionRuntime,
    load_entity_resolution_release,
)
from pc_build_recommender.evaluation.manifest import sha256_file, sha256_json
from pc_build_recommender.performance_models import (
    PerformanceModelArtifact,
    load_performance_artifact,
)
from pc_build_recommender.ranking import (
    LambdaMARTRanker,
    ProductRanker,
    RankerPromotionPolicy,
    ranker_artifact_manifest_path,
)
from pc_build_recommender.retrieval import (
    PostgresHybridRetriever,
    SentenceTransformerEmbeddingEncoder,
    ValidatedEncoderBundle,
    bm25_index_from_embedding_artifact,
    validate_embedding_artifact,
    validate_encoder_bundle,
)

SERVING_RELEASE_SCHEMA_VERSION = "pc-build-recommender.serving-release.v3"
_SHA256_LENGTH = 64
_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema_version",
        "catalog_data_version",
        "catalog",
        "catalog_inputs",
        "embedding",
        "retrieval",
        "entity_resolution",
        "ranker",
        "ranker_promotion",
        "performance",
        "content_sha256",
    }
)


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _SHA256_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _reject_nonfinite(value: str) -> None:
    raise ValueError(f"non-finite JSON value is forbidden: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key is forbidden: {key!r}")
        result[key] = value
    return result


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ServingConfigurationError(f"{field} must be an object")
    return value


def _exact_fields(
    value: object,
    *,
    field: str,
    expected: frozenset[str],
) -> Mapping[str, Any]:
    result = _mapping(value, field=field)
    actual = set(result)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ServingConfigurationError(
            f"{field} fields do not match the serving contract; missing={missing}, extra={extra}"
        )
    return result


def _string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ServingConfigurationError(f"{field} must be a non-empty string")
    return value.strip()


def _sha256(value: object, *, field: str) -> str:
    if not _is_sha256(value):
        raise ServingConfigurationError(f"{field} must be a lowercase SHA-256 digest")
    return str(value)


def _positive_integer(value: object, *, field: str, maximum: int) -> int:
    if type(value) is not int or not 1 <= value <= maximum:
        raise ServingConfigurationError(f"{field} must be an integer from 1 to {maximum}")
    return value


@dataclass(frozen=True, slots=True)
class _ContentDigest:
    size_bytes: int
    sha256: str

    @classmethod
    def parse(cls, value: object, *, field: str) -> _ContentDigest:
        payload = _exact_fields(
            value,
            field=field,
            expected=frozenset({"size_bytes", "sha256"}),
        )
        size = payload["size_bytes"]
        if type(size) is not int or size < 0:
            raise ServingConfigurationError(f"{field}.size_bytes must be non-negative")
        return cls(size_bytes=size, sha256=_sha256(payload["sha256"], field=f"{field}.sha256"))

    def verify(self, path: Path, *, field: str) -> None:
        if not path.is_file():
            raise ServingConfigurationError(f"{field} does not exist: {path}")
        if path.stat().st_size != self.size_bytes:
            raise ServingConfigurationError(f"{field} size does not match serving manifest")
        if sha256_file(path) != self.sha256:
            raise ServingConfigurationError(f"{field} SHA-256 does not match serving manifest")


@dataclass(frozen=True, slots=True)
class _ArtifactReference:
    path: str
    digest: _ContentDigest

    @classmethod
    def parse(cls, value: object, *, field: str) -> _ArtifactReference:
        payload = _exact_fields(
            value,
            field=field,
            expected=frozenset({"path", "size_bytes", "sha256"}),
        )
        path = _string(payload["path"], field=f"{field}.path")
        candidate = Path(path)
        if candidate.is_absolute() or any(part == ".." for part in candidate.parts):
            raise ServingConfigurationError(f"{field}.path must be relative and confined")
        return cls(
            path=path,
            digest=_ContentDigest(
                size_bytes=_positive_integer_or_zero(
                    payload["size_bytes"], field=f"{field}.size_bytes"
                ),
                sha256=_sha256(payload["sha256"], field=f"{field}.sha256"),
            ),
        )

    def resolve(self, root: Path, *, field: str) -> Path:
        candidate = (root / self.path).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as error:
            raise ServingConfigurationError(f"{field}.path escapes the serving release") from error
        self.digest.verify(candidate, field=field)
        return candidate


@dataclass(frozen=True, slots=True)
class _EncoderBundleReference:
    path: str
    sha256: str
    file_count: int
    size_bytes: int

    @classmethod
    def parse(cls, value: object, *, field: str) -> _EncoderBundleReference:
        payload = _exact_fields(
            value,
            field=field,
            expected=frozenset({"path", "sha256", "file_count", "size_bytes"}),
        )
        path = _string(payload["path"], field=f"{field}.path")
        candidate = Path(path)
        if candidate.is_absolute() or any(part == ".." for part in candidate.parts):
            raise ServingConfigurationError(f"{field}.path must be relative and confined")
        return cls(
            path=path,
            sha256=_sha256(payload["sha256"], field=f"{field}.sha256"),
            file_count=_positive_integer(
                payload["file_count"], field=f"{field}.file_count", maximum=4096
            ),
            size_bytes=_positive_integer(
                payload["size_bytes"],
                field=f"{field}.size_bytes",
                maximum=2 * 1024 * 1024 * 1024,
            ),
        )

    def unresolved_path(self, root: Path, *, field: str) -> Path:
        candidate = Path(os.path.abspath(root / self.path))
        try:
            candidate.relative_to(root)
        except ValueError as error:
            raise ServingConfigurationError(f"{field}.path escapes the serving release") from error
        return candidate


def _positive_integer_or_zero(value: object, *, field: str) -> int:
    if type(value) is not int or value < 0:
        raise ServingConfigurationError(f"{field} must be a non-negative integer")
    return value


@dataclass(frozen=True, slots=True)
class ProductionCatalogRelease:
    """Exact catalogue inputs and authorized ER runtime from one pinned release."""

    manifest_path: Path
    manifest_sha256: str
    manifest: Mapping[str, Any]
    catalog_path: Path
    offers_path: Path
    reviewed_mappings_path: Path
    review_evidence_path: Path
    entity_resolution: EntityResolutionRelease
    entity_resolution_evaluation_path: Path
    entity_resolution_policy_path: Path
    entity_resolution_rights_path: Path

    @property
    def entity_resolution_runtime(self) -> EntityResolutionRuntime:
        return self.entity_resolution.runtime

    @property
    def entity_resolution_policy(self) -> EntityResolutionPolicy:
        return self.entity_resolution.policy


@dataclass(frozen=True, slots=True)
class ProductionServingRelease:
    """Fully loaded serving components validated against reviewed evidence."""

    manifest_path: Path
    manifest_sha256: str
    retriever: PostgresHybridRetriever
    ranker: ProductRanker
    performance_artifacts: tuple[PerformanceModelArtifact, ...]
    active_models: ActiveServingModels
    catalog_release: ProductionCatalogRelease
    encoder_bundle: ValidatedEncoderBundle
    semantic_encoder_ready: bool


def production_catalog_policy_from_entity_resolution(
    policy: EntityResolutionPolicy,
) -> ProductionCatalogPolicy:
    """Project the exact ER release policy into the catalogue readiness contract."""

    return ProductionCatalogPolicy(**policy.production_catalog_policy_kwargs())


def _read_manifest(path: Path, *, expected_content_sha256: str) -> Mapping[str, Any]:
    if not path.is_file():
        raise ServingConfigurationError(f"serving manifest does not exist: {path}")
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_nonfinite,
            object_pairs_hook=_unique_object,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ServingConfigurationError(f"serving manifest is invalid JSON: {error}") from error
    manifest = _exact_fields(payload, field="manifest", expected=_TOP_LEVEL_FIELDS)
    if manifest["schema_version"] != SERVING_RELEASE_SCHEMA_VERSION:
        raise ServingConfigurationError("unsupported serving manifest schema")
    stored_hash = _sha256(manifest["content_sha256"], field="manifest.content_sha256")
    unhashed = dict(manifest)
    unhashed.pop("content_sha256")
    if sha256_json(unhashed) != stored_hash:
        raise ServingConfigurationError("serving manifest content hash verification failed")
    expected_hash = _sha256(
        expected_content_sha256,
        field="expected serving manifest SHA-256",
    )
    if stored_hash != expected_hash:
        raise ServingConfigurationError(
            "serving manifest does not match the operator-pinned SHA-256"
        )
    return manifest


def _release_reference(
    value: object,
    *,
    field: str,
    root: Path,
    expected_name: str | None = None,
) -> Path:
    reference = _ArtifactReference.parse(value, field=field)
    path = reference.resolve(root, field=field)
    if expected_name is not None and path.name != expected_name:
        raise ServingConfigurationError(f"{field}.path must end with {expected_name!r}")
    return path


def _catalog_release_from_manifest(
    manifest: Mapping[str, Any],
    *,
    manifest_path: Path,
    catalog_path: Path,
    offers_path: Path,
    reviewed_mappings_path: Path,
    review_evidence_path: Path,
    expected_catalog_data_version: str | None,
) -> ProductionCatalogRelease:
    root = manifest_path.parent.resolve()
    manifest_catalog_version = _string(
        manifest["catalog_data_version"], field="manifest.catalog_data_version"
    )
    if (
        expected_catalog_data_version is not None
        and manifest_catalog_version != expected_catalog_data_version
    ):
        raise ServingConfigurationError(
            "serving manifest catalogue version does not match runtime configuration"
        )

    _ContentDigest.parse(manifest["catalog"], field="manifest.catalog").verify(
        catalog_path,
        field="catalog artifact",
    )
    catalog_inputs = _exact_fields(
        manifest["catalog_inputs"],
        field="manifest.catalog_inputs",
        expected=frozenset({"offers", "reviewed_mappings", "review_evidence"}),
    )
    _ContentDigest.parse(catalog_inputs["offers"], field="manifest.catalog_inputs.offers").verify(
        offers_path, field="governed offers artifact"
    )
    _ContentDigest.parse(
        catalog_inputs["reviewed_mappings"],
        field="manifest.catalog_inputs.reviewed_mappings",
    ).verify(reviewed_mappings_path, field="reviewed mappings artifact")
    _ContentDigest.parse(
        catalog_inputs["review_evidence"],
        field="manifest.catalog_inputs.review_evidence",
    ).verify(review_evidence_path, field="review evidence artifact")

    er = _exact_fields(
        manifest["entity_resolution"],
        field="manifest.entity_resolution",
        expected=frozenset(
            {
                "metadata",
                "model",
                "serving_evidence",
                "evaluation",
                "policy",
                "rights",
                "model_version",
                "artifact_core_sha256",
                "model_file_sha256",
                "metadata_sha256",
                "calibrator_sha256",
                "serving_evidence_sha256",
                "model_release_sha256",
                "evaluation_sha256",
                "policy_sha256",
                "rights_sha256",
                "binding_sha256",
            }
        ),
    )
    metadata_path = _release_reference(
        er["metadata"],
        field="manifest.entity_resolution.metadata",
        root=root,
        expected_name="metadata.json",
    )
    model_path = _release_reference(
        er["model"],
        field="manifest.entity_resolution.model",
        root=root,
        expected_name="model.txt",
    )
    evidence_path = _release_reference(
        er["serving_evidence"],
        field="manifest.entity_resolution.serving_evidence",
        root=root,
        expected_name="serving_evidence.json",
    )
    if len({metadata_path.parent, model_path.parent, evidence_path.parent}) != 1:
        raise ServingConfigurationError(
            "entity-resolution model, metadata, and serving evidence must share one artifact dir"
        )
    evaluation_path = _release_reference(
        er["evaluation"],
        field="manifest.entity_resolution.evaluation",
        root=root,
    )
    policy_path = _release_reference(
        er["policy"],
        field="manifest.entity_resolution.policy",
        root=root,
    )
    rights_path = _release_reference(
        er["rights"],
        field="manifest.entity_resolution.rights",
        root=root,
    )
    try:
        entity_release = load_entity_resolution_release(
            metadata_path.parent,
            evaluation_path,
            policy_path,
            rights_path,
        )
    except Exception as error:
        raise ServingConfigurationError(
            f"entity-resolution release failed validation: {error}"
        ) from error
    identity = entity_release.identity
    expected_identity = {
        "artifact_core_sha256": identity.artifact_core_sha256,
        "model_file_sha256": identity.model_file_sha256,
        "metadata_sha256": identity.metadata_sha256,
        "calibrator_sha256": identity.calibrator_sha256,
        "serving_evidence_sha256": identity.serving_evidence_sha256,
        "model_release_sha256": identity.model_release_sha256,
        "evaluation_sha256": identity.evaluation_sha256,
        "policy_sha256": identity.policy_sha256,
        "rights_sha256": identity.rights_sha256,
        "binding_sha256": identity.binding_sha256,
    }
    for field_name, actual_digest in expected_identity.items():
        configured_digest = _sha256(
            er[field_name], field=f"manifest.entity_resolution.{field_name}"
        )
        if configured_digest != actual_digest:
            raise ServingConfigurationError(
                f"manifest.entity_resolution.{field_name} does not match loaded release"
            )
    model_version = _string(er["model_version"], field="manifest.entity_resolution.model_version")
    if model_version != entity_release.runtime.model_version:
        raise ServingConfigurationError(
            "manifest.entity_resolution.model_version does not match loaded release"
        )
    return ProductionCatalogRelease(
        manifest_path=manifest_path,
        manifest_sha256=_sha256(manifest["content_sha256"], field="manifest.content_sha256"),
        manifest=manifest,
        catalog_path=catalog_path,
        offers_path=offers_path,
        reviewed_mappings_path=reviewed_mappings_path,
        review_evidence_path=review_evidence_path,
        entity_resolution=entity_release,
        entity_resolution_evaluation_path=evaluation_path,
        entity_resolution_policy_path=policy_path,
        entity_resolution_rights_path=rights_path,
    )


def load_production_catalog_release(
    manifest_path: str | Path,
    *,
    catalog_path: str | Path,
    offers_path: str | Path,
    reviewed_mappings_path: str | Path | None,
    review_evidence_path: str | Path | None,
    expected_catalog_data_version: str | None,
    expected_manifest_sha256: str,
) -> ProductionCatalogRelease:
    """Load only catalogue/ER authority for release import and API bootstrap parity."""

    if reviewed_mappings_path is None:
        raise ServingConfigurationError(
            "production serving requires an exact reviewed-mappings artifact"
        )
    if review_evidence_path is None:
        raise ServingConfigurationError(
            "production serving requires an exact review-evidence artifact"
        )
    resolved_manifest = Path(manifest_path).resolve()
    manifest = _read_manifest(
        resolved_manifest,
        expected_content_sha256=expected_manifest_sha256,
    )
    try:
        return _catalog_release_from_manifest(
            manifest,
            manifest_path=resolved_manifest,
            catalog_path=Path(catalog_path).resolve(),
            offers_path=Path(offers_path).resolve(),
            reviewed_mappings_path=Path(reviewed_mappings_path).resolve(),
            review_evidence_path=Path(review_evidence_path).resolve(),
            expected_catalog_data_version=expected_catalog_data_version,
        )
    except ServingConfigurationError:
        raise
    except Exception as error:
        raise ServingConfigurationError(
            f"production catalogue release failed validation: {error}"
        ) from error


def _load_release(
    *,
    manifest_path: Path,
    catalog_path: Path,
    offers_path: Path,
    reviewed_mappings_path: Path,
    review_evidence_path: Path,
    session_factory: sessionmaker[Session],
    expected_catalog_data_version: str,
    expected_ranker_version: str,
    expected_manifest_sha256: str,
    expected_encoder_bundle_path: Path,
    expected_encoder_bundle_sha256: str,
) -> ProductionServingRelease:
    manifest = _read_manifest(
        manifest_path,
        expected_content_sha256=expected_manifest_sha256,
    )
    root = manifest_path.parent.resolve()
    catalog_release = _catalog_release_from_manifest(
        manifest,
        manifest_path=manifest_path,
        catalog_path=catalog_path,
        offers_path=offers_path,
        reviewed_mappings_path=reviewed_mappings_path,
        review_evidence_path=review_evidence_path,
        expected_catalog_data_version=expected_catalog_data_version,
    )
    manifest_catalog_version = _string(
        manifest["catalog_data_version"], field="manifest.catalog_data_version"
    )

    embedding = _exact_fields(
        manifest["embedding"],
        field="manifest.embedding",
        expected=frozenset(
            {
                "artifact_manifest",
                "data_version",
                "index_version",
                "embedding_model",
                "encoder_revision",
                "encoder_fingerprint",
                "dataset_content_hash",
                "manifest_schema_version",
                "device",
                "batch_size",
                "rrf_k",
                "encoder_bundle",
            }
        ),
    )
    embedding_manifest_path = _release_reference(
        embedding["artifact_manifest"],
        field="manifest.embedding.artifact_manifest",
        root=root,
        expected_name="manifest.json",
    )
    encoder_bundle_reference = _EncoderBundleReference.parse(
        embedding["encoder_bundle"], field="manifest.embedding.encoder_bundle"
    )
    operator_bundle_sha256 = _sha256(
        expected_encoder_bundle_sha256,
        field="expected semantic encoder bundle SHA-256",
    )
    if encoder_bundle_reference.sha256 != operator_bundle_sha256:
        raise ServingConfigurationError(
            "semantic encoder bundle does not match the operator-pinned SHA-256"
        )
    encoder_bundle_path = encoder_bundle_reference.unresolved_path(
        root, field="manifest.embedding.encoder_bundle"
    )
    operator_bundle_path = Path(os.path.abspath(expected_encoder_bundle_path))
    if encoder_bundle_path != operator_bundle_path:
        raise ServingConfigurationError(
            "semantic encoder bundle path does not match the operator-pinned path"
        )
    encoder_bundle = validate_encoder_bundle(
        encoder_bundle_path,
        expected_sha256=encoder_bundle_reference.sha256,
        expected_file_count=encoder_bundle_reference.file_count,
        expected_size_bytes=encoder_bundle_reference.size_bytes,
    )
    embedding_artifact = validate_embedding_artifact(
        catalog_path,
        embedding_manifest_path.parent,
    )
    model_name = _string(embedding["embedding_model"], field="embedding.embedding_model")
    revision = _string(embedding["encoder_revision"], field="embedding.encoder_revision")
    device = _string(embedding["device"], field="embedding.device").casefold()
    if device not in {"auto", "cpu", "cuda"}:
        raise ServingConfigurationError("embedding.device must be auto, cpu, or cuda")
    encoder = SentenceTransformerEmbeddingEncoder(
        model_name,
        revision=revision,
        device=device,
        batch_size=_positive_integer(
            embedding["batch_size"], field="embedding.batch_size", maximum=512
        ),
        model_path=encoder_bundle.path,
        local_files_only=True,
    )
    encoder.warmup(expected_dimension=int(embedding_artifact.vectors.shape[1]))
    embedding_expectation = EmbeddingReleaseExpectation(
        data_version=_string(embedding["data_version"], field="embedding.data_version"),
        index_version=_string(embedding["index_version"], field="embedding.index_version"),
        embedding_model=model_name,
        encoder_revision=revision,
        encoder_fingerprint=_sha256(
            embedding["encoder_fingerprint"], field="embedding.encoder_fingerprint"
        ),
        dataset_content_hash=_sha256(
            embedding["dataset_content_hash"], field="embedding.dataset_content_hash"
        ),
        manifest_schema_version=_string(
            embedding["manifest_schema_version"], field="embedding.manifest_schema_version"
        ),
    )
    retriever = PostgresHybridRetriever(
        session_factory,
        encoder=encoder,
        data_version=embedding_expectation.data_version,
        index_version=embedding_expectation.index_version,
        encoder_fingerprint=embedding_expectation.encoder_fingerprint,
        dataset_content_hash=embedding_expectation.dataset_content_hash,
        embedding_model=embedding_expectation.embedding_model,
        rrf_k=_positive_integer(embedding["rrf_k"], field="embedding.rrf_k", maximum=1000),
        bm25_index=bm25_index_from_embedding_artifact(embedding_artifact),
    )

    retrieval = _exact_fields(
        manifest["retrieval"],
        field="manifest.retrieval",
        expected=frozenset({"comparison_report", "evaluation_model"}),
    )
    comparison_report_path = _release_reference(
        retrieval["comparison_report"],
        field="manifest.retrieval.comparison_report",
        root=root,
    )
    retrieval_evaluation_model = _string(
        retrieval["evaluation_model"], field="manifest.retrieval.evaluation_model"
    )

    ranker_config = _exact_fields(
        manifest["ranker"],
        field="manifest.ranker",
        expected=frozenset({"model", "artifact_manifest", "ranker_version"}),
    )
    model_path = _release_reference(
        ranker_config["model"], field="manifest.ranker.model", root=root
    )
    ranker_manifest_path = _release_reference(
        ranker_config["artifact_manifest"],
        field="manifest.ranker.artifact_manifest",
        root=root,
    )
    if ranker_manifest_path != ranker_artifact_manifest_path(model_path).resolve():
        raise ServingConfigurationError(
            "ranker artifact manifest reference does not belong to the configured model"
        )
    manifest_ranker_version = _string(
        ranker_config["ranker_version"], field="manifest.ranker.ranker_version"
    )
    if manifest_ranker_version != expected_ranker_version:
        raise ServingConfigurationError(
            "serving manifest ranker version does not match runtime configuration"
        )
    ranker = LambdaMARTRanker.load(model_path)

    promotion = _exact_fields(
        manifest["ranker_promotion"],
        field="manifest.ranker_promotion",
        expected=frozenset({"decision", "decision_sha256", "policy"}),
    )
    promotion_decision_path = _release_reference(
        promotion["decision"],
        field="manifest.ranker_promotion.decision",
        root=root,
    )
    expected_decision_sha256 = _sha256(
        promotion["decision_sha256"],
        field="manifest.ranker_promotion.decision_sha256",
    )
    try:
        expected_promotion_policy = RankerPromotionPolicy.from_dict(
            _mapping(promotion["policy"], field="manifest.ranker_promotion.policy")
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ServingConfigurationError(
            f"manifest.ranker_promotion.policy is invalid: {error}"
        ) from error

    performance_config = manifest["performance"]
    if not isinstance(performance_config, list) or not performance_config:
        raise ServingConfigurationError("manifest.performance must be a non-empty array")
    artifacts: list[PerformanceModelArtifact] = []
    expected_performance_versions: dict[str, str] = {}
    for index, raw_entry in enumerate(performance_config):
        field = f"manifest.performance[{index}]"
        entry = _exact_fields(
            raw_entry,
            field=field,
            expected=frozenset({"artifact_manifest", "route", "model_version"}),
        )
        artifact_manifest_path = _release_reference(
            entry["artifact_manifest"],
            field=f"{field}.artifact_manifest",
            root=root,
            expected_name="artifact_manifest.json",
        )
        artifact = load_performance_artifact(artifact_manifest_path.parent)
        route = _string(entry["route"], field=f"{field}.route")
        version = _string(entry["model_version"], field=f"{field}.model_version")
        actual_route = f"{artifact.config.category}/{artifact.config.workload}"
        if route != actual_route or version != artifact.model_version:
            raise ServingConfigurationError(
                f"{field} route/version does not match the loaded performance artifact"
            )
        if route in expected_performance_versions:
            raise ServingConfigurationError(f"duplicate performance route in manifest: {route}")
        expected_performance_versions[route] = version
        artifacts.append(artifact)

    active_models = validate_promoted_serving_models(
        catalog_data_version=manifest_catalog_version,
        expected_catalog_data_version=expected_catalog_data_version,
        retriever=retriever,
        embedding_artifact=embedding_artifact,
        embedding_expectation=embedding_expectation,
        retrieval_comparison_report_path=comparison_report_path,
        retrieval_evaluation_model=retrieval_evaluation_model,
        ranker=ranker,
        expected_ranker_version=expected_ranker_version,
        ranker_promotion_decision_path=promotion_decision_path,
        expected_ranker_promotion_decision_sha256=expected_decision_sha256,
        expected_ranker_promotion_policy=expected_promotion_policy,
        performance_artifacts=tuple(artifacts),
        expected_performance_versions=expected_performance_versions,
    )
    return ProductionServingRelease(
        manifest_path=manifest_path,
        manifest_sha256=_sha256(manifest["content_sha256"], field="manifest.content_sha256"),
        retriever=retriever,
        ranker=ranker,
        performance_artifacts=tuple(artifacts),
        active_models=active_models,
        catalog_release=catalog_release,
        encoder_bundle=encoder_bundle,
        semantic_encoder_ready=True,
    )


def load_production_serving_release(
    manifest_path: str | Path,
    *,
    catalog_path: str | Path,
    offers_path: str | Path,
    reviewed_mappings_path: str | Path,
    review_evidence_path: str | Path,
    session_factory: sessionmaker[Session],
    expected_catalog_data_version: str,
    expected_ranker_version: str,
    expected_manifest_sha256: str,
    expected_encoder_bundle_path: str | Path,
    expected_encoder_bundle_sha256: str,
) -> ProductionServingRelease:
    """Verify and load the exact production release, or abort startup."""

    resolved_manifest = Path(manifest_path).resolve()
    resolved_catalog = Path(catalog_path).resolve()
    resolved_offers = Path(offers_path).resolve()
    resolved_reviewed_mappings = Path(reviewed_mappings_path).resolve()
    resolved_review_evidence = Path(review_evidence_path).resolve()
    try:
        return _load_release(
            manifest_path=resolved_manifest,
            catalog_path=resolved_catalog,
            offers_path=resolved_offers,
            reviewed_mappings_path=resolved_reviewed_mappings,
            review_evidence_path=resolved_review_evidence,
            session_factory=session_factory,
            expected_catalog_data_version=expected_catalog_data_version,
            expected_ranker_version=expected_ranker_version,
            expected_manifest_sha256=expected_manifest_sha256,
            expected_encoder_bundle_path=Path(expected_encoder_bundle_path),
            expected_encoder_bundle_sha256=expected_encoder_bundle_sha256,
        )
    except ServingConfigurationError:
        raise
    except Exception as error:
        raise ServingConfigurationError(
            f"production serving release failed validation: {error}"
        ) from error
