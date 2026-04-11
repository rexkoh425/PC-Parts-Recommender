"""Composition root for deterministic local, test, and demo applications."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from pc_build_recommender.compatibility import (
    AUTHORITATIVE_COMPATIBILITY_POLICY,
    CONTROLLED_NON_PRODUCTION_POLICY,
    CompatibilityEngine,
)
from pc_build_recommender.optimizer import BuildOptimizer
from pc_build_recommender.performance_models import PerformanceModelArtifact
from pc_build_recommender.ranking import HeuristicRanker, ProductRanker, RankerArtifactIdentity
from pc_build_recommender.retrieval import (
    HybridProductRetriever,
    ProductRetriever,
    VectorSearchBackend,
)

from .catalog import ApplicationCatalog, CatalogReader, CompatibilityEvidencePolicy
from .models import ApplicationVersions, CandidateLimits
from .performance import ArtifactPerformanceProvider
from .pipeline import CandidatePipeline
from .services import (
    GenerateBuildsService,
    ReplaceComponentService,
    SearchProductsService,
)
from .serving import ActiveServingModels
from .store import InMemoryResultStore, ResultStore


@dataclass(frozen=True, slots=True)
class ApplicationServices:
    """Use cases and shared serving state exposed to an API adapter."""

    catalog: ApplicationCatalog
    search_products: SearchProductsService
    generate_builds: GenerateBuildsService
    replace_component: ReplaceComponentService
    results: ResultStore
    versions: ApplicationVersions


def create_application_services(
    repository: CatalogReader,
    *,
    data_version: str | None = None,
    ranker: ProductRanker | None = None,
    vector_backend: VectorSearchBackend | None = None,
    retriever: ProductRetriever | None = None,
    retrieval_model_version: str | None = None,
    performance_model_version: str | None = None,
    performance_artifacts: Sequence[PerformanceModelArtifact] = (),
    allow_unpromoted_performance_models: bool = False,
    require_promoted_models: bool = False,
    promoted_serving_models: ActiveServingModels | None = None,
    compatibility_engine: CompatibilityEngine | None = None,
    optimizer: BuildOptimizer | None = None,
    result_store: ResultStore | None = None,
    candidate_limits: CandidateLimits | None = None,
    random_seed: int = 42,
    compatibility_evidence_policy: CompatibilityEvidencePolicy = AUTHORITATIVE_COMPATIBILITY_POLICY,
) -> ApplicationServices:
    """Build services from persisted canonical catalogue data.

    This function never manufactures demo products and defaults to authoritative-only
    compatibility evidence.  An empty repository raises
    :class:`EmptyCatalogError` from :meth:`ApplicationCatalog.from_repository`,
    making missing ingestion or seeding explicit to operators.
    """

    catalog = ApplicationCatalog.from_repository(
        repository,
        data_version=data_version,
        compatibility_evidence_policy=compatibility_evidence_policy,
    )
    if retriever is not None and vector_backend is not None:
        raise ValueError("retriever and vector_backend are mutually exclusive")
    active_retriever = retriever or HybridProductRetriever(
        catalog.documents, vector_backend=vector_backend
    )
    derived_retrieval_version = active_retriever.retrieval_model_version
    if retrieval_model_version is not None and retrieval_model_version != derived_retrieval_version:
        raise ValueError("retrieval_model_version does not match the configured runtime retriever")
    active_ranker = ranker or HeuristicRanker(ranker_version="heuristic-v1")
    performance_provider = (
        ArtifactPerformanceProvider(performance_artifacts) if performance_artifacts else None
    )
    if performance_provider is not None:
        performance_provider.validate_catalog(catalog)
    if allow_unpromoted_performance_models and require_promoted_models:
        raise RuntimeError("production serving cannot allow unpromoted performance models")

    if performance_provider is None:
        derived_performance_version = "observed-only-v1"
    elif require_promoted_models:
        if promoted_serving_models is None:
            raise RuntimeError("production serving requires validated promoted-serving evidence")
        if not performance_provider.all_promotable:
            reasons = "; ".join(performance_provider.promotion_block_reasons)
            raise RuntimeError(f"production performance models are not promotable: {reasons}")
        if dict(performance_provider.model_versions) != dict(
            promoted_serving_models.performance_models
        ):
            raise RuntimeError(
                "production performance model routes/versions do not match promoted-serving "
                "evidence"
            )
        derived_performance_version = promoted_serving_models.performance_model_label
    elif not performance_provider.all_promotable:
        if not allow_unpromoted_performance_models:
            reasons = "; ".join(performance_provider.promotion_block_reasons)
            raise RuntimeError(
                "unpromoted performance artifacts require explicit development opt-in: " + reasons
            )
        derived_performance_version = _performance_label(
            "development-relative", performance_provider.model_versions
        )
    elif promoted_serving_models is not None:
        if dict(performance_provider.model_versions) != dict(
            promoted_serving_models.performance_models
        ):
            raise RuntimeError(
                "performance model routes/versions do not match promoted-serving evidence"
            )
        derived_performance_version = _performance_label(
            "promotion-eligible", performance_provider.model_versions
        )
    else:
        derived_performance_version = _performance_label(
            "promotion-eligible", performance_provider.model_versions
        )

    if performance_provider is None and performance_model_version not in (None, "observed-only-v1"):
        raise RuntimeError("performance model activation requires a runtime inference provider")
    if performance_model_version is not None and (
        performance_model_version != derived_performance_version
    ):
        raise ValueError(
            "performance_model_version does not match the configured runtime inference provider"
        )
    if require_promoted_models:
        if promoted_serving_models is None:
            raise RuntimeError("production serving requires validated promoted-serving evidence")
        if retriever is None or ranker is None:
            raise RuntimeError(
                "production serving requires an explicitly configured retriever and ranker"
            )
        if not active_ranker.metadata.promotion_eligible:
            reasons = "; ".join(active_ranker.metadata.promotion_block_reasons) or "not promoted"
            raise RuntimeError(f"production ranker is not promotion-eligible: {reasons}")
        if catalog.data_version != promoted_serving_models.catalog_data_version:
            raise RuntimeError(
                "production catalog version does not match promoted-serving evidence"
            )
        if active_ranker.metadata.ranker_version != promoted_serving_models.ranking_model:
            raise RuntimeError("production ranker version does not match promoted-serving evidence")
        try:
            ranker_identity = active_ranker.artifact_identity
        except RuntimeError as error:
            raise RuntimeError("production ranker lacks verified artifact identity") from error
        if not isinstance(ranker_identity, RankerArtifactIdentity):
            raise RuntimeError("production ranker lacks verified artifact identity")
        expected_ranker_identity = (
            promoted_serving_models.ranker_model_sha256,
            promoted_serving_models.ranker_metadata_sha256,
            promoted_serving_models.ranker_manifest_sha256,
        )
        if (
            ranker_identity.model_sha256,
            ranker_identity.metadata_sha256,
            ranker_identity.manifest_sha256,
        ) != expected_ranker_identity:
            raise RuntimeError(
                "production ranker artifact identity does not match promoted-serving evidence"
            )
        if derived_retrieval_version != promoted_serving_models.retrieval_model:
            raise RuntimeError(
                "production retrieval version does not match promoted-serving evidence"
            )
        if performance_provider is None:
            raise RuntimeError("production serving requires promoted performance artifacts")
    active_compatibility = compatibility_engine or CompatibilityEngine()
    active_optimizer = optimizer or BuildOptimizer()
    store = result_store if result_store is not None else InMemoryResultStore()
    pipeline = CandidatePipeline(
        catalog,
        active_retriever,
        active_ranker,
        active_compatibility,
        candidate_limits=candidate_limits,
        performance_provider=performance_provider,
    )
    versions = ApplicationVersions(
        data_version=catalog.data_version,
        ranking_model=active_ranker.metadata.ranker_version,
        rule_version=active_compatibility.rule_version,
        optimizer_version="cp-sat-v1",
        retrieval_model=derived_retrieval_version,
        performance_model=derived_performance_version,
    )
    generator = GenerateBuildsService(
        catalog,
        pipeline,
        active_compatibility,
        optimizer=active_optimizer,
        store=store,
        versions=versions,
        random_seed=random_seed,
    )
    search = SearchProductsService(catalog, pipeline, store=store)
    replacement = ReplaceComponentService(generator)
    return ApplicationServices(
        catalog=catalog,
        search_products=search,
        generate_builds=generator,
        replace_component=replacement,
        results=store,
        versions=versions,
    )


def create_seeded_demo_services(
    repository: CatalogReader,
    **kwargs: Any,
) -> ApplicationServices:
    """Readable demo alias that still requires a genuinely seeded repository."""

    kwargs.setdefault("compatibility_evidence_policy", CONTROLLED_NON_PRODUCTION_POLICY)
    return create_application_services(repository, **kwargs)


def _performance_label(prefix: str, versions: Mapping[str, str]) -> str:
    return (
        prefix
        + "["
        + ",".join(f"{route}={version}" for route, version in sorted(dict(versions).items()))
        + "]"
    )
