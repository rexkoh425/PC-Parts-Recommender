"""Public application use cases for the PC Build Recommender."""

from .catalog import ApplicationCatalog, CatalogReader, CompatibilityEvidencePolicy
from .factory import (
    ApplicationServices,
    create_application_services,
    create_seeded_demo_services,
)
from .models import (
    ApplicationBuildGenerationResponse,
    ApplicationError,
    ApplicationVersions,
    CandidateLimits,
    CatalogIntegrityError,
    CatalogItem,
    EmptyCatalogError,
    OptimizerProfileStatus,
    ReplacementMode,
    RequestConflictError,
    ResultNotFoundError,
    SearchProductResult,
    SearchProductsOutcome,
)
from .performance import ArtifactPerformanceProvider
from .pipeline import CandidatePipeline, PreparedCandidates, RankedCatalogItem
from .services import (
    GenerateBuildsService,
    ReplaceComponentService,
    SearchProductsService,
)
from .serving import (
    ActiveServingModels,
    EmbeddingReleaseExpectation,
    ServingConfigurationError,
    validate_promoted_serving_models,
)
from .store import InMemoryResultStore, ResultStore, StoredGeneration

__all__ = [
    "ApplicationCatalog",
    "ApplicationBuildGenerationResponse",
    "ApplicationError",
    "ApplicationServices",
    "ApplicationVersions",
    "ActiveServingModels",
    "ArtifactPerformanceProvider",
    "CandidateLimits",
    "CandidatePipeline",
    "CatalogIntegrityError",
    "CatalogItem",
    "CatalogReader",
    "CompatibilityEvidencePolicy",
    "EmptyCatalogError",
    "EmbeddingReleaseExpectation",
    "GenerateBuildsService",
    "InMemoryResultStore",
    "ResultStore",
    "OptimizerProfileStatus",
    "PreparedCandidates",
    "RankedCatalogItem",
    "ReplaceComponentService",
    "ReplacementMode",
    "RequestConflictError",
    "ResultNotFoundError",
    "SearchProductResult",
    "SearchProductsOutcome",
    "SearchProductsService",
    "ServingConfigurationError",
    "StoredGeneration",
    "create_application_services",
    "create_seeded_demo_services",
    "validate_promoted_serving_models",
]
