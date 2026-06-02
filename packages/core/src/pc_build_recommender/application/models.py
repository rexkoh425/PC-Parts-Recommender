"""Application-layer contracts for recommendation use cases.

These contracts intentionally sit above persistence and below HTTP.  They keep
the online recommendation path usable from FastAPI, tests, and command-line
demonstrations without importing transport-specific request models.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any

from pydantic import Field

from pc_build_recommender.domain import (
    BenchmarkResult,
    BuildGenerationResponse,
    BuildPreset,
    MasterProduct,
    CompatVerdict,
    DomainModel,
    RetailerOffering,
    ReviewNote,
)
from pc_build_recommender.optimizer import OptimizationStatus


class ApplicationError(RuntimeError):
    """Base class for errors that are safe to translate at an API boundary."""


class EmptyCatalogError(ApplicationError):
    """Raised when an application is created without real catalogue products."""


class CatalogIntegrityError(ApplicationError):
    """Raised when catalogue identity or category invariants are broken."""


class RequestConflictError(ApplicationError):
    """Raised when an idempotency key is reused for a different request."""


class ResultNotFoundError(ApplicationError, KeyError):
    """Raised when a stored request or build cannot be found."""


class ReplacementMode(StrEnum):
    """Whether a component replacement keeps or reoptimises planned parts."""

    LOCK_OTHER_COMPONENTS = "lock_other_components"
    REOPTIMIZE_UNLOCKED = "reoptimize_unlocked"


class OptimizerProfileStatus(DomainModel):
    """One profile-specific CP-SAT outcome retained for transport adapters."""

    profile: BuildPreset
    status: OptimizationStatus
    wall_time_seconds: float = Field(ge=0)
    objective_value: int | None = None


class ApplicationBuildGenerationResponse(BuildGenerationResponse):
    """Domain build response plus truthful optimisation execution metadata."""

    optimizer_status: OptimizationStatus
    optimizer_version: str = Field(min_length=1)
    retrieval_model: str = Field(min_length=1)
    performance_model: str = Field(min_length=1)
    optimizer_ran: bool
    optimizer_profile_statuses: list[OptimizerProfileStatus] = Field(default_factory=list)
    optimizer_validator_rejections: int = Field(default=0, ge=0)


@dataclass(frozen=True, slots=True)
class CatalogItem:
    """One canonical product and the evidence used during online serving."""

    product: MasterProduct
    listing: RetailerOffering | None
    compatibility_record: Mapping[str, Any]
    workload_scores: Mapping[str, float] = field(default_factory=dict)
    workload_benchmarks: Mapping[str, tuple[BenchmarkResult, ...]] = field(default_factory=dict)
    review_evidence: tuple[ReviewNote, ...] = ()
    ranking_signals: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.listing is not None and self.listing.product_id != self.product.product_id:
            raise CatalogIntegrityError(
                "listing product_id does not match its canonical product: "
                f"{self.listing.listing_id}"
            )
        object.__setattr__(
            self,
            "compatibility_record",
            MappingProxyType(dict(self.compatibility_record)),
        )
        object.__setattr__(self, "workload_scores", MappingProxyType(dict(self.workload_scores)))
        object.__setattr__(
            self,
            "workload_benchmarks",
            MappingProxyType(
                {
                    workload: tuple(observations)
                    for workload, observations in self.workload_benchmarks.items()
                }
            ),
        )
        if any(item.product_id != self.product.product_id for item in self.review_evidence):
            raise CatalogIntegrityError(
                "review evidence product_id does not match its canonical product: "
                f"{self.product.product_id}"
            )
        evidence_ids = [item.evidence_id for item in self.review_evidence]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise CatalogIntegrityError(
                f"catalog item contains duplicate review evidence IDs: {self.product.product_id}"
            )
        object.__setattr__(
            self,
            "review_evidence",
            tuple(sorted(self.review_evidence, key=lambda item: (item.aspect, item.evidence_id))),
        )
        object.__setattr__(self, "ranking_signals", MappingProxyType(dict(self.ranking_signals)))

    @property
    def price_sgd(self) -> float | None:
        return float(self.listing.total_price) if self.listing is not None else None


@dataclass(frozen=True, slots=True)
class SearchProductResult:
    """Transparent hybrid-search result returned by :class:`SearchProductsService`."""

    product: MasterProduct
    listing: RetailerOffering | None
    rank: int
    rrf_score: float
    bm25_score: float
    vector_similarity: float
    compatibility_status: CompatVerdict | None = None
    workload_scores: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.rank < 1:
            raise ValueError("rank must start at one")
        object.__setattr__(self, "workload_scores", MappingProxyType(dict(self.workload_scores)))

    @property
    def product_id(self) -> str:
        return self.product.product_id


@dataclass(frozen=True, slots=True)
class SearchProductsOutcome:
    """A bounded search result set with authoritative compatibility-filter counts.

    ``retrieved_candidates`` counts candidates returned by the retriever after its
    structured filters and before an optional complete-build compatibility check.
    The two filtered counts partition candidates withheld by that compatibility
    check, so callers can observe the compatibility funnel without inspecting
    individual product identifiers.
    """

    results: tuple[SearchProductResult, ...]
    retrieved_candidates: int
    filtered_incompatible: int = 0
    filtered_unknown: int = 0

    def __post_init__(self) -> None:
        counts = (
            self.retrieved_candidates,
            self.filtered_incompatible,
            self.filtered_unknown,
        )
        if any(count < 0 for count in counts):
            raise ValueError("search outcome counts must be non-negative")
        if (
            len(self.results) + self.filtered_incompatible + self.filtered_unknown
            != self.retrieved_candidates
        ):
            raise ValueError(
                "search outcome counts must account for every retrieved candidate"
            )


@dataclass(frozen=True, slots=True)
class CandidateLimits:
    """Bounded candidate pools matching the product specification."""

    cpu: int = 30
    gpu: int = 30
    motherboard: int = 50
    memory: int = 40
    storage: int = 40
    power_supply: int = 40
    cooler: int = 30
    case: int = 30

    def __post_init__(self) -> None:
        if any(value < 1 for value in self.as_mapping().values()):
            raise ValueError("candidate limits must be positive")

    def as_mapping(self) -> dict[str, int]:
        return {
            "cpu": self.cpu,
            "gpu": self.gpu,
            "motherboard": self.motherboard,
            "memory": self.memory,
            "storage": self.storage,
            "power_supply": self.power_supply,
            "cooler": self.cooler,
            "case": self.case,
        }


@dataclass(frozen=True, slots=True)
class ApplicationVersions:
    """Versions stamped onto every generated response."""

    data_version: str
    ranking_model: str
    rule_version: str
    optimizer_version: str
    retrieval_model: str = "unspecified-retrieval-v1"
    performance_model: str = "observed-only-v1"

    def __post_init__(self) -> None:
        for name in (
            "data_version",
            "ranking_model",
            "rule_version",
            "optimizer_version",
            "retrieval_model",
            "performance_model",
        ):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must not be empty")
                print("DEBUG", locals())  # noqa
