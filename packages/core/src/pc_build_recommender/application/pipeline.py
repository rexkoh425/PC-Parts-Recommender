"""Deterministic retrieval, hard-filter, compatibility, and ranking stages."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from pc_build_recommender.compatibility import CompatibilityEngine
from pc_build_recommender.domain import (
    BuildRequestSpec,
    ComponentKind,
    WorkloadLabel,
    WorkloadPerformanceSignal,
)
from pc_build_recommender.ranking import (
    ProductRanker,
    RankedCandidate,
    ScoredCandidate,
    RankingContext,
)
from pc_build_recommender.retrieval import (
    ProductRetriever,
    RetrievedCandidate,
    StructuredFilterSpec,
    product_matches_filters,
)

from .catalog import ApplicationCatalog
from .models import CandidateLimits, CatalogItem
from .performance import ArtifactPerformanceProvider

REQUIRED_CATEGORIES: tuple[str, ...] = tuple(category.value for category in ComponentKind)

# Pairwise relations that can safely reject a component before optimisation.
# Full-build power and retained-component checks still run independently after
# CP-SAT has selected a complete build.
COMPATIBILITY_PAIRS: tuple[tuple[str, str], ...] = (
    ("cpu", "motherboard"),
    ("memory", "motherboard"),
    ("motherboard", "case"),
    ("gpu", "case"),
    ("cooler", "case"),
    ("cpu", "cooler"),
    ("gpu", "power_supply"),
    ("storage", "motherboard"),
)


@dataclass(frozen=True, slots=True)
class RankedCatalogItem:
    item: CatalogItem
    rank: int
    raw_score: float
    component_score: float
    ranker_version: str
    ranking_basis: str
    feature_contributions: Mapping[str, float] = field(default_factory=dict)
    performance_signals: Mapping[str, WorkloadPerformanceSignal] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.rank < 1:
            raise ValueError("rank must start at one")
        if not 0.0 <= self.component_score <= 100.0:
            raise ValueError("component_score must be between zero and 100")
        object.__setattr__(
            self,
            "feature_contributions",
            MappingProxyType(dict(self.feature_contributions)),
        )
        object.__setattr__(
            self,
            "performance_signals",
            MappingProxyType(dict(self.performance_signals)),
        )

    @property
    def product_id(self) -> str:
        return self.item.product.product_id

    @property
    def effective_workload_scores(self) -> Mapping[str, float]:
        """Scores usable for ranking while retaining their evidence basis separately."""

        return MappingProxyType(
            {
                workload: signal.relative_score
                for workload, signal in self.performance_signals.items()
            }
        )


@dataclass(frozen=True, slots=True)
class PreparedCandidates:
    pools: Mapping[str, tuple[RankedCatalogItem, ...]]
    locked_product_ids: frozenset[str]
    infeasibility_reasons: tuple[str, ...] = ()

    @property
    def is_feasible(self) -> bool:
        return not self.infeasibility_reasons and all(
            self.pools.get(category) for category in REQUIRED_CATEGORIES
        )


def build_query_text(request: BuildRequestSpec) -> str:
    """Create a stable search query from the authoritative structured request."""

    parts: list[str] = []
    if request.raw_query:
        parts.append(request.raw_query)
    parts.extend(
        workload.name.value.replace("_", " ")
        for workload in sorted(request.workloads, key=lambda item: item.name.value)
    )
    if request.performance_target:
        parts.append(request.performance_target)
    requirements = request.requirements
    if requirements.minimum_gpu_vram_gb is not None:
        parts.append(f"GPU {requirements.minimum_gpu_vram_gb} GB VRAM")
    if requirements.minimum_memory_gb is not None:
        parts.append(f"memory {requirements.minimum_memory_gb} GB")
    if requirements.storage_gb is not None:
        parts.append(f"storage {requirements.storage_gb} GB")
    if requirements.wifi_required:
        parts.append("Wi-Fi")
    if requirements.case_size is not None:
        parts.append(requirements.case_size.value.replace("_", " "))
    if request.preferences.noise:
        parts.append(f"{request.preferences.noise} noise")
    if request.preferences.upgradeability:
        parts.append(f"{request.preferences.upgradeability} upgradeability")
    if request.preferences.power_efficiency:
        parts.append(f"{request.preferences.power_efficiency} power efficiency")
    return " ".join(parts) or "desktop PC component"


def filters_by_category(
    request: BuildRequestSpec,
    *,
    locked_by_category: Mapping[str, str] | None = None,
) -> dict[str, StructuredFilterSpec]:
    """Translate authoritative structured requirements into scoped filters."""

    locked = locked_by_category or {}
    requirements = request.requirements
    excluded = frozenset(request.preferences.excluded_brands)
    result: dict[str, StructuredFilterSpec] = {}
    for category in REQUIRED_CATEGORIES:
        attribute_equals: dict[str, Any] = {}
        attribute_minimums: dict[str, float] = {}
        required_form_factor: str | None = None
        if category == ComponentKind.STORAGE.value and requirements.storage_gb is not None:
            attribute_minimums["capacity_gb"] = float(requirements.storage_gb)
        if category == ComponentKind.CASE.value and requirements.case_size is not None:
            attribute_equals["case_size"] = requirements.case_size.value
        if (
            category == ComponentKind.MOTHERBOARD.value
            and requirements.required_motherboard_form_factor is not None
        ):
            required_form_factor = requirements.required_motherboard_form_factor.value

        locked_id = locked.get(category)
        result[category] = StructuredFilterSpec(
            maximum_price_sgd=(None if locked_id is not None else float(request.budget_sgd)),
            minimum_gpu_vram_gb=(
                float(requirements.minimum_gpu_vram_gb)
                if requirements.minimum_gpu_vram_gb is not None
                else None
            ),
            minimum_memory_gb=(
                float(requirements.minimum_memory_gb)
                if requirements.minimum_memory_gb is not None
                else None
            ),
            required_memory_type=(
                requirements.required_memory_type.value
                if requirements.required_memory_type is not None
                else None
            ),
            required_form_factor=required_form_factor,
            wifi_required=requirements.wifi_required,
            excluded_brands=excluded,
            in_stock_only=requirements.in_stock_only and locked_id is None,
            allowed_product_ids=(frozenset({locked_id}) if locked_id is not None else None),
            attribute_equals=attribute_equals,
            attribute_minimums=attribute_minimums,
        )
    return result

# TODO: rest of this module still to come.
