"""Deterministic retrieval, hard-filter, compatibility, and ranking stages."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from pc_build_recommender.compatibility import CompatibilityEngine
from pc_build_recommender.domain import (
    BuildGenerationRequest,
    ComponentKind,
    WorkloadName,
    WorkloadPerformanceSignal,
)
from pc_build_recommender.ranking import (
    ProductRanker,
    RankedCandidate,
    RankingCandidate,
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


def build_query_text(request: BuildGenerationRequest) -> str:
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
    request: BuildGenerationRequest,
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


class CandidatePipeline:
    """Prepare bounded, compatibility-eligible, ranked CP-SAT candidates."""

    def __init__(
        self,
        catalog: ApplicationCatalog,
        retriever: ProductRetriever,
        ranker: ProductRanker,
        compatibility_engine: CompatibilityEngine,
        *,
        candidate_limits: CandidateLimits | None = None,
        performance_provider: ArtifactPerformanceProvider | None = None,
    ) -> None:
        self.catalog = catalog
        self.retriever = retriever
        self.ranker = ranker
        self.compatibility_engine = compatibility_engine
        self.candidate_limits = candidate_limits or CandidateLimits()
        self.performance_provider = performance_provider
        self._documents_by_id = {document.product_id: document for document in catalog.documents}

    def prepare(
        self,
        request: BuildGenerationRequest,
        *,
        request_id: str,
    ) -> PreparedCandidates:
        locked_by_category = {
            existing.category.value: existing.product_id for existing in request.existing_products
        }
        reasons = self._validate_locked_products(request, locked_by_category)
        if reasons:
            return PreparedCandidates(
                pools=MappingProxyType({}),
                locked_product_ids=frozenset(locked_by_category.values()),
                infeasibility_reasons=tuple(reasons),
            )

        query_text = build_query_text(request)
        filters = filters_by_category(request, locked_by_category=locked_by_category)
        limits = self.candidate_limits.as_mapping()
        retrieved: dict[str, list[RetrievedCandidate]] = {}
        for category in REQUIRED_CATEGORIES:
            locked_id = locked_by_category.get(category)
            if locked_id is not None:
                document = self.catalog.document_for(locked_id)
                retrieved[category] = [
                    RetrievedCandidate(
                        product=document,
                        rank=1,
                        rrf_score=0.0,
                        bm25_score=0.0,
                        vector_similarity=0.0,
                    )
                ]
                continue
            retrieved[category] = self.retriever.retrieve(
                query_text,
                category=category,
                filters=filters[category],
                top_k=max(limits[category] * 3, limits[category]),
            )

        missing = [category for category in REQUIRED_CATEGORIES if not retrieved[category]]
        if missing:
            reasons = [
                f"No {category} products satisfy the structured filters and availability rules."
                for category in missing
            ]
            reasons.extend(self._suggest_relaxations(request, missing))
            return PreparedCandidates(
                pools=MappingProxyType({}),
                locked_product_ids=frozenset(locked_by_category.values()),
                infeasibility_reasons=tuple(dict.fromkeys(reasons)),
            )

        eligible, compatibility_reasons = self._arc_consistency_prune(retrieved)
        if compatibility_reasons:
            compatibility_reasons.extend(
                self._suggest_relaxations(
                    request,
                    [category for category in REQUIRED_CATEGORIES if not eligible[category]],
                )
            )
            return PreparedCandidates(
                pools=MappingProxyType({}),
                locked_product_ids=frozenset(locked_by_category.values()),
                infeasibility_reasons=tuple(dict.fromkeys(compatibility_reasons)),
            )

        context = RankingContext(
            query_id=request_id,
            query_text=query_text,
            budget_sgd=float(request.budget_sgd),
            workload_weights={item.name.value: item.weight for item in request.workloads},
            requirements=request.requirements.model_dump(mode="json"),
            preferences=request.preferences.model_dump(mode="json"),
            data_version=self.catalog.data_version,
            candidate_set_version=self.catalog.data_version,
        )
        ranked_pools: dict[str, tuple[RankedCatalogItem, ...]] = {}
        for category in REQUIRED_CATEGORIES:
            ranked = self._rank(context, eligible[category])
            ranked_pools[category] = tuple(ranked[: limits[category]])

        return PreparedCandidates(
            pools=MappingProxyType(ranked_pools),
            locked_product_ids=frozenset(locked_by_category.values()),
        )

    def _validate_locked_products(
        self,
        request: BuildGenerationRequest,
        locked_by_category: Mapping[str, str],
    ) -> list[str]:
        reasons: list[str] = []
        filters = filters_by_category(request, locked_by_category=locked_by_category)
        for category, product_id in sorted(locked_by_category.items()):
            item = self.catalog.get(product_id)
            if item is None:
                reasons.append(
                    f"Retained {category} product {product_id!r} is not in the canonical catalogue."
                )
                continue
            if item.product.category.value != category:
                reasons.append(
                    f"Retained product {product_id!r} is {item.product.category.value}, "
                    f"not {category}."
                )
                continue
            document = self.catalog.document_for(product_id)
            if not product_matches_filters(document, filters[category]):
                reasons.append(
                    f"Retained {category} {item.product.canonical_name!r} violates a hard "
                    "structured requirement or brand exclusion."
                )
        return reasons

    def _arc_consistency_prune(
        self,
        pools: Mapping[str, Sequence[RetrievedCandidate]],
    ) -> tuple[dict[str, list[RetrievedCandidate]], list[str]]:
        eligible = {category: list(candidates) for category, candidates in pools.items()}
        cache: dict[tuple[str, str, str, str], bool] = {}

        def pair_is_feasible(
            left_category: str,
            left: RetrievedCandidate,
            right_category: str,
            right: RetrievedCandidate,
        ) -> bool:
            key = (left_category, left.product_id, right_category, right.product_id)
            if key not in cache:
                left_item = self.catalog.require(left.product_id)
                right_item = self.catalog.require(right.product_id)
                report = self.compatibility_engine.check_pair(
                    left_category,
                    left_item.compatibility_record,
                    right_category,
                    right_item.compatibility_record,
                )
                cache[key] = report.is_feasible
            return cache[key]

        changed = True
        while changed:
            changed = False
            for left_category, right_category in COMPATIBILITY_PAIRS:
                left_pool = eligible[left_category]
                right_pool = eligible[right_category]
                supported_left = [
                    left
                    for left in left_pool
                    if any(
                        pair_is_feasible(left_category, left, right_category, right)
                        for right in right_pool
                    )
                ]
                supported_right = [
                    right
                    for right in right_pool
                    if any(
                        pair_is_feasible(left_category, left, right_category, right)
                        for left in supported_left
                    )
                ]
                if len(supported_left) != len(left_pool):
                    eligible[left_category] = supported_left
                    changed = True
                if len(supported_right) != len(right_pool):
                    eligible[right_category] = supported_right
                    changed = True

        empty = [category for category in REQUIRED_CATEGORIES if not eligible[category]]
        reasons = [
            f"Compatibility pruning removed every {category} candidate; required fields may "
            "be missing or all available pairs conflict."
            for category in empty
        ]
        return eligible, reasons

    def _rank(
        self,
        context: RankingContext,
        candidates: Sequence[RetrievedCandidate],
    ) -> list[RankedCatalogItem]:
        ranking_candidates: list[RankingCandidate] = []
        performance_by_product: dict[str, Mapping[str, WorkloadPerformanceSignal]] = {}
        for candidate in candidates:
            item = self.catalog.require(candidate.product_id)
            performance = self._performance_signals(item, context)
            performance_by_product[candidate.product_id] = performance
            request_workloads = {
                workload: signal.relative_score for workload, signal in performance.items()
            }
            signals = dict(item.ranking_signals)
            observed = {
                workload: signal.relative_score
                for workload, signal in performance.items()
                if signal.basis == "observed"
            }
            predicted = {
                workload: signal.relative_score
                for workload, signal in performance.items()
                if signal.basis != "observed"
            }
            observed_score = self._weighted_score(context, observed)
            predicted_score = self._weighted_score(context, predicted)
            if observed_score is not None:
                signals["observed_benchmark_score"] = observed_score
            else:
                signals.pop("observed_benchmark_score", None)
            if predicted_score is not None:
                signals["predicted_workload_score"] = predicted_score
            ranking_candidates.append(
                RankingCandidate.from_retrieved(
                    candidate,
                    workload_scores=request_workloads,
                    signals=signals,
                )
            )
        ranked = self.ranker.rank_query(context, ranking_candidates)
        return self._normalise_ranked(ranked, performance_by_product)

    def _performance_signals(
        self,
        item: CatalogItem,
        context: RankingContext,
    ) -> Mapping[str, WorkloadPerformanceSignal]:
        result: dict[str, WorkloadPerformanceSignal] = {}
        for workload in context.workload_weights:
            observed_score = item.workload_scores.get(workload)
            observations = item.workload_benchmarks.get(workload, ())
            if observed_score is not None and observations:
                result[workload] = WorkloadPerformanceSignal(
                    workload=WorkloadName(workload),
                    metric="normalised comparable benchmark score",
                    unit="relative index",
                    score=observed_score,
                    relative_score=observed_score,
                    basis="observed",
                    confidence="observed",
                    decision="observed_benchmark",
                    supporting_sources=list(
                        dict.fromkeys(observation.source_url for observation in observations)
                    ),
                    supporting_benchmark_ids=[
                        observation.benchmark_id for observation in observations
                    ],
                )
                continue
            if self.performance_provider is None:
                continue
            estimate = self.performance_provider.estimate(item.product, workload)
            if estimate is not None:
                result[workload] = estimate
        return MappingProxyType(result)

    @staticmethod
    def _weighted_score(
        context: RankingContext,
        scores: Mapping[str, float],
    ) -> float | None:
        total_weight = sum(context.workload_weights.get(name, 0.0) for name in scores)
        if total_weight <= 0:
            return None
        return (
            sum(context.workload_weights.get(name, 0.0) * score for name, score in scores.items())
            / total_weight
        )

    def _normalise_ranked(
        self,
        ranked: Sequence[RankedCandidate],
        performance_by_product: Mapping[str, Mapping[str, WorkloadPerformanceSignal]],
    ) -> list[RankedCatalogItem]:
        if not ranked:
            return []
        scores = [item.score for item in ranked]
        minimum = min(scores)
        maximum = max(scores)
        result: list[RankedCatalogItem] = []
        for item in ranked:
            component_score = (
                100.0
                if len(ranked) == 1
                else 100.0 * (item.score - minimum) / max(maximum - minimum, 1e-12)
            )
            result.append(
                RankedCatalogItem(
                    item=self.catalog.require(item.product_id),
                    rank=item.rank,
                    raw_score=item.score,
                    component_score=round(component_score, 6),
                    ranker_version=item.ranker_version,
                    ranking_basis=item.ranking_basis,
                    feature_contributions=item.feature_contributions,
                    performance_signals=performance_by_product.get(item.product_id, {}),
                )
            )
        return result

    @staticmethod
    def _suggest_relaxations(
        request: BuildGenerationRequest,
        empty_categories: Sequence[str],
    ) -> list[str]:
        suggestions: list[str] = []
        categories = set(empty_categories)
        requirements = request.requirements
        if "gpu" in categories and requirements.minimum_gpu_vram_gb is not None:
            suggestions.append("Suggested relaxation: lower the minimum GPU VRAM requirement.")
        if "memory" in categories and requirements.minimum_memory_gb is not None:
            suggestions.append("Suggested relaxation: lower the minimum system-memory capacity.")
        if "storage" in categories and requirements.storage_gb is not None:
            suggestions.append("Suggested relaxation: lower the required storage capacity.")
        if "motherboard" in categories and requirements.wifi_required:
            suggestions.append("Suggested relaxation: allow a separate Wi-Fi adapter.")
        if request.preferences.excluded_brands:
            suggestions.append("Suggested relaxation: remove one or more brand exclusions.")
        if requirements.in_stock_only:
            suggestions.append("Suggested relaxation: include backorder or unknown-stock listings.")
        suggestions.append("Suggested relaxation: increase the acquisition budget.")
        return suggestions
