"""Core application use cases for search, generation, and component replacement."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from decimal import Decimal

from pc_build_recommender.compatibility import (
    CompatibilityEngine,
    CompatibilityReport,
)
from pc_build_recommender.compatibility import (
    CompatVerdict as EngineCompatibilityStatus,
)
from pc_build_recommender.domain import (
    BuildRequestSpec,
    BuildRecommendation,
    CompatVerdict,
    ComponentAlternative,
    ComponentKind,
    ExistingComponent,
    RetailerListing,
    StockState,
    new_id,
)
from pc_build_recommender.optimizer import (
    BuildOptimizer,
    CandidateScores,
    OptimizationCandidate,
    OptimizationProblem,
    OptimizationSolution,
    OptimizationStatus,
    PairwiseCompatibility,
)
from pc_build_recommender.retrieval import StructuredFilterSpec

from .catalog import ApplicationCatalog
from .models import (
    ApplicationBuildGenerationResponse,
    ApplicationVersions,
    CatalogIntegrityError,
    OptimizerProfileStatus,
    ReplacementMode,
    RequestConflictError,
    SearchProductResult,
    SearchProductsOutcome,
)
from .pipeline import (
    COMPATIBILITY_PAIRS,
    CandidatePipeline,
    PreparedCandidates,
    RankedCatalogItem,
)
from .presenter import recommendation_from_solution
from .store import InMemoryResultStore, ResultStore

_CATEGORY_ALIASES = {
    "psu": ComponentKind.POWER_SUPPLY.value,
    "power_supply_unit": ComponentKind.POWER_SUPPLY.value,
    "cpu_cooler": ComponentKind.COOLER.value,
    "ram": ComponentKind.MEMORY.value,
    "chassis": ComponentKind.CASE.value,
}


def _category(value: ComponentKind | str) -> ComponentKind:
    raw = value.value if isinstance(value, ComponentKind) else value
    normalised = str(raw).strip().casefold().replace("-", "_").replace(" ", "_")
    return ComponentKind(_CATEGORY_ALIASES.get(normalised, normalised))


def _pair_status(report: CompatibilityReport) -> CompatVerdict:
    if report.status == EngineCompatibilityStatus.FAIL:
        return CompatVerdict.FAIL
    if report.status == EngineCompatibilityStatus.UNKNOWN:
        return CompatVerdict.UNKNOWN
    if report.status == EngineCompatibilityStatus.WARNING:
        return CompatVerdict.WARNING
    return CompatVerdict.PASS


def _non_pass_message(report: CompatibilityReport) -> str:
    messages = [
        result.message
        for result in report.results
        if result.status != EngineCompatibilityStatus.PASS
    ]
    return " ".join(dict.fromkeys(messages))


class SearchProductsService:
    """Category-scoped hybrid product search with optional build compatibility."""

    def __init__(
        self,
        catalog: ApplicationCatalog,
        pipeline: CandidatePipeline,
        *,
        store: ResultStore | None = None,
    ) -> None:
        self.catalog = catalog
        self.pipeline = pipeline
        self.store = store

    def search(
        self,
        query: str,
        *,
        category: ComponentKind | str,
        filters: StructuredFilterSpec | None = None,
        top_k: int = 20,
        compatible_with_build_id: str | None = None,
    ) -> list[SearchProductResult]:
        """Return search results while preserving the original list-only API."""

        return list(
            self.search_with_outcome(
                query,
                category=category,
                filters=filters,
                top_k=top_k,
                compatible_with_build_id=compatible_with_build_id,
            ).results
        )

    def search_with_outcome(
        self,
        query: str,
        *,
        category: ComponentKind | str,
        filters: StructuredFilterSpec | None = None,
        top_k: int = 20,
        compatible_with_build_id: str | None = None,
    ) -> SearchProductsOutcome:
        """Return search results plus conservative compatibility-filter diagnostics."""

        if top_k < 1:
            return SearchProductsOutcome(results=(), retrieved_candidates=0)
        category_value = _category(category).value
        hits = self.pipeline.retriever.retrieve(
            query or category_value.replace("_", " "),
            category=category_value,
            filters=filters or StructuredFilterSpec(),
            top_k=top_k,
        )
        retrieved_candidates = len(hits)
        compatibility_statuses: dict[str, CompatVerdict] = {}
        filtered_incompatible = 0
        filtered_unknown = 0
        if compatible_with_build_id is not None:
            if self.store is None:
                raise RuntimeError("compatible build search requires a result store")
            generation = self.store.generation_for_build(compatible_with_build_id)
            build = self.store.require_build(compatible_with_build_id)
            existing_records = [
                self.catalog.require(component.product_id).compatibility_record
                for component in generation.request.existing_products
                if component.category.value != category_value
            ]
            compatible_hits = []
            for hit in hits:
                build_records = {
                    component.category.value: self.catalog.require(
                        component.product_id
                    ).compatibility_record
                    for component in build.components
                }
                build_records[category_value] = self.catalog.require(
                    hit.product_id
                ).compatibility_record
                report = self.pipeline.compatibility_engine.check_complete_build(
                    build_records,
                    existing_components=existing_records,
                )
                if report.is_feasible:
                    compatible_hits.append(hit)
                    compatibility_statuses[hit.product_id] = _pair_status(report)
                elif report.has_failures:
                    # A combined FAIL + UNKNOWN report stays in the FAIL bucket,
                    # matching the public API's fail-first filter semantics.
                    filtered_incompatible += 1
                else:
                    filtered_unknown += 1
            hits = compatible_hits

        return SearchProductsOutcome(
            results=tuple(
                SearchProductResult(
                    product=self.catalog.require(hit.product_id).product,
                    listing=self.catalog.require(hit.product_id).listing,
                    rank=rank,
                    rrf_score=hit.rrf_score,
                    bm25_score=hit.bm25_score,
                    vector_similarity=hit.vector_similarity,
                    compatibility_status=compatibility_statuses.get(hit.product_id),
                    workload_scores=self.catalog.require(hit.product_id).workload_scores,
                )
                for rank, hit in enumerate(hits, start=1)
            ),
            retrieved_candidates=retrieved_candidates,
            filtered_incompatible=filtered_incompatible,
            filtered_unknown=filtered_unknown,
        )


class GenerateBuildsService:
    """Execute retrieval through independently validated diverse builds."""

    def __init__(
        self,
        catalog: ApplicationCatalog,
        pipeline: CandidatePipeline,
        compatibility_engine: CompatibilityEngine,
        *,
        optimizer: BuildOptimizer | None = None,
        store: ResultStore | None = None,
        versions: ApplicationVersions | None = None,
        random_seed: int = 42,
    ) -> None:
        self.catalog = catalog
        self.pipeline = pipeline
        self.compatibility_engine = compatibility_engine
        self.optimizer = optimizer or BuildOptimizer()
        self.store = store if store is not None else InMemoryResultStore()
        self.versions = versions or ApplicationVersions(
            data_version=catalog.data_version,
            ranking_model=pipeline.ranker.metadata.ranker_version,
            rule_version=compatibility_engine.rule_version,
            optimizer_version="cp-sat-v1",
        )
        self.random_seed = random_seed

    def generate(
        self,
        request: BuildRequestSpec,
        *,
        request_id: str | None = None,
        included_existing_product_ids: frozenset[str] = frozenset(),
    ) -> ApplicationBuildGenerationResponse:
        existing_ids = frozenset(item.product_id for item in request.existing_products)
        unknown_included = included_existing_product_ids - existing_ids
        if unknown_included:
            raise ValueError(
                "included_existing_product_ids must reference retained products: "
                f"{sorted(unknown_included)}"
            )
        unpriced_included = [
            product_id
            for product_id in included_existing_product_ids
            if self.catalog.require(product_id).listing is None
        ]
        if unpriced_included:
            raise CatalogIntegrityError(
                "retained products included in budget require a catalog listing: "
                f"{sorted(unpriced_included)}"
            )
        no_cost_ids = existing_ids - included_existing_product_ids
        return self._generate(
            request,
            request_id=request_id,
            no_cost_product_ids=no_cost_ids,
            owned_product_ids=existing_ids,
            exclude_locked_from_budget=not included_existing_product_ids,
        )

    def _generate(
        self,
        request: BuildRequestSpec,
        *,
        request_id: str | None,
        no_cost_product_ids: frozenset[str],
        owned_product_ids: frozenset[str],
        exclude_locked_from_budget: bool,
        selection_labels: Mapping[str, str] | None = None,
    ) -> ApplicationBuildGenerationResponse:
        resolved_request_id = request_id or new_id("req")
        if not resolved_request_id.strip():
            raise ValueError("request_id must not be empty")
        prior = self.store.get_generation(resolved_request_id)
        if prior is not None:
            if (
                prior.request != request
                or prior.no_cost_product_ids != no_cost_product_ids
                or prior.owned_product_ids != owned_product_ids
            ):
                raise RequestConflictError(
                    f"request_id is already bound to another request: {resolved_request_id}"
                )
            return prior.response

        prepared = self.pipeline.prepare(request, request_id=resolved_request_id)
        if not prepared.is_feasible:
            response = self._response(
                resolved_request_id,
                builds=[],
                infeasibility_reasons=list(prepared.infeasibility_reasons),
                optimizer_status=OptimizationStatus.INFEASIBLE,
                optimizer_ran=False,
            )
            return self.store.save(
                request,
                response,
                no_cost_product_ids=no_cost_product_ids,
                owned_product_ids=owned_product_ids,
            )

        problem = self._problem(
            request,
            prepared,
            no_cost_product_ids=no_cost_product_ids,
            exclude_locked_from_budget=exclude_locked_from_budget,
        )
        result = self.optimizer.optimize(
            problem,
            max_solutions=min(5, len(request.requested_profiles)),
        )
        builds: list[BuildRecommendation] = []
        rejected_reasons: list[str] = []
        for solution in result.solutions:
            report = self._complete_report(solution, request)
            if not report.is_feasible:
                rejected_reasons.extend(
                    result_item.message
                    for result_item in report.results
                    if result_item.status
                    in (EngineCompatibilityStatus.FAIL, EngineCompatibilityStatus.UNKNOWN)
                )
                continue
            if not self._is_diverse(solution, builds, request):
                rejected_reasons.append(
                    "A solved build was withheld because it differed from an earlier result "
                    "in fewer than two unlocked meaningful components."
                )
                continue
            recommendation = recommendation_from_solution(
                solution,
                report,
                request,
                prepared,
                no_cost_product_ids=no_cost_product_ids,
                selection_labels=selection_labels,
            )
            recommendation = self._with_alternatives(
                recommendation,
                solution,
                request,
                prepared,
                no_cost_product_ids=no_cost_product_ids,
            )
            builds.append(recommendation)

        reasons = list(dict.fromkeys((*result.infeasibility_reasons, *rejected_reasons)))
        if not builds and not reasons:
            reasons.append("No build passed the independent complete-build compatibility recheck.")
        if len(builds) < len(request.requested_profiles):
            reasons.append(
                f"Only {len(builds)} of {len(request.requested_profiles)} requested profile "
                "builds satisfy all hard constraints and diversity rules."
            )
        if not builds:
            reasons.extend(self._general_relaxations(request))

        response = self._response(
            resolved_request_id,
            builds=builds,
            infeasibility_reasons=list(dict.fromkeys(reasons)),
            optimizer_status=result.status,
            optimizer_ran=True,
            optimizer_profile_statuses=[
                OptimizerProfileStatus(
                    profile=record.profile,
                    status=record.status,
                    wall_time_seconds=record.wall_time_seconds,
                    objective_value=record.objective_value,
                )
                for record in result.profile_statuses
            ],
            optimizer_validator_rejections=result.rejected_by_validator,
        )
        return self.store.save(
            request,
            response,
            no_cost_product_ids=no_cost_product_ids,
            owned_product_ids=owned_product_ids,
        )

    def get_response(self, request_id: str) -> ApplicationBuildGenerationResponse | None:
        return self.store.get_response(request_id)

    def get_build(self, build_id: str) -> BuildRecommendation | None:
        return self.store.get_build(build_id)

    def _problem(
        self,
        request: BuildRequestSpec,
        prepared: PreparedCandidates,
        *,
        no_cost_product_ids: frozenset[str],
        exclude_locked_from_budget: bool,
    ) -> OptimizationProblem:
        ranked = [candidate for pool in prepared.pools.values() for candidate in pool]
        products = [candidate.item.product for candidate in ranked]
        listings: list[RetailerListing] = []
        for candidate in ranked:
            listing = candidate.item.listing
            if listing is None:
                continue
            if not exclude_locked_from_budget and candidate.product_id in no_cost_product_ids:
                listing = listing.model_copy(
                    update={
                        "base_price": Decimal("0"),
                        "shipping_price": Decimal("0"),
                        "stock_status": StockState.IN_STOCK,
                    }
                )
            listings.append(listing)

        scores = {
            candidate.product_id: self._optimizer_scores(candidate, request) for candidate in ranked
        }
        pairwise = self._pairwise_constraints(prepared)
        existing_records = [
            self.catalog.require(existing.product_id).compatibility_record
            for existing in request.existing_products
        ]

        def independent_validator(
            selected: Mapping[ComponentKind, OptimizationCandidate],
        ) -> CompatibilityReport:
            records = {
                category.value: self.catalog.require(candidate.product_id).compatibility_record
                for category, candidate in selected.items()
            }
            return self.compatibility_engine.check_complete_build(
                records,
                existing_components=existing_records,
            )

        return OptimizationProblem.from_domain(
            request,
            products,
            listings,
            scores_by_product=scores,
            pairwise_compatibility=pairwise,
            independent_validator=independent_validator,
            exclude_locked_from_budget=exclude_locked_from_budget,
            random_seed=self.random_seed,
        )

    def _pairwise_constraints(
        self, prepared: PreparedCandidates
    ) -> tuple[PairwiseCompatibility, ...]:
        constraints: list[PairwiseCompatibility] = []
        for left_category, right_category in COMPATIBILITY_PAIRS:
            for left in prepared.pools[left_category]:
                for right in prepared.pools[right_category]:
                    report = self.compatibility_engine.check_pair(
                        left_category,
                        left.item.compatibility_record,
                        right_category,
                        right.item.compatibility_record,
                    )
                    status = _pair_status(report)
                    if status == CompatVerdict.PASS:
                        continue
                    constraints.append(
                        PairwiseCompatibility(
                            left_product_id=left.product_id,
                            right_product_id=right.product_id,
                            status=status,
                            message=_non_pass_message(report),
                            hard=True,
                        )
                    )
        return tuple(constraints)

    @staticmethod
    def _optimizer_scores(
        candidate: RankedCatalogItem,
        request: BuildRequestSpec,
    ) -> CandidateScores:
        workload_values = [
            (
                workload.weight,
                candidate.effective_workload_scores[workload.name.value],
            )
            for workload in request.workloads
            if workload.name.value in candidate.effective_workload_scores
        ]
        observed_weight = sum(weight for weight, _ in workload_values)
        performance = (
            sum(weight * value for weight, value in workload_values) / observed_weight
            if observed_weight > 0
            else 0.0
        )
        signals = candidate.item.ranking_signals
        preferred = {brand.casefold() for brand in request.preferences.preferred_brands}
        preference = (
            100.0
            if preferred and candidate.item.product.brand.casefold() in preferred
            else 100.0 * float(signals.get("preference_match_score", 0.0))
        )

        def percentage(name: str) -> float:
            value = float(signals.get(name, 0.0))
            return max(0.0, min(100.0, value * 100.0 if value <= 1.0 else value))

        return CandidateScores(
            performance=max(0.0, min(100.0, performance)),
            value=candidate.component_score,
            reliability=percentage("reliability_score"),
            upgradeability=percentage("upgradeability_score"),
            efficiency=percentage("power_efficiency_score"),
            preference=max(0.0, min(100.0, preference)),
        )

    def _complete_report(
        self,
        solution: OptimizationSolution,
        request: BuildRequestSpec,
    ) -> CompatibilityReport:
        records = {
            category.value: self.catalog.require(candidate.product_id).compatibility_record
            for category, candidate in solution.selected.items()
        }
        existing_records = [
            self.catalog.require(existing.product_id).compatibility_record
            for existing in request.existing_products
        ]
        return self.compatibility_engine.check_complete_build(
            records,
            existing_components=existing_records,
        )

    @staticmethod
    def _is_diverse(
        solution: OptimizationSolution,
        accepted: Sequence[BuildRecommendation],
        request: BuildRequestSpec,
    ) -> bool:
        if not accepted:
            return True
        locked_categories = {item.category for item in request.existing_products}
        selected = {
            category: candidate.product_id for category, candidate in solution.selected.items()
        }
        for build in accepted:
            previous = {component.category: component.product_id for component in build.components}
            differences = sum(
                selected[category] != previous[category]
                for category in ComponentKind
                if category not in locked_categories
            )
            if differences < 2:
                return False
        return True

    def _with_alternatives(
        self,
        build: BuildRecommendation,
        solution: OptimizationSolution,
        request: BuildRequestSpec,
        prepared: PreparedCandidates,
        *,
        no_cost_product_ids: frozenset[str],
    ) -> BuildRecommendation:
        alternatives: list[ComponentAlternative] = []
        selected_ids = {candidate.product_id for candidate in solution.selected.values()}
        acquisition_total = build.total_price_sgd
        for category in ComponentKind:
            selected_candidate = solution.selected[category]
            if selected_candidate.product_id in no_cost_product_ids:
                continue
            selected_ranked = next(
                item
                for item in prepared.pools[category.value]
                if item.product_id == selected_candidate.product_id
            )
            for candidate in prepared.pools[category.value]:
                if candidate.product_id in selected_ids:
                    continue
                replacement_price = Decimal(str(candidate.item.price_sgd or 0.0))
                current_price = Decimal(selected_candidate.price_cents) / Decimal(100)
                price_delta = replacement_price - current_price
                if acquisition_total + price_delta > request.budget_sgd:
                    continue
                records = {
                    selected_category.value: self.catalog.require(
                        selected.product_id
                    ).compatibility_record
                    for selected_category, selected in solution.selected.items()
                }
                records[category.value] = candidate.item.compatibility_record
                report = self.compatibility_engine.check_complete_build(records)
                if not report.is_feasible:
                    continue
                performance_delta = self._weighted_workload_delta(
                    candidate,
                    selected_ranked,
                    request,
                )
                alternatives.append(
                    ComponentAlternative(
                        category=category,
                        product_id=candidate.product_id,
                        listing_id=(
                            candidate.item.listing.listing_id
                            if candidate.item.listing is not None
                            else None
                        ),
                        canonical_name=candidate.item.product.canonical_name,
                        price_delta_sgd=price_delta,
                        performance_delta=performance_delta,
                        explanation=(
                            f"Compatible rank #{candidate.rank} alternative; swapping it changes "
                            f"acquisition cost by S${price_delta:+.2f}."
                        ),
                    )
                )
                break
            if len(alternatives) >= 3:
                break
        return build.model_copy(update={"alternatives": alternatives}, deep=True)

    @staticmethod
    def _weighted_workload_delta(
        candidate: RankedCatalogItem,
        selected: RankedCatalogItem,
        request: BuildRequestSpec,
    ) -> float | None:
        paired = [
            (
                workload.weight,
                candidate.effective_workload_scores[workload.name.value],
                selected.effective_workload_scores[workload.name.value],
            )
            for workload in request.workloads
            if workload.name.value in candidate.effective_workload_scores
            and workload.name.value in selected.effective_workload_scores
        ]
        if not paired:
            return None
        weight = sum(item[0] for item in paired)
        return round(
            sum(item_weight * (new - prior) for item_weight, new, prior in paired) / weight,
            6,
        )

    def _response(
        self,
        request_id: str,
        *,
        builds: Sequence[BuildRecommendation],
        infeasibility_reasons: Sequence[str],
        optimizer_status: OptimizationStatus,
        optimizer_ran: bool,
        optimizer_profile_statuses: Sequence[OptimizerProfileStatus] = (),
        optimizer_validator_rejections: int = 0,
    ) -> ApplicationBuildGenerationResponse:
        return ApplicationBuildGenerationResponse(
            request_id=request_id,
            data_version=self.versions.data_version,
            ranking_model=self.versions.ranking_model,
            rule_version=self.versions.rule_version,
            builds=list(builds),
            infeasibility_reasons=list(infeasibility_reasons),
            optimizer_status=optimizer_status,
            optimizer_version=self.versions.optimizer_version,
            retrieval_model=self.versions.retrieval_model,
            performance_model=self.versions.performance_model,
            optimizer_ran=optimizer_ran,
            optimizer_profile_statuses=list(optimizer_profile_statuses),
            optimizer_validator_rejections=optimizer_validator_rejections,
        )

    @staticmethod
    def _general_relaxations(request: BuildRequestSpec) -> list[str]:
        suggestions = ["Suggested relaxation: increase the acquisition budget."]
        if request.requirements.minimum_gpu_vram_gb is not None:
            suggestions.append("Suggested relaxation: lower the minimum GPU VRAM requirement.")
        if request.requirements.minimum_memory_gb is not None:
            suggestions.append("Suggested relaxation: lower the minimum memory requirement.")
        if request.preferences.excluded_brands:
            suggestions.append("Suggested relaxation: remove one or more brand exclusions.")
        return suggestions


class ReplaceComponentService:
    """Keep a build fixed, replace one category, and re-run all hard gates."""

    def __init__(self, generator: GenerateBuildsService) -> None:
        self.generator = generator

    def replace(
        self,
        build_id: str,
        *,
        category: ComponentKind | str,
        replacement_product_id: str,
        mode: ReplacementMode | str = ReplacementMode.LOCK_OTHER_COMPONENTS,
        request_id: str | None = None,
    ) -> ApplicationBuildGenerationResponse:
        generation = self.generator.store.generation_for_build(build_id)
        build = self.generator.store.require_build(build_id)
        replacement = self.generator.catalog.get(replacement_product_id)
        if replacement is None:
            raise CatalogIntegrityError(f"replacement product not found: {replacement_product_id}")
        requested_category = _category(category)
        replacement_mode = ReplacementMode(mode)
        if replacement.product.category != requested_category:
            raise CatalogIntegrityError(
                f"replacement product is {replacement.product.category.value}, not "
                f"{requested_category.value}"
            )
        current_by_category = {component.category: component for component in build.components}
        if requested_category not in current_by_category:
            raise CatalogIntegrityError(
                f"build does not contain category {requested_category.value}"
            )

        old_product_id = current_by_category[requested_category].product_id
        locked: list[ExistingComponent] = []
        labels: dict[str, str] = {}
        for component_category in ComponentKind:
            current = current_by_category[component_category]
            product_id = (
                replacement_product_id
                if component_category == requested_category
                else current.product_id
            )
            item = self.generator.catalog.require(product_id)
            listing_id = item.listing.listing_id if item.listing is not None else None
            should_lock = (
                replacement_mode == ReplacementMode.LOCK_OTHER_COMPONENTS
                or component_category == requested_category
                or product_id in generation.no_cost_product_ids
                or product_id in generation.owned_product_ids
            )
            if should_lock:
                locked.append(
                    ExistingComponent(
                        category=component_category,
                        product_id=product_id,
                        listing_id=listing_id,
                    )
                )
            if component_category == requested_category:
                labels[product_id] = (
                    f"User-selected replacement for {current.canonical_name}; all other "
                    "components were kept fixed."
                )
            elif product_id in generation.no_cost_product_ids:
                labels[product_id] = (
                    "Retained original user-owned component; its acquisition cost remains "
                    "excluded from the budget."
                )
            elif product_id in generation.owned_product_ids:
                labels[product_id] = (
                    "Retained original user-owned component; its recorded price remains "
                    "included in the budget."
                )
            else:
                labels[product_id] = (
                    f"Kept fixed from build {build_id}."
                    if replacement_mode == ReplacementMode.LOCK_OTHER_COMPONENTS
                    else "Re-selected while optimising unlocked components around the replacement."
                )

        no_cost_ids = generation.no_cost_product_ids - {old_product_id}
        owned_ids = generation.owned_product_ids - {old_product_id}
        updated_request = generation.request.model_copy(
            update={
                "existing_products": locked,
                "requested_profiles": [build.profile],
            },
            deep=True,
        )
        return self.generator._generate(
            updated_request,
            request_id=request_id,
            no_cost_product_ids=no_cost_ids,
            owned_product_ids=owned_ids,
            exclude_locked_from_budget=False,
            selection_labels=labels,
        )
