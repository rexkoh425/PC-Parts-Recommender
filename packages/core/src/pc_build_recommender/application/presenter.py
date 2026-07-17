"""Deterministic conversion of validated optimiser solutions to domain responses."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from decimal import Decimal
from typing import Any

from pc_build_recommender.compatibility import (
    CompatibilityReport,
)
from pc_build_recommender.compatibility import (
    CompatVerdict as EngineCompatibilityStatus,
)
from pc_build_recommender.domain import (
    BuildComponentSelection,
    BuildGenerationRequest,
    BuildRecommendation,
    CompatibilityCheck,
    CompatVerdict,
    WorkloadLabel,
)
from pc_build_recommender.optimizer.models import OptimizationSolution

from .pipeline import PreparedCandidates, RankedCatalogItem


def _domain_status(status: EngineCompatibilityStatus) -> CompatVerdict:
    return CompatVerdict(status.value.casefold())


def _component_ids(evidence: Mapping[str, Any]) -> list[str]:
    values: set[str] = set()

    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            for key, nested in value.items():
                if key == "product_id" and nested is not None:
                    values.add(str(nested))
                else:
                    visit(nested)
        elif isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
            for nested in value:
                visit(nested)

    visit(evidence)
    return sorted(values)


def compatibility_checks(report: CompatibilityReport) -> list[CompatibilityCheck]:
    return [
        CompatibilityCheck(
            rule_id=result.rule_id,
            status=_domain_status(result.status),
            message=result.message,
            component_ids=_component_ids(result.evidence),
        )
        for result in report.results
    ]


def _workload_scores(
    request: BuildGenerationRequest,
    selected: Iterable[RankedCatalogItem],
) -> dict[WorkloadLabel, float]:
    items = tuple(selected)
    result: dict[WorkloadLabel, float] = {}
    for workload in request.workloads:
        signals = [
            signal
            for item in items
            if (signal := item.performance_signals.get(workload.name.value)) is not None
            and signal.score is not None
            and signal.score <= 100.0
        ]
        comparable_targets = {(signal.metric, signal.unit) for signal in signals}
        if signals and len(comparable_targets) == 1:
            result[workload.name] = round(
                sum(signal.score for signal in signals if signal.score is not None) / len(signals),
                6,
            )
    return result


def recommendation_from_solution(
    solution: OptimizationSolution,
    report: CompatibilityReport,
    request: BuildGenerationRequest,
    prepared: PreparedCandidates,
    *,
    no_cost_product_ids: frozenset[str] = frozenset(),
    selection_labels: Mapping[str, str] | None = None,
) -> BuildRecommendation:
    """Present one independently rechecked solution without generative text."""

    if not report.is_feasible:
        raise ValueError("a failed or unknown compatibility report cannot be presented")
    labels = selection_labels or {}
    ranked_by_id = {
        candidate.product_id: candidate for pool in prepared.pools.values() for candidate in pool
    }
    existing_by_id = {item.product_id: item for item in request.existing_products}
    components: list[BuildComponentSelection] = []
    selected_ranked: list[RankedCatalogItem] = []
    for category, candidate in sorted(solution.selected.items(), key=lambda item: item[0].value):
        ranked = ranked_by_id[candidate.product_id]
        selected_ranked.append(ranked)
        is_no_cost = candidate.product_id in no_cost_product_ids
        existing = existing_by_id.get(candidate.product_id)
        listing_id = (
            existing.listing_id
            if existing is not None and existing.listing_id is not None
            else candidate.listing_id
        )
        if candidate.product_id in labels:
            reason = labels[candidate.product_id]
        elif is_no_cost:
            reason = (
                "Retained existing component as requested; its acquisition cost is excluded "
                "from this budget."
            )
        else:
            reason = (
                f"Ranked #{ranked.rank} in {category.value} and selected by the "
                f"{solution.profile.value.replace('_', ' ')} objective."
            )
        components.append(
            BuildComponentSelection(
                category=category,
                product_id=candidate.product_id,
                listing_id=listing_id,
                canonical_name=candidate.canonical_name or candidate.product_id,
                price_sgd=(
                    Decimal("0") if is_no_cost else Decimal(candidate.price_cents) / Decimal(100)
                ),
                component_score=round(ranked.component_score, 6),
                selection_reason=reason,
                performance_signals=[
                    ranked.performance_signals[workload.name.value]
                    for workload in request.workloads
                    if workload.name.value in ranked.performance_signals
                ],
            )
        )

    checks = compatibility_checks(report)
    warning_messages = [
        check.message for check in checks if check.status == CompatVerdict.WARNING
    ]
    status = CompatVerdict.WARNING if warning_messages else CompatVerdict.PASS
    overall_score = round(
        sum(component.component_score for component in components) / len(components),
        6,
    )
    workload_scores = _workload_scores(request, selected_ranked)
    performance_bases = {
        signal.basis for item in selected_ranked for signal in item.performance_signals.values()
    }
    explanation = [
        (
            f"The {solution.profile.value.replace('_', ' ')} CP-SAT objective selected one "
            "eligible product from each required category."
        ),
        (
            f"Acquisition cost is S${Decimal(solution.total_price_cents) / Decimal(100):.2f} "
            f"against a S${request.budget_sgd:.2f} budget."
        ),
        (
            f"The complete build was independently rechecked with compatibility rules "
            f"{report.rule_version}; no FAIL or hard UNKNOWN result is returned."
        ),
    ]
    if "observed" in performance_bases:
        explanation.append(
            "Displayed workload scores use benchmark observations normalised only within "
            "comparable benchmark configurations."
        )
    if "predicted" in performance_bases:
        explanation.append(
            "Predicted component performance comes from promotion-eligible, versioned models; "
            "component evidence identifies the exact model version."
        )
    if "relative" in performance_bases:
        explanation.append(
            "Relative-only model outputs may influence ranking but are not presented as precise "
            "build performance estimates."
        )
    if not performance_bases:
        explanation.append(
            "No comparable observed workload benchmark covers this complete build; ranking "
            "therefore relies on retrieval, specifications, price, and available evidence."
        )
    if no_cost_product_ids:
        explanation.append(
            "Retained user-owned component prices are shown as S$0 in the acquisition total."
        )

    return BuildRecommendation(
        profile=solution.profile,
        total_price_sgd=Decimal(solution.total_price_cents) / Decimal(100),
        overall_score=max(0.0, min(100.0, overall_score)),
        components=components,
        workload_scores=workload_scores,
        compatibility_status=status,
        compatibility_checks=checks,
        estimated_power_watts=float(solution.estimated_load_watts),
        warnings=list(dict.fromkeys((*solution.warnings, *warning_messages))),
        explanation=explanation,
        alternatives=[],
    )
