"""Narrow adapters from the stable public domain API to optimiser projections."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import replace
from decimal import ROUND_HALF_UP, Decimal
from typing import Any

from pc_build_recommender.domain import (
    BuildComponentSelection,
    BuildRequestSpec,
    BuildProfile,
    BuildRecommendation,
    MasterProduct,
    CompatibilityCheck,
    CompatVerdict,
    ComponentKind,
    RetailerListing,
    StockStatus,
)

from .models import (
    CandidateScores,
    IndependentValidator,
    OptimizationCandidate,
    OptimizationProblem,
    OptimizationSolution,
    PairwiseCompatibility,
    _domain_compatibility_status,
)


def money_to_cents(value: object) -> int:
    amount = Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return int(amount * 100)


def _as_mapping(value: object) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        payload: object = model_dump(mode="python")
        if isinstance(payload, Mapping):
            return payload
        raise TypeError("model_dump() must return a mapping")
    return vars(value) if hasattr(value, "__dict__") else {}


def _value(value: object) -> object:
    return getattr(value, "value", value)


def _score_value(value: object, *names: str) -> float:
    source = _as_mapping(value)
    for name in names:
        if name in source and source[name] is not None:
            return float(source[name])
        attribute = getattr(value, name, None)
        if attribute is not None:
            return float(attribute)
    return 0.0


def coerce_scores(value: CandidateScores | Mapping[str, float] | object | None) -> CandidateScores:
    if value is None:
        return CandidateScores()
    if isinstance(value, CandidateScores):
        return value
    return CandidateScores(
        performance=_score_value(
            value, "performance", "performance_score", "workload_score", "predicted_workload_score"
        ),
        value=_score_value(value, "value", "value_score", "price_to_performance"),
        reliability=_score_value(value, "reliability", "reliability_score"),
        upgradeability=_score_value(value, "upgradeability", "upgradeability_score"),
        efficiency=_score_value(value, "efficiency", "efficiency_score", "power_efficiency"),
        preference=_score_value(value, "preference", "preference_score", "preference_match"),
        warning_penalty=_score_value(value, "warning_penalty"),
    )


def candidate_from_domain(
    product: MasterProduct,
    listing: RetailerListing | None,
    *,
    scores: CandidateScores | Mapping[str, float] | object | None = None,
) -> OptimizationCandidate:
    """Project a canonical product and offer using only public domain fields."""

    category = ComponentKind(product.category)
    attributes_object = product.category_attributes
    attributes = dict(_as_mapping(attributes_object))

    price_cents = 0
    listing_id: str | None = None
    in_stock = False
    if listing is not None:
        total_price = getattr(listing, "total_price", None)
        if total_price is None:
            total_price = listing.base_price + getattr(listing, "shipping_price", 0)
        price_cents = money_to_cents(total_price)
        listing_id = str(listing.listing_id)
        in_stock = _value(getattr(listing, "stock_status", None)) == StockStatus.IN_STOCK.value

    power_draw: int | None = None
    psu_wattage: int | None = None
    required_connectors: Mapping[str, int] = {}
    provided_connectors: Mapping[str, int] = {}
    eps_connectors: int | None = None
    recommended_psu: int | None = None
    if category == ComponentKind.CPU:
        power = attributes.get("peak_power_watts") or attributes.get("tdp_watts")
        power_draw = None if power is None else int(float(power) + 0.999999)
    elif category == ComponentKind.GPU:
        power = attributes.get("board_power_watts")
        power_draw = None if power is None else int(float(power) + 0.999999)
        required_connectors = dict(attributes.get("power_connectors") or {})
        recommended_psu_value = attributes.get("recommended_psu_watts")
        recommended_psu = None if recommended_psu_value is None else int(recommended_psu_value)
    elif category == ComponentKind.POWER_SUPPLY:
        wattage = attributes.get("wattage")
        psu_wattage = None if wattage is None else int(wattage)
        provided_connectors = dict(attributes.get("pcie_connectors") or {})
        eps = attributes.get("eps_connectors")
        eps_connectors = None if eps is None else int(eps)

    return OptimizationCandidate(
        product_id=str(product.product_id),
        category=category,
        price_cents=price_cents,
        brand=str(getattr(product, "brand", "")),
        canonical_name=str(
            getattr(product, "canonical_name", None) or getattr(product, "model", "")
        ),
        listing_id=listing_id,
        in_stock=in_stock,
        scores=coerce_scores(scores),
        attributes=attributes,
        power_draw_watts=power_draw,
        psu_wattage=psu_wattage,
        required_power_connectors=required_connectors,
        provided_power_connectors=provided_connectors,
        eps_connectors=eps_connectors,
        recommended_psu_watts=recommended_psu,
        source_product=product,
        source_listing=listing,
    )


def _coerce_pairwise(
    pairwise: Iterable[PairwiseCompatibility] | Mapping[tuple[str, str], object],
) -> tuple[PairwiseCompatibility, ...]:
    if isinstance(pairwise, Mapping):
        converted: list[PairwiseCompatibility] = []
        for (left_id, right_id), value in pairwise.items():
            if isinstance(value, PairwiseCompatibility):
                converted.append(value)
                continue
            status = getattr(value, "status", value)
            converted.append(
                PairwiseCompatibility(
                    left_id,
                    right_id,
                    _domain_compatibility_status(status),
                    message=str(getattr(value, "message", "")),
                )
            )
        return tuple(converted)
    return tuple(pairwise)


def _compatibility_validator(engine: object) -> IndependentValidator:
    checker = getattr(engine, "check_complete_build", None) or getattr(engine, "check_build", None)
    if not callable(checker):
        raise TypeError("compatibility_engine must expose check_complete_build or check_build")

    def validate(selected: Mapping[ComponentKind, OptimizationCandidate]) -> object:
        components: dict[str, Mapping[str, Any]] = {}
        for category, candidate in selected.items():
            source = candidate.source_product
            model_dump = getattr(source, "model_dump", None)
            if source is None or not callable(model_dump):
                raise TypeError("domain compatibility validation requires source products")
            payload: object = model_dump(mode="json")
            if not isinstance(payload, Mapping):
                raise TypeError("source product model_dump() must return a mapping")
            components[category.value] = payload
        return checker(components)

    return validate


def problem_from_domain(
    request: BuildRequestSpec,
    products: Iterable[MasterProduct],
    listings: Iterable[RetailerListing],
    *,
    scores_by_product: Mapping[str, CandidateScores | Mapping[str, float] | object] | None = None,
    pairwise_compatibility: Iterable[PairwiseCompatibility] | Mapping[tuple[str, str], object] = (),
    independent_validator: IndependentValidator | None = None,
    compatibility_engine: object | None = None,
    **overrides: Any,
) -> OptimizationProblem:
    """Choose one offer per product and convert a domain build request."""

    products_tuple = tuple(products)
    scores_by_product = scores_by_product or {}
    listings_by_product: dict[str, list[RetailerListing]] = defaultdict(list)
    for listing in listings:
        if str(getattr(listing, "currency", "")) == "SGD":
            listings_by_product[str(listing.product_id)].append(listing)

    existing = tuple(getattr(request, "existing_products", ()))
    locked_ids = frozenset(str(component.product_id) for component in existing)
    candidates: list[OptimizationCandidate] = []
    for product in products_tuple:
        product_id = str(product.product_id)
        offers = listings_by_product.get(product_id, [])
        offers.sort(
            key=lambda offer: (
                _value(getattr(offer, "stock_status", None)) != StockStatus.IN_STOCK.value,
                money_to_cents(offer.total_price),
                str(offer.listing_id),
            )
        )
        selected_listing = offers[0] if offers else None
        if selected_listing is None and product_id not in locked_ids:
            continue
        candidate = candidate_from_domain(
            product,
            selected_listing,
            scores=scores_by_product.get(product_id),
        )
        if selected_listing is None and product_id in locked_ids:
            candidate = replace(candidate, price_cents=0, in_stock=True)
        candidates.append(candidate)

    requirements = request.requirements
    preferences = request.preferences
    validator = independent_validator
    if validator is None and compatibility_engine is not None:
        validator = _compatibility_validator(compatibility_engine)

    values: dict[str, Any] = {
        "candidates": tuple(candidates),
        "budget_cents": money_to_cents(request.budget_sgd),
        "profiles": tuple(BuildProfile(profile) for profile in request.requested_profiles),
        "locked_product_ids": locked_ids,
        "minimum_gpu_vram_gb": getattr(requirements, "minimum_gpu_vram_gb", None),
        "minimum_memory_gb": getattr(requirements, "minimum_memory_gb", None),
        "minimum_storage_gb": getattr(requirements, "storage_gb", None),
        "required_memory_type": getattr(requirements, "required_memory_type", None),
        "required_motherboard_form_factor": getattr(
            requirements, "required_motherboard_form_factor", None
        ),
        "wifi_required": bool(getattr(requirements, "wifi_required", False)),
        "required_case_size": getattr(requirements, "case_size", None),
        "in_stock_only": bool(getattr(requirements, "in_stock_only", True)),
        "excluded_brands": frozenset(getattr(preferences, "excluded_brands", ())),
        "pairwise_compatibility": _coerce_pairwise(pairwise_compatibility),
        "independent_validator": validator,
    }
    values.update(overrides)
    return OptimizationProblem(**values)


def _report_payload(report: object) -> Mapping[str, Any]:
    if isinstance(report, Mapping):
        return report
    to_dict = getattr(report, "to_dict", None)
    if callable(to_dict):
        payload = to_dict()
        if isinstance(payload, Mapping):
            return payload
    raise TypeError("compatibility report must expose a mapping or to_dict()")


def _component_ids_from_evidence(evidence: object) -> list[str]:
    if not isinstance(evidence, Mapping):
        return []
    identifiers: list[str] = []
    for key, value in evidence.items():
        normalised_key = str(key).casefold()
        if "product_id" not in normalised_key and normalised_key != "component_ids":
            continue
        if isinstance(value, (list, tuple, set, frozenset)):
            identifiers.extend(str(item) for item in value)
        elif value is not None:
            identifiers.append(str(value))
    return list(dict.fromkeys(identifiers))


def solution_to_domain(
    solution: OptimizationSolution,
    *,
    compatibility_report: object | None = None,
    compatibility_engine: object | None = None,
) -> BuildRecommendation:
    """Convert a solution only after a structured compatibility report passes."""

    selected = solution.selected
    report = (
        compatibility_report if compatibility_report is not None else solution.compatibility_report
    )
    if compatibility_engine is not None:
        checker = getattr(compatibility_engine, "check_complete_build", None) or getattr(
            compatibility_engine, "check_build", None
        )
        if not callable(checker):
            raise TypeError("compatibility_engine must expose check_complete_build or check_build")
        records = {}
        for category, candidate in selected.items():
            source = candidate.source_product
            model_dump = getattr(source, "model_dump", None)
            if source is None or not callable(model_dump):
                raise TypeError("fresh compatibility validation requires source products")
            records[category.value] = model_dump(mode="json")
        report = checker(records)
    if report is None:
        raise ValueError("a structured compatibility report is required for domain conversion")

    from .validation import normalise_validator_result

    report_errors = normalise_validator_result(report)
    if report_errors:
        raise ValueError(
            "cannot convert an incompatible or unknown build: " + "; ".join(report_errors)
        )
    payload = _report_payload(report)
    compatibility_checks = []
    for raw_result in payload.get("results", ()):
        item = raw_result if isinstance(raw_result, Mapping) else _report_payload(raw_result)
        status = _domain_compatibility_status(item.get("status", "unknown"))
        compatibility_checks.append(
            CompatibilityCheck(
                rule_id=str(item["rule_id"]) if item.get("rule_id") is not None else None,
                status=status,
                message=str(item.get("message") or f"Compatibility check: {status.value}"),
                component_ids=_component_ids_from_evidence(item.get("evidence", {})),
            )
        )
    overall_status = (
        CompatVerdict.WARNING
        if any(check.status == CompatVerdict.WARNING for check in compatibility_checks)
        else CompatVerdict.PASS
    )
    report_warnings = [
        check.message
        for check in compatibility_checks
        if check.status == CompatVerdict.WARNING
    ]
    components = []
    for category in ComponentKind:
        # ComponentKind has an alias, so iterating yields only canonical values.
        candidate = selected[category]
        weighted_score = (
            candidate.scores.performance
            + candidate.scores.value
            + candidate.scores.reliability
            + candidate.scores.upgradeability
            + candidate.scores.efficiency
            + candidate.scores.preference
        ) / 6
        components.append(
            BuildComponentSelection(
                category=category,
                product_id=candidate.product_id,
                listing_id=candidate.listing_id,
                canonical_name=candidate.canonical_name or candidate.product_id,
                price_sgd=Decimal(candidate.price_cents) / 100,
                component_score=max(0.0, min(100.0, weighted_score)),
                selection_reason=f"Selected by the {solution.profile.value} objective",
            )
        )
    overall = sum(component.component_score for component in components) / len(components)
    return BuildRecommendation(
        profile=solution.profile,
        total_price_sgd=Decimal(solution.total_price_cents) / 100,
        overall_score=max(0.0, min(100.0, overall)),
        components=components,
        workload_scores={},
        compatibility_status=overall_status,
        compatibility_checks=compatibility_checks,
        estimated_power_watts=float(solution.estimated_load_watts),
        warnings=list(dict.fromkeys((*solution.warnings, *report_warnings))),
        explanation=[
            f"Optimised for {solution.profile.value.replace('_', ' ')} within the hard constraints."
        ],
        alternatives=[],
    )
