"""Independent hard-constraint checks and pre-solve diagnostics."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence

from pc_build_recommender.domain import CompatVerdict, ComponentKind

from .models import (
    REQUIRED_CATEGORIES,
    FeatureOperator,
    FeatureRequirement,
    OptimizationCandidate,
    OptimizationProblem,
    _domain_compatibility_status,
)
from .scoring import candidate_power_watts


def _normalise_scalar(value: object) -> object:
    if hasattr(value, "value"):
        value = value.value
    return value.casefold() if isinstance(value, str) else value


def _connector_key(value: str) -> str:
    return value.strip().casefold().replace("-", "_").replace(" ", "_")


def _connectors_satisfy(
    required: Mapping[str, int],
    provided: Mapping[str, int],
) -> bool:
    available = {_connector_key(name): int(count) for name, count in provided.items()}
    return all(
        available.get(_connector_key(name), 0) >= int(count) for name, count in required.items()
    )


def _feature_matches(candidate: OptimizationCandidate, requirement: FeatureRequirement) -> bool:
    value = candidate.attribute(requirement.attribute)
    expected = requirement.expected
    operator = FeatureOperator(requirement.operator)
    if value is None:
        return False
    if operator == FeatureOperator.TRUTHY:
        return bool(value)
    if operator == FeatureOperator.EQUALS:
        return _normalise_scalar(value) == _normalise_scalar(expected)
    if operator == FeatureOperator.AT_LEAST:
        try:
            return bool(value >= expected)
        except TypeError:
            return False
    if operator == FeatureOperator.CONTAINS:
        try:
            return _normalise_scalar(expected) in {_normalise_scalar(item) for item in value}
        except TypeError:
            return False
    return False


def candidate_eligibility_reasons(
    problem: OptimizationProblem,
    candidate: OptimizationCandidate,
) -> tuple[str, ...]:
    """Return every direct hard-filter reason that excludes a candidate."""

    reasons: list[str] = []
    locked = candidate.product_id in problem.locked_product_ids
    if problem.in_stock_only and not locked and not candidate.in_stock:
        reasons.append("not in stock")
    if candidate.brand.casefold() in problem.excluded_brands:
        reasons.append(f"brand {candidate.brand!r} is excluded")

    if candidate.category == ComponentKind.GPU:
        vram = candidate.attribute("vram_gb")
        if problem.minimum_gpu_vram_gb is not None and (
            vram is None or int(vram) < problem.minimum_gpu_vram_gb
        ):
            reasons.append(f"GPU VRAM is below {problem.minimum_gpu_vram_gb} GB or unknown")
        if candidate.power_draw_watts is None:
            reasons.append("GPU board power is unknown")

    if candidate.category == ComponentKind.CPU and candidate.power_draw_watts is None:
        reasons.append("CPU peak power is unknown")

    if candidate.category == ComponentKind.MEMORY:
        capacity = candidate.attribute("capacity_gb")
        if problem.minimum_memory_gb is not None and (
            capacity is None or int(capacity) < problem.minimum_memory_gb
        ):
            reasons.append(f"memory capacity is below {problem.minimum_memory_gb} GB or unknown")
        if problem.required_memory_type is not None and _normalise_scalar(
            candidate.attribute("memory_type")
        ) != _normalise_scalar(problem.required_memory_type):
            reasons.append(f"memory type is not {problem.required_memory_type}")

    if candidate.category == ComponentKind.STORAGE:
        capacity = candidate.attribute("capacity_gb")
        if problem.minimum_storage_gb is not None and (
            capacity is None or int(capacity) < problem.minimum_storage_gb
        ):
            reasons.append(f"storage capacity is below {problem.minimum_storage_gb} GB or unknown")

    if candidate.category == ComponentKind.MOTHERBOARD:
        if problem.wifi_required and candidate.attribute("wifi_support") is not True:
            reasons.append("motherboard Wi-Fi support is absent or unknown")
        if problem.required_memory_type is not None and _normalise_scalar(
            candidate.attribute("memory_type")
        ) != _normalise_scalar(problem.required_memory_type):
            reasons.append(f"motherboard memory type is not {problem.required_memory_type}")
        if problem.required_motherboard_form_factor is not None and _normalise_scalar(
            candidate.attribute("form_factor")
        ) != _normalise_scalar(problem.required_motherboard_form_factor):
            reasons.append(
                f"motherboard form factor is not {problem.required_motherboard_form_factor}"
            )

    if (
        candidate.category == ComponentKind.CASE
        and problem.required_case_size is not None
        and _normalise_scalar(candidate.attribute("case_size"))
        != _normalise_scalar(problem.required_case_size)
    ):
        reasons.append(f"case size is not {problem.required_case_size}")

    if candidate.category == ComponentKind.POWER_SUPPLY:
        if candidate.psu_wattage is None:
            reasons.append("PSU wattage is unknown")
        if candidate.eps_connectors is None:
            reasons.append("PSU EPS connector count is unknown")
        elif candidate.eps_connectors < problem.required_eps_connectors:
            reasons.append(f"PSU has fewer than {problem.required_eps_connectors} EPS connectors")

    for requirement in problem.required_features:
        if requirement.category == candidate.category and not _feature_matches(
            candidate, requirement
        ):
            description = requirement.description or (
                f"{requirement.attribute} {requirement.operator.value} {requirement.expected!r}"
            )
            reasons.append(f"required feature not satisfied: {description}")
    return tuple(reasons)


def eligible_candidates(problem: OptimizationProblem) -> tuple[OptimizationCandidate, ...]:
    return tuple(
        candidate
        for candidate in problem.candidates
        if not candidate_eligibility_reasons(problem, candidate)
    )


def estimated_load_watts(
    problem: OptimizationProblem,
    selected: Mapping[ComponentKind, OptimizationCandidate],
) -> int:
    return problem.base_power_watts + sum(
        candidate_power_watts(problem, candidate)
        for candidate in selected.values()
        if candidate.category != ComponentKind.POWER_SUPPLY
    )


def required_psu_watts(load_watts: int, headroom_percent: int) -> int:
    return math.ceil(load_watts * (100 + headroom_percent) / 100)


def validate_selected_build(
    problem: OptimizationProblem,
    selected: Mapping[ComponentKind, OptimizationCandidate] | Sequence[OptimizationCandidate],
) -> tuple[str, ...]:
    """Re-evaluate every optimiser hard constraint without consulting the CP model."""

    if not isinstance(selected, Mapping):
        selected_by_category = {candidate.category: candidate for candidate in selected}
    else:
        selected_by_category = dict(selected)
    errors: list[str] = []
    categories = Counter(candidate.category for candidate in selected_by_category.values())
    for category in REQUIRED_CATEGORIES:
        if categories[category] != 1 or category not in selected_by_category:
            errors.append(f"build must contain exactly one {category.value}")
    if errors:
        return tuple(errors)

    chosen = tuple(selected_by_category[category] for category in REQUIRED_CATEGORIES)
    chosen_ids = {candidate.product_id for candidate in chosen}
    missing_locks = problem.locked_product_ids - chosen_ids
    if missing_locks:
        errors.append(f"locked products are missing: {sorted(missing_locks)}")
    for candidate in chosen:
        errors.extend(
            f"{candidate.product_id}: {reason}"
            for reason in candidate_eligibility_reasons(problem, candidate)
        )

    acquisition_cost = sum(problem.acquisition_price_cents(candidate) for candidate in chosen)
    if acquisition_cost > problem.budget_cents:
        errors.append(
            f"build price {acquisition_cost} cents exceeds budget {problem.budget_cents} cents"
        )

    for pair in problem.pairwise_compatibility:
        if pair.is_forbidden and pair.key <= chosen_ids:
            detail = pair.message or pair.status.value
            errors.append(
                "forbidden compatibility pair "
                f"{pair.left_product_id}/{pair.right_product_id}: {detail}"
            )

    gpu = selected_by_category[ComponentKind.GPU]
    psu = selected_by_category[ComponentKind.POWER_SUPPLY]
    if not _connectors_satisfy(
        gpu.required_power_connectors,
        psu.provided_power_connectors,
    ):
        errors.append("PSU does not provide all GPU power connectors")
    if (
        gpu.recommended_psu_watts is not None
        and psu.psu_wattage is not None
        and psu.psu_wattage < gpu.recommended_psu_watts
    ):
        errors.append(f"PSU wattage is below GPU recommendation of {gpu.recommended_psu_watts} W")

    load = estimated_load_watts(problem, selected_by_category)
    required_watts = required_psu_watts(load, problem.power_headroom_percent)
    if psu.psu_wattage is None or psu.psu_wattage < required_watts:
        errors.append(
            f"PSU capacity must be at least {required_watts} W for {load} W load and "
            f"{problem.power_headroom_percent}% headroom"
        )
    return tuple(errors)


def normalise_validator_result(result: object) -> tuple[str, ...]:
    """Accept compatibility reports, booleans, or explicit error collections."""

    if result is True:
        return ()
    if result is None:
        return ("independent validator returned no result",)
    if result is False:
        return ("independent validator rejected the build",)
    if isinstance(result, str):
        return (result,)
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[0], bool):
        if result[0]:
            return ()
        reasons = result[1]
        if isinstance(reasons, str):
            return (reasons,)
        return tuple(str(reason) for reason in reasons)

    if isinstance(result, Mapping):
        feasibility_value: object | None = None
        for key in ("is_feasible", "is_compatible", "valid"):
            if key in result:
                feasibility_value = result[key]
                break
        blocking_messages: list[str] = []
        raw_status = result.get("status")
        if raw_status is not None:
            try:
                status = _domain_compatibility_status(raw_status)
            except ValueError:
                return (f"independent validator returned unrecognized status {raw_status!r}",)
            if status in (CompatVerdict.FAIL, CompatVerdict.UNKNOWN):
                blocking_messages.append(f"compatibility status is {status.value}")
        for item in result.get("results", ()):
            item_mapping = item if isinstance(item, Mapping) else {}
            item_status = item_mapping.get("status", getattr(item, "status", None))
            if item_status is None:
                blocking_messages.append("compatibility result is missing a status")
                continue
            try:
                status = _domain_compatibility_status(item_status)
            except ValueError:
                blocking_messages.append(
                    f"compatibility result has unrecognized status {item_status!r}"
                )
                continue
            if status in (CompatVerdict.FAIL, CompatVerdict.UNKNOWN):
                message = item_mapping.get("message", getattr(item, "message", status.value))
                blocking_messages.append(str(message))
        if feasibility_value is False or blocking_messages:
            raw_reasons = result.get("reasons", ())
            if isinstance(raw_reasons, str):
                blocking_messages.append(raw_reasons)
            else:
                blocking_messages.extend(str(reason) for reason in raw_reasons)
            return tuple(dict.fromkeys(blocking_messages)) or (
                "independent validator rejected the build",
            )
        if feasibility_value is True:
            return ()
        return ("independent validator mapping has no explicit feasibility result",)

    to_dict = getattr(result, "to_dict", None)
    if callable(to_dict):
        payload = to_dict()
        if not isinstance(payload, Mapping):
            return ("independent validator to_dict() did not return a mapping",)
        return normalise_validator_result(payload)

    has_failures = getattr(result, "has_failures", None)
    has_unknowns = getattr(result, "has_unknowns", None)
    is_feasible = getattr(result, "is_feasible", None)
    if is_feasible is None:
        is_feasible = getattr(result, "is_compatible", None)
    if isinstance(is_feasible, bool):
        if is_feasible and has_failures is not True and has_unknowns is not True:
            return ()
        return ("independent compatibility validation failed or is unknown",)
    if isinstance(result, Sequence):
        return tuple(str(reason) for reason in result)
    return (f"unrecognized independent validator result type: {type(result).__name__}",)


def diagnose_problem(problem: OptimizationProblem) -> tuple[str, ...]:
    """Return actionable necessary-condition failures before or after solving."""

    reasons: list[str] = []
    by_id = {candidate.product_id: candidate for candidate in problem.candidates}
    unknown_locks = problem.locked_product_ids - by_id.keys()
    if unknown_locks:
        reasons.append(
            f"locked products are absent from the candidate catalogue: {sorted(unknown_locks)}"
        )
    for product_id in sorted(problem.locked_product_ids & by_id.keys()):
        lock_reasons = candidate_eligibility_reasons(problem, by_id[product_id])
        if lock_reasons:
            reasons.append(
                f"locked product {product_id} violates hard requirements: "
                + ", ".join(lock_reasons)
            )

    locked_categories = [
        by_id[product_id].category
        for product_id in problem.locked_product_ids
        if product_id in by_id
    ]
    duplicate_locked_categories = [
        category.value for category, count in Counter(locked_categories).items() if count > 1
    ]
    if duplicate_locked_categories:
        reasons.append(
            f"multiple products are locked in one category: {sorted(duplicate_locked_categories)}"
        )

    eligible = eligible_candidates(problem)
    grouped = {
        category: [candidate for candidate in eligible if candidate.category == category]
        for category in REQUIRED_CATEGORIES
    }
    for category, candidates in grouped.items():
        if not candidates:
            all_in_category = [
                candidate for candidate in problem.candidates if candidate.category == category
            ]
            if not all_in_category:
                reasons.append(f"no candidates were supplied for {category.value}")
            else:
                counts = Counter(
                    reason
                    for candidate in all_in_category
                    for reason in candidate_eligibility_reasons(problem, candidate)
                )
                detail = ", ".join(f"{reason} ({count})" for reason, count in counts.items())
                reasons.append(f"no eligible {category.value} candidates: {detail}")

    if all(grouped.values()):
        cheapest = sum(
            min(problem.acquisition_price_cents(candidate) for candidate in candidates)
            for candidates in grouped.values()
        )
        if cheapest > problem.budget_cents:
            reasons.append(
                f"minimum category-by-category price {cheapest} cents exceeds budget "
                f"{problem.budget_cents} cents"
            )
    return tuple(dict.fromkeys(reasons))
