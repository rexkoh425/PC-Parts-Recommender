"""Deterministic integer objective functions shared by CP-SAT and exhaustive checks."""

from __future__ import annotations

from collections.abc import Iterable

from pc_build_recommender.domain import BuildPreset, CompatVerdict

from .models import (
    PROFILE_WEIGHTS,
    OptimizationCandidate,
    OptimizationProblem,
    PairwiseCompatibility,
)


def candidate_objective_coefficient(
    problem: OptimizationProblem,
    candidate: OptimizationCandidate,
    profile: BuildPreset,
) -> int:
    """Return the integer CP-SAT coefficient for one selected candidate."""

    weights = PROFILE_WEIGHTS[BuildPreset(profile)]
    scores = candidate.scores
    weighted_score = (
        weights.performance * scores.performance
        + weights.value * scores.value
        + weights.reliability * scores.reliability
        + weights.upgradeability * scores.upgradeability
        + weights.efficiency * scores.efficiency
        + weights.preference * scores.preference
    )
    # Scores may contain two decimal places.  Scaling by 100 preserves useful precision
    # while keeping the model entirely integral.
    coefficient = round(weighted_score * 100)
    coefficient -= round(scores.warning_penalty * 100)
    if weights.price_penalty_divisor is not None:
        coefficient -= problem.acquisition_price_cents(candidate) // weights.price_penalty_divisor
    if weights.power_penalty_per_watt:
        coefficient -= candidate_power_watts(problem, candidate) * weights.power_penalty_per_watt
    return coefficient


def candidate_power_watts(
    problem: OptimizationProblem,
    candidate: OptimizationCandidate,
) -> int:
    """Return candidate load, using conservative explicit allowances where configured."""

    if candidate.category.value == "power_supply":
        return 0
    if candidate.power_draw_watts is not None:
        return int(candidate.power_draw_watts)
    return int(problem.category_power_allowances_watts.get(candidate.category, 0))


def pair_penalty(pair: PairwiseCompatibility) -> int:
    if pair.status == CompatVerdict.WARNING or (
        pair.status == CompatVerdict.UNKNOWN and not pair.hard
    ):
        return pair.penalty_points * 100
    return 0


def selected_objective_value(
    problem: OptimizationProblem,
    selected: Iterable[OptimizationCandidate],
    profile: BuildPreset,
) -> int:
    chosen = tuple(selected)
    product_ids = {candidate.product_id for candidate in chosen}
    value = sum(
        candidate_objective_coefficient(problem, candidate, profile) for candidate in chosen
    )
    value -= sum(
        pair_penalty(pair) for pair in problem.pairwise_compatibility if pair.key <= product_ids
    )
    return value
