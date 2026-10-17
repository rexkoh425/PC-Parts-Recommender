"""Deterministic non-CP-SAT baselines for optimizer quality comparisons."""

from __future__ import annotations

import itertools
from collections.abc import Mapping

from pc_build_recommender.domain import BuildProfile, ComponentCategory

from .engine import _solution_from_selected
from .models import (
    REQUIRED_CATEGORIES,
    OptimizationCandidate,
    OptimizationProblem,
    OptimizationResult,
    OptimizationStatus,
)
from .scoring import selected_objective_value
from .validation import eligible_candidates, validate_selected_build


def _ranked(
    problem: OptimizationProblem, category: ComponentCategory, profile: BuildProfile
) -> tuple[OptimizationCandidate, ...]:
    from .scoring import candidate_objective_coefficient

    return tuple(
        sorted(
            (
                candidate
                for candidate in eligible_candidates(problem)
                if candidate.category == category
            ),
            key=lambda candidate: (
                -candidate_objective_coefficient(problem, candidate, profile),
                candidate.price_cents,
                candidate.product_id,
            ),
        )
    )


def greedy_solution(
    problem: OptimizationProblem, profile: BuildProfile | None = None, *, local_width: int = 2
):
    """Select a feasible build without invoking OR-Tools.

    The first pass is greedy by category. A bounded neighbourhood of the top
    ``local_width`` candidates per category then repairs choices that made the
    complete build infeasible. Every returned build is independently validated.
    """
    selected_profile = BuildProfile(profile or problem.profiles[0])
    if local_width < 1:
        raise ValueError("local_width must be positive")
    categories = tuple(REQUIRED_CATEGORIES)
    ranked = {category: _ranked(problem, category, selected_profile) for category in categories}
    if any(not values for values in ranked.values()):
        return None
    greedy = {category: values[0] for category, values in ranked.items()}
    if not validate_selected_build(problem, greedy):
        return _make_solution(problem, greedy, selected_profile)
    choices = [values[:local_width] for values in ranked.values()]
    best = None
    for combination in itertools.product(*choices):
        candidate_selection = dict(zip(categories, combination, strict=True))
        if validate_selected_build(problem, candidate_selection):
            continue
        score = selected_objective_value(problem, combination, selected_profile)
        if best is None or score > best[0]:
            best = (score, candidate_selection)
    return None if best is None else _make_solution(problem, best[1], selected_profile)


def _make_solution(
    problem: OptimizationProblem,
    selected: Mapping[ComponentCategory, OptimizationCandidate],
    profile: BuildProfile,
):
    objective = selected_objective_value(problem, selected.values(), profile)
    return _solution_from_selected(
        problem,
        selected,
        profile=profile,
        objective_value=objective,
        solver_status=OptimizationStatus.FEASIBLE,
    )


def greedy_optimize(problem: OptimizationProblem, *, local_width: int = 2) -> OptimizationResult:
    solution = greedy_solution(problem, local_width=local_width)
    if solution is None:
        return OptimizationResult(status=OptimizationStatus.INFEASIBLE, solutions=())
    return OptimizationResult(status=OptimizationStatus.FEASIBLE, solutions=(solution,))
