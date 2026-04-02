"""Small-catalogue exhaustive oracle used to validate CP-SAT optimality."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

from pc_build_recommender.domain import BuildPreset

from .engine import _solution_from_selected
from .models import (
    REQUIRED_CATEGORIES,
    OptimizationProblem,
    OptimizationSolution,
    OptimizationStatus,
)
from .scoring import selected_objective_value
from .validation import eligible_candidates, normalise_validator_result, validate_selected_build


@dataclass(frozen=True, slots=True)
class ExhaustiveResult:
    """All feasible solutions in descending objective order for a reduced catalogue."""

    profile: BuildPreset
    solutions: tuple[OptimizationSolution, ...]
    combinations_evaluated: int

    @property
    def best(self) -> OptimizationSolution | None:
        return self.solutions[0] if self.solutions else None


def enumerate_feasible_builds(
    problem: OptimizationProblem,
    *,
    profile: BuildPreset | None = None,
    max_combinations: int = 1_000_000,
) -> ExhaustiveResult:
    """Enumerate a deliberately small catalogue as an independent optimiser oracle."""

    selected_profile = BuildPreset(profile or problem.profiles[0])
    eligible = eligible_candidates(problem)
    grouped = tuple(
        tuple(candidate for candidate in eligible if candidate.category == category)
        for category in REQUIRED_CATEGORIES
    )
    if any(not candidates for candidates in grouped):
        return ExhaustiveResult(selected_profile, (), 0)
    combination_count = math.prod(len(candidates) for candidates in grouped)
    if combination_count > max_combinations:
        raise ValueError(
            f"exhaustive enumeration would inspect {combination_count} combinations; "
            f"limit is {max_combinations}"
        )

    solutions: list[OptimizationSolution] = []
    evaluated = 0
    for combination in itertools.product(*grouped):
        evaluated += 1
        selected = {candidate.category: candidate for candidate in combination}
        if validate_selected_build(problem, selected):
            continue
        if problem.independent_validator is not None:
            result = problem.independent_validator(selected)
            if normalise_validator_result(result):
                continue
        objective = selected_objective_value(problem, combination, selected_profile)
        solutions.append(
            _solution_from_selected(
                problem,
                selected,
                profile=selected_profile,
                objective_value=objective,
                solver_status=OptimizationStatus.OPTIMAL,
            )
        )
    solutions.sort(
        key=lambda solution: (
            -solution.objective_value,
            solution.total_price_cents,
            tuple(sorted(solution.product_ids)),
        )
    )
    return ExhaustiveResult(selected_profile, tuple(solutions), evaluated)
