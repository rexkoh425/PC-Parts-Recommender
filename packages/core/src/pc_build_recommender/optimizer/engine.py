"""OR-Tools CP-SAT implementation for complete, diverse PC builds."""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from typing import Any

from ortools.sat.python import cp_model

from pc_build_recommender.domain import BuildPreset, CompatVerdict, ComponentKind

from .models import (
    REQUIRED_CATEGORIES,
    OptimizationCandidate,
    OptimizationProblem,
    OptimizationResult,
    OptimizationSolution,
    OptimizationStatus,
    ProfileSolveRecord,
)
from .scoring import candidate_objective_coefficient, candidate_power_watts, pair_penalty
from .validation import (
    _connectors_satisfy,
    diagnose_problem,
    eligible_candidates,
    estimated_load_watts,
    normalise_validator_result,
    required_psu_watts,
    validate_selected_build,
)


def _solver_status(status: cp_model.CpSolverStatus) -> OptimizationStatus:
    if status == cp_model.OPTIMAL:
        return OptimizationStatus.OPTIMAL
    if status == cp_model.FEASIBLE:
        return OptimizationStatus.FEASIBLE
    if status == cp_model.INFEASIBLE:
        return OptimizationStatus.INFEASIBLE
    if status == cp_model.MODEL_INVALID:
        return OptimizationStatus.MODEL_INVALID
    return OptimizationStatus.UNKNOWN


def _warning_messages(
    problem: OptimizationProblem,
    product_ids: set[str],
) -> tuple[str, ...]:
    messages = []
    for pair in problem.pairwise_compatibility:
        if pair.key <= product_ids and (
            pair.status == CompatVerdict.WARNING
            or (pair.status == CompatVerdict.UNKNOWN and not pair.hard)
        ):
            messages.append(
                pair.message
                or f"{pair.status.value} compatibility for "
                f"{pair.left_product_id}/{pair.right_product_id}"
            )
    return tuple(dict.fromkeys(messages))


def _solution_from_selected(
    problem: OptimizationProblem,
    selected: Mapping[ComponentKind, OptimizationCandidate],
    *,
    profile: BuildPreset,
    objective_value: int,
    solver_status: OptimizationStatus,
    compatibility_report: object | None = None,
) -> OptimizationSolution:
    load = estimated_load_watts(problem, selected)
    return OptimizationSolution(
        profile=profile,
        selected=dict(selected),
        total_price_cents=sum(
            problem.acquisition_price_cents(candidate) for candidate in selected.values()
        ),
        catalog_total_price_cents=sum(candidate.price_cents for candidate in selected.values()),
        objective_value=objective_value,
        estimated_load_watts=load,
        required_psu_watts=required_psu_watts(load, problem.power_headroom_percent),
        solver_status=solver_status,
        warnings=_warning_messages(
            problem, {candidate.product_id for candidate in selected.values()}
        ),
        compatibility_report=compatibility_report,
    )


class _CpModelState:
    def __init__(
        self,
        problem: OptimizationProblem,
        candidates: Sequence[OptimizationCandidate],
        previous_solutions: Sequence[OptimizationSolution],
    ) -> None:
        self.problem = problem
        self.candidates = tuple(candidates)
        self.model = cp_model.CpModel()
        self.variables = {
            candidate.product_id: self.model.new_bool_var(f"select_{candidate.product_id}")
            for candidate in self.candidates
        }
        self.warning_variables: list[tuple[cp_model.IntVar, int]] = []
        self._add_cardinality()
        self._add_locks()
        self._add_budget()
        self._add_pairwise_compatibility()
        self._add_power_and_connectors()
        self._add_diversity(previous_solutions)

    def _add_cardinality(self) -> None:
        for category in REQUIRED_CATEGORIES:
            category_vars = [
                self.variables[candidate.product_id]
                for candidate in self.candidates
                if candidate.category == category
            ]
            self.model.add_exactly_one(category_vars)

    def _add_locks(self) -> None:
        for product_id in self.problem.locked_product_ids:
            variable = self.variables.get(product_id)
            if variable is not None:
                self.model.add(variable == 1)

    def _add_budget(self) -> None:
        self.model.add(
            sum(
                self.problem.acquisition_price_cents(candidate)
                * self.variables[candidate.product_id]
                for candidate in self.candidates
            )
            <= self.problem.budget_cents
        )

    def _add_pairwise_compatibility(self) -> None:
        for index, pair in enumerate(self.problem.pairwise_compatibility):
            left = self.variables.get(pair.left_product_id)
            right = self.variables.get(pair.right_product_id)
            if left is None or right is None:
                continue
            if pair.is_forbidden:
                self.model.add(left + right <= 1)
                continue
            penalty = pair_penalty(pair)
            if penalty:
                both = self.model.new_bool_var(f"warning_pair_{index}")
                self.model.add(both <= left)
                self.model.add(both <= right)
                self.model.add(both >= left + right - 1)
                self.warning_variables.append((both, penalty))

    def _add_power_and_connectors(self) -> None:
        load_expression = self.problem.base_power_watts + sum(
            candidate_power_watts(self.problem, candidate) * self.variables[candidate.product_id]
            for candidate in self.candidates
            if candidate.category != ComponentKind.POWER_SUPPLY
        )
        capacity_expression = sum(
            int(candidate.psu_wattage or 0) * self.variables[candidate.product_id]
            for candidate in self.candidates
            if candidate.category == ComponentKind.POWER_SUPPLY
        )
        self.model.add(
            load_expression * (100 + self.problem.power_headroom_percent)
            <= capacity_expression * 100
        )

        gpus = [
            candidate
            for candidate in self.candidates
            if candidate.category == ComponentKind.GPU
        ]
        power_supplies = [
            candidate
            for candidate in self.candidates
            if candidate.category == ComponentKind.POWER_SUPPLY
        ]
        for gpu in gpus:
            for psu in power_supplies:
                connectors_ok = _connectors_satisfy(
                    gpu.required_power_connectors,
                    psu.provided_power_connectors,
                )
                recommendation_ok = (
                    gpu.recommended_psu_watts is None
                    or psu.psu_wattage is not None
                    and psu.psu_wattage >= gpu.recommended_psu_watts
                )
                if not connectors_ok or not recommendation_ok:
                    self.model.add(
                        self.variables[gpu.product_id] + self.variables[psu.product_id] <= 1
                    )

    def _add_diversity(self, previous_solutions: Sequence[OptimizationSolution]) -> None:
        if not previous_solutions:
            return
        meaningful = tuple(
            category
            for category in self.problem.meaningful_categories
            if all(
                candidate.category != category
                for candidate in self.problem.candidates
                if candidate.product_id in self.problem.locked_product_ids
            )
        )
        for solution in previous_solutions:
            prior_vars = [
                self.variables[solution.selected[category].product_id]
                for category in meaningful
                if solution.selected[category].product_id in self.variables
            ]
            self.model.add(sum(prior_vars) <= len(prior_vars) - self.problem.diversity_distance)

    def set_objective(self, profile: BuildPreset) -> None:
        terms: list[Any] = [
            candidate_objective_coefficient(self.problem, candidate, profile)
            * self.variables[candidate.product_id]
            for candidate in self.candidates
        ]
        terms.extend(-penalty * variable for variable, penalty in self.warning_variables)
        self.model.maximize(sum(terms))

    def forbid_exact_solution(
        self, selected: Mapping[ComponentKind, OptimizationCandidate]
    ) -> None:
        self.model.add(
            sum(self.variables[candidate.product_id] for candidate in selected.values())
            <= len(REQUIRED_CATEGORIES) - 1
        )

# TODO: rest of this module still to come.
