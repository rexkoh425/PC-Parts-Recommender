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


class BuildOptimizer:
    """Generate objective-profile builds with fail-closed validation and diversity."""

    def __init__(self, *, max_validator_rejections: int = 50) -> None:
        if max_validator_rejections < 1:
            raise ValueError("max_validator_rejections must be positive")
        self.max_validator_rejections = max_validator_rejections

    def optimize(
        self,
        problem: OptimizationProblem,
        *,
        max_solutions: int | None = None,
    ) -> OptimizationResult:
        """Solve for up to ``max_solutions`` diverse, independently validated builds."""

        requested_count = len(problem.profiles) if max_solutions is None else max_solutions
        if requested_count < 1:
            raise ValueError("max_solutions must be positive")
        precheck_reasons = list(diagnose_problem(problem))
        eligible = eligible_candidates(problem)
        if precheck_reasons:
            return OptimizationResult(
                status=OptimizationStatus.INFEASIBLE,
                solutions=(),
                infeasibility_reasons=tuple(precheck_reasons),
            )

        profiles = tuple(itertools.islice(itertools.cycle(problem.profiles), requested_count))
        solutions: list[OptimizationSolution] = []
        records: list[ProfileSolveRecord] = []
        validator_reasons: list[str] = []
        rejected_by_validator = 0

        unlocked_meaningful_count = sum(
            1
            for category in problem.meaningful_categories
            if not any(
                candidate.category == category
                for candidate in problem.candidates
                if candidate.product_id in problem.locked_product_ids
            )
        )
        for profile_index, profile in enumerate(profiles):
            if profile_index > 0 and unlocked_meaningful_count < problem.diversity_distance:
                precheck_reasons.append(
                    "fewer than two unlocked meaningful categories are available "
                    "for a diverse build"
                )
                break

            state = _CpModelState(problem, eligible, solutions)
            state.set_objective(profile)
            final_status = OptimizationStatus.UNKNOWN
            final_wall_time = 0.0
            final_objective: int | None = None
            accepted: OptimizationSolution | None = None

            for _ in range(self.max_validator_rejections + 1):
                solver = cp_model.CpSolver()
                solver.parameters.max_time_in_seconds = problem.time_limit_seconds
                solver.parameters.num_search_workers = 1
                solver.parameters.random_seed = problem.random_seed
                status_code = solver.solve(state.model)
                final_status = _solver_status(status_code)
                final_wall_time += solver.wall_time
                if final_status not in (OptimizationStatus.OPTIMAL, OptimizationStatus.FEASIBLE):
                    break

                selected = {
                    candidate.category: candidate
                    for candidate in eligible
                    if solver.boolean_value(state.variables[candidate.product_id])
                }
                objective_value = round(solver.objective_value)
                validation_errors = list(validate_selected_build(problem, selected))
                validator_result: object | None = None
                if not validation_errors and problem.independent_validator is not None:
                    try:
                        validator_result = problem.independent_validator(selected)
                    except Exception as exc:  # defensive boundary around a caller-supplied hook
                        validation_errors = [
                            f"independent validator raised {type(exc).__name__}: {exc}"
                        ]
                    else:
                        validation_errors = list(normalise_validator_result(validator_result))
                if not validation_errors:
                    accepted = _solution_from_selected(
                        problem,
                        selected,
                        profile=profile,
                        objective_value=objective_value,
                        solver_status=final_status,
                        compatibility_report=validator_result,
                    )
                    final_objective = objective_value
                    break

                rejected_by_validator += 1
                validator_reasons.extend(validation_errors)
                state.forbid_exact_solution(selected)
                if rejected_by_validator >= self.max_validator_rejections:
                    break

            records.append(
                ProfileSolveRecord(
                    profile=profile,
                    status=final_status,
                    wall_time_seconds=final_wall_time,
                    objective_value=final_objective,
                )
            )
            if accepted is None:
                if final_status == OptimizationStatus.INFEASIBLE and solutions:
                    precheck_reasons.append(
                        f"no additional build can differ by at least "
                        f"{problem.diversity_distance} meaningful components"
                    )
                break
            solutions.append(accepted)

        reasons = list(dict.fromkeys((*precheck_reasons, *validator_reasons)))
        if not solutions and not reasons:
            reasons.append(
                "no combination satisfies all budget, compatibility, and power constraints"
            )
        if solutions and len(solutions) < requested_count and not reasons:
            reasons.append(
                f"only {len(solutions)} of {requested_count} requested builds are feasible"
            )

        if not solutions:
            final_result_status = records[-1].status if records else OptimizationStatus.INFEASIBLE
            if final_result_status == OptimizationStatus.OPTIMAL:
                final_result_status = OptimizationStatus.INFEASIBLE
        elif len(solutions) == requested_count and all(
            record.status == OptimizationStatus.OPTIMAL for record in records
        ):
            final_result_status = OptimizationStatus.OPTIMAL
        else:
            final_result_status = OptimizationStatus.FEASIBLE

        return OptimizationResult(
            status=final_result_status,
            solutions=tuple(solutions),
            infeasibility_reasons=tuple(reasons),
            profile_statuses=tuple(records),
            rejected_by_validator=rejected_by_validator,
        )

    # A readable alias for application services.
    generate = optimize
