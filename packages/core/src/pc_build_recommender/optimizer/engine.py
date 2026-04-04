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

# TODO: rest of this module still to come.
