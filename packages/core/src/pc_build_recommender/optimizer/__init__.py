"""Constraint-based complete-build generation public API."""

from .adapters import candidate_from_domain, money_to_cents, problem_from_domain, solution_to_domain
from .engine import BuildOptimizer
from .exhaustive import ExhaustiveResult, enumerate_feasible_builds
from .models import (
    DEFAULT_POWER_ALLOWANCES_WATTS,
    PROFILE_WEIGHTS,
    REQUIRED_CATEGORIES,
    CandidateScores,
    FeatureOperator,
    FeatureRequirement,
    OptimizationCandidate,
    OptimizationProblem,
    OptimizationResult,
    OptimizationSolution,
    OptimizationStatus,
    PairwiseCompatibility,
    ProfileSolveRecord,
    ProfileWeights,
)
from .scoring import candidate_objective_coefficient, selected_objective_value
from .validation import (
    candidate_eligibility_reasons,
    diagnose_problem,
    eligible_candidates,
    estimated_load_watts,
    required_psu_watts,
    validate_selected_build,
)

__all__ = [
    "DEFAULT_POWER_ALLOWANCES_WATTS",
    "PROFILE_WEIGHTS",
    "REQUIRED_CATEGORIES",
    "BuildOptimizer",
    "CandidateScores",
    "ExhaustiveResult",
    "FeatureOperator",
    "FeatureRequirement",
    "OptimizationCandidate",
    "OptimizationProblem",
    "OptimizationResult",
    "OptimizationSolution",
    "OptimizationStatus",
    "PairwiseCompatibility",
    "ProfileSolveRecord",
    "ProfileWeights",
    "candidate_eligibility_reasons",
    "candidate_from_domain",
    "candidate_objective_coefficient",
    "diagnose_problem",
    "eligible_candidates",
    "enumerate_feasible_builds",
    "estimated_load_watts",
    "money_to_cents",
    "problem_from_domain",
    "required_psu_watts",
    "selected_objective_value",
    "solution_to_domain",
    "validate_selected_build",
]
