"""Bounded, auditable evaluation of builds actually returned by CP-SAT.

This module is deliberately separate from the generated compatibility-rule evaluation.
Its evidence unit is an :class:`OptimizationSolution` returned by :class:`BuildOptimizer`,
not an arbitrary configuration passed directly to the compatibility engine.

The post-solve oracle below does not call ``validate_selected_build`` or inspect the CP-SAT
model.  It independently re-evaluates every hard constraint from the immutable problem and
solution contracts.  Compact, self-hashed records retain enough information to reproduce a
scenario from its deterministic seed and audit exactly which products were returned.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from importlib.metadata import version as distribution_version
from pathlib import Path

from pc_build_recommender.compatibility import (
    DEFAULT_RULE_VERSION,
    CompatibilityEngine,
)
from pc_build_recommender.domain import BuildProfile, CompatVerdict, ComponentCategory

from .engine import BuildOptimizer
from .models import (
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
)

ARTIFACT_SCHEMA_VERSION = "pc-build-recommender.optimizer-evaluation.v1"
GENERATOR_VERSION = "deterministic-generated-optimizer-requests.v1"
CLAIM_VALID_BUILD_TARGET = 10_000
MAX_SCENARIO_COUNT = 10_000
MAX_CANDIDATES_PER_CATEGORY = 4
MAX_SOLUTIONS_PER_SCENARIO = 5
MAX_RETAINED_OUTPUT_RECORDS = MAX_SCENARIO_COUNT * MAX_SOLUTIONS_PER_SCENARIO
MAX_SOLVER_TIME_LIMIT_SECONDS = 5.0
DEFAULT_OUTPUT_DIR = Path("artifacts/evaluation/optimizer-generated-builds-v1")
_COMPATIBILITY_ENGINE = CompatibilityEngine(rule_version=DEFAULT_RULE_VERSION)


class OptimizerEvaluationError(RuntimeError):
    """Raised when an optimizer evaluation artifact is internally inconsistent."""


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _source_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _current_source_sha256() -> dict[str, str]:
    """Return the exact source-byte digests required by optimizer evidence."""

    module_path = Path(__file__).resolve()
    source_paths = {
        "optimizer_engine": module_path.with_name("engine.py"),
        "compatibility_engine": module_path.parent.parent.joinpath("compatibility", "engine.py"),
        "evaluation_harness_and_independent_oracle": module_path,
    }
    return {name: _source_sha256(path) for name, path in source_paths.items()}


def _verify_source_sha256(payload: Mapping[str, object]) -> None:
    source_hashes = payload.get("source_sha256")
    if not isinstance(source_hashes, Mapping):
        raise OptimizerEvaluationError("source_sha256 must be an object")
    expected_source_hashes = _current_source_sha256()
    if set(source_hashes) != set(expected_source_hashes):
        raise OptimizerEvaluationError(
            "source_sha256 must contain exactly the required implementation sources"
        )
    if not all(_valid_sha256(value) for value in source_hashes.values()):
        raise OptimizerEvaluationError("source_sha256 must contain valid SHA-256 digests")
    mismatched_sources = sorted(
        name
        for name, expected_digest in expected_source_hashes.items()
        if source_hashes[name] != expected_digest
    )
    if mismatched_sources:
        raise OptimizerEvaluationError(
            "source_sha256 does not match current implementation sources: "
            + ", ".join(mismatched_sources)
        )


@dataclass(frozen=True, slots=True)
class OptimizerEvaluationConfig:
    """Hard-bounded controls for deterministic optimizer evaluation."""

    scenario_count: int = 100
    seed: int = 20_260_723
    candidates_per_category: int = 2
    solutions_per_scenario: int = 1
    solver_time_limit_seconds: float = 0.5
    infeasible_every: int | None = None

    def __post_init__(self) -> None:
        if not 1 <= self.scenario_count <= MAX_SCENARIO_COUNT:
            raise ValueError(f"scenario_count must be between 1 and {MAX_SCENARIO_COUNT}")
        if not 0 <= self.seed <= 2_147_483_647:
            raise ValueError("seed must be between 0 and 2^31-1")
        if not 1 <= self.candidates_per_category <= MAX_CANDIDATES_PER_CATEGORY:
            raise ValueError(
                f"candidates_per_category must be between 1 and {MAX_CANDIDATES_PER_CATEGORY}"
            )
        if not 1 <= self.solutions_per_scenario <= MAX_SOLUTIONS_PER_SCENARIO:
            raise ValueError(
                f"solutions_per_scenario must be between 1 and {MAX_SOLUTIONS_PER_SCENARIO}"
            )
        if self.solutions_per_scenario > 1 and self.candidates_per_category < 2:
            raise ValueError("multiple solutions require at least two candidates per category")
        if self.scenario_count * self.solutions_per_scenario > MAX_RETAINED_OUTPUT_RECORDS:
            raise ValueError(f"the run may retain at most {MAX_RETAINED_OUTPUT_RECORDS} outputs")
        if not 0 < self.solver_time_limit_seconds <= MAX_SOLVER_TIME_LIMIT_SECONDS:
            raise ValueError(
                "solver_time_limit_seconds must be greater than zero and at most "
                f"{MAX_SOLVER_TIME_LIMIT_SECONDS}"
            )
        if self.infeasible_every is not None and self.infeasible_every < 1:
            raise ValueError("infeasible_every must be positive when supplied")


@dataclass(frozen=True, slots=True)
class ClaimAssessment:
    """Machine-checkable assessment for the narrow 10,000-build engineering claim."""

    eligible: bool
    blockers: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "claim": (
                "OR-Tools generated at least 10,000 complete PC builds that were retained "
                "and independently revalidated"
            ),
            "eligible": self.eligible,
            "required_valid_build_count": CLAIM_VALID_BUILD_TARGET,
            "blockers": list(self.blockers),
            "scope": (
                "Engineering evidence from deterministic generated optimizer requests "
                f"independently rechecked with {DEFAULT_RULE_VERSION}; not evidence of "
                "10,000 observed customer or market builds."
            ),
        }


def assess_10k_valid_build_claim(
    *,
    optimizer_output_count: int,
    independently_checked_output_count: int,
    independently_valid_output_count: int,
    unique_output_count: int,
    retained_output_record_count: int,
    invalid_output_count: int,
    evaluation_passed: bool,
    records_verified: bool,
) -> ClaimAssessment:
    """Fail closed unless every returned output is retained, checked, and valid."""

    counts = (
        optimizer_output_count,
        independently_checked_output_count,
        independently_valid_output_count,
        unique_output_count,
        retained_output_record_count,
        invalid_output_count,
    )
    if any(type(value) is not int or value < 0 for value in counts):
        raise ValueError("claim evidence counts must be non-negative integers")
    blockers: list[str] = []
    if not evaluation_passed:
        blockers.append("the optimizer evaluation did not pass")
    if not records_verified:
        blockers.append("retained scenario and output record hashes were not verified")
    if optimizer_output_count < CLAIM_VALID_BUILD_TARGET:
        blockers.append(
            f"only {optimizer_output_count} optimizer outputs were returned; "
            f"{CLAIM_VALID_BUILD_TARGET} are required"
        )
    if independently_checked_output_count != optimizer_output_count:
        blockers.append("not every optimizer output was independently checked")
    if independently_valid_output_count != optimizer_output_count or invalid_output_count:
        blockers.append("one or more optimizer outputs failed independent validation")
    if unique_output_count < CLAIM_VALID_BUILD_TARGET:
        blockers.append(
            f"only {unique_output_count} unique product-ID build tuples were retained; "
            f"{CLAIM_VALID_BUILD_TARGET} are required"
        )
    if retained_output_record_count != optimizer_output_count:
        blockers.append("not every optimizer output has a retained evidence record")
    return ClaimAssessment(eligible=not blockers, blockers=tuple(blockers))


@dataclass(frozen=True, slots=True)
class GeneratedOptimizerEvaluation:
    """Content-addressed report containing compact per-scenario evidence."""

    payload: Mapping[str, object]
    artifact_sha256: str

    def to_dict(self) -> dict[str, object]:
        return {**self.payload, "artifact_sha256": self.artifact_sha256}

    def verify(self) -> None:
        _verify_report(self.payload, self.artifact_sha256)


def _normalise_scalar(value: object) -> object:
    if isinstance(value, Enum):
        value = value.value
    return value.casefold() if isinstance(value, str) else value


def _connector_key(value: str) -> str:
    return value.strip().casefold().replace("-", "_").replace(" ", "_")


def _connectors_satisfy_independently(
    required: Mapping[str, int], provided: Mapping[str, int]
) -> bool:
    available = {_connector_key(name): int(count) for name, count in provided.items()}
    return all(
        available.get(_connector_key(name), 0) >= int(count) for name, count in required.items()
    )


def _feature_matches_independently(
    candidate: OptimizationCandidate, requirement: FeatureRequirement
) -> bool:
    value = candidate.attributes.get(requirement.attribute)
    expected = requirement.expected
    if value is None:
        return False
    operator = FeatureOperator(requirement.operator)
    if operator is FeatureOperator.TRUTHY:
        return bool(value)
    if operator is FeatureOperator.EQUALS:
        return _normalise_scalar(value) == _normalise_scalar(expected)
    if operator is FeatureOperator.AT_LEAST:
        try:
            return bool(value >= expected)
        except TypeError:
            return False
    if operator is FeatureOperator.CONTAINS:
        try:
            return _normalise_scalar(expected) in {_normalise_scalar(item) for item in value}
        except TypeError:
            return False
    return False


def _candidate_errors_independently(
    problem: OptimizationProblem, candidate: OptimizationCandidate
) -> list[str]:
    errors: list[str] = []
    locked = candidate.product_id in problem.locked_product_ids
    if problem.in_stock_only and not locked and not candidate.in_stock:
        errors.append("candidate is not in stock")
    if candidate.brand.casefold() in problem.excluded_brands:
        errors.append("candidate brand is excluded")

    if candidate.category is ComponentCategory.CPU and candidate.power_draw_watts is None:
        errors.append("CPU peak power is unknown")
    elif candidate.category is ComponentCategory.GPU:
        vram = candidate.attributes.get("vram_gb")
        if problem.minimum_gpu_vram_gb is not None and (
            vram is None or int(vram) < problem.minimum_gpu_vram_gb
        ):
            errors.append("GPU VRAM does not satisfy the minimum")
        if candidate.power_draw_watts is None:
            errors.append("GPU board power is unknown")
    elif candidate.category is ComponentCategory.MEMORY:
        capacity = candidate.attributes.get("capacity_gb")
        if problem.minimum_memory_gb is not None and (
            capacity is None or int(capacity) < problem.minimum_memory_gb
        ):
            errors.append("memory capacity does not satisfy the minimum")
        if problem.required_memory_type is not None and _normalise_scalar(
            candidate.attributes.get("memory_type")
        ) != _normalise_scalar(problem.required_memory_type):
            errors.append("memory type does not satisfy the requirement")
    elif candidate.category is ComponentCategory.STORAGE:
        capacity = candidate.attributes.get("capacity_gb")
        if problem.minimum_storage_gb is not None and (
            capacity is None or int(capacity) < problem.minimum_storage_gb
        ):
            errors.append("storage capacity does not satisfy the minimum")
    elif candidate.category is ComponentCategory.MOTHERBOARD:
        if problem.wifi_required and candidate.attributes.get("wifi_support") is not True:
            errors.append("motherboard Wi-Fi support is absent or unknown")
        if problem.required_memory_type is not None and _normalise_scalar(
            candidate.attributes.get("memory_type")
        ) != _normalise_scalar(problem.required_memory_type):
            errors.append("motherboard memory type does not satisfy the requirement")
        if problem.required_motherboard_form_factor is not None and _normalise_scalar(
            candidate.attributes.get("form_factor")
        ) != _normalise_scalar(problem.required_motherboard_form_factor):
            errors.append("motherboard form factor does not satisfy the requirement")
    elif (
        candidate.category is ComponentCategory.CASE
        and problem.required_case_size is not None
        and _normalise_scalar(candidate.attributes.get("case_size"))
        != _normalise_scalar(problem.required_case_size)
    ):
        errors.append("case size does not satisfy the requirement")
    elif candidate.category is ComponentCategory.POWER_SUPPLY:
        if candidate.psu_wattage is None:
            errors.append("PSU wattage is unknown")
        if candidate.eps_connectors is None:
            errors.append("PSU EPS connector count is unknown")
        elif candidate.eps_connectors < problem.required_eps_connectors:
            errors.append("PSU EPS connector count is insufficient")

    for requirement in problem.required_features:
        if requirement.category is candidate.category and not _feature_matches_independently(
            candidate, requirement
        ):
            errors.append(f"required feature {requirement.attribute!r} is not satisfied")
    return errors


def _candidate_power_watts_independently(
    problem: OptimizationProblem, candidate: OptimizationCandidate
) -> int:
    if candidate.category is ComponentCategory.POWER_SUPPLY:
        return 0
    if candidate.power_draw_watts is not None:
        return int(candidate.power_draw_watts)
    return int(problem.category_power_allowances_watts.get(candidate.category, 0))


def _objective_value_independently(
    problem: OptimizationProblem, solution: OptimizationSolution
) -> int:
    weights = PROFILE_WEIGHTS[solution.profile]
    total = 0
    for candidate in solution.selected.values():
        scores = candidate.scores
        weighted_score = (
            weights.performance * scores.performance
            + weights.value * scores.value
            + weights.reliability * scores.reliability
            + weights.upgradeability * scores.upgradeability
            + weights.efficiency * scores.efficiency
            + weights.preference * scores.preference
        )
        coefficient = round(weighted_score * 100) - round(scores.warning_penalty * 100)
        if weights.price_penalty_divisor is not None:
            acquisition_price = (
                0
                if problem.exclude_locked_from_budget
                and candidate.product_id in problem.locked_product_ids
                else candidate.price_cents
            )
            coefficient -= acquisition_price // weights.price_penalty_divisor
        if weights.power_penalty_per_watt:
            coefficient -= (
                _candidate_power_watts_independently(problem, candidate)
                * weights.power_penalty_per_watt
            )
        total += coefficient

    selected_ids = {candidate.product_id for candidate in solution.selected.values()}
    total -= sum(
        pair.penalty_points * 100
        for pair in problem.pairwise_compatibility
        if {pair.left_product_id, pair.right_product_id} <= selected_ids
        and (
            pair.status is CompatVerdict.WARNING
            or (pair.status is CompatVerdict.UNKNOWN and not pair.hard)
        )
    )
    return total


def _warning_messages_independently(
    problem: OptimizationProblem, selected_ids: set[str]
) -> tuple[str, ...]:
    messages = [
        pair.message
        or (f"{pair.status.value} compatibility for {pair.left_product_id}/{pair.right_product_id}")
        for pair in problem.pairwise_compatibility
        if {pair.left_product_id, pair.right_product_id} <= selected_ids
        and (
            pair.status is CompatVerdict.WARNING
            or (pair.status is CompatVerdict.UNKNOWN and not pair.hard)
        )
    ]
    return tuple(dict.fromkeys(messages))


def _compatibility_component(candidate: OptimizationCandidate) -> dict[str, object]:
    component = {
        "product_id": candidate.product_id,
        "category": candidate.category.value,
        "status": "active",
        **candidate.attributes,
    }
    if candidate.category is ComponentCategory.CPU:
        component["peak_power_w"] = candidate.power_draw_watts
    elif candidate.category is ComponentCategory.GPU:
        component["board_power_w"] = candidate.power_draw_watts
        component["required_power_connectors"] = dict(candidate.required_power_connectors)
    elif candidate.category is ComponentCategory.POWER_SUPPLY:
        component["wattage"] = candidate.psu_wattage
        component["pcie_connectors"] = dict(candidate.provided_power_connectors)
        component["eps_connectors"] = candidate.eps_connectors
    return component


def _versioned_compatibility_errors(
    selected: Mapping[ComponentCategory, OptimizationCandidate],
) -> tuple[str, ...]:
    components = {
        category.value: _compatibility_component(candidate)
        for category, candidate in selected.items()
    }
    report = _COMPATIBILITY_ENGINE.check_complete_build(components)
    errors: list[str] = []
    if not report.results:
        errors.append("versioned compatibility engine returned no rule results")
        return tuple(errors)
    if report.rule_version != DEFAULT_RULE_VERSION or any(
        result.rule_version != DEFAULT_RULE_VERSION for result in report.results
    ):
        errors.append("versioned compatibility results used an unexpected rule version")
    observed_rule_ids = {result.rule_id for result in report.results}
    required_cardinality_rules = {
        f"compat.build.cardinality.{category.value}" for category in REQUIRED_CATEGORIES
    }
    if not required_cardinality_rules <= observed_rule_ids:
        errors.append("versioned compatibility report omitted required cardinality rules")
    if report.has_failures:
        failures = sorted(
            result.rule_id for result in report.results if result.status.value == "FAIL"
        )
        errors.append(f"versioned compatibility engine reported failures: {failures}")
    if report.has_unknowns:
        unknowns = sorted(
            result.rule_id for result in report.results if result.status.value == "UNKNOWN"
        )
        errors.append(f"versioned compatibility engine reported unknowns: {unknowns}")
    return tuple(errors)


def _versioned_compatibility_evidence(
    selected: Mapping[ComponentCategory, OptimizationCandidate],
) -> dict[str, object]:
    components = {
        category.value: _compatibility_component(candidate)
        for category, candidate in selected.items()
    }
    report = _COMPATIBILITY_ENGINE.check_complete_build(components)
    return {
        "rule_version": report.rule_version,
        "rule_result_count": len(report.results),
        "status_counts": dict(report.status_counts),
        "report_sha256": _sha256_json(report.to_dict()),
    }


def independently_validate_solution(
    problem: OptimizationProblem, solution: OptimizationSolution
) -> tuple[str, ...]:
    """Recheck a returned build without using optimizer validation helpers or CP-SAT state."""

    errors: list[str] = []
    selected = dict(solution.selected)
    required = set(REQUIRED_CATEGORIES)
    if set(selected) != required:
        missing = sorted(category.value for category in required - set(selected))
        extra = sorted(category.value for category in set(selected) - required)
        errors.append(
            f"selected category keys are incomplete or unexpected: missing={missing}, extra={extra}"
        )
        return tuple(errors)

    catalogue_by_id = {candidate.product_id: candidate for candidate in problem.candidates}
    selected_ids: set[str] = set()
    for category in REQUIRED_CATEGORIES:
        candidate = selected[category]
        if candidate.category is not category:
            errors.append(
                f"selected {category.value} key contains a {candidate.category.value} candidate"
            )
        if candidate.product_id in selected_ids:
            errors.append(f"product {candidate.product_id!r} was selected more than once")
        selected_ids.add(candidate.product_id)
        catalogue_candidate = catalogue_by_id.get(candidate.product_id)
        if catalogue_candidate is None:
            errors.append(f"product {candidate.product_id!r} is absent from the request catalogue")
        elif catalogue_candidate != candidate:
            errors.append(f"product {candidate.product_id!r} differs from the request candidate")
        errors.extend(
            f"{candidate.product_id}: {message}"
            for message in _candidate_errors_independently(problem, candidate)
        )

    missing_locks = problem.locked_product_ids - selected_ids
    if missing_locks:
        errors.append(f"locked products are missing: {sorted(missing_locks)}")

    acquisition_total = sum(
        0
        if problem.exclude_locked_from_budget and candidate.product_id in problem.locked_product_ids
        else candidate.price_cents
        for candidate in selected.values()
    )
    catalogue_total = sum(candidate.price_cents for candidate in selected.values())
    if acquisition_total > problem.budget_cents:
        errors.append("selected build exceeds the request budget")
    if solution.total_price_cents != acquisition_total:
        errors.append("reported acquisition total does not match selected candidates")
    if solution.catalog_total_price_cents != catalogue_total:
        errors.append("reported catalogue total does not match selected candidates")

    for pair in problem.pairwise_compatibility:
        forbidden = pair.status is CompatVerdict.FAIL or (
            pair.status is CompatVerdict.UNKNOWN and pair.hard
        )
        if forbidden and {pair.left_product_id, pair.right_product_id} <= selected_ids:
            errors.append(
                f"forbidden pair {pair.left_product_id}/{pair.right_product_id} was selected"
            )

    gpu = selected[ComponentCategory.GPU]
    psu = selected[ComponentCategory.POWER_SUPPLY]
    if not _connectors_satisfy_independently(
        gpu.required_power_connectors, psu.provided_power_connectors
    ):
        errors.append("PSU does not provide the GPU power connectors")
    if gpu.recommended_psu_watts is not None and (
        psu.psu_wattage is None or psu.psu_wattage < gpu.recommended_psu_watts
    ):
        errors.append("PSU wattage is below the GPU recommendation")

    estimated_load = problem.base_power_watts + sum(
        _candidate_power_watts_independently(problem, candidate)
        for candidate in selected.values()
        if candidate.category is not ComponentCategory.POWER_SUPPLY
    )
    required_psu = math.ceil(estimated_load * (100 + problem.power_headroom_percent) / 100)
    if psu.psu_wattage is None or psu.psu_wattage < required_psu:
        errors.append("PSU capacity is below independently calculated peak load plus headroom")
    if solution.estimated_load_watts != estimated_load:
        errors.append("reported estimated load does not match independent calculation")
    if solution.required_psu_watts != required_psu:
        errors.append("reported required PSU wattage does not match independent calculation")
    if solution.objective_value != _objective_value_independently(problem, solution):
        errors.append("reported objective value does not match independent calculation")
    expected_warnings = _warning_messages_independently(problem, selected_ids)
    if solution.warnings != expected_warnings:
        errors.append("reported warnings do not match selected soft compatibility pairs")
    if solution.profile not in problem.profiles:
        errors.append("solution profile was not requested")
    if solution.solver_status not in (OptimizationStatus.OPTIMAL, OptimizationStatus.FEASIBLE):
        errors.append("a returned solution has a non-feasible solver status")
    errors.extend(_versioned_compatibility_errors(selected))
    return tuple(dict.fromkeys(errors))


def _independent_result_errors(
    problem: OptimizationProblem,
    result: OptimizationResult,
    *,
    requested_solution_count: int,
) -> tuple[tuple[tuple[str, ...], ...], tuple[str, ...]]:
    per_solution = [
        list(independently_validate_solution(problem, solution)) for solution in result.solutions
    ]
    result_errors: list[str] = []
    if result.is_feasible is not bool(result.solutions):
        result_errors.append("result feasibility flag disagrees with returned solutions")
    expected_record_counts = {len(result.solutions), len(result.solutions) + 1}
    if len(result.profile_statuses) not in expected_record_counts:
        result_errors.append("profile solve record count is inconsistent with returned solutions")
    for record in result.profile_statuses:
        if not math.isfinite(record.wall_time_seconds) or record.wall_time_seconds < 0:
            result_errors.append("profile solve wall time is not finite and non-negative")
    for index, solution in enumerate(result.solutions):
        expected_profile = problem.profiles[index % len(problem.profiles)]
        if solution.profile is not expected_profile:
            per_solution[index].append(
                "solution profile does not follow the requested profile cycle"
            )
        if index >= len(result.profile_statuses):
            per_solution[index].append("solution has no corresponding profile solve record")
            continue
        record = result.profile_statuses[index]
        if record.profile is not solution.profile:
            per_solution[index].append("solution profile disagrees with profile solve record")
        if record.status is not solution.solver_status:
            per_solution[index].append("solution status disagrees with profile solve record")
        if record.objective_value != solution.objective_value:
            per_solution[index].append("solution objective disagrees with profile solve record")

    for left_index, left in enumerate(result.solutions):
        for right_index, right in enumerate(
            result.solutions[left_index + 1 :], start=left_index + 1
        ):
            meaningful = tuple(
                category
                for category in problem.meaningful_categories
                if not any(
                    candidate.category is category
                    and candidate.product_id in problem.locked_product_ids
                    for candidate in problem.candidates
                )
            )
            differences = sum(
                left.selected[category].product_id != right.selected[category].product_id
                for category in meaningful
            )
            if differences < problem.diversity_distance:
                message = (
                    f"solution diversity distance {differences} is below "
                    f"{problem.diversity_distance}"
                )
                per_solution[left_index].append(message)
                per_solution[right_index].append(message)

    if not result.solutions:
        expected_status = (
            result.profile_statuses[-1].status
            if result.profile_statuses
            else OptimizationStatus.INFEASIBLE
        )
        if expected_status is OptimizationStatus.OPTIMAL:
            expected_status = OptimizationStatus.INFEASIBLE
    elif len(result.solutions) == requested_solution_count and all(
        record.status is OptimizationStatus.OPTIMAL for record in result.profile_statuses
    ):
        expected_status = OptimizationStatus.OPTIMAL
    else:
        expected_status = OptimizationStatus.FEASIBLE
    if result.status is not expected_status:
        result_errors.append(
            f"aggregate result status {result.status.name} should be {expected_status.name}"
        )
    if result.rejected_by_validator != 0:
        result_errors.append("generated evaluation request unexpectedly used validator rejections")
    return (
        tuple(tuple(dict.fromkeys(errors)) for errors in per_solution),
        tuple(dict.fromkeys(result_errors)),
    )


_PRICE_RANGES: Mapping[ComponentCategory, tuple[int, int]] = {
    ComponentCategory.CPU: (15_000, 35_000),
    ComponentCategory.GPU: (35_000, 85_000),
    ComponentCategory.MOTHERBOARD: (12_000, 28_000),
    ComponentCategory.MEMORY: (8_000, 20_000),
    ComponentCategory.STORAGE: (6_000, 16_000),
    ComponentCategory.POWER_SUPPLY: (8_000, 18_000),
    ComponentCategory.COOLER: (3_000, 12_000),
    ComponentCategory.CASE: (5_000, 16_000),
}


def _scores(rng: random.Random) -> CandidateScores:
    return CandidateScores(
        performance=rng.randint(35, 100),
        value=rng.randint(35, 100),
        reliability=rng.randint(50, 100),
        upgradeability=rng.randint(35, 100),
        efficiency=rng.randint(35, 100),
        preference=rng.randint(35, 100),
        warning_penalty=rng.randint(0, 10),
    )


def _generated_candidate(
    *,
    scenario_index: int,
    category: ComponentCategory,
    candidate_index: int,
    rng: random.Random,
    minimum_gpu_vram_gb: int,
    minimum_memory_gb: int,
    minimum_storage_gb: int,
) -> OptimizationCandidate:
    low, high = _PRICE_RANGES[category]
    attributes: dict[str, object] = {}
    power_draw_watts: int | None = None
    psu_wattage: int | None = None
    required_power_connectors: Mapping[str, int] = {}
    provided_power_connectors: Mapping[str, int] = {}
    eps_connectors: int | None = None
    recommended_psu_watts: int | None = None
    if category is ComponentCategory.CPU:
        power_draw_watts = rng.randint(65, 125)
        attributes.update(
            socket="AM5",
            generation="Ryzen 7000",
            model="Generated CPU",
            supported_chipsets=["B650"],
            peak_power_w=power_draw_watts,
        )
    elif category is ComponentCategory.GPU:
        attributes["vram_gb"] = minimum_gpu_vram_gb + candidate_index * 4
        power_draw_watts = rng.randint(180, 320)
        required_power_connectors = {"8-pin PCIe": 1 + candidate_index % 2}
        recommended_psu_watts = 650
        attributes.update(
            host_interface="PCIe x16",
            length_mm=240 + candidate_index * 20,
            slot_width=2 + candidate_index,
            board_power_w=power_draw_watts,
            required_power_connectors=dict(required_power_connectors),
        )
    elif category is ComponentCategory.MOTHERBOARD:
        attributes.update(
            socket="AM5",
            chipset="B650",
            supported_cpu_generations=["Ryzen 7000"],
            memory_type="DDR5",
            wifi_support=True,
            maximum_memory_gb=192,
            memory_slots=4,
            form_factor="ATX",
            pcie_slots=3,
            m2_slots=3,
            sata_ports=6,
        )
    elif category is ComponentCategory.MEMORY:
        attributes.update(
            memory_type="DDR5",
            capacity_gb=minimum_memory_gb + candidate_index * 16,
            module_count=2,
        )
    elif category is ComponentCategory.STORAGE:
        attributes.update(
            capacity_gb=minimum_storage_gb + candidate_index * 500,
            interface="M.2 NVMe",
            form_factor="M.2 2280",
        )
    elif category is ComponentCategory.POWER_SUPPLY:
        psu_wattage = 850 + candidate_index * 100
        provided_power_connectors = {"8-pin PCIe": 3}
        eps_connectors = 2
        attributes.update(
            wattage=psu_wattage,
            form_factor="ATX",
            pcie_connectors=dict(provided_power_connectors),
        )
    elif category is ComponentCategory.COOLER:
        attributes.update(
            cooler_type="air",
            supported_sockets=["AM5"],
            height_mm=150 + candidate_index * 5,
        )
    elif category is ComponentCategory.CASE:
        attributes.update(
            case_size="mid_tower",
            dust_filter=True,
            supported_motherboard_sizes=["ATX", "Micro-ATX", "Mini-ITX"],
            maximum_gpu_length_mm=360,
            maximum_gpu_slot_width=4,
            maximum_cooler_height_mm=180,
            supported_psu_sizes=["ATX", "SFX"],
            radiator_support_mm=[120, 240, 280, 360],
        )

    product_id = f"s{scenario_index:05d}-{category.value}-c{candidate_index}"
    return OptimizationCandidate(
        product_id=product_id,
        category=category,
        price_cents=rng.randint(low, high),
        brand=f"GeneratedBrand{candidate_index % 3}",
        canonical_name=f"Generated {category.value} {scenario_index}-{candidate_index}",
        listing_id=f"generated-listing-{product_id}",
        in_stock=True,
        scores=_scores(rng),
        attributes=attributes,
        power_draw_watts=power_draw_watts,
        psu_wattage=psu_wattage,
        required_power_connectors=required_power_connectors,
        provided_power_connectors=provided_power_connectors,
        eps_connectors=eps_connectors,
        recommended_psu_watts=recommended_psu_watts,
    )


def generate_optimizer_problem(
    config: OptimizerEvaluationConfig,
    *,
    scenario_index: int,
) -> tuple[str, OptimizationProblem]:
    """Create one deterministic request with a known feasible or infeasible budget oracle."""

    if not 0 <= scenario_index < config.scenario_count:
        raise ValueError("scenario_index is outside the configured run")
    scenario_seed = (config.seed * 1_000_003 + scenario_index) % 2_147_483_647
    rng = random.Random(scenario_seed)
    minimum_gpu_vram_gb = (12, 16)[(scenario_index + config.seed) % 2]
    minimum_memory_gb = (32, 64)[(scenario_index + config.seed // 3) % 2]
    minimum_storage_gb = (1_000, 2_000)[(scenario_index + config.seed // 5) % 2]
    candidates = tuple(
        _generated_candidate(
            scenario_index=scenario_index,
            category=category,
            candidate_index=candidate_index,
            rng=rng,
            minimum_gpu_vram_gb=minimum_gpu_vram_gb,
            minimum_memory_gb=minimum_memory_gb,
            minimum_storage_gb=minimum_storage_gb,
        )
        for category in REQUIRED_CATEGORIES
        for candidate_index in range(config.candidates_per_category)
    )
    locked_product_ids = (
        frozenset({f"s{scenario_index:05d}-gpu-c0"})
        if (scenario_index + 1) % 7 == 0
        else frozenset()
    )
    pairwise: list[PairwiseCompatibility] = []
    if config.candidates_per_category >= 2:
        pairwise.extend(
            (
                PairwiseCompatibility(
                    f"s{scenario_index:05d}-cpu-c1",
                    f"s{scenario_index:05d}-motherboard-c1",
                    CompatVerdict.FAIL,
                    message="generated socket-family negative pair",
                ),
                PairwiseCompatibility(
                    f"s{scenario_index:05d}-gpu-c1",
                    f"s{scenario_index:05d}-case-c1",
                    CompatVerdict.FAIL,
                    message="generated clearance negative pair",
                ),
                PairwiseCompatibility(
                    f"s{scenario_index:05d}-memory-c1",
                    f"s{scenario_index:05d}-storage-c1",
                    CompatVerdict.WARNING,
                    message="generated shared-resource warning",
                    hard=False,
                ),
            )
        )

    def acquisition_price(candidate: OptimizationCandidate) -> int:
        if candidate.product_id in locked_product_ids:
            return 0
        return candidate.price_cents

    category_minimums = [
        min(
            acquisition_price(candidate)
            for candidate in candidates
            if candidate.category is category
        )
        for category in REQUIRED_CATEGORIES
    ]
    minimum_total = sum(category_minimums)
    expected_kind = (
        "budget_infeasible"
        if config.infeasible_every is not None
        and (scenario_index + 1) % config.infeasible_every == 0
        else "feasible"
    )
    budget_cents = (
        max(0, minimum_total - 1)
        if expected_kind == "budget_infeasible"
        else minimum_total + max(10_000, minimum_total // 3)
    )
    profile = tuple(BuildProfile)[(scenario_index + config.seed) % len(BuildProfile)]
    return expected_kind, OptimizationProblem(
        candidates=candidates,
        budget_cents=budget_cents,
        profiles=(profile,),
        locked_product_ids=locked_product_ids,
        minimum_gpu_vram_gb=minimum_gpu_vram_gb,
        minimum_memory_gb=minimum_memory_gb,
        minimum_storage_gb=minimum_storage_gb,
        required_memory_type="ddr5",
        required_motherboard_form_factor="atx",
        wifi_required=True,
        required_case_size="mid_tower",
        excluded_brands=frozenset({"ExcludedGeneratedBrand"}),
        required_features=(
            FeatureRequirement(
                ComponentCategory.CASE,
                "dust_filter",
                operator=FeatureOperator.TRUTHY,
            ),
        ),
        pairwise_compatibility=tuple(pairwise),
        power_headroom_percent=25,
        diversity_distance=2,
        time_limit_seconds=config.solver_time_limit_seconds,
        random_seed=scenario_seed,
    )


def _candidate_payload(candidate: OptimizationCandidate) -> dict[str, object]:
    return {
        "product_id": candidate.product_id,
        "category": candidate.category.value,
        "price_cents": candidate.price_cents,
        "brand": candidate.brand,
        "canonical_name": candidate.canonical_name,
        "listing_id": candidate.listing_id,
        "in_stock": candidate.in_stock,
        "scores": {
            "performance": candidate.scores.performance,
            "value": candidate.scores.value,
            "reliability": candidate.scores.reliability,
            "upgradeability": candidate.scores.upgradeability,
            "efficiency": candidate.scores.efficiency,
            "preference": candidate.scores.preference,
            "warning_penalty": candidate.scores.warning_penalty,
        },
        "attributes": dict(candidate.attributes),
        "power_draw_watts": candidate.power_draw_watts,
        "psu_wattage": candidate.psu_wattage,
        "required_power_connectors": dict(candidate.required_power_connectors),
        "provided_power_connectors": dict(candidate.provided_power_connectors),
        "eps_connectors": candidate.eps_connectors,
        "recommended_psu_watts": candidate.recommended_psu_watts,
    }


def _problem_payload(problem: OptimizationProblem) -> dict[str, object]:
    return {
        "candidates": [_candidate_payload(candidate) for candidate in problem.candidates],
        "budget_cents": problem.budget_cents,
        "profiles": [profile.value for profile in problem.profiles],
        "locked_product_ids": sorted(problem.locked_product_ids),
        "minimum_gpu_vram_gb": problem.minimum_gpu_vram_gb,
        "minimum_memory_gb": problem.minimum_memory_gb,
        "minimum_storage_gb": problem.minimum_storage_gb,
        "required_memory_type": _normalise_scalar(problem.required_memory_type),
        "required_motherboard_form_factor": _normalise_scalar(
            problem.required_motherboard_form_factor
        ),
        "wifi_required": problem.wifi_required,
        "required_case_size": _normalise_scalar(problem.required_case_size),
        "in_stock_only": problem.in_stock_only,
        "excluded_brands": sorted(problem.excluded_brands),
        "required_features": [
            {
                "category": requirement.category.value,
                "attribute": requirement.attribute,
                "expected": _normalise_scalar(requirement.expected),
                "operator": requirement.operator.value,
                "description": requirement.description,
            }
            for requirement in problem.required_features
        ],
        "pairwise_compatibility": [
            {
                "left_product_id": pair.left_product_id,
                "right_product_id": pair.right_product_id,
                "status": pair.status.value,
                "message": pair.message,
                "hard": pair.hard,
                "penalty_points": pair.penalty_points,
            }
            for pair in problem.pairwise_compatibility
        ],
        "power_headroom_percent": problem.power_headroom_percent,
        "base_power_watts": problem.base_power_watts,
        "category_power_allowances_watts": {
            category.value: watts
            for category, watts in sorted(
                problem.category_power_allowances_watts.items(), key=lambda item: item[0].value
            )
        },
        "required_eps_connectors": problem.required_eps_connectors,
        "diversity_distance": problem.diversity_distance,
        "meaningful_categories": [category.value for category in problem.meaningful_categories],
        "exclude_locked_from_budget": problem.exclude_locked_from_budget,
        "time_limit_seconds": problem.time_limit_seconds,
        "random_seed": problem.random_seed,
        "independent_validator_present": problem.independent_validator is not None,
    }


def _solution_record(solution: OptimizationSolution, errors: Sequence[str]) -> dict[str, object]:
    payload: dict[str, object] = {
        "profile": solution.profile.value,
        "solver_status": solution.solver_status.name,
        "selected_product_ids": {
            category.value: solution.selected[category].product_id
            for category in REQUIRED_CATEGORIES
        },
        "total_price_cents": solution.total_price_cents,
        "catalog_total_price_cents": solution.catalog_total_price_cents,
        "objective_value": solution.objective_value,
        "estimated_load_watts": solution.estimated_load_watts,
        "required_psu_watts": solution.required_psu_watts,
        "warnings": list(solution.warnings),
        "versioned_compatibility": _versioned_compatibility_evidence(solution.selected),
        "independently_checked": True,
        "independent_validation_passed": not errors,
        "independent_validation_errors": list(errors),
    }
    payload["output_record_sha256"] = _sha256_json(payload)
    return payload


def _scenario_record(
    *,
    scenario_index: int,
    expected_kind: str,
    problem: OptimizationProblem,
    result: OptimizationResult,
    requested_solution_count: int,
) -> dict[str, object]:
    errors_by_solution, result_errors = _independent_result_errors(
        problem,
        result,
        requested_solution_count=requested_solution_count,
    )
    output_records = [
        _solution_record(solution, errors)
        for solution, errors in zip(result.solutions, errors_by_solution, strict=True)
    ]
    if expected_kind == "feasible":
        scenario_passed = (
            result.status in (OptimizationStatus.OPTIMAL, OptimizationStatus.FEASIBLE)
            and len(result.solutions) == requested_solution_count
            and all(not errors for errors in errors_by_solution)
            and not result_errors
        )
        infeasibility_oracle_confirmed = False
    else:
        minimum_total = sum(
            min(
                0
                if problem.exclude_locked_from_budget
                and candidate.product_id in problem.locked_product_ids
                else candidate.price_cents
                for candidate in problem.candidates
                if candidate.category is category
            )
            for category in REQUIRED_CATEGORIES
        )
        infeasibility_oracle_confirmed = minimum_total > problem.budget_cents
        scenario_passed = (
            result.status is OptimizationStatus.INFEASIBLE
            and not result.solutions
            and infeasibility_oracle_confirmed
            and not result_errors
        )
    payload: dict[str, object] = {
        "scenario_index": scenario_index,
        "scenario_seed": problem.random_seed,
        "expected_kind": expected_kind,
        "request_sha256": _sha256_json(_problem_payload(problem)),
        "result_status": result.status.name,
        "profile_solve_records": [
            {
                "profile": record.profile.value,
                "status": record.status.name,
                "objective_value": record.objective_value,
            }
            for record in result.profile_statuses
        ],
        "requested_solution_count": requested_solution_count,
        "returned_solution_count": len(result.solutions),
        "rejected_by_validator": result.rejected_by_validator,
        "infeasibility_reasons_sha256": _sha256_json(list(result.infeasibility_reasons)),
        "infeasibility_oracle_confirmed": infeasibility_oracle_confirmed,
        "independent_result_validation_errors": list(result_errors),
        "scenario_passed": scenario_passed,
        "outputs": output_records,
    }
    payload["scenario_record_sha256"] = _sha256_json(payload)
    return payload


def _record_counters(
    records: Sequence[Mapping[str, object]],
) -> tuple[dict[str, int], dict[str, int], int, int, int, int, int, int]:
    status_counts: Counter[str] = Counter({status.name: 0 for status in OptimizationStatus})
    profile_status_counts: Counter[str] = Counter({status.name: 0 for status in OptimizationStatus})
    optimizer_output_count = 0
    independently_checked_count = 0
    independently_valid_count = 0
    invalid_output_count = 0
    infeasibility_oracle_confirmed_count = 0
    unique_outputs: set[tuple[str, ...]] = set()
    for record in records:
        status_counts[str(record["result_status"])] += 1
        raw_profile_records = record["profile_solve_records"]
        if not isinstance(raw_profile_records, list):
            raise OptimizerEvaluationError("profile_solve_records must be a list")
        for profile_record in raw_profile_records:
            if not isinstance(profile_record, Mapping):
                raise OptimizerEvaluationError("profile solve record must be an object")
            profile_status_counts[str(profile_record["status"])] += 1
        outputs = record["outputs"]
        if not isinstance(outputs, list):
            raise OptimizerEvaluationError("scenario outputs must be a list")
        optimizer_output_count += len(outputs)
        for output in outputs:
            if not isinstance(output, Mapping):
                raise OptimizerEvaluationError("optimizer output record must be an object")
            independently_checked_count += output.get("independently_checked") is True
            if output.get("independent_validation_passed") is True:
                independently_valid_count += 1
            else:
                invalid_output_count += 1
            selected_ids = output.get("selected_product_ids")
            if not isinstance(selected_ids, Mapping):
                raise OptimizerEvaluationError("selected_product_ids must be an object")
            try:
                unique_outputs.add(
                    tuple(str(selected_ids[category.value]) for category in REQUIRED_CATEGORIES)
                )
            except KeyError as exc:
                raise OptimizerEvaluationError(
                    "selected_product_ids omits a required category"
                ) from exc
        infeasibility_oracle_confirmed_count += record.get("infeasibility_oracle_confirmed") is True
    return (
        dict(sorted(status_counts.items())),
        dict(sorted(profile_status_counts.items())),
        optimizer_output_count,
        independently_checked_count,
        independently_valid_count,
        invalid_output_count,
        infeasibility_oracle_confirmed_count,
        len(unique_outputs),
    )


def run_optimizer_evaluation(
    config: OptimizerEvaluationConfig = OptimizerEvaluationConfig(),
) -> GeneratedOptimizerEvaluation:
    """Invoke BuildOptimizer for every generated request and retain bounded evidence."""

    optimizer = BuildOptimizer()
    records: list[dict[str, object]] = []
    for scenario_index in range(config.scenario_count):
        expected_kind, problem = generate_optimizer_problem(config, scenario_index=scenario_index)
        result = optimizer.optimize(problem, max_solutions=config.solutions_per_scenario)
        records.append(
            _scenario_record(
                scenario_index=scenario_index,
                expected_kind=expected_kind,
                problem=problem,
                result=result,
                requested_solution_count=config.solutions_per_scenario,
            )
        )

    stream_hash = hashlib.sha256()
    for record in records:
        stream_hash.update(_canonical_json_bytes(record))
        stream_hash.update(b"\n")
    (
        status_counts,
        profile_status_counts,
        optimizer_output_count,
        independently_checked_count,
        independently_valid_count,
        invalid_output_count,
        infeasibility_oracle_confirmed_count,
        unique_output_count,
    ) = _record_counters(records)
    evaluation_passed = all(record["scenario_passed"] is True for record in records)
    claim = assess_10k_valid_build_claim(
        optimizer_output_count=optimizer_output_count,
        independently_checked_output_count=independently_checked_count,
        independently_valid_output_count=independently_valid_count,
        unique_output_count=unique_output_count,
        retained_output_record_count=optimizer_output_count,
        invalid_output_count=invalid_output_count,
        evaluation_passed=evaluation_passed,
        records_verified=True,
    )
    payload: dict[str, object] = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "artifact_kind": "optimizer_generated_build_evaluation",
        "generator_version": GENERATOR_VERSION,
        "evaluation_passed": evaluation_passed,
        "scenario_provenance": "deterministic_generated_optimizer_requests",
        "optimizer": "OR-Tools CP-SAT via BuildOptimizer",
        "ortools_version": distribution_version("ortools"),
        "compatibility_rule_version": DEFAULT_RULE_VERSION,
        "scenario_count": config.scenario_count,
        "optimizer_invocation_count": config.scenario_count,
        "seed": config.seed,
        "candidates_per_category": config.candidates_per_category,
        "solutions_per_scenario": config.solutions_per_scenario,
        "solver_time_limit_seconds": config.solver_time_limit_seconds,
        "infeasible_every": config.infeasible_every,
        "result_status_counts": status_counts,
        "profile_solve_status_counts": profile_status_counts,
        "optimizer_output_count": optimizer_output_count,
        "independently_checked_output_count": independently_checked_count,
        "independently_valid_output_count": independently_valid_count,
        "invalid_output_count": invalid_output_count,
        "unique_output_count": unique_output_count,
        "retained_scenario_record_count": len(records),
        "retained_output_record_count": optimizer_output_count,
        "infeasibility_oracle_confirmed_count": infeasibility_oracle_confirmed_count,
        "scenario_record_stream_sha256": stream_hash.hexdigest(),
        "claim_assessment": claim.to_dict(),
        "memory_strategy": {
            "mode": "compact_bounded_records_and_incremental_sha256",
            "maximum_scenarios": MAX_SCENARIO_COUNT,
            "maximum_candidates_per_category": MAX_CANDIDATES_PER_CATEGORY,
            "maximum_outputs_per_scenario": MAX_SOLUTIONS_PER_SCENARIO,
            "maximum_retained_output_records": MAX_RETAINED_OUTPUT_RECORDS,
            "full_candidate_payloads_retained": False,
        },
        "source_sha256": _current_source_sha256(),
        "scenario_records": records,
    }
    artifact_sha256 = _sha256_json(payload)
    report = GeneratedOptimizerEvaluation(payload=payload, artifact_sha256=artifact_sha256)
    report.verify()
    return report


def _verify_output_record(output: Mapping[str, object]) -> None:
    stored_hash = output.get("output_record_sha256")
    if not _valid_sha256(stored_hash):
        raise OptimizerEvaluationError("optimizer output record hash is invalid")
    unhashed = dict(output)
    del unhashed["output_record_sha256"]
    if _sha256_json(unhashed) != stored_hash:
        raise OptimizerEvaluationError("optimizer output record hash mismatch")


def _record_int(record: Mapping[str, object], field: str) -> int:
    value = record.get(field)
    if type(value) is not int:
        raise OptimizerEvaluationError(f"record field {field!r} must be an integer")
    return value


def _solution_from_output_record(
    problem: OptimizationProblem, output: Mapping[str, object]
) -> OptimizationSolution:
    raw_selected = output.get("selected_product_ids")
    if not isinstance(raw_selected, Mapping):
        raise OptimizerEvaluationError("selected_product_ids must be an object")
    catalogue = {candidate.product_id: candidate for candidate in problem.candidates}
    selected: dict[ComponentCategory, OptimizationCandidate] = {}
    for category in REQUIRED_CATEGORIES:
        product_id = raw_selected.get(category.value)
        if not isinstance(product_id, str):
            raise OptimizerEvaluationError(
                f"selected_product_ids.{category.value} must be a string"
            )
        candidate = catalogue.get(product_id)
        if candidate is None:
            raise OptimizerEvaluationError(
                f"retained output references unknown product {product_id!r}"
            )
        selected[category] = candidate
    raw_profile = output.get("profile")
    raw_status = output.get("solver_status")
    try:
        profile = BuildProfile(str(raw_profile))
        solver_status = OptimizationStatus[str(raw_status)]
    except (KeyError, ValueError) as exc:
        raise OptimizerEvaluationError("retained output profile or status is invalid") from exc
    raw_warnings = output.get("warnings")
    if not isinstance(raw_warnings, list) or not all(
        isinstance(warning, str) for warning in raw_warnings
    ):
        raise OptimizerEvaluationError("retained output warnings must be a string list")
    return OptimizationSolution(
        profile=profile,
        selected=selected,
        total_price_cents=_record_int(output, "total_price_cents"),
        catalog_total_price_cents=_record_int(output, "catalog_total_price_cents"),
        objective_value=_record_int(output, "objective_value"),
        estimated_load_watts=_record_int(output, "estimated_load_watts"),
        required_psu_watts=_record_int(output, "required_psu_watts"),
        solver_status=solver_status,
        warnings=tuple(raw_warnings),
    )


def _profile_records_from_scenario(
    record: Mapping[str, object],
) -> tuple[ProfileSolveRecord, ...]:
    raw_records = record.get("profile_solve_records")
    if not isinstance(raw_records, list):
        raise OptimizerEvaluationError("profile_solve_records must be a list")
    parsed: list[ProfileSolveRecord] = []
    for raw_record in raw_records:
        if not isinstance(raw_record, Mapping):
            raise OptimizerEvaluationError("profile solve record must be an object")
        raw_objective = raw_record.get("objective_value")
        if raw_objective is not None and type(raw_objective) is not int:
            raise OptimizerEvaluationError("profile objective_value must be an integer or null")
        try:
            profile = BuildProfile(str(raw_record.get("profile")))
            status = OptimizationStatus[str(raw_record.get("status"))]
        except (KeyError, ValueError) as exc:
            raise OptimizerEvaluationError("profile solve record is invalid") from exc
        parsed.append(
            ProfileSolveRecord(
                profile=profile,
                status=status,
                wall_time_seconds=0.0,
                objective_value=raw_objective,
            )
        )
    return tuple(parsed)


def _config_from_payload(payload: Mapping[str, object]) -> OptimizerEvaluationConfig:
    raw_time_limit = payload.get("solver_time_limit_seconds")
    if isinstance(raw_time_limit, bool) or not isinstance(raw_time_limit, (int, float)):
        raise OptimizerEvaluationError("solver_time_limit_seconds must be numeric")
    raw_infeasible_every = payload.get("infeasible_every")
    if raw_infeasible_every is not None and type(raw_infeasible_every) is not int:
        raise OptimizerEvaluationError("infeasible_every must be an integer or null")
    try:
        return OptimizerEvaluationConfig(
            scenario_count=_record_int(payload, "scenario_count"),
            seed=_record_int(payload, "seed"),
            candidates_per_category=_record_int(payload, "candidates_per_category"),
            solutions_per_scenario=_record_int(payload, "solutions_per_scenario"),
            solver_time_limit_seconds=float(raw_time_limit),
            infeasible_every=raw_infeasible_every,
        )
    except ValueError as exc:
        raise OptimizerEvaluationError(f"artifact configuration is invalid: {exc}") from exc


def _verify_report(payload: Mapping[str, object], artifact_sha256: str) -> None:
    if not _valid_sha256(artifact_sha256) or _sha256_json(payload) != artifact_sha256:
        raise OptimizerEvaluationError("optimizer evaluation artifact hash mismatch")
    if payload.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise OptimizerEvaluationError("unsupported optimizer evaluation schema")
    if payload.get("generator_version") != GENERATOR_VERSION:
        raise OptimizerEvaluationError("unsupported optimizer request generator version")
    if payload.get("compatibility_rule_version") != DEFAULT_RULE_VERSION:
        raise OptimizerEvaluationError("unsupported compatibility rule version")
    _verify_source_sha256(payload)
    config = _config_from_payload(payload)
    raw_records = payload.get("scenario_records")
    if not isinstance(raw_records, list):
        raise OptimizerEvaluationError("scenario_records must be a list")
    if len(raw_records) != payload.get("scenario_count"):
        raise OptimizerEvaluationError("scenario record count does not match scenario_count")
    if len(raw_records) > MAX_SCENARIO_COUNT:
        raise OptimizerEvaluationError("scenario record count exceeds the safety bound")

    records: list[Mapping[str, object]] = []
    stream_hash = hashlib.sha256()
    for expected_index, raw_record in enumerate(raw_records):
        if not isinstance(raw_record, Mapping):
            raise OptimizerEvaluationError("scenario record must be an object")
        stored_hash = raw_record.get("scenario_record_sha256")
        if not _valid_sha256(stored_hash):
            raise OptimizerEvaluationError("scenario record hash is invalid")
        unhashed = dict(raw_record)
        del unhashed["scenario_record_sha256"]
        if _sha256_json(unhashed) != stored_hash:
            raise OptimizerEvaluationError("scenario record hash mismatch")
        outputs = raw_record.get("outputs")
        if not isinstance(outputs, list) or len(outputs) > MAX_SOLUTIONS_PER_SCENARIO:
            raise OptimizerEvaluationError("scenario output record count exceeds the safety bound")
        scenario_index = _record_int(raw_record, "scenario_index")
        if scenario_index != expected_index:
            raise OptimizerEvaluationError("scenario records are missing, duplicated, or reordered")
        expected_kind, problem = generate_optimizer_problem(config, scenario_index=scenario_index)
        if raw_record.get("expected_kind") != expected_kind:
            raise OptimizerEvaluationError("scenario expected kind disagrees with the generator")
        if raw_record.get("scenario_seed") != problem.random_seed:
            raise OptimizerEvaluationError("scenario seed disagrees with the generator")
        if raw_record.get("request_sha256") != _sha256_json(_problem_payload(problem)):
            raise OptimizerEvaluationError("scenario request digest disagrees with regeneration")
        if raw_record.get("requested_solution_count") != config.solutions_per_scenario:
            raise OptimizerEvaluationError("requested solution count disagrees with configuration")
        if raw_record.get("returned_solution_count") != len(outputs):
            raise OptimizerEvaluationError("returned solution count disagrees with output records")
        if not _valid_sha256(raw_record.get("infeasibility_reasons_sha256")):
            raise OptimizerEvaluationError("infeasibility reason digest is invalid")

        solutions: list[OptimizationSolution] = []
        for output in outputs:
            if not isinstance(output, Mapping):
                raise OptimizerEvaluationError("optimizer output record must be an object")
            _verify_output_record(output)
            solution = _solution_from_output_record(problem, output)
            expected_compatibility = _versioned_compatibility_evidence(solution.selected)
            if output.get("versioned_compatibility") != expected_compatibility:
                raise OptimizerEvaluationError(
                    "retained compatibility digest disagrees with versioned revalidation"
                )
            raw_errors = output.get("independent_validation_errors")
            if not isinstance(raw_errors, list) or not all(
                isinstance(error, str) for error in raw_errors
            ):
                raise OptimizerEvaluationError(
                    "independent_validation_errors must be a string list"
                )
            if output.get("independently_checked") is not True:
                raise OptimizerEvaluationError("retained optimizer output was not checked")
            solutions.append(solution)

        raw_result_status = raw_record.get("result_status")
        try:
            result_status = OptimizationStatus[str(raw_result_status)]
        except KeyError as exc:
            raise OptimizerEvaluationError("scenario result status is invalid") from exc
        rejected_by_validator = _record_int(raw_record, "rejected_by_validator")
        if rejected_by_validator < 0:
            raise OptimizerEvaluationError("rejected_by_validator must be non-negative")
        reconstructed_result = OptimizationResult(
            status=result_status,
            solutions=tuple(solutions),
            profile_statuses=_profile_records_from_scenario(raw_record),
            rejected_by_validator=rejected_by_validator,
        )
        errors_by_solution, result_errors = _independent_result_errors(
            problem,
            reconstructed_result,
            requested_solution_count=config.solutions_per_scenario,
        )
        for output, errors in zip(outputs, errors_by_solution, strict=True):
            raw_errors = output["independent_validation_errors"]
            if not isinstance(raw_errors, list) or tuple(raw_errors) != errors:
                raise OptimizerEvaluationError(
                    "cross-solution validation result disagrees with retained output"
                )
            if output.get("independent_validation_passed") is not (not errors):
                raise OptimizerEvaluationError(
                    "independent validation pass flag disagrees with validation errors"
                )
        raw_result_errors = raw_record.get("independent_result_validation_errors")
        if not isinstance(raw_result_errors, list) or tuple(raw_result_errors) != result_errors:
            raise OptimizerEvaluationError("independent result validation errors do not match")

        if expected_kind == "feasible":
            infeasibility_confirmed = False
            scenario_passed = (
                result_status in (OptimizationStatus.OPTIMAL, OptimizationStatus.FEASIBLE)
                and len(solutions) == config.solutions_per_scenario
                and all(not errors for errors in errors_by_solution)
                and not result_errors
            )
        else:
            minimum_total = sum(
                min(
                    0
                    if problem.exclude_locked_from_budget
                    and candidate.product_id in problem.locked_product_ids
                    else candidate.price_cents
                    for candidate in problem.candidates
                    if candidate.category is category
                )
                for category in REQUIRED_CATEGORIES
            )
            infeasibility_confirmed = minimum_total > problem.budget_cents
            scenario_passed = (
                result_status is OptimizationStatus.INFEASIBLE
                and not solutions
                and infeasibility_confirmed
                and not result_errors
            )
        if raw_record.get("infeasibility_oracle_confirmed") is not infeasibility_confirmed:
            raise OptimizerEvaluationError("infeasibility oracle flag disagrees with request")
        if raw_record.get("scenario_passed") is not scenario_passed:
            raise OptimizerEvaluationError("scenario pass flag disagrees with retained evidence")
        stream_hash.update(_canonical_json_bytes(raw_record))
        stream_hash.update(b"\n")
        records.append(raw_record)
    if stream_hash.hexdigest() != payload.get("scenario_record_stream_sha256"):
        raise OptimizerEvaluationError("scenario record stream hash mismatch")

    (
        status_counts,
        profile_status_counts,
        output_count,
        checked_count,
        valid_count,
        invalid_count,
        infeasible_confirmed_count,
        unique_output_count,
    ) = _record_counters(records)
    expected_fields: Mapping[str, object] = {
        "optimizer_invocation_count": config.scenario_count,
        "result_status_counts": status_counts,
        "profile_solve_status_counts": profile_status_counts,
        "optimizer_output_count": output_count,
        "independently_checked_output_count": checked_count,
        "independently_valid_output_count": valid_count,
        "invalid_output_count": invalid_count,
        "unique_output_count": unique_output_count,
        "retained_scenario_record_count": len(records),
        "retained_output_record_count": output_count,
        "infeasibility_oracle_confirmed_count": infeasible_confirmed_count,
    }
    for field, expected in expected_fields.items():
        if payload.get(field) != expected:
            raise OptimizerEvaluationError(f"aggregate field {field!r} does not match records")
    if not isinstance(payload.get("ortools_version"), str):
        raise OptimizerEvaluationError("ortools_version must be retained")
    evaluation_passed = all(record.get("scenario_passed") is True for record in records)
    if payload.get("evaluation_passed") is not evaluation_passed:
        raise OptimizerEvaluationError("evaluation_passed does not match scenario records")
    expected_claim = assess_10k_valid_build_claim(
        optimizer_output_count=output_count,
        independently_checked_output_count=checked_count,
        independently_valid_output_count=valid_count,
        unique_output_count=unique_output_count,
        retained_output_record_count=output_count,
        invalid_output_count=invalid_count,
        evaluation_passed=evaluation_passed,
        records_verified=True,
    ).to_dict()
    if payload.get("claim_assessment") != expected_claim:
        raise OptimizerEvaluationError("claim assessment does not match retained evidence")


def write_optimizer_evaluation(
    report: GeneratedOptimizerEvaluation, output_dir: str | Path = DEFAULT_OUTPUT_DIR
) -> Path:
    """Atomically write a verified content-addressed report."""

    report.verify()
    destination_dir = Path(output_dir).resolve()
    destination_dir.mkdir(parents=True, exist_ok=True)
    filename = (
        f"optimizer-generated-seed-{report.payload['seed']}-"
        f"n-{report.payload['scenario_count']}-{report.artifact_sha256[:16]}.json"
    )
    destination = destination_dir / filename
    serialised = (
        json.dumps(report.to_dict(), allow_nan=False, ensure_ascii=True, indent=2, sort_keys=True)
        + "\n"
    )
    if destination.exists():
        if destination.read_text(encoding="utf-8") != serialised:
            raise OptimizerEvaluationError(
                "existing content-addressed optimizer report has different bytes"
            )
        load_optimizer_evaluation(destination)
        return destination
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=destination_dir,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(serialised)
            handle.flush()
            os.fsync(handle.fileno())
            temporary_name = handle.name
        os.replace(temporary_name, destination)
    finally:
        if temporary_name is not None:
            temporary_path = Path(temporary_name)
            if temporary_path.exists():
                temporary_path.unlink()
    load_optimizer_evaluation(destination)
    return destination


def load_optimizer_evaluation(path: str | Path) -> GeneratedOptimizerEvaluation:
    source = Path(path)
    raw = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise OptimizerEvaluationError("optimizer evaluation root must be an object")
    artifact_sha256 = raw.pop("artifact_sha256", None)
    if not isinstance(artifact_sha256, str):
        raise OptimizerEvaluationError("optimizer evaluation is missing artifact_sha256")
    report = GeneratedOptimizerEvaluation(payload=raw, artifact_sha256=artifact_sha256)
    report.verify()
    if artifact_sha256[:16] not in source.name:
        raise OptimizerEvaluationError("optimizer evaluation filename omits its content hash")
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate bounded optimizer requests, invoke CP-SAT, independently validate every "
            "returned build, and write compact content-addressed evidence."
        )
    )
    parser.add_argument(
        "--count",
        type=int,
        default=100,
        help=f"Optimizer requests to run (1-{MAX_SCENARIO_COUNT}; default: 100)",
    )
    parser.add_argument("--seed", type=int, default=20_260_723)
    parser.add_argument(
        "--candidates-per-category",
        type=int,
        default=2,
        help=f"Generated candidates in each category (1-{MAX_CANDIDATES_PER_CATEGORY})",
    )
    parser.add_argument(
        "--solutions-per-scenario",
        type=int,
        default=1,
        help=f"Returned builds requested from each solve (1-{MAX_SOLUTIONS_PER_SCENARIO})",
    )
    parser.add_argument(
        "--solver-time-limit-seconds",
        type=float,
        default=0.5,
        help=f"Per-profile CP-SAT limit (maximum {MAX_SOLVER_TIME_LIMIT_SECONDS:g}s)",
    )
    parser.add_argument(
        "--infeasible-every",
        type=int,
        default=None,
        help="Make every Nth request independently budget-infeasible (diagnostic mode)",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        config = OptimizerEvaluationConfig(
            scenario_count=args.count,
            seed=args.seed,
            candidates_per_category=args.candidates_per_category,
            solutions_per_scenario=args.solutions_per_scenario,
            solver_time_limit_seconds=args.solver_time_limit_seconds,
            infeasible_every=args.infeasible_every,
        )
    except ValueError as exc:
        parser.error(str(exc))
    report = run_optimizer_evaluation(config)
    output = write_optimizer_evaluation(report, args.output_dir)
    claim = report.payload["claim_assessment"]
    if not isinstance(claim, Mapping):
        raise OptimizerEvaluationError("claim assessment is malformed")
    print(
        json.dumps(
            {
                "artifact": str(output),
                "artifact_sha256": report.artifact_sha256,
                "evaluation_passed": report.payload["evaluation_passed"],
                "optimizer_invocation_count": report.payload["optimizer_invocation_count"],
                "optimizer_output_count": report.payload["optimizer_output_count"],
                "result_status_counts": report.payload["result_status_counts"],
                "eligible_for_10k_valid_build_claim": claim["eligible"],
            },
            sort_keys=True,
        )
    )
    return 0 if report.payload["evaluation_passed"] is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
