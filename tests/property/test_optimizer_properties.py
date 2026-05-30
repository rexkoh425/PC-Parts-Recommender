from __future__ import annotations

from dataclasses import replace

from hypothesis import given, settings
from hypothesis import strategies as st

from pc_build_recommender.domain import (
    BuildProfile,
    CompatVerdict,
    ComponentCategory,
)
from pc_build_recommender.optimizer import (
    BuildOptimizer,
    CandidateScores,
    OptimizationCandidate,
    OptimizationProblem,
    PairwiseCompatibility,
    enumerate_feasible_builds,
    validate_selected_build,
)


def _candidate(
    category: ComponentCategory,
    suffix: str,
    *,
    price_cents: int = 10_000,
    score: float = 50,
) -> OptimizationCandidate:
    attributes: dict[str, object] = {}
    extra: dict[str, object] = {}
    if category == ComponentCategory.CPU:
        extra["power_draw_watts"] = 100
    elif category == ComponentCategory.GPU:
        attributes["vram_gb"] = 16
        extra.update(
            power_draw_watts=200,
            required_power_connectors={"pcie_8pin": 1},
            recommended_psu_watts=600,
        )
    elif category == ComponentCategory.MEMORY:
        attributes.update(capacity_gb=32, memory_type="ddr5")
    elif category == ComponentCategory.STORAGE:
        attributes["capacity_gb"] = 2_000
    elif category == ComponentCategory.MOTHERBOARD:
        attributes.update(wifi_support=True, memory_type="ddr5")
    elif category == ComponentCategory.POWER_SUPPLY:
        extra.update(
            psu_wattage=750,
            provided_power_connectors={"pcie_8pin": 2},
            eps_connectors=2,
        )
    return OptimizationCandidate(
        product_id=f"{category.value}-{suffix}",
        category=category,
        price_cents=price_cents,
        scores=CandidateScores(
            performance=score,
            value=score,
            reliability=score,
            upgradeability=score,
            efficiency=score,
            preference=score,
        ),
        attributes=attributes,
        **extra,
    )


def _catalogue() -> tuple[OptimizationCandidate, ...]:
    return tuple(_candidate(category, "a") for category in ComponentCategory)


def _problem(
    catalogue: tuple[OptimizationCandidate, ...],
    **overrides: object,
) -> OptimizationProblem:
    values: dict[str, object] = {
        "candidates": catalogue,
        "budget_cents": 200_000,
        "profiles": (BuildProfile.BEST_OVERALL,),
        "minimum_gpu_vram_gb": 16,
        "minimum_memory_gb": 32,
        "minimum_storage_gb": 2_000,
        "required_memory_type": "ddr5",
        "wifi_required": True,
    }
    values.update(overrides)
    return OptimizationProblem(**values)


@given(
    budget=st.integers(min_value=80_000, max_value=150_000),
    gpu_price=st.integers(min_value=5_000, max_value=50_000),
)
@settings(max_examples=25, deadline=None)
def test_every_returned_build_has_exact_cardinality_and_respects_integer_budget(
    budget: int,
    gpu_price: int,
) -> None:
    catalogue = tuple(
        replace(item, price_cents=gpu_price) if item.category == ComponentCategory.GPU else item
        for item in _catalogue()
    )
    request = _problem(catalogue, budget_cents=budget)

    result = BuildOptimizer().optimize(request)

    if result.solutions:
        solution = result.solutions[0]
        assert len(solution.selected) == 8
        assert solution.total_price_cents <= budget
        assert validate_selected_build(request, solution.selected) == ()


@given(
    cpu_score_a=st.integers(min_value=0, max_value=100),
    cpu_score_b=st.integers(min_value=0, max_value=100),
    gpu_score_a=st.integers(min_value=0, max_value=100),
    gpu_score_b=st.integers(min_value=0, max_value=100),
)
@settings(max_examples=30, deadline=None)
def test_cp_sat_objective_matches_exhaustive_search(
    cpu_score_a: int,
    cpu_score_b: int,
    gpu_score_a: int,
    gpu_score_b: int,
) -> None:
    base = _catalogue()
    cpu = next(item for item in base if item.category == ComponentCategory.CPU)
    gpu = next(item for item in base if item.category == ComponentCategory.GPU)
    catalogue = tuple(
        replace(item, scores=replace(item.scores, performance=cpu_score_a))
        if item.category == ComponentCategory.CPU
        else replace(item, scores=replace(item.scores, performance=gpu_score_a))
        if item.category == ComponentCategory.GPU
        else item
        for item in base
    ) + (
        replace(
            cpu,
            product_id="cpu-b",
            scores=replace(cpu.scores, performance=cpu_score_b),
        ),
        replace(
            gpu,
            product_id="gpu-b",
            scores=replace(gpu.scores, performance=gpu_score_b),
        ),
    )
    request = _problem(catalogue)

    solved = BuildOptimizer().optimize(request).solutions[0]
    exhaustive = enumerate_feasible_builds(request)

    assert exhaustive.best is not None
    assert solved.objective_value == exhaustive.best.objective_value


@given(
    left_suffix=st.sampled_from(("a", "b")),
    right_suffix=st.sampled_from(("a", "b")),
)
@settings(max_examples=8, deadline=None)
def test_adding_a_hard_incompatibility_never_increases_feasible_set(
    left_suffix: str,
    right_suffix: str,
) -> None:
    base = _catalogue()
    cpu = next(item for item in base if item.category == ComponentCategory.CPU)
    motherboard = next(item for item in base if item.category == ComponentCategory.MOTHERBOARD)
    catalogue = base + (
        replace(cpu, product_id="cpu-b"),
        replace(motherboard, product_id="motherboard-b"),
    )
    unconstrained = _problem(catalogue)
    constrained = _problem(
        catalogue,
        pairwise_compatibility=(
            PairwiseCompatibility(
                f"cpu-{left_suffix}",
                f"motherboard-{right_suffix}",
                CompatVerdict.FAIL,
            ),
        ),
    )

    before = enumerate_feasible_builds(unconstrained)
    after = enumerate_feasible_builds(constrained)

    assert len(after.solutions) <= len(before.solutions)


@given(
    cpu_power=st.integers(min_value=65, max_value=250),
    gpu_power=st.integers(min_value=100, max_value=450),
    lower_wattage=st.integers(min_value=600, max_value=900),
    increase=st.integers(min_value=1, max_value=400),
)
@settings(max_examples=25, deadline=None)
def test_increasing_psu_wattage_cannot_create_a_capacity_failure(
    cpu_power: int,
    gpu_power: int,
    lower_wattage: int,
    increase: int,
) -> None:
    base = _catalogue()
    powered = tuple(
        replace(item, power_draw_watts=cpu_power)
        if item.category == ComponentCategory.CPU
        else replace(item, power_draw_watts=gpu_power)
        if item.category == ComponentCategory.GPU
        else replace(item, psu_wattage=lower_wattage)
        if item.category == ComponentCategory.POWER_SUPPLY
        else item
        for item in base
    )
    stronger = tuple(
        replace(item, psu_wattage=lower_wattage + increase)
        if item.category == ComponentCategory.POWER_SUPPLY
        else item
        for item in powered
    )

    lower_result = BuildOptimizer().optimize(_problem(powered))
    stronger_result = BuildOptimizer().optimize(_problem(stronger))

    if lower_result.is_feasible:
        assert stronger_result.is_feasible
