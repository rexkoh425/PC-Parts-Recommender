from __future__ import annotations

from dataclasses import replace
from decimal import Decimal

import pytest

from pc_build_recommender.compatibility import (
    CompatibilityReport,
    CompatibilityResult,
)
from pc_build_recommender.compatibility import (
    CompatVerdict as EngineCompatibilityStatus,
)
from pc_build_recommender.domain import (
    BuildRequestSpec,
    BuildPreferences,
    BuildPreset,
    BuildRequirements,
    MasterProduct,
    CaseAttributes,
    CompatVerdict,
    ComponentKind,
    CPUAttributes,
    ExistingComponent,
    GPUAttributes,
    ListingCondition,
    MemoryAttributes,
    MotherboardAttributes,
    PowerSupplyAttributes,
    RetailerOffering,
    StockState,
    StorageAttributes,
    WorkloadLabel,
    WorkloadPreference,
)
from pc_build_recommender.domain import CoolerAttributes as DomainCoolerAttributes
from pc_build_recommender.optimizer import (
    BuildOptimizer,
    CandidateScores,
    FeatureRequirement,
    OptimizationCandidate,
    OptimizationProblem,
    OptimizationStatus,
    PairwiseCompatibility,
    enumerate_feasible_builds,
    solution_to_domain,
    validate_selected_build,
)


def candidate(
    category: ComponentKind,
    suffix: str,
    *,
    price: int = 10_000,
    performance: float = 50,
    value: float = 50,
    efficiency: float = 50,
    power: int | None = None,
    attributes: dict[str, object] | None = None,
    in_stock: bool = True,
    brand: str = "Allowed",
) -> OptimizationCandidate:
    kwargs: dict[str, object] = {}
    if category == ComponentKind.CPU:
        kwargs["power_draw_watts"] = 120 if power is None else power
    elif category == ComponentKind.GPU:
        kwargs.update(
            power_draw_watts=250 if power is None else power,
            required_power_connectors={"pcie_8pin": 1},
            recommended_psu_watts=650,
        )
    elif category == ComponentKind.POWER_SUPPLY:
        kwargs.update(
            psu_wattage=750,
            provided_power_connectors={"pcie_8pin": 2},
            eps_connectors=2,
        )
    return OptimizationCandidate(
        product_id=f"{category.value}-{suffix}",
        category=category,
        price_cents=price,
        brand=brand,
        canonical_name=f"{category.value} {suffix}",
        listing_id=f"listing-{category.value}-{suffix}",
        in_stock=in_stock,
        scores=CandidateScores(
            performance=performance,
            value=value,
            reliability=50,
            upgradeability=50,
            efficiency=efficiency,
            preference=50,
        ),
        attributes=attributes or {},
        **kwargs,
    )


def complete_catalogue(*, alternatives: bool = False) -> tuple[OptimizationCandidate, ...]:
    attributes = {
        ComponentKind.GPU: {"vram_gb": 16},
        ComponentKind.MEMORY: {"capacity_gb": 32, "memory_type": "ddr5"},
        ComponentKind.STORAGE: {"capacity_gb": 2_000},
        ComponentKind.MOTHERBOARD: {
            "wifi_support": True,
            "memory_type": "ddr5",
            "form_factor": "atx",
        },
        ComponentKind.CASE: {"case_size": "mid_tower"},
    }
    result: list[OptimizationCandidate] = []
    for category in (
        ComponentKind.CPU,
        ComponentKind.GPU,
        ComponentKind.MOTHERBOARD,
        ComponentKind.MEMORY,
        ComponentKind.STORAGE,
        ComponentKind.POWER_SUPPLY,
        ComponentKind.COOLER,
        ComponentKind.CASE,
    ):
        result.append(
            candidate(
                category,
                "a",
                performance=80,
                value=70,
                attributes=attributes.get(category),
            )
        )
        if alternatives:
            result.append(
                candidate(
                    category,
                    "b",
                    price=9_000,
                    performance=60,
                    value=80,
                    attributes=attributes.get(category),
                )
            )
    return tuple(result)


def problem(
    candidates: tuple[OptimizationCandidate, ...],
    **kwargs: object,
) -> OptimizationProblem:
    values: dict[str, object] = {
        "candidates": candidates,
        "budget_cents": 200_000,
        "profiles": (BuildPreset.BEST_OVERALL,),
        "minimum_gpu_vram_gb": 16,
        "minimum_memory_gb": 32,
        "minimum_storage_gb": 2_000,
        "required_memory_type": "ddr5",
        "wifi_required": True,
    }
    values.update(kwargs)
    return OptimizationProblem(**values)


def test_cp_sat_selects_exactly_one_per_category_and_respects_budget() -> None:
    request = problem(complete_catalogue(alternatives=True), budget_cents=75_000)

    result = BuildOptimizer().optimize(request)

    assert result.status == OptimizationStatus.OPTIMAL
    assert len(result.solutions) == 1
    solution = result.solutions[0]
    assert set(solution.selected) == {
        ComponentKind.CPU,
        ComponentKind.GPU,
        ComponentKind.MOTHERBOARD,
        ComponentKind.MEMORY,
        ComponentKind.STORAGE,
        ComponentKind.POWER_SUPPLY,
        ComponentKind.COOLER,
        ComponentKind.CASE,
    }
    assert solution.total_price_cents <= 75_000
    assert validate_selected_build(request, solution.selected) == ()


def test_locked_existing_component_is_selected_and_excluded_from_purchase_budget() -> None:
    catalogue = complete_catalogue(alternatives=True)
    locked_gpu = next(item for item in catalogue if item.product_id == "gpu-a")
    locked_gpu = replace(locked_gpu, price_cents=150_000)
    catalogue = tuple(locked_gpu if item.product_id == "gpu-a" else item for item in catalogue)
    request = problem(
        catalogue,
        budget_cents=70_000,
        locked_product_ids=frozenset({"gpu-a"}),
    )

    result = BuildOptimizer().optimize(request)

    assert result.is_feasible
    solution = result.solutions[0]
    assert solution.selected[ComponentKind.GPU].product_id == "gpu-a"
    assert solution.total_price_cents <= 70_000
    assert solution.catalog_total_price_cents > solution.total_price_cents


def test_hard_unknown_and_failure_pairs_are_infeasible() -> None:
    catalogue = complete_catalogue(alternatives=True)
    pairs = (
        PairwiseCompatibility(
            "cpu-a",
            "motherboard-a",
            CompatVerdict.UNKNOWN,
            message="BIOS support missing",
        ),
        PairwiseCompatibility(
            "cpu-a",
            "motherboard-b",
            CompatVerdict.FAIL,
            message="socket mismatch",
        ),
    )
    request = problem(catalogue, pairwise_compatibility=pairs)

    solution = BuildOptimizer().optimize(request).solutions[0]

    assert solution.selected[ComponentKind.CPU].product_id == "cpu-b"


def test_excluded_brand_and_required_feature_are_hard_filters() -> None:
    catalogue = list(complete_catalogue(alternatives=True))
    catalogue = [
        replace(item, brand="Blocked") if item.product_id == "gpu-a" else item for item in catalogue
    ]
    catalogue = [
        replace(item, attributes={**item.attributes, "dust_filter": True})
        if item.product_id == "case-b"
        else item
        for item in catalogue
    ]
    request = problem(
        tuple(catalogue),
        excluded_brands=frozenset({"blocked"}),
        required_features=(FeatureRequirement(ComponentKind.CASE, "dust_filter", True),),
    )

    solution = BuildOptimizer().optimize(request).solutions[0]

    assert solution.selected[ComponentKind.GPU].product_id == "gpu-b"
    assert solution.selected[ComponentKind.CASE].product_id == "case-b"


def test_power_headroom_gpu_recommendation_and_connectors_are_hard_constraints() -> None:
    catalogue = list(complete_catalogue())
    old_psu = next(item for item in catalogue if item.category == ComponentKind.POWER_SUPPLY)
    weak_psu = replace(
        old_psu,
        product_id="power_supply-weak",
        psu_wattage=500,
        provided_power_connectors={},
        scores=replace(old_psu.scores, performance=100),
    )
    catalogue.append(weak_psu)
    request = problem(tuple(catalogue))

    solution = BuildOptimizer().optimize(request).solutions[0]

    assert solution.selected[ComponentKind.POWER_SUPPLY].product_id == "power_supply-a"
    assert solution.required_psu_watts <= 750


def test_profile_objectives_choose_performance_and_power_differently() -> None:
    base = list(complete_catalogue())
    gpu = next(item for item in base if item.category == ComponentKind.GPU)
    efficient_gpu = replace(
        gpu,
        product_id="gpu-efficient",
        power_draw_watts=120,
        recommended_psu_watts=500,
        scores=replace(gpu.scores, performance=40, efficiency=100),
    )
    fast_gpu = replace(gpu, scores=replace(gpu.scores, performance=100, efficiency=0))
    catalogue = tuple(
        fast_gpu if item.category == ComponentKind.GPU else item for item in base
    ) + (efficient_gpu,)

    performance = (
        BuildOptimizer()
        .optimize(problem(catalogue, profiles=(BuildPreset.HIGHEST_PERFORMANCE,)))
        .solutions[0]
    )
    low_power = (
        BuildOptimizer()
        .optimize(problem(catalogue, profiles=(BuildPreset.LOWEST_POWER,)))
        .solutions[0]
    )

    assert performance.selected[ComponentKind.GPU].product_id == "gpu-a"
    assert low_power.selected[ComponentKind.GPU].product_id == "gpu-efficient"
    assert low_power.estimated_load_watts < performance.estimated_load_watts


def test_diverse_solutions_differ_by_at_least_two_unlocked_components() -> None:
    request = problem(
        complete_catalogue(alternatives=True),
        profiles=(BuildPreset.BEST_OVERALL, BuildPreset.BEST_VALUE, BuildPreset.LOWEST_POWER),
    )

    result = BuildOptimizer().optimize(request)

    assert len(result.solutions) == 3
    for index, left in enumerate(result.solutions):
        for right in result.solutions[index + 1 :]:
            differences = sum(
                left.selected[category].product_id != right.selected[category].product_id
                for category in left.selected
            )
            assert differences >= 2


def test_independent_validator_rejects_candidate_and_solver_reoptimises() -> None:
    catalogue = complete_catalogue() + (
        replace(
            next(item for item in complete_catalogue() if item.category == ComponentKind.GPU),
            product_id="gpu-b",
            scores=CandidateScores(performance=10),
        ),
    )

    def validator(
        selected: dict[ComponentKind, OptimizationCandidate],
    ) -> tuple[bool, list[str]]:
        accepted = selected[ComponentKind.GPU].product_id != "gpu-a"
        return accepted, [] if accepted else ["independent GPU check failed"]

    result = BuildOptimizer().optimize(problem(catalogue, independent_validator=validator))

    assert result.solutions[0].selected[ComponentKind.GPU].product_id == "gpu-b"
    assert result.rejected_by_validator == 1
    assert "independent GPU check failed" in result.infeasibility_reasons


def test_cp_sat_matches_exhaustive_oracle_on_reduced_catalogue() -> None:
    request = problem(complete_catalogue(alternatives=True))

    cp_solution = BuildOptimizer().optimize(request).solutions[0]
    exhaustive = enumerate_feasible_builds(request)

    assert exhaustive.best is not None
    assert cp_solution.objective_value == exhaustive.best.objective_value
    assert cp_solution.product_ids == exhaustive.best.product_ids


def test_infeasibility_diagnostics_name_missing_eligible_category() -> None:
    catalogue = tuple(
        replace(item, in_stock=False) if item.category == ComponentKind.GPU else item
        for item in complete_catalogue()
    )

    result = BuildOptimizer().optimize(problem(catalogue))

    assert result.status == OptimizationStatus.INFEASIBLE
    assert not result.solutions
    assert any("no eligible gpu" in reason for reason in result.infeasibility_reasons)


def _product(category: ComponentKind, product_id: str) -> MasterProduct:
    attributes = {
        ComponentKind.CPU: CPUAttributes(socket="AM5", peak_power_watts=120),
        ComponentKind.GPU: GPUAttributes(
            vram_gb=16,
            board_power_watts=250,
            recommended_psu_watts=650,
            power_connectors={"pcie_8pin": 1},
        ),
        ComponentKind.MOTHERBOARD: MotherboardAttributes(
            socket="AM5", memory_type="ddr5", wifi_support=True
        ),
        ComponentKind.MEMORY: MemoryAttributes(
            memory_type="ddr5", capacity_gb=32, module_count=2
        ),
        ComponentKind.STORAGE: StorageAttributes(capacity_gb=2_000, interface="nvme_pcie"),
        ComponentKind.POWER_SUPPLY: PowerSupplyAttributes(
            wattage=750,
            pcie_connectors={"pcie_8pin": 2},
            eps_connectors=2,
        ),
        ComponentKind.COOLER: DomainCoolerAttributes(supported_sockets=["AM5"]),
        ComponentKind.CASE: CaseAttributes(case_size="mid_tower"),
    }[category]
    return MasterProduct(
        product_id=product_id,
        category=category,
        brand="Brand",
        model=product_id,
        canonical_name=product_id,
        category_attributes=attributes,
    )


def _listing(product_id: str, price: Decimal = Decimal("100")) -> RetailerOffering:
    return RetailerOffering(
        listing_id=f"listing-{product_id}",
        product_id=product_id,
        retailer="retailer",
        source_listing_id=product_id,
        title=product_id,
        condition=ListingCondition.NEW,
        base_price=price,
        stock_status=StockState.IN_STOCK,
        listing_url=f"https://example.test/{product_id}",
    )


class PassingCompatibilityEngine:
    def __init__(self) -> None:
        self.received_mapping = False

    def check_complete_build(self, components: object) -> CompatibilityReport:
        assert isinstance(components, dict)
        assert all(isinstance(value, dict) for value in components.values())
        self.received_mapping = True
        return CompatibilityReport(
            "compat_v_test",
            (
                CompatibilityResult(
                    rule_id="compat.test",
                    rule_version="compat_v_test",
                    status=EngineCompatibilityStatus.PASS,
                    message="Independent compatibility check passed.",
                    evidence={"selected_product_ids": ["cpu", "gpu"]},
                ),
            ),
        )


def test_domain_adapter_uses_cheapest_listing_and_owned_lock_costs_zero() -> None:
    products = tuple(_product(category, category.value) for category in ComponentKind)
    listings = [_listing(product.product_id) for product in products]
    listings.append(_listing("gpu", Decimal("90")))
    listings[-1] = listings[-1].model_copy(update={"listing_id": "listing-gpu-cheap"})
    request = BuildRequestSpec(
        budget_sgd=Decimal("700"),
        workloads=[WorkloadPreference(name=WorkloadLabel.LOCAL_AI, weight=1)],
        existing_products=[ExistingComponent(category=ComponentKind.GPU, product_id="gpu")],
        requirements=BuildRequirements(
            minimum_gpu_vram_gb=16,
            minimum_memory_gb=32,
            storage_gb=2_000,
            required_memory_type="ddr5",
            wifi_required=True,
        ),
        preferences=BuildPreferences(),
        requested_profiles=[BuildPreset.BEST_OVERALL],
    )

    engine = PassingCompatibilityEngine()
    optimization_problem = OptimizationProblem.from_domain(
        request, products, listings, compatibility_engine=engine
    )
    result = BuildOptimizer().optimize(optimization_problem)

    assert result.is_feasible
    assert optimization_problem.locked_product_ids == {"gpu"}
    assert result.solutions[0].total_price_cents == 70_000
    recommendation = solution_to_domain(result.solutions[0])
    assert engine.received_mapping
    assert recommendation.compatibility_status == CompatVerdict.PASS
    assert len(recommendation.components) == 8
    assert recommendation.compatibility_checks[0].rule_id == "compat.test"
    assert recommendation.compatibility_checks[0].component_ids == ["cpu", "gpu"]


def test_uppercase_compatibility_statuses_are_normalised_in_pairwise_adapter() -> None:
    pair = PairwiseCompatibility("cpu-a", "motherboard-a", EngineCompatibilityStatus.FAIL)
    assert pair.status == CompatVerdict.FAIL
    assert pair.is_forbidden


def test_validator_mappings_and_unknown_objects_fail_closed() -> None:
    catalogue = complete_catalogue()

    result = BuildOptimizer().optimize(
        problem(catalogue, independent_validator=lambda _selected: {})
    )

    assert not result.solutions
    assert any("no explicit feasibility" in reason for reason in result.infeasibility_reasons)


def test_solution_to_domain_rejects_unknown_compatibility_report() -> None:
    solution = BuildOptimizer().optimize(problem(complete_catalogue())).solutions[0]
    report = CompatibilityReport(
        "compat_v_test",
        (
            CompatibilityResult(
                rule_id="compat.unknown",
                rule_version="compat_v_test",
                status=EngineCompatibilityStatus.UNKNOWN,
                message="GPU clearance is unknown.",
            ),
        ),
    )

    with pytest.raises(ValueError, match="GPU clearance is unknown"):
        solution_to_domain(solution, compatibility_report=report)
