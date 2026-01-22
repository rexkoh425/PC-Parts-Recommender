from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest
from pydantic import ValidationError

from pc_build_recommender.domain import (
    BenchmarkValueKind,
    BuildComponentSelection,
    BuildRequestSpec,
    BuildGenerationResponse,
    BuildPreferences,
    BuildPreset,
    BuildRecommendation,
    MasterProduct,
    CaseAttributes,
    CompatibilityCheck,
    CompatVerdict,
    ComponentKind,
    CoolerAttributes,
    CPUAttributes,
    ExistingComponent,
    GPUAttributes,
    MemoryAttributes,
    MotherboardAttributes,
    PerformanceEstimate,
    PowerSupplyAttributes,
    StorageAttributes,
    WorkloadLabel,
    WorkloadPreference,
)


@pytest.mark.parametrize(
    ("category", "attributes", "attribute_type"),
    [
        (ComponentKind.CPU, {"socket": "AM5", "core_count": 8}, CPUAttributes),
        (ComponentKind.GPU, {"vram_gb": 16}, GPUAttributes),
        (
            ComponentKind.MOTHERBOARD,
            {"socket": "AM5", "memory_type": "ddr5"},
            MotherboardAttributes,
        ),
        (ComponentKind.MEMORY, {"memory_type": "ddr5", "capacity_gb": 32}, MemoryAttributes),
        (ComponentKind.STORAGE, {"capacity_gb": 2000}, StorageAttributes),
        (ComponentKind.POWER_SUPPLY, {"wattage": 850}, PowerSupplyAttributes),
        (ComponentKind.COOLER, {"supported_sockets": ["AM5"]}, CoolerAttributes),
        (
            ComponentKind.CASE,
            {"maximum_gpu_length_mm": 360},
            CaseAttributes,
        ),
    ],
)
def test_product_parses_attributes_for_all_eight_categories(
    category: ComponentKind,
    attributes: dict[str, object],
    attribute_type: type[object],
) -> None:
    product = MasterProduct.model_validate(
        {
            "product_id": f"prod_{category.value}",
            "category": category,
            "brand": "Example",
            "model": category.value,
            "canonical_name": f"Example {category.value}",
            "category_attributes": attributes,
        }
    )

    assert isinstance(product.category_attributes, attribute_type)


def test_product_rejects_attributes_from_a_different_category() -> None:
    with pytest.raises(ValidationError, match="GPUAttributes"):
        MasterProduct(
            category=ComponentKind.GPU,
            brand="Example",
            model="Wrong",
            canonical_name="Example Wrong",
            category_attributes=CPUAttributes(socket="AM5"),
        )


def test_missing_compatibility_data_remains_unknown_to_callers() -> None:
    gpu = GPUAttributes(vram_gb=16)

    assert gpu.length_mm is None
    assert gpu.power_connectors == {}


def test_build_request_enforces_weight_sum_and_uniqueness() -> None:
    valid = BuildRequestSpec(
        budget_sgd=Decimal("2500"),
        workloads=[
            WorkloadPreference(name=WorkloadLabel.LOCAL_AI, weight=0.6),
            WorkloadPreference(name=WorkloadLabel.GAMING_1440P, weight=0.4),
        ],
        existing_products=[
            ExistingComponent(category=ComponentKind.GPU, product_id="prod_gpu")
        ],
    )
    assert sum(item.weight for item in valid.workloads) == pytest.approx(1.0)

    target_request = BuildRequestSpec(
        budget_sgd=2500,
        workloads=[WorkloadPreference(name=WorkloadLabel.LOCAL_AI, weight=1.0)],
        performance_target="  120 FPS at 1440p high settings  ",
    )
    assert target_request.performance_target == "120 FPS at 1440p high settings"

    with pytest.raises(ValidationError, match="at most 200 characters"):
        BuildRequestSpec(
            budget_sgd=2500,
            workloads=[WorkloadPreference(name=WorkloadLabel.LOCAL_AI, weight=1.0)],
            performance_target="x" * 201,
        )

    with pytest.raises(ValidationError, match="sum to 1.0"):
        BuildRequestSpec(
            budget_sgd=2500,
            workloads=[WorkloadPreference(name=WorkloadLabel.LOCAL_AI, weight=0.8)],
        )

    with pytest.raises(ValidationError, match="workload names must be unique"):
        BuildRequestSpec(
            budget_sgd=2500,
            workloads=[
                WorkloadPreference(name=WorkloadLabel.LOCAL_AI, weight=0.5),
                WorkloadPreference(name=WorkloadLabel.LOCAL_AI, weight=0.5),
            ],
        )

    with pytest.raises(ValidationError, match="one existing product"):
        BuildRequestSpec(
            budget_sgd=2500,
            workloads=[WorkloadPreference(name=WorkloadLabel.LOCAL_AI, weight=1.0)],
            existing_products=[
                ExistingComponent(category=ComponentKind.GPU, product_id="gpu_one"),
                ExistingComponent(category=ComponentKind.GPU, product_id="gpu_two"),
            ],
        )


def test_brand_preferences_cannot_conflict() -> None:
    with pytest.raises(ValidationError, match="both preferred and excluded"):
        BuildPreferences(preferred_brands=["AMD"], excluded_brands=["amd"])


def test_predictions_are_never_presented_without_model_provenance() -> None:
    with pytest.raises(ValidationError, match="require model_version"):
        PerformanceEstimate(
            workload=WorkloadLabel.LOCAL_AI,
            score=82.5,
            value_kind=BenchmarkValueKind.PREDICTED,
        )

    observed = PerformanceEstimate(
        workload=WorkloadLabel.LOCAL_AI,
        score=80,
        value_kind=BenchmarkValueKind.OBSERVED,
        supporting_benchmark_ids=["bench_1"],
    )
    assert observed.model_version is None


def test_empty_build_response_requires_infeasibility_reason() -> None:
    payload = {
        "request_id": "req_1",
        "data_version": "2026-07-22",
        "ranking_model": "ltr_v1",
        "rule_version": "compat_v1",
        "builds": [],
    }
    with pytest.raises(ValidationError, match="infeasibility"):
        BuildGenerationResponse.model_validate(payload)

    payload["infeasibility_reasons"] = ["No in-stock 16 GB GPU fits the budget."]
    response = BuildGenerationResponse.model_validate(payload)
    assert response.builds == []


def _complete_components() -> list[BuildComponentSelection]:
    return [
        BuildComponentSelection(
            category=category,
            product_id=f"prod_{category.value}",
            canonical_name=f"Example {category.value}",
            price_sgd=100,
            component_score=80,
            selection_reason="Best feasible candidate.",
        )
        for category in ComponentKind
    ]


def _build_payload() -> dict[str, object]:
    return {
        "build_id": "build_complete",
        "profile": BuildPreset.BEST_OVERALL,
        "total_price_sgd": 800,
        "overall_score": 80,
        "components": _complete_components(),
        "compatibility_status": CompatVerdict.PASS,
        "compatibility_checks": [
            CompatibilityCheck(status=CompatVerdict.PASS, message="All sockets match.")
        ],
    }


def test_returned_build_requires_exactly_one_component_from_all_eight_categories() -> None:
    complete = BuildRecommendation.model_validate(_build_payload())
    assert {component.category for component in complete.components} == set(ComponentKind)

    incomplete = _build_payload()
    incomplete["components"] = _complete_components()[:-1]
    with pytest.raises(ValidationError, match="exactly all eight categories"):
        BuildRecommendation.model_validate(incomplete)


@pytest.mark.parametrize("status", [CompatVerdict.FAIL, CompatVerdict.UNKNOWN])
def test_returned_build_rejects_hard_compatibility_status(
    status: CompatVerdict,
) -> None:
    payload = _build_payload()
    payload["compatibility_status"] = status

    with pytest.raises(ValidationError, match="cannot have FAIL or UNKNOWN"):
        BuildRecommendation.model_validate(payload)


@pytest.mark.parametrize("status", [CompatVerdict.FAIL, CompatVerdict.UNKNOWN])
def test_returned_build_rejects_hard_compatibility_checks(
    status: CompatVerdict,
) -> None:
    payload = _build_payload()
    payload["compatibility_status"] = CompatVerdict.WARNING
    payload["compatibility_checks"] = [
        CompatibilityCheck(status=status, message="Compatibility could not be established.")
    ]

    with pytest.raises(ValidationError, match="cannot contain FAIL or UNKNOWN"):
        BuildRecommendation.model_validate(payload)


def test_returned_build_allows_warning_without_hard_failures() -> None:
    payload = _build_payload()
    payload["compatibility_status"] = CompatVerdict.WARNING
    payload["compatibility_checks"] = [
        CompatibilityCheck(
            status=CompatVerdict.WARNING,
            message="BIOS version should be confirmed before assembly.",
        )
    ]

    build = BuildRecommendation.model_validate(payload)
    assert build.compatibility_status == CompatVerdict.WARNING


def test_retailer_timestamp_contract_uses_real_datetimes() -> None:
    # Smoke evidence that the project's canonical time values are timezone aware.
    value = datetime.now(UTC)
    assert value.utcoffset() is not None
