from __future__ import annotations

from copy import deepcopy

from hypothesis import given
from hypothesis import strategies as st

from pc_build_recommender.compatibility import CompatibilityEngine, CompatVerdict


def base_build() -> dict[str, dict[str, object]]:
    return {
        "cpu": {
            "product_id": "cpu",
            "category": "cpu",
            "socket": "AM5",
            "generation": "Zen 4",
            "supported_chipsets": ["B650"],
            "peak_power_w": 140,
        },
        "gpu": {
            "product_id": "gpu",
            "category": "gpu",
            "length_mm": 300,
            "slot_width": 3,
            "board_power_w": 250,
            "required_power_connectors": {"8-pin PCIe": 2},
        },
        "motherboard": {
            "product_id": "board",
            "category": "motherboard",
            "socket": "AM5",
            "chipset": "B650",
            "supported_cpu_generations": ["Zen 4"],
            "memory_type": "DDR5",
            "maximum_memory_gb": 128,
            "memory_slots": 4,
            "form_factor": "ATX",
            "pcie_slots": 3,
            "m2_slots": 2,
            "sata_ports": 4,
        },
        "memory": {
            "product_id": "memory",
            "category": "memory",
            "memory_type": "DDR5",
            "capacity_gb": 32,
            "module_count": 2,
        },
        "storage": {
            "product_id": "storage",
            "category": "storage",
            "interface": "NVMe",
        },
        "power_supply": {
            "product_id": "psu",
            "category": "power_supply",
            "wattage": 850,
            "form_factor": "ATX",
            "pcie_connectors": {"8-pin PCIe": 4},
        },
        "cooler": {
            "product_id": "cooler",
            "category": "cooler",
            "cooler_type": "air",
            "height_mm": 150,
            "supported_sockets": ["AM5"],
        },
        "case": {
            "product_id": "case",
            "category": "case",
            "supported_motherboard_sizes": ["ATX"],
            "maximum_gpu_length_mm": 350,
            "maximum_gpu_slot_width": 4,
            "maximum_cooler_height_mm": 160,
            "supported_psu_sizes": ["ATX"],
        },
    }


def status_for(report: object, rule_id: str) -> CompatVerdict:
    results = report.by_rule(rule_id)  # type: ignore[attr-defined]
    assert len(results) == 1
    return results[0].status


@given(
    gpu_length=st.integers(min_value=1, max_value=700),
    deficit=st.integers(min_value=1, max_value=300),
    further_reduction=st.integers(min_value=0, max_value=300),
)
def test_reducing_case_clearance_cannot_turn_failed_gpu_into_pass(
    gpu_length: int, deficit: int, further_reduction: int
) -> None:
    engine = CompatibilityEngine()
    first_clearance = max(0, gpu_length - deficit)
    smaller_clearance = max(0, first_clearance - further_reduction)
    gpu = {"length_mm": gpu_length, "slot_width": 2}

    first = engine.check_pair(
        "gpu",
        gpu,
        "case",
        {"maximum_gpu_length_mm": first_clearance, "maximum_gpu_slot_width": 4},
    )
    second = engine.check_pair(
        "gpu",
        gpu,
        "case",
        {"maximum_gpu_length_mm": smaller_clearance, "maximum_gpu_slot_width": 4},
    )

    assert status_for(first, "compat.gpu_case.length") is CompatVerdict.FAIL
    assert status_for(second, "compat.gpu_case.length") is CompatVerdict.FAIL


@given(
    cpu_power=st.integers(min_value=1, max_value=400),
    gpu_power=st.integers(min_value=1, max_value=700),
    extra_capacity=st.integers(min_value=0, max_value=1_000),
)
def test_increasing_psu_wattage_cannot_create_a_wattage_failure(
    cpu_power: int, gpu_power: int, extra_capacity: int
) -> None:
    engine = CompatibilityEngine()
    required = int((cpu_power + gpu_power + 100) * 1.25 + 0.999999)
    build = base_build()
    build["cpu"]["peak_power_w"] = cpu_power
    build["gpu"]["board_power_w"] = gpu_power
    build["power_supply"]["wattage"] = required
    at_threshold = engine.check_build(build)
    build["power_supply"]["wattage"] = required + extra_capacity
    increased = engine.check_build(build)

    assert status_for(at_threshold, "compat.power_supply.capacity") is CompatVerdict.PASS
    assert status_for(increased, "compat.power_supply.capacity") is CompatVerdict.PASS


@given(
    memory_capacity=st.integers(min_value=1, max_value=1_024),
    motherboard_capacity=st.integers(min_value=1, max_value=1_024),
)
def test_memory_capacity_rule_matches_numeric_boundary(
    memory_capacity: int, motherboard_capacity: int
) -> None:
    report = CompatibilityEngine().check_pair(
        "memory",
        {"memory_type": "DDR5", "capacity_gb": memory_capacity, "module_count": 1},
        "motherboard",
        {
            "memory_type": "DDR5",
            "maximum_memory_gb": motherboard_capacity,
            "memory_slots": 1,
        },
    )

    expected = (
        CompatVerdict.PASS
        if memory_capacity <= motherboard_capacity
        else CompatVerdict.FAIL
    )
    assert status_for(report, "compat.memory_motherboard.capacity") is expected


@given(ddr4_spelling=st.sampled_from(["DDR4", "ddr-4", "D D R 4"]))
def test_ddr4_memory_never_passes_ddr5_only_motherboard(ddr4_spelling: str) -> None:
    report = CompatibilityEngine().check_pair(
        "memory",
        {"memory_type": ddr4_spelling, "capacity_gb": 32, "module_count": 2},
        "motherboard",
        {"memory_type": "DDR5", "maximum_memory_gb": 128, "memory_slots": 4},
    )

    assert status_for(report, "compat.memory_motherboard.generation") is CompatVerdict.FAIL


@given(wrong_socket=st.sampled_from(["AM4", "LGA1700", "LGA1851", "sTR5"]))
def test_adding_hard_incompatibility_cannot_increase_feasible_build_set(
    wrong_socket: str,
) -> None:
    engine = CompatibilityEngine()
    build = base_build()
    valid_report = engine.check_build(build)
    invalid_build = deepcopy(build)
    invalid_build["motherboard"]["socket"] = wrong_socket
    invalid_report = engine.check_build(invalid_build)

    assert valid_report.is_feasible
    assert not invalid_report.is_feasible
    assert invalid_report.has_failures


@given(category_to_remove=st.sampled_from(tuple(base_build())))
def test_removing_required_category_always_makes_complete_build_infeasible(
    category_to_remove: str,
) -> None:
    build = base_build()
    del build[category_to_remove]

    report = CompatibilityEngine().check_complete_build(build)

    rule_id = f"compat.build.cardinality.{category_to_remove}"
    assert status_for(report, rule_id) is CompatVerdict.FAIL
    assert not report.is_feasible


@given(
    cpu_power=st.integers(min_value=1, max_value=400),
    gpu_power=st.integers(min_value=1, max_value=700),
    additional_capacity=st.integers(min_value=0, max_value=1_000),
)
def test_increasing_transient_capacity_cannot_create_transient_failure(
    cpu_power: int, gpu_power: int, additional_capacity: int
) -> None:
    engine = CompatibilityEngine()
    build = base_build()
    build["cpu"]["peak_power_w"] = cpu_power
    build["gpu"]["board_power_w"] = gpu_power
    required = int(cpu_power + gpu_power * 1.25 + 100 + 0.999999)
    build["power_supply"]["wattage"] = required
    at_threshold = engine.check_build(build)
    build["power_supply"]["wattage"] = required + additional_capacity
    increased = engine.check_build(build)

    assert (
        status_for(at_threshold, "compat.power_supply.transient_capacity")
        is CompatVerdict.PASS
    )
    assert (
        status_for(increased, "compat.power_supply.transient_capacity") is CompatVerdict.PASS
    )


@given(
    initial_bays=st.integers(min_value=1, max_value=20),
    added_bays=st.integers(min_value=0, max_value=20),
)
def test_increasing_compatible_drive_bays_cannot_create_drive_bay_failure(
    initial_bays: int, added_bays: int
) -> None:
    engine = CompatibilityEngine()
    storage = {"interface": "SATA", "form_factor": "3.5-inch"}
    initial = engine.check_pair(
        "storage",
        storage,
        "case",
        {"drive_bays_by_form_factor": {"3.5-inch": initial_bays}},
    )
    increased = engine.check_pair(
        "storage",
        storage,
        "case",
        {"drive_bays_by_form_factor": {"3.5-inch": initial_bays + added_bays}},
    )

    assert status_for(initial, "compat.storage_case.drive_bay") is CompatVerdict.PASS
    assert status_for(increased, "compat.storage_case.drive_bay") is CompatVerdict.PASS
