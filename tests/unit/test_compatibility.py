from __future__ import annotations

from copy import deepcopy

import pytest

from pc_build_recommender.compatibility import (
    CompatibilityEngine,
    CompatibilityStatus,
    PowerPolicy,
    check_build_compatibility,
)


def valid_build() -> dict[str, dict[str, object]]:
    return {
        "cpu": {
            "product_id": "cpu-1",
            "category": "cpu",
            "socket": "AM5",
            "generation": "Zen 4",
            "model": "Ryzen 7 7700X",
            "supported_chipsets": ["B650"],
            "peak_power_watts": 142,
            "source_url": "https://manufacturer.invalid/cpu-1",
        },
        "gpu": {
            "product_id": "gpu-1",
            "category": "gpu",
            "length_mm": 300,
            "slot_width": 2.5,
            "board_power_watts": 250,
            "required_power_connectors": {"8-pin PCIe": 2},
        },
        "motherboard": {
            "product_id": "motherboard-1",
            "category": "motherboard",
            "socket": "AM5",
            "chipset": "B650",
            "supported_cpu_generations": ["Zen 4"],
            "memory_type": "DDR5",
            "maximum_memory_gb": 128,
            "memory_slots": 4,
            "form_factor": "ATX",
            "pcie_slots": 3,
            "m2_slots": 3,
            "sata_ports": 4,
            "required_eps_connectors": {"8-pin EPS": 1},
        },
        "memory": {
            "product_id": "memory-1",
            "category": "memory",
            "memory_type": "DDR5",
            "capacity_gb": 32,
            "module_count": 2,
        },
        "storage": {
            "product_id": "storage-1",
            "category": "storage",
            "interface": "M.2 PCIe 4.0 NVMe",
            "capacity_gb": 2_000,
        },
        "power_supply": {
            "product_id": "psu-1",
            "category": "power_supply",
            "wattage": 850,
            "form_factor": "ATX",
            "pcie_connectors": {"8-pin PCIe": 4},
            "eps_connectors": {"8-pin EPS": 2},
        },
        "cooler": {
            "product_id": "cooler-1",
            "category": "cooler",
            "cooler_type": "air",
            "supported_sockets": ["AM5", "AM4"],
            "height_mm": 155,
        },
        "case": {
            "product_id": "case-1",
            "category": "case",
            "supported_motherboard_sizes": ["ATX", "Micro-ATX", "Mini-ITX"],
            "maximum_gpu_length_mm": 340,
            "maximum_gpu_slot_width": 3.5,
            "maximum_cooler_height_mm": 165,
            "supported_radiator_sizes_mm": [120, 240, 280, 360],
            "supported_psu_sizes": ["ATX"],
        },
    }


def result_for(report: object, rule_id: str):
    matches = report.by_rule(rule_id)  # type: ignore[attr-defined]
    assert len(matches) == 1
    return matches[0]


def test_complete_known_valid_build_passes_every_rule() -> None:
    report = CompatibilityEngine().check_complete_build(valid_build())

    assert report.status is CompatibilityStatus.PASS
    assert report.is_compatible
    assert report.is_feasible
    assert all(result.rule_version == "compat_v2" for result in report.results)
    assert all(result.evidence for result in report.results)


def test_missing_critical_field_is_unknown_and_never_compatible() -> None:
    build = valid_build()
    del build["case"]["maximum_gpu_length_mm"]

    report = CompatibilityEngine().check_build(build)
    result = result_for(report, "compat.gpu_case.length")

    assert result.status is CompatibilityStatus.UNKNOWN
    assert "case_maximum_gpu_length_mm" in result.evidence["missing_fields"]
    assert report.status is CompatibilityStatus.UNKNOWN
    assert not report.is_compatible


@pytest.mark.parametrize(
    ("left_category", "left", "right_category", "right", "rule_id"),
    [
        (
            "cpu",
            {"socket": "AM5", "generation": "Zen 4"},
            "motherboard",
            {"socket": "AM5", "chipset": "B650", "supported_cpu_generations": []},
            "compat.cpu_motherboard.chipset_bios",
        ),
        (
            "motherboard",
            {"form_factor": "ATX"},
            "case",
            {"supported_motherboard_sizes": []},
            "compat.motherboard_case.form_factor",
        ),
        (
            "cooler",
            {"supported_sockets": []},
            "cpu",
            {"socket": "AM5"},
            "compat.cooler_cpu.socket",
        ),
    ],
)
def test_empty_default_support_lists_are_missing_evidence_not_hard_failures(
    left_category: str,
    left: dict[str, object],
    right_category: str,
    right: dict[str, object],
    rule_id: str,
) -> None:
    report = CompatibilityEngine().check_pair(left_category, left, right_category, right)

    assert result_for(report, rule_id).status is CompatibilityStatus.UNKNOWN


@pytest.mark.parametrize(
    ("mutate", "rule_id"),
    [
        (
            lambda build: build["motherboard"].update(socket="LGA1700"),
            "compat.cpu_motherboard.socket",
        ),
        (
            lambda build: build["motherboard"].update(supported_cpu_generations=["Zen 3"]),
            "compat.cpu_motherboard.chipset_bios",
        ),
        (
            lambda build: build["memory"].update(memory_type="DDR4"),
            "compat.memory_motherboard.generation",
        ),
        (
            lambda build: build["memory"].update(capacity_gb=256),
            "compat.memory_motherboard.capacity",
        ),
        (
            lambda build: build["memory"].update(module_count=8),
            "compat.memory_motherboard.modules",
        ),
        (
            lambda build: build["case"].update(supported_motherboard_sizes=["Mini-ITX"]),
            "compat.motherboard_case.form_factor",
        ),
        (
            lambda build: build["case"].update(maximum_gpu_length_mm=299),
            "compat.gpu_case.length",
        ),
        (
            lambda build: build["case"].update(maximum_gpu_slot_width=2),
            "compat.gpu_case.slot_width",
        ),
        (
            lambda build: build["case"].update(maximum_cooler_height_mm=150),
            "compat.cooler_case.clearance",
        ),
        (
            lambda build: build["cooler"].update(supported_sockets=["LGA1700"]),
            "compat.cooler_cpu.socket",
        ),
        (
            lambda build: build["power_supply"].update(wattage=500),
            "compat.power_supply.capacity",
        ),
        (
            lambda build: build["power_supply"].update(pcie_connectors={"8-pin PCIe": 1}),
            "compat.power_supply.gpu_connectors",
        ),
        (
            lambda build: build["power_supply"].update(eps_connectors={"8-pin EPS": 0}),
            "compat.power_supply.eps_connectors",
        ),
        (
            lambda build: build["storage"].update(interface="U.2"),
            "compat.storage_motherboard.interface",
        ),
    ],
)
def test_known_hard_incompatibilities_fail(mutate, rule_id: str) -> None:
    build = valid_build()
    mutate(build)

    report = CompatibilityEngine().check_build(build)

    assert result_for(report, rule_id).status is CompatibilityStatus.FAIL
    assert report.status is CompatibilityStatus.FAIL
    assert not report.is_compatible


def test_bios_update_is_a_feasible_warning_with_versioned_evidence() -> None:
    build = valid_build()
    build["motherboard"].update(
        minimum_bios_versions={"Zen 4": "F10"},
        bios_version="F5",
        bios_update_available=True,
    )

    report = CompatibilityEngine(rule_version="compat_test_7").check_build(build)
    result = result_for(report, "compat.cpu_motherboard.chipset_bios")

    assert result.status is CompatibilityStatus.WARNING
    assert result.rule_version == "compat_test_7"
    assert result.evidence["minimum_bios_version"] == "F10"
    assert report.status is CompatibilityStatus.WARNING
    assert report.is_compatible


def test_liquid_cooler_uses_explicit_radiator_support() -> None:
    engine = CompatibilityEngine()
    case = valid_build()["case"]

    passes = engine.check_pair(
        "cooler",
        {"cooler_type": "aio", "radiator_size_mm": 280},
        "case",
        case,
    )
    fails = engine.check_pair(
        "cooler",
        {"cooler_type": "aio", "radiator_size_mm": 420},
        "case",
        case,
    )

    assert passes.status is CompatibilityStatus.PASS
    assert fails.status is CompatibilityStatus.FAIL


def test_storage_requires_supported_interface_and_available_slot() -> None:
    engine = CompatibilityEngine()
    motherboard = {
        "supported_storage_interfaces": ["NVMe", "SATA"],
        "storage_slot_counts": {"NVMe": 0, "SATA": 4},
    }

    report = engine.check_pair("storage", {"interface": "NVMe"}, "motherboard", motherboard)

    assert (
        result_for(report, "compat.storage_motherboard.interface").status
        is CompatibilityStatus.PASS
    )
    assert result_for(report, "compat.storage_motherboard.slots").status is CompatibilityStatus.FAIL


def test_existing_component_must_be_retained_by_stable_identity() -> None:
    build = valid_build()
    retained = deepcopy(build["gpu"])

    kept = CompatibilityEngine().check_build(build, existing_components=[retained])
    replaced_build = deepcopy(build)
    replaced_build["gpu"]["product_id"] = "gpu-2"
    replaced = CompatibilityEngine().check_build(replaced_build, existing_components=[retained])

    assert result_for(kept, "compat.existing.retained").status is CompatibilityStatus.PASS
    assert result_for(replaced, "compat.existing.retained").status is CompatibilityStatus.FAIL


def test_existing_component_with_no_identity_is_unknown() -> None:
    report = CompatibilityEngine().check_build(
        valid_build(), existing_components=[{"category": "gpu", "model": "unkeyed"}]
    )

    assert result_for(report, "compat.existing.retained").status is CompatibilityStatus.UNKNOWN
    assert not report.is_compatible


def test_complete_build_enforces_exactly_one_component_per_required_category() -> None:
    build = valid_build()
    build["gpu"] = [build["gpu"], deepcopy(build["gpu"])]
    del build["storage"]

    report = CompatibilityEngine().check_build(build)

    assert result_for(report, "compat.build.cardinality.gpu").status is CompatibilityStatus.FAIL
    assert result_for(report, "compat.build.cardinality.storage").status is CompatibilityStatus.FAIL


def test_nested_attribute_mappings_and_category_aliases_are_supported() -> None:
    report = CompatibilityEngine().check_pair(
        "RAM",
        {
            "product_id": "ram",
            "category_attributes": {
                "Memory Type": "DDR5",
                "Capacity GB": 64,
                "Module Count": 2,
            },
        },
        "mainboard",
        {
            "product_id": "board",
            "category_attributes": {
                "DDR Type": "DDR5",
                "Maximum Memory GB": 128,
                "DIMM Slots": 4,
            },
        },
    )

    assert report.status is CompatibilityStatus.PASS


def test_connector_parser_accepts_common_catalogue_strings() -> None:
    report = CompatibilityEngine().check_pair(
        "gpu",
        {"power_connectors": "2x 8-pin PCIe"},
        "psu",
        {"pcie_connectors": ["PCIe 6+2-pin", "PCIe 6+2-pin", "PCIe 6+2-pin"]},
    )

    assert report.status is CompatibilityStatus.PASS


def test_one_convertible_pcie_lead_cannot_satisfy_two_connector_requirements() -> None:
    report = CompatibilityEngine().check_pair(
        "gpu",
        {"power_connectors": {"8-pin PCIe": 1, "6-pin PCIe": 1}},
        "psu",
        {"pcie_connectors": ["PCIe 6+2-pin"]},
    )

    assert report.status is CompatibilityStatus.FAIL


def test_absent_optional_motherboard_eps_requirement_does_not_invent_a_rule() -> None:
    build = valid_build()
    del build["motherboard"]["required_eps_connectors"]
    del build["power_supply"]["eps_connectors"]

    report = CompatibilityEngine().check_build(build)

    assert not report.by_rule("compat.power_supply.eps_connectors")
    assert report.status is CompatibilityStatus.PASS


def test_power_policy_is_configurable_and_validated() -> None:
    build = valid_build()
    build["power_supply"]["wattage"] = 600

    strict = CompatibilityEngine().check_build(build)
    relaxed = CompatibilityEngine(
        power_policy=PowerPolicy(headroom_ratio=0.10, accessory_allowance_w=50)
    ).check_build(build)

    assert result_for(strict, "compat.power_supply.capacity").status is CompatibilityStatus.FAIL
    assert result_for(relaxed, "compat.power_supply.capacity").status is CompatibilityStatus.PASS
    with pytest.raises(ValueError):
        PowerPolicy(headroom_ratio=1.1)
    with pytest.raises(ValueError):
        PowerPolicy(gpu_transient_multiplier=0.9)


def test_convenience_function_and_json_representation() -> None:
    report = check_build_compatibility(valid_build(), rule_version="compat_serialized")

    payload = report.to_dict()

    assert payload["rule_version"] == "compat_serialized"
    assert payload["status"] == "PASS"
    assert payload["is_compatible"] is True
    assert payload["status_counts"]["UNKNOWN"] == 0
    assert payload["missing_data_risk"]["level"] == "NONE"
    assert payload["results"][0]["rule_id"].startswith("compat.build.cardinality")


def test_invalid_component_shape_raises_clear_type_error() -> None:
    with pytest.raises(TypeError, match="must be a component mapping"):
        CompatibilityEngine().check_build({"cpu": "not a component"})  # type: ignore[dict-item]


def test_required_bios_with_unknown_installed_version_is_fail_closed() -> None:
    build = valid_build()
    build["motherboard"]["minimum_bios_versions"] = {"Zen 4": "F10"}

    report = CompatibilityEngine().check_build(build)
    result = result_for(report, "compat.cpu_motherboard.chipset_bios")

    assert result.status is CompatibilityStatus.UNKNOWN
    assert result.evidence["missing_fields"] == ["installed_bios_version"]
    assert report.missing_data_risk.blocks_feasibility
    assert "installed_bios_version" in report.missing_data_risk.missing_fields


def test_bios_support_matrix_can_block_cpu_with_no_update_path() -> None:
    build = valid_build()
    build["motherboard"].update(
        cpu_support_matrix={
            "Ryzen 7 7700X": {
                "supported": True,
                "minimum_bios_version": "F10",
                "bios_update_available": False,
            }
        },
        bios_version="F5",
    )

    result = result_for(
        CompatibilityEngine().check_build(build),
        "compat.cpu_motherboard.chipset_bios",
    )

    assert result.status is CompatibilityStatus.FAIL


def test_beta_bios_support_is_an_auditable_warning() -> None:
    build = valid_build()
    build["motherboard"]["cpu_support_matrix"] = {
        "Ryzen 7 7700X": {"supported": True, "support_status": "beta"}
    }

    result = result_for(
        CompatibilityEngine().check_build(build),
        "compat.cpu_motherboard.chipset_bios",
    )

    assert result.status is CompatibilityStatus.WARNING
    assert result.evidence["cpu_support_status"] == "beta"


def test_radiator_position_and_thickness_are_both_enforced() -> None:
    engine = CompatibilityEngine()
    case = {
        "radiator_support_mm": {"front": [280, 360], "top": [240]},
        "maximum_radiator_thickness_mm": {"front": 65, "top": 55},
    }
    cooler = {
        "cooler_type": "aio",
        "radiator_size_mm": 240,
        "required_radiator_position": "top",
        "radiator_thickness_mm": 52,
    }

    passes = engine.check_pair("cooler", cooler, "case", case)
    too_thick = engine.check_pair("cooler", {**cooler, "radiator_thickness_mm": 60}, "case", case)

    assert passes.status is CompatibilityStatus.PASS
    assert too_thick.status is CompatibilityStatus.FAIL


def test_pcie_slot_evidence_is_required_for_a_discrete_gpu() -> None:
    engine = CompatibilityEngine()
    gpu = {"host_interface": "PCIe 4.0 x16"}

    unknown = engine.check_pair("gpu", gpu, "motherboard", {})
    failed = engine.check_pair("gpu", gpu, "motherboard", {"pcie_slots": 0})
    passed = engine.check_pair("gpu", gpu, "motherboard", {"pcie_x16_slots": 1})

    assert unknown.status is CompatibilityStatus.UNKNOWN
    assert failed.status is CompatibilityStatus.FAIL
    assert passed.status is CompatibilityStatus.PASS


def test_storage_drive_bay_type_and_count_are_enforced() -> None:
    engine = CompatibilityEngine()
    storage = {"interface": "SATA", "form_factor": "3.5-inch"}

    passed = engine.check_pair(
        "storage", storage, "case", {"drive_bays_by_form_factor": {"3.5-inch": 2}}
    )
    failed = engine.check_pair(
        "storage", storage, "case", {"drive_bays_by_form_factor": {"2.5-inch": 2}}
    )

    assert passed.status is CompatibilityStatus.PASS
    assert failed.status is CompatibilityStatus.FAIL


def test_published_shared_resource_conflict_blocks_complete_build() -> None:
    build = valid_build()
    build["storage"].update(interface="PCIe", form_factor="add-in-card")
    build["motherboard"].update(
        supported_storage_interfaces=["PCIe"],
        resource_conflicts=[
            {
                "resources": ["gpu_pcie", "storage_pcie"],
                "message": "Second expansion device disables the GPU slot.",
                "evidence_source": "manufacturer-manual",
            }
        ],
    )

    result = result_for(
        CompatibilityEngine().check_build(build),
        "compat.motherboard.resource_conflicts",
    )

    assert result.status is CompatibilityStatus.FAIL
    assert result.evidence["triggered_conflicts"][0]["evidence_source"] == ("manufacturer-manual")


def test_front_radiator_drive_bay_loss_is_a_hard_conflict() -> None:
    build = valid_build()
    build["cooler"].update(
        cooler_type="aio",
        radiator_size_mm=360,
        required_radiator_position="front",
    )
    build["storage"].update(interface="SATA", form_factor="3.5-inch")
    build["case"].update(
        radiator_support_mm={"front": [360]},
        drive_bays_by_form_factor={"3.5-inch": 1},
        radiator_drive_bay_loss={"front": {"360": 1}},
    )

    result = result_for(
        CompatibilityEngine().check_build(build),
        "compat.cooler_case.radiator_drive_bays",
    )

    assert result.status is CompatibilityStatus.FAIL


def test_discontinued_new_part_fails_but_retained_user_part_warns() -> None:
    build = valid_build()
    build["gpu"]["status"] = "discontinued"
    retained = deepcopy(build["gpu"])

    new_selection = CompatibilityEngine().check_build(build)
    retained_selection = CompatibilityEngine().check_build(build, existing_components=[retained])

    assert (
        result_for(new_selection, "compat.product.lifecycle.gpu").status is CompatibilityStatus.FAIL
    )
    assert (
        result_for(retained_selection, "compat.product.lifecycle.gpu").status
        is CompatibilityStatus.WARNING
    )


def test_unrecognised_connector_type_is_unknown_not_no_connector_required() -> None:
    report = CompatibilityEngine().check_pair(
        "gpu",
        {"power_connectors": {"vendor-mystery-link": 1}},
        "psu",
        {"pcie_connectors": {"8-pin PCIe": 4}},
    )

    assert report.status is CompatibilityStatus.UNKNOWN
    assert report.results[0].evidence["unparsed_required_connectors"]


def test_explicit_gpu_transient_can_fail_when_continuous_headroom_passes() -> None:
    build = valid_build()
    build["power_supply"]["wattage"] = 650
    build["gpu"]["transient_power_w"] = 500

    report = CompatibilityEngine().check_build(build)

    assert result_for(report, "compat.power_supply.capacity").status is CompatibilityStatus.PASS
    assert (
        result_for(report, "compat.power_supply.transient_capacity").status
        is CompatibilityStatus.FAIL
    )


def test_power_supply_case_form_factor_is_hard_compatibility() -> None:
    report = CompatibilityEngine().check_pair(
        "power_supply",
        {"form_factor": "ATX"},
        "case",
        {"supported_psu_sizes": ["SFX", "SFX-L"]},
    )

    assert report.status is CompatibilityStatus.FAIL
