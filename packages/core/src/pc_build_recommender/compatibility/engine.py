"""Deterministic, versioned compatibility rules for complete PC builds."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Final

from ._access import (
    MISSING,
    Component,
    connector_inventory,
    form_factor,
    form_factors,
    integer,
    is_missing,
    lookup,
    number,
    product_identity,
    source_evidence,
    storage_interface,
    storage_interfaces,
    token,
    tokens,
)
from .models import CompatibilityReport, CompatibilityResult, CompatibilityStatus, PowerPolicy

DEFAULT_RULE_VERSION: Final = "compat_v2"
COMPATIBILITY_AUTHORITY_KEY: Final = "_compatibility_authority"
AUTHORITATIVE_COMPATIBILITY_POLICY: Final = "authoritative_only"
CONTROLLED_NON_PRODUCTION_POLICY: Final = "controlled_non_production"
DEFAULT_REQUIRED_CATEGORIES: Final = (
    "cpu",
    "gpu",
    "motherboard",
    "memory",
    "storage",
    "power_supply",
    "cooler",
    "case",
)

_CATEGORY_ALIASES = {
    "processor": "cpu",
    "graphics": "gpu",
    "graphics_card": "gpu",
    "video_card": "gpu",
    "mainboard": "motherboard",
    "mother_board": "motherboard",
    "ram": "memory",
    "memory_kit": "memory",
    "ssd": "storage",
    "hdd": "storage",
    "power_supply_unit": "power_supply",
    "psu": "power_supply",
    "cpu_cooler": "cooler",
    "chassis": "case",
}


def _category(value: object) -> str:
    raw = re.sub(r"[^a-z0-9]+", "_", str(value).casefold()).strip("_")
    return _CATEGORY_ALIASES.get(raw, raw)


class CompatibilityEngine:
    """Evaluate pairwise and full-build compatibility from component mappings.

    Rule evaluation is intentionally deterministic and fail-closed.  An absent dimension,
    socket, connector requirement, or other critical field is reported as ``UNKNOWN`` and can
    never become a ``PASS`` merely because no conflict was observed.
    """

    def __init__(
        self,
        *,
        rule_version: str = DEFAULT_RULE_VERSION,
        power_policy: PowerPolicy | None = None,
        required_categories: Sequence[str] = DEFAULT_REQUIRED_CATEGORIES,
    ) -> None:
        if not rule_version.strip():
            raise ValueError("rule_version must not be empty")
        self.rule_version = rule_version
        self.power_policy = power_policy or PowerPolicy()
        self.required_categories = tuple(_category(item) for item in required_categories)
        if len(self.required_categories) != len(set(self.required_categories)):
            raise ValueError("required_categories must be unique")

    def check_pair(
        self,
        left_category: str,
        left: Component,
        right_category: str,
        right: Component,
    ) -> CompatibilityReport:
        """Run all rules applicable to a component pair."""

        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            raise TypeError("components must be mappings")
        authority_unknowns = self._authority_unknowns(
            ((_category(left_category), left), (_category(right_category), right))
        )
        if authority_unknowns:
            return CompatibilityReport(self.rule_version, tuple(authority_unknowns))
        results = self._pair_results(
            _category(left_category), left, _category(right_category), right
        )
        return CompatibilityReport(self.rule_version, tuple(results))

    def check_build(
        self,
        components: Mapping[str, Component | Sequence[Component]],
        *,
        existing_components: Iterable[Component] = (),
    ) -> CompatibilityReport:
        """Independently validate cardinality, every pairwise rule, and full-build power."""

        grouped = self._normalise_build(components)
        results: list[CompatibilityResult] = []

        for category_name in self.required_categories:
            selected = grouped.get(category_name, ())
            count = len(selected)
            status = CompatibilityStatus.PASS if count == 1 else CompatibilityStatus.FAIL
            message = (
                f"Exactly one {category_name} is selected."
                if count == 1
                else f"A complete build requires exactly one {category_name}; found {count}."
            )
            results.append(
                self._result(
                    f"compat.build.cardinality.{category_name}",
                    status,
                    message,
                    {
                        "category": category_name,
                        "required_count": 1,
                        "actual_count": count,
                        "selected_product_ids": [
                            identity
                            for component in selected
                            if (identity := product_identity(component)) is not None
                        ],
                    },
                )
            )

        existing = tuple(existing_components)
        authority_components = [
            (category_name, component)
            for category_name, selected in grouped.items()
            for component in selected
        ]
        authority_components.extend(("existing", component) for component in existing)
        authority_unknowns = self._authority_unknowns(authority_components)
        if authority_unknowns:
            # Cardinality is structural and does not rely on product specifications.  Every
            # data-derived rule is suppressed until the selected records have authoritative
            # compatibility provenance, so community fields cannot create PASS or FAIL.
            results.extend(authority_unknowns)
            return CompatibilityReport(self.rule_version, tuple(results))

        pairings = (
            ("cpu", "motherboard"),
            ("gpu", "motherboard"),
            ("memory", "motherboard"),
            ("motherboard", "case"),
            ("gpu", "case"),
            ("cooler", "case"),
            ("storage", "case"),
            ("power_supply", "case"),
            ("cpu", "cooler"),
            ("gpu", "power_supply"),
            ("storage", "motherboard"),
        )
        for left_category, right_category in pairings:
            left = self._one(grouped, left_category)
            right = self._one(grouped, right_category)
            if left is not None and right is not None:
                results.extend(self._pair_results(left_category, left, right_category, right))

        # EPS requirements vary by board and are not part of the minimum motherboard schema.
        # Enforce the rule when an authoritative requirement is present, rather than inventing
        # a connector count for records that do not publish one.
        motherboard = self._one(grouped, "motherboard")
        power_supply = self._one(grouped, "power_supply")
        if motherboard is not None and power_supply is not None:
            required_eps = lookup(
                motherboard, "required_eps_connectors", "cpu_power_connectors", "eps_connectors"
            )
            if not is_missing(required_eps):
                results.append(self._motherboard_psu_connectors(motherboard, power_supply))

        cpu = self._one(grouped, "cpu")
        gpu = self._one(grouped, "gpu")
        if cpu is not None and gpu is not None and power_supply is not None:
            results.append(self._power_capacity(cpu, gpu, power_supply))
            results.append(self._power_transient_capacity(cpu, gpu, power_supply))

        conflict_result = self._published_resource_conflicts(grouped)
        if conflict_result is not None:
            results.append(conflict_result)
        radiator_bay_result = self._radiator_drive_bay_constraint(grouped)
        if radiator_bay_result is not None:
            results.append(radiator_bay_result)

        results.extend(self._product_lifecycle_results(grouped, existing))
        results.extend(self._retained_component_results(grouped, existing))
        return CompatibilityReport(self.rule_version, tuple(results))

    def check_complete_build(
        self,
        components: Mapping[str, Component | Sequence[Component]],
        *,
        existing_components: Iterable[Component] = (),
    ) -> CompatibilityReport:
        """Explicit alias for callers that want complete-build validation."""

        return self.check_build(components, existing_components=existing_components)

    def _normalise_build(
        self, components: Mapping[str, Component | Sequence[Component]]
    ) -> dict[str, tuple[Component, ...]]:
        if not isinstance(components, Mapping):
            raise TypeError("components must be a mapping keyed by category")
        grouped: dict[str, list[Component]] = {}
        for raw_category, value in components.items():
            category_name = _category(raw_category)
            if isinstance(value, Mapping):
                selected: Sequence[Component] = (value,)
            elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                if not all(isinstance(item, Mapping) for item in value):
                    raise TypeError(f"all {category_name} components must be mappings")
                selected = value
            else:
                raise TypeError(f"{category_name} must be a component mapping or sequence")
            grouped.setdefault(category_name, []).extend(selected)
        return {key: tuple(value) for key, value in grouped.items()}

    @staticmethod
    def _one(grouped: Mapping[str, tuple[Component, ...]], category_name: str) -> Component | None:
        selected = grouped.get(category_name, ())
        return selected[0] if len(selected) == 1 else None

    def _pair_results(
        self,
        left_category: str,
        left: Component,
        right_category: str,
        right: Component,
    ) -> list[CompatibilityResult]:
        pair = (left_category, right_category)
        if pair == ("cpu", "motherboard"):
            return self._cpu_motherboard(left, right)
        if pair == ("motherboard", "cpu"):
            return self._cpu_motherboard(right, left)
        if pair == ("memory", "motherboard"):
            return self._memory_motherboard(left, right)
        if pair == ("motherboard", "memory"):
            return self._memory_motherboard(right, left)
        if pair == ("gpu", "motherboard"):
            return [self._gpu_motherboard_pcie(left, right)]
        if pair == ("motherboard", "gpu"):
            return [self._gpu_motherboard_pcie(right, left)]
        if pair == ("motherboard", "case"):
            return [self._motherboard_case(left, right)]
        if pair == ("case", "motherboard"):
            return [self._motherboard_case(right, left)]
        if pair == ("gpu", "case"):
            return self._gpu_case(left, right)
        if pair == ("case", "gpu"):
            return self._gpu_case(right, left)
        if pair == ("cooler", "case"):
            return [self._cooler_case(left, right)]
        if pair == ("case", "cooler"):
            return [self._cooler_case(right, left)]
        if pair == ("storage", "case"):
            return [self._storage_case(left, right)]
        if pair == ("case", "storage"):
            return [self._storage_case(right, left)]
        if pair == ("power_supply", "case"):
            return [self._power_supply_case(left, right)]
        if pair == ("case", "power_supply"):
            return [self._power_supply_case(right, left)]
        if pair == ("cpu", "cooler"):
            return [self._cooler_cpu(right, left)]
        if pair == ("cooler", "cpu"):
            return [self._cooler_cpu(left, right)]
        if pair == ("gpu", "power_supply"):
            return [self._gpu_psu_connectors(left, right)]
        if pair == ("power_supply", "gpu"):
            return [self._gpu_psu_connectors(right, left)]
        if pair == ("motherboard", "power_supply"):
            return [self._motherboard_psu_connectors(left, right)]
        if pair == ("power_supply", "motherboard"):
            return [self._motherboard_psu_connectors(right, left)]
        if pair == ("storage", "motherboard"):
            return self._storage_motherboard(left, right)
        if pair == ("motherboard", "storage"):
            return self._storage_motherboard(right, left)
        return []

    def _cpu_motherboard(self, cpu: Component, motherboard: Component) -> list[CompatibilityResult]:
        cpu_socket_raw = lookup(cpu, "socket")
        motherboard_socket_raw = lookup(motherboard, "socket", "cpu_socket")
        socket_evidence = self._pair_evidence(
            "cpu",
            cpu,
            "motherboard",
            motherboard,
            cpu_socket=cpu_socket_raw if not is_missing(cpu_socket_raw) else None,
            motherboard_socket=(
                motherboard_socket_raw if not is_missing(motherboard_socket_raw) else None
            ),
        )
        missing_socket = self._missing_fields(
            cpu_socket=cpu_socket_raw, motherboard_socket=motherboard_socket_raw
        )
        if missing_socket:
            socket_result = self._unknown(
                "compat.cpu_motherboard.socket", missing_socket, socket_evidence
            )
        elif token(cpu_socket_raw) == token(motherboard_socket_raw):
            socket_result = self._result(
                "compat.cpu_motherboard.socket",
                CompatibilityStatus.PASS,
                "CPU and motherboard sockets match.",
                socket_evidence,
            )
        else:
            socket_result = self._result(
                "compat.cpu_motherboard.socket",
                CompatibilityStatus.FAIL,
                "CPU and motherboard sockets do not match.",
                socket_evidence,
            )

        platform_result = self._cpu_platform_bios(cpu, motherboard)
        return [socket_result, platform_result]

    def _cpu_platform_bios(self, cpu: Component, motherboard: Component) -> CompatibilityResult:
        generation_raw = lookup(cpu, "generation", "cpu_generation", "architecture")
        chipset_raw = lookup(motherboard, "chipset")
        cpu_model_raw = lookup(cpu, "model", "canonical_name", "name")
        supported_generations_raw = lookup(
            motherboard,
            "supported_cpu_generations",
            "cpu_generations",
            "supported_generations",
        )
        supported_models_raw = lookup(motherboard, "supported_cpu_models", "supported_processors")
        supported_chipsets_raw = lookup(cpu, "supported_chipsets", "compatible_chipsets")
        support_matrix_raw = lookup(
            motherboard, "cpu_support_matrix", "cpu_support", "processor_support"
        )
        evidence = self._pair_evidence(
            "cpu",
            cpu,
            "motherboard",
            motherboard,
            cpu_generation=None if is_missing(generation_raw) else generation_raw,
            cpu_model=None if is_missing(cpu_model_raw) else cpu_model_raw,
            motherboard_chipset=None if is_missing(chipset_raw) else chipset_raw,
            supported_cpu_generations=(
                None if is_missing(supported_generations_raw) else supported_generations_raw
            ),
            supported_cpu_models=None if is_missing(supported_models_raw) else supported_models_raw,
            cpu_supported_chipsets=(
                None if is_missing(supported_chipsets_raw) else supported_chipsets_raw
            ),
            cpu_support_matrix=None if is_missing(support_matrix_raw) else support_matrix_raw,
        )
        missing = self._missing_fields(
            cpu_generation=generation_raw, motherboard_chipset=chipset_raw
        )
        generation = token(generation_raw)
        chipset = token(chipset_raw)
        cpu_model = token(cpu_model_raw)
        supported_generations = tokens(supported_generations_raw)
        supported_models = tokens(supported_models_raw)
        supported_chipsets = tokens(supported_chipsets_raw)
        supported_generations = supported_generations or None
        supported_models = supported_models or None
        supported_chipsets = supported_chipsets or None
        support_entry: Any = MISSING
        if isinstance(support_matrix_raw, Mapping) and support_matrix_raw:
            normalised_matrix = {token(key): value for key, value in support_matrix_raw.items()}
            support_entry = normalised_matrix.get(cpu_model, MISSING)
            if is_missing(support_entry):
                support_entry = normalised_matrix.get(generation, MISSING)

        has_support_matrix = (
            any(
                value is not None
                for value in (supported_generations, supported_models, supported_chipsets)
            )
            or isinstance(support_matrix_raw, Mapping)
            and bool(support_matrix_raw)
        )
        if not has_support_matrix:
            missing.append("cpu_support_matrix")
        if missing:
            return self._unknown("compat.cpu_motherboard.chipset_bios", missing, evidence)

        supported = False
        contradicted_by: list[str] = []
        if supported_generations is not None:
            if generation in supported_generations:
                supported = True
            else:
                contradicted_by.append("supported_cpu_generations")
        if supported_models is not None and cpu_model is not None:
            if cpu_model in supported_models:
                supported = True
            else:
                contradicted_by.append("supported_cpu_models")
        if supported_chipsets is not None:
            if chipset in supported_chipsets:
                supported = True
            else:
                contradicted_by.append("cpu_supported_chipsets")
        if isinstance(support_matrix_raw, Mapping) and support_matrix_raw:
            if is_missing(support_entry):
                contradicted_by.append("cpu_support_matrix")
            elif support_entry is False or (
                isinstance(support_entry, Mapping) and support_entry.get("supported") is False
            ):
                contradicted_by.append("cpu_support_matrix.supported")
            else:
                supported = True
        evidence["matched_cpu_support_entry"] = None if is_missing(support_entry) else support_entry

        # An authoritative negative in any supplied compatibility matrix is a hard conflict.
        if contradicted_by:
            evidence["conflicting_support_fields"] = contradicted_by
            return self._result(
                "compat.cpu_motherboard.chipset_bios",
                CompatibilityStatus.FAIL,
                "The motherboard chipset or CPU support matrix excludes this CPU.",
                evidence,
            )
        if not supported:
            return self._unknown(
                "compat.cpu_motherboard.chipset_bios", ["matching_cpu_support_entry"], evidence
            )

        minimum_versions_raw = lookup(
            motherboard, "minimum_bios_versions", "bios_support", "cpu_minimum_bios"
        )
        bios_version_raw = lookup(motherboard, "bios_version", "current_bios_version")
        update_required_raw = lookup(
            motherboard, "bios_update_required_for", "bios_update_cpu_generations"
        )
        minimum_version: Any = MISSING
        if isinstance(minimum_versions_raw, Mapping):
            normalised_minimums = {token(key): value for key, value in minimum_versions_raw.items()}
            minimum_version = normalised_minimums.get(cpu_model, MISSING)
            if is_missing(minimum_version):
                minimum_version = normalised_minimums.get(generation, MISSING)
        if isinstance(support_entry, Mapping):
            entry_minimum = support_entry.get(
                "minimum_bios_version", support_entry.get("min_bios_version", MISSING)
            )
            if not is_missing(entry_minimum):
                minimum_version = entry_minimum
        elif isinstance(support_entry, str):
            minimum_version = support_entry
        update_available_raw = lookup(
            motherboard, "bios_update_available", "cpu_bios_update_available"
        )
        support_status: str | None = None
        if isinstance(support_entry, Mapping):
            if "bios_update_available" in support_entry:
                update_available_raw = support_entry["bios_update_available"]
            support_status = token(support_entry.get("support_status"))
        evidence["minimum_bios_version"] = None if is_missing(minimum_version) else minimum_version
        evidence["installed_bios_version"] = (
            None if is_missing(bios_version_raw) else bios_version_raw
        )
        evidence["bios_update_available"] = (
            None if is_missing(update_available_raw) else bool(update_available_raw)
        )
        evidence["cpu_support_status"] = support_status

        if not is_missing(minimum_version):
            if is_missing(bios_version_raw):
                return self._unknown(
                    "compat.cpu_motherboard.chipset_bios",
                    ["installed_bios_version"],
                    evidence,
                )
            installed_key = self._version_key(bios_version_raw)
            minimum_key = self._version_key(minimum_version)
            if not installed_key or not minimum_key:
                return self._result(
                    "compat.cpu_motherboard.chipset_bios",
                    CompatibilityStatus.UNKNOWN,
                    "The installed or minimum BIOS version cannot be compared safely.",
                    {**evidence, "missing_fields": ["comparable_bios_versions"]},
                )
            if installed_key < minimum_key:
                if update_available_raw is False:
                    return self._result(
                        "compat.cpu_motherboard.chipset_bios",
                        CompatibilityStatus.FAIL,
                        "The installed BIOS is too old and the support data reports "
                        "no update path.",
                        evidence,
                    )
                if is_missing(update_available_raw):
                    return self._unknown(
                        "compat.cpu_motherboard.chipset_bios",
                        ["bios_update_availability"],
                        evidence,
                    )
                return self._result(
                    "compat.cpu_motherboard.chipset_bios",
                    CompatibilityStatus.WARNING,
                    "The CPU is supported after a motherboard BIOS update.",
                    evidence,
                )

        if support_status in {"beta", "preview", "experimental"}:
            return self._result(
                "compat.cpu_motherboard.chipset_bios",
                CompatibilityStatus.WARNING,
                "CPU support is published as beta or preview BIOS support.",
                evidence,
            )

        update_required = tokens(update_required_raw)
        if update_required is not None and (
            generation in update_required or cpu_model in update_required
        ):
            return self._result(
                "compat.cpu_motherboard.chipset_bios",
                CompatibilityStatus.WARNING,
                "The support data marks this CPU as requiring a BIOS update.",
                evidence,
            )

        return self._result(
            "compat.cpu_motherboard.chipset_bios",
            CompatibilityStatus.PASS,
            "The motherboard chipset and CPU support data include this CPU.",
            evidence,
        )

    def _memory_motherboard(
        self, memory: Component, motherboard: Component
    ) -> list[CompatibilityResult]:
        memory_type_raw = lookup(memory, "memory_type", "ddr_type", "type")
        motherboard_type_raw = lookup(motherboard, "memory_type", "ddr_type")
        type_evidence = self._pair_evidence(
            "memory",
            memory,
            "motherboard",
            motherboard,
            memory_type=None if is_missing(memory_type_raw) else memory_type_raw,
            motherboard_memory_type=(
                None if is_missing(motherboard_type_raw) else motherboard_type_raw
            ),
        )
        missing_type = self._missing_fields(
            memory_type=memory_type_raw, motherboard_memory_type=motherboard_type_raw
        )
        if missing_type:
            type_result = self._unknown(
                "compat.memory_motherboard.generation", missing_type, type_evidence
            )
        elif token(memory_type_raw) == token(motherboard_type_raw):
            type_result = self._result(
                "compat.memory_motherboard.generation",
                CompatibilityStatus.PASS,
                "Memory generation matches the motherboard.",
                type_evidence,
            )
        else:
            type_result = self._result(
                "compat.memory_motherboard.generation",
                CompatibilityStatus.FAIL,
                "Memory generation does not match the motherboard.",
                type_evidence,
            )

        capacity_raw = lookup(memory, "capacity_gb", "capacity")
        maximum_raw = lookup(motherboard, "maximum_memory_gb", "max_memory_gb", "maximum_memory")
        capacity = number(capacity_raw)
        maximum = number(maximum_raw)
        capacity_evidence = self._pair_evidence(
            "memory",
            memory,
            "motherboard",
            motherboard,
            memory_capacity_gb=capacity,
            motherboard_maximum_memory_gb=maximum,
        )
        missing_capacity = [
            name
            for name, value in {
                "memory_capacity_gb": capacity,
                "motherboard_maximum_memory_gb": maximum,
            }.items()
            if value is None
        ]
        if capacity is None or maximum is None:
            capacity_result = self._unknown(
                "compat.memory_motherboard.capacity", missing_capacity, capacity_evidence
            )
        elif capacity <= 0 or maximum <= 0:
            capacity_result = self._result(
                "compat.memory_motherboard.capacity",
                CompatibilityStatus.UNKNOWN,
                "Memory capacity data is not a positive measurement.",
                {**capacity_evidence, "invalid_fields": ["capacity_gb"]},
            )
        elif capacity <= maximum:
            capacity_result = self._result(
                "compat.memory_motherboard.capacity",
                CompatibilityStatus.PASS,
                "Memory capacity is within the motherboard limit.",
                capacity_evidence,
            )
        else:
            capacity_result = self._result(
                "compat.memory_motherboard.capacity",
                CompatibilityStatus.FAIL,
                "Memory capacity exceeds the motherboard limit.",
                capacity_evidence,
            )

        module_count_raw = lookup(memory, "module_count", "modules", "kit_module_count")
        slot_count_raw = lookup(motherboard, "memory_slots", "dimm_slots")
        module_count = integer(module_count_raw)
        slot_count = integer(slot_count_raw)
        modules_evidence = self._pair_evidence(
            "memory",
            memory,
            "motherboard",
            motherboard,
            memory_module_count=module_count,
            motherboard_memory_slots=slot_count,
        )
        missing_modules = [
            name
            for name, value in {
                "memory_module_count": module_count,
                "motherboard_memory_slots": slot_count,
            }.items()
            if value is None
        ]
        if module_count is None or slot_count is None:
            modules_result = self._unknown(
                "compat.memory_motherboard.modules", missing_modules, modules_evidence
            )
        elif module_count <= 0 or slot_count <= 0:
            modules_result = self._result(
                "compat.memory_motherboard.modules",
                CompatibilityStatus.UNKNOWN,
                "Memory module or slot count is not a positive integer.",
                {**modules_evidence, "invalid_fields": ["module_count_or_memory_slots"]},
            )
        elif module_count <= slot_count:
            modules_result = self._result(
                "compat.memory_motherboard.modules",
                CompatibilityStatus.PASS,
                "The memory kit fits the available DIMM slots.",
                modules_evidence,
            )
        else:
            modules_result = self._result(
                "compat.memory_motherboard.modules",
                CompatibilityStatus.FAIL,
                "The memory kit contains more modules than the motherboard has DIMM slots.",
                modules_evidence,
            )
        return [type_result, capacity_result, modules_result]

    def _gpu_motherboard_pcie(self, gpu: Component, motherboard: Component) -> CompatibilityResult:
        slot_count = integer(
            lookup(motherboard, "pcie_x16_slots", "pcie_slots", "pci_express_slots")
        )
        interface_raw = lookup(gpu, "host_interface", "interface", "pcie_interface")
        interface_name = token(interface_raw) if not is_missing(interface_raw) else "pcie"
        evidence = self._pair_evidence(
            "gpu",
            gpu,
            "motherboard",
            motherboard,
            gpu_host_interface=interface_name,
            motherboard_pcie_slot_count=slot_count,
            required_pcie_slots=1,
        )
        if slot_count is None:
            return self._unknown(
                "compat.gpu_motherboard.pcie_slot", ["motherboard_pcie_slot_count"], evidence
            )
        if interface_name is None or "pcie" not in interface_name:
            return self._result(
                "compat.gpu_motherboard.pcie_slot",
                CompatibilityStatus.UNKNOWN,
                "The GPU host interface is not recognised as PCI Express.",
                {**evidence, "missing_fields": ["recognised_gpu_host_interface"]},
            )
        if slot_count >= 1:
            return self._result(
                "compat.gpu_motherboard.pcie_slot",
                CompatibilityStatus.PASS,
                "The motherboard provides a PCI Express slot for the GPU.",
                evidence,
            )
        return self._result(
            "compat.gpu_motherboard.pcie_slot",
            CompatibilityStatus.FAIL,
            "The motherboard has no PCI Express slot available for the GPU.",
            evidence,
        )

    def _motherboard_case(self, motherboard: Component, case: Component) -> CompatibilityResult:
        motherboard_form_raw = lookup(motherboard, "form_factor", "motherboard_form_factor")
        supported_raw = lookup(
            case, "supported_motherboard_sizes", "motherboard_support", "supported_form_factors"
        )
        motherboard_form = form_factor(motherboard_form_raw)
        supported = form_factors(supported_raw)
        supported = supported or None
        evidence = self._pair_evidence(
            "motherboard",
            motherboard,
            "case",
            case,
            motherboard_form_factor=motherboard_form,
            case_supported_motherboard_sizes=(sorted(supported) if supported is not None else None),
        )
        missing = [
            name
            for name, value in {
                "motherboard_form_factor": motherboard_form,
                "case_supported_motherboard_sizes": supported,
            }.items()
            if value is None
        ]
        if motherboard_form is None or supported is None:
            return self._unknown("compat.motherboard_case.form_factor", missing, evidence)
        if motherboard_form in supported:
            return self._result(
                "compat.motherboard_case.form_factor",
                CompatibilityStatus.PASS,
                "The case supports the motherboard form factor.",
                evidence,
            )
        return self._result(
            "compat.motherboard_case.form_factor",
            CompatibilityStatus.FAIL,
            "The case does not support the motherboard form factor.",
            evidence,
        )

    def _gpu_case(self, gpu: Component, case: Component) -> list[CompatibilityResult]:
        gpu_length = number(lookup(gpu, "length_mm", "gpu_length_mm", "length"))
        maximum_length = number(
            lookup(case, "maximum_gpu_length_mm", "max_gpu_length_mm", "gpu_clearance_mm")
        )
        length_evidence = self._pair_evidence(
            "gpu",
            gpu,
            "case",
            case,
            gpu_length_mm=gpu_length,
            case_maximum_gpu_length_mm=maximum_length,
        )
        missing_length = [
            name
            for name, value in {
                "gpu_length_mm": gpu_length,
                "case_maximum_gpu_length_mm": maximum_length,
            }.items()
            if value is None
        ]
        if gpu_length is None or maximum_length is None:
            length_result = self._unknown("compat.gpu_case.length", missing_length, length_evidence)
        elif gpu_length < 0 or maximum_length < 0:
            length_result = self._result(
                "compat.gpu_case.length",
                CompatibilityStatus.UNKNOWN,
                "GPU length or case clearance is an invalid negative measurement.",
                {**length_evidence, "invalid_fields": ["gpu_length_or_clearance"]},
            )
        elif gpu_length <= maximum_length:
            length_result = self._result(
                "compat.gpu_case.length",
                CompatibilityStatus.PASS,
                "GPU length is within the case clearance.",
                length_evidence,
            )
        else:
            length_result = self._result(
                "compat.gpu_case.length",
                CompatibilityStatus.FAIL,
                "GPU length exceeds the case clearance.",
                length_evidence,
            )

        gpu_slots = number(lookup(gpu, "slot_width", "slots", "gpu_slot_width"))
        case_slots = number(
            lookup(
                case,
                "maximum_gpu_slot_width",
                "max_gpu_slot_width",
                "gpu_slot_clearance",
                "expansion_slots",
            )
        )
        slot_evidence = self._pair_evidence(
            "gpu",
            gpu,
            "case",
            case,
            gpu_slot_width=gpu_slots,
            case_gpu_slot_clearance=case_slots,
        )
        missing_slots = [
            name
            for name, value in {
                "gpu_slot_width": gpu_slots,
                "case_gpu_slot_clearance": case_slots,
            }.items()
            if value is None
        ]
        if gpu_slots is None or case_slots is None:
            slot_result = self._unknown("compat.gpu_case.slot_width", missing_slots, slot_evidence)
        elif gpu_slots <= 0 or case_slots <= 0:
            slot_result = self._result(
                "compat.gpu_case.slot_width",
                CompatibilityStatus.UNKNOWN,
                "GPU slot width or case slot clearance is not positive.",
                {**slot_evidence, "invalid_fields": ["gpu_slot_width_or_clearance"]},
            )
        elif gpu_slots <= case_slots:
            slot_result = self._result(
                "compat.gpu_case.slot_width",
                CompatibilityStatus.PASS,
                "GPU slot width is within the case clearance.",
                slot_evidence,
            )
        else:
            slot_result = self._result(
                "compat.gpu_case.slot_width",
                CompatibilityStatus.FAIL,
                "GPU slot width exceeds the case clearance.",
                slot_evidence,
            )
        return [length_result, slot_result]

    def _cooler_case(self, cooler: Component, case: Component) -> CompatibilityResult:
        cooler_type_raw = lookup(cooler, "cooler_type", "type")
        cooler_type = token(cooler_type_raw)
        evidence = self._pair_evidence(
            "cooler",
            cooler,
            "case",
            case,
            cooler_type=None if is_missing(cooler_type_raw) else cooler_type_raw,
        )
        if cooler_type is None:
            return self._unknown("compat.cooler_case.clearance", ["cooler_type"], evidence)

        if cooler_type in {"air", "aircooler", "tower", "lowprofile", "passive"}:
            height = number(lookup(cooler, "height_mm", "cooler_height_mm", "height"))
            maximum = number(
                lookup(
                    case, "maximum_cooler_height_mm", "max_cooler_height_mm", "cooler_clearance_mm"
                )
            )
            evidence.update(cooler_height_mm=height, case_maximum_cooler_height_mm=maximum)
            missing = [
                name
                for name, value in {
                    "cooler_height_mm": height,
                    "case_maximum_cooler_height_mm": maximum,
                }.items()
                if value is None
            ]
            if height is None or maximum is None:
                return self._unknown("compat.cooler_case.clearance", missing, evidence)
            if height < 0 or maximum < 0:
                return self._result(
                    "compat.cooler_case.clearance",
                    CompatibilityStatus.UNKNOWN,
                    "Cooler height or case clearance is an invalid negative measurement.",
                    {**evidence, "invalid_fields": ["cooler_height_or_clearance"]},
                )
            if height <= maximum:
                return self._result(
                    "compat.cooler_case.clearance",
                    CompatibilityStatus.PASS,
                    "Air-cooler height is within the case clearance.",
                    evidence,
                )
            return self._result(
                "compat.cooler_case.clearance",
                CompatibilityStatus.FAIL,
                "Air-cooler height exceeds the case clearance.",
                evidence,
            )

        if cooler_type in {"aio", "liquid", "liquidcooler", "water", "watercooler"}:
            radiator = integer(lookup(cooler, "radiator_size_mm", "radiator_size"))
            supported_raw = lookup(
                case,
                "supported_radiator_sizes_mm",
                "radiator_support_mm",
                "radiator_support",
                "supported_radiator_sizes",
            )
            maximum = number(lookup(case, "maximum_radiator_size_mm", "max_radiator_size_mm"))
            supported = self._measurement_set(supported_raw)
            supported = supported or None
            support_by_position = self._radiator_support_by_position(supported_raw)
            required_position = token(
                lookup(cooler, "required_radiator_position", "radiator_position")
            )
            compatible_positions = (
                sorted(
                    position
                    for position, sizes in support_by_position.items()
                    if radiator is not None and radiator in sizes
                )
                if support_by_position is not None
                else None
            )
            radiator_thickness = number(
                lookup(cooler, "radiator_thickness_mm", "total_radiator_thickness_mm")
            )
            maximum_thickness_raw = lookup(
                case,
                "maximum_radiator_thickness_mm",
                "max_radiator_thickness_mm",
                default=MISSING,
            )
            evidence.update(
                radiator_size_mm=radiator,
                case_supported_radiator_sizes_mm=(
                    sorted(supported) if supported is not None else None
                ),
                case_maximum_radiator_size_mm=maximum,
                required_radiator_position=required_position,
                compatible_radiator_positions=compatible_positions,
                radiator_thickness_mm=radiator_thickness,
                case_maximum_radiator_thickness_mm=(
                    None if is_missing(maximum_thickness_raw) else maximum_thickness_raw
                ),
            )
            if radiator is None:
                return self._unknown("compat.cooler_case.clearance", ["radiator_size_mm"], evidence)
            if supported is None and maximum is None:
                return self._unknown(
                    "compat.cooler_case.clearance", ["case_radiator_support"], evidence
                )
            if required_position is not None:
                if support_by_position is None:
                    return self._unknown(
                        "compat.cooler_case.clearance",
                        ["case_radiator_support_by_position"],
                        evidence,
                    )
                if required_position not in support_by_position:
                    return self._result(
                        "compat.cooler_case.clearance",
                        CompatibilityStatus.FAIL,
                        "The case does not provide the required radiator mounting position.",
                        evidence,
                    )
                if radiator not in support_by_position[required_position]:
                    return self._result(
                        "compat.cooler_case.clearance",
                        CompatibilityStatus.FAIL,
                        "The required case position does not support this radiator size.",
                        evidence,
                    )
            if radiator_thickness is not None or not is_missing(maximum_thickness_raw):
                if radiator_thickness is None or is_missing(maximum_thickness_raw):
                    return self._unknown(
                        "compat.cooler_case.clearance",
                        ["radiator_and_case_thickness_measurements"],
                        evidence,
                    )
                maximum_thickness = self._radiator_thickness_limit(
                    maximum_thickness_raw,
                    required_position=required_position,
                    compatible_positions=compatible_positions,
                )
                if maximum_thickness is None:
                    return self._unknown(
                        "compat.cooler_case.clearance",
                        ["applicable_case_radiator_thickness_limit"],
                        evidence,
                    )
                evidence["applicable_maximum_radiator_thickness_mm"] = maximum_thickness
                if radiator_thickness > maximum_thickness:
                    return self._result(
                        "compat.cooler_case.clearance",
                        CompatibilityStatus.FAIL,
                        "The radiator assembly is thicker than the case mounting clearance.",
                        evidence,
                    )
            if supported is not None:
                fits = radiator in supported
            else:
                assert maximum is not None
                fits = radiator <= maximum
            if fits:
                return self._result(
                    "compat.cooler_case.clearance",
                    CompatibilityStatus.PASS,
                    "The case explicitly supports the cooler radiator size.",
                    evidence,
                )
            return self._result(
                "compat.cooler_case.clearance",
                CompatibilityStatus.FAIL,
                "The case does not support the cooler radiator size.",
                evidence,
            )

        return self._result(
            "compat.cooler_case.clearance",
            CompatibilityStatus.UNKNOWN,
            "The cooler type is not recognised by the clearance rule.",
            {**evidence, "unsupported_cooler_type": cooler_type},
        )

    def _cooler_cpu(self, cooler: Component, cpu: Component) -> CompatibilityResult:
        cpu_socket_raw = lookup(cpu, "socket")
        supported_raw = lookup(cooler, "supported_sockets", "socket_support", "sockets")
        supported = tokens(supported_raw)
        supported = supported or None
        cpu_socket = token(cpu_socket_raw)
        evidence = self._pair_evidence(
            "cooler",
            cooler,
            "cpu",
            cpu,
            cpu_socket=None if is_missing(cpu_socket_raw) else cpu_socket_raw,
            cooler_supported_sockets=(sorted(supported) if supported is not None else None),
        )
        missing = [
            name
            for name, value in {
                "cpu_socket": cpu_socket,
                "cooler_supported_sockets": supported,
            }.items()
            if value is None
        ]
        if cpu_socket is None or supported is None:
            return self._unknown("compat.cooler_cpu.socket", missing, evidence)
        if cpu_socket in supported:
            return self._result(
                "compat.cooler_cpu.socket",
                CompatibilityStatus.PASS,
                "The cooler includes mounting support for the CPU socket.",
                evidence,
            )
        return self._result(
            "compat.cooler_cpu.socket",
            CompatibilityStatus.FAIL,
            "The cooler does not support the CPU socket.",
            evidence,
        )

    def _power_capacity(
        self, cpu: Component, gpu: Component, power_supply: Component
    ) -> CompatibilityResult:
        cpu_power = number(
            lookup(
                cpu,
                "peak_power_w",
                "peak_power_watts",
                "maximum_turbo_power_w",
                "max_turbo_power_w",
                "package_power_w",
                "ppt_w",
            )
        )
        gpu_power = number(
            lookup(
                gpu,
                "board_power_w",
                "board_power_watts",
                "total_board_power_w",
                "total_board_power_watts",
                "tbp_w",
            )
        )
        psu_wattage = number(lookup(power_supply, "wattage", "wattage_w", "rated_power_w"))
        missing = [
            name
            for name, value in {
                "cpu_peak_power_w": cpu_power,
                "gpu_board_power_w": gpu_power,
                "psu_wattage": psu_wattage,
            }.items()
            if value is None
        ]
        evidence = {
            "components": {
                "cpu": source_evidence(cpu),
                "gpu": source_evidence(gpu),
                "power_supply": source_evidence(power_supply),
            },
            "cpu_peak_power_w": cpu_power,
            "gpu_board_power_w": gpu_power,
            "accessory_allowance_w": self.power_policy.accessory_allowance_w,
            "headroom_ratio": self.power_policy.headroom_ratio,
            "psu_wattage": psu_wattage,
        }
        if cpu_power is None or gpu_power is None or psu_wattage is None:
            return self._unknown("compat.power_supply.capacity", missing, evidence)
        if cpu_power < 0 or gpu_power < 0 or psu_wattage <= 0:
            return self._result(
                "compat.power_supply.capacity",
                CompatibilityStatus.UNKNOWN,
                "Power inputs contain an invalid measurement.",
                {**evidence, "invalid_fields": ["power_measurement"]},
            )
        estimated_peak = cpu_power + gpu_power + self.power_policy.accessory_allowance_w
        required_wattage = math.ceil(estimated_peak * (1.0 + self.power_policy.headroom_ratio))
        evidence.update(estimated_peak_load_w=estimated_peak, required_psu_wattage=required_wattage)
        if psu_wattage >= required_wattage:
            return self._result(
                "compat.power_supply.capacity",
                CompatibilityStatus.PASS,
                "Power-supply capacity meets the estimated peak load and configured headroom.",
                evidence,
            )
        return self._result(
            "compat.power_supply.capacity",
            CompatibilityStatus.FAIL,
            "Power-supply capacity is below the estimated peak load with configured headroom.",
            evidence,
        )

    def _power_transient_capacity(
        self, cpu: Component, gpu: Component, power_supply: Component
    ) -> CompatibilityResult:
        cpu_power = number(
            lookup(
                cpu,
                "peak_power_w",
                "peak_power_watts",
                "maximum_turbo_power_w",
                "max_turbo_power_w",
                "package_power_w",
                "ppt_w",
            )
        )
        gpu_board_power = number(
            lookup(
                gpu,
                "board_power_w",
                "board_power_watts",
                "total_board_power_w",
                "total_board_power_watts",
                "tbp_w",
            )
        )
        explicit_gpu_transient = number(
            lookup(
                gpu,
                "transient_power_w",
                "transient_power_watts",
                "peak_transient_power_w",
            )
        )
        psu_rated = number(lookup(power_supply, "wattage", "wattage_w", "rated_power_w"))
        psu_transient = number(
            lookup(
                power_supply,
                "transient_capacity_w",
                "transient_capacity_watts",
                "peak_capacity_w",
            )
        )
        missing = [
            name
            for name, value in {
                "cpu_peak_power_w": cpu_power,
                "gpu_board_power_w": gpu_board_power,
                "psu_wattage": psu_rated,
            }.items()
            if value is None
        ]
        evidence: dict[str, Any] = {
            "components": {
                "cpu": source_evidence(cpu),
                "gpu": source_evidence(gpu),
                "power_supply": source_evidence(power_supply),
            },
            "cpu_peak_power_w": cpu_power,
            "gpu_board_power_w": gpu_board_power,
            "explicit_gpu_transient_power_w": explicit_gpu_transient,
            "gpu_transient_multiplier": self.power_policy.gpu_transient_multiplier,
            "accessory_allowance_w": self.power_policy.accessory_allowance_w,
            "psu_rated_wattage": psu_rated,
            "psu_explicit_transient_capacity_w": psu_transient,
        }
        if cpu_power is None or gpu_board_power is None or psu_rated is None:
            return self._unknown("compat.power_supply.transient_capacity", missing, evidence)
        if (
            cpu_power < 0
            or gpu_board_power < 0
            or psu_rated <= 0
            or (explicit_gpu_transient is not None and explicit_gpu_transient < 0)
            or (psu_transient is not None and psu_transient <= 0)
        ):
            return self._result(
                "compat.power_supply.transient_capacity",
                CompatibilityStatus.UNKNOWN,
                "Transient-power inputs contain an invalid measurement.",
                {**evidence, "invalid_fields": ["power_measurement"]},
            )

        estimated_gpu_transient = max(
            gpu_board_power,
            explicit_gpu_transient
            if explicit_gpu_transient is not None
            else gpu_board_power * self.power_policy.gpu_transient_multiplier,
        )
        required_transient_capacity = math.ceil(
            cpu_power + estimated_gpu_transient + self.power_policy.accessory_allowance_w
        )
        available_transient_capacity = psu_transient if psu_transient is not None else psu_rated
        evidence.update(
            estimated_gpu_transient_power_w=estimated_gpu_transient,
            required_transient_capacity_w=required_transient_capacity,
            available_transient_capacity_w=available_transient_capacity,
            transient_capacity_basis=(
                "published" if psu_transient is not None else "rated_wattage"
            ),
        )
        if available_transient_capacity >= required_transient_capacity:
            return self._result(
                "compat.power_supply.transient_capacity",
                CompatibilityStatus.PASS,
                "The power supply covers the configured GPU transient-load policy.",
                evidence,
            )
        return self._result(
            "compat.power_supply.transient_capacity",
            CompatibilityStatus.FAIL,
            "The power supply does not cover the configured GPU transient-load policy.",
            evidence,
        )

    def _gpu_psu_connectors(self, gpu: Component, power_supply: Component) -> CompatibilityResult:
        required_raw = lookup(gpu, "required_power_connectors", "power_connectors")
        available_raw = lookup(power_supply, "pcie_connectors", "gpu_power_connectors")
        required, unparsed_required = connector_inventory(required_raw, family="pcie")
        available, unparsed_available = connector_inventory(available_raw, family="pcie")
        evidence = self._pair_evidence(
            "gpu",
            gpu,
            "power_supply",
            power_supply,
            required_gpu_connectors=dict(required) if required is not None else None,
            available_pcie_connectors=dict(available) if available is not None else None,
            unparsed_required_connectors=list(unparsed_required),
            unparsed_available_connectors=list(unparsed_available),
        )
        if unparsed_required or unparsed_available:
            return self._result(
                "compat.power_supply.gpu_connectors",
                CompatibilityStatus.UNKNOWN,
                "One or more GPU power connector entries could not be interpreted safely.",
                {
                    **evidence,
                    "missing_fields": ["recognised_gpu_and_psu_connector_types"],
                },
            )
        if required is None:
            return self._unknown(
                "compat.power_supply.gpu_connectors", ["gpu_required_power_connectors"], evidence
            )
        if not required:
            return self._result(
                "compat.power_supply.gpu_connectors",
                CompatibilityStatus.PASS,
                "The GPU explicitly requires no auxiliary power connector.",
                evidence,
            )
        if available is None:
            return self._unknown(
                "compat.power_supply.gpu_connectors", ["psu_pcie_connectors"], evidence
            )
        shortages = self._connector_shortages(required, available)
        if not shortages:
            return self._result(
                "compat.power_supply.gpu_connectors",
                CompatibilityStatus.PASS,
                "The power supply provides all GPU power connectors.",
                evidence,
            )
        return self._result(
            "compat.power_supply.gpu_connectors",
            CompatibilityStatus.FAIL,
            "The power supply is missing one or more required GPU power connectors.",
            {**evidence, "connector_shortages": shortages},
        )

    def _motherboard_psu_connectors(
        self, motherboard: Component, power_supply: Component
    ) -> CompatibilityResult:
        required_raw = lookup(
            motherboard, "required_eps_connectors", "cpu_power_connectors", "eps_connectors"
        )
        available_raw = lookup(power_supply, "eps_connectors", "cpu_power_connectors")
        required, unparsed_required = connector_inventory(required_raw, family="eps")
        available, unparsed_available = connector_inventory(available_raw, family="eps")
        evidence = self._pair_evidence(
            "motherboard",
            motherboard,
            "power_supply",
            power_supply,
            required_eps_connectors=dict(required) if required is not None else None,
            available_eps_connectors=dict(available) if available is not None else None,
            unparsed_required_connectors=list(unparsed_required),
            unparsed_available_connectors=list(unparsed_available),
        )
        if unparsed_required or unparsed_available:
            return self._result(
                "compat.power_supply.eps_connectors",
                CompatibilityStatus.UNKNOWN,
                "One or more EPS connector entries could not be interpreted safely.",
                {
                    **evidence,
                    "missing_fields": ["recognised_eps_connector_types"],
                },
            )
        missing = []
        if required is None:
            missing.append("motherboard_required_eps_connectors")
        if available is None:
            missing.append("psu_eps_connectors")
        if required is None or available is None:
            return self._unknown("compat.power_supply.eps_connectors", missing, evidence)
        shortages = self._connector_shortages(required, available)
        if not shortages:
            return self._result(
                "compat.power_supply.eps_connectors",
                CompatibilityStatus.PASS,
                "The power supply provides the motherboard EPS connectors.",
                evidence,
            )
        return self._result(
            "compat.power_supply.eps_connectors",
            CompatibilityStatus.FAIL,
            "The power supply is missing one or more required motherboard EPS connectors.",
            {**evidence, "connector_shortages": shortages},
        )

    def _storage_motherboard(
        self, storage: Component, motherboard: Component
    ) -> list[CompatibilityResult]:
        interface_raw = lookup(storage, "interface", "storage_interface")
        interface = storage_interface(interface_raw)
        supported_raw = lookup(
            motherboard,
            "supported_storage_interfaces",
            "storage_interfaces",
            "supported_interfaces",
        )
        supported = storage_interfaces(supported_raw)
        supported = supported or None
        if supported is None:
            supported = self._derived_storage_interfaces(motherboard)
        evidence = self._pair_evidence(
            "storage",
            storage,
            "motherboard",
            motherboard,
            storage_interface=interface,
            motherboard_supported_storage_interfaces=(
                sorted(supported) if supported is not None else None
            ),
        )
        missing = [
            name
            for name, value in {
                "storage_interface": interface,
                "motherboard_supported_storage_interfaces": supported,
            }.items()
            if value is None
        ]
        if interface is None or supported is None:
            interface_result = self._unknown(
                "compat.storage_motherboard.interface", missing, evidence
            )
        elif interface in supported:
            interface_result = self._result(
                "compat.storage_motherboard.interface",
                CompatibilityStatus.PASS,
                "The motherboard supports the storage interface.",
                evidence,
            )
        else:
            interface_result = self._result(
                "compat.storage_motherboard.interface",
                CompatibilityStatus.FAIL,
                "The motherboard does not support the storage interface.",
                evidence,
            )

        required_slots = Counter[str]()
        if interface in {"m2_nvme", "m2_sata"}:
            required_slots["m2"] = 1
        elif interface == "sata":
            required_slots["sata"] = 1
        elif interface == "pcie":
            required_slots["pcie"] = 1
        available_slots = self._storage_slot_counts(motherboard)
        slot_evidence = self._pair_evidence(
            "storage",
            storage,
            "motherboard",
            motherboard,
            required_storage_slots=dict(required_slots),
            available_storage_slots=(
                dict(available_slots) if available_slots is not None else None
            ),
        )
        if interface is None:
            slot_result = self._unknown(
                "compat.storage_motherboard.slots", ["storage_interface"], slot_evidence
            )
        elif not required_slots:
            slot_result = self._result(
                "compat.storage_motherboard.slots",
                CompatibilityStatus.UNKNOWN,
                "The storage interface has no recognised slot-allocation rule.",
                {**slot_evidence, "unsupported_storage_interface": interface},
            )
        elif available_slots is None:
            slot_result = self._unknown(
                "compat.storage_motherboard.slots",
                ["motherboard_storage_slot_counts"],
                slot_evidence,
            )
        else:
            shortages = {
                name: required_count - available_slots.get(name, 0)
                for name, required_count in required_slots.items()
                if available_slots.get(name, 0) < required_count
            }
            if shortages:
                slot_result = self._result(
                    "compat.storage_motherboard.slots",
                    CompatibilityStatus.FAIL,
                    "The motherboard has no available slot for the selected storage device.",
                    {**slot_evidence, "slot_shortages": shortages},
                )
            else:
                slot_result = self._result(
                    "compat.storage_motherboard.slots",
                    CompatibilityStatus.PASS,
                    "The motherboard has an available slot for the selected storage device.",
                    slot_evidence,
                )
        return [interface_result, slot_result]

    def _storage_case(self, storage: Component, case: Component) -> CompatibilityResult:
        form_raw = lookup(storage, "form_factor", "storage_form_factor")
        interface = storage_interface(lookup(storage, "interface", "storage_interface"))
        storage_form = self._storage_form_factor(form_raw)
        if storage_form is None and interface in {"m2_nvme", "m2_sata"}:
            storage_form = "m2"

        bays_raw = lookup(case, "drive_bays_by_form_factor", "drive_bay_counts")
        total_bays = integer(lookup(case, "drive_bays", "drive_bay_count"))
        supported_forms = tokens(
            lookup(case, "supported_drive_form_factors", "drive_bay_form_factors")
        )
        bay_counts: dict[str, int] | None = None
        malformed_bay_entries: list[str] = []
        if isinstance(bays_raw, Mapping):
            bay_counts = {}
            for raw_form, raw_count in bays_raw.items():
                normalised = self._storage_form_factor(raw_form)
                count = integer(raw_count)
                if normalised is not None and count is not None:
                    bay_counts[normalised] = bay_counts.get(normalised, 0) + count
                else:
                    malformed_bay_entries.append(f"{raw_form}={raw_count}")

        evidence = self._pair_evidence(
            "storage",
            storage,
            "case",
            case,
            storage_form_factor=storage_form,
            storage_interface=interface,
            case_drive_bays_by_form_factor=bay_counts,
            case_total_drive_bays=total_bays,
            case_supported_drive_form_factors=(
                sorted(supported_forms) if supported_forms else None
            ),
            malformed_drive_bay_entries=malformed_bay_entries,
        )
        if storage_form is None:
            return self._unknown("compat.storage_case.drive_bay", ["storage_form_factor"], evidence)
        if storage_form == "m2":
            return self._result(
                "compat.storage_case.drive_bay",
                CompatibilityStatus.PASS,
                "The M.2 storage device does not consume a case drive bay.",
                evidence,
            )
        if storage_form not in {"2_5_inch", "3_5_inch"}:
            return self._result(
                "compat.storage_case.drive_bay",
                CompatibilityStatus.UNKNOWN,
                "The storage form factor has no recognised case-bay allocation rule.",
                {**evidence, "missing_fields": ["recognised_storage_form_factor"]},
            )
        if malformed_bay_entries:
            return self._unknown(
                "compat.storage_case.drive_bay",
                ["valid_case_drive_bay_counts"],
                evidence,
            )

        if bay_counts is not None:
            available = bay_counts.get(storage_form, 0)
        elif total_bays is not None and supported_forms is not None:
            available = total_bays if token(storage_form) in supported_forms else 0
        else:
            return self._unknown(
                "compat.storage_case.drive_bay",
                ["case_drive_bays_by_form_factor"],
                evidence,
            )
        evidence["available_compatible_drive_bays"] = available
        if available >= 1:
            return self._result(
                "compat.storage_case.drive_bay",
                CompatibilityStatus.PASS,
                "The case provides a compatible bay for the storage device.",
                evidence,
            )
        return self._result(
            "compat.storage_case.drive_bay",
            CompatibilityStatus.FAIL,
            "The case has no compatible bay for the storage device.",
            evidence,
        )

    def _power_supply_case(self, power_supply: Component, case: Component) -> CompatibilityResult:
        power_supply_form_raw = lookup(power_supply, "form_factor", "psu_form_factor")
        supported_raw = lookup(case, "supported_psu_sizes", "supported_psu_form_factors")
        power_supply_form = token(power_supply_form_raw)
        supported = tokens(supported_raw)
        supported = supported or None
        evidence = self._pair_evidence(
            "power_supply",
            power_supply,
            "case",
            case,
            power_supply_form_factor=power_supply_form,
            case_supported_psu_form_factors=(sorted(supported) if supported else None),
        )
        missing = [
            name
            for name, value in {
                "power_supply_form_factor": power_supply_form,
                "case_supported_psu_form_factors": supported,
            }.items()
            if value is None
        ]
        if power_supply_form is None or supported is None:
            return self._unknown("compat.power_supply_case.form_factor", missing, evidence)
        if power_supply_form in supported:
            return self._result(
                "compat.power_supply_case.form_factor",
                CompatibilityStatus.PASS,
                "The case supports the power-supply form factor.",
                evidence,
            )
        return self._result(
            "compat.power_supply_case.form_factor",
            CompatibilityStatus.FAIL,
            "The case does not support the power-supply form factor.",
            evidence,
        )

    def _published_resource_conflicts(
        self, grouped: Mapping[str, tuple[Component, ...]]
    ) -> CompatibilityResult | None:
        motherboard = self._one(grouped, "motherboard")
        if motherboard is None:
            return None
        conflicts_raw = lookup(
            motherboard, "resource_conflicts", "incompatible_resource_pairs", default=MISSING
        )
        if is_missing(conflicts_raw):
            return None

        active_resources = {
            normalised
            for category, values in grouped.items()
            if values and (normalised := token(category)) is not None
        }
        storage = self._one(grouped, "storage")
        if storage is not None:
            interface = storage_interface(lookup(storage, "interface", "storage_interface"))
            if interface is not None:
                active_resources.add(interface)
                active_resources.add(f"storage{token(interface)}")
        cooler = self._one(grouped, "cooler")
        if cooler is not None and token(lookup(cooler, "cooler_type", "type")) == "aio":
            active_resources.add("aioradiator")
        if self._one(grouped, "gpu") is not None:
            active_resources.add("gpupcie")

        if not isinstance(conflicts_raw, Sequence) or isinstance(conflicts_raw, (str, bytes)):
            return self._unknown(
                "compat.motherboard.resource_conflicts",
                ["valid_resource_conflict_definitions"],
                {
                    "components": {"motherboard": source_evidence(motherboard)},
                    "published_resource_conflicts": conflicts_raw,
                    "active_resources": sorted(active_resources),
                },
            )

        malformed: list[int] = []
        triggered: list[dict[str, Any]] = []
        normalised_rules: list[dict[str, Any]] = []
        for index, raw_rule in enumerate(conflicts_raw):
            if isinstance(raw_rule, Mapping):
                resources_raw = raw_rule.get("resources", raw_rule.get("when_all"))
                message = str(raw_rule.get("message", "Published shared-resource conflict."))
                source = raw_rule.get("evidence_source")
            else:
                resources_raw = raw_rule
                message = "Published shared-resource conflict."
                source = None
            resources = tokens(resources_raw)
            if resources is None or len(resources) < 2:
                malformed.append(index)
                continue
            normalised_rule = {
                "index": index,
                "resources": sorted(resources),
                "message": message,
                "evidence_source": source,
            }
            normalised_rules.append(normalised_rule)
            if resources.issubset(active_resources):
                triggered.append(normalised_rule)

        evidence = {
            "components": {"motherboard": source_evidence(motherboard)},
            "active_resources": sorted(active_resources),
            "published_resource_conflicts": normalised_rules,
            "triggered_conflicts": triggered,
        }
        if malformed:
            return self._result(
                "compat.motherboard.resource_conflicts",
                CompatibilityStatus.UNKNOWN,
                "One or more shared-resource conflict definitions are malformed.",
                {
                    **evidence,
                    "malformed_rule_indexes": malformed,
                    "missing_fields": ["valid_resource_conflict_definitions"],
                },
            )
        if triggered:
            return self._result(
                "compat.motherboard.resource_conflicts",
                CompatibilityStatus.FAIL,
                "The selected components trigger a published motherboard resource conflict.",
                evidence,
            )
        return self._result(
            "compat.motherboard.resource_conflicts",
            CompatibilityStatus.PASS,
            "No published motherboard shared-resource conflict is triggered.",
            evidence,
        )

    def _product_lifecycle_results(
        self,
        grouped: Mapping[str, tuple[Component, ...]],
        existing_components: Sequence[Component],
    ) -> list[CompatibilityResult]:
        existing_ids = {
            identity
            for component in existing_components
            if (identity := product_identity(component)) is not None
        }
        results: list[CompatibilityResult] = []
        evaluated_existing_ids: set[str] = set()
        for category_name, selected in grouped.items():
            for component in selected:
                identity = product_identity(component)
                status_raw = lookup(component, "status", "product_status", default=MISSING)
                retained = identity is not None and identity in existing_ids
                if is_missing(status_raw) and not retained:
                    continue
                rule_id = f"compat.product.lifecycle.{category_name}"
                evidence = {
                    "components": {category_name: source_evidence(component)},
                    "category": category_name,
                    "product_identity": identity,
                    "product_status": None if is_missing(status_raw) else status_raw,
                    "retained_user_owned_part": retained,
                }
                if retained and identity is not None:
                    evaluated_existing_ids.add(identity)
                if is_missing(status_raw):
                    results.append(self._unknown(rule_id, ["product_status"], evidence))
                    continue
                status = token(status_raw)
                if status in {"active", "current", "available"}:
                    results.append(
                        self._result(
                            rule_id,
                            CompatibilityStatus.PASS,
                            "The selected product is active.",
                            evidence,
                        )
                    )
                elif status in {"discontinued", "eol", "endoflife"} and retained:
                    results.append(
                        self._result(
                            rule_id,
                            CompatibilityStatus.WARNING,
                            "The discontinued product is allowed only because it is user-owned "
                            "and retained.",
                            evidence,
                        )
                    )
                elif status in {"discontinued", "eol", "endoflife", "upcoming", "notreleased"}:
                    results.append(
                        self._result(
                            rule_id,
                            CompatibilityStatus.FAIL,
                            "A non-active product cannot be selected as a new component.",
                            evidence,
                        )
                    )
                else:
                    results.append(
                        self._result(
                            rule_id,
                            CompatibilityStatus.UNKNOWN,
                            "The product lifecycle status is not recognised.",
                            {**evidence, "missing_fields": ["recognised_product_status"]},
                        )
                    )

        for existing in existing_components:
            identity = product_identity(existing)
            if identity is None or identity in evaluated_existing_ids:
                continue
            category_raw = lookup(existing, "category", default="existing")
            category_name = _category(category_raw)
            results.append(
                self._unknown(
                    f"compat.product.lifecycle.{category_name}",
                    ["selected_retained_product_status"],
                    {
                        "components": {category_name: source_evidence(existing)},
                        "category": category_name,
                        "product_identity": identity,
                    },
                )
            )
        return results

    def _radiator_drive_bay_constraint(
        self, grouped: Mapping[str, tuple[Component, ...]]
    ) -> CompatibilityResult | None:
        cooler = self._one(grouped, "cooler")
        case = self._one(grouped, "case")
        storage = self._one(grouped, "storage")
        if cooler is None or case is None or storage is None:
            return None
        loss_raw = lookup(
            case,
            "radiator_drive_bay_loss",
            "radiator_drive_bay_loss_by_size",
            default=MISSING,
        )
        if is_missing(loss_raw) or token(lookup(cooler, "cooler_type", "type")) != "aio":
            return None

        radiator_size = integer(lookup(cooler, "radiator_size_mm", "radiator_size"))
        radiator_position = token(lookup(cooler, "required_radiator_position", "radiator_position"))
        storage_form = self._storage_form_factor(
            lookup(storage, "form_factor", "storage_form_factor")
        )
        interface = storage_interface(lookup(storage, "interface", "storage_interface"))
        if storage_form is None and interface in {"m2_nvme", "m2_sata"}:
            storage_form = "m2"
        bays_raw = lookup(case, "drive_bays_by_form_factor", "drive_bay_counts")
        evidence: dict[str, Any] = {
            "components": {
                "cooler": source_evidence(cooler),
                "case": source_evidence(case),
                "storage": source_evidence(storage),
            },
            "radiator_size_mm": radiator_size,
            "radiator_position": radiator_position,
            "storage_form_factor": storage_form,
            "published_radiator_drive_bay_loss": loss_raw,
        }
        if storage_form == "m2":
            return self._result(
                "compat.cooler_case.radiator_drive_bays",
                CompatibilityStatus.PASS,
                "Radiator installation cannot consume a bay needed by the M.2 storage device.",
                evidence,
            )
        if radiator_size is None or storage_form is None:
            return self._unknown(
                "compat.cooler_case.radiator_drive_bays",
                [
                    name
                    for name, value in {
                        "radiator_size_mm": radiator_size,
                        "storage_form_factor": storage_form,
                    }.items()
                    if value is None
                ],
                evidence,
            )
        if not isinstance(loss_raw, Mapping) or not isinstance(bays_raw, Mapping):
            return self._unknown(
                "compat.cooler_case.radiator_drive_bays",
                ["radiator_bay_loss_and_drive_bay_counts"],
                evidence,
            )

        applicable_loss_raw: Any = MISSING
        if radiator_position is not None:
            position_losses = next(
                (
                    value
                    for key, value in loss_raw.items()
                    if token(key) == radiator_position and isinstance(value, Mapping)
                ),
                MISSING,
            )
            if isinstance(position_losses, Mapping):
                applicable_loss_raw = next(
                    (
                        value
                        for key, value in position_losses.items()
                        if integer(key) == radiator_size
                    ),
                    MISSING,
                )
        if is_missing(applicable_loss_raw):
            applicable_loss_raw = next(
                (value for key, value in loss_raw.items() if integer(key) == radiator_size),
                MISSING,
            )
        lost_bays = integer(applicable_loss_raw)
        available_before = next(
            (
                integer(value)
                for key, value in bays_raw.items()
                if self._storage_form_factor(key) == storage_form
            ),
            None,
        )
        evidence.update(
            compatible_bays_before_radiator=available_before,
            compatible_bays_lost_to_radiator=lost_bays,
        )
        if lost_bays is None or available_before is None:
            return self._unknown(
                "compat.cooler_case.radiator_drive_bays",
                ["applicable_radiator_bay_loss"],
                evidence,
            )
        available_after = max(0, available_before - lost_bays)
        evidence["compatible_bays_after_radiator"] = available_after
        if available_after >= 1:
            return self._result(
                "compat.cooler_case.radiator_drive_bays",
                CompatibilityStatus.PASS,
                "A compatible drive bay remains after radiator installation.",
                evidence,
            )
        return self._result(
            "compat.cooler_case.radiator_drive_bays",
            CompatibilityStatus.FAIL,
            "Radiator installation removes every bay compatible with the selected storage device.",
            evidence,
        )

    def _retained_component_results(
        self,
        grouped: Mapping[str, tuple[Component, ...]],
        existing_components: Iterable[Component],
    ) -> list[CompatibilityResult]:
        results: list[CompatibilityResult] = []
        for index, existing in enumerate(existing_components):
            if not isinstance(existing, Mapping):
                raise TypeError("existing_components must contain mappings")
            category_raw = lookup(existing, "category")
            identity = product_identity(existing)
            evidence = {
                "existing_index": index,
                "existing_component": source_evidence(existing),
                "category": None if is_missing(category_raw) else category_raw,
                "product_identity": identity,
            }
            missing = []
            if is_missing(category_raw):
                missing.append("existing_component.category")
            if identity is None:
                missing.append("existing_component.product_id_or_mpn")
            if missing:
                results.append(self._unknown("compat.existing.retained", missing, evidence))
                continue
            category_name = _category(category_raw)
            selected_identities = {
                selected_identity
                for component in grouped.get(category_name, ())
                if (selected_identity := product_identity(component)) is not None
            }
            evidence["selected_product_identities"] = sorted(selected_identities)
            if identity in selected_identities:
                results.append(
                    self._result(
                        "compat.existing.retained",
                        CompatibilityStatus.PASS,
                        "The retained user-owned component remains selected.",
                        evidence,
                    )
                )
            else:
                results.append(
                    self._result(
                        "compat.existing.retained",
                        CompatibilityStatus.FAIL,
                        "A required user-owned component was not retained in the build.",
                        evidence,
                    )
                )
        return results

    def _result(
        self,
        rule_id: str,
        status: CompatibilityStatus,
        message: str,
        evidence: Mapping[str, Any],
    ) -> CompatibilityResult:
        return CompatibilityResult(rule_id, self.rule_version, status, message, evidence)

    def _authority_unknowns(
        self, components: Iterable[tuple[str, Component]]
    ) -> list[CompatibilityResult]:
        """Block specification rules when an authoritative-only record is unverified."""

        results: list[CompatibilityResult] = []
        seen: set[tuple[str, str]] = set()
        for category_name, component in components:
            raw_authority = component.get(COMPATIBILITY_AUTHORITY_KEY)
            if not isinstance(raw_authority, Mapping):
                continue
            if raw_authority.get("policy") != AUTHORITATIVE_COMPATIBILITY_POLICY:
                continue
            if raw_authority.get("decision") == "authoritative":
                continue

            identity = product_identity(component) or "unknown-product"
            key = (_category(category_name), identity)
            if key in seen:
                continue
            seen.add(key)
            reason = str(
                raw_authority.get("reason")
                or "Manufacturer-verified or explicitly authoritative field provenance is absent."
            )
            unverified_fields = raw_authority.get("unverified_fields")
            missing_fields = (
                [str(field) for field in unverified_fields]
                if isinstance(unverified_fields, Sequence)
                and not isinstance(unverified_fields, (str, bytes))
                else ["authoritative_compatibility_provenance"]
            )
            results.append(
                self._result(
                    "compat.evidence.authority",
                    CompatibilityStatus.UNKNOWN,
                    "Compatibility rule evaluation was suppressed because authoritative "
                    f"evidence is unavailable for {category_name}: {reason}",
                    {
                        "category": _category(category_name),
                        "component": source_evidence(component),
                        "authority": dict(raw_authority),
                        "missing_fields": sorted(set(missing_fields)),
                    },
                )
            )
        return results

    def _unknown(
        self, rule_id: str, missing_fields: Sequence[str], evidence: Mapping[str, Any]
    ) -> CompatibilityResult:
        fields = sorted(set(missing_fields))
        return self._result(
            rule_id,
            CompatibilityStatus.UNKNOWN,
            "Compatibility cannot be determined because required data is missing: "
            f"{', '.join(fields)}.",
            {**evidence, "missing_fields": fields},
        )

    @staticmethod
    def _missing_fields(**values: Any) -> list[str]:
        return [name for name, value in values.items() if is_missing(value)]

    @staticmethod
    def _pair_evidence(
        left_name: str,
        left: Component,
        right_name: str,
        right: Component,
        **values: Any,
    ) -> dict[str, Any]:
        return {
            "components": {
                left_name: source_evidence(left),
                right_name: source_evidence(right),
            },
            **values,
        }

    @staticmethod
    def _version_key(value: Any) -> tuple[tuple[int, int | str], ...]:
        parts = re.findall(r"\d+|[a-z]+", str(value).casefold())
        return tuple((0, int(part)) if part.isdigit() else (1, part) for part in parts)

    @staticmethod
    def _measurement_set(value: Any) -> set[int] | None:
        if is_missing(value):
            return None
        if isinstance(value, Mapping):
            raw_values: Iterable[Any] = (
                size
                for raw_size in value.values()
                for size in (
                    raw_size
                    if isinstance(raw_size, Sequence) and not isinstance(raw_size, (str, bytes))
                    else (raw_size,)
                )
            )
        elif isinstance(value, str):
            raw_values = re.split(r"[,;|]", value)
        elif isinstance(value, Iterable):
            raw_values = value
        else:
            raw_values = (value,)
        parsed = {size for item in raw_values if (size := integer(item)) is not None}
        return parsed

    @classmethod
    def _radiator_support_by_position(cls, value: Any) -> dict[str, set[int]] | None:
        if not isinstance(value, Mapping):
            return None
        result: dict[str, set[int]] = {}
        for raw_position, raw_sizes in value.items():
            position = token(raw_position)
            sizes = cls._measurement_set(raw_sizes)
            if position is not None and sizes:
                result[position] = sizes
        return result or None

    @staticmethod
    def _radiator_thickness_limit(
        value: Any,
        *,
        required_position: str | None,
        compatible_positions: Sequence[str] | None,
    ) -> float | None:
        scalar = number(value)
        if scalar is not None:
            return scalar
        if not isinstance(value, Mapping):
            return None
        limits = {
            position: limit
            for raw_position, raw_limit in value.items()
            if (position := token(raw_position)) is not None
            and (limit := number(raw_limit)) is not None
        }
        if required_position is not None:
            return limits.get(required_position)
        eligible = [
            limits[position] for position in compatible_positions or () if position in limits
        ]
        return max(eligible) if eligible else None

    @staticmethod
    def _storage_form_factor(value: Any) -> str | None:
        normalised = token(value)
        if normalised is None:
            return None
        if normalised.startswith("m2") or normalised in {"2230", "2242", "2260", "2280", "22110"}:
            return "m2"
        if normalised in {"25", "25inch", "25in", "2_5_inch"} or "25inch" in normalised:
            return "2_5_inch"
        if normalised in {"35", "35inch", "35in", "3_5_inch"} or "35inch" in normalised:
            return "3_5_inch"
        if normalised in {"addin", "addincard", "aic"}:
            return "add_in_card"
        return normalised

    @staticmethod
    def _connector_shortages(required: Counter[str], available: Counter[str]) -> dict[str, int]:
        remaining = Counter(available)
        shortages: dict[str, int] = {}
        # Allocate exact connectors first. A 6+2-pin lead may cover a 6-pin requirement only
        # after all native 8-pin requirements have reserved their leads.
        ordered_requirements = sorted(required.items(), key=lambda item: item[0] == "pcie_6_pin")
        for connector, required_count in ordered_requirements:
            exact_supplied = min(required_count, remaining.get(connector, 0))
            supplied = exact_supplied
            remaining[connector] -= exact_supplied
            if connector == "pcie_6_pin" and supplied < required_count:
                convertible = remaining.get("pcie_8_pin", 0)
                used = min(required_count - supplied, convertible)
                supplied += used
                remaining["pcie_8_pin"] -= used
            if supplied < required_count:
                shortages[connector] = required_count - supplied
        return shortages

    @staticmethod
    def _derived_storage_interfaces(motherboard: Component) -> set[str] | None:
        interface_set: set[str] = set()
        observed = False
        for aliases, interface_name in (
            (("m2_slots", "m.2_slots"), "m2_nvme"),
            (("sata_ports", "sata_connectors"), "sata"),
            (("pcie_slots", "pci_express_slots"), "pcie"),
        ):
            raw = lookup(motherboard, *aliases)
            count = integer(raw)
            if count is not None:
                observed = True
                if count > 0:
                    interface_set.add(interface_name)
        return interface_set if observed else None

    @staticmethod
    def _storage_slot_counts(motherboard: Component) -> Counter[str] | None:
        explicit_raw = lookup(motherboard, "storage_slot_counts", "available_storage_slots")
        if isinstance(explicit_raw, Mapping):
            result: Counter[str] = Counter()
            for raw_name, raw_count in explicit_raw.items():
                count = integer(raw_count)
                normalised = storage_interface(raw_name)
                if count is None or normalised is None:
                    continue
                slot_name = "m2" if normalised in {"m2_nvme", "m2_sata"} else normalised
                result[slot_name] += count
            return result

        result = Counter()
        observed = False
        for aliases, slot_name in (
            (("m2_slots", "m.2_slots"), "m2"),
            (("sata_ports", "sata_connectors"), "sata"),
            (("pcie_slots", "pci_express_slots"), "pcie"),
        ):
            raw = lookup(motherboard, *aliases)
            count = integer(raw)
            if count is not None:
                observed = True
                result[slot_name] = count
        return result if observed else None


def check_build_compatibility(
    components: Mapping[str, Component | Sequence[Component]],
    *,
    existing_components: Iterable[Component] = (),
    rule_version: str = DEFAULT_RULE_VERSION,
    power_policy: PowerPolicy | None = None,
) -> CompatibilityReport:
    """Convenience entry point for stateless complete-build validation."""

    return CompatibilityEngine(rule_version=rule_version, power_policy=power_policy).check_build(
        components, existing_components=existing_components
    )
