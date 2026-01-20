"""Typed component attribute schemas.

Compatibility-critical values are optional because real catalog data can be incomplete.
The compatibility engine must interpret absent values as UNKNOWN, never as PASS.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .enums import (
    CaseSize,
    CoolerType,
    EfficiencyRating,
    MemoryType,
    ModularType,
    MotherboardFormFactor,
    PowerSupplyFormFactor,
    StorageFormFactor,
    StorageInterface,
)

PositiveFloat = Annotated[float, Field(gt=0)]
NonNegativeFloat = Annotated[float, Field(ge=0)]
PositiveInt = Annotated[int, Field(gt=0)]
NonNegativeInt = Annotated[int, Field(ge=0)]


class DomainModel(BaseModel):
    """Strict base used for durable API and persistence contracts."""

    # ``model_version`` is a deliberate durable-contract field.  Pydantic 2.9
    # protects every ``model_`` prefix by default, which emits a warning for
    # that harmless field.  Keep the actual validation/serialization roots
    # protected instead, matching the narrower modern Pydantic default.
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
        protected_namespaces=("model_validate", "model_dump"),
    )


class CommonProductAttributes(DomainModel):
    warranty_years: NonNegativeFloat | None = None
    width_mm: PositiveFloat | None = None
    height_mm: PositiveFloat | None = None
    depth_mm: PositiveFloat | None = None
    colour: str | None = None
    msrp_sgd: Annotated[Decimal, Field(ge=0)] | None = None
    tags: list[str] = Field(default_factory=list)


class CPUAttributes(DomainModel):
    socket: str | None = None
    architecture: str | None = None
    generation: str | None = None
    core_count: PositiveInt | None = None
    thread_count: PositiveInt | None = None
    base_clock_ghz: PositiveFloat | None = None
    boost_clock_ghz: PositiveFloat | None = None
    tdp_watts: PositiveFloat | None = None
    peak_power_watts: PositiveFloat | None = None
    maximum_memory_gb: PositiveInt | None = None
    integrated_graphics: str | None = None
    included_cooler: bool | None = None

    @model_validator(mode="after")
    def threads_cover_cores(self) -> CPUAttributes:
        if (
            self.core_count is not None
            and self.thread_count is not None
            and self.thread_count < self.core_count
        ):
            raise ValueError("thread_count cannot be lower than core_count")
        return self


class GPUAttributes(DomainModel):
    architecture: str | None = None
    vram_gb: PositiveInt | None = None
    vram_type: str | None = None
    memory_bus_bits: PositiveInt | None = None
    memory_bandwidth_gbps: PositiveFloat | None = None
    compute_unit_count: PositiveInt | None = None
    base_clock_mhz: PositiveFloat | None = None
    boost_clock_mhz: PositiveFloat | None = None
    length_mm: PositiveFloat | None = None
    height_mm: PositiveFloat | None = None
    slot_width: PositiveFloat | None = None
    board_power_watts: PositiveFloat | None = None
    recommended_psu_watts: PositiveInt | None = None
    power_connectors: dict[str, NonNegativeInt] = Field(default_factory=dict)


class MotherboardAttributes(DomainModel):
    socket: str | None = None
    chipset: str | None = None
    supported_cpu_generations: list[str] = Field(default_factory=list)
    form_factor: MotherboardFormFactor | None = None
    memory_type: MemoryType | None = None
    maximum_memory_gb: PositiveInt | None = None
    memory_slots: PositiveInt | None = None
    pcie_slots: NonNegativeInt | None = None
    m2_slots: NonNegativeInt | None = None
    sata_ports: NonNegativeInt | None = None
    wifi_support: bool | None = None
    bios_version: str | None = None


class MemoryAttributes(DomainModel):
    memory_type: MemoryType | None = None
    capacity_gb: PositiveInt | None = None
    module_count: PositiveInt | None = None
    speed_mt_s: PositiveInt | None = None
    cas_latency: PositiveFloat | None = None
    voltage: PositiveFloat | None = None
    module_height_mm: PositiveFloat | None = None


class StorageAttributes(DomainModel):
    capacity_gb: PositiveInt | None = None
    interface: StorageInterface | None = None
    form_factor: StorageFormFactor | None = None
    sequential_read_mbps: PositiveFloat | None = None
    sequential_write_mbps: PositiveFloat | None = None
    random_read_iops: PositiveFloat | None = None
    random_write_iops: PositiveFloat | None = None
    endurance_tbw: PositiveFloat | None = None


class PowerSupplyAttributes(DomainModel):
    wattage: PositiveInt | None = None
    efficiency_rating: EfficiencyRating | None = None
    form_factor: PowerSupplyFormFactor | None = None
    modular_type: ModularType | None = None
    pcie_connectors: dict[str, NonNegativeInt] = Field(default_factory=dict)
    eps_connectors: NonNegativeInt | None = None
    atx_version: str | None = None
    warranty_years: NonNegativeFloat | None = None


class CoolerAttributes(DomainModel):
    cooler_type: CoolerType | None = None
    supported_sockets: list[str] = Field(default_factory=list)
    height_mm: PositiveFloat | None = None
    radiator_size_mm: PositiveInt | None = None
    fan_count: PositiveInt | None = None
    estimated_cooling_capacity_watts: PositiveFloat | None = None

    @field_validator("supported_sockets")
    @classmethod
    def sockets_are_unique(cls, sockets: list[str]) -> list[str]:
        normalised = [socket.strip() for socket in sockets]
        if len({socket.casefold() for socket in normalised}) != len(normalised):
            raise ValueError("supported_sockets must not contain duplicates")
        return normalised


class CaseAttributes(DomainModel):
    case_size: CaseSize | None = None
    supported_motherboard_sizes: list[MotherboardFormFactor] = Field(default_factory=list)
    maximum_gpu_length_mm: PositiveFloat | None = None
    maximum_gpu_slot_width: PositiveFloat | None = None
    maximum_cooler_height_mm: PositiveFloat | None = None
    supported_psu_sizes: list[PowerSupplyFormFactor] = Field(default_factory=list)
    radiator_support_mm: list[PositiveInt] = Field(default_factory=list)
    drive_bays: NonNegativeInt | None = None
    included_fans: NonNegativeInt | None = None

    @field_validator(
        "supported_motherboard_sizes", "supported_psu_sizes", "radiator_support_mm"
    )
    @classmethod
    def sequence_values_are_unique(cls, values: list[object]) -> list[object]:
        if len(set(values)) != len(values):
            raise ValueError("supported values must not contain duplicates")
        return values


ComponentAttributes = (
    CPUAttributes
    | GPUAttributes
    | MotherboardAttributes
    | MemoryAttributes
    | StorageAttributes
    | PowerSupplyAttributes
    | CoolerAttributes
    | CaseAttributes
)
