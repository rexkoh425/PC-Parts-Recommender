"""Application-facing domain models for catalog, requests, and recommendations."""

from __future__ import annotations

import math
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Annotated, Any, Literal
from uuid import uuid4

from pydantic import Field, field_validator, model_validator

from .components import (
    CaseAttributes,
    CommonProductAttributes,
    ComponentAttributes,
    CoolerAttributes,
    CPUAttributes,
    DomainModel,
    GPUAttributes,
    MemoryAttributes,
    MotherboardAttributes,
    PowerSupplyAttributes,
    StorageAttributes,
)
from .enums import (
    BenchmarkValueKind,
    BuildPreset,
    CaseSize,
    CompatVerdict,
    ComponentKind,
    InteractionType,
    ListingCondition,
    MemoryType,
    MotherboardFormFactor,
    ProductStatus,
    SourceType,
    StockState,
    WorkloadLabel,
)

Money = Annotated[Decimal, Field(ge=0, decimal_places=2)]
Score = Annotated[float, Field(ge=0, le=100)]
Probability = Annotated[float, Field(ge=0, le=1)]


def new_id(prefix: str) -> str:
    """Return a sortable-by-prefix opaque identifier suitable for external APIs."""

    return f"{prefix}_{uuid4().hex}"


def utc_now() -> datetime:
    return datetime.now(UTC)


_ATTRIBUTE_TYPES: dict[ComponentKind, type[ComponentAttributes]] = {
    ComponentKind.CPU: CPUAttributes,
    ComponentKind.GPU: GPUAttributes,
    ComponentKind.MOTHERBOARD: MotherboardAttributes,
    ComponentKind.MEMORY: MemoryAttributes,
    ComponentKind.STORAGE: StorageAttributes,
    ComponentKind.POWER_SUPPLY: PowerSupplyAttributes,
    ComponentKind.COOLER: CoolerAttributes,
    ComponentKind.CASE: CaseAttributes,
}


class SourceProvenance(DomainModel):
    provenance_id: str = Field(default_factory=lambda: new_id("src"), min_length=1)
    product_id: str | None = None
    listing_id: str | None = None
    source_name: str = Field(min_length=1)
    source_url: str = Field(min_length=1)
    source_type: SourceType
    retrieved_at: datetime = Field(default_factory=utc_now)
    raw_content_hash: str = Field(min_length=1)
    parser_version: str = Field(min_length=1)
    licence_or_access_note: str = Field(min_length=1)
    last_verified_at: datetime | None = None
    extraction_confidence: Probability = 1.0


class MasterProduct(DomainModel):
    product_id: str = Field(default_factory=lambda: new_id("prod"), min_length=1)
    category: ComponentKind
    brand: str = Field(min_length=1)
    model: str = Field(min_length=1)
    manufacturer_part_number: str | None = None
    gtin: str | None = None
    canonical_name: str = Field(min_length=1)
    release_date: date | None = None
    status: ProductStatus = ProductStatus.ACTIVE
    common_attributes: CommonProductAttributes = Field(default_factory=CommonProductAttributes)
    category_attributes: ComponentAttributes
    source_confidence: Probability = 1.0
    provenance: list[SourceProvenance] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="before")
    @classmethod
    def parse_category_attributes(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        category_value = value.get("category")
        attributes = value.get("category_attributes")
        if category_value is None or not isinstance(attributes, dict):
            return value
        category = ComponentKind(category_value)
        parsed = dict(value)
        parsed["category_attributes"] = _ATTRIBUTE_TYPES[category].model_validate(attributes)
        return parsed

    @model_validator(mode="after")
    def attributes_match_category(self) -> MasterProduct:
        expected_type = _ATTRIBUTE_TYPES[self.category]
        if not isinstance(self.category_attributes, expected_type):
            raise ValueError(f"{self.category.value} products require {expected_type.__name__}")
        return self


class RetailerOffering(DomainModel):
    listing_id: str = Field(default_factory=lambda: new_id("listing"), min_length=1)
    product_id: str = Field(min_length=1)
    retailer: str = Field(min_length=1)
    source_listing_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    condition: ListingCondition = ListingCondition.NEW
    currency: str = Field(default="SGD", pattern=r"^[A-Z]{3}$")
    base_price: Money
    shipping_price: Money = Decimal("0")
    stock_status: StockState = StockState.UNKNOWN
    seller_name: str | None = None
    listing_url: str = Field(min_length=1)
    first_seen_at: datetime = Field(default_factory=utc_now)
    last_seen_at: datetime = Field(default_factory=utc_now)

    @property
    def total_price(self) -> Decimal:
        return self.base_price + self.shipping_price

    @model_validator(mode="after")
    def seen_times_are_ordered(self) -> RetailerOffering:
        if self.last_seen_at < self.first_seen_at:
            raise ValueError("last_seen_at cannot be earlier than first_seen_at")
        return self


class PriceSample(DomainModel):
    snapshot_id: str = Field(default_factory=lambda: new_id("price"), min_length=1)
    listing_id: str = Field(min_length=1)
    observed_at: datetime = Field(default_factory=utc_now)
    base_price: Money
    shipping_price: Money = Decimal("0")
    stock_status: StockState
    promotion_text: str | None = None

    @property
    def total_price(self) -> Decimal:
        return self.base_price + self.shipping_price


class BenchmarkResult(DomainModel):
    benchmark_id: str = Field(default_factory=lambda: new_id("bench"), min_length=1)
    product_id: str = Field(min_length=1)
    workload: WorkloadLabel
    benchmark_name: str = Field(min_length=1)
    benchmark_version: str = Field(min_length=1)
    score: float
    unit: str = Field(min_length=1)
    higher_is_better: bool = True
    resolution: str | None = None
    preset: str | None = None
    operating_system: str | None = None
    driver_version: str | None = None
    source_url: str = Field(min_length=1)
    observed_at: datetime


class PerformanceEstimate(DomainModel):
    workload: WorkloadLabel
    score: float
    value_kind: BenchmarkValueKind
    model_version: str | None = None
    confidence: Probability | None = None
    supporting_benchmark_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def predicted_values_name_the_model(self) -> PerformanceEstimate:
        if self.value_kind == BenchmarkValueKind.PREDICTED and not self.model_version:
            raise ValueError("predicted performance values require model_version")
        return self

# TODO: rest of this module still to come.
