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


class WorkloadPerformanceSignal(DomainModel):
    """Request-scoped measured or model-derived component performance evidence."""

    workload: WorkloadLabel
    metric: str = Field(min_length=1)
    unit: str | None = None
    score: float | None = Field(default=None, ge=0)
    relative_score: float = Field(ge=0)
    basis: Literal["observed", "predicted", "relative"]
    confidence: Literal["observed", "high", "medium", "low"]
    decision: Literal[
        "observed_benchmark",
        "precise_model_prediction",
        "model_not_promotion_eligible",
        "input_outside_training_contract",
        "model_not_promotion_eligible_and_input_outside_training_contract",
        "precise_predictions_disabled",
        "precise_predictions_disabled_and_input_outside_training_contract",
    ]
    model_version: str | None = None
    supporting_sources: list[str] = Field(default_factory=list)
    supporting_benchmark_ids: list[str] = Field(default_factory=list)
    lower_score: float | None = Field(default=None, ge=0)
    upper_score: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def basis_has_truthful_provenance(self) -> WorkloadPerformanceSignal:
        numeric_values = [self.relative_score]
        numeric_values.extend(
            value for value in (self.score, self.lower_score, self.upper_score) if value is not None
        )
        if not all(math.isfinite(value) for value in numeric_values):
            raise ValueError("performance signal values must be finite")
        if self.basis == "observed":
            if self.score is None:
                raise ValueError("observed performance requires a score")
            if self.model_version is not None:
                raise ValueError("observed performance cannot claim a model version")
            if not self.supporting_sources or not self.supporting_benchmark_ids:
                raise ValueError("observed performance requires benchmark provenance")
        elif self.model_version is None:
            raise ValueError("model-derived performance requires model_version")
        if self.basis == "observed" and self.confidence != "observed":
            raise ValueError("observed performance requires observed confidence")
        if self.basis != "observed" and self.confidence == "observed":
            raise ValueError("model-derived performance cannot claim observed confidence")
        if self.basis == "relative" and self.score is not None:
            raise ValueError("relative performance cannot expose a precise score")
        if self.basis == "predicted" and self.score is None:
            raise ValueError("predicted performance requires a precise score")
        if (self.lower_score is None) != (self.upper_score is None):
            raise ValueError("prediction interval bounds must be supplied together")
        if self.lower_score is not None and self.upper_score is not None:
            if self.basis != "predicted" or self.score is None:
                raise ValueError("only predicted performance may expose an interval")
            if not self.lower_score <= self.score <= self.upper_score:
                raise ValueError("prediction interval must contain the score")
        if self.basis == "predicted" and self.lower_score is None:
            raise ValueError("predicted performance requires a calibrated interval")
        if self.basis != "predicted" and self.lower_score is not None:
            raise ValueError("only predicted performance may expose an interval")
        if self.basis == "observed" and self.decision != "observed_benchmark":
            raise ValueError("observed performance requires the observed_benchmark decision")
        if self.basis == "predicted" and self.decision != "precise_model_prediction":
            raise ValueError("predicted performance requires the precise_model_prediction decision")
        if self.basis == "relative" and self.decision not in {
            "model_not_promotion_eligible",
            "input_outside_training_contract",
            "model_not_promotion_eligible_and_input_outside_training_contract",
            "precise_predictions_disabled",
            "precise_predictions_disabled_and_input_outside_training_contract",
        }:
            raise ValueError("relative performance requires a bounded fallback decision")
        return self


class CompatibilityRule(DomainModel):
    rule_id: str = Field(default_factory=lambda: new_id("rule"), min_length=1)
    rule_version: str = Field(min_length=1)
    left_category: ComponentKind
    right_category: ComponentKind
    rule_type: str = Field(min_length=1)
    severity: CompatVerdict
    required_fields: list[str] = Field(default_factory=list)
    message_template: str = Field(min_length=1)
    evidence_source: str = Field(min_length=1)
    effective_from: datetime


class ReviewNote(DomainModel):
    """One short, permitted and citable statement about a canonical product.

    The bounded fields prevent the catalogue from becoming an unbounded
    review-text store. Source-specific rights and release binding are checked
    by the processed-catalog loader before an instance reaches public serving.
    """

    evidence_id: str = Field(
        default_factory=lambda: new_id("evidence"), min_length=1, max_length=80
    )
    product_id: str = Field(min_length=1, max_length=80)
    aspect: str = Field(min_length=1, max_length=80)
    sentiment: float = Field(ge=-1, le=1)
    evidence_text: str = Field(min_length=1, max_length=500)
    source_url: str = Field(min_length=1, max_length=2048)
    published_at: datetime | None = None
    confidence: Probability


class SearchQuery(DomainModel):
    query_id: str = Field(default_factory=lambda: new_id("query"), min_length=1)
    raw_query: str | None = None
    structured_constraints: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class WorkloadPreference(DomainModel):
    name: WorkloadLabel
    weight: float = Field(gt=0, le=1)


class ExistingComponent(DomainModel):
    category: ComponentKind
    product_id: str = Field(min_length=1)
    listing_id: str | None = None
    purchase_price_sgd: Money | None = None


class BuildRequirements(DomainModel):
    minimum_gpu_vram_gb: int | None = Field(default=None, ge=1)
    minimum_memory_gb: int | None = Field(default=None, ge=1)
    storage_gb: int | None = Field(default=None, ge=1)
    required_memory_type: MemoryType | None = None
    required_motherboard_form_factor: MotherboardFormFactor | None = None
    wifi_required: bool = False
    case_size: CaseSize | None = None
    in_stock_only: bool = True


class BuildPreferences(DomainModel):
    noise: str | None = None
    upgradeability: str | None = None
    power_efficiency: str | None = None
    preferred_brands: list[str] = Field(default_factory=list)
    excluded_brands: list[str] = Field(default_factory=list)

    @field_validator("preferred_brands", "excluded_brands")
    @classmethod
    def brands_are_unique(cls, brands: list[str]) -> list[str]:
        if len({brand.casefold() for brand in brands}) != len(brands):
            raise ValueError("brand lists must not contain duplicates")
        return brands

    @model_validator(mode="after")
    def preferred_and_excluded_do_not_overlap(self) -> BuildPreferences:
        preferred = {brand.casefold() for brand in self.preferred_brands}
        excluded = {brand.casefold() for brand in self.excluded_brands}
        overlap = preferred & excluded
        if overlap:
            raise ValueError(f"brands cannot be both preferred and excluded: {sorted(overlap)}")
        return self

# TODO: rest of this module still to come.
