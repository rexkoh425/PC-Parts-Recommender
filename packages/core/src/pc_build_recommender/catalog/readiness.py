"""Measured catalog coverage and fail-closed production readiness policy."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date
from types import MappingProxyType
from typing import Any

from pc_build_recommender.data_rights import (
    DataUse,
    production_catalog_rights_are_valid,
)
from pc_build_recommender.domain import (
    MasterProduct,
    ComponentKind,
    RetailerListing,
    SourceProvenance,
    StockStatus,
)

from .canonical_identity import CanonicalIdentityPreflightReport
from .er_gate import EntityResolutionEvaluation

_DATA_USE_RIGHT_FIELDS = tuple(use.field_name for use in DataUse)

# Each inner tuple is an OR group: at least one field in the group must be present.
CRITICAL_COMPATIBILITY_FIELDS: Mapping[ComponentKind, tuple[tuple[str, ...], ...]] = {
    ComponentKind.CPU: (
        ("socket",),
        ("generation",),
        ("peak_power_watts", "tdp_watts"),
    ),
    ComponentKind.GPU: (
        ("vram_gb",),
        ("length_mm",),
        ("slot_width",),
        ("board_power_watts",),
        ("power_connectors",),
    ),
    ComponentKind.MOTHERBOARD: (
        ("socket",),
        ("chipset",),
        ("supported_cpu_generations", "bios_version"),
        ("form_factor",),
        ("memory_type",),
        ("maximum_memory_gb",),
        ("memory_slots",),
        ("m2_slots",),
        ("sata_ports",),
        ("wifi_support",),
    ),
    ComponentKind.MEMORY: (
        ("memory_type",),
        ("capacity_gb",),
        ("module_count",),
    ),
    ComponentKind.STORAGE: (
        ("capacity_gb",),
        ("interface",),
        ("form_factor",),
    ),
    ComponentKind.POWER_SUPPLY: (
        ("wattage",),
        ("form_factor",),
        ("pcie_connectors",),
        ("eps_connectors",),
    ),
    ComponentKind.COOLER: (
        ("supported_sockets",),
        ("height_mm", "radiator_size_mm"),
        ("estimated_cooling_capacity_watts",),
    ),
    ComponentKind.CASE: (
        ("supported_motherboard_sizes",),
        ("maximum_gpu_length_mm",),
        ("maximum_gpu_slot_width",),
        ("maximum_cooler_height_mm", "radiator_support_mm"),
        ("supported_psu_sizes",),
    ),
}


def _group_name(fields: tuple[str, ...]) -> str:
    return "|".join(fields)


def _is_present(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, str | bytes | Mapping | Iterable):
        # bool is not iterable, so False remains a known/present compatibility value.
        try:
            return len(value) > 0  # type: ignore[arg-type]
        except TypeError:
            return True
    return True


def _provenance_is_complete(
    provenance: SourceProvenance,
    *,
    product_id: str | None = None,
    listing_id: str | None = None,
) -> bool:
    if product_id is not None and provenance.product_id != product_id:
        return False
    if listing_id is not None and provenance.listing_id != listing_id:
        return False
    return all(
        (
            provenance.source_name,
            provenance.source_url,
            provenance.source_type.value,
            provenance.retrieved_at,
            provenance.raw_content_hash,
            provenance.parser_version,
            provenance.licence_or_access_note,
        )
    )


@dataclass(frozen=True, slots=True)
class ProductionCatalogPolicy:
    """Conservative defaults for enabling real recommendation traffic."""

    minimum_products: int = 750
    minimum_products_per_category: int = 1
    minimum_mapping_rate: float = 0.80
    minimum_critical_field_rate: float = 0.90
    require_complete_priced_coverage: bool = True
    require_complete_in_stock_coverage: bool = True
    require_complete_product_provenance: bool = True
    require_complete_offer_provenance: bool = True
    require_explicit_offer_rights: bool = True
    require_production_offer_rights: bool = True
    require_complete_listing_provenance: bool = True
    minimum_er_precision: float = 0.99
    minimum_er_labelled_pairs: int = 1000
    require_promoted_entity_resolution_model: bool = True

    def __post_init__(self) -> None:
        if self.minimum_products < 0 or self.minimum_products_per_category < 0:
            raise ValueError("minimum product counts cannot be negative")
        for name in (
            "minimum_mapping_rate",
            "minimum_critical_field_rate",
            "minimum_er_precision",
        ):
            value = getattr(self, name)
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1")
        if self.minimum_er_labelled_pairs < 1:
            raise ValueError("minimum_er_labelled_pairs must be positive")


@dataclass(frozen=True, slots=True)
class CatalogReadinessReport:
    data_version: str
    product_count: int
    products_by_category: Mapping[str, int]
    compatibility_ready_products_by_category: Mapping[str, int]
    critical_field_present_by_category: Mapping[str, Mapping[str, int]]
    product_provenance_complete_count: int
    offer_provenance_complete_count: int
    offer_rights_explicit_count: int
    offer_rights_production_valid_count: int
    rights_evaluated_on: date
    rights_territory: str
    entity_resolution_evaluation: EntityResolutionEvaluation | None
    entity_resolution_model_version: str | None
    entity_resolution_model_production_authorized: bool
    listing_count: int
    listing_provenance_complete_count: int
    offer_count: int
    mapping_outcomes: Mapping[str, int]
    matched_listings_by_category: Mapping[str, int]
    in_stock_listings_by_category: Mapping[str, int]
    canonical_identity_preflight: CanonicalIdentityPreflightReport | None = None

    def __post_init__(self) -> None:
        if sum(self.products_by_category.values()) != self.product_count:
            raise ValueError("products_by_category must sum to product_count")
        if sum(self.matched_listings_by_category.values()) != self.listing_count:
            raise ValueError("matched_listings_by_category must sum to listing_count")
        if sum(self.mapping_outcomes.values()) != self.offer_count:
            raise ValueError("mapping_outcomes must sum to offer_count")
        if not 0 <= self.product_provenance_complete_count <= self.product_count:
            raise ValueError("product provenance count is outside the product count")
        if not 0 <= self.offer_provenance_complete_count <= self.offer_count:
            raise ValueError("offer provenance count is outside the offer count")
        if not 0 <= self.offer_rights_explicit_count <= self.offer_count:
            raise ValueError("offer rights count is outside the offer count")
        if not 0 <= self.offer_rights_production_valid_count <= self.offer_count:
            raise ValueError("production-valid offer rights count is outside the offer count")
        if self.offer_rights_production_valid_count > self.offer_rights_explicit_count:
            raise ValueError("production-valid offer rights cannot exceed explicit rights")
        if not self.rights_territory.strip():
            raise ValueError("rights_territory is required")
        if not 0 <= self.listing_provenance_complete_count <= self.listing_count:
            raise ValueError("listing provenance count is outside the listing count")
        if (
            self.canonical_identity_preflight is not None
            and self.canonical_identity_preflight.record_count != self.product_count
        ):
            raise ValueError(
                "canonical identity preflight record_count must equal product_count"
            )
        for name in (
            "products_by_category",
            "compatibility_ready_products_by_category",
            "mapping_outcomes",
            "matched_listings_by_category",
            "in_stock_listings_by_category",
        ):
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))
        nested = {
            category: MappingProxyType(dict(fields))
            for category, fields in self.critical_field_present_by_category.items()
        }
        object.__setattr__(
            self,
            "critical_field_present_by_category",
            MappingProxyType(nested),
        )

    @property
    def product_provenance_missing_count(self) -> int:
        return self.product_count - self.product_provenance_complete_count

    @property
    def listing_provenance_missing_count(self) -> int:
        return self.listing_count - self.listing_provenance_complete_count

    @property
    def offer_provenance_missing_count(self) -> int:
        return self.offer_count - self.offer_provenance_complete_count

    @property
    def offer_rights_missing_count(self) -> int:
        return self.offer_count - self.offer_rights_explicit_count

    @property
    def offer_rights_production_invalid_count(self) -> int:
        return self.offer_count - self.offer_rights_production_valid_count

    @property
    def mapping_rate(self) -> float:
        return self.listing_count / self.offer_count if self.offer_count else 0.0

    @property
    def has_complete_priced_coverage(self) -> bool:
        required = {category.value for category in ComponentKind}
        actual = {
            category for category, count in self.matched_listings_by_category.items() if count > 0
        }
        return actual == required

    @property
    def has_complete_in_stock_coverage(self) -> bool:
        required = {category.value for category in ComponentKind}
        actual = {
            category for category, count in self.in_stock_listings_by_category.items() if count > 0
        }
        return actual == required

    def compatibility_ready_rate(self, category: str) -> float:
        count = self.products_by_category.get(category, 0)
        return (
            self.compatibility_ready_products_by_category.get(category, 0) / count if count else 0.0
        )

    def critical_field_rate(self, category: str, field_group: str) -> float:
        count = self.products_by_category.get(category, 0)
        present = self.critical_field_present_by_category.get(category, {}).get(field_group, 0)
        return present / count if count else 0.0

    def blockers(
        self,
        policy: ProductionCatalogPolicy | None = None,
    ) -> tuple[str, ...]:
        active = policy or ProductionCatalogPolicy()
        blockers: list[str] = []
        if self.canonical_identity_preflight is None:
            blockers.append("canonical identity preflight is missing")
        else:
            blockers.extend(self.canonical_identity_preflight.blockers())
        if self.product_count < active.minimum_products:
            blockers.append(
                f"product_count={self.product_count} below minimum={active.minimum_products}"
            )
        for category in ComponentKind:
            category_name = category.value
            count = self.products_by_category.get(category_name, 0)
            if count < active.minimum_products_per_category:
                blockers.append(
                    f"{category_name} product_count={count} below "
                    f"minimum={active.minimum_products_per_category}"
                )
                continue
            for field_group in self.critical_field_present_by_category.get(category_name, {}):
                rate = self.critical_field_rate(category_name, field_group)
                if rate < active.minimum_critical_field_rate:
                    blockers.append(
                        f"{category_name}.{field_group} completeness={rate:.3f} below "
                        f"minimum={active.minimum_critical_field_rate:.3f}"
                    )
        if self.mapping_rate < active.minimum_mapping_rate:
            blockers.append(
                f"offer_mapping_rate={self.mapping_rate:.3f} below "
                f"minimum={active.minimum_mapping_rate:.3f}"
            )
        if active.require_complete_priced_coverage and not self.has_complete_priced_coverage:
            blockers.append("priced listing coverage is incomplete across required categories")
        if active.require_complete_in_stock_coverage and not self.has_complete_in_stock_coverage:
            blockers.append(
                "known in-stock listing coverage is incomplete across required categories"
            )
        if active.require_complete_product_provenance and self.product_provenance_missing_count:
            blockers.append(
                f"{self.product_provenance_missing_count} products lack complete provenance"
            )
        if active.require_complete_offer_provenance and self.offer_provenance_missing_count:
            blockers.append(
                f"{self.offer_provenance_missing_count} offers lack complete provenance"
            )
        if active.require_explicit_offer_rights and self.offer_rights_missing_count:
            blockers.append(
                f"{self.offer_rights_missing_count} offers lack explicit data-use rights"
            )
        if active.require_production_offer_rights and self.offer_rights_production_invalid_count:
            blockers.append(
                f"{self.offer_rights_production_invalid_count} offers lack active "
                f"{self.rights_territory} production grants for display, cache, history, "
                "and derivation"
            )
        if self.entity_resolution_evaluation is None:
            blockers.append(
                "no versioned human-labelled entity-resolution evaluation is configured"
            )
        else:
            blockers.extend(
                self.entity_resolution_evaluation.blockers(
                    minimum_precision=active.minimum_er_precision,
                    minimum_labelled_pairs=active.minimum_er_labelled_pairs,
                )
            )
        if active.require_promoted_entity_resolution_model:
            if not self.entity_resolution_model_version:
                blockers.append("no entity-resolution serving model is configured")
            elif not self.entity_resolution_model_production_authorized:
                blockers.append("entity-resolution serving model is not production authorized")
        if (
            self.entity_resolution_model_version
            and self.entity_resolution_evaluation is not None
            and self.entity_resolution_evaluation.model_version
            != self.entity_resolution_model_version
        ):
            blockers.append(
                "entity-resolution evaluation model_version does not match serving model"
            )
        if active.require_complete_listing_provenance and self.listing_provenance_missing_count:
            blockers.append(
                f"{self.listing_provenance_missing_count} listings lack complete provenance"
            )
        return tuple(blockers)

    def to_dict(
        self,
        policy: ProductionCatalogPolicy | None = None,
    ) -> dict[str, Any]:
        active = policy or ProductionCatalogPolicy()
        field_rates = {
            category: {
                field_group: self.critical_field_rate(category, field_group)
                for field_group in fields
            }
            for category, fields in self.critical_field_present_by_category.items()
        }
        ready_rates = {
            category: self.compatibility_ready_rate(category)
            for category in self.products_by_category
        }
        blockers = self.blockers(active)
        return {
            "data_version": self.data_version,
            "canonical_identity_preflight": (
                self.canonical_identity_preflight.to_dict()
                if self.canonical_identity_preflight is not None
                else None
            ),
            "product_count": self.product_count,
            "products_by_category": dict(self.products_by_category),
            "compatibility_ready_products_by_category": dict(
                self.compatibility_ready_products_by_category
            ),
            "compatibility_ready_rate_by_category": ready_rates,
            "critical_field_present_by_category": {
                category: dict(fields)
                for category, fields in self.critical_field_present_by_category.items()
            },
            "critical_field_rate_by_category": field_rates,
            "product_provenance_complete_count": self.product_provenance_complete_count,
            "product_provenance_missing_count": self.product_provenance_missing_count,
            "offer_provenance_complete_count": self.offer_provenance_complete_count,
            "offer_provenance_missing_count": self.offer_provenance_missing_count,
            "offer_rights_explicit_count": self.offer_rights_explicit_count,
            "offer_rights_missing_count": self.offer_rights_missing_count,
            "offer_rights_production_valid_count": self.offer_rights_production_valid_count,
            "offer_rights_production_invalid_count": (self.offer_rights_production_invalid_count),
            "rights_evaluated_on": self.rights_evaluated_on.isoformat(),
            "rights_territory": self.rights_territory,
            "entity_resolution_evaluation": (
                self.entity_resolution_evaluation.to_dict()
                if self.entity_resolution_evaluation is not None
                else None
            ),
            "entity_resolution_model_version": self.entity_resolution_model_version,
            "entity_resolution_model_production_authorized": (
                self.entity_resolution_model_production_authorized
            ),
            "listing_count": self.listing_count,
            "listing_provenance_complete_count": self.listing_provenance_complete_count,
            "listing_provenance_missing_count": self.listing_provenance_missing_count,
            "offer_count": self.offer_count,
            "mapping_rate": self.mapping_rate,
            "mapping_outcomes": dict(self.mapping_outcomes),
            "matched_listings_by_category": dict(self.matched_listings_by_category),
            "in_stock_listings_by_category": dict(self.in_stock_listings_by_category),
            "has_complete_priced_coverage": self.has_complete_priced_coverage,
            "has_complete_in_stock_coverage": self.has_complete_in_stock_coverage,
            "production_ready": not blockers,
            "production_blockers": list(blockers),
        }


class ProductionCatalogReadinessError(RuntimeError):
    def __init__(self, report: CatalogReadinessReport, blockers: tuple[str, ...]) -> None:
        super().__init__("catalog is not production-ready: " + "; ".join(blockers))
        self.report = report
        self.blockers = blockers


class CatalogReadinessAccumulator:
    """Streaming accumulator; it never retains products or listings."""

    def __init__(
        self,
        *,
        rights_territory: str = "SG",
        rights_evaluated_on: date | None = None,
    ) -> None:
        self.product_count = 0
        self.products_by_category: Counter[str] = Counter()
        self.ready_by_category: Counter[str] = Counter()
        self.field_counts: dict[str, Counter[str]] = defaultdict(Counter)
        self.product_provenance_complete_count = 0
        self.offer_provenance_complete_count = 0
        self.offer_rights_explicit_count = 0
        self.offer_rights_production_valid_count = 0
        self.rights_territory = rights_territory.strip().upper()
        self.rights_evaluated_on = rights_evaluated_on or date.today()
        self.listing_count = 0
        self.listing_provenance_complete_count = 0
        self.matched_by_category: Counter[str] = Counter()
        self.in_stock_by_category: Counter[str] = Counter()
        self.canonical_identity_preflight: CanonicalIdentityPreflightReport | None = None

    def observe_canonical_identity_preflight(
        self,
        report: CanonicalIdentityPreflightReport,
    ) -> None:
        if self.canonical_identity_preflight is not None:
            raise ValueError("canonical identity preflight was already observed")
        self.canonical_identity_preflight = report

    def observe_product(self, product: MasterProduct) -> None:
        category = product.category.value
        self.product_count += 1
        self.products_by_category[category] += 1
        attributes = product.category_attributes.model_dump(mode="python")
        ready = True
        for group in CRITICAL_COMPATIBILITY_FIELDS[product.category]:
            group_name = _group_name(group)
            present = any(_is_present(attributes.get(field)) for field in group)
            if present:
                self.field_counts[category][group_name] += 1
            else:
                ready = False
        if ready:
            self.ready_by_category[category] += 1
        if any(
            _provenance_is_complete(item, product_id=product.product_id)
            for item in product.provenance
        ):
            self.product_provenance_complete_count += 1

    def observe_listing(
        self,
        listing: RetailerListing,
        *,
        category: ComponentKind,
        provenance: SourceProvenance | None,
    ) -> None:
        self.listing_count += 1
        self.matched_by_category[category.value] += 1
        if listing.stock_status is StockStatus.IN_STOCK:
            self.in_stock_by_category[category.value] += 1
        if provenance is not None and _provenance_is_complete(
            provenance, listing_id=listing.listing_id
        ):
            self.listing_provenance_complete_count += 1

    def observe_offer_provenance(
        self,
        provenance: SourceProvenance,
        *,
        listing_id: str,
    ) -> None:
        if _provenance_is_complete(provenance, listing_id=listing_id):
            self.offer_provenance_complete_count += 1

    def observe_offer_rights(self, rights: object) -> None:
        if isinstance(rights, Mapping) and all(
            type(rights.get(field)) is bool for field in _DATA_USE_RIGHT_FIELDS
        ):
            self.offer_rights_explicit_count += 1
        if production_catalog_rights_are_valid(
            rights,
            territory=self.rights_territory,
            on_date=self.rights_evaluated_on,
        ):
            self.offer_rights_production_valid_count += 1

    def finish(
        self,
        *,
        data_version: str,
        offer_count: int,
        mapping_outcomes: Mapping[str, int],
        entity_resolution_evaluation: EntityResolutionEvaluation | None = None,
        entity_resolution_model_version: str | None = None,
        entity_resolution_model_production_authorized: bool = False,
    ) -> CatalogReadinessReport:
        field_counts: dict[str, dict[str, int]] = {}
        product_counts: dict[str, int] = {}
        ready_counts: dict[str, int] = {}
        matched_counts: dict[str, int] = {}
        in_stock_counts: dict[str, int] = {}
        for category in ComponentKind:
            category_name = category.value
            product_counts[category_name] = self.products_by_category.get(category_name, 0)
            ready_counts[category_name] = self.ready_by_category.get(category_name, 0)
            matched_counts[category_name] = self.matched_by_category.get(category_name, 0)
            in_stock_counts[category_name] = self.in_stock_by_category.get(category_name, 0)
            field_counts[category_name] = {
                _group_name(group): self.field_counts[category_name].get(_group_name(group), 0)
                for group in CRITICAL_COMPATIBILITY_FIELDS[category]
            }
        return CatalogReadinessReport(
            data_version=data_version,
            product_count=self.product_count,
            products_by_category=product_counts,
            compatibility_ready_products_by_category=ready_counts,
            critical_field_present_by_category=field_counts,
            product_provenance_complete_count=self.product_provenance_complete_count,
            offer_provenance_complete_count=self.offer_provenance_complete_count,
            offer_rights_explicit_count=self.offer_rights_explicit_count,
            offer_rights_production_valid_count=(self.offer_rights_production_valid_count),
            rights_evaluated_on=self.rights_evaluated_on,
            rights_territory=self.rights_territory,
            entity_resolution_evaluation=entity_resolution_evaluation,
            entity_resolution_model_version=entity_resolution_model_version,
            entity_resolution_model_production_authorized=(
                entity_resolution_model_production_authorized
            ),
            listing_count=self.listing_count,
            listing_provenance_complete_count=self.listing_provenance_complete_count,
            offer_count=offer_count,
            mapping_outcomes=dict(sorted(mapping_outcomes.items())),
            matched_listings_by_category=matched_counts,
            in_stock_listings_by_category=in_stock_counts,
            canonical_identity_preflight=self.canonical_identity_preflight,
        )


def validate_production_readiness(
    report: CatalogReadinessReport,
    policy: ProductionCatalogPolicy | None = None,
) -> None:
    blockers = report.blockers(policy)
    if blockers:
        raise ProductionCatalogReadinessError(report, blockers)
