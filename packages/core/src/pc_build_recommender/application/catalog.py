"""Immutable online-serving snapshot built from the canonical catalogue."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import fmean
from types import MappingProxyType
from typing import Any, Literal, Protocol

from pc_build_recommender.compatibility import (
    AUTHORITATIVE_COMPATIBILITY_POLICY,
    COMPATIBILITY_AUTHORITY_KEY,
    CONTROLLED_NON_PRODUCTION_POLICY,
)
from pc_build_recommender.domain import (
    BenchmarkResult,
    CanonicalProduct,
    ComponentKind,
    ListingCondition,
    ProductStatus,
    RetailerListing,
    ReviewNote,
    SourceType,
    StockStatus,
)
from pc_build_recommender.retrieval import ProductDocument

from .models import CatalogIntegrityError, CatalogItem, EmptyCatalogError

type CompatibilityEvidencePolicy = Literal["controlled_non_production", "authoritative_only"]

_COMPATIBILITY_EVIDENCE_POLICIES = frozenset(
    {CONTROLLED_NON_PRODUCTION_POLICY, AUTHORITATIVE_COMPATIBILITY_POLICY}
)
_COMPATIBILITY_CRITICAL_FIELDS: Mapping[ComponentKind, frozenset[str]] = {
    ComponentKind.CPU: frozenset({"socket", "generation", "peak_power_watts"}),
    ComponentKind.GPU: frozenset(
        {"length_mm", "slot_width", "board_power_watts", "power_connectors"}
    ),
    ComponentKind.MOTHERBOARD: frozenset(
        {
            "socket",
            "chipset",
            "supported_cpu_generations",
            "form_factor",
            "memory_type",
            "maximum_memory_gb",
            "memory_slots",
            "pcie_slots",
            "m2_slots",
            "sata_ports",
            "bios_version",
        }
    ),
    ComponentKind.MEMORY: frozenset({"memory_type", "capacity_gb", "module_count"}),
    ComponentKind.STORAGE: frozenset({"interface", "form_factor"}),
    ComponentKind.POWER_SUPPLY: frozenset(
        {"wattage", "form_factor", "pcie_connectors", "eps_connectors", "atx_version"}
    ),
    ComponentKind.COOLER: frozenset(
        {
            "cooler_type",
            "supported_sockets",
            "height_mm",
            "radiator_size_mm",
            "estimated_cooling_capacity_watts",
        }
    ),
    ComponentKind.CASE: frozenset(
        {
            "supported_motherboard_sizes",
            "maximum_gpu_length_mm",
            "maximum_gpu_slot_width",
            "maximum_cooler_height_mm",
            "supported_psu_sizes",
            "radiator_support_mm",
            "drive_bays",
        }
    ),
}


class CatalogReader(Protocol):
    """The persistence methods needed to construct an online snapshot."""

    def list_products(
        self,
        *,
        category: ComponentKind | None = None,
        brand: str | None = None,
        status: ProductStatus | None = ProductStatus.ACTIVE,
        offset: int = 0,
        limit: int = 100,
    ) -> list[CanonicalProduct]: ...

    def get_product(self, product_id: str) -> CanonicalProduct | None: ...

    def list_listings(
        self,
        *,
        product_id: str | None = None,
        retailer: str | None = None,
        stock_status: StockStatus | None = None,
        limit: int = 100,
    ) -> list[RetailerListing]: ...

    def list_benchmarks(
        self, product_id: str, *, workload: str | None = None
    ) -> list[BenchmarkResult]: ...

    def list_review_evidence(self, product_id: str) -> list[ReviewNote]: ...


def _all_products(repository: CatalogReader) -> list[CanonicalProduct]:
    products: list[CanonicalProduct] = []
    offset = 0
    while True:
        page = repository.list_products(
            status=None,
            offset=offset,
            limit=1000,
        )
        products.extend(page)
        if len(page) < 1000:
            break
        offset += len(page)
    return products


def _preferred_listing(listings: Sequence[RetailerListing]) -> RetailerListing | None:
    # The product scope is new components.  A used/open-box/refurbished offer
    # must never become an acquisition candidate merely because it is cheap.
    new_listings = [item for item in listings if item.condition == ListingCondition.NEW]
    in_stock = [item for item in new_listings if item.stock_status == StockStatus.IN_STOCK]
    return min(in_stock, key=lambda item: (item.total_price, item.listing_id), default=None)


def _benchmark_signature(product: CanonicalProduct, benchmark: BenchmarkResult) -> tuple[str, ...]:
    return (
        product.category.value,
        benchmark.workload.value,
        benchmark.benchmark_name.casefold(),
        benchmark.benchmark_version.casefold(),
        (benchmark.resolution or "").casefold(),
        (benchmark.preset or "").casefold(),
        (benchmark.operating_system or "").casefold(),
        (benchmark.driver_version or "").casefold(),
        str(benchmark.higher_is_better),
    )


def _normalised_workload_scores(
    products: Sequence[CanonicalProduct],
    benchmarks_by_product: Mapping[str, Sequence[BenchmarkResult]],
) -> dict[str, dict[str, float]]:
    """Normalise only within comparable benchmark configurations.

    A product can contribute several comparable observations.  Each benchmark
    family is first scaled independently to 0..100; product/workload scores are
    then the mean of those family-local values.  This avoids combining raw FPS,
    throughput, and compilation values as if their units were interchangeable.
    """

    products_by_id = {product.product_id: product for product in products}
    groups: dict[tuple[str, ...], list[tuple[str, BenchmarkResult]]] = defaultdict(list)
    for product_id, observations in benchmarks_by_product.items():
        product = products_by_id[product_id]
        for observation in observations:
            groups[_benchmark_signature(product, observation)].append((product_id, observation))

    by_product_workload: dict[tuple[str, str], list[float]] = defaultdict(list)
    for signature in sorted(groups):
        group_members = groups[signature]
        values = [item.score for _, item in group_members if math.isfinite(item.score)]
        if not values:
            continue
        minimum = min(values)
        maximum = max(values)
        higher_is_better = group_members[0][1].higher_is_better
        for product_id, observation in group_members:
            if not math.isfinite(observation.score):
                continue
            if math.isclose(maximum, minimum):
                normalised = 50.0
            else:
                normalised = 100.0 * (observation.score - minimum) / (maximum - minimum)
                if not higher_is_better:
                    normalised = 100.0 - normalised
            by_product_workload[(product_id, observation.workload.value)].append(normalised)

    result: dict[str, dict[str, float]] = defaultdict(dict)
    for (product_id, workload), scores in sorted(by_product_workload.items()):
        result[product_id][workload] = round(fmean(scores), 6)
    return dict(result)


def _compatibility_authority(
    product: CanonicalProduct,
    attributes: Mapping[str, Any],
    *,
    evidence_policy: CompatibilityEvidencePolicy,
) -> dict[str, Any]:
    provenance = sorted(product.provenance, key=lambda item: item.provenance_id)
    sources = [
        {
            "provenance_id": item.provenance_id,
            "source_name": item.source_name,
            "source_url": item.source_url,
            "source_type": item.source_type.value,
            "last_verified_at": (
                item.last_verified_at.isoformat() if item.last_verified_at is not None else None
            ),
        }
        for item in provenance
    ]
    critical_fields = sorted(
        field
        for field in _COMPATIBILITY_CRITICAL_FIELDS[product.category]
        if attributes.get(field) is not None
    )
    if evidence_policy == CONTROLLED_NON_PRODUCTION_POLICY:
        return {
            "policy": CONTROLLED_NON_PRODUCTION_POLICY,
            "decision": "controlled_non_production",
            "production_eligible": False,
            "authoritative_fields": [],
            "unverified_fields": critical_fields,
            "sources": sources,
            "reason": "Controlled demo and test records are explicitly excluded from production.",
        }

    manufacturer_only = bool(provenance) and all(
        item.source_type is SourceType.MANUFACTURER for item in provenance
    )
    if manufacturer_only:
        return {
            "policy": AUTHORITATIVE_COMPATIBILITY_POLICY,
            "decision": "authoritative",
            "production_eligible": True,
            "authoritative_fields": critical_fields,
            "unverified_fields": [],
            "sources": sources,
            "reason": "All product specification provenance is manufacturer-authored.",
        }

    if not provenance:
        reason = "The product has no specification provenance."
    elif any(item.source_type is SourceType.MANUFACTURER for item in provenance):
        reason = (
            "Product-level provenance mixes manufacturer and non-authoritative sources; "
            "field-level lineage is unavailable."
        )
    else:
        reason = "Product specifications are sourced only from non-authoritative community data."
    return {
        "policy": AUTHORITATIVE_COMPATIBILITY_POLICY,
        "decision": "unknown",
        "production_eligible": False,
        "authoritative_fields": [],
        "unverified_fields": critical_fields or ["category_attributes"],
        "sources": sources,
        "reason": reason,
    }


def _compatibility_record(
    product: CanonicalProduct,
    *,
    evidence_policy: CompatibilityEvidencePolicy,
) -> dict[str, Any]:
    payload = product.model_dump(mode="json", exclude={"provenance"})
    attributes = product.category_attributes.model_dump(mode="json")

    # Compatibility accepts storage-agnostic mappings.  These aliases bridge
    # explicit domain field names without making up missing evidence.
    if "peak_power_watts" in attributes:
        payload["peak_power_w"] = attributes["peak_power_watts"]
    if "board_power_watts" in attributes:
        payload["board_power_w"] = attributes["board_power_watts"]

    provenance = sorted(product.provenance, key=lambda item: item.provenance_id)
    if provenance:
        payload["source_url"] = provenance[0].source_url
        payload["source"] = provenance[0].source_name
    payload[COMPATIBILITY_AUTHORITY_KEY] = _compatibility_authority(
        product,
        attributes,
        evidence_policy=evidence_policy,
    )
    return payload


def _document(item: CatalogItem) -> ProductDocument:
    product = item.product
    common = product.common_attributes.model_dump(mode="json")
    category = product.category_attributes.model_dump(mode="json")
    attributes = {
        **common,
        **category,
        "canonical_name": product.canonical_name,
        "model": product.model,
        "manufacturer_part_number": product.manufacturer_part_number,
        "workload_scores": dict(item.workload_scores),
    }
    attribute_text = " ".join(
        f"{name.replace('_', ' ')} {value}"
        for name, value in sorted(attributes.items())
        if value is not None and not isinstance(value, Mapping)
    )
    search_text = " ".join(
        part
        for part in (
            product.category.value,
            product.brand,
            product.canonical_name,
            product.model,
            product.manufacturer_part_number,
            attribute_text,
            " ".join(sorted(item.workload_scores)),
        )
        if part
    )
    return ProductDocument(
        product_id=product.product_id,
        category=product.category.value,
        text=search_text,
        brand=product.brand,
        price_sgd=item.price_sgd,
        stock_status=item.listing.stock_status.value if item.listing else None,
        attributes=attributes,
    )


def _content_version(
    products: Sequence[CanonicalProduct],
    listings_by_product: Mapping[str, Sequence[RetailerListing]],
    benchmarks_by_product: Mapping[str, Sequence[BenchmarkResult]],
    reviews_by_product: Mapping[str, Sequence[ReviewNote]],
) -> str:
    payload = {
        "products": [
            product.model_dump(mode="json")
            for product in sorted(products, key=lambda item: item.product_id)
        ],
        "listings": [
            listing.model_dump(mode="json")
            for product_id in sorted(listings_by_product)
            for listing in sorted(listings_by_product[product_id], key=lambda item: item.listing_id)
        ],
        "benchmarks": [
            benchmark.model_dump(mode="json")
            for product_id in sorted(benchmarks_by_product)
            for benchmark in sorted(
                benchmarks_by_product[product_id], key=lambda item: item.benchmark_id
            )
        ],
        "review_evidence": [
            evidence.model_dump(mode="json")
            for product_id in sorted(reviews_by_product)
            for evidence in sorted(
                reviews_by_product[product_id], key=lambda item: item.evidence_id
            )
        ],
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return f"catalog-{digest[:16]}"


@dataclass(frozen=True, slots=True)
class ApplicationCatalog:
    """Immutable product/listing/benchmark snapshot for deterministic serving."""

    items: tuple[CatalogItem, ...]
    documents: tuple[ProductDocument, ...]
    data_version: str
    compatibility_evidence_policy: CompatibilityEvidencePolicy
    _items_by_id: Mapping[str, CatalogItem]

    @classmethod
    def from_repository(
        cls,
        repository: CatalogReader,
        *,
        data_version: str | None = None,
        compatibility_evidence_policy: CompatibilityEvidencePolicy = (
            AUTHORITATIVE_COMPATIBILITY_POLICY
        ),
    ) -> ApplicationCatalog:
        if compatibility_evidence_policy not in _COMPATIBILITY_EVIDENCE_POLICIES:
            raise ValueError(
                f"unsupported compatibility evidence policy: {compatibility_evidence_policy}"
            )
        products = _all_products(repository)
        if not products:
            raise EmptyCatalogError(
                "cannot create recommendation services: the canonical catalogue is empty"
            )
        ids = [product.product_id for product in products]
        if len(ids) != len(set(ids)):
            raise CatalogIntegrityError("canonical catalogue contains duplicate product IDs")

        listings_by_product: dict[str, list[RetailerListing]] = {}
        benchmarks_by_product: dict[str, list[BenchmarkResult]] = {}
        reviews_by_product: dict[str, list[ReviewNote]] = {}
        for product in products:
            listings_by_product[product.product_id] = repository.list_listings(
                product_id=product.product_id,
                limit=1000,
            )
            benchmarks_by_product[product.product_id] = repository.list_benchmarks(
                product.product_id
            )
            reviews_by_product[product.product_id] = repository.list_review_evidence(
                product.product_id
            )
        workload_scores = _normalised_workload_scores(products, benchmarks_by_product)

        items: list[CatalogItem] = []
        for product in sorted(products, key=lambda item: item.product_id):
            listing = _preferred_listing(listings_by_product[product.product_id])
            product_scores = workload_scores.get(product.product_id, {})
            product_benchmarks: dict[str, tuple[BenchmarkResult, ...]] = {}
            for benchmark in benchmarks_by_product[product.product_id]:
                product_benchmarks.setdefault(benchmark.workload.value, ())
                product_benchmarks[benchmark.workload.value] += (benchmark,)
            signals: dict[str, float] = {
                "availability_score": float(
                    listing is not None and listing.stock_status == StockStatus.IN_STOCK
                ),
                "freshness_score": 1.0,
            }
            if product_scores:
                signals["observed_benchmark_score"] = fmean(product_scores.values())
            if product.common_attributes.warranty_years is not None:
                signals["warranty_years"] = product.common_attributes.warranty_years
            items.append(
                CatalogItem(
                    product=product,
                    listing=listing,
                    compatibility_record=_compatibility_record(
                        product,
                        evidence_policy=compatibility_evidence_policy,
                    ),
                    workload_scores=product_scores,
                    workload_benchmarks=product_benchmarks,
                    review_evidence=tuple(reviews_by_product[product.product_id]),
                    ranking_signals=signals,
                )
            )

        item_tuple = tuple(items)
        version = data_version or _content_version(
            products, listings_by_product, benchmarks_by_product, reviews_by_product
        )
        return cls(
            items=item_tuple,
            documents=tuple(
                _document(item)
                for item in item_tuple
                if item.product.status is ProductStatus.ACTIVE
            ),
            data_version=version,
            compatibility_evidence_policy=compatibility_evidence_policy,
            _items_by_id=MappingProxyType({item.product.product_id: item for item in item_tuple}),
        )

    def get(self, product_id: str) -> CatalogItem | None:
        return self._items_by_id.get(product_id)

    def require(self, product_id: str) -> CatalogItem:
        item = self.get(product_id)
        if item is None:
            raise CatalogIntegrityError(f"canonical product not found: {product_id}")
        return item

    def document_for(self, product_id: str) -> ProductDocument:
        """Return a filter document even for a retained inactive product."""

        return _document(self.require(product_id))

    def categories(self) -> frozenset[str]:
        return frozenset(item.product.category.value for item in self.items)

    @property
    def has_authoritative_compatibility_coverage(self) -> bool:
        """Whether every item may participate in production compatibility decisions."""

        if self.compatibility_evidence_policy != AUTHORITATIVE_COMPATIBILITY_POLICY:
            return False
        return all(
            isinstance(
                authority := item.compatibility_record.get(COMPATIBILITY_AUTHORITY_KEY),
                Mapping,
            )
            and authority.get("decision") == "authoritative"
            for item in self.items
        )
