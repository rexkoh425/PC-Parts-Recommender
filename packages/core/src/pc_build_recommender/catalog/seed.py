"""Deterministic, idempotent JSON seed loading for local and test catalogues."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID, uuid5

from sqlalchemy.orm import Session

from pc_build_recommender.domain import (
    BenchmarkResult,
    MasterProduct,
    PriceSample,
    RetailerListing,
    SourceProvenance,
)

from .repository import CatalogRepository

SEED_NAMESPACE = UUID("e639af92-166e-5fcb-a18f-f9f1b902e8de")
EPOCH = "1970-01-01T00:00:00+00:00"


@dataclass(frozen=True, slots=True)
class SeedLoadResult:
    products: int = 0
    listings: int = 0
    price_snapshots: int = 0
    benchmarks: int = 0
    provenance_records: int = 0

    @property
    def total_records(self) -> int:
        return (
            self.products
            + self.listings
            + self.price_snapshots
            + self.benchmarks
            + self.provenance_records
        )


def deterministic_id(prefix: str, *parts: object) -> str:
    """Generate a stable identifier from a record's natural identity."""

    normalised = "|".join(str(part).strip().casefold() for part in parts)
    return f"{prefix}_{uuid5(SEED_NAMESPACE, f'{prefix}|{normalised}').hex}"


def _require(mapping: dict[str, Any], key: str, record_type: str) -> Any:
    value = mapping.get(key)
    if value in (None, ""):
        raise ValueError(f"{record_type} seed record requires {key}")
    return value


def _prepare_product(raw: dict[str, Any]) -> dict[str, Any]:
    product = deepcopy(raw)
    category = _require(product, "category", "product")
    brand = _require(product, "brand", "product")
    model = _require(product, "model", "product")
    identity = product.get("manufacturer_part_number") or model
    product.setdefault("product_id", deterministic_id("prod", category, brand, identity))
    product.setdefault("canonical_name", f"{brand} {model}")
    product.setdefault("category_attributes", {})
    product.setdefault("created_at", EPOCH)
    product.setdefault("updated_at", EPOCH)
    provenance_items = product.get("provenance", [])
    for provenance in provenance_items:
        provenance.setdefault("product_id", product["product_id"])
        _prepare_provenance(provenance)
    return product


def _prepare_listing(raw: dict[str, Any]) -> dict[str, Any]:
    listing = deepcopy(raw)
    product_id = _require(listing, "product_id", "listing")
    retailer = _require(listing, "retailer", "listing")
    source_id = _require(listing, "source_listing_id", "listing")
    listing.setdefault("listing_id", deterministic_id("listing", retailer, source_id))
    listing.setdefault("title", source_id)
    listing.setdefault("listing_url", f"seed://{retailer}/{source_id}")
    listing.setdefault("first_seen_at", EPOCH)
    listing.setdefault("last_seen_at", listing["first_seen_at"])
    if not product_id:
        raise ValueError("listing product_id cannot be empty")
    return listing


def _prepare_price(raw: dict[str, Any]) -> dict[str, Any]:
    price = deepcopy(raw)
    listing_id = _require(price, "listing_id", "price snapshot")
    observed_at = _require(price, "observed_at", "price snapshot")
    price.setdefault("snapshot_id", deterministic_id("price", listing_id, observed_at))
    return price


def _prepare_benchmark(raw: dict[str, Any]) -> dict[str, Any]:
    benchmark = deepcopy(raw)
    product_id = _require(benchmark, "product_id", "benchmark")
    workload = _require(benchmark, "workload", "benchmark")
    name = _require(benchmark, "benchmark_name", "benchmark")
    version = _require(benchmark, "benchmark_version", "benchmark")
    discriminator = (
        benchmark.get("resolution"),
        benchmark.get("preset"),
        benchmark.get("operating_system"),
        benchmark.get("driver_version"),
    )
    benchmark.setdefault(
        "benchmark_id",
        deterministic_id("bench", product_id, workload, name, version, *discriminator),
    )
    return benchmark


def _prepare_provenance(raw: dict[str, Any]) -> dict[str, Any]:
    source_name = _require(raw, "source_name", "provenance")
    source_url = _require(raw, "source_url", "provenance")
    content_hash = _require(raw, "raw_content_hash", "provenance")
    target = raw.get("product_id") or raw.get("listing_id") or "unmapped"
    raw.setdefault(
        "provenance_id",
        deterministic_id("src", target, source_name, source_url, content_hash),
    )
    raw.setdefault("retrieved_at", EPOCH)
    raw.setdefault("last_verified_at", raw["retrieved_at"])
    return raw


def load_seed_data(session: Session, data: dict[str, Any]) -> SeedLoadResult:
    """Validate and upsert seed data in stable order without committing the transaction."""

    if not isinstance(data, dict):
        raise TypeError("seed data must be a JSON object")
    repository = CatalogRepository(session)

    products = [_prepare_product(item) for item in data.get("products", [])]
    listings = [_prepare_listing(item) for item in data.get("listings", [])]
    price_inputs = data.get("price_snapshots")
    if price_inputs is None:
        price_inputs = data.get("prices", [])
    if not isinstance(price_inputs, list):
        raise TypeError("price_snapshots must be a JSON array")
    prices = [_prepare_price(item) for item in price_inputs]
    benchmarks = [_prepare_benchmark(item) for item in data.get("benchmarks", [])]
    provenance = [deepcopy(item) for item in data.get("provenance", [])]
    for item in provenance:
        _prepare_provenance(item)

    for item in sorted(products, key=lambda value: value["product_id"]):
        repository.upsert_product(MasterProduct.model_validate(item))
    for item in sorted(listings, key=lambda value: value["listing_id"]):
        repository.upsert_listing(RetailerListing.model_validate(item))
    for item in sorted(prices, key=lambda value: value["snapshot_id"]):
        repository.upsert_price_snapshot(PriceSample.model_validate(item))
    for item in sorted(benchmarks, key=lambda value: value["benchmark_id"]):
        repository.upsert_benchmark(BenchmarkResult.model_validate(item))
    for item in sorted(provenance, key=lambda value: value["provenance_id"]):
        repository.upsert_provenance(SourceProvenance.model_validate(item))

    return SeedLoadResult(
        products=len(products),
        listings=len(listings),
        price_snapshots=len(prices),
        benchmarks=len(benchmarks),
        provenance_records=len(provenance)
        + sum(len(item.get("provenance", [])) for item in products),
    )


def load_seed_file(session: Session, path: str | Path) -> SeedLoadResult:
    seed_path = Path(path)
    with seed_path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    return load_seed_data(session, data)


# Friendly alias for CLI scripts.
seed_catalog = load_seed_file
