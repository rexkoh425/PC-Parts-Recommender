"""Deterministic synthetic fixtures for engineering tests only.

Nothing produced here is measured retailer or manufacturer evidence.  Every record and
pair carries ``is_synthetic=True`` and evaluation helpers refuse to promote its metrics.
"""

from __future__ import annotations

import hashlib
import random
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .records import CanonicalProductRecord, ListingRow, LabelledPair

SYNTHETIC_PROVENANCE = (
    "synthetic engineering fixture; not retailer, manufacturer, benchmark, or user evidence"
)


@dataclass(frozen=True, slots=True)
class SyntheticEntityResolutionDataset:
    products: tuple[CanonicalProductRecord, ...]
    listings: tuple[ListingRow, ...]
    pairs: tuple[LabelledPair, ...]
    seed: int
    provenance: str = SYNTHETIC_PROVENANCE
    is_synthetic: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "provenance": self.provenance,
            "is_synthetic": True,
            "products": [product.to_dict() for product in self.products],
            "listings": [listing.to_dict() for listing in self.listings],
            "pairs": [pair.to_dict() for pair in self.pairs],
        }


_BRANDS = ("Aster", "Boreal", "Cinder", "Dovetail")


def _embedding(identifier: str, dimensions: int = 8) -> tuple[float, ...]:
    digest = hashlib.sha256(identifier.encode("utf-8")).digest()
    values = [((digest[index] / 255.0) * 2.0) - 1.0 for index in range(dimensions)]
    norm = sum(value * value for value in values) ** 0.5 or 1.0
    return tuple(value / norm for value in values)


def _catalog_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    memory_capacities = (16, 32, 48, 64)
    gpu_vram = (8, 12, 16, 24)
    storage_capacities = (512, 1024, 2048, 4096)
    psu_wattages = (550, 650, 750, 850)

    for brand_index, brand in enumerate(_BRANDS):
        for variant_index in range(4):
            capacity = memory_capacities[variant_index]
            speed = 5600 + brand_index * 200
            rows.append(
                {
                    "category": "memory",
                    "brand": brand,
                    "model": f"Velocity M{brand_index + 1}{variant_index + 1}",
                    "variant": f"{capacity}GB DDR5-{speed}",
                    "attributes": {
                        "capacity_gb": capacity,
                        "module_count": 2,
                        "memory_type": "DDR5",
                        "speed_mts": speed,
                    },
                    "price": 55.0 + capacity * 2.0,
                }
            )
            vram = gpu_vram[variant_index]
            rows.append(
                {
                    "category": "gpu",
                    "brand": brand,
                    "model": f"Nebula GX{60 + brand_index}{variant_index} XT",
                    "variant": f"{vram}GB GDDR6",
                    "attributes": {"vram_gb": vram, "slot_width": 2.5},
                    "price": 320.0 + variant_index * 170.0 + brand_index * 15.0,
                }
            )
            storage_gb = storage_capacities[variant_index]
            storage_label = f"{storage_gb // 1024}TB" if storage_gb >= 1024 else f"{storage_gb}GB"
            rows.append(
                {
                    "category": "storage",
                    "brand": brand,
                    "model": f"Flashline S{brand_index + 1}{variant_index + 1}",
                    "variant": f"{storage_label} NVMe",
                    "attributes": {"capacity_gb": storage_gb, "interface": "PCIe 4.0"},
                    "price": 45.0 + storage_gb * 0.06,
                }
            )
            watts = psu_wattages[variant_index]
            rows.append(
                {
                    "category": "psu",
                    "brand": brand,
                    "model": f"Current P{brand_index + 1}{variant_index + 1}",
                    "variant": f"{watts}W 80 Plus Gold",
                    "attributes": {"wattage_w": watts, "form_factor": "ATX"},
                    "price": 65.0 + watts * 0.08,
                }
            )
    return rows


def _make_products(product_count: int) -> tuple[CanonicalProductRecord, ...]:
    if product_count < 4:
        raise ValueError("product_count must be at least four")
    catalog_rows = _catalog_rows()
    by_category: dict[str, list[dict[str, Any]]] = {}
    for row in catalog_rows:
        by_category.setdefault(str(row["category"]), []).append(row)
    # Interleave two-variant chunks.  Even the minimum four-product fixture then contains
    # useful same-category negatives, while the default remains balanced across categories.
    rows: list[dict[str, Any]] = []
    categories = tuple(by_category)
    for offset in range(0, max(map(len, by_category.values())), 2):
        for category in categories:
            rows.extend(by_category[category][offset : offset + 2])
    if product_count > len(rows):
        raise ValueError(f"product_count cannot exceed {len(rows)}")
    products: list[CanonicalProductRecord] = []
    for index, row in enumerate(rows[:product_count]):
        product_id = f"synthetic-product-{index:04d}"
        mpn = f"SYN-{row['category'][:3].upper()}-{index:04d}"
        canonical_name = f"{row['brand']} {row['model']} {row['variant']}"
        products.append(
            CanonicalProductRecord(
                product_id=product_id,
                category=str(row["category"]),
                brand=str(row["brand"]),
                model=str(row["model"]),
                canonical_name=canonical_name,
                manufacturer_part_number=mpn,
                gtin=f"0990000{index:06d}",
                attributes=dict(row["attributes"]),
                price_sgd=float(row["price"]),
                embedding=_embedding(product_id),
                is_synthetic=True,
            )
        )
    return tuple(products)


def _title_variants(product: CanonicalProductRecord) -> tuple[str, ...]:
    canonical = product.canonical_name
    compact = re.sub(r"\s+", " ", canonical.replace("-", " ")).strip()
    return (
        canonical.upper(),
        f"{product.model} by {product.brand} {compact.split(product.model, 1)[-1].strip()}",
        (
            f"{product.brand} {compact.replace(product.brand, '', 1).strip()} "
            f"{product.manufacturer_part_number}"
        ),
    )


def synthetic_catalog(
    *,
    seed: int = 7,
    product_count: int = 48,
    positive_variants: int = 2,
    negatives_per_listing: int = 2,
) -> SyntheticEntityResolutionDataset:
    """Build a deterministic, explicitly synthetic pair dataset.

    Negative examples favour same-brand, same-category neighbouring variants, producing
    hard negatives such as 32 GB versus 64 GB rather than only trivial category mismatches.
    """

    if not 1 <= positive_variants <= 3:
        raise ValueError("positive_variants must be between one and three")
    if negatives_per_listing < 1:
        raise ValueError("negatives_per_listing must be at least one")
    rng = random.Random(seed)
    products = _make_products(product_count)
    by_category: dict[str, list[CanonicalProductRecord]] = {}
    for product in products:
        by_category.setdefault(product.category, []).append(product)

    listings: list[ListingRow] = []
    pairs: list[LabelledPair] = []
    for product_index, product in enumerate(products):
        variants = _title_variants(product)
        for variant_index in range(positive_variants):
            listing_id = f"synthetic-listing-{product_index:04d}-{variant_index}"
            jitter = rng.uniform(-0.08, 0.08)
            listing = ListingRow(
                listing_id=listing_id,
                title=variants[variant_index],
                category=product.category,
                brand=product.brand if variant_index != 1 else "",
                manufacturer_part_number=(
                    product.manufacturer_part_number if variant_index == 0 else None
                ),
                gtin=product.gtin if variant_index == 0 else None,
                attributes=dict(product.attributes) if variant_index == 0 else {},
                current_price_sgd=(product.price_sgd or 0.0) * (1.0 + jitter),
                embedding=product.embedding,
                retailer=f"Synthetic Retailer {variant_index + 1}",
                is_synthetic=True,
            )
            listings.append(listing)
            pairs.append(
                LabelledPair(
                    pair_id=f"{listing_id}--positive",
                    listing=listing,
                    product=product,
                    label=1,
                    is_synthetic=True,
                )
            )

            candidate_negatives = [
                candidate
                for candidate in by_category[product.category]
                if candidate.product_id != product.product_id
            ]
            candidate_negatives.sort(
                key=lambda candidate: (
                    candidate.brand != product.brand,
                    candidate.product_id,
                )
            )
            hard_pool = candidate_negatives[: max(negatives_per_listing * 3, 6)]
            selected = rng.sample(
                hard_pool,
                k=min(negatives_per_listing, len(hard_pool)),
            )
            for negative_index, candidate in enumerate(selected):
                pairs.append(
                    LabelledPair(
                        pair_id=f"{listing_id}--negative-{negative_index}",
                        listing=listing,
                        product=candidate,
                        label=0,
                        is_synthetic=True,
                    )
                )

    return SyntheticEntityResolutionDataset(
        products=products,
        listings=tuple(listings),
        pairs=tuple(pairs),
        seed=seed,
    )


def synthetic_pairs(
    *,
    seed: int = 7,
    product_count: int = 48,
    positive_variants: int = 2,
    negatives_per_listing: int = 2,
) -> tuple[LabelledPair, ...]:
    """Convenience API returning only the explicitly synthetic labelled pairs."""

    return synthetic_catalog(
        seed=seed,
        product_count=product_count,
        positive_variants=positive_variants,
        negatives_per_listing=negatives_per_listing,
    ).pairs


def generate_synthetic_entity_resolution_data(
    **kwargs: int,
) -> SyntheticEntityResolutionDataset:
    """Descriptive alias used by training and smoke-test CLIs."""

    return synthetic_catalog(**kwargs)


def assert_all_synthetic(examples: Sequence[LabelledPair]) -> None:
    """Fail if an engineering fixture accidentally contains undeclared provenance."""

    if not examples or not all(example.is_synthetic for example in examples):
        raise ValueError("expected a non-empty, explicitly synthetic pair dataset")
