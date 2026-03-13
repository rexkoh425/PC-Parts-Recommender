"""Strict adapters for the existing BuildCores/Dynacore ER pilot inputs."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .candidate_generation import canonical_pc_category
from .records import CanonicalProductRecord, ListingRecord
from .review import SourceUsePolicy

_CATALOGUE_SOURCE = "buildcores_open_db"
_LISTING_SOURCE = "dynacore_controlled_pdf"


@dataclass(frozen=True, slots=True)
class PCWorkflowInputs:
    products: tuple[CanonicalProductRecord, ...]
    listings: tuple[ListingRecord, ...]
    source_policy: SourceUsePolicy


def _read_json_lines(path: Path) -> Iterable[tuple[int, Mapping[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload: Any = json.loads(line)
            if not isinstance(payload, Mapping):
                raise TypeError(f"{path}:{line_number}: expected a JSON object")
            yield line_number, payload


def _boolean(row: Mapping[str, Any], field: str, *, location: str) -> bool:
    value = row.get(field)
    if not isinstance(value, bool):
        raise TypeError(f"{location}: {field} must be a JSON boolean")
    return value


def _catalogue_product(row: Mapping[str, Any], *, location: str) -> CanonicalProductRecord:
    if row.get("record_type") != "canonical_product":
        raise ValueError(f"{location}: expected canonical_product record")
    data = row.get("data")
    if not isinstance(data, Mapping):
        raise TypeError(f"{location}: data must be an object")
    provenance = data.get("provenance")
    if not isinstance(provenance, list) or not provenance:
        raise ValueError(f"{location}: canonical product needs provenance")
    if any(
        not isinstance(item, Mapping) or item.get("source_name") != _CATALOGUE_SOURCE
        for item in provenance
    ):
        raise ValueError(f"{location}: only {_CATALOGUE_SOURCE} catalogue rows are allowed")
    category = canonical_pc_category(str(data.get("category", "")))
    if category is None:
        raise ValueError(f"{location}: unsupported PC category")
    common = data.get("common_attributes")
    category_attributes = data.get("category_attributes")
    if not isinstance(common, Mapping) or not isinstance(category_attributes, Mapping):
        raise TypeError(f"{location}: product attributes must be objects")
    attributes = {**dict(common), **dict(category_attributes)}
    msrp = common.get("msrp_sgd")
    return CanonicalProductRecord(
        product_id=str(data["product_id"]),
        category=category,
        brand=str(data.get("brand", "")),
        model=str(data.get("model", "")),
        canonical_name=str(data["canonical_name"]),
        manufacturer_part_number=(
            str(data["manufacturer_part_number"]) if data.get("manufacturer_part_number") else None
        ),
        gtin=str(data["gtin"]) if data.get("gtin") else None,
        attributes=attributes,
        price_sgd=float(msrp) if msrp is not None else None,
        is_synthetic=False,
    )


def _controlled_listing(row: Mapping[str, Any], *, location: str) -> ListingRecord:
    if row.get("record_type") != "retailer_listing":
        raise ValueError(f"{location}: expected retailer_listing record")
    provenance = row.get("provenance")
    if not isinstance(provenance, Mapping) or provenance.get("source_name") != _LISTING_SOURCE:
        raise ValueError(f"{location}: only {_LISTING_SOURCE} listing rows are allowed")
    data = row.get("data")
    metadata = row.get("normalisation_metadata")
    if not isinstance(data, Mapping) or not isinstance(metadata, Mapping):
        raise TypeError(f"{location}: data and normalisation_metadata must be objects")
    listing = data.get("listing")
    if not isinstance(listing, Mapping):
        raise TypeError(f"{location}: data.listing must be an object")
    category = canonical_pc_category(str(metadata.get("category", "")))
    if category is None:
        raise ValueError(f"{location}: unsupported PC category")
    if str(listing.get("currency", "")).upper() != "SGD":
        raise ValueError(f"{location}: controlled pilot listing must use SGD")
    # Brand and MPN stay missing unless explicitly supplied by the controlled record. We do
    # not convert title-token guesses into authoritative entity-resolution fields.
    return ListingRecord(
        listing_id=str(listing["listing_id"]),
        title=str(listing["title"]),
        category=category,
        brand=str(listing.get("brand", "")),
        manufacturer_part_number=(
            str(listing["manufacturer_part_number"])
            if listing.get("manufacturer_part_number")
            else None
        ),
        attributes={
            "source_record_id": row.get("source_record_id"),
            "source_section": metadata.get("section"),
            "source_variant": metadata.get("variant"),
            "source_confidence_flags": metadata.get("confidence_flags", []),
        },
        current_price_sgd=float(listing["base_price"]),
        retailer=str(listing.get("retailer", "Dynacore")),
        is_synthetic=False,
    )


def load_controlled_pc_workflow_inputs(
    catalogue_jsonl: str | Path,
    listings_jsonl: str | Path,
) -> PCWorkflowInputs:
    """Load only the already-registered BuildCores catalogue and Dynacore pilot offers."""

    catalogue_path = Path(catalogue_jsonl)
    listing_path = Path(listings_jsonl)
    products: list[CanonicalProductRecord] = []
    listings: list[ListingRecord] = []
    catalogue_versions: set[str] = set()
    listing_versions: set[str] = set()
    catalogue_training_flags: list[bool] = []
    listing_training_flags: list[bool] = []
    catalogue_claim_flags: list[bool] = []
    listing_claim_flags: list[bool] = []

    for line_number, row in _read_json_lines(catalogue_path):
        location = f"{catalogue_path}:{line_number}"
        products.append(_catalogue_product(row, location=location))
        catalogue_versions.add(str(row["archive_snapshot_sha256"]))
        catalogue_training_flags.append(_boolean(row, "training_eligible", location=location))
        catalogue_claim_flags.append(_boolean(row, "published_claims_eligible", location=location))
    for line_number, row in _read_json_lines(listing_path):
        location = f"{listing_path}:{line_number}"
        listings.append(_controlled_listing(row, location=location))
        listing_versions.add(str(row["archive_snapshot_sha256"]))
        listing_training_flags.append(_boolean(row, "training_eligible", location=location))
        listing_claim_flags.append(_boolean(row, "published_claims_eligible", location=location))
    if not products or not listings:
        raise ValueError("catalogue and listing inputs must both contain records")
    if len(catalogue_versions) != 1 or len(listing_versions) != 1:
        raise ValueError("each workflow input must contain exactly one source snapshot version")
    if len({product.product_id for product in products}) != len(products):
        raise ValueError("catalogue contains duplicate product_id values")
    if len({listing.listing_id for listing in listings}) != len(listings):
        raise ValueError("controlled offers contain duplicate listing_id values")

    catalogue_version = next(iter(catalogue_versions))
    listing_version = next(iter(listing_versions))
    policy = SourceUsePolicy(
        listing_source=_LISTING_SOURCE,
        catalogue_source=_CATALOGUE_SOURCE,
        data_version=f"buildcores:{catalogue_version}+dynacore:{listing_version}",
        training_eligible=all(catalogue_training_flags) and all(listing_training_flags),
        published_metrics_eligible=all(catalogue_claim_flags) and all(listing_claim_flags),
        scope_note=(
            "Controlled Dynacore offers are development-only. Candidate queues and human "
            "review evidence are allowed, but training and published metric claims remain "
            "disabled unless source permission changes."
        ),
    )
    return PCWorkflowInputs(
        products=tuple(products),
        listings=tuple(listings),
        source_policy=policy,
    )
