"""Deterministic canonical-product identity preflight.

The preflight is deliberately diagnostic: it reports every source row and never
merges or drops products. Production publication and persistence can then fail
closed until ambiguous identities have been reviewed.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from pc_build_recommender.domain import CanonicalProduct
from pc_build_recommender.entity_resolution.normalization import normalize_identifier

CANONICAL_IDENTITY_PREFLIGHT_SCHEMA_VERSION = (
    "pc-build-recommender.canonical-identity-preflight.v1"
)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


@dataclass(frozen=True, slots=True, order=True)
class CanonicalIdentityMember:
    """One source product represented in an identity finding."""

    product_id: str
    category: str
    brand: str
    manufacturer_part_number: str | None

    @classmethod
    def from_product(cls, product: CanonicalProduct) -> CanonicalIdentityMember:
        return cls(
            product_id=product.product_id,
            category=product.category.value,
            brand=product.brand,
            manufacturer_part_number=product.manufacturer_part_number,
        )

    def to_dict(self) -> dict[str, str | None]:
        return {
            "product_id": self.product_id,
            "category": self.category,
            "brand": self.brand,
            "manufacturer_part_number": self.manufacturer_part_number,
        }


@dataclass(frozen=True, slots=True)
class CanonicalIdentityConflictGroup:
    """Products sharing one normalized identity key."""

    normalized_brand: str
    normalized_manufacturer_part_number: str
    products: tuple[CanonicalIdentityMember, ...]

    def __post_init__(self) -> None:
        if len(self.products) < 2:
            raise ValueError("an identity conflict group requires at least two products")
        if not self.normalized_brand or not self.normalized_manufacturer_part_number:
            raise ValueError("an identity conflict group requires a complete normalized key")

    def to_dict(self) -> dict[str, Any]:
        return {
            "normalized_brand": self.normalized_brand,
            "normalized_manufacturer_part_number": (
                self.normalized_manufacturer_part_number
            ),
            "product_count": len(self.products),
            "products": [product.to_dict() for product in self.products],
            "required_resolution": "manual_identity_review",
        }


@dataclass(frozen=True, slots=True)
class DuplicateProductIdGroup:
    """Repeated product IDs retained as a reportable source-data conflict."""

    product_id: str
    products: tuple[CanonicalIdentityMember, ...]

    def __post_init__(self) -> None:
        if len(self.products) < 2:
            raise ValueError("a duplicate product ID group requires at least two products")

    def to_dict(self) -> dict[str, Any]:
        return {
            "product_id": self.product_id,
            "record_count": len(self.products),
            "products": [product.to_dict() for product in self.products],
            "required_resolution": "manual_identity_review",
        }


@dataclass(frozen=True, slots=True)
class CanonicalIdentityPreflightReport:
    """Complete, deterministic identity findings for one canonical catalogue."""

    record_count: int
    complete_identity_count: int
    missing_brand_products: tuple[CanonicalIdentityMember, ...]
    missing_mpn_products: tuple[CanonicalIdentityMember, ...]
    duplicate_identity_groups: tuple[CanonicalIdentityConflictGroup, ...]
    duplicate_product_id_groups: tuple[DuplicateProductIdGroup, ...]

    @property
    def duplicate_identity_product_count(self) -> int:
        return sum(len(group.products) for group in self.duplicate_identity_groups)

    @property
    def duplicate_product_id_record_count(self) -> int:
        return sum(len(group.products) for group in self.duplicate_product_id_groups)

    @property
    def production_ready(self) -> bool:
        return not self.blockers()

    def blockers(self) -> tuple[str, ...]:
        blockers: list[str] = []
        if self.missing_brand_products:
            blockers.append(
                f"{len(self.missing_brand_products)} canonical products lack a normalized brand"
            )
        if self.missing_mpn_products:
            blockers.append(
                f"{len(self.missing_mpn_products)} canonical products lack a normalized "
                "manufacturer part number"
            )
        if self.duplicate_identity_groups:
            blockers.append(
                f"{len(self.duplicate_identity_groups)} normalized brand/MPN identities "
                f"conflict across {self.duplicate_identity_product_count} canonical products"
            )
        if self.duplicate_product_id_groups:
            blockers.append(
                f"{len(self.duplicate_product_id_groups)} product IDs are duplicated across "
                f"{self.duplicate_product_id_record_count} source records"
            )
        return tuple(blockers)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": CANONICAL_IDENTITY_PREFLIGHT_SCHEMA_VERSION,
            "record_count": self.record_count,
            "complete_identity_count": self.complete_identity_count,
            "missing_brand_count": len(self.missing_brand_products),
            "missing_mpn_count": len(self.missing_mpn_products),
            "duplicate_identity_group_count": len(self.duplicate_identity_groups),
            "duplicate_identity_product_count": self.duplicate_identity_product_count,
            "duplicate_product_id_group_count": len(self.duplicate_product_id_groups),
            "duplicate_product_id_record_count": self.duplicate_product_id_record_count,
            "missing_brand_products": [
                product.to_dict() for product in self.missing_brand_products
            ],
            "missing_mpn_products": [
                product.to_dict() for product in self.missing_mpn_products
            ],
            "duplicate_identity_groups": [
                group.to_dict() for group in self.duplicate_identity_groups
            ],
            "duplicate_product_id_groups": [
                group.to_dict() for group in self.duplicate_product_id_groups
            ],
            "production_ready": self.production_ready,
            "production_blockers": list(self.blockers()),
            "resolution_policy": (
                "retain_all_source_rows_and_require_manual_review; "
                "never_auto_merge_or_drop"
            ),
        }
        payload["content_sha256"] = hashlib.sha256(_canonical_json(payload)).hexdigest()
        return payload


class CanonicalIdentityImportError(RuntimeError):
    """Raised before persistence when canonical identities are ambiguous."""

    def __init__(self, report: CanonicalIdentityPreflightReport) -> None:
        super().__init__(
            "canonical identity preflight blocked database import: "
            + "; ".join(report.blockers())
        )
        self.report = report


def audit_canonical_product_identities(
    products: Iterable[CanonicalProduct],
) -> CanonicalIdentityPreflightReport:
    """Audit normalized brand/MPN identity without mutating the source catalogue."""

    identity_groups: dict[tuple[str, str], list[CanonicalIdentityMember]] = defaultdict(list)
    product_id_groups: dict[str, list[CanonicalIdentityMember]] = defaultdict(list)
    missing_brand: list[CanonicalIdentityMember] = []
    missing_mpn: list[CanonicalIdentityMember] = []
    record_count = 0
    complete_identity_count = 0

    for product in products:
        record_count += 1
        member = CanonicalIdentityMember.from_product(product)
        product_id_groups[member.product_id].append(member)
        brand = normalize_identifier(member.brand)
        mpn = normalize_identifier(member.manufacturer_part_number)
        if not brand:
            missing_brand.append(member)
        if not mpn:
            missing_mpn.append(member)
        if brand and mpn:
            complete_identity_count += 1
            identity_groups[(brand, mpn)].append(member)

    duplicate_identities = tuple(
        CanonicalIdentityConflictGroup(
            normalized_brand=brand,
            normalized_manufacturer_part_number=mpn,
            products=tuple(sorted(members)),
        )
        for (brand, mpn), members in sorted(identity_groups.items())
        if len(members) > 1
    )
    duplicate_product_ids = tuple(
        DuplicateProductIdGroup(
            product_id=product_id,
            products=tuple(sorted(members)),
        )
        for product_id, members in sorted(product_id_groups.items())
        if len(members) > 1
    )
    return CanonicalIdentityPreflightReport(
        record_count=record_count,
        complete_identity_count=complete_identity_count,
        missing_brand_products=tuple(sorted(missing_brand)),
        missing_mpn_products=tuple(sorted(missing_mpn)),
        duplicate_identity_groups=duplicate_identities,
        duplicate_product_id_groups=duplicate_product_ids,
    )


def audit_canonical_envelopes(
    envelopes: Iterable[Mapping[str, Any]],
) -> CanonicalIdentityPreflightReport:
    """Validate and audit normalized canonical-product envelopes."""

    products: list[CanonicalProduct] = []
    for envelope in envelopes:
        if envelope.get("record_type") != "canonical_product":
            raise ValueError("identity preflight requires canonical_product records")
        data = envelope.get("data")
        if not isinstance(data, Mapping):
            raise ValueError("canonical_product identity preflight requires a data object")
        products.append(CanonicalProduct.model_validate(data))
    return audit_canonical_product_identities(products)


__all__ = [
    "CANONICAL_IDENTITY_PREFLIGHT_SCHEMA_VERSION",
    "CanonicalIdentityConflictGroup",
    "CanonicalIdentityImportError",
    "CanonicalIdentityMember",
    "CanonicalIdentityPreflightReport",
    "DuplicateProductIdGroup",
    "audit_canonical_envelopes",
    "audit_canonical_product_identities",
]
