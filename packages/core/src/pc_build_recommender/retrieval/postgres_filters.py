"""Shared PostgreSQL predicates for retrieval-time structured constraints."""

from __future__ import annotations

from collections.abc import Collection
from decimal import Decimal
from typing import Any

from sqlalchemy import Float, String, case, cast, exists, false, func, select
from sqlalchemy.sql.elements import ColumnElement

from pc_build_recommender.catalog.orm import (
    CanonicalProductRecord,
    RetailerListingRecord,
)

from .models import StructuredFilters

AVAILABLE_STOCK_VALUES = ("in_stock",)
CATEGORY_ALIASES = {
    "cases": "case",
    "cpu_cooler": "cooler",
    "gpus": "gpu",
    "memory_kit": "memory",
    "memory_kits": "memory",
    "motherboards": "motherboard",
    "psu": "power_supply",
    "ram": "memory",
}


def normalize_postgres_category(category: str) -> str:
    key = category.strip().casefold()
    return CATEGORY_ALIASES.get(key, key)


def _listing_conditions(*, in_stock_only: bool) -> list[ColumnElement[bool]]:
    conditions: list[ColumnElement[bool]] = [
        RetailerListingRecord.product_id == CanonicalProductRecord.product_id,
        RetailerListingRecord.condition == "new",
        RetailerListingRecord.currency == "SGD",
    ]
    if in_stock_only:
        conditions.append(RetailerListingRecord.stock_status.in_(AVAILABLE_STOCK_VALUES))
    return conditions


def cheapest_price_expression(*, in_stock_only: bool) -> ColumnElement[Decimal]:
    """Return a correlated cheapest-listing scalar used by filters and hydration."""

    total_price = RetailerListingRecord.base_price + RetailerListingRecord.shipping_price
    availability_order = [] if in_stock_only else [
        case(
            (RetailerListingRecord.stock_status.in_(AVAILABLE_STOCK_VALUES), 0),
            else_=1,
        )
    ]
    return (
        select(total_price)
        .where(*_listing_conditions(in_stock_only=in_stock_only))
        .order_by(
            *availability_order,
            total_price,
            RetailerListingRecord.listing_id,
        )
        .limit(1)
        .correlate(CanonicalProductRecord)
        .scalar_subquery()
    )


def cheapest_stock_expression(*, in_stock_only: bool) -> ColumnElement[str | None]:
    total_price = RetailerListingRecord.base_price + RetailerListingRecord.shipping_price
    availability_order = [] if in_stock_only else [
        case(
            (RetailerListingRecord.stock_status.in_(AVAILABLE_STOCK_VALUES), 0),
            else_=1,
        )
    ]
    return (
        select(RetailerListingRecord.stock_status)
        .where(*_listing_conditions(in_stock_only=in_stock_only))
        .order_by(
            *availability_order,
            total_price,
            RetailerListingRecord.listing_id,
        )
        .limit(1)
        .correlate(CanonicalProductRecord)
        .scalar_subquery()
    )


def _json_text(field_name: str) -> ColumnElement[str | None]:
    path = tuple(part for part in field_name.split(".") if part)
    if not path:
        raise ValueError("attribute field names must not be empty")
    category_value = CanonicalProductRecord.category_attributes[path].as_string()
    common_value = CanonicalProductRecord.common_attributes[path].as_string()
    return func.coalesce(category_value, common_value)


def _normalized_text(value: ColumnElement[str | None]) -> ColumnElement[str]:
    return func.lower(
        func.replace(func.replace(cast(value, String), "-", "_"), " ", "_")
    )


def _safe_number(value: ColumnElement[str | None]) -> ColumnElement[float | None]:
    numeric_pattern = r"^[+-]?([0-9]+([.][0-9]*)?|[.][0-9]+)$"
    return case(
        (cast(value, String).op("~")(numeric_pattern), cast(value, Float)),
        else_=None,
    )


def _attribute_equals(field_name: str, required: Any) -> ColumnElement[bool]:
    value = _json_text(field_name)
    if isinstance(required, bool):
        return _normalized_text(value).in_(("true", "yes", "included", "integrated")) \
            if required else _normalized_text(value).in_(("false", "no"))
    if isinstance(required, (int, float)) and not isinstance(required, bool):
        return _safe_number(value) == float(required)
    if isinstance(required, Collection) and not isinstance(required, (str, bytes, dict)):
        raise ValueError("attribute_equals values must be scalar")
    normalized = str(required).strip().casefold().replace("-", "_").replace(" ", "_")
    return _normalized_text(value) == normalized


def postgres_structured_predicates(
    *,
    category: str,
    filters: StructuredFilters | None,
    candidate_ids: set[str] | frozenset[str] | None = None,
) -> list[ColumnElement[bool]]:
    """Compile direct requirements into fail-closed SQL predicates."""

    category_key = normalize_postgres_category(category)
    predicates: list[ColumnElement[bool]] = [
        CanonicalProductRecord.category == category_key,
        CanonicalProductRecord.status == "active",
    ]
    if candidate_ids is not None:
        if not candidate_ids:
            predicates.append(false())
        else:
            predicates.append(CanonicalProductRecord.product_id.in_(candidate_ids))
    if filters is None:
        return predicates
    if filters.allowed_product_ids is not None:
        if not filters.allowed_product_ids:
            predicates.append(false())
        else:
            predicates.append(
                CanonicalProductRecord.product_id.in_(filters.allowed_product_ids)
            )
    if filters.excluded_brands:
        predicates.append(
            func.lower(CanonicalProductRecord.brand).not_in(
                tuple(sorted(filters.excluded_brands))
            )
        )
    if filters.in_stock_only:
        predicates.append(exists().where(*_listing_conditions(in_stock_only=True)))
    if filters.maximum_price_sgd is not None:
        predicates.append(
            cheapest_price_expression(in_stock_only=filters.in_stock_only)
            <= filters.maximum_price_sgd
        )
    if filters.minimum_gpu_vram_gb is not None and category_key == "gpu":
        predicates.append(
            _safe_number(_json_text("vram_gb")) >= filters.minimum_gpu_vram_gb
        )
    if filters.minimum_memory_gb is not None and category_key == "memory":
        predicates.append(
            _safe_number(_json_text("capacity_gb")) >= filters.minimum_memory_gb
        )
    if filters.required_memory_type is not None and category_key in {
        "memory",
        "motherboard",
    }:
        predicates.append(
            _attribute_equals("memory_type", filters.required_memory_type)
        )
    if filters.wifi_required and category_key == "motherboard":
        predicates.append(_attribute_equals("wifi_support", True))
    if filters.required_form_factor is not None and category_key in {
        "case",
        "motherboard",
    }:
        field_name = "case_size" if category_key == "case" else "form_factor"
        predicates.append(_attribute_equals(field_name, filters.required_form_factor))
    predicates.extend(
        _attribute_equals(field_name, required)
        for field_name, required in sorted(filters.attribute_equals.items())
    )
    predicates.extend(
        _safe_number(_json_text(field_name)) >= required
        for field_name, required in sorted(filters.attribute_minimums.items())
    )
    return predicates
