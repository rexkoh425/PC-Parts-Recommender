"""Structured pre-retrieval product filters."""

from __future__ import annotations

import math
from collections.abc import Collection
from typing import Any

from .models import IndexedDocument, StructuredFilterSpec

GPU_CATEGORIES = frozenset({"gpu", "gpus", "graphics_card", "graphics_cards"})
MEMORY_CATEGORIES = frozenset({"memory", "ram", "memory_kit", "memory_kits"})
MOTHERBOARD_CATEGORIES = frozenset({"motherboard", "motherboards"})
CASE_CATEGORIES = frozenset({"case", "cases", "chassis"})
AVAILABLE_STOCK_STATES = frozenset(
    {"in_stock", "in stock", "available", "limited_stock", "limited stock"}
)


def _first(document: IndexedDocument, *names: str) -> Any:
    for name in names:
        value = document.get(name)
        if value is not None:
            return value
    return None


def _as_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _normalised(value: Any) -> str:
    return str(value).strip().casefold().replace("-", "_").replace(" ", "_")


def _equals(actual: Any, required: Any) -> bool:
    if isinstance(actual, Collection) and not isinstance(actual, (str, bytes, dict)):
        return any(_equals(item, required) for item in actual)
    if isinstance(actual, str) or isinstance(required, str):
        return _normalised(actual) == _normalised(required)
    return bool(actual == required)


def product_matches_filters(document: IndexedDocument, filters: StructuredFilterSpec) -> bool:
    """Return whether a product satisfies every applicable direct requirement.

    Required fields are strict for the component category they govern: missing
    VRAM on a GPU, for example, fails a minimum-VRAM filter.  The same global
    requirement is ignored for unrelated categories such as CPUs.
    """

    if (
        filters.allowed_product_ids is not None
        and document.product_id not in filters.allowed_product_ids
    ):
        return False

    if document.brand and document.brand.casefold() in filters.excluded_brands:
        return False

    if filters.maximum_price_sgd is not None and (
        document.price_sgd is None or document.price_sgd > filters.maximum_price_sgd
    ):
        return False

    if filters.in_stock_only:
        if document.stock_status is None:
            return False
        if _normalised(document.stock_status) not in {
            _normalised(state) for state in AVAILABLE_STOCK_STATES
        }:
            return False

    if filters.minimum_gpu_vram_gb is not None and document.category in GPU_CATEGORIES:
        vram = _as_number(_first(document, "vram_gb", "gpu_vram_gb", "vram_capacity_gb"))
        if vram is None or vram < filters.minimum_gpu_vram_gb:
            return False

    if filters.minimum_memory_gb is not None and document.category in MEMORY_CATEGORIES:
        capacity = _as_number(_first(document, "capacity_gb", "memory_gb", "total_capacity_gb"))
        if capacity is None or capacity < filters.minimum_memory_gb:
            return False

    memory_scoped = document.category in MEMORY_CATEGORIES | MOTHERBOARD_CATEGORIES
    if filters.required_memory_type is not None and memory_scoped:
        memory_type = _first(document, "memory_type", "supported_memory_type")
        if memory_type is None or not _equals(memory_type, filters.required_memory_type):
            return False

    if filters.wifi_required and document.category in MOTHERBOARD_CATEGORIES:
        wifi = _first(document, "wifi_support", "wifi", "has_wifi")
        if wifi is not True and _normalised(wifi) not in {"yes", "true", "integrated", "included"}:
            return False

    if filters.required_form_factor is not None:
        form_factor: Any = None
        if document.category in MOTHERBOARD_CATEGORIES:
            form_factor = _first(document, "form_factor", "motherboard_form_factor")
        elif document.category in CASE_CATEGORIES:
            form_factor = _first(document, "case_size", "form_factor", "supported_case_size")
        if document.category in MOTHERBOARD_CATEGORIES | CASE_CATEGORIES and (
            form_factor is None or not _equals(form_factor, filters.required_form_factor)
        ):
            return False

    for field_name, required in filters.attribute_equals.items():
        actual = document.get(field_name)
        if actual is None or not _equals(actual, required):
            return False

    for field_name, required in filters.attribute_minimums.items():
        actual = _as_number(document.get(field_name))
        if actual is None or actual < required:
            return False

    return True


def filter_products(
    documents: Collection[IndexedDocument], filters: StructuredFilterSpec
) -> list[IndexedDocument]:
    """Filter a product collection while preserving its input order."""

    return [document for document in documents if product_matches_filters(document, filters)]
