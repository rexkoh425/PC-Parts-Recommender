"""Deterministic, versioned compatibility rules for complete PC builds."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Final

from ._access import (
    MISSING,
    Component,
    connector_inventory,
    form_factor,
    form_factors,
    integer,
    is_missing,
    lookup,
    number,
    product_identity,
    source_evidence,
    storage_interface,
    storage_interfaces,
    token,
    tokens,
)
from .models import CompatibilityReport, CompatibilityResult, CompatVerdict, PowerPolicy

DEFAULT_RULE_VERSION: Final = "compat_v2"
COMPATIBILITY_AUTHORITY_KEY: Final = "_compatibility_authority"
AUTHORITATIVE_COMPATIBILITY_POLICY: Final = "authoritative_only"
CONTROLLED_NON_PRODUCTION_POLICY: Final = "controlled_non_production"
DEFAULT_REQUIRED_CATEGORIES: Final = (
    "cpu",
    "gpu",
    "motherboard",
    "memory",
    "storage",
    "power_supply",
    "cooler",
    "case",
)

_CATEGORY_ALIASES = {
    "processor": "cpu",
    "graphics": "gpu",
    "graphics_card": "gpu",
    "video_card": "gpu",
    "mainboard": "motherboard",
    "mother_board": "motherboard",
    "ram": "memory",
    "memory_kit": "memory",
    "ssd": "storage",
    "hdd": "storage",
    "power_supply_unit": "power_supply",
    "psu": "power_supply",
    "cpu_cooler": "cooler",
    "chassis": "case",
}


def _category(value: object) -> str:
    raw = re.sub(r"[^a-z0-9]+", "_", str(value).casefold()).strip("_")
    return _CATEGORY_ALIASES.get(raw, raw)

# TODO: rest of this module still to come.
