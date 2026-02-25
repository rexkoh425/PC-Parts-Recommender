"""Normalisation helpers for dictionary-shaped component records."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Final

type Component = Mapping[str, Any]
MISSING: Final = object()

_NESTED_ATTRIBUTE_KEYS = (
    "category_attributes",
    "common_attributes",
    "attributes",
    "specifications",
    "specs",
)


def _normalise_key(value: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).casefold())


def lookup(component: Component, *aliases: str, default: Any = MISSING) -> Any:
    """Read an attribute by tolerant key matching from supported attribute containers."""

    alias_keys = {_normalise_key(alias) for alias in aliases}
    containers: list[Mapping[str, Any]] = [component]
    for nested_key in _NESTED_ATTRIBUTE_KEYS:
        nested = _direct_lookup(component, nested_key)
        if isinstance(nested, Mapping):
            containers.append(nested)

    for container in containers:
        for key, value in container.items():
            if _normalise_key(key) in alias_keys:
                return value
    return default


def _direct_lookup(component: Mapping[str, Any], alias: str) -> Any:
    wanted = _normalise_key(alias)
    for key, value in component.items():
        if _normalise_key(key) == wanted:
            return value
    return MISSING


def is_missing(value: Any) -> bool:
    return value is MISSING or value is None or (isinstance(value, str) and not value.strip())


def number(value: Any) -> float | None:
    """Coerce a scalar measurement such as ``"335 mm"`` to a finite float."""

    if is_missing(value) or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        numeric = float(value)
    elif isinstance(value, str):
        match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", value.replace(",", ""))
        if match is None:
            return None
        numeric = float(match.group())
    else:
        return None
    return numeric if math.isfinite(numeric) else None


def integer(value: Any) -> int | None:
    numeric = number(value)
    if numeric is None or numeric < 0 or not numeric.is_integer():
        return None
    return int(numeric)


def token(value: Any) -> str | None:
    if is_missing(value):
        return None
    normalised = re.sub(r"[^a-z0-9]+", "", str(value).casefold())
    return normalised or None


def tokens(value: Any) -> set[str] | None:
    """Convert an explicit scalar/sequence/mapping of supported values to normalised tokens."""

    if is_missing(value):
        return None
    if isinstance(value, Mapping):
        items = [key for key, enabled in value.items() if bool(enabled)]
    elif isinstance(value, str):
        items = re.split(r"[,;|]", value)
    elif isinstance(value, Iterable):
        items = list(value)
    else:
        items = [value]
    return {normalised for item in items if (normalised := token(item)) is not None}


def form_factor(value: Any) -> str | None:
    normalised = token(value)
    if normalised is None:
        return None
    aliases = {
        "microatx": "microatx",
        "matx": "microatx",
        "uatx": "microatx",
        "miniitx": "miniitx",
        "mitx": "miniitx",
        "extendedatx": "eatx",
        "eatx": "eatx",
        "atx": "atx",
        "ssi": "ssi",
        "ssiceb": "ssiceb",
        "ssieeb": "ssieeb",
    }
    return aliases.get(normalised, normalised)


def form_factors(value: Any) -> set[str] | None:
    raw = tokens(value)
    if raw is None:
        return None
    return {normalised for item in raw if (normalised := form_factor(item)) is not None}


def storage_interface(value: Any) -> str | None:
    normalised = token(value)
    if normalised is None:
        return None
    if "nvme" in normalised or normalised.startswith("m2pcie"):
        return "m2_nvme"
    if "sata" in normalised and ("m2" in normalised or "m.2" in str(value).casefold()):
        return "m2_sata"
    if "sata" in normalised:
        return "sata"
    if "pcie" in normalised:
        return "pcie"
    return normalised


def storage_interfaces(value: Any) -> set[str] | None:
    if is_missing(value):
        return None
    if isinstance(value, Mapping):
        items = [key for key, enabled in value.items() if bool(enabled)]
    elif isinstance(value, str):
        items = re.split(r"[,;|]", value)
    elif isinstance(value, Iterable):
        items = list(value)
    else:
        items = [value]
    return {normalised for item in items if (normalised := storage_interface(item)) is not None}


def _connector_name(value: object, *, family: str | None = None) -> str | None:
    raw = str(value).casefold()
    normalised = token(value)
    if normalised is None:
        return None
    if "12v2x6" in normalised:
        return "12v_2x6"
    if "12vhpwr" in normalised or "16pin" in normalised:
        return "12vhpwr"
    if "eps" in normalised or "cpu" in normalised or family == "eps":
        if (
            "4plus4" in normalised
            or "4+4" in raw
            or "44pin" in normalised
            or "8pin" in normalised
            or normalised in {"8", "8p"}
        ):
            return "eps_8_pin"
        if "4pin" in normalised or normalised in {"4", "4p"}:
            return "eps_4_pin"
    if (
        "6plus2" in normalised
        or "6+2" in raw
        or "62pin" in normalised
        or "8pin" in normalised
        or normalised in {"8", "8p"}
    ):
        return "pcie_8_pin"
    if "6pin" in normalised or normalised in {"6", "6p"}:
        return "pcie_6_pin"
    # Preserve an explicit vendor connector rather than pretending it is absent.
    if any(marker in raw for marker in ("pin", "pcie", "eps")):
        return normalised
    return None


def connector_inventory(
    value: Any, *, family: str | None = None
) -> tuple[Counter[str] | None, tuple[str, ...]]:
    """Parse connectors while retaining any entries that could not be interpreted."""

    if is_missing(value):
        return None, ()
    result: Counter[str] = Counter()
    unknown_entries: list[str] = []
    if isinstance(value, Mapping):
        for raw_name, raw_count in value.items():
            name = _connector_name(raw_name, family=family)
            count = integer(raw_count)
            if name is not None and count is not None:
                result[name] += count
            else:
                unknown_entries.append(f"{raw_name}={raw_count}")
        return result, tuple(unknown_entries)

    if isinstance(value, int) and not isinstance(value, bool):
        default_name = "eps_8_pin" if family == "eps" else "pcie_8_pin"
        if value >= 0:
            result[default_name] = value
        else:
            unknown_entries.append(str(value))
        return result, tuple(unknown_entries)

    values: Sequence[Any]
    if isinstance(value, str):
        values = re.split(r"[,;|]", value)
    elif isinstance(value, Sequence):
        values = value
    else:
        values = [value]
    for raw_item in values:
        item = str(raw_item).strip()
        multiplier_match = re.match(r"^(\d+)\s*[x×]\s*(.+)$", item, flags=re.IGNORECASE)
        count = int(multiplier_match.group(1)) if multiplier_match else 1
        name_text = multiplier_match.group(2) if multiplier_match else item
        name = _connector_name(name_text, family=family)
        if name is not None:
            result[name] += count
        elif item:
            unknown_entries.append(item)
    return result, tuple(unknown_entries)


def connector_counts(value: Any, *, family: str | None = None) -> Counter[str] | None:
    """Backward-compatible connector-count view used by lower-level callers."""

    inventory, _ = connector_inventory(value, family=family)
    return inventory


def product_identity(component: Component) -> str | None:
    value = lookup(
        component,
        "product_id",
        "id",
        "manufacturer_part_number",
        "mpn",
        "sku",
        default=MISSING,
    )
    if is_missing(value):
        return None
    return str(value).strip().casefold()


def source_evidence(component: Component) -> dict[str, Any]:
    evidence: dict[str, Any] = {}
    for output_key, aliases in {
        "product_id": ("product_id", "id"),
        "manufacturer_part_number": ("manufacturer_part_number", "mpn"),
        "source_url": ("source_url", "manufacturer_url"),
        "source": ("source", "evidence_source", "source_name"),
    }.items():
        value = lookup(component, *aliases, default=MISSING)
        if not is_missing(value):
            evidence[output_key] = value
    return evidence
