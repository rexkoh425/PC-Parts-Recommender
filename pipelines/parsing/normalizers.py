"""Strict mappings from source records into the shared domain contracts."""

from __future__ import annotations

import hashlib
import re
from datetime import date, datetime
from typing import Any

from pc_build_recommender.domain.components import (
    CaseAttributes,
    CommonProductAttributes,
    CoolerAttributes,
    CPUAttributes,
    GPUAttributes,
    MemoryAttributes,
    MotherboardAttributes,
    PowerSupplyAttributes,
    StorageAttributes,
)
from pc_build_recommender.domain.enums import (
    CaseSize,
    CoolerType,
    EfficiencyRating,
    MemoryType,
    ModularType,
    MotherboardFormFactor,
    PowerSupplyFormFactor,
    ProductStatus,
    SourceType,
    StorageFormFactor,
    StorageInterface,
)
from pc_build_recommender.domain.models import CanonicalProduct, SourceProvenance
from pipelines.sources.base import RawSnapshot, sha256_bytes

NORMALISED_RECORD_SCHEMA_VERSION = "pc-build-recommender.normalised-record.v1"

BUILDCORES_CATEGORY_MAP = {
    "CPU": "cpu",
    "GPU": "gpu",
    "Motherboard": "motherboard",
    "RAM": "memory",
    "Storage": "storage",
    "PSU": "power_supply",
    "CPUCooler": "cooler",
    "PCCase": "case",
}

_UUID_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


def stable_identifier(prefix: str, *parts: object, length: int = 24) -> str:
    value = "\x1f".join(str(part).strip() for part in parts)
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]
    return f"{prefix}_{digest}"


def _mapping(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _positive_float(value: object) -> float | None:
    try:
        parsed = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _positive_int(value: object) -> int | None:
    parsed = _positive_float(value)
    return int(parsed) if parsed is not None and parsed.is_integer() else None


def _nonnegative_int(value: object) -> int | None:
    if not isinstance(value, (str, bytes, bytearray, int, float)):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _sum_known_nonnegative(mapping: dict[str, Any], keys: tuple[str, ...]) -> int | None:
    values = [_nonnegative_int(mapping.get(key)) for key in keys]
    known_values = [value for value in values if value is not None]
    return sum(known_values) if known_values else None


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalise_memory_type(value: object) -> MemoryType | None:
    text = str(value).strip().lower().replace(" ", "")
    return {
        "ddr3": MemoryType.DDR3,
        "ddr4": MemoryType.DDR4,
        "ddr5": MemoryType.DDR5,
    }.get(text)


def _normalise_motherboard_form_factor(
    value: object,
) -> MotherboardFormFactor | None:
    text = re.sub(r"[^a-z0-9]", "", str(value).lower())
    return {
        "miniitx": MotherboardFormFactor.MINI_ITX,
        "microatx": MotherboardFormFactor.MICRO_ATX,
        "matx": MotherboardFormFactor.MICRO_ATX,
        "atx": MotherboardFormFactor.ATX,
        "eatx": MotherboardFormFactor.E_ATX,
        "extendedatx": MotherboardFormFactor.E_ATX,
    }.get(text)


def _normalise_psu_form_factor(value: object) -> PowerSupplyFormFactor | None:
    text = re.sub(r"[^a-z0-9]", "", str(value).lower())
    if text == "atx":
        return PowerSupplyFormFactor.ATX
    if text == "sfx":
        return PowerSupplyFormFactor.SFX
    if text in {"sfxl", "sfxlarge"}:
        return PowerSupplyFormFactor.SFX_L
    return None


def _normalise_efficiency(value: object) -> EfficiencyRating | None:
    text = re.sub(r"[^a-z0-9]", "", str(value).lower())
    if "titanium" in text:
        return EfficiencyRating.TITANIUM
    if "platinum" in text:
        return EfficiencyRating.PLATINUM
    if "gold" in text:
        return EfficiencyRating.GOLD
    if "silver" in text:
        return EfficiencyRating.SILVER
    if "bronze" in text:
        return EfficiencyRating.BRONZE
    if text in {"80", "80plus", "standard"}:
        return EfficiencyRating.STANDARD
    return None


def _normalise_modularity(value: object) -> ModularType | None:
    text = re.sub(r"[^a-z]", "", str(value).lower())
    if text in {"fullmodular", "fullymodular"}:
        return ModularType.FULLY_MODULAR
    if text == "semimodular":
        return ModularType.SEMI_MODULAR
    if text in {"nonmodular", "fixed"}:
        return ModularType.NON_MODULAR
    return None


def _normalise_storage_interface(record: dict[str, Any]) -> StorageInterface | None:
    text = str(record.get("interface", "")).lower()
    if record.get("nvme") is True or "nvme" in text:
        return StorageInterface.NVME_PCIE
    if "sata" in text:
        return StorageInterface.SATA
    if "pcie" in text or "pci-e" in text:
        return StorageInterface.PCIE
    if "usb" in text:
        return StorageInterface.USB
    return None


def _normalise_storage_form_factor(value: object) -> StorageFormFactor | None:
    text = re.sub(r"[^a-z0-9]", "", str(value).lower())
    mappings = {
        "m22230": StorageFormFactor.M2_2230,
        "m22242": StorageFormFactor.M2_2242,
        "m22260": StorageFormFactor.M2_2260,
        "m22280": StorageFormFactor.M2_2280,
        "m222110": StorageFormFactor.M2_22110,
        "25": StorageFormFactor.DRIVE_2_5_INCH,
        "25inch": StorageFormFactor.DRIVE_2_5_INCH,
        "35": StorageFormFactor.DRIVE_3_5_INCH,
        "35inch": StorageFormFactor.DRIVE_3_5_INCH,
        "addincard": StorageFormFactor.ADD_IN_CARD,
    }
    return mappings.get(text)


def _normalise_case_size(value: object) -> CaseSize | None:
    text = str(value).lower()
    if "full" in text and "tower" in text:
        return CaseSize.FULL_TOWER
    if "mid" in text and "tower" in text:
        return CaseSize.MID_TOWER
    if "mini" in text and "tower" in text:
        return CaseSize.MINI_TOWER
    if any(token in text for token in ("small form", "sff", "mini-itx", "mini itx")):
        return CaseSize.SMALL_FORM_FACTOR
    return None


def _part_number(metadata: dict[str, Any], canonical_name: str) -> str | None:
    values = metadata.get("part_numbers")
    if not isinstance(values, list):
        return None
    compact_name = re.sub(r"\s+", "", canonical_name).casefold()
    for value in values:
        candidate = _optional_text(value)
        if candidate is None or _UUID_PATTERN.fullmatch(candidate):
            continue
        if re.sub(r"\s+", "", candidate).casefold() == compact_name:
            continue
        return candidate
    return None


def _colour(record: dict[str, Any], metadata: dict[str, Any]) -> str | None:
    values = record.get("color")
    if isinstance(values, list) and values:
        return "/".join(str(value).title() for value in values if str(value).strip()) or None
    return _optional_text(metadata.get("manufacturer_color"))


def _build_cpu(record: dict[str, Any]) -> CPUAttributes:
    cores = _mapping(record.get("cores"))
    clocks = _mapping(_mapping(record.get("clocks")).get("performance"))
    specifications = _mapping(record.get("specifications"))
    memory = _mapping(specifications.get("memory"))
    integrated_graphics = _mapping(specifications.get("integratedGraphics"))
    graphics_model = _optional_text(integrated_graphics.get("model"))
    if graphics_model is not None and graphics_model.casefold() in {"none", "n/a"}:
        graphics_model = None
    return CPUAttributes(
        socket=_optional_text(record.get("socket")),
        architecture=_optional_text(record.get("microarchitecture")),
        generation=_optional_text(record.get("series")),
        core_count=_positive_int(cores.get("total")),
        thread_count=_positive_int(cores.get("threads")),
        base_clock_ghz=_positive_float(clocks.get("base")),
        boost_clock_ghz=_positive_float(clocks.get("boost")),
        tdp_watts=_positive_float(specifications.get("tdp")),
        peak_power_watts=_positive_float(specifications.get("ppt")),
        maximum_memory_gb=_positive_int(memory.get("maxSupport")),
        integrated_graphics=graphics_model,
        included_cooler=(
            specifications.get("includesCooler")
            if isinstance(specifications.get("includesCooler"), bool)
            else None
        ),
    )


def _build_gpu(record: dict[str, Any]) -> GPUAttributes:
    connectors = _mapping(record.get("power_connectors"))
    connector_mapping = {
        "6_pin": "pcie_6_pin",
        "8_pin": "pcie_8_pin",
        "12vhpwr": "pcie_12VHPWR",
        "12v_2x6": "pcie_12V_2x6",
    }
    normalised_connectors = {
        target: value
        for target, source in connector_mapping.items()
        if (value := _nonnegative_int(connectors.get(source))) is not None
    }
    return GPUAttributes(
        architecture=_optional_text(record.get("chipset")),
        vram_gb=_positive_int(record.get("memory")),
        vram_type=_optional_text(record.get("memory_type")),
        memory_bus_bits=_positive_int(record.get("memory_bus")),
        compute_unit_count=_positive_int(record.get("core_count")),
        base_clock_mhz=_positive_float(record.get("core_base_clock")),
        boost_clock_mhz=_positive_float(record.get("core_boost_clock")),
        length_mm=_positive_float(record.get("length")),
        height_mm=_positive_float(record.get("height")),
        slot_width=_positive_float(record.get("total_slot_width")),
        board_power_watts=_positive_float(record.get("tdp")),
        recommended_psu_watts=_positive_int(record.get("recommended_psu_wattage")),
        power_connectors=normalised_connectors,
    )


def _build_motherboard(record: dict[str, Any]) -> MotherboardAttributes:
    memory = _mapping(record.get("memory"))
    storage = _mapping(record.get("storage_devices"))
    pcie_slots = record.get("pcie_slots")
    pcie_count = None
    if isinstance(pcie_slots, list):
        pcie_count = sum(
            _nonnegative_int(_mapping(slot).get("quantity")) or 0 for slot in pcie_slots
        )
    m2_slots = record.get("m2_slots")
    wireless = _optional_text(record.get("wireless_networking"))
    return MotherboardAttributes(
        socket=_optional_text(record.get("socket")),
        chipset=_optional_text(record.get("chipset")),
        form_factor=_normalise_motherboard_form_factor(record.get("form_factor")),
        memory_type=_normalise_memory_type(memory.get("ram_type")),
        maximum_memory_gb=_positive_int(memory.get("max")),
        memory_slots=_positive_int(memory.get("slots")),
        pcie_slots=pcie_count,
        m2_slots=len(m2_slots) if isinstance(m2_slots, list) else None,
        sata_ports=_sum_known_nonnegative(storage, ("sata_6_gb_s", "sata_3_gb_s")),
        wifi_support=bool(wireless) if wireless is not None else None,
        bios_version=_optional_text(record.get("bios_version")),
    )


def _build_memory(record: dict[str, Any]) -> MemoryAttributes:
    modules = _mapping(record.get("modules"))
    return MemoryAttributes(
        memory_type=_normalise_memory_type(record.get("ram_type")),
        capacity_gb=_positive_int(record.get("capacity")),
        module_count=_positive_int(modules.get("quantity")),
        speed_mt_s=_positive_int(record.get("speed")),
        cas_latency=_positive_float(record.get("cas_latency")),
        voltage=_positive_float(record.get("voltage")),
        module_height_mm=_positive_float(record.get("height")),
    )


def _build_storage(record: dict[str, Any]) -> StorageAttributes:
    return StorageAttributes(
        capacity_gb=_positive_int(record.get("capacity")),
        interface=_normalise_storage_interface(record),
        form_factor=_normalise_storage_form_factor(record.get("form_factor")),
        sequential_read_mbps=_positive_float(record.get("sequential_read")),
        sequential_write_mbps=_positive_float(record.get("sequential_write")),
        random_read_iops=_positive_float(record.get("random_read")),
        random_write_iops=_positive_float(record.get("random_write")),
        endurance_tbw=_positive_float(record.get("endurance")),
    )


def _build_power_supply(record: dict[str, Any]) -> PowerSupplyAttributes:
    connectors = _mapping(record.get("connectors"))
    connector_mapping = {
        "6_plus_2_pin": "pcie_6_plus_2_pin",
        "12vhpwr": "pcie_12vhpwr",
    }
    pcie_connectors = {
        target: value
        for target, source in connector_mapping.items()
        if (value := _nonnegative_int(connectors.get(source))) is not None
    }
    return PowerSupplyAttributes(
        wattage=_positive_int(record.get("wattage")),
        efficiency_rating=_normalise_efficiency(record.get("efficiency_rating")),
        form_factor=_normalise_psu_form_factor(record.get("form_factor")),
        modular_type=_normalise_modularity(record.get("modular")),
        pcie_connectors=pcie_connectors,
        eps_connectors=_nonnegative_int(connectors.get("eps_8_pin")),
        atx_version=_optional_text(record.get("atx_version")),
        warranty_years=_positive_float(record.get("warranty")),
    )


def _build_cooler(record: dict[str, Any]) -> CoolerAttributes:
    sockets = record.get("cpu_sockets")
    supported_sockets = (
        [str(value).strip() for value in sockets] if isinstance(sockets, list) else []
    )
    supported_sockets = list(dict.fromkeys(value for value in supported_sockets if value))
    water_cooled = record.get("water_cooled")
    cooler_type = None
    if water_cooled is True:
        cooler_type = CoolerType.AIO
    elif water_cooled is False:
        cooler_type = CoolerType.AIR
    return CoolerAttributes(
        cooler_type=cooler_type,
        supported_sockets=supported_sockets,
        height_mm=_positive_float(record.get("height")),
        radiator_size_mm=_positive_int(record.get("radiator_size")),
        fan_count=_positive_int(record.get("fan_quantity")),
        estimated_cooling_capacity_watts=_positive_float(record.get("cooling_capacity")),
    )


def _build_case(record: dict[str, Any]) -> CaseAttributes:
    motherboard_values = record.get("supported_motherboard_form_factors")
    motherboard_sizes: list[MotherboardFormFactor] = []
    if isinstance(motherboard_values, list):
        motherboard_sizes = [
            motherboard_size
            for raw_value in motherboard_values
            if (
                motherboard_size := _normalise_motherboard_form_factor(raw_value)
            )
            is not None
        ]
    motherboard_sizes = list(dict.fromkeys(motherboard_sizes))
    psu_values = record.get("supported_power_supply_form_factors")
    psu_sizes: list[PowerSupplyFormFactor] = []
    if isinstance(psu_values, list):
        psu_sizes = [
            psu_size
            for raw_value in psu_values
            if (psu_size := _normalise_psu_form_factor(raw_value)) is not None
        ]
    psu_sizes = list(dict.fromkeys(psu_sizes))
    radiators = record.get("radiator_support")
    radiator_support: list[int] = []
    if isinstance(radiators, list):
        radiator_support = [
            radiator_size
            for raw_value in radiators
            if (radiator_size := _positive_int(raw_value)) is not None
        ]
    return CaseAttributes(
        case_size=_normalise_case_size(record.get("form_factor")),
        supported_motherboard_sizes=motherboard_sizes,
        maximum_gpu_length_mm=_positive_float(record.get("max_video_card_length")),
        maximum_gpu_slot_width=_positive_float(record.get("max_video_card_slot_width")),
        maximum_cooler_height_mm=_positive_float(record.get("max_cpu_cooler_height")),
        supported_psu_sizes=psu_sizes,
        radiator_support_mm=list(dict.fromkeys(radiator_support)),
        drive_bays=_sum_known_nonnegative(
            record, ("internal_3_5_bays", "internal_2_5_bays")
        ),
        included_fans=_nonnegative_int(record.get("included_fans")),
    )


_ATTRIBUTE_BUILDERS = {
    "CPU": _build_cpu,
    "GPU": _build_gpu,
    "Motherboard": _build_motherboard,
    "RAM": _build_memory,
    "Storage": _build_storage,
    "PSU": _build_power_supply,
    "CPUCooler": _build_cooler,
    "PCCase": _build_case,
}


def _last_verified(metadata: dict[str, Any]) -> datetime | None:
    value = _optional_text(metadata.get("last_manually_spec_verified_at"))
    if value is None:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else None


def normalise_buildcores_product(
    *,
    record: dict[str, Any],
    source_category: str,
    source_record_path: str,
    raw_record_bytes: bytes,
    snapshot: RawSnapshot,
    commit: str,
) -> dict[str, Any]:
    """Map one BuildCores record and validate it through ``CanonicalProduct``."""

    if source_category not in BUILDCORES_CATEGORY_MAP:
        raise ValueError(f"unsupported BuildCores category: {source_category}")
    opendb_id = _optional_text(record.get("opendb_id"))
    if opendb_id is None:
        raise ValueError("missing opendb_id")
    metadata = _mapping(record.get("metadata"))
    canonical_name = _optional_text(metadata.get("name"))
    brand = _optional_text(metadata.get("manufacturer"))
    if canonical_name is None or brand is None:
        raise ValueError("metadata.name and metadata.manufacturer are required")
    model = (
        _optional_text(metadata.get("variant"))
        or _optional_text(metadata.get("series"))
        or canonical_name
    )
    release_year = _positive_int(metadata.get("releaseYear"))
    if release_year is not None and not 1970 <= release_year <= date.today().year + 2:
        release_year = None
    general_information = _mapping(record.get("general_product_information"))
    manufacturer_url = _optional_text(general_information.get("manufacturer_url"))
    raw_url = (
        "https://raw.githubusercontent.com/buildcores/buildcores-open-db/"
        f"{commit}/open-db/{source_category}/{PathLike(source_record_path).name}"
    )
    verified_at = _last_verified(metadata)
    confidence = 0.95 if verified_at is not None else 0.90 if manufacturer_url else 0.80
    product_id = f"prod_buildcores_{opendb_id}"
    provenance = SourceProvenance(
        provenance_id=f"src_buildcores_{opendb_id}",
        product_id=product_id,
        source_name="buildcores_open_db",
        source_url=manufacturer_url or raw_url,
        source_type=SourceType.IMPORT,
        retrieved_at=snapshot.retrieved_at,
        raw_content_hash=sha256_bytes(raw_record_bytes),
        parser_version=snapshot.parser_version,
        licence_or_access_note=snapshot.licence_or_access_note,
        last_verified_at=verified_at,
        extraction_confidence=confidence,
    )
    dimensions = _mapping(record.get("dimensions_mm"))
    series = _optional_text(metadata.get("series"))
    common_attributes = CommonProductAttributes(
        width_mm=_positive_float(dimensions.get("width")),
        height_mm=_positive_float(dimensions.get("height")),
        depth_mm=_positive_float(dimensions.get("depth")),
        colour=_colour(record, metadata),
        tags=list(dict.fromkeys(tag for tag in (series, source_category) if tag)),
    )
    category_attributes = _ATTRIBUTE_BUILDERS[source_category](record)
    product = CanonicalProduct(
        product_id=product_id,
        category=BUILDCORES_CATEGORY_MAP[source_category],
        brand=brand,
        model=model,
        manufacturer_part_number=_part_number(metadata, canonical_name),
        gtin=None,
        canonical_name=canonical_name,
        release_date=date(release_year, 1, 1) if release_year is not None else None,
        status=ProductStatus.ACTIVE,
        common_attributes=common_attributes,
        category_attributes=category_attributes,
        source_confidence=confidence,
        provenance=[provenance],
        created_at=snapshot.retrieved_at,
        updated_at=snapshot.retrieved_at,
    )
    return {
        "schema_version": NORMALISED_RECORD_SCHEMA_VERSION,
        "record_type": "canonical_product",
        "source_record_id": opendb_id,
        "source_record_path": source_record_path,
        "archive_snapshot_sha256": snapshot.content_sha256,
        "training_eligible": True,
        "published_claims_eligible": True,
        "normalisation_metadata": {
            "source_category": source_category,
            "release_date_precision": "year" if release_year is not None else None,
            "manufacturer_url_present": manufacturer_url is not None,
        },
        "data": product.model_dump(mode="json"),
    }


class PathLike:
    """Tiny POSIX/Windows filename helper without resolving untrusted archive paths."""

    def __init__(self, value: str) -> None:
        self.value = value

    @property
    def name(self) -> str:
        return self.value.replace("\\", "/").rsplit("/", maxsplit=1)[-1]
