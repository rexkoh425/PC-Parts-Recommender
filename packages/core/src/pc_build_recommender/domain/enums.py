"""Shared enumerations for the PC build recommendation domain."""

from __future__ import annotations

from enum import StrEnum


class ComponentCategory(StrEnum):
    """The eight component categories required by a complete desktop build."""

    CPU = "cpu"
    GPU = "gpu"
    MOTHERBOARD = "motherboard"
    MEMORY = "memory"
    STORAGE = "storage"
    POWER_SUPPLY = "power_supply"
    COOLER = "cooler"
    CPU_COOLER = "cooler"  # Readable alias used by some callers.
    CASE = "case"


class ProductStatus(StrEnum):
    ACTIVE = "active"
    DISCONTINUED = "discontinued"
    UPCOMING = "upcoming"
    UNKNOWN = "unknown"


class StockStatus(StrEnum):
    IN_STOCK = "in_stock"
    OUT_OF_STOCK = "out_of_stock"
    BACKORDER = "backorder"
    PREORDER = "preorder"
    UNKNOWN = "unknown"


class ListingCondition(StrEnum):
    NEW = "new"
    OPEN_BOX = "open_box"
    REFURBISHED = "refurbished"
    USED = "used"
    UNKNOWN = "unknown"


class SourceType(StrEnum):
    MANUFACTURER = "manufacturer"
    RETAILER = "retailer"
    BENCHMARK = "benchmark"
    REVIEW = "review"
    IMPORT = "import"


class MemoryType(StrEnum):
    DDR3 = "ddr3"
    DDR4 = "ddr4"
    DDR5 = "ddr5"


class MotherboardFormFactor(StrEnum):
    MINI_ITX = "mini_itx"
    MICRO_ATX = "micro_atx"
    ATX = "atx"
    E_ATX = "e_atx"


class CaseSize(StrEnum):
    SMALL_FORM_FACTOR = "small_form_factor"
    MINI_TOWER = "mini_tower"
    MID_TOWER = "mid_tower"
    FULL_TOWER = "full_tower"


class StorageInterface(StrEnum):
    SATA = "sata"
    NVME_PCIE = "nvme_pcie"
    PCIE = "pcie"
    USB = "usb"


class StorageFormFactor(StrEnum):
    M2_2230 = "m2_2230"
    M2_2242 = "m2_2242"
    M2_2260 = "m2_2260"
    M2_2280 = "m2_2280"
    M2_22110 = "m2_22110"
    DRIVE_2_5_INCH = "2_5_inch"
    DRIVE_3_5_INCH = "3_5_inch"
    ADD_IN_CARD = "add_in_card"


class PowerSupplyFormFactor(StrEnum):
    ATX = "atx"
    SFX = "sfx"
    SFX_L = "sfx_l"


class EfficiencyRating(StrEnum):
    STANDARD = "80_plus"
    BRONZE = "80_plus_bronze"
    SILVER = "80_plus_silver"
    GOLD = "80_plus_gold"
    PLATINUM = "80_plus_platinum"
    TITANIUM = "80_plus_titanium"


class ModularType(StrEnum):
    NON_MODULAR = "non_modular"
    SEMI_MODULAR = "semi_modular"
    FULLY_MODULAR = "fully_modular"


class CoolerType(StrEnum):
    AIR = "air"
    AIO = "aio"


class WorkloadName(StrEnum):
    GAMING_1080P = "gaming_1080p"
    GAMING_1440P = "gaming_1440p"
    GAMING_4K = "gaming_4k"
    LOCAL_AI = "local_ai"
    SOFTWARE_DEVELOPMENT = "software_development"
    CONTENT_CREATION = "content_creation"


class BenchmarkValueKind(StrEnum):
    OBSERVED = "observed"
    PREDICTED = "predicted"


class CompatVerdict(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"
    UNKNOWN = "unknown"


class BuildProfile(StrEnum):
    BEST_OVERALL = "best_overall"
    BEST_VALUE = "best_value"
    HIGHEST_PERFORMANCE = "highest_performance"
    MOST_UPGRADEABLE = "most_upgradeable"
    LOWEST_POWER = "lowest_power"


class InteractionType(StrEnum):
    SEARCH_SUBMITTED = "search_submitted"
    BUILD_GENERATED = "build_generated"
    BUILD_VIEWED = "build_viewed"
    BUILD_SAVED = "build_saved"
    BUILD_SHARED = "build_shared"
    COMPONENT_VIEWED = "component_viewed"
    COMPONENT_REPLACED = "component_replaced"
    COMPARISON_OPENED = "comparison_opened"
    RETAILER_CLICKED = "retailer_clicked"
    RECOMMENDATION_DISMISSED = "recommendation_dismissed"
    FEEDBACK_SUBMITTED = "feedback_submitted"

