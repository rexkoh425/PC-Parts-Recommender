"""Machine-readable source-use rights shared by ingestion and serving gates."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from typing import Any


class DataUse(StrEnum):
    DISPLAY = "display"
    CACHE = "cache"
    STORE_HISTORY = "store_history"
    REDISTRIBUTE = "redistribute"
    EMBED = "embed"
    TRAIN = "train"
    DERIVE = "derive"

    @property
    def field_name(self) -> str:
        return f"may_{self.value}"


PRODUCTION_CATALOG_USES = (
    DataUse.DISPLAY,
    DataUse.CACHE,
    DataUse.STORE_HISTORY,
    DataUse.DERIVE,
)

# TODO: rest of this module still to come.
