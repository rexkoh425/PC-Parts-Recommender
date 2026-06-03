"""Dagster assets and dependency-free pipeline helpers."""

from .ingestion import ASSETS as INGESTION_ASSETS
from .ingestion import persist_checked_batch
from .observability import (
    ASSETS as OBSERVABILITY_ASSETS,
)
from .observability import (
    DAGSTER_AVAILABLE,
    directory_inventory,
    quality_summary,
)
from .retention import ASSETS as RETENTION_ASSETS

ASSETS = (*INGESTION_ASSETS, *OBSERVABILITY_ASSETS, *RETENTION_ASSETS)

__all__ = [
    "ASSETS",
    "DAGSTER_AVAILABLE",
    "directory_inventory",
    "persist_checked_batch",
    "quality_summary",
]
