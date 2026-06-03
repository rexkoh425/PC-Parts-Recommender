"""Small data-observability assets that also work without Dagster installed.

The pure functions are intentionally dependency-free so unit tests and the API image can import
pipeline definitions without installing the optional ``pipeline`` dependency group.
"""

from __future__ import annotations

import os
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pipelines.assets.operation_observation import instrument_pipeline_operation


def directory_inventory(root: Path) -> dict[str, Any]:
    """Return cheap aggregate metadata for regular files below ``root``."""

    marker_files = {".gitkeep", ".DS_Store", "Thumbs.db"}
    files = (
        sorted(path for path in root.rglob("*") if path.is_file() and path.name not in marker_files)
        if root.exists()
        else []
    )
    suffix_counts = Counter(path.suffix.casefold() or "<none>" for path in files)
    total_bytes = sum(path.stat().st_size for path in files)
    latest_timestamp = max((path.stat().st_mtime for path in files), default=None)
    latest_modified_at = (
        datetime.fromtimestamp(latest_timestamp, tz=UTC).isoformat()
        if latest_timestamp is not None
        else None
    )
    return {
        "root": str(root.resolve()),
        "file_count": len(files),
        "total_bytes": total_bytes,
        "files_by_extension": dict(sorted(suffix_counts.items())),
        "latest_modified_at": latest_modified_at,
    }


def quality_summary(
    raw_inventory: dict[str, Any], processed_inventory: dict[str, Any]
) -> dict[str, Any]:
    """Summarise pipeline readiness without pretending that an empty catalogue is healthy."""

    raw_count = int(raw_inventory.get("file_count", 0))
    processed_count = int(processed_inventory.get("file_count", 0))
    checks = {
        "raw_snapshots_present": raw_count > 0,
        "processed_outputs_present": processed_count > 0,
        "processed_not_without_raw": processed_count == 0 or raw_count > 0,
    }
    status = "pass" if all(checks.values()) else "warning"
    return {
        "status": status,
        "checks": checks,
        "raw_file_count": raw_count,
        "processed_file_count": processed_count,
        "evaluated_at": datetime.now(UTC).isoformat(),
    }


try:
    from dagster import asset
except ModuleNotFoundError as exc:
    if exc.name != "dagster":
        raise
    DAGSTER_AVAILABLE = False
    ASSETS: tuple[object, ...] = ()
else:
    DAGSTER_AVAILABLE = True

    @asset(group_name="data_observability", compute_kind="filesystem")
    @instrument_pipeline_operation("raw_snapshot_inventory")
    def raw_snapshot_inventory(context) -> dict[str, Any]:  # type: ignore[no-untyped-def]
        """Expose raw snapshot volume and freshness to every pipeline run."""

        summary = directory_inventory(Path(os.getenv("RAW_DATA_DIR", "data/raw")))
        context.add_output_metadata(summary)
        return summary

    @asset(group_name="data_observability", compute_kind="filesystem")
    @instrument_pipeline_operation("processed_dataset_inventory")
    def processed_dataset_inventory(context) -> dict[str, Any]:  # type: ignore[no-untyped-def]
        """Expose processed dataset volume and freshness to every pipeline run."""

        summary = directory_inventory(Path(os.getenv("PROCESSED_DATA_DIR", "data/processed")))
        context.add_output_metadata(summary)
        return summary

    @asset(group_name="data_observability")
    @instrument_pipeline_operation("pipeline_data_quality_summary")
    def pipeline_data_quality_summary(  # type: ignore[no-untyped-def]
        context,
        raw_snapshot_inventory: dict[str, Any],
        processed_dataset_inventory: dict[str, Any],
    ) -> dict[str, Any]:
        """Record explicit warnings for empty or disconnected pipeline outputs."""

        summary = quality_summary(raw_snapshot_inventory, processed_dataset_inventory)
        context.add_output_metadata(
            {
                "status": summary["status"],
                "raw_file_count": summary["raw_file_count"],
                "processed_file_count": summary["processed_file_count"],
            }
        )
        return summary

    ASSETS = (
        raw_snapshot_inventory,
        processed_dataset_inventory,
        pipeline_data_quality_summary,
    )
