"""Scheduled governed-web retention with registry-derived source coverage."""

from __future__ import annotations

import os
from datetime import timedelta
from pathlib import Path
from typing import Any

from pipelines.assets.operation_observation import instrument_pipeline_operation
from pipelines.retention.registry import (
    GOVERNED_WEB_RETENTION_CRON,
    RetentionRegistryError,
    load_governed_web_retention_sources,
)
from pipelines.retention.wdc import WDCRetentionError, maintain_wdc_research_retention
from pipelines.retention.web import WebRetentionError, maintain_web_retention

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
WDC_RESEARCH_RETENTION_CRON = "15 0 * * *"


def execute_governed_web_retention(
    *,
    registry_path: Path,
    raw_root: Path,
    processed_root: Path,
    orphan_grace_hours: int = 24,
    maximum_entries: int = 100_000,
) -> dict[str, Any]:
    """Run the sole destructive engine for every registry-managed web source."""

    if type(orphan_grace_hours) is not int or not 1 <= orphan_grace_hours <= 24 * 30:
        raise ValueError("WEB_RETENTION_ORPHAN_GRACE_HOURS must be between 1 and 720")
    if type(maximum_entries) is not int or not 1 <= maximum_entries <= 1_000_000:
        raise ValueError("WEB_RETENTION_MAXIMUM_ENTRIES must be between 1 and 1,000,000")
    source_names = load_governed_web_retention_sources(registry_path)
    reports = maintain_web_retention(
        raw_root=raw_root,
        processed_root=processed_root,
        source_names=source_names,
        orphan_grace=timedelta(hours=orphan_grace_hours),
        maximum_entries=maximum_entries,
        dry_run=False,
    )
    counters = {
        "raw_receipts_scanned": sum(item.raw.receipts_scanned for item in reports),
        "raw_receipts_removed": sum(item.raw.expired_receipts_removed for item in reports),
        "raw_bodies_removed": sum(
            item.raw.expired_bodies_removed + item.raw.orphan_bodies_removed for item in reports
        ),
        "processed_runs_scanned": sum(item.processed.runs_scanned for item in reports),
        "processed_runs_removed": sum(item.processed.expired_runs_removed for item in reports),
        "publication_operations_scanned": sum(
            item.processed.publication_operations_scanned for item in reports
        ),
        "publication_operations_removed": sum(
            item.processed.publication_operations_removed for item in reports
        ),
        "published_residues_removed": sum(
            item.processed.published_residues_removed for item in reports
        ),
        "preserved_unknown_files": sum(
            item.raw.unrelated_files_preserved + item.processed.unrelated_files_preserved
            for item in reports
        ),
        "grace_leftovers": sum(
            item.raw.orphan_bodies_in_grace
            + item.raw.crash_leftovers_in_grace
            + item.processed.publication_operations_in_grace
            for item in reports
        ),
    }
    for report in reports:
        mismatches = (
            report.raw.expired_receipts_eligible != report.raw.expired_receipts_removed,
            report.raw.expired_bodies_eligible != report.raw.expired_bodies_removed,
            report.raw.orphan_bodies_eligible != report.raw.orphan_bodies_removed,
            report.raw.crash_leftovers_eligible != report.raw.crash_leftovers_removed,
            report.raw.cache_files_eligible != report.raw.cache_files_removed,
            report.processed.expired_runs_eligible != report.processed.expired_runs_removed,
            (
                report.processed.publication_operations_eligible
                != report.processed.publication_operations_removed
            ),
            (
                report.processed.published_residues_detected
                < report.processed.published_residues_removed
            ),
        )
        if any(mismatches):
            raise WebRetentionError(
                f"retention removal count mismatch for source {report.source_name!r}"
            )
    return {
        "status": "ok",
        "source_names": list(source_names),
        "source_count": len(source_names),
        **counters,
    }


def execute_wdc_research_retention(
    *,
    raw_root: Path,
    output_root: Path,
    category_index: Path,
    maximum_entries: int = 100_000,
) -> dict[str, Any]:
    """Run the independently governed WDC quarantine retention engine.

    WDC is intentionally absent from the production governed-web registry: it has
    a different receipt/manifest format and must never be conflated with a source
    authorised for catalogue display or model training.
    """

    if type(maximum_entries) is not int or not 1 <= maximum_entries <= 1_000_000:
        raise ValueError("WDC_RETENTION_MAXIMUM_ENTRIES must be between 1 and 1,000,000")
    report = maintain_wdc_research_retention(
        raw_root=raw_root,
        output_root=output_root,
        category_index=category_index,
        maximum_entries=maximum_entries,
        dry_run=False,
    )
    mismatches = (
        report.raw_pairs_eligible != report.raw_pairs_removed,
        report.category_index_eligible != report.category_index_removed,
        report.sealed_runs_eligible != report.sealed_runs_removed,
        report.working_runs_eligible != report.working_runs_removed,
    )
    if any(mismatches):
        raise WDCRetentionError("WDC retention removal count mismatch")
    return {"status": "ok", **report.to_dict()}


try:
    from dagster import Failure, asset
except ModuleNotFoundError as exc:
    if exc.name != "dagster":
        raise
    DAGSTER_AVAILABLE = False
    ASSETS: tuple[object, ...] = ()
else:
    DAGSTER_AVAILABLE = True

    @asset(group_name="data_retention", compute_kind="filesystem")
    @instrument_pipeline_operation("governed_web_retention_maintenance")
    def governed_web_retention_maintenance(  # type: ignore[no-untyped-def]
        context,
    ) -> dict[str, Any]:
        """Enforce every governed-web receipt on the running hourly schedule."""

        try:
            result = execute_governed_web_retention(
                registry_path=Path(
                    os.getenv(
                        "SOURCE_REGISTRY_PATH",
                        str(REPOSITORY_ROOT / "data" / "source_registry.yaml"),
                    )
                ),
                raw_root=Path(os.getenv("RAW_DATA_DIR", "data/raw")),
                processed_root=Path(os.getenv("PROCESSED_DATA_DIR", "data/processed")),
                orphan_grace_hours=int(os.getenv("WEB_RETENTION_ORPHAN_GRACE_HOURS", "24")),
                maximum_entries=int(os.getenv("WEB_RETENTION_MAXIMUM_ENTRIES", "100000")),
            )
        except (OSError, ValueError, RetentionRegistryError, WebRetentionError) as exc:
            context.log.error("governed_web_retention_failed error=%s", str(exc))
            raise Failure(f"governed_web_retention_failed: {exc}") from exc
        context.add_output_metadata(result)
        if result["preserved_unknown_files"] or result["grace_leftovers"]:
            context.log.warning(
                "governed_web_retention_attention preserved_unknown_files=%s grace_leftovers=%s",
                result["preserved_unknown_files"],
                result["grace_leftovers"],
            )
        return result

    @asset(group_name="data_retention", compute_kind="filesystem")
    @instrument_pipeline_operation("wdc_research_retention_maintenance")
    def wdc_research_retention_maintenance(  # type: ignore[no-untyped-def]
        context,
    ) -> dict[str, Any]:
        """Delete expired WDC research-quarantine artifacts once per day."""

        quarantine_root = Path(os.getenv("WDC_QUARANTINE_DIR", "data/quarantine"))
        try:
            result = execute_wdc_research_retention(
                raw_root=Path(os.getenv("RAW_DATA_DIR", "data/raw")),
                output_root=quarantine_root,
                category_index=Path(
                    os.getenv(
                        "WDC_CATEGORY_INDEX_PATH",
                        str(quarantine_root / "wdc-products-category-index.sqlite3"),
                    )
                ),
                maximum_entries=int(os.getenv("WDC_RETENTION_MAXIMUM_ENTRIES", "100000")),
            )
        except (OSError, ValueError, WDCRetentionError) as exc:
            context.log.error("wdc_research_retention_failed error=%s", str(exc))
            raise Failure(f"wdc_research_retention_failed: {exc}") from exc
        context.add_output_metadata(result)
        if result["unrelated_entries_preserved"]:
            context.log.warning(
                "wdc_research_retention_attention preserved_unknown_files=%s",
                result["unrelated_entries_preserved"],
            )
        return result

    ASSETS = (governed_web_retention_maintenance, wdc_research_retention_maintenance)


__all__ = [
    "ASSETS",
    "DAGSTER_AVAILABLE",
    "GOVERNED_WEB_RETENTION_CRON",
    "WDC_RESEARCH_RETENTION_CRON",
    "execute_governed_web_retention",
    "execute_wdc_research_retention",
]
