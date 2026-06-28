"""Manual Dagster ingestion assets for the permitted starter datasets."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from pipelines.assets.operation_observation import instrument_pipeline_operation
from pipelines.checks.quality import evaluate_batch_quality_against_previous, write_quality_report
from pipelines.parsing.writer import write_parsed_batch
from pipelines.sources.base import ParsedBatch
from pipelines.sources.blender import BlenderOpenDataAdapter
from pipelines.sources.buildcores import BuildCoresOpenDBAdapter
from pipelines.sources.mlperf import MLPerfInferenceAdapter
from pipelines.sources.retailer_csv import (
    ConsentedRetailerCSVAdapter,
    RetailerFeedPolicy,
)


def persist_checked_batch(
    batch: ParsedBatch,
    *,
    processed_root: str | Path,
    variant: str | None = None,
) -> dict[str, Any]:
    """Persist a parsed batch, its manifest, and its quality report."""

    artifacts = write_parsed_batch(
        batch,
        processed_root=processed_root,
        prefer_parquet=True,
        variant=variant,
    )
    quality = evaluate_batch_quality_against_previous(
        batch,
        processed_root=processed_root,
        variant=variant,
    )
    quality_path = write_quality_report(quality, artifacts.output_directory / "data-quality.json")
    return {
        "source_name": batch.source_name,
        "snapshot_sha256": batch.snapshot_sha256,
        "accepted_count": batch.accepted_count,
        "rejected_count": batch.rejected_count,
        "quality_status": quality.status,
        "manifest_path": str(artifacts.manifest_json.resolve()),
        "quality_report_path": str(quality_path.resolve()),
        "records_path": str(artifacts.records_jsonl.resolve()),
    }


def _optional_path(environment_name: str) -> Path | None:
    value = os.getenv(environment_name)
    return Path(value) if value else None


try:
    from dagster import Failure, asset
except ModuleNotFoundError as exc:
    if exc.name != "dagster":
        raise
    DAGSTER_AVAILABLE = False
    ASSETS: tuple[object, ...] = ()
else:
    DAGSTER_AVAILABLE = True

    def _record_metadata(context: Any, result: dict[str, Any]) -> None:
        context.add_output_metadata(
            {
                "snapshot_sha256": result["snapshot_sha256"],
                "accepted_count": result["accepted_count"],
                "rejected_count": result["rejected_count"],
                "quality_status": result["quality_status"],
                "manifest_path": result["manifest_path"],
                "quality_report_path": result["quality_report_path"],
            }
        )
        if result["quality_status"] == "fail":
            raise Failure(
                f"{result['source_name']} failed data-quality checks; "
                f"see {result['quality_report_path']}"
            )

    @asset(group_name="catalog_ingestion", compute_kind="python")
    @instrument_pipeline_operation("buildcores_catalog_ingestion")
    def buildcores_catalog_ingestion(context) -> dict[str, Any]:  # type: ignore[no-untyped-def]
        """Fetch and parse the pinned, attributed BuildCores catalogue snapshot."""

        raw_root = Path(os.getenv("RAW_DATA_DIR", "data/raw"))
        processed_root = Path(os.getenv("PROCESSED_DATA_DIR", "data/processed"))
        per_category_limit = int(os.getenv("BUILDCORES_PER_CATEGORY_LIMIT", "100"))
        adapter = BuildCoresOpenDBAdapter(raw_root=raw_root)
        snapshot = adapter.fetch(archive_path=_optional_path("BUILDCORES_ARCHIVE_PATH"))
        batch = adapter.parse(snapshot, per_category_limit=per_category_limit)
        result = persist_checked_batch(
            batch,
            processed_root=processed_root,
            variant=f"limit_{per_category_limit}",
        )
        _record_metadata(context, result)
        return result

    @asset(group_name="benchmark_ingestion", compute_kind="python")
    @instrument_pipeline_operation("blender_benchmark_ingestion")
    def blender_benchmark_ingestion(context) -> dict[str, Any]:  # type: ignore[no-untyped-def]
        """Fetch and parse a bounded Blender Open Data benchmark snapshot."""

        raw_root = Path(os.getenv("RAW_DATA_DIR", "data/raw"))
        processed_root = Path(os.getenv("PROCESSED_DATA_DIR", "data/processed"))
        limit = int(os.getenv("BLENDER_MAX_OBSERVATIONS", "3000"))
        adapter = BlenderOpenDataAdapter(raw_root=raw_root)
        snapshot = adapter.fetch(archive_path=_optional_path("BLENDER_ARCHIVE_PATH"))
        batch = adapter.parse(snapshot, max_observations=limit)
        result = persist_checked_batch(
            batch,
            processed_root=processed_root,
            variant=f"limit_{limit}",
        )
        _record_metadata(context, result)
        return result

    @asset(group_name="benchmark_ingestion", compute_kind="python")
    @instrument_pipeline_operation("mlperf_benchmark_ingestion")
    def mlperf_benchmark_ingestion(context) -> dict[str, Any]:  # type: ignore[no-untyped-def]
        """Fetch and parse the pinned official MLPerf Inference summary."""

        raw_root = Path(os.getenv("RAW_DATA_DIR", "data/raw"))
        processed_root = Path(os.getenv("PROCESSED_DATA_DIR", "data/processed"))
        max_records_value = os.getenv("MLPERF_MAX_RECORDS")
        max_records = int(max_records_value) if max_records_value else None
        adapter = MLPerfInferenceAdapter(raw_root=raw_root)
        snapshot = adapter.fetch(summary_path=_optional_path("MLPERF_SUMMARY_PATH"))
        batch = adapter.parse(snapshot, max_records=max_records)
        result = persist_checked_batch(
            batch,
            processed_root=processed_root,
            variant=f"limit_{max_records}" if max_records is not None else "complete",
        )
        _record_metadata(context, result)
        return result

    @asset(group_name="retailer_ingestion", compute_kind="python")
    @instrument_pipeline_operation("consented_retailer_feed_ingestion")
    def consented_retailer_feed_ingestion(  # type: ignore[no-untyped-def]
        context,
    ) -> dict[str, Any]:
        """Import a local retailer CSV only with an explicit, auditable consent policy."""

        csv_path = _optional_path("RETAILER_FEED_CSV")
        policy_json = os.getenv("RETAILER_FEED_POLICY_JSON")
        if csv_path is None or policy_json is None:
            raise Failure(
                "Set RETAILER_FEED_CSV and RETAILER_FEED_POLICY_JSON before running this asset."
            )
        try:
            policy_payload = json.loads(policy_json)
            if not isinstance(policy_payload, dict):
                raise TypeError("policy root must be an object")
            policy = RetailerFeedPolicy.from_mapping(policy_payload)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise Failure(f"Invalid RETAILER_FEED_POLICY_JSON: {exc}") from exc

        raw_root = Path(os.getenv("RAW_DATA_DIR", "data/raw"))
        processed_root = Path(os.getenv("PROCESSED_DATA_DIR", "data/processed"))
        adapter = ConsentedRetailerCSVAdapter(raw_root=raw_root, policy=policy)
        batch = adapter.parse(adapter.fetch(csv_path=csv_path))
        result = persist_checked_batch(batch, processed_root=processed_root)
        _record_metadata(context, result)
        return result

    ASSETS = (
        buildcores_catalog_ingestion,
        blender_benchmark_ingestion,
        mlperf_benchmark_ingestion,
        consented_retailer_feed_ingestion,
    )
