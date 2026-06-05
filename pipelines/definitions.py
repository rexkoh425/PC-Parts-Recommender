"""Dagster code location with a safe import path when optional dependencies are absent."""

from __future__ import annotations

from typing import Any, cast

from pipelines.assets import ASSETS, DAGSTER_AVAILABLE

if DAGSTER_AVAILABLE:
    from dagster import (
        AssetSelection,
        DefaultScheduleStatus,
        Definitions,
        ScheduleDefinition,
        define_asset_job,
    )

    from pipelines.assets.retention import GOVERNED_WEB_RETENTION_CRON, WDC_RESEARCH_RETENTION_CRON

    def _select_asset(asset_name: str) -> AssetSelection:
        assets_selector = getattr(AssetSelection, "assets", None)
        if assets_selector is not None:
            return cast(AssetSelection, assets_selector(asset_name))
        return AssetSelection.keys(asset_name)

    data_observability_job = define_asset_job(
        name="data_observability",
        selection=AssetSelection.groups("data_observability"),
    )
    buildcores_import_job = define_asset_job(
        name="buildcores_import",
        selection=_select_asset("buildcores_catalog_ingestion"),
    )
    benchmark_import_job = define_asset_job(
        name="benchmark_import",
        selection=AssetSelection.groups("benchmark_ingestion"),
    )
    retailer_feed_import_job = define_asset_job(
        name="retailer_feed_import",
        selection=_select_asset("consented_retailer_feed_ingestion"),
    )
    governed_web_retention_job = define_asset_job(
        name="governed_web_retention",
        selection=_select_asset("governed_web_retention_maintenance"),
    )
    wdc_research_retention_job = define_asset_job(
        name="wdc_research_retention",
        selection=_select_asset("wdc_research_retention_maintenance"),
    )
    data_observability_schedule = ScheduleDefinition(
        job=data_observability_job,
        cron_schedule="0 */12 * * *",
        execution_timezone="Asia/Singapore",
    )
    governed_web_retention_schedule = ScheduleDefinition(
        name="governed_web_retention_hourly",
        job=governed_web_retention_job,
        cron_schedule=GOVERNED_WEB_RETENTION_CRON,
        execution_timezone="Asia/Singapore",
        default_status=DefaultScheduleStatus.RUNNING,
    )
    wdc_research_retention_schedule = ScheduleDefinition(
        name="wdc_research_retention_daily",
        job=wdc_research_retention_job,
        cron_schedule=WDC_RESEARCH_RETENTION_CRON,
        execution_timezone="Asia/Singapore",
        default_status=DefaultScheduleStatus.RUNNING,
    )
    defs: object | None = Definitions(
        assets=cast(Any, list(ASSETS)),
        jobs=[
            data_observability_job,
            buildcores_import_job,
            benchmark_import_job,
            retailer_feed_import_job,
            governed_web_retention_job,
            wdc_research_retention_job,
        ],
        schedules=[
            data_observability_schedule,
            governed_web_retention_schedule,
            wdc_research_retention_schedule,
        ],
    )
else:
    # ``defs`` remains discoverable for tools that inspect this module in the base environment.
    # The Dagster image installs the optional dependency and receives a real Definitions object.
    defs = None

__all__ = ["defs"]
