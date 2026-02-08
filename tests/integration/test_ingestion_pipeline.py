from __future__ import annotations

import csv
import json
from datetime import date

import pytest
from pipelines.checks.quality import evaluate_batch_quality, write_quality_report
from pipelines.parsing.writer import write_parsed_batch
from pipelines.sources.base import sha256_file
from pipelines.sources.retailer_csv import ConsentedRetailerCSVAdapter, RetailerFeedPolicy
from pipelines.sources.rights import DataUseRights


@pytest.mark.integration
def test_csv_snapshot_parse_write_and_quality_are_idempotent(tmp_path) -> None:
    csv_path = tmp_path / "feed.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "source_listing_id",
                "title",
                "currency",
                "base_price",
                "stock_status",
                "listing_url",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "source_listing_id": "fixture-1",
                "title": "Fixture CPU",
                "currency": "SGD",
                "base_price": "399",
                "stock_status": "available",
                "listing_url": "https://example.test/fixture-1",
            }
        )
    policy = RetailerFeedPolicy(
        retailer="Fixture Retailer",
        feed_id="integration",
        source_url="controlled://integration-feed",
        licence_or_access_note="Testing only.",
        rights=DataUseRights(
            contract_reference="test-agreement",
            contract_version_url="contract://fixture/test-agreement",
            consent_effective_on=date(2026, 1, 1),
            consent_expires_on=None,
            retention_days=30,
            deletion_required_on_termination=True,
            deletion_sla_days=7,
            territories=("SG",),
            may_display=True,
            may_cache=True,
            may_store_history=True,
            may_redistribute=False,
            may_embed=False,
            may_train=False,
            may_derive=True,
        ),
    )
    adapter = ConsentedRetailerCSVAdapter(raw_root=tmp_path / "raw", policy=policy)

    first_snapshot = adapter.fetch(csv_path=csv_path)
    first_batch = adapter.parse(first_snapshot)
    first_artifacts = write_parsed_batch(
        first_batch,
        processed_root=tmp_path / "processed",
        prefer_parquet=False,
        variant="fixture",
    )
    first_hash = sha256_file(first_artifacts.records_jsonl)
    first_manifest = json.loads(first_artifacts.manifest_json.read_text(encoding="utf-8"))

    second_snapshot = adapter.fetch(csv_path=csv_path)
    second_batch = adapter.parse(second_snapshot)
    second_artifacts = write_parsed_batch(
        second_batch,
        processed_root=tmp_path / "processed",
        prefer_parquet=False,
        variant="fixture",
    )
    report = evaluate_batch_quality(second_batch, maximum_rejection_rate=0.1)
    report_path = write_quality_report(
        report, second_artifacts.output_directory / "data-quality.json"
    )

    assert first_snapshot.reused is False
    assert second_snapshot.reused is True
    assert first_hash == sha256_file(second_artifacts.records_jsonl)
    assert first_manifest == json.loads(second_artifacts.manifest_json.read_text(encoding="utf-8"))
    assert report.status == "pass"
    assert json.loads(report_path.read_text(encoding="utf-8"))["status"] == "pass"
