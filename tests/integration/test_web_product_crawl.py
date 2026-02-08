from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest
from pipelines.parsing.writer import write_parsed_batch
from pipelines.sources.base import sha256_file
from pipelines.sources.web_product import (
    WebCrawlError,
    WebCrawlSecurityError,
    WebProductCrawlerAdapter,
)
from tests.unit.test_sources_web_product import (
    PRODUCT_URL,
    ROBOTS_BODY,
    TERMS_BODY,
    VirtualClock,
    _policy,
    _product_html,
    _transport,
)


def _crawl(tmp_path, *, conditional: bool):
    clock = VirtualClock()
    adapter = WebProductCrawlerAdapter(
        raw_root=tmp_path / "raw",
        policy=_policy(),
        transport=_transport(conditional=conditional),
        resolver=lambda _host: ("1.1.1.1",),
        clock=clock,
        sleeper=clock.sleep,
    )
    return adapter.crawl([PRODUCT_URL])


def test_conditional_crawl_reuses_body_but_writes_a_new_observation_receipt(tmp_path) -> None:
    first = _crawl(tmp_path, conditional=False)
    first_artifacts = write_parsed_batch(
        first.batch,
        processed_root=tmp_path / "processed",
        prefer_parquet=False,
    )
    records_before = first_artifacts.records_jsonl.read_bytes()

    second = _crawl(tmp_path, conditional=True)
    second_artifacts = write_parsed_batch(
        second.batch,
        processed_root=tmp_path / "processed",
        prefer_parquet=False,
    )

    assert first.batch.snapshot_sha256 != second.batch.snapshot_sha256
    assert (
        first.batch.records[0]["archive_snapshot_sha256"]
        == (second.batch.records[0]["archive_snapshot_sha256"])
    )
    assert second.pages[0].not_modified is True
    assert second.pages[0].snapshot.reused is True
    assert second.pages[0].snapshot.path == first.pages[0].snapshot.path
    assert second.pages[0].snapshot.metadata_path != first.pages[0].snapshot.metadata_path
    assert second.pages[0].snapshot.retrieved_at >= first.pages[0].snapshot.retrieved_at
    assert second_artifacts.records_jsonl.read_bytes() != records_before
    assert sha256_file(first_artifacts.records_jsonl) != sha256_file(second_artifacts.records_jsonl)
    assert (
        json.loads(second_artifacts.manifest_json.read_text(encoding="utf-8"))[
            "source_snapshot_sha256"
        ]
        == second.batch.snapshot_sha256
    )


def test_v3_listing_identity_is_stable_across_conditional_observations(tmp_path) -> None:
    first = _crawl(tmp_path, conditional=True)
    second = _crawl(tmp_path, conditional=True)

    first_record = first.batch.records[0]
    second_record = second.batch.records[0]
    first_listing = first_record["data"]["listing"]
    second_listing = second_record["data"]["listing"]
    assert first.pages[0].snapshot.parser_version == "schemaorg-product-offer-v3"
    assert second.pages[0].not_modified is True
    assert first_record["source_record_id"] == second_record["source_record_id"]
    assert first_listing["listing_id"] == second_listing["listing_id"]
    assert first_record["source_record_id"].startswith("web_listing_")
    assert first_record["source_record_id"] != "FIXTURE-GPU-16"


def test_conditional_cached_bodies_are_charged_to_the_total_byte_budget(tmp_path) -> None:
    clock = VirtualClock()
    adapter = WebProductCrawlerAdapter(
        raw_root=tmp_path / "raw",
        policy=_policy(),
        transport=_transport(conditional=True),
        resolver=lambda _host: ("1.1.1.1",),
        clock=clock,
        sleeper=clock.sleep,
    )
    expected_bytes = len(ROBOTS_BODY) + (2 * len(TERMS_BODY)) + len(_product_html())

    adapter.crawl([PRODUCT_URL])
    assert adapter._total_bytes == expected_bytes
    second = adapter.crawl([PRODUCT_URL])

    assert second.pages[0].not_modified is True
    assert adapter._total_bytes == expected_bytes


def test_identical_200_responses_get_distinct_immutable_receipts(tmp_path) -> None:
    first = _crawl(tmp_path, conditional=False)
    second = _crawl(tmp_path, conditional=False)

    assert first.pages[0].snapshot.path == second.pages[0].snapshot.path
    assert first.pages[0].snapshot.metadata_path != second.pages[0].snapshot.metadata_path
    assert first.pages[0].snapshot.metadata_path.is_file()
    assert second.pages[0].snapshot.metadata_path.is_file()
    assert second.pages[0].snapshot.retrieved_at >= first.pages[0].snapshot.retrieved_at
    assert first.batch.snapshot_sha256 != second.batch.snapshot_sha256


def _expire_receipt(path, *, legacy_missing_expiry: bool = False) -> dict[str, object]:
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata["retrieved_at"] = (datetime.now(UTC) - timedelta(days=60)).isoformat()
    metadata["retention_expires_at"] = (
        None if legacy_missing_expiry else (datetime.now(UTC) - timedelta(days=1)).isoformat()
    )
    path.write_text(json.dumps(metadata), encoding="utf-8")
    return metadata


def test_expired_cache_reference_is_dropped_without_deleting_raw_evidence(
    tmp_path,
) -> None:
    first = _crawl(tmp_path, conditional=False)
    receipt_path = first.pages[0].snapshot.metadata_path
    old_metadata = _expire_receipt(receipt_path)

    second = _crawl(tmp_path, conditional=True)
    new_metadata = json.loads(second.pages[0].snapshot.metadata_path.read_text(encoding="utf-8"))

    assert receipt_path.exists()
    assert first.pages[0].snapshot.path.exists()
    assert second.pages[0].snapshot.metadata_path != receipt_path
    assert second.pages[0].not_modified is False
    assert second.pages[0].snapshot.reused is True
    assert new_metadata["retrieved_at"] != old_metadata["retrieved_at"]
    assert datetime.fromisoformat(str(new_metadata["retrieved_at"])) > datetime.now(UTC) - (
        timedelta(minutes=1)
    )


def test_cached_receipt_without_expiry_fails_closed_without_deleting_evidence(
    tmp_path,
) -> None:
    first = _crawl(tmp_path, conditional=False)
    receipt = first.pages[0].snapshot.metadata_path
    _expire_receipt(receipt, legacy_missing_expiry=True)

    with pytest.raises(WebCrawlError, match="has no retention expiry"):
        _crawl(tmp_path, conditional=True)

    assert receipt.exists()
    assert first.pages[0].snapshot.path.exists()


def test_retention_pruning_rejects_unsafe_raw_paths_without_deleting_outside_store(
    tmp_path,
) -> None:
    first = _crawl(tmp_path, conditional=False)
    receipt = first.pages[0].snapshot.metadata_path
    metadata = _expire_receipt(receipt)
    metadata["raw_file"] = "..\\..\\victim.html"
    receipt.write_text(json.dumps(metadata), encoding="utf-8")
    victim = tmp_path / "victim.html"
    victim.write_text("keep", encoding="utf-8")

    with pytest.raises(WebCrawlSecurityError, match="unsafe raw file"):
        _crawl(tmp_path, conditional=False)
    assert victim.read_text(encoding="utf-8") == "keep"
