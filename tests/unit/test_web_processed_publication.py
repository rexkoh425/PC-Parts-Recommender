from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pipelines.retention.publication as publication_module
import pytest
import scripts.fetch_open_data as fetch_open_data
from pipelines.checks.quality import evaluate_batch_quality, write_quality_report
from pipelines.parsing.writer import write_parsed_batch
from pipelines.retention.publication import (
    WEB_PUBLICATION_CONTROL_DIRECTORY,
    WebProcessedPublication,
    WebPublicationError,
    begin_web_processed_publication,
    execute_web_publication_recovery,
    plan_web_publication_recovery,
    publish_web_processed_publication,
    seal_web_processed_publication,
)
from pipelines.retention.web import (
    WEB_PROCESSED_RETENTION_RECEIPT,
    write_web_processed_retention_receipt,
)
from pipelines.sources.base import ParsedBatch, RawSnapshot
from pipelines.sources.web_product import (
    CrawledPage,
    WebAcquisitionAuthority,
    WebCrawlResult,
    WebSourcePolicy,
    WebUsageScope,
)

from pc_build_recommender.data_rights import DataUseRights
from pc_build_recommender.domain.enums import ComponentCategory

SOURCE_NAME = "fixture_web_publication"
RUN_SHA256 = "7" * 64
PRODUCT_URL = "https://shop.example.test/products/gpu"
TERMS_URL = "https://shop.example.test/terms"
NOW = datetime(2026, 7, 22, 12, 0, tzinfo=UTC)


def _stale_recovery_now() -> datetime:
    """Keep fixed receipt timestamps while placing filesystem mtimes beyond the grace window."""

    return max(NOW, datetime.now(UTC)) + timedelta(days=4)


def _policy() -> WebSourcePolicy:
    return WebSourcePolicy(
        source_name=SOURCE_NAME,
        retailer="Fixture Shop",
        allowed_hosts=("shop.example.test",),
        terms_url=TERMS_URL,
        terms_selector="#terms",
        canonical_terms_sha256="a" * 64,
        terms_verified_on=date(2026, 7, 1),
        licence_or_access_note="Fixture governed-web publication test.",
        rights=DataUseRights(
            contract_reference="fixture-rights-v1",
            contract_version_url=TERMS_URL,
            consent_effective_on=date(2026, 1, 1),
            consent_expires_on=None,
            retention_days=7,
            deletion_required_on_termination=True,
            deletion_sla_days=1,
            territories=("SG",),
            may_display=False,
            may_cache=False,
            may_store_history=False,
            may_redistribute=False,
            may_embed=False,
            may_train=False,
            may_derive=False,
        ),
        acquisition_authority=WebAcquisitionAuthority(
            authority_reference="fixture-authority-v1",
            reviewed_on=date(2026, 1, 1),
            expires_on=None,
            permits_automated_retrieval=True,
            permits_raw_snapshot_storage=True,
            permits_internal_analysis=True,
            retention_days=7,
            deletion_required=True,
        ),
        url_categories={PRODUCT_URL: ComponentCategory.GPU},
        usage_scope=WebUsageScope.INTERNAL_RESEARCH,
    )


def _write_source_registry(
    path: Path,
    policy: WebSourcePolicy,
    *,
    restricted: bool = False,
) -> None:
    hosts = "\n".join(f"      - {host}" for host in policy.allowed_hosts)
    restricted_block = ""
    if restricted:
        restricted_block = f"""\
blocked_or_restricted_sources:
  fixture_shop:
    reason: Fixture terms prohibit automated extraction.
    hosts:
{hosts}
    terms_url: {TERMS_URL}
    reviewed_on: 2026-07-22
"""
    path.write_text(
        f"""\
schema_version: pc-build-recommender.source-registry.v1
sources:
  {policy.source_name}:
    kind: exact_url_schema_org_product_offer_crawl
    template: governed_web_product
    source_url: {PRODUCT_URL}
    allowed_hosts:
{hosts}
    usage_scope: {policy.usage_scope.value}
    retention_maintenance:
      engine: governed_web_receipts_v2
      required: true
      maximum_interval_minutes: 60
{restricted_block}""",
        encoding="utf-8",
    )


def _prepare_complete_stage(processed_root: Path) -> WebProcessedPublication:
    publication = begin_web_processed_publication(
        processed_root=processed_root,
        source_name=SOURCE_NAME,
        run_sha256=RUN_SHA256,
        created_at=NOW,
    )
    batch = ParsedBatch(
        source_name=SOURCE_NAME,
        snapshot_sha256=RUN_SHA256,
        records=[],
        rejected=[],
        statistics={},
    )
    artifacts = write_parsed_batch(
        batch,
        processed_root=publication.workspace_processed_root,
        prefer_parquet=False,
    )
    quality = evaluate_batch_quality(batch, maximum_rejection_rate=0.25)
    write_quality_report(quality, artifacts.output_directory / "data-quality.json")
    write_web_processed_retention_receipt(
        processed_root=publication.workspace_processed_root,
        output_directory=artifacts.output_directory,
        policy=_policy(),
        retrieval_started_at=NOW - timedelta(hours=2),
        retrieval_completed_at=NOW - timedelta(hours=1),
        created_at=NOW,
    )
    seal_web_processed_publication(publication, sealed_at=NOW)
    return publication


def _run_bytes(run_directory: Path) -> dict[str, bytes]:
    return {child.name: child.read_bytes() for child in run_directory.iterdir() if child.is_file()}


def test_crash_before_publish_never_exposes_a_final_run(tmp_path: Path) -> None:
    processed_root = tmp_path / "processed"
    publication = _prepare_complete_stage(processed_root)

    assert not publication.final_directory.exists()
    assert publication.staged_run_directory.is_dir()
    assert publication.intent_path.is_file()
    assert publication.ready_path.is_file()
    assert publication.staged_run_directory.joinpath(WEB_PROCESSED_RETENTION_RECEIPT).is_file()
    assert list(publication.source_root.iterdir()) == []
    assert publication.control_root == processed_root / WEB_PUBLICATION_CONTROL_DIRECTORY
    assert publication.control_root.is_dir()


def test_two_publishers_never_overwrite_the_winning_run(tmp_path: Path) -> None:
    processed_root = tmp_path / "processed"
    first = _prepare_complete_stage(processed_root)
    second = _prepare_complete_stage(processed_root)
    expected_bytes = _run_bytes(first.staged_run_directory)
    assert expected_bytes == _run_bytes(second.staged_run_directory)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(publish_web_processed_publication, publication)
            for publication in (first, second)
        ]
        outcomes: list[Path | BaseException] = []
        for future in futures:
            try:
                outcomes.append(future.result())
            except BaseException as exc:  # noqa: BLE001 - assert the exact concurrent outcome.
                outcomes.append(exc)

    successes = [outcome for outcome in outcomes if isinstance(outcome, Path)]
    failures = [outcome for outcome in outcomes if isinstance(outcome, BaseException)]
    assert successes == [processed_root / SOURCE_NAME / RUN_SHA256]
    assert len(failures) == 1
    assert isinstance(failures[0], WebPublicationError)
    assert "already exists" in str(failures[0])
    assert _run_bytes(successes[0]) == expected_bytes
    losing_stages = [
        publication for publication in (first, second) if publication.staged_run_directory.is_dir()
    ]
    assert len(losing_stages) == 1


def test_publish_revalidates_a_sealed_stage_before_rename(tmp_path: Path) -> None:
    publication = _prepare_complete_stage(tmp_path / "processed")
    records = publication.staged_run_directory / "records.jsonl"
    records.write_text('{"tampered":true}\n', encoding="utf-8")

    with pytest.raises(WebPublicationError, match="changed after it was sealed"):
        publish_web_processed_publication(publication)

    assert not publication.final_directory.exists()
    assert publication.staged_run_directory.is_dir()


def test_cleanup_failure_does_not_report_a_committed_publication_as_failed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    publication = _prepare_complete_stage(tmp_path / "processed")

    def fail_cleanup(_publication: WebProcessedPublication) -> None:
        raise OSError("injected control cleanup failure")

    monkeypatch.setattr(publication_module, "_clean_published_operation", fail_cleanup)

    final_directory = publish_web_processed_publication(publication)

    assert final_directory == publication.final_directory
    assert final_directory.is_dir()
    assert final_directory.joinpath(WEB_PROCESSED_RETENTION_RECEIPT).is_file()
    assert publication.operation_directory.is_dir()


def test_recovery_reclaims_a_stale_published_control_residue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    publication = _prepare_complete_stage(tmp_path / "processed")

    def fail_cleanup(_publication: WebProcessedPublication) -> None:
        raise OSError("injected control cleanup failure")

    monkeypatch.setattr(publication_module, "_clean_published_operation", fail_cleanup)
    assert publish_web_processed_publication(publication) == publication.final_directory
    assert publication.operation_directory.is_dir()

    plan = plan_web_publication_recovery(
        processed_root=tmp_path / "processed",
        source_names=(SOURCE_NAME,),
        now=_stale_recovery_now(),
        orphan_grace=timedelta(hours=24),
    )
    reports = execute_web_publication_recovery(plan, dry_run=False)

    assert not publication.operation_directory.exists()
    assert publication.final_directory.is_dir()
    assert reports[0].published_residues_detected == 1
    assert reports[0].published_residues_removed == 1
    assert reports[0].operations_removed == 1


def test_recovery_revalidates_a_stale_operation_before_deletion(tmp_path: Path) -> None:
    processed_root = tmp_path / "processed"
    publication = begin_web_processed_publication(
        processed_root=processed_root,
        source_name=SOURCE_NAME,
        run_sha256=RUN_SHA256,
        created_at=NOW - timedelta(days=2),
    )
    plan = plan_web_publication_recovery(
        processed_root=processed_root,
        source_names=(SOURCE_NAME,),
        now=_stale_recovery_now(),
        orphan_grace=timedelta(hours=24),
    )
    publication.operation_directory.joinpath(SOURCE_NAME).mkdir()

    with pytest.raises(WebPublicationError, match="changed before recovery deletion"):
        execute_web_publication_recovery(plan, dry_run=False)

    assert publication.operation_directory.is_dir()
    assert publication.intent_path.is_file()


def test_publish_rejects_a_linklike_stage_swap_after_sealing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    publication = _prepare_complete_stage(tmp_path / "processed")
    original_is_linklike = publication_module._is_linklike
    monkeypatch.setattr(
        publication_module,
        "_is_linklike",
        lambda path: path == publication.staged_run_directory or original_is_linklike(path),
    )

    with pytest.raises(WebPublicationError, match="link-like"):
        publish_web_processed_publication(publication)

    assert not publication.final_directory.exists()
    assert publication.staged_run_directory.is_dir()


def test_web_ingestion_script_publishes_only_the_complete_final_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current_time = datetime.now(UTC)
    policy = _policy()
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps(policy.to_dict()), encoding="utf-8")
    source_registry = tmp_path / "source-registry.yaml"
    _write_source_registry(source_registry, policy)
    raw_page = tmp_path / "page.html"
    raw_page.write_bytes(b"fixture page")
    raw_metadata = tmp_path / "page.json"
    raw_metadata.write_text("{}", encoding="utf-8")
    snapshot = RawSnapshot(
        source_name=SOURCE_NAME,
        source_url=PRODUCT_URL,
        source_type="retailer",
        retrieved_at=current_time - timedelta(hours=1),
        content_sha256=hashlib.sha256(raw_page.read_bytes()).hexdigest(),
        byte_count=raw_page.stat().st_size,
        media_type="text/html",
        parser_version="fixture-v1",
        licence_or_access_note="Fixture.",
        path=raw_page,
        metadata_path=raw_metadata,
    )
    result = WebCrawlResult(
        batch=ParsedBatch(
            source_name=SOURCE_NAME,
            snapshot_sha256=RUN_SHA256,
            records=[],
            rejected=[],
            statistics={},
        ),
        pages=(
            CrawledPage(
                requested_url=PRODUCT_URL,
                final_url=PRODUCT_URL,
                snapshot=snapshot,
                etag=None,
                last_modified=None,
                not_modified=False,
            ),
        ),
        retrieval_started_at=current_time - timedelta(hours=2),
        retrieval_completed_at=current_time - timedelta(hours=1),
        robots_sha256_by_host={"shop.example.test": "b" * 64},
        terms_snapshot_sha256="c" * 64,
        terms_post_snapshot_sha256="c" * 64,
        terms_canonical_sha256="a" * 64,
        policy_fingerprint=policy.fingerprint,
    )

    class FakeCrawler:
        def __init__(self, *, raw_root: Path, policy: WebSourcePolicy) -> None:
            assert raw_root == tmp_path / "raw"
            assert policy.source_name == SOURCE_NAME

        def crawl(self, urls: list[str]) -> WebCrawlResult:
            assert urls == [PRODUCT_URL]
            return result

    monkeypatch.setattr(fetch_open_data, "WebProductCrawlerAdapter", FakeCrawler)
    processed_root = tmp_path / "processed"
    args = argparse.Namespace(
        web_policy_json=policy_path,
        web_url=[PRODUCT_URL],
        raw_root=tmp_path / "raw",
        processed_root=processed_root,
        source_registry=source_registry,
        no_parquet=True,
    )

    summary = fetch_open_data._run_web_product(args)

    final_directory = processed_root / SOURCE_NAME / RUN_SHA256
    assert final_directory.is_dir()
    assert Path(summary["records_path"]) == final_directory / "records.jsonl"
    assert Path(summary["retention_receipt_path"]) == (
        final_directory / WEB_PROCESSED_RETENTION_RECEIPT
    )
    assert not processed_root.joinpath(WEB_PUBLICATION_CONTROL_DIRECTORY).exists()


def test_web_ingestion_rejects_a_registry_restricted_host_before_crawling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = _policy()
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps(policy.to_dict()), encoding="utf-8")
    registry_path = tmp_path / "source-registry.yaml"
    _write_source_registry(registry_path, policy, restricted=True)

    class UnexpectedCrawler:
        def __init__(self, **_: object) -> None:
            raise AssertionError("a restricted source must fail before crawler construction")

    monkeypatch.setattr(fetch_open_data, "WebProductCrawlerAdapter", UnexpectedCrawler)
    args = argparse.Namespace(
        web_policy_json=policy_path,
        web_url=[PRODUCT_URL],
        raw_root=tmp_path / "raw",
        processed_root=tmp_path / "processed",
        source_registry=registry_path,
        no_parquet=True,
    )

    with pytest.raises(ValueError, match="restricted by source registry entry 'fixture_shop'"):
        fetch_open_data._run_web_product(args)


def test_web_ingestion_rejects_an_unregistered_policy_before_crawling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = _policy()
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps(policy.to_dict()), encoding="utf-8")
    registry_path = tmp_path / "source-registry.yaml"
    registry_path.write_text(
        "schema_version: pc-build-recommender.source-registry.v1\nsources: {}\n",
        encoding="utf-8",
    )

    class UnexpectedCrawler:
        def __init__(self, **_: object) -> None:
            raise AssertionError("an unregistered source must fail before crawler construction")

    monkeypatch.setattr(fetch_open_data, "WebProductCrawlerAdapter", UnexpectedCrawler)
    args = argparse.Namespace(
        web_policy_json=policy_path,
        web_url=[PRODUCT_URL],
        raw_root=tmp_path / "raw",
        processed_root=tmp_path / "processed",
        source_registry=registry_path,
        no_parquet=True,
    )

    with pytest.raises(ValueError, match="not a registered governed-web source"):
        fetch_open_data._run_web_product(args)
