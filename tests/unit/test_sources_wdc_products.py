from __future__ import annotations

import gzip
import hashlib
import json
import os
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import pytest
from pipelines.sources.base import FetchedSnapshot, utc_now
from pipelines.sources.wdc_products import (
    WDC_RESEARCH_DATA_USE_RIGHTS,
    WDC_RESEARCH_MANIFEST_SCHEMA,
    WDCProductsResearchSource,
    WDCResearchLimitError,
    build_wdc_category_index,
    import_wdc_research_candidates,
    infer_pc_component_categories,
)
from scripts.import_wdc_research_corpus import main as import_wdc_main

from pc_build_recommender.data_rights import production_catalog_rights_are_valid


def _write_jsonl_gzip(path: Path, rows: list[dict[str, object]]) -> list[bytes]:
    encoded_rows = [
        json.dumps(row, sort_keys=True, separators=(",", ":")).encode() + b"\n" for row in rows
    ]
    with gzip.open(path, "wb") as handle:
        for row in encoded_rows:
            handle.write(row)
    return encoded_rows


def _fixture_snapshots(tmp_path: Path) -> tuple[FetchedSnapshot, FetchedSnapshot, list[bytes]]:
    category_path = tmp_path / "categories.json.gz"
    corpus_path = tmp_path / "corpus.json.gz"
    category_rows = [
        {
            "cluster_id": cluster_id,
            "predicted_CategoryLabel_majority_voted": category,
        }
        for cluster_id, category in (
            (100, "Computers_and_Accessories"),
            (101, "Tools_and_Home_Improvement"),
            (102, "Computers_and_Accessories"),
            (103, "Computers_and_Accessories"),
            (104, "Computers_and_Accessories"),
            (105, "Computers_and_Accessories"),
        )
    ]
    corpus_rows = [
        {
            "id": 1,
            "cluster_id": 100,
            "brand": "Corsair",
            "title": "Corsair ATX mid tower computer case",
            "description": "Historical case offer",
            "price": "131.99",
            "priceCurrency": "USD",
            "url": "https://historical.example/case",
        },
        {
            "id": 2,
            "cluster_id": 101,
            "brand": "Fixture",
            "title": "RTX styled workshop hammer",
            "description": None,
            "price": "12",
            "priceCurrency": "EUR",
            "url": "https://historical.example/hammer",
        },
        {
            "id": 3,
            "cluster_id": 102,
            "brand": "Fixture",
            "title": "15 inch gaming laptop",
            "description": "A complete notebook, not a desktop component",
            "price": "999",
            "priceCurrency": "GBP",
            "url": "https://historical.example/laptop",
        },
        {
            "id": 4,
            "cluster_id": 103,
            "brand": "NVIDIA",
            "title": "GeForce RTX 4070 graphics card 12 GB",
            "description": "Historical GPU offer",
            "price": "749",
            "priceCurrency": "USD",
            "url": "https://historical.example/gpu",
        },
        {
            "id": 5,
            "cluster_id": 104,
            "brand": "APC",
            "title": "APC UPS uninterruptible power supply 2000W",
            "description": "Must not be classified as a desktop PSU",
            "price": "500",
            "priceCurrency": "USD",
            "url": "https://historical.example/ups",
        },
        {
            "id": 6,
            "cluster_id": 105,
            "brand": "Bundle",
            "title": "GeForce RTX 4070 graphics card with 2TB NVMe SSD",
            "description": "Ambiguous bundle",
            "price": "900",
            "priceCurrency": "USD",
            "url": "https://historical.example/bundle",
        },
    ]
    corpus_source_lines = _write_jsonl_gzip(corpus_path, corpus_rows)
    _write_jsonl_gzip(category_path, category_rows)
    source = WDCProductsResearchSource(tmp_path / "raw")
    return (
        source.fetch_corpus(corpus_path=corpus_path),
        source.fetch_categories(category_path=category_path),
        corpus_source_lines,
    )


def test_category_index_and_research_import_are_resumable_and_idempotent(
    tmp_path: Path,
) -> None:
    corpus, categories, source_lines = _fixture_snapshots(tmp_path)
    index_path = tmp_path / "quarantine" / "categories.sqlite3"

    first_index = build_wdc_category_index(
        categories,
        index_path=index_path,
        record_budget=2,
        checkpoint_interval=1,
    )
    assert first_index.complete is False
    assert first_index.completed_source_lines == 2

    second_index = build_wdc_category_index(
        categories,
        index_path=index_path,
        record_budget=None,
        checkpoint_interval=1,
    )
    assert second_index.complete is True
    assert second_index.indexed_clusters == 6
    assert build_wdc_category_index(categories, index_path=index_path).reused is True

    output_root = tmp_path / "quarantine"
    first = import_wdc_research_candidates(
        corpus,
        category_index_path=index_path,
        output_root=output_root,
        record_budget=2,
        checkpoint_interval=1,
    )
    assert first.complete is False
    assert first.selected_records == 1
    assert first.work_path is not None

    # An interrupted append after the durable checkpoint is discarded on resume.
    with (first.work_path / "records.jsonl.part").open("ab") as handle:
        handle.write(b"crash-tail")

    second = import_wdc_research_candidates(
        corpus,
        category_index_path=index_path,
        output_root=output_root,
        record_budget=2,
        checkpoint_interval=1,
    )
    assert second.complete is False
    assert second.selected_records == 2

    completed = import_wdc_research_candidates(
        corpus,
        category_index_path=index_path,
        output_root=output_root,
        record_budget=2,
        checkpoint_interval=1,
    )
    assert completed.complete is True
    assert completed.selected_records == 2
    assert completed.rejected_non_computer_category == 1
    assert completed.rejected_not_component_like == 2
    assert completed.rejected_ambiguous_component == 1
    assert completed.output_path is not None
    assert completed.manifest_path is not None

    records = [
        json.loads(line) for line in completed.output_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [record["data"]["component_candidate_category"] for record in records] == [
        "case",
        "gpu",
    ]
    assert records[0]["provenance"]["source_snapshot_sha256"] == corpus.content_sha256
    assert records[0]["provenance"]["source_record_sha256"]
    assert records[0]["data"]["historical_price_text"] == "131.99"
    assert "listing" not in records[0]
    assert "stock_status" not in records[0]["data"]
    assert records[0]["quarantine"]["production_eligible"] is False
    assert records[0]["quarantine"]["singapore_market_evidence"] is False

    # Exact source-line provenance survives normalization.
    assert (
        records[0]["provenance"]["source_record_sha256"]
        == hashlib.sha256(source_lines[0]).hexdigest()
    )

    manifest = json.loads(completed.manifest_path.read_text(encoding="utf-8"))
    assert manifest["source_snapshot"]["content_sha256"] == corpus.content_sha256
    assert manifest["quarantine"]["model_training_eligible"] is False
    assert manifest["limits"]["maximum_selected_records"] == 100_000
    assert any(
        "current Singapore retailer" in item
        for item in manifest["evidence_scope"]["cannot_support"]
    )

    reused = import_wdc_research_candidates(
        corpus,
        category_index_path=index_path,
        output_root=output_root,
        record_budget=1,
    )
    assert reused.complete is True
    assert reused.reused is True
    assert reused.output_path == completed.output_path


def test_wdc_rights_and_component_rules_fail_closed() -> None:
    assert production_catalog_rights_are_valid(WDC_RESEARCH_DATA_USE_RIGHTS) is False
    assert infer_pc_component_categories("GeForce RTX 4080 graphics card") == ("gpu",)
    assert infer_pc_component_categories("APC UPS power supply 2000W") == ()
    assert infer_pc_component_categories("DDR5 32GB memory kit") == ("memory",)
    assert set(
        infer_pc_component_categories("GeForce RTX 4070 graphics card and 2TB NVMe SSD")
    ) == {"gpu", "storage"}


def test_wdc_parser_rejects_oversized_lines_before_publishing(tmp_path: Path) -> None:
    corpus, categories, _source_lines = _fixture_snapshots(tmp_path)
    index_path = tmp_path / "categories.sqlite3"
    build_wdc_category_index(categories, index_path=index_path)

    with pytest.raises(WDCResearchLimitError, match="line exceeds"):
        import_wdc_research_candidates(
            corpus,
            category_index_path=index_path,
            output_root=tmp_path / "quarantine",
            max_line_bytes=128,
        )
    assert not list((tmp_path / "quarantine").rglob("manifest.json"))


def test_wdc_parser_refuses_expired_research_snapshot(tmp_path: Path) -> None:
    corpus, categories, _source_lines = _fixture_snapshots(tmp_path)
    index_path = tmp_path / "categories.sqlite3"
    build_wdc_category_index(categories, index_path=index_path)
    expired = replace(corpus, retrieved_at=utc_now() - timedelta(days=366))

    with pytest.raises(PermissionError, match="retention limit"):
        import_wdc_research_candidates(
            expired,
            category_index_path=index_path,
            output_root=tmp_path / "quarantine",
        )


def test_wdc_cli_completes_small_local_fixture(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    corpus, categories, _source_lines = _fixture_snapshots(tmp_path)
    exit_code = import_wdc_main(
        [
            "--corpus",
            str(corpus.path),
            "--categories",
            str(categories.path),
            "--raw-root",
            str(tmp_path / "cli-raw"),
            "--output-root",
            str(tmp_path / "cli-quarantine"),
            "--category-index",
            str(tmp_path / "cli-quarantine" / "categories.sqlite3"),
        ]
    )

    assert exit_code == 0
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "complete"
    assert report["production_eligible"] is False
    assert report["corpus_import"]["selected_records"] == 2


@pytest.mark.parametrize(
    "seal_stage",
    ["records_renamed", "manifest_written", "checkpoint_removed"],
)
def test_wdc_import_recovers_each_interrupted_seal_stage(
    tmp_path: Path,
    seal_stage: str,
) -> None:
    corpus, categories, _source_lines = _fixture_snapshots(tmp_path)
    index_path = tmp_path / "categories.sqlite3"
    build_wdc_category_index(categories, index_path=index_path)
    output_root = tmp_path / "quarantine"
    completed = import_wdc_research_candidates(
        corpus,
        category_index_path=index_path,
        output_root=output_root,
    )
    assert completed.output_path is not None
    assert completed.manifest_path is not None
    final_root = completed.output_path.parent
    work_root = final_root.parent / ".work" / final_root.name
    work_root.parent.mkdir(parents=True, exist_ok=True)
    os.replace(final_root, work_root)

    manifest_path = work_root / "manifest.json"
    records_path = work_root / "records.jsonl"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    chain = bytes(32)
    with records_path.open("rb") as handle:
        for line in handle:
            chain = hashlib.sha256(chain + line).digest()
    checkpoint = {
        "schema_version": WDC_RESEARCH_MANIFEST_SCHEMA,
        "state": "working",
        "source_sha256": manifest["source_sha256"],
        "category_source_sha256": manifest["category_source_sha256"],
        "policy_sha256": manifest["policy_sha256"],
        "completed_source_lines": manifest["completed_source_lines"],
        "selected_records": manifest["selected_records"],
        "rejected_non_computer_category": manifest["rejected_non_computer_category"],
        "rejected_not_component_like": manifest["rejected_not_component_like"],
        "rejected_ambiguous_component": manifest["rejected_ambiguous_component"],
        "output_bytes": manifest["output_bytes"],
        "output_chain_sha256": chain.hex(),
    }
    if seal_stage in {"records_renamed", "manifest_written"}:
        (work_root / "checkpoint.json").write_text(
            json.dumps(checkpoint),
            encoding="utf-8",
        )
    if seal_stage == "records_renamed":
        manifest_path.unlink()

    recovered = import_wdc_research_candidates(
        corpus,
        category_index_path=index_path,
        output_root=output_root,
    )
    assert recovered.complete is True
    assert recovered.output_path is not None and recovered.output_path.is_file()
    assert recovered.manifest_path is not None and recovered.manifest_path.is_file()
    assert not work_root.exists()
