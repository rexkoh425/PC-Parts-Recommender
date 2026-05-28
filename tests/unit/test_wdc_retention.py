from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pipelines.retention.wdc import WDCRetentionError, maintain_wdc_research_retention
from pipelines.sources.base import RAW_SNAPSHOT_SCHEMA_VERSION
from pipelines.sources.wdc_products import (
    WDC_CATEGORY_INDEX_SCHEMA,
    WDC_CATEGORY_SOURCE_NAME,
    WDC_CORPUS_SOURCE_NAME,
    WDC_PARSER_VERSION,
    WDC_RESEARCH_MANIFEST_SCHEMA,
    WDC_RESEARCH_RECORD_SCHEMA,
    WDC_RESEARCH_RETENTION_DAYS,
    WDC_SELECTION_POLICY_VERSION,
)
from scripts.maintain_wdc_research_retention import main as retention_main

_RETRIEVED_AT = datetime(2025, 7, 1, 12, tzinfo=UTC)
_EXPIRED_NOW = _RETRIEVED_AT + timedelta(days=WDC_RESEARCH_RETENTION_DAYS + 1)


def _write_raw_snapshot(
    raw_root: Path,
    *,
    source_name: str,
    body: bytes,
    retrieved_at: datetime = _RETRIEVED_AT,
) -> tuple[str, Path, Path]:
    digest = hashlib.sha256(body).hexdigest()
    root = raw_root / source_name
    root.mkdir(parents=True)
    raw_path = root / f"{digest}.json"
    receipt_path = root / f"{raw_path.name}.metadata.json"
    raw_path.write_bytes(body)
    receipt_path.write_text(
        json.dumps(
            {
                "schema_version": RAW_SNAPSHOT_SCHEMA_VERSION,
                "source_name": source_name,
                "source_url": f"https://example.invalid/{source_name}",
                "source_type": "historical_research_fixture",
                "retrieved_at": retrieved_at.isoformat(),
                "content_sha256": digest,
                "byte_count": len(body),
                "media_type": "application/x-ndjson",
                "parser_version": WDC_PARSER_VERSION,
                "licence_or_access_note": "Research-only fixture.",
                "raw_file": raw_path.name,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return digest, raw_path, receipt_path


def _write_category_index(path: Path, *, source_sha256: str, deadline: datetime) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            "CREATE TABLE import_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        connection.executemany(
            "INSERT INTO import_metadata(key, value) VALUES (?, ?)",
            (
                ("schema_version", WDC_CATEGORY_INDEX_SCHEMA),
                ("parser_version", WDC_PARSER_VERSION),
                ("source_sha256", source_sha256),
                ("retention_deadline", deadline.isoformat()),
            ),
        )
        connection.commit()
    finally:
        connection.close()


def _write_sealed_run(
    output_root: Path,
    *,
    corpus_sha256: str,
    category_sha256: str,
    retrieved_at: datetime = _RETRIEVED_AT,
) -> Path:
    run = output_root / WDC_CORPUS_SOURCE_NAME / ("a" * 32)
    run.mkdir(parents=True)
    records = run / "records.jsonl"
    records.write_bytes(b'{"research_only":true}\n')
    records_sha256 = hashlib.sha256(records.read_bytes()).hexdigest()
    (run / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": WDC_RESEARCH_MANIFEST_SCHEMA,
                "status": "complete",
                "source_sha256": corpus_sha256,
                "category_source_sha256": category_sha256,
                "policy_sha256": "c" * 64,
                "parser_version": WDC_PARSER_VERSION,
                "record_schema_version": WDC_RESEARCH_RECORD_SCHEMA,
                "selection_policy_version": WDC_SELECTION_POLICY_VERSION,
                "source_snapshot": {
                    "source_name": WDC_CORPUS_SOURCE_NAME,
                    "content_sha256": corpus_sha256,
                    "retrieved_at": retrieved_at.isoformat(),
                },
                "retention_deadline": (
                    retrieved_at + timedelta(days=WDC_RESEARCH_RETENTION_DAYS)
                ).isoformat(),
                "quarantine": {
                    "production_eligible": False,
                    "singapore_market_evidence": False,
                    "current_price_or_stock_evidence": False,
                    "model_training_eligible": False,
                    "published_metric_claim_eligible": False,
                },
                "output_sha256": records_sha256,
                "output_bytes": records.stat().st_size,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return run


def _write_working_run(output_root: Path, *, corpus_sha256: str, category_sha256: str) -> Path:
    run = output_root / WDC_CORPUS_SOURCE_NAME / ".work" / ("b" * 32)
    run.mkdir(parents=True)
    (run / "checkpoint.json").write_text(
        json.dumps(
            {
                "schema_version": WDC_RESEARCH_MANIFEST_SCHEMA,
                "state": "working",
                "source_sha256": corpus_sha256,
                "category_source_sha256": category_sha256,
                "policy_sha256": "d" * 64,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return run


def _expired_fixture(tmp_path: Path) -> dict[str, Path]:
    raw_root = tmp_path / "raw"
    output_root = tmp_path / "quarantine"
    corpus_sha256, corpus_raw, corpus_receipt = _write_raw_snapshot(
        raw_root,
        source_name=WDC_CORPUS_SOURCE_NAME,
        body=b"corpus fixture",
    )
    category_sha256, category_raw, category_receipt = _write_raw_snapshot(
        raw_root,
        source_name=WDC_CATEGORY_SOURCE_NAME,
        body=b"category fixture",
    )
    index = output_root / "wdc-products-category-index.sqlite3"
    index.parent.mkdir(parents=True)
    deadline = _RETRIEVED_AT + timedelta(days=WDC_RESEARCH_RETENTION_DAYS)
    _write_category_index(index, source_sha256=category_sha256, deadline=deadline)
    Path(f"{index}-wal").write_bytes(b"sidecar")
    sealed = _write_sealed_run(
        output_root,
        corpus_sha256=corpus_sha256,
        category_sha256=category_sha256,
    )
    working = _write_working_run(
        output_root,
        corpus_sha256=corpus_sha256,
        category_sha256=category_sha256,
    )
    return {
        "raw_root": raw_root,
        "output_root": output_root,
        "index": index,
        "corpus_raw": corpus_raw,
        "corpus_receipt": corpus_receipt,
        "category_raw": category_raw,
        "category_receipt": category_receipt,
        "sealed": sealed,
        "working": working,
    }


def test_wdc_retention_dry_run_and_execution_only_remove_expired_validated_artifacts(
    tmp_path: Path,
) -> None:
    paths = _expired_fixture(tmp_path)
    preserve = paths["output_root"] / WDC_CORPUS_SOURCE_NAME / "operator-note.txt"
    preserve.write_text("do not manage this unrelated file", encoding="utf-8")

    dry_run = maintain_wdc_research_retention(
        raw_root=paths["raw_root"],
        output_root=paths["output_root"],
        category_index=paths["index"],
        now=_EXPIRED_NOW,
        dry_run=True,
    )

    assert dry_run.raw_receipts_scanned == 2
    assert dry_run.raw_pairs_eligible == 2
    assert dry_run.raw_pairs_removed == 0
    assert dry_run.category_index_eligible is True
    assert dry_run.category_index_removed is False
    assert dry_run.sealed_runs_scanned == 1
    assert dry_run.sealed_runs_eligible == 1
    assert dry_run.working_runs_scanned == 1
    assert dry_run.working_runs_eligible == 1
    assert dry_run.unrelated_entries_preserved == 1
    assert paths["corpus_raw"].exists()
    assert paths["sealed"].exists()
    assert paths["working"].exists()

    completed = maintain_wdc_research_retention(
        raw_root=paths["raw_root"],
        output_root=paths["output_root"],
        category_index=paths["index"],
        now=_EXPIRED_NOW,
    )

    assert completed.raw_pairs_removed == 2
    assert completed.category_index_removed is True
    assert completed.sealed_runs_removed == 1
    assert completed.working_runs_removed == 1
    assert not paths["corpus_raw"].exists()
    assert not paths["corpus_receipt"].exists()
    assert not paths["category_raw"].exists()
    assert not paths["category_receipt"].exists()
    assert not paths["index"].exists()
    assert not Path(f"{paths['index']}-wal").exists()
    assert not paths["sealed"].exists()
    assert not paths["working"].exists()
    assert preserve.exists()


def test_wdc_retention_preserves_current_research_artifacts(tmp_path: Path) -> None:
    paths = _expired_fixture(tmp_path)
    before_deadline = _RETRIEVED_AT + timedelta(days=WDC_RESEARCH_RETENTION_DAYS - 1)

    report = maintain_wdc_research_retention(
        raw_root=paths["raw_root"],
        output_root=paths["output_root"],
        category_index=paths["index"],
        now=before_deadline,
    )

    assert report.raw_pairs_eligible == 0
    assert report.category_index_eligible is False
    assert report.sealed_runs_eligible == 0
    assert report.working_runs_eligible == 0
    assert paths["corpus_raw"].exists()
    assert paths["sealed"].exists()
    assert paths["working"].exists()


def test_wdc_retention_fails_closed_before_deletion_for_an_unknown_work_entry(
    tmp_path: Path,
) -> None:
    paths = _expired_fixture(tmp_path)
    (paths["working"] / "unrecognised.bin").write_bytes(b"must preserve")

    with pytest.raises(WDCRetentionError, match="unknown entries"):
        maintain_wdc_research_retention(
            raw_root=paths["raw_root"],
            output_root=paths["output_root"],
            category_index=paths["index"],
            now=_EXPIRED_NOW,
        )

    assert paths["corpus_raw"].exists()
    assert paths["corpus_receipt"].exists()
    assert paths["index"].exists()
    assert paths["sealed"].exists()
    assert (paths["working"] / "unrecognised.bin").exists()


def test_wdc_retention_cli_reports_a_dry_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    paths = _expired_fixture(tmp_path)

    status = retention_main(
        [
            "--raw-root",
            str(paths["raw_root"]),
            "--output-root",
            str(paths["output_root"]),
            "--category-index",
            str(paths["index"]),
            "--now",
            _EXPIRED_NOW.isoformat(),
            "--dry-run",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert status == 0
    assert payload["status"] == "ok"
    assert payload["dry_run"] is True
    assert payload["raw_pairs_eligible"] == 2
    assert paths["corpus_raw"].exists()
