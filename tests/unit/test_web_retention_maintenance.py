from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pipelines.retention.web as retention
import pytest
from pipelines.retention.publication import (
    WEB_PUBLICATION_CONTROL_DIRECTORY,
    begin_web_processed_publication,
)
from pipelines.retention.web import (
    WEB_PROCESSED_RETENTION_RECEIPT,
    WEB_PROCESSED_RETENTION_SCHEMA_VERSION,
    WEB_RAW_METADATA_SCHEMA_VERSION,
    WebRetentionError,
    maintain_web_retention,
)
from scripts.maintain_web_retention import main

NOW = datetime(2030, 1, 15, 12, 0, tzinfo=UTC)
SOURCE_NAME = "fixture_web_production"
SOURCE_URL = "https://shop.example.test/products/gpu"
POLICY_FINGERPRINT = "a" * 64


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _raw_body_name(body: bytes) -> str:
    url_digest = _digest(SOURCE_URL.encode("utf-8"))
    return f"{url_digest[:32]}-{_digest(body)}.html"


def _write_raw_receipt(
    *,
    raw_root: Path,
    body: bytes,
    retrieved_at: datetime,
    nonce: str,
    write_body: bool = True,
    overrides: dict[str, Any] | None = None,
) -> tuple[Path, Path]:
    pages = raw_root / SOURCE_NAME / "pages"
    pages.mkdir(parents=True, exist_ok=True)
    url_digest = _digest(SOURCE_URL.encode("utf-8"))
    content_digest = _digest(body)
    body_path = pages / _raw_body_name(body)
    if write_body:
        body_path.write_bytes(body)
    receipt_path = pages / (
        f"{url_digest[:32]}-{content_digest}-{POLICY_FINGERPRINT[:16]}-{nonce}.json"
    )
    payload: dict[str, Any] = {
        "schema_version": WEB_RAW_METADATA_SCHEMA_VERSION,
        "source_name": SOURCE_NAME,
        "source_url": SOURCE_URL,
        "source_url_sha256": url_digest,
        "final_url": SOURCE_URL,
        "source_type": "retailer",
        "retrieved_at": retrieved_at.isoformat(),
        "retention_expires_at": (retrieved_at + timedelta(days=7)).isoformat(),
        "content_sha256": content_digest,
        "byte_count": len(body),
        "media_type": "text/html",
        "parser_version": "schemaorg-product-offer-v1",
        "licence_or_access_note": "Fixture governed-web acquisition.",
        "policy_fingerprint": POLICY_FINGERPRINT,
        "usage_scope": "production_catalog",
        "acquisition_authority": {
            "authority_reference": "fixture-authority-v1",
            "reviewed_on": "2029-01-01",
            "expires_on": None,
            "permits_automated_retrieval": True,
            "permits_raw_snapshot_storage": True,
            "permits_internal_analysis": True,
            "retention_days": 7,
            "deletion_required": True,
        },
        "data_use_rights": {
            "contract_reference": "fixture-rights-v1",
            "contract_version_url": "https://shop.example.test/terms",
            "consent_effective_on": "2029-01-01",
            "consent_expires_on": None,
            "retention_days": 7,
            "deletion_required_on_termination": True,
            "deletion_sla_days": 1,
            "territories": ["SG"],
            "may_display": True,
            "may_cache": True,
            "may_store_history": True,
            "may_redistribute": False,
            "may_embed": False,
            "may_train": False,
            "may_derive": True,
        },
        "etag": None,
        "last_modified": None,
        "raw_file": body_path.name,
    }
    if overrides:
        payload.update(overrides)
    receipt_path.write_text(json.dumps(payload), encoding="utf-8")
    return receipt_path, body_path


def _write_processed_run(
    *,
    processed_root: Path,
    digest: str,
    retrieved_at: datetime,
    usage_scope: str = "production_catalog",
    deletion_required: bool = True,
    write_receipt: bool = True,
) -> Path:
    run = processed_root / SOURCE_NAME / digest
    run.mkdir(parents=True)
    records = run / "records.jsonl"
    rejections = run / "rejections.jsonl"
    records.write_text("{}\n", encoding="utf-8")
    rejections.write_text("", encoding="utf-8")
    manifest_payload = {
        "schema_version": "pc-build-recommender.processed-batch.v1",
        "source_name": SOURCE_NAME,
        "source_snapshot_sha256": digest,
        "accepted_count": 1,
        "rejected_count": 0,
        "statistics": {},
        "files": {
            "records.jsonl": {
                "sha256": _digest(records.read_bytes()),
                "byte_count": records.stat().st_size,
            },
            "rejections.jsonl": {
                "sha256": _digest(rejections.read_bytes()),
                "byte_count": rejections.stat().st_size,
            },
        },
    }
    manifest_payload["content_sha256"] = _digest(
        json.dumps(
            manifest_payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    )
    (run / "manifest.json").write_text(
        json.dumps(manifest_payload),
        encoding="utf-8",
    )
    (run / "data-quality.json").write_text(
        json.dumps(
            {
                "schema_version": "pc-build-recommender.data-quality.v1",
                "source_name": SOURCE_NAME,
                "snapshot_sha256": digest,
                "status": "pass",
                "accepted_count": 1,
                "rejected_count": 0,
                "rejection_rate": 0.0,
                "checks": [],
                "record_type_counts": {},
                "category_counts": {},
                "eligibility_counts": {},
                "source_statistics": {},
            }
        ),
        encoding="utf-8",
    )
    if write_receipt:
        payload = {
            "schema_version": WEB_PROCESSED_RETENTION_SCHEMA_VERSION,
            "source_name": SOURCE_NAME,
            "run_sha256": digest,
            "manifest_sha256": _digest((run / "manifest.json").read_bytes()),
            "data_quality_sha256": _digest((run / "data-quality.json").read_bytes()),
            "policy_fingerprint": POLICY_FINGERPRINT,
            "usage_scope": usage_scope,
            "created_at": (retrieved_at + timedelta(hours=1)).isoformat(),
            "retrieval_started_at": retrieved_at.isoformat(),
            "retrieval_completed_at": (retrieved_at + timedelta(minutes=30)).isoformat(),
            "retention_days": 7,
            "retention_expires_at": (retrieved_at + timedelta(days=7)).isoformat(),
            "deletion_required": deletion_required,
            "acquisition_authority": {
                "authority_reference": "fixture-authority-v1",
                "reviewed_on": "2029-01-01",
                "expires_on": None,
                "permits_automated_retrieval": True,
                "permits_raw_snapshot_storage": True,
                "permits_internal_analysis": True,
                "retention_days": 7,
                "deletion_required": deletion_required,
            },
            "data_use_rights": {
                "contract_reference": "fixture-rights-v1",
                "contract_version_url": "https://shop.example.test/terms",
                "consent_effective_on": "2029-01-01",
                "consent_expires_on": None,
                "retention_days": 7,
                "deletion_required_on_termination": True,
                "deletion_sla_days": 1,
                "territories": ["SG"],
                "may_display": True,
                "may_cache": True,
                "may_store_history": True,
                "may_redistribute": False,
                "may_embed": False,
                "may_train": False,
                "may_derive": True,
            },
        }
        (run / WEB_PROCESSED_RETENTION_RECEIPT).write_text(json.dumps(payload), encoding="utf-8")
    return run


def _maintain(
    tmp_path: Path,
    *,
    dry_run: bool = False,
    grace: timedelta = timedelta(hours=24),
) -> tuple[retention.SourceRetentionReport, ...]:
    return maintain_web_retention(
        raw_root=tmp_path / "raw",
        processed_root=tmp_path / "processed",
        source_names=(SOURCE_NAME,),
        now=NOW,
        orphan_grace=grace,
        dry_run=dry_run,
    )


def _update_json(path: Path, update: Callable[[dict[str, Any]], object]) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    update(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_maintenance_runs_after_acquisition_authority_expiry(tmp_path: Path) -> None:
    receipt, body = _write_raw_receipt(
        raw_root=tmp_path / "raw",
        body=b"expired page",
        retrieved_at=NOW - timedelta(days=8),
        nonce="1" * 12,
    )
    run = _write_processed_run(
        processed_root=tmp_path / "processed",
        digest="1" * 64,
        retrieved_at=NOW - timedelta(days=8),
    )
    cache = tmp_path / "raw" / SOURCE_NAME / "http-cache.json"
    cache.write_text(
        json.dumps(
            {
                "schema_version": "pc-build-recommender.web-crawl-cache.v1",
                "source_name": SOURCE_NAME,
                "policy_fingerprint": POLICY_FINGERPRINT,
                "entries": {},
            }
        ),
        encoding="utf-8",
    )

    report = _maintain(tmp_path)[0]

    assert not receipt.exists()
    assert not body.exists()
    assert not run.exists()
    assert not cache.exists()
    assert report.raw.expired_receipts_removed == 1
    assert report.raw.expired_bodies_removed == 1
    assert report.processed.expired_runs_removed == 1
    assert report.raw.cache_files_removed == 1


def test_stale_private_publication_operation_is_reclaimed_after_grace(tmp_path: Path) -> None:
    publication = begin_web_processed_publication(
        processed_root=tmp_path / "processed",
        source_name=SOURCE_NAME,
        run_sha256="a" * 64,
        created_at=NOW - timedelta(days=2),
    )

    report = _maintain(tmp_path)[0]

    assert not publication.operation_directory.exists()
    assert not (tmp_path / "processed" / WEB_PUBLICATION_CONTROL_DIRECTORY).exists()
    assert report.processed.publication_operations_scanned == 1
    assert report.processed.publication_operations_eligible == 1
    assert report.processed.publication_operations_removed == 1
    assert report.processed.publication_operations_in_grace == 0
    assert "publication-operation:" in " ".join(report.action_sample)


def test_recent_private_publication_operation_remains_in_grace(tmp_path: Path) -> None:
    publication = begin_web_processed_publication(
        processed_root=tmp_path / "processed",
        source_name=SOURCE_NAME,
        run_sha256="b" * 64,
        created_at=NOW - timedelta(hours=1),
    )

    report = _maintain(tmp_path)[0]

    assert publication.operation_directory.is_dir()
    assert report.processed.publication_operations_scanned == 1
    assert report.processed.publication_operations_eligible == 0
    assert report.processed.publication_operations_removed == 0
    assert report.processed.publication_operations_in_grace == 1


def test_unknown_private_publication_content_blocks_all_retention_deletions(tmp_path: Path) -> None:
    receipt, body = _write_raw_receipt(
        raw_root=tmp_path / "raw",
        body=b"expired page",
        retrieved_at=NOW - timedelta(days=8),
        nonce="8" * 12,
    )
    publication = begin_web_processed_publication(
        processed_root=tmp_path / "processed",
        source_name=SOURCE_NAME,
        run_sha256="c" * 64,
        created_at=NOW - timedelta(days=2),
    )
    (publication.operation_directory / "unexpected.bin").write_bytes(b"operator review required")

    with pytest.raises(WebRetentionError, match="unknown root entries"):
        _maintain(tmp_path)

    assert receipt.is_file()
    assert body.is_file()
    assert publication.operation_directory.is_dir()


def test_unreceipted_production_run_fails_closed_before_any_delete(tmp_path: Path) -> None:
    receipt, body = _write_raw_receipt(
        raw_root=tmp_path / "raw",
        body=b"must remain after planning failure",
        retrieved_at=NOW - timedelta(days=8),
        nonce="2" * 12,
    )
    run = _write_processed_run(
        processed_root=tmp_path / "processed",
        digest="2" * 64,
        retrieved_at=NOW,
        write_receipt=False,
    )

    with pytest.raises(WebRetentionError, match="has no retention receipt"):
        _maintain(tmp_path)

    assert receipt.is_file()
    assert body.is_file()
    assert run.is_dir()


def test_old_orphan_body_and_recognised_crash_file_are_removed_only_after_grace(
    tmp_path: Path,
) -> None:
    pages = tmp_path / "raw" / SOURCE_NAME / "pages"
    pages.mkdir(parents=True)
    body_name = _raw_body_name(b"orphan")
    orphan = pages / body_name
    orphan.write_bytes(b"orphan")
    crash = pages / f".{body_name}.deadbeef.tmp"
    crash.write_bytes(b"partial")
    source_crash = pages.parent / ".http-cache.json.cafebabe.tmp"
    source_crash.write_bytes(b"partial cache")
    unrelated = pages / "operator-note.txt"
    unrelated.write_text("preserve", encoding="utf-8")
    old_timestamp = (NOW - timedelta(days=2)).timestamp()
    os.utime(orphan, (old_timestamp, old_timestamp))
    os.utime(crash, (old_timestamp, old_timestamp))
    os.utime(source_crash, (old_timestamp, old_timestamp))
    os.utime(unrelated, (old_timestamp, old_timestamp))

    report = _maintain(tmp_path)[0]

    assert not orphan.exists()
    assert not crash.exists()
    assert not source_crash.exists()
    assert unrelated.read_text(encoding="utf-8") == "preserve"
    assert report.raw.orphan_bodies_removed == 1
    assert report.raw.crash_leftovers_removed == 2
    assert report.raw.unrelated_files_preserved == 1


def test_recent_orphan_is_held_in_grace_window(tmp_path: Path) -> None:
    pages = tmp_path / "raw" / SOURCE_NAME / "pages"
    pages.mkdir(parents=True)
    orphan = pages / _raw_body_name(b"recent orphan")
    orphan.write_bytes(b"recent orphan")
    recent_timestamp = (NOW - timedelta(hours=2)).timestamp()
    os.utime(orphan, (recent_timestamp, recent_timestamp))

    report = _maintain(tmp_path)[0]

    assert orphan.is_file()
    assert report.raw.orphan_bodies_in_grace == 1
    assert report.raw.orphan_bodies_removed == 0


def test_body_shared_by_expired_and_active_receipts_is_preserved(tmp_path: Path) -> None:
    expired_receipt, body = _write_raw_receipt(
        raw_root=tmp_path / "raw",
        body=b"shared body",
        retrieved_at=NOW - timedelta(days=8),
        nonce="3" * 12,
    )
    active_receipt, active_body = _write_raw_receipt(
        raw_root=tmp_path / "raw",
        body=b"shared body",
        retrieved_at=NOW - timedelta(days=1),
        nonce="4" * 12,
    )

    report = _maintain(tmp_path)[0]

    assert not expired_receipt.exists()
    assert active_receipt.is_file()
    assert body == active_body
    assert body.is_file()
    assert report.raw.shared_bodies_preserved == 1
    assert report.raw.expired_bodies_removed == 0


def test_legacy_active_receipt_filename_preserves_its_body(tmp_path: Path) -> None:
    receipt, body = _write_raw_receipt(
        raw_root=tmp_path / "raw",
        body=b"legacy shared body",
        retrieved_at=NOW - timedelta(days=1),
        nonce="9" * 12,
    )
    legacy_receipt = receipt.with_name(receipt.name.replace(f"-{'9' * 12}.json", ".json"))
    receipt.replace(legacy_receipt)

    report = _maintain(tmp_path)[0]

    assert legacy_receipt.is_file()
    assert body.is_file()
    assert report.raw.active_receipts == 1
    assert report.raw.orphan_bodies_detected == 0


def test_malicious_raw_file_path_is_rejected_without_touching_outside_file(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside.html"
    outside.write_text("must survive", encoding="utf-8")
    receipt, _body = _write_raw_receipt(
        raw_root=tmp_path / "raw",
        body=b"malicious receipt",
        retrieved_at=NOW - timedelta(days=8),
        nonce="5" * 12,
        overrides={"raw_file": "../../outside.html"},
    )

    with pytest.raises(WebRetentionError, match="unsafe raw_file"):
        _maintain(tmp_path)

    assert receipt.is_file()
    assert outside.read_text(encoding="utf-8") == "must survive"


def test_linklike_processed_run_is_rejected_and_confined(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = _write_processed_run(
        processed_root=tmp_path / "processed",
        digest="6" * 64,
        retrieved_at=NOW - timedelta(days=8),
    )
    original = retention._is_linklike
    monkeypatch.setattr(retention, "_is_linklike", lambda path: path == run or original(path))

    with pytest.raises(WebRetentionError, match="symlink or junction"):
        _maintain(tmp_path)

    assert (run / "records.jsonl").is_file()


def test_processed_run_with_nested_directory_is_rejected_without_traversal(
    tmp_path: Path,
) -> None:
    run = _write_processed_run(
        processed_root=tmp_path / "processed",
        digest="9" * 64,
        retrieved_at=NOW - timedelta(days=8),
    )
    nested = run / "unexpected"
    nested.mkdir()
    (nested / "outside-contract.txt").write_text("preserve", encoding="utf-8")

    with pytest.raises(WebRetentionError, match="non-file entry|non-regular entry"):
        _maintain(tmp_path)

    assert (nested / "outside-contract.txt").read_text(encoding="utf-8") == "preserve"


def test_processed_run_mount_point_is_rejected_before_deletion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = _write_processed_run(
        processed_root=tmp_path / "processed",
        digest="8" * 64,
        retrieved_at=NOW - timedelta(days=8),
    )
    original = os.path.ismount
    monkeypatch.setattr(os.path, "ismount", lambda path: Path(path) == run or original(path))

    with pytest.raises(WebRetentionError, match="escaped|regular directory"):
        _maintain(tmp_path)

    assert run.is_dir()


def test_interrupted_processed_delete_resumes_from_tombstone(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = _write_processed_run(
        processed_root=tmp_path / "processed",
        digest="4" * 64,
        retrieved_at=NOW - timedelta(days=8),
    )
    source_root = run.parent
    original_unlink = Path.unlink
    deletion_calls = 0

    def flaky_unlink(path: Path, *args: object, **kwargs: object) -> None:
        nonlocal deletion_calls
        if path.parent.name.endswith(".deleting"):
            deletion_calls += 1
            if deletion_calls == 2:
                raise PermissionError("injected mid-delete failure")
        original_unlink(path, *args, **kwargs)

    with monkeypatch.context() as patch_context:
        patch_context.setattr(Path, "unlink", flaky_unlink)
        with pytest.raises(PermissionError, match="injected mid-delete"):
            _maintain(tmp_path)

    assert not run.exists()
    tombstones = tuple(source_root.glob(".*.deleting"))
    assert len(tombstones) == 1

    report = _maintain(tmp_path)[0]

    assert not tombstones[0].exists()
    assert report.processed.expired_runs_removed == 1
    assert report.processed.deletion_tombstones_resumed == 1


def test_tombstone_name_alone_cannot_authorize_deleting_an_active_run(
    tmp_path: Path,
) -> None:
    run = _write_processed_run(
        processed_root=tmp_path / "processed",
        digest="3" * 64,
        retrieved_at=NOW - timedelta(days=1),
    )
    tombstone = run.with_name(f".{run.name}.{'a' * 24}.deleting")
    run.rename(tombstone)

    with pytest.raises(WebRetentionError, match="not authorized for deletion"):
        _maintain(tmp_path)

    assert tombstone.is_dir()
    assert (tombstone / "records.jsonl").is_file()


def test_processed_receipt_must_require_deletion(tmp_path: Path) -> None:
    run = _write_processed_run(
        processed_root=tmp_path / "processed",
        digest="7" * 64,
        retrieved_at=NOW - timedelta(days=8),
        deletion_required=False,
    )

    with pytest.raises(WebRetentionError, match="must require deletion"):
        _maintain(tmp_path)

    assert run.is_dir()


def test_dry_run_reports_without_deleting(tmp_path: Path) -> None:
    receipt, body = _write_raw_receipt(
        raw_root=tmp_path / "raw",
        body=b"dry run",
        retrieved_at=NOW - timedelta(days=8),
        nonce="8" * 12,
    )

    report = _maintain(tmp_path, dry_run=True)[0]

    assert receipt.is_file()
    assert body.is_file()
    assert report.dry_run is True
    assert report.raw.expired_receipts_eligible == 1
    assert report.raw.expired_receipts_removed == 0


def test_cli_emits_machine_readable_result(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(
        [
            "--source-name",
            SOURCE_NAME,
            "--raw-root",
            str(tmp_path / "raw"),
            "--processed-root",
            str(tmp_path / "processed"),
            "--now",
            NOW.isoformat(),
            "--dry-run",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["status"] == "ok"
    assert payload["dry_run"] is True
    assert payload["sources"][0]["source_name"] == SOURCE_NAME


def test_copied_processed_receipt_cannot_delete_a_different_run(tmp_path: Path) -> None:
    expired = _write_processed_run(
        processed_root=tmp_path / "processed",
        digest="a" * 64,
        retrieved_at=NOW - timedelta(days=8),
    )
    active = _write_processed_run(
        processed_root=tmp_path / "processed",
        digest="b" * 64,
        retrieved_at=NOW - timedelta(days=1),
    )
    (active / WEB_PROCESSED_RETENTION_RECEIPT).write_bytes(
        (expired / WEB_PROCESSED_RETENTION_RECEIPT).read_bytes()
    )

    with pytest.raises(WebRetentionError, match="run_sha256 does not match"):
        _maintain(tmp_path)

    assert expired.is_dir()
    assert active.is_dir()


def test_tampered_processed_data_blocks_automatic_deletion(tmp_path: Path) -> None:
    run = _write_processed_run(
        processed_root=tmp_path / "processed",
        digest="9" * 64,
        retrieved_at=NOW - timedelta(days=8),
    )
    (run / "records.jsonl").write_text('{"tampered":true}\n', encoding="utf-8")

    with pytest.raises(WebRetentionError, match="does not match its manifest"):
        _maintain(tmp_path)

    assert run.is_dir()


def test_undeclared_processed_file_blocks_automatic_deletion(tmp_path: Path) -> None:
    run = _write_processed_run(
        processed_root=tmp_path / "processed",
        digest="8" * 64,
        retrieved_at=NOW - timedelta(days=8),
    )
    (run / "operator-note.txt").write_text("preserve and stop", encoding="utf-8")

    with pytest.raises(WebRetentionError, match="undeclared files"):
        _maintain(tmp_path)

    assert run.is_dir()
    assert (run / "operator-note.txt").is_file()


@pytest.mark.parametrize(
    ("field_name", "value", "error"),
    [
        ("retention_days", 8, "expiry does not match retention_days"),
        (
            "retention_expires_at",
            (NOW + timedelta(days=30)).isoformat(),
            "expiry does not match retention_days",
        ),
    ],
)
def test_tampered_processed_retention_contract_blocks_all_deletion(
    tmp_path: Path,
    field_name: str,
    value: object,
    error: str,
) -> None:
    expired = _write_processed_run(
        processed_root=tmp_path / "processed",
        digest="c" * 64,
        retrieved_at=NOW - timedelta(days=8),
    )
    later = _write_processed_run(
        processed_root=tmp_path / "processed",
        digest="d" * 64,
        retrieved_at=NOW - timedelta(days=1),
    )
    _update_json(
        later / WEB_PROCESSED_RETENTION_RECEIPT,
        lambda payload: payload.__setitem__(field_name, value),
    )

    with pytest.raises(WebRetentionError, match=error):
        _maintain(tmp_path)

    assert expired.is_dir()
    assert later.is_dir()


def test_authority_expiry_is_an_independent_deletion_deadline(tmp_path: Path) -> None:
    retrieved_at = NOW - timedelta(days=2)
    receipt, body = _write_raw_receipt(
        raw_root=tmp_path / "raw",
        body=b"authority expired",
        retrieved_at=retrieved_at,
        nonce="a" * 12,
    )
    run = _write_processed_run(
        processed_root=tmp_path / "processed",
        digest="e" * 64,
        retrieved_at=retrieved_at,
    )
    authority_expiry = (NOW - timedelta(days=1)).date().isoformat()
    _update_json(
        receipt,
        lambda payload: payload["acquisition_authority"].__setitem__(
            "expires_on", authority_expiry
        ),
    )
    _update_json(
        run / WEB_PROCESSED_RETENTION_RECEIPT,
        lambda payload: payload["acquisition_authority"].__setitem__(
            "expires_on", authority_expiry
        ),
    )

    report = _maintain(tmp_path)[0]

    assert not receipt.exists()
    assert not body.exists()
    assert not run.exists()
    assert report.raw.expired_receipts_removed == 1
    assert report.processed.expired_runs_removed == 1


def test_rights_expiry_uses_the_recorded_deletion_sla(tmp_path: Path) -> None:
    retrieved_at = NOW - timedelta(days=5)
    receipt, body = _write_raw_receipt(
        raw_root=tmp_path / "raw",
        body=b"rights expired",
        retrieved_at=retrieved_at,
        nonce="b" * 12,
    )
    run = _write_processed_run(
        processed_root=tmp_path / "processed",
        digest="f" * 64,
        retrieved_at=retrieved_at,
    )
    rights_expiry = (NOW - timedelta(days=3)).date().isoformat()
    for path in (receipt, run / WEB_PROCESSED_RETENTION_RECEIPT):
        _update_json(
            path,
            lambda payload: (
                payload["data_use_rights"].__setitem__("consent_expires_on", rights_expiry),
                payload["data_use_rights"].__setitem__("deletion_sla_days", 1),
            ),
        )

    report = _maintain(tmp_path)[0]

    assert not receipt.exists()
    assert not body.exists()
    assert not run.exists()
    assert report.raw.expired_receipts_removed == 1
    assert report.processed.expired_runs_removed == 1


def test_processed_retrieval_after_authority_expiry_is_rejected(tmp_path: Path) -> None:
    run = _write_processed_run(
        processed_root=tmp_path / "processed",
        digest="0" * 64,
        retrieved_at=NOW - timedelta(days=1),
    )
    _update_json(
        run / WEB_PROCESSED_RETENTION_RECEIPT,
        lambda payload: payload["acquisition_authority"].__setitem__(
            "expires_on", (NOW - timedelta(days=2)).date().isoformat()
        ),
    )

    with pytest.raises(WebRetentionError, match="retrieved after acquisition authority expiry"):
        _maintain(tmp_path)

    assert run.is_dir()


def test_legacy_v1_receipt_blocks_automatic_deletion(tmp_path: Path) -> None:
    run = _write_processed_run(
        processed_root=tmp_path / "processed",
        digest="1" * 64,
        retrieved_at=NOW - timedelta(days=8),
    )
    _update_json(
        run / WEB_PROCESSED_RETENTION_RECEIPT,
        lambda payload: payload.__setitem__(
            "schema_version", "pc-build-recommender.web-processed-retention.v1"
        ),
    )

    with pytest.raises(WebRetentionError, match="unsupported processed retention receipt schema"):
        _maintain(tmp_path)

    assert run.is_dir()


def test_global_entry_budget_spans_sources_and_processed_trees(tmp_path: Path) -> None:
    first = _write_processed_run(
        processed_root=tmp_path / "processed",
        digest="2" * 64,
        retrieved_at=NOW - timedelta(days=8),
    )
    second = _write_processed_run(
        processed_root=tmp_path / "processed",
        digest="3" * 64,
        retrieved_at=NOW - timedelta(days=8),
    )

    with pytest.raises(WebRetentionError, match="global 17-entry work limit"):
        maintain_web_retention(
            raw_root=tmp_path / "raw",
            processed_root=tmp_path / "processed",
            source_names=(SOURCE_NAME,),
            now=NOW,
            maximum_entries=17,
            dry_run=False,
        )

    assert first.is_dir()
    assert second.is_dir()


def test_global_entry_budget_includes_private_publication_workspaces(tmp_path: Path) -> None:
    publication = begin_web_processed_publication(
        processed_root=tmp_path / "processed",
        source_name=SOURCE_NAME,
        run_sha256="d" * 64,
        created_at=NOW - timedelta(days=2),
    )

    with pytest.raises(WebRetentionError, match="global 1-entry work limit"):
        maintain_web_retention(
            raw_root=tmp_path / "raw",
            processed_root=tmp_path / "processed",
            source_names=(SOURCE_NAME,),
            now=NOW,
            maximum_entries=1,
            dry_run=False,
        )

    assert publication.operation_directory.is_dir()


@pytest.mark.parametrize("invalid_limit", [True, False, 3.0, "3"])
def test_global_entry_budget_requires_a_strict_integer(
    tmp_path: Path,
    invalid_limit: object,
) -> None:
    with pytest.raises(ValueError, match="maximum_entries"):
        maintain_web_retention(
            raw_root=tmp_path / "raw",
            processed_root=tmp_path / "processed",
            source_names=(SOURCE_NAME,),
            now=NOW,
            maximum_entries=invalid_limit,  # type: ignore[arg-type]
            dry_run=True,
        )


def test_research_receipt_with_downstream_grant_is_rejected(tmp_path: Path) -> None:
    receipt, body = _write_raw_receipt(
        raw_root=tmp_path / "raw",
        body=b"research grant mismatch",
        retrieved_at=NOW - timedelta(days=1),
        nonce="c" * 12,
        overrides={"usage_scope": "internal_research"},
    )
    _update_json(
        receipt,
        lambda payload: payload["data_use_rights"].__setitem__("may_display", True),
    )

    with pytest.raises(WebRetentionError, match="research scope contains downstream rights"):
        _maintain(tmp_path)

    assert receipt.is_file()
    assert body.is_file()
