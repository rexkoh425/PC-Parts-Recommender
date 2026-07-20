from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from pipelines.retention.legacy_web import (
    LEGACY_WEB_PROCESSED_RETENTION_SCHEMA_VERSION,
    LEGACY_WEB_RAW_METADATA_SCHEMA_VERSION,
    plan_legacy_web_retention_migration,
)
from pipelines.retention.web import WebRetentionError
from scripts.maintain_web_retention import main

NOW = datetime(2030, 1, 15, 12, 0, tzinfo=UTC)
SOURCE_NAME = "fixture_web_research"
SOURCE_URL = "https://shop.example.test/products/gpu"
POLICY_FINGERPRINT = "a" * 64


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_legacy_raw(
    root: Path,
    *,
    policy_fingerprint: str = POLICY_FINGERPRINT,
    body: bytes = b"legacy governed evidence",
    retrieved_at: datetime | None = None,
) -> tuple[Path, Path]:
    pages = root / SOURCE_NAME / "pages"
    pages.mkdir(parents=True)
    observed_at = retrieved_at or NOW - timedelta(days=8)
    url_sha = _digest(SOURCE_URL.encode("utf-8"))
    body_sha = _digest(body)
    body_path = pages / f"{url_sha[:32]}-{body_sha}.html"
    body_path.write_bytes(body)
    receipt_path = pages / (
        f"{url_sha[:32]}-{body_sha}-{policy_fingerprint[:16]}-{'1' * 12}.json"
    )
    payload: dict[str, Any] = {
        "schema_version": LEGACY_WEB_RAW_METADATA_SCHEMA_VERSION,
        "source_name": SOURCE_NAME,
        "source_url": SOURCE_URL,
        "source_url_sha256": url_sha,
        "final_url": SOURCE_URL,
        "source_type": "retailer",
        "retrieved_at": observed_at.isoformat(),
        "retention_expires_at": (observed_at + timedelta(days=7)).isoformat(),
        "content_sha256": body_sha,
        "byte_count": len(body),
        "media_type": "text/html",
        "parser_version": "schemaorg-product-offer-v1",
        "licence_or_access_note": "Fixture internal-research authority.",
        "policy_fingerprint": policy_fingerprint,
        "usage_scope": "internal_research",
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
        "etag": None,
        "last_modified": None,
        "raw_file": body_path.name,
    }
    receipt_path.write_text(json.dumps(payload), encoding="utf-8")
    return receipt_path, body_path


def _write_legacy_processed(
    root: Path,
    *,
    policy_fingerprint: str = POLICY_FINGERPRINT,
    run_sha: str = "b" * 64,
) -> Path:
    run = root / SOURCE_NAME / run_sha
    run.mkdir(parents=True)
    records = run / "records.jsonl"
    rejections = run / "rejections.jsonl"
    records.write_text("{}\n", encoding="utf-8")
    rejections.write_text("", encoding="utf-8")
    authority = {
        "authority_reference": "fixture-authority-v1",
        "reviewed_on": "2029-01-01",
        "expires_on": None,
        "permits_automated_retrieval": True,
        "permits_raw_snapshot_storage": True,
        "permits_internal_analysis": True,
        "retention_days": 7,
        "deletion_required": True,
    }
    rights = {
        "contract_reference": "fixture-rights-v1",
        "contract_version_url": "https://shop.example.test/terms",
        "consent_effective_on": "2029-01-01",
        "consent_expires_on": None,
        "retention_days": 7,
        "deletion_required_on_termination": True,
        "deletion_sla_days": 1,
        "territories": ["SG"],
        "may_display": False,
        "may_cache": False,
        "may_store_history": False,
        "may_redistribute": False,
        "may_embed": False,
        "may_train": False,
        "may_derive": False,
    }
    source_statistics = {
        "policy_fingerprint": policy_fingerprint,
        "usage_scope": "internal_research",
        "acquisition_authority": authority,
        "data_use_rights": rights,
        "pages_requested": 0,
        "robots_sha256_by_host": {},
    }
    files = {
        "records.jsonl": {
            "sha256": _digest(records.read_bytes()),
            "byte_count": records.stat().st_size,
        },
        "rejections.jsonl": {
            "sha256": _digest(rejections.read_bytes()),
            "byte_count": rejections.stat().st_size,
        },
    }
    manifest: dict[str, Any] = {
        "schema_version": "pc-build-recommender.processed-batch.v1",
        "source_name": SOURCE_NAME,
        "source_snapshot_sha256": run_sha,
        "accepted_count": 1,
        "rejected_count": 0,
        "statistics": source_statistics,
        "files": files,
    }
    manifest["content_sha256"] = _digest(
        json.dumps(
            manifest,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    )
    (run / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    quality = {
        "schema_version": "pc-build-recommender.data-quality.v1",
        "source_name": SOURCE_NAME,
        "snapshot_sha256": run_sha,
        "status": "pass",
        "accepted_count": 1,
        "rejected_count": 0,
        "rejection_rate": 0.0,
        "checks": [],
        "record_type_counts": {},
        "category_counts": {},
        "eligibility_counts": {},
        "source_statistics": source_statistics,
    }
    (run / "data-quality.json").write_text(json.dumps(quality), encoding="utf-8")
    retrieved_at = NOW - timedelta(days=8)
    receipt = {
        "schema_version": LEGACY_WEB_PROCESSED_RETENTION_SCHEMA_VERSION,
        "source_name": SOURCE_NAME,
        "policy_fingerprint": policy_fingerprint,
        "usage_scope": "internal_research",
        "created_at": (retrieved_at + timedelta(hours=1)).isoformat(),
        "retrieved_at": retrieved_at.isoformat(),
        "retention_expires_at": (retrieved_at + timedelta(days=7)).isoformat(),
        "deletion_required": True,
    }
    (run / "retention-receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    return run


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_plan_recovers_retained_evidence_and_performs_zero_writes(tmp_path: Path) -> None:
    _write_legacy_raw(tmp_path / "raw")
    _write_legacy_processed(tmp_path / "processed")
    before = _tree_bytes(tmp_path)

    report = plan_legacy_web_retention_migration(
        raw_root=tmp_path / "raw",
        processed_root=tmp_path / "processed",
        source_names=(SOURCE_NAME,),
        now=NOW,
    )[0]

    assert report.legacy_raw_receipts_scanned == 1
    assert report.legacy_processed_runs_scanned == 1
    assert report.expired_legacy_raw_receipts == 1
    assert report.expired_legacy_processed_runs == 1
    assert report.write_actions_planned == 0
    assert report.migration_required is True
    assert report.migration_ready is True
    assert len(report.evidence_sha256) == 64
    assert report.policy_plans[0].raw_authority_evidence_sha256 is not None
    assert report.policy_plans[0].processed_authority_evidence_sha256 is not None
    assert (
        report.policy_plans[0].processed_data_use_rights_evidence_sha256 is not None
    )
    assert (
        report.policy_plans[0].processed_retrieval_interval_evidence_sha256
        is not None
    )
    assert report.policy_plans[0].missing_evidence == ()
    assert report.blockers == ()
    assert _tree_bytes(tmp_path) == before


def test_processed_manifest_supplies_bound_authority_and_rights_without_raw_receipt(
    tmp_path: Path,
) -> None:
    _write_legacy_raw(tmp_path / "raw")
    missing_authority_policy = "c" * 64
    _write_legacy_processed(
        tmp_path / "processed",
        policy_fingerprint=missing_authority_policy,
    )

    report = plan_legacy_web_retention_migration(
        raw_root=tmp_path / "raw",
        processed_root=tmp_path / "processed",
        source_names=(SOURCE_NAME,),
        now=NOW,
    )[0]

    plan = next(
        item
        for item in report.policy_plans
        if item.policy_fingerprint == missing_authority_policy
    )
    assert plan.raw_authority_evidence_sha256 is None
    assert plan.processed_authority_evidence_sha256 is not None
    assert plan.processed_data_use_rights_evidence_sha256 is not None
    assert plan.processed_retrieval_interval_evidence_sha256 is None
    assert plan.missing_evidence == ("processed_retrieval_interval",)


def test_raw_only_policy_reports_missing_rights_evidence(tmp_path: Path) -> None:
    _write_legacy_raw(tmp_path / "raw")

    report = plan_legacy_web_retention_migration(
        raw_root=tmp_path / "raw",
        processed_root=tmp_path / "processed",
        source_names=(SOURCE_NAME,),
        now=NOW,
    )[0]

    assert report.policy_plans[0].missing_evidence == ("data_use_rights",)
    assert "no matching processed manifest evidence remains" in report.blockers[0]


def test_recovered_interval_with_different_retention_anchor_requires_review(
    tmp_path: Path,
) -> None:
    _write_legacy_raw(
        tmp_path / "raw",
        retrieved_at=NOW - timedelta(days=8, minutes=2),
    )
    _write_legacy_processed(tmp_path / "processed")

    report = plan_legacy_web_retention_migration(
        raw_root=tmp_path / "raw",
        processed_root=tmp_path / "processed",
        source_names=(SOURCE_NAME,),
        now=NOW,
    )[0]

    assert (
        report.policy_plans[0].processed_retrieval_interval_evidence_sha256
        is not None
    )
    assert report.policy_plans[0].missing_evidence == ("processed_retention_anchor",)
    assert "reviewed migration decision" in report.blockers[0]


def test_evidence_digest_is_stable_across_evaluation_times(tmp_path: Path) -> None:
    _write_legacy_raw(tmp_path / "raw")
    _write_legacy_processed(tmp_path / "processed")

    first = plan_legacy_web_retention_migration(
        raw_root=tmp_path / "raw",
        processed_root=tmp_path / "processed",
        source_names=(SOURCE_NAME,),
        now=NOW,
    )[0]
    later = plan_legacy_web_retention_migration(
        raw_root=tmp_path / "raw",
        processed_root=tmp_path / "processed",
        source_names=(SOURCE_NAME,),
        now=NOW + timedelta(days=1),
    )[0]

    assert first.evidence_sha256 == later.evidence_sha256


def test_manifest_policy_mismatch_blocks_even_after_rehash(tmp_path: Path) -> None:
    run = _write_legacy_processed(tmp_path / "processed")
    manifest_path = run / "manifest.json"
    quality_path = run / "data-quality.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    manifest["statistics"]["policy_fingerprint"] = "c" * 64
    quality["source_statistics"] = manifest["statistics"]
    semantic_manifest = dict(manifest)
    semantic_manifest.pop("content_sha256")
    manifest["content_sha256"] = _digest(
        json.dumps(
            semantic_manifest,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    quality_path.write_text(json.dumps(quality), encoding="utf-8")

    with pytest.raises(WebRetentionError, match="receipt policy fingerprint"):
        plan_legacy_web_retention_migration(
            raw_root=tmp_path / "raw",
            processed_root=tmp_path / "processed",
            source_names=(SOURCE_NAME,),
            now=NOW,
        )


def test_tampered_legacy_raw_body_blocks_the_plan(tmp_path: Path) -> None:
    _receipt, body = _write_legacy_raw(tmp_path / "raw")
    body.write_bytes(b"tampered")

    with pytest.raises(WebRetentionError, match="does not match its receipt"):
        plan_legacy_web_retention_migration(
            raw_root=tmp_path / "raw",
            processed_root=tmp_path / "processed",
            source_names=(SOURCE_NAME,),
            now=NOW,
        )


def test_cli_migration_plan_is_read_only_and_returns_blocked(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write_legacy_processed(tmp_path / "processed")
    before = _tree_bytes(tmp_path)

    exit_code = main(
        [
            "--source-name",
            SOURCE_NAME,
            "--raw-root",
            str(tmp_path / "raw"),
            "--processed-root",
            str(tmp_path / "processed"),
            "--plan-legacy-migration",
            "--now",
            NOW.isoformat(),
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 2
    assert payload["status"] == "blocked"
    assert payload["mode"] == "legacy-migration-plan"
    assert payload["dry_run"] is True
    assert payload["write_actions_planned"] == 0
    assert _tree_bytes(tmp_path) == before
