from __future__ import annotations

import hashlib
import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest
from pipelines.retention.web import (
    WEB_PROCESSED_RETENTION_RECEIPT,
    WEB_PROCESSED_RETENTION_SCHEMA_VERSION,
    WebRetentionError,
    write_web_processed_retention_receipt,
)
from pipelines.sources.web_product import (
    WebAcquisitionAuthority,
    WebSourcePolicy,
    WebUsageScope,
)

from pc_build_recommender.data_rights import DataUseRights
from pc_build_recommender.domain.enums import ComponentCategory

SOURCE_NAME = "fixture_web_research"
TERMS_URL = "https://shop.example.test/terms"
PRODUCT_URL = "https://shop.example.test/products/gpu"
NOW = datetime(2026, 7, 22, 12, 0, tzinfo=UTC)


def _policy(*, retention_days: int = 7) -> WebSourcePolicy:
    rights = DataUseRights(
        contract_reference="fixture-research-review-v1",
        contract_version_url=TERMS_URL,
        consent_effective_on=date(2020, 1, 1),
        consent_expires_on=None,
        retention_days=retention_days,
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
    )
    authority = WebAcquisitionAuthority(
        authority_reference="fixture-internal-analysis-review-v1",
        reviewed_on=date(2020, 1, 1),
        expires_on=None,
        permits_automated_retrieval=True,
        permits_raw_snapshot_storage=True,
        permits_internal_analysis=True,
        retention_days=retention_days,
        deletion_required=True,
    )
    return WebSourcePolicy(
        source_name=SOURCE_NAME,
        retailer="Fixture Shop",
        allowed_hosts=("shop.example.test",),
        terms_url=TERMS_URL,
        terms_selector="#terms",
        canonical_terms_sha256="a" * 64,
        terms_verified_on=date(2020, 1, 1),
        licence_or_access_note="Fixture internal-research authority only.",
        rights=rights,
        acquisition_authority=authority,
        url_categories={PRODUCT_URL: ComponentCategory.GPU},
        usage_scope=WebUsageScope.INTERNAL_RESEARCH,
    )


def _run_directory(processed_root: Path, digest: str = "5" * 64) -> Path:
    run_directory = processed_root / SOURCE_NAME / digest
    run_directory.mkdir(parents=True)
    records = run_directory / "records.jsonl"
    rejections = run_directory / "rejections.jsonl"
    records.write_text("{}\n", encoding="utf-8")
    rejections.write_text("", encoding="utf-8")
    files = {
        "records.jsonl": {
            "sha256": hashlib.sha256(records.read_bytes()).hexdigest(),
            "byte_count": records.stat().st_size,
        },
        "rejections.jsonl": {
            "sha256": hashlib.sha256(rejections.read_bytes()).hexdigest(),
            "byte_count": rejections.stat().st_size,
        },
    }
    manifest_payload = {
        "schema_version": "pc-build-recommender.processed-batch.v1",
        "source_name": SOURCE_NAME,
        "source_snapshot_sha256": digest,
        "accepted_count": 1,
        "rejected_count": 0,
        "statistics": {},
        "files": files,
    }
    manifest_payload["content_sha256"] = hashlib.sha256(
        json.dumps(
            manifest_payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    (run_directory / "manifest.json").write_text(
        json.dumps(manifest_payload),
        encoding="utf-8",
    )
    (run_directory / "data-quality.json").write_text(
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
    return run_directory


def test_v2_receipt_binds_run_authority_rights_and_retention(tmp_path: Path) -> None:
    processed_root = tmp_path / "processed"
    policy = _policy(retention_days=7)
    run_directory = _run_directory(processed_root)
    started = NOW - timedelta(hours=2)
    completed = NOW - timedelta(hours=1)

    receipt_path = write_web_processed_retention_receipt(
        processed_root=processed_root,
        output_directory=run_directory,
        policy=policy,
        retrieval_started_at=started,
        retrieval_completed_at=completed,
        created_at=NOW,
    )

    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt_path == run_directory / WEB_PROCESSED_RETENTION_RECEIPT
    assert payload["schema_version"] == WEB_PROCESSED_RETENTION_SCHEMA_VERSION
    assert payload["run_sha256"] == run_directory.name
    assert payload["retention_days"] == 7
    assert payload["retention_expires_at"] == (started + timedelta(days=7)).isoformat()
    assert payload["acquisition_authority"] == policy.acquisition_authority.to_dict()
    assert payload["data_use_rights"] == policy.rights.to_dict()


def test_second_write_is_idempotent_and_cannot_reset_expiry(tmp_path: Path) -> None:
    processed_root = tmp_path / "processed"
    policy = _policy()
    run_directory = _run_directory(processed_root)
    started = NOW - timedelta(hours=2)
    completed = NOW - timedelta(hours=1)
    receipt = write_web_processed_retention_receipt(
        processed_root=processed_root,
        output_directory=run_directory,
        policy=policy,
        retrieval_started_at=started,
        retrieval_completed_at=completed,
        created_at=NOW,
    )
    before = receipt.read_bytes()

    repeated = write_web_processed_retention_receipt(
        processed_root=processed_root,
        output_directory=run_directory,
        policy=policy,
        retrieval_started_at=started,
        retrieval_completed_at=completed,
        created_at=NOW + timedelta(hours=1),
    )

    assert repeated == receipt
    assert receipt.read_bytes() == before


def test_conflicting_receipt_replay_is_rejected_without_rewrite(tmp_path: Path) -> None:
    processed_root = tmp_path / "processed"
    policy = _policy()
    run_directory = _run_directory(processed_root)
    receipt = write_web_processed_retention_receipt(
        processed_root=processed_root,
        output_directory=run_directory,
        policy=policy,
        retrieval_started_at=NOW - timedelta(hours=2),
        retrieval_completed_at=NOW - timedelta(hours=1),
        created_at=NOW,
    )
    before = receipt.read_bytes()

    with pytest.raises(WebRetentionError, match="conflicts"):
        write_web_processed_retention_receipt(
            processed_root=processed_root,
            output_directory=run_directory,
            policy=policy,
            retrieval_started_at=NOW - timedelta(hours=3),
            retrieval_completed_at=NOW - timedelta(hours=1),
            created_at=NOW,
        )

    assert receipt.read_bytes() == before


def test_writer_rejects_manifest_run_hash_mismatch(tmp_path: Path) -> None:
    processed_root = tmp_path / "processed"
    run_directory = _run_directory(processed_root)
    manifest = run_directory / "manifest.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["source_snapshot_sha256"] = "6" * 64
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(WebRetentionError, match="snapshot does not match"):
        write_web_processed_retention_receipt(
            processed_root=processed_root,
            output_directory=run_directory,
            policy=_policy(),
            retrieval_started_at=NOW - timedelta(hours=2),
            retrieval_completed_at=NOW - timedelta(hours=1),
            created_at=NOW,
        )


def test_writer_rejects_manifest_semantic_hash_tampering(tmp_path: Path) -> None:
    processed_root = tmp_path / "processed"
    run_directory = _run_directory(processed_root)
    manifest = run_directory / "manifest.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["accepted_count"] = 2
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(WebRetentionError, match="content_sha256 does not match"):
        write_web_processed_retention_receipt(
            processed_root=processed_root,
            output_directory=run_directory,
            policy=_policy(),
            retrieval_started_at=NOW - timedelta(hours=2),
            retrieval_completed_at=NOW - timedelta(hours=1),
            created_at=NOW,
        )


def test_existing_receipt_rejects_data_file_tampering(tmp_path: Path) -> None:
    processed_root = tmp_path / "processed"
    run_directory = _run_directory(processed_root)
    receipt = write_web_processed_retention_receipt(
        processed_root=processed_root,
        output_directory=run_directory,
        policy=_policy(),
        retrieval_started_at=NOW - timedelta(hours=2),
        retrieval_completed_at=NOW - timedelta(hours=1),
        created_at=NOW,
    )
    before = receipt.read_bytes()
    (run_directory / "records.jsonl").write_text('{"tampered":true}\n', encoding="utf-8")

    with pytest.raises(WebRetentionError, match="does not match its manifest"):
        write_web_processed_retention_receipt(
            processed_root=processed_root,
            output_directory=run_directory,
            policy=_policy(),
            retrieval_started_at=NOW - timedelta(hours=2),
            retrieval_completed_at=NOW - timedelta(hours=1),
            created_at=NOW,
        )

    assert receipt.read_bytes() == before


@pytest.mark.parametrize("entry_kind", ["file", "directory"])
def test_writer_rejects_undeclared_run_entries(tmp_path: Path, entry_kind: str) -> None:
    processed_root = tmp_path / "processed"
    run_directory = _run_directory(processed_root)
    extra = run_directory / "undeclared"
    if entry_kind == "file":
        extra.write_text("operator material", encoding="utf-8")
    else:
        extra.mkdir()

    with pytest.raises(WebRetentionError, match="undeclared"):
        write_web_processed_retention_receipt(
            processed_root=processed_root,
            output_directory=run_directory,
            policy=_policy(),
            retrieval_started_at=NOW - timedelta(hours=2),
            retrieval_completed_at=NOW - timedelta(hours=1),
            created_at=NOW,
        )

    assert not (run_directory / WEB_PROCESSED_RETENTION_RECEIPT).exists()
