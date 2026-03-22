from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from pc_build_recommender.catalog import (
    REVIEW_EVIDENCE_RECORD_TYPE,
    REVIEW_EVIDENCE_SCHEMA_VERSION,
    load_review_evidence,
)

NOW = datetime(2026, 7, 22, tzinfo=UTC)
PRODUCT_ID = "prod_fixture_gpu"
SOURCE_URL = "https://reviews.example.test/fixture-gpu"


def _rights(*, may_display: bool = True) -> dict[str, object]:
    return {
        "contract_reference": "fixture-review-display-v1",
        "contract_version_url": "https://reviews.example.test/licence",
        "consent_effective_on": "2026-01-01",
        "consent_expires_on": None,
        "retention_days": None,
        "deletion_required_on_termination": True,
        "deletion_sla_days": 30,
        "territories": ["SG"],
        "may_display": may_display,
        "may_cache": True,
        "may_store_history": True,
        "may_redistribute": False,
        "may_embed": False,
        "may_train": False,
        "may_derive": True,
    }


def _record(
    *,
    evidence_id: str = "review-noise-1",
    aspect: str = "noise",
    sentiment: float = -0.8,
    product_id: str = PRODUCT_ID,
    may_display: bool = True,
    source_url: str = SOURCE_URL,
) -> dict[str, object]:
    return {
        "schema_version": REVIEW_EVIDENCE_SCHEMA_VERSION,
        "record_type": REVIEW_EVIDENCE_RECORD_TYPE,
        "data": {
            "evidence_id": evidence_id,
            "product_id": product_id,
            "aspect": aspect,
            "sentiment": sentiment,
            "evidence_text": "The fixture cooler becomes audible under sustained load.",
            "source_url": source_url,
            "published_at": NOW.isoformat(),
            "confidence": 0.92,
        },
        "data_use_rights": _rights(may_display=may_display),
        "provenance": {
            "source_name": "fixture_permitted_reviews",
            "source_url": source_url,
            "retrieved_at": NOW.isoformat(),
            "raw_content_hash": "a" * 64,
            "parser_version": "fixture-review-parser-v1",
            "licence_or_access_note": "Fixture contract allows cited Singapore display.",
        },
    }


def _write(path: Path, *records: dict[str, object]) -> None:
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def test_review_artifact_accepts_only_permitted_cited_known_product_evidence(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "reviews.jsonl"
    _write(
        artifact,
        _record(evidence_id="review-performance-1", aspect="performance", sentiment=0.8),
        _record(),
    )

    loaded = load_review_evidence(artifact, known_product_ids=[PRODUCT_ID])

    assert [item.evidence_id for item in loaded] == [
        "review-noise-1",
        "review-performance-1",
    ]
    assert loaded[0].published_at == NOW
    assert loaded[0].source_url == SOURCE_URL


@pytest.mark.parametrize(
    ("record", "message"),
    [
        (_record(may_display=False), "lacks active SG display"),
        (_record(product_id="prod_unknown"), "unknown canonical product"),
        (_record(aspect="generic_sentiment"), "aspect must be one of"),
        (_record(source_url="http://reviews.example.test/fixture-gpu"), "HTTPS URL"),
    ],
)
def test_review_artifact_fails_closed_on_unsafe_or_unlicensed_evidence(
    tmp_path: Path,
    record: dict[str, object],
    message: str,
) -> None:
    artifact = tmp_path / "reviews.jsonl"
    _write(artifact, record)

    with pytest.raises(ValueError, match=message):
        load_review_evidence(artifact, known_product_ids=[PRODUCT_ID])


def test_review_artifact_rejects_duplicate_evidence_ids(tmp_path: Path) -> None:
    artifact = tmp_path / "reviews.jsonl"
    _write(
        artifact,
        _record(),
        _record(aspect="performance", sentiment=0.8),
    )

    with pytest.raises(ValueError, match="duplicate review evidence ID"):
        load_review_evidence(artifact, known_product_ids=[PRODUCT_ID])
