from __future__ import annotations

import json

import pytest
from pipelines.sources.base import (
    SnapshotError,
    SnapshotTooLargeError,
    sha256_bytes,
    snapshot_local_file,
)


def test_local_snapshot_is_content_addressed_and_idempotent(tmp_path) -> None:
    source = tmp_path / "catalog.json"
    source.write_bytes(b'{"value":1}\n')
    raw_root = tmp_path / "raw"

    first = snapshot_local_file(
        source_name="fixture_catalog",
        source_url="fixture://catalog",
        source_type="import",
        source_path=source,
        raw_root=raw_root,
        parser_version="fixture-v1",
        licence_or_access_note="Test fixture only.",
        media_type="application/json",
    )
    metadata_before = first.metadata_path.read_bytes()
    second = snapshot_local_file(
        source_name="fixture_catalog",
        source_url="fixture://catalog",
        source_type="import",
        source_path=source,
        raw_root=raw_root,
        parser_version="fixture-v1",
        licence_or_access_note="Test fixture only.",
        media_type="application/json",
    )

    assert first.content_sha256 == sha256_bytes(source.read_bytes())
    assert first.path == second.path
    assert first.reused is False
    assert second.reused is True
    assert first.metadata_path.read_bytes() == metadata_before
    metadata = json.loads(first.metadata_path.read_text(encoding="utf-8"))
    assert metadata["content_sha256"] == first.content_sha256
    assert metadata["raw_file"] == first.path.name


def test_local_snapshot_enforces_expected_hash_before_publication(tmp_path) -> None:
    source = tmp_path / "catalog.csv"
    source.write_bytes(b"listing_id,price\n1,100\n")
    raw_root = tmp_path / "raw"

    with pytest.raises(SnapshotError, match="SHA-256 mismatch"):
        snapshot_local_file(
            source_name="authorized_feed",
            source_url="awin://advertisers/1/feeds/2",
            source_type="authorized_retailer_feed",
            source_path=source,
            raw_root=raw_root,
            parser_version="awin-v1",
            licence_or_access_note="Test fixture only.",
            expected_sha256="0" * 64,
            maximum_bytes=1024,
        )

    assert not list(raw_root.rglob("*.csv"))
    assert not list(raw_root.rglob("*.metadata.json"))


def test_local_snapshot_enforces_size_limit_before_publication(tmp_path) -> None:
    source = tmp_path / "catalog.csv"
    source.write_bytes(b"0123456789")
    raw_root = tmp_path / "raw"

    with pytest.raises(SnapshotTooLargeError, match="limit is 9"):
        snapshot_local_file(
            source_name="authorized_feed",
            source_url="awin://advertisers/1/feeds/2",
            source_type="authorized_retailer_feed",
            source_path=source,
            raw_root=raw_root,
            parser_version="awin-v1",
            licence_or_access_note="Test fixture only.",
            maximum_bytes=9,
        )

    assert not list(raw_root.rglob("*.csv"))
    assert not list(raw_root.rglob("*.metadata.json"))
