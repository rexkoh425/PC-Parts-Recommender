from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest
from pipelines.sources.base import ParsedBatch, RawSnapshot, SnapshotError, sha256_bytes
from pipelines.sources.pci_ids import (
    PCI_IDS_GZIP_URL,
    PCI_IDS_LICENSE,
    PCI_IDS_PLAIN_URL,
    PCI_IDS_SOURCE_NAME,
    PCI_IDS_USER_AGENT,
    PRIORITY_PC_VENDOR_IDS,
    PCIIDRepositoryAdapter,
    PCIIdsParseLimitError,
)
from scripts.fetch_open_data import main as fetch_open_data_main

PCI_IDS_FIXTURE = b"""#
# PCI ID Repository fixture
#
8086  Intel Corporation
\t1234  Fixture Graphics Device
\t\t1028 0001  Dell Fixture Board
10DE  NVIDIA Corporation
\t1ABC  Fixture Accelerator
C 03  Display controller
\t00  VGA compatible controller
\t\t00  VGA controller
"""


def _snapshot(
    tmp_path: Path,
    *,
    payload: bytes = PCI_IDS_FIXTURE,
    compressed: bool = False,
) -> RawSnapshot:
    tmp_path.mkdir(parents=True, exist_ok=True)
    source = tmp_path / ("pci.ids.gz" if compressed else "pci.ids")
    source.write_bytes(gzip.compress(payload, mtime=0) if compressed else payload)
    adapter = PCIIDRepositoryAdapter(raw_root=tmp_path / "raw")
    return adapter.fetch(
        snapshot_path=source,
        snapshot_format="gzip" if compressed else "plain",
    )


def test_pci_ids_parser_streams_vendor_device_and_subsystem_aliases(tmp_path: Path) -> None:
    adapter = PCIIDRepositoryAdapter(raw_root=tmp_path / "raw")
    snapshot = _snapshot(tmp_path)

    first = adapter.parse(snapshot, max_records=20)
    second = adapter.parse(snapshot, max_records=20)

    assert first.records == second.records
    assert first.accepted_count == 5
    assert first.rejected_count == 0
    assert [record["source_record_id"] for record in first.records] == [
        "pci:vendor:8086",
        "pci:device:8086:1234",
        "pci:subsystem:8086:1234:1028:0001",
        "pci:vendor:10de",
        "pci:device:10de:1abc",
    ]
    subsystem = first.records[2]
    assert subsystem["record_type"] == "hardware_identifier_alias"
    assert subsystem["training_eligible"] is False
    assert subsystem["published_claims_eligible"] is False
    assert "data_use_rights" not in subsystem
    assert subsystem["rights_metadata"] == {
        "rights_basis": "open_licence",
        "licence": "BSD-3-Clause",
        "copyright_holders": "Martin Mares and Albert Pool",
        "third_party_notice": "docs/third-party/pci-id-repository-BSD-3-Clause.txt",
        "attribution_required": True,
        "may_display": True,
        "may_cache": True,
        "may_store_history": True,
        "may_redistribute": True,
        "may_derive": True,
        "may_embed": False,
        "may_train": False,
        "use_scope": "entity_resolution_blocking_only",
    }
    assert subsystem["provenance"]["licence"] == PCI_IDS_LICENSE
    assert subsystem["provenance"]["source_line"] == 6
    assert subsystem["data"] == {
        "namespace": "pci",
        "identifier_type": "subsystem",
        "identifiers": {
            "vendor_id": "8086",
            "device_id": "1234",
            "subsystem_vendor_id": "1028",
            "subsystem_device_id": "0001",
        },
        "canonical_label": "Dell Fixture Board",
        "aliases": ["Dell Fixture Board"],
        "use_scope": "entity_resolution_blocking_only",
        "authoritative_for": ["pci_identifier_to_label"],
        "not_authoritative_for": [
            "canonical_product_identity",
            "compatibility",
            "price",
            "stock",
            "performance",
        ],
    }
    assert subsystem["normalisation_metadata"]["compatibility_authoritative"] is False
    assert subsystem["normalisation_metadata"]["contains_price_or_stock"] is False
    assert first.statistics["accepted_by_identifier_type"] == {
        "vendor": 2,
        "device": 2,
        "subsystem": 1,
    }
    assert first.statistics["class_lines_ignored"] == 3
    assert first.statistics["streaming_parser"] is True


def test_pci_ids_gzip_and_plain_snapshots_have_equivalent_records(tmp_path: Path) -> None:
    adapter = PCIIDRepositoryAdapter(raw_root=tmp_path / "raw")
    plain = adapter.parse(_snapshot(tmp_path / "plain"), max_records=20)
    compressed = adapter.parse(
        _snapshot(tmp_path / "compressed", compressed=True),
        max_records=20,
    )

    def semantic_rows(batch: ParsedBatch) -> list[tuple[str, object]]:
        return [(str(record["source_record_id"]), record["data"]) for record in batch.records]

    assert semantic_rows(plain) == semantic_rows(compressed)


def test_pci_ids_parser_caps_records_and_rejects_malformed_lines(tmp_path: Path) -> None:
    malformed = b"""\t1234  Device without vendor
zzzz  Invalid vendor identifier
8086  Intel Corporation
\t1234  Valid Device
\t\t1028 0001  Valid Subsystem
\t\t1028 nope  Invalid Subsystem
8086  Duplicate Vendor
"""
    adapter = PCIIDRepositoryAdapter(raw_root=tmp_path / "raw")
    batch = adapter.parse(_snapshot(tmp_path, payload=malformed), max_records=3)

    assert batch.accepted_count == 3
    assert batch.statistics["record_limit_reached"] is True
    assert [row["reason"] for row in batch.rejected] == [
        "device_without_vendor",
        "malformed_vendor_line",
        "malformed_subsystem_line",
        "duplicate_identifier",
    ]
    assert [row["source_record_id"] for row in batch.records] == [
        "pci:vendor:8086",
        "pci:device:8086:1234",
        "pci:subsystem:8086:1234:1028:0001",
    ]


def test_pci_ids_bounded_selection_scans_full_file_and_keeps_major_vendor_anchors(
    tmp_path: Path,
) -> None:
    payload = b"""0001  Early Vendor One
\t0001  Early Device One
0002  Early Vendor Two
\t0002  Early Device Two
1002  Advanced Micro Devices, Inc.
\t1111  AMD Device
10de  NVIDIA Corporation
\t2222  NVIDIA Device
10ec  Realtek Semiconductor Co., Ltd.
\t3333  Realtek Device
8086  Intel Corporation
\t4444  Intel Device
"""
    adapter = PCIIDRepositoryAdapter(raw_root=tmp_path / "raw")
    snapshot = _snapshot(tmp_path, payload=payload)

    first = adapter.parse(snapshot, max_records=6)
    second = adapter.parse(snapshot, max_records=6)

    selected_ids = {str(record["source_record_id"]) for record in first.records}
    assert first.records == second.records
    assert first.accepted_count == 6
    assert {f"pci:vendor:{vendor_id}" for vendor_id in PRIORITY_PC_VENDOR_IDS}.issubset(
        selected_ids
    )
    assert first.statistics["lines_scanned"] == len(payload.splitlines())
    assert first.statistics["full_snapshot_scanned"] is True
    assert first.statistics["record_limit_reached"] is True
    assert first.statistics["priority_vendor_anchors_selected"] == list(PRIORITY_PC_VENDOR_IDS)
    assert first.statistics["selection_strategy"] == (
        "priority_vendor_anchors_plus_deterministic_stratified_hash_v1"
    )
    assert first.statistics["retained_candidate_capacity"] <= 2 * 6


def test_pci_ids_parser_audits_duplicate_invalid_utf8_and_oversized_lines(
    tmp_path: Path,
) -> None:
    payload = (
        b"8086  Intel Corporation\n"
        b"8086  Duplicate Intel\n"
        b"\t1234  Invalid UTF-8 \xff\n"
        b"\t1235  " + b"x" * 40 + b"\n"
        b"\t1236  Valid Device\n"
    )
    adapter = PCIIDRepositoryAdapter(raw_root=tmp_path / "raw")
    batch = adapter.parse(
        _snapshot(tmp_path, payload=payload),
        max_records=10,
        maximum_line_bytes=32,
    )

    assert [record["source_record_id"] for record in batch.records] == [
        "pci:vendor:8086",
        "pci:device:8086:1236",
    ]
    assert [row["reason"] for row in batch.rejected] == [
        "duplicate_identifier",
        "invalid_utf8",
        "line_too_long",
    ]
    assert batch.rejected[2]["details"]["byte_count"] > 32


def test_pci_ids_parser_rejects_empty_labels_without_leaking_context(tmp_path: Path) -> None:
    payload = b"8086    \n\t1234  Orphaned Device\n10de  NVIDIA Corporation\n"
    adapter = PCIIDRepositoryAdapter(raw_root=tmp_path / "raw")
    batch = adapter.parse(_snapshot(tmp_path, payload=payload), max_records=10)

    assert [row["reason"] for row in batch.rejected] == [
        "empty_label",
        "device_without_vendor",
    ]
    assert [row["source_record_id"] for row in batch.records] == ["pci:vendor:10de"]


def test_pci_ids_parser_fails_closed_on_decompression_budget(tmp_path: Path) -> None:
    adapter = PCIIDRepositoryAdapter(raw_root=tmp_path / "raw")
    snapshot = _snapshot(tmp_path, compressed=True)

    with pytest.raises(PCIIdsParseLimitError, match="decompressed bytes exceeded"):
        adapter.parse(snapshot, max_records=20, maximum_uncompressed_bytes=20)


def test_pci_ids_parser_bounds_rejection_memory_and_lines_scanned(tmp_path: Path) -> None:
    payload = b"".join(f"invalid-{index}\n".encode() for index in range(10))
    adapter = PCIIDRepositoryAdapter(raw_root=tmp_path / "raw")
    snapshot = _snapshot(tmp_path, payload=payload)

    batch = adapter.parse(
        snapshot,
        max_records=20,
        maximum_recorded_rejections=2,
    )
    assert batch.rejected_count == 2
    assert batch.statistics["total_rejections"] == 10
    assert batch.statistics["recorded_rejections_truncated"] == 8

    with pytest.raises(PCIIdsParseLimitError, match="lines scanned exceeded 3"):
        adapter.parse(snapshot, max_records=20, maximum_lines_scanned=3)


@pytest.mark.parametrize(
    ("snapshot_format", "expected_url", "expected_suffix", "maximum_bytes"),
    [
        ("gzip", PCI_IDS_GZIP_URL, ".ids.gz", 8 * 1024 * 1024),
        ("plain", PCI_IDS_PLAIN_URL, ".ids", 32 * 1024 * 1024),
    ],
)
def test_pci_ids_remote_fetch_uses_official_url_named_agent_and_hash_pin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    snapshot_format: str,
    expected_url: str,
    expected_suffix: str,
    maximum_bytes: int,
) -> None:
    captured: dict[str, object] = {}
    sentinel = object()

    def fake_fetch_http_snapshot(**kwargs: object) -> object:
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(
        "pipelines.sources.pci_ids.fetch_http_snapshot",
        fake_fetch_http_snapshot,
    )
    adapter = PCIIDRepositoryAdapter(raw_root=tmp_path / "raw")
    result = adapter.fetch(
        snapshot_format=snapshot_format,  # type: ignore[arg-type]
        expected_sha256="a" * 64,
    )

    assert result is sentinel
    assert captured["source_name"] == PCI_IDS_SOURCE_NAME
    assert captured["source_url"] == expected_url
    assert captured["suffix"] == expected_suffix
    assert captured["expected_sha256"] == "a" * 64
    assert captured["maximum_bytes"] == maximum_bytes
    assert captured["headers"] == {
        "User-Agent": PCI_IDS_USER_AGENT,
        "Accept-Encoding": "gzip",
    }


def test_pci_ids_local_snapshot_can_be_sha256_pinned(tmp_path: Path) -> None:
    source = tmp_path / "pci.ids"
    source.write_bytes(PCI_IDS_FIXTURE)
    adapter = PCIIDRepositoryAdapter(raw_root=tmp_path / "raw")
    expected = sha256_bytes(PCI_IDS_FIXTURE)

    snapshot = adapter.fetch(
        snapshot_path=source,
        snapshot_format="plain",
        expected_sha256=expected,
    )
    assert snapshot.content_sha256 == expected
    with pytest.raises(SnapshotError, match="SHA-256 mismatch"):
        adapter.fetch(
            snapshot_path=source,
            snapshot_format="plain",
            expected_sha256="0" * 64,
        )
    with pytest.raises(ValueError, match="64-character hexadecimal"):
        adapter.fetch(snapshot_format="gzip", expected_sha256="not-a-digest")


def test_pci_ids_parse_requires_its_own_snapshot(tmp_path: Path) -> None:
    snapshot = _snapshot(tmp_path)
    foreign = RawSnapshot(
        source_name="fixture_foreign",
        source_url=snapshot.source_url,
        source_type=snapshot.source_type,
        retrieved_at=snapshot.retrieved_at,
        content_sha256=snapshot.content_sha256,
        byte_count=snapshot.byte_count,
        media_type=snapshot.media_type,
        parser_version=snapshot.parser_version,
        licence_or_access_note=snapshot.licence_or_access_note,
        path=snapshot.path,
        metadata_path=snapshot.metadata_path,
    )

    with pytest.raises(ValueError, match="unexpected PCI IDs source"):
        PCIIDRepositoryAdapter(raw_root=tmp_path / "raw").parse(foreign)


def test_fetch_open_data_cli_materialises_bounded_pci_aliases(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "pci.ids"
    source.write_bytes(PCI_IDS_FIXTURE)

    exit_code = fetch_open_data_main(
        [
            "--source",
            "pci_ids",
            "--pci-ids-snapshot",
            str(source),
            "--pci-ids-format",
            "plain",
            "--pci-ids-sha256",
            sha256_bytes(PCI_IDS_FIXTURE),
            "--pci-ids-record-limit",
            "3",
            "--raw-root",
            str(tmp_path / "raw"),
            "--processed-root",
            str(tmp_path / "processed"),
            "--no-parquet",
        ]
    )

    assert exit_code == 0
    summary = json.loads(capsys.readouterr().out)
    source_summary = summary["sources"][0]
    assert source_summary["source_name"] == PCI_IDS_SOURCE_NAME
    assert source_summary["accepted_count"] == 3
    assert source_summary["statistics"]["record_limit"] == 3
    assert source_summary["statistics"]["record_limit_reached"] is True
    rows = [
        json.loads(line)
        for line in Path(source_summary["records_path"]).read_text(encoding="utf-8").splitlines()
    ]
    assert [row["source_record_id"] for row in rows] == [
        "pci:device:10de:1abc",
        "pci:vendor:10de",
        "pci:vendor:8086",
    ]
