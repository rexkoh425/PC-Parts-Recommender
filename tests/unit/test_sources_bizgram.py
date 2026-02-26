from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pipelines.checks.quality import evaluate_batch_quality
from pipelines.sources import bizgram_pdf
from pipelines.sources.base import FetchedSnapshot
from pipelines.sources.bizgram_pdf import (
    BIZGRAM_DATA_USE_RIGHTS,
    BIZGRAM_EXPECTED_SHA256,
    BIZGRAM_PARSER_VERSION,
    BIZGRAM_SOURCE_URL,
    BizgramControlledPDFAdapter,
)

from pc_build_recommender.data_rights import production_catalog_rights_are_valid

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BIZGRAM_PDF = REPOSITORY_ROOT / "tmp" / "pdfs" / "bizgram-2026-07-21.pdf"


class _FakePage:
    def __init__(self, text: str) -> None:
        self._text = text

    def extract_text(self) -> str:
        return self._text


class _FakeReader:
    def __init__(self, pages: list[str]) -> None:
        self.pages = [_FakePage(page) for page in pages]


def _snapshot(tmp_path: Path) -> FetchedSnapshot:
    return FetchedSnapshot(
        source_name="bizgram_controlled_pdf",
        source_url=BIZGRAM_SOURCE_URL,
        source_type="retailer",
        retrieved_at=datetime(2026, 7, 22, tzinfo=UTC),
        content_sha256=BIZGRAM_EXPECTED_SHA256,
        byte_count=512_000,
        media_type="application/pdf",
        parser_version=BIZGRAM_PARSER_VERSION,
        licence_or_access_note="controlled test fixture",
        path=tmp_path / "fixture.pdf",
        metadata_path=tmp_path / "fixture.pdf.metadata.json",
    )


def _fixture_pages() -> list[str]:
    return [
        "\n".join(
            (
                "BIZGRAM ASIA #05-50",
                "Asus STRIX RTX Nvidia",
                "ASUS DUAL RTX 5070 12GB ........999",
                "Header without a price",
            )
        ),
        "\n".join(
            (
                "Price for Bundles Only",
                "AMD 7  Motherboard and CPU DDR5",
                "Bundle row ........1200",
            )
        ),
        "\n".join(
            (
                "AMD Radeon Graphic Card",
                "PSU / SMPS / Power Supply  by Bizgram",
                "ASROCK RX 9070 XT 16GB ........899",
                "ASROCK RX 9070 XT 16GB ........899",
                "CORSAIR NAUTILUS AIO CPU COOLER ........105",
                "ASUS TUF 850W GOLD PSU ........139",
                "DEEPCOOL MATREXX 30 MATX Casing ........45",
                "ASUS RTX 5070 / MSI RTX 5070 ........900",
                "TP-Link Router ........79",
                "Broken Product ........PROMO",
            )
        ),
        "\n".join(
            (
                "Thermal Grizzly",
                "NVIDIA RTX PRO",
                "NVIDIA RTX PRO 6000 96GB ........26999",
            )
        ),
        "TP-Link Deco BE65 Pro\nTP-Link Router ........729",
        "Dlink Wifi / Wired Networking before GST\nD-link Switch ........29",
        "Ubiquiti UACC-Cable-Patch-Outdoor\nUbiquiti Cable ........22",
        "\n".join(
            (
                "Motorola Zebra DS2208 Scanner",
                "Silverstone PS16B",
                "Intel H61 OEM Motherboard 3rd Gen ........109",
                "Silverstone PS16B mATX Casing ........59",
                "Silverstone FTZ01-B Mini-ITX motherboard ........199",
            )
        ),
        "Fortune PC\nBizgram AU3\nComplete PC ........2705",
    ]


def test_bizgram_terminal_price_and_component_guards() -> None:
    parsed = BizgramControlledPDFAdapter._parse_terminal_price(
        "ASUS DUAL RTX 5070 12GB ........S$ 1,299.90"
    )
    assert parsed is not None
    assert parsed[0] == "ASUS DUAL RTX 5070 12GB"
    assert str(parsed[1]) == "1299.90"
    assert BizgramControlledPDFAdapter._parse_terminal_price("GPU 1299") is None
    assert BizgramControlledPDFAdapter._parse_terminal_price("GPU ........PROMO") is None
    assert BizgramControlledPDFAdapter._parse_terminal_price("GPU ........0") is None

    assert BizgramControlledPDFAdapter._category_from_title("ASUS DUAL RTX 5070 12GB") == "gpu"
    assert (
        BizgramControlledPDFAdapter._category_from_title("Intel H61 OEM Motherboard")
        == "motherboard"
    )
    assert (
        BizgramControlledPDFAdapter._category_from_title("Silverstone Mini-ITX motherboard") is None
    )
    assert (
        BizgramControlledPDFAdapter._title_rejection_reason("Tecware case with 250W PSU", 3)
        == "ambiguous_multi_product_or_inline_price"
    )


def test_bizgram_parse_is_deterministic_bounded_and_rights_quarantined(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pages = _fixture_pages()
    monkeypatch.setattr(bizgram_pdf, "PdfReader", lambda _path: _FakeReader(pages))
    adapter = BizgramControlledPDFAdapter(raw_root=tmp_path / "raw")
    snapshot = _snapshot(tmp_path)

    first = adapter.parse(snapshot)
    second = adapter.parse(snapshot)

    assert first.records == second.records
    assert first.rejected == second.rejected
    assert first.accepted_count == 7
    assert first.statistics["accepted_by_category"] == {
        "case": 2,
        "cooler": 1,
        "gpu": 2,
        "motherboard": 1,
        "power_supply": 1,
    }
    assert first.statistics["rejections_dropped_due_to_budget"] == 0
    assert first.statistics["network_fetch_used"] is False
    assert first.statistics["stock_asserted"] is False
    reasons = {item["reason"] for item in first.rejected}
    assert "exact_duplicate_source_offer" in reasons
    assert "bundle_or_matrix_offer" in reasons
    assert "ambiguous_multi_product_or_inline_price" in reasons
    assert "non_numeric_or_ambiguous_terminal_price" in reasons
    assert "excluded_professional_or_server_component" in reasons
    assert "unsupported_or_ambiguous_component_category" in reasons

    for record in first.records:
        assert record["training_eligible"] is False
        assert record["published_claims_eligible"] is False
        assert record["normalisation_metadata"]["development_only"] is True
        assert record["data"]["listing"]["stock_status"] == "unknown"
        assert record["data"]["price_snapshot"]["stock_status"] == "unknown"
        assert record["archive_snapshot_sha256"] == BIZGRAM_EXPECTED_SHA256
        assert len(record["raw_record_sha256"]) == 64
        assert production_catalog_rights_are_valid(record["data_use_rights"]) is False
        assert not any(
            value for key, value in record["data_use_rights"].items() if key.startswith("may_")
        )

    first.records[0]["training_eligible"] = True
    quality = evaluate_batch_quality(first, maximum_rejection_rate=1.0)
    controlled_check = next(
        check for check in quality.checks if check["name"] == "controlled_import_use_restricted"
    )
    assert quality.status == "fail"
    assert controlled_check["count"] == 1


def test_bizgram_fetch_is_local_fingerprinted_and_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b"%PDF-1.7\ncontrolled fixture\n"
    source = tmp_path / "bizgram.pdf"
    source.write_bytes(payload)
    fingerprint = hashlib.sha256(payload).hexdigest()
    monkeypatch.setattr(bizgram_pdf, "BIZGRAM_EXPECTED_SHA256", fingerprint)
    adapter = BizgramControlledPDFAdapter(raw_root=tmp_path / "raw")

    first = adapter.fetch(pdf_path=source)
    second = adapter.fetch(pdf_path=source)

    assert first.content_sha256 == fingerprint
    assert first.reused is False
    assert second.reused is True
    assert first.path == second.path
    assert not hasattr(adapter, "fetch_url")


def test_bizgram_wrong_fingerprint_is_rejected_before_snapshot(
    tmp_path: Path,
) -> None:
    source = tmp_path / "bizgram.pdf"
    source.write_bytes(b"%PDF-1.7\nunreviewed fixture\n")
    raw_root = tmp_path / "raw"

    with pytest.raises(ValueError, match="fingerprint changed"):
        BizgramControlledPDFAdapter(raw_root=raw_root).fetch(pdf_path=source)

    assert not raw_root.exists()
    assert production_catalog_rights_are_valid(BIZGRAM_DATA_USE_RIGHTS) is False


@pytest.mark.slow
@pytest.mark.skipif(not BIZGRAM_PDF.exists(), reason="controlled PDF is not present")
def test_current_bizgram_pdf_has_reviewed_measured_counts(tmp_path: Path) -> None:
    adapter = BizgramControlledPDFAdapter(raw_root=tmp_path / "raw")
    batch = adapter.parse(adapter.fetch(pdf_path=BIZGRAM_PDF))

    assert batch.accepted_count == 192
    assert batch.rejected_count == 4_257
    assert batch.statistics["accepted_by_category"] == {
        "case": 6,
        "cooler": 24,
        "gpu": 139,
        "motherboard": 2,
        "power_supply": 21,
    }
    assert batch.statistics["rejections_dropped_due_to_budget"] == 0
