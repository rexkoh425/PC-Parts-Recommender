from __future__ import annotations

from pathlib import Path

import pytest
from pipelines.sources.dynacore_pdf import DynacoreControlledPDFAdapter

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DYNACORE_PDF = REPOSITORY_ROOT / "tmp" / "pdfs" / "dynacore-2026-07-17.pdf"


def test_dynacore_price_and_category_guards() -> None:
    assert DynacoreControlledPDFAdapter._strict_price("599") == 599
    assert DynacoreControlledPDFAdapter._strict_price("PROMO") is None
    assert DynacoreControlledPDFAdapter._strict_price("#REF!") is None
    assert DynacoreControlledPDFAdapter._strict_price("0") is None
    assert DynacoreControlledPDFAdapter._category_from_title("850W ATX3.1 PSU") == ("power_supply")
    assert DynacoreControlledPDFAdapter._category_from_title("Phantom Spirit cooler") == ("cooler")


def test_dynacore_overlay_price_character_detection() -> None:
    characters = [
        {"text": "M", "x0": 817.9},
        {"text": "o", "x0": 822.2},
        {"text": "d", "x0": 825.3},
        {"text": "3", "x0": 818.52},
        {"text": "4", "x0": 821.76},
        {"text": "0", "x0": 825.00},
    ]
    assert (
        DynacoreControlledPDFAdapter._price_character_start(characters, 818.52, tolerance=0.25) == 3
    )


@pytest.mark.slow
@pytest.mark.skipif(not DYNACORE_PDF.exists(), reason="controlled PDF is not present")
def test_current_dynacore_pdf_has_reviewed_measured_counts(tmp_path) -> None:
    adapter = DynacoreControlledPDFAdapter(raw_root=tmp_path / "raw")
    batch = adapter.parse(adapter.fetch(pdf_path=DYNACORE_PDF))

    assert batch.accepted_count == 485
    assert batch.statistics["manual_review_count"] == 69
    assert batch.statistics["hard_rejected_count"] == 68
    assert batch.statistics["rejection_counts"]["invalid_ref_cell"] == 67
    assert batch.statistics["accepted_by_category"] == {
        "case": 110,
        "cooler": 57,
        "gpu": 158,
        "memory": 76,
        "power_supply": 68,
        "storage": 16,
    }
    assert all(record["training_eligible"] is False for record in batch.records)
    assert all(record["published_claims_eligible"] is False for record in batch.records)
    assert all(
        not any(value for key, value in record["data_use_rights"].items() if key.startswith("may_"))
        for record in batch.records
    )
    assert all(record["data"]["listing"]["stock_status"] == "unknown" for record in batch.records)
