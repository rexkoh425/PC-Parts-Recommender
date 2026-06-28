"""Quarantined local import for one reviewed Bizgram price-list PDF.

The adapter deliberately has no network-fetch method.  It accepts only an exact,
reviewed PDF fingerprint and emits development-only offers whose downstream rights
are all denied.  The parser is intentionally conservative: a row needs a dotted
leader, one unambiguous terminal SGD price, and a page-appropriate component model.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from pypdf import PdfReader

from pc_build_recommender.domain.enums import ListingCondition, StockState
from pc_build_recommender.domain.models import PriceSample, RetailerOffering
from pipelines.parsing.normalizers import NORMALISED_RECORD_SCHEMA_VERSION, stable_identifier
from pipelines.sources.base import (
    ParsedBatch,
    RawSnapshot,
    rejected_record,
    sha256_bytes,
    sha256_file,
    snapshot_local_file,
)

BIZGRAM_EXPECTED_SHA256 = "011bcf9471b9aefb4747d7fd9486680af71282c67016dac87698a90ca5f76c12"
BIZGRAM_SOURCE_URL = "controlled-import://bizgram/2026-07-21"
BIZGRAM_PRICE_LIST_URL = "https://www.bizgram.com/pricelist-download/"
BIZGRAM_PARSER_VERSION = "bizgram-controlled-pdf-2026-07-21-layout-v1"
BIZGRAM_ACCESS_NOTE = (
    "Development-only local import. No open data licence or written downstream-use "
    "permission was established. Do not display, publish, redistribute, retain price "
    "history, embed, derive from, or train models on these rows without written permission "
    "from Bizgram. The adapter never downloads the PDF."
)
BIZGRAM_DOCUMENT_UPDATED_AT = datetime(
    2026,
    7,
    21,
    14,
    24,
    tzinfo=timezone(timedelta(hours=8)),
)

# This intentionally is not a complete DataUseRights contract: there is no consent
# document to cite.  Every capability is false, and strict production-rights parsing
# consequently fails closed as well.
BIZGRAM_DATA_USE_RIGHTS: dict[str, Any] = {
    "rights_status": "barred_no_written_permission",
    "consent_reference": None,
    "retention_days": 0,
    "deletion_required_on_termination": False,
    "deletion_sla_days": None,
    "may_display": False,
    "may_cache": False,
    "may_store_history": False,
    "may_redistribute": False,
    "may_embed": False,
    "may_train": False,
    "may_derive": False,
}

BIZGRAM_EXPECTED_PAGES = 9
BIZGRAM_MAX_PDF_BYTES = 2 * 1024 * 1024
BIZGRAM_MAX_PAGE_TEXT_CHARS = 100_000
BIZGRAM_MAX_LINES_PER_PAGE = 2_000
BIZGRAM_MAX_LINE_CHARS = 600
BIZGRAM_MAX_ACCEPTED_RECORDS = 1_000
BIZGRAM_MAX_REJECTIONS = 8_000

_LAYOUT_ANCHORS: dict[int, tuple[str, ...]] = {
    1: ("BIZGRAM ASIA #05-50", "Asus STRIX RTX Nvidia"),
    2: ("Price for Bundles Only", "AMD 7  Motherboard and CPU DDR5"),
    3: ("AMD Radeon Graphic Card", "PSU / SMPS / Power Supply  by Bizgram"),
    4: ("Thermal Grizzly", "NVIDIA RTX PRO"),
    5: ("TP-Link Deco BE65 Pro",),
    6: ("Dlink Wifi / Wired Networking before GST",),
    7: ("Ubiquiti UACC-Cable-Patch-Outdoor",),
    8: ("Motorola Zebra DS2208 Scanner", "Silverstone PS16B"),
    9: ("Fortune PC", "Bizgram AU3"),
}

_ALLOWED_CATEGORIES_BY_PAGE: dict[int, frozenset[str]] = {
    1: frozenset({"gpu"}),
    2: frozenset(),
    3: frozenset({"gpu", "cooler", "power_supply", "case"}),
    4: frozenset(),
    5: frozenset(),
    6: frozenset(),
    7: frozenset(),
    8: frozenset({"motherboard", "case"}),
    9: frozenset(),
}

_DOTTED_LEADER = re.compile(r"\.{3,}")
_TERMINAL_PRICE_LINE = re.compile(
    r"\A(?P<title>.+?)\s*\.{3,}\s*"
    r"(?:(?:SGD|S\$|\$)\s*)?"
    r"(?P<price>(?:[1-9]\d{0,2}(?:,\d{3})+|[1-9]\d{0,4})(?:\.\d{1,2})?)\s*\Z",
    re.IGNORECASE,
)
_GPU = re.compile(
    r"\b(?:"
    r"(?:GeForce\s+)?(?:RTX|GTX)\s*(?:PRO\W*)?\d{3,4}[A-Z0-9-]*"
    r"|(?:Radeon\s+)?RX\s*\d{4}[A-Z0-9-]*"
    r"|(?:Intel\s+)?Arc\s+[AB]\d{3}"
    r")\b",
    re.IGNORECASE,
)
_CPU = re.compile(
    r"\b(?:Ryzen\s+(?:[3579]\s+)?\d{4}[A-Z0-9-]*|Threadripper\s+[A-Z0-9-]+|"
    r"Core\s+(?:Ultra\s+)?(?:i[3579][ -])?\d{4,5}[A-Z0-9-]*)\b",
    re.IGNORECASE,
)
_MOTHERBOARD = re.compile(r"\b(?:motherboard|mainboard)\b", re.IGNORECASE)
_MOTHERBOARD_CHIPSET = re.compile(
    r"\b(?:H[678]\d{1,2}|B[568]\d{2}|Z[6789]\d{2}|X[568]\d{2}|WRX\d{2,3})[A-Z-]*\b",
    re.IGNORECASE,
)
_MEMORY = re.compile(r"\b(?:DDR[345]|SO-?DIMM|UDIMM)\b", re.IGNORECASE)
_STORAGE = re.compile(r"\b(?:SSD|NVMe|hard\s+disk|HDD)\b", re.IGNORECASE)
_POWER_SUPPLY = re.compile(r"\b(?:PSU|power\s+supply|SMPS)\b", re.IGNORECASE)
_COOLER = re.compile(
    r"\b(?:CPU\s+(?:air\s+|liquid\s+)?cooler|AIO\s+CPU(?:\s+cooler)?|"
    r"liquid\s+CPU\s+cooler|heatsink|Peerless\s+Assassin|Phantom\s+Spirit)\b",
    re.IGNORECASE,
)
_CASE = re.compile(r"\b(?:PC\s+case|computer\s+case|casing|chassis)\b", re.IGNORECASE)
_FRAGMENT = re.compile(r"\A(?:for|with|and|or|suitable|supports?)\b", re.IGNORECASE)
_MULTI_PRODUCT = re.compile(
    r"(?:\s/\s|\s\+\s|\b(?:bundle|combo|with\s+free|bring\s+your\s+own)\b|\$)",
    re.IGNORECASE,
)


class BizgramControlledPDFAdapter:
    """Parse a bounded high-confidence subset of one local, fingerprinted PDF."""

    def __init__(self, *, raw_root: str | Path) -> None:
        self.raw_root = Path(raw_root)

    def fetch(self, *, pdf_path: str | Path) -> RawSnapshot:
        """Snapshot a reviewed local PDF; this method performs no network request."""

        source_path = Path(pdf_path).resolve()
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        if source_path.suffix.casefold() != ".pdf":
            raise ValueError("Bizgram controlled import requires a local .pdf file")
        byte_count = source_path.stat().st_size
        if byte_count < 5 or byte_count > BIZGRAM_MAX_PDF_BYTES:
            raise ValueError(f"Bizgram PDF must be between 5 and {BIZGRAM_MAX_PDF_BYTES} bytes")
        with source_path.open("rb") as source:
            if source.read(5) != b"%PDF-":
                raise ValueError("Bizgram controlled import input is not a PDF")
        if sha256_file(source_path) != BIZGRAM_EXPECTED_SHA256:
            raise ValueError(
                "Bizgram PDF fingerprint changed; review a new layout profile before import"
            )

        snapshot = snapshot_local_file(
            source_name="bizgram_controlled_pdf",
            source_url=BIZGRAM_SOURCE_URL,
            source_type="retailer",
            source_path=source_path,
            raw_root=self.raw_root,
            parser_version=BIZGRAM_PARSER_VERSION,
            licence_or_access_note=BIZGRAM_ACCESS_NOTE,
            suffix=".pdf",
            media_type="application/pdf",
        )
        if snapshot.content_sha256 != BIZGRAM_EXPECTED_SHA256:
            raise ValueError("snapshotted Bizgram PDF fingerprint changed during import")
        return snapshot

    def parse(self, snapshot: RawSnapshot) -> ParsedBatch:
        self._validate_snapshot(snapshot)
        batch = ParsedBatch(
            source_name=snapshot.source_name,
            snapshot_sha256=snapshot.content_sha256,
        )
        candidates: list[dict[str, Any]] = []
        counters: dict[str, int] = defaultdict(int)
        reader = PdfReader(snapshot.path)
        if len(reader.pages) != BIZGRAM_EXPECTED_PAGES:
            raise ValueError(f"expected {BIZGRAM_EXPECTED_PAGES} pages, found {len(reader.pages)}")

        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if len(text) > BIZGRAM_MAX_PAGE_TEXT_CHARS:
                raise ValueError(f"page {page_number} exceeds the reviewed text-size budget")
            self._validate_layout_page(page_number, text)
            lines = text.splitlines()
            if len(lines) > BIZGRAM_MAX_LINES_PER_PAGE:
                raise ValueError(f"page {page_number} exceeds the reviewed line budget")

            for line_number, raw_line in enumerate(lines, start=1):
                counters["total_lines_seen"] += 1
                line = self._clean_text(raw_line)
                if not line or _DOTTED_LEADER.search(line) is None:
                    counters["ignored_non_dotted_lines"] += 1
                    continue
                counters["dotted_candidate_lines"] += 1
                source_record_id = f"p{page_number}-l{line_number:04d}"
                if len(line) > BIZGRAM_MAX_LINE_CHARS:
                    self._reject(
                        batch,
                        counters,
                        source_record_id,
                        "line_exceeds_reviewed_length",
                        raw_line_prefix=line[:200],
                        raw_line_sha256=sha256_bytes(line.encode("utf-8")),
                    )
                    continue

                parsed = self._parse_terminal_price(line)
                if parsed is None:
                    self._reject(
                        batch,
                        counters,
                        source_record_id,
                        "non_numeric_or_ambiguous_terminal_price",
                        raw_line=line,
                        page=page_number,
                    )
                    continue
                title, price, raw_price = parsed
                rejection_reason = self._title_rejection_reason(title, page_number)
                if rejection_reason is not None:
                    self._reject(
                        batch,
                        counters,
                        source_record_id,
                        rejection_reason,
                        title=title,
                        raw_price=raw_price,
                        page=page_number,
                    )
                    continue
                category = self._category_from_title(title)
                allowed_categories = _ALLOWED_CATEGORIES_BY_PAGE[page_number]
                if category is None or category not in allowed_categories:
                    self._reject(
                        batch,
                        counters,
                        source_record_id,
                        "unsupported_or_ambiguous_component_category",
                        title=title,
                        raw_price=raw_price,
                        page=page_number,
                    )
                    continue
                if len(candidates) >= BIZGRAM_MAX_ACCEPTED_RECORDS:
                    self._reject(
                        batch,
                        counters,
                        source_record_id,
                        "accepted_record_budget_exhausted",
                        title=title,
                        page=page_number,
                    )
                    continue
                candidates.append(
                    self._listing_record(
                        snapshot=snapshot,
                        source_record_id=source_record_id,
                        title=title,
                        price=price,
                        raw_price=raw_price,
                        raw_line=line,
                        page_number=page_number,
                        line_number=line_number,
                        category=category,
                    )
                )

        batch.records = self._quarantine_duplicates_and_conflicts(
            candidates,
            batch,
            counters,
        )
        category_counts: dict[str, int] = defaultdict(int)
        for record in batch.records:
            category_counts[str(record["normalisation_metadata"]["category"])] += 1
        rejection_counts: dict[str, int] = defaultdict(int)
        for rejected in batch.rejected:
            rejection_counts[str(rejected["reason"])] += 1
        batch.statistics = {
            "document_sha256": snapshot.content_sha256,
            "document_updated_at": BIZGRAM_DOCUMENT_UPDATED_AT.isoformat(),
            "page_count": BIZGRAM_EXPECTED_PAGES,
            "born_digital_text_layer": True,
            "ocr_used": False,
            "network_fetch_used": False,
            "stock_asserted": False,
            "training_eligible": False,
            "published_claims_eligible": False,
            "accepted_by_category": dict(sorted(category_counts.items())),
            "rejection_counts": dict(sorted(rejection_counts.items())),
            "rejections_dropped_due_to_budget": counters["rejections_dropped_due_to_budget"],
            "dotted_candidate_lines": counters["dotted_candidate_lines"],
            "ignored_non_dotted_lines": counters["ignored_non_dotted_lines"],
            "total_lines_seen": counters["total_lines_seen"],
            "limits": {
                "maximum_pdf_bytes": BIZGRAM_MAX_PDF_BYTES,
                "maximum_page_text_chars": BIZGRAM_MAX_PAGE_TEXT_CHARS,
                "maximum_lines_per_page": BIZGRAM_MAX_LINES_PER_PAGE,
                "maximum_line_chars": BIZGRAM_MAX_LINE_CHARS,
                "maximum_accepted_records": BIZGRAM_MAX_ACCEPTED_RECORDS,
                "maximum_rejections": BIZGRAM_MAX_REJECTIONS,
            },
        }
        return batch

    @staticmethod
    def _validate_snapshot(snapshot: RawSnapshot) -> None:
        if snapshot.source_name != "bizgram_controlled_pdf":
            raise ValueError("snapshot source is not the controlled Bizgram adapter")
        if snapshot.source_url != BIZGRAM_SOURCE_URL:
            raise ValueError("snapshot source URL does not match the reviewed Bizgram profile")
        if snapshot.parser_version != BIZGRAM_PARSER_VERSION:
            raise ValueError("snapshot parser version does not match the reviewed Bizgram profile")
        if snapshot.content_sha256 != BIZGRAM_EXPECTED_SHA256:
            raise ValueError("snapshot does not match the reviewed Bizgram PDF fingerprint")
        if snapshot.byte_count > BIZGRAM_MAX_PDF_BYTES:
            raise ValueError("snapshot exceeds the reviewed Bizgram byte budget")

    @staticmethod
    def _validate_layout_page(page_number: int, text: str) -> None:
        missing = [anchor for anchor in _LAYOUT_ANCHORS[page_number] if anchor not in text]
        if missing:
            raise ValueError(
                f"page {page_number} layout anchors do not match the reviewed profile: {missing}"
            )

    @staticmethod
    def _parse_terminal_price(line: str) -> tuple[str, Decimal, str] | None:
        match = _TERMINAL_PRICE_LINE.fullmatch(line)
        if match is None:
            return None
        title = BizgramControlledPDFAdapter._clean_text(match.group("title"))
        raw_price = match.group("price")
        if not title or len(title) < 6:
            return None
        try:
            price = Decimal(raw_price.replace(",", "")).quantize(Decimal("0.01"))
        except InvalidOperation:
            return None
        if price < Decimal("1.00") or price > Decimal("50000.00"):
            return None
        return title, price, raw_price

    @staticmethod
    def _title_rejection_reason(title: str, page_number: int) -> str | None:
        if page_number in {2, 9}:
            return "bundle_or_matrix_offer"
        if _FRAGMENT.search(title):
            return "wrapped_or_incomplete_title_fragment"
        if _MULTI_PRODUCT.search(title):
            return "ambiguous_multi_product_or_inline_price"
        if re.search(
            r"\b(?:eGPU|enclosure|adapter|cable|riser|tray|barebone|mini\s+(?:PC|desktop)|"
            r"desktop\s+computer|laptop|notebook)\b",
            title,
            re.IGNORECASE,
        ):
            return "excluded_accessory_or_complete_system"
        if re.search(r"\b(?:NVIDIA\s+)?RTX\s+PRO\b", title, re.IGNORECASE):
            return "excluded_professional_or_server_component"
        if re.search(r"(?:\bwith|\bw/)\s+.*\bPSU\b", title, re.IGNORECASE):
            return "ambiguous_multi_product_or_inline_price"
        return None

    @staticmethod
    def _category_from_title(title: str) -> str | None:
        categories: list[str] = []
        if _GPU.search(title):
            categories.append("gpu")
        if _CPU.search(title):
            categories.append("cpu")
        if _MOTHERBOARD.search(title) and _MOTHERBOARD_CHIPSET.search(title):
            categories.append("motherboard")
        if _MEMORY.search(title):
            categories.append("memory")
        if _STORAGE.search(title):
            categories.append("storage")
        if _POWER_SUPPLY.search(title):
            categories.append("power_supply")
        if _COOLER.search(title):
            categories.append("cooler")
        if _CASE.search(title):
            categories.append("case")
        return categories[0] if len(categories) == 1 else None

    def _listing_record(
        self,
        *,
        snapshot: RawSnapshot,
        source_record_id: str,
        title: str,
        price: Decimal,
        raw_price: str,
        raw_line: str,
        page_number: int,
        line_number: int,
        category: str,
    ) -> dict[str, Any]:
        listing_id = stable_identifier("listing_bizgram", source_record_id, length=32)
        product_id = stable_identifier("unmatched_product", "bizgram", title, category)
        listing = RetailerOffering(
            listing_id=listing_id,
            product_id=product_id,
            retailer="Bizgram",
            source_listing_id=source_record_id,
            title=title,
            condition=ListingCondition.NEW,
            currency="SGD",
            base_price=price,
            shipping_price=Decimal("0.00"),
            stock_status=StockState.UNKNOWN,
            seller_name="Bizgram",
            listing_url=BIZGRAM_PRICE_LIST_URL,
            first_seen_at=BIZGRAM_DOCUMENT_UPDATED_AT,
            last_seen_at=BIZGRAM_DOCUMENT_UPDATED_AT,
        )
        price_snapshot = PriceSample(
            snapshot_id=stable_identifier(
                "price_bizgram",
                listing_id,
                BIZGRAM_DOCUMENT_UPDATED_AT.isoformat(),
                price,
                length=32,
            ),
            listing_id=listing_id,
            observed_at=BIZGRAM_DOCUMENT_UPDATED_AT,
            base_price=price,
            shipping_price=Decimal("0.00"),
            stock_status=StockState.UNKNOWN,
            promotion_text="Bizgram price-list amount; stock and shipping not asserted",
        )
        raw_row = {
            "raw_line": raw_line,
            "raw_price": raw_price,
            "page": page_number,
            "line": line_number,
        }
        return {
            "schema_version": NORMALISED_RECORD_SCHEMA_VERSION,
            "record_type": "retailer_listing",
            "source_record_id": source_record_id,
            "archive_snapshot_sha256": snapshot.content_sha256,
            "raw_record_sha256": sha256_bytes(
                json.dumps(raw_row, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ),
            "training_eligible": False,
            "published_claims_eligible": False,
            "data_use_rights": dict(BIZGRAM_DATA_USE_RIGHTS),
            "provenance": {
                "source_name": snapshot.source_name,
                "source_url": snapshot.source_url,
                "source_type": "retailer",
                "retrieved_at": snapshot.retrieved_at.isoformat(),
                "parser_version": snapshot.parser_version,
                "licence_or_access_note": snapshot.licence_or_access_note,
                "extraction_confidence": 0.94,
            },
            "normalisation_metadata": {
                "page_number": page_number,
                "line_number": line_number,
                "category": category,
                "raw_title": title,
                "raw_price_text": raw_price,
                "extraction_method": "pypdf_dotted_leader_terminal_price",
                "confidence_flags": [
                    "known_document_fingerprint",
                    "known_page_layout",
                    "dotted_leader",
                    "single_terminal_numeric_price",
                    "no_ocr",
                ],
                "canonical_mapping_status": "unmatched",
                "development_only": True,
            },
            "data": {
                "listing": listing.model_dump(mode="json"),
                "price_snapshot": price_snapshot.model_dump(mode="json"),
            },
        }

    def _quarantine_duplicates_and_conflicts(
        self,
        candidates: list[dict[str, Any]],
        batch: ParsedBatch,
        counters: dict[str, int],
    ) -> list[dict[str, Any]]:
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for record in candidates:
            metadata = record["normalisation_metadata"]
            listing = record["data"]["listing"]
            key = (
                str(metadata["category"]),
                re.sub(r"\W+", " ", str(listing["title"]).casefold()).strip(),
            )
            grouped[key].append(record)

        accepted: list[dict[str, Any]] = []
        for records in grouped.values():
            prices = {str(record["data"]["listing"]["base_price"]) for record in records}
            if len(prices) > 1:
                for record in records:
                    self._reject(
                        batch,
                        counters,
                        str(record["source_record_id"]),
                        "conflicting_duplicate_price",
                        title=record["data"]["listing"]["title"],
                        observed_prices=sorted(prices),
                    )
                continue
            accepted.append(records[0])
            for duplicate in records[1:]:
                self._reject(
                    batch,
                    counters,
                    str(duplicate["source_record_id"]),
                    "exact_duplicate_source_offer",
                    title=duplicate["data"]["listing"]["title"],
                )
        return accepted

    @staticmethod
    def _reject(
        batch: ParsedBatch,
        counters: dict[str, int],
        record_id: str,
        reason: str,
        **details: object,
    ) -> None:
        if len(batch.rejected) >= BIZGRAM_MAX_REJECTIONS:
            counters["rejections_dropped_due_to_budget"] += 1
            return
        batch.rejected.append(rejected_record(record_id, reason, **details))

    @staticmethod
    def _clean_text(value: object) -> str | None:
        if value is None:
            return None
        text = re.sub(r"\s+", " ", str(value)).strip()
        return text or None
