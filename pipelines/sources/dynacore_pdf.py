"""Development-only controlled import for the pinned Dynacore price-list PDF."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

import pdfplumber
from pdfplumber.pdf import PDF

from pc_build_recommender.domain.enums import ListingCondition, StockStatus
from pc_build_recommender.domain.models import PriceSample, RetailerListing
from pipelines.parsing.normalizers import NORMALISED_RECORD_SCHEMA_VERSION, stable_identifier
from pipelines.sources.base import (
    ParsedBatch,
    RawSnapshot,
    rejected_record,
    sha256_bytes,
    snapshot_local_file,
)

DYNACORE_EXPECTED_SHA256 = "6e243d7bf1cba090f529b09a9276fac03fedddcadb8c11cf9ce7ec1e674bb9ba"
DYNACORE_SOURCE_URL = "controlled-import://dynacore/2026-07-17"
DYNACORE_RETAILER_URL = "https://www.dynacoretech.com/"
DYNACORE_PARSER_VERSION = "dynacore-controlled-pdf-2026-07-layout-v1"
DYNACORE_ACCESS_NOTE = (
    "Development-only controlled import. No open data licence was established. "
    "Do not redistribute source data, train models, or use rows in published metric claims "
    "without written permission from Dynacore."
)
DYNACORE_OBSERVED_AT = datetime(2026, 7, 18, 0, 0, tzinfo=timezone(timedelta(hours=8)))
DYNACORE_DATA_USE_RIGHTS: dict[str, Any] = {
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

_NUMERIC_PRICE = re.compile(r"^[1-9][0-9]{1,4}$")
_CAPACITY = re.compile(r"\b(\d+)\s*(GB|TB)\b", re.IGNORECASE)
_GPU_PRODUCT = re.compile(
    r"\b(?:GT\s?\d|RTX|GeForce|Radeon|Quadro|Intel\s+ARC|Arc\s+[AB]\d)\b",
    re.IGNORECASE,
)
_GPU_EXCLUSION = re.compile(
    r"\b(?:eGPU|enclosure|Jetson|DGX|developer\s+kit|Studio-G)\b", re.IGNORECASE
)
_MEMORY_PRODUCT = re.compile(r"\b(?:DDR[345]|SODIMM|UDIMM|SO-DIMM)\b", re.IGNORECASE)


class DynacoreControlledPDFAdapter:
    """Parse only high-confidence rows from one fingerprinted, born-digital PDF."""

    def __init__(self, *, raw_root: str | Path) -> None:
        self.raw_root = Path(raw_root)

    def fetch(self, *, pdf_path: str | Path) -> RawSnapshot:
        snapshot = snapshot_local_file(
            source_name="dynacore_controlled_pdf",
            source_url=DYNACORE_SOURCE_URL,
            source_type="retailer",
            source_path=pdf_path,
            raw_root=self.raw_root,
            parser_version=DYNACORE_PARSER_VERSION,
            licence_or_access_note=DYNACORE_ACCESS_NOTE,
            suffix=".pdf",
            media_type="application/pdf",
        )
        if snapshot.content_sha256 != DYNACORE_EXPECTED_SHA256:
            raise ValueError(
                "Dynacore PDF fingerprint changed; create and review a new layout "
                "profile before import"
            )
        return snapshot

    def parse(self, snapshot: RawSnapshot) -> ParsedBatch:
        if snapshot.content_sha256 != DYNACORE_EXPECTED_SHA256:
            raise ValueError("snapshot does not match the reviewed Dynacore PDF fingerprint")
        batch = ParsedBatch(
            source_name=snapshot.source_name,
            snapshot_sha256=snapshot.content_sha256,
        )
        candidates: list[dict[str, Any]] = []
        with pdfplumber.open(snapshot.path) as document:
            self._validate_layout(document)
            self._quarantine_ref_cells(document, batch)
            candidates.extend(self._parse_nvidia_gpu_tables(document.pages[1], batch, snapshot))
            candidates.extend(self._parse_amd_gpu_and_memory(document.pages[2], batch, snapshot))
            candidates.extend(self._parse_storage(document.pages[2], batch, snapshot))
            candidates.extend(
                self._parse_coordinate_column(
                    page=document.pages[2],
                    page_number=3,
                    snapshot=snapshot,
                    batch=batch,
                    section="case",
                    x_min=541.0,
                    x_max=688.0,
                    price_x=671.52,
                    price_x_tolerance=2.0,
                    category="case",
                    title_filter=lambda value: bool(
                        re.search(r"\b(?:case|casing)\b", value, re.IGNORECASE)
                    ),
                )
            )
            candidates.extend(
                self._parse_coordinate_column(
                    page=document.pages[2],
                    page_number=3,
                    snapshot=snapshot,
                    batch=batch,
                    section="psu_or_cooler",
                    x_min=688.0,
                    x_max=930.0,
                    price_x=818.52,
                    price_x_tolerance=0.25,
                    category=None,
                    title_filter=lambda value: bool(
                        re.search(
                            r"\b(?:PSU|power\s+supply|\d{3,4}W|cooler|heatsink|AIO)\b",
                            value,
                            re.IGNORECASE,
                        )
                    ),
                )
            )

        batch.records = self._quarantine_duplicates_and_conflicts(candidates, batch)
        total_rejections = len(batch.rejected)
        category_counts: dict[str, int] = {}
        method_counts: dict[str, int] = {}
        for record in batch.records:
            metadata = record["normalisation_metadata"]
            category = str(metadata["category"])
            method = str(metadata["extraction_method"])
            category_counts[category] = category_counts.get(category, 0) + 1
            method_counts[method] = method_counts.get(method, 0) + 1
        rejection_counts: dict[str, int] = {}
        for rejected in batch.rejected:
            reason = str(rejected.get("reason", "unknown"))
            rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
        manual_review_reasons = {
            "ambiguous_component_category",
            "ambiguous_multi_product_or_inline_price",
            "conflicting_duplicate_price",
            "non_numeric_or_ambiguous_price",
        }
        manual_review_count = sum(
            count for reason, count in rejection_counts.items() if reason in manual_review_reasons
        )
        batch.statistics = {
            "document_sha256": snapshot.content_sha256,
            "page_count": 12,
            "born_digital_text_layer": True,
            "ocr_used": False,
            "training_eligible": False,
            "published_claims_eligible": False,
            "accepted_by_category": dict(sorted(category_counts.items())),
            "accepted_by_method": dict(sorted(method_counts.items())),
            "rejection_counts": dict(sorted(rejection_counts.items())),
            "total_rejections": total_rejections,
            "manual_review_count": manual_review_count,
            "hard_rejected_count": total_rejections - manual_review_count,
        }
        return batch

    @staticmethod
    def _validate_layout(document: PDF) -> None:
        if len(document.pages) != 12:
            raise ValueError(f"expected 12 pages, found {len(document.pages)}")
        page_two_text = document.pages[1].extract_text() or ""
        page_three_text = document.pages[2].extract_text() or ""
        required_page_two = ("NVIDIA GRAPHICS CARDS", "UPDATED AS OF: 18 Jul 2026")
        required_page_three = ("AMD GRAPHICS CARDS", "M.2 NVMe INTERNAL SSD")
        if any(value not in page_two_text for value in required_page_two):
            raise ValueError("page 2 layout anchors do not match the reviewed profile")
        if any(value not in page_three_text for value in required_page_three):
            raise ValueError("page 3 layout anchors do not match the reviewed profile")

    @staticmethod
    def _quarantine_ref_cells(document: PDF, batch: ParsedBatch) -> None:
        for page_index in (1, 2):
            for table_index, table in enumerate(document.pages[page_index].extract_tables()):
                for row_index, row in enumerate(table):
                    for column_index, value in enumerate(row):
                        if "#REF!" not in str(value or "").upper():
                            continue
                        batch.rejected.append(
                            rejected_record(
                                f"p{page_index + 1}-t{table_index}-r{row_index}-c{column_index}",
                                "invalid_ref_cell",
                                page=page_index + 1,
                                table=table_index,
                                row=row_index,
                                column=column_index,
                                raw_value=value,
                            )
                        )

    def _parse_nvidia_gpu_tables(
        self,
        page: Any,
        batch: ParsedBatch,
        snapshot: RawSnapshot,
    ) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        tables = page.crop((546, 0, 815, page.height)).extract_tables()
        for table_index, table in enumerate(tables):
            for row_index, row in enumerate(table):
                if len(row) < 2:
                    continue
                title = self._clean_text(row[0])
                raw_price = self._clean_text(row[-1])
                if not title or not _GPU_PRODUCT.search(title):
                    continue
                record_id = f"p2-gpu-t{table_index}-r{row_index}"
                if _GPU_EXCLUSION.search(title):
                    batch.rejected.append(
                        rejected_record(record_id, "excluded_non_desktop_gpu", title=title)
                    )
                    continue
                price = self._strict_price(raw_price)
                if price is None or raw_price is None:
                    batch.rejected.append(
                        rejected_record(
                            record_id,
                            "non_numeric_or_ambiguous_price",
                            title=title,
                            raw_price=raw_price,
                        )
                    )
                    continue
                records.append(
                    self._listing_record(
                        snapshot=snapshot,
                        source_record_id=record_id,
                        title=title,
                        price=price,
                        page_number=2,
                        section="nvidia_gpu",
                        category="gpu",
                        raw_price=raw_price,
                        extraction_method="pdf_table_cell",
                        extraction_confidence=0.97,
                        bounding_box=[546.0, 0.0, 815.0, float(page.height)],
                    )
                )
        return records

    def _parse_amd_gpu_and_memory(
        self,
        page: Any,
        batch: ParsedBatch,
        snapshot: RawSnapshot,
    ) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        tables = page.extract_tables()
        if not tables:
            raise ValueError("page 3 component table was not found")
        for row_index, row in enumerate(tables[0]):
            title = self._clean_text(row[0] if row else None)
            raw_price = self._clean_text(row[-1] if row else None)
            if not title or not _GPU_PRODUCT.search(title):
                continue
            record_id = f"p3-amd-gpu-r{row_index}"
            if _GPU_EXCLUSION.search(title):
                batch.rejected.append(
                    rejected_record(record_id, "excluded_non_desktop_gpu", title=title)
                )
                continue
            price = self._strict_price(raw_price)
            if price is None or raw_price is None:
                batch.rejected.append(
                    rejected_record(
                        record_id,
                        "non_numeric_or_ambiguous_price",
                        title=title,
                        raw_price=raw_price,
                    )
                )
                continue
            records.append(
                self._listing_record(
                    snapshot=snapshot,
                    source_record_id=record_id,
                    title=title,
                    price=price,
                    page_number=3,
                    section="amd_gpu",
                    category="gpu",
                    raw_price=raw_price,
                    extraction_method="pdf_table_cell",
                    extraction_confidence=0.97,
                    bounding_box=[6.0, 0.0, 284.0, float(page.height)],
                )
            )

        for table_index, table in enumerate(tables[:2]):
            capacity_by_column: dict[int, str] = {}
            for row_index, row in enumerate(table):
                title = self._clean_text(row[0] if row else None)
                if not title:
                    continue
                if "BUNDLE" in title.upper() or "CHASSIS" in title.upper():
                    capacity_by_column = {}
                    continue
                header_capacities = {
                    column: match.group(0).upper().replace(" ", "")
                    for column, value in enumerate(row)
                    if (match := _CAPACITY.fullmatch(self._clean_text(value) or "")) is not None
                }
                if header_capacities and re.search(r"\b(?:RAM|MEMORY)\b", title, re.IGNORECASE):
                    capacity_by_column = header_capacities
                    continue
                if not capacity_by_column or not _MEMORY_PRODUCT.search(title):
                    continue
                for column, capacity in capacity_by_column.items():
                    raw_price = self._clean_text(row[column] if column < len(row) else None)
                    price = self._strict_price(raw_price)
                    if price is None or raw_price is None:
                        continue
                    record_id = f"p3-ram-t{table_index}-r{row_index}-c{column}"
                    records.append(
                        self._listing_record(
                            snapshot=snapshot,
                            source_record_id=record_id,
                            title=f"{title} [{capacity}]",
                            price=price,
                            page_number=3,
                            section="memory_matrix",
                            category="memory",
                            raw_price=raw_price,
                            extraction_method="pdf_capacity_matrix",
                            extraction_confidence=0.95,
                            variant=capacity,
                            bounding_box=[6.0, 0.0, 284.0, float(page.height)],
                        )
                    )
        return records

    def _parse_storage(
        self,
        page: Any,
        batch: ParsedBatch,
        snapshot: RawSnapshot,
    ) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        tables = page.extract_tables()
        if len(tables) < 2:
            raise ValueError("page 3 storage table was not found")
        table = tables[1]
        reviewed_sections = (
            (2, 30, {4: "500GB", 5: "1TB", 6: "2TB", 7: "4TB"}, "nvme_ssd"),
            (35, 45, {3: "250GB", 4: "500GB", 5: "1TB", 6: "2TB", 7: "4TB"}, "sata_ssd"),
        )
        for start, end, capacity_by_column, section in reviewed_sections:
            for row_index in range(start, min(end, len(table))):
                row = table[row_index]
                title = self._clean_text(row[0] if row else None)
                if not title:
                    continue
                if "/" in title or "$" in title or "#REF!" in title.upper():
                    batch.rejected.append(
                        rejected_record(
                            f"p3-storage-r{row_index}",
                            "ambiguous_multi_product_or_inline_price",
                            title=title,
                        )
                    )
                    continue
                for column, capacity in capacity_by_column.items():
                    raw_price = self._clean_text(row[column] if column < len(row) else None)
                    if not raw_price or raw_price == "-":
                        continue
                    price = self._strict_price(raw_price)
                    record_id = f"p3-storage-r{row_index}-c{column}"
                    if price is None:
                        batch.rejected.append(
                            rejected_record(
                                record_id,
                                "non_numeric_or_ambiguous_price",
                                title=title,
                                raw_price=raw_price,
                            )
                        )
                        continue
                    records.append(
                        self._listing_record(
                            snapshot=snapshot,
                            source_record_id=record_id,
                            title=f"{title} [{capacity}]",
                            price=price,
                            page_number=3,
                            section=section,
                            category="storage",
                            raw_price=raw_price,
                            extraction_method="pdf_capacity_matrix",
                            extraction_confidence=0.96,
                            variant=capacity,
                            bounding_box=[289.0, 0.0, 539.0, float(page.height)],
                        )
                    )
        return records

    def _parse_coordinate_column(
        self,
        *,
        page: Any,
        page_number: int,
        snapshot: RawSnapshot,
        batch: ParsedBatch,
        section: str,
        x_min: float,
        x_max: float,
        price_x: float,
        price_x_tolerance: float,
        category: str | None,
        title_filter: Any,
    ) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        lines: dict[float, list[dict[str, Any]]] = defaultdict(list)
        for character in page.chars:
            if x_min <= float(character["x0"]) < x_max:
                lines[round(float(character["top"]), 3)].append(character)
        for line_number, (top, characters) in enumerate(sorted(lines.items())):
            price_start = self._price_character_start(
                characters, price_x, tolerance=price_x_tolerance
            )
            if price_start is None:
                continue
            price_characters = []
            for character in characters[price_start:]:
                if not str(character.get("text", "")).isdigit():
                    break
                price_characters.append(character)
            raw_price = "".join(str(value["text"]) for value in price_characters)
            price = self._strict_price(raw_price)
            title = self._clean_text(
                "".join(str(value.get("text", "")) for value in characters[:price_start])
            )
            if price is None or not title or not title_filter(title):
                continue
            inferred_category = category or self._category_from_title(title)
            if inferred_category is None:
                batch.rejected.append(
                    rejected_record(
                        f"p{page_number}-{section}-l{line_number}",
                        "ambiguous_component_category",
                        title=title,
                        raw_price=raw_price,
                    )
                )
                continue
            source_record_id = f"p{page_number}-{section}-l{line_number}"
            records.append(
                self._listing_record(
                    snapshot=snapshot,
                    source_record_id=source_record_id,
                    title=title,
                    price=price,
                    page_number=page_number,
                    section=section,
                    category=inferred_category,
                    raw_price=raw_price,
                    extraction_method="pdf_coordinate_price_overlay",
                    extraction_confidence=0.88,
                    bounding_box=[x_min, top, x_max, top + float(characters[0]["height"])],
                    confidence_flags=[
                        "known_layout_fingerprint",
                        "price_starts_at_reviewed_column_coordinate",
                        "no_ocr",
                    ],
                )
            )
        return records

    @staticmethod
    def _price_character_start(
        characters: list[dict[str, Any]], price_x: float, *, tolerance: float
    ) -> int | None:
        candidates = [
            index
            for index, character in enumerate(characters)
            if str(character.get("text", "")).isdigit()
            and abs(float(character["x0"]) - price_x) <= tolerance
        ]
        for index in reversed(candidates):
            digits = 0
            for character in characters[index:]:
                if not str(character.get("text", "")).isdigit():
                    break
                digits += 1
            if 2 <= digits <= 5:
                return index
        return None

    @staticmethod
    def _category_from_title(title: str) -> str | None:
        if re.search(r"\b(?:cooler|heatsink|AIO)\b", title, re.IGNORECASE):
            return "cooler"
        if re.search(
            r"\b(?:PSU|power\s+supply|\d{3,4}W|80\+|ATX3(?:\.\d)?)\b",
            title,
            re.IGNORECASE,
        ):
            return "power_supply"
        return None

    def _listing_record(
        self,
        *,
        snapshot: RawSnapshot,
        source_record_id: str,
        title: str,
        price: int,
        page_number: int,
        section: str,
        category: str,
        raw_price: str,
        extraction_method: str,
        extraction_confidence: float,
        variant: str | None = None,
        bounding_box: list[float] | None = None,
        confidence_flags: list[str] | None = None,
    ) -> dict[str, Any]:
        listing_id = stable_identifier("listing_dynacore", source_record_id, length=32)
        product_id = stable_identifier("unmatched_product", "dynacore", title, category)
        amount = Decimal(price).quantize(Decimal("0.01"))
        listing = RetailerListing(
            listing_id=listing_id,
            product_id=product_id,
            retailer="Dynacore",
            source_listing_id=source_record_id,
            title=title,
            condition=ListingCondition.NEW,
            currency="SGD",
            base_price=amount,
            shipping_price=Decimal("0.00"),
            stock_status=StockStatus.UNKNOWN,
            seller_name="Dynacore",
            listing_url=DYNACORE_RETAILER_URL,
            first_seen_at=DYNACORE_OBSERVED_AT,
            last_seen_at=DYNACORE_OBSERVED_AT,
        )
        price_snapshot = PriceSample(
            snapshot_id=stable_identifier(
                "price_dynacore",
                listing_id,
                DYNACORE_OBSERVED_AT.isoformat(),
                amount,
                length=32,
            ),
            listing_id=listing_id,
            observed_at=DYNACORE_OBSERVED_AT,
            base_price=amount,
            shipping_price=Decimal("0.00"),
            stock_status=StockStatus.UNKNOWN,
            promotion_text="Dynacore price list #05-73; availability not asserted",
        )
        raw_row = {
            "title": title,
            "raw_price": raw_price,
            "page": page_number,
            "section": section,
            "variant": variant,
            "bounding_box": bounding_box,
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
            "data_use_rights": dict(DYNACORE_DATA_USE_RIGHTS),
            "provenance": {
                "source_name": snapshot.source_name,
                "source_url": snapshot.source_url,
                "source_type": "retailer",
                "retrieved_at": snapshot.retrieved_at.isoformat(),
                "parser_version": snapshot.parser_version,
                "licence_or_access_note": snapshot.licence_or_access_note,
                "extraction_confidence": extraction_confidence,
            },
            "normalisation_metadata": {
                "page_number": page_number,
                "section": section,
                "category": category,
                "variant": variant,
                "raw_title": title,
                "raw_price_text": raw_price,
                "bounding_box": bounding_box,
                "extraction_method": extraction_method,
                "confidence_flags": confidence_flags
                or ["known_layout_fingerprint", "numeric_price_cell", "no_ocr"],
                "canonical_mapping_status": "unmatched",
                "development_only": True,
            },
            "data": {
                "listing": listing.model_dump(mode="json"),
                "price_snapshot": price_snapshot.model_dump(mode="json"),
            },
        }

    def _quarantine_duplicates_and_conflicts(
        self, candidates: list[dict[str, Any]], batch: ParsedBatch
    ) -> list[dict[str, Any]]:
        grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
        for record in candidates:
            metadata = record["normalisation_metadata"]
            listing = record["data"]["listing"]
            key = (
                str(metadata["category"]),
                re.sub(r"\W+", " ", str(listing["title"]).casefold()).strip(),
                str(metadata.get("variant") or ""),
            )
            grouped[key].append(record)

        accepted: list[dict[str, Any]] = []
        for records in grouped.values():
            prices = {str(record["data"]["listing"]["base_price"]) for record in records}
            if len(prices) > 1:
                for record in records:
                    batch.rejected.append(
                        rejected_record(
                            str(record["source_record_id"]),
                            "conflicting_duplicate_price",
                            title=record["data"]["listing"]["title"],
                            observed_prices=sorted(prices),
                        )
                    )
                continue
            accepted.append(records[0])
            for duplicate in records[1:]:
                batch.rejected.append(
                    rejected_record(
                        str(duplicate["source_record_id"]),
                        "exact_duplicate_source_offer",
                        title=duplicate["data"]["listing"]["title"],
                    )
                )
        return accepted

    @staticmethod
    def _strict_price(value: str | None) -> int | None:
        if value is None or _NUMERIC_PRICE.fullmatch(value) is None:
            return None
        price = int(value)
        return price if 10 <= price <= 50_000 else None

    @staticmethod
    def _clean_text(value: object) -> str | None:
        if value is None:
            return None
        text = re.sub(r"\s+", " ", str(value)).strip()
        return text or None
