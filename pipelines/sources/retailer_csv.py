"""Controlled CSV adapter for a retailer that has explicitly consented to ingestion."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from datetime import datetime
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from pathlib import Path
from typing import Any

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
from pipelines.sources.rights import DataUse, DataUseRights

RETAILER_CSV_PARSER_VERSION = "consented-retailer-csv-v1"
REQUIRED_COLUMNS = {
    "source_listing_id",
    "title",
    "currency",
    "base_price",
    "stock_status",
    "listing_url",
}


@dataclass(frozen=True, slots=True)
class RetailerFeedPolicy:
    """Auditable authority and data-use limits supplied with one retailer feed."""

    retailer: str
    feed_id: str
    source_url: str
    licence_or_access_note: str
    rights: DataUseRights
    training_eligible: bool = False
    published_claims_eligible: bool = False
    allow_non_new: bool = False

    def __post_init__(self) -> None:
        if not self.retailer.strip():
            raise ValueError("retailer must not be empty")
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", self.feed_id):
            raise ValueError("feed_id must be a lowercase slug")
        if not self.source_url.strip() or not self.licence_or_access_note.strip():
            raise ValueError("source_url and licence_or_access_note must not be empty")
        self.rights.assert_consent_active()
        for required_use in (
            DataUse.DISPLAY,
            DataUse.CACHE,
            DataUse.STORE_HISTORY,
            DataUse.DERIVE,
        ):
            self.rights.assert_allowed(required_use)
        if self.training_eligible:
            self.rights.assert_allowed(DataUse.TRAIN)
        if self.published_claims_eligible:
            self.rights.assert_allowed(DataUse.DISPLAY)
            self.rights.assert_allowed(DataUse.DERIVE)

    @property
    def consent_reference(self) -> str:
        return self.rights.contract_reference

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> RetailerFeedPolicy:
        values = dict(payload)
        raw_rights = values.get("rights")
        if not isinstance(raw_rights, dict):
            raise TypeError("retailer feed policy requires a rights object")
        values["rights"] = DataUseRights.from_mapping(raw_rights)
        return cls(**values)


class ConsentedRetailerCSVAdapter:
    """Normalise listings only when a concrete retailer consent reference is present."""

    def __init__(self, *, raw_root: str | Path, policy: RetailerFeedPolicy) -> None:
        self.raw_root = Path(raw_root)
        self.policy = policy

    @property
    def source_name(self) -> str:
        return f"retailer_feed_{self.policy.feed_id}"

    def fetch(self, *, csv_path: str | Path) -> RawSnapshot:
        return snapshot_local_file(
            source_name=self.source_name,
            source_url=self.policy.source_url,
            source_type="retailer",
            source_path=csv_path,
            raw_root=self.raw_root,
            parser_version=RETAILER_CSV_PARSER_VERSION,
            licence_or_access_note=self.policy.licence_or_access_note,
            suffix=".csv",
            media_type="text/csv",
        )

    def parse(self, snapshot: RawSnapshot) -> ParsedBatch:
        batch = ParsedBatch(
            source_name=snapshot.source_name,
            snapshot_sha256=snapshot.content_sha256,
        )
        seen_source_ids: set[str] = set()
        with snapshot.path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            columns = set(reader.fieldnames or [])
            missing_columns = REQUIRED_COLUMNS - columns
            if missing_columns:
                raise ValueError(f"retailer CSV missing columns: {sorted(missing_columns)}")
            for row_number, row in enumerate(reader, start=2):
                source_listing_id = str(row.get("source_listing_id", "")).strip()
                record_id = source_listing_id or f"row-{row_number}"
                if source_listing_id in seen_source_ids:
                    batch.rejected.append(
                        rejected_record(record_id, "duplicate_source_listing_id", row=row_number)
                    )
                    continue
                try:
                    normalised = self._normalise_row(
                        row=row,
                        row_number=row_number,
                        snapshot=snapshot,
                    )
                except (InvalidOperation, TypeError, ValueError) as exc:
                    batch.rejected.append(
                        rejected_record(
                            record_id,
                            "invalid_retailer_listing",
                            row=row_number,
                            error=f"{type(exc).__name__}: {exc}",
                        )
                    )
                    continue
                seen_source_ids.add(source_listing_id)
                batch.records.append(normalised)
        batch.statistics = {
            "retailer": self.policy.retailer,
            "feed_id": self.policy.feed_id,
            "consent_reference": self.policy.consent_reference,
            "training_eligible": self.policy.training_eligible,
            "published_claims_eligible": self.policy.published_claims_eligible,
            "data_use_rights": self.policy.rights.to_dict(),
            "unique_source_listing_ids": len(seen_source_ids),
        }
        return batch

    def _normalise_row(
        self,
        *,
        row: dict[str, str | None],
        row_number: int,
        snapshot: RawSnapshot,
    ) -> dict[str, Any]:
        source_listing_id = self._required(row, "source_listing_id")
        title = self._required(row, "title")
        csv_retailer = str(row.get("retailer") or "").strip()
        if csv_retailer and csv_retailer.casefold() != self.policy.retailer.casefold():
            raise ValueError(
                f"row retailer {csv_retailer!r} does not match policy retailer "
                f"{self.policy.retailer!r}"
            )
        currency = self._required(row, "currency").upper()
        if not re.fullmatch(r"[A-Z]{3}", currency):
            raise ValueError(f"invalid currency: {currency!r}")
        base_price = self._money(row.get("base_price"), positive=True)
        shipping_price = self._money(row.get("shipping_price") or "0", positive=False)
        stock_status = self._stock_status(row.get("stock_status"))
        condition = self._condition(row.get("condition") or "new")
        if not self.policy.allow_non_new and condition != ListingCondition.NEW:
            raise ValueError(f"non-new condition is outside this feed policy: {condition.value}")
        listing_url = self._required(row, "listing_url")
        observed_at = self._datetime(row.get("observed_at")) or snapshot.retrieved_at
        product_id = str(row.get("product_id") or "").strip() or stable_identifier(
            "unmatched_product", self.policy.retailer, source_listing_id
        )
        listing_id = stable_identifier(
            "listing", self.policy.retailer, source_listing_id, length=32
        )
        listing = RetailerListing(
            listing_id=listing_id,
            product_id=product_id,
            retailer=self.policy.retailer,
            source_listing_id=source_listing_id,
            title=title,
            condition=condition,
            currency=currency,
            base_price=base_price,
            shipping_price=shipping_price,
            stock_status=stock_status,
            seller_name=str(row.get("seller_name") or "").strip() or None,
            listing_url=listing_url,
            first_seen_at=observed_at,
            last_seen_at=observed_at,
        )
        price_snapshot = PriceSample(
            snapshot_id=stable_identifier(
                "price",
                listing_id,
                observed_at.isoformat(),
                base_price,
                shipping_price,
                stock_status.value,
                length=32,
            ),
            listing_id=listing_id,
            observed_at=observed_at,
            base_price=base_price,
            shipping_price=shipping_price,
            stock_status=stock_status,
            promotion_text=str(row.get("promotion_text") or "").strip() or None,
        )
        raw_row_bytes = json.dumps(
            row,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return {
            "schema_version": NORMALISED_RECORD_SCHEMA_VERSION,
            "record_type": "retailer_listing",
            "source_record_id": source_listing_id,
            "archive_snapshot_sha256": snapshot.content_sha256,
            "raw_record_sha256": sha256_bytes(raw_row_bytes),
            "training_eligible": self.policy.training_eligible,
            "published_claims_eligible": self.policy.published_claims_eligible,
            "data_use_rights": self.policy.rights.to_dict(),
            "provenance": {
                "source_name": snapshot.source_name,
                "source_url": listing_url,
                "source_type": "retailer",
                "retrieved_at": snapshot.retrieved_at.isoformat(),
                "parser_version": snapshot.parser_version,
                "licence_or_access_note": snapshot.licence_or_access_note,
                "extraction_confidence": 1.0,
            },
            "normalisation_metadata": {
                "row_number": row_number,
                "canonical_mapping_status": (
                    "provided" if str(row.get("product_id") or "").strip() else "unmatched"
                ),
                "manufacturer_part_number": str(row.get("manufacturer_part_number") or "").strip()
                or None,
            },
            "data": {
                "listing": listing.model_dump(mode="json"),
                "price_snapshot": price_snapshot.model_dump(mode="json"),
            },
        }

    @staticmethod
    def _required(row: dict[str, str | None], field_name: str) -> str:
        value = str(row.get(field_name) or "").strip()
        if not value:
            raise ValueError(f"{field_name} is required")
        return value

    @staticmethod
    def _money(value: object, *, positive: bool) -> Decimal:
        text = str(value).strip().replace(",", "")
        amount = Decimal(text).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        if amount < 0 or (positive and amount == 0):
            qualifier = "positive" if positive else "non-negative"
            raise ValueError(f"money must be {qualifier}: {value!r}")
        return amount

    @staticmethod
    def _stock_status(value: object) -> StockStatus:
        text = re.sub(r"[^a-z]", "_", str(value).strip().lower()).strip("_")
        mappings = {
            "in_stock": StockStatus.IN_STOCK,
            "instock": StockStatus.IN_STOCK,
            "available": StockStatus.IN_STOCK,
            "out_of_stock": StockStatus.OUT_OF_STOCK,
            "outofstock": StockStatus.OUT_OF_STOCK,
            "sold_out": StockStatus.OUT_OF_STOCK,
            "backorder": StockStatus.BACKORDER,
            "back_order": StockStatus.BACKORDER,
            "preorder": StockStatus.PREORDER,
            "pre_order": StockStatus.PREORDER,
            "unknown": StockStatus.UNKNOWN,
        }
        if text not in mappings:
            raise ValueError(f"unrecognised stock_status: {value!r}")
        return mappings[text]

    @staticmethod
    def _condition(value: object) -> ListingCondition:
        text = re.sub(r"[^a-z]", "_", str(value).strip().lower()).strip("_")
        mappings = {
            "new": ListingCondition.NEW,
            "open_box": ListingCondition.OPEN_BOX,
            "refurbished": ListingCondition.REFURBISHED,
            "used": ListingCondition.USED,
            "unknown": ListingCondition.UNKNOWN,
        }
        if text not in mappings:
            raise ValueError(f"unrecognised condition: {value!r}")
        return mappings[text]

    @staticmethod
    def _datetime(value: object) -> datetime | None:
        text = str(value or "").strip()
        if not text:
            return None
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            raise ValueError("observed_at must include a timezone")
        return parsed
