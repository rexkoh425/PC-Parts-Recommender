"""Durable, auditable manual decisions for retailer-to-canonical mappings."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from tempfile import NamedTemporaryFile
from types import MappingProxyType
from typing import Any

REVIEWED_MAPPING_SCHEMA_VERSION = "pc-build-recommender.reviewed-mappings.v1"


class ReviewStatus(StrEnum):
    APPROVED = "approved"
    REJECTED = "rejected"


class MappingOutcome(StrEnum):
    AUTO_MATCHED = "auto_matched"
    REVIEWED_MATCHED = "reviewed_matched"
    REVIEW_REJECTED = "review_rejected"
    MANUAL_REVIEW = "manual_review"
    MODEL_REJECTED = "model_rejected"
    AMBIGUOUS = "ambiguous"
    HARD_CONFLICT = "hard_conflict"
    UNMATCHED = "unmatched"


@dataclass(frozen=True, slots=True)
class MappingReview:
    listing_id: str
    review_status: ReviewStatus
    reviewed_by: str
    evidence: str
    product_id: str | None = None
    reviewed_at: str | None = None

    def __post_init__(self) -> None:
        for name in ("listing_id", "reviewed_by", "evidence"):
            if not getattr(self, name).strip():
                raise ValueError(f"mapping review requires {name}")
        if self.review_status is ReviewStatus.APPROVED and not self.product_id:
            raise ValueError("approved mapping reviews require product_id")
        if self.review_status is ReviewStatus.REJECTED and self.product_id is not None:
            raise ValueError("rejected mapping reviews cannot name a product_id")

    def to_dict(self) -> dict[str, Any]:
        return {
            "listing_id": self.listing_id,
            "product_id": self.product_id,
            "review_status": self.review_status.value,
            "reviewed_by": self.reviewed_by,
            "evidence": self.evidence,
            "reviewed_at": self.reviewed_at,
        }


@dataclass(frozen=True, slots=True)
class MappingDecision:
    listing_id: str
    source_listing_id: str
    title: str
    category: str
    outcome: MappingOutcome
    matched_product_id: str | None = None
    candidate_product_ids: tuple[str, ...] = ()
    method: str | None = None
    reason: str | None = None
    probability: float | None = None
    model_version: str | None = None
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.probability is not None and not 0.0 <= self.probability <= 1.0:
            raise ValueError("mapping probability must be between zero and one")
        if self.model_version is not None and not self.model_version.strip():
            raise ValueError("model_version must not be blank")
        object.__setattr__(self, "evidence", MappingProxyType(dict(self.evidence)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "listing_id": self.listing_id,
            "source_listing_id": self.source_listing_id,
            "title": self.title,
            "category": self.category,
            "outcome": self.outcome.value,
            "matched_product_id": self.matched_product_id,
            "candidate_product_ids": list(self.candidate_product_ids),
            "method": self.method,
            "reason": self.reason,
            "probability": self.probability,
            "model_version": self.model_version,
            "evidence": dict(self.evidence),
        }


def load_mapping_reviews(path: str | Path | None) -> dict[str, MappingReview]:
    if path is None:
        return {}
    review_path = Path(path)
    payload = json.loads(review_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("reviewed mapping manifest must be a JSON object")
    if payload.get("schema_version") != REVIEWED_MAPPING_SCHEMA_VERSION:
        raise ValueError("unsupported reviewed mapping schema")
    rows = payload.get("mappings")
    if not isinstance(rows, list):
        raise ValueError("reviewed mapping manifest requires a mappings array")
    result: dict[str, MappingReview] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("mapping review rows must be JSON objects")
        try:
            status = ReviewStatus(str(row.get("review_status", "")))
        except ValueError as error:
            raise ValueError("review_status must be approved or rejected") from error
        product_id = row.get("product_id")
        review = MappingReview(
            listing_id=str(row.get("listing_id", "")).strip(),
            product_id=str(product_id).strip() if product_id not in (None, "") else None,
            review_status=status,
            reviewed_by=str(row.get("reviewed_by", "")).strip(),
            evidence=str(row.get("evidence", "")).strip(),
            reviewed_at=(
                str(row.get("reviewed_at")).strip()
                if row.get("reviewed_at") not in (None, "")
                else None
            ),
        )
        if review.listing_id in result:
            raise ValueError(f"duplicate reviewed listing mapping: {review.listing_id}")
        result[review.listing_id] = review
    return result


def save_mapping_reviews(path: str | Path, reviews: dict[str, MappingReview]) -> None:
    """Atomically replace a review manifest with deterministic ordering."""

    review_path = Path(path).resolve()
    review_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": REVIEWED_MAPPING_SCHEMA_VERSION,
        "updated_at": datetime.now(UTC).isoformat(),
        "mappings": [reviews[key].to_dict() for key in sorted(reviews)],
    }
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=review_path.parent,
            prefix=f".{review_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            json.dump(payload, temporary, indent=2, sort_keys=True)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_path = Path(temporary.name)
        os.replace(temporary_path, review_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def upsert_mapping_review(
    path: str | Path,
    *,
    listing_id: str,
    status: ReviewStatus,
    reviewed_by: str,
    evidence: str,
    product_id: str | None = None,
) -> MappingReview:
    review_path = Path(path)
    reviews = load_mapping_reviews(review_path) if review_path.exists() else {}
    review = MappingReview(
        listing_id=listing_id.strip(),
        product_id=product_id.strip() if product_id else None,
        review_status=status,
        reviewed_by=reviewed_by.strip(),
        evidence=evidence.strip(),
        reviewed_at=datetime.now(UTC).isoformat(),
    )
    reviews[review.listing_id] = review
    save_mapping_reviews(review_path, reviews)
    return review
