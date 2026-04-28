"""Durable, provenance-aware manual review queues for entity resolution.

Candidate generation never creates labels. Only a review carrying an explicit reviewer and
timezone-aware timestamp can become a binary training example. Source-use policy travels
with the queue so controlled imports cannot silently enter training or published metrics.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Self

from .candidate_generation import PCBlockingCandidate
from .conflicts import NumericConflict
from .records import CanonicalProductRecord, ListingRecord, PairExample

REVIEW_QUEUE_SCHEMA_VERSION = "pc-build-recommender.er-review-queue.v1"
_CSV_COLUMNS = (
    "queue_id",
    "queue_item_id",
    "snapshot_sha256",
    "listing_id",
    "listing_title",
    "product_id",
    "product_name",
    "category",
    "blocking_score",
    "blocking_reasons",
    "conflicts",
    "state",
    "human_label",
    "reviewer_id",
    "reviewed_at",
    "reviewer_note",
)


class ReviewState(StrEnum):
    PENDING = "PENDING"
    LABELLED = "LABELLED"
    SKIPPED = "SKIPPED"
    INVALID = "INVALID"


class HumanMatchLabel(StrEnum):
    MATCH = "MATCH"
    NON_MATCH = "NON_MATCH"
    UNCERTAIN = "UNCERTAIN"


class ReviewConflictError(ValueError):
    """Raised when an import attempts to rewrite an existing human decision."""


@dataclass(frozen=True, slots=True)
class SourceUsePolicy:
    """Usage rights and evidence scope inherited by every item in one queue."""

    listing_source: str
    catalogue_source: str
    data_version: str
    training_eligible: bool
    published_metrics_eligible: bool
    scope_note: str
    model_serving_eligible: bool = False

    def __post_init__(self) -> None:
        for name, value in (
            ("listing_source", self.listing_source),
            ("catalogue_source", self.catalogue_source),
            ("data_version", self.data_version),
            ("scope_note", self.scope_note),
        ):
            if not value.strip():
                raise ValueError(f"{name} must not be empty")

    def to_dict(self) -> dict[str, object]:
        return {
            "listing_source": self.listing_source,
            "catalogue_source": self.catalogue_source,
            "data_version": self.data_version,
            "training_eligible": self.training_eligible,
            "published_metrics_eligible": self.published_metrics_eligible,
            "model_serving_eligible": self.model_serving_eligible,
            "scope_note": self.scope_note,
        }

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> Self:
        training_eligible = row["training_eligible"]
        published_metrics_eligible = row["published_metrics_eligible"]
        model_serving_eligible = row.get("model_serving_eligible", False)
        if (
            not isinstance(training_eligible, bool)
            or not isinstance(published_metrics_eligible, bool)
            or not isinstance(model_serving_eligible, bool)
        ):
            raise TypeError("source-use eligibility fields must be JSON booleans")
        return cls(
            listing_source=str(row["listing_source"]),
            catalogue_source=str(row["catalogue_source"]),
            data_version=str(row["data_version"]),
            training_eligible=training_eligible,
            published_metrics_eligible=published_metrics_eligible,
            scope_note=str(row["scope_note"]),
            model_serving_eligible=model_serving_eligible,
        )


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_payload(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _validate_timestamp(value: str, *, field_name: str) -> None:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"{field_name} must be an ISO-8601 timestamp") from error
    if parsed.tzinfo is None:
        raise ValueError(f"{field_name} must include a timezone")


def _conflict_payload(conflicts: Iterable[NumericConflict]) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "field": conflict.field,
            "listing_value": conflict.listing_value,
            "product_value": conflict.product_value,
            "message": conflict.message,
        }
        for conflict in conflicts
    )


@dataclass(frozen=True, slots=True)
class ReviewQueueItem:
    """A frozen candidate snapshot plus an optional, attributable human decision."""

    queue_item_id: str
    listing: ListingRecord
    product: CanonicalProductRecord
    blocking_score: float
    blocking_reasons: tuple[str, ...]
    conflicts: tuple[Mapping[str, object], ...]
    data_version: str
    snapshot_sha256: str
    state: ReviewState = ReviewState.PENDING
    human_label: HumanMatchLabel | None = None
    reviewer_id: str | None = None
    reviewed_at: str | None = None
    reviewer_note: str | None = None

    def __post_init__(self) -> None:
        if not self.queue_item_id.strip() or not self.data_version.strip():
            raise ValueError("queue_item_id and data_version must not be empty")
        if not 0.0 <= self.blocking_score <= 1.0:
            raise ValueError("blocking_score must be between zero and one")
        if not self.blocking_reasons or any(not reason.strip() for reason in self.blocking_reasons):
            raise ValueError("blocking_reasons must contain non-empty evidence codes")
        object.__setattr__(
            self,
            "conflicts",
            tuple(MappingProxyType(dict(conflict)) for conflict in self.conflicts),
        )
        expected_hash = self.calculate_snapshot_hash()
        if self.snapshot_sha256 != expected_hash:
            raise ValueError("review candidate snapshot hash mismatch")
        if self.state is ReviewState.PENDING:
            if any(
                value is not None
                for value in (
                    self.human_label,
                    self.reviewer_id,
                    self.reviewed_at,
                    self.reviewer_note,
                )
            ):
                raise ValueError("pending review items cannot contain review fields")
            return
        if not self.reviewer_id or not self.reviewed_at:
            raise ValueError("completed review actions require reviewer_id and reviewed_at")
        _validate_timestamp(self.reviewed_at, field_name="reviewed_at")
        if self.state is ReviewState.LABELLED and self.human_label is None:
            raise ValueError("labelled review items require a human label")
        if self.state is not ReviewState.LABELLED and self.human_label is not None:
            raise ValueError("skipped or invalid review items cannot contain a match label")

    def snapshot_payload(self) -> dict[str, Any]:
        return {
            "listing": self.listing.to_dict(),
            "product": self.product.to_dict(),
            "blocking_score": self.blocking_score,
            "blocking_reasons": list(self.blocking_reasons),
            "conflicts": [dict(conflict) for conflict in self.conflicts],
            "data_version": self.data_version,
        }

    def calculate_snapshot_hash(self) -> str:
        return _sha256_payload(self.snapshot_payload())

    @classmethod
    def pending(
        cls,
        candidate: PCBlockingCandidate,
        *,
        data_version: str,
    ) -> Self:
        conflicts = _conflict_payload(candidate.conflicts)
        payload = {
            "listing": candidate.listing.to_dict(),
            "product": candidate.product.to_dict(),
            "blocking_score": candidate.blocking_score,
            "blocking_reasons": list(candidate.reasons),
            "conflicts": [dict(conflict) for conflict in conflicts],
            "data_version": data_version,
        }
        snapshot_sha256 = _sha256_payload(payload)
        identity_hash = hashlib.sha256(
            (
                f"{data_version}\0{candidate.listing.listing_id}\0{candidate.product.product_id}"
            ).encode()
        ).hexdigest()
        return cls(
            queue_item_id=f"erq-{identity_hash[:24]}",
            listing=candidate.listing,
            product=candidate.product,
            blocking_score=candidate.blocking_score,
            blocking_reasons=candidate.reasons,
            conflicts=conflicts,
            data_version=data_version,
            snapshot_sha256=snapshot_sha256,
        )

    def with_review(
        self,
        *,
        state: ReviewState,
        reviewer_id: str,
        reviewed_at: str,
        human_label: HumanMatchLabel | None = None,
        reviewer_note: str | None = None,
    ) -> Self:
        if self.state is not ReviewState.PENDING:
            candidate = replace(
                self,
                state=state,
                reviewer_id=reviewer_id,
                reviewed_at=reviewed_at,
                human_label=human_label,
                reviewer_note=reviewer_note,
            )
            if candidate == self:
                return self
            raise ReviewConflictError(
                f"review item {self.queue_item_id} already has a different decision"
            )
        return replace(
            self,
            state=state,
            reviewer_id=reviewer_id,
            reviewed_at=reviewed_at,
            human_label=human_label,
            reviewer_note=reviewer_note,
        )

    @property
    def is_binary_human_label(self) -> bool:
        return self.state is ReviewState.LABELLED and self.human_label in {
            HumanMatchLabel.MATCH,
            HumanMatchLabel.NON_MATCH,
        }

    def to_pair_example(self) -> PairExample:
        """Convert only an attributable binary human judgment into model input."""

        if not self.is_binary_human_label:
            raise ValueError("only binary human labels can become PairExample rows")
        assert self.human_label is not None
        return PairExample(
            pair_id=f"human-{self.queue_item_id}",
            listing=self.listing,
            product=self.product,
            label=int(self.human_label is HumanMatchLabel.MATCH),
            is_synthetic=False,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "queue_item_id": self.queue_item_id,
            **self.snapshot_payload(),
            "snapshot_sha256": self.snapshot_sha256,
            "state": self.state.value,
            "human_label": self.human_label.value if self.human_label is not None else None,
            "reviewer_id": self.reviewer_id,
            "reviewed_at": self.reviewed_at,
            "reviewer_note": self.reviewer_note,
        }

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> Self:
        raw_conflicts = row.get("conflicts", ())
        if not isinstance(raw_conflicts, list | tuple) or not all(
            isinstance(conflict, Mapping) for conflict in raw_conflicts
        ):
            raise TypeError("conflicts must be a list of objects")
        label = row.get("human_label")
        return cls(
            queue_item_id=str(row["queue_item_id"]),
            listing=ListingRecord.from_dict(row["listing"]),
            product=CanonicalProductRecord.from_dict(row["product"]),
            blocking_score=float(row["blocking_score"]),
            blocking_reasons=tuple(str(value) for value in row.get("blocking_reasons", ())),
            conflicts=tuple(dict(conflict) for conflict in raw_conflicts),
            data_version=str(row["data_version"]),
            snapshot_sha256=str(row["snapshot_sha256"]),
            state=ReviewState(str(row.get("state", ReviewState.PENDING.value))),
            human_label=HumanMatchLabel(str(label)) if label else None,
            reviewer_id=str(row["reviewer_id"]) if row.get("reviewer_id") else None,
            reviewed_at=str(row["reviewed_at"]) if row.get("reviewed_at") else None,
            reviewer_note=str(row["reviewer_note"]) if row.get("reviewer_note") else None,
        )


@dataclass(frozen=True, slots=True)
class ReviewQueue:
    queue_id: str
    created_at: str
    source_policy: SourceUsePolicy
    items: tuple[ReviewQueueItem, ...]

    def __post_init__(self) -> None:
        if not self.queue_id.strip():
            raise ValueError("queue_id must not be empty")
        _validate_timestamp(self.created_at, field_name="created_at")
        if not self.items:
            raise ValueError("review queue must contain at least one candidate")
        item_ids = [item.queue_item_id for item in self.items]
        if len(item_ids) != len(set(item_ids)):
            raise ValueError("review queue contains duplicate queue_item_id values")
        pair_keys = [(item.listing.listing_id, item.product.product_id) for item in self.items]
        if len(pair_keys) != len(set(pair_keys)):
            raise ValueError("review queue contains duplicate listing/product pairs")
        if any(item.data_version != self.source_policy.data_version for item in self.items):
            raise ValueError("review item data version does not match queue source policy")
        if self.queue_id != self.calculate_queue_id():
            raise ValueError("review queue identifier does not match its candidate manifest")

    def calculate_queue_id(self) -> str:
        queue_payload = {
            "schema_version": REVIEW_QUEUE_SCHEMA_VERSION,
            "created_at": self.created_at,
            "source_policy": self.source_policy.to_dict(),
            "item_snapshot_hashes": sorted(item.snapshot_sha256 for item in self.items),
        }
        return f"er-review-{_sha256_payload(queue_payload)[:24]}"

    @classmethod
    def from_candidates(
        cls,
        candidates: Iterable[PCBlockingCandidate],
        *,
        source_policy: SourceUsePolicy,
        created_at: str,
    ) -> Self:
        _validate_timestamp(created_at, field_name="created_at")
        items = tuple(
            ReviewQueueItem.pending(candidate, data_version=source_policy.data_version)
            for candidate in candidates
        )
        queue_payload = {
            "schema_version": REVIEW_QUEUE_SCHEMA_VERSION,
            "created_at": created_at,
            "source_policy": source_policy.to_dict(),
            "item_snapshot_hashes": sorted(item.snapshot_sha256 for item in items),
        }
        queue_id = f"er-review-{_sha256_payload(queue_payload)[:24]}"
        return cls(
            queue_id=queue_id,
            created_at=created_at,
            source_policy=source_policy,
            items=items,
        )

    def item(self, queue_item_id: str) -> ReviewQueueItem:
        for item in self.items:
            if item.queue_item_id == queue_item_id:
                return item
        raise KeyError(queue_item_id)

    def replace_item(self, updated: ReviewQueueItem) -> Self:
        if updated.queue_item_id not in {item.queue_item_id for item in self.items}:
            raise KeyError(updated.queue_item_id)
        return replace(
            self,
            items=tuple(
                updated if item.queue_item_id == updated.queue_item_id else item
                for item in self.items
            ),
        )

    def human_labelled_examples(self) -> tuple[PairExample, ...]:
        """Return training rows only when source policy and human evidence both allow it."""

        if not self.source_policy.training_eligible:
            raise PermissionError(
                "queue source policy forbids model training; labels remain review evidence only"
            )
        return tuple(item.to_pair_example() for item in self.items if item.is_binary_human_label)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": REVIEW_QUEUE_SCHEMA_VERSION,
            "record_type": "manifest",
            "queue_id": self.queue_id,
            "created_at": self.created_at,
            "source_policy": self.source_policy.to_dict(),
            "item_count": len(self.items),
            "state_counts": {
                state.value: sum(item.state is state for item in self.items)
                for state in ReviewState
            },
            "label_provenance": "human_only; blank candidate rows are never implicit negatives",
        }

    def export_jsonl(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        rows = (
            self.manifest(),
            *(
                {
                    "schema_version": REVIEW_QUEUE_SCHEMA_VERSION,
                    "record_type": "item",
                    **item.to_dict(),
                }
                for item in self.items
            ),
        )
        _atomic_json_lines(destination, rows)
        return destination

    @classmethod
    def import_jsonl(cls, path: str | Path) -> Self:
        source = Path(path)
        decoded_rows: list[Any] = [
            json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line
        ]
        if not all(isinstance(row, Mapping) for row in decoded_rows):
            raise TypeError("every review queue JSONL row must be an object")
        rows = [dict(row) for row in decoded_rows]
        if not rows or rows[0].get("record_type") != "manifest":
            raise ValueError("review queue JSONL must begin with a manifest")
        manifest = rows[0]
        if manifest.get("schema_version") != REVIEW_QUEUE_SCHEMA_VERSION:
            raise ValueError("unsupported review queue schema")
        for row_number, row in enumerate(rows[1:], start=2):
            if row.get("schema_version") != REVIEW_QUEUE_SCHEMA_VERSION:
                raise ValueError(f"row {row_number}: unsupported review queue schema")
            if row.get("record_type") != "item":
                raise ValueError(f"row {row_number}: unknown review queue record type")
        items = tuple(ReviewQueueItem.from_dict(row) for row in rows[1:])
        if len(items) != int(manifest["item_count"]):
            raise ValueError("review queue item count does not match its manifest")
        return cls(
            queue_id=str(manifest["queue_id"]),
            created_at=str(manifest["created_at"]),
            source_policy=SourceUsePolicy.from_dict(manifest["source_policy"]),
            items=items,
        )

    def export_label_sheet(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8-sig",
                newline="",
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                writer = csv.DictWriter(handle, fieldnames=_CSV_COLUMNS)
                writer.writeheader()
                for item in self.items:
                    writer.writerow(
                        {
                            "queue_id": self.queue_id,
                            "queue_item_id": item.queue_item_id,
                            "snapshot_sha256": item.snapshot_sha256,
                            "listing_id": item.listing.listing_id,
                            "listing_title": item.listing.title,
                            "product_id": item.product.product_id,
                            "product_name": item.product.canonical_name,
                            "category": item.listing.category,
                            "blocking_score": item.blocking_score,
                            "blocking_reasons": "|".join(item.blocking_reasons),
                            "conflicts": _canonical_json(
                                {"values": [dict(value) for value in item.conflicts]}
                            ),
                            "state": item.state.value,
                            "human_label": item.human_label.value if item.human_label else "",
                            "reviewer_id": item.reviewer_id or "",
                            "reviewed_at": item.reviewed_at or "",
                            "reviewer_note": item.reviewer_note or "",
                        }
                    )
            os.replace(temporary, destination)
            temporary = None
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
        return destination

    def import_label_sheet(self, path: str | Path) -> Self:
        source = Path(path)
        updates: dict[str, ReviewQueueItem] = {}
        with source.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            missing = set(_CSV_COLUMNS) - set(reader.fieldnames or ())
            if missing:
                raise ValueError(f"label sheet is missing columns: {sorted(missing)}")
            for row_number, row in enumerate(reader, start=2):
                if row["queue_id"] != self.queue_id:
                    raise ValueError(f"row {row_number}: queue_id mismatch")
                queue_item_id = row["queue_item_id"]
                if queue_item_id in updates:
                    raise ValueError(f"row {row_number}: duplicate queue_item_id")
                current = self.item(queue_item_id)
                if row["snapshot_sha256"] != current.snapshot_sha256:
                    raise ValueError(f"row {row_number}: candidate snapshot hash mismatch")
                state = ReviewState(row["state"].strip() or ReviewState.PENDING.value)
                label_text = row["human_label"].strip()
                if state is ReviewState.PENDING:
                    if label_text or row["reviewer_id"].strip() or row["reviewed_at"].strip():
                        raise ValueError(f"row {row_number}: pending row contains review fields")
                    updates[queue_item_id] = current
                    continue
                label = HumanMatchLabel(label_text) if label_text else None
                updates[queue_item_id] = current.with_review(
                    state=state,
                    human_label=label,
                    reviewer_id=row["reviewer_id"].strip(),
                    reviewed_at=row["reviewed_at"].strip(),
                    reviewer_note=row["reviewer_note"].strip() or None,
                )
        if set(updates) != {item.queue_item_id for item in self.items}:
            raise ValueError("label sheet must contain every queue item exactly once")
        return replace(self, items=tuple(updates[item.queue_item_id] for item in self.items))


def _atomic_json_lines(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            for row in rows:
                handle.write(_canonical_json(row))
                handle.write("\n")
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
