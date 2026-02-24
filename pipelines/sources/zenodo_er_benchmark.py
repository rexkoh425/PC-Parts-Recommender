"""Pinned, licensed entity-resolution transfer benchmark from Zenodo.

The source is record 8164151, version v3.  Its description lists Dn7 as the
Walmart-Amazon clean-clean product benchmark and the record applies CC BY 4.0
to the deposited files.  This adapter deliberately labels the output as a
*transfer benchmark*: it is not evidence about Singapore PC-retailer listings.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import re
import tempfile
import zipfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pc_build_recommender.entity_resolution import (
    CanonicalProductRecord,
    ListingRow,
    LabelledPair,
)
from pipelines.sources.base import (
    RawSnapshot,
    SnapshotError,
    fetch_http_snapshot,
    sha256_file,
    snapshot_local_file,
)

ZENODO_RECORD_ID = "8164151"
ZENODO_VERSION = "v3"
ZENODO_DOI = "10.5281/zenodo.8164151"
ZENODO_RECORD_URL = f"https://zenodo.org/records/{ZENODO_RECORD_ID}"
ZENODO_DN7_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}/files/Dn7.zip/content"
ZENODO_DN7_MD5 = "ec3f4f1a09aa434b40b266d16799535d"
ZENODO_DN7_SHA256 = "3e3fd6951ab4c4ed6aa741c2594d3ab496b63aeca6b41b8a1e639bc6d9895980"
ZENODO_LICENSE = "Creative Commons Attribution 4.0 International (CC BY 4.0)"
ZENODO_CREATOR = "George Papadakis, University of Athens"
ZENODO_DATASET_NAME = "Dn7 / Walmart-Amazon product entity-matching benchmark"
ZENODO_PARSER_VERSION = "zenodo-er-dn7-v1"
ZENODO_LICENSE_NOTE = (
    f"{ZENODO_LICENSE}; creator: {ZENODO_CREATOR}; DOI: {ZENODO_DOI}. "
    "Attribution is required. Metrics are transfer-benchmark-only and do not validate "
    "PC-retailer entity resolution."
)

_MEMBERS = {
    "left": "Dn7/tableA.csv",
    "right": "Dn7/tableB.csv",
    "train": "Dn7/train_set.csv",
    "validation": "Dn7/valid_set.csv",
    "test": "Dn7/test_set.csv",
}
_TABLE_FIELDS = ("id", "title", "modelno", "price", "shipweight", "brand", "dimensions")
_PAIR_FIELDS = ("id", "left_id", "right_id", "label")
_SPLIT_NAMES = ("train", "validation", "test")


@dataclass(frozen=True, slots=True)
class SourceProductRecord:
    """One row from a source-side product table."""

    record_id: str
    title: str
    model_number: str | None
    price_usd: float | None
    shipping_weight_lb: float | None
    brand: str
    dimensions: str | None
    price_source_text: str | None = None
    shipping_weight_source_text: str | None = None


@dataclass(frozen=True, slots=True)
class SourcePairLabel:
    """One deposited pair label, retaining its original split for provenance."""

    source_pair_id: str
    left_id: str
    right_id: str
    label: int
    original_split: str


@dataclass(frozen=True, slots=True)
class EntityMatchingSourceDataset:
    left: Mapping[str, SourceProductRecord]
    right: Mapping[str, SourceProductRecord]
    pairs: tuple[SourcePairLabel, ...]


@dataclass(frozen=True, slots=True)
class LeakageSafeTransferSplit:
    """Record-disjoint pair splits and the audit data proving the boundary."""

    pairs: Mapping[str, tuple[SourcePairLabel, ...]]
    left_assignments: Mapping[str, str]
    right_assignments: Mapping[str, str]
    dropped_cross_split_pairs: int
    dropped_cross_split_positives: int


def _clean_text(value: str | None) -> str:
    result = (value or "").strip()
    # The deposited CSV represents several text values with one literal quote layer.
    while len(result) >= 2 and result[0] == result[-1] == '"':
        result = result[1:-1].strip()
    return result


def _optional_text(value: str | None) -> str | None:
    cleaned = _clean_text(value)
    return cleaned or None


_PLAIN_NONNEGATIVE_NUMBER = re.compile(r"^\d+(?:\.\d+)?$")


def _source_number(value: str | None) -> tuple[float | None, str | None]:
    """Parse only plain non-negative decimal literals and retain the source text.

    Values such as ``1 206`` may be human-formatted thousands, malformed decimals, or
    another source convention. Interpreting them would invent semantics, so they remain
    available as raw provenance while the model-facing numeric value is missing.
    """

    cleaned = _clean_text(value)
    if not cleaned:
        return None, None
    if not _PLAIN_NONNEGATIVE_NUMBER.fullmatch(cleaned):
        return None, cleaned
    result = float(cleaned)
    if not math.isfinite(result):
        return None, cleaned
    return result, cleaned


def _read_csv(archive: zipfile.ZipFile, member: str) -> list[dict[str, str]]:
    with (
        archive.open(member) as binary,
        io.TextIOWrapper(binary, encoding="utf-8-sig", newline="") as text,
    ):
        return [dict(row) for row in csv.DictReader(text)]


def _parse_table(
    rows: Iterable[Mapping[str, str]], *, member: str
) -> dict[str, SourceProductRecord]:
    result: dict[str, SourceProductRecord] = {}
    for row_number, row in enumerate(rows, start=2):
        missing = set(_TABLE_FIELDS) - set(row)
        if missing:
            raise ValueError(f"{member}:{row_number}: missing columns {sorted(missing)}")
        record_id = _clean_text(row["id"])
        title = _clean_text(row["title"])
        if not record_id or not title:
            raise ValueError(f"{member}:{row_number}: id and title are required")
        if record_id in result:
            raise ValueError(f"{member}:{row_number}: duplicate id {record_id!r}")
        price, price_source_text = _source_number(row.get("price"))
        shipping_weight, shipping_weight_source_text = _source_number(row.get("shipweight"))
        result[record_id] = SourceProductRecord(
            record_id=record_id,
            title=title,
            model_number=_optional_text(row.get("modelno")),
            price_usd=price,
            shipping_weight_lb=shipping_weight,
            brand=_clean_text(row.get("brand")),
            dimensions=_optional_text(row.get("dimensions")),
            price_source_text=price_source_text,
            shipping_weight_source_text=shipping_weight_source_text,
        )
    if not result:
        raise ValueError(f"{member}: source table is empty")
    return result


def _parse_pairs(
    rows: Iterable[Mapping[str, str]],
    *,
    member: str,
    original_split: str,
    left: Mapping[str, SourceProductRecord],
    right: Mapping[str, SourceProductRecord],
) -> list[SourcePairLabel]:
    result: list[SourcePairLabel] = []
    seen: set[tuple[str, str]] = set()
    for row_number, row in enumerate(rows, start=2):
        missing = set(_PAIR_FIELDS) - set(row)
        if missing:
            raise ValueError(f"{member}:{row_number}: missing columns {sorted(missing)}")
        source_id = _clean_text(row["id"])
        left_id = _clean_text(row["left_id"])
        right_id = _clean_text(row["right_id"])
        try:
            label = int(_clean_text(row["label"]))
        except ValueError as error:
            raise ValueError(f"{member}:{row_number}: invalid binary label") from error
        if label not in (0, 1):
            raise ValueError(f"{member}:{row_number}: invalid binary label {label}")
        if left_id not in left or right_id not in right:
            raise ValueError(
                f"{member}:{row_number}: pair references an unknown source-table record"
            )
        key = (left_id, right_id)
        if key in seen:
            raise ValueError(f"{member}:{row_number}: duplicate pair {key!r}")
        seen.add(key)
        result.append(
            SourcePairLabel(
                source_pair_id=f"{original_split}:{source_id}",
                left_id=left_id,
                right_id=right_id,
                label=label,
                original_split=original_split,
            )
        )
    if not result:
        raise ValueError(f"{member}: pair table is empty")
    return result


class ZenodoEntityMatchingDn7Adapter:
    """Fetch and parse the immutable Dn7 deposit with hash verification."""

    def __init__(self, *, raw_root: str | Path) -> None:
        self.raw_root = Path(raw_root)

    def fetch(self, *, archive_path: str | Path | None = None) -> RawSnapshot:
        if archive_path is None:
            return fetch_http_snapshot(
                source_name="zenodo_er_dn7",
                source_url=ZENODO_DN7_URL,
                source_type="open_benchmark_import",
                raw_root=self.raw_root,
                parser_version=ZENODO_PARSER_VERSION,
                licence_or_access_note=ZENODO_LICENSE_NOTE,
                suffix=".zip",
                expected_sha256=ZENODO_DN7_SHA256,
                maximum_bytes=16 * 1024 * 1024,
            )
        snapshot = snapshot_local_file(
            source_name="zenodo_er_dn7",
            source_url=ZENODO_DN7_URL,
            source_type="open_benchmark_import",
            source_path=archive_path,
            raw_root=self.raw_root,
            parser_version=ZENODO_PARSER_VERSION,
            licence_or_access_note=ZENODO_LICENSE_NOTE,
            suffix=".zip",
            media_type="application/zip",
        )
        if snapshot.content_sha256 != ZENODO_DN7_SHA256:
            raise SnapshotError(
                "local Dn7 archive does not match the pinned Zenodo v3 SHA-256: "
                f"{snapshot.content_sha256}"
            )
        return snapshot

    def parse(self, snapshot: RawSnapshot) -> EntityMatchingSourceDataset:
        if snapshot.content_sha256 != ZENODO_DN7_SHA256:
            raise SnapshotError("refusing to parse an unpinned Dn7 snapshot")
        return _parse_archive(snapshot.path)


def _parse_archive(path: str | Path) -> EntityMatchingSourceDataset:
    """Parse an archive after its caller has established trust in the bytes.

    Production calls always go through :meth:`ZenodoEntityMatchingDn7Adapter.parse`, which
    checks the pinned SHA-256 first. Keeping byte trust separate from CSV parsing lets tiny
    synthetic archives exercise the parser without weakening the ingestion boundary.
    """

    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        missing = set(_MEMBERS.values()) - names
        if missing:
            raise ValueError(f"Dn7 archive is missing required members: {sorted(missing)}")
        left = _parse_table(_read_csv(archive, _MEMBERS["left"]), member=_MEMBERS["left"])
        right = _parse_table(_read_csv(archive, _MEMBERS["right"]), member=_MEMBERS["right"])
        pairs: list[SourcePairLabel] = []
        global_pairs: set[tuple[str, str]] = set()
        for split_name in _SPLIT_NAMES:
            parsed = _parse_pairs(
                _read_csv(archive, _MEMBERS[split_name]),
                member=_MEMBERS[split_name],
                original_split=split_name,
                left=left,
                right=right,
            )
            for pair in parsed:
                key = (pair.left_id, pair.right_id)
                if key in global_pairs:
                    raise ValueError(f"pair {key!r} occurs in multiple deposited splits")
                global_pairs.add(key)
            pairs.extend(parsed)
    return EntityMatchingSourceDataset(left=left, right=right, pairs=tuple(pairs))


class _UnionFind:
    def __init__(self, nodes: Iterable[str]) -> None:
        self.parent = {node: node for node in nodes}

    def find(self, node: str) -> str:
        parent = self.parent[node]
        if parent != node:
            self.parent[node] = self.find(parent)
        return self.parent[node]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        smaller, larger = sorted((left_root, right_root))
        self.parent[larger] = smaller


def _stable_split(component_key: str, *, seed: int) -> str:
    digest = hashlib.sha256(f"{seed}:{component_key}".encode()).digest()
    fraction = int.from_bytes(digest[:8], byteorder="big") / 2**64
    if fraction < 0.60:
        return "train"
    if fraction < 0.80:
        return "validation"
    return "test"


def record_disjoint_split(
    dataset: EntityMatchingSourceDataset,
    *,
    seed: int = 20260722,
) -> LeakageSafeTransferSplit:
    """Re-split pairs so neither source record can leak across train/validation/test.

    Positive links first define identity components. Every raw record then receives one
    deterministic split. Cross-split negative candidate pairs are discarded; positive
    pairs can never be discarded because both ends belong to the same identity component.
    """

    left_nodes = {record_id: f"left:{record_id}" for record_id in dataset.left}
    right_nodes = {record_id: f"right:{record_id}" for record_id in dataset.right}
    union_find = _UnionFind((*left_nodes.values(), *right_nodes.values()))
    for pair in dataset.pairs:
        if pair.label == 1:
            union_find.union(left_nodes[pair.left_id], right_nodes[pair.right_id])

    component_splits: dict[str, str] = {}

    def assignment(node: str) -> str:
        root = union_find.find(node)
        return component_splits.setdefault(root, _stable_split(root, seed=seed))

    left_assignments = {record_id: assignment(node) for record_id, node in left_nodes.items()}
    right_assignments = {record_id: assignment(node) for record_id, node in right_nodes.items()}
    split_pairs: dict[str, list[SourcePairLabel]] = {name: [] for name in _SPLIT_NAMES}
    dropped = 0
    dropped_positives = 0
    for pair in dataset.pairs:
        left_split = left_assignments[pair.left_id]
        right_split = right_assignments[pair.right_id]
        if left_split != right_split:
            dropped += 1
            dropped_positives += pair.label
            continue
        split_pairs[left_split].append(pair)

    if dropped_positives:
        raise AssertionError("positive identity pairs must remain in the same split")
    for split_name, rows in split_pairs.items():
        if {row.label for row in rows} != {0, 1}:
            raise ValueError(f"{split_name} lacks both labels after record-disjoint splitting")

    return LeakageSafeTransferSplit(
        pairs={name: tuple(rows) for name, rows in split_pairs.items()},
        left_assignments=left_assignments,
        right_assignments=right_assignments,
        dropped_cross_split_pairs=dropped,
        dropped_cross_split_positives=dropped_positives,
    )


def _source_attributes(record: SourceProductRecord, *, side: str) -> dict[str, Any]:
    # Prices stay explicitly USD and do not enter price_sgd fields. This avoids silently
    # contaminating the production price feature with a false currency interpretation.
    return {
        "benchmark_source_side": side,
        "source_currency": "USD",
        "source_price_usd": record.price_usd,
        "source_price_text": record.price_source_text,
        "source_shipping_weight_lb": record.shipping_weight_lb,
        "source_shipping_weight_text": record.shipping_weight_source_text,
        "source_dimensions": record.dimensions,
    }


def adapt_pair(
    pair: SourcePairLabel,
    *,
    dataset: EntityMatchingSourceDataset,
    assigned_split: str,
) -> LabelledPair:
    """Adapt one deposited label to the project's typed pair contract."""

    left = dataset.left[pair.left_id]
    right = dataset.right[pair.right_id]
    listing = ListingRow(
        listing_id=f"zenodo-dn7-left-{left.record_id}",
        title=left.title,
        category="transfer_consumer_products",
        brand=left.brand,
        manufacturer_part_number=left.model_number,
        attributes=_source_attributes(left, side="left"),
        current_price_sgd=None,
        retailer="transfer_benchmark_source_a",
        is_synthetic=False,
    )
    product = CanonicalProductRecord(
        product_id=f"zenodo-dn7-right-{right.record_id}",
        category="transfer_consumer_products",
        brand=right.brand,
        model=right.model_number or "",
        canonical_name=right.title,
        manufacturer_part_number=right.model_number,
        attributes=_source_attributes(right, side="right"),
        price_sgd=None,
        is_synthetic=False,
    )
    return LabelledPair(
        pair_id=f"zenodo-dn7-{assigned_split}-{pair.source_pair_id}",
        listing=listing,
        product=product,
        label=pair.label,
        is_synthetic=False,
    )


def _write_json_lines(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
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
            count = 0
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False))
                handle.write("\n")
                count += 1
        os.replace(temporary, path)
        temporary = None
        return count
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def materialize_transfer_pairs(
    dataset: EntityMatchingSourceDataset,
    snapshot: RawSnapshot,
    *,
    processed_root: str | Path,
    seed: int = 20260722,
) -> Path:
    """Write deterministic, record-disjoint LabelledPair JSONL files and a manifest."""

    split = record_disjoint_split(dataset, seed=seed)
    destination = Path(processed_root) / "zenodo_er_dn7" / snapshot.content_sha256
    destination.mkdir(parents=True, exist_ok=True)
    file_evidence: dict[str, dict[str, Any]] = {}
    for split_name in _SPLIT_NAMES:
        path = destination / f"pairs.{split_name}.jsonl"
        pairs = [
            adapt_pair(row, dataset=dataset, assigned_split=split_name)
            for row in split.pairs[split_name]
        ]
        row_count = _write_json_lines(path, (pair.to_dict() for pair in pairs))
        file_evidence[split_name] = {
            "path": path.name,
            "rows": row_count,
            "positives": sum(pair.label for pair in pairs),
            "left_entities": len({pair.listing.listing_id for pair in pairs}),
            "right_entities": len({pair.product.product_id for pair in pairs}),
            "sha256": sha256_file(path),
        }

    left_sets = {
        split_name: {
            record_id
            for record_id, assigned in split.left_assignments.items()
            if assigned == split_name
        }
        for split_name in _SPLIT_NAMES
    }
    right_sets = {
        split_name: {
            record_id
            for record_id, assigned in split.right_assignments.items()
            if assigned == split_name
        }
        for split_name in _SPLIT_NAMES
    }
    overlap_checks = {
        "left_train_validation": len(left_sets["train"] & left_sets["validation"]),
        "left_train_test": len(left_sets["train"] & left_sets["test"]),
        "left_validation_test": len(left_sets["validation"] & left_sets["test"]),
        "right_train_validation": len(right_sets["train"] & right_sets["validation"]),
        "right_train_test": len(right_sets["train"] & right_sets["test"]),
        "right_validation_test": len(right_sets["validation"] & right_sets["test"]),
    }
    if any(overlap_checks.values()):
        raise AssertionError("record-disjoint split contains source-entity leakage")

    manifest = {
        "schema_version": "pc-build-recommender.er-transfer-dataset.v1",
        "claim_scope": "transfer_benchmark_only",
        "pc_retailer_production_claim_eligible": False,
        "pc_retailer_production_claim_block_reason": (
            "The source domain and candidate-generation process differ from Singapore PC listings."
        ),
        "source": {
            "dataset_name": ZENODO_DATASET_NAME,
            "creator": ZENODO_CREATOR,
            "record_id": ZENODO_RECORD_ID,
            "version": ZENODO_VERSION,
            "doi": ZENODO_DOI,
            "record_url": ZENODO_RECORD_URL,
            "file_url": ZENODO_DN7_URL,
            "file_name": "Dn7.zip",
            "license": ZENODO_LICENSE,
            "deposited_md5": ZENODO_DN7_MD5,
            "verified_sha256": snapshot.content_sha256,
            "bytes": snapshot.byte_count,
            "retrieved_at": snapshot.retrieved_at.isoformat(),
            "parser_version": ZENODO_PARSER_VERSION,
        },
        "original": {
            "left_records": len(dataset.left),
            "right_records": len(dataset.right),
            "pairs": len(dataset.pairs),
            "positives": sum(pair.label for pair in dataset.pairs),
            "deposited_split_counts": {
                name: sum(pair.original_split == name for pair in dataset.pairs)
                for name in _SPLIT_NAMES
            },
            "unparsed_numeric_source_values": {
                "left_price": sum(
                    record.price_source_text is not None and record.price_usd is None
                    for record in dataset.left.values()
                ),
                "left_shipping_weight": sum(
                    record.shipping_weight_source_text is not None
                    and record.shipping_weight_lb is None
                    for record in dataset.left.values()
                ),
                "right_price": sum(
                    record.price_source_text is not None and record.price_usd is None
                    for record in dataset.right.values()
                ),
                "right_shipping_weight": sum(
                    record.shipping_weight_source_text is not None
                    and record.shipping_weight_lb is None
                    for record in dataset.right.values()
                ),
            },
        },
        "split_policy": {
            "name": "positive_identity_components_then_record_disjoint_hash_v1",
            "seed": seed,
            "weights": {"train": 0.6, "validation": 0.2, "test": 0.2},
            "leakage_units": ["left_source_record_id", "right_source_record_id"],
            "cross_split_negative_pairs_dropped": split.dropped_cross_split_pairs,
            "cross_split_positive_pairs_dropped": split.dropped_cross_split_positives,
            "overlap_checks": overlap_checks,
        },
        "files": file_evidence,
        "currency_policy": (
            "Deposited USD values are retained only in source attributes; SGD fields are null."
        ),
        "numeric_parse_policy": (
            "Only plain non-negative decimal literals become numeric features. Other nonblank "
            "values remain verbatim source-text attributes with a missing numeric value."
        ),
    }
    _write_json(destination / "dataset_manifest.json", manifest)
    return destination
