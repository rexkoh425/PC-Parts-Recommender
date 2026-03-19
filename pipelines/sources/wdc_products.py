"""Bounded, resumable import of the historical WDC Products research corpus.

This module deliberately does not emit :class:`RetailerOffering` objects.  WDC
Products is a historical, multi-market research corpus extracted from 2020 web
pages.  Its rows are useful for exercising ingestion and candidate-discovery
code, but they are not evidence of current Singapore prices or availability.

The public download page does not establish the production/commercial rights
required by this project.  Consequently every normalized row is quarantined and
all downstream rights other than retaining the immutable research download are
denied.  In particular, the output is ineligible for serving, embeddings,
training, published model claims, and production entity resolution.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import sqlite3
import tempfile
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, BinaryIO, Final, cast

from pc_build_recommender.data_rights import production_catalog_rights_are_valid
from pipelines.sources.base import (
    FetchedSnapshot,
    fetch_http_snapshot,
    sha256_bytes,
    sha256_file,
    snapshot_local_file,
    utc_now,
)

WDC_PRODUCTS_PAGE_URL: Final = "https://webdatacommons.org/largescaleproductcorpus/wdc-products/"
WDC_PRODUCTS_CORPUS_URL: Final = (
    "https://data.dws.informatik.uni-mannheim.de/largescaleproductcorpus/"
    "data/wdc-products/wdcproducts_corpus_with_url.json.gz"
)
WDC_PRODUCTS_CATEGORIES_URL: Final = (
    "https://data.dws.informatik.uni-mannheim.de/largescaleproductcorpus/"
    "data/wdc-products/categorization/"
    "WDC_Corpus_LargeScaleExperiment_MajorityVoting.json.gz"
)
WDC_CORPUS_SOURCE_NAME: Final = "wdc_products_pdc2020c_research"
WDC_CATEGORY_SOURCE_NAME: Final = "wdc_products_pdc2020c_categories_research"
WDC_PARSER_VERSION: Final = "wdc-pdc2020c-research-quarantine-v1"
WDC_CATEGORY_INDEX_SCHEMA: Final = "pc-build-recommender.wdc-category-index.v1"
WDC_RESEARCH_RECORD_SCHEMA: Final = "pc-build-recommender.wdc-research-offer.v1"
WDC_RESEARCH_MANIFEST_SCHEMA: Final = "pc-build-recommender.wdc-research-manifest.v1"
WDC_SELECTION_POLICY_VERSION: Final = "pc-component-candidate-rules-v1"

# The upstream corpus is currently about 5.1 GB compressed.  These are hard
# safety ceilings, not expected consumption.  The parser never materializes the
# decompressed corpus and keeps at most one bounded source line in memory.
MAX_CORPUS_DOWNLOAD_BYTES: Final = 6 * 1024**3
MAX_CATEGORY_DOWNLOAD_BYTES: Final = 256 * 1024**2
DEFAULT_MAX_CORPUS_DECOMPRESSED_BYTES: Final = 64 * 1024**3
DEFAULT_MAX_CATEGORY_DECOMPRESSED_BYTES: Final = 4 * 1024**3
DEFAULT_MAX_LINE_BYTES: Final = 2 * 1024**2
DEFAULT_CHECKPOINT_INTERVAL: Final = 10_000
DEFAULT_MAX_SELECTED_RECORDS: Final = 100_000
HARD_MAX_SELECTED_RECORDS: Final = 250_000
DEFAULT_MAX_OUTPUT_BYTES: Final = 1024**3
DESCRIPTION_EXCERPT_CHARACTERS: Final = 4096
WDC_RESEARCH_RETENTION_DAYS: Final = 365

WDC_ACCESS_NOTE: Final = (
    "Historical PDC2020-C research corpus made available for public download by Web Data "
    "Commons. The project has not established a licence or contract granting production "
    "catalogue display, retained price history, redistribution, embeddings, model training, "
    "derived commercial data, or Singapore-market use. Immutable source caching is limited "
    "to an internal research quarantine."
)

# This is intentionally a complete DataUseRights-shaped mapping so every generic
# gate can fail closed.  Public availability supports retaining the research
# download; it is not treated as a grant for any downstream product use.
WDC_RESEARCH_DATA_USE_RIGHTS: Final[dict[str, object]] = {
    "contract_reference": "wdc-products-public-research-download-no-production-contract",
    "contract_version_url": WDC_PRODUCTS_PAGE_URL,
    "consent_effective_on": "2022-12-22",
    "consent_expires_on": None,
    "retention_days": WDC_RESEARCH_RETENTION_DAYS,
    "deletion_required_on_termination": False,
    "deletion_sla_days": None,
    "territories": ["INTERNAL_RESEARCH"],
    "may_display": False,
    "may_cache": True,
    "may_store_history": False,
    "may_redistribute": False,
    "may_embed": False,
    "may_train": False,
    "may_derive": False,
}

_COMPUTERS_CATEGORY = "Computers_and_Accessories"
_CATEGORY_FIELD = "predicted_CategoryLabel_majority_voted"
_INITIAL_CHAIN = bytes(32)


class WDCResearchImportError(RuntimeError):
    """Raised when a WDC research import cannot remain bounded and auditable."""


class WDCResearchLimitError(WDCResearchImportError):
    """Raised before an input or output resource ceiling is exceeded."""


@dataclass(frozen=True, slots=True)
class WDCCategoryIndexResult:
    index_path: Path
    source_sha256: str
    completed_source_lines: int
    indexed_clusters: int
    complete: bool
    reused: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "index_path": str(self.index_path),
            "source_sha256": self.source_sha256,
            "completed_source_lines": self.completed_source_lines,
            "indexed_clusters": self.indexed_clusters,
            "complete": self.complete,
            "reused": self.reused,
        }


@dataclass(frozen=True, slots=True)
class WDCResearchImportResult:
    source_sha256: str
    category_source_sha256: str
    completed_source_lines: int
    selected_records: int
    rejected_non_computer_category: int
    rejected_not_component_like: int
    rejected_ambiguous_component: int
    complete: bool
    reused: bool
    output_path: Path | None
    manifest_path: Path | None
    work_path: Path | None

    def to_dict(self) -> dict[str, object]:
        return {
            "source_sha256": self.source_sha256,
            "category_source_sha256": self.category_source_sha256,
            "completed_source_lines": self.completed_source_lines,
            "selected_records": self.selected_records,
            "rejected_non_computer_category": self.rejected_non_computer_category,
            "rejected_not_component_like": self.rejected_not_component_like,
            "rejected_ambiguous_component": self.rejected_ambiguous_component,
            "complete": self.complete,
            "reused": self.reused,
            "output_path": str(self.output_path) if self.output_path is not None else None,
            "manifest_path": str(self.manifest_path) if self.manifest_path is not None else None,
            "work_path": str(self.work_path) if self.work_path is not None else None,
            "production_eligible": False,
            "singapore_market_evidence": False,
            "current_price_or_stock_evidence": False,
            "model_training_eligible": False,
        }


class WDCProductsResearchSource:
    """Create content-addressed snapshots of explicitly requested WDC downloads."""

    def __init__(self, raw_root: str | Path) -> None:
        self.raw_root = Path(raw_root)

    def fetch_corpus(self, *, corpus_path: str | Path | None = None) -> FetchedSnapshot:
        if corpus_path is not None:
            suffix, media_type = _local_jsonl_format(corpus_path)
            return snapshot_local_file(
                source_name=WDC_CORPUS_SOURCE_NAME,
                source_url=WDC_PRODUCTS_CORPUS_URL,
                source_type="historical_research_web_product_corpus",
                source_path=corpus_path,
                raw_root=self.raw_root,
                parser_version=WDC_PARSER_VERSION,
                licence_or_access_note=WDC_ACCESS_NOTE,
                suffix=suffix,
                media_type=media_type,
            )
        return fetch_http_snapshot(
            source_name=WDC_CORPUS_SOURCE_NAME,
            source_url=WDC_PRODUCTS_CORPUS_URL,
            source_type="historical_research_web_product_corpus",
            raw_root=self.raw_root,
            parser_version=WDC_PARSER_VERSION,
            licence_or_access_note=WDC_ACCESS_NOTE,
            suffix=".json.gz",
            maximum_bytes=MAX_CORPUS_DOWNLOAD_BYTES,
            timeout_seconds=1800,
        )

    def fetch_categories(self, *, category_path: str | Path | None = None) -> FetchedSnapshot:
        if category_path is not None:
            suffix, media_type = _local_jsonl_format(category_path)
            return snapshot_local_file(
                source_name=WDC_CATEGORY_SOURCE_NAME,
                source_url=WDC_PRODUCTS_CATEGORIES_URL,
                source_type="historical_research_product_category_predictions",
                source_path=category_path,
                raw_root=self.raw_root,
                parser_version=WDC_PARSER_VERSION,
                licence_or_access_note=WDC_ACCESS_NOTE,
                suffix=suffix,
                media_type=media_type,
            )
        return fetch_http_snapshot(
            source_name=WDC_CATEGORY_SOURCE_NAME,
            source_url=WDC_PRODUCTS_CATEGORIES_URL,
            source_type="historical_research_product_category_predictions",
            raw_root=self.raw_root,
            parser_version=WDC_PARSER_VERSION,
            licence_or_access_note=WDC_ACCESS_NOTE,
            suffix=".json.gz",
            maximum_bytes=MAX_CATEGORY_DOWNLOAD_BYTES,
            timeout_seconds=900,
        )


def _local_jsonl_format(path: str | Path) -> tuple[str, str]:
    source = Path(path)
    with source.open("rb") as handle:
        gzip_encoded = handle.read(2) == b"\x1f\x8b"
    if gzip_encoded:
        return ".json.gz", "application/gzip"
    return ".json", "application/x-ndjson"


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
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
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
            temporary_path = Path(handle.name)
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _retention_deadline(snapshot: FetchedSnapshot) -> datetime:
    return snapshot.retrieved_at + timedelta(days=WDC_RESEARCH_RETENTION_DAYS)


def _assert_snapshot_within_retention(snapshot: FetchedSnapshot) -> None:
    if utc_now() > _retention_deadline(snapshot):
        raise PermissionError(
            f"WDC research snapshot exceeded its {WDC_RESEARCH_RETENTION_DAYS}-day "
            "internal retention limit; remove it and perform a new rights review before reuse"
        )


def _open_maybe_gzip(path: Path) -> BinaryIO:
    with path.open("rb") as probe:
        magic = probe.read(2)
    if magic == b"\x1f\x8b":
        return cast(BinaryIO, gzip.open(path, "rb"))
    return path.open("rb")


def _iter_json_lines(
    path: Path,
    *,
    max_line_bytes: int,
    max_decompressed_bytes: int,
    start_after_line: int = 0,
) -> Iterator[tuple[int, bytes, Mapping[str, Any]]]:
    if max_line_bytes < 128:
        raise ValueError("max_line_bytes must be at least 128")
    if max_decompressed_bytes < max_line_bytes:
        raise ValueError("max_decompressed_bytes must be at least max_line_bytes")
    decompressed_bytes = 0
    with _open_maybe_gzip(path) as handle:
        line_number = 0
        while True:
            raw_line = handle.readline(max_line_bytes + 1)
            if not raw_line:
                break
            line_number += 1
            if len(raw_line) > max_line_bytes:
                raise WDCResearchLimitError(
                    f"{path}:{line_number}: JSONL line exceeds {max_line_bytes} bytes"
                )
            decompressed_bytes += len(raw_line)
            if decompressed_bytes > max_decompressed_bytes:
                raise WDCResearchLimitError(
                    f"{path}: decompressed input exceeds {max_decompressed_bytes} bytes"
                )
            if not raw_line.strip():
                raise WDCResearchImportError(f"{path}:{line_number}: blank JSONL line")
            # A gzip resume must replay decompression from the beginning, but
            # it need not allocate/parse JSON for already committed rows.
            if line_number <= start_after_line:
                continue
            try:
                payload = json.loads(raw_line)
            except json.JSONDecodeError as error:
                raise WDCResearchImportError(
                    f"{path}:{line_number}: invalid JSON: {error}"
                ) from error
            if not isinstance(payload, Mapping):
                raise WDCResearchImportError(f"{path}:{line_number}: JSON root must be an object")
            yield line_number, raw_line, payload


def _metadata(connection: sqlite3.Connection) -> dict[str, str]:
    return {
        str(key): str(value)
        for key, value in connection.execute("SELECT key, value FROM import_metadata")
    }


def _set_metadata(connection: sqlite3.Connection, key: str, value: object) -> None:
    connection.execute(
        "INSERT INTO import_metadata(key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, str(value)),
    )


def _initialise_category_database(
    connection: sqlite3.Connection,
    *,
    snapshot: FetchedSnapshot,
) -> dict[str, str]:
    connection.execute("PRAGMA journal_mode=DELETE")
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute(
        "CREATE TABLE IF NOT EXISTS import_metadata ("
        "key TEXT PRIMARY KEY NOT NULL, value TEXT NOT NULL)"
    )
    connection.execute(
        "CREATE TABLE IF NOT EXISTS category_map ("
        "cluster_id TEXT PRIMARY KEY NOT NULL, category TEXT NOT NULL)"
    )
    metadata = _metadata(connection)
    if not metadata:
        for key, value in {
            "schema_version": WDC_CATEGORY_INDEX_SCHEMA,
            "parser_version": WDC_PARSER_VERSION,
            "source_sha256": snapshot.content_sha256,
            "source_url": snapshot.source_url,
            "source_retrieved_at": snapshot.retrieved_at.isoformat(),
            "retention_deadline": _retention_deadline(snapshot).isoformat(),
            "completed_source_lines": 0,
            "complete": 0,
        }.items():
            _set_metadata(connection, key, value)
        connection.commit()
        return _metadata(connection)
    if metadata.get("schema_version") != WDC_CATEGORY_INDEX_SCHEMA:
        raise WDCResearchImportError("unsupported WDC category-index schema")
    if metadata.get("parser_version") != WDC_PARSER_VERSION:
        raise WDCResearchImportError("WDC category index was built by a different parser version")
    if metadata.get("source_sha256") != snapshot.content_sha256:
        raise WDCResearchImportError(
            "category index is pinned to a different source snapshot; choose a new index path"
        )
    return metadata


def _required_identifier(payload: Mapping[str, Any], field: str, *, context: str) -> str:
    value = payload.get(field)
    if isinstance(value, bool) or not isinstance(value, str | int):
        raise WDCResearchImportError(f"{context}: {field} must be a string or integer")
    result = str(value).strip()
    if not result:
        raise WDCResearchImportError(f"{context}: {field} is required")
    return result


def build_wdc_category_index(
    snapshot: FetchedSnapshot,
    *,
    index_path: str | Path,
    record_budget: int | None = None,
    checkpoint_interval: int = DEFAULT_CHECKPOINT_INTERVAL,
    max_line_bytes: int = DEFAULT_MAX_LINE_BYTES,
    max_decompressed_bytes: int = DEFAULT_MAX_CATEGORY_DECOMPRESSED_BYTES,
) -> WDCCategoryIndexResult:
    """Build a disk-backed category lookup without retaining the mapping in RAM.

    A finite ``record_budget`` pauses after that many new input rows. Re-running
    against the same snapshot and index path resumes from the committed line.
    Gzip streams are replayed to that line, so resumption is memory bounded but
    not constant-time.
    """

    if record_budget is not None and record_budget < 1:
        raise ValueError("record_budget must be positive or null")
    if checkpoint_interval < 1:
        raise ValueError("checkpoint_interval must be positive")
    _assert_snapshot_within_retention(snapshot)
    destination = Path(index_path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(destination) as connection:
        metadata = _initialise_category_database(connection, snapshot=snapshot)
        completed_line = int(metadata["completed_source_lines"])
        if metadata["complete"] == "1":
            indexed = int(connection.execute("SELECT COUNT(*) FROM category_map").fetchone()[0])
            return WDCCategoryIndexResult(
                index_path=destination,
                source_sha256=snapshot.content_sha256,
                completed_source_lines=completed_line,
                indexed_clusters=indexed,
                complete=True,
                reused=True,
            )

        new_records = 0
        reached_eof = True
        last_line = completed_line
        rows = _iter_json_lines(
            snapshot.path,
            max_line_bytes=max_line_bytes,
            max_decompressed_bytes=max_decompressed_bytes,
            start_after_line=completed_line,
        )
        for source_offset, (line_number, _raw_line, payload) in enumerate(rows):
            if record_budget is not None and source_offset >= record_budget:
                reached_eof = False
                break
            context = f"{snapshot.path}:{line_number}"
            cluster_id = _required_identifier(payload, "cluster_id", context=context)
            raw_category = payload.get(_CATEGORY_FIELD)
            if not isinstance(raw_category, str) or not raw_category.strip():
                raise WDCResearchImportError(f"{context}: {_CATEGORY_FIELD} is required")
            category = raw_category.strip()
            cursor = connection.execute(
                "INSERT OR IGNORE INTO category_map(cluster_id, category) VALUES (?, ?)",
                (cluster_id, category),
            )
            if cursor.rowcount == 0:
                existing = connection.execute(
                    "SELECT category FROM category_map WHERE cluster_id = ?",
                    (cluster_id,),
                ).fetchone()
                if existing is None or str(existing[0]) != category:
                    raise WDCResearchImportError(
                        f"{context}: conflicting category for cluster {cluster_id}"
                    )
            last_line = line_number
            new_records = source_offset + 1
            if new_records % checkpoint_interval == 0:
                _set_metadata(connection, "completed_source_lines", last_line)
                connection.commit()

        _set_metadata(connection, "completed_source_lines", last_line)
        _set_metadata(connection, "complete", 1 if reached_eof else 0)
        connection.commit()
        indexed = int(connection.execute("SELECT COUNT(*) FROM category_map").fetchone()[0])
        return WDCCategoryIndexResult(
            index_path=destination,
            source_sha256=snapshot.content_sha256,
            completed_source_lines=last_line,
            indexed_clusters=indexed,
            complete=reached_eof,
            reused=False,
        )


_COMPONENT_PATTERNS: Final[tuple[tuple[str, tuple[re.Pattern[str], ...]], ...]] = (
    (
        "gpu",
        (
            re.compile(r"\b(?:geforce\s+(?:rtx|gtx)|radeon\s+rx)\s*\d", re.I),
            re.compile(r"\bgraphics\s+card\b", re.I),
        ),
    ),
    (
        "cpu",
        (
            re.compile(r"\b(?:ryzen|threadripper)\s+\d", re.I),
            re.compile(r"\bintel\s+core\s+(?:ultra\s+)?[i3579]?\s*\d", re.I),
            re.compile(r"\b(?:desktop|computer)\s+processor\b", re.I),
        ),
    ),
    (
        "motherboard",
        (
            re.compile(r"\bmotherboard\b", re.I),
            re.compile(r"\bmainboard\b", re.I),
        ),
    ),
    (
        "memory",
        (
            re.compile(r"\bddr[345]\b.{0,60}\b(?:ram|memory|dimm|sodimm|module|kit)\b", re.I),
            re.compile(r"\b(?:ram|memory|dimm|sodimm)\b.{0,60}\bddr[345]\b", re.I),
        ),
    ),
    (
        "storage",
        (
            re.compile(r"\b(?:nvme|solid[ -]state\s+drive|ssd)\b", re.I),
            re.compile(r"\bm\.2\b.{0,50}\b(?:drive|storage|[0-9]+\s*(?:gb|tb))\b", re.I),
        ),
    ),
    (
        "power_supply",
        (
            re.compile(r"\b(?:pc\s+)?power\s+supply\b.{0,80}\b\d{3,4}\s*w(?:att)?\b", re.I),
            re.compile(r"\bpsu\b.{0,80}\b\d{3,4}\s*w(?:att)?\b", re.I),
        ),
    ),
    (
        "cpu_cooler",
        (
            re.compile(r"\bcpu\s+(?:air\s+|liquid\s+)?cooler\b", re.I),
            re.compile(r"\b(?:aio|all[ -]in[ -]one)\b.{0,60}\b(?:cpu|radiator|cooler)\b", re.I),
            re.compile(r"\b(?:cpu\s+)?heat\s*sink\b", re.I),
        ),
    ),
    (
        "case",
        (
            re.compile(r"\b(?:pc|computer|gaming)\s+(?:tower\s+)?case\b", re.I),
            re.compile(r"\b(?:mini|mid|full)[ -]?tower\b", re.I),
            re.compile(r"\batx\s+(?:computer\s+)?case\b", re.I),
            re.compile(r"\bcomputer\s+chassis\b", re.I),
        ),
    ),
)
_UPS_PATTERN = re.compile(r"\b(?:ups|uninterruptible\s+power)\b", re.I)


def infer_pc_component_categories(title: str, description: str | None = None) -> tuple[str, ...]:
    """Return transparent heuristic component candidates, never a canonical label."""

    text = f"{title} {description or ''}"[: 32 * 1024]
    matches: list[str] = []
    for category, patterns in _COMPONENT_PATTERNS:
        if category == "power_supply" and _UPS_PATTERN.search(text):
            continue
        if any(pattern.search(text) for pattern in patterns):
            matches.append(category)
    return tuple(matches)


def _optional_text(value: object, *, maximum_characters: int | None = None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise WDCResearchImportError("optional WDC text field must be a string or null")
    result = value.strip()
    if not result:
        return None
    if maximum_characters is not None:
        return result[:maximum_characters]
    return result


def _chain_next(previous: bytes, line: bytes) -> bytes:
    return hashlib.sha256(previous + line).digest()


def _scan_partial_output(
    path: Path,
    *,
    expected_bytes: int,
) -> tuple[bytes, set[str]]:
    chain = _INITIAL_CHAIN
    source_ids: set[str] = set()
    consumed = 0
    if not path.exists():
        if expected_bytes:
            raise WDCResearchImportError("partial output is missing")
        return chain, source_ids
    with path.open("rb") as handle:
        for line_number, line in enumerate(handle, start=1):
            consumed += len(line)
            if consumed > expected_bytes:
                raise WDCResearchImportError("partial output exceeds its checkpoint boundary")
            chain = _chain_next(chain, line)
            try:
                payload = json.loads(line)
                source_id = payload["data"]["source_offer_id"]
            except (KeyError, TypeError, json.JSONDecodeError) as error:
                raise WDCResearchImportError(
                    f"partial output line {line_number} is invalid"
                ) from error
            if not isinstance(source_id, str) or source_id in source_ids:
                raise WDCResearchImportError("partial output contains an invalid duplicate ID")
            source_ids.add(source_id)
    if consumed != expected_bytes:
        raise WDCResearchImportError("partial output byte count does not match checkpoint")
    return chain, source_ids


def _checkpoint_payload(
    *,
    source_sha256: str,
    category_source_sha256: str,
    policy_sha256: str,
    completed_source_lines: int,
    selected_records: int,
    rejected_non_computer_category: int,
    rejected_not_component_like: int,
    rejected_ambiguous_component: int,
    output_bytes: int,
    output_chain_sha256: str,
) -> dict[str, object]:
    return {
        "schema_version": WDC_RESEARCH_MANIFEST_SCHEMA,
        "state": "working",
        "source_sha256": source_sha256,
        "category_source_sha256": category_source_sha256,
        "policy_sha256": policy_sha256,
        "completed_source_lines": completed_source_lines,
        "selected_records": selected_records,
        "rejected_non_computer_category": rejected_non_computer_category,
        "rejected_not_component_like": rejected_not_component_like,
        "rejected_ambiguous_component": rejected_ambiguous_component,
        "output_bytes": output_bytes,
        "output_chain_sha256": output_chain_sha256,
    }


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise WDCResearchImportError(f"cannot read JSON metadata {path}: {error}") from error
    if not isinstance(payload, dict):
        raise WDCResearchImportError(f"JSON metadata must be an object: {path}")
    return payload


def _category_index_metadata(index_path: Path) -> dict[str, str]:
    if not index_path.is_file():
        raise FileNotFoundError(f"WDC category index not found: {index_path}")
    with sqlite3.connect(f"file:{index_path}?mode=ro", uri=True) as connection:
        metadata = _metadata(connection)
    if metadata.get("schema_version") != WDC_CATEGORY_INDEX_SCHEMA:
        raise WDCResearchImportError("category index has an unsupported schema")
    if metadata.get("parser_version") != WDC_PARSER_VERSION:
        raise WDCResearchImportError("category index has a different parser version")
    if metadata.get("complete") != "1":
        raise WDCResearchImportError("category index is incomplete; resume it before corpus import")
    raw_deadline = metadata.get("retention_deadline")
    if raw_deadline is None:
        raise WDCResearchImportError("category index has no retention deadline")
    try:
        deadline = datetime.fromisoformat(raw_deadline)
    except ValueError as error:
        raise WDCResearchImportError("category index has an invalid retention deadline") from error
    if deadline.tzinfo is None:
        raise WDCResearchImportError("category-index retention deadline must be timezone aware")
    if utc_now() > deadline:
        raise PermissionError("WDC category-index research retention period has expired")
    return metadata


def _selection_policy_sha256(
    *,
    source_sha256: str,
    category_source_sha256: str,
    maximum_selected_records: int,
    maximum_output_bytes: int,
) -> str:
    payload = {
        "source_sha256": source_sha256,
        "category_source_sha256": category_source_sha256,
        "parser_version": WDC_PARSER_VERSION,
        "category_index_schema": WDC_CATEGORY_INDEX_SCHEMA,
        "record_schema": WDC_RESEARCH_RECORD_SCHEMA,
        "manifest_schema": WDC_RESEARCH_MANIFEST_SCHEMA,
        "selection_policy_version": WDC_SELECTION_POLICY_VERSION,
        "required_broad_category": _COMPUTERS_CATEGORY,
        "description_excerpt_characters": DESCRIPTION_EXCERPT_CHARACTERS,
        "maximum_selected_records": maximum_selected_records,
        "maximum_output_bytes": maximum_output_bytes,
        "rights": WDC_RESEARCH_DATA_USE_RIGHTS,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return sha256_bytes(encoded)


def _validate_finished_artifact(
    final_root: Path,
    *,
    source_sha256: str,
    category_source_sha256: str,
    policy_sha256: str,
) -> WDCResearchImportResult:
    manifest_path = final_root / "manifest.json"
    output_path = final_root / "records.jsonl"
    if not manifest_path.is_file() or not output_path.is_file():
        raise WDCResearchImportError(f"incomplete sealed WDC artifact: {final_root}")
    manifest = _load_json_object(manifest_path)
    expected = {
        "schema_version": WDC_RESEARCH_MANIFEST_SCHEMA,
        "status": "complete",
        "source_sha256": source_sha256,
        "category_source_sha256": category_source_sha256,
        "policy_sha256": policy_sha256,
        "parser_version": WDC_PARSER_VERSION,
        "record_schema_version": WDC_RESEARCH_RECORD_SCHEMA,
        "selection_policy_version": WDC_SELECTION_POLICY_VERSION,
    }
    for field, value in expected.items():
        if manifest.get(field) != value:
            raise WDCResearchImportError(
                f"sealed WDC artifact has mismatched {field}: {manifest_path}"
            )
    if manifest.get("output_sha256") != sha256_file(output_path):
        raise WDCResearchImportError(f"sealed WDC artifact hash mismatch: {output_path}")
    if manifest.get("output_bytes") != output_path.stat().st_size:
        raise WDCResearchImportError(f"sealed WDC artifact size mismatch: {output_path}")
    return WDCResearchImportResult(
        source_sha256=source_sha256,
        category_source_sha256=category_source_sha256,
        completed_source_lines=int(manifest["completed_source_lines"]),
        selected_records=int(manifest["selected_records"]),
        rejected_non_computer_category=int(manifest["rejected_non_computer_category"]),
        rejected_not_component_like=int(manifest["rejected_not_component_like"]),
        rejected_ambiguous_component=int(manifest["rejected_ambiguous_component"]),
        complete=True,
        reused=True,
        output_path=output_path,
        manifest_path=manifest_path,
        work_path=None,
    )


def _normalise_research_record(
    payload: Mapping[str, Any],
    *,
    source_line_number: int,
    raw_line: bytes,
    snapshot: FetchedSnapshot,
    broad_category: str,
    component_category: str,
) -> dict[str, object]:
    context = f"{snapshot.path}:{source_line_number}"
    source_offer_id = _required_identifier(payload, "id", context=context)
    cluster_id = _required_identifier(payload, "cluster_id", context=context)
    title = _optional_text(payload.get("title"))
    if title is None:
        raise WDCResearchImportError(f"{context}: title is required")
    description = _optional_text(
        payload.get("description"),
        maximum_characters=DESCRIPTION_EXCERPT_CHARACTERS,
    )
    source_url = _optional_text(payload.get("url"))
    return {
        "schema_version": WDC_RESEARCH_RECORD_SCHEMA,
        "record_type": "historical_research_product_offer",
        "quarantine": {
            "research_only": True,
            "production_eligible": False,
            "singapore_market_evidence": False,
            "current_price_evidence": False,
            "stock_evidence": False,
            "canonical_product_evidence": False,
            "entity_resolution_production_claim_eligible": False,
            "model_training_eligible": False,
            "published_metric_claim_eligible": False,
            "reason": (
                "Historical multi-market web research data with no reviewed production rights; "
                "component category is a conservative heuristic candidate, not ground truth."
            ),
        },
        "data_use_rights": dict(WDC_RESEARCH_DATA_USE_RIGHTS),
        "data": {
            "research_offer_id": "wdc-research-" + sha256_bytes(source_offer_id.encode())[:24],
            "source_offer_id": source_offer_id,
            "source_cluster_id": cluster_id,
            "title": title,
            "brand": _optional_text(payload.get("brand")),
            "description_excerpt": description,
            "description_was_truncated": (
                isinstance(payload.get("description"), str)
                and len(str(payload["description"]).strip()) > DESCRIPTION_EXCERPT_CHARACTERS
            ),
            "historical_price_text": _optional_text(payload.get("price")),
            "historical_price_currency": _optional_text(payload.get("priceCurrency")),
            "historical_source_url": source_url,
            "historical_observation_year": 2020,
            "wdc_broad_category": broad_category,
            "component_candidate_category": component_category,
            "component_candidate_rule_version": WDC_SELECTION_POLICY_VERSION,
        },
        "provenance": {
            "source_name": snapshot.source_name,
            "source_dataset_url": snapshot.source_url,
            "source_information_url": WDC_PRODUCTS_PAGE_URL,
            "source_snapshot_sha256": snapshot.content_sha256,
            "source_snapshot_retrieved_at": snapshot.retrieved_at.isoformat(),
            "source_record_line": source_line_number,
            "source_record_sha256": sha256_bytes(raw_line),
            "parser_version": WDC_PARSER_VERSION,
            "licence_or_access_note": snapshot.licence_or_access_note,
        },
    }


def import_wdc_research_candidates(
    snapshot: FetchedSnapshot,
    *,
    category_index_path: str | Path,
    output_root: str | Path,
    record_budget: int | None = None,
    checkpoint_interval: int = DEFAULT_CHECKPOINT_INTERVAL,
    max_line_bytes: int = DEFAULT_MAX_LINE_BYTES,
    max_decompressed_bytes: int = DEFAULT_MAX_CORPUS_DECOMPRESSED_BYTES,
    maximum_selected_records: int = DEFAULT_MAX_SELECTED_RECORDS,
    maximum_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
) -> WDCResearchImportResult:
    """Select quarantined PC-component-like candidates from PDC2020-C.

    The operation is content-addressed and idempotent. A finite
    ``record_budget`` pauses after that many newly examined source rows and can
    be resumed by invoking the same configuration again. No source row is
    represented as a current retailer listing.
    """

    if production_catalog_rights_are_valid(WDC_RESEARCH_DATA_USE_RIGHTS):
        raise AssertionError("WDC research rights must never pass the production catalogue gate")
    if record_budget is not None and record_budget < 1:
        raise ValueError("record_budget must be positive or null")
    if checkpoint_interval < 1:
        raise ValueError("checkpoint_interval must be positive")
    if not 1 <= maximum_selected_records <= HARD_MAX_SELECTED_RECORDS:
        raise ValueError(
            f"maximum_selected_records must be between 1 and {HARD_MAX_SELECTED_RECORDS}"
        )
    if maximum_output_bytes < 1024:
        raise ValueError("maximum_output_bytes must be at least 1024")
    _assert_snapshot_within_retention(snapshot)

    category_path = Path(category_index_path).resolve()
    category_metadata = _category_index_metadata(category_path)
    category_source_sha256 = category_metadata["source_sha256"]
    policy_sha256 = _selection_policy_sha256(
        source_sha256=snapshot.content_sha256,
        category_source_sha256=category_source_sha256,
        maximum_selected_records=maximum_selected_records,
        maximum_output_bytes=maximum_output_bytes,
    )
    run_id = sha256_bytes(
        f"{snapshot.content_sha256}|{category_source_sha256}|{policy_sha256}".encode()
    )
    root = Path(output_root).resolve() / WDC_CORPUS_SOURCE_NAME
    # A full pair of 64-character hashes exceeds the legacy Windows MAX_PATH
    # limit in common pytest/worktree roots. The directory uses a 128-bit
    # content-derived run prefix; the manifest retains and validates all hashes.
    final_root = root / run_id[:32]
    if final_root.exists():
        return _validate_finished_artifact(
            final_root,
            source_sha256=snapshot.content_sha256,
            category_source_sha256=category_source_sha256,
            policy_sha256=policy_sha256,
        )

    work_root = root / ".work" / run_id[:32]
    work_root.mkdir(parents=True, exist_ok=True)
    partial_path = work_root / "records.jsonl.part"
    checkpoint_path = work_root / "checkpoint.json"
    staged_records_path = work_root / "records.jsonl"
    staged_manifest_path = work_root / "manifest.json"

    # Crash recovery for the three-step seal transition:
    # 1. records.part -> records, 2. write manifest, 3. remove checkpoint and
    # rename the whole work directory.  A checkpoint means replay can safely
    # restore ``records`` to ``records.part``.  A valid manifest with no
    # checkpoint means the directory is already sealed and only its final
    # atomic rename remains.
    if (
        not checkpoint_path.exists()
        and staged_records_path.exists()
        and staged_manifest_path.exists()
    ):
        _validate_finished_artifact(
            work_root,
            source_sha256=snapshot.content_sha256,
            category_source_sha256=category_source_sha256,
            policy_sha256=policy_sha256,
        )
        final_root.parent.mkdir(parents=True, exist_ok=True)
        if final_root.exists():
            raise WDCResearchImportError(f"sealed output appeared concurrently: {final_root}")
        os.replace(work_root, final_root)
        return _validate_finished_artifact(
            final_root,
            source_sha256=snapshot.content_sha256,
            category_source_sha256=category_source_sha256,
            policy_sha256=policy_sha256,
        )
    if not checkpoint_path.exists() and (
        partial_path.exists() or staged_records_path.exists() or staged_manifest_path.exists()
    ):
        raise WDCResearchImportError("unrecoverable WDC work directory without checkpoint")
    if checkpoint_path.exists() and staged_records_path.exists():
        if partial_path.exists():
            raise WDCResearchImportError(
                "working WDC artifact contains both partial and staged rows"
            )
        os.replace(staged_records_path, partial_path)
        staged_manifest_path.unlink(missing_ok=True)
    elif checkpoint_path.exists() and staged_manifest_path.exists():
        raise WDCResearchImportError("working WDC manifest exists without staged rows")

    if checkpoint_path.exists():
        checkpoint = _load_json_object(checkpoint_path)
        expected_identity = {
            "schema_version": WDC_RESEARCH_MANIFEST_SCHEMA,
            "state": "working",
            "source_sha256": snapshot.content_sha256,
            "category_source_sha256": category_source_sha256,
            "policy_sha256": policy_sha256,
        }
        for field, value in expected_identity.items():
            if checkpoint.get(field) != value:
                raise WDCResearchImportError(f"working checkpoint has mismatched {field}")
    else:
        checkpoint = _checkpoint_payload(
            source_sha256=snapshot.content_sha256,
            category_source_sha256=category_source_sha256,
            policy_sha256=policy_sha256,
            completed_source_lines=0,
            selected_records=0,
            rejected_non_computer_category=0,
            rejected_not_component_like=0,
            rejected_ambiguous_component=0,
            output_bytes=0,
            output_chain_sha256=_INITIAL_CHAIN.hex(),
        )
        _atomic_json(checkpoint_path, checkpoint)

    completed_line = int(checkpoint["completed_source_lines"])
    selected = int(checkpoint["selected_records"])
    rejected_non_computer = int(checkpoint["rejected_non_computer_category"])
    rejected_not_component = int(checkpoint["rejected_not_component_like"])
    rejected_ambiguous = int(checkpoint["rejected_ambiguous_component"])
    output_bytes = int(checkpoint["output_bytes"])
    if partial_path.exists() and partial_path.stat().st_size > output_bytes:
        with partial_path.open("r+b") as handle:
            handle.truncate(output_bytes)
            handle.flush()
            os.fsync(handle.fileno())
    chain, selected_source_ids = _scan_partial_output(
        partial_path,
        expected_bytes=output_bytes,
    )
    if chain.hex() != checkpoint["output_chain_sha256"]:
        raise WDCResearchImportError("partial output content does not match checkpoint chain")
    if len(selected_source_ids) != selected:
        raise WDCResearchImportError("partial output row count does not match checkpoint")

    reached_eof = True
    new_examined = 0
    last_line = completed_line
    with (
        sqlite3.connect(f"file:{category_path}?mode=ro", uri=True) as categories,
        partial_path.open("ab") as output,
    ):
        for line_number, raw_line, payload in _iter_json_lines(
            snapshot.path,
            max_line_bytes=max_line_bytes,
            max_decompressed_bytes=max_decompressed_bytes,
            start_after_line=completed_line,
        ):
            if record_budget is not None and new_examined >= record_budget:
                reached_eof = False
                break
            context = f"{snapshot.path}:{line_number}"
            source_offer_id = _required_identifier(payload, "id", context=context)
            cluster_id = _required_identifier(payload, "cluster_id", context=context)
            title = _optional_text(payload.get("title"))
            if title is None:
                raise WDCResearchImportError(f"{context}: title is required")
            raw_description = _optional_text(
                payload.get("description"),
                maximum_characters=32 * 1024,
            )
            row = categories.execute(
                "SELECT category FROM category_map WHERE cluster_id = ?",
                (cluster_id,),
            ).fetchone()
            broad_category = str(row[0]) if row is not None else None
            if broad_category != _COMPUTERS_CATEGORY:
                rejected_non_computer += 1
            else:
                component_categories = infer_pc_component_categories(title, raw_description)
                if not component_categories:
                    rejected_not_component += 1
                elif len(component_categories) != 1:
                    rejected_ambiguous += 1
                else:
                    if source_offer_id in selected_source_ids:
                        raise WDCResearchImportError(
                            f"{context}: duplicate selected source offer ID {source_offer_id}"
                        )
                    if selected >= maximum_selected_records:
                        raise WDCResearchLimitError(
                            "selected-record limit reached; no complete artifact was published"
                        )
                    record = _normalise_research_record(
                        payload,
                        source_line_number=line_number,
                        raw_line=raw_line,
                        snapshot=snapshot,
                        broad_category=broad_category,
                        component_category=component_categories[0],
                    )
                    encoded = (
                        json.dumps(
                            record,
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        ).encode("utf-8")
                        + b"\n"
                    )
                    if output_bytes + len(encoded) > maximum_output_bytes:
                        raise WDCResearchLimitError(
                            "output-byte limit reached; no complete artifact was published"
                        )
                    output.write(encoded)
                    chain = _chain_next(chain, encoded)
                    output_bytes += len(encoded)
                    selected += 1
                    selected_source_ids.add(source_offer_id)
            new_examined += 1
            last_line = line_number
            if new_examined % checkpoint_interval == 0:
                output.flush()
                os.fsync(output.fileno())
                checkpoint = _checkpoint_payload(
                    source_sha256=snapshot.content_sha256,
                    category_source_sha256=category_source_sha256,
                    policy_sha256=policy_sha256,
                    completed_source_lines=last_line,
                    selected_records=selected,
                    rejected_non_computer_category=rejected_non_computer,
                    rejected_not_component_like=rejected_not_component,
                    rejected_ambiguous_component=rejected_ambiguous,
                    output_bytes=output_bytes,
                    output_chain_sha256=chain.hex(),
                )
                _atomic_json(checkpoint_path, checkpoint)

        output.flush()
        os.fsync(output.fileno())

    checkpoint = _checkpoint_payload(
        source_sha256=snapshot.content_sha256,
        category_source_sha256=category_source_sha256,
        policy_sha256=policy_sha256,
        completed_source_lines=last_line,
        selected_records=selected,
        rejected_non_computer_category=rejected_non_computer,
        rejected_not_component_like=rejected_not_component,
        rejected_ambiguous_component=rejected_ambiguous,
        output_bytes=output_bytes,
        output_chain_sha256=chain.hex(),
    )
    _atomic_json(checkpoint_path, checkpoint)
    if not reached_eof:
        return WDCResearchImportResult(
            source_sha256=snapshot.content_sha256,
            category_source_sha256=category_source_sha256,
            completed_source_lines=last_line,
            selected_records=selected,
            rejected_non_computer_category=rejected_non_computer,
            rejected_not_component_like=rejected_not_component,
            rejected_ambiguous_component=rejected_ambiguous,
            complete=False,
            reused=False,
            output_path=None,
            manifest_path=None,
            work_path=work_root,
        )

    records_path = work_root / "records.jsonl"
    os.replace(partial_path, records_path)
    manifest = {
        "schema_version": WDC_RESEARCH_MANIFEST_SCHEMA,
        "status": "complete",
        "source_sha256": snapshot.content_sha256,
        "category_source_sha256": category_source_sha256,
        "policy_sha256": policy_sha256,
        "source_snapshot": snapshot.metadata(),
        "parser_version": WDC_PARSER_VERSION,
        "record_schema_version": WDC_RESEARCH_RECORD_SCHEMA,
        "selection_policy_version": WDC_SELECTION_POLICY_VERSION,
        "completed_source_lines": last_line,
        "selected_records": selected,
        "rejected_non_computer_category": rejected_non_computer,
        "rejected_not_component_like": rejected_not_component,
        "rejected_ambiguous_component": rejected_ambiguous,
        "output_file": records_path.name,
        "output_bytes": records_path.stat().st_size,
        "output_sha256": sha256_file(records_path),
        "data_use_rights": dict(WDC_RESEARCH_DATA_USE_RIGHTS),
        "retention_deadline": _retention_deadline(snapshot).isoformat(),
        "quarantine": {
            "research_only": True,
            "production_eligible": False,
            "singapore_market_evidence": False,
            "current_price_or_stock_evidence": False,
            "model_training_eligible": False,
            "published_metric_claim_eligible": False,
        },
        "limits": {
            "max_line_bytes": max_line_bytes,
            "max_decompressed_bytes": max_decompressed_bytes,
            "maximum_selected_records": maximum_selected_records,
            "maximum_output_bytes": maximum_output_bytes,
            "in_memory_selected_id_ceiling": HARD_MAX_SELECTED_RECORDS,
        },
        "evidence_scope": {
            "can_support": [
                "bounded ingestion engineering",
                "historical schema and provenance tests",
                "research-only PC-component candidate discovery",
            ],
            "cannot_support": [
                "current Singapore retailer listings, prices, shipping, or stock",
                "production canonical-product counts or entity-resolution accuracy",
                "training or published model metrics without a separate rights review",
                "manufacturer-authoritative compatibility specifications",
            ],
        },
    }
    manifest_path = work_root / "manifest.json"
    _atomic_json(manifest_path, manifest)
    checkpoint_path.unlink()
    final_root.parent.mkdir(parents=True, exist_ok=True)
    if final_root.exists():
        raise WDCResearchImportError(f"sealed output appeared concurrently: {final_root}")
    os.replace(work_root, final_root)
    return WDCResearchImportResult(
        source_sha256=snapshot.content_sha256,
        category_source_sha256=category_source_sha256,
        completed_source_lines=last_line,
        selected_records=selected,
        rejected_non_computer_category=rejected_non_computer,
        rejected_not_component_like=rejected_not_component,
        rejected_ambiguous_component=rejected_ambiguous,
        complete=True,
        reused=False,
        output_path=final_root / "records.jsonl",
        manifest_path=final_root / "manifest.json",
        work_path=None,
    )


__all__ = [
    "DEFAULT_CHECKPOINT_INTERVAL",
    "DEFAULT_MAX_CATEGORY_DECOMPRESSED_BYTES",
    "DEFAULT_MAX_CORPUS_DECOMPRESSED_BYTES",
    "DEFAULT_MAX_LINE_BYTES",
    "DEFAULT_MAX_OUTPUT_BYTES",
    "DEFAULT_MAX_SELECTED_RECORDS",
    "HARD_MAX_SELECTED_RECORDS",
    "MAX_CATEGORY_DOWNLOAD_BYTES",
    "MAX_CORPUS_DOWNLOAD_BYTES",
    "WDC_PRODUCTS_CATEGORIES_URL",
    "WDC_PRODUCTS_CORPUS_URL",
    "WDC_PRODUCTS_PAGE_URL",
    "WDC_RESEARCH_DATA_USE_RIGHTS",
    "WDC_RESEARCH_RETENTION_DAYS",
    "WDCCategoryIndexResult",
    "WDCProductsResearchSource",
    "WDCResearchImportError",
    "WDCResearchImportResult",
    "WDCResearchLimitError",
    "build_wdc_category_index",
    "import_wdc_research_candidates",
    "infer_pc_component_categories",
]
