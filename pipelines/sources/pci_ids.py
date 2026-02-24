"""Bounded PCI ID Repository aliases for entity-resolution candidate blocking.

The upstream ``pci.ids`` snapshot is a daily, human-readable mapping from PCI
identifiers to labels.  This adapter deliberately does not create canonical
products, prices, stock observations, benchmarks, or compatibility facts.  Its
records may only enrich deterministic entity-resolution blocking features.
"""

from __future__ import annotations

import gzip
import heapq
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

from pipelines.parsing.normalizers import NORMALISED_RECORD_SCHEMA_VERSION
from pipelines.sources.base import (
    ParseResult,
    RawSnapshot,
    SnapshotError,
    fetch_http_snapshot,
    rejected_record,
    sha256_bytes,
    sha256_file,
    snapshot_local_file,
)

PCI_IDS_SOURCE_NAME = "pci_id_repository"
PCI_IDS_HOME_URL = "https://pci-ids.ucw.cz/"
PCI_IDS_PLAIN_URL = "https://pci-ids.ucw.cz/v2.2/pci.ids"
PCI_IDS_GZIP_URL = "https://pci-ids.ucw.cz/v2.2/pci.ids.gz"
PCI_IDS_PARSER_VERSION = "pci-id-repository-v2"
PCI_IDS_LICENSE = "BSD-3-Clause"
PCI_IDS_COPYRIGHT_HOLDERS = "Martin Mares and Albert Pool"
PCI_IDS_LICENSE_NOTE = (
    "PCI ID Repository database and generated files, copyright Martin Mares and "
    "Albert Pool, used under the BSD 3-Clause "
    "licence option. Retain the licence notice and attribution. Records are auxiliary "
    "PCI identifier labels for entity-resolution blocking only; they are not product, "
    "price, stock, benchmark, or compatibility evidence."
)
PCI_IDS_USER_AGENT = (
    "BuildSignal-PC-Build-Recommender/0.1 "
    "(PCI-ID alias cache; https://buildsignal-pc-recommender.tendra425.chatgpt.site)"
)

DEFAULT_RECORD_LIMIT = 20_000
DEFAULT_MAXIMUM_UNCOMPRESSED_BYTES = 32 * 1024 * 1024
DEFAULT_MAXIMUM_LINE_BYTES = 16 * 1024
DEFAULT_MAXIMUM_LINES_SCANNED = 500_000
DEFAULT_MAXIMUM_RECORDED_REJECTIONS = 10_000
PRIORITY_PC_VENDOR_IDS = ("1002", "10de", "10ec", "8086")
_GZIP_MAGIC = b"\x1f\x8b"

_VENDOR = re.compile(rb"^([0-9A-Fa-f]{4}) {2,}(.+?)\s*$")
_DEVICE = re.compile(rb"^\t([0-9A-Fa-f]{4}) {2,}(.+?)\s*$")
_SUBSYSTEM = re.compile(rb"^\t\t([0-9A-Fa-f]{4}) ([0-9A-Fa-f]{4}) {2,}(.+?)\s*$")
_CLASS = re.compile(rb"^C [0-9A-Fa-f]{2} {2,}.+?\s*$")
_SHA256 = re.compile(r"^[0-9A-Fa-f]{64}$")

_SELECTION_STRATA = tuple(
    f"priority:{vendor_id}:{identifier_type}"
    for vendor_id in PRIORITY_PC_VENDOR_IDS
    for identifier_type in ("device", "subsystem")
) + ("general:vendor", "general:device", "general:subsystem")


class PCIIdsParseLimitError(SnapshotError):
    """Raised when decompressed input exceeds a parser safety budget."""


class _BinaryLineReader(Protocol):
    def read(self, size: int = -1, /) -> bytes: ...

    def readline(self, size: int = -1, /) -> bytes: ...


def _iter_bounded_lines(
    stream: _BinaryLineReader,
    *,
    maximum_line_bytes: int,
    maximum_uncompressed_bytes: int,
    maximum_lines_scanned: int,
) -> Iterator[tuple[int, bytes, int, int, bool]]:
    """Read lines in bounded fragments and drain oversized lines without retaining them."""

    line_number = 0
    uncompressed_bytes = 0
    fragment_limit = maximum_line_bytes + 1
    while True:
        if line_number >= maximum_lines_scanned:
            if stream.read(1):
                raise PCIIdsParseLimitError(
                    f"PCI IDs lines scanned exceeded {maximum_lines_scanned}"
                )
            return
        prefix = stream.readline(fragment_limit)
        if not prefix:
            return
        line_number += 1
        line_byte_count = len(prefix)
        uncompressed_bytes += len(prefix)
        if uncompressed_bytes > maximum_uncompressed_bytes:
            raise PCIIdsParseLimitError(
                "PCI IDs decompressed bytes exceeded "
                f"{maximum_uncompressed_bytes} at line {line_number}"
            )
        line_too_long = len(prefix) > maximum_line_bytes
        line_complete = prefix.endswith(b"\n")
        while not line_complete:
            fragment = stream.readline(fragment_limit)
            if not fragment:
                line_complete = True
                break
            line_byte_count += len(fragment)
            uncompressed_bytes += len(fragment)
            if uncompressed_bytes > maximum_uncompressed_bytes:
                raise PCIIdsParseLimitError(
                    "PCI IDs decompressed bytes exceeded "
                    f"{maximum_uncompressed_bytes} at line {line_number}"
                )
            line_too_long = line_too_long or line_byte_count > maximum_line_bytes
            line_complete = fragment.endswith(b"\n")
            if not line_too_long:
                prefix += fragment
        yield (
            line_number,
            prefix,
            line_byte_count,
            uncompressed_bytes,
            line_too_long,
        )


def _iter_binary_lines(
    path: Path,
    *,
    maximum_line_bytes: int,
    maximum_uncompressed_bytes: int,
    maximum_lines_scanned: int,
) -> Iterator[tuple[int, bytes, int, int, bool]]:
    """Yield plain or gzip lines with decompression budgets enforced while reading."""

    try:
        with path.open("rb") as raw:
            magic = raw.read(2)
            raw.seek(0)
            if magic == _GZIP_MAGIC:
                with gzip.GzipFile(fileobj=raw, mode="rb") as stream:
                    yield from _iter_bounded_lines(
                        stream,
                        maximum_line_bytes=maximum_line_bytes,
                        maximum_uncompressed_bytes=maximum_uncompressed_bytes,
                        maximum_lines_scanned=maximum_lines_scanned,
                    )
            else:
                yield from _iter_bounded_lines(
                    raw,
                    maximum_line_bytes=maximum_line_bytes,
                    maximum_uncompressed_bytes=maximum_uncompressed_bytes,
                    maximum_lines_scanned=maximum_lines_scanned,
                )
    except (EOFError, gzip.BadGzipFile, OSError) as exc:
        raise SnapshotError(f"invalid PCI IDs snapshot {path}: {exc}") from exc


def _identifier_record(
    *,
    snapshot: RawSnapshot,
    line_number: int,
    raw_line: bytes,
    identifier_type: Literal["vendor", "device", "subsystem"],
    canonical_label: str,
    vendor_id: str,
    device_id: str | None = None,
    subsystem_vendor_id: str | None = None,
    subsystem_device_id: str | None = None,
) -> dict[str, Any]:
    identifiers = {"vendor_id": vendor_id}
    identifier_parts = ["pci", identifier_type, vendor_id]
    if device_id is not None:
        identifiers["device_id"] = device_id
        identifier_parts.append(device_id)
    if subsystem_vendor_id is not None and subsystem_device_id is not None:
        identifiers["subsystem_vendor_id"] = subsystem_vendor_id
        identifiers["subsystem_device_id"] = subsystem_device_id
        identifier_parts.extend((subsystem_vendor_id, subsystem_device_id))
    source_record_id = ":".join(identifier_parts)
    return {
        "schema_version": NORMALISED_RECORD_SCHEMA_VERSION,
        "record_type": "hardware_identifier_alias",
        "source_record_id": source_record_id,
        "archive_snapshot_sha256": snapshot.content_sha256,
        "raw_record_sha256": sha256_bytes(raw_line.rstrip(b"\r\n")),
        # Licence eligibility is intentionally narrower than the technical licence grant.
        # These facts feed deterministic blocking only and are not training examples or
        # evidence for published model-quality claims.
        "training_eligible": False,
        "published_claims_eligible": False,
        "rights_metadata": {
            "rights_basis": "open_licence",
            "licence": PCI_IDS_LICENSE,
            "copyright_holders": PCI_IDS_COPYRIGHT_HOLDERS,
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
        },
        "provenance": {
            "source_name": snapshot.source_name,
            "source_url": snapshot.source_url,
            "source_type": "identity_aliases",
            "retrieved_at": snapshot.retrieved_at.isoformat(),
            "parser_version": PCI_IDS_PARSER_VERSION,
            "licence": PCI_IDS_LICENSE,
            "copyright_holders": PCI_IDS_COPYRIGHT_HOLDERS,
            "licence_or_access_note": snapshot.licence_or_access_note,
            "extraction_confidence": 1.0,
            "source_line": line_number,
        },
        "normalisation_metadata": {
            "entity_resolution_role": "candidate_blocking_enrichment",
            "compatibility_authoritative": False,
            "contains_price_or_stock": False,
            "contains_product_identity": False,
        },
        "data": {
            "namespace": "pci",
            "identifier_type": identifier_type,
            "identifiers": identifiers,
            "canonical_label": canonical_label,
            "aliases": [canonical_label],
            "use_scope": "entity_resolution_blocking_only",
            "authoritative_for": ["pci_identifier_to_label"],
            "not_authoritative_for": [
                "canonical_product_identity",
                "compatibility",
                "price",
                "stock",
                "performance",
            ],
        },
    }


@dataclass(frozen=True)
class _SelectionCandidate:
    """One parsed alias retained by a bounded deterministic selector."""

    score: int
    line_number: int
    source_record_id: str
    record: dict[str, Any]


class _DeterministicReservoir:
    """Keep the lowest stable-hash candidates using at most ``capacity`` slots."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self._heap: list[tuple[int, int, str, _SelectionCandidate]] = []

    def offer(
        self,
        candidate: _SelectionCandidate,
    ) -> tuple[bool, _SelectionCandidate | None]:
        if self.capacity == 0:
            return False, None
        entry = (
            -candidate.score,
            -candidate.line_number,
            candidate.source_record_id,
            candidate,
        )
        if len(self._heap) < self.capacity:
            heapq.heappush(self._heap, entry)
            return True, None

        worst = self._heap[0]
        worst_rank = (-worst[0], -worst[1], worst[2])
        candidate_rank = (
            candidate.score,
            candidate.line_number,
            candidate.source_record_id,
        )
        if candidate_rank >= worst_rank:
            return False, None
        evicted = heapq.heapreplace(self._heap, entry)[3]
        return True, evicted

    def candidates(self) -> list[_SelectionCandidate]:
        return sorted(
            (entry[3] for entry in self._heap),
            key=lambda candidate: (
                candidate.score,
                candidate.line_number,
                candidate.source_record_id,
            ),
        )


class _BoundedStratifiedSelector:
    """Select representative aliases while guaranteeing major-vendor anchors.

    The full input is scanned under the parser byte/line budgets. Four vendor
    anchors are retained separately, the remaining primary budget is shared
    deterministically across identifier/vendor strata, and one global fallback
    reservoir fills capacity left by absent or sparse strata. The number of
    retained references is bounded by roughly twice ``max_records``.
    """

    def __init__(self, max_records: int) -> None:
        self.max_records = max_records
        stratified_budget = max(0, max_records - len(PRIORITY_PC_VENDOR_IDS))
        base_capacity, extra_capacity = divmod(
            stratified_budget,
            len(_SELECTION_STRATA),
        )
        self.stratum_capacities = {
            stratum: base_capacity + (index < extra_capacity)
            for index, stratum in enumerate(_SELECTION_STRATA)
        }
        self._strata = {
            stratum: _DeterministicReservoir(capacity)
            for stratum, capacity in self.stratum_capacities.items()
        }
        self._fallback = _DeterministicReservoir(max_records)
        self._anchors: dict[str, _SelectionCandidate] = {}
        self._retained_references: dict[str, int] = {}
        self.candidates_considered = 0

    @property
    def retained_candidate_capacity(self) -> int:
        return (
            len(PRIORITY_PC_VENDOR_IDS) + sum(self.stratum_capacities.values()) + self.max_records
        )

    def _retain(self, candidate: _SelectionCandidate) -> None:
        source_record_id = candidate.source_record_id
        self._retained_references[source_record_id] = (
            self._retained_references.get(source_record_id, 0) + 1
        )

    def _release(self, candidate: _SelectionCandidate | None) -> None:
        if candidate is None:
            return
        source_record_id = candidate.source_record_id
        references = self._retained_references[source_record_id] - 1
        if references == 0:
            del self._retained_references[source_record_id]
        else:
            self._retained_references[source_record_id] = references

    @staticmethod
    def _stratum(record: dict[str, Any]) -> str:
        data = record["data"]
        identifier_type = str(data["identifier_type"])
        vendor_id = str(data["identifiers"]["vendor_id"])
        if vendor_id in PRIORITY_PC_VENDOR_IDS and identifier_type != "vendor":
            return f"priority:{vendor_id}:{identifier_type}"
        return f"general:{identifier_type}"

    def offer(self, record: dict[str, Any], *, line_number: int) -> bool:
        """Offer a record, returning ``False`` only for a retained duplicate."""

        self.candidates_considered += 1
        source_record_id = str(record["source_record_id"])
        if source_record_id in self._retained_references:
            return False
        candidate = _SelectionCandidate(
            score=int(sha256_bytes(source_record_id.encode("ascii")), 16),
            line_number=line_number,
            source_record_id=source_record_id,
            record=record,
        )
        data = record["data"]
        identifier_type = str(data["identifier_type"])
        vendor_id = str(data["identifiers"]["vendor_id"])
        if identifier_type == "vendor" and vendor_id in PRIORITY_PC_VENDOR_IDS:
            self._anchors[vendor_id] = candidate
            self._retain(candidate)
            return True

        stratum = self._stratum(record)
        retained, evicted = self._strata[stratum].offer(candidate)
        if retained:
            self._retain(candidate)
            self._release(evicted)
        retained, evicted = self._fallback.offer(candidate)
        if retained:
            self._retain(candidate)
            self._release(evicted)
        return True

    def selected(self) -> list[_SelectionCandidate]:
        selected: dict[str, _SelectionCandidate] = {}
        for vendor_id in PRIORITY_PC_VENDOR_IDS:
            candidate = self._anchors.get(vendor_id)
            if candidate is not None and len(selected) < self.max_records:
                selected[candidate.source_record_id] = candidate

        for stratum in _SELECTION_STRATA:
            for candidate in self._strata[stratum].candidates():
                if len(selected) >= self.max_records:
                    break
                selected.setdefault(candidate.source_record_id, candidate)

        for candidate in self._fallback.candidates():
            if len(selected) >= self.max_records:
                break
            selected.setdefault(candidate.source_record_id, candidate)

        return sorted(
            selected.values(),
            key=lambda candidate: (
                candidate.line_number,
                candidate.source_record_id,
            ),
        )

    def selected_stratum_counts(
        self,
        selected: list[_SelectionCandidate],
    ) -> dict[str, int]:
        counts = {stratum: 0 for stratum in _SELECTION_STRATA}
        counts["priority_vendor_anchor"] = 0
        for candidate in selected:
            data = candidate.record["data"]
            identifier_type = str(data["identifier_type"])
            vendor_id = str(data["identifiers"]["vendor_id"])
            if identifier_type == "vendor" and vendor_id in PRIORITY_PC_VENDOR_IDS:
                counts["priority_vendor_anchor"] += 1
            else:
                counts[self._stratum(candidate.record)] += 1
        return counts


class PCIIDRepositoryAdapter:
    """Fetch and stream a content-addressed PCI ID Repository snapshot."""

    def __init__(self, *, raw_root: str | Path) -> None:
        self.raw_root = Path(raw_root)

    def fetch(
        self,
        *,
        snapshot_path: str | Path | None = None,
        snapshot_format: Literal["gzip", "plain"] = "gzip",
        expected_sha256: str | None = None,
    ) -> RawSnapshot:
        if snapshot_format not in {"gzip", "plain"}:
            raise ValueError("snapshot_format must be 'gzip' or 'plain'")
        if expected_sha256 is not None and _SHA256.fullmatch(expected_sha256) is None:
            raise ValueError("expected_sha256 must be a 64-character hexadecimal digest")
        expected_sha256 = expected_sha256.lower() if expected_sha256 is not None else None
        source_url = PCI_IDS_GZIP_URL if snapshot_format == "gzip" else PCI_IDS_PLAIN_URL
        suffix = ".ids.gz" if snapshot_format == "gzip" else ".ids"
        media_type = "application/gzip" if snapshot_format == "gzip" else "text/plain"
        if snapshot_path is None:
            return fetch_http_snapshot(
                source_name=PCI_IDS_SOURCE_NAME,
                source_url=source_url,
                source_type="open_identity_alias_import",
                raw_root=self.raw_root,
                parser_version=PCI_IDS_PARSER_VERSION,
                licence_or_access_note=PCI_IDS_LICENSE_NOTE,
                suffix=suffix,
                expected_sha256=expected_sha256,
                maximum_bytes=(8 * 1024 * 1024 if snapshot_format == "gzip" else 32 * 1024 * 1024),
                headers={
                    "User-Agent": PCI_IDS_USER_AGENT,
                    "Accept-Encoding": "gzip",
                },
            )
        local_path = Path(snapshot_path)
        if expected_sha256 is not None:
            local_sha256 = sha256_file(local_path)
            if local_sha256 != expected_sha256:
                raise SnapshotError(
                    f"PCI IDs SHA-256 mismatch: expected {expected_sha256}, received {local_sha256}"
                )
        snapshot = snapshot_local_file(
            source_name=PCI_IDS_SOURCE_NAME,
            source_url=source_url,
            source_type="open_identity_alias_import",
            source_path=local_path,
            raw_root=self.raw_root,
            parser_version=PCI_IDS_PARSER_VERSION,
            licence_or_access_note=PCI_IDS_LICENSE_NOTE,
            suffix=suffix,
            media_type=media_type,
        )
        if expected_sha256 is not None and snapshot.content_sha256 != expected_sha256:
            raise SnapshotError(
                "PCI IDs SHA-256 mismatch: "
                f"expected {expected_sha256}, received {snapshot.content_sha256}"
            )
        return snapshot

    def parse(
        self,
        snapshot: RawSnapshot,
        *,
        max_records: int = DEFAULT_RECORD_LIMIT,
        maximum_uncompressed_bytes: int = DEFAULT_MAXIMUM_UNCOMPRESSED_BYTES,
        maximum_line_bytes: int = DEFAULT_MAXIMUM_LINE_BYTES,
        maximum_lines_scanned: int = DEFAULT_MAXIMUM_LINES_SCANNED,
        maximum_recorded_rejections: int = DEFAULT_MAXIMUM_RECORDED_REJECTIONS,
    ) -> ParseResult:
        if snapshot.source_name != PCI_IDS_SOURCE_NAME:
            raise ValueError(f"unexpected PCI IDs source: {snapshot.source_name}")
        if max_records <= 0:
            raise ValueError("max_records must be positive")
        if maximum_uncompressed_bytes <= 0:
            raise ValueError("maximum_uncompressed_bytes must be positive")
        if maximum_line_bytes <= 0:
            raise ValueError("maximum_line_bytes must be positive")
        if maximum_lines_scanned <= 0:
            raise ValueError("maximum_lines_scanned must be positive")
        if maximum_recorded_rejections <= 0:
            raise ValueError("maximum_recorded_rejections must be positive")

        batch = ParseResult(
            source_name=snapshot.source_name,
            snapshot_sha256=snapshot.content_sha256,
        )
        current_vendor: str | None = None
        current_device: str | None = None
        in_class_section = False
        uncompressed_bytes = 0
        lines_scanned = 0
        class_lines_ignored = 0
        selector = _BoundedStratifiedSelector(max_records)
        total_rejections = 0

        def reject(record_id: str, reason: str, **details: object) -> None:
            nonlocal total_rejections
            total_rejections += 1
            if len(batch.rejected) < maximum_recorded_rejections:
                batch.rejected.append(rejected_record(record_id, reason, **details))

        for (
            line_number,
            raw_line,
            line_byte_count,
            current_uncompressed_bytes,
            line_too_long,
        ) in _iter_binary_lines(
            snapshot.path,
            maximum_line_bytes=maximum_line_bytes,
            maximum_uncompressed_bytes=maximum_uncompressed_bytes,
            maximum_lines_scanned=maximum_lines_scanned,
        ):
            lines_scanned = line_number
            uncompressed_bytes = current_uncompressed_bytes
            if line_too_long:
                reject(
                    f"line:{line_number}",
                    "line_too_long",
                    byte_count=line_byte_count,
                    maximum_line_bytes=maximum_line_bytes,
                )
                indentation = len(raw_line) - len(raw_line.lstrip(b"\t"))
                if indentation == 0:
                    current_vendor = None
                    current_device = None
                elif indentation == 1:
                    current_device = None
                continue
            stripped = raw_line.strip()
            if not stripped or stripped.startswith(b"#"):
                continue
            if _CLASS.fullmatch(raw_line.rstrip(b"\r\n")):
                in_class_section = True
                current_vendor = None
                current_device = None
                class_lines_ignored += 1
                continue
            if in_class_section:
                class_lines_ignored += 1
                continue

            line = raw_line.rstrip(b"\r\n")
            match = _SUBSYSTEM.fullmatch(line)
            if match is not None:
                if current_vendor is None or current_device is None:
                    reject(f"line:{line_number}", "subsystem_without_device")
                    continue
                try:
                    label = match.group(3).decode("utf-8").strip()
                except UnicodeDecodeError as exc:
                    reject(
                        f"line:{line_number}",
                        "invalid_utf8",
                        byte_offset=exc.start,
                    )
                    continue
                if not label:
                    reject(f"line:{line_number}", "empty_label")
                    continue
                subsystem_vendor = match.group(1).decode("ascii").lower()
                subsystem_device = match.group(2).decode("ascii").lower()
                record = _identifier_record(
                    snapshot=snapshot,
                    line_number=line_number,
                    raw_line=raw_line,
                    identifier_type="subsystem",
                    canonical_label=label,
                    vendor_id=current_vendor,
                    device_id=current_device,
                    subsystem_vendor_id=subsystem_vendor,
                    subsystem_device_id=subsystem_device,
                )
            else:
                match = _DEVICE.fullmatch(line)
                if match is not None:
                    if current_vendor is None:
                        reject(f"line:{line_number}", "device_without_vendor")
                        current_device = None
                        continue
                    try:
                        label = match.group(2).decode("utf-8").strip()
                    except UnicodeDecodeError as exc:
                        reject(
                            f"line:{line_number}",
                            "invalid_utf8",
                            byte_offset=exc.start,
                        )
                        current_device = None
                        continue
                    if not label:
                        reject(f"line:{line_number}", "empty_label")
                        current_device = None
                        continue
                    current_device = match.group(1).decode("ascii").lower()
                    record = _identifier_record(
                        snapshot=snapshot,
                        line_number=line_number,
                        raw_line=raw_line,
                        identifier_type="device",
                        canonical_label=label,
                        vendor_id=current_vendor,
                        device_id=current_device,
                    )
                else:
                    match = _VENDOR.fullmatch(line)
                    if match is not None:
                        try:
                            label = match.group(2).decode("utf-8").strip()
                        except UnicodeDecodeError as exc:
                            reject(
                                f"line:{line_number}",
                                "invalid_utf8",
                                byte_offset=exc.start,
                            )
                            current_vendor = None
                            current_device = None
                            continue
                        if not label:
                            reject(f"line:{line_number}", "empty_label")
                            current_vendor = None
                            current_device = None
                            continue
                        current_vendor = match.group(1).decode("ascii").lower()
                        current_device = None
                        record = _identifier_record(
                            snapshot=snapshot,
                            line_number=line_number,
                            raw_line=raw_line,
                            identifier_type="vendor",
                            canonical_label=label,
                            vendor_id=current_vendor,
                        )
                    else:
                        indentation = len(line) - len(line.lstrip(b"\t"))
                        reason = {
                            0: "malformed_vendor_line",
                            1: "malformed_device_line",
                            2: "malformed_subsystem_line",
                        }.get(indentation, "unsupported_indentation")
                        reject(
                            f"line:{line_number}",
                            reason,
                            indentation=indentation,
                        )
                        if indentation == 0:
                            current_vendor = None
                            current_device = None
                        elif indentation == 1:
                            current_device = None
                        continue

            if not selector.offer(record, line_number=line_number):
                reject(
                    f"line:{line_number}",
                    "duplicate_identifier",
                    source_record_id=str(record["source_record_id"]),
                )
                continue

        selected = selector.selected()
        batch.records = [candidate.record for candidate in selected]
        counts = {"vendor": 0, "device": 0, "subsystem": 0}
        for record in batch.records:
            counts[str(record["data"]["identifier_type"])] += 1
        selected_anchor_ids = [
            vendor_id
            for vendor_id in PRIORITY_PC_VENDOR_IDS
            if any(
                candidate.record["data"]["identifier_type"] == "vendor"
                and candidate.record["data"]["identifiers"]["vendor_id"] == vendor_id
                for candidate in selected
            )
        ]
        record_limit_reached = selector.candidates_considered > max_records

        batch.statistics = {
            "licence": PCI_IDS_LICENSE,
            "use_scope": "entity_resolution_blocking_only",
            "record_limit": max_records,
            "record_limit_reached": record_limit_reached,
            "lines_scanned": lines_scanned,
            "maximum_lines_scanned": maximum_lines_scanned,
            "uncompressed_bytes_scanned": uncompressed_bytes,
            "class_lines_ignored": class_lines_ignored,
            "accepted_by_identifier_type": counts,
            "total_rejections": total_rejections,
            "recorded_rejections": batch.rejected_count,
            "recorded_rejections_truncated": total_rejections - batch.rejected_count,
            "streaming_parser": True,
            "full_snapshot_scanned": True,
            "selection_strategy": ("priority_vendor_anchors_plus_deterministic_stratified_hash_v1"),
            "priority_vendor_ids": list(PRIORITY_PC_VENDOR_IDS),
            "priority_vendor_anchors_selected": selected_anchor_ids,
            "selection_stratum_capacities": selector.stratum_capacities,
            "selection_stratum_counts": selector.selected_stratum_counts(selected),
            "candidates_considered": selector.candidates_considered,
            "retained_candidate_capacity": selector.retained_candidate_capacity,
        }
        return batch


__all__ = [
    "DEFAULT_RECORD_LIMIT",
    "PCI_IDS_COPYRIGHT_HOLDERS",
    "PCI_IDS_GZIP_URL",
    "PCI_IDS_HOME_URL",
    "PCI_IDS_LICENSE",
    "PCI_IDS_PARSER_VERSION",
    "PCI_IDS_PLAIN_URL",
    "PCI_IDS_SOURCE_NAME",
    "PCI_IDS_USER_AGENT",
    "PCIIdsParseLimitError",
    "PCIIDRepositoryAdapter",
    "PRIORITY_PC_VENDOR_IDS",
]
