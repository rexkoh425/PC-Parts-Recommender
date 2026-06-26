"""Pinned BuildCores OpenDB catalogue adapter."""

from __future__ import annotations

import json
import os
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath

from pc_build_recommender.catalog.canonical_identity import audit_canonical_envelopes
from pipelines.parsing.normalizers import BUILDCORES_CATEGORY_MAP, normalise_buildcores_product
from pipelines.sources.base import (
    ParsedBatch,
    RawSnapshot,
    fetch_http_snapshot,
    rejected_record,
    sha256_file,
    snapshot_local_file,
)

BUILDCORES_COMMIT = "6a64ab14fb1ab1bc1f3030d36b70bddcc2afeb0f"
BUILDCORES_ARCHIVE_URL = (
    "https://github.com/buildcores/buildcores-open-db/archive/"
    f"{BUILDCORES_COMMIT}.zip"
)
BUILDCORES_ARCHIVE_SHA256 = "f3ee75dd07ffdd7725da7b056229e0df12838c571b2372bd59563f3a79fd383f"
BUILDCORES_LICENSE_NOTE = (
    "BuildCores OpenDB database licensed ODC-By 1.0; attribution required. "
    "Community-maintained specifications require field-level verification for hard compatibility."
)
BUILDCORES_PARSER_VERSION = "buildcores-open-db-v1"
DEFAULT_CATEGORIES = tuple(BUILDCORES_CATEGORY_MAP)
MAXIMUM_ARCHIVE_BYTES = 150 * 1024 * 1024
MAXIMUM_ARCHIVE_MEMBERS = 100_000
MAXIMUM_UNCOMPRESSED_BYTES = 512 * 1024 * 1024
MAXIMUM_MEMBER_BYTES = 16 * 1024 * 1024
MAXIMUM_COMPRESSION_RATIO = 100
_ALLOWED_COMPRESSION = frozenset({zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED})


class BuildCoresOpenDBAdapter:
    """Fetch and normalise a deterministic subset of the pinned OpenDB commit."""

    def __init__(self, *, raw_root: str | Path) -> None:
        self.raw_root = Path(raw_root)

    def fetch(self, *, archive_path: str | Path | None = None) -> RawSnapshot:
        if archive_path is not None:
            return snapshot_local_file(
                source_name="buildcores_open_db",
                source_url=BUILDCORES_ARCHIVE_URL,
                source_type="import",
                source_path=archive_path,
                raw_root=self.raw_root,
                parser_version=BUILDCORES_PARSER_VERSION,
                licence_or_access_note=BUILDCORES_LICENSE_NOTE,
                suffix=".zip",
                media_type="application/zip",
                expected_sha256=BUILDCORES_ARCHIVE_SHA256,
                maximum_bytes=MAXIMUM_ARCHIVE_BYTES,
            )
        return fetch_http_snapshot(
            source_name="buildcores_open_db",
            source_url=BUILDCORES_ARCHIVE_URL,
            source_type="import",
            raw_root=self.raw_root,
            parser_version=BUILDCORES_PARSER_VERSION,
            licence_or_access_note=BUILDCORES_LICENSE_NOTE,
            suffix=".zip",
            expected_sha256=BUILDCORES_ARCHIVE_SHA256,
            maximum_bytes=MAXIMUM_ARCHIVE_BYTES,
        )

    def parse(
        self,
        snapshot: RawSnapshot,
        *,
        categories: Sequence[str] = DEFAULT_CATEGORIES,
        per_category_limit: int | None = 100,
        per_category_limits: Mapping[str, int] | None = None,
    ) -> ParsedBatch:
        self._validate_snapshot(snapshot)
        if per_category_limit is not None and per_category_limit <= 0:
            raise ValueError("per_category_limit must be positive or None")
        unknown_categories = set(categories) - set(BUILDCORES_CATEGORY_MAP)
        if unknown_categories:
            raise ValueError(f"unsupported BuildCores categories: {sorted(unknown_categories)}")
        limits = dict(per_category_limits or {})
        for category, limit in limits.items():
            if category not in BUILDCORES_CATEGORY_MAP or limit <= 0:
                raise ValueError(f"invalid category limit: {category}={limit}")

        batch = ParsedBatch(
            source_name=snapshot.source_name,
            snapshot_sha256=snapshot.content_sha256,
        )
        accepted_by_category = {category: 0 for category in categories}
        available_by_category = {category: 0 for category in categories}
        with zipfile.ZipFile(snapshot.path) as archive:
            member_count, uncompressed_bytes = self._validate_archive(archive)
            entries = self._category_entries(archive, categories)
            for category in categories:
                category_entries = entries[category]
                available_by_category[category] = len(category_entries)
                selected_limit = limits.get(category, per_category_limit)
                selected_entries = (
                    category_entries
                    if selected_limit is None
                    else category_entries[:selected_limit]
                )
                for entry in selected_entries:
                    source_record_path = self._relative_record_path(entry.filename)
                    try:
                        raw_record = archive.read(entry)
                        payload = json.loads(raw_record)
                        if not isinstance(payload, dict):
                            raise TypeError("product JSON root must be an object")
                        normalised = normalise_buildcores_product(
                            record=payload,
                            source_category=category,
                            source_record_path=source_record_path,
                            raw_record_bytes=raw_record,
                            snapshot=snapshot,
                            commit=BUILDCORES_COMMIT,
                        )
                    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                        batch.rejected.append(
                            rejected_record(
                                source_record_path,
                                "invalid_buildcores_product",
                                category=category,
                                error=f"{type(exc).__name__}: {exc}",
                            )
                        )
                        continue
                    batch.records.append(normalised)
                    accepted_by_category[category] += 1

        identity_preflight = audit_canonical_envelopes(batch.records)
        batch.statistics = {
            "commit": BUILDCORES_COMMIT,
            "archive_sha256": BUILDCORES_ARCHIVE_SHA256,
            "archive_member_count": member_count,
            "archive_uncompressed_bytes": uncompressed_bytes,
            "available_records": sum(available_by_category.values()),
            "selected_records": batch.accepted_count + batch.rejected_count,
            "accepted_by_category": accepted_by_category,
            "available_by_category": available_by_category,
            "per_category_limit": per_category_limit,
            "canonical_identity_preflight": identity_preflight.to_dict(),
        }
        return batch

    @staticmethod
    def _validate_snapshot(snapshot: RawSnapshot) -> None:
        is_junction = getattr(os.path, "isjunction", None)
        if (
            snapshot.source_name != "buildcores_open_db"
            or snapshot.source_url != BUILDCORES_ARCHIVE_URL
            or snapshot.source_type != "import"
            or snapshot.parser_version != BUILDCORES_PARSER_VERSION
            or snapshot.media_type != "application/zip"
            or snapshot.licence_or_access_note != BUILDCORES_LICENSE_NOTE
            or snapshot.content_sha256 != BUILDCORES_ARCHIVE_SHA256
            or snapshot.byte_count > MAXIMUM_ARCHIVE_BYTES
            or snapshot.path.is_symlink()
            or bool(is_junction is not None and is_junction(snapshot.path))
            or not snapshot.path.is_file()
        ):
            raise ValueError("BuildCores snapshot does not match the pinned archive authority")
        if (
            snapshot.path.stat().st_size != snapshot.byte_count
            or sha256_file(snapshot.path) != BUILDCORES_ARCHIVE_SHA256
        ):
            raise ValueError("BuildCores snapshot bytes changed after acquisition")

    @staticmethod
    def _validate_archive(archive: zipfile.ZipFile) -> tuple[int, int]:
        entries = archive.infolist()
        if len(entries) > MAXIMUM_ARCHIVE_MEMBERS:
            raise ValueError("BuildCores archive exceeds its member-count limit")
        names: set[str] = set()
        uncompressed_bytes = 0
        for entry in entries:
            filename = entry.filename
            path = PurePosixPath(filename)
            if (
                not filename
                or "\\" in filename
                or path.is_absolute()
                or any(part in {"", ".", ".."} for part in path.parts)
                or filename in names
            ):
                raise ValueError("BuildCores archive contains an unsafe or duplicate member")
            names.add(filename)
            if entry.flag_bits & 0x1:
                raise ValueError("BuildCores archive contains an encrypted member")
            if entry.compress_type not in _ALLOWED_COMPRESSION:
                raise ValueError("BuildCores archive uses an unsupported compression method")
            if entry.file_size < 0 or entry.compress_size < 0:
                raise ValueError("BuildCores archive contains an invalid member size")
            if entry.file_size > MAXIMUM_MEMBER_BYTES:
                raise ValueError("BuildCores archive member exceeds its size limit")
            uncompressed_bytes += entry.file_size
            if uncompressed_bytes > MAXIMUM_UNCOMPRESSED_BYTES:
                raise ValueError("BuildCores archive exceeds its uncompressed-size limit")
            if entry.file_size and (
                entry.compress_size == 0
                or entry.file_size / entry.compress_size > MAXIMUM_COMPRESSION_RATIO
            ):
                raise ValueError("BuildCores archive member exceeds its compression-ratio limit")
        return len(entries), uncompressed_bytes

    @staticmethod
    def _category_entries(
        archive: zipfile.ZipFile, categories: Sequence[str]
    ) -> dict[str, list[zipfile.ZipInfo]]:
        grouped: dict[str, list[zipfile.ZipInfo]] = {category: [] for category in categories}
        for entry in archive.infolist():
            path = PurePosixPath(entry.filename)
            parts = path.parts
            if entry.is_dir() or path.suffix.lower() != ".json" or "open-db" not in parts:
                continue
            open_db_index = parts.index("open-db")
            if len(parts) != open_db_index + 3:
                continue
            category = parts[open_db_index + 1]
            if category in grouped:
                grouped[category].append(entry)
        for category_entries in grouped.values():
            category_entries.sort(key=lambda item: item.filename)
        return grouped

    @staticmethod
    def _relative_record_path(archive_path: str) -> str:
        normalised = archive_path.replace("\\", "/")
        marker = "/open-db/"
        if marker not in normalised:
            raise ValueError(f"record is outside open-db: {archive_path}")
        return f"open-db/{normalised.split(marker, maxsplit=1)[1]}"
