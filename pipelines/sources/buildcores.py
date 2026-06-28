"""Pinned BuildCores OpenDB catalogue adapter."""

from __future__ import annotations

import json
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath

from pipelines.parsing.normalizers import BUILDCORES_CATEGORY_MAP, normalise_buildcores_product
from pipelines.sources.base import (
    ParsedBatch,
    RawSnapshot,
    fetch_http_snapshot,
    rejected_record,
    snapshot_local_file,
)
from pc_build_recommender.catalog.canonical_identity import audit_canonical_envelopes

BUILDCORES_COMMIT = "6a64ab14fb1ab1bc1f3030d36b70bddcc2afeb0f"
BUILDCORES_ARCHIVE_URL = (
    "https://github.com/buildcores/buildcores-open-db/archive/"
    f"{BUILDCORES_COMMIT}.zip"
)
BUILDCORES_LICENSE_NOTE = (
    "BuildCores OpenDB database licensed ODC-By 1.0; attribution required. "
    "Community-maintained specifications require field-level verification for hard compatibility."
)
BUILDCORES_PARSER_VERSION = "buildcores-open-db-v1"
DEFAULT_CATEGORIES = tuple(BUILDCORES_CATEGORY_MAP)


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
            )
        return fetch_http_snapshot(
            source_name="buildcores_open_db",
            source_url=BUILDCORES_ARCHIVE_URL,
            source_type="import",
            raw_root=self.raw_root,
            parser_version=BUILDCORES_PARSER_VERSION,
            licence_or_access_note=BUILDCORES_LICENSE_NOTE,
            suffix=".zip",
            maximum_bytes=150 * 1024 * 1024,
        )

    def parse(
        self,
        snapshot: RawSnapshot,
        *,
        categories: Sequence[str] = DEFAULT_CATEGORIES,
        per_category_limit: int | None = 100,
        per_category_limits: Mapping[str, int] | None = None,
    ) -> ParsedBatch:
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
            "available_records": sum(available_by_category.values()),
            "selected_records": batch.accepted_count + batch.rejected_count,
            "accepted_by_category": accepted_by_category,
            "available_by_category": available_by_category,
            "per_category_limit": per_category_limit,
            "canonical_identity_preflight": identity_preflight.to_dict(),
        }
        return batch

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
