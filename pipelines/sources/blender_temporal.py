"""Bounded, exact temporal diffs for immutable Blender Open Data snapshots.

The normal source adapter deliberately selects a bounded sample from a large
snapshot.  External performance evaluation needs a different contract: identify
submissions that are present only in a later immutable snapshot, retain only a
pre-registered comparable cohort, and never treat a later observation of a
development family as an independent product test.

This module keeps the old-submission index on disk in SQLite.  Memory use is
therefore bounded by the configured novel-submission, novel-observation, and
retained-cohort limits rather than by the 400k+ source population.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import tempfile
import zipfile
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from pipelines.sources.base import RAW_SNAPSHOT_SCHEMA_VERSION, RawSnapshot
from pipelines.sources.blender import (
    BLENDER_LICENSE_NOTE,
    BLENDER_PARSER_VERSION,
    BLENDER_SNAPSHOT_URL,
    BlenderOpenDataAdapter,
)

TEMPORAL_DIFF_SCHEMA_VERSION = "pc-build-recommender.blender-temporal-diff.v1"
_SQLITE_LOOKUP_BATCH = 400


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_set_sha256(values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in sorted(values):
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def load_blender_raw_snapshot(archive_path: str | Path) -> RawSnapshot:
    """Load and fully verify one content-addressed Blender raw snapshot."""

    path = Path(archive_path).resolve()
    metadata_path = path.with_suffix(f"{path.suffix}.metadata.json")
    if not path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(path if not path.is_file() else metadata_path)
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != RAW_SNAPSHOT_SCHEMA_VERSION
    ):
        raise ValueError(f"unsupported raw snapshot metadata: {metadata_path}")
    if payload.get("source_name") != "blender_open_data":
        raise ValueError("snapshot source_name must be blender_open_data")
    if payload.get("parser_version") != BLENDER_PARSER_VERSION:
        raise ValueError("snapshot parser version does not match the active Blender parser")
    if payload.get("raw_file") != path.name:
        raise ValueError("snapshot metadata raw_file does not match the archive path")
    byte_count = path.stat().st_size
    if int(payload.get("byte_count", -1)) != byte_count:
        raise ValueError("snapshot byte count does not match metadata")
    content_sha256 = _sha256_file(path)
    if payload.get("content_sha256") != content_sha256:
        raise ValueError("snapshot content hash does not match metadata")
    retrieved_at = datetime.fromisoformat(str(payload["retrieved_at"]))
    if retrieved_at.tzinfo is None:
        raise ValueError("snapshot retrieval time must be timezone-aware")
    return RawSnapshot(
        source_name="blender_open_data",
        source_url=str(payload.get("source_url") or BLENDER_SNAPSHOT_URL),
        source_type=str(payload.get("source_type") or "benchmark"),
        retrieved_at=retrieved_at,
        content_sha256=content_sha256,
        byte_count=byte_count,
        media_type=str(payload.get("media_type") or "application/zip"),
        parser_version=BLENDER_PARSER_VERSION,
        licence_or_access_note=str(payload.get("licence_or_access_note") or BLENDER_LICENSE_NOTE),
        path=path,
        metadata_path=metadata_path,
    )


@dataclass(frozen=True, slots=True)
class BlenderCohortContract:
    """Complete comparable-cohort identity used before viewing external scores."""

    benchmark_version: str
    scene: str
    backend: str
    operating_system: str
    source_unit: str
    source_higher_is_better: bool
    source_score_field: str
    blender_build_hash: str
    benchmark_script: str
    scene_checksum: str

    def __post_init__(self) -> None:
        text_values = (
            self.benchmark_version,
            self.scene,
            self.backend,
            self.operating_system,
            self.source_unit,
            self.source_score_field,
            self.blender_build_hash,
            self.benchmark_script,
            self.scene_checksum,
        )
        if any(not value.strip() for value in text_values):
            raise ValueError("cohort contract text fields must not be blank")

    def to_dict(self) -> dict[str, object]:
        return {
            "benchmark_version": self.benchmark_version,
            "scene": self.scene,
            "backend": self.backend,
            "operating_system": self.operating_system,
            "source_unit": self.source_unit,
            "source_higher_is_better": self.source_higher_is_better,
            "source_score_field": self.source_score_field,
            "blender_build_hash": self.blender_build_hash,
            "benchmark_script": self.benchmark_script,
            "scene_checksum": self.scene_checksum,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> BlenderCohortContract:
        higher_is_better = payload.get("source_higher_is_better")
        if not isinstance(higher_is_better, bool):
            raise ValueError("source_higher_is_better must be an explicit boolean")
        return cls(
            benchmark_version=str(payload["benchmark_version"]),
            scene=str(payload["scene"]),
            backend=str(payload["backend"]),
            operating_system=str(payload["operating_system"]),
            source_unit=str(payload["source_unit"]),
            source_higher_is_better=higher_is_better,
            source_score_field=str(payload["source_score_field"]),
            blender_build_hash=str(payload["blender_build_hash"]),
            benchmark_script=str(payload["benchmark_script"]),
            scene_checksum=str(payload["scene_checksum"]),
        )

    def mismatch_reasons(self, record: Mapping[str, Any]) -> tuple[str, ...]:
        data = record.get("data")
        metadata = record.get("normalisation_metadata")
        if not isinstance(data, Mapping) or not isinstance(metadata, Mapping):
            return ("missing_normalised_contract",)
        actual = {
            "benchmark_version": data.get("benchmark_version"),
            "scene": data.get("preset"),
            "backend": metadata.get("device_type"),
            "operating_system": data.get("operating_system"),
            "source_unit": data.get("unit"),
            "source_higher_is_better": data.get("higher_is_better"),
            "source_score_field": metadata.get("score_source_field"),
            "blender_build_hash": metadata.get("blender_build_hash"),
            "benchmark_script": metadata.get("benchmark_script"),
            "scene_checksum": metadata.get("scene_checksum"),
        }
        expected = self.to_dict()
        return tuple(name for name, value in expected.items() if actual.get(name) != value)


@dataclass(frozen=True, slots=True)
class BlenderTemporalDiff:
    """Auditable novel-submission result and the bounded exact-cohort records."""

    old_submission_count: int
    new_submission_count: int
    removed_submission_count: int
    novel_submission_ids: tuple[str, ...]
    novel_raw_observation_count: int
    novel_normalised_observation_count: int
    retained_cohort_records: tuple[dict[str, Any], ...]
    normalisation_rejection_counts: dict[str, int]
    cohort_mismatch_counts: dict[str, int]
    device_counts: dict[str, int]
    unique_hardware_name_count: int
    old_max_created_at: str | None
    new_max_created_at: str | None
    novel_min_created_at: str | None
    novel_max_created_at: str | None
    peak_retained_record_count: int
    schema_version: str = TEMPORAL_DIFF_SCHEMA_VERSION

    @property
    def novel_submission_count(self) -> int:
        return len(self.novel_submission_ids)

    @property
    def novel_submission_ids_sha256(self) -> str:
        return _stable_set_sha256(self.novel_submission_ids)

    @property
    def retained_source_record_ids_sha256(self) -> str:
        values = [str(record["source_record_id"]) for record in self.retained_cohort_records]
        return _stable_set_sha256(values)

    def summary(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "old_submission_count": self.old_submission_count,
            "new_submission_count": self.new_submission_count,
            "removed_submission_count": self.removed_submission_count,
            "new_snapshot_is_strict_superset": (
                self.removed_submission_count == 0 and self.novel_submission_count > 0
            ),
            "novel_submission_count": self.novel_submission_count,
            "novel_submission_ids_sha256": self.novel_submission_ids_sha256,
            "novel_raw_observation_count": self.novel_raw_observation_count,
            "novel_normalised_observation_count": self.novel_normalised_observation_count,
            "retained_cohort_record_count": len(self.retained_cohort_records),
            "retained_source_record_ids_sha256": self.retained_source_record_ids_sha256,
            "normalisation_rejection_counts": dict(
                sorted(self.normalisation_rejection_counts.items())
            ),
            "cohort_mismatch_counts": dict(sorted(self.cohort_mismatch_counts.items())),
            "device_counts": dict(sorted(self.device_counts.items())),
            "unique_hardware_name_count": self.unique_hardware_name_count,
            "old_max_created_at": self.old_max_created_at,
            "new_max_created_at": self.new_max_created_at,
            "novel_min_created_at": self.novel_min_created_at,
            "novel_max_created_at": self.novel_max_created_at,
            "bounded_memory": {
                "old_submission_index": "sqlite_on_disk",
                "peak_retained_record_count": self.peak_retained_record_count,
            },
        }


def _submission_stream(snapshot: RawSnapshot) -> Iterator[tuple[int, dict[str, Any]]]:
    with zipfile.ZipFile(snapshot.path) as archive:
        members = [item for item in archive.infolist() if item.filename.lower().endswith(".jsonl")]
        if len(members) != 1:
            raise ValueError(f"expected exactly one Blender JSONL member, found {len(members)}")
        with archive.open(members[0]) as source:
            for line_number, raw_line in enumerate(source, start=1):
                try:
                    value = json.loads(raw_line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"{snapshot.path}:{members[0].filename}:{line_number}: invalid JSON"
                    ) from exc
                if not isinstance(value, dict):
                    raise ValueError(
                        f"{snapshot.path}:{members[0].filename}:{line_number}: expected object"
                    )
                yield line_number, value


def _created_at(value: Mapping[str, Any]) -> str | None:
    raw = value.get("created_at")
    if raw is None:
        return None
    try:
        parsed = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.isoformat() if parsed.tzinfo is not None else None


def _maximum_timestamp(current: str | None, candidate: str | None) -> str | None:
    if candidate is None:
        return current
    return candidate if current is None or candidate > current else current


def _minimum_timestamp(current: str | None, candidate: str | None) -> str | None:
    if candidate is None:
        return current
    return candidate if current is None or candidate < current else current


def _existing_ids(connection: sqlite3.Connection, identifiers: Sequence[str]) -> set[str]:
    result: set[str] = set()
    for offset in range(0, len(identifiers), _SQLITE_LOOKUP_BATCH):
        values = identifiers[offset : offset + _SQLITE_LOOKUP_BATCH]
        placeholders = ",".join("?" for _ in values)
        rows = connection.execute(
            f"SELECT submission_id FROM old_ids WHERE submission_id IN ({placeholders})",  # noqa: S608
            tuple(values),
        )
        result.update(str(row[0]) for row in rows)
    return result


def _process_novel_submission(
    submission: dict[str, Any],
    *,
    snapshot: RawSnapshot,
    contract: BlenderCohortContract,
    retained: list[dict[str, Any]],
    rejection_counts: Counter[str],
    mismatch_counts: Counter[str],
    device_counts: Counter[str],
    hardware_names: set[str],
    max_novel_observations: int,
    max_retained_records: int,
    counters: Counter[str],
) -> None:
    observations = submission.get("data")
    if not isinstance(observations, list):
        rejection_counts["missing_observation_list"] += 1
        return
    for observation_index, observation in enumerate(observations):
        counters["raw_observations"] += 1
        if counters["raw_observations"] > max_novel_observations:
            raise MemoryError(
                f"novel observation count exceeds bounded limit {max_novel_observations}"
            )
        if not isinstance(observation, dict):
            rejection_counts["observation_not_object"] += 1
            continue
        try:
            normalised = BlenderOpenDataAdapter._normalise_observation(
                submission=submission,
                observation=observation,
                observation_index=observation_index,
                snapshot=snapshot,
            )
        except (TypeError, ValueError) as exc:
            rejection_counts[f"{type(exc).__name__}:{str(exc).split(':', 1)[0]}"] += 1
            continue
        counters["normalised_observations"] += 1
        metadata = normalised["normalisation_metadata"]
        device_counts[str(metadata["device_type"])] += 1
        hardware_names.add(str(metadata["hardware_name"]))
        reasons = contract.mismatch_reasons(normalised)
        if reasons:
            mismatch_counts.update(reasons)
            continue
        retained.append(normalised)
        if len(retained) > max_retained_records:
            raise MemoryError(f"retained cohort exceeds bounded limit {max_retained_records}")


def diff_blender_snapshots(
    old_snapshot: RawSnapshot,
    new_snapshot: RawSnapshot,
    *,
    contract: BlenderCohortContract,
    max_submissions: int = 2_000_000,
    max_novel_submissions: int = 100_000,
    max_novel_observations: int = 500_000,
    max_retained_records: int = 100_000,
) -> BlenderTemporalDiff:
    """Return an exact ID diff while retaining only the pre-registered cohort."""

    if old_snapshot.source_name != new_snapshot.source_name:
        raise ValueError("snapshot sources do not match")
    if old_snapshot.content_sha256 == new_snapshot.content_sha256:
        raise ValueError("temporal snapshots must have different content hashes")
    if new_snapshot.retrieved_at <= old_snapshot.retrieved_at:
        raise ValueError("new snapshot retrieval time must be later than old snapshot")
    limits = (max_submissions, max_novel_submissions, max_novel_observations, max_retained_records)
    if any(value < 1 for value in limits):
        raise ValueError("all temporal diff limits must be positive")

    retained: list[dict[str, Any]] = []
    novel_ids: list[str] = []
    rejection_counts: Counter[str] = Counter()
    mismatch_counts: Counter[str] = Counter()
    device_counts: Counter[str] = Counter()
    hardware_names: set[str] = set()
    counters: Counter[str] = Counter()
    old_max_created_at: str | None = None
    new_max_created_at: str | None = None
    novel_min_created_at: str | None = None
    novel_max_created_at: str | None = None

    with tempfile.TemporaryDirectory(prefix="pcbr-blender-diff-") as temporary:
        database = Path(temporary) / "submission-index.sqlite3"
        connection = sqlite3.connect(database)
        try:
            connection.execute("PRAGMA journal_mode=OFF")
            connection.execute("PRAGMA synchronous=OFF")
            connection.execute("PRAGMA temp_store=FILE")
            connection.execute(
                "CREATE TABLE old_ids ("
                "submission_id TEXT PRIMARY KEY, "
                "seen_in_new INTEGER NOT NULL DEFAULT 0)"
            )
            connection.execute("CREATE TABLE new_ids (submission_id TEXT PRIMARY KEY)")
            old_batch: list[tuple[str]] = []
            for _, submission in _submission_stream(old_snapshot):
                counters["old_submissions"] += 1
                if counters["old_submissions"] > max_submissions:
                    raise MemoryError(f"old snapshot exceeds submission limit {max_submissions}")
                submission_id = str(submission.get("id", "")).strip()
                if not submission_id:
                    raise ValueError("old snapshot contains a missing submission id")
                old_max_created_at = _maximum_timestamp(old_max_created_at, _created_at(submission))
                old_batch.append((submission_id,))
                if len(old_batch) >= 10_000:
                    before = connection.total_changes
                    connection.executemany("INSERT OR IGNORE INTO old_ids VALUES (?, 0)", old_batch)
                    if connection.total_changes - before != len(old_batch):
                        raise ValueError("old snapshot contains duplicate submission ids")
                    old_batch.clear()
            if old_batch:
                before = connection.total_changes
                connection.executemany("INSERT OR IGNORE INTO old_ids VALUES (?, 0)", old_batch)
                if connection.total_changes - before != len(old_batch):
                    raise ValueError("old snapshot contains duplicate submission ids")
            connection.commit()

            new_batch: list[tuple[str, dict[str, Any]]] = []

            def flush_new_batch() -> None:
                nonlocal novel_min_created_at, novel_max_created_at
                if not new_batch:
                    return
                identifiers = [submission_id for submission_id, _ in new_batch]
                before = connection.total_changes
                connection.executemany(
                    "INSERT OR IGNORE INTO new_ids (submission_id) VALUES (?)",
                    ((value,) for value in identifiers),
                )
                if connection.total_changes - before != len(identifiers):
                    raise ValueError("new snapshot contains duplicate submission ids")
                existing = _existing_ids(connection, identifiers)
                if existing:
                    connection.executemany(
                        "UPDATE old_ids SET seen_in_new = 1 WHERE submission_id = ?",
                        ((value,) for value in existing),
                    )
                for submission_id, submission in new_batch:
                    if submission_id in existing:
                        continue
                    novel_ids.append(submission_id)
                    if len(novel_ids) > max_novel_submissions:
                        raise MemoryError(
                            f"novel submission count exceeds bounded limit {max_novel_submissions}"
                        )
                    created_at = _created_at(submission)
                    novel_min_created_at = _minimum_timestamp(novel_min_created_at, created_at)
                    novel_max_created_at = _maximum_timestamp(novel_max_created_at, created_at)
                    _process_novel_submission(
                        submission,
                        snapshot=new_snapshot,
                        contract=contract,
                        retained=retained,
                        rejection_counts=rejection_counts,
                        mismatch_counts=mismatch_counts,
                        device_counts=device_counts,
                        hardware_names=hardware_names,
                        max_novel_observations=max_novel_observations,
                        max_retained_records=max_retained_records,
                        counters=counters,
                    )
                new_batch.clear()

            for _, submission in _submission_stream(new_snapshot):
                counters["new_submissions"] += 1
                if counters["new_submissions"] > max_submissions:
                    raise MemoryError(f"new snapshot exceeds submission limit {max_submissions}")
                submission_id = str(submission.get("id", "")).strip()
                if not submission_id:
                    raise ValueError("new snapshot contains a missing submission id")
                new_max_created_at = _maximum_timestamp(new_max_created_at, _created_at(submission))
                new_batch.append((submission_id, submission))
                if len(new_batch) >= _SQLITE_LOOKUP_BATCH:
                    flush_new_batch()
            flush_new_batch()
            connection.commit()
            removed = int(
                connection.execute("SELECT COUNT(*) FROM old_ids WHERE seen_in_new = 0").fetchone()[
                    0
                ]
            )
        finally:
            connection.close()

    return BlenderTemporalDiff(
        old_submission_count=int(counters["old_submissions"]),
        new_submission_count=int(counters["new_submissions"]),
        removed_submission_count=removed,
        novel_submission_ids=tuple(sorted(novel_ids)),
        novel_raw_observation_count=int(counters["raw_observations"]),
        novel_normalised_observation_count=int(counters["normalised_observations"]),
        retained_cohort_records=tuple(retained),
        normalisation_rejection_counts=dict(rejection_counts),
        cohort_mismatch_counts=dict(mismatch_counts),
        device_counts=dict(device_counts),
        unique_hardware_name_count=len(hardware_names),
        old_max_created_at=old_max_created_at,
        new_max_created_at=new_max_created_at,
        novel_min_created_at=novel_min_created_at,
        novel_max_created_at=novel_max_created_at,
        peak_retained_record_count=len(retained),
    )
