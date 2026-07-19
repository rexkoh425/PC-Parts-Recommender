"""Fail-closed retention maintenance for the quarantined WDC research corpus.

The WDC import deliberately remains outside the production catalogue, but its raw
snapshots, category index, paused work, and sealed research artifacts are still
subject to a 365-day internal retention deadline.  This module derives deletion
eligibility from the immutable snapshot metadata and WDC manifests; it never
accepts a caller-selected path below a broad root and it leaves malformed or
unrecognised data untouched.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import stat as stat_module
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Final

from pipelines.sources.base import RAW_SNAPSHOT_SCHEMA_VERSION, sha256_file
from pipelines.sources.wdc_products import (
    WDC_CATEGORY_INDEX_SCHEMA,
    WDC_CATEGORY_SOURCE_NAME,
    WDC_CORPUS_SOURCE_NAME,
    WDC_PARSER_VERSION,
    WDC_RESEARCH_MANIFEST_SCHEMA,
    WDC_RESEARCH_RECORD_SCHEMA,
    WDC_RESEARCH_RETENTION_DAYS,
    WDC_SELECTION_POLICY_VERSION,
)

WDC_RETENTION_SCHEMA_VERSION: Final = "pc-build-recommender.wdc-retention.v1"
_MAX_METADATA_BYTES: Final = 512 * 1024
_DEFAULT_MAXIMUM_ENTRIES: Final = 100_000
_ACTION_SAMPLE_LIMIT: Final = 20
_SHA256_PATTERN: Final = re.compile(r"^[0-9a-f]{64}$")
_RUN_PATTERN: Final = re.compile(r"^[0-9a-f]{32}$")
_RAW_FILE_PATTERN: Final = re.compile(r"^(?P<sha>[0-9a-f]{64})\.json(?:\.gz)?$")
_RAW_METADATA_SUFFIX: Final = ".metadata.json"
_WORK_ALLOWED_FILES: Final = frozenset(
    {"checkpoint.json", "records.jsonl.part", "records.jsonl", "manifest.json"}
)


class WDCRetentionError(RuntimeError):
    """Raised when WDC retention cannot prove a deletion target is safe."""


@dataclass(frozen=True, slots=True)
class WDCRetentionReport:
    """Aggregate result of a bounded WDC retention maintenance run."""

    dry_run: bool
    evaluated_at: str
    raw_receipts_scanned: int
    raw_pairs_eligible: int
    raw_pairs_removed: int
    category_index_eligible: bool
    category_index_removed: bool
    sealed_runs_scanned: int
    sealed_runs_eligible: int
    sealed_runs_removed: int
    working_runs_scanned: int
    working_runs_eligible: int
    working_runs_removed: int
    unrelated_entries_preserved: int
    action_sample: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["action_sample"] = list(self.action_sample)
        return result


@dataclass(frozen=True, slots=True)
class _PlannedFile:
    path: Path
    size: int
    mtime_ns: int
    sha256: str | None


@dataclass(frozen=True, slots=True)
class _PlannedTree:
    root: Path
    files: tuple[_PlannedFile, ...]
    directories: tuple[_PlannedFile, ...]


@dataclass(slots=True)
class _WorkBudget:
    maximum: int
    consumed: int = 0

    def consume(self, label: str) -> None:
        self.consumed += 1
        if self.consumed > self.maximum:
            raise WDCRetentionError(
                f"WDC retention exceeded the {self.maximum}-entry work limit while scanning {label}"
            )


def _is_linklike(path: Path) -> bool:
    is_junction = getattr(os.path, "isjunction", None)
    return path.is_symlink() or bool(is_junction is not None and is_junction(path))


def _require_regular(path: Path, *, label: str) -> os.stat_result:
    if _is_linklike(path):
        raise WDCRetentionError(f"{label} must not be a symlink or junction: {path}")
    try:
        metadata = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise WDCRetentionError(f"cannot inspect {label}: {path}") from exc
    if not stat_module.S_ISREG(metadata.st_mode):
        raise WDCRetentionError(f"{label} must be a regular file: {path}")
    return metadata


def _require_directory(path: Path, *, label: str) -> os.stat_result:
    if _is_linklike(path):
        raise WDCRetentionError(f"{label} must not be a symlink or junction: {path}")
    try:
        metadata = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise WDCRetentionError(f"cannot inspect {label}: {path}") from exc
    if not stat_module.S_ISDIR(metadata.st_mode):
        raise WDCRetentionError(f"{label} must be a directory: {path}")
    return metadata


def _contain(path: Path, root: Path, *, label: str) -> None:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise WDCRetentionError(f"{label} escaped its exact retention root") from exc


def _retention_root(value: str | Path, *, label: str) -> Path:
    """Return a lexical absolute root without silently traversing a root symlink."""

    path = Path(value).absolute()
    if path.exists():
        _require_directory(path, label=label)
    return path


def _read_json_object_with_digest(path: Path, *, label: str) -> tuple[dict[str, Any], str]:
    metadata = _require_regular(path, label=label)
    if metadata.st_size > _MAX_METADATA_BYTES:
        raise WDCRetentionError(f"{label} exceeds {_MAX_METADATA_BYTES} bytes: {path}")
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise WDCRetentionError(f"cannot read {label}: {path}") from exc
    after = _require_regular(path, label=label)
    if metadata.st_size != after.st_size or metadata.st_mtime_ns != after.st_mtime_ns:
        raise WDCRetentionError(f"{label} changed while it was read: {path}")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WDCRetentionError(f"{label} is not valid UTF-8 JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise WDCRetentionError(f"{label} must be a JSON object: {path}")
    return payload, hashlib.sha256(raw).hexdigest()


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    return _read_json_object_with_digest(path, label=label)[0]


def _timestamp(value: object, *, label: str) -> datetime:
    if not isinstance(value, str):
        raise WDCRetentionError(f"{label} must be an ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise WDCRetentionError(f"{label} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise WDCRetentionError(f"{label} must include a UTC offset")
    return parsed.astimezone(UTC)


def _sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise WDCRetentionError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _plan_file(path: Path, *, label: str, sha256: str | None = None) -> _PlannedFile:
    metadata = _require_regular(path, label=label)
    if sha256 is not None:
        observed = sha256_file(path)
        if observed != sha256:
            raise WDCRetentionError(f"{label} SHA-256 does not match its receipt: {path}")
    return _PlannedFile(
        path=path,
        size=metadata.st_size,
        mtime_ns=metadata.st_mtime_ns,
        sha256=sha256,
    )


def _validate_planned_file(target: _PlannedFile, *, label: str) -> None:
    metadata = _require_regular(target.path, label=label)
    if metadata.st_size != target.size or metadata.st_mtime_ns != target.mtime_ns:
        raise WDCRetentionError(f"{label} changed before deletion: {target.path}")
    if target.sha256 is not None and sha256_file(target.path) != target.sha256:
        raise WDCRetentionError(f"{label} content changed before deletion: {target.path}")


def _iter_entries(path: Path, *, budget: _WorkBudget, label: str) -> tuple[Path, ...]:
    _require_directory(path, label=label)
    try:
        entries = tuple(sorted(path.iterdir(), key=lambda entry: entry.name))
    except OSError as exc:
        raise WDCRetentionError(f"cannot scan {label}: {path}") from exc
    for entry in entries:
        budget.consume(f"{label}/{entry.name}")
    return entries


def _raw_deadline(
    payload: dict[str, Any], *, source_name: str, metadata_path: Path
) -> tuple[str, datetime, str, int]:
    expected_fields = {
        "schema_version",
        "source_name",
        "source_url",
        "source_type",
        "retrieved_at",
        "content_sha256",
        "byte_count",
        "media_type",
        "parser_version",
        "licence_or_access_note",
        "raw_file",
    }
    if set(payload) != expected_fields:
        raise WDCRetentionError(
            f"raw WDC receipt fields are incomplete or unknown: {metadata_path}"
        )
    if payload.get("schema_version") != RAW_SNAPSHOT_SCHEMA_VERSION:
        raise WDCRetentionError(f"raw WDC receipt has an unsupported schema: {metadata_path}")
    if payload.get("source_name") != source_name:
        raise WDCRetentionError(f"raw WDC receipt source does not match its root: {metadata_path}")
    if payload.get("parser_version") != WDC_PARSER_VERSION:
        raise WDCRetentionError(
            f"raw WDC receipt has an unexpected parser version: {metadata_path}"
        )
    digest = _sha256(payload.get("content_sha256"), label="raw receipt content_sha256")
    raw_file = payload.get("raw_file")
    if not isinstance(raw_file, str) or _RAW_FILE_PATTERN.fullmatch(raw_file) is None:
        raise WDCRetentionError(f"raw WDC receipt has an unsafe raw_file: {metadata_path}")
    raw_match = _RAW_FILE_PATTERN.fullmatch(raw_file)
    assert raw_match is not None
    if raw_match.group("sha") != digest:
        raise WDCRetentionError(
            f"raw WDC receipt file digest disagrees with its content digest: {metadata_path}"
        )
    if metadata_path.name != raw_file + _RAW_METADATA_SUFFIX:
        raise WDCRetentionError(f"raw WDC receipt name does not bind its raw file: {metadata_path}")
    retrieved_at = _timestamp(payload.get("retrieved_at"), label="raw receipt retrieved_at")
    byte_count = payload.get("byte_count")
    if type(byte_count) is not int or byte_count < 0:
        raise WDCRetentionError(f"raw WDC receipt byte_count is invalid: {metadata_path}")
    return digest, retrieved_at + timedelta(days=WDC_RESEARCH_RETENTION_DAYS), raw_file, byte_count


def _scan_raw_source(
    root: Path,
    *,
    source_name: str,
    now: datetime,
    budget: _WorkBudget,
) -> tuple[list[tuple[_PlannedFile | None, _PlannedFile]], dict[str, datetime], int, int]:
    if not root.exists():
        return [], {}, 0, 0
    entries = _iter_entries(root, budget=budget, label=f"raw/{source_name}")
    plans: list[tuple[_PlannedFile | None, _PlannedFile]] = []
    deadlines: dict[str, datetime] = {}
    receipts_scanned = 0
    unrelated = 0
    for entry in entries:
        if entry.name.endswith(_RAW_METADATA_SUFFIX):
            receipts_scanned += 1
            payload, receipt_digest = _read_json_object_with_digest(
                entry,
                label="raw WDC receipt",
            )
            digest, deadline, raw_name, byte_count = _raw_deadline(
                payload,
                source_name=source_name,
                metadata_path=entry,
            )
            previous = deadlines.setdefault(digest, deadline)
            if previous != deadline:
                raise WDCRetentionError(
                    f"WDC snapshot digest has inconsistent retention deadlines: {digest}"
                )
            raw_path = root / raw_name
            if (
                raw_path.exists()
                and _require_regular(
                    raw_path,
                    label="raw WDC body",
                ).st_size
                != byte_count
            ):
                raise WDCRetentionError(f"raw WDC body size does not match its receipt: {raw_path}")
            if deadline > now:
                if not raw_path.exists():
                    raise WDCRetentionError(f"active raw WDC receipt is missing its body: {entry}")
                continue
            receipt_plan = _plan_file(
                entry,
                label="expired raw WDC receipt",
                sha256=receipt_digest,
            )
            body_plan = (
                _plan_file(raw_path, label="expired raw WDC body", sha256=digest)
                if raw_path.exists()
                else None
            )
            plans.append((body_plan, receipt_plan))
            continue
        if _RAW_FILE_PATTERN.fullmatch(entry.name) is None:
            unrelated += 1
            continue
        expected_receipt = root / (entry.name + _RAW_METADATA_SUFFIX)
        if not expected_receipt.exists():
            unrelated += 1
    return plans, deadlines, receipts_scanned, unrelated


def _category_index_plan(
    index_path: Path,
    *,
    now: datetime,
    deadlines: dict[str, datetime],
) -> tuple[tuple[_PlannedFile, ...], bool]:
    if not index_path.exists():
        return (), False
    _require_regular(index_path, label="WDC category index")
    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(f"file:{index_path}?mode=ro", uri=True)
        rows = connection.execute("SELECT key, value FROM import_metadata").fetchall()
    except sqlite3.Error as exc:
        raise WDCRetentionError(f"cannot read WDC category index metadata: {index_path}") from exc
    finally:
        if connection is not None:
            connection.close()
    metadata = {str(key): str(value) for key, value in rows}
    if metadata.get("schema_version") != WDC_CATEGORY_INDEX_SCHEMA:
        raise WDCRetentionError(f"WDC category index has an unsupported schema: {index_path}")
    if metadata.get("parser_version") != WDC_PARSER_VERSION:
        raise WDCRetentionError(
            f"WDC category index has an unexpected parser version: {index_path}"
        )
    source_sha = _sha256(metadata.get("source_sha256"), label="WDC category index source_sha256")
    deadline = _timestamp(metadata.get("retention_deadline"), label="WDC category index deadline")
    source_deadline = deadlines.get(source_sha)
    if source_deadline is not None and source_deadline != deadline:
        raise WDCRetentionError("WDC category index deadline disagrees with its source receipt")
    if deadline > now:
        return (), False
    paths = [index_path]
    for suffix in ("-wal", "-shm"):
        sidecar = Path(f"{index_path}{suffix}")
        if sidecar.exists():
            paths.append(sidecar)
    return tuple(
        _plan_file(path, label="expired WDC category-index artifact") for path in paths
    ), True


def _capture_tree(root: Path, *, containment_root: Path, budget: _WorkBudget) -> _PlannedTree:
    _contain(root, containment_root, label="WDC deletion tree")
    _require_directory(root, label="WDC deletion tree")
    files: list[_PlannedFile] = []
    directories: list[_PlannedFile] = []

    def visit(directory: Path) -> None:
        metadata = _require_directory(directory, label="WDC deletion directory")
        directories.append(
            _PlannedFile(directory, metadata.st_size, metadata.st_mtime_ns, sha256=None)
        )
        for entry in _iter_entries(directory, budget=budget, label="WDC deletion tree"):
            _contain(entry, containment_root, label="WDC deletion tree")
            if _is_linklike(entry):
                raise WDCRetentionError(
                    f"WDC deletion tree contains a symlink or junction: {entry}"
                )
            metadata = entry.stat(follow_symlinks=False)
            if stat_module.S_ISDIR(metadata.st_mode):
                visit(entry)
            elif stat_module.S_ISREG(metadata.st_mode):
                files.append(_PlannedFile(entry, metadata.st_size, metadata.st_mtime_ns, None))
            else:
                raise WDCRetentionError(f"WDC deletion tree contains a non-regular entry: {entry}")

    visit(root)
    return _PlannedTree(
        root=root,
        files=tuple(files),
        directories=tuple(sorted(directories, key=lambda item: len(item.path.parts), reverse=True)),
    )


def _manifest_deadline(
    manifest_path: Path,
    *,
    deadlines: dict[str, datetime],
    verify_output: bool,
) -> datetime:
    manifest = _read_json_object(manifest_path, label="sealed WDC manifest")
    expected = {
        "schema_version": WDC_RESEARCH_MANIFEST_SCHEMA,
        "status": "complete",
        "parser_version": WDC_PARSER_VERSION,
        "record_schema_version": WDC_RESEARCH_RECORD_SCHEMA,
        "selection_policy_version": WDC_SELECTION_POLICY_VERSION,
    }
    for field, value in expected.items():
        if manifest.get(field) != value:
            raise WDCRetentionError(f"sealed WDC manifest has invalid {field}: {manifest_path}")
    source_sha = _sha256(manifest.get("source_sha256"), label="sealed WDC source_sha256")
    _sha256(manifest.get("category_source_sha256"), label="sealed WDC category_source_sha256")
    _sha256(manifest.get("policy_sha256"), label="sealed WDC policy_sha256")
    source_snapshot = manifest.get("source_snapshot")
    if not isinstance(source_snapshot, dict):
        raise WDCRetentionError(f"sealed WDC manifest has no source snapshot: {manifest_path}")
    if source_snapshot.get("source_name") != WDC_CORPUS_SOURCE_NAME:
        raise WDCRetentionError(f"sealed WDC manifest has an unexpected source: {manifest_path}")
    if (
        _sha256(source_snapshot.get("content_sha256"), label="sealed WDC snapshot digest")
        != source_sha
    ):
        raise WDCRetentionError(f"sealed WDC manifest source digest mismatch: {manifest_path}")
    retrieved_at = _timestamp(
        source_snapshot.get("retrieved_at"), label="sealed WDC retrieval time"
    )
    deadline = _timestamp(manifest.get("retention_deadline"), label="sealed WDC retention deadline")
    if deadline != retrieved_at + timedelta(days=WDC_RESEARCH_RETENTION_DAYS):
        raise WDCRetentionError(
            f"sealed WDC manifest retention deadline is invalid: {manifest_path}"
        )
    source_deadline = deadlines.get(source_sha)
    if source_deadline is not None and source_deadline != deadline:
        raise WDCRetentionError("sealed WDC manifest deadline disagrees with its source receipt")
    quarantine = manifest.get("quarantine")
    if not isinstance(quarantine, dict) or any(
        quarantine.get(field) is not False
        for field in (
            "production_eligible",
            "singapore_market_evidence",
            "current_price_or_stock_evidence",
            "model_training_eligible",
            "published_metric_claim_eligible",
        )
    ):
        raise WDCRetentionError(
            f"sealed WDC manifest is not a research quarantine: {manifest_path}"
        )
    output_path = manifest_path.parent / "records.jsonl"
    expected_hash = _sha256(manifest.get("output_sha256"), label="sealed WDC output_sha256")
    expected_bytes = manifest.get("output_bytes")
    if type(expected_bytes) is not int or expected_bytes < 0:
        raise WDCRetentionError(f"sealed WDC manifest output_bytes is invalid: {manifest_path}")
    output_metadata = _require_regular(output_path, label="sealed WDC output")
    if output_metadata.st_size != expected_bytes:
        raise WDCRetentionError(
            f"sealed WDC output size does not match its manifest: {output_path}"
        )
    if verify_output and sha256_file(output_path) != expected_hash:
        raise WDCRetentionError(
            f"sealed WDC output hash does not match its manifest: {output_path}"
        )
    return deadline


def _checkpoint_deadline(path: Path, *, deadlines: dict[str, datetime]) -> datetime:
    checkpoint = _read_json_object(path, label="working WDC checkpoint")
    if checkpoint.get("schema_version") != WDC_RESEARCH_MANIFEST_SCHEMA:
        raise WDCRetentionError(f"working WDC checkpoint has an unsupported schema: {path}")
    if checkpoint.get("state") != "working":
        raise WDCRetentionError(f"working WDC checkpoint has an invalid state: {path}")
    source_sha = _sha256(checkpoint.get("source_sha256"), label="working WDC source_sha256")
    _sha256(checkpoint.get("category_source_sha256"), label="working WDC category_source_sha256")
    _sha256(checkpoint.get("policy_sha256"), label="working WDC policy_sha256")
    deadline = deadlines.get(source_sha)
    if deadline is None:
        raise WDCRetentionError(
            "working WDC checkpoint cannot prove its retention deadline without its raw receipt: "
            f"{path}"
        )
    return deadline


def _scan_quarantine(
    output_root: Path,
    *,
    now: datetime,
    deadlines: dict[str, datetime],
    budget: _WorkBudget,
) -> tuple[list[_PlannedTree], list[_PlannedTree], int, int, int]:
    root = output_root / WDC_CORPUS_SOURCE_NAME
    if not root.exists():
        return [], [], 0, 0, 0
    entries = _iter_entries(root, budget=budget, label="WDC quarantine root")
    sealed: list[_PlannedTree] = []
    working: list[_PlannedTree] = []
    sealed_scanned = 0
    working_scanned = 0
    unrelated = 0
    for entry in entries:
        if entry.name == ".work":
            for work in _iter_entries(entry, budget=budget, label="WDC quarantine work root"):
                if _RUN_PATTERN.fullmatch(work.name) is None:
                    unrelated += 1
                    continue
                working_scanned += 1
                _require_directory(work, label="WDC working run")
                work_entries = _iter_entries(work, budget=budget, label="WDC working run")
                unknown = {child.name for child in work_entries} - _WORK_ALLOWED_FILES
                if unknown:
                    raise WDCRetentionError(
                        f"working WDC run has unknown entries and will not be deleted: {work}"
                    )
                checkpoint = work / "checkpoint.json"
                if not checkpoint.is_file():
                    raise WDCRetentionError(f"working WDC run has no checkpoint: {work}")
                if _checkpoint_deadline(checkpoint, deadlines=deadlines) <= now:
                    working.append(_capture_tree(work, containment_root=root, budget=budget))
            continue
        if _RUN_PATTERN.fullmatch(entry.name) is None:
            unrelated += 1
            continue
        sealed_scanned += 1
        _require_directory(entry, label="sealed WDC run")
        manifest_path = entry / "manifest.json"
        deadline = _manifest_deadline(
            manifest_path,
            deadlines=deadlines,
            verify_output=False,
        )
        if deadline <= now:
            _manifest_deadline(manifest_path, deadlines=deadlines, verify_output=True)
            sealed.append(_capture_tree(entry, containment_root=root, budget=budget))
    return sealed, working, sealed_scanned, working_scanned, unrelated


def _delete_file(target: _PlannedFile, *, label: str) -> None:
    _validate_planned_file(target, label=label)
    try:
        target.path.unlink()
    except OSError as exc:
        raise WDCRetentionError(f"cannot delete {label}: {target.path}") from exc


def _delete_tree(tree: _PlannedTree) -> None:
    for target in tree.files:
        _delete_file(target, label="expired WDC artifact file")
    for target in tree.directories:
        _require_directory(target.path, label="expired WDC artifact directory")
        try:
            target.path.rmdir()
        except OSError as exc:
            raise WDCRetentionError(
                f"cannot delete expired WDC artifact directory: {target.path}"
            ) from exc


def maintain_wdc_research_retention(
    *,
    raw_root: str | Path,
    output_root: str | Path,
    category_index: str | Path,
    now: datetime | None = None,
    dry_run: bool = False,
    maximum_entries: int = _DEFAULT_MAXIMUM_ENTRIES,
) -> WDCRetentionReport:
    """Delete only WDC artifacts whose immutable retention evidence has expired.

    Planning completes before any deletion.  Unknown files are preserved, malformed
    governed artifacts fail the run closed, and every destructive target is rechecked
    immediately before removal.
    """

    if maximum_entries < 1:
        raise ValueError("maximum_entries must be positive")
    timestamp = now or datetime.now(UTC)
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("retention maintenance time must be timezone aware")
    timestamp = timestamp.astimezone(UTC)
    raw_root_path = _retention_root(raw_root, label="WDC raw root")
    output_root_path = _retention_root(output_root, label="WDC quarantine root")
    index_path = Path(category_index).absolute()
    if index_path.parent.exists():
        _require_directory(index_path.parent, label="WDC category-index parent")
    budget = _WorkBudget(maximum_entries)

    raw_plans: list[tuple[_PlannedFile | None, _PlannedFile]] = []
    deadlines: dict[str, datetime] = {}
    raw_receipts_scanned = 0
    unrelated = 0
    for source_name in (WDC_CORPUS_SOURCE_NAME, WDC_CATEGORY_SOURCE_NAME):
        plans, source_deadlines, scanned, preserved = _scan_raw_source(
            raw_root_path / source_name,
            source_name=source_name,
            now=timestamp,
            budget=budget,
        )
        for digest, deadline in source_deadlines.items():
            existing = deadlines.setdefault(digest, deadline)
            if existing != deadline:
                raise WDCRetentionError(f"WDC source digest has inconsistent deadlines: {digest}")
        raw_plans.extend(plans)
        raw_receipts_scanned += scanned
        unrelated += preserved

    index_plan, index_eligible = _category_index_plan(
        index_path,
        now=timestamp,
        deadlines=deadlines,
    )
    sealed_plans, working_plans, sealed_scanned, working_scanned, quarantine_unrelated = (
        _scan_quarantine(
            output_root_path,
            now=timestamp,
            deadlines=deadlines,
            budget=budget,
        )
    )
    unrelated += quarantine_unrelated
    action_paths = [
        *(f"raw:{receipt.path}" for _body, receipt in raw_plans),
        *(f"category-index:{target.path}" for target in index_plan),
        *(f"sealed:{tree.root}" for tree in sealed_plans),
        *(f"working:{tree.root}" for tree in working_plans),
    ]
    if not dry_run:
        for tree in (*sealed_plans, *working_plans):
            _delete_tree(tree)
        for target in index_plan:
            _delete_file(target, label="expired WDC category-index artifact")
        for body, receipt in raw_plans:
            if body is not None:
                _delete_file(body, label="expired raw WDC body")
            _delete_file(receipt, label="expired raw WDC receipt")
    return WDCRetentionReport(
        dry_run=dry_run,
        evaluated_at=timestamp.isoformat(),
        raw_receipts_scanned=raw_receipts_scanned,
        raw_pairs_eligible=len(raw_plans),
        raw_pairs_removed=0 if dry_run else len(raw_plans),
        category_index_eligible=index_eligible,
        category_index_removed=index_eligible and not dry_run,
        sealed_runs_scanned=sealed_scanned,
        sealed_runs_eligible=len(sealed_plans),
        sealed_runs_removed=0 if dry_run else len(sealed_plans),
        working_runs_scanned=working_scanned,
        working_runs_eligible=len(working_plans),
        working_runs_removed=0 if dry_run else len(working_plans),
        unrelated_entries_preserved=unrelated,
        action_sample=tuple(action_paths[:_ACTION_SAMPLE_LIMIT]),
    )


__all__ = [
    "WDCRetentionError",
    "WDCRetentionReport",
    "WDC_RETENTION_SCHEMA_VERSION",
    "maintain_wdc_research_retention",
]
