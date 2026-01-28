"""Deterministic JSONL output with optional Parquet acceleration."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pipelines.sources.base import ParseResult, sha256_file

PROCESSED_MANIFEST_SCHEMA_VERSION = "pc-build-recommender.processed-batch.v1"


@dataclass(frozen=True, slots=True)
class ProcessedArtifacts:
    output_directory: Path
    records_jsonl: Path
    rejections_jsonl: Path
    manifest_json: Path
    parquet_path: Path | None
    accepted_count: int
    rejected_count: int


def _canonical_json(record: object) -> str:
    return json.dumps(
        record,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _record_sort_key(record: dict[str, Any]) -> tuple[str, str]:
    return (
        str(record.get("record_type", "")),
        str(record.get("source_record_id", record.get("record_id", ""))),
    )


def _write_jsonl_atomic(path: Path, records: list[dict[str, Any]]) -> None:
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
            temporary_path = Path(handle.name)
            for record in sorted(records, key=_record_sort_key):
                handle.write(_canonical_json(record))
                handle.write("\n")
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _write_json_atomic(path: Path, payload: object) -> None:
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
            temporary_path = Path(handle.name)
            json.dump(
                payload,
                handle,
                allow_nan=False,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _write_optional_parquet(path: Path, records: list[dict[str, Any]]) -> Path | None:
    if importlib.util.find_spec("pyarrow") is None:
        return None
    import pyarrow as pa  # type: ignore[import-not-found]
    import pyarrow.parquet as pq  # type: ignore[import-not-found]

    table = pa.Table.from_pylist(sorted(records, key=_record_sort_key))
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        pq.write_table(table, temporary_path, compression="zstd")
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return path


def write_parsed_batch(
    batch: ParseResult,
    *,
    processed_root: str | Path,
    prefer_parquet: bool = True,
    variant: str | None = None,
) -> ProcessedArtifacts:
    """Write one batch under a snapshot-hash directory, making reruns idempotent."""

    output_directory = Path(processed_root) / batch.source_name / batch.snapshot_sha256
    if variant is not None:
        if re.fullmatch(r"[a-z0-9][a-z0-9_-]*", variant) is None:
            raise ValueError("variant must be a lowercase slug")
        output_directory /= variant
    output_directory.mkdir(parents=True, exist_ok=True)
    records_jsonl = output_directory / "records.jsonl"
    rejections_jsonl = output_directory / "rejections.jsonl"
    _write_jsonl_atomic(records_jsonl, batch.records)
    _write_jsonl_atomic(rejections_jsonl, batch.rejected)
    parquet_path = (
        _write_optional_parquet(output_directory / "records.parquet", batch.records)
        if prefer_parquet and batch.records
        else None
    )
    files = {
        "records.jsonl": {
            "sha256": sha256_file(records_jsonl),
            "byte_count": records_jsonl.stat().st_size,
        },
        "rejections.jsonl": {
            "sha256": sha256_file(rejections_jsonl),
            "byte_count": rejections_jsonl.stat().st_size,
        },
    }
    if parquet_path is not None:
        files[parquet_path.name] = {
            "sha256": sha256_file(parquet_path),
            "byte_count": parquet_path.stat().st_size,
        }
    semantic_payload = {
        "schema_version": PROCESSED_MANIFEST_SCHEMA_VERSION,
        "source_name": batch.source_name,
        "source_snapshot_sha256": batch.snapshot_sha256,
        "accepted_count": batch.accepted_count,
        "rejected_count": batch.rejected_count,
        "statistics": batch.statistics,
        "files": files,
    }
    semantic_payload["content_sha256"] = hashlib.sha256(
        _canonical_json(semantic_payload).encode("utf-8")
    ).hexdigest()
    manifest_json = output_directory / "manifest.json"
    _write_json_atomic(manifest_json, semantic_payload)
    return ProcessedArtifacts(
        output_directory=output_directory,
        records_jsonl=records_jsonl,
        rejections_jsonl=rejections_jsonl,
        manifest_json=manifest_json,
        parquet_path=parquet_path,
        accepted_count=batch.accepted_count,
        rejected_count=batch.rejected_count,
    )
