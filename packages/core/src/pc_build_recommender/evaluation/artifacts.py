"""Strict JSON artifacts for auditable model-evaluation runs."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .contracts import EvaluationResult
from .manifest import sha256_json

ARTIFACT_SCHEMA_VERSION = "pc-build-recommender.evaluation-artifact.v1"


def _valid_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def build_evaluation_artifact(
    *,
    task: str,
    run_id: str,
    dataset_manifest_sha256: str,
    result: EvaluationResult,
    metadata: Mapping[str, object] | None = None,
    created_at: datetime | None = None,
) -> dict[str, object]:
    """Build an artifact and make reportability impossible to overlook."""

    if not task or not run_id:
        raise ValueError("task and run_id must not be empty")
    if not _valid_sha256(dataset_manifest_sha256):
        raise ValueError("dataset_manifest_sha256 must be a lowercase SHA-256 digest")
    timestamp = created_at or datetime.now(UTC)
    if timestamp.tzinfo is None:
        raise ValueError("created_at must be timezone-aware")
    payload: dict[str, object] = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "task": task,
        "run_id": run_id,
        "created_at": timestamp.astimezone(UTC).isoformat().replace("+00:00", "Z"),
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "eligible_for_reported_metrics": result.data_use.eligible_for_reported_metrics,
        "reporting_block_reason": result.data_use.reporting_block_reason,
        "synthetic_data": result.data_use.to_dict(),
        "metrics": [metric.to_dict() for metric in result.metrics],
        "evaluation_metadata": result.metadata,
        "run_metadata": dict(metadata or {}),
    }
    payload["artifact_sha256"] = sha256_json(payload)
    return payload


def verify_evaluation_artifact(payload: Mapping[str, object]) -> bool:
    """Verify the artifact hash without trusting the stored metric content."""

    stored_hash = payload.get("artifact_sha256")
    if not isinstance(stored_hash, str) or not _valid_sha256(stored_hash):
        return False
    unhashed = dict(payload)
    del unhashed["artifact_sha256"]
    try:
        return sha256_json(unhashed) == stored_hash
    except (TypeError, ValueError):
        return False


def write_evaluation_artifact(payload: Mapping[str, object], path: str | Path) -> Path:
    """Atomically write a verified artifact as strict JSON."""

    if not verify_evaluation_artifact(payload):
        raise ValueError("evaluation artifact hash is missing or invalid")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    serialised = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(serialised)
            handle.write("\n")
            temporary_name = handle.name
        os.replace(temporary_name, target)
    finally:
        if temporary_name is not None:
            temporary_path = Path(temporary_name)
            if temporary_path.exists():
                temporary_path.unlink()
    return target


def load_evaluation_artifact(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError("artifact root must be an object")
    if not verify_evaluation_artifact(payload):
        raise ValueError("artifact hash verification failed")
    return payload
