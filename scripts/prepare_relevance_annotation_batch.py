"""Compile rights-cleared, label-free retrieval candidates into blinded review tasks.

This is the collection boundary between real retrieval output and the trusted
human-annotation service.  It intentionally cannot consume frozen qrels or
silver candidates: reviewer labels, ranks, and model-derived scores are rejected
before any batch is written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pc_build_recommender.annotation import validate_blinded_annotation_payload
from pc_build_recommender.evaluation.manifest import (
    canonical_json_bytes,
    sha256_file,
    sha256_json,
)
from pc_build_recommender.evaluation.splits import DEFAULT_SPLIT_WEIGHTS, deterministic_group_split

INPUT_SCHEMA_VERSION = "pc-build-recommender.relevance-annotation-candidates.v1"
GROUP_SCHEMA_VERSION = "pc-build-recommender.annotation-group-import.v1"
PROJECT_SCHEMA_VERSION = "pc-build-recommender.annotation-project-import.v1"
CAPTURE_MANIFEST_SCHEMA_VERSION = "pc-build-recommender.relevance-annotation-capture.v2"
PRELABEL_QUERY_SCHEMA_VERSION = "pc-build-recommender.ranking-prelabel-query.v1"
PRELABEL_SNAPSHOT_SCHEMA_VERSION = (
    "pc-build-recommender.ranking-prelabel-snapshot-manifest.v1"
)
MANIFEST_SCHEMA_VERSION = "pc-build-recommender.relevance-annotation-batch.v2"
COMPILER_VERSION = "relevance-annotation-batch-compiler.v2"
DEFAULT_SEED = 20260723
MAX_INPUT_BYTES = 64 * 1024 * 1024
MAX_PRELABEL_LINE_BYTES = 8 * 1024 * 1024


class RelevanceAnnotationBatchInputError(ValueError):
    """Raised when a collection input would not make a safe review batch."""


@dataclass(frozen=True, slots=True)
class _Candidate:
    product_id: str
    evidence_payload: dict[str, Any]


@dataclass(frozen=True, slots=True)
class _Query:
    query_id: str
    query_group_id: str
    category: str
    query_text: str
    structured_constraints: dict[str, Any]
    candidates: tuple[_Candidate, ...]


@dataclass(frozen=True, slots=True)
class _PrelabelBinding:
    manifest_file_sha256: str
    snapshot_sha256: str
    candidate_universe_sha256: str
    feature_contract_sha256: str
    query_row_sha256: Mapping[str, str]
    candidate_ids_sha256: Mapping[str, str]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Label-free candidate JSON.")
    parser.add_argument(
        "--capture-manifest",
        type=Path,
        required=True,
        help="Version-2 capture manifest that binds the pre-label feature snapshot.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New immutable batch directory; an existing path is always rejected.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Deterministic query-group split seed.",
    )
    return parser


def _object(value: object, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RelevanceAnnotationBatchInputError(f"{name} must be an object")
    return {str(key): nested for key, nested in value.items()}


def _string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RelevanceAnnotationBatchInputError(f"{name} must be a non-empty string")
    result = value.strip()
    if result.casefold().startswith("replace"):
        raise RelevanceAnnotationBatchInputError(
            f"{name} contains a template placeholder; replace it with real collection evidence"
        )
    return result


def _false(value: object, *, name: str) -> None:
    if value is not False:
        raise RelevanceAnnotationBatchInputError(
            f"{name} must be explicitly false; synthetic collection evidence is not allowed"
        )


def _only_keys(payload: Mapping[str, Any], *, allowed: set[str], name: str) -> None:
    unexpected = sorted(set(payload) - allowed)
    if unexpected:
        raise RelevanceAnnotationBatchInputError(
            f"{name} contains unsupported fields: {', '.join(unexpected)}"
        )
    missing = sorted(allowed - set(payload))
    if missing:
        raise RelevanceAnnotationBatchInputError(
            f"{name} is missing required fields: {', '.join(missing)}"
        )


def _json_object(value: object, *, name: str) -> dict[str, Any]:
    payload = _object(value, name=name)
    try:
        return _object(json.loads(canonical_json_bytes(payload)), name=name)
    except (TypeError, ValueError) as error:
        raise RelevanceAnnotationBatchInputError(f"{name} must contain finite JSON data") from error


def _source_policy(value: object) -> dict[str, Any]:
    policy = _json_object(value, name="source_policy")
    for field in ("training_eligible", "published_metrics_eligible"):
        if policy.get(field) is not True:
            raise RelevanceAnnotationBatchInputError(
                f"source_policy.{field} must be true for a trainable relevance collection"
            )
    serving = policy.get("model_serving_eligible", False)
    if not isinstance(serving, bool):
        raise RelevanceAnnotationBatchInputError(
            "source_policy.model_serving_eligible must be a boolean when provided"
        )
    if serving and (
        not isinstance(policy.get("serving_attribution_notice"), str)
        or not policy["serving_attribution_notice"].strip()
    ):
        raise RelevanceAnnotationBatchInputError(
            "source_policy.serving_attribution_notice must be a non-empty string "
            "when derived-model serving is eligible"
        )
    if not isinstance(policy.get("scope_note"), str) or not policy["scope_note"].strip():
        raise RelevanceAnnotationBatchInputError(
            "source_policy.scope_note must be a non-empty string"
        )
    policy["model_serving_eligible"] = serving
    return policy


def _provenance(value: object, *, name: str) -> dict[str, str]:
    payload = _object(value, name=name)
    _only_keys(
        payload,
        allowed={"source_name", "source_url", "license_or_access_note", "retrieved_at"},
        name=name,
    )
    source_url = _string(payload["source_url"], name=f"{name}.source_url")
    if not source_url.startswith("https://"):
        raise RelevanceAnnotationBatchInputError(f"{name}.source_url must use https")
    if "replace-with" in source_url.casefold():
        raise RelevanceAnnotationBatchInputError(
            f"{name}.source_url contains a template placeholder; use the actual evidence URL"
        )
    return {
        "source_name": _string(payload["source_name"], name=f"{name}.source_name"),
        "source_url": source_url,
        "license_or_access_note": _string(
            payload["license_or_access_note"], name=f"{name}.license_or_access_note"
        ),
        "retrieved_at": _string(payload["retrieved_at"], name=f"{name}.retrieved_at"),
    }


def _candidate(value: object, *, query_id: str, index: int) -> _Candidate:
    name = f"queries[{query_id!r}].candidates[{index}]"
    payload = _object(value, name=name)
    _only_keys(
        payload,
        allowed={"product_id", "evidence_payload", "provenance", "is_synthetic"},
        name=name,
    )
    _false(payload["is_synthetic"], name=f"{name}.is_synthetic")
    evidence = _json_object(payload["evidence_payload"], name=f"{name}.evidence_payload")
    if not evidence:
        raise RelevanceAnnotationBatchInputError(f"{name}.evidence_payload must not be empty")
    if "provenance" in evidence:
        raise RelevanceAnnotationBatchInputError(
            f"{name}.evidence_payload.provenance is not allowed; use {name}.provenance"
        )
    try:
        validate_blinded_annotation_payload(evidence, path=f"{name}.evidence_payload")
    except ValueError as error:
        raise RelevanceAnnotationBatchInputError(str(error)) from error
    evidence["provenance"] = _provenance(payload["provenance"], name=f"{name}.provenance")
    return _Candidate(
        product_id=_string(payload["product_id"], name=f"{name}.product_id"),
        evidence_payload=evidence,
    )


def _query(value: object, *, index: int) -> _Query:
    name = f"queries[{index}]"
    payload = _object(value, name=name)
    _only_keys(
        payload,
        allowed={
            "query_id",
            "query_group_id",
            "category",
            "query_text",
            "structured_constraints",
            "candidates",
            "is_synthetic",
        },
        name=name,
    )
    _false(payload["is_synthetic"], name=f"{name}.is_synthetic")
    query_id = _string(payload["query_id"], name=f"{name}.query_id")
    constraints = _json_object(
        payload["structured_constraints"], name=f"{name}.structured_constraints"
    )
    try:
        validate_blinded_annotation_payload(constraints, path=f"{name}.structured_constraints")
    except ValueError as error:
        raise RelevanceAnnotationBatchInputError(str(error)) from error
    raw_candidates = payload["candidates"]
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise RelevanceAnnotationBatchInputError(f"{name}.candidates must be a non-empty array")
    candidates = tuple(
        _candidate(candidate, query_id=query_id, index=candidate_index)
        for candidate_index, candidate in enumerate(raw_candidates)
    )
    product_ids = [candidate.product_id for candidate in candidates]
    if len(product_ids) != len(set(product_ids)):
        raise RelevanceAnnotationBatchInputError(
            f"{name}.candidates product_id values must be unique"
        )
    return _Query(
        query_id=query_id,
        query_group_id=_string(payload["query_group_id"], name=f"{name}.query_group_id"),
        category=_string(payload["category"], name=f"{name}.category").casefold(),
        query_text=_string(payload["query_text"], name=f"{name}.query_text"),
        structured_constraints=constraints,
        candidates=tuple(sorted(candidates, key=lambda candidate: candidate.product_id)),
    )


def _load_input(path: Path) -> tuple[dict[str, Any], tuple[_Query, ...], dict[str, Any]]:
    resolved = path.resolve(strict=True)
    if resolved.stat().st_size > MAX_INPUT_BYTES:
        raise RelevanceAnnotationBatchInputError(
            f"input exceeds the {MAX_INPUT_BYTES} byte safety limit"
        )
    try:
        raw = json.loads(resolved.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise RelevanceAnnotationBatchInputError(f"input is not valid JSON: {error.msg}") from error
    payload = _object(raw, name="input")
    _only_keys(
        payload,
        allowed={
            "schema_version",
            "dataset_name",
            "dataset_version",
            "rubric_version",
            "data_version",
            "source_policy",
            "queries",
        },
        name="input",
    )
    if payload["schema_version"] != INPUT_SCHEMA_VERSION:
        raise RelevanceAnnotationBatchInputError(
            f"input.schema_version must be {INPUT_SCHEMA_VERSION!r}"
        )
    raw_queries = payload["queries"]
    if not isinstance(raw_queries, list) or not raw_queries:
        raise RelevanceAnnotationBatchInputError("input.queries must be a non-empty array")
    metadata = {
        "dataset_name": _string(payload["dataset_name"], name="input.dataset_name"),
        "dataset_version": _string(payload["dataset_version"], name="input.dataset_version"),
        "rubric_version": _string(payload["rubric_version"], name="input.rubric_version"),
        "data_version": _string(payload["data_version"], name="input.data_version"),
        "source_policy": _source_policy(payload["source_policy"]),
    }
    queries = tuple(_query(query, index=index) for index, query in enumerate(raw_queries))
    query_ids = [query.query_id for query in queries]
    if len(query_ids) != len(set(query_ids)):
        raise RelevanceAnnotationBatchInputError("input query_id values must be unique")
    if len({query.query_group_id for query in queries}) < len(DEFAULT_SPLIT_WEIGHTS):
        raise RelevanceAnnotationBatchInputError(
            "at least three distinct query_group_id values are required to cover "
            "train, validation, and test"
        )
    return metadata, tuple(sorted(queries, key=lambda query: query.query_id)), payload


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _digest(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RelevanceAnnotationBatchInputError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _load_prelabel_binding(
    *,
    input_path: Path,
    manifest_path: Path,
    queries: Sequence[_Query],
) -> _PrelabelBinding:
    resolved_manifest = manifest_path.resolve(strict=True)
    try:
        manifest = _object(
            json.loads(resolved_manifest.read_text(encoding="utf-8")),
            name="capture manifest",
        )
    except json.JSONDecodeError as error:
        raise RelevanceAnnotationBatchInputError(
            f"capture manifest is not valid JSON: {error.msg}"
        ) from error
    if manifest.get("schema_version") != CAPTURE_MANIFEST_SCHEMA_VERSION:
        raise RelevanceAnnotationBatchInputError(
            f"capture manifest schema_version must be {CAPTURE_MANIFEST_SCHEMA_VERSION!r}"
        )
    expected_candidates_sha256 = _digest(
        manifest.get("candidate_file_sha256"),
        name="capture manifest candidate_file_sha256",
    )
    if sha256_file(input_path) != expected_candidates_sha256:
        raise RelevanceAnnotationBatchInputError(
            "candidate input SHA-256 does not match the capture manifest"
        )

    snapshot = _object(
        manifest.get("prelabel_ranking_snapshot"),
        name="capture manifest prelabel_ranking_snapshot",
    )
    if snapshot.get("schema_version") != PRELABEL_SNAPSHOT_SCHEMA_VERSION:
        raise RelevanceAnnotationBatchInputError(
            f"pre-label snapshot schema_version must be {PRELABEL_SNAPSHOT_SCHEMA_VERSION!r}"
        )
    if snapshot.get("label_state") != "absent":
        raise RelevanceAnnotationBatchInputError("pre-label snapshot label_state must be absent")
    feature_contract = _object(
        snapshot.get("feature_contract"),
        name="pre-label feature contract",
    )
    if feature_contract.get("contains_relevance_labels") is not False:
        raise RelevanceAnnotationBatchInputError(
            "pre-label feature contract must explicitly exclude relevance labels"
        )
    if feature_contract.get("label_free_by_construction") is not True:
        raise RelevanceAnnotationBatchInputError(
            "pre-label feature contract must be label-free by construction"
        )
    expected_feature_contract_sha256 = _digest(
        snapshot.get("feature_contract_sha256"),
        name="pre-label feature_contract_sha256",
    )
    if sha256_json(feature_contract) != expected_feature_contract_sha256:
        raise RelevanceAnnotationBatchInputError("pre-label feature contract hash mismatch")
    snapshot_sha256 = _digest(
        snapshot.get("snapshot_sha256"),
        name="pre-label snapshot_sha256",
    )
    snapshot_without_hash = dict(snapshot)
    snapshot_without_hash.pop("snapshot_sha256")
    if sha256_json(snapshot_without_hash) != snapshot_sha256:
        raise RelevanceAnnotationBatchInputError("pre-label snapshot self-hash mismatch")

    feature_file_name = snapshot.get("file_name")
    if (
        not isinstance(feature_file_name, str)
        or not feature_file_name
        or Path(feature_file_name).name != feature_file_name
    ):
        raise RelevanceAnnotationBatchInputError("pre-label feature file name is unsafe")
    feature_path = resolved_manifest.parent / feature_file_name
    if not feature_path.is_file():
        raise RelevanceAnnotationBatchInputError("pre-label feature file is missing")
    if feature_path.stat().st_size != snapshot.get("size_bytes"):
        raise RelevanceAnnotationBatchInputError("pre-label feature file size mismatch")
    expected_feature_sha256 = _digest(
        snapshot.get("file_sha256"),
        name="pre-label feature file_sha256",
    )
    if sha256_file(feature_path) != expected_feature_sha256:
        raise RelevanceAnnotationBatchInputError("pre-label feature file SHA-256 mismatch")

    expected_row_hashes = _object(
        snapshot.get("query_row_sha256"),
        name="pre-label query_row_sha256",
    )
    rows: dict[str, dict[str, Any]] = {}
    with feature_path.open("rb") as stream:
        for line_number, line in enumerate(stream, start=1):
            if len(line) > MAX_PRELABEL_LINE_BYTES:
                raise RelevanceAnnotationBatchInputError(
                    f"pre-label feature line {line_number} exceeds the safety limit"
                )
            if not line.strip():
                continue
            try:
                row = _object(json.loads(line), name=f"pre-label feature line {line_number}")
            except json.JSONDecodeError as error:
                raise RelevanceAnnotationBatchInputError(
                    f"pre-label feature line {line_number} is invalid JSON"
                ) from error
            if row.get("schema_version") != PRELABEL_QUERY_SCHEMA_VERSION:
                raise RelevanceAnnotationBatchInputError(
                    f"pre-label feature line {line_number} has an unsupported schema"
                )
            query_id = _string(row.get("query_id"), name=f"pre-label line {line_number} query_id")
            if query_id in rows:
                raise RelevanceAnnotationBatchInputError(
                    f"pre-label feature query_id is duplicated: {query_id!r}"
                )
            if any(
                forbidden in json.dumps(row, sort_keys=True).casefold()
                for forbidden in (
                    '"relevance_grade"',
                    '"reviewer_id"',
                    '"adjudication"',
                    '"human_label"',
                )
            ):
                raise RelevanceAnnotationBatchInputError(
                    f"pre-label feature row {query_id!r} contains label or reviewer data"
                )
            expected_row_sha256 = _digest(
                expected_row_hashes.get(query_id),
                name=f"pre-label query_row_sha256[{query_id!r}]",
            )
            if sha256_json(row) != expected_row_sha256:
                raise RelevanceAnnotationBatchInputError(
                    f"pre-label feature row hash mismatch for {query_id!r}"
                )
            rows[query_id] = row

    expected_queries = {query.query_id: query for query in queries}
    if set(rows) != set(expected_queries):
        raise RelevanceAnnotationBatchInputError(
            "pre-label feature query IDs do not match the annotation candidates"
        )
    candidate_hashes: dict[str, str] = {}
    for query_id, query in expected_queries.items():
        row = rows[query_id]
        if (
            row.get("query_group_id") != query.query_group_id
            or str(row.get("category", "")).casefold() != query.category
        ):
            raise RelevanceAnnotationBatchInputError(
                f"pre-label query identity differs for {query_id!r}"
            )
        raw_candidates = row.get("candidates")
        if not isinstance(raw_candidates, list):
            raise RelevanceAnnotationBatchInputError(
                f"pre-label candidates must be an array for {query_id!r}"
            )
        feature_ids = [
            _string(candidate.get("product_id"), name=f"pre-label {query_id!r} product_id")
            for candidate in raw_candidates
            if isinstance(candidate, Mapping)
        ]
        expected_ids = [candidate.product_id for candidate in query.candidates]
        if sorted(feature_ids) != sorted(expected_ids) or len(feature_ids) != len(raw_candidates):
            raise RelevanceAnnotationBatchInputError(
                f"pre-label candidate universe differs for {query_id!r}"
            )
        candidate_digest = _digest(
            row.get("candidate_ids_sha256"),
            name=f"pre-label candidate_ids_sha256[{query_id!r}]",
        )
        if sha256_json(sorted(feature_ids)) != candidate_digest:
            raise RelevanceAnnotationBatchInputError(
                f"pre-label candidate hash mismatch for {query_id!r}"
            )
        candidate_hashes[query_id] = candidate_digest

    return _PrelabelBinding(
        manifest_file_sha256=sha256_file(resolved_manifest),
        snapshot_sha256=snapshot_sha256,
        candidate_universe_sha256=_digest(
            snapshot.get("candidate_universe_sha256"),
            name="pre-label candidate_universe_sha256",
        ),
        feature_contract_sha256=expected_feature_contract_sha256,
        query_row_sha256={
            query_id: _digest(
                expected_row_hashes[query_id],
                name=f"pre-label query_row_sha256[{query_id!r}]",
            )
            for query_id in sorted(expected_queries)
        },
        candidate_ids_sha256=candidate_hashes,
    )


def _group_record(
    query: _Query,
    *,
    split_name: str,
    prelabel_binding: _PrelabelBinding,
) -> dict[str, Any]:
    return {
        "schema_version": GROUP_SCHEMA_VERSION,
        "group_key": query.query_id,
        "leakage_group_id": query.query_group_id,
        "category": query.category,
        "split_name": split_name,
        "context_payload": {
            "query_text": query.query_text,
            "structured_constraints": query.structured_constraints,
            "ranking_prelabel_binding": {
                "snapshot_sha256": prelabel_binding.snapshot_sha256,
                "query_row_sha256": prelabel_binding.query_row_sha256[query.query_id],
                "candidate_ids_sha256": prelabel_binding.candidate_ids_sha256[query.query_id],
            },
        },
        "is_synthetic": False,
        "items": [
            {
                "target_id": candidate.product_id,
                "evidence_payload": candidate.evidence_payload,
                "priority": 0,
                "is_synthetic": False,
            }
            for candidate in query.candidates
        ],
    }


def _write_batch(
    *,
    destination: Path,
    project_spec: Mapping[str, Any],
    groups: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> None:
    output = destination.resolve()
    if output.exists():
        raise RelevanceAnnotationBatchInputError(
            f"output directory already exists and will not be overwritten: {output}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        groups_bytes = b"".join(canonical_json_bytes(group) + b"\n" for group in groups)
        (temporary / "groups.jsonl").write_bytes(groups_bytes)
        (temporary / "project-spec.json").write_bytes(
            canonical_json_bytes(project_spec) + b"\n"
        )
        (temporary / "manifest.json").write_bytes(canonical_json_bytes(manifest) + b"\n")
        os.replace(temporary, output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def compile_relevance_annotation_batch(
    *,
    input_path: Path,
    capture_manifest_path: Path,
    output_dir: Path,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Create an immutable, blinded JSONL batch and its import project specification."""

    if isinstance(seed, bool):
        raise RelevanceAnnotationBatchInputError("seed must be an integer")
    metadata, queries, source_input = _load_input(input_path)
    prelabel_binding = _load_prelabel_binding(
        input_path=input_path,
        manifest_path=capture_manifest_path,
        queries=queries,
    )
    split = deterministic_group_split(
        [query.query_group_id for query in queries], weights=DEFAULT_SPLIT_WEIGHTS, seed=seed
    )
    assignments = {query.query_id: split.split_for(query.query_group_id) for query in queries}
    groups = [
        _group_record(
            query,
            split_name=assignments[query.query_id],
            prelabel_binding=prelabel_binding,
        )
        for query in queries
    ]
    project_spec = {
        "schema_version": PROJECT_SCHEMA_VERSION,
        "task_type": "relevance",
        **metadata,
    }
    groups_bytes = b"".join(canonical_json_bytes(group) + b"\n" for group in groups)
    split_payload = {
        "seed": seed,
        "weights": dict(DEFAULT_SPLIT_WEIGHTS),
        "query_group_ids": {query.query_id: query.query_group_id for query in queries},
        "assignments": assignments,
    }
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "compiler_version": COMPILER_VERSION,
        "input_sha256": sha256_json(source_input),
        "project_spec_sha256": sha256_json(project_spec),
        "source_policy_sha256": sha256_json(metadata["source_policy"]),
        "ranking_prelabel_binding": {
            "capture_manifest_file_sha256": prelabel_binding.manifest_file_sha256,
            "snapshot_sha256": prelabel_binding.snapshot_sha256,
            "candidate_universe_sha256": prelabel_binding.candidate_universe_sha256,
            "feature_contract_sha256": prelabel_binding.feature_contract_sha256,
            "query_row_sha256": dict(prelabel_binding.query_row_sha256),
        },
        "query_group_split_sha256": sha256_json(split_payload),
        "query_count": len(queries),
        "query_group_count": len(split.assignments),
        "item_count": sum(len(query.candidates) for query in queries),
        "split_counts": split.group_counts(),
        "files": {
            "groups.jsonl": _sha256_bytes(groups_bytes),
            "project-spec.json": _sha256_bytes(canonical_json_bytes(project_spec) + b"\n"),
        },
    }
    _write_batch(
        destination=output_dir,
        project_spec=project_spec,
        groups=groups,
        manifest=manifest,
    )
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "status": "ok",
        "output_dir": str(output_dir.resolve()),
        "manifest_sha256": sha256_json(manifest),
        "query_count": manifest["query_count"],
        "item_count": manifest["item_count"],
        "split_counts": manifest["split_counts"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = compile_relevance_annotation_batch(
            input_path=args.input,
            capture_manifest_path=args.capture_manifest,
            output_dir=args.output_dir,
            seed=args.seed,
        )
    except (OSError, RelevanceAnnotationBatchInputError, ValueError) as error:
        print(
            json.dumps({"status": "error", "message": str(error)}, sort_keys=True),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
