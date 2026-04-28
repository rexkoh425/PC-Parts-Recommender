"""Join a pre-label feature commitment with a frozen human annotation release.

The command never computes ranking features.  It verifies the immutable feature
snapshot captured before annotation, verifies the complete annotation release,
and copies every pre-label field unchanged while appending only adjudicated
``relevance_grade`` values.  Human LambdaMART training consumes the resulting
manifest-bound JSONL rather than a manually authored feature file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pc_build_recommender.evaluation.manifest import (
    canonical_json_bytes,
    sha256_file,
    sha256_json,
)
from pc_build_recommender.ranking import (
    RankingCandidate,
    RankingContext,
    RankingFeatureBuilder,
)
from pc_build_recommender.retrieval import (
    FrozenCandidateSet,
    QueryGroupSplit,
    HumanJudgmentSet,
    load_human_judgment_set,
)

CAPTURE_MANIFEST_SCHEMA_VERSION = "pc-build-recommender.relevance-annotation-capture.v2"
PRELABEL_QUERY_SCHEMA_VERSION = "pc-build-recommender.ranking-prelabel-query.v1"
PRELABEL_SNAPSHOT_SCHEMA_VERSION = (
    "pc-build-recommender.ranking-prelabel-snapshot-manifest.v1"
)
ANNOTATION_RELEASE_SCHEMA_VERSION = "pc-build-recommender.annotation-release.v1"
EVIDENCE_SNAPSHOT_SCHEMA_VERSION = (
    "pc-build-recommender.annotation-evidence-snapshots.v1"
)
LABELED_SNAPSHOT_MANIFEST_SCHEMA_VERSION = (
    "pc-build-recommender.ranking-labeled-snapshot-manifest.v1"
)
RANKING_DATA_FILENAME = "ranking.jsonl"
MAX_MANIFEST_BYTES = 8 * 1024 * 1024
MAX_FEATURE_BYTES = 256 * 1024 * 1024
MAX_FEATURE_LINE_BYTES = 8 * 1024 * 1024
_FORBIDDEN_PRELABEL_KEYS = frozenset(
    {
        "adjudication",
        "grade",
        "human_label",
        "judgment",
        "judgments",
        "label",
        "relevance_grade",
        "relevance_label",
        "reviewer",
        "reviewer_id",
    }
)
_RELEVANCE_RELEASE_FILES = frozenset(
    {
        "evidence-snapshots.json",
        "human-judgments.json",
        "qrels.json",
        "query-split.json",
    }
)


class RankingSnapshotMaterializationError(ValueError):
    """Raised when pre-label or annotation lineage cannot be proven."""


@dataclass(frozen=True, slots=True)
class VerifiedLabeledRankingSnapshot:
    manifest_path: Path
    manifest_file_sha256: str
    manifest_sha256: str
    ranking_file_sha256: str
    prelabel_snapshot_sha256: str
    feature_contract_sha256: str
    annotation_release_sha256: str


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--annotation-release-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _object(value: object, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RankingSnapshotMaterializationError(f"{name} must be an object")
    return {str(key): nested for key, nested in value.items()}


def _string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RankingSnapshotMaterializationError(f"{name} must be a non-empty string")
    return value.strip()


def _digest(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RankingSnapshotMaterializationError(
            f"{name} must be a lowercase SHA-256 digest"
        )
    return value


def _load_json_object(path: Path, *, name: str) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    if resolved.stat().st_size > MAX_MANIFEST_BYTES:
        raise RankingSnapshotMaterializationError(f"{name} exceeds the safety limit")
    try:
        return _object(json.loads(resolved.read_text(encoding="utf-8")), name=name)
    except json.JSONDecodeError as error:
        raise RankingSnapshotMaterializationError(
            f"{name} is not valid JSON: {error.msg}"
        ) from error


def _assert_prelabel_value(value: object, *, path: str) -> None:
    if isinstance(value, Mapping):
        for raw_key, nested in value.items():
            key = str(raw_key).strip().casefold()
            if key in _FORBIDDEN_PRELABEL_KEYS:
                raise RankingSnapshotMaterializationError(
                    f"{path}.{raw_key} contains post-label evidence"
                )
            _assert_prelabel_value(nested, path=f"{path}.{raw_key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _assert_prelabel_value(nested, path=f"{path}[{index}]")


def _load_prelabel_rows(
    capture_dir: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    tuple[dict[str, Any], ...],
    str,
]:
    root = capture_dir.resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(root)
    manifest_path = root / "manifest.json"
    manifest = _load_json_object(manifest_path, name="capture manifest")
    if manifest.get("schema_version") != CAPTURE_MANIFEST_SCHEMA_VERSION:
        raise RankingSnapshotMaterializationError(
            f"capture manifest schema_version must be {CAPTURE_MANIFEST_SCHEMA_VERSION!r}"
        )
    snapshot = _object(
        manifest.get("prelabel_ranking_snapshot"),
        name="capture pre-label snapshot",
    )
    if snapshot.get("schema_version") != PRELABEL_SNAPSHOT_SCHEMA_VERSION:
        raise RankingSnapshotMaterializationError(
            f"pre-label schema_version must be {PRELABEL_SNAPSHOT_SCHEMA_VERSION!r}"
        )
    if snapshot.get("label_state") != "absent":
        raise RankingSnapshotMaterializationError("pre-label label_state must be absent")
    feature_contract = _object(
        snapshot.get("feature_contract"),
        name="pre-label feature contract",
    )
    if (
        feature_contract.get("contains_relevance_labels") is not False
        or feature_contract.get("label_free_by_construction") is not True
    ):
        raise RankingSnapshotMaterializationError(
            "pre-label feature contract does not prove label independence"
        )
    feature_builder = RankingFeatureBuilder()
    if (
        feature_contract.get("feature_version") != feature_builder.feature_version
        or feature_contract.get("feature_names") != list(feature_builder.feature_names)
    ):
        raise RankingSnapshotMaterializationError(
            "pre-label feature contract does not match the current feature builder"
        )
    if sha256_json(feature_contract) != _digest(
        snapshot.get("feature_contract_sha256"),
        name="pre-label feature_contract_sha256",
    ):
        raise RankingSnapshotMaterializationError("pre-label feature contract hash mismatch")
    snapshot_sha256 = _digest(
        snapshot.get("snapshot_sha256"),
        name="pre-label snapshot_sha256",
    )
    semantic_snapshot = dict(snapshot)
    semantic_snapshot.pop("snapshot_sha256")
    if sha256_json(semantic_snapshot) != snapshot_sha256:
        raise RankingSnapshotMaterializationError("pre-label snapshot self-hash mismatch")

    feature_name = snapshot.get("file_name")
    if (
        not isinstance(feature_name, str)
        or Path(feature_name).name != feature_name
        or not feature_name
    ):
        raise RankingSnapshotMaterializationError("pre-label feature file name is unsafe")
    feature_path = root / feature_name
    if not feature_path.is_file():
        raise RankingSnapshotMaterializationError("pre-label feature file is missing")
    if feature_path.stat().st_size > MAX_FEATURE_BYTES:
        raise RankingSnapshotMaterializationError("pre-label feature file exceeds safety limit")
    if feature_path.stat().st_size != snapshot.get("size_bytes"):
        raise RankingSnapshotMaterializationError("pre-label feature file size mismatch")
    if sha256_file(feature_path) != _digest(
        snapshot.get("file_sha256"),
        name="pre-label file_sha256",
    ):
        raise RankingSnapshotMaterializationError("pre-label feature file hash mismatch")
    expected_row_hashes = _object(
        snapshot.get("query_row_sha256"),
        name="pre-label query row hashes",
    )
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    with feature_path.open("rb") as stream:
        for line_number, line in enumerate(stream, start=1):
            if len(line) > MAX_FEATURE_LINE_BYTES:
                raise RankingSnapshotMaterializationError(
                    f"pre-label feature line {line_number} exceeds safety limit"
                )
            if not line.strip():
                continue
            try:
                row = _object(json.loads(line), name=f"pre-label feature line {line_number}")
            except json.JSONDecodeError as error:
                raise RankingSnapshotMaterializationError(
                    f"pre-label feature line {line_number} is invalid JSON"
                ) from error
            if row.get("schema_version") != PRELABEL_QUERY_SCHEMA_VERSION:
                raise RankingSnapshotMaterializationError(
                    f"pre-label feature line {line_number} has an unsupported schema"
                )
            _assert_prelabel_value(row, path=f"pre-label feature line {line_number}")
            query_id = _string(row.get("query_id"), name=f"pre-label line {line_number} query_id")
            if query_id in seen:
                raise RankingSnapshotMaterializationError(
                    f"duplicate pre-label query_id: {query_id!r}"
                )
            seen.add(query_id)
            if sha256_json(row) != _digest(
                expected_row_hashes.get(query_id),
                name=f"pre-label row hash for {query_id!r}",
            ):
                raise RankingSnapshotMaterializationError(
                    f"pre-label row hash mismatch for {query_id!r}"
                )
            raw_candidates = row.get("candidates")
            if not isinstance(raw_candidates, list) or not raw_candidates:
                raise RankingSnapshotMaterializationError(
                    f"pre-label candidates are missing for {query_id!r}"
                )
            candidate_payloads = [
                _object(candidate, name=f"pre-label candidate {index}")
                for index, candidate in enumerate(raw_candidates)
            ]
            product_ids = [
                _string(
                    candidate.get("product_id"),
                    name=f"pre-label candidate {index} product_id",
                )
                for index, candidate in enumerate(candidate_payloads)
            ]
            if len(product_ids) != len(set(product_ids)):
                raise RankingSnapshotMaterializationError(
                    f"pre-label candidates are duplicated for {query_id!r}"
                )
            if sha256_json(product_ids) != _digest(
                row.get("candidate_ids_sha256"),
                name=f"pre-label candidate hash for {query_id!r}",
            ):
                raise RankingSnapshotMaterializationError(
                    f"pre-label candidate hash mismatch for {query_id!r}"
                )
            context_payload = _object(
                row.get("context"),
                name=f"pre-label context for {query_id!r}",
            )
            try:
                context = RankingContext(**context_payload)
                ranking_candidates = tuple(
                    RankingCandidate(**candidate) for candidate in candidate_payloads
                )
                feature_batch = feature_builder.build(context, ranking_candidates)
            except (TypeError, ValueError) as error:
                raise RankingSnapshotMaterializationError(
                    f"pre-label ranking inputs are invalid for {query_id!r}: {error}"
                ) from error
            matrix_payload = {
                "feature_version": feature_builder.feature_version,
                "feature_names": list(feature_builder.feature_names),
                "query_id": query_id,
                "rows": [
                    {
                        "product_id": product_id,
                        "values_hex": [
                            float(value).hex()
                            for value in feature_batch.values[index]
                        ],
                    }
                    for index, product_id in enumerate(feature_batch.product_ids)
                ],
            }
            if sha256_json(matrix_payload) != _digest(
                row.get("feature_matrix_sha256"),
                name=f"pre-label feature matrix hash for {query_id!r}",
            ):
                raise RankingSnapshotMaterializationError(
                    f"pre-label feature matrix changed for {query_id!r}"
                )
            rows.append(row)
    if set(expected_row_hashes) != seen:
        raise RankingSnapshotMaterializationError(
            "pre-label row hash keys do not match feature queries"
        )
    candidate_universe = [
        {
            "query_id": row["query_id"],
            "query_group_id": row["query_group_id"],
            "category": row["category"],
            "candidate_ids_sha256": row["candidate_ids_sha256"],
        }
        for row in sorted(rows, key=lambda item: str(item["query_id"]))
    ]
    if sha256_json(candidate_universe) != _digest(
        snapshot.get("candidate_universe_sha256"),
        name="pre-label candidate_universe_sha256",
    ):
        raise RankingSnapshotMaterializationError(
            "pre-label candidate-universe hash mismatch"
        )
    return manifest, snapshot, tuple(rows), sha256_file(manifest_path)


def _verify_annotation_release(
    release_dir: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    HumanJudgmentSet,
    FrozenCandidateSet,
    QueryGroupSplit,
    str,
]:
    root = release_dir.resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(root)
    manifest_path = root / "manifest.json"
    manifest = _load_json_object(manifest_path, name="annotation release manifest")
    if manifest.get("schema_version") != ANNOTATION_RELEASE_SCHEMA_VERSION:
        raise RankingSnapshotMaterializationError(
            f"annotation release schema_version must be {ANNOTATION_RELEASE_SCHEMA_VERSION!r}"
        )
    if manifest.get("task_type") != "relevance":
        raise RankingSnapshotMaterializationError(
            "annotation release must contain relevance labels"
        )
    release_sha256 = _digest(
        manifest.get("release_sha256"),
        name="annotation release_sha256",
    )
    identity = dict(manifest)
    identity.pop("release_sha256")
    if sha256_json(identity) != release_sha256:
        raise RankingSnapshotMaterializationError("annotation release self-hash mismatch")
    files = _object(manifest.get("files"), name="annotation release files")
    if set(files) != _RELEVANCE_RELEASE_FILES:
        raise RankingSnapshotMaterializationError(
            "annotation release files do not match the relevance release contract"
        )
    for name, raw_evidence in files.items():
        evidence = _object(raw_evidence, name=f"annotation release file {name}")
        path = root / name
        if not path.is_file():
            raise RankingSnapshotMaterializationError(
                f"annotation release file is missing: {name}"
            )
        if path.stat().st_size != evidence.get("size_bytes"):
            raise RankingSnapshotMaterializationError(
                f"annotation release file size mismatch: {name}"
            )
        if sha256_file(path) != _digest(
            evidence.get("sha256"),
            name=f"annotation release {name} sha256",
        ):
            raise RankingSnapshotMaterializationError(
                f"annotation release file hash mismatch: {name}"
            )

    human = load_human_judgment_set(root / "human-judgments.json")
    adjudicated = human.adjudicate()
    qrels = FrozenCandidateSet.load(root / "qrels.json")
    expected_qrels = adjudicated.frozen_candidates
    if (
        qrels.version != expected_qrels.version
        or qrels.checksum != expected_qrels.checksum
        or qrels.evidence_checksum != expected_qrels.evidence_checksum
    ):
        raise RankingSnapshotMaterializationError(
            "qrels do not match the independently reviewed human judgments"
        )
    if not qrels.eligible_for_promotion:
        raise RankingSnapshotMaterializationError(
            "qrels are not adjudicated, non-synthetic human evidence"
        )
    split = QueryGroupSplit.load(root / "query-split.json")
    split.validate_dataset(qrels)
    expected_groups = {query.query_id: query.query_group_id for query in qrels.queries}
    if dict(split.query_group_ids) != expected_groups:
        raise RankingSnapshotMaterializationError(
            "annotation split query groups do not match qrels"
        )
    reviewers: dict[tuple[str, str], set[str]] = defaultdict(set)
    for judgment in human.judgments:
        reviewers[(judgment.query_id, judgment.product_id)].add(judgment.reviewer_id)
    expected_pairs = {
        (query.query_id, product_id)
        for query in human.queries
        for product_id in query.candidate_ids
    }
    if set(reviewers) != expected_pairs or any(
        len(reviewers[pair]) < 2 for pair in expected_pairs
    ):
        raise RankingSnapshotMaterializationError(
            "every relevance pair requires two independent human reviewers"
        )

    evidence = _load_json_object(
        root / "evidence-snapshots.json",
        name="annotation evidence snapshots",
    )
    if evidence.get("schema_version") != EVIDENCE_SNAPSHOT_SCHEMA_VERSION:
        raise RankingSnapshotMaterializationError(
            f"evidence schema_version must be {EVIDENCE_SNAPSHOT_SCHEMA_VERSION!r}"
        )
    return manifest, evidence, human, qrels, split, sha256_file(manifest_path)


def _materialized_rows(
    *,
    prelabel_rows: Sequence[Mapping[str, Any]],
    snapshot: Mapping[str, Any],
    evidence: Mapping[str, Any],
    qrels: FrozenCandidateSet,
) -> tuple[dict[str, Any], ...]:
    prelabel_by_id = {
        _string(row.get("query_id"), name="pre-label query_id"): row
        for row in prelabel_rows
    }
    qrels_by_id = {query.query_id: query for query in qrels.queries}
    raw_groups = evidence.get("groups")
    if not isinstance(raw_groups, list):
        raise RankingSnapshotMaterializationError("annotation evidence groups must be an array")
    evidence_by_id = {
        _string(
            _object(group, name=f"annotation evidence group {index}").get("group_key"),
            name=f"annotation evidence group {index} key",
        ): _object(group, name=f"annotation evidence group {index}")
        for index, group in enumerate(raw_groups)
    }
    if set(prelabel_by_id) != set(qrels_by_id) or set(evidence_by_id) != set(qrels_by_id):
        raise RankingSnapshotMaterializationError(
            "pre-label, evidence, and qrels query universes differ"
        )
    row_hashes = _object(snapshot.get("query_row_sha256"), name="pre-label row hashes")
    materialized: list[dict[str, Any]] = []
    for query_id in sorted(qrels_by_id):
        source_row = _object(prelabel_by_id[query_id], name=f"pre-label row {query_id!r}")
        qrel = qrels_by_id[query_id]
        group = evidence_by_id[query_id]
        if (
            source_row.get("query_group_id") != qrel.query_group_id
            or str(source_row.get("category", "")).casefold() != qrel.category
            or group.get("leakage_group_id") != qrel.query_group_id
            or str(group.get("category", "")).casefold() != qrel.category
        ):
            raise RankingSnapshotMaterializationError(
                f"query identity differs across lineage for {query_id!r}"
            )
        context = _object(
            group.get("context_payload"),
            name=f"annotation evidence context {query_id!r}",
        )
        binding = _object(
            context.get("ranking_prelabel_binding"),
            name=f"annotation pre-label binding {query_id!r}",
        )
        if (
            binding.get("snapshot_sha256") != snapshot.get("snapshot_sha256")
            or binding.get("query_row_sha256") != row_hashes.get(query_id)
            or binding.get("candidate_ids_sha256")
            != source_row.get("candidate_ids_sha256")
        ):
            raise RankingSnapshotMaterializationError(
                f"annotation evidence lost the pre-label commitment for {query_id!r}"
            )
        raw_candidates = source_row.get("candidates")
        if not isinstance(raw_candidates, list):
            raise RankingSnapshotMaterializationError(
                f"pre-label candidates are invalid for {query_id!r}"
            )
        candidate_ids = [
            _string(
                _object(candidate, name=f"pre-label candidate {query_id!r}").get(
                    "product_id"
                ),
                name=f"pre-label candidate {query_id!r} product_id",
            )
            for candidate in raw_candidates
        ]
        evidence_items = group.get("items")
        if not isinstance(evidence_items, list):
            raise RankingSnapshotMaterializationError(
                f"annotation evidence items are invalid for {query_id!r}"
            )
        evidence_ids = {
            _string(
                _object(item, name=f"annotation evidence item {query_id!r}").get(
                    "target_id"
                ),
                name=f"annotation evidence item {query_id!r} target_id",
            )
            for item in evidence_items
        }
        if (
            set(candidate_ids) != set(qrel.candidate_ids)
            or evidence_ids != set(qrel.candidate_ids)
            or set(qrel.relevance_labels) != set(qrel.candidate_ids)
        ):
            raise RankingSnapshotMaterializationError(
                f"candidate universe differs across lineage for {query_id!r}"
            )
        labeled_candidates = []
        for raw_candidate in raw_candidates:
            candidate = _object(raw_candidate, name=f"pre-label candidate {query_id!r}")
            product_id = str(candidate["product_id"])
            labeled_candidates.append(
                {
                    **candidate,
                    "relevance_grade": qrel.relevance_labels[product_id],
                }
            )
        materialized.append(
            {
                **source_row,
                "candidates": labeled_candidates,
            }
        )
    return tuple(materialized)


def _write_output(
    *,
    output_dir: Path,
    ranking_bytes: bytes,
    manifest: Mapping[str, Any],
) -> None:
    destination = output_dir.resolve()
    if destination.exists():
        raise RankingSnapshotMaterializationError(
            f"output directory already exists and will not be overwritten: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        (temporary / RANKING_DATA_FILENAME).write_bytes(ranking_bytes)
        (temporary / "manifest.json").write_bytes(canonical_json_bytes(manifest) + b"\n")
        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def verify_labeled_ranking_snapshot(
    *,
    ranking_path: Path,
    manifest_path: Path,
    human_judgments_path: Path,
    qrels_path: Path,
    query_split_path: Path,
) -> VerifiedLabeledRankingSnapshot:
    """Verify the exact materialized file and every declared human lineage input."""

    manifest = _load_json_object(manifest_path, name="labeled ranking snapshot manifest")
    if manifest.get("schema_version") != LABELED_SNAPSHOT_MANIFEST_SCHEMA_VERSION:
        raise RankingSnapshotMaterializationError(
            "labeled ranking snapshot manifest has an unsupported schema"
        )
    manifest_sha256 = _digest(
        manifest.get("manifest_sha256"),
        name="labeled snapshot manifest_sha256",
    )
    semantic_manifest = dict(manifest)
    semantic_manifest.pop("manifest_sha256")
    if sha256_json(semantic_manifest) != manifest_sha256:
        raise RankingSnapshotMaterializationError(
            "labeled ranking snapshot manifest self-hash mismatch"
        )
    files = _object(manifest.get("files"), name="labeled snapshot files")
    if set(files) != {RANKING_DATA_FILENAME}:
        raise RankingSnapshotMaterializationError(
            "labeled snapshot must contain exactly one ranking JSONL file"
        )
    ranking_evidence = _object(
        files[RANKING_DATA_FILENAME],
        name="labeled snapshot ranking file",
    )
    resolved_ranking = ranking_path.resolve(strict=True)
    if resolved_ranking.name != RANKING_DATA_FILENAME:
        raise RankingSnapshotMaterializationError(
            f"labeled ranking input must be named {RANKING_DATA_FILENAME!r}"
        )
    if resolved_ranking.stat().st_size != ranking_evidence.get("size_bytes"):
        raise RankingSnapshotMaterializationError("labeled ranking file size mismatch")
    ranking_file_sha256 = _digest(
        ranking_evidence.get("sha256"),
        name="labeled ranking file sha256",
    )
    if sha256_file(resolved_ranking) != ranking_file_sha256:
        raise RankingSnapshotMaterializationError("labeled ranking file hash mismatch")
    prelabel = _object(manifest.get("prelabel"), name="labeled snapshot pre-label binding")
    expected_prelabel_rows = _object(
        prelabel.get("query_row_sha256"),
        name="labeled snapshot pre-label row hashes",
    )
    actual_prelabel_rows: dict[str, str] = {}
    with resolved_ranking.open("rb") as stream:
        for line_number, line in enumerate(stream, start=1):
            if len(line) > MAX_FEATURE_LINE_BYTES:
                raise RankingSnapshotMaterializationError(
                    f"labeled ranking line {line_number} exceeds safety limit"
                )
            if not line.strip():
                continue
            try:
                row = _object(json.loads(line), name=f"labeled ranking line {line_number}")
            except json.JSONDecodeError as error:
                raise RankingSnapshotMaterializationError(
                    f"labeled ranking line {line_number} is invalid JSON"
                ) from error
            query_id = _string(
                row.get("query_id"),
                name=f"labeled ranking line {line_number} query_id",
            )
            if query_id in actual_prelabel_rows:
                raise RankingSnapshotMaterializationError(
                    f"labeled ranking query_id is duplicated: {query_id!r}"
                )
            raw_candidates = row.get("candidates")
            if not isinstance(raw_candidates, list) or not raw_candidates:
                raise RankingSnapshotMaterializationError(
                    f"labeled ranking candidates are missing for {query_id!r}"
                )
            prelabel_candidates: list[dict[str, Any]] = []
            for candidate_index, raw_candidate in enumerate(raw_candidates):
                candidate = _object(
                    raw_candidate,
                    name=f"labeled ranking candidate {candidate_index}",
                )
                grade = candidate.pop("relevance_grade", None)
                if isinstance(grade, bool) or not isinstance(grade, int) or not 0 <= grade <= 4:
                    raise RankingSnapshotMaterializationError(
                        f"labeled ranking candidate {candidate_index} has an invalid grade"
                    )
                prelabel_candidates.append(candidate)
            prelabel_row = {**row, "candidates": prelabel_candidates}
            actual_prelabel_rows[query_id] = sha256_json(prelabel_row)
    if set(actual_prelabel_rows) != set(expected_prelabel_rows):
        raise RankingSnapshotMaterializationError(
            "labeled ranking queries do not match the pre-label row commitment"
        )
    for query_id, actual_hash in actual_prelabel_rows.items():
        if actual_hash != _digest(
            expected_prelabel_rows[query_id],
            name=f"labeled snapshot pre-label row hash {query_id!r}",
        ):
            raise RankingSnapshotMaterializationError(
                f"labeled ranking changed pre-label features for {query_id!r}"
            )

    annotation = _object(
        manifest.get("annotation_release"),
        name="labeled snapshot annotation release",
    )
    annotation_files = _object(
        annotation.get("files"),
        name="labeled snapshot annotation files",
    )
    if set(annotation_files) != _RELEVANCE_RELEASE_FILES:
        raise RankingSnapshotMaterializationError(
            "labeled snapshot annotation files do not match the release contract"
        )
    supplied_paths = {
        "human-judgments.json": human_judgments_path,
        "qrels.json": qrels_path,
        "query-split.json": query_split_path,
    }
    resolved_supplied_paths = {
        name: path.resolve(strict=True) for name, path in supplied_paths.items()
    }
    release_roots = {path.parent for path in resolved_supplied_paths.values()}
    if len(release_roots) != 1:
        raise RankingSnapshotMaterializationError(
            "human judgments, qrels, and query split must come from one release directory"
        )
    release_root = next(iter(release_roots))
    evidence_path = release_root / "evidence-snapshots.json"
    if not evidence_path.is_file():
        raise RankingSnapshotMaterializationError(
            "annotation evidence snapshots are missing"
        )
    all_annotation_paths = {
        **resolved_supplied_paths,
        "evidence-snapshots.json": evidence_path,
    }
    for name, resolved in all_annotation_paths.items():
        evidence = _object(
            annotation_files.get(name),
            name=f"labeled snapshot annotation file {name}",
        )
        if resolved.stat().st_size != evidence.get("size_bytes"):
            raise RankingSnapshotMaterializationError(
                f"labeled snapshot annotation file size mismatch: {name}"
            )
        if sha256_file(resolved) != _digest(
            evidence.get("sha256"),
            name=f"labeled snapshot annotation file hash {name}",
        ):
            raise RankingSnapshotMaterializationError(
                f"labeled snapshot annotation file hash mismatch: {name}"
            )
    evidence_payload = _load_json_object(
        evidence_path,
        name="labeled snapshot annotation evidence",
    )
    raw_groups = evidence_payload.get("groups")
    if not isinstance(raw_groups, list):
        raise RankingSnapshotMaterializationError(
            "labeled snapshot annotation evidence groups must be an array"
        )
    evidence_bindings: dict[str, dict[str, Any]] = {}
    for index, raw_group in enumerate(raw_groups):
        group = _object(raw_group, name=f"labeled snapshot evidence group {index}")
        query_id = _string(
            group.get("group_key"),
            name=f"labeled snapshot evidence group {index} key",
        )
        context = _object(
            group.get("context_payload"),
            name=f"labeled snapshot evidence group {query_id!r} context",
        )
        evidence_bindings[query_id] = _object(
            context.get("ranking_prelabel_binding"),
            name=f"labeled snapshot evidence group {query_id!r} pre-label binding",
        )
    if set(evidence_bindings) != set(expected_prelabel_rows):
        raise RankingSnapshotMaterializationError(
            "annotation evidence queries do not match the pre-label row commitment"
        )
    prelabel_snapshot_sha256 = _digest(
        prelabel.get("snapshot_sha256"),
        name="labeled snapshot pre-label snapshot hash",
    )
    for query_id, binding in evidence_bindings.items():
        if (
            binding.get("snapshot_sha256") != prelabel_snapshot_sha256
            or binding.get("query_row_sha256") != expected_prelabel_rows[query_id]
        ):
            raise RankingSnapshotMaterializationError(
                f"annotation evidence pre-label binding mismatch for {query_id!r}"
            )

    human = load_human_judgment_set(human_judgments_path)
    human_manifest = _object(
        manifest.get("human_judgments"),
        name="labeled snapshot human judgments",
    )
    if human.content_sha256 != _digest(
        human_manifest.get("content_sha256"),
        name="labeled snapshot human judgment content hash",
    ):
        raise RankingSnapshotMaterializationError(
            "labeled snapshot human judgment content hash mismatch"
        )
    if human_manifest.get("minimum_independent_reviewers_per_pair") != 2:
        raise RankingSnapshotMaterializationError(
            "labeled snapshot does not require two independent reviewers"
        )
    qrels = FrozenCandidateSet.load(qrels_path)
    qrels_manifest = _object(manifest.get("qrels"), name="labeled snapshot qrels")
    if (
        qrels.version != qrels_manifest.get("version")
        or qrels.checksum != qrels_manifest.get("checksum")
        or qrels.evidence_checksum != qrels_manifest.get("evidence_checksum")
        or qrels.judgment_manifest_sha256
        != qrels_manifest.get("judgment_manifest_sha256")
    ):
        raise RankingSnapshotMaterializationError(
            "labeled snapshot qrels binding mismatch"
        )
    split = QueryGroupSplit.load(query_split_path)
    split_manifest = _object(
        manifest.get("query_split"),
        name="labeled snapshot query split",
    )
    if (
        split.version != split_manifest.get("version")
        or split.checksum != split_manifest.get("checksum")
        or dict(sorted(split.assignments.items())) != split_manifest.get("assignments")
    ):
        raise RankingSnapshotMaterializationError(
            "labeled snapshot query-split binding mismatch"
        )
    split.validate_dataset(qrels)
    return VerifiedLabeledRankingSnapshot(
        manifest_path=manifest_path.resolve(strict=True),
        manifest_file_sha256=sha256_file(manifest_path),
        manifest_sha256=manifest_sha256,
        ranking_file_sha256=ranking_file_sha256,
        prelabel_snapshot_sha256=prelabel_snapshot_sha256,
        feature_contract_sha256=_digest(
            prelabel.get("feature_contract_sha256"),
            name="labeled snapshot feature contract hash",
        ),
        annotation_release_sha256=_digest(
            annotation.get("release_sha256"),
            name="labeled snapshot annotation release hash",
        ),
    )


def materialize_ranking_snapshot(
    *,
    capture_dir: Path,
    annotation_release_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Create an immutable, manifest-bound human ranking training snapshot."""

    capture_manifest, snapshot, prelabel_rows, capture_manifest_sha256 = (
        _load_prelabel_rows(capture_dir)
    )
    (
        annotation_manifest,
        evidence,
        human,
        qrels,
        split,
        annotation_manifest_sha256,
    ) = _verify_annotation_release(annotation_release_dir)
    if capture_manifest.get("source_policy_sha256") != annotation_manifest.get(
        "source_policy_sha256"
    ):
        raise RankingSnapshotMaterializationError(
            "capture and annotation release source policies differ"
        )
    if capture_manifest.get("query_count") != len(qrels.queries):
        raise RankingSnapshotMaterializationError(
            "capture and annotation release query counts differ"
        )
    rows = _materialized_rows(
        prelabel_rows=prelabel_rows,
        snapshot=snapshot,
        evidence=evidence,
        qrels=qrels,
    )
    ranking_bytes = b"".join(canonical_json_bytes(row) + b"\n" for row in rows)
    release_root = annotation_release_dir.resolve(strict=True)
    file_bindings = {
        name: {
            "sha256": sha256_file(release_root / name),
            "size_bytes": (release_root / name).stat().st_size,
        }
        for name in sorted(_RELEVANCE_RELEASE_FILES)
    }
    manifest: dict[str, Any] = {
        "schema_version": LABELED_SNAPSHOT_MANIFEST_SCHEMA_VERSION,
        "prelabel": {
            "capture_manifest_file_sha256": capture_manifest_sha256,
            "snapshot_sha256": snapshot["snapshot_sha256"],
            "feature_file_sha256": snapshot["file_sha256"],
            "candidate_universe_sha256": snapshot["candidate_universe_sha256"],
            "feature_contract_sha256": snapshot["feature_contract_sha256"],
            "query_row_sha256": snapshot["query_row_sha256"],
        },
        "annotation_release": {
            "manifest_file_sha256": annotation_manifest_sha256,
            "release_sha256": annotation_manifest["release_sha256"],
            "files": file_bindings,
        },
        "human_judgments": {
            "content_sha256": human.content_sha256,
            "minimum_independent_reviewers_per_pair": 2,
        },
        "qrels": {
            "version": qrels.version,
            "checksum": qrels.checksum,
            "evidence_checksum": qrels.evidence_checksum,
            "judgment_manifest_sha256": qrels.judgment_manifest_sha256,
        },
        "query_split": {
            "version": split.version,
            "checksum": split.checksum,
            "assignments": dict(sorted(split.assignments.items())),
        },
        "dataset": {
            "query_count": len(rows),
            "row_count": sum(len(row["candidates"]) for row in rows),
            "query_group_count": len(set(split.query_group_ids.values())),
            "grade_counts": {
                str(grade): sum(
                    1
                    for row in rows
                    for candidate in row["candidates"]
                    if candidate["relevance_grade"] == grade
                )
                for grade in range(5)
            },
        },
        "files": {
            RANKING_DATA_FILENAME: {
                "sha256": sha256_file_bytes(ranking_bytes),
                "size_bytes": len(ranking_bytes),
            }
        },
    }
    manifest["manifest_sha256"] = sha256_json(manifest)
    _write_output(
        output_dir=output_dir,
        ranking_bytes=ranking_bytes,
        manifest=manifest,
    )
    return {
        "schema_version": LABELED_SNAPSHOT_MANIFEST_SCHEMA_VERSION,
        "status": "ok",
        "output_dir": str(output_dir.resolve()),
        "manifest_sha256": manifest["manifest_sha256"],
        "query_count": manifest["dataset"]["query_count"],
        "row_count": manifest["dataset"]["row_count"],
    }


def sha256_file_bytes(payload: bytes) -> str:
    """Hash bytes without writing a temporary file."""

    return hashlib.sha256(payload).hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = materialize_ranking_snapshot(
            capture_dir=args.capture_dir,
            annotation_release_dir=args.annotation_release_dir,
            output_dir=args.output_dir,
        )
    except (
        OSError,
        RankingSnapshotMaterializationError,
        TypeError,
        ValueError,
    ) as error:
        print(
            json.dumps({"status": "error", "message": str(error)}, sort_keys=True),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
