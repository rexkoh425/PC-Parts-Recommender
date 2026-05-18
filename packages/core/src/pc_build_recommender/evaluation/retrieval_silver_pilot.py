"""Reproducible retrieval baselines on the real catalog with silver judgments.

The judgments used here are deliberately weak: a checked-in query definition
declares category/specification predicates, and products satisfying those
predicates receive grade 3 or 4.  The same labels must never be presented as
human relevance judgments, used to promote a learned ranker, or used for
resume/production metric claims.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any

import numpy as np
from numpy.typing import NDArray

from pc_build_recommender.retrieval import (
    BM25ProductIndex,
    FrozenCandidateQuery,
    PinnedCandidateSet,
    ProductDocument,
    RelevanceLabelSource,
    SentenceTransformerEmbeddingEncoder,
    evaluate_ranked_candidates,
    reciprocal_rank_fusion,
)
from pc_build_recommender.retrieval.embedding_index import (
    build_product_embedding_text,
    load_normalized_product_jsonl,
)

from .manifest import canonical_json_bytes, sha256_file
from .metrics import bootstrap_confidence_interval

SILVER_QUERY_SCHEMA_VERSION = "pc-build-recommender.silver-query-set.v1"
SILVER_RESULT_SCHEMA_VERSION = "pc-build-recommender.retrieval-silver-pilot.v1"
DEFAULT_SCORE_DECIMALS = 8
REPORTING_BLOCK_REASON = (
    "Relevance grades are deterministic silver labels derived from the same declared "
    "specification predicates used to define each query; no human relevance review was done."
)


def _normalise_scalar(value: object) -> object:
    return value.casefold() if isinstance(value, str) else value


def _field_value(record: Mapping[str, Any], field: str) -> Any:
    current: Any = record
    for part in field.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


@dataclass(frozen=True, slots=True)
class SilverConstraint:
    field: str
    operator: str
    value: Any

    def __post_init__(self) -> None:
        if not self.field:
            raise ValueError("constraint field must not be empty")
        supported = {"eq", "ge", "le", "one_of", "contains", "contains_all", "not_null"}
        if self.operator not in supported:
            raise ValueError(f"unsupported silver constraint operator: {self.operator}")
        if self.operator in {"one_of", "contains_all"} and (
            not isinstance(self.value, list) or not self.value
        ):
            raise ValueError(f"{self.operator} requires a non-empty list value")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> SilverConstraint:
        return cls(
            field=str(payload["field"]),
            operator=str(payload["operator"]),
            value=payload.get("value"),
        )

    def matches(self, record: Mapping[str, Any]) -> bool:
        actual = _field_value(record, self.field)
        if self.operator == "not_null":
            return actual is not None and actual != ""
        if actual is None:
            return False
        if self.operator == "eq":
            return _normalise_scalar(actual) == _normalise_scalar(self.value)
        if self.operator in {"ge", "le"}:
            if isinstance(actual, bool) or isinstance(self.value, bool):
                return False
            try:
                left = float(actual)
                right = float(self.value)
            except (TypeError, ValueError):
                return False
            if not math.isfinite(left) or not math.isfinite(right):
                return False
            return left >= right if self.operator == "ge" else left <= right
        if self.operator == "one_of":
            expected = {_normalise_scalar(item) for item in self.value}
            return _normalise_scalar(actual) in expected
        if self.operator == "contains":
            if not isinstance(actual, Sequence) or isinstance(actual, (str, bytes, bytearray)):
                return False
            expected_item = _normalise_scalar(self.value)
            return expected_item in {_normalise_scalar(item) for item in actual}
        if self.operator == "contains_all":
            if not isinstance(actual, Sequence) or isinstance(actual, (str, bytes, bytearray)):
                return False
            actual_values = {_normalise_scalar(item) for item in actual}
            return all(_normalise_scalar(item) in actual_values for item in self.value)
        raise AssertionError("validated operator was not implemented")


@dataclass(frozen=True, slots=True)
class SilverQuery:
    query_id: str
    query_text: str
    category: str
    must: tuple[SilverConstraint, ...]
    excellent: tuple[SilverConstraint, ...]

    def __post_init__(self) -> None:
        if not self.query_id or not self.query_text or not self.category:
            raise ValueError("silver query ID, text, and category must not be empty")
        if not self.must:
            raise ValueError("each silver query must declare at least one required predicate")
        object.__setattr__(self, "category", self.category.casefold())

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> SilverQuery:
        must = payload.get("must")
        excellent = payload.get("excellent", [])
        if not isinstance(must, list) or not isinstance(excellent, list):
            raise TypeError("query must and excellent predicates must be lists")
        return cls(
            query_id=str(payload["query_id"]),
            query_text=str(payload["query_text"]),
            category=str(payload["category"]),
            must=tuple(SilverConstraint.from_mapping(item) for item in must),
            excellent=tuple(SilverConstraint.from_mapping(item) for item in excellent),
        )


@dataclass(frozen=True, slots=True)
class SilverQuerySet:
    dataset_name: str
    judgment_method: str
    queries: tuple[SilverQuery, ...]
    source_sha256: str

    def __post_init__(self) -> None:
        if not self.dataset_name or not self.judgment_method or not self.queries:
            raise ValueError("silver query set metadata and queries must not be empty")
        query_ids = [query.query_id for query in self.queries]
        if len(query_ids) != len(set(query_ids)):
            raise ValueError("silver query IDs must be unique")


def load_silver_query_set(path: str | Path) -> SilverQuerySet:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError("silver query set root must be an object")
    if payload.get("schema_version") != SILVER_QUERY_SCHEMA_VERSION:
        raise ValueError("unsupported silver query set schema")
    query_payloads = payload.get("queries")
    if not isinstance(query_payloads, list):
        raise TypeError("silver query set queries must be a list")
    return SilverQuerySet(
        dataset_name=str(payload["dataset_name"]),
        judgment_method=str(payload["judgment_method"]),
        queries=tuple(SilverQuery.from_mapping(item) for item in query_payloads),
        source_sha256=sha256_file(source),
    )


def silver_grade(record: Mapping[str, Any], query: SilverQuery) -> int:
    """Return 0, 3, or 4 from the declared weak-label predicates."""

    if str(record.get("category", "")).casefold() != query.category:
        return 0
    if not all(constraint.matches(record) for constraint in query.must):
        return 0
    if query.excellent and all(constraint.matches(record) for constraint in query.excellent):
        return 4
    return 3


def build_frozen_silver_dataset(
    records: Sequence[Mapping[str, Any]],
    query_set: SilverQuerySet,
    *,
    catalog_sha256: str,
) -> PinnedCandidateSet:
    """Create a checksummed candidate universe and silver qrels."""

    by_category: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        by_category[str(record["category"]).casefold()].append(record)
    queries: list[FrozenCandidateQuery] = []
    for query in query_set.queries:
        category_records = sorted(
            by_category.get(query.category, []), key=lambda row: str(row["product_id"])
        )
        if not category_records:
            raise ValueError(f"query {query.query_id!r} references an empty category")
        labels = {
            str(record["product_id"]): grade
            for record in category_records
            if (grade := silver_grade(record, query)) > 0
        }
        if not labels:
            raise ValueError(f"query {query.query_id!r} produced no silver-relevant products")
        queries.append(
            FrozenCandidateQuery(
                query_id=query.query_id,
                query_text=query.query_text,
                category=query.category,
                candidate_ids=tuple(str(record["product_id"]) for record in category_records),
                relevance_labels=labels,
            )
        )
    version = f"buildcores-silver-pilot-v1-{catalog_sha256[:12]}-{query_set.source_sha256[:12]}"
    return PinnedCandidateSet.create(
        version,
        queries,
        label_source=RelevanceLabelSource.SILVER,
        adjudication_complete=False,
        contains_synthetic_labels=False,
    )


def _verify_file_contract(path: Path, metadata: Mapping[str, Any]) -> None:
    if metadata.get("sha256") != sha256_file(path):
        raise ValueError(f"artifact checksum mismatch: {path}")
    if metadata.get("bytes") != path.stat().st_size:
        raise ValueError(f"artifact size mismatch: {path}")


def _load_embedding_artifact(
    directory: str | Path,
) -> tuple[NDArray[np.float32], tuple[Mapping[str, Any], ...], Mapping[str, Any]]:
    root = Path(directory)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping):
        raise TypeError("embedding manifest root must be an object")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise TypeError("embedding manifest artifacts must be an object")
    embeddings_path = root / str(artifacts["embeddings"]["path"])
    id_map_path = root / str(artifacts["id_map"]["path"])
    _verify_file_contract(embeddings_path, artifacts["embeddings"])
    _verify_file_contract(id_map_path, artifacts["id_map"])
    matrix = np.load(embeddings_path, allow_pickle=False)
    if matrix.dtype != np.float32 or matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError("embedding matrix must be a finite float32 matrix")
    rows = tuple(
        json.loads(line)
        for line in id_map_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if len(rows) != matrix.shape[0]:
        raise ValueError("embedding row map does not match matrix rows")
    ordered_rows = tuple(sorted(rows, key=lambda row: int(row["row_index"])))
    if [int(row["row_index"]) for row in ordered_rows] != list(range(matrix.shape[0])):
        raise ValueError("embedding row indices are not a complete contiguous range")
    return matrix, ordered_rows, manifest


def _complete_bm25_rankings(
    dataset: PinnedCandidateSet, documents: Sequence[ProductDocument]
) -> dict[str, list[str]]:
    index = BM25ProductIndex(documents)
    rankings: dict[str, list[str]] = {}
    for query in dataset.queries:
        assert query.category is not None
        hits = index.search(
            query.query_text,
            category=query.category,
            top_k=len(query.candidate_ids),
            candidate_ids=set(query.candidate_ids),
        )
        ranking = [hit.product_id for hit in hits]
        if set(ranking) != set(query.candidate_ids):
            raise ValueError(f"BM25 did not rank the frozen universe for {query.query_id}")
        rankings[query.query_id] = ranking
    return rankings


def _complete_vector_rankings(
    dataset: PinnedCandidateSet,
    *,
    matrix: NDArray[np.float32],
    id_rows: Sequence[Mapping[str, Any]],
    model_name: str,
    device: str,
    batch_size: int,
    score_decimals: int,
) -> tuple[dict[str, list[str]], Mapping[str, object]]:
    encoder = SentenceTransformerEmbeddingEncoder(
        model_name,
        device=device,
        batch_size=batch_size,
    )
    query_vectors = encoder.encode([query.query_text for query in dataset.queries])
    if query_vectors.shape != (len(dataset.queries), matrix.shape[1]):
        raise ValueError("query embeddings do not match the catalog embedding dimension")
    index_by_id = {str(row["product_id"]): int(row["row_index"]) for row in id_rows}
    if len(index_by_id) != len(id_rows):
        raise ValueError("embedding row map contains duplicate product IDs")
    rankings: dict[str, list[str]] = {}
    for query_index, query in enumerate(dataset.queries):
        missing = set(query.candidate_ids) - set(index_by_id)
        if missing:
            raise ValueError(f"embedding artifact is missing candidates: {sorted(missing)[:3]}")
        scored = [
            (
                product_id,
                round(
                    float(np.dot(matrix[index_by_id[product_id]], query_vectors[query_index])),
                    score_decimals,
                ),
            )
            for product_id in query.candidate_ids
        ]
        scored.sort(key=lambda item: (-item[1], item[0]))
        rankings[query.query_id] = [product_id for product_id, _ in scored]
    metadata: dict[str, object] = {
        "model_name": model_name,
        "requested_device": device,
        "resolved_device": encoder.resolved_device,
        "batch_size": batch_size,
        "dimension": matrix.shape[1],
        "cosine_score_rounding_decimals": score_decimals,
    }
    return rankings, metadata


def _complete_rrf_rankings(
    dataset: PinnedCandidateSet,
    bm25: Mapping[str, Sequence[str]],
    vector: Mapping[str, Sequence[str]],
    *,
    rrf_k: int,
) -> dict[str, list[str]]:
    rankings: dict[str, list[str]] = {}
    for query in dataset.queries:
        fused = reciprocal_rank_fusion(
            {"bm25": bm25[query.query_id], "vector": vector[query.query_id]},
            k=rrf_k,
            limit=len(query.candidate_ids),
        )
        ranking = [hit.product_id for hit in fused]
        if set(ranking) != set(query.candidate_ids):
            raise ValueError(f"RRF did not rank the frozen universe for {query.query_id}")
        rankings[query.query_id] = ranking
    return rankings


def _single_query_metrics(query: FrozenCandidateQuery, ranking: Sequence[str]) -> dict[str, float]:
    single = PinnedCandidateSet.create("single-query-slice-v1", [query])
    result = evaluate_ranked_candidates(
        single,
        {query.query_id: ranking},
        recall_ks=(20, 50),
    )
    return {
        "recall_at_20": result.recall_at[20],
        "recall_at_50": result.recall_at[50],
        "mean_reciprocal_rank": result.mean_reciprocal_rank,
        "ndcg_at_10": result.ndcg_at_10,
    }


def _model_metrics(
    dataset: PinnedCandidateSet,
    rankings: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    aggregate = evaluate_ranked_candidates(dataset, rankings, recall_ks=(20, 50))
    per_query = {
        query.query_id: _single_query_metrics(query, rankings[query.query_id])
        for query in dataset.queries
    }
    metric_names = ("recall_at_20", "recall_at_50", "mean_reciprocal_rank", "ndcg_at_10")
    confidence_intervals: dict[str, object] = {}
    for metric_name in metric_names:
        values = [per_query[query.query_id][metric_name] for query in dataset.queries]
        lower, upper = bootstrap_confidence_interval(values, n_resamples=1_000)
        confidence_intervals[metric_name] = {
            "lower": lower,
            "upper": upper,
            "confidence_level": 0.95,
            "resamples": 1_000,
            "seed": 20260722,
        }
    by_category: dict[str, dict[str, float]] = {}
    categories = sorted({query.category for query in dataset.queries if query.category})
    for category in categories:
        members = [query for query in dataset.queries if query.category == category]
        by_category[str(category)] = {
            metric_name: fmean(per_query[query.query_id][metric_name] for query in members)
            for metric_name in metric_names
        }
    return {
        "aggregate": aggregate.to_dict(),
        "macro_query_bootstrap_ci": confidence_intervals,
        "by_category": by_category,
        "per_query": per_query,
    }


def _paired_baseline_comparison(
    baseline: Mapping[str, Any],
    challenger: Mapping[str, Any],
) -> dict[str, object]:
    metric_names = ("recall_at_20", "recall_at_50", "mean_reciprocal_rank", "ndcg_at_10")
    baseline_per_query = baseline["per_query"]
    challenger_per_query = challenger["per_query"]
    query_ids = sorted(baseline_per_query)
    if query_ids != sorted(challenger_per_query):
        raise ValueError("baseline and challenger query sets do not match")
    comparisons: dict[str, object] = {}
    for metric_name in metric_names:
        differences = [
            float(challenger_per_query[query_id][metric_name])
            - float(baseline_per_query[query_id][metric_name])
            for query_id in query_ids
        ]
        absolute_delta = fmean(differences)
        baseline_value = fmean(
            float(baseline_per_query[query_id][metric_name]) for query_id in query_ids
        )
        lower, upper = bootstrap_confidence_interval(differences, n_resamples=1_000)
        comparisons[metric_name] = {
            "baseline_value": baseline_value,
            "challenger_value": baseline_value + absolute_delta,
            "absolute_delta": absolute_delta,
            "relative_delta_percent": (
                absolute_delta / baseline_value * 100.0 if baseline_value else None
            ),
            "paired_query_bootstrap_delta_ci": {
                "lower": lower,
                "upper": upper,
                "confidence_level": 0.95,
                "resamples": 1_000,
                "seed": 20260722,
            },
        }
    return comparisons


def _portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
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
            handle.write(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
            handle.write("\n")
            temporary_name = handle.name
        os.replace(temporary_name, path)
    finally:
        if temporary_name and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _artifact_hash(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def run_retrieval_silver_pilot(
    *,
    catalog_path: str | Path,
    query_set_path: str | Path,
    embedding_dir: str | Path,
    output_dir: str | Path,
    device: str = "auto",
    batch_size: int = 64,
    rrf_k: int = 60,
    score_decimals: int = DEFAULT_SCORE_DECIMALS,
) -> Mapping[str, object]:
    """Run BM25, vector, RRF, and a deterministic negative control."""

    if rrf_k < 1 or batch_size < 1 or score_decimals < 0:
        raise ValueError("rrf_k/batch_size must be positive and score_decimals non-negative")
    catalog = Path(catalog_path)
    queries_path = Path(query_set_path)
    embeddings_root = Path(embedding_dir)
    output = Path(output_dir)
    records, source_files = load_normalized_product_jsonl(catalog)
    if len(source_files) != 1:
        raise ValueError("the pilot requires one frozen normalized catalog JSONL")
    catalog_hash = sha256_file(source_files[0])
    query_set = load_silver_query_set(queries_path)
    dataset = build_frozen_silver_dataset(
        records,
        query_set,
        catalog_sha256=catalog_hash,
    )
    documents = tuple(
        ProductDocument(
            product_id=str(record["product_id"]),
            category=str(record["category"]),
            text=build_product_embedding_text(record),
            brand=str(record["brand"]) if record.get("brand") is not None else None,
            attributes={
                **dict(record.get("common_attributes", {})),
                **dict(record.get("category_attributes", {})),
            },
        )
        for record in records
    )
    matrix, id_rows, embedding_manifest = _load_embedding_artifact(embeddings_root)
    embedding_encoder = embedding_manifest.get("encoder")
    if not isinstance(embedding_encoder, Mapping):
        raise TypeError("embedding manifest encoder must be an object")
    if embedding_encoder.get("kind") != "sentence_transformer":
        raise ValueError("vector pilot requires a SentenceTransformer embedding artifact")
    model_name = str(embedding_encoder["model_name"])

    bm25_rankings = _complete_bm25_rankings(dataset, documents)
    vector_rankings, query_encoder_metadata = _complete_vector_rankings(
        dataset,
        matrix=matrix,
        id_rows=id_rows,
        model_name=model_name,
        device=device,
        batch_size=batch_size,
        score_decimals=score_decimals,
    )
    rrf_rankings = _complete_rrf_rankings(
        dataset,
        bm25_rankings,
        vector_rankings,
        rrf_k=rrf_k,
    )
    negative_control_rankings = {
        query.query_id: list(query.candidate_ids) for query in dataset.queries
    }
    model_rankings: dict[str, Mapping[str, Sequence[str]]] = {
        "stable_product_id_negative_control": negative_control_rankings,
        "bm25": bm25_rankings,
        "vector_only": vector_rankings,
        "rrf_hybrid": rrf_rankings,
    }
    model_results = {
        model_name_key: _model_metrics(dataset, rankings)
        for model_name_key, rankings in model_rankings.items()
    }

    frozen_path = output / "frozen-candidates.json"
    output.mkdir(parents=True, exist_ok=True)
    dataset.save(frozen_path)
    grade_counts = Counter(
        grade for query in dataset.queries for grade in query.relevance_labels.values()
    )
    query_statistics = {
        query.query_id: {
            "category": query.category,
            "candidate_count": len(query.candidate_ids),
            "relevant_count": len(query.relevance_labels),
            "grade_4_count": sum(grade == 4 for grade in query.relevance_labels.values()),
            "grade_3_count": sum(grade == 3 for grade in query.relevance_labels.values()),
        }
        for query in dataset.queries
    }
    recall_ceilings = {
        str(cutoff): fmean(
            min(cutoff, len(query.relevance_labels)) / len(query.relevance_labels)
            for query in dataset.queries
        )
        for cutoff in (20, 50)
    }
    queries_above_cutoff = {
        str(cutoff): sum(len(query.relevance_labels) > cutoff for query in dataset.queries)
        for cutoff in (20, 50)
    }
    embedding_manifest_path = embeddings_root / "manifest.json"
    positive_qrel_count = sum(len(query.relevance_labels) for query in dataset.queries)
    payload: dict[str, object] = {
        "schema_version": SILVER_RESULT_SCHEMA_VERSION,
        "run_id": dataset.version,
        "evidence_status": "diagnostic_silver_pilot_only",
        "eligible_for_production_or_resume_metric_claims": False,
        "reporting_block_reason": REPORTING_BLOCK_REASON,
        "human_relevance_judgments": False,
        "training_performed": False,
        "learning_to_rank_status": {
            "trained": False,
            "reason": (
                "Training LambdaMART on labels generated from these same predicates would "
                "measure label reconstruction and leak the evaluation target."
            ),
        },
        "dataset": {
            "name": query_set.dataset_name,
            "version": dataset.version,
            "candidate_checksum": dataset.checksum,
            "catalog_product_count": len(records),
            "query_count": len(dataset.queries),
            "candidate_rows_across_queries": sum(
                len(query.candidate_ids) for query in dataset.queries
            ),
            "positive_qrel_count": positive_qrel_count,
            "grade_counts": {str(grade): count for grade, count in sorted(grade_counts.items())},
            "judgment_method": query_set.judgment_method,
            "query_statistics": query_statistics,
            "macro_recall_cutoff_ceilings": recall_ceilings,
            "queries_with_more_relevant_items_than_cutoff": queries_above_cutoff,
        },
        "sources": {
            "catalog": {
                "path": _portable_path(source_files[0]),
                "sha256": catalog_hash,
            },
            "query_set": {
                "path": _portable_path(queries_path),
                "sha256": query_set.source_sha256,
            },
            "embedding_manifest": {
                "path": _portable_path(embedding_manifest_path),
                "sha256": sha256_file(embedding_manifest_path),
                "content_hash": embedding_manifest.get("content_hash"),
                "index_version": embedding_manifest.get("index_version"),
                "built_on_device": embedding_encoder.get("resolved_device"),
            },
            "frozen_candidates": {
                "path": _portable_path(frozen_path),
                "sha256": sha256_file(frozen_path),
            },
        },
        "evaluation_parameters": {
            "rrf_k": rrf_k,
            "recall_cutoffs": [20, 50],
            "ndcg_cutoff": 10,
            "relevance_threshold_for_recall_and_mrr": 1,
            "query_encoder": query_encoder_metadata,
        },
        "models": {
            "stable_product_id_negative_control": {
                "basis": "ascending product ID; diagnostic negative control",
                "metrics": model_results["stable_product_id_negative_control"],
            },
            "bm25": {
                "basis": "rank_bm25 over the same labelled product text as the embedding index",
                "metrics": model_results["bm25"],
            },
            "vector_only": {
                "basis": "cosine similarity against the existing frozen embedding matrix",
                "metrics": model_results["vector_only"],
            },
            "rrf_hybrid": {
                "basis": "reciprocal rank fusion of complete BM25 and vector rankings",
                "metrics": model_results["rrf_hybrid"],
            },
        },
        "paired_baseline_comparisons": {
            "vector_only_minus_bm25": _paired_baseline_comparison(
                model_results["bm25"], model_results["vector_only"]
            ),
            "rrf_hybrid_minus_bm25": _paired_baseline_comparison(
                model_results["bm25"], model_results["rrf_hybrid"]
            ),
        },
        "limitations": [
            REPORTING_BLOCK_REASON,
            (
                "Queries cover declared catalog fields, not subjective workload value "
                "or compatibility."
            ),
            (
                "Catalog specifications are community-maintained and may contain missing "
                "or stale fields."
            ),
            "No prices, availability, observed user behavior, or reviewer agreement are evaluated.",
            (
                "Some broad queries have more silver-relevant products than a retrieval cutoff, "
                "so the artifact records the query-level macro Recall@K ceiling."
            ),
            "Heuristic and LambdaMART rankers are omitted because required independent price, "
            "performance, preference, and human-label evidence is not present in this pilot.",
        ],
    }
    payload["artifact_sha256"] = _artifact_hash(payload)
    metrics_path = output / "metrics.json"
    _atomic_json(metrics_path, payload)
    summary = {
        "metrics_path": str(metrics_path),
        "frozen_candidates_path": str(frozen_path),
        "artifact_sha256": payload["artifact_sha256"],
        "dataset_version": dataset.version,
        "candidate_checksum": dataset.checksum,
        "query_count": len(dataset.queries),
        "positive_qrel_count": positive_qrel_count,
        "eligible_for_production_or_resume_metric_claims": False,
        "aggregate_metrics": {name: result["aggregate"] for name, result in model_results.items()},
    }
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--queries", required=True, type=Path)
    parser.add_argument("--embedding-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--score-decimals", type=int, default=DEFAULT_SCORE_DECIMALS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    summary = run_retrieval_silver_pilot(
        catalog_path=args.catalog,
        query_set_path=args.queries,
        embedding_dir=args.embedding_dir,
        output_dir=args.output_dir,
        device=args.device,
        batch_size=args.batch_size,
        rrf_k=args.rrf_k,
        score_decimals=args.score_decimals,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
