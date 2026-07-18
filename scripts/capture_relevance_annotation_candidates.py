"""Create a blinded relevance-annotation candidate pool from a verified catalogue.

This command is deliberately a collection aid, not an evaluator.  It retrieves
category-scoped candidates with the development hybrid retriever, strips every
rank, score, and source-membership hint before writing reviewer input, and
binds the result to the exact catalogue and query-set bytes.  Human annotation
remains the only source of relevance grades.
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
from pc_build_recommender.evaluation.manifest import canonical_json_bytes, sha256_file, sha256_json
from pc_build_recommender.ranking import (
    RankingCandidate,
    RankingContext,
    RankingFeatureBuilder,
)
from pc_build_recommender.retrieval import (
    HybridProductRetriever,
    ProductDocument,
    RetrievedCandidate,
    SearchHit,
    StructuredFilterSpec,
)

QUERY_SET_SCHEMA_VERSION = "pc-build-recommender.relevance-annotation-query-set.v1"
CANDIDATE_SET_SCHEMA_VERSION = "pc-build-recommender.relevance-annotation-candidates.v1"
CAPTURE_MANIFEST_SCHEMA_VERSION = "pc-build-recommender.relevance-annotation-capture.v2"
CAPTURE_POLICY_VERSION = "hybrid-blinded-multi-retriever-candidate-pooling-v2"
PRELABEL_QUERY_SCHEMA_VERSION = "pc-build-recommender.ranking-prelabel-query.v1"
PRELABEL_SNAPSHOT_SCHEMA_VERSION = (
    "pc-build-recommender.ranking-prelabel-snapshot-manifest.v1"
)
CONTEXT_MAPPING_VERSION = "ranking-context-from-structured-query-v1"
PRELABEL_FEATURE_FILENAME = "prelabel-features.jsonl"
MAX_QUERY_SET_BYTES = 8 * 1025 * 1024
DEFAULT_MAX_CATALOG_RECORDS = 10_000
DEFAULT_MAX_CATALOG_BYTES = 64 * 1024 * 1024
DEFAULT_TOP_K = 20
DEFAULT_PER_SOURCE_TOP_K = 20
DEFAULT_MAX_CANDIDATES_PER_QUERY = 30
MAX_TOP_K = 50
MAX_CANDIDATES_PER_QUERY = 100
_POOL_SOURCE_NAMES = ("bm25", "vector")
_PREFERENCE_KEYS = frozenset(
    {
        "excluded_brands",
        "noise",
        "power_efficiency",
        "preferred_brands",
        "upgradeability",
    }
)


class RelevanceCandidateCaptureError(ValueError):
    """Raised when a candidate pool cannot be produced safely and reproducibly."""


@dataclass(frozen=True, slots=True)
class _Query:
    query_id: str
    query_group_id: str
    category: str
    query_text: str
    structured_constraints: dict[str, Any]


@dataclass(frozen=True, slots=True)
class _CatalogueProduct:
    document: ProductDocument
    evidence_payload: dict[str, Any]
    provenance: dict[str, str]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-set", type=Path, required=True)
    parser.add_argument("--catalog-records", type=Path, required=True)
    parser.add_argument("--catalog-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Maximum number of fused RRF candidates retained for each query.",
    )
    parser.add_argument(
        "--per-source-top-k",
        type=int,
        default=DEFAULT_PER_SOURCE_TOP_K,
        help="Maximum candidate depth retrieved from each individual discovery source.",
    )
    parser.add_argument(
        "--max-candidates-per-query",
        type=int,
        default=DEFAULT_MAX_CANDIDATES_PER_QUERY,
        help="Bounded total candidate pool after fused and source-exclusive discovery.",
    )
    parser.add_argument("--max-catalog-records", type=int, default=DEFAULT_MAX_CATALOG_RECORDS)
    parser.add_argument("--max-catalog-bytes", type=int, default=DEFAULT_MAX_CATALOG_BYTES)
    return parser


def _object(value: object, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RelevanceCandidateCaptureError(f"{name} must be an object")
    return {str(key): nested for key, nested in value.items()}


def _string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RelevanceCandidateCaptureError(f"{name} must be a non-empty string")
    result = value.strip()
    if result.casefold().startswith("replace"):
        raise RelevanceCandidateCaptureError(
            f"{name} contains a template placeholder; replace it with collection input"
        )
    return result


def _explicit_false(value: object, *, name: str) -> None:
    if value is not False:
        raise RelevanceCandidateCaptureError(f"{name} must be explicitly false")


def _only_keys(payload: Mapping[str, Any], *, allowed: set[str], name: str) -> None:
    unexpected = sorted(set(payload).difference(allowed))
    missing = sorted(allowed.difference(payload))
    if unexpected:
        raise RelevanceCandidateCaptureError(
            f"{name} contains unsupported fields: {', '.join(unexpected)}"
        )
    if missing:
        raise RelevanceCandidateCaptureError(
            f"{name} is missing required fields: {', '.join(missing)}"
        )


def _json_object(value: object, *, name: str) -> dict[str, Any]:
    payload = _object(value, name=name)
    try:
        return _object(json.loads(canonical_json_bytes(payload)), name=name)
    except (TypeError, ValueError) as error:
        raise RelevanceCandidateCaptureError(f"{name} must contain finite JSON data") from error


def _source_policy(value: object) -> dict[str, Any]:
    policy = _json_object(value, name="source_policy")
    for name in ("training_eligible", "published_metrics_eligible"):
        if policy.get(name) is not True:
            raise RelevanceCandidateCaptureError(
                f"source_policy.{name} must be true for human relevance collection"
            )
    serving = policy.get("model_serving_eligible", False)
    if not isinstance(serving, bool):
        raise RelevanceCandidateCaptureError(
            "source_policy.model_serving_eligible must be a boolean when provided"
        )
    if serving and (
        not isinstance(policy.get("serving_attribution_notice"), str)
        or not policy["serving_attribution_notice"].strip()
    ):
        raise RelevanceCandidateCaptureError(
            "source_policy.serving_attribution_notice must be a non-empty string "
            "when derived-model serving is eligible"
        )
    if not isinstance(policy.get("scope_note"), str) or not policy["scope_note"].strip():
        raise RelevanceCandidateCaptureError(
            "source_policy.scope_note must be a non-empty string"
        )
    policy["model_serving_eligible"] = serving
    return policy


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
            "is_synthetic",
        },
        name=name,
    )
    _explicit_false(payload["is_synthetic"], name=f"{name}.is_synthetic")
    constraints = _json_object(
        payload["structured_constraints"], name=f"{name}.structured_constraints"
    )
    try:
        validate_blinded_annotation_payload(constraints, path=f"{name}.structured_constraints")
    except ValueError as error:
        raise RelevanceCandidateCaptureError(str(error)) from error
    return _Query(
        query_id=_string(payload["query_id"], name=f"{name}.query_id"),
        query_group_id=_string(payload["query_group_id"], name=f"{name}.query_group_id"),
        category=_string(payload["category"], name=f"{name}.category").casefold(),
        query_text=_string(payload["query_text"], name=f"{name}.query_text"),
        structured_constraints=constraints,
    )


def _load_query_set(path: Path) -> tuple[dict[str, Any], tuple[_Query, ...], dict[str, Any]]:
    resolved = path.resolve(strict=True)
    if resolved.stat().st_size > MAX_QUERY_SET_BYTES:
        raise RelevanceCandidateCaptureError(
            f"query set exceeds the {MAX_QUERY_SET_BYTES} byte safety limit"
        )
    try:
        raw = json.loads(resolved.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise RelevanceCandidateCaptureError(f"query set is not valid JSON: {error.msg}") from error
    payload = _object(raw, name="query set")
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
        name="query set",
    )
    if payload["schema_version"] != QUERY_SET_SCHEMA_VERSION:
        raise RelevanceCandidateCaptureError(
            f"query set schema_version must be {QUERY_SET_SCHEMA_VERSION!r}"
        )
    raw_queries = payload["queries"]
    if not isinstance(raw_queries, list) or not raw_queries:
        raise RelevanceCandidateCaptureError("query set queries must be a non-empty array")
    queries = tuple(_query(query, index=index) for index, query in enumerate(raw_queries))
    query_ids = [query.query_id for query in queries]
    if len(query_ids) != len(set(query_ids)):
        raise RelevanceCandidateCaptureError("query set query_id values must be unique")
    if len({query.query_group_id for query in queries}) < 3:
        raise RelevanceCandidateCaptureError(
            "query set requires at least three distinct query_group_id values for later splitting"
        )
    metadata = {
        "dataset_name": _string(payload["dataset_name"], name="query set dataset_name"),
        "dataset_version": _string(
            payload["dataset_version"], name="query set dataset_version"
        ),
        "rubric_version": _string(payload["rubric_version"], name="query set rubric_version"),
        "data_version": _string(payload["data_version"], name="query set data_version"),
        "source_policy": _source_policy(payload["source_policy"]),
    }
    return metadata, tuple(sorted(queries, key=lambda query: query.query_id)), payload


def _verify_catalogue_manifest(records_path: Path, manifest_path: Path) -> dict[str, Any]:
    manifest = _json_object(
        json.loads(manifest_path.resolve(strict=True).read_text(encoding="utf-8")),
        name="catalog manifest",
    )
    files = manifest.get("files")
    if not isinstance(files, Mapping) or not isinstance(files.get("records.jsonl"), Mapping):
        raise RelevanceCandidateCaptureError("catalog manifest lacks files.records.jsonl")
    expected_sha256 = files["records.jsonl"].get("sha256")
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise RelevanceCandidateCaptureError("catalog manifest records.jsonl SHA-256 is invalid")
    actual_sha256 = sha256_file(records_path)
    if actual_sha256 != expected_sha256:
        raise RelevanceCandidateCaptureError(
            "catalog records SHA-256 does not match the supplied catalog manifest"
        )
    return manifest


def _candidate_provenance(value: object, *, product_id: str) -> dict[str, str]:
    if not isinstance(value, list):
        raise RelevanceCandidateCaptureError(f"catalogue product {product_id} lacks provenance")
    candidates: list[dict[str, str]] = []
    for raw in value:
        if not isinstance(raw, Mapping):
            continue
        try:
            source_url = _string(raw.get("source_url"), name=f"product {product_id} source URL")
            if not source_url.startswith("https://"):
                continue
            candidates.append(
                {
                    "source_name": _string(
                        raw.get("source_name"), name=f"product {product_id} source name"
                    ),
                    "source_url": source_url,
                    "license_or_access_note": _string(
                        raw.get("licence_or_access_note"),
                        name=f"product {product_id} licence note",
                    ),
                    "retrieved_at": _string(
                        raw.get("retrieved_at"),
                        name=f"product {product_id} retrieval time",
                    ),
                }
            )
        except RelevanceCandidateCaptureError:
            continue
    if not candidates:
        raise RelevanceCandidateCaptureError(
            f"catalogue product {product_id} has no reviewable HTTPS provenance"
        )
    return min(candidates, key=lambda item: (item["source_url"], item["source_name"]))


def _read_catalogue(
    records_path: Path,
    *,
    max_records: int,
    max_bytes: int,
) -> tuple[dict[str, _CatalogueProduct], dict[str, int]]:
    if max_records < 1 or max_bytes < 1:
        raise ValueError("catalogue limits must be positive")
    if records_path.stat().st_size > max_bytes:
        raise MemoryError(f"catalog records exceed the {max_bytes} byte safety limit")
    products: dict[str, _CatalogueProduct] = {}
    stats = {"examined_records": 0, "accepted_products": 0, "rejected_products": 0}
    with records_path.open("rb") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            stats["examined_records"] += 1
            if stats["examined_records"] > max_records:
                raise MemoryError(f"catalogue exceeds the {max_records} record safety limit")
            try:
                envelope = _object(json.loads(line), name=f"catalogue line {line_number}")
                if envelope.get("record_type") != "canonical_product":
                    raise RelevanceCandidateCaptureError("record is not a canonical product")
                if envelope.get("training_eligible") is not True:
                    raise RelevanceCandidateCaptureError("record is not training eligible")
                if envelope.get("published_claims_eligible") is not True:
                    raise RelevanceCandidateCaptureError(
                        "record is not eligible for published metrics"
                    )
                data = _json_object(envelope.get("data"), name=f"catalogue line {line_number}.data")
                document = ProductDocument.from_mapping(data)
                provenance = _candidate_provenance(
                    data.get("provenance"), product_id=document.product_id
                )
                evidence = {
                    name: data.get(name)
                    for name in (
                        "canonical_name",
                        "brand",
                        "model",
                        "manufacturer_part_number",
                        "gtin",
                        "common_attributes",
                        "category_attributes",
                        "source_confidence",
                    )
                    if data.get(name) is not None
                }
                validate_blinded_annotation_payload(
                    evidence, path=f"catalogue product {document.product_id}"
                )
                if document.product_id in products:
                    raise RelevanceCandidateCaptureError("duplicate canonical product ID")
                products[document.product_id] = _CatalogueProduct(
                    document=document,
                    evidence_payload=evidence,
                    provenance=provenance,
                )
                stats["accepted_products"] += 1
            except (TypeError, ValueError, RelevanceCandidateCaptureError):
                stats["rejected_products"] += 1
    if not products:
        raise RelevanceCandidateCaptureError(
            "catalogue contained no rights-cleared reviewable products"
        )
    return products, stats


def _select_pooled_candidate_ids(
    *,
    fused_candidates: Sequence[RetrievedCandidate],
    source_pools: Mapping[str, Sequence[SearchHit]],
    max_candidates: int,
) -> tuple[tuple[str, ...], dict[str, int]]:
    """Select fused candidates then source-exclusive candidates round-robin.

    The returned order is an internal collection decision only.  The caller
    subsequently sorts the reviewer-facing records by canonical product ID and
    does not serialize retrieval ranks, scores, or source membership.
    """

    if max_candidates < 1:
        raise ValueError("max_candidates must be positive")
    if len(fused_candidates) > max_candidates:
        raise ValueError("max_candidates must retain every fused candidate")

    selected_ids: list[str] = []
    selected_set: set[str] = set()
    for candidate in fused_candidates:
        if candidate.product_id not in selected_set:
            selected_ids.append(candidate.product_id)
            selected_set.add(candidate.product_id)

    source_positions = {source: 0 for source in _POOL_SOURCE_NAMES}
    source_additions = {source: 0 for source in _POOL_SOURCE_NAMES}
    while len(selected_ids) < max_candidates:
        added_this_round = False
        for source in _POOL_SOURCE_NAMES:
            hits = source_pools.get(source, ())
            position = source_positions[source]
            while position < len(hits) and hits[position].product_id in selected_set:
                position += 1
            source_positions[source] = position
            if position >= len(hits):
                continue

            product_id = hits[position].product_id
            source_positions[source] = position + 1
            selected_ids.append(product_id)
            selected_set.add(product_id)
            source_additions[source] += 1
            added_this_round = True
            if len(selected_ids) >= max_candidates:
                break
        if not added_this_round:
            break

    return (
        tuple(selected_ids),
        {
            "rrf_candidates": len(fused_candidates),
            "bm25_candidates": len(source_pools.get("bm25", ())),
            "vector_candidates": len(source_pools.get("vector", ())),
            "bm25_source_exclusive_added": source_additions["bm25"],
            "vector_source_exclusive_added": source_additions["vector"],
            "pooled_candidates": len(selected_ids),
        },
    )


def _workload_weights(constraints: dict[str, Any]) -> dict[str, float]:
    raw_weights = constraints.pop("workload_weights", None)
    if isinstance(raw_weights, Mapping):
        weights = {
            str(name): float(weight)
            for name, weight in raw_weights.items()
            if not isinstance(weight, bool)
            and isinstance(weight, int | float)
            and float(weight) >= 0.0
        }
        if weights and sum(weights.values()) > 0:
            return weights

    raw_workload = constraints.pop("workload", None)
    if isinstance(raw_workload, str) and raw_workload.strip():
        return {raw_workload.strip(): 1.0}
    raw_workloads = constraints.pop("workloads", None)
    if isinstance(raw_workloads, list):
        list_weights: dict[str, float] = {}
        for item in raw_workloads:
            if isinstance(item, str) and item.strip():
                list_weights[item.strip()] = 1.0
            elif isinstance(item, Mapping):
                name = item.get("name")
                weight = item.get("weight", 1.0)
                if (
                    isinstance(name, str)
                    and name.strip()
                    and not isinstance(weight, bool)
                    and isinstance(weight, int | float)
                    and float(weight) >= 0.0
                ):
                    list_weights[name.strip()] = float(weight)
        if list_weights and sum(list_weights.values()) > 0:
            return list_weights
    return {}


def _ranking_context(
    query: _Query,
    *,
    metadata: Mapping[str, Any],
) -> RankingContext:
    constraints = dict(query.structured_constraints)
    raw_budget = constraints.pop("budget_sgd", None)
    budget_sgd = (
        float(raw_budget)
        if not isinstance(raw_budget, bool)
        and isinstance(raw_budget, int | float)
        and float(raw_budget) > 0.0
        else None
    )
    workload_weights = _workload_weights(constraints)
    preferences = {
        key: constraints.pop(key)
        for key in sorted(_PREFERENCE_KEYS)
        if key in constraints
    }
    return RankingContext(
        query_id=query.query_id,
        query_text=query.query_text,
        budget_sgd=budget_sgd,
        workload_weights=workload_weights,
        requirements=constraints,
        preferences=preferences,
        data_version=str(metadata["data_version"]),
        candidate_set_version=str(metadata["dataset_version"]),
    )


def _context_payload(context: RankingContext) -> dict[str, Any]:
    return {
        "query_id": context.query_id,
        "query_text": context.query_text,
        "budget_sgd": context.budget_sgd,
        "workload_weights": dict(context.workload_weights),
        "requirements": dict(context.requirements),
        "preferences": dict(context.preferences),
        "data_version": context.data_version,
        "candidate_set_version": context.candidate_set_version,
    }


def _ranking_candidate(
    *,
    product: _CatalogueProduct,
    bm25_hit: SearchHit | None,
    vector_hit: SearchHit | None,
    fused_candidate: RetrievedCandidate | None,
) -> RankingCandidate:
    evidence = product.evidence_payload
    attributes: dict[str, Any] = {}
    for field in ("common_attributes", "category_attributes"):
        value = evidence.get(field)
        if isinstance(value, Mapping):
            attributes.update({str(key): nested for key, nested in value.items()})
    for field in ("canonical_name", "model", "manufacturer_part_number", "gtin"):
        if evidence.get(field) is not None:
            attributes[field] = evidence[field]
    bm25_score = bm25_hit.score if bm25_hit is not None else 0.0
    return RankingCandidate(
        product_id=product.document.product_id,
        category=product.document.category,
        price_sgd=product.document.price_sgd,
        brand=product.document.brand,
        retrieval_scores={
            "bm25_score": bm25_score,
            "lexical_score": bm25_score,
            "vector_similarity": vector_hit.score if vector_hit is not None else 0.0,
            "rrf_score": fused_candidate.rrf_score if fused_candidate is not None else 0.0,
        },
        workload_scores={},
        signals={},
        attributes=attributes,
    )


def _ranking_candidate_payload(candidate: RankingCandidate) -> dict[str, Any]:
    return {
        "product_id": candidate.product_id,
        "category": candidate.category,
        "price_sgd": candidate.price_sgd,
        "brand": candidate.brand,
        "retrieval_scores": dict(candidate.retrieval_scores),
        "workload_scores": dict(candidate.workload_scores),
        "signals": dict(candidate.signals),
        "attributes": dict(candidate.attributes),
    }


def _prelabel_query_row(
    *,
    query: _Query,
    context: RankingContext,
    candidates: Sequence[RankingCandidate],
    feature_builder: RankingFeatureBuilder,
) -> dict[str, Any]:
    ordered = tuple(sorted(candidates, key=lambda candidate: candidate.product_id))
    feature_batch = feature_builder.build(context, ordered)
    candidate_ids = [candidate.product_id for candidate in ordered]
    matrix_payload = {
        "feature_version": feature_builder.feature_version,
        "feature_names": list(feature_builder.feature_names),
        "query_id": query.query_id,
        "rows": [
            {
                "product_id": product_id,
                "values_hex": [float(value).hex() for value in feature_batch.values[index]],
            }
            for index, product_id in enumerate(feature_batch.product_ids)
        ],
    }
    return {
        "schema_version": PRELABEL_QUERY_SCHEMA_VERSION,
        "query_id": query.query_id,
        "query_group_id": query.query_group_id,
        "category": query.category,
        "context": _context_payload(context),
        "candidates": [_ranking_candidate_payload(candidate) for candidate in ordered],
        "candidate_ids_sha256": sha256_json(candidate_ids),
        "feature_matrix_sha256": sha256_json(matrix_payload),
    }


def _candidate_set(
    *,
    metadata: Mapping[str, Any],
    queries: Sequence[_Query],
    products: Mapping[str, _CatalogueProduct],
    top_k: int,
    per_source_top_k: int,
    max_candidates_per_query: int,
) -> tuple[
    dict[str, Any],
    dict[str, str],
    dict[str, dict[str, int]],
    tuple[dict[str, Any], ...],
]:
    retriever = HybridProductRetriever([product.document for product in products.values()])
    feature_builder = RankingFeatureBuilder()
    query_payloads: list[dict[str, Any]] = []
    prelabel_rows: list[dict[str, Any]] = []
    candidate_digests: dict[str, str] = {}
    pooling_counts: dict[str, dict[str, int]] = {}
    for query in queries:
        fused_candidates, source_pools = retriever.retrieve_with_source_pools(
            query.query_text,
            category=query.category,
            filters=StructuredFilterSpec(in_stock_only=False),
            top_k=top_k,
            per_source_k=per_source_top_k,
        )
        selected_ids, query_pooling_counts = _select_pooled_candidate_ids(
            fused_candidates=fused_candidates,
            source_pools=source_pools,
            max_candidates=max_candidates_per_query,
        )
        if not selected_ids:
            raise RelevanceCandidateCaptureError(
                f"no rights-cleared catalogue candidates for query {query.query_id!r}"
            )
        unknown_product_ids = sorted(set(selected_ids).difference(products))
        if unknown_product_ids:
            raise RelevanceCandidateCaptureError(
                "retrieval returned candidates outside the verified catalogue: "
                + ", ".join(unknown_product_ids)
            )
        candidates = [
            {
                "product_id": product_id,
                "evidence_payload": products[product_id].evidence_payload,
                "provenance": products[product_id].provenance,
                "is_synthetic": False,
            }
            for product_id in selected_ids
        ]
        candidate_digests[query.query_id] = sha256_json(
            sorted(str(candidate["product_id"]) for candidate in candidates)
        )
        pooling_counts[query.query_id] = query_pooling_counts
        fused_by_id = {candidate.product_id: candidate for candidate in fused_candidates}
        bm25_by_id = {
            candidate.product_id: candidate for candidate in source_pools.get("bm25", ())
        }
        vector_by_id = {
            candidate.product_id: candidate for candidate in source_pools.get("vector", ())
        }
        ranking_candidates = tuple(
            _ranking_candidate(
                product=products[product_id],
                bm25_hit=bm25_by_id.get(product_id),
                vector_hit=vector_by_id.get(product_id),
                fused_candidate=fused_by_id.get(product_id),
            )
            for product_id in selected_ids
        )
        prelabel_rows.append(
            _prelabel_query_row(
                query=query,
                context=_ranking_context(query, metadata=metadata),
                candidates=ranking_candidates,
                feature_builder=feature_builder,
            )
        )
        # Reviewer-facing candidate order is stable but unrelated to retrieval rank.
        candidates.sort(key=lambda candidate: str(candidate["product_id"]))
        query_payloads.append(
            {
                "query_id": query.query_id,
                "query_group_id": query.query_group_id,
                "category": query.category,
                "query_text": query.query_text,
                "structured_constraints": query.structured_constraints,
                "candidates": candidates,
                "is_synthetic": False,
            }
        )
    return (
        {
            "schema_version": CANDIDATE_SET_SCHEMA_VERSION,
            **metadata,
            "queries": query_payloads,
        },
        candidate_digests,
        pooling_counts,
        tuple(sorted(prelabel_rows, key=lambda row: str(row["query_id"]))),
    )


def _write_output(
    *,
    output_dir: Path,
    candidates: Mapping[str, Any],
    manifest: Mapping[str, Any],
    prelabel_feature_bytes: bytes,
) -> None:
    destination = output_dir.resolve()
    if destination.exists():
        raise RelevanceCandidateCaptureError(
            f"output directory already exists and will not be overwritten: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        (temporary / "candidates.json").write_bytes(canonical_json_bytes(candidates) + b"\n")
        (temporary / PRELABEL_FEATURE_FILENAME).write_bytes(prelabel_feature_bytes)
        (temporary / "manifest.json").write_bytes(canonical_json_bytes(manifest) + b"\n")
        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def capture_relevance_annotation_candidates(
    *,
    query_set_path: Path,
    catalog_records_path: Path,
    catalog_manifest_path: Path,
    output_dir: Path,
    top_k: int = DEFAULT_TOP_K,
    per_source_top_k: int = DEFAULT_PER_SOURCE_TOP_K,
    max_candidates_per_query: int = DEFAULT_MAX_CANDIDATES_PER_QUERY,
    max_catalog_records: int = DEFAULT_MAX_CATALOG_RECORDS,
    max_catalog_bytes: int = DEFAULT_MAX_CATALOG_BYTES,
) -> dict[str, Any]:
    """Capture a rights-cleared, score-blinded candidate universe for human review."""

    if not 1 <= top_k <= MAX_TOP_K:
        raise ValueError(f"top_k must be between 1 and {MAX_TOP_K}")
    if not 1 <= per_source_top_k <= MAX_TOP_K:
        raise ValueError(f"per_source_top_k must be between 1 and {MAX_TOP_K}")
    if not 1 <= max_candidates_per_query <= MAX_CANDIDATES_PER_QUERY:
        raise ValueError(
            "max_candidates_per_query must be between 1 and "
            f"{MAX_CANDIDATES_PER_QUERY}"
        )
    if max_candidates_per_query < top_k:
        raise ValueError(
            "max_candidates_per_query must be at least top_k so every fused candidate is retained"
        )
    records = catalog_records_path.resolve(strict=True)
    manifest_path = catalog_manifest_path.resolve(strict=True)
    catalog_manifest = _verify_catalogue_manifest(records, manifest_path)
    metadata, queries, query_set = _load_query_set(query_set_path)
    products, catalog_stats = _read_catalogue(
        records,
        max_records=max_catalog_records,
        max_bytes=max_catalog_bytes,
    )
    candidates, candidate_digests, pooling_counts, prelabel_rows = _candidate_set(
        metadata=metadata,
        queries=queries,
        products=products,
        top_k=top_k,
        per_source_top_k=per_source_top_k,
        max_candidates_per_query=max_candidates_per_query,
    )
    prelabel_feature_bytes = b"".join(
        canonical_json_bytes(row) + b"\n" for row in prelabel_rows
    )
    feature_builder = RankingFeatureBuilder()
    feature_contract = {
        "feature_version": feature_builder.feature_version,
        "feature_names": list(feature_builder.feature_names),
        "context_mapping_version": CONTEXT_MAPPING_VERSION,
        "missing_value_policy": "RankingFeatureBuilder defaults; no post-label imputation",
        "contains_relevance_labels": False,
        "label_free_by_construction": True,
    }
    candidate_universe = [
        {
            "query_id": row["query_id"],
            "query_group_id": row["query_group_id"],
            "category": row["category"],
            "candidate_ids_sha256": row["candidate_ids_sha256"],
        }
        for row in prelabel_rows
    ]
    prelabel_snapshot = {
        "schema_version": PRELABEL_SNAPSHOT_SCHEMA_VERSION,
        "file_name": PRELABEL_FEATURE_FILENAME,
        "file_sha256": hashlib.sha256(prelabel_feature_bytes).hexdigest(),
        "size_bytes": len(prelabel_feature_bytes),
        "query_row_sha256": {
            str(row["query_id"]): sha256_json(row) for row in prelabel_rows
        },
        "candidate_universe_sha256": sha256_json(candidate_universe),
        "feature_contract": feature_contract,
        "feature_contract_sha256": sha256_json(feature_contract),
        "label_state": "absent",
        "promotion_eligible_retrieval": False,
        "promotion_block_reasons": [
            "candidate capture uses the deterministic stable-hash development vector backend"
        ],
    }
    prelabel_snapshot["snapshot_sha256"] = sha256_json(prelabel_snapshot)
    manifest = {
        "schema_version": CAPTURE_MANIFEST_SCHEMA_VERSION,
        "capture_policy_version": CAPTURE_POLICY_VERSION,
        "query_set_sha256": sha256_json(query_set),
        "catalog_records_sha256": sha256_file(records),
        "catalog_manifest_sha256": sha256_file(manifest_path),
        "catalog_manifest_content_sha256": catalog_manifest.get("content_sha256"),
        "source_policy_sha256": sha256_json(metadata["source_policy"]),
        "candidate_set_sha256": sha256_json(candidates),
        "candidate_file_sha256": hashlib.sha256(
            canonical_json_bytes(candidates) + b"\n"
        ).hexdigest(),
        "query_count": len(queries),
        "candidate_count": sum(len(query["candidates"]) for query in candidates["queries"]),
        "top_k_per_query": top_k,
        "per_source_top_k": per_source_top_k,
        "max_candidates_per_query": max_candidates_per_query,
        "catalogue": catalog_stats,
        "reviewer_blinding": {
            "retrieval_scores_excluded": True,
            "retrieval_ranks_excluded": True,
            "retrieval_source_membership_excluded": True,
            "reviewer_file": "candidates.json",
            "prelabel_feature_file_is_not_reviewer_input": True,
            "candidate_identifiers_sha256": candidate_digests,
        },
        "retrieval_pooling": {
            "model": "bm25+stable-hash-vector+rrf-development-v1",
            "scope": "candidate discovery only; not a serving or evaluation claim",
            "strategy": "rrf-plus-source-union",
            "bm25_implementation": "rank_bm25.BM25Okapi",
            "tokenizer": "pc_build_recommender.retrieval.text.tokenize",
            "vector_backend": "StableHashEmbeddingEncoder",
            "vector_model": "stable-lexical-hash-v1-512",
            "vector_dimension": 512,
            "rrf_k": 60,
            "structured_filter_policy": "category + in_stock_only=false",
            "fused_top_k_per_query": top_k,
            "per_source_top_k": per_source_top_k,
            "max_candidates_per_query": max_candidates_per_query,
            "source_names": ["bm25", "vector", "rrf"],
            "per_query_counts": pooling_counts,
        },
        "prelabel_ranking_snapshot": prelabel_snapshot,
    }
    _write_output(
        output_dir=output_dir,
        candidates=candidates,
        manifest=manifest,
        prelabel_feature_bytes=prelabel_feature_bytes,
    )
    return {
        "schema_version": CAPTURE_MANIFEST_SCHEMA_VERSION,
        "status": "ok",
        "output_dir": str(output_dir.resolve()),
        "manifest_sha256": sha256_json(manifest),
        "query_count": manifest["query_count"],
        "candidate_count": manifest["candidate_count"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = capture_relevance_annotation_candidates(
            query_set_path=args.query_set,
            catalog_records_path=args.catalog_records,
            catalog_manifest_path=args.catalog_manifest,
            output_dir=args.output_dir,
            top_k=args.top_k,
            per_source_top_k=args.per_source_top_k,
            max_candidates_per_query=args.max_candidates_per_query,
            max_catalog_records=args.max_catalog_records,
            max_catalog_bytes=args.max_catalog_bytes,
        )
    except (OSError, RelevanceCandidateCaptureError, ValueError, MemoryError) as error:
        print(
            json.dumps({"status": "error", "message": str(error)}, sort_keys=True),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
