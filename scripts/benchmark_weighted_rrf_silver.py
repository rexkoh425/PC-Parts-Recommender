"""Tune weighted RRF on silver-query validation and report one sealed holdout."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean
from typing import Any

from pc_build_recommender.evaluation.manifest import sha256_json
from pc_build_recommender.evaluation.metrics import bootstrap_confidence_interval
from pc_build_recommender.evaluation.retrieval_silver_pilot import (
    _complete_bm25_rankings,
    _complete_vector_rankings,
    _load_embedding_artifact,
)
from pc_build_recommender.evaluation.splits import assert_group_disjoint, deterministic_group_split
from pc_build_recommender.retrieval import (
    FrozenCandidateSet,
    ProductDocument,
    ndcg_at_k,
    reciprocal_rank_fusion,
)
from pc_build_recommender.retrieval.embedding_index import (
    build_product_embedding_text,
    load_normalized_product_jsonl,
)

DEFAULT_CATALOG = Path(
    "data/processed/buildcores_open_db/"
    "f3ee75dd07ffdd7725da7b056229e0df12838c571b2372bd59563f3a79fd383f/full/records.jsonl"
)
DEFAULT_QRELS = Path("artifacts/evaluation/retrieval-silver-full-v2/frozen-candidates.json")
DEFAULT_EMBEDDINGS = Path("artifacts/retrieval/buildcores-full-embeddings-pinned")
WEIGHT_GRID = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
K_GRID = (10, 20, 40, 60, 100)
SEED = 20261108
BOOTSTRAP_RESAMPLES = 1_000


def _rankings(
    dataset: FrozenCandidateSet,
    bm25: dict[str, list[str]],
    vector: dict[str, list[str]],
    *,
    bm25_weight: float,
    k: int,
) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for query in dataset.queries:
        fused = reciprocal_rank_fusion(
            {"bm25": bm25[query.query_id], "vector": vector[query.query_id]},
            weights={"bm25": bm25_weight, "vector": 1.0},
            k=k,
            limit=len(query.candidate_ids),
        )
        ranking = [hit.product_id for hit in fused]
        if len(ranking) != len(query.candidate_ids) or set(ranking) != set(query.candidate_ids):
            raise ValueError(f"weighted RRF changed the frozen candidate set for {query.query_id}")
        result[query.query_id] = ranking
    return result


def _query_ndcg(dataset: FrozenCandidateSet, rankings: dict[str, list[str]]) -> dict[str, float]:
    return {
        query.query_id: ndcg_at_k(
            {
                product_id: query.relevance_labels.get(product_id, 0)
                for product_id in query.candidate_ids
            },
            rankings[query.query_id],
            k=10,
        )
        for query in dataset.queries
    }


def _mean_for_split(values: dict[str, float], query_ids: list[str]) -> float:
    return fmean(values[query_id] for query_id in query_ids)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _build_documents(
    catalog_path: Path,
) -> tuple[list[dict[str, Any]], tuple[ProductDocument, ...]]:
    records, source_files = load_normalized_product_jsonl(catalog_path)
    if len(source_files) != 1:
        raise ValueError("expected one frozen BuildCores catalogue file")
    docs = tuple(
        ProductDocument(
            product_id=str(row["product_id"]),
            category=str(row["category"]),
            text=build_product_embedding_text(row),
            brand=str(row["brand"]) if row.get("brand") is not None else None,
            attributes={
                **dict(row.get("common_attributes", {})),
                **dict(row.get("category_attributes", {})),
            },
        )
        for row in records
    )
    return records, docs


def run_experiment(
    *, catalog_path: Path, qrels_path: Path, embedding_dir: Path, output_dir: Path, device: str
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    dataset = FrozenCandidateSet.load(qrels_path)
    if dataset.label_source.value != "silver":
        raise ValueError("this diagnostic is specifically scoped to the existing silver qrels")
    groups = [query.query_id for query in dataset.queries]
    strata = {query.query_id: str(query.category) for query in dataset.queries}
    split = deterministic_group_split(
        groups, weights={"train": 0.6, "validation": 0.2, "test": 0.2}, seed=SEED, strata=strata
    )
    assignments = {group: split.split_for(group) for group in groups}
    query_ids_by_split = {
        name: [query.query_id for query in dataset.queries if assignments[query.query_id] == name]
        for name in ("train", "validation", "test")
    }
    assert_group_disjoint(
        groups,
        [assignments[group] for group in groups],
    )
    if any(not query_ids_by_split[name] for name in query_ids_by_split):
        raise ValueError("intent-disjoint train/validation/test split produced an empty split")

    records, documents = _build_documents(catalog_path)
    bm25 = _complete_bm25_rankings(dataset, documents)
    matrix, id_rows, embedding_manifest = _load_embedding_artifact(embedding_dir)
    encoder = embedding_manifest["encoder"]
    vector, encoder_metadata = _complete_vector_rankings(
        dataset,
        matrix=matrix,
        id_rows=id_rows,
        model_name=str(encoder["model_name"]),
        device=device,
        batch_size=64,
        score_decimals=8,
    )
    baseline = _rankings(dataset, bm25, vector, bm25_weight=1.0, k=60)
    baseline_values = _query_ndcg(dataset, baseline)

    tuning: list[dict[str, Any]] = []
    for bm25_weight in WEIGHT_GRID:
        for k in K_GRID:
            candidate_rankings = _rankings(dataset, bm25, vector, bm25_weight=bm25_weight, k=k)
            values = _query_ndcg(dataset, candidate_rankings)
            tuning.append(
                {
                    "bm25_weight": bm25_weight,
                    "vector_weight": 1.0,
                    "rrf_k": k,
                    "validation_ndcg_at_10": _mean_for_split(
                        values, query_ids_by_split["validation"]
                    ),
                    "train_ndcg_at_10": _mean_for_split(values, query_ids_by_split["train"]),
                }
            )
    tuning.sort(
        key=lambda row: (
            -row["validation_ndcg_at_10"],
            abs(row["bm25_weight"] - 1.0),
            abs(row["rrf_k"] - 60),
        )
    )
    selected = tuning[0]
    challenger = _rankings(
        dataset,
        bm25,
        vector,
        bm25_weight=float(selected["bm25_weight"]),
        k=int(selected["rrf_k"]),
    )
    challenger_values = _query_ndcg(dataset, challenger)
    test_ids = query_ids_by_split["test"]
    baseline_test = [baseline_values[query_id] for query_id in test_ids]
    challenger_test = [challenger_values[query_id] for query_id in test_ids]
    deltas = [new - old for old, new in zip(baseline_test, challenger_test, strict=True)]
    ci_low, ci_high = bootstrap_confidence_interval(
        deltas, n_resamples=BOOTSTRAP_RESAMPLES, seed=SEED
    )
    baseline_ndcg = fmean(baseline_test)
    challenger_ndcg = fmean(challenger_test)
    improvement = (challenger_ndcg - baseline_ndcg) / baseline_ndcg if baseline_ndcg > 0 else None

    categories: dict[str, dict[str, float]] = {}
    for query in dataset.queries:
        if query.query_id not in test_ids:
            continue
        category = str(query.category)
        categories[category] = {
            "baseline_ndcg_at_10": baseline_values[query.query_id],
            "weighted_rrf_ndcg_at_10": challenger_values[query.query_id],
            "delta": challenger_values[query.query_id] - baseline_values[query.query_id],
        }

    summary: dict[str, Any] = {
        "schema_version": "pc-build-recommender.weighted-rrf-silver-diagnostic.v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "git_revision": _git_revision(),
        "label_source": dataset.label_source.value,
        "eligible_for_production_or_resume_claims": False,
        "claim_boundary": "32-query deterministic specification-silver diagnostic only",
        "catalogue_count": len(records),
        "catalogue_sha256": _sha256_file(catalog_path),
        "qrels_sha256": _sha256_file(qrels_path),
        "embedding_manifest_sha256": _sha256_file(embedding_dir / "manifest.json"),
        "query_count": len(dataset.queries),
        "query_product_pair_count": sum(len(query.candidate_ids) for query in dataset.queries),
        "split_seed": SEED,
        "split_sizes": {name: len(ids) for name, ids in query_ids_by_split.items()},
        "split_assignments": assignments,
        "candidate_set_fixed": True,
        "retrieval_first_stage_changed": False,
        "query_encoder": encoder_metadata,
        "tuning_grid_count": len(tuning),
        "selected_on_validation_only": {
            "bm25_weight": selected["bm25_weight"],
            "vector_weight": selected["vector_weight"],
            "rrf_k": selected["rrf_k"],
            "validation_ndcg_at_10": selected["validation_ndcg_at_10"],
        },
        "test": {
            "query_count": len(test_ids),
            "baseline_rrf_ndcg_at_10": baseline_ndcg,
            "weighted_rrf_ndcg_at_10": challenger_ndcg,
            "absolute_delta": challenger_ndcg - baseline_ndcg,
            "relative_improvement": improvement,
            "paired_bootstrap_95_ci_absolute_delta": [ci_low, ci_high],
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "query_win_rate": fmean(delta > 0 for delta in deltas),
            "by_category": categories,
            "baseline_per_query": {query_id: baseline_values[query_id] for query_id in test_ids},
            "weighted_rrf_per_query": {
                query_id: challenger_values[query_id] for query_id in test_ids
            },
        },
        "baseline_configuration": {"bm25_weight": 1.0, "vector_weight": 1.0, "rrf_k": 60},
        "tuning_results": tuning,
        "random_seed": SEED,
        "limitations": [
            "Specification-generated silver labels are not human relevance judgments.",
            "Only eight query groups are in the sealed holdout; confidence intervals are wide.",
            "The experiment adjusts fusion weights only; first-stage retrieval is unchanged.",
            "This diagnostic cannot establish production or resume-quality recommendation lift.",
        ],
    }
    summary["artifact_sha256"] = sha256_json(summary)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output_dir / "rankings.json").write_text(
        json.dumps(
            {
                "split_assignments": assignments,
                "test_baseline": {query_id: baseline[query_id] for query_id in test_ids},
                "test_weighted_rrf": {query_id: challenger[query_id] for query_id in test_ids},
            },
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    return summary


def _git_revision() -> str | None:
    import subprocess

    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--qrels", type=Path, default=DEFAULT_QRELS)
    parser.add_argument("--embedding-dir", type=Path, default=DEFAULT_EMBEDDINGS)
    parser.add_argument(
        "--output-root", type=Path, default=Path("artifacts/evaluation/weighted-rrf-silver")
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    destination = args.output_root / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    result = run_experiment(
        catalog_path=args.catalog,
        qrels_path=args.qrels,
        embedding_dir=args.embedding_dir,
        output_dir=destination,
        device=args.device,
    )
    print(
        json.dumps(
            {
                "output_dir": str(destination),
                "test": result["test"],
                "selected": result["selected_on_validation_only"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
