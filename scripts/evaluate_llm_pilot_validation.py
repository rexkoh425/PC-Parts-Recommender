"""Evaluate the real pilot model on its previously used, seven-query validation split.

This is an LLM-weak technical diagnostic; the pilot queries were used by older
silver experiments and are not an independent final holdout.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean

import numpy as np
from training.train_ranking import _load_queries, _split_queries

from pc_build_recommender.evaluation.manifest import sha256_file
from pc_build_recommender.evaluation.metrics import bootstrap_confidence_interval
from pc_build_recommender.evaluation.retrieval_silver_pilot import _load_embedding_artifact
from pc_build_recommender.ranking import LambdaMARTRanker
from pc_build_recommender.retrieval import (
    BM25ProductIndex,
    SentenceTransformerEmbeddingEncoder,
    reciprocal_rank_fusion,
)
from scripts.benchmark_weighted_rrf_silver import _build_documents
from scripts.build_llm_judge_queue import CATALOG, EMBEDDINGS
from scripts.llm_ranking_experiment import model_ranking, ndcg


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="auto")
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    queries = _split_queries(_load_queries(a.dataset), seed=20261108)["validation"]
    records, docs = _build_documents(CATALOG)
    index = BM25ProductIndex(docs)
    matrix, id_rows, manifest = _load_embedding_artifact(EMBEDDINGS)
    positions = {str(r["product_id"]): int(r["row_index"]) for r in id_rows}
    encoder = SentenceTransformerEmbeddingEncoder(
        manifest["encoder"]["model_name"], device=a.device, batch_size=64
    )
    vectors = encoder.encode([q.context.query_text for q in queries])
    ranker = LambdaMARTRanker.load(a.model)
    per_query = []
    for i, q in enumerate(queries):
        category = q.candidates[0].category
        universe = [r["product_id"] for r in records if r["category"] == category]
        lexical = index.search(q.context.query_text, category=category, top_k=len(universe))
        vector = sorted(
            universe,
            key=lambda pid: (-round(float(np.dot(matrix[positions[pid]], vectors[i])), 8), pid),
        )
        fused = reciprocal_rank_fusion({"bm25": lexical[:50], "vector": vector[:50]}, k=60)
        fused_scores = {hit.product_id: hit.score for hit in fused}
        for candidate in q.candidates:
            if (
                abs(
                    fused_scores.get(candidate.product_id, 0.0)
                    - candidate.retrieval_scores["rrf_score"]
                )
                > 1e-12
            ):
                raise ValueError("reconstructed RRF differs from the frozen pilot score")
        ids = {c.product_id for c in q.candidates}
        rrf = [h.product_id for h in fused if h.product_id in ids]
        rrf += sorted(ids - set(rrf))
        bm25 = [h.product_id for h in lexical if h.product_id in ids]
        vector = [pid for pid in vector if pid in ids]
        rankings = {
            "bm25": bm25,
            "vector": vector,
            "rrf": rrf,
            "lambdamart": model_ranking(ranker, q),
        }
        labels = {c.product_id: g for c, g in zip(q.candidates, q.relevance_grades, strict=True)}
        relevant = {pid for pid, grade in labels.items() if grade > 0}
        metrics = {}
        for method, ranking in rankings.items():
            first = next((n for n, pid in enumerate(ranking, 1) if pid in relevant), None)
            metrics[method] = {
                "ndcg_at_10": ndcg(q, ranking),
                "mrr": 1 / first if first else 0,
                "recall_at_50": len(relevant & set(ranking[:50])) / len(relevant)
                if relevant
                else None,
            }
        per_query.append({"query_id": q.context.query_id, "metrics": metrics})
    aggregate = {}
    for name in ("bm25", "vector", "rrf", "lambdamart"):
        aggregate[name] = {
            key: fmean(v for row in per_query if (v := row["metrics"][name][key]) is not None)
            for key in ("ndcg_at_10", "mrr", "recall_at_50")
        }
    delta = [
        row["metrics"]["lambdamart"]["ndcg_at_10"] - row["metrics"]["rrf"]["ndcg_at_10"]
        for row in per_query
    ]
    interval = bootstrap_confidence_interval(delta, n_resamples=1000, seed=20261108)
    report = {
        "schema_version": "pcbr.llm-pilot-validation.v1",
        "label_source": "llm_weak",
        "dataset_sha256": sha256_file(a.dataset),
        "model_sha256": sha256_file(a.model),
        "query_count": len(queries),
        "models": aggregate,
        "absolute_ndcg_delta_vs_rrf": fmean(delta),
        "relative_ndcg_delta_vs_rrf": fmean(delta) / aggregate["rrf"]["ndcg_at_10"]
        if aggregate["rrf"]["ndcg_at_10"]
        else None,
        "paired_bootstrap_95_ci": interval,
        "bootstrap_resamples": 1000,
        "wins": sum(x > 1e-12 for x in delta),
        "losses": sum(x < -1e-12 for x in delta),
        "ties": sum(abs(x) <= 1e-12 for x in delta),
        "per_query": per_query,
        "limitation": (
            "previously used pilot queries; seven validation groups; "
            "LLM weak labels; conditional candidate-pool recall"
        ),
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "per_query"}, indent=2))


if __name__ == "__main__":
    main()
