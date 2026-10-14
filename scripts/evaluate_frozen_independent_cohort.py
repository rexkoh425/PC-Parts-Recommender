"""Evaluate the frozen LambdaMART model on a completed independent label set."""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from training.train_ranking import _load_queries

from pc_build_recommender.ranking import LambdaMARTRanker
from pc_build_recommender.retrieval import ndcg_at_k
from scripts.llm_ranking_experiment import DiagnosticFeatureBuilder, model_ranking


def metrics(query, ranking):
    labels = {
        c.product_id: g for c, g in zip(query.candidates, query.relevance_grades, strict=True)
    }
    relevant = {pid for pid, grade in labels.items() if grade > 0}
    first = next((i for i, pid in enumerate(ranking, 1) if pid in relevant), None)
    return {
        "ndcg_at_5": ndcg_at_k(labels, ranking, k=5),
        "ndcg_at_10": ndcg_at_k(labels, ranking, k=10),
        "mrr": 1 / first if first else 0.0,
        "recall_at_10": len(set(ranking[:10]) & relevant) / len(relevant) if relevant else None,
        "recall_at_50": len(set(ranking[:50]) & relevant) / len(relevant) if relevant else None,
    }


def bootstrap(values, seed=20261110, resamples=10000):
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    draws = np.empty(resamples)
    for i in range(resamples):
        draws[i] = values[rng.integers(0, len(values), len(values))].mean()
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--query-manifest", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    queries = _load_queries(args.dataset)
    evaluation_categories = {}
    if args.query_manifest:
        manifest = json.loads(args.query_manifest.read_text(encoding="utf-8"))
        evaluation_categories = {
            row["query_id"]: row.get("evaluation_category") for row in manifest["queries"]
        }
    snapshot = {
        row["query_id"]: row for row in json.loads(args.snapshot.read_text(encoding="utf-8"))
    }
    ranker = LambdaMARTRanker.load(args.model, feature_builder=DiagnosticFeatureBuilder())
    rows = []
    for query in queries:
        frozen = snapshot[query.context.query_id]
        rankings = dict(frozen["rankings"])
        start = time.perf_counter()
        rankings["lambdamart"] = model_ranking(ranker, query)
        inference_ms = 1000 * (time.perf_counter() - start)
        for method, order in rankings.items():
            row = metrics(query, order)
            row.update(
                query_id=query.context.query_id,
                evaluation_category=evaluation_categories.get(
                    query.context.query_id,
                    getattr(query.context, "evaluation_category", query.candidates[0].category),
                ),
                judge_model=getattr(query.context, "judge_model", "gpt-6-luna"),
                method=method,
                inference_ms=inference_ms if method == "lambdamart" else None,
            )
            rows.append(row)
    by_method = defaultdict(list)
    for row in rows:
        by_method[row["method"]].append(row)
    summary = {}
    for method, method_rows in by_method.items():
        summary[method] = {
            key: float(np.mean([r[key] for r in method_rows if r[key] is not None]))
            for key in ("ndcg_at_5", "ndcg_at_10", "mrr", "recall_at_10", "recall_at_50")
        }
    query_map = defaultdict(dict)
    for row in rows:
        query_map[row["query_id"]][row["method"]] = row
    deltas = {
        metric: [
            query_map[q]["lambdamart"][metric] - query_map[q]["rrf"][metric]
            for q in query_map
            if query_map[q]["lambdamart"][metric] is not None
            and query_map[q]["rrf"][metric] is not None
        ]
        for metric in ("ndcg_at_5", "ndcg_at_10", "mrr", "recall_at_10", "recall_at_50")
    }
    wins = sum(d > 0 for d in deltas["ndcg_at_10"])
    losses = sum(d < 0 for d in deltas["ndcg_at_10"])
    ties = sum(d == 0 for d in deltas["ndcg_at_10"])
    report = {
        "schema_version": "pcbr.frozen-independent-evaluation.v1",
        "label_source": "llm_weak",
        "judge_model": "gpt-6-luna",
        "query_count": len(query_map),
        "models": summary,
        "paired_deltas": {
            metric: {"mean": float(np.mean(values)), "ci95": bootstrap(values)}
            for metric, values in deltas.items()
        },
        "ndcg10_wins": wins,
        "ndcg10_losses": losses,
        "ndcg10_ties": ties,
        "relative_ndcg10_delta": float(
            np.mean(deltas["ndcg_at_10"]) / summary["rrf"]["ndcg_at_10"]
        ),
        "inference_p95_ms": float(
            np.percentile([r["inference_ms"] for r in by_method["lambdamart"]], 95)
        ),
        "inference_p99_ms": float(
            np.percentile([r["inference_ms"] for r in by_method["lambdamart"]], 99)
        ),
        "scope": "fixed 28-candidate pools per query; LLM-only diagnostic, not human validation",
    }
    category_rows = defaultdict(lambda: defaultdict(list))
    for query_id, methods in query_map.items():
        category = next(row["evaluation_category"] for row in rows if row["query_id"] == query_id)
        category_rows[category]["rrf"].append(methods["rrf"]["ndcg_at_10"])
        category_rows[category]["lambdamart"].append(methods["lambdamart"]["ndcg_at_10"])
    report["category_ndcg10"] = {
        category: {method: float(np.mean(values)) for method, values in methods.items()}
        for category, methods in category_rows.items()
    }
    report["worst_regressions"] = sorted(
        (
            {
                "query_id": q,
                "delta_ndcg10": float(
                    query_map[q]["lambdamart"]["ndcg_at_10"] - query_map[q]["rrf"]["ndcg_at_10"]
                ),
            }
            for q in query_map
        ),
        key=lambda row: row["delta_ndcg10"],
    )[:20]
    report["largest_improvements"] = sorted(
        (
            {
                "query_id": q,
                "delta_ndcg10": float(
                    query_map[q]["lambdamart"]["ndcg_at_10"] - query_map[q]["rrf"]["ndcg_at_10"]
                ),
            }
            for q in query_map
        ),
        key=lambda row: row["delta_ndcg10"],
        reverse=True,
    )[:20]
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with (args.output / "ranking-results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
