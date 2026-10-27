"""Tune and evaluate a non-promotable LLM-judged LambdaMART diagnostic.

The test subcommand writes one immutable holdout report. It measures ranking
within the frozen judged candidate pools, not full-catalogue retrieval recall.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from statistics import fmean

import numpy as np
from training.train_ranking import _load_queries

from pc_build_recommender.evaluation.manifest import sha256_file
from pc_build_recommender.evaluation.metrics import bootstrap_confidence_interval
from pc_build_recommender.ranking import LambdaMARTRanker, RankingFeatureBuilder
from pc_build_recommender.ranking.features import FeatureBatch
from pc_build_recommender.retrieval import RelevanceLabelSource, ndcg_at_k

SEED = 20261108
PARAMETERS = (
    {"n_estimators": 150, "num_leaves": 15, "min_child_samples": 15},
    {"n_estimators": 250, "num_leaves": 31, "min_child_samples": 30},
    {"n_estimators": 200, "num_leaves": 7, "min_child_samples": 50},
)


class DiagnosticFeatureBuilder(RankingFeatureBuilder):
    """Add an explicit missing-specification indicator to the existing contract."""

    feature_version = "ranking-features-llm-diagnostic-v1"
    feature_names = (*RankingFeatureBuilder.feature_names, "specification_unknown_fraction")

    def build(self, context, candidates) -> FeatureBatch:
        base = RankingFeatureBuilder().build(context, candidates)
        unknown = np.asarray(
            [
                candidate.signals.get("specification_unknown_fraction", 0.0)
                for candidate in candidates
            ],
            dtype=np.float64,
        )
        if not np.isfinite(unknown).all() or np.any((unknown < 0) | (unknown > 1)):
            raise ValueError("specification unknown fraction must be finite and between 0 and 1")
        values = np.column_stack((base.values, unknown))
        return FeatureBatch(base.query_id, base.product_ids, self.feature_names, values)


class WithoutRetrievalFeatures(DiagnosticFeatureBuilder):
    """Validation-only ablation, never published as a serving model."""

    def build(self, context, candidates) -> FeatureBatch:
        batch = super().build(context, candidates)
        values = batch.values.copy()
        for name in ("bm25_score", "vector_similarity", "rrf_score"):
            values[:, batch.feature_names.index(name)] = 0
        return FeatureBatch(batch.query_id, batch.product_ids, batch.feature_names, values)


class WithoutSpecificationFeature(DiagnosticFeatureBuilder):
    """Validation-only ablation of explicit specification satisfaction."""

    def build(self, context, candidates) -> FeatureBatch:
        batch = super().build(context, candidates)
        values = batch.values.copy()
        values[:, batch.feature_names.index("specification_match")] = 0
        values[:, batch.feature_names.index("specification_unknown_fraction")] = 0
        return FeatureBatch(batch.query_id, batch.product_ids, batch.feature_names, values)


def load_inputs(dataset: Path, summary_path: Path, snapshot_path: Path):
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    manifest = json.loads((snapshot_path.parent / "manifest.json").read_text(encoding="utf-8"))
    audit_path = snapshot_path.parent.parent / "query-split-audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if not summary.get("training_dataset_ready") or summary.get("unresolved_task_count") != 0:
        raise ValueError("all real judgments must be completed before training")
    if summary.get("label_source") != "llm_weak":
        raise ValueError("this diagnostic requires explicit llm_weak provenance")
    if sha256_file(dataset) != summary["ranking_dataset_sha256"]:
        raise ValueError("ranking dataset does not match judgment summary")
    if (
        sha256_file(snapshot_path) != manifest["snapshot_sha256"]
        or summary["input_sha256"] != manifest["tasks_sha256"]
        or manifest["seed"] != SEED
    ):
        raise ValueError("queue, split snapshot, or judgment input differs from frozen manifest")
    if (
        audit["snapshot_sha256"] != sha256_file(snapshot_path)
        or audit["cross_split_matches"]
        or audit["previous_query_matches"]
    ):
        raise ValueError("query split did not pass its frozen similarity audit")
    queries = _load_queries(dataset)
    rows = json.loads(snapshot_path.read_text(encoding="utf-8"))
    snapshot = {row["query_id"]: row for row in rows}
    if len(snapshot) != len(rows) or set(snapshot) != {q.context.query_id for q in queries}:
        raise ValueError("query snapshot and labeled dataset differ")
    splits = {name: [] for name in ("train", "validation", "test")}
    group_splits = {}
    for q in queries:
        frozen = snapshot[q.context.query_id]
        name = frozen["split"]
        group = q.context.query_group_id or q.context.query_id
        if (
            name not in splits
            or frozen["query_group_id"] != group
            or group_splits.setdefault(group, name) != name
            or set(frozen["candidate_ids"]) != {c.product_id for c in q.candidates}
        ):
            raise ValueError("split, query group, or candidate pool differs from frozen snapshot")
        splits[name].append(q)
        for ranking in frozen["rankings"].values():
            if len(ranking) != len(q.candidates) or set(ranking) != set(frozen["candidate_ids"]):
                raise ValueError("baseline ranking changed the candidate pool")
    return summary, splits, snapshot


def ndcg(query, ranking: list[str]) -> float:
    labels = {
        c.product_id: g for c, g in zip(query.candidates, query.relevance_grades, strict=True)
    }
    return ndcg_at_k(labels, ranking, k=10)


def model_ranking(ranker: LambdaMARTRanker, query) -> list[str]:
    scores = ranker.predict(query.context, query.candidates)
    return [
        c.product_id
        for c, _ in sorted(
            zip(query.candidates, scores, strict=True),
            key=lambda item: (-float(item[1]), item[0].product_id),
        )
    ]


def tune(args) -> None:
    if args.output.exists():
        raise FileExistsError(args.output)
    summary, split, snapshot = load_inputs(args.dataset, args.summary, args.snapshot)
    full_split = split
    primary_judge = summary.get("primary_judge_model")
    if summary["model_id"] == "mixed-by-query" and not primary_judge:
        raise ValueError("mixed-model labels require a prespecified primary judge")
    if primary_judge:
        split = {
            name: [
                q
                for q in queries
                if snapshot[q.context.query_id].get("judge_model") == primary_judge
            ]
            for name, queries in full_split.items()
        }
    if len(split["validation"]) < 1 or len(split["test"]) < 50:
        raise ValueError("need validation queries and at least 50 held-out query groups")
    if len({q.context.query_group_id for q in split["test"]}) < 50:
        raise ValueError("test split has fewer than 50 independent intent groups")
    args.output.mkdir(parents=True)
    data_hash = sha256_file(args.dataset)
    snapshot_hash = sha256_file(args.snapshot)
    groups = {
        q.context.query_id: q.context.query_group_id or q.context.query_id
        for q in (*split["train"], *split["validation"])
    }
    rrf_val = fmean(
        ndcg(q, snapshot[q.context.query_id]["rankings"]["rrf"]) for q in split["validation"]
    )
    results = []
    for index, params in enumerate(PARAMETERS):
        start = time.perf_counter()
        ranker = LambdaMARTRanker(
            feature_builder=DiagnosticFeatureBuilder(),
            parameters={**params, "random_state": SEED, "n_jobs": 4},
            ranker_version=f"llm-diagnostic-{index}",
        )
        ranker.fit(
            split["train"],
            validation_queries=split["validation"],
            training_data_version=f"sha256:{data_hash}",
            candidate_set_version=snapshot_hash,
            training_label_source=RelevanceLabelSource.LLM_WEAK,
            query_group_ids=groups,
        )
        elapsed = time.perf_counter() - start
        score = fmean(ndcg(q, model_ranking(ranker, q)) for q in split["validation"])
        path = args.output / f"model-{index}.txt"
        ranker.save(path)
        loaded = LambdaMARTRanker.load(path, feature_builder=DiagnosticFeatureBuilder())
        if loaded.metadata.promotion_eligible:
            raise ValueError("weak-label model unexpectedly eligible for promotion")
        gain = dict(
            zip(
                loaded.metadata.feature_names,
                map(float, loaded._booster.feature_importance(importance_type="gain")),
                strict=True,
            )
        )
        results.append(
            {
                "model": str(path),
                "model_sha256": sha256_file(path),
                "parameters": params,
                "validation_ndcg_at_10": score,
                "training_seconds": elapsed,
                "boosting_iterations": loaded._booster.current_iteration(),
                "feature_gain": gain,
            }
        )
    selected = max(results, key=lambda row: row["validation_ndcg_at_10"])
    ablation = LambdaMARTRanker(
        feature_builder=WithoutRetrievalFeatures(),
        parameters={**selected["parameters"], "random_state": SEED, "n_jobs": 4},
    )
    ablation.fit(
        split["train"],
        validation_queries=split["validation"],
        training_data_version=f"sha256:{data_hash}",
        candidate_set_version=snapshot_hash,
        training_label_source=RelevanceLabelSource.LLM_WEAK,
        query_group_ids=groups,
    )
    ablated_score = fmean(ndcg(q, model_ranking(ablation, q)) for q in split["validation"])
    no_spec = LambdaMARTRanker(
        feature_builder=WithoutSpecificationFeature(),
        parameters={**selected["parameters"], "random_state": SEED, "n_jobs": 4},
    )
    no_spec.fit(
        split["train"],
        validation_queries=split["validation"],
        training_data_version=f"sha256:{data_hash}",
        candidate_set_version=snapshot_hash,
        training_label_source=RelevanceLabelSource.LLM_WEAK,
        query_group_ids=groups,
    )
    no_spec_score = fmean(ndcg(q, model_ranking(no_spec, q)) for q in split["validation"])
    report = {
        "schema_version": "pcbr.llm-ranking-tuning.v1",
        "label_source": "llm_weak",
        "eligible_for_promotion": False,
        "dataset_sha256": data_hash,
        "judgment_summary_sha256": sha256_file(args.summary),
        "snapshot_sha256": snapshot_hash,
        "query_split_audit_sha256": sha256_file(
            args.snapshot.parent.parent / "query-split-audit.json"
        ),
        "seed": SEED,
        "model_id": summary["model_id"],
        "primary_judge_model": primary_judge,
        "prompt_version": summary["prompt_version"],
        "split_query_counts": {k: len(v) for k, v in split.items()},
        "split_group_counts": {
            k: len({q.context.query_group_id for q in v}) for k, v in split.items()
        },
        "all_judges_split_query_counts": {k: len(v) for k, v in full_split.items()},
        "validation_rrf_ndcg_at_10": rrf_val,
        "candidates": results,
        "selected": selected,
        "validation_no_retrieval_feature_ndcg_at_10": ablated_score,
        "validation_no_specification_feature_ndcg_at_10": no_spec_score,
    }
    (args.output / "selection.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


def test(args) -> None:
    target = args.output / "sealed-test.json"
    if target.exists():
        raise FileExistsError("sealed test has already been written")
    selection_path = args.output / "selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if (
        selection["dataset_sha256"] != sha256_file(args.dataset)
        or selection["snapshot_sha256"] != sha256_file(args.snapshot)
        or selection["query_split_audit_sha256"]
        != sha256_file(args.snapshot.parent.parent / "query-split-audit.json")
        or selection["judgment_summary_sha256"] != sha256_file(args.summary)
    ):
        raise ValueError("test inputs differ from the preselected tuning inputs")
    _, split, snapshot = load_inputs(args.dataset, args.summary, args.snapshot)
    model_path = Path(selection["selected"]["model"])
    if sha256_file(model_path) != selection["selected"]["model_sha256"]:
        raise ValueError("selected model digest changed")
    ranker = LambdaMARTRanker.load(model_path, feature_builder=DiagnosticFeatureBuilder())
    rows = []
    for q in split["test"]:
        frozen = snapshot[q.context.query_id]
        labels = {c.product_id: g for c, g in zip(q.candidates, q.relevance_grades, strict=True)}
        relevant = {pid for pid, grade in labels.items() if grade > 0}
        start = time.perf_counter()
        model_order = model_ranking(ranker, q)
        inference_ms = 1000 * (time.perf_counter() - start)
        rankings = {**frozen["rankings"], "lambdamart": model_order}
        metrics = {}
        for name, order in rankings.items():
            first = next((i for i, pid in enumerate(order, 1) if pid in relevant), None)
            metrics[name] = {
                "ndcg_at_10": ndcg(q, order),
                "mrr": 1 / first if first else 0,
                "recall_at_20": len(set(order[:20]) & relevant) / len(relevant)
                if relevant
                else None,
                "recall_at_50": len(set(order[:50]) & relevant) / len(relevant)
                if relevant
                else None,
            }
        rows.append(
            {
                "query_id": q.context.query_id,
                "query_group_id": q.context.query_group_id,
                "category": q.candidates[0].category,
                "judge_model": frozen.get("judge_model", selection["model_id"]),
                "relevant_count": len(relevant),
                "distinct_grade_count": len(set(labels.values())),
                "inference_ms": inference_ms,
                "metrics": metrics,
            }
        )
    primary_judge = selection.get("primary_judge_model")
    primary_rows = [r for r in rows if not primary_judge or r["judge_model"] == primary_judge]
    if len({r["query_group_id"] for r in primary_rows}) < 50:
        raise ValueError("primary single-judge holdout has fewer than 50 intent groups")
    results = {}
    for model in ("bm25", "vector", "rrf", "lambdamart"):
        metrics = {}
        for metric in ("ndcg_at_10", "mrr", "recall_at_20", "recall_at_50"):
            values = [
                r["metrics"][model][metric]
                for r in primary_rows
                if r["metrics"][model][metric] is not None
            ]
            metrics[metric] = fmean(values) if values else None
        results[model] = metrics
    deltas = [
        r["metrics"]["lambdamart"]["ndcg_at_10"] - r["metrics"]["rrf"]["ndcg_at_10"]
        for r in primary_rows
    ]
    groups = [r["query_group_id"] for r in primary_rows]
    low, high = bootstrap_confidence_interval(deltas, groups=groups, n_resamples=1000, seed=SEED)
    base = results["rrf"]["ndcg_at_10"]
    by_category = {}
    for category in sorted({r["category"] for r in primary_rows}):
        selected = [r for r in primary_rows if r["category"] == category]
        by_category[category] = {
            name: fmean(r["metrics"][name]["ndcg_at_10"] for r in selected) for name in results
        }
    by_judge = {}
    for judge in sorted({r["judge_model"] for r in rows}):
        selected = [r for r in rows if r["judge_model"] == judge]
        by_judge[judge] = {
            "query_count": len(selected),
            "rrf_ndcg_at_10": fmean(r["metrics"]["rrf"]["ndcg_at_10"] for r in selected),
            "lambdamart_ndcg_at_10": fmean(
                r["metrics"]["lambdamart"]["ndcg_at_10"] for r in selected
            ),
        }
    informative = [r for r in primary_rows if r["distinct_grade_count"] > 1]
    informative_delta = [
        r["metrics"]["lambdamart"]["ndcg_at_10"] - r["metrics"]["rrf"]["ndcg_at_10"]
        for r in informative
    ]
    informative_groups = [r["query_group_id"] for r in informative]
    informative_interval = (
        list(
            bootstrap_confidence_interval(
                informative_delta, groups=informative_groups, n_resamples=1000, seed=SEED
            )
        )
        if len(set(informative_groups)) >= 2
        else None
    )
    report = {
        "schema_version": "pcbr.llm-ranking-holdout.v1",
        "label_source": "llm_weak",
        "eligible_for_promotion": False,
        "selection_sha256": sha256_file(selection_path),
        "primary_judge_model": primary_judge,
        "query_count": len(primary_rows),
        "all_judges_query_count": len(rows),
        "group_count": len(set(groups)),
        "models": results,
        "absolute_ndcg_delta_vs_rrf": fmean(deltas),
        "relative_ndcg_delta_vs_rrf": fmean(deltas) / base if base else None,
        "paired_group_bootstrap_95_ci": [low, high],
        "bootstrap_resamples": 1000,
        "ndcg_95_ci_by_model": {
            model: list(
                bootstrap_confidence_interval(
                    [r["metrics"][model]["ndcg_at_10"] for r in primary_rows],
                    groups=groups,
                    n_resamples=1000,
                    seed=SEED,
                )
            )
            for model in results
        },
        "wins": sum(d > 1e-12 for d in deltas),
        "losses": sum(d < -1e-12 for d in deltas),
        "ties": sum(abs(d) <= 1e-12 for d in deltas),
        "win_rate": sum(d > 1e-12 for d in deltas) / len(deltas),
        "queries_without_relevant_candidates": sum(r["relevant_count"] == 0 for r in primary_rows),
        "homogeneous_grade_query_count": sum(r["distinct_grade_count"] == 1 for r in primary_rows),
        "informative_query_count": len(informative),
        "informative_query_ndcg_delta_vs_rrf": fmean(informative_delta)
        if informative_delta
        else None,
        "informative_query_delta_95_ci": informative_interval,
        "by_category": by_category,
        "by_judge_model": by_judge,
        "inference_p95_ms": float(np.percentile([r["inference_ms"] for r in rows], 95)),
        "scope": "conditional on 28 judged candidates per query; recall@50 is not catalogue recall",
        "queries": rows,
    }
    with target.open("x", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    print(json.dumps({k: v for k, v in report.items() if k != "queries"}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("tune", "test"))
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    (tune if args.phase == "tune" else test)(args)


if __name__ == "__main__":
    main()
