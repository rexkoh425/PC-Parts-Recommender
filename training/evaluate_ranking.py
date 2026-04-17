"""Evaluate and promote one persisted LambdaMART ranker on frozen human evidence.

The command loads the exact content-addressed model artifact, rebuilds the frozen
feature matrix, produces scores itself, and binds those bytes and outputs into the
comparison report.  Caller-supplied challenger rankings are never accepted.
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

from pc_build_recommender.ranking import (
    LabeledRankingQuery,
    LambdaMARTRanker,
    RankerPromotionPolicy,
    RankingCandidate,
    evaluate_ranker_promotion,
    generate_artifact_bound_rankings,
    write_ranker_promotion_decision,
)
from pc_build_recommender.retrieval import (
    FrozenCandidateSet,
    QueryGroupSplit,
    compare_ranked_models,
    write_ranking_comparison_report,
)
from training._common import print_json
from training.materialize_ranking_snapshot import verify_labeled_ranking_snapshot
from training.train_ranking import _load_queries, _validate_feature_snapshot_against_qrels


def _retrieval_ranking(
    queries: Sequence[LabeledRankingQuery],
    *,
    score_name: str,
) -> dict[str, list[str]]:
    rankings: dict[str, list[str]] = {}
    for query in queries:
        query_id = query.context.query_id
        candidate_rows = query.candidates
        scores: dict[str, float] = {}
        for candidate in candidate_rows:
            if not isinstance(candidate, RankingCandidate):
                raise TypeError("ranking feature snapshot contains an invalid candidate")
            raw_score = candidate.retrieval_scores.get(score_name)
            if raw_score is None or not math.isfinite(float(raw_score)):
                raise ValueError(
                    f"query {query_id!r} candidate {candidate.product_id!r} lacks finite "
                    f"{score_name}"
                )
            scores[candidate.product_id] = float(raw_score)
        rankings[query_id] = sorted(
            scores,
            key=lambda product_id: (-scores[product_id], product_id),
        )
    return rankings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-snapshot", required=True, type=Path)
    parser.add_argument("--dataset-manifest", required=True, type=Path)
    parser.add_argument("--human-judgments", required=True, type=Path)
    parser.add_argument("--qrels", required=True, type=Path)
    parser.add_argument("--frozen-query-split", required=True, type=Path)
    parser.add_argument("--ranker-model", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--challenger-model", default="lambdamart")
    parser.add_argument("--n-resamples", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--minimum-test-query-groups", type=int, default=50)
    parser.add_argument("--minimum-recall-at-50", type=float, default=0.95)
    parser.add_argument(
        "--minimum-relative-ndcg-lift-percent-over-bm25",
        type=float,
        default=15.0,
    )
    parser.add_argument("--minimum-bm25-ndcg-delta-ci-lower", type=float, default=0.0)
    parser.add_argument("--rrf-ndcg-noninferiority-margin", type=float, default=0.01)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    for path in (
        args.feature_snapshot,
        args.dataset_manifest,
        args.human_judgments,
        args.qrels,
        args.frozen_query_split,
        args.ranker_model,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.n_resamples < 1:
        raise ValueError("n_resamples must be positive")

    qrels = FrozenCandidateSet.load(args.qrels)
    if (
        qrels.label_source.value != "human"
        or not qrels.adjudication_complete
        or qrels.contains_synthetic_labels
        or qrels.judgment_manifest_sha256 is None
    ):
        raise ValueError("ranking promotion requires adjudicated, non-synthetic human qrels")
    query_split = QueryGroupSplit.load(args.frozen_query_split)
    query_split.validate_dataset(qrels)
    if set(query_split.weights) != {"train", "validation", "test"}:
        raise ValueError("frozen query split must contain train, validation, and test")

    verified_snapshot = verify_labeled_ranking_snapshot(
        ranking_path=args.feature_snapshot,
        manifest_path=args.dataset_manifest,
        human_judgments_path=args.human_judgments,
        qrels_path=args.qrels,
        query_split_path=args.frozen_query_split,
    )
    queries = _load_queries(args.feature_snapshot)
    _validate_feature_snapshot_against_qrels(queries, qrels)
    expected_groups = {query.query_id: query.query_group_id for query in qrels.queries}
    if dict(query_split.query_group_ids) != expected_groups:
        raise ValueError("frozen split query groups do not match the human qrels")

    ranker = LambdaMARTRanker.load(args.ranker_model)
    if ranker.metadata.query_group_split_checksum != query_split.checksum:
        raise ValueError("ranker was trained against a different frozen query split")
    if ranker.metadata.training_judgment_manifest_sha256 != qrels.judgment_manifest_sha256:
        raise ValueError("ranker was trained against different human judgments")
    if (
        ranker.metadata.training_dataset_manifest_sha256
        != verified_snapshot.manifest_sha256
        or ranker.metadata.training_prelabel_snapshot_sha256
        != verified_snapshot.prelabel_snapshot_sha256
        or ranker.metadata.training_feature_contract_sha256
        != verified_snapshot.feature_contract_sha256
    ):
        raise ValueError("ranker was trained against a different pre-label feature snapshot")

    contexts = {query.context.query_id: query.context for query in queries}
    candidates = {query.context.query_id: query.candidates for query in queries}
    bound = generate_artifact_bound_rankings(
        ranker,
        qrels,
        model_name=args.challenger_model,
        contexts=contexts,
        candidates=candidates,
        query_split=query_split,
        split_name="test",
    )
    rankings: dict[str, Mapping[str, Sequence[str]]] = {
        "bm25": _retrieval_ranking(queries, score_name="bm25_score"),
        "rrf_hybrid": _retrieval_ranking(queries, score_name="rrf_score"),
        args.challenger_model: bound.rankings,
    }
    report = compare_ranked_models(
        qrels,
        rankings,
        artifact_bound_rankings={args.challenger_model: bound.evidence},
        baseline_model="bm25",
        reference_models=("bm25", "rrf_hybrid"),
        query_split=query_split,
        split_name="test",
        n_resamples=args.n_resamples,
        seed=args.seed,
    )
    policy = RankerPromotionPolicy(
        minimum_test_query_groups=args.minimum_test_query_groups,
        minimum_recall_at_50=args.minimum_recall_at_50,
        minimum_relative_ndcg_lift_percent_over_bm25=(
            args.minimum_relative_ndcg_lift_percent_over_bm25
        ),
        minimum_bm25_ndcg_delta_ci_lower=args.minimum_bm25_ndcg_delta_ci_lower,
        rrf_ndcg_noninferiority_margin=args.rrf_ndcg_noninferiority_margin,
    )
    decision = evaluate_ranker_promotion(
        report,
        challenger_model=args.challenger_model,
        ranker=ranker,
        policy=policy,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = write_ranking_comparison_report(
        report,
        args.output_dir / "ranking-comparison.json",
    )
    decision_path = write_ranker_promotion_decision(
        decision,
        args.output_dir / "ranker-promotion-decision.json",
    )
    print_json(
        {
            "comparison_report": str(report_path.resolve()),
            "promotion_decision": str(decision_path.resolve()),
            "report_sha256": report.report_sha256,
            "decision_sha256": decision.decision_sha256,
            "artifact_binding_sha256": bound.evidence.evidence_sha256,
            "passed": decision.passed,
            "failures": list(decision.failures),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
