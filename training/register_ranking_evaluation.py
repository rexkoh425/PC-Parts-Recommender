"""Preregister one LambdaMART artifact for one frozen human test cohort."""

from __future__ import annotations

import argparse
from pathlib import Path

from pc_build_recommender.ranking import (
    MINIMUM_PRODUCTION_BOOTSTRAP_RESAMPLES,
    LambdaMARTRanker,
    RankerPromotionPolicy,
    assert_production_ranker_promotion_policy,
)
from training._common import print_json
from training.ranking_evaluation_gate import (
    RankingEvaluationIntent,
    RankingEvaluationLedger,
    bind_unopened_ranking_test_cohort,
    preregister_evaluation_intent,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-snapshot", required=True, type=Path)
    parser.add_argument("--dataset-manifest", required=True, type=Path)
    parser.add_argument("--human-judgments", required=True, type=Path)
    parser.add_argument("--qrels", required=True, type=Path)
    parser.add_argument("--frozen-query-split", required=True, type=Path)
    parser.add_argument("--ranker-model", required=True, type=Path)
    parser.add_argument("--intent-root", required=True, type=Path)
    parser.add_argument("--ledger-dir", required=True, type=Path)
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
    if args.n_resamples < MINIMUM_PRODUCTION_BOOTSTRAP_RESAMPLES:
        raise ValueError(
            "production n_resamples must be at least "
            f"{MINIMUM_PRODUCTION_BOOTSTRAP_RESAMPLES}"
        )

    ranker = LambdaMARTRanker.load(args.ranker_model)
    cohort = bind_unopened_ranking_test_cohort(
        ranking_path=args.feature_snapshot,
        manifest_path=args.dataset_manifest,
        human_judgments_path=args.human_judgments,
        qrels_path=args.qrels,
        query_split_path=args.frozen_query_split,
        ranker=ranker,
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
    assert_production_ranker_promotion_policy(policy)
    intent = RankingEvaluationIntent.create(
        cohort=cohort,
        ranker=ranker,
        challenger_model=args.challenger_model,
        n_resamples=args.n_resamples,
        bootstrap_seed=args.seed,
        policy=policy,
    )
    ledger = RankingEvaluationLedger(args.ledger_dir)
    intent_path = preregister_evaluation_intent(
        intent,
        intent_root=args.intent_root,
        ledger=ledger,
    )
    print_json(
        {
            "status": "preregistered",
            "intent_path": str(intent_path.resolve()),
            "intent_sha256": intent.intent_sha256,
            "cohort_sha256": intent.cohort_sha256,
            "ranker_model_sha256": intent.ranker_model_sha256,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
