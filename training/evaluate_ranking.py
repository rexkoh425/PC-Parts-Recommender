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
    RankingCandidate,
    evaluate_ranker_promotion,
    generate_artifact_bound_rankings,
)
from pc_build_recommender.retrieval import (
    compare_ranked_models,
)
from training._common import print_json
from training.ranking_evaluation_gate import (
    RankingEvaluationIntent,
    RankingEvaluationLedger,
    assert_ranker_matches_lineage,
    publish_ranking_evaluation,
    verify_human_ranking_lineage,
)
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
    parser.add_argument("--evaluation-intent", required=True, type=Path)
    parser.add_argument("--ledger-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
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
        args.evaluation_intent,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    intent = RankingEvaluationIntent.load(args.evaluation_intent)
    intent.assert_runtime_bindings()
    ranker = LambdaMARTRanker.load(args.ranker_model)
    intent.assert_file_bindings(
        ranking_path=args.feature_snapshot,
        manifest_path=args.dataset_manifest,
        human_judgments_path=args.human_judgments,
        qrels_path=args.qrels,
        query_split_path=args.frozen_query_split,
        ranker=ranker,
    )
    ledger = RankingEvaluationLedger(args.ledger_dir)
    # Claim before parsing judgments or producing test scores.  A semantic
    # validation failure is therefore an auditable test access, not a free probe.
    access_path = ledger.claim_access(intent)
    lineage = verify_human_ranking_lineage(
        ranking_path=args.feature_snapshot,
        manifest_path=args.dataset_manifest,
        human_judgments_path=args.human_judgments,
        qrels_path=args.qrels,
        query_split_path=args.frozen_query_split,
    )
    intent.assert_lineage(lineage)
    assert_ranker_matches_lineage(ranker, lineage)
    qrels = lineage.qrels
    query_split = lineage.query_split
    queries = _load_queries(args.feature_snapshot)
    _validate_feature_snapshot_against_qrels(queries, qrels)

    contexts = {query.context.query_id: query.context for query in queries}
    candidates = {query.context.query_id: query.candidates for query in queries}
    bound = generate_artifact_bound_rankings(
        ranker,
        qrels,
        model_name=intent.challenger_model,
        contexts=contexts,
        candidates=candidates,
        query_split=query_split,
        split_name="test",
    )
    rankings: dict[str, Mapping[str, Sequence[str]]] = {
        "bm25": _retrieval_ranking(queries, score_name="bm25_score"),
        "rrf_hybrid": _retrieval_ranking(queries, score_name="rrf_score"),
        intent.challenger_model: bound.rankings,
    }
    report = compare_ranked_models(
        qrels,
        rankings,
        artifact_bound_rankings={intent.challenger_model: bound.evidence},
        baseline_model="bm25",
        reference_models=("bm25", "rrf_hybrid"),
        query_split=query_split,
        split_name="test",
        n_resamples=intent.n_resamples,
        seed=intent.bootstrap_seed,
    )
    decision = evaluate_ranker_promotion(
        report,
        challenger_model=intent.challenger_model,
        ranker=ranker,
        policy=intent.policy,
    )

    completion_path = ledger.record_completion(
        intent,
        comparison_report_sha256=report.report_sha256,
        promotion_decision_sha256=decision.decision_sha256,
    )
    registration_path, recorded_access_path, recorded_completion_path = (
        ledger.evidence_paths(intent)
    )
    if recorded_access_path != access_path or recorded_completion_path != completion_path:
        raise RuntimeError("ranking ledger returned inconsistent evidence paths")
    # Publication happens only after the append-only completion record binds the
    # exact report and decision. A passing decision can never escape half-sealed.
    published = publish_ranking_evaluation(
        output_root=args.output_dir,
        intent=intent,
        registration_path=registration_path,
        access_path=access_path,
        completion_path=completion_path,
        comparison_report=report.to_dict(),
        promotion_decision=decision.to_dict(),
    )
    report_path = published.output_dir / "ranking-comparison.json"
    decision_path = published.output_dir / "ranker-promotion-decision.json"
    print_json(
        {
            "evaluation_intent": str(args.evaluation_intent.resolve()),
            "intent_sha256": intent.intent_sha256,
            "cohort_sha256": intent.cohort_sha256,
            "test_access_record": str(access_path.resolve()),
            "completion_record": str(completion_path.resolve()),
            "sealed_output_dir": str(published.output_dir.resolve()),
            "bundle_manifest_sha256": published.bundle_manifest_sha256,
            "evaluation_payload_sha256": published.evaluation_payload_sha256,
            "ledger_identity_sha256": published.ledger_identity_sha256,
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
