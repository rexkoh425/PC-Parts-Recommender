from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

import pytest

from pc_build_recommender.evaluation.contracts import MetricEstimate
from pc_build_recommender.evaluation.manifest import sha256_json
from pc_build_recommender.retrieval import (
    CandidatePopulationDeclaration,
    CandidatePopulationScope,
    FrozenCandidateQuery,
    FrozenCandidateSet,
    FrozenPruningStage,
    FrozenQueryGroupSplit,
    PruningClaimPolicy,
    PruningEvaluationProvenance,
    PruningEvaluationReport,
    PruningPromotionError,
    PruningStageKind,
    RelevanceLabelSource,
    evaluate_candidate_pruning,
    evaluate_pruning_claim,
    load_frozen_pruning_trace,
    load_pruning_evaluation_report,
    write_frozen_pruning_trace,
    write_pruning_evaluation_report,
)


def _dataset(
    *,
    label_source: RelevanceLabelSource = RelevanceLabelSource.HUMAN,
    complete_qrels: bool = True,
) -> FrozenCandidateSet:
    candidate_ids = tuple(f"p-{index}" for index in range(10))
    labels = {candidate_id: 0 for candidate_id in candidate_ids}
    labels["p-0"] = 4
    labels["p-1"] = 2
    if not complete_qrels:
        labels.pop("p-9")
    queries = [
        FrozenCandidateQuery(
            query_id=f"q-{index}",
            query_group_id=f"intent-{index}",
            query_text=f"query {index}",
            category="gpu",
            candidate_ids=candidate_ids,
            relevance_labels=labels,
        )
        for index in range(5)
    ]
    human = label_source is RelevanceLabelSource.HUMAN
    return FrozenCandidateSet.create(
        "pruning-human-v1",
        queries,
        label_source=label_source,
        adjudication_complete=human,
        contains_synthetic_labels=False,
        judgment_manifest_sha256="a" * 64 if human else None,
    )


def _stages(dataset: FrozenCandidateSet) -> tuple[FrozenPruningStage, ...]:
    initial = {query.query_id: query.candidate_ids for query in dataset.queries}
    first = {query.query_id: ("p-0", "p-1", "p-2", "p-3", "p-4") for query in dataset.queries}
    second = {query.query_id: ("p-0", "p-4") for query in dataset.queries}
    return (
        FrozenPruningStage.create(
            "structured_requirements",
            first,
            kind=PruningStageKind.STRUCTURED_REQUIREMENTS,
            version="requirements-v1",
            removal_reasons=_removal_reasons(initial, first, "structured:requirements"),
        ),
        FrozenPruningStage.create(
            "compatibility",
            second,
            kind=PruningStageKind.COMPATIBILITY,
            version="compat_v2",
            removal_reasons=_removal_reasons(first, second, "compat_v2:pairwise"),
        ),
    )


def _removal_reasons(
    before: Mapping[str, Sequence[str]],
    after: Mapping[str, Sequence[str]],
    reason: str,
) -> dict[str, dict[str, tuple[str, ...]]]:
    return {
        query_id: {
            candidate_id: (reason,) for candidate_id in set(candidate_ids) - set(after[query_id])
        }
        for query_id, candidate_ids in before.items()
    }


def _population(
    dataset: FrozenCandidateSet,
    *,
    scope: CandidatePopulationScope = CandidatePopulationScope.FULL_ELIGIBLE_CORPUS,
) -> CandidatePopulationDeclaration:
    return CandidatePopulationDeclaration.create(
        dataset,
        scope=scope,
        catalog_manifest_sha256="b" * 64,
        construction_method="all category-eligible catalog products before filtering",
        complete_eligible_corpus=scope is CandidatePopulationScope.FULL_ELIGIBLE_CORPUS,
        catalog_candidate_ids_by_query=(
            {query.query_id: query.candidate_ids for query in dataset.queries}
            if scope is CandidatePopulationScope.FULL_ELIGIBLE_CORPUS
            else None
        ),
    )


def _provenance() -> PruningEvaluationProvenance:
    return PruningEvaluationProvenance.create(
        run_id="pruning-eval-001",
        evaluated_at_utc="2026-07-23T00:00:00Z",
        pipeline_version="candidate-pruning-v1",
        compatibility_rule_version="compat_v2",
        data_version="catalog-v1",
        code_revision="deadbeef",
        catalog_manifest_sha256="b" * 64,
        filter_configuration_sha256="c" * 64,
        metadata={"environment": "offline-evaluation"},
    )


def _report(
    dataset: FrozenCandidateSet,
    *,
    population: CandidatePopulationDeclaration | None = None,
) -> PruningEvaluationReport:
    split = FrozenQueryGroupSplit.create(dataset, version="pruning-split-v1", seed=13)
    return evaluate_candidate_pruning(
        dataset,
        _stages(dataset),
        population=population or _population(dataset),
        provenance=_provenance(),
        query_split=split,
        split_name="test",
        n_resamples=40,
        seed=7,
    )


def test_pruning_report_records_sequential_counts_pooled_metrics_and_intervals() -> None:
    dataset = _dataset()
    report = _report(dataset)
    evaluated_queries = report.query_count

    assert report.eligible_for_promotion
    assert report.corpus_claim_qualified
    assert report.final_stage.candidate_count_before == evaluated_queries * 5
    assert report.final_stage.candidate_count_after == evaluated_queries * 2
    assert report.final_stage.relevant_count_before == evaluated_queries * 2
    assert report.final_stage.relevant_count_after == evaluated_queries

    first_pruning = report.stages[0].metrics["incremental_pooled_pruning_fraction"]
    final_pruning = report.final_stage.metrics["cumulative_pooled_pruning_fraction"]
    final_recall = report.final_stage.metrics["cumulative_pooled_retained_relevant_recall"]
    assert first_pruning.value == pytest.approx(0.5)
    assert final_pruning.value == pytest.approx(0.8)
    assert final_pruning.numerator == evaluated_queries * 8
    assert final_pruning.denominator == evaluated_queries * 10
    assert final_pruning.ci_lower is not None
    assert final_recall.value == pytest.approx(0.5)
    assert final_recall.numerator == evaluated_queries
    assert final_recall.denominator == evaluated_queries * 2
    assert final_recall.ci_lower is not None
    report.require_promotable()


def test_claim_policy_is_separate_and_fails_closed_on_sample_and_metric_targets() -> None:
    report = _report(_dataset())

    default_decision = evaluate_pruning_claim(report)
    assert not default_decision.passed
    assert any("query-group count" in reason for reason in default_decision.failures)
    assert any("bootstrap resample count" in reason for reason in default_decision.failures)
    assert any("retained-relevant recall" in reason for reason in default_decision.failures)

    fixture_policy = PruningClaimPolicy(
        minimum_test_query_groups=1,
        minimum_fully_judged_candidate_pairs=1,
        minimum_bootstrap_resamples=40,
        minimum_pooled_pruning_fraction=0.8,
        minimum_pooled_retained_relevant_recall=0.5,
        minimum_pruning_ci_lower=0.8,
        minimum_recall_ci_lower=0.5,
    )
    fixture_decision = evaluate_pruning_claim(report, policy=fixture_policy)
    assert fixture_decision.passed
    assert fixture_decision.measured_values["pooled_pruning_fraction"] == pytest.approx(0.8)
    assert fixture_decision.measured_values["pooled_retained_relevant_recall"] == pytest.approx(0.5)


def test_claim_decision_rejects_a_report_mutated_after_hashing() -> None:
    report = _report(_dataset())
    metrics = cast(dict[str, MetricEstimate], report.final_stage.metrics)
    metrics["cumulative_pooled_retained_relevant_recall"] = MetricEstimate(
        name="cumulative_pooled_retained_relevant_recall",
        value=1.0,
        sample_count=report.query_count,
        ci_lower=1.0,
        ci_upper=1.0,
        confidence_level=0.95,
        numerator=2,
        denominator=2,
    )

    with pytest.raises(ValueError, match="changed after hashing"):
        evaluate_pruning_claim(
            report,
            policy=PruningClaimPolicy(
                minimum_test_query_groups=1,
                minimum_fully_judged_candidate_pairs=1,
                minimum_bootstrap_resamples=2,
                minimum_pooled_pruning_fraction=0.0,
                minimum_pooled_retained_relevant_recall=0.0,
                minimum_pruning_ci_lower=0.0,
                minimum_recall_ci_lower=0.0,
            ),
        )


def test_claim_uses_terminal_compatibility_stage_not_later_rank_truncation() -> None:
    dataset = _dataset()
    split = FrozenQueryGroupSplit.create(dataset, version="pruning-split-v1", seed=13)
    stages = (
        *_stages(dataset),
        FrozenPruningStage.create(
            "top_k",
            {query.query_id: ("p-0",) for query in dataset.queries},
            kind=PruningStageKind.RANKING_TRUNCATION,
            version="top-k-v1",
            removal_reasons={
                query.query_id: {"p-4": ("ranking:top_k",)} for query in dataset.queries
            },
        ),
    )
    report = evaluate_candidate_pruning(
        dataset,
        stages,
        population=_population(dataset),
        provenance=_provenance(),
        query_split=split,
        split_name="test",
        n_resamples=40,
    )
    decision = evaluate_pruning_claim(
        report,
        policy=PruningClaimPolicy(
            minimum_test_query_groups=1,
            minimum_fully_judged_candidate_pairs=1,
            minimum_bootstrap_resamples=40,
            minimum_pooled_pruning_fraction=0.8,
            minimum_pooled_retained_relevant_recall=0.5,
            minimum_pruning_ci_lower=0.8,
            minimum_recall_ci_lower=0.5,
        ),
    )

    assert not decision.passed
    assert decision.measured_values["claim_stage_kind"] == "compatibility"
    assert any("terminal claim stage" in reason for reason in decision.failures)


def test_claim_rejects_pruning_from_a_prefix_retrieval_stage() -> None:
    dataset = _dataset()
    original = _stages(dataset)
    retained = {query.query_id: ("p-0",) for query in dataset.queries}
    initial = {query.query_id: query.candidate_ids for query in dataset.queries}
    no_removals: dict[str, dict[str, tuple[str, ...]]] = {
        query.query_id: {} for query in dataset.queries
    }
    stages = (
        FrozenPruningStage.create(
            "retrieval_cap",
            retained,
            kind=PruningStageKind.RETRIEVAL,
            version="retrieval-v1",
            removal_reasons=_removal_reasons(initial, retained, "retrieval:cap"),
        ),
        FrozenPruningStage.create(
            "structured_requirements",
            retained,
            kind=PruningStageKind.STRUCTURED_REQUIREMENTS,
            version=original[0].version,
            removal_reasons=no_removals,
        ),
        FrozenPruningStage.create(
            "compatibility",
            retained,
            kind=PruningStageKind.COMPATIBILITY,
            version=original[1].version,
            removal_reasons=no_removals,
        ),
    )
    split = FrozenQueryGroupSplit.create(dataset, version="pruning-split-v1", seed=13)
    report = evaluate_candidate_pruning(
        dataset,
        stages,
        population=_population(dataset),
        provenance=_provenance(),
        query_split=split,
        split_name="test",
        n_resamples=40,
    )
    decision = evaluate_pruning_claim(
        report,
        policy=PruningClaimPolicy(
            minimum_test_query_groups=1,
            minimum_fully_judged_candidate_pairs=1,
            minimum_bootstrap_resamples=40,
            minimum_pooled_pruning_fraction=0.9,
            minimum_pooled_retained_relevant_recall=0.5,
            minimum_pruning_ci_lower=0.9,
            minimum_recall_ci_lower=0.5,
        ),
    )

    assert not decision.passed
    assert any("exactly match" in reason for reason in decision.failures)


def test_claim_rejects_underpowered_confidence_protocol() -> None:
    dataset = _dataset()
    split = FrozenQueryGroupSplit.create(dataset, version="pruning-split-v1", seed=13)
    report = evaluate_candidate_pruning(
        dataset,
        _stages(dataset),
        population=_population(dataset),
        provenance=_provenance(),
        query_split=split,
        split_name="test",
        confidence_level=0.01,
        n_resamples=2,
    )
    decision = evaluate_pruning_claim(
        report,
        policy=PruningClaimPolicy(
            minimum_test_query_groups=1,
            minimum_fully_judged_candidate_pairs=1,
            minimum_bootstrap_resamples=2,
            minimum_pooled_pruning_fraction=0.8,
            minimum_pooled_retained_relevant_recall=0.5,
            minimum_pruning_ci_lower=0.8,
            minimum_recall_ci_lower=0.5,
        ),
    )

    assert not decision.passed
    assert any("confidence level" in reason for reason in decision.failures)


def test_claim_rejects_compatibility_stage_version_mismatch() -> None:
    dataset = _dataset()
    stages = list(_stages(dataset))
    stages[1] = FrozenPruningStage.create(
        "compatibility",
        stages[1].retained_candidate_ids,
        kind=PruningStageKind.COMPATIBILITY,
        version="compat-wrong",
        removal_reasons=stages[1].removal_reasons,
    )
    split = FrozenQueryGroupSplit.create(dataset, version="pruning-split-v1", seed=13)
    report = evaluate_candidate_pruning(
        dataset,
        stages,
        population=_population(dataset),
        provenance=_provenance(),
        query_split=split,
        split_name="test",
        n_resamples=40,
    )
    decision = evaluate_pruning_claim(
        report,
        policy=PruningClaimPolicy(
            minimum_test_query_groups=1,
            minimum_fully_judged_candidate_pairs=1,
            minimum_bootstrap_resamples=40,
            minimum_pooled_pruning_fraction=0.8,
            minimum_pooled_retained_relevant_recall=0.5,
            minimum_pruning_ci_lower=0.8,
            minimum_recall_ci_lower=0.5,
        ),
    )

    assert not decision.passed
    assert any("version does not match" in reason for reason in decision.failures)


def test_pruning_report_is_content_addressed_and_tamper_evident(tmp_path: Path) -> None:
    dataset = _dataset()
    stages = _stages(dataset)
    report = _report(dataset)
    trace_path = write_frozen_pruning_trace(stages, tmp_path / "trace.json")
    assert load_frozen_pruning_trace(trace_path) == stages
    trace_payload = json.loads(trace_path.read_text(encoding="utf-8"))
    assert trace_payload["trace_sha256"] == report.pruning_trace_sha256
    output = write_pruning_evaluation_report(report, tmp_path / "pruning.json")

    loaded = load_pruning_evaluation_report(output)
    assert loaded["report_sha256"] == report.report_sha256
    assert loaded["pruning_trace_sha256"] == report.pruning_trace_sha256
    assert loaded["population"]["checksum"] == report.population.checksum
    assert loaded["provenance"]["checksum"] == report.provenance.checksum

    loaded["stages"][0]["counts"]["candidate_query_pairs"]["after"] = 999
    output.write_text(json.dumps(loaded), encoding="utf-8")
    with pytest.raises(ValueError, match="hash verification failed"):
        load_pruning_evaluation_report(output)


def test_loader_rejects_rehashed_but_internally_inconsistent_counts(tmp_path: Path) -> None:
    report = _report(_dataset())
    payload = report.to_dict()
    stages = cast(list[dict[str, object]], payload["stages"])
    counts = cast(dict[str, object], stages[0]["counts"])
    candidate_counts = cast(dict[str, object], counts["candidate_query_pairs"])
    candidate_counts["after"] = cast(int, candidate_counts["after"]) - 1
    candidate_counts["pruned"] = cast(int, candidate_counts["pruned"]) + 1
    payload["report_sha256"] = sha256_json(
        {key: value for key, value in payload.items() if key != "report_sha256"}
    )
    output = tmp_path / "fabricated.json"
    output.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="aggregate counts do not equal"):
        load_pruning_evaluation_report(output)


def test_pruning_report_fails_closed_without_human_labels() -> None:
    dataset = _dataset(label_source=RelevanceLabelSource.SILVER)
    report = _report(dataset)

    assert not report.eligible_for_promotion
    assert any("silver" in reason for reason in report.promotion_block_reasons)
    with pytest.raises(PruningPromotionError, match="not promotable"):
        report.require_promotable()


def test_retrieval_pool_metrics_are_explicitly_not_corpus_qualified() -> None:
    dataset = _dataset()
    report = _report(
        dataset,
        population=_population(
            dataset,
            scope=CandidatePopulationScope.FROZEN_RETRIEVAL_POOL,
        ),
    )

    assert not report.corpus_claim_qualified
    assert not report.eligible_for_promotion
    assert report.evidence_eligible
    payload = report.to_dict()
    claim = cast(dict[str, object], payload["claim_qualification"])
    assert claim["population_scope"] == "frozen_retrieval_pool"
    assert claim["corpus_claim_qualified"] is False


def test_pooled_and_macro_metrics_use_declared_weighting() -> None:
    dataset = FrozenCandidateSet.create(
        "weighting-v1",
        (
            FrozenCandidateQuery(
                query_id="large",
                candidate_ids=tuple(f"p-{index}" for index in range(10)),
                relevance_labels={
                    **{f"p-{index}": 0 for index in range(10)},
                    "p-0": 4,
                    "p-1": 2,
                },
            ),
            FrozenCandidateQuery(
                query_id="small",
                candidate_ids=("s-0", "s-1"),
                relevance_labels={"s-0": 4, "s-1": 0},
            ),
        ),
        label_source=RelevanceLabelSource.HUMAN,
        adjudication_complete=True,
        judgment_manifest_sha256="d" * 64,
    )
    stage = FrozenPruningStage.create(
        "structured",
        {"large": ("p-0", "p-2"), "small": ("s-0", "s-1")},
        kind=PruningStageKind.STRUCTURED_REQUIREMENTS,
        version="requirements-v1",
        removal_reasons={
            "large": {
                candidate_id: ("structured:requirements",)
                for candidate_id in ("p-1", "p-3", "p-4", "p-5", "p-6", "p-7", "p-8", "p-9")
            },
            "small": {},
        },
    )
    report = evaluate_candidate_pruning(
        dataset,
        (stage,),
        population=_population(dataset),
        provenance=_provenance(),
        n_resamples=40,
        seed=5,
    )

    metrics = report.final_stage.metrics
    assert metrics["cumulative_pooled_pruning_fraction"].value == pytest.approx(8 / 12)
    assert metrics["cumulative_macro_pruning_fraction"].value == pytest.approx(0.4)
    assert metrics["cumulative_pooled_retained_relevant_recall"].value == pytest.approx(2 / 3)
    assert metrics["cumulative_macro_retained_relevant_recall"].value == pytest.approx(0.75)


def test_incomplete_qrels_are_diagnostic_even_if_declared_human() -> None:
    dataset = _dataset(complete_qrels=False)
    report = _report(dataset)

    assert not report.eligible_for_promotion
    assert any("every frozen candidate pair" in reason for reason in report.promotion_block_reasons)


def test_stages_must_be_exact_monotonic_subsets() -> None:
    dataset = _dataset()
    split = FrozenQueryGroupSplit.create(dataset, version="pruning-split-v1", seed=13)
    first = FrozenPruningStage.create(
        "first",
        {query.query_id: ("p-0",) for query in dataset.queries},
        kind=PruningStageKind.STRUCTURED_REQUIREMENTS,
        version="requirements-v1",
        removal_reasons={
            query.query_id: {
                candidate_id: ("structured:requirements",)
                for candidate_id in query.candidate_ids
                if candidate_id != "p-0"
            }
            for query in dataset.queries
        },
    )
    reintroduced = FrozenPruningStage.create(
        "second",
        {query.query_id: ("p-0", "p-1") for query in dataset.queries},
        kind=PruningStageKind.COMPATIBILITY,
        version="compat_v2",
        removal_reasons={query.query_id: {} for query in dataset.queries},
    )

    with pytest.raises(ValueError, match="not a monotonic subset"):
        evaluate_candidate_pruning(
            dataset,
            (first, reintroduced),
            population=_population(dataset),
            provenance=_provenance(),
            query_split=split,
            split_name="test",
            n_resamples=10,
        )


def test_stage_constructor_rejects_a_string_as_a_candidate_collection() -> None:
    with pytest.raises(TypeError, match="sequences of IDs"):
        FrozenPruningStage.create(
            "broken",
            {"q-1": "product-id"},
            kind=PruningStageKind.OTHER_DIAGNOSTIC,
            version="broken-v1",
        )


def test_mutated_frozen_qrels_and_stage_snapshots_are_rejected(tmp_path: Path) -> None:
    dataset = _dataset()
    cast(dict[str, int], dataset.queries[0].relevance_labels)["p-0"] = 0
    with pytest.raises(ValueError, match="changed after hashing"):
        _report(dataset)

    clean_dataset = _dataset()
    stages = _stages(clean_dataset)
    cast(dict[str, tuple[str, ...]], stages[0].retained_candidate_ids)["q-0"] = ("p-9",)
    with pytest.raises(ValueError, match="changed after it was frozen"):
        write_frozen_pruning_trace(stages, tmp_path / "mutated-trace.json")
    split = FrozenQueryGroupSplit.create(clean_dataset, version="pruning-split-v1", seed=13)
    with pytest.raises(ValueError, match="changed after it was frozen"):
        evaluate_candidate_pruning(
            clean_dataset,
            stages,
            population=_population(clean_dataset),
            provenance=_provenance(),
            query_split=split,
            split_name="test",
            n_resamples=10,
        )


def test_mutated_frozen_query_split_is_rejected() -> None:
    dataset = _dataset()
    split = FrozenQueryGroupSplit.create(dataset, version="pruning-split-v1", seed=13)
    first_query = dataset.queries[0].query_id
    assignments = cast(dict[str, str], split.assignments)
    assignments[first_query] = "test" if assignments[first_query] != "test" else "validation"

    with pytest.raises(ValueError, match="split changed after hashing"):
        evaluate_candidate_pruning(
            dataset,
            _stages(dataset),
            population=_population(dataset),
            provenance=_provenance(),
            query_split=split,
            split_name="test",
            n_resamples=10,
        )
