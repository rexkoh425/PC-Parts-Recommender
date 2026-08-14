from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
from training import ranking_evaluation_gate as evaluation_gate
from training._common import sha256_file
from training.evaluate_ranking import main as evaluation_main
from training.ranking_evaluation_gate import (
    RankingEvaluationGateError,
    verify_human_ranking_lineage,
)
from training.register_ranking_evaluation import main as registration_main
from training.train_ranking import main as ranking_main

from pc_build_recommender.evaluation.manifest import sha256_json
from pc_build_recommender.ranking import (
    LambdaMARTRanker,
    load_ranker_promotion_decision,
    verify_ranking_evaluation_bundle,
)
from pc_build_recommender.retrieval import (
    FrozenCandidateSet,
    FrozenQueryGroupSplit,
    HumanJudgmentSet,
    LabelingQuery,
    ReviewerJudgment,
    load_human_judgment_set,
    load_ranking_comparison_report,
    write_human_judgment_set,
)


def _human_judgments(query_count: int = 12) -> HumanJudgmentSet:
    queries = tuple(
        LabelingQuery(
            query_id=f"q{index}",
            query_group_id=f"intent-{index}",
            query_text=f"gaming gpu request {index}",
            category="gpu",
            candidate_ids=(f"q{index}-weak", f"q{index}-strong"),
        )
        for index in range(query_count)
    )
    judgments = tuple(
        ReviewerJudgment(
            query_id=query.query_id,
            product_id=product_id,
            reviewer_id=reviewer_id,
            grade=grade,
            rationale=f"{reviewer_id} graded the frozen candidate {grade}",
            reviewed_at_utc="2026-07-23T01:00:00Z",
        )
        for query in queries
        for product_id, grade in zip(query.candidate_ids, (1, 4), strict=True)
        for reviewer_id in ("reviewer-1", "reviewer-2")
    )
    return HumanJudgmentSet(
        dataset_name="human-ranking-evaluation",
        dataset_version="human-ranking-v1",
        queries=queries,
        judgments=judgments,
        adjudications=(),
    )


def _feature_rows(query_count: int = 12) -> list[dict[str, object]]:
    return [
        {
            "schema_version": "pc-build-recommender.ranking-prelabel-query.v1",
            "query_id": f"q{index}",
            "query_group_id": f"intent-{index}",
            "category": "gpu",
            "context": {
                "query_id": f"q{index}",
                "budget_sgd": 2000,
                "workload_weights": {"gaming": 1.0},
            },
            "candidates": [
                {
                    "product_id": f"q{index}-weak",
                    "category": "gpu",
                    "price_sgd": 900,
                    "retrieval_scores": {"bm25_score": 1.0},
                    "workload_scores": {"gaming": 50},
                    "relevance_grade": 1,
                },
                {
                    "product_id": f"q{index}-strong",
                    "category": "gpu",
                    "price_sgd": 1200,
                    "retrieval_scores": {"bm25_score": 2.0},
                    "workload_scores": {"gaming": 90},
                    "relevance_grade": 4,
                },
            ],
            "candidate_ids_sha256": sha256_json(
                [f"q{index}-weak", f"q{index}-strong"]
            ),
            "feature_matrix_sha256": "f" * 64,
        }
        for index in range(query_count)
    ]


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_dataset_manifest(
    *,
    input_path: Path,
    human_path: Path,
    qrels_path: Path,
    split_path: Path,
) -> Path:
    rows = [
        json.loads(line)
        for line in input_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    row_hashes: dict[str, str] = {}
    candidate_hashes: dict[str, str] = {}
    for row in rows:
        prelabel_candidates = []
        product_ids = []
        for raw_candidate in row["candidates"]:
            candidate = dict(raw_candidate)
            candidate.pop("relevance_grade")
            prelabel_candidates.append(candidate)
            product_ids.append(candidate["product_id"])
        prelabel_row = {**row, "candidates": prelabel_candidates}
        row_hashes[row["query_id"]] = sha256_json(prelabel_row)
        candidate_hashes[row["query_id"]] = sha256_json(product_ids)
    snapshot_sha256 = sha256_json({"query_row_sha256": row_hashes})
    evidence_path = input_path.parent / "evidence-snapshots.json"
    evidence_path.write_text(
        json.dumps(
            {
                "schema_version": (
                    "pc-build-recommender.annotation-evidence-snapshots.v1"
                ),
                "groups": [
                    {
                        "group_key": row["query_id"],
                        "context_payload": {
                            "ranking_prelabel_binding": {
                                "snapshot_sha256": snapshot_sha256,
                                "query_row_sha256": row_hashes[row["query_id"]],
                                "candidate_ids_sha256": candidate_hashes[
                                    row["query_id"]
                                ],
                            }
                        },
                    }
                    for row in rows
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    release_files = {
        name: {
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for name, path in {
            "evidence-snapshots.json": evidence_path,
            "human-judgments.json": human_path,
            "qrels.json": qrels_path,
            "query-split.json": split_path,
        }.items()
    }
    human = load_human_judgment_set(human_path)
    qrels = FrozenCandidateSet.load(qrels_path)
    split = FrozenQueryGroupSplit.load(split_path)
    manifest: dict[str, object] = {
        "schema_version": (
            "pc-build-recommender.ranking-labeled-snapshot-manifest.v1"
        ),
        "prelabel": {
            "capture_manifest_file_sha256": "a" * 64,
            "snapshot_sha256": snapshot_sha256,
            "feature_file_sha256": "b" * 64,
            "candidate_universe_sha256": "c" * 64,
            "feature_contract_sha256": "d" * 64,
            "query_row_sha256": row_hashes,
        },
        "annotation_release": {
            "manifest_file_sha256": "e" * 64,
            "release_sha256": sha256_json(release_files),
            "files": release_files,
        },
        "human_judgments": {
            "content_sha256": human.content_sha256,
            "minimum_independent_reviewers_per_pair": 2,
        },
        "qrels": {
            "version": qrels.version,
            "checksum": qrels.checksum,
            "evidence_checksum": qrels.evidence_checksum,
            "judgment_manifest_sha256": qrels.judgment_manifest_sha256,
        },
        "query_split": {
            "version": split.version,
            "checksum": split.checksum,
            "assignments": dict(sorted(split.assignments.items())),
        },
        "dataset": {
            "query_count": len(rows),
            "row_count": sum(len(row["candidates"]) for row in rows),
        },
        "files": {
            "ranking.jsonl": {
                "sha256": sha256_file(input_path),
                "size_bytes": input_path.stat().st_size,
            }
        },
    }
    manifest["manifest_sha256"] = sha256_json(manifest)
    manifest_path = input_path.parent / "dataset-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def _write_lineage(
    tmp_path: Path,
    *,
    query_count: int = 12,
) -> tuple[HumanJudgmentSet, Path, Path, Path, FrozenQueryGroupSplit]:
    judgments = _human_judgments(query_count)
    adjudicated = judgments.adjudicate()
    human_path = tmp_path / "human-judgments.json"
    qrels_path = tmp_path / "qrels.json"
    split_path = tmp_path / "query-split.json"
    write_human_judgment_set(judgments, human_path)
    adjudicated.frozen_candidates.save(qrels_path)
    split = FrozenQueryGroupSplit.create(
        adjudicated.frozen_candidates,
        version="human-ranking-split-v1",
        weights={"train": 0.6, "validation": 0.2, "test": 0.2},
        seed=7,
    )
    assert set(split.assignments.values()) == {"train", "validation", "test"}
    split.save(split_path)
    return judgments, human_path, qrels_path, split_path, split


def _human_cli_args(
    *,
    input_path: Path,
    artifact_dir: Path,
    human_path: Path,
    qrels_path: Path,
    split_path: Path,
) -> list[str]:
    dataset_manifest = _write_dataset_manifest(
        input_path=input_path,
        human_path=human_path,
        qrels_path=qrels_path,
        split_path=split_path,
    )
    return [
        "--input",
        str(input_path),
        "--artifact-dir",
        str(artifact_dir),
        "--candidate-set-version",
        "human-ranking-v1",
        "--label-provenance",
        "human",
        "--human-judgments",
        str(human_path),
        "--qrels",
        str(qrels_path),
        "--frozen-query-split",
        str(split_path),
        "--dataset-manifest",
        str(dataset_manifest),
        "--early-stopping-rounds",
        "1",
        "--parameters",
        json.dumps(
            {
                "n_estimators": 2,
                "num_leaves": 3,
                "min_child_samples": 1,
                "n_jobs": 1,
            }
        ),
    ]


def _registration_cli_args(
    *,
    input_path: Path,
    artifact_dir: Path,
    human_path: Path,
    qrels_path: Path,
    split_path: Path,
    intent_root: Path,
    ledger_dir: Path,
    seed: int = 20260723,
) -> list[str]:
    return [
        "--feature-snapshot",
        str(input_path),
        "--dataset-manifest",
        str(input_path.parent / "dataset-manifest.json"),
        "--human-judgments",
        str(human_path),
        "--qrels",
        str(qrels_path),
        "--frozen-query-split",
        str(split_path),
        "--ranker-model",
        str(artifact_dir / "ranker-artifact" / "ranker.txt"),
        "--intent-root",
        str(intent_root),
        "--ledger-dir",
        str(ledger_dir),
        "--n-resamples",
        "50",
        "--seed",
        str(seed),
    ]


@pytest.mark.model
def test_human_ranking_cli_persists_verified_lineage_and_uses_frozen_split(
    tmp_path: Path,
) -> None:
    judgments, human_path, qrels_path, split_path, split = _write_lineage(tmp_path)
    input_path = tmp_path / "ranking.jsonl"
    artifact_dir = tmp_path / "artifact"
    _write_jsonl(input_path, _feature_rows())
    cli_args = _human_cli_args(
        input_path=input_path,
        artifact_dir=artifact_dir,
        human_path=human_path,
        qrels_path=qrels_path,
        split_path=split_path,
    )

    assert ranking_main(cli_args) == 0

    model_path = artifact_dir / "ranker-artifact" / "ranker.txt"
    loaded = LambdaMARTRanker.load(model_path)
    metadata = loaded.metadata
    assert metadata.training_label_source == "human"
    assert metadata.training_adjudication_complete
    assert not metadata.contains_synthetic_labels
    assert metadata.training_judgment_manifest_sha256 == judgments.content_sha256
    assert metadata.query_group_split_checksum == split.checksum
    assert metadata.query_split_membership_verified
    assert metadata.promotion_eligible

    evidence = json.loads((artifact_dir / "training_evidence.json").read_text("utf-8"))
    assert evidence["schema_version"] == ("pc-build-recommender.ranking-training-evidence.v3")
    assert evidence["verified_label_source"] == "human"
    assert evidence["adjudication_complete"] is True
    assert evidence["minimum_independent_reviewers_per_pair"] == 2
    assert evidence["judgment_manifest_sha256"] == judgments.content_sha256
    assert evidence["qrels_manifest_sha256"] == sha256_file(qrels_path)
    assert evidence["human_judgments"]["content_sha256"] == judgments.content_sha256
    assert evidence["qrels"]["file_sha256"] == sha256_file(qrels_path)
    assert evidence["frozen_query_split"]["file_sha256"] == sha256_file(split_path)
    assert evidence["frozen_query_split"]["checksum"] == split.checksum
    assert evidence["dataset_manifest"]["manifest_sha256"]
    assert evidence["dataset_manifest"]["prelabel_snapshot_sha256"]
    assert evidence["ranker_metadata"] == metadata.to_dict()

    report = json.loads((artifact_dir / "training_report.json").read_text("utf-8"))
    assert Path(report["artifact"]["model_path"]) == model_path.resolve()
    assert report["artifact"]["publication_intent_sha256"] == loaded.publication_intent_sha256
    assert report["ranker"]["promotion_eligible"] is True
    assert "heldout_test" not in report
    assert report["validation_diagnostic"]["promotion_evidence"] is False
    assert report["promotion_decision"]["emitted"] is False
    assert "test_query_hashes" not in evidence
    assert evidence["frozen_test_cohort"]["metrics_accessed"] is False
    assert report["resources"]["materialization_memory_expansion_factor"] == 12.0
    assert report["resources"]["materialization_runtime_memory_mb"] == 512.0
    assert report["resources"]["host_memory_preflight"]["max_used_gib"] == 55.0

    # A command retry after the bundle commit must converge on the exact same
    # immutable bytes even though the replacement training attempt has a new timestamp.
    committed_identity = loaded.artifact_identity
    assert ranking_main(cli_args) == 0
    assert LambdaMARTRanker.load(model_path).artifact_identity == committed_identity


@pytest.mark.model
def test_ranking_evaluation_cli_binds_loaded_artifact_and_writes_promotion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, human_path, qrels_path, split_path, _ = _write_lineage(
        tmp_path, query_count=400
    )
    rows = _feature_rows(400)
    for row in rows:
        candidates = row["candidates"]
        assert isinstance(candidates, list)
        weak, strong = candidates
        assert isinstance(weak, dict) and isinstance(strong, dict)
        weak["retrieval_scores"] = {"bm25_score": 2.0, "rrf_score": 1.0}
        strong["retrieval_scores"] = {"bm25_score": 1.0, "rrf_score": 2.0}
    input_path = tmp_path / "ranking.jsonl"
    artifact_dir = tmp_path / "artifact"
    evaluation_dir = tmp_path / "evaluation"
    intent_root = tmp_path / "evaluation-intents"
    ledger_dir = tmp_path / "evaluation-ledger"
    _write_jsonl(input_path, rows)
    assert (
        ranking_main(
            _human_cli_args(
                input_path=input_path,
                artifact_dir=artifact_dir,
                human_path=human_path,
                qrels_path=qrels_path,
                split_path=split_path,
            )
        )
        == 0
    )

    registration_args = _registration_cli_args(
        input_path=input_path,
        artifact_dir=artifact_dir,
        human_path=human_path,
        qrels_path=qrels_path,
        split_path=split_path,
        intent_root=intent_root,
        ledger_dir=ledger_dir,
    )
    original_loader = evaluation_gate.load_human_judgment_set

    def _unexpected_label_parse(_path: Path) -> None:
        raise AssertionError("preregistration must not parse human test labels")

    monkeypatch.setattr(evaluation_gate, "load_human_judgment_set", _unexpected_label_parse)
    assert registration_main(registration_args) == 0
    monkeypatch.setattr(evaluation_gate, "load_human_judgment_set", original_loader)
    intent_paths = list(intent_root.glob("*.json"))
    assert len(intent_paths) == 1
    intent_path = intent_paths[0]
    sealed_dir = evaluation_dir / intent_path.stem
    evaluation_args = [
        "--feature-snapshot",
        str(input_path),
        "--dataset-manifest",
        str(input_path.parent / "dataset-manifest.json"),
        "--human-judgments",
        str(human_path),
        "--qrels",
        str(qrels_path),
        "--frozen-query-split",
        str(split_path),
        "--ranker-model",
        str(artifact_dir / "ranker-artifact" / "ranker.txt"),
        "--evaluation-intent",
        str(intent_path),
        "--ledger-dir",
        str(ledger_dir),
        "--output-dir",
        str(evaluation_dir),
    ]
    assert (
        evaluation_main(evaluation_args)
        == 0
    )

    loaded = LambdaMARTRanker.load(artifact_dir / "ranker-artifact" / "ranker.txt")
    report_path = sealed_dir / "ranking-comparison.json"
    decision_path = sealed_dir / "ranker-promotion-decision.json"
    report = load_ranking_comparison_report(report_path)
    decision = load_ranker_promotion_decision(decision_path)
    binding = report["artifact_bound_rankings"]["lambdamart"]
    assert decision.passed
    assert binding["artifact_identity"] == loaded.artifact_identity.to_dict()
    assert decision.ranker_model_sha256 == loaded.artifact_identity.model_sha256
    assert decision.measured_values["artifact_binding_sha256"] == binding["evidence_sha256"]
    assert len(list((ledger_dir / "accesses").glob("*.json"))) == 1
    assert len(list((ledger_dir / "completions").glob("*.json"))) == 1
    verified_bundle = verify_ranking_evaluation_bundle(sealed_dir / "manifest.json")
    assert verified_bundle.promotion_decision == decision
    assert verified_bundle.comparison_report_path == report_path.resolve()
    assert verified_bundle.n_resamples == 50
    assert {
        path.name for path in sealed_dir.iterdir() if path.is_file()
    } == {
        "evaluation-intent.json",
        "ranking-test-registration.json",
        "ranking-test-access.json",
        "ranking-test-completion.json",
        "ranking-comparison.json",
        "ranker-promotion-decision.json",
        "manifest.json",
    }

    sealed_bytes = {path.name: path.read_bytes() for path in sealed_dir.iterdir()}
    assert evaluation_main(evaluation_args) == 0
    assert {path.name: path.read_bytes() for path in sealed_dir.iterdir()} == sealed_bytes
    assert len(list((ledger_dir / "accesses").glob("*.json"))) == 1

    with pytest.raises(RankingEvaluationGateError, match="already registered"):
        registration_main(
            _registration_cli_args(
                input_path=input_path,
                artifact_dir=artifact_dir,
                human_path=human_path,
                qrels_path=qrels_path,
                split_path=split_path,
                intent_root=intent_root,
                ledger_dir=ledger_dir,
                seed=99,
            )
        )

    report_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(RankingEvaluationGateError, match="existing sealed evaluation"):
        evaluation_main(evaluation_args)


def test_human_ranking_cli_requires_complete_authoritative_lineage(tmp_path: Path) -> None:
    input_path = tmp_path / "ranking.jsonl"
    artifact_dir = tmp_path / "artifact"
    _write_jsonl(input_path, _feature_rows(3))

    with pytest.raises(ValueError, match="require --human-judgments, --qrels"):
        ranking_main(
            [
                "--input",
                str(input_path),
                "--artifact-dir",
                str(artifact_dir),
                "--candidate-set-version",
                "human-ranking-v1",
                "--label-provenance",
                "human",
            ]
        )

    assert not artifact_dir.exists()


def test_human_ranking_cli_rejects_feature_grade_drift(tmp_path: Path) -> None:
    _, human_path, qrels_path, split_path, _ = _write_lineage(tmp_path)
    rows = _feature_rows()
    first_candidates = rows[0]["candidates"]
    assert isinstance(first_candidates, list)
    assert isinstance(first_candidates[0], dict)
    first_candidates[0]["relevance_grade"] = 0
    input_path = tmp_path / "ranking.jsonl"
    artifact_dir = tmp_path / "artifact"
    _write_jsonl(input_path, rows)

    with pytest.raises(ValueError, match="grades differ from adjudicated qrels"):
        ranking_main(
            _human_cli_args(
                input_path=input_path,
                artifact_dir=artifact_dir,
                human_path=human_path,
                qrels_path=qrels_path,
                split_path=split_path,
            )
        )

    assert not artifact_dir.exists()


def test_human_ranking_cli_rejects_post_label_feature_rewrite_even_if_rehashed(
    tmp_path: Path,
) -> None:
    _, human_path, qrels_path, split_path, _ = _write_lineage(tmp_path)
    input_path = tmp_path / "ranking.jsonl"
    artifact_dir = tmp_path / "artifact"
    rows = _feature_rows()
    _write_jsonl(input_path, rows)
    cli_args = _human_cli_args(
        input_path=input_path,
        artifact_dir=artifact_dir,
        human_path=human_path,
        qrels_path=qrels_path,
        split_path=split_path,
    )

    candidates = rows[0]["candidates"]
    assert isinstance(candidates, list)
    assert isinstance(candidates[0], dict)
    candidates[0]["price_sgd"] = 1
    _write_jsonl(input_path, rows)
    manifest_path = tmp_path / "dataset-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["ranking.jsonl"] = {
        "sha256": sha256_file(input_path),
        "size_bytes": input_path.stat().st_size,
    }
    manifest.pop("manifest_sha256")
    manifest["manifest_sha256"] = sha256_json(manifest)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="changed pre-label features"):
        ranking_main(cli_args)

    assert not artifact_dir.exists()


def test_human_ranking_cli_rejects_qrels_from_different_judgment_manifest(
    tmp_path: Path,
) -> None:
    judgments, human_path, qrels_path, split_path, _ = _write_lineage(tmp_path)
    changed_first = replace(
        judgments.judgments[0],
        rationale="a materially different reviewer rationale",
    )
    changed = replace(
        judgments,
        judgments=(changed_first, *judgments.judgments[1:]),
    )
    write_human_judgment_set(changed, human_path)
    input_path = tmp_path / "ranking.jsonl"
    artifact_dir = tmp_path / "artifact"
    _write_jsonl(input_path, _feature_rows())

    with pytest.raises(ValueError, match="frozen qrels do not match"):
        ranking_main(
            _human_cli_args(
                input_path=input_path,
                artifact_dir=artifact_dir,
                human_path=human_path,
                qrels_path=qrels_path,
                split_path=split_path,
            )
        )

    assert not artifact_dir.exists()


def test_sealed_evaluator_independently_rejects_forged_two_reviewer_manifest(
    tmp_path: Path,
) -> None:
    judgments, human_path, qrels_path, split_path, _ = _write_lineage(tmp_path)
    input_path = tmp_path / "ranking.jsonl"
    _write_jsonl(input_path, _feature_rows())
    manifest_path = _write_dataset_manifest(
        input_path=input_path,
        human_path=human_path,
        qrels_path=qrels_path,
        split_path=split_path,
    )

    one_reviewer = replace(
        judgments,
        judgments=tuple(
            judgment
            for judgment in judgments.judgments
            if judgment.reviewer_id == "reviewer-1"
        ),
    )
    write_human_judgment_set(one_reviewer, human_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["human_judgments"]["content_sha256"] = one_reviewer.content_sha256
    manifest["annotation_release"]["files"]["human-judgments.json"] = {
        "sha256": sha256_file(human_path),
        "size_bytes": human_path.stat().st_size,
    }
    manifest.pop("manifest_sha256")
    manifest["manifest_sha256"] = sha256_json(manifest)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RankingEvaluationGateError, match="two independent"):
        verify_human_ranking_lineage(
            ranking_path=input_path,
            manifest_path=manifest_path,
            human_judgments_path=human_path,
            qrels_path=qrels_path,
            query_split_path=split_path,
        )


def test_human_ranking_cli_rejects_rechecksummed_split_group_drift(
    tmp_path: Path,
) -> None:
    _, human_path, qrels_path, split_path, split = _write_lineage(tmp_path)
    changed_groups = dict(split.query_group_ids)
    changed_groups["q0"] = "different-intent"
    content = split.content_payload()
    content["query_group_ids"] = dict(sorted(changed_groups.items()))
    changed_split = FrozenQueryGroupSplit(
        version=split.version,
        dataset_checksum=split.dataset_checksum,
        dataset_evidence_checksum=split.dataset_evidence_checksum,
        label_source=split.label_source,
        adjudication_complete=split.adjudication_complete,
        contains_synthetic_labels=split.contains_synthetic_labels,
        judgment_manifest_sha256=split.judgment_manifest_sha256,
        query_group_ids=changed_groups,
        assignments=split.assignments,
        weights=split.weights,
        seed=split.seed,
        checksum=sha256_json(content),
    )
    changed_split.save(split_path)
    input_path = tmp_path / "ranking.jsonl"
    artifact_dir = tmp_path / "artifact"
    _write_jsonl(input_path, _feature_rows())

    with pytest.raises(ValueError, match="split query groups do not match"):
        ranking_main(
            _human_cli_args(
                input_path=input_path,
                artifact_dir=artifact_dir,
                human_path=human_path,
                qrels_path=qrels_path,
                split_path=split_path,
            )
        )

    assert not artifact_dir.exists()
