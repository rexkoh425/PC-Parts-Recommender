from __future__ import annotations

import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import numpy as np
import pytest

import pc_build_recommender.ranking.lambdamart as lambdamart_module
from pc_build_recommender.evaluation.manifest import sha256_file
from pc_build_recommender.ranking import (
    LabeledRankingQuery,
    LambdaMARTRanker,
    RankerMetadata,
    ScoredCandidate,
    RankingContext,
    RankingFeatureBuilder,
    prepare_lgbm_data,
    ranker_artifact_manifest_path,
)
from pc_build_recommender.retrieval import (
    FrozenCandidateQuery,
    PinnedCandidateSet,
    QueryGroupSplit,
    RelevanceLabelSource,
)


def _candidate(query_number: int, grade: int) -> ScoredCandidate:
    # Correlate multiple independent features with relevance so the tiny model
    # has a real split to learn while still exercising query groups.
    return ScoredCandidate(
        product_id=f"q{query_number}-g{grade}",
        category="gpu",
        price_sgd=1500 - grade * 100 + query_number,
        retrieval_scores={
            "bm25_score": grade * 2 + query_number * 0.01,
            "vector_similarity": grade / 4,
            "rrf_score": (grade + 1) / 100,
        },
        workload_scores={"gaming": grade * 25},
        signals={
            "availability_score": 1,
            "reliability_score": grade / 4,
            "predicted_workload_score": grade * 25,
        },
        attributes={"vram_gb": 8 + grade * 4},
    )


def _queries(count: int = 5, *, start: int = 0) -> list[LabeledRankingQuery]:
    queries = []
    for query_number in range(start, start + count):
        # Alternate row order to prove labels are associated within groups, not
        # inferred from global row position.
        grades = [0, 1, 2, 3, 4]
        if query_number % 2:
            grades.reverse()
        candidates = [_candidate(query_number, grade) for grade in grades]
        queries.append(
            LabeledRankingQuery.create(
                RankingContext(
                    query_id=f"q{query_number}",
                    budget_sgd=2000,
                    workload_weights={"gaming": 1},
                ),
                candidates,
                grades,
            )
        )
    return queries


def _trained_artifact_ranker(*, ranker_version: str = "ltr-artifact-test-v1") -> LambdaMARTRanker:
    return LambdaMARTRanker(
        ranker_version=ranker_version,
        parameters={
            "n_estimators": 3,
            "num_leaves": 3,
            "min_child_samples": 1,
            "n_jobs": 1,
        },
    ).fit(
        _queries(3),
        training_data_version="artifact-labels-v1",
    )


def _persisted_ranker(tmp_path: Path) -> tuple[Path, Path, Path, LambdaMARTRanker]:
    ranker = _trained_artifact_ranker()
    model_path, metadata_path = ranker.save(tmp_path / "ranker.txt")
    return model_path, metadata_path, ranker_artifact_manifest_path(model_path), ranker


_CRASH_PUBLISH_WORKER = r"""
import os
import sys
from pathlib import Path

import pc_build_recommender.ranking.lambdamart as module
from pc_build_recommender.ranking import LambdaMARTRanker

source_model = Path(sys.argv[1])
bundle = Path(sys.argv[2])
crash_point = sys.argv[3]
ranker = LambdaMARTRanker.load(source_model)
publication_intent_sha256 = sys.argv[4]

if crash_point == "before_rename":
    def crash_before(_source, _destination):
        os._exit(73)
    module._rename_directory_noreplace = crash_before
elif crash_point == "after_rename":
    original_fsync = module._fsync_directory
    resolved_parent = bundle.parent.resolve()
    def crash_after(path):
        if Path(path) == resolved_parent and bundle.is_dir():
            os._exit(74)
        original_fsync(path)
    module._fsync_directory = crash_after
else:
    raise ValueError(crash_point)

ranker.publish_bundle(bundle, publication_intent_sha256=publication_intent_sha256)
"""


_PUBLICATION_INTENT_SHA256 = "a" * 64


def _run_crash_publish_worker(
    *,
    source_model: Path,
    bundle: Path,
    crash_point: str,
    publication_intent_sha256: str = _PUBLICATION_INTENT_SHA256,
) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ)
    environment.update(
        {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    return subprocess.run(
        [
            sys.executable,
            "-c",
            _CRASH_PUBLISH_WORKER,
            str(source_model),
            str(bundle),
            crash_point,
            publication_intent_sha256,
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
        env=environment,
    )


def test_prepare_lgbm_data_keeps_query_rows_contiguous() -> None:
    prepared = prepare_lgbm_data(_queries(3), RankingFeatureBuilder())

    assert prepared.group_sizes == (5, 5, 5)
    assert prepared.features.shape == (15, len(RankingFeatureBuilder.feature_names))
    assert prepared.labels.tolist()[:5] == [0, 1, 2, 3, 4]
    assert prepared.labels.tolist()[5:10] == [4, 3, 2, 1, 0]
    assert [query_id for query_id, _ in prepared.row_keys[:5]] == ["q0"] * 5


@pytest.mark.model
def test_lambdamart_trains_round_trips_and_preserves_scores(tmp_path) -> None:
    ranker = LambdaMARTRanker(
        ranker_version="ltr-test-v1",
        parameters={
            "n_estimators": 30,
            "learning_rate": 0.15,
            "num_leaves": 7,
            "min_child_samples": 1,
            "n_jobs": 1,
        },
    ).fit(
        _queries(4),
        validation_queries=_queries(1, start=50),
        training_data_version="labels-test-v1",
        candidate_set_version="frozen-test-v1",
        early_stopping_rounds=5,
    )
    context = RankingContext(
        query_id="serve",
        budget_sgd=2000,
        workload_weights={"gaming": 1},
    )
    candidates = [_candidate(99, grade) for grade in [2, 0, 4, 1, 3]]
    before = ranker.predict(context, candidates)
    model_path, metadata_path = ranker.save(tmp_path / "ranker.txt")
    manifest_path = ranker_artifact_manifest_path(model_path)

    loaded = LambdaMARTRanker.load(model_path)
    after = loaded.predict(context, candidates)

    np.testing.assert_allclose(after, before, rtol=0, atol=1e-12)
    assert metadata_path.exists()
    assert manifest_path.exists()
    assert loaded.artifact_identity == ranker.artifact_identity
    assert loaded.artifact_identity.model_sha256 == sha256_file(model_path)
    assert loaded.artifact_identity.metadata_sha256 == sha256_file(metadata_path)
    assert loaded.artifact_identity.manifest_sha256 == sha256_file(manifest_path)
    assert loaded.metadata.ranker_version == "ltr-test-v1"
    assert loaded.metadata.ranking_basis == "lightgbm_lambdamart"
    assert loaded.metadata.training_query_count == 4
    assert loaded.metadata.training_row_count == 20
    assert loaded.metadata.training_data_version == "labels-test-v1"
    assert [item.product_id for item in loaded.rank_query(context, candidates)][0] == "q99-g4"


@pytest.mark.model
def test_lambdamart_artifact_rejects_model_metadata_and_manifest_tampering(
    tmp_path: Path,
) -> None:
    for tampered_file in ("model", "metadata", "manifest"):
        case_path = tmp_path / tampered_file
        case_path.mkdir()
        model_path, metadata_path, manifest_path, _ = _persisted_ranker(case_path)
        if tampered_file == "model":
            model_path.write_bytes(model_path.read_bytes() + b"\n# tampered\n")
            expected = "size does not match manifest"
        elif tampered_file == "metadata":
            payload = json.loads(metadata_path.read_text("utf-8"))
            payload["training_query_count"] = 999_999
            metadata_path.write_text(json.dumps(payload), encoding="utf-8")
            expected = "size does not match manifest|digest does not match manifest"
        else:
            payload = json.loads(manifest_path.read_text("utf-8"))
            payload["ranker_version"] = "same-model-tampered-manifest"
            manifest_path.write_text(json.dumps(payload), encoding="utf-8")
            expected = "manifest hash does not match"

        with pytest.raises(ValueError, match=expected):
            LambdaMARTRanker.load(model_path)


@pytest.mark.model
def test_lambdamart_artifact_is_immutable_and_requires_commit_manifest(
    tmp_path: Path,
) -> None:
    model_path, _, manifest_path, ranker = _persisted_ranker(tmp_path)
    loaded_legacy = LambdaMARTRanker.load(model_path)

    assert loaded_legacy.publication_intent_sha256 is None

    with pytest.raises(FileExistsError, match="immutable"):
        ranker.save(model_path)

    manifest_path.unlink()
    with pytest.raises(FileNotFoundError):
        LambdaMARTRanker.load(model_path)


@pytest.mark.model
def test_bundle_publication_round_trips_a_v2_manifest(tmp_path: Path) -> None:
    ranker = _trained_artifact_ranker()
    bundle = tmp_path / "ranker-artifact"

    model_path, metadata_path = ranker.publish_bundle(
        bundle,
        publication_intent_sha256=_PUBLICATION_INTENT_SHA256,
    )
    loaded = LambdaMARTRanker.load(model_path)
    manifest = json.loads(ranker_artifact_manifest_path(model_path).read_text("utf-8"))

    assert model_path == bundle / "ranker.txt"
    assert metadata_path == bundle / "ranker.txt.metadata.json"
    assert manifest["schema_version"].endswith("artifact-manifest.v2")
    assert manifest["publication_intent_sha256"] == ranker.publication_intent_sha256
    assert loaded.publication_intent_sha256 == ranker.publication_intent_sha256
    assert loaded.artifact_identity == ranker.artifact_identity
    assert ranker.verified_artifact_loaded


@pytest.mark.model
def test_bundle_retry_after_crash_before_rename_ignores_orphan_stage(tmp_path: Path) -> None:
    source_model, _, _, ranker = _persisted_ranker(tmp_path / "source")
    bundle = tmp_path / "ranker-artifact"

    crashed = _run_crash_publish_worker(
        source_model=source_model,
        bundle=bundle,
        crash_point="before_rename",
    )

    assert crashed.returncode == 73, crashed.stderr
    assert not bundle.exists()
    assert list(tmp_path.glob(".ranker-artifact.publish-*"))

    model_path, _ = ranker.publish_bundle(
        bundle,
        publication_intent_sha256=_PUBLICATION_INTENT_SHA256,
    )

    assert LambdaMARTRanker.load(model_path).artifact_identity == ranker.artifact_identity


@pytest.mark.model
def test_bundle_retry_after_crash_after_rename_adopts_committed_bytes(tmp_path: Path) -> None:
    source_model, _, _, _ = _persisted_ranker(tmp_path / "source")
    bundle = tmp_path / "ranker-artifact"

    crashed = _run_crash_publish_worker(
        source_model=source_model,
        bundle=bundle,
        crash_point="after_rename",
    )

    assert crashed.returncode == 74, crashed.stderr
    committed = LambdaMARTRanker.load(bundle / "ranker.txt")
    retry = _trained_artifact_ranker()
    retry_created_at = retry.metadata.created_at_utc
    assert retry_created_at != committed.metadata.created_at_utc

    model_path, _ = retry.publish_bundle(
        bundle,
        publication_intent_sha256=_PUBLICATION_INTENT_SHA256,
    )

    assert model_path == bundle / "ranker.txt"
    assert retry.artifact_identity == committed.artifact_identity
    assert retry.metadata.created_at_utc == committed.metadata.created_at_utc
    assert retry.verified_artifact_loaded


@pytest.mark.model
def test_concurrent_same_intent_bundle_publishers_converge_without_overwrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_model, _, _, _ = _persisted_ranker(tmp_path / "source")
    publishers = [LambdaMARTRanker.load(source_model) for _ in range(2)]
    bundle = tmp_path / "ranker-artifact"
    barrier = Barrier(2)
    original_rename = lambdamart_module._rename_directory_noreplace

    def synchronized_rename(source: Path, destination: Path) -> None:
        barrier.wait(timeout=10)
        original_rename(source, destination)

    monkeypatch.setattr(
        lambdamart_module,
        "_rename_directory_noreplace",
        synchronized_rename,
    )
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                ranker.publish_bundle,
                bundle,
                publication_intent_sha256=_PUBLICATION_INTENT_SHA256,
            )
            for ranker in publishers
        ]
        results = [future.result(timeout=15) for future in futures]

    assert results == [(bundle / "ranker.txt", bundle / "ranker.txt.metadata.json")] * 2
    assert publishers[0].artifact_identity == publishers[1].artifact_identity
    assert not list(tmp_path.glob(".ranker-artifact.publish-*"))


@pytest.mark.model
def test_existing_bundle_with_different_intent_or_tampering_is_never_replaced(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "ranker-artifact"
    model_path, _ = _trained_artifact_ranker().publish_bundle(
        bundle,
        publication_intent_sha256=_PUBLICATION_INTENT_SHA256,
    )
    committed_manifest = ranker_artifact_manifest_path(model_path)
    original_manifest = committed_manifest.read_bytes()

    incompatible = _trained_artifact_ranker(ranker_version="ltr-different-intent-v1")
    with pytest.raises(FileExistsError, match="different publication intent"):
        incompatible.publish_bundle(bundle, publication_intent_sha256="b" * 64)
    assert committed_manifest.read_bytes() == original_manifest

    model_path.write_bytes(model_path.read_bytes() + b"\n# tampered\n")
    tampered_model = model_path.read_bytes()
    with pytest.raises(ValueError, match="size does not match manifest"):
        _trained_artifact_ranker().publish_bundle(
            bundle,
            publication_intent_sha256=_PUBLICATION_INTENT_SHA256,
        )
    assert model_path.read_bytes() == tampered_model


@pytest.mark.model
def test_lambdamart_artifact_rejects_oversized_manifest_before_json_parse(
    tmp_path: Path,
) -> None:
    model_path, _, manifest_path, _ = _persisted_ranker(tmp_path)
    manifest_path.write_bytes(b"{" + b" " * (1024 * 1024) + b"}")

    with pytest.raises(ValueError, match="exceeds the .* safety limit"):
        LambdaMARTRanker.load(model_path)


def test_untrained_lambdamart_cannot_predict() -> None:
    with pytest.raises(RuntimeError, match="not been trained"):
        LambdaMARTRanker().predict(
            RankingContext(query_id="q"),
            [_candidate(1, 1)],
        )


@pytest.mark.parametrize(
    "field",
    [
        "training_adjudication_complete",
        "contains_synthetic_labels",
        "query_split_membership_verified",
        "promotion_eligible",
    ],
)
def test_ranker_metadata_rejects_string_booleans(field: str) -> None:
    payload = RankerMetadata(
        ranker_version="ltr-strict-v1",
        ranking_basis="lightgbm_lambdamart",
        feature_version="ranking-features-v1",
        model_type="LGBMRanker",
        feature_names=("bm25_score",),
        created_at_utc="2026-07-23T00:00:00+00:00",
    ).to_dict()
    payload[field] = "false"

    with pytest.raises(TypeError, match=f"{field} must be a boolean"):
        RankerMetadata.from_dict(payload)


def test_lambdamart_rejects_query_leakage_between_splits() -> None:
    with pytest.raises(ValueError, match="must be disjoint"):
        LambdaMARTRanker(parameters={"n_estimators": 2}).fit(
            _queries(2),
            validation_queries=_queries(1),
            training_data_version="labels-test-v1",
        )


def test_lambdamart_rejects_intent_group_leakage_between_splits() -> None:
    with pytest.raises(ValueError, match="query groups must be disjoint"):
        LambdaMARTRanker(parameters={"n_estimators": 2, "n_jobs": 1}).fit(
            _queries(1),
            validation_queries=_queries(1, start=50),
            training_data_version="labels-test-v1",
            query_group_ids={"q0": "same-intent", "q50": "same-intent"},
            query_group_split_checksum="a" * 64,
        )


@pytest.mark.model
def test_silver_lambdamart_training_metadata_is_never_promotable() -> None:
    ranker = LambdaMARTRanker(
        parameters={
            "n_estimators": 2,
            "num_leaves": 3,
            "min_child_samples": 1,
            "n_jobs": 1,
        }
    ).fit(
        _queries(2),
        training_data_version="silver-v1",
        training_label_source="silver",
    )

    assert ranker.metadata.training_label_source == "silver"
    assert not ranker.metadata.promotion_eligible
    assert any("silver" in reason for reason in ranker.metadata.promotion_block_reasons)


@pytest.mark.model
def test_human_lambdamart_verifies_exact_frozen_train_and_validation_membership() -> None:
    labeled_queries = _queries(6)
    frozen_dataset = PinnedCandidateSet.create(
        "human-labels-v1",
        [
            FrozenCandidateQuery(
                query_id=query.context.query_id,
                query_group_id=f"intent-{query.context.query_id}",
                candidate_ids=tuple(candidate.product_id for candidate in query.candidates),
                relevance_labels={
                    candidate.product_id: grade
                    for candidate, grade in zip(
                        query.candidates, query.relevance_grades, strict=True
                    )
                },
            )
            for query in labeled_queries
        ],
        label_source=RelevanceLabelSource.HUMAN,
        adjudication_complete=True,
        judgment_manifest_sha256="b" * 64,
    )
    split = QueryGroupSplit.create(frozen_dataset, version="human-split-v1", seed=3)
    by_id = {query.context.query_id: query for query in labeled_queries}
    training = [
        by_id[query_id]
        for query_id, split_name in split.assignments.items()
        if split_name == "train"
    ]
    validation = [
        by_id[query_id]
        for query_id, split_name in split.assignments.items()
        if split_name == "validation"
    ]

    ranker = LambdaMARTRanker(
        parameters={
            "n_estimators": 2,
            "num_leaves": 3,
            "min_child_samples": 1,
            "n_jobs": 1,
        }
    ).fit(
        training,
        validation_queries=validation,
        training_data_version="human-labels-v1",
        training_label_source="human",
        training_adjudication_complete=True,
        training_judgment_manifest_sha256="b" * 64,
        training_dataset_manifest_sha256="c" * 64,
        training_prelabel_snapshot_sha256="d" * 64,
        training_feature_contract_sha256="e" * 64,
        frozen_query_split=split,
        early_stopping_rounds=1,
    )

    assert ranker.metadata.promotion_eligible
    assert ranker.metadata.query_split_membership_verified
    assert ranker.metadata.query_group_split_checksum == split.checksum

    silver_dataset = PinnedCandidateSet.create(
        "silver-labels-v1",
        frozen_dataset.queries,
        label_source=RelevanceLabelSource.SILVER,
    )
    silver_split = QueryGroupSplit.create(
        silver_dataset,
        version="silver-split-v1",
        seed=3,
    )
    with pytest.raises(ValueError, match="label source conflicts"):
        LambdaMARTRanker(parameters={"n_estimators": 2, "n_jobs": 1}).fit(
            training,
            validation_queries=validation,
            training_data_version="silver-labels-v1",
            training_label_source="human",
            frozen_query_split=silver_split,
        )
