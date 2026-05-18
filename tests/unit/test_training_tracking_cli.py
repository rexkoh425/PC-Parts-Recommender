from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from training import benchmark_entity_resolution_transfer as transfer_benchmark
from training.train_entity_resolution import (
    build_parser as entity_parser,
)
from training.train_entity_resolution import (
    main as entity_main,
)
from training.train_entity_resolution_human import main as human_entity_main
from training.train_performance import build_parser as performance_parser
from training.train_ranking import (
    _load_queries,
    _promotion_blockers,
)
from training.train_ranking import (
    build_parser as ranking_parser,
)
from training.train_ranking import (
    main as ranking_main,
)

from pc_build_recommender.ranking import RankerMetadata


def _ranking_query(query_id: str) -> dict[str, object]:
    return {
        "context": {
            "query_id": query_id,
            "budget_sgd": 2000,
            "workload_weights": {"gaming": 1.0},
        },
        "candidates": [
            {
                "product_id": f"{query_id}-weak",
                "category": "gpu",
                "price_sgd": 900,
                "retrieval_scores": {"bm25_score": 1.0},
                "workload_scores": {"gaming": 50},
                "relevance_grade": 1,
            },
            {
                "product_id": f"{query_id}-strong",
                "category": "gpu",
                "price_sgd": 1200,
                "retrieval_scores": {"bm25_score": 2.0},
                "workload_scores": {"gaming": 90},
                "relevance_grade": 4,
            },
        ],
    }


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    "parser,required",
    [
        (entity_parser(), ["--input", "pairs.jsonl", "--artifact-dir", "artifact"]),
        (performance_parser(), ["--input", "scores.csv", "--artifact-dir", "artifact"]),
        (
            ranking_parser(),
            [
                "--input",
                "queries.jsonl",
                "--artifact-dir",
                "artifact",
                "--candidate-set-version",
                "frozen-v1",
                "--label-provenance",
                "human",
            ],
        ),
    ],
)
def test_training_clis_do_not_auto_enable_mlflow(
    parser: argparse.ArgumentParser, required: list[str]
) -> None:
    args = parser.parse_args(required)

    assert args.track_mlflow is False
    assert args.mlflow_tracking_uri is None


def test_ranking_loader_preserves_query_groups_and_grades(tmp_path: Path) -> None:
    path = tmp_path / "ranking.jsonl"
    _write_jsonl(path, [_ranking_query(f"q{index}") for index in range(3)])

    queries = _load_queries(path)

    assert [query.context.query_id for query in queries] == ["q0", "q1", "q2"]
    assert queries[0].relevance_grades == (1, 4)
    assert [candidate.product_id for candidate in queries[0].candidates] == [
        "q0-weak",
        "q0-strong",
    ]


@pytest.mark.parametrize("invalid_grade", [True, 1.0, "1"])
def test_ranking_loader_does_not_coerce_relevance_grades(
    tmp_path: Path,
    invalid_grade: object,
) -> None:
    row = _ranking_query("q0")
    candidates = row["candidates"]
    assert isinstance(candidates, list)
    assert isinstance(candidates[0], dict)
    candidates[0]["relevance_grade"] = invalid_grade
    path = tmp_path / "ranking.jsonl"
    _write_jsonl(path, [row])

    with pytest.raises(ValueError, match="must be an integer from 0 to 4"):
        _load_queries(path)


def test_silver_ranking_data_is_permanently_non_promotable() -> None:
    metadata = RankerMetadata(
        ranker_version="ltr-diagnostic",
        ranking_basis="lightgbm_lambdamart",
        feature_version="ranking-features-v1",
        model_type="LGBMRanker",
        feature_names=("bm25_score",),
        created_at_utc="2026-07-23T00:00:00+00:00",
        training_label_source="silver",
        promotion_block_reasons=("training label source is silver, not human",),
    )

    blockers = _promotion_blockers(
        ranker_metadata=metadata,
        minimum_independent_reviewers=0,
        query_count=150,
        row_count=2000,
        relative_improvement=25.0,
    )

    assert any("silver" in blocker for blocker in blockers)
    assert any("independent human" in blocker for blocker in blockers)


def test_silver_training_requires_explicit_diagnostic_opt_in(tmp_path: Path) -> None:
    path = tmp_path / "ranking.jsonl"
    _write_jsonl(path, [_ranking_query(f"q{index}") for index in range(3)])

    with pytest.raises(ValueError, match="non-human relevance labels require"):
        ranking_main(
            [
                "--input",
                str(path),
                "--artifact-dir",
                str(tmp_path / "artifact"),
                "--candidate-set-version",
                "silver-frozen-v1",
                "--label-provenance",
                "silver",
            ]
        )

    assert not (tmp_path / "artifact").exists()


def test_ranking_cli_checks_host_memory_before_parsing_materialized_input(tmp_path: Path) -> None:
    path = tmp_path / "ranking.jsonl"
    path.write_text("this is deliberately not JSON\n", encoding="utf-8")

    with pytest.raises(MemoryError, match="host memory preflight refused training"):
        ranking_main(
            [
                "--input",
                str(path),
                "--artifact-dir",
                str(tmp_path / "artifact"),
                "--candidate-set-version",
                "silver-frozen-v1",
                "--label-provenance",
                "silver",
                "--allow-non-human-labels",
                "--max-host-used-gb",
                "0.01",
            ]
        )


def test_entity_cli_checks_host_memory_before_parsing_materialized_input(tmp_path: Path) -> None:
    path = tmp_path / "pairs.jsonl"
    path.write_text("this is deliberately not JSON\n", encoding="utf-8")

    with pytest.raises(MemoryError, match="host memory preflight refused training"):
        entity_main(
            [
                "--input",
                str(path),
                "--artifact-dir",
                str(tmp_path / "artifact"),
                "--max-host-used-gb",
                "0.01",
            ]
        )


def test_human_entity_cli_checks_host_memory_before_importing_review_queue(tmp_path: Path) -> None:
    path = tmp_path / "reviewed-queue.jsonl"
    path.write_text("this is deliberately not JSON\n", encoding="utf-8")

    with pytest.raises(MemoryError, match="host memory preflight refused training"):
        human_entity_main(
            [
                "--review-queue",
                str(path),
                "--artifact-dir",
                str(tmp_path / "artifact"),
                "--max-host-used-gb",
                "0.01",
            ]
        )


def test_transfer_benchmark_checks_host_memory_before_loading_materialized_splits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    materialized = tmp_path / "materialized"
    materialized.mkdir()
    for split_name in ("train", "validation", "test"):
        (materialized / f"pairs.{split_name}.jsonl").write_text(
            "this is deliberately not JSON\n",
            encoding="utf-8",
        )
    (materialized / "dataset_manifest.json").write_text("{}\n", encoding="utf-8")

    class _Adapter:
        def __init__(self, *, raw_root: Path) -> None:
            assert raw_root == tmp_path / "raw"

        def fetch(self, *, archive_path: Path | None) -> SimpleNamespace:
            assert archive_path is None
            return SimpleNamespace(content_sha256="a" * 64)

        def parse(self, snapshot: SimpleNamespace) -> object:
            assert snapshot.content_sha256 == "a" * 64
            return object()

    monkeypatch.setattr(transfer_benchmark, "ZenodoEntityMatchingDn7Adapter", _Adapter)
    monkeypatch.setattr(
        transfer_benchmark,
        "materialize_transfer_pairs",
        lambda *args, **kwargs: materialized,
    )
    monkeypatch.setattr(
        transfer_benchmark,
        "load_materialized_splits",
        lambda _path: pytest.fail("split parsing must not run after a rejected memory preflight"),
    )

    with pytest.raises(MemoryError, match="host memory preflight refused training"):
        transfer_benchmark.main(
            [
                "--raw-root",
                str(tmp_path / "raw"),
                "--processed-root",
                str(tmp_path / "processed"),
                "--artifact-dir",
                str(tmp_path / "artifact"),
                "--max-host-used-gb",
                "0.01",
            ]
        )
