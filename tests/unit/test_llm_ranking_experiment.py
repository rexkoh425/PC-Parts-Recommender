from __future__ import annotations

import json
from argparse import Namespace

import pytest
from scripts.llm_ranking_experiment import SEED, tune
from scripts.llm_ranking_experiment import test as evaluate_test
from training.train_ranking import _load_queries, _split_queries

from pc_build_recommender.evaluation.manifest import sha256_file


def test_diagnostic_runs_and_seals_the_holdout(tmp_path, capsys):
    dataset = tmp_path / "labeled.jsonl"
    rows = []
    for i in range(300):
        rows.append(
            {
                "context": {"query_id": f"q{i}", "query_group_id": f"q{i}"},
                "candidates": [
                    {
                        "product_id": f"p{j}",
                        "category": "cpu",
                        "relevance_grade": j,
                        "retrieval_scores": {
                            "bm25_score": float(j),
                            "vector_similarity": j / 3,
                            "rrf_score": j / 10,
                        },
                    }
                    for j in range(3)
                ],
            }
        )
    dataset.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    split = _split_queries(_load_queries(dataset), seed=SEED)
    snapshot = []
    for name, queries in split.items():
        for q in queries:
            snapshot.append(
                {
                    "query_id": q.context.query_id,
                    "query_group_id": q.context.query_group_id,
                    "judge_model": "original" if q.context.query_id == "q0" else "alternative",
                    "split": name,
                    "candidate_ids": ["p0", "p1", "p2"],
                    "rankings": {k: ["p2", "p1", "p0"] for k in ("bm25", "vector", "rrf")},
                }
            )
    queue_dir = tmp_path / "queue"
    queue_dir.mkdir()
    snapshot_path = queue_dir / "snapshot.json"
    snapshot_path.write_text(json.dumps(snapshot), encoding="utf-8")
    (queue_dir / "manifest.json").write_text(
        json.dumps(
            {"snapshot_sha256": sha256_file(snapshot_path), "tasks_sha256": "a" * 64, "seed": SEED}
        ),
        encoding="utf-8",
    )
    (tmp_path / "query-split-audit.json").write_text(
        json.dumps(
            {
                "snapshot_sha256": sha256_file(snapshot_path),
                "cross_split_matches": [],
                "previous_query_matches": [],
            }
        ),
        encoding="utf-8",
    )
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "training_dataset_ready": True,
                "unresolved_task_count": 0,
                "label_source": "llm_weak",
                "ranking_dataset_sha256": sha256_file(dataset),
                "input_sha256": "a" * 64,
                "model_id": "mixed-by-query",
                "primary_judge_model": "alternative",
                "prompt_version": "test-only",
            }
        ),
        encoding="utf-8",
    )
    args = Namespace(
        dataset=dataset, snapshot=snapshot_path, summary=summary, output=tmp_path / "out"
    )
    tune(args)
    selection = json.loads((args.output / "selection.json").read_text(encoding="utf-8"))
    assert selection["primary_judge_model"] == "alternative"
    assert sum(selection["split_query_counts"].values()) == 299
    evaluate_test(args)
    report = json.loads((args.output / "sealed-test.json").read_text(encoding="utf-8"))
    assert report["group_count"] >= 50
    assert report["label_source"] == "llm_weak"
    assert not report["eligible_for_promotion"]
    with pytest.raises(FileExistsError):
        evaluate_test(args)
    capsys.readouterr()
