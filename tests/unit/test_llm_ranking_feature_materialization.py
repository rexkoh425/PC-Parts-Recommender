from __future__ import annotations

import json

from scripts.materialize_llm_ranking_features import materialize
from scripts.snapshot_llm_ranking_features import snapshot

from pc_build_recommender.evaluation.manifest import sha256_file


def test_features_use_requirements_and_specs_without_changing_labels(tmp_path):
    queue = tmp_path / "tasks.jsonl"
    tasks = []
    for product_id, cores in (("a", 12), ("b", None)):
        tasks.append(
            {
                "task_id": f"q:{product_id}",
                "query_id": "q",
                "candidate": {"product_id": product_id},
                "judge_input": {
                    "specifications": {"cores": cores},
                    "requirements": [
                        {"field": "cores", "operator": "ge", "value": 8, "hard": True}
                    ],
                },
            }
        )
    queue.write_text("".join(json.dumps(t) + "\n" for t in tasks), encoding="utf-8")
    feature_dir = tmp_path / "features"
    snapshot(queue, feature_dir)
    dataset = tmp_path / "ranking.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "context": {"query_id": "q"},
                "candidates": [
                    {"product_id": "a", "category": "cpu", "relevance_grade": 4},
                    {"product_id": "b", "category": "cpu", "relevance_grade": 2},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "training_dataset_ready": True,
                "unresolved_task_count": 0,
                "ranking_dataset_sha256": sha256_file(dataset),
                "input_sha256": sha256_file(queue),
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "out"
    materialize(dataset, summary, feature_dir, out)
    row = json.loads((out / "ranking_dataset.llm_weak.jsonl").read_text(encoding="utf-8"))
    assert [c["relevance_grade"] for c in row["candidates"]] == [4, 2]
    assert [c["signals"]["specification_match"] for c in row["candidates"]] == [1.0, 0.0]
    assert [c["signals"]["specification_unknown_fraction"] for c in row["candidates"]] == [0.0, 1.0]
