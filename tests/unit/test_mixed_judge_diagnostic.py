from __future__ import annotations

import json
from pathlib import Path

from scripts.combine_mixed_judgments import combine
from scripts.prepare_mixed_judge_queue import prepare
from training.llm_judge import PROMPT_VERSION, RUBRIC, _canonical_hash

from pc_build_recommender.evaluation.manifest import sha256_file


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(x) + "\n" for x in rows), encoding="utf-8")


def result(task: dict, model: str) -> dict:
    return {
        "task_id": task["task_id"],
        "task_hash": _canonical_hash(
            {
                "task": task,
                "model_id": model,
                "prompt_version": PROMPT_VERSION,
                "rubric_sha256": _canonical_hash(RUBRIC),
            }
        ),
        "query_id": task["query_id"],
        "query_group_id": task["query_group_id"],
        "product_id": task["candidate"]["product_id"],
        "candidate": task["candidate"],
        "context": task["context"],
        "judgment": {"relevance_grade": 3, "model_id": model, "prompt_version": PROMPT_VERSION},
        "usage": {"prompt_tokens": 2, "completion_tokens": 1},
        "hard_requirement_violated": False,
        "hard_requirement_unknown": False,
        "authorized_evidence_sha256": _canonical_hash([]),
    }


def test_preserves_split_and_one_judge_per_group(tmp_path):
    source = tmp_path / "source"
    queue = source / "queue"
    queue.mkdir(parents=True)
    (source / "judged").mkdir()
    tasks = []
    snapshots = []
    for i in range(260):
        qid = f"q{i}"
        split = "test" if i < 52 else "validation" if i < 104 else "train"
        snapshots.append(
            {
                "query_id": qid,
                "query_group_id": qid,
                "split": split,
                "candidate_ids": ["a", "b"],
                "rankings": {k: ["a", "b"] for k in ("bm25", "vector", "rrf")},
            }
        )
        for pid in ("a", "b"):
            tasks.append(
                {
                    "task_id": f"{qid}:{pid}",
                    "query_id": qid,
                    "query_group_id": qid,
                    "context": {"query_id": qid, "query_group_id": qid},
                    "candidate": {"product_id": pid, "category": "cpu"},
                    "judge_input": {"specifications": {}, "requirements": []},
                }
            )
    write_jsonl(queue / "tasks.jsonl", tasks)
    (queue / "query-candidates.json").write_text(json.dumps(snapshots), encoding="utf-8")
    (queue / "manifest.json").write_text(
        json.dumps(
            {
                "seed": 20261108,
                "tasks_sha256": sha256_file(queue / "tasks.jsonl"),
                "snapshot_sha256": sha256_file(queue / "query-candidates.json"),
            }
        ),
        encoding="utf-8",
    )
    write_jsonl(
        source / "judged/judgments.checkpoint.jsonl", [result(t, "original") for t in tasks[:3]]
    )
    root = tmp_path / "mixed"
    prepared = prepare(source, root, "original", "alternative")
    assert len(prepared["excluded_partial_group_ids"]) == 1
    assert prepared["split_group_counts"]["test"] == 51
    alt_tasks = [json.loads(x) for x in (root / "alternative-tasks.jsonl").read_text().splitlines()]
    alt_checkpoint = tmp_path / "alternative.jsonl"
    write_jsonl(alt_checkpoint, [result(t, "alternative") for t in alt_tasks])
    alt_summary = tmp_path / "summary.json"
    alt_summary.write_text(
        json.dumps(
            {
                "training_dataset_ready": True,
                "unresolved_task_count": 0,
                "input_sha256": sha256_file(root / "alternative-tasks.jsonl"),
                "model_id": "alternative",
                "judgments_sha256": sha256_file(alt_checkpoint),
                "prompt_version": PROMPT_VERSION,
            }
        ),
        encoding="utf-8",
    )
    (root / "constraint-overrides.json").write_text(
        json.dumps(
            {
                "source_snapshot_sha256": sha256_file(root / "queue/query-candidates.json"),
                "overrides": {
                    "q0": [
                        {"field": "memory_type", "operator": "eq", "value": "ddr4", "hard": True}
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    manifest = json.loads((root / "queue/manifest.json").read_text(encoding="utf-8"))
    (root / "final-judgment-audit.json").write_text(
        json.dumps(
            {
                "complete": True,
                "frozen_queue_sha256": manifest["tasks_sha256"],
                "snapshot_sha256": manifest["snapshot_sha256"],
                "judge_assignments_sha256": manifest["judge_assignments_sha256"],
                "original_judgments_sha256": sha256_file(root / "original-judgments.jsonl"),
                "alternative_judgments_sha256": sha256_file(alt_checkpoint),
            }
        ),
        encoding="utf-8",
    )
    combined = combine(root, alt_summary, alt_checkpoint, tmp_path / "combined")
    assert combined["model_counts"]["original"] == 2
    assert combined["model_counts"]["alternative"] == len(alt_tasks)
    assert combined["query_count"] == 259
    assert combined["supplemental_grade_corrections"] == 2
