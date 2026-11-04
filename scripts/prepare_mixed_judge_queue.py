"""Prepare a diagnostic fallback while keeping each frozen intent with one judge.

Only completely judged groups are retained from the original model. Groups with
partially judged candidates are excluded, and existing split assignments remain
unchanged. Selection uses completion state only, never grades or explanations.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from training.llm_judge import PROMPT_VERSION, RUBRIC, _canonical_hash

from pc_build_recommender.evaluation.manifest import sha256_file


def prepare(source: Path, output: Path, original_model: str, alternative_model: str) -> dict:
    if output.exists():
        raise FileExistsError(output)
    queue = source / "queue/tasks.jsonl"
    snapshot_file = source / "queue/query-candidates.json"
    manifest = json.loads((source / "queue/manifest.json").read_text(encoding="utf-8"))
    if (
        sha256_file(queue) != manifest["tasks_sha256"]
        or sha256_file(snapshot_file) != manifest["snapshot_sha256"]
    ):
        raise ValueError("source queue or split snapshot changed")
    tasks = [json.loads(line) for line in queue.read_text(encoding="utf-8").splitlines()]
    task_by_id = {t["task_id"]: t for t in tasks}
    checkpoint = source / "judged/judgments.checkpoint.jsonl"
    records = {}
    for line in checkpoint.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        task = task_by_id.get(row["task_id"])
        if task is None:
            continue
        expected_hash = _canonical_hash(
            {
                "task": task,
                "model_id": original_model,
                "prompt_version": PROMPT_VERSION,
                "rubric_sha256": _canonical_hash(RUBRIC),
            }
        )
        if (
            row["task_hash"] != expected_hash
            or row["judgment"].get("model_id") != original_model
            or row["judgment"].get("prompt_version") != PROMPT_VERSION
        ):
            raise ValueError("source checkpoint task, model or prompt mismatch")
        records[row["task_id"]] = row
    expected = Counter(t["query_group_id"] for t in tasks)
    completed = Counter(task_by_id[task_id]["query_group_id"] for task_id in records)
    partial = {group for group, count in completed.items() if count != expected[group]}
    original_groups = {group for group, count in completed.items() if count == expected[group]}
    kept = [t for t in tasks if t["query_group_id"] not in partial]
    snapshots = [
        q
        for q in json.loads(snapshot_file.read_text(encoding="utf-8"))
        if q["query_group_id"] not in partial
    ]
    assignment = {
        q["query_id"]: original_model
        if q["query_group_id"] in original_groups
        else alternative_model
        for q in snapshots
    }
    for row in snapshots:
        row["judge_model"] = assignment[row["query_id"]]
    groups_by_split = {
        name: len({q["query_group_id"] for q in snapshots if q["split"] == name})
        for name in ("train", "validation", "test")
    }
    if groups_by_split["test"] < 50:
        raise ValueError("fewer than 50 test intent groups remain")
    output.mkdir(parents=True)
    output_queue = output / "queue"
    output_queue.mkdir()
    full_queue = output_queue / "tasks.jsonl"
    full_queue.write_text(
        "".join(json.dumps(t, sort_keys=True, ensure_ascii=False) + "\n" for t in kept),
        encoding="utf-8",
    )
    full_snapshot = output_queue / "query-candidates.json"
    full_snapshot.write_text(
        json.dumps(snapshots, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    alternative = [t for t in kept if t["query_group_id"] not in original_groups]
    alternative_path = output / "alternative-tasks.jsonl"
    alternative_path.write_text(
        "".join(json.dumps(t, sort_keys=True, ensure_ascii=False) + "\n" for t in alternative),
        encoding="utf-8",
    )
    prior_path = output / "original-judgments.jsonl"
    prior_ids = {t["task_id"] for t in kept if t["query_group_id"] in original_groups}
    prior_path.write_text(
        "".join(
            json.dumps(records[k], sort_keys=True, ensure_ascii=False) + "\n"
            for k in sorted(prior_ids)
        ),
        encoding="utf-8",
    )
    assignment_path = output / "judge-assignments.json"
    assignment_path.write_text(json.dumps(assignment, indent=2) + "\n", encoding="utf-8")
    result = {
        "schema_version": "pcbr.mixed-judge-queue.v1",
        "seed": manifest["seed"],
        "source_queue_sha256": sha256_file(queue),
        "source_checkpoint_sha256": sha256_file(checkpoint),
        "tasks_sha256": sha256_file(full_queue),
        "snapshot_sha256": sha256_file(full_snapshot),
        "alternative_tasks_sha256": sha256_file(alternative_path),
        "original_judgments_sha256": sha256_file(prior_path),
        "judge_assignments_sha256": sha256_file(assignment_path),
        "query_count": len(snapshots),
        "query_group_count": len(expected) - len(partial),
        "task_count": len(kept),
        "original_model": original_model,
        "alternative_model": alternative_model,
        "original_model_group_count": len(original_groups),
        "excluded_partial_group_ids": sorted(partial),
        "excluded_partial_judgment_count": sum(completed[g] for g in partial),
        "split_group_counts": groups_by_split,
        "purpose": "LLM-only diagnostic with frozen group split and per-query judge provenance",
    }
    (output_queue / "manifest.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--original-model", default="gpt-6-luna")
    p.add_argument("--alternative-model", default="gpt-5.6-luna")
    a = p.parse_args()
    print(json.dumps(prepare(a.source, a.output, a.original_model, a.alternative_model), indent=2))


if __name__ == "__main__":
    main()
