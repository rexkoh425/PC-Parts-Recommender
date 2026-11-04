"""Audit frozen tasks, judge assignments and completed weak judgments before training.

This checks data integrity only; it does not calculate model or holdout metrics.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from training.llm_judge import PROMPT_VERSION, RUBRIC, _canonical_hash, hard_requirement_outcome

from pc_build_recommender.evaluation.manifest import sha256_file


def audit(root: Path, output: Path, require_complete: bool) -> dict:
    if output.exists():
        raise FileExistsError(output)
    queue = root / "queue/tasks.jsonl"
    snapshot = root / "queue/query-candidates.json"
    manifest = json.loads((root / "queue/manifest.json").read_text(encoding="utf-8"))
    if (
        sha256_file(queue) != manifest["tasks_sha256"]
        or sha256_file(snapshot) != manifest["snapshot_sha256"]
    ):
        raise ValueError("frozen queue or split snapshot digest mismatch")
    assignment_path = root / "judge-assignments.json"
    if sha256_file(assignment_path) != manifest["judge_assignments_sha256"]:
        raise ValueError("judge assignments digest mismatch")
    assignments = json.loads(assignment_path.read_text(encoding="utf-8"))
    rows = [json.loads(x) for x in queue.read_text(encoding="utf-8").splitlines()]
    tasks = {t["task_id"]: t for t in rows}
    if len(tasks) != len(rows) or len(tasks) != manifest["task_count"]:
        raise ValueError("duplicate or missing task IDs")
    snapshots = json.loads(snapshot.read_text(encoding="utf-8"))
    by_query = {q["query_id"]: q for q in snapshots}
    groups = {}
    for q in snapshots:
        group = q["query_group_id"]
        value = (q["split"], q["judge_model"])
        if (
            groups.setdefault(group, value) != value
            or assignments[q["query_id"]] != q["judge_model"]
        ):
            raise ValueError("intent group leaks across split or judge")
    task_groups = Counter(t["query_id"] for t in rows)
    if set(task_groups) != set(by_query):
        raise ValueError("query set differs from frozen snapshot")
    for qid, count in task_groups.items():
        ids = {t["candidate"]["product_id"] for t in rows if t["query_id"] == qid}
        if count != len(ids) or ids != set(by_query[qid]["candidate_ids"]):
            raise ValueError("candidate pool differs from frozen snapshot")
    records = {}
    sources = [
        root / "original-judgments.jsonl",
        root / "alternative-judged/judgments.checkpoint.jsonl",
    ]
    for path in sources:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            task = tasks.get(row["task_id"])
            if task is None or row["task_id"] in records:
                raise ValueError("unknown or duplicate judgment")
            model = assignments[task["query_id"]]
            expected_hash = _canonical_hash(
                {
                    "task": task,
                    "model_id": model,
                    "prompt_version": PROMPT_VERSION,
                    "rubric_sha256": _canonical_hash(RUBRIC),
                }
            )
            violated, unknown, _ = hard_requirement_outcome(task)
            grade = row["judgment"]["relevance_grade"]
            if (
                row["task_hash"] != expected_hash
                or row["judgment"]["model_id"] != model
                or row["judgment"]["prompt_version"] != PROMPT_VERSION
                or row.get("hard_requirement_violated") != violated
                or row.get("hard_requirement_unknown") != unknown
                or row.get("authorized_evidence_sha256")
                != _canonical_hash(task["judge_input"].get("source_evidence", []))
                or (violated and grade != 0)
                or (unknown and not violated and grade > 2)
            ):
                raise ValueError("judgment provenance or deterministic check mismatch")
            records[row["task_id"]] = row
    missing = len(tasks) - len(records)
    if require_complete and missing:
        raise ValueError(f"{missing} required judgments are missing")
    split_audit = root / "query-split-audit.json"
    q_audit = json.loads(split_audit.read_text(encoding="utf-8"))
    if (
        q_audit["snapshot_sha256"] != sha256_file(snapshot)
        or q_audit["cross_split_matches"]
        or q_audit["previous_query_matches"]
    ):
        raise ValueError("frozen query split audit failed")
    feature_manifest = json.loads(
        (root / "constraint-features-v1/manifest.json").read_text(encoding="utf-8")
    )
    feature_file = root / "constraint-features-v1/features.jsonl"
    if feature_manifest["queue_sha256"] != sha256_file(queue) or feature_manifest[
        "features_sha256"
    ] != sha256_file(feature_file):
        raise ValueError("feature snapshot integrity failed")
    overlay = root / "constraint-overrides.json"
    if overlay.exists():
        data = json.loads(overlay.read_text(encoding="utf-8"))
        if data["source_snapshot_sha256"] != sha256_file(snapshot) or not set(
            data["overrides"]
        ).issubset(by_query):
            raise ValueError("constraint override is not bound to the frozen snapshot")
    completed_counts = Counter(tasks[k]["query_id"] for k in records)
    report = {
        "schema_version": "pcbr.mixed-llm-judgment-audit.v1",
        "complete": missing == 0,
        "task_count": len(tasks),
        "valid_judgments": len(records),
        "missing_judgments": missing,
        "query_count": len(snapshots),
        "query_group_count": len(groups),
        "complete_query_count": sum(completed_counts[q] == n for q, n in task_groups.items()),
        "split_group_counts": {
            s: sum(v[0] == s for v in groups.values()) for s in ("train", "validation", "test")
        },
        "judge_group_counts": dict(Counter(v[1] for v in groups.values())),
        "split_judge_group_counts": {
            f"{split}:{model}": sum(value == (split, model) for value in groups.values())
            for split, model in sorted(set(groups.values()))
        },
        "frozen_queue_sha256": sha256_file(queue),
        "snapshot_sha256": sha256_file(snapshot),
        "judge_assignments_sha256": sha256_file(assignment_path),
        "original_judgments_sha256": sha256_file(sources[0]) if sources[0].exists() else None,
        "alternative_judgments_sha256": sha256_file(sources[1]) if sources[1].exists() else None,
        "prompt_version": PROMPT_VERSION,
        "rubric_sha256": _canonical_hash(RUBRIC),
        "query_split_audit_sha256": sha256_file(split_audit),
        "feature_snapshot_sha256": sha256_file(feature_file),
        "constraint_overrides_sha256": sha256_file(overlay) if overlay.exists() else None,
        "runtime_note_sha256": sha256_file(root / "judging-runtime-note.json")
        if (root / "judging-runtime-note.json").exists()
        else None,
        "limitation": (
            "validates recorded hard checks and provenance, "
            "not every free-text rationale or human relevance"
        ),
    }
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--require-complete", action="store_true")
    a = p.parse_args()
    print(json.dumps(audit(a.root, a.output, a.require_complete), indent=2))


if __name__ == "__main__":
    main()
