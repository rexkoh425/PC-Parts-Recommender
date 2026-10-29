"""Audit a frozen pilot against all authored requirements without new model calls."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from training.llm_judge import hard_requirement_outcome

from pc_build_recommender.evaluation.manifest import sha256_file


def audit(queue: Path, checkpoint: Path, query_file: Path, output: Path) -> dict:
    queries = {
        x["query_id"]: x for x in json.loads(query_file.read_text(encoding="utf-8"))["queries"]
    }
    tasks = {
        x["task_id"]: x for x in map(json.loads, queue.read_text(encoding="utf-8").splitlines())
    }
    results = {
        x["task_id"]: x
        for x in map(json.loads, checkpoint.read_text(encoding="utf-8").splitlines())
    }
    if len(results) != len(tasks) or set(results) != set(tasks):
        raise ValueError("pilot must finish all frozen tasks before validation")
    rows = []
    for task_id, result in sorted(results.items()):
        task = tasks[task_id]
        task["judge_input"]["requirements"] = [
            dict(x, hard=True) for x in queries[task["query_id"]]["must"]
        ]
        violated, unknown, details = hard_requirement_outcome(task)
        before = result["judgment"]["relevance_grade"]
        after = 0 if violated else min(before, 2) if unknown else before
        rows.append(
            {
                "task_id": task_id,
                "query_id": task["query_id"],
                "category": task["candidate"]["category"],
                "original_grade": before,
                "audited_grade": after,
                "hard_violation": violated,
                "hard_unknown": unknown,
                "raw_grade": result.get("judge_raw_relevance_grade"),
                "details": details,
                "confidence": result["judgment"]["confidence"],
                "missing_evidence": bool(result["judgment"]["missing_or_unsupported_information"]),
            }
        )
    output.mkdir(parents=True, exist_ok=False)
    path = output / "audited-grades.jsonl"
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    grades = Counter(r["audited_grade"] for r in rows)
    categories = Counter(r["category"] for r in rows)
    summary = {
        "schema_version": "pcbr.llm-pilot-audit.v1",
        "label_source": "llm_weak",
        "eligible_for_promotion": False,
        "queue_sha256": sha256_file(queue),
        "checkpoint_sha256": sha256_file(checkpoint),
        "query_file_sha256": sha256_file(query_file),
        "audited_grades_sha256": sha256_file(path),
        "task_count": len(rows),
        "query_count": len({r["query_id"] for r in rows}),
        "categories": dict(categories),
        "audited_grade_counts": {str(g): grades[g] for g in range(5)},
        "grade_corrections": sum(r["original_grade"] != r["audited_grade"] for r in rows),
        "severe_raw_hard_contradictions": sum(
            r["hard_violation"] and r["raw_grade"] is not None and r["raw_grade"] >= 3 for r in rows
        ),
        "unresolved_hard_contradictions": sum(
            r["hard_violation"] and r["audited_grade"] != 0 for r in rows
        ),
        "unknown_hard_requirements": sum(r["hard_unknown"] for r in rows),
        "missing_evidence_count": sum(r["missing_evidence"] for r in rows),
        "low_confidence_count": sum(r["confidence"] < 0.5 for r in rows),
        "limitations": (
            "authored technical pilot; an automated hard-constraint audit does not establish "
            "factual accuracy of every free-text rationale"
        ),
    }
    (output / "audit-summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--queries", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.queue, args.checkpoint, args.queries, args.output), indent=2))


if __name__ == "__main__":
    main()
