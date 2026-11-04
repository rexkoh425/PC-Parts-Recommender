"""Combine complete, query-disjoint weak judgments while preserving judge provenance."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from training.llm_judge import PROMPT_VERSION, RUBRIC, _canonical_hash, hard_requirement_outcome

from pc_build_recommender.evaluation.manifest import sha256_file


def combine(
    root: Path, alternative_summary: Path, alternative_checkpoint: Path, output: Path
) -> dict:
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads((root / "queue/manifest.json").read_text(encoding="utf-8"))
    summary = json.loads(alternative_summary.read_text(encoding="utf-8"))
    audit_path = root / "final-judgment-audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if (
        not summary.get("training_dataset_ready")
        or summary.get("unresolved_task_count") != 0
        or summary["input_sha256"] != manifest["alternative_tasks_sha256"]
        or summary["model_id"] != manifest["alternative_model"]
        or summary["judgments_sha256"] != sha256_file(alternative_checkpoint)
        or sha256_file(root / "queue/tasks.jsonl") != manifest["tasks_sha256"]
        or sha256_file(root / "judge-assignments.json") != manifest["judge_assignments_sha256"]
        or sha256_file(root / "original-judgments.jsonl") != manifest["original_judgments_sha256"]
        or not audit.get("complete")
        or audit["frozen_queue_sha256"] != manifest["tasks_sha256"]
        or audit["snapshot_sha256"] != manifest["snapshot_sha256"]
        or audit["judge_assignments_sha256"] != manifest["judge_assignments_sha256"]
        or audit["original_judgments_sha256"] != sha256_file(root / "original-judgments.jsonl")
        or audit["alternative_judgments_sha256"] != sha256_file(alternative_checkpoint)
    ):
        raise ValueError("source judgments are incomplete or fail integrity checks")
    tasks = {
        t["task_id"]: t
        for t in map(
            json.loads, (root / "queue/tasks.jsonl").read_text(encoding="utf-8").splitlines()
        )
    }
    assignments = json.loads((root / "judge-assignments.json").read_text(encoding="utf-8"))
    override_path = root / "constraint-overrides.json"
    overrides = {}
    if override_path.exists():
        overlay = json.loads(override_path.read_text(encoding="utf-8"))
        if overlay["source_snapshot_sha256"] != sha256_file(root / "queue/query-candidates.json"):
            raise ValueError("constraint overrides do not match the frozen query snapshot")
        overrides = overlay["overrides"]
        if not set(overrides).issubset(assignments):
            raise ValueError("constraint override references an unknown query")
    records = {}
    for path in (root / "original-judgments.jsonl", alternative_checkpoint):
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row["task_id"] in records:
                raise ValueError("duplicate judgment task")
            records[row["task_id"]] = row
    if set(records) != set(tasks):
        raise ValueError("judgments do not cover the frozen candidate tasks")
    groups = defaultdict(list)
    group_models = {}
    effective_grades = {}
    override_audit = []
    for task_id, task in tasks.items():
        row = records[task_id]
        expected_hash = _canonical_hash(
            {
                "task": task,
                "model_id": assignments[task["query_id"]],
                "prompt_version": PROMPT_VERSION,
                "rubric_sha256": _canonical_hash(RUBRIC),
            }
        )
        if (
            row["query_id"] != task["query_id"]
            or row["product_id"] != task["candidate"]["product_id"]
            or row["judgment"]["model_id"] != assignments[task["query_id"]]
            or row["judgment"].get("prompt_version") != PROMPT_VERSION
            or row["task_hash"] != expected_hash
        ):
            raise ValueError("judgment identity or per-query model assignment differs")
        expected_violated, expected_unknown, _ = hard_requirement_outcome(task)
        if (
            row.get("hard_requirement_violated") != expected_violated
            or row.get("hard_requirement_unknown") != expected_unknown
            or row.get("authorized_evidence_sha256")
            != _canonical_hash(task["judge_input"].get("source_evidence", []))
        ):
            raise ValueError("hard-check or source-evidence provenance does not match frozen task")
        model = assignments[task["query_id"]]
        group = task["query_group_id"]
        if group_models.setdefault(group, model) != model:
            raise ValueError("one intent group was assigned multiple judge models")
        grade = row["judgment"]["relevance_grade"]
        if (row.get("hard_requirement_violated") and grade != 0) or (
            row.get("hard_requirement_unknown")
            and not row.get("hard_requirement_violated")
            and grade > 2
        ):
            raise ValueError("a final judgment contradicts a deterministic hard requirement")
        if task["query_id"] in overrides:
            supplemental_task = {
                **task,
                "judge_input": {**task["judge_input"], "requirements": overrides[task["query_id"]]},
            }
            violated, unknown, details = hard_requirement_outcome(supplemental_task)
            adjusted = 0 if violated else min(grade, 2) if unknown else grade
            override_audit.append(
                {
                    "task_id": task_id,
                    "original_grade": grade,
                    "adjusted_grade": adjusted,
                    "hard_violation": violated,
                    "hard_unknown": unknown,
                    "details": details,
                }
            )
            grade = adjusted
        effective_grades[task_id] = grade
        candidate = dict(task["candidate"])
        candidate["relevance_grade"] = grade
        groups[task["query_id"]].append(candidate)
    output.mkdir(parents=True)
    dataset = output / "ranking_dataset.llm_weak.jsonl"
    contexts = {t["query_id"]: t["context"] for t in tasks.values()}
    dataset.write_text(
        "".join(
            json.dumps(
                {
                    "context": contexts[q],
                    "candidates": sorted(groups[q], key=lambda x: x["product_id"]),
                },
                sort_keys=True,
                ensure_ascii=False,
            )
            + "\n"
            for q in sorted(groups)
        ),
        encoding="utf-8",
    )
    grades = Counter(effective_grades.values())
    counts = Counter(r["judgment"]["model_id"] for r in records.values())
    by_model = defaultdict(Counter)
    by_category = defaultdict(Counter)
    confidence = defaultdict(Counter)
    for task_id, row in records.items():
        model = row["judgment"]["model_id"]
        grade = effective_grades[task_id]
        by_model[model][grade] += 1
        by_category[tasks[task_id]["candidate"]["category"]][grade] += 1
        value = row["judgment"].get("confidence")
        if value is not None:
            confidence[model][
                "below_0.5" if value < 0.5 else "0.5_to_below_0.8" if value < 0.8 else "0.8_to_1.0"
            ] += 1
    usage = {
        "prompt_tokens": sum(r["usage"]["prompt_tokens"] for r in records.values()),
        "completion_tokens": sum(r["usage"]["completion_tokens"] for r in records.values()),
    }
    result = {
        "schema_version": "pcbr.mixed-llm-weak-judgments.v1",
        "label_source": "llm_weak",
        "eligible_for_production_or_human_quality_claims": False,
        "training_dataset_ready": True,
        "unresolved_task_count": 0,
        "input_sha256": manifest["tasks_sha256"],
        "ranking_dataset_sha256": sha256_file(dataset),
        "source_manifest_sha256": sha256_file(root / "queue/manifest.json"),
        "final_judgment_audit_sha256": sha256_file(audit_path),
        "alternative_summary_sha256": sha256_file(alternative_summary),
        "judge_assignments_sha256": sha256_file(root / "judge-assignments.json"),
        "model_id": "mixed-by-query",
        "primary_judge_model": manifest["alternative_model"],
        "model_counts": dict(counts),
        "prompt_version": summary["prompt_version"],
        "task_count": len(records),
        "query_count": len(groups),
        "grade_counts": {str(g): grades[g] for g in range(5)},
        "constraint_overrides_sha256": sha256_file(override_path)
        if override_path.exists()
        else None,
        "supplemental_constraint_task_count": len(override_audit),
        "supplemental_grade_corrections": sum(
            r["original_grade"] != r["adjusted_grade"] for r in override_audit
        ),
        "supplemental_hard_violation_count": sum(r["hard_violation"] for r in override_audit),
        "supplemental_hard_unknown_count": sum(r["hard_unknown"] for r in override_audit),
        "supplemental_high_grade_contradictions": sum(
            r["hard_violation"] and r["original_grade"] >= 3 for r in override_audit
        ),
        "grade_counts_by_model": {k: {str(g): v[g] for g in range(5)} for k, v in by_model.items()},
        "grade_counts_by_category": {
            k: {str(g): v[g] for g in range(5)} for k, v in by_category.items()
        },
        "confidence_buckets_by_model": {k: dict(v) for k, v in confidence.items()},
        "confidence_recorded_count": sum(sum(v.values()) for v in confidence.values()),
        "missing_evidence_count": sum(
            bool(r["judgment"].get("missing_or_unsupported_information")) for r in records.values()
        ),
        "hard_violation_count": sum(
            bool(r.get("hard_requirement_violated")) for r in records.values()
        ),
        "hard_unknown_count": sum(
            bool(r.get("hard_requirement_unknown")) for r in records.values()
        ),
        "raw_high_grade_hard_contradiction_count": sum(
            bool(r.get("hard_requirement_violated")) and r.get("judge_raw_relevance_grade", 0) >= 3
            for r in records.values()
        ),
        "final_hard_contradiction_count": 0,
        **usage,
        "estimated_cost_usd": None,
        "limitation": (
            "each query uses one judge; model assignment follows route availability "
            "and is not randomized"
        ),
    }
    if override_audit:
        audit_path = output / "constraint-override-audit.jsonl"
        audit_path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in override_audit),
            encoding="utf-8",
        )
        result["constraint_override_audit_sha256"] = sha256_file(audit_path)
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--alternative-summary", required=True, type=Path)
    p.add_argument("--alternative-checkpoint", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    print(
        json.dumps(
            combine(a.root, a.alternative_summary, a.alternative_checkpoint, a.output), indent=2
        )
    )


if __name__ == "__main__":
    main()
