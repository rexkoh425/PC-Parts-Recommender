"""Reuse judgments only when the blinded judge input and model/prompt match."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from training.llm_judge import (
    PROMPT_VERSION,
    RUBRIC,
    _blind,
    _canonical_hash,
    hard_requirement_outcome,
)

from pc_build_recommender.evaluation.manifest import sha256_file


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--queue", required=True, type=Path)
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--source-queue", type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--model", required=True)
    args = p.parse_args()
    tasks = [json.loads(line) for line in args.queue.read_text(encoding="utf-8").splitlines()]
    expected = {
        t["task_id"]: _canonical_hash(
            {
                "task": t,
                "model_id": args.model,
                "prompt_version": PROMPT_VERSION,
                "rubric_sha256": _canonical_hash(RUBRIC),
            }
        )
        for t in tasks
    }
    source = [json.loads(line) for line in args.source.read_text(encoding="utf-8").splitlines()]
    matched = {r["task_id"]: r for r in source if expected.get(r["task_id"]) == r["task_hash"]}
    rebound = 0
    if args.source_queue is not None:
        old_tasks = {
            t["task_id"]: t
            for t in map(json.loads, args.source_queue.read_text(encoding="utf-8").splitlines())
        }
        new_tasks = {t["task_id"]: t for t in tasks}
        for result in source:
            task_id = result["task_id"]
            old, new = old_tasks.get(task_id), new_tasks.get(task_id)
            if task_id in matched or old is None or new is None:
                continue
            prior_hash = _canonical_hash(
                {
                    "task": old,
                    "model_id": args.model,
                    "prompt_version": PROMPT_VERSION,
                    "rubric_sha256": _canonical_hash(RUBRIC),
                }
            )
            if (
                result["task_hash"] != prior_hash
                or result["judgment"].get("model_id") != args.model
                or result["judgment"].get("prompt_version") != PROMPT_VERSION
                or _canonical_hash(_blind(old["judge_input"]))
                != _canonical_hash(_blind(new["judge_input"]))
                or hard_requirement_outcome(old) != hard_requirement_outcome(new)
                or old["candidate"]["product_id"] != new["candidate"]["product_id"]
            ):
                continue
            updated = dict(result)
            updated.update(
                {
                    "task_hash": expected[task_id],
                    "candidate": new["candidate"],
                    "context": new["context"],
                    "query_group_id": new["query_group_id"],
                    "rebound_from_task_hash": prior_hash,
                }
            )
            matched[task_id] = updated
            rebound += 1
    args.output_dir.mkdir(parents=True, exist_ok=False)
    target = args.output_dir / "judgments.checkpoint.jsonl"
    target.write_text(
        "".join(
            json.dumps(matched[k], sort_keys=True, ensure_ascii=False) + "\n"
            for k in sorted(matched)
        ),
        encoding="utf-8",
    )
    report = {
        "source_sha256": sha256_file(args.source),
        "queue_sha256": sha256_file(args.queue),
        "matched_count": len(matched),
        "rebound_identical_blinded_inputs": rebound,
        "source_queue_sha256": sha256_file(args.source_queue) if args.source_queue else None,
        "target_sha256": sha256_file(target),
        "model_id": args.model,
        "prompt_version": PROMPT_VERSION,
    }
    (args.output_dir / "reused-checkpoint.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
