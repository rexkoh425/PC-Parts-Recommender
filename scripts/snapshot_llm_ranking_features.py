"""Freeze label-independent requirement-match features for an LLM ranking queue."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from training.llm_judge import _matches, _nested

from pc_build_recommender.evaluation.manifest import sha256_file


def snapshot(queue: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    rows = []
    for line in queue.read_text(encoding="utf-8").splitlines():
        task = json.loads(line)
        info = task["judge_input"]
        requirements = [r for r in info["requirements"] if r.get("hard")]
        outcomes = [
            _matches(_nested(info["specifications"], r["field"]), r["operator"], r.get("value"))
            for r in requirements
        ]
        count = len(outcomes)
        rows.append(
            {
                "task_id": task["task_id"],
                "query_id": task["query_id"],
                "product_id": task["candidate"]["product_id"],
                "specification_match": sum(x is True for x in outcomes) / count if count else 0.0,
                "unknown_fraction": sum(x is None for x in outcomes) / count if count else 0.0,
            }
        )
    output.mkdir(parents=True)
    data = output / "features.jsonl"
    data.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    report = {
        "schema_version": "pcbr.label-independent-constraint-features.v1",
        "queue_sha256": sha256_file(queue),
        "features_sha256": sha256_file(data),
        "pair_count": len(rows),
        "formula": (
            "fraction of explicit hard requirements verified satisfied; "
            "unknowns are not counted as satisfied"
        ),
        "evidence": (
            "computed only from frozen query requirements and catalogue fields; "
            "no labels or judge explanations"
        ),
    }
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--queue", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(snapshot(a.queue, a.output), indent=2))


if __name__ == "__main__":
    main()
