"""Attach frozen, label-independent specification features to completed labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pc_build_recommender.evaluation.manifest import sha256_file


def materialize(dataset: Path, source_summary: Path, feature_dir: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    summary = json.loads(source_summary.read_text(encoding="utf-8"))
    manifest = json.loads((feature_dir / "manifest.json").read_text(encoding="utf-8"))
    features_path = feature_dir / "features.jsonl"
    if (
        not summary.get("training_dataset_ready")
        or summary.get("unresolved_task_count") != 0
        or summary["ranking_dataset_sha256"] != sha256_file(dataset)
        or summary["input_sha256"] != manifest["queue_sha256"]
        or manifest["features_sha256"] != sha256_file(features_path)
    ):
        raise ValueError("judgment or feature snapshot integrity check failed")
    features = {
        x["task_id"]: x
        for x in map(json.loads, features_path.read_text(encoding="utf-8").splitlines())
    }
    if len(features) != manifest["pair_count"]:
        raise ValueError("duplicate or missing feature snapshot tasks")
    rows = [json.loads(line) for line in dataset.read_text(encoding="utf-8").splitlines()]
    used: set[str] = set()
    for row in rows:
        qid = row["context"]["query_id"]
        for candidate in row["candidates"]:
            task_id = f"{qid}:{candidate['product_id']}"
            feature = features[task_id]
            if feature["query_id"] != qid or feature["product_id"] != candidate["product_id"]:
                raise ValueError("feature candidate does not match judgment candidate")
            candidate.setdefault("signals", {})["specification_match"] = feature[
                "specification_match"
            ]
            candidate["signals"]["specification_unknown_fraction"] = feature["unknown_fraction"]
            used.add(task_id)
    if used != set(features):
        raise ValueError("feature snapshot and judgment pairs differ")
    output.mkdir(parents=True)
    enriched = output / "ranking_dataset.llm_weak.jsonl"
    enriched.write_text(
        "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    summary["source_judgment_summary_sha256"] = sha256_file(source_summary)
    summary["source_ranking_dataset_sha256"] = sha256_file(dataset)
    summary["feature_manifest_sha256"] = sha256_file(feature_dir / "manifest.json")
    summary["feature_snapshot_sha256"] = sha256_file(features_path)
    summary["ranking_dataset_sha256"] = sha256_file(enriched)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True, type=Path)
    p.add_argument("--summary", required=True, type=Path)
    p.add_argument("--features", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    print(json.dumps(materialize(a.dataset, a.summary, a.features, a.output), indent=2))


if __name__ == "__main__":
    main()
