"""Audit frozen query-group splits and semantic similarity without relevance labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pc_build_recommender.evaluation.manifest import sha256_file
from pc_build_recommender.retrieval import SentenceTransformerEmbeddingEncoder


def audit(snapshot: Path, previous: Path, output: Path, threshold: float, device: str) -> dict:
    if output.exists():
        raise FileExistsError(output)
    current = json.loads(snapshot.read_text(encoding="utf-8"))
    old = json.loads(previous.read_text(encoding="utf-8"))["queries"]
    groups: dict[str, str] = {}
    for row in current:
        group, split = row["query_group_id"], row["split"]
        if groups.setdefault(group, split) != split:
            raise ValueError("query group appears in multiple splits")
    encoder = SentenceTransformerEmbeddingEncoder(
        "sentence-transformers/all-MiniLM-L6-v2", device=device, batch_size=64
    )
    vectors = encoder.encode([r["query_text"] for r in current + old])
    n = len(current)
    cross_split = []
    previous_matches = []
    max_cross = 0.0
    max_previous = 0.0
    for i, row in enumerate(current):
        for j in range(i + 1, n):
            other = current[j]
            if row["category"] != other["category"] or row["split"] == other["split"]:
                continue
            score = float(np.dot(vectors[i], vectors[j]))
            max_cross = max(max_cross, score)
            if score >= threshold:
                cross_split.append([row["query_id"], other["query_id"], score])
        for j, other in enumerate(old):
            if row["category"] != other["category"]:
                continue
            score = float(np.dot(vectors[i], vectors[n + j]))
            max_previous = max(max_previous, score)
            if score >= threshold:
                previous_matches.append([row["query_id"], other["query_id"], score])
    report = {
        "schema_version": "pcbr.query-split-audit.v1",
        "snapshot_sha256": sha256_file(snapshot),
        "previous_queries_sha256": sha256_file(previous),
        "query_count": n,
        "group_count": len(groups),
        "threshold": threshold,
        "max_cross_split_similarity": max_cross,
        "max_previous_similarity": max_previous,
        "cross_split_matches": cross_split,
        "previous_query_matches": previous_matches,
        "encoder_device": encoder.resolved_device,
        "limitation": (
            "embedding similarity is an automated screen and cannot prove semantic independence"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if cross_split or previous_matches:
        raise ValueError("similarity audit found overlapping intents")
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--snapshot", required=True, type=Path)
    p.add_argument(
        "--previous", type=Path, default=Path("evals/retrieval/real_catalog_silver_queries.v1.json")
    )
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--threshold", type=float, default=0.80)
    p.add_argument("--device", default="auto")
    a = p.parse_args()
    print(json.dumps(audit(a.snapshot, a.previous, a.output, a.threshold, a.device), indent=2))


if __name__ == "__main__":
    main()
