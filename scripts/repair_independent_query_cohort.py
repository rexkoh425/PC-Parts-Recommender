"""Replace flagged independent-cohort queries with unused screened candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from scripts.freeze_independent_query_cohort import QUOTAS, classify


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", required=True, type=Path)
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    current = json.loads(args.queries.read_text(encoding="utf-8"))
    candidates = json.loads(args.candidates.read_text(encoding="utf-8"))["queries"]
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    flagged = {row[0] for row in audit["cross_split_matches"]}
    flagged.update(row[0] for row in audit["previous_query_matches"])
    kept = [row for row in current["queries"] if row["query_id"] not in flagged]
    counts = {name: 0 for name in QUOTAS}
    for row in kept:
        counts[row["evaluation_category"]] += 1
    used = {row["query_id"] for row in kept}
    pool = [
        row
        for row in candidates
        if row["query_id"] not in used
        and row["query_id"] not in flagged
        and not row["query_id"].startswith("fresh-v1-")
    ]
    for row in pool:
        bucket = classify(row)
        if counts[bucket] >= QUOTAS[bucket]:
            continue
        copy = dict(row)
        copy["evaluation_category"] = bucket
        kept.append(copy)
        used.add(copy["query_id"])
        counts[bucket] += 1
        if len(kept) == sum(QUOTAS.values()):
            break
    if len(kept) < sum(QUOTAS.values()):
        for row in pool:
            if row["query_id"] in used:
                continue
            for bucket, quota in QUOTAS.items():
                if counts[bucket] < quota:
                    copy = dict(row)
                    copy["evaluation_category"] = bucket
                    kept.append(copy)
                    used.add(copy["query_id"])
                    counts[bucket] += 1
                    break
            if len(kept) == sum(QUOTAS.values()):
                break
    if len(kept) != sum(QUOTAS.values()) or counts != QUOTAS:
        raise ValueError({"counts": counts, "flagged": len(flagged), "kept": len(kept)})
    payload = {
        "schema_version": "pcbr.independent-heldout-query-cohort.v2",
        "seed": 20261110,
        "query_count": len(kept),
        "quotas": QUOTAS,
        "queries": kept,
        "replaced_query_ids": sorted(flagged),
        "source_sha256": hashlib.sha256(args.candidates.read_bytes()).hexdigest(),
        "prior_manifest_sha256": hashlib.sha256(args.queries.read_bytes()).hexdigest(),
        "audit_sha256": hashlib.sha256(args.audit.read_bytes()).hexdigest(),
        "limitation": "synthetic model-authored queries; independence is subject to the strict automated audit",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"query_count": len(kept), "counts": counts, "replaced": len(flagged)}, indent=2))


if __name__ == "__main__":
    main()
