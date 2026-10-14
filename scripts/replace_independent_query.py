"""Replace one flagged query with an unused screened candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from scripts.freeze_independent_query_cohort import classify


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", required=True, type=Path)
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--query-id", required=True)
    parser.add_argument("--exclude-id", action="append", default=[])
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    data = json.loads(args.queries.read_text(encoding="utf-8"))
    candidates = json.loads(args.candidates.read_text(encoding="utf-8"))["queries"]
    old = next(row for row in data["queries"] if row["query_id"] == args.query_id)
    used = {row["query_id"] for row in data["queries"]}
    replacement = next(
        row
        for row in candidates
        if row["query_id"] not in used
        and row["query_id"] not in set(args.exclude_id)
        and not row["query_id"].startswith("fresh-v1-")
        and classify(row) == old["evaluation_category"]
    )
    replacement = dict(replacement)
    replacement["evaluation_category"] = old["evaluation_category"]
    data["queries"] = [
        replacement if row["query_id"] == args.query_id else row for row in data["queries"]
    ]
    data["schema_version"] = "pcbr.independent-heldout-query-cohort.v3"
    data["replaced_query_ids"] = sorted(set(data.get("replaced_query_ids", [])) | {args.query_id})
    data["replacement_source_sha256"] = hashlib.sha256(args.candidates.read_bytes()).hexdigest()
    args.output.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"replaced": args.query_id, "with": replacement["query_id"]}))


if __name__ == "__main__":
    main()
