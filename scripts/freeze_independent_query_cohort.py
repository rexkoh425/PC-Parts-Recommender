"""Freeze a quota-controlled independent query cohort from screened proposals."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


QUOTAS = {
    "gaming_gpu": 35,
    "cpu_productivity": 30,
    "budget": 30,
    "compatibility_upgrade": 35,
    "specific_specifications": 25,
    "ambiguous_natural_language": 25,
    "complex_multi_constraint": 20,
}


def classify(row: dict) -> str:
    text = row["query_text"].casefold()
    must_count = len(row.get("must", []))
    if must_count >= 3:
        return "complex_multi_constraint"
    if any(token in text for token in ("gaming", "fps", "1440p", "4k", "graphics")):
        return "gaming_gpu"
    if any(token in text for token in ("compile", "render", "workstation", "development", "productivity")):
        return "cpu_productivity"
    if any(token in text for token in ("budget", "affordable", "value", "cheap")):
        return "budget"
    if any(token in text for token in ("upgrade", "compatible", "compatibility", "existing", "fit", "socket")):
        return "compatibility_upgrade"
    if any(token in text for token in ("quiet", "suitable", "modern", "prefer", "need")):
        return "ambiguous_natural_language"
    return "specific_specifications"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    source = json.loads(args.input.read_text(encoding="utf-8"))
    candidates = [row for row in source["queries"] if row["query_id"].startswith("diverse-v1-")]
    buckets = {name: [] for name in QUOTAS}
    for row in candidates:
        buckets[classify(row)].append(row)
    selected = []
    used = set()
    deficits = {}
    for name, quota in QUOTAS.items():
        rows = [row for row in buckets[name] if id(row) not in used][:quota]
        deficits[name] = quota - len(rows)
        for row in rows:
            used.add(id(row))
            copy = dict(row)
            copy["evaluation_category"] = name
            selected.append(copy)
    fallback = [row for row in candidates if id(row) not in used]
    for name, deficit in deficits.items():
        for row in fallback[:deficit]:
            used.add(id(row))
            copy = dict(row)
            copy["evaluation_category"] = name
            selected.append(copy)
        fallback = fallback[deficit:]
    if len(selected) != sum(QUOTAS.values()):
        raise ValueError("quota selection did not produce exactly 200 rows")
    payload = {
        "schema_version": "pcbr.independent-heldout-query-cohort.v1",
        "seed": 20261110,
        "query_count": len(selected),
        "quotas": QUOTAS,
        "queries": selected,
        "source_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "classification": "deterministic keyword and must-count buckets; not human intent annotation",
        "limitation": "synthetic model-authored queries with automated similarity screening; manual semantic review remains outstanding",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"query_count": len(selected), "category_counts": {name: quota for name, quota in QUOTAS.items()}}, indent=2))


if __name__ == "__main__":
    main()
