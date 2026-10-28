"""Freeze an LLM-judge pilot queue from authored queries and the BuildCores catalogue.

The supplied query file is a pilot, previously used in silver diagnostics. It must
not be reused as an independent final test set. No silver labels are loaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np

from pc_build_recommender.evaluation.manifest import sha256_file
from pc_build_recommender.evaluation.retrieval_silver_pilot import _load_embedding_artifact
from pc_build_recommender.evaluation.splits import deterministic_group_split
from pc_build_recommender.retrieval import (
    BM25ProductIndex,
    ProductDocument,
    SentenceTransformerEmbeddingEncoder,
    reciprocal_rank_fusion,
)
from pc_build_recommender.retrieval.embedding_index import (
    build_product_embedding_text,
    load_normalized_product_jsonl,
)

CATALOG = Path(
    "data/processed/buildcores_open_db/f3ee75dd07ffdd7725da7b056229e0df12838c571b2372bd59563f3a79fd383f/full/records.jsonl"
)
QUERIES = Path("evals/retrieval/real_catalog_silver_queries.v1.json")
EMBEDDINGS = Path("artifacts/retrieval/buildcores-full-embeddings-pinned")


def explicit_unknown_price_requirement(query_text: str) -> dict | None:
    """Represent an explicit currency limit as unknown when no live price exists."""
    match = re.search(
        r"\b(?:under|below|less than|up to|at most|max(?:imum)?)\s*"
        r"(?:s\$|\$|sgd\s*)\s*([\d,]+(?:\.\d+)?)",
        query_text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    return {
        "name": "explicit price limit; current price and currency unverified",
        "field": "known_price",
        "operator": "le",
        "value": float(match.group(1).replace(",", "")),
        "hard": True,
    }


def build_queue(
    catalog: Path, queries_path: Path, embeddings: Path, output: Path, seed: int, device: str
) -> dict:
    if output.exists():
        raise FileExistsError(output)
    queries = json.loads(queries_path.read_text(encoding="utf-8"))["queries"]
    ids = [q["query_id"] for q in queries]
    if len(ids) != len(set(ids)) or len(
        {q["query_text"].casefold().strip() for q in queries}
    ) != len(ids):
        raise ValueError("query IDs and normalized query texts must be unique")
    records, _ = load_normalized_product_jsonl(catalog)
    by_id = {r["product_id"]: r for r in records}
    docs = [
        ProductDocument(
            product_id=r["product_id"],
            category=r["category"],
            text=build_product_embedding_text(r),
            brand=r.get("brand"),
            attributes={**r.get("common_attributes", {}), **r.get("category_attributes", {})},
        )
        for r in records
    ]
    index = BM25ProductIndex(docs)
    matrix, id_rows, manifest = _load_embedding_artifact(embeddings)
    positions = {str(r["product_id"]): int(r["row_index"]) for r in id_rows}
    if set(positions) != set(by_id):
        raise ValueError("catalogue and embedding product IDs differ")
    encoder = SentenceTransformerEmbeddingEncoder(
        str(manifest["encoder"]["model_name"]), device=device, batch_size=64
    )
    vectors = encoder.encode([q["query_text"] for q in queries])
    split = deterministic_group_split(
        [q.get("query_group_id", q["query_id"]) for q in queries],
        weights={"train": 0.6, "validation": 0.2, "test": 0.2},
        seed=seed,
    )
    tasks = []
    snapshot = []
    for i, q in enumerate(queries):
        qid, category = q["query_id"], q["category"]
        group_id = q.get("query_group_id", qid)
        universe = sorted(r["product_id"] for r in records if r["category"] == category)
        lexical = index.search(q["query_text"], category=category, top_k=len(universe))
        bm = {h.product_id: h.score for h in lexical}
        vec = {pid: round(float(np.dot(matrix[positions[pid]], vectors[i])), 8) for pid in universe}
        vec_order = sorted(universe, key=lambda pid: (-vec[pid], pid))
        fused = reciprocal_rank_fusion({"bm25": lexical[:50], "vector": vec_order[:50]}, k=60)
        rrf = {hit.product_id: hit.score for hit in fused}
        chosen: list[str] = []
        for ranking in ([h.product_id for h in lexical], vec_order, [h.product_id for h in fused]):
            added = 0
            for pid in ranking:
                if pid not in chosen:
                    chosen.append(pid)
                    added += 1
                if added == 8:
                    break
        rng = random.Random(int(hashlib.sha256(f"{seed}:{qid}".encode()).hexdigest(), 16))
        chosen.extend(rng.sample([pid for pid in universe if pid not in chosen], 4))
        requirements = [
            dict(c, hard=True)
            for c in q.get("must", [])
            if c["operator"]
            in {"eq", "ge", "le", "gt", "lt", "one_of", "contains", "contains_all", "not_null"}
        ]
        price_requirement = explicit_unknown_price_requirement(q["query_text"])
        if price_requirement is not None:
            requirements.append(price_requirement)
        feature_requirements = {
            c["field"].split(".")[-1]: c["value"]
            for c in requirements
            if c["operator"] == "eq" and isinstance(c["value"], (str, int, float, bool))
        }
        for c in requirements:
            if c["operator"] == "ge" and c["field"] == "category_attributes.vram_gb":
                feature_requirements["minimum_gpu_vram_gb"] = c["value"]
            if (
                c["operator"] == "ge"
                and c["field"] == "category_attributes.capacity_gb"
                and category == "memory"
            ):
                feature_requirements["minimum_memory_gb"] = c["value"]
        context = {
            "query_id": qid,
            "query_group_id": group_id,
            "query_text": q["query_text"],
            "requirements": feature_requirements,
        }
        for pid in chosen:
            r = by_id[pid]
            attrs = {**r.get("common_attributes", {}), **r.get("category_attributes", {})}
            evidence = [
                {
                    "source_name": p.get("source_name"),
                    "source_url": p.get("source_url"),
                    "licence_or_access_note": p.get("licence_or_access_note"),
                }
                for p in r.get("provenance", [])
            ]
            candidate = {
                "product_id": pid,
                "category": category,
                "brand": r.get("brand"),
                "retrieval_scores": {
                    "bm25_score": bm[pid],
                    "vector_similarity": vec[pid],
                    "rrf_score": rrf.get(pid, 0.0),
                },
                "attributes": attrs,
            }
            judge_input = {
                "query_text": q["query_text"],
                "category": category,
                "requirements": requirements,
                "product": {"name": r.get("canonical_name"), "category": category},
                "specifications": {
                    "common_attributes": r.get("common_attributes", {}),
                    "category_attributes": r.get("category_attributes", {}),
                },
                "price_sgd": None,
                "stock_status": "unknown",
                "source_evidence": evidence,
            }
            tasks.append(
                {
                    "task_id": f"{qid}:{pid}",
                    "query_id": qid,
                    "query_group_id": group_id,
                    "context": context,
                    "candidate": candidate,
                    "judge_input": judge_input,
                }
            )
        snapshot.append(
            {
                "query_id": qid,
                "query_group_id": group_id,
                "query_text": q["query_text"],
                "category": category,
                "split": split.split_for(group_id),
                "candidate_ids": sorted(chosen),
                "rankings": {
                    "bm25": [h.product_id for h in lexical if h.product_id in chosen],
                    "vector": [pid for pid in vec_order if pid in chosen],
                    "rrf": [h.product_id for h in fused if h.product_id in chosen]
                    + sorted(pid for pid in chosen if pid not in rrf),
                },
            }
        )
    tasks.sort(key=lambda t: t["task_id"])
    output.mkdir(parents=True)
    queue = output / "tasks.jsonl"
    queue.write_text(
        "".join(json.dumps(t, sort_keys=True, ensure_ascii=False) + "\n" for t in tasks),
        encoding="utf-8",
    )
    snapshot_path = output / "query-candidates.json"
    snapshot_path.write_text(
        json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    pilot = queries_path.resolve() == QUERIES.resolve()
    report = {
        "schema_version": "pcbr.llm-judge-pilot-queue.v1",
        "seed": seed,
        "purpose": (
            "pilot only; query intents previously used in silver diagnostics; "
            "no independent holdout"
            if pilot
            else "synthetic authored diagnostic; semantic intent independence not human verified"
        ),
        "label_source": "unjudged",
        "catalogue_sha256": sha256_file(catalog),
        "query_file_sha256": sha256_file(queries_path),
        "embedding_manifest_sha256": sha256_file(embeddings / "manifest.json"),
        "tasks_sha256": sha256_file(queue),
        "snapshot_sha256": sha256_file(snapshot_path),
        "query_count": len(queries),
        "query_group_count": len({q.get("query_group_id", q["query_id"]) for q in queries}),
        "task_count": len(tasks),
        "categories": dict(Counter(q["category"] for q in queries)),
        "splits": dict(Counter(row["split"] for row in snapshot)),
        "encoder_device": encoder.resolved_device,
        "selection": "8 unique each from BM25, vector and RRF, plus 4 seeded category controls",
        "label_note": (
            "existing silver predicates inform explicit requirements; no grades included"
            if pilot
            else "query-authored requirements; no relevance grades included"
        ),
    }
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=CATALOG)
    parser.add_argument("--queries", type=Path, default=QUERIES)
    parser.add_argument("--embeddings", type=Path, default=EMBEDDINGS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261108)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    print(
        json.dumps(
            build_queue(
                args.catalog, args.queries, args.embeddings, args.output, args.seed, args.device
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
