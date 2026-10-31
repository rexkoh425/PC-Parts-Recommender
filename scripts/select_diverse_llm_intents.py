"""Freeze a diverse synthetic query set, excluding near-duplicate intent wording.

Embeddings provide a conservative automated screen, not proof of semantic independence.
The selector does not inspect any relevance grades or judge explanations.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from pc_build_recommender.retrieval import SentenceTransformerEmbeddingEncoder
from pc_build_recommender.retrieval.embedding_index import load_normalized_product_jsonl
from scripts.build_llm_judge_queue import CATALOG
from scripts.generate_diverse_llm_intents import FIELDS


def normalize(value: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).casefold())


def valid_proposal(category: str, row: object) -> bool:
    if not isinstance(row, dict) or not isinstance(row.get("query_text"), str):
        return False
    text = row["query_text"].strip()
    if not 15 <= len(text) <= 500 or not isinstance(row.get("intent_family"), str):
        return False
    requirements = row.get("must")
    if not isinstance(requirements, list):
        return False
    normalized = normalize(text)
    list_fields = {"supported_sockets", "supported_motherboard_sizes"}
    numeric_fields = {
        "core_count",
        "thread_count",
        "tdp_watts",
        "base_clock_ghz",
        "maximum_memory_gb",
        "vram_gb",
        "board_power_watts",
        "length_mm",
        "slot_width",
        "memory_bus_bits",
        "capacity_gb",
        "speed_mt_s",
        "cas_latency",
        "module_count",
        "module_height_mm",
        "memory_slots",
        "m2_slots",
        "sata_ports",
        "wattage",
        "eps_connectors",
        "height_mm",
        "radiator_size_mm",
        "fan_count",
        "maximum_gpu_length_mm",
        "maximum_cooler_height_mm",
        "drive_bays",
    }
    for requirement in requirements:
        if not isinstance(requirement, dict):
            return False
        field = str(requirement.get("field", ""))
        if not field.startswith("category_attributes."):
            return False
        name = field.removeprefix("category_attributes.")
        op, value = requirement.get("operator"), requirement.get("value")
        if name not in FIELDS[category] or op not in {"eq", "ge", "le", "contains"}:
            return False
        if name in list_fields:
            if op != "contains" or not isinstance(value, str):
                return False
        elif name in numeric_fields:
            if (
                op not in {"eq", "ge", "le"}
                or isinstance(value, bool)
                or not isinstance(value, (int, float))
            ):
                return False
            numbers = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", text.replace(",", ""))]
            if not math.isfinite(value) or float(value) not in numbers:
                return False
        elif name == "included_cooler":
            if op != "eq" or not isinstance(value, bool) or "cooler" not in text.casefold():
                return False
            negative = bool(re.search(r"\b(no|without)\b.{0,24}\bcooler\b", text.casefold()))
            positive = bool(
                re.search(r"\b(included|include|bundled|with)\b.{0,24}\bcooler\b", text.casefold())
            )
            if (value and (not positive or negative)) or (not value and not negative):
                return False
            continue
        elif op != "eq" or not isinstance(value, str):
            return False
        if name not in numeric_fields and (
            not normalize(value) or normalize(value) not in normalized
        ):
            return False
    return True


def canonicalize_proposal(category: str, row: dict, vocabulary: dict) -> dict | None:
    """Ground string constraints in catalogue vocabulary or reject the proposal."""
    result = copy.deepcopy(row)
    for requirement in result["must"]:
        value = requirement["value"]
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            literal = re.escape(f"{value:g}")
            text = result["query_text"].replace(",", "").casefold()
            if requirement["operator"] == "le" and re.search(
                rf"\b(?:under|below|less than)\s+{literal}\b", text
            ):
                requirement["operator"] = "lt"
            elif requirement["operator"] == "ge" and re.search(
                rf"\b(?:over|above|more than|greater than)\s+{literal}\b", text
            ):
                requirement["operator"] = "gt"
        if not isinstance(value, str):
            continue
        field = requirement["field"].removeprefix("category_attributes.")
        normalized = normalize(value)
        if category == "storage" and field == "interface" and normalized == "nvme":
            normalized = normalize("nvme_pcie")
        if category == "cooler" and field == "cooler_type":
            if normalized == "lowprofile" and "air" in result["query_text"].casefold():
                normalized = "air"
            if normalized in {"liquidaio", "liquid", "allinone"}:
                normalized = "aio"
        values = vocabulary.get((category, field), {}).get(normalized, set())
        if len(values) != 1:
            return None
        requirement["value"] = next(iter(values))
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--proposals", required=True, action="append", type=Path)
    p.add_argument(
        "--previous", type=Path, default=Path("evals/retrieval/real_catalog_silver_queries.v1.json")
    )
    p.add_argument("--existing", required=True, type=Path)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--per-category", type=int, default=37)
    p.add_argument("--similarity-threshold", type=float, default=0.80)
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    old = json.loads(args.previous.read_text(encoding="utf-8"))["queries"]
    existing = json.loads(args.existing.read_text(encoding="utf-8"))["queries"]
    records, _ = load_normalized_product_jsonl(CATALOG)
    vocabulary: dict[tuple[str, str], dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for record in records:
        for field, value in record.get("category_attributes", {}).items():
            for item in value if isinstance(value, list) else [value]:
                if isinstance(item, str) and item:
                    vocabulary[(record["category"], field)][normalize(item)].add(item)
    completed = Counter(
        json.loads(line)["query_id"]
        for line in args.checkpoint.read_text(encoding="utf-8").splitlines()
    )
    by_family: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in existing:
        by_family[(row["category"], row["family"])].append(row)
    retained = [
        max(rows, key=lambda x: (completed[x["query_id"]], x["query_id"]))
        for rows in by_family.values()
    ]
    proposals = []
    invalid = Counter()
    proposal_hashes = {}
    for category in FIELDS:
        for batch, proposal_dir in enumerate(args.proposals):
            proposal_path = proposal_dir / f"{category}.json"
            proposal_hashes[f"{batch}:{category}"] = hashlib.sha256(
                proposal_path.read_bytes()
            ).hexdigest()
            source = json.loads(proposal_path.read_text(encoding="utf-8"))
            if source["category"] != category:
                raise ValueError("proposal file category mismatch")
            for index, row in enumerate(source["queries"]):
                if not valid_proposal(category, row):
                    invalid[category] += 1
                    continue
                row = canonicalize_proposal(category, row, vocabulary)
                if row is None:
                    invalid[category] += 1
                    continue
                qid = f"diverse-v1-{category}-{batch:02d}-{index:03d}"
                proposals.append(
                    {
                        "query_id": qid,
                        "query_group_id": qid,
                        "category": category,
                        "query_text": row["query_text"].strip(),
                        "must": row["must"],
                        "family": row["intent_family"].casefold().strip(),
                        "query_generation": source["model_id"],
                    }
                )
    all_rows = old + retained + proposals
    encoder = SentenceTransformerEmbeddingEncoder(
        "sentence-transformers/all-MiniLM-L6-v2", device=args.device, batch_size=64
    )
    matrix = encoder.encode([r["query_text"] for r in all_rows])
    old_count, old_end = len(old), len(old) + len(retained)
    selected = []
    report = {}
    for category in FIELDS:
        previous_indices = [i for i, r in enumerate(old) if r["category"] == category]
        candidates = [
            i for i in range(old_count, len(all_rows)) if all_rows[i]["category"] == category
        ]
        candidates.sort(
            key=lambda i: (
                i >= old_end,
                -completed[all_rows[i]["query_id"]],
                all_rows[i]["query_id"],
            )
        )
        chosen: list[int] = []
        family_names = set()
        previous_vectors = matrix[previous_indices]
        for i in candidates:
            row = all_rows[i]
            if previous_indices and np.any(
                previous_vectors @ matrix[i] >= args.similarity_threshold
            ):
                continue
            family = normalize(row["family"])
            if family in family_names or (
                chosen and np.any(matrix[chosen] @ matrix[i] >= args.similarity_threshold)
            ):
                continue
            chosen.append(i)
            family_names.add(family)
            if len(chosen) == args.per_category:
                break
        selected.extend(all_rows[i] for i in chosen)
        report[category] = {
            "selected": len(chosen),
            "retained_existing": sum(i < old_end for i in chosen),
            "invalid_proposals": invalid[category],
        }
    if len(selected) < 250:
        raise ValueError(f"only {len(selected)} distinct screened intents; need at least 250")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "pcbr.diverse-synthetic-intents.v1",
        "queries": selected,
        "selection": report,
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "similarity_threshold": args.similarity_threshold,
        "proposal_file_sha256": proposal_hashes,
        "previous_query_file_sha256": hashlib.sha256(args.previous.read_bytes()).hexdigest(),
        "existing_query_file_sha256": hashlib.sha256(args.existing.read_bytes()).hexdigest(),
        "query_generator_script_sha256": hashlib.sha256(
            Path("scripts/generate_diverse_llm_intents.py").read_bytes()
        ).hexdigest(),
        "selector_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "checkpoint_query_counts_only_sha256": hashlib.sha256(
            args.checkpoint.read_bytes()
        ).hexdigest(),
        "limitation": (
            "synthetic model-authored intents; automated similarity is not "
            "human independence review"
        ),
    }
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"query_count": len(selected), "categories": report}, indent=2))


if __name__ == "__main__":
    main()
