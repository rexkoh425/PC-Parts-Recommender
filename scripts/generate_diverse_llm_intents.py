"""Generate and checkpoint synthetic, unlabeled query proposals by component category.

Uses a model separate from the judge. All proposals are unverified synthetic
requests; callers must audit, group, freeze and deduplicate them before judging.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

FIELDS = {
    "cpu": [
        "socket",
        "core_count",
        "thread_count",
        "tdp_watts",
        "base_clock_ghz",
        "maximum_memory_gb",
        "included_cooler",
    ],
    "gpu": [
        "vram_gb",
        "vram_type",
        "board_power_watts",
        "length_mm",
        "slot_width",
        "memory_bus_bits",
    ],
    "memory": [
        "capacity_gb",
        "memory_type",
        "speed_mt_s",
        "cas_latency",
        "module_count",
        "module_height_mm",
    ],
    "motherboard": [
        "socket",
        "form_factor",
        "memory_type",
        "memory_slots",
        "maximum_memory_gb",
        "m2_slots",
        "sata_ports",
    ],
    "power_supply": [
        "wattage",
        "form_factor",
        "efficiency_rating",
        "eps_connectors",
        "modular_type",
    ],
    "storage": ["capacity_gb", "form_factor", "interface"],
    "cooler": ["cooler_type", "height_mm", "radiator_size_mm", "fan_count", "supported_sockets"],
    "case": [
        "case_size",
        "maximum_gpu_length_mm",
        "maximum_cooler_height_mm",
        "drive_bays",
        "supported_motherboard_sizes",
    ],
}


def request(model: str, category: str, count: int, avoid: list[str]) -> tuple[list[dict], dict]:
    base = os.environ["OPENAI_BASE_URL"].rstrip("/")
    if not base.endswith("/v1"):
        base += "/v1"
    prompt = (
        f"Produce {count} distinct realistic PC component search intents for category {category}. "
        "Return JSON object with key queries, an array of objects with keys query_text, "
        "intent_family, and must. intent_family must describe the underlying user need, "
        "not just a rewording or a changed number. Avoid threshold variants of one template. "
        "Mix gaming, work, upgrades, constraints, ambiguity and difficult tradeoffs. "
        "Avoid requiring market prices, stock or benchmark claims that are not supplied. "
        "must is an array of only explicit essential requirements from query_text; "
        "each element has field, operator, value. Allowed field names are category_attributes. "
        f"followed by one of {', '.join(FIELDS[category])}; "
        "allowed operators are eq, ge, le, contains. "
        "Use contains only for list-valued supported_sockets or supported_motherboard_sizes. "
        "Use numbers for numeric fields, boolean for included_cooler and lowercase snake_case for "
        "memory_type, form_factor, interface, case_size and cooler_type. "
        "For ambiguous requirements or unverifiable workload suitability, use an empty must array. "
        "Do not invent product facts. Return no grades, examples, or markdown."
    )
    if avoid:
        prompt += " Avoid repeating these existing intent families: " + ", ".join(avoid) + "."
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"},
        "temperature": 0,
    }
    req = urllib.request.Request(
        base + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={
            "Authorization": "Bearer " + os.environ["OPENAI_API_KEY"],
            "Content-Type": "application/json",
        },
        method="POST",
    )
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=180) as response:
                payload = json.load(response)
            value = json.loads(payload["choices"][0]["message"]["content"])
            if not isinstance(value.get("queries"), list):
                raise ValueError("missing queries array")
            return value["queries"], payload.get("usage", {})
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError, ValueError):
            if attempt == 2:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--model", default="gpt-5.6-luna")
    p.add_argument("--count", type=int, default=45)
    p.add_argument("--avoid-dir", type=Path)
    p.add_argument("--concurrency", type=int, default=1)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    def generate_category(category: str) -> tuple[str, int | None]:
        path = args.output / f"{category}.json"
        if path.exists():
            return category, None
        avoid = []
        if args.avoid_dir:
            prior = json.loads((args.avoid_dir / f"{category}.json").read_text(encoding="utf-8"))
            avoid = [str(row.get("intent_family", "")) for row in prior["queries"]]
        proposals, usage = request(args.model, category, args.count, avoid)
        data = {
            "category": category,
            "model_id": args.model,
            "usage": usage,
            "queries": proposals,
            "provenance": "synthetic LLM-authored query proposals; no relevance labels",
        }
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return category, len(proposals)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(generate_category, category) for category in FIELDS]
        for future in concurrent.futures.as_completed(futures):
            category, count = future.result()
            print(category, "resumed" if count is None else f"proposals {count}", flush=True)


if __name__ == "__main__":
    main()
