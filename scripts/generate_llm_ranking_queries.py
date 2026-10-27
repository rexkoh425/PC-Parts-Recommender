"""Create a reproducible, synthetic set of distinct specification requests.

Run only after the existing judge pilot passes. These authored intents are not
customer observations. Review semantic overlap before treating groups as independent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def constraint(field: str, operator: str, value: object) -> dict:
    return {"field": f"category_attributes.{field}", "operator": operator, "value": value}


def intent_signature(query: dict) -> str:
    return json.dumps(
        [query["category"], sorted(query["must"], key=lambda item: item["field"])],
        sort_keys=True,
        separators=(",", ":"),
    )


def exclude_previous(queries: list[dict], previous: list[dict]) -> list[dict]:
    signatures = {intent_signature(query) for query in previous}
    texts = {query["query_text"].casefold().strip() for query in previous}
    return [
        q
        for q in queries
        if intent_signature(q) not in signatures and q["query_text"].casefold().strip() not in texts
    ]


def generate() -> list[dict]:
    queries: list[dict] = []

    def add(category: str, family: str, text: str, requirements: list[dict]) -> None:
        index = sum(q["category"] == category and q["family"] == family for q in queries)
        query_id = f"fresh-v1-{category}-{family}-{index:02d}"
        signature = json.dumps(
            [category, sorted(requirements, key=lambda item: item["field"])],
            sort_keys=True,
            separators=(",", ":"),
        )
        group_id = "intent-" + hashlib.sha256(signature.encode()).hexdigest()[:16]
        queries.append(
            {
                "query_id": query_id,
                "query_group_id": group_id,
                "family": family,
                "category": category,
                "query_text": text,
                "must": requirements,
            }
        )

    capacities = [4, 6, 8, 10, 12, 16, 20, 24]
    for i, cores in enumerate(capacities):
        power = [45, 65, 95, 125, 170, 220, 250, 280][i]
        socket = ["AM4", "AM5", "LGA 1200", "LGA 1700", "LGA 1851", "LGA 2066", "AM4", "AM5"][i]
        add(
            "cpu",
            "render-power",
            f"CPU for rendering with at least {cores} cores and TDP at most {power} W",
            [constraint("core_count", "ge", cores), constraint("tdp_watts", "le", power)],
        )
        add(
            "cpu",
            "upgrade-socket",
            f"Upgrade processor for {socket}, at least {cores} cores for development",
            [constraint("socket", "eq", socket), constraint("core_count", "ge", cores)],
        )
        threads = cores * 2
        add(
            "cpu",
            "threads-clock",
            f"CPU for compiling code with at least {threads} threads and base clock of 2.5 GHz",
            [constraint("thread_count", "ge", threads), constraint("base_clock_ghz", "ge", 2.5)],
        )
        memory = [32, 64, 96, 128, 192, 256, 512, 1024][i]
        add(
            "cpu",
            "memory-support",
            f"Workstation CPU supporting at least {memory} GB memory and {cores} cores",
            [constraint("maximum_memory_gb", "ge", memory), constraint("core_count", "ge", cores)],
        )
        add(
            "cpu",
            "cooler-included",
            f"Processor with bundled cooler, {socket} socket, TDP no more than {power} W",
            [
                constraint("included_cooler", "eq", True),
                constraint("socket", "eq", socket),
                constraint("tdp_watts", "le", power),
            ],
        )

    for i, vram in enumerate([4, 6, 8, 10, 12, 16, 20, 24]):
        power = [75, 120, 150, 180, 220, 260, 300, 350][i]
        length = [180, 200, 220, 240, 260, 280, 300, 320][i]
        bus = [64, 96, 128, 160, 192, 256, 320, 384][i]
        add(
            "gpu",
            "vram-power",
            f"Graphics card for 1440p gaming with at least {vram} GB VRAM "
            f"and board power at most {power} W",
            [constraint("vram_gb", "ge", vram), constraint("board_power_watts", "le", power)],
        )
        add(
            "gpu",
            "compact-vram",
            f"GPU for a small PC with {vram} GB or more VRAM and length at most {length} mm",
            [constraint("vram_gb", "ge", vram), constraint("length_mm", "le", length)],
        )
        typ = ["GDDR5", "GDDR6", "GDDR6X", "GDDR7"][i % 4]
        add(
            "gpu",
            "memory-type",
            f"Graphics card using {typ} memory with at least {vram} GB for media work",
            [constraint("vram_type", "eq", typ), constraint("vram_gb", "ge", vram)],
        )
        slots = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5][i]
        add(
            "gpu",
            "case-clearance",
            f"GPU fitting {length} mm length and {slots} slots for an existing case",
            [constraint("length_mm", "le", length), constraint("slot_width", "le", slots)],
        )
        add(
            "gpu",
            "bus-vram",
            f"Graphics card with a memory bus at least {bus} bits and {vram} GB VRAM for compute",
            [constraint("memory_bus_bits", "ge", bus), constraint("vram_gb", "ge", vram)],
        )

    for i, capacity in enumerate([8, 16, 24, 32, 48, 64, 96, 128]):
        speed = [2400, 2666, 3200, 3600, 4800, 5600, 6000, 6400][i]
        typ = "ddr4" if i < 4 else "ddr5"
        modules = [1, 2, 4, 1, 2, 4, 2, 4][i]
        latency = [14, 16, 18, 20, 28, 32, 36, 40][i]
        height = [25, 28, 30, 32, 34, 36, 40, 45][i]
        add(
            "memory",
            "capacity-speed",
            f"Memory for development with at least {capacity} GB and speed at least {speed} MT/s",
            [constraint("capacity_gb", "ge", capacity), constraint("speed_mt_s", "ge", speed)],
        )
        add(
            "memory",
            "type-capacity",
            f"Upgrade RAM for a {typ.upper()} motherboard, at least {capacity} GB",
            [constraint("memory_type", "eq", typ), constraint("capacity_gb", "ge", capacity)],
        )
        add(
            "memory",
            "latency-speed",
            f"RAM with CAS latency at most {latency} and speed at least {speed} MT/s",
            [constraint("cas_latency", "le", latency), constraint("speed_mt_s", "ge", speed)],
        )
        add(
            "memory",
            "modules-capacity",
            f"Memory kit with {modules} modules and at least {capacity} GB total",
            [constraint("module_count", "eq", modules), constraint("capacity_gb", "ge", capacity)],
        )
        add(
            "memory",
            "clearance",
            f"Low clearance memory at least {capacity} GB and at most {height} mm tall",
            [
                constraint("capacity_gb", "ge", capacity),
                constraint("module_height_mm", "le", height),
            ],
        )

    for i, socket in enumerate(
        ["AM4", "AM5", "LGA 1151", "LGA 1200", "LGA 1700", "LGA 1851", "LGA 2066", "AM4"]
    ):
        typ = "ddr5" if socket in {"AM5", "LGA 1851"} else "ddr4"
        factor = ["atx", "micro_atx", "mini_itx", "e_atx"][i % 4]
        slots = [1, 2, 3, 4, 5, 6, 7, 8][i]
        mem = [32, 64, 96, 128, 192, 256, 512, 1024][i]
        add(
            "motherboard",
            "socket-memory",
            f"Motherboard for {socket} CPU using {typ.upper()} and supporting {mem} GB memory",
            [
                constraint("socket", "eq", socket),
                constraint("memory_type", "eq", typ),
                constraint("maximum_memory_gb", "ge", mem),
            ],
        )
        add(
            "motherboard",
            "case-slots",
            f"{factor} motherboard with at least {slots} memory slots for an existing case",
            [constraint("form_factor", "eq", factor), constraint("memory_slots", "ge", slots)],
        )
        add(
            "motherboard",
            "m2-capacity",
            f"Motherboard with at least {slots} M.2 slots and {mem} GB maximum memory",
            [constraint("m2_slots", "ge", slots), constraint("maximum_memory_gb", "ge", mem)],
        )
        add(
            "motherboard",
            "storage-ports",
            f"Motherboard for {socket} with at least {slots} SATA ports and two M.2 slots",
            [
                constraint("socket", "eq", socket),
                constraint("sata_ports", "ge", slots),
                constraint("m2_slots", "ge", 2),
            ],
        )
        add(
            "motherboard",
            "memory-slots",
            f"{typ.upper()} motherboard with at least {slots} memory slots "
            f"and {mem} GB maximum RAM",
            [
                constraint("memory_type", "eq", typ),
                constraint("memory_slots", "ge", slots),
                constraint("maximum_memory_gb", "ge", mem),
            ],
        )

    watts = [350, 450, 550, 650, 750, 850, 1000, 1200]
    for i, watt in enumerate(watts):
        factor = "sfx" if i % 2 else "atx"
        efficiency = ["80_plus_bronze", "80_plus_gold", "80_plus_platinum", "80_plus_titanium"][
            i % 4
        ]
        plugs = 1 + i % 4
        modular = "semi_modular" if i % 2 else "non_modular"
        add(
            "power_supply",
            "size-wattage",
            f"{factor.upper()} power supply rated at least {watt} W for a compact build",
            [constraint("form_factor", "eq", factor), constraint("wattage", "ge", watt)],
        )
        add(
            "power_supply",
            "efficiency",
            f"Power supply rated at least {watt} W with {efficiency.replace('_', ' ')} efficiency",
            [constraint("wattage", "ge", watt), constraint("efficiency_rating", "eq", efficiency)],
        )
        add(
            "power_supply",
            "gpu-connectors",
            f"PSU rated at least {watt} W with at least {plugs} 6+2 pin PCIe connectors for a GPU",
            [
                constraint("wattage", "ge", watt),
                constraint("pcie_connectors.6_plus_2_pin", "ge", plugs),
            ],
        )
        add(
            "power_supply",
            "cpu-connectors",
            f"PSU with at least {plugs} EPS connectors and {watt} W capacity",
            [constraint("eps_connectors", "ge", plugs), constraint("wattage", "ge", watt)],
        )
        add(
            "power_supply",
            "cabling",
            f"{modular.replace('_', ' ')} power supply with at least {watt} W output",
            [constraint("modular_type", "eq", modular), constraint("wattage", "ge", watt)],
        )

    for i, capacity in enumerate([250, 500, 1000, 1500, 2000, 4000, 8000, 12000]):
        factor = ["m2_2280", "2_5_inch", "3_5_inch", "m2_2230"][i % 4]
        interface = "nvme_pcie" if factor.startswith("m2") else "sata"
        add(
            "storage",
            "workload-interface",
            f"Storage for video projects, at least {capacity} GB using {interface}",
            [constraint("capacity_gb", "ge", capacity), constraint("interface", "eq", interface)],
        )
        add(
            "storage",
            "physical-form",
            f"Replacement storage of form factor {factor} and capacity at least {capacity} GB",
            [constraint("form_factor", "eq", factor), constraint("capacity_gb", "ge", capacity)],
        )
        add(
            "storage",
            "backup",
            f"Backup drive with at least {capacity} GB using SATA",
            [constraint("capacity_gb", "ge", capacity), constraint("interface", "eq", "sata")],
        )
        add(
            "storage",
            "operating-system",
            f"NVMe boot drive with at least {capacity} GB for a development PC",
            [constraint("capacity_gb", "ge", capacity), constraint("interface", "eq", "nvme_pcie")],
        )
        add(
            "storage",
            "bay",
            f"Storage fitting a 3.5 inch bay with at least {capacity} GB for archival data",
            [
                constraint("capacity_gb", "ge", capacity),
                constraint("form_factor", "eq", "3_5_inch"),
            ],
        )

    for i, socket in enumerate(
        ["AM4", "AM5", "LGA 1200", "LGA 1700", "LGA 1851", "LGA 2066", "AM3", "LGA 1151"]
    ):
        height = [45, 60, 80, 100, 120, 140, 160, 180][i]
        radiator = [120, 140, 180, 240, 280, 360, 420, 480][i]
        fans = 1 + i % 3
        add(
            "cooler",
            "socket-height",
            f"Air cooler compatible with {socket}, at most {height} mm high",
            [
                constraint("supported_sockets", "contains", socket),
                constraint("height_mm", "le", height),
                constraint("cooler_type", "eq", "air"),
            ],
        )
        add(
            "cooler",
            "socket-radiator",
            f"AIO cooler for {socket} with a {radiator} mm radiator",
            [
                constraint("supported_sockets", "contains", socket),
                constraint("radiator_size_mm", "eq", radiator),
                constraint("cooler_type", "eq", "aio"),
            ],
        )
        add(
            "cooler",
            "fan-socket",
            f"CPU cooler supporting {socket} with at least {fans} fans",
            [
                constraint("supported_sockets", "contains", socket),
                constraint("fan_count", "ge", fans),
            ],
        )
        add(
            "cooler",
            "height-fans",
            f"Low height air cooler up to {height} mm and at least {fans} fans",
            [
                constraint("height_mm", "le", height),
                constraint("fan_count", "ge", fans),
                constraint("cooler_type", "eq", "air"),
            ],
        )
        add(
            "cooler",
            "radiator-fans",
            f"Liquid cooler with a {radiator} mm radiator and at least {fans} fans",
            [constraint("radiator_size_mm", "eq", radiator), constraint("fan_count", "ge", fans)],
        )

    for i, clearance in enumerate([220, 240, 260, 280, 300, 320, 340, 360]):
        factor = ["atx", "micro_atx", "mini_itx", "e_atx"][i % 4]
        size = ["mid_tower", "mini_tower", "full_tower", "small_form_factor"][i % 4]
        bays = 1 + i
        height = [100, 110, 120, 130, 140, 150, 160, 170][i]
        add(
            "case",
            "gpu-drive-bays",
            f"PC case for a {clearance} mm graphics card with at least {bays} drive bays",
            [
                constraint("maximum_gpu_length_mm", "ge", clearance),
                constraint("drive_bays", "ge", bays),
            ],
        )
        add(
            "case",
            "board-gpu",
            f"Case supporting {factor} motherboard and a GPU at least {clearance} mm long",
            [
                constraint("supported_motherboard_sizes", "contains", factor),
                constraint("maximum_gpu_length_mm", "ge", clearance),
            ],
        )
        add(
            "case",
            "tower-clearance",
            f"{size.replace('_', ' ')} case with GPU clearance at least {clearance} mm",
            [
                constraint("case_size", "eq", size),
                constraint("maximum_gpu_length_mm", "ge", clearance),
            ],
        )
        add(
            "case",
            "cooler-gpu",
            f"Case fitting a {height} mm CPU cooler and {clearance} mm GPU",
            [
                constraint("maximum_cooler_height_mm", "ge", height),
                constraint("maximum_gpu_length_mm", "ge", clearance),
            ],
        )
        add(
            "case",
            "board-storage",
            f"Case supporting {factor} motherboard with at least {bays} drive bays",
            [
                constraint("supported_motherboard_sizes", "contains", factor),
                constraint("drive_bays", "ge", bays),
            ],
        )
    return queries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--previous", type=Path, default=Path("evals/retrieval/real_catalog_silver_queries.v1.json")
    )
    args = parser.parse_args()
    previous = json.loads(args.previous.read_text(encoding="utf-8"))["queries"]
    queries = exclude_previous(generate(), previous)
    if args.output.exists():
        raise FileExistsError(args.output)
    ids = [q["query_id"] for q in queries]
    texts = [q["query_text"].casefold().strip() for q in queries]
    if len(set(ids)) != len(ids) or len(set(texts)) != len(texts):
        raise ValueError("duplicate generated ID or text")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "schema_version": "pcbr.authored-ranking-intents.v1",
                "prior_query_file_sha256": hashlib.sha256(args.previous.read_bytes()).hexdigest(),
                "excluded_prior_constraint_matches": len(generate()) - len(queries),
                "queries": queries,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(queries)} synthetic authored queries to {args.output}")


if __name__ == "__main__":
    main()
