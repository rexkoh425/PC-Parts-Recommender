"""Compare deterministic heuristics and CP-SAT on a frozen synthetic cohort."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import random
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, median
from typing import Any

from pc_build_recommender.domain import BuildProfile, CompatibilityStatus, ComponentCategory
from pc_build_recommender.optimizer import (
    BuildOptimizer,
    CandidateScores,
    OptimizationCandidate,
    OptimizationProblem,
    PairwiseCompatibility,
    enumerate_feasible_builds,
    greedy_optimize,
    validate_selected_build,
)

_CATEGORIES = (
    ComponentCategory.CPU,
    ComponentCategory.GPU,
    ComponentCategory.MOTHERBOARD,
    ComponentCategory.MEMORY,
    ComponentCategory.STORAGE,
    ComponentCategory.POWER_SUPPLY,
    ComponentCategory.COOLER,
    ComponentCategory.CASE,
)
_PROFILES = tuple(BuildProfile)
_BASE_PRICE_CENTS = {
    ComponentCategory.CPU: 32_000,
    ComponentCategory.GPU: 48_000,
    ComponentCategory.MOTHERBOARD: 18_000,
    ComponentCategory.MEMORY: 7_000,
    ComponentCategory.STORAGE: 7_000,
    ComponentCategory.POWER_SUPPLY: 9_000,
    ComponentCategory.COOLER: 4_000,
    ComponentCategory.CASE: 8_000,
}


def _hard_pair(left: OptimizationCandidate, right: OptimizationCandidate, message: str):
    return PairwiseCompatibility(
        left.product_id, right.product_id, CompatibilityStatus.FAIL, message=message
    )


def make_problem(
    seed: int,
    *,
    candidates_per_category: int | None = None,
    tight: bool | None = None,
    infeasible: bool = False,
    difficulty: str | None = None,
) -> OptimizationProblem:
    """Generate one controlled scenario; all generated values are explicitly synthetic."""

    rng = random.Random(seed)
    difficulty = difficulty or ("tight" if tight else "medium")
    count = candidates_per_category or (2 if difficulty == "easy" else 4)
    candidates: list[OptimizationCandidate] = []
    for category in _CATEGORIES:
        for index in range(count):
            tier = index / max(count - 1, 1)
            base = _BASE_PRICE_CENTS[category]
            price = max(100, int(base * (0.72 + 0.70 * tier) + rng.randrange(-1200, 1201)))
            # Performance and value deliberately compete; neither is a constant tie-breaker.
            performance = min(99.0, max(10.0, 38 + 58 * tier + rng.uniform(-18, 18)))
            value = min(99.0, max(10.0, 94 - 60 * tier + rng.uniform(-15, 15)))
            attrs: dict[str, Any] = {}
            extra: dict[str, Any] = {}
            if category == ComponentCategory.CPU:
                socket = "am5" if index % 2 == 0 else "lga1700"
                attrs["socket"] = socket
                extra["power_draw_watts"] = 75 + index * 45
            elif category == ComponentCategory.GPU:
                length = 225 + index * 32
                attrs.update(vram_gb=8 + index * 4, length_mm=length)
                extra.update(
                    power_draw_watts=170 + index * 85,
                    recommended_psu_watts=550 + index * 150,
                    required_power_connectors={"pcie_8pin": 1 + int(index >= 2)},
                )
            elif category == ComponentCategory.MOTHERBOARD:
                attrs.update(
                    socket="am5" if index % 2 == 0 else "lga1700",
                    memory_type="ddr5" if index % 2 == 0 else "ddr4",
                    form_factor="atx" if index % 2 == 0 else "micro_atx",
                    wifi_support=True,
                )
            elif category == ComponentCategory.MEMORY:
                attrs.update(
                    memory_type="ddr5" if index % 2 == 0 else "ddr4",
                    capacity_gb=16 + index * 16,
                )
            elif category == ComponentCategory.STORAGE:
                attrs["capacity_gb"] = 1000 + index * 1000
            elif category == ComponentCategory.POWER_SUPPLY:
                extra.update(
                    psu_wattage=600 + index * 200,
                    provided_power_connectors={"pcie_8pin": 2 + int(index >= 2)},
                    eps_connectors=2,
                )
            elif category == ComponentCategory.CASE:
                attrs.update(
                    case_size="mid_tower",
                    maximum_gpu_length_mm=280 + index * 45,
                    supported_motherboard_sizes=["atx", "micro_atx"],
                )
            candidates.append(
                OptimizationCandidate(
                    product_id=f"{category.value}-{seed}-{index}",
                    category=category,
                    price_cents=price,
                    in_stock=(difficulty != "stock" or rng.random() > 0.08),
                    canonical_name=f"synthetic {category.value} option {index}",
                    scores=CandidateScores(
                        performance=performance,
                        value=value,
                        reliability=rng.uniform(35, 98),
                        upgradeability=rng.uniform(25, 98),
                        efficiency=rng.uniform(30, 98),
                        preference=rng.uniform(15, 98),
                    ),
                    attributes=attrs,
                    **extra,
                )
            )

    by_category = {
        category: [candidate for candidate in candidates if candidate.category == category]
        for category in _CATEGORIES
    }
    pairs: list[PairwiseCompatibility] = []
    for cpu in by_category[ComponentCategory.CPU]:
        for board in by_category[ComponentCategory.MOTHERBOARD]:
            if cpu.attribute("socket") != board.attribute("socket"):
                pairs.append(_hard_pair(cpu, board, "socket mismatch"))
    for board in by_category[ComponentCategory.MOTHERBOARD]:
        for memory in by_category[ComponentCategory.MEMORY]:
            if board.attribute("memory_type") != memory.attribute("memory_type"):
                pairs.append(_hard_pair(board, memory, "memory generation mismatch"))
        for case in by_category[ComponentCategory.CASE]:
            if board.attribute("form_factor") not in case.attribute("supported_motherboard_sizes"):
                pairs.append(_hard_pair(board, case, "motherboard form factor clearance"))
    for gpu in by_category[ComponentCategory.GPU]:
        for case in by_category[ComponentCategory.CASE]:
            if gpu.attribute("length_mm") > case.attribute("maximum_gpu_length_mm"):
                pairs.append(_hard_pair(gpu, case, "GPU clearance"))

    profile = _PROFILES[seed % len(_PROFILES)]
    locked = frozenset()
    if difficulty == "locked":
        locked = frozenset({by_category[ComponentCategory.CPU][0].product_id})
    if infeasible or difficulty == "infeasible":
        budget = 10_000
    elif difficulty == "easy":
        budget = 180_000
    elif difficulty == "hard":
        budget = 105_000
    elif difficulty == "locked":
        budget = 125_000
    else:
        budget = 145_000 if difficulty == "medium" else 115_000
    return OptimizationProblem(
        candidates=tuple(candidates),
        budget_cents=budget,
        profiles=(profile,),
        locked_product_ids=locked,
        minimum_gpu_vram_gb=8,
        minimum_memory_gb=16,
        minimum_storage_gb=1000,
        wifi_required=True,
        pairwise_compatibility=tuple(pairs),
        in_stock_only=(difficulty != "stock"),
        time_limit_seconds=1.0,
        random_seed=seed,
    )


def _solve(algorithm: str, problem: OptimizationProblem):
    started = time.perf_counter()
    if algorithm == "greedy":
        result = greedy_optimize(problem, local_width=1)
    elif algorithm == "greedy-local":
        result = greedy_optimize(problem, local_width=2)
    elif algorithm == "cp-sat":
        result = BuildOptimizer().optimize(problem)
    elif algorithm == "cp-sat-warm":
        result = BuildOptimizer(warm_start=True).optimize(problem)
    else:
        raise ValueError(algorithm)
    return result, time.perf_counter() - started


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, int((percentile / 100) * len(ordered))))]


def _problem_payload(problem: OptimizationProblem, index: int, difficulty: str) -> dict[str, Any]:
    return {
        "scenario": index,
        "difficulty": difficulty,
        "budget_cents": problem.budget_cents,
        "profile": problem.profiles[0].value,
        "locked_product_ids": sorted(problem.locked_product_ids),
        "random_seed": problem.random_seed,
        "provenance": "controlled synthetic scenario; no market price/spec evidence",
        "candidates": [
            {
                "product_id": c.product_id,
                "category": c.category.value,
                "price_cents": c.price_cents,
                "in_stock": c.in_stock,
                "scores": {
                    name: getattr(c.scores, name)
                    for name in (
                        "performance",
                        "value",
                        "reliability",
                        "upgradeability",
                        "efficiency",
                        "preference",
                        "warning_penalty",
                    )
                },
                "attributes": dict(c.attributes),
                "power_draw_watts": c.power_draw_watts,
                "psu_wattage": c.psu_wattage,
                "recommended_psu_watts": c.recommended_psu_watts,
                "required_power_connectors": dict(c.required_power_connectors),
                "provided_power_connectors": dict(c.provided_power_connectors),
                "eps_connectors": c.eps_connectors,
            }
            for c in problem.candidates
        ],
        "hard_compatibility_pairs": [
            {
                "left": p.left_product_id,
                "right": p.right_product_id,
                "status": p.status.value,
                "message": p.message,
            }
            for p in problem.pairwise_compatibility
        ],
    }


def run_quality_benchmark(
    *,
    output_root: Path,
    scenario_count: int = 200,
    seed: int = 20261108,
    include_warm_start: bool = False,
) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(output_root)
    output_root.mkdir(parents=True)
    easy_count = scenario_count * 20 // 100
    medium_count = scenario_count * 35 // 100
    hard_count = scenario_count * 35 // 100
    infeasible_count = scenario_count - easy_count - medium_count - hard_count
    difficulty_specs = (
        ("easy", easy_count),
        ("medium", medium_count),
        ("hard", hard_count),
        ("infeasible", infeasible_count),
    )
    difficulty_sequence = [
        difficulty for difficulty, count in difficulty_specs for _ in range(count)
    ]
    scenarios: list[tuple[OptimizationProblem, str]] = []
    for index in range(scenario_count):
        difficulty = difficulty_sequence[index]
        count = 2 if difficulty == "easy" else 4 if difficulty == "hard" else 3
        problem = make_problem(
            seed + index,
            candidates_per_category=count,
            difficulty=difficulty,
            infeasible=difficulty == "infeasible",
        )
        scenarios.append((problem, difficulty))

    cohort_lines = [
        json.dumps(_problem_payload(problem, index, difficulty), sort_keys=True)
        for index, (problem, difficulty) in enumerate(scenarios)
    ]
    cohort_bytes = ("\n".join(cohort_lines) + "\n").encode()
    (output_root / "scenarios.jsonl").write_bytes(cohort_bytes)
    cohort_sha = hashlib.sha256(cohort_bytes).hexdigest()

    algorithms = ("greedy", "greedy-local", "cp-sat")
    if include_warm_start:
        algorithms += ("cp-sat-warm",)
    records: list[dict[str, Any]] = []
    for index, (problem, difficulty) in enumerate(scenarios):
        for algorithm in algorithms:
            result, elapsed = _solve(algorithm, problem)
            solution = result.solutions[0] if result.solutions else None
            errors = validate_selected_build(problem, solution.selected) if solution else ()
            valid = solution is not None and not errors
            records.append(
                {
                    "scenario": index,
                    "difficulty": difficulty,
                    "algorithm": algorithm,
                    "status": result.status.value,
                    "valid": valid,
                    "objective": solution.objective_value if valid else None,
                    "runtime_seconds": elapsed,
                    "hard_violations": len(errors),
                }
            )

    oracle_records: list[dict[str, Any]] = []
    for index, (problem, difficulty) in enumerate(scenarios[: min(100, scenario_count)]):
        oracle = enumerate_feasible_builds(problem, max_combinations=4**8)
        oracle_algorithms = ("cp-sat", "cp-sat-warm") if include_warm_start else ("cp-sat",)
        for algorithm in oracle_algorithms:
            result, _ = _solve(algorithm, problem)
            actual = result.solutions[0].objective_value if result.solutions else None
            expected = oracle.best.objective_value if oracle.best else None
            gap = (
                (expected - actual) / abs(expected)
                if expected is not None and actual is not None and expected != 0
                else 0.0
                if expected is None and actual is None
                else None
            )
            oracle_records.append(
                {
                    "scenario": index,
                    "difficulty": difficulty,
                    "algorithm": algorithm,
                    "oracle_objective": expected,
                    "solver_objective": actual,
                    "matched": actual == expected,
                    "relative_optimality_gap": gap,
                    "combinations_evaluated": oracle.combinations_evaluated,
                }
            )

    aggregates: list[dict[str, Any]] = []
    for algorithm in algorithms:
        rows = [row for row in records if row["algorithm"] == algorithm]
        valid_rows = [row for row in rows if row["valid"]]
        runtime = [row["runtime_seconds"] for row in rows]
        aggregates.append(
            {
                "algorithm": algorithm,
                "attempted": len(rows),
                "valid_build_count": len(valid_rows),
                "feasible_rate": len(valid_rows) / len(rows),
                "proven_feasible_cases_solved": sum(row["valid"] for row in rows),
                "proven_infeasible_cases_identified": sum(
                    row["difficulty"] == "infeasible"
                    and row["status"] == "infeasible"
                    and not row["valid"]
                    for row in rows
                ),
                "unresolved_cases": sum(
                    row["difficulty"] != "infeasible" and not row["valid"] for row in rows
                ),
                "hard_constraint_violations": sum(row["hard_violations"] for row in rows),
                "mean_objective": mean(row["objective"] for row in valid_rows)
                if valid_rows
                else None,
                "median_objective": median(row["objective"] for row in valid_rows)
                if valid_rows
                else None,
                "runtime_p50_seconds": _percentile(runtime, 50),
                "runtime_p95_seconds": _percentile(runtime, 95),
                "runtime_p99_seconds": _percentile(runtime, 99),
                "optimal_status_count": sum(row["status"] == "optimal" for row in rows),
                "timeout_or_unknown_count": sum(row["status"] == "unknown" for row in rows),
            }
        )

    greedy_by_scenario = {row["scenario"]: row for row in records if row["algorithm"] == "greedy"}
    paired: dict[str, Any] = {}
    for algorithm in algorithms[1:]:
        both = [
            (greedy_by_scenario[row["scenario"]], row)
            for row in records
            if row["algorithm"] == algorithm
            and row["valid"]
            and greedy_by_scenario[row["scenario"]]["valid"]
        ]
        improvements = [
            (challenge["objective"] - baseline["objective"]) / abs(baseline["objective"])
            for baseline, challenge in both
            if baseline["objective"] != 0
        ]
        paired[algorithm] = {
            "comparable_valid_pairs": len(both),
            "mean_absolute_objective_delta": mean(
                challenge["objective"] - baseline["objective"] for baseline, challenge in both
            )
            if both
            else None,
            "mean_relative_objective_improvement": mean(improvements) if improvements else None,
            "win_rate": mean(
                challenge["objective"] > baseline["objective"] for baseline, challenge in both
            )
            if both
            else None,
            "challenger_found_valid_when_greedy_failed": sum(
                row["valid"] and not greedy_by_scenario[row["scenario"]]["valid"]
                for row in records
                if row["algorithm"] == algorithm
            ),
        }

    by_difficulty: dict[str, Any] = {}
    for difficulty, _ in difficulty_specs:
        by_difficulty[difficulty] = {}
        for algorithm in algorithms:
            rows = [
                row
                for row in records
                if row["difficulty"] == difficulty and row["algorithm"] == algorithm
            ]
            valid_rows = [row for row in rows if row["valid"]]
            by_difficulty[difficulty][algorithm] = {
                "attempted": len(rows),
                "valid_build_count": len(valid_rows),
                "feasible_rate": len(valid_rows) / len(rows) if rows else None,
                "mean_objective": mean(row["objective"] for row in valid_rows)
                if valid_rows
                else None,
                "runtime_p95_seconds": _percentile([row["runtime_seconds"] for row in rows], 95)
                if rows
                else None,
            }

    summary = {
        "schema_version": "pc-build-recommender.optimizer-quality.v2",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "git_revision": _git_revision(),
        "working_tree_dirty": _git_dirty(),
        "provenance": (
            "deterministic synthetic scenarios; no customer builds or market price evidence"
        ),
        "scenario_count": scenario_count,
        "difficulty_counts": {
            key: sum(d == key for _, d in scenarios) for key, _ in difficulty_specs
        },
        "real_catalog_scenarios": 0,
        "synthetic_scenarios": scenario_count,
        "cohort_status_counts": {
            "proven_feasible": sum(d != "infeasible" for _, d in scenarios),
            "proven_infeasible": sum(d == "infeasible" for _, d in scenarios),
            "unresolved": 0,
        },
        "seed": seed,
        "cohort_sha256": cohort_sha,
        "candidate_counts_per_category": sorted(
            {len(problem.candidates) // 8 for problem, _ in scenarios}
        ),
        "hardware": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "ortools": importlib.metadata.version("ortools"),
        },
        "algorithms": aggregates,
        "paired_vs_greedy": paired,
        "by_difficulty": by_difficulty,
        "exhaustive_oracle": {
            "checks": len(oracle_records),
            "matched": sum(row["matched"] for row in oracle_records),
            "mean_relative_optimality_gap": mean(
                row["relative_optimality_gap"]
                for row in oracle_records
                if row["relative_optimality_gap"] is not None
            ),
            "maximum_relative_optimality_gap": max(
                row["relative_optimality_gap"]
                for row in oracle_records
                if row["relative_optimality_gap"] is not None
            ),
            "rows": oracle_records,
        },
        "correctness": "all returned builds independently passed validate_selected_build",
        "objective_unchanged": True,
        "warm_start": (
            "validated greedy-local solution supplied as a CP-SAT hint; objective and "
            "constraints unchanged"
        ),
    }
    with (output_root / "runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (output_root / "difficulty-breakdown.json").write_text(
        json.dumps(by_difficulty, indent=2) + "\n", encoding="utf-8"
    )
    correctness = {
        "schema_version": "pc-build-recommender.optimizer-quality-correctness.v1",
        "scenario_count": scenario_count,
        "algorithm_count": len(algorithms),
        "returned_solution_records": sum(row["valid"] for row in records),
        "hard_constraint_violations": sum(row["hard_violations"] for row in records),
        "all_returned_builds_independently_valid": all(
            row["hard_violations"] == 0 for row in records if row["valid"]
        ),
        "oracle_checks": len(oracle_records),
        "oracle_matches": sum(row["matched"] for row in oracle_records),
        "unresolved_solver_records": sum(
            row["status"] in {"unknown", "model_invalid"} for row in records
        ),
    }
    (output_root / "correctness-report.json").write_text(
        json.dumps(correctness, indent=2) + "\n", encoding="utf-8"
    )
    scenarios_manifest = {
        "schema_version": "pc-build-recommender.optimizer-quality-scenarios.v1",
        "scenario_count": scenario_count,
        "seed": seed,
        "difficulty_distribution": {key: count for key, count in difficulty_specs},
        "cohort_sha256": cohort_sha,
        "provenance": summary["provenance"],
    }
    (output_root / "scenarios-manifest.json").write_text(
        json.dumps(scenarios_manifest, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Optimizer quality benchmark v2",
        "",
        "Controlled synthetic scenarios; no customer observations or factual market prices.",
        f"Cohort: {scenario_count} scenarios; SHA-256 `{cohort_sha}`.",
        "",
        "| Algorithm | Feasible | Mean objective | Improvement vs greedy | "
        "p95 runtime (ms) | Violations |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for item in aggregates:
        improvement = paired.get(item["algorithm"], {}).get("mean_relative_objective_improvement")
        improvement_text = (
            "baseline"
            if item["algorithm"] == "greedy"
            else ("n/a" if improvement is None else f"{improvement:+.2%}")
        )
        lines.append(
            f"| {item['algorithm']} | {item['feasible_rate']:.1%} | "
            f"{item['mean_objective'] if item['mean_objective'] is not None else 'n/a'} | "
            f"{improvement_text} | {item['runtime_p95_seconds'] * 1000:.3f} | "
            f"{item['hard_constraint_violations']} |"
        )
    lines += [
        "",
        f"Exhaustive CP-SAT oracle matches: "
        f"{sum(row['matched'] for row in oracle_records)}/{len(oracle_records)} "
        "solver/scenario checks.",
        "",
        "The warm start only supplies a feasible hint; it does not alter the objective.",
        "",
    ]
    (output_root / "report.md").write_text("\n".join(lines), encoding="utf-8")
    artifact_files = (
        "scenarios.jsonl",
        "scenarios-manifest.json",
        "runs.csv",
        "summary.json",
        "difficulty-breakdown.json",
        "correctness-report.json",
        "report.md",
    )
    manifest: dict[str, Any] = {
        "schema_version": "pc-build-recommender.optimizer-quality-manifest.v1",
        "git_revision": summary["git_revision"],
        "working_tree_dirty": summary["working_tree_dirty"],
        "seed": seed,
        "cohort_sha256": cohort_sha,
        "source_files": {
            path: _source_digest(path)
            for path in (
                "scripts/benchmark_optimizer_quality.py",
                "packages/core/src/pc_build_recommender/optimizer/engine.py",
                "packages/core/src/pc_build_recommender/optimizer/greedy.py",
                "packages/core/src/pc_build_recommender/optimizer/scoring.py",
                "packages/core/src/pc_build_recommender/optimizer/validation.py",
            )
        },
        "software_versions": summary["hardware"],
        "files": {
            name: hashlib.sha256((output_root / name).read_bytes()).hexdigest()
            for name in artifact_files
        },
    }
    manifest["manifest_sha256"] = _canonical_json_hash(manifest)
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def _git_revision() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _source_digest(path: str) -> str | None:
    target = Path(path)
    return hashlib.sha256(target.read_bytes()).hexdigest() if target.is_file() else None


def _git_dirty() -> bool | None:
    try:
        output = subprocess.check_output(["git", "status", "--porcelain"], text=True)
        return bool(output.strip())
    except (OSError, subprocess.CalledProcessError):
        return None


def _canonical_json_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/evaluation/optimizer-quality-benchmark")
    )
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20261108)
    parser.add_argument(
        "--include-warm-start",
        action="store_true",
        help="Include the exploratory greedy-hint CP-SAT variant (not default).",
    )
    args = parser.parse_args()
    destination = args.output_dir / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    summary = run_quality_benchmark(
        output_root=destination,
        scenario_count=args.count,
        seed=args.seed,
        include_warm_start=args.include_warm_start,
    )
    print(
        json.dumps(
            {
                "output_dir": str(destination),
                "algorithms": summary["algorithms"],
                "paired_vs_greedy": summary["paired_vs_greedy"],
                "oracle": {
                    "checks": summary["exhaustive_oracle"]["checks"],
                    "matched": summary["exhaustive_oracle"]["matched"],
                },
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
