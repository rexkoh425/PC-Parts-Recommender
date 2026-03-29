"""Deterministic generated-scenario evaluation for the compat_v2 ruleset.

This module evaluates rule behavior on generated configurations.  It deliberately does not
claim that the configurations are observed market builds or that the report is a substitute
for a curated compatibility corpus.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from .engine import DEFAULT_RULE_VERSION, CompatibilityEngine
from .models import CompatibilityReport, CompatVerdict, PowerPolicy

SCENARIO_KINDS: Final = (
    "valid",
    "socket_failure",
    "ddr_failure",
    "gpu_length_failure",
    "gpu_slot_failure",
    "connector_failure",
    "bios_failure",
    "power_failure",
    "resource_failure",
    "missing_data",
)
GENERATED_SCENARIO_LABEL: Final = "deterministically_generated_not_observed_market_builds"

type BuildMapping = dict[str, dict[str, Any]]
type ExpectedOutcome = tuple[str, CompatVerdict]


class CompatibilityEvaluationError(AssertionError):
    """Raised as soon as the engine disagrees with the independent scenario oracle."""


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    """Reproducible evaluation inputs."""

    scenario_count: int = 10_000
    seed: int = 20_260_722
    rule_version: str = DEFAULT_RULE_VERSION

    def __post_init__(self) -> None:
        if self.scenario_count < len(SCENARIO_KINDS):
            raise ValueError(
                f"scenario_count must be at least {len(SCENARIO_KINDS)} to cover every family"
            )
        if self.rule_version != "compat_v2":
            raise ValueError("this frozen harness evaluates compat_v2 only")


@dataclass(frozen=True, slots=True)
class GeneratedCompatibilityEvaluation:
    """Content-addressed aggregate report; no per-scenario records are retained."""

    payload: Mapping[str, Any]
    artifact_sha256: str

    def __post_init__(self) -> None:
        expected = _sha256_json(self.payload)
        if self.artifact_sha256 != expected:
            raise ValueError("artifact_sha256 does not match the evaluation payload")

    def to_dict(self) -> dict[str, Any]:
        return {**self.payload, "artifact_sha256": self.artifact_sha256}

    def verify(self) -> None:
        if _sha256_json(self.payload) != self.artifact_sha256:
            raise ValueError("evaluation payload failed SHA-256 verification")


@dataclass(frozen=True, slots=True)
class _Scenario:
    kind: str
    build: BuildMapping
    expected_nonpass: tuple[ExpectedOutcome, ...]


def _canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_json(payload: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _source_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(64 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _round_up(value: float, step: int) -> int:
    return int(math.ceil(value / step) * step)


def _make_valid_build(rng: random.Random, index: int, policy: PowerPolicy) -> BuildMapping:
    cpu_power = rng.randint(65, 190)
    gpu_power = rng.randint(120, 420)
    gpu_length = rng.randint(190, 345)
    gpu_slots = rng.randint(2, 4)
    connector_count = rng.randint(1, 3)
    cooler_height = rng.randint(120, 165)
    memory_capacity = rng.choice((16, 32, 64, 96))
    motherboard_capacity = rng.choice((128, 192, 256))

    continuous_required = math.ceil(
        (cpu_power + gpu_power + policy.accessory_allowance_w) * (1.0 + policy.headroom_ratio)
    )
    transient_required = math.ceil(
        cpu_power + gpu_power * policy.gpu_transient_multiplier + policy.accessory_allowance_w
    )
    psu_wattage = _round_up(max(continuous_required, transient_required) + rng.randint(50, 250), 50)
    suffix = f"{index:05d}"
    return {
        "cpu": {
            "product_id": f"generated-cpu-{suffix}",
            "category": "cpu",
            "status": "active",
            "socket": "AM5",
            "generation": "Ryzen 7000",
            "model": "Generated CPU",
            "supported_chipsets": ["B650"],
            "peak_power_w": cpu_power,
        },
        "gpu": {
            "product_id": f"generated-gpu-{suffix}",
            "category": "gpu",
            "status": "active",
            "host_interface": "PCIe x16",
            "length_mm": gpu_length,
            "slot_width": gpu_slots,
            "board_power_w": gpu_power,
            "required_power_connectors": {"8-pin PCIe": connector_count},
        },
        "motherboard": {
            "product_id": f"generated-motherboard-{suffix}",
            "category": "motherboard",
            "status": "active",
            "socket": "AM5",
            "chipset": "B650",
            "supported_cpu_generations": ["Ryzen 7000"],
            "memory_type": "DDR5",
            "maximum_memory_gb": motherboard_capacity,
            "memory_slots": 4,
            "form_factor": "ATX",
            "pcie_slots": rng.randint(2, 4),
            "m2_slots": rng.randint(2, 5),
            "sata_ports": rng.randint(4, 8),
        },
        "memory": {
            "product_id": f"generated-memory-{suffix}",
            "category": "memory",
            "status": "active",
            "memory_type": "DDR5",
            "capacity_gb": memory_capacity,
            "module_count": rng.choice((1, 2, 4)),
        },
        "storage": {
            "product_id": f"generated-storage-{suffix}",
            "category": "storage",
            "status": "active",
            "interface": "M.2 NVMe",
            "form_factor": "M.2 2280",
        },
        "power_supply": {
            "product_id": f"generated-psu-{suffix}",
            "category": "power_supply",
            "status": "active",
            "wattage": psu_wattage,
            "form_factor": "ATX",
            "pcie_connectors": {"8-pin PCIe": connector_count + rng.randint(0, 2)},
        },
        "cooler": {
            "product_id": f"generated-cooler-{suffix}",
            "category": "cooler",
            "status": "active",
            "cooler_type": "air",
            "supported_sockets": ["AM5"],
            "height_mm": cooler_height,
        },
        "case": {
            "product_id": f"generated-case-{suffix}",
            "category": "case",
            "status": "active",
            "supported_motherboard_sizes": ["ATX", "Micro-ATX", "Mini-ITX"],
            "maximum_gpu_length_mm": gpu_length + rng.randint(10, 100),
            "maximum_gpu_slot_width": gpu_slots + rng.randint(1, 3),
            "maximum_cooler_height_mm": cooler_height + rng.randint(5, 35),
            "supported_psu_sizes": ["ATX", "SFX"],
            "radiator_support_mm": [120, 240, 280, 360],
        },
    }


def _independent_baseline_oracle(build: BuildMapping, policy: PowerPolicy) -> int:
    """Assert baseline facts without calling any compatibility implementation."""

    cpu = build["cpu"]
    gpu = build["gpu"]
    motherboard = build["motherboard"]
    memory = build["memory"]
    storage = build["storage"]
    power_supply = build["power_supply"]
    cooler = build["cooler"]
    case = build["case"]
    required_connectors = int(gpu["required_power_connectors"]["8-pin PCIe"])
    available_connectors = int(power_supply["pcie_connectors"]["8-pin PCIe"])
    continuous_required = math.ceil(
        (int(cpu["peak_power_w"]) + int(gpu["board_power_w"]) + policy.accessory_allowance_w)
        * (1.0 + policy.headroom_ratio)
    )
    transient_required = math.ceil(
        int(cpu["peak_power_w"])
        + int(gpu["board_power_w"]) * policy.gpu_transient_multiplier
        + policy.accessory_allowance_w
    )
    conditions = (
        cpu["socket"] == motherboard["socket"],
        cpu["generation"] in motherboard["supported_cpu_generations"],
        cpu["supported_chipsets"][0] == motherboard["chipset"],
        memory["memory_type"] == motherboard["memory_type"],
        int(memory["capacity_gb"]) <= int(motherboard["maximum_memory_gb"]),
        int(memory["module_count"]) <= int(motherboard["memory_slots"]),
        motherboard["form_factor"] in case["supported_motherboard_sizes"],
        int(gpu["length_mm"]) <= int(case["maximum_gpu_length_mm"]),
        int(gpu["slot_width"]) <= int(case["maximum_gpu_slot_width"]),
        int(cooler["height_mm"]) <= int(case["maximum_cooler_height_mm"]),
        cpu["socket"] in cooler["supported_sockets"],
        int(motherboard["pcie_slots"]) >= 1,
        storage["interface"] == "M.2 NVMe" and int(motherboard["m2_slots"]) >= 1,
        power_supply["form_factor"] in case["supported_psu_sizes"],
        required_connectors <= available_connectors,
        int(power_supply["wattage"]) >= continuous_required,
        int(power_supply["wattage"]) >= transient_required,
    )
    if not all(conditions):
        raise CompatibilityEvaluationError("generated baseline violated its independent oracle")
    return len(conditions)


def _inject_scenario(
    kind: str,
    baseline: BuildMapping,
    rng: random.Random,
    policy: PowerPolicy,
) -> _Scenario:
    build = deepcopy(baseline)
    expected: tuple[ExpectedOutcome, ...]
    if kind == "valid":
        expected = ()
    elif kind == "socket_failure":
        build["motherboard"]["socket"] = "LGA1700"
        if build["cpu"]["socket"] == build["motherboard"]["socket"]:
            raise CompatibilityEvaluationError("socket injection did not create a mismatch")
        expected = (("compat.cpu_motherboard.socket", CompatVerdict.FAIL),)
    elif kind == "ddr_failure":
        build["memory"]["memory_type"] = "DDR4"
        if build["memory"]["memory_type"] == build["motherboard"]["memory_type"]:
            raise CompatibilityEvaluationError("DDR injection did not create a mismatch")
        expected = (("compat.memory_motherboard.generation", CompatVerdict.FAIL),)
    elif kind == "gpu_length_failure":
        build["case"]["maximum_gpu_length_mm"] = int(build["gpu"]["length_mm"]) - rng.randint(1, 20)
        if int(build["gpu"]["length_mm"]) <= int(build["case"]["maximum_gpu_length_mm"]):
            raise CompatibilityEvaluationError("length injection did not exceed clearance")
        expected = (("compat.gpu_case.length", CompatVerdict.FAIL),)
    elif kind == "gpu_slot_failure":
        build["case"]["maximum_gpu_slot_width"] = int(build["gpu"]["slot_width"]) - 1
        if int(build["gpu"]["slot_width"]) <= int(build["case"]["maximum_gpu_slot_width"]):
            raise CompatibilityEvaluationError("slot injection did not exceed clearance")
        expected = (("compat.gpu_case.slot_width", CompatVerdict.FAIL),)
    elif kind == "connector_failure":
        required = int(build["gpu"]["required_power_connectors"]["8-pin PCIe"])
        build["power_supply"]["pcie_connectors"] = {"8-pin PCIe": required - 1}
        if required <= int(build["power_supply"]["pcie_connectors"]["8-pin PCIe"]):
            raise CompatibilityEvaluationError("connector injection did not create a shortage")
        expected = (("compat.power_supply.gpu_connectors", CompatVerdict.FAIL),)
    elif kind == "bios_failure":
        generation = str(build["cpu"]["generation"])
        build["motherboard"].update(
            minimum_bios_versions={generation: "F20"},
            bios_version="F1",
            bios_update_available=False,
        )
        if not (1 < 20 and build["motherboard"]["bios_update_available"] is False):
            raise CompatibilityEvaluationError("BIOS injection oracle failed")
        expected = (("compat.cpu_motherboard.chipset_bios", CompatVerdict.FAIL),)
    elif kind == "power_failure":
        cpu_power = int(build["cpu"]["peak_power_w"])
        gpu_power = int(build["gpu"]["board_power_w"])
        transient_required = math.ceil(
            cpu_power + gpu_power * policy.gpu_transient_multiplier + policy.accessory_allowance_w
        )
        build["power_supply"]["wattage"] = transient_required - 1
        continuous_required = math.ceil(
            (cpu_power + gpu_power + policy.accessory_allowance_w) * (1.0 + policy.headroom_ratio)
        )
        if not int(build["power_supply"]["wattage"]) < min(transient_required, continuous_required):
            raise CompatibilityEvaluationError("power injection did not violate both policies")
        expected = (
            ("compat.power_supply.capacity", CompatVerdict.FAIL),
            ("compat.power_supply.transient_capacity", CompatVerdict.FAIL),
        )
    elif kind == "resource_failure":
        build["motherboard"]["resource_conflicts"] = [
            {
                "resources": ["gpu_pcie", "storage_m2_nvme"],
                "message": "Generated negative-control resource conflict.",
                "evidence_source": "generated-oracle",
            }
        ]
        expected = (("compat.motherboard.resource_conflicts", CompatVerdict.FAIL),)
    elif kind == "missing_data":
        del build["case"]["maximum_gpu_length_mm"]
        if "maximum_gpu_length_mm" in build["case"]:
            raise CompatibilityEvaluationError("missing-data injection retained its field")
        expected = (("compat.gpu_case.length", CompatVerdict.UNKNOWN),)
    else:
        raise ValueError(f"unsupported scenario kind: {kind}")
    return _Scenario(kind=kind, build=build, expected_nonpass=expected)


def _nonpass_outcomes(report: CompatibilityReport) -> tuple[ExpectedOutcome, ...]:
    return tuple(
        sorted(
            (
                (result.rule_id, result.status)
                for result in report.results
                if result.status is not CompatVerdict.PASS
            ),
            key=lambda item: (item[0], item[1].value),
        )
    )


def _require_rule_status(
    report: CompatibilityReport, rule_id: str, expected: CompatVerdict
) -> None:
    matches = report.by_rule(rule_id)
    if len(matches) != 1 or matches[0].status is not expected:
        actual = [(result.rule_id, result.status.value) for result in matches]
        raise CompatibilityEvaluationError(
            f"expected {rule_id}={expected.value}; observed {actual}"
        )


def _monotonic_checks(
    engine: CompatibilityEngine,
    scenario: _Scenario,
    *,
    occurrence: int,
) -> Counter[str]:
    """Run deterministic invariants without retaining additional scenario records."""

    build = scenario.build
    counts: Counter[str] = Counter()
    if scenario.kind == "gpu_length_failure":
        smaller_case = deepcopy(build["case"])
        smaller_case["maximum_gpu_length_mm"] = max(
            0, int(smaller_case["maximum_gpu_length_mm"]) - 10
        )
        _require_rule_status(
            engine.check_pair("gpu", build["gpu"], "case", smaller_case),
            "compat.gpu_case.length",
            CompatVerdict.FAIL,
        )
        counts["reducing_gpu_clearance_cannot_repair_failure"] += 1
    elif scenario.kind == "gpu_slot_failure":
        smaller_case = deepcopy(build["case"])
        smaller_case["maximum_gpu_slot_width"] = max(
            0.25, float(smaller_case["maximum_gpu_slot_width"]) - 0.25
        )
        _require_rule_status(
            engine.check_pair("gpu", build["gpu"], "case", smaller_case),
            "compat.gpu_case.slot_width",
            CompatVerdict.FAIL,
        )
        counts["reducing_gpu_slot_clearance_cannot_repair_failure"] += 1
    elif scenario.kind == "ddr_failure":
        _require_rule_status(
            engine.check_pair("memory", build["memory"], "motherboard", build["motherboard"]),
            "compat.memory_motherboard.generation",
            CompatVerdict.FAIL,
        )
        counts["ddr4_never_passes_ddr5_only_motherboard"] += 1
    elif scenario.kind == "connector_failure":
        required = int(build["gpu"]["required_power_connectors"]["8-pin PCIe"])
        repaired = deepcopy(build["power_supply"])
        repaired["pcie_connectors"] = {"8-pin PCIe": required}
        expanded = deepcopy(repaired)
        expanded["pcie_connectors"] = {"8-pin PCIe": required + 2}
        for power_supply in (repaired, expanded):
            _require_rule_status(
                engine.check_pair("gpu", build["gpu"], "power_supply", power_supply),
                "compat.power_supply.gpu_connectors",
                CompatVerdict.PASS,
            )
        counts["adding_required_connectors_cannot_create_shortage"] += 1
    elif scenario.kind == "missing_data":
        _require_rule_status(
            engine.check_pair("gpu", build["gpu"], "case", build["case"]),
            "compat.gpu_case.length",
            CompatVerdict.UNKNOWN,
        )
        counts["missing_clearance_never_passes"] += 1

    # Full-build invariants are sampled every tenth occurrence to bound runtime while keeping
    # the sample deterministic and balanced within each scenario family.
    if occurrence % 10 == 0 and scenario.kind in {"valid", "power_failure", "resource_failure"}:
        repaired_build = deepcopy(build)
        if scenario.kind == "power_failure":
            repaired_build["power_supply"]["wattage"] = 2_000
        elif scenario.kind == "resource_failure":
            repaired_build["motherboard"].pop("resource_conflicts", None)
        else:
            repaired_build["power_supply"]["wattage"] = (
                int(repaired_build["power_supply"]["wattage"]) + 500
            )
            repaired_build["case"]["maximum_gpu_length_mm"] = (
                int(repaired_build["case"]["maximum_gpu_length_mm"]) + 100
            )
        repaired_report = engine.check_build(repaired_build)
        if not repaired_report.is_feasible:
            raise CompatibilityEvaluationError(
                f"monotonic repair was unexpectedly infeasible for {scenario.kind}"
            )
        invariant_name = {
            "valid": "increasing_capacity_and_clearance_preserves_feasibility",
            "power_failure": "increasing_psu_wattage_repairs_power_failure",
            "resource_failure": "removing_hard_resource_conflict_restores_feasibility",
        }[scenario.kind]
        counts[invariant_name] += 1
    return counts


def run_generated_evaluation(
    config: EvaluationConfig = EvaluationConfig(),
) -> GeneratedCompatibilityEvaluation:
    """Stream generated configurations through compat_v2 and aggregate exact outcomes."""

    policy = PowerPolicy()
    engine = CompatibilityEngine(rule_version=config.rule_version, power_policy=policy)
    rng = random.Random(config.seed)
    scenario_stream_hash = hashlib.sha256()
    scenario_counts: Counter[str] = Counter()
    overall_status_counts: Counter[str] = Counter()
    observed_nonpass_counts: Counter[str] = Counter()
    expected_nonpass_counts: Counter[str] = Counter()
    monotonic_counts: Counter[str] = Counter()
    family_occurrences: Counter[str] = Counter()
    oracle_assertion_count = 0
    engine_rule_assertion_count = 0
    total_rule_results = 0

    for index in range(config.scenario_count):
        kind = SCENARIO_KINDS[(index + config.seed) % len(SCENARIO_KINDS)]
        baseline = _make_valid_build(rng, index, policy)
        oracle_assertion_count += _independent_baseline_oracle(baseline, policy)
        scenario = _inject_scenario(kind, baseline, rng, policy)
        family_occurrences[kind] += 1
        scenario_counts[kind] += 1
        scenario_stream_hash.update(
            _canonical_json_bytes({"scenario_index": index, "kind": kind, "build": scenario.build})
        )
        scenario_stream_hash.update(b"\n")

        report = engine.check_build(scenario.build)
        total_rule_results += len(report.results)
        if any(result.rule_version != config.rule_version for result in report.results):
            raise CompatibilityEvaluationError(
                f"scenario {index} emitted a result with the wrong rule version"
            )
        engine_rule_assertion_count += len(report.results)
        observed = _nonpass_outcomes(report)
        expected = tuple(
            sorted(scenario.expected_nonpass, key=lambda item: (item[0], item[1].value))
        )
        if observed != expected:
            raise CompatibilityEvaluationError(
                f"scenario {index} ({kind}) expected "
                f"{[(rule, status.value) for rule, status in expected]} but observed "
                f"{[(rule, status.value) for rule, status in observed]}"
            )
        engine_rule_assertion_count += 1
        for rule_id, status in expected:
            expected_nonpass_counts[f"{rule_id}|{status.value}"] += 1
        for rule_id, status in observed:
            observed_nonpass_counts[f"{rule_id}|{status.value}"] += 1
        overall_status_counts[report.status.value] += 1
        monotonic_counts.update(
            _monotonic_checks(
                engine,
                scenario,
                occurrence=family_occurrences[kind],
            )
        )

    if scenario_counts.total() != config.scenario_count:
        raise CompatibilityEvaluationError("scenario counter did not match requested count")
    if expected_nonpass_counts != observed_nonpass_counts:
        raise CompatibilityEvaluationError("aggregate expected and observed outcomes differ")
    if not monotonic_counts or any(count <= 0 for count in monotonic_counts.values()):
        raise CompatibilityEvaluationError("one or more monotonic invariants were not exercised")

    module_path = Path(__file__).resolve()
    engine_path = module_path.with_name("engine.py")
    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "compatibility_generated_scenario_evaluation",
        "evaluation_passed": True,
        "scenario_provenance": GENERATED_SCENARIO_LABEL,
        "market_builds_evaluated": 0,
        "claim_scope": (
            "Engineering validation on deterministically generated configurations; "
            "not evidence of 10,000 observed market builds."
        ),
        "rule_version": config.rule_version,
        "seed": config.seed,
        "scenario_count": config.scenario_count,
        "scenario_kinds": list(SCENARIO_KINDS),
        "scenario_counts": dict(sorted(scenario_counts.items())),
        "scenario_stream_sha256": scenario_stream_hash.hexdigest(),
        "expected_nonpass_outcomes": dict(sorted(expected_nonpass_counts.items())),
        "observed_nonpass_outcomes": dict(sorted(observed_nonpass_counts.items())),
        "overall_status_counts": dict(sorted(overall_status_counts.items())),
        "oracle_mismatch_count": 0,
        "assertions": {
            "independent_oracle": oracle_assertion_count,
            "engine_rule_and_exact_outcome": engine_rule_assertion_count,
            "monotonic": monotonic_counts.total(),
            "total": (
                oracle_assertion_count + engine_rule_assertion_count + monotonic_counts.total()
            ),
            "failed": 0,
        },
        "monotonic_invariants": dict(sorted(monotonic_counts.items())),
        "total_rule_results_checked": total_rule_results,
        "power_policy": {
            "headroom_ratio": policy.headroom_ratio,
            "accessory_allowance_w": policy.accessory_allowance_w,
            "gpu_transient_multiplier": policy.gpu_transient_multiplier,
        },
        "memory_strategy": {
            "mode": "streaming_counters_and_incremental_sha256",
            "retained_scenario_records": 0,
            "writes_per_scenario": 0,
        },
        "source_sha256": {
            "compatibility_engine": _source_sha256(engine_path),
            "evaluation_harness": _source_sha256(module_path),
        },
    }
    return GeneratedCompatibilityEvaluation(payload=payload, artifact_sha256=_sha256_json(payload))


def write_evaluation_report(
    report: GeneratedCompatibilityEvaluation,
    output_dir: str | Path,
) -> Path:
    """Atomically persist a content-addressed report and verify any existing artifact."""

    report.verify()
    destination_dir = Path(output_dir).resolve()
    destination_dir.mkdir(parents=True, exist_ok=True)
    filename = (
        f"compatibility-generated-{report.payload['rule_version']}-"
        f"seed-{report.payload['seed']}-n-{report.payload['scenario_count']}-"
        f"{report.artifact_sha256[:16]}.json"
    )
    destination = destination_dir / filename
    serialised = json.dumps(report.to_dict(), indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    if destination.exists():
        existing = destination.read_text(encoding="utf-8")
        if existing != serialised:
            raise ValueError("existing content-addressed report has different bytes")
        load_evaluation_report(destination)
        return destination
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(serialised, encoding="utf-8", newline="\n")
    temporary.replace(destination)
    load_evaluation_report(destination)
    return destination


def load_evaluation_report(path: str | Path) -> GeneratedCompatibilityEvaluation:
    source = Path(path)
    raw = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("evaluation report must be a JSON object")
    artifact_sha256 = raw.pop("artifact_sha256", None)
    if not isinstance(artifact_sha256, str):
        raise ValueError("evaluation report is missing artifact_sha256")
    report = GeneratedCompatibilityEvaluation(payload=raw, artifact_sha256=artifact_sha256)
    report.verify()
    if artifact_sha256[:16] not in source.name:
        raise ValueError("evaluation filename does not contain its content hash")
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the deterministic compat_v2 generated-scenario evaluation."
    )
    parser.add_argument("--count", type=int, default=10_000, help="Generated scenario count")
    parser.add_argument("--seed", type=int, default=20_260_722, help="Deterministic RNG seed")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/evaluation/compatibility-generated-v2"),
        help="Directory for the content-addressed JSON report",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = EvaluationConfig(scenario_count=args.count, seed=args.seed)
    report = run_generated_evaluation(config)
    output = write_evaluation_report(report, args.output_dir)
    print(
        json.dumps(
            {
                "artifact": str(output),
                "artifact_sha256": report.artifact_sha256,
                "evaluation_passed": report.payload["evaluation_passed"],
                "rule_version": report.payload["rule_version"],
                "scenario_count": report.payload["scenario_count"],
                "scenario_provenance": report.payload["scenario_provenance"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
