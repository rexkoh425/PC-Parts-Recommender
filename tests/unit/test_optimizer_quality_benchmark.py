from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

_SCRIPT = Path(__file__).parents[2] / "scripts" / "benchmark_optimizer_quality.py"
_SPEC = importlib.util.spec_from_file_location("benchmark_optimizer_quality", _SCRIPT)
assert _SPEC and _SPEC.loader
benchmark = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(benchmark)


def test_greedy_baselines_are_non_cp_sat_and_valid() -> None:
    problem = benchmark.make_problem(12, candidates_per_category=2, difficulty="easy")
    for algorithm in ("greedy", "greedy-local", "cp-sat", "cp-sat-warm"):
        result, _ = benchmark._solve(algorithm, problem)
        if result.solutions:
            assert not benchmark.validate_selected_build(problem, result.solutions[0].selected)
        if algorithm.startswith("cp-sat"):
            assert result.solutions


def test_warm_start_preserves_optimal_objective() -> None:
    problem = benchmark.make_problem(15, candidates_per_category=2, difficulty="easy")
    baseline, _ = benchmark._solve("cp-sat", problem)
    warm, _ = benchmark._solve("cp-sat-warm", problem)
    assert baseline.solutions and warm.solutions
    assert baseline.solutions[0].objective_value == warm.solutions[0].objective_value


def test_quality_benchmark_writes_synthetic_evidence(tmp_path: Path) -> None:
    summary = benchmark.run_quality_benchmark(
        output_root=tmp_path / "quality", scenario_count=4, seed=9
    )
    assert summary["provenance"].startswith("deterministic synthetic")
    assert len(summary["algorithms"]) == 3
    assert (tmp_path / "quality" / "runs.csv").is_file()
    assert (tmp_path / "quality" / "scenarios.jsonl").is_file()
    assert (tmp_path / "quality" / "summary.json").is_file()
    assert (tmp_path / "quality" / "report.md").is_file()
    manifest = json.loads((tmp_path / "quality" / "manifest.json").read_text())
    manifest_hash = manifest.pop("manifest_sha256")
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    assert hashlib.sha256(canonical).hexdigest() == manifest_hash
    for filename, expected_hash in manifest["files"].items():
        assert (
            hashlib.sha256((tmp_path / "quality" / filename).read_bytes()).hexdigest()
            == expected_hash
        )
