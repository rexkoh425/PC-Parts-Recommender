"""Run the isolated, permanently non-promotable CPU performance v3 experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]

from pc_build_recommender.performance_models import PerformanceModelConfig
from pc_build_recommender.performance_models.v3_diagnostic import (
    CPU_BASE_FEATURES,
    run_v3_diagnostic,
    save_v3_diagnostic,
)
from training._common import print_json, sha256_file
from training.train_performance import (
    _dataset_evidence_from_manifest,
    _explicit_boolean_series,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--dataset-manifest", required=True, type=Path)
    parser.add_argument("--artifact-dir", required=True, type=Path)
    parser.add_argument("--workload", required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    frame = pd.read_csv(args.input)
    for column in ("is_synthetic", "eligible_for_external_claims"):
        if column in frame:
            frame[column] = _explicit_boolean_series(frame[column], name=column)
    config = PerformanceModelConfig(
        category="cpu",
        workload=args.workload,
        feature_columns=CPU_BASE_FEATURES,
        max_cpu_threads=1,
        bootstrap_resamples=max(100, args.bootstrap_resamples),
    )
    evidence = _dataset_evidence_from_manifest(
        args.dataset_manifest,
        input_path=args.input,
        frame=frame,
        config=config,
    )
    result = run_v3_diagnostic(
        frame,
        config,
        dataset_evidence=evidence,
        input_csv_sha256=sha256_file(args.input),
        dataset_manifest_sha256=sha256_file(args.dataset_manifest),
        bootstrap_resamples=args.bootstrap_resamples,
    )
    artifact_path = save_v3_diagnostic(result, args.artifact_dir)
    print_json({"artifact_path": str(artifact_path.resolve()), **result.report})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
