"""Record non-secret software, hardware and source hashes for a ranking experiment."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import torch

from pc_build_recommender.evaluation.manifest import sha256_file

FILES = (
    "training/llm_judge.py",
    "packages/core/src/pc_build_recommender/ranking/features.py",
    "packages/core/src/pc_build_recommender/ranking/lambdamart.py",
    "packages/core/src/pc_build_recommender/ranking/models.py",
    "scripts/build_llm_judge_queue.py",
    "scripts/audit_llm_query_split.py",
    "scripts/audit_llm_judge_pilot.py",
    "scripts/audit_mixed_llm_judgments.py",
    "scripts/evaluate_llm_pilot_validation.py",
    "scripts/generate_diverse_llm_intents.py",
    "scripts/select_diverse_llm_intents.py",
    "scripts/prepare_mixed_judge_queue.py",
    "scripts/reuse_llm_judgments.py",
    "scripts/combine_mixed_judgments.py",
    "scripts/snapshot_llm_ranking_features.py",
    "scripts/materialize_llm_ranking_features.py",
    "scripts/llm_ranking_experiment.py",
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    versions = {}
    for name in ("numpy", "lightgbm", "sentence-transformers", "torch", "scikit-learn"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    report = {
        "schema_version": "pcbr.llm-ranking-environment.v1",
        "recorded_at_utc": datetime.now(UTC).isoformat(),
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "git_worktree_dirty": bool(
            subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        ),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "software_versions": versions,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "source_sha256": {name: sha256_file(Path(name)) for name in FILES},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
