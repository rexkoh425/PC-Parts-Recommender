"""Generate deterministic *synthetic* smoke-test datasets.

These rows exercise the training stack before licensed, source-provenanced data is
available.  They are explicitly marked synthetic and must never be used for promotion,
portfolio, or resume metrics.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pc_build_recommender.entity_resolution import synthetic_pairs
from pc_build_recommender.performance_models import make_synthetic_performance_dataset
from training._common import sha256_file, write_json, write_json_lines


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/starter"))
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--entity-products", type=int, default=48)
    parser.add_argument("--performance-families", type=int, default=60)
    parser.add_argument("--variants-per-family", type=int, default=4)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.entity_products < 4:
        raise ValueError("--entity-products must be at least four")
    if args.performance_families < 15:
        raise ValueError("--performance-families must be at least 15")
    if args.variants_per_family < 2:
        raise ValueError("--variants-per-family must be at least two")

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = synthetic_pairs(seed=args.seed, product_count=args.entity_products)
    entity_path = output_dir / "entity_resolution.synthetic.jsonl"
    entity_count = write_json_lines(entity_path, (pair.to_dict() for pair in pairs))

    performance_frame = make_synthetic_performance_dataset(
        n_families=args.performance_families,
        variants_per_family=args.variants_per_family,
        seed=args.seed,
    )
    performance_path = output_dir / "performance_gpu.synthetic.csv"
    performance_frame.to_csv(performance_path, index=False, lineterminator="\n")

    manifest = {
        "seed": args.seed,
        "synthetic": True,
        "eligible_for_external_claims": False,
        "promotion_block_reason": (
            "deterministic starter data is synthetic and exists only for pipeline smoke tests"
        ),
        "datasets": {
            "entity_resolution": {
                "path": entity_path.name,
                "rows": entity_count,
                "sha256": sha256_file(entity_path),
            },
            "performance_gpu": {
                "path": performance_path.name,
                "rows": len(performance_frame),
                "sha256": sha256_file(performance_path),
            },
        },
    }
    write_json(output_dir / "manifest.synthetic.json", manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
