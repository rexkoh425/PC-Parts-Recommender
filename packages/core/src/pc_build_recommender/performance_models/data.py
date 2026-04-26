"""Dataset construction, validation, and leakage-safe performance splits."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from pc_build_recommender.evaluation.splits import (
    assert_group_disjoint,
    deterministic_group_split,
)

from .contracts import PerformanceModelConfig

SYNTHETIC_GPU_FEATURE_COLUMNS: tuple[str, ...] = (
    "generation_index",
    "compute_units",
    "vram_gb",
    "memory_bandwidth_gbps",
    "boost_clock_mhz",
    "board_power_w",
)

SYNTHETIC_DATASET_PROVENANCE = (
    "deterministic_generator:pc_build_recommender.performance_models."
    "make_synthetic_performance_dataset:v1"
)


def make_synthetic_performance_dataset(
    *,
    n_families: int = 60,
    variants_per_family: int = 4,
    n_generations: int = 6,
    seed: int = 20260722,
    category: str = "gpu",
    workload: str = "gaming_1440p",
) -> pd.DataFrame:
    """Create deterministic development-only data for plumbing and smoke tests.

    The generated rows are intentionally conspicuous: every row carries both an
    ``is_synthetic`` flag and an ``eligible_for_external_claims=False`` marker.
    Training on this frame can exercise the full pipeline, but the resulting
    artifact is always non-promotable.
    """

    if n_families < 15:
        raise ValueError("n_families must be at least 15 for stable grouped evaluation")
    if variants_per_family < 2:
        raise ValueError("variants_per_family must be at least two")
    if n_generations < 3 or n_generations > n_families:
        raise ValueError("n_generations must be between three and n_families")
    if not category or not workload:
        raise ValueError("category and workload must not be empty")

    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for family_index in range(n_families):
        generation_index = family_index % n_generations + 1
        tier = family_index // n_generations + 1
        family_jitter = rng.normal(0.0, 0.6)
        base_compute = 18.0 + 5.5 * tier + 3.2 * generation_index + family_jitter
        base_vram = float((8, 10, 12, 16, 20, 24)[min(tier - 1, 5)])
        base_bandwidth = 115.0 + 7.4 * base_compute + 24.0 * generation_index
        base_clock = 1325.0 + 58.0 * generation_index + 31.0 * tier
        base_power = 90.0 + 4.0 * base_compute + 7.0 * tier
        family_name = f"synthetic-{category}-family-{family_index:03d}"
        generation_name = f"synthetic-generation-{generation_index:02d}"

        for variant_index in range(variants_per_family):
            variant_scale = 0.97 + 0.02 * variant_index
            compute_units = max(8.0, round(base_compute + rng.normal(0.0, 0.3), 3))
            bandwidth = max(
                50.0,
                round(base_bandwidth * variant_scale + rng.normal(0.0, 2.5), 3),
            )
            boost_clock = max(
                800.0,
                round(base_clock * variant_scale + rng.normal(0.0, 5.0), 3),
            )
            board_power = max(
                50.0,
                round(base_power * (0.985 + 0.01 * variant_index), 3),
            )
            vram_gb = base_vram + (4.0 if variant_index == variants_per_family - 1 else 0.0)

            # A non-linear relationship gives the tree model meaningful
            # signal without pretending to reproduce any real benchmark suite.
            signal = (
                0.48 * compute_units
                + 0.014 * bandwidth
                + 0.006 * boost_clock
                + 0.82 * np.sqrt(vram_gb * bandwidth)
                + 0.095 * compute_units * generation_index
                + 13.0 * float(vram_gb >= 16.0)
                + 7.0 * float(bandwidth >= 560.0)
                + 22.0 * float(vram_gb >= 16.0 and generation_index >= 4)
                + 15.0 * float(45.0 <= compute_units < 65.0)
                - 14.0 * float(board_power > 340.0 and generation_index <= 3)
                + 5.5 * np.sin(compute_units / 8.5)
                - 0.014 * max(board_power - 320.0, 0.0) ** 1.22
            )
            target_score = max(1.0, signal + rng.normal(0.0, 0.45))
            rows.append(
                {
                    "product_id": f"synthetic-{category}-{family_index:03d}-{variant_index:02d}",
                    "product_family": family_name,
                    "hardware_generation": generation_name,
                    "category": category,
                    "workload": workload,
                    "generation_index": float(generation_index),
                    "compute_units": compute_units,
                    "vram_gb": vram_gb,
                    "memory_bandwidth_gbps": bandwidth,
                    "boost_clock_mhz": boost_clock,
                    "board_power_w": board_power,
                    "target_score": round(float(target_score), 6),
                    "is_synthetic": True,
                    "eligible_for_external_claims": False,
                    "dataset_role": "development_only_non_promotable",
                    "data_provenance": SYNTHETIC_DATASET_PROVENANCE,
                }
            )

    frame = pd.DataFrame.from_records(rows)
    frame.attrs.update(
        {
            "is_synthetic": True,
            "eligible_for_external_claims": False,
            "dataset_role": "development_only_non_promotable",
            "data_provenance": SYNTHETIC_DATASET_PROVENANCE,
            "seed": seed,
        }
    )
    return frame


def _required_columns(config: PerformanceModelConfig) -> tuple[str, ...]:
    return (
        config.product_id_column,
        config.family_column,
        config.generation_column,
        config.synthetic_column,
        "category",
        "workload",
        config.target_column,
        *config.feature_columns,
    )


def validate_performance_frame(
    frame: pd.DataFrame,
    config: PerformanceModelConfig,
) -> pd.DataFrame:
    """Validate and return a stable-order copy suitable for model fitting."""

    if frame.empty:
        raise ValueError("performance training data must not be empty")
    if frame.columns.duplicated().any():
        duplicates = sorted(set(frame.columns[frame.columns.duplicated()].astype(str)))
        raise ValueError(f"performance data contains duplicate columns: {duplicates}")
    missing = sorted(set(_required_columns(config)).difference(frame.columns))
    if missing:
        raise ValueError(f"performance data is missing required columns: {missing}")

    prepared = frame.copy()
    if prepared[config.product_id_column].isna().any():
        raise ValueError("product identifiers cannot be missing")
    if prepared[config.product_id_column].astype(str).duplicated().any():
        raise ValueError("product identifiers must be unique")
    for column in (config.family_column, config.generation_column):
        if prepared[column].isna().any() or (prepared[column].astype(str).str.len() == 0).any():
            raise ValueError(f"{column} cannot contain missing or empty values")
    for column, expected in (("category", config.category), ("workload", config.workload)):
        actual = set(prepared[column].dropna().astype(str))
        if prepared[column].isna().any() or actual != {expected}:
            raise ValueError(
                f"{column} must contain only the configured value {expected!r}; "
                f"found {sorted(actual)!r}"
            )

    generations_per_family = prepared.groupby(config.family_column, dropna=False)[
        config.generation_column
    ].nunique(dropna=False)
    spanning_families = generations_per_family[generations_per_family != 1]
    if not spanning_families.empty:
        raise ValueError(
            "each product family must belong to exactly one hardware generation; "
            f"violations: {list(spanning_families.index.astype(str))[:5]}"
        )

    synthetic_values = prepared[config.synthetic_column]
    if not pd.api.types.is_bool_dtype(synthetic_values.dtype):
        if not synthetic_values.map(lambda value: isinstance(value, (bool, np.bool_))).all():
            raise TypeError(f"{config.synthetic_column} must contain explicit booleans")
        prepared[config.synthetic_column] = synthetic_values.astype(bool)

    for column in (*config.feature_columns, config.target_column):
        if pd.api.types.is_bool_dtype(prepared[column].dtype):
            raise TypeError(f"model column {column!r} must be numeric and not boolean")
        if not pd.api.types.is_numeric_dtype(prepared[column]):
            raise TypeError(f"model column {column!r} must be numeric")
        numeric = prepared[column].to_numpy(dtype=float)
        if np.isinf(numeric).any():
            raise ValueError(f"model column {column!r} cannot contain infinite values")
    target = prepared[config.target_column].to_numpy(dtype=float)
    if np.isnan(target).any():
        raise ValueError("target values cannot be missing")
    if (target <= 0).any():
        raise ValueError("target values must be positive so MAPE remains meaningful")
    if prepared.loc[:, config.feature_columns].isna().all(axis=0).any():
        empty_features = prepared.loc[:, config.feature_columns].columns[
            prepared.loc[:, config.feature_columns].isna().all(axis=0)
        ]
        raise ValueError(f"feature columns cannot be entirely missing: {list(empty_features)}")
    for feature in config.feature_columns:
        values = prepared[feature].to_numpy(dtype=float)
        missing_fraction = float(np.isnan(values).mean())
        if missing_fraction > config.max_training_missing_fraction:
            raise ValueError(
                f"feature {feature!r} missing fraction {missing_fraction:.4f} exceeds "
                f"{config.max_training_missing_fraction:.4f}"
            )
        unique_count = int(np.unique(values[np.isfinite(values)]).size)
        if unique_count < config.min_feature_unique_values:
            raise ValueError(
                f"feature {feature!r} has {unique_count} finite unique values; "
                f"at least {config.min_feature_unique_values} are required"
            )

    prepared[config.product_id_column] = prepared[config.product_id_column].astype(str)
    prepared[config.family_column] = prepared[config.family_column].astype(str)
    prepared[config.generation_column] = prepared[config.generation_column].astype(str)
    return prepared.sort_values(
        [config.generation_column, config.family_column, config.product_id_column],
        kind="mergesort",
    ).reset_index(drop=True)


def _group_keys(frame: pd.DataFrame, config: PerformanceModelConfig) -> list[str]:
    return [str(value) for value in frame[config.family_column].tolist()]


def split_performance_frame(
    frame: pd.DataFrame,
    config: PerformanceModelConfig,
) -> pd.DataFrame:
    """Split whole product families, stratified by their hardware generation.

    Validation enforces the invariant that one family belongs to exactly one
    generation.  This prevents variants of the same family from crossing splits
    while keeping generation coverage as balanced as the group counts permit.
    """

    prepared = validate_performance_frame(frame, config)
    group_keys = _group_keys(prepared, config)
    family_generation = dict(
        prepared.loc[:, [config.family_column, config.generation_column]]
        .drop_duplicates()
        .astype(str)
        .itertuples(index=False, name=None)
    )
    split = deterministic_group_split(
        group_keys,
        weights=config.split_weights,
        seed=config.split_seed,
        strata=family_generation,
    )
    prepared[config.split_column] = split.row_assignments(group_keys)
    assert_group_disjoint(group_keys, prepared[config.split_column].tolist())
    row_counts = prepared[config.split_column].value_counts().to_dict()
    too_small = {
        split_name: int(row_counts.get(split_name, 0))
        for split_name in config.split_weights
        if int(row_counts.get(split_name, 0)) < 2
    }
    if too_small:
        raise ValueError(
            "each grouped split needs at least two rows for regression metrics; "
            f"too small: {too_small}"
        )
    return prepared


def performance_frame_sha256(
    frame: pd.DataFrame,
    config: PerformanceModelConfig,
) -> str:
    """Hash the semantically relevant, stable-order training frame."""

    columns: Sequence[str] = (
        config.product_id_column,
        config.family_column,
        config.generation_column,
        config.synthetic_column,
        config.target_column,
        *config.feature_columns,
        config.split_column,
    )
    missing = set(columns).difference(frame.columns)
    if missing:
        raise ValueError(f"cannot hash frame with missing columns: {sorted(missing)}")
    digest = hashlib.sha256()
    digest.update(
        json.dumps(list(columns), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    )
    digest.update(b"\n")
    for row in frame.loc[:, columns].itertuples(index=False, name=None):
        normalized: list[object] = []
        for value in row:
            if value is None or pd.isna(value):
                normalized.append(["missing"])
            elif isinstance(value, (bool, np.bool_)):
                normalized.append(["bool", bool(value)])
            elif isinstance(value, (int, np.integer)):
                normalized.append(["int", int(value)])
            elif isinstance(value, (float, np.floating)):
                number = float(value)
                if not np.isfinite(number):
                    raise ValueError("cannot hash a frame containing infinite numeric values")
                normalized.append(["float_hex", number.hex()])
            else:
                normalized.append(["str", str(value)])
        digest.update(
            json.dumps(normalized, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def estimate_peak_training_memory_mb(
    frame: pd.DataFrame,
    config: PerformanceModelConfig,
) -> float:
    """Conservatively estimate host memory before materialising model copies.

    The bound covers the source frame, dense feature copies used by the baseline
    and learner, histogram workspaces, and a fixed runtime allowance.  It is a
    fail-fast budget hook, not a claim of exact peak RSS or GPU VRAM usage.
    """

    row_count = max(1, len(frame))
    feature_count = max(1, len(config.feature_columns))
    frame_bytes = int(frame.memory_usage(index=True, deep=True).sum())
    dense_feature_bytes = row_count * feature_count * 8
    learner_copies = dense_feature_bytes * 12
    histogram_bytes = feature_count * config.gpu_max_bin * config.max_cpu_threads * 32
    runtime_allowance_bytes = 32 * 1024 * 1024
    total_bytes = frame_bytes + learner_copies + histogram_bytes + runtime_allowance_bytes
    return total_bytes / (1024.0 * 1024.0)
