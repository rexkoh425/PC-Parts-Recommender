from __future__ import annotations

import pandas as pd

from pc_build_recommender.performance_models import PerformanceModelConfig
from pc_build_recommender.performance_models.v3_diagnostic import (
    CPU_BASE_FEATURES,
    _outer_split,
    fit_cpu_feature_contract,
    transform_cpu_features,
    v3_candidate_grid,
)


def _cpu_rows() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for family_index in range(18):
        for variant in range(2):
            rows.append(
                {
                    "product_id": f"cpu-{family_index}-{variant}",
                    "product_family": f"family-{family_index}",
                    "hardware_generation": f"generation-{family_index % 3}",
                    "category": "cpu",
                    "workload": "cpu-workload",
                    "core_count": float(4 + family_index),
                    "thread_count": float(8 + family_index * 2),
                    "base_clock_ghz": 3.0 + variant * 0.1,
                    "boost_clock_ghz": 4.0 + variant * 0.1,
                    "tdp_watts": float(65 + family_index),
                    "target_score": float(50 + family_index * 3 + variant),
                    "is_synthetic": False,
                    "eligible_for_external_claims": True,
                }
            )
    return pd.DataFrame(rows)


def _config() -> PerformanceModelConfig:
    return PerformanceModelConfig(
        category="cpu",
        workload="cpu-workload",
        feature_columns=CPU_BASE_FEATURES,
        bootstrap_resamples=100,
    )


def test_v3_feature_engineering_is_target_independent_and_handles_unknown_generation() -> None:
    frame = _cpu_rows()
    contract = fit_cpu_feature_contract(
        frame.loc[frame["hardware_generation"] == "generation-0"]
    )
    first = transform_cpu_features(frame, contract, engineered=True)
    changed_target = frame.copy()
    changed_target["target_score"] *= 1000.0
    second = transform_cpu_features(changed_target, contract, engineered=True)

    pd.testing.assert_frame_equal(first, second)
    assert first.loc[20:, "generation__unknown"].eq(1.0).any()
    assert contract.to_dict()["target_columns_used"] == []


def test_v3_outer_holdout_is_deterministic_and_family_disjoint() -> None:
    frame = _cpu_rows()
    first = _outer_split(frame, _config())
    second = _outer_split(frame.sample(frac=1.0, random_state=4), _config())

    first_map = first.set_index("product_id")["v3_split"].to_dict()
    second_map = second.set_index("product_id")["v3_split"].to_dict()
    assert first_map == second_map
    assert first.groupby("product_family")["v3_split"].nunique().eq(1).all()
    assert set(first["v3_split"]) == {"development", "calibration", "holdout"}


def test_v3_grid_has_one_fixed_baseline_and_log_engineered_candidates() -> None:
    grid = v3_candidate_grid()

    assert sum(candidate.candidate_id == "v2_like_base_identity" for candidate in grid) == 1
    assert any(
        candidate.engineered_features and candidate.target_transform == "log"
        for candidate in grid
    )
