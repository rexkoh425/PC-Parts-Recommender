"""Leakage-aware metrics and confidence intervals for recommender experiments."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Callable, Hashable, Sequence
from statistics import NormalDist, fmean

from .contracts import DataUseDeclaration, EvaluationResult, MetricEstimate

DEFAULT_BOOTSTRAP_RESAMPLES = 1_000
DEFAULT_CONFIDENCE_LEVEL = 0.95
DEFAULT_RANDOM_SEED = 20260722


def _validate_confidence(confidence_level: float) -> None:
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between zero and one")


def _percentile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("cannot calculate a percentile from no values")
    position = (len(sorted_values) - 1) * probability
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return sorted_values[lower_index]
    fraction = position - lower_index
    return sorted_values[lower_index] * (1.0 - fraction) + sorted_values[upper_index] * fraction


def bootstrap_confidence_interval(
    values: Sequence[float],
    *,
    statistic: Callable[[Sequence[float]], float] = fmean,
    groups: Sequence[Hashable] | None = None,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = DEFAULT_RANDOM_SEED,
) -> tuple[float, float]:
    """Return a deterministic percentile-bootstrap interval.

    When ``groups`` is supplied, whole groups are sampled with replacement. This
    supports family-level or query-level confidence intervals without row leakage.
    """

    _validate_confidence(confidence_level)
    if n_resamples < 2:
        raise ValueError("n_resamples must be at least two")
    numeric_values = [float(value) for value in values]
    if not numeric_values or not all(math.isfinite(value) for value in numeric_values):
        raise ValueError("values must be a non-empty finite sequence")
    if groups is not None and len(groups) != len(numeric_values):
        raise ValueError("groups and values must have the same length")

    if groups is None:
        sampling_units = [[index] for index in range(len(numeric_values))]
    else:
        grouped_indices: dict[Hashable, list[int]] = defaultdict(list)
        for index, group_id in enumerate(groups):
            grouped_indices[group_id].append(index)
        sampling_units = list(grouped_indices.values())

    generator = random.Random(seed)
    estimates: list[float] = []
    for _ in range(n_resamples):
        sampled_indices: list[int] = []
        for _ in range(len(sampling_units)):
            sampled_indices.extend(generator.choice(sampling_units))
        estimate = float(statistic([numeric_values[index] for index in sampled_indices]))
        if math.isfinite(estimate):
            estimates.append(estimate)
    if len(estimates) < 2:
        raise ValueError("bootstrap statistic did not produce enough finite estimates")
    estimates.sort()
    alpha = 1.0 - confidence_level
    return _percentile(estimates, alpha / 2.0), _percentile(estimates, 1.0 - alpha / 2.0)


def wilson_confidence_interval(
    successes: int,
    trials: int,
    *,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
) -> tuple[float, float] | None:
    """Wilson score interval for a binomial proportion."""

    _validate_confidence(confidence_level)
    if trials < 0 or successes < 0 or successes > trials:
        raise ValueError("successes and trials must satisfy 0 <= successes <= trials")
    if trials == 0:
        return None
    probability = successes / trials
    z_score = NormalDist().inv_cdf(1.0 - (1.0 - confidence_level) / 2.0)
    z_squared = z_score * z_score
    denominator = 1.0 + z_squared / trials
    centre = (probability + z_squared / (2.0 * trials)) / denominator
    margin = (
        z_score
        * math.sqrt(
            probability * (1.0 - probability) / trials + z_squared / (4.0 * trials * trials)
        )
        / denominator
    )
    return max(0.0, centre - margin), min(1.0, centre + margin)


def _prepare_indices(
    length: int,
    is_synthetic: Sequence[bool],
    *,
    include_synthetic: bool,
) -> tuple[list[int], DataUseDeclaration]:
    if len(is_synthetic) != length:
        raise ValueError("is_synthetic must contain one explicit flag per row")
    flags = [bool(flag) for flag in is_synthetic]
    declaration = DataUseDeclaration.from_flags(flags, include_synthetic=include_synthetic)
    indices = (
        list(range(length))
        if include_synthetic
        else [index for index, flag in enumerate(flags) if not flag]
    )
    if not indices:
        raise ValueError("no evaluation rows remain after synthetic-data exclusion")
    return indices, declaration


def _validate_equal_lengths(*columns: Sequence[object]) -> int:
    if not columns:
        raise ValueError("at least one input column is required")
    length = len(columns[0])
    if any(len(column) != length for column in columns[1:]):
        raise ValueError("all input columns must have the same length")
    if length == 0:
        raise ValueError("evaluation inputs must not be empty")
    return length


def _bounded_ci(
    point_estimate: float,
    interval: tuple[float, float] | None,
) -> tuple[float | None, float | None]:
    if interval is None:
        return None, None
    return min(point_estimate, interval[0]), max(point_estimate, interval[1])


def _metric_with_interval(
    *,
    name: str,
    value: float,
    sample_count: int,
    interval: tuple[float, float] | None,
    confidence_level: float,
    numerator: int | None = None,
    denominator: int | None = None,
    unit: str = "ratio",
) -> MetricEstimate:
    lower, upper = _bounded_ci(value, interval)
    return MetricEstimate(
        name=name,
        value=value,
        sample_count=sample_count,
        ci_lower=lower,
        ci_upper=upper,
        confidence_level=confidence_level if lower is not None else None,
        numerator=numerator,
        denominator=denominator,
        unit=unit,
    )


def _binary_counts(labels: Sequence[int], predictions: Sequence[bool]) -> tuple[int, int, int]:
    true_positives = sum(
        label == 1 and prediction for label, prediction in zip(labels, predictions, strict=True)
    )
    false_positives = sum(
        label == 0 and prediction for label, prediction in zip(labels, predictions, strict=True)
    )
    false_negatives = sum(
        label == 1 and not prediction for label, prediction in zip(labels, predictions, strict=True)
    )
    return true_positives, false_positives, false_negatives


def _average_precision(labels: Sequence[int], scores: Sequence[float]) -> float:
    positive_count = sum(labels)
    if positive_count == 0:
        return 0.0
    order = sorted(range(len(labels)), key=lambda index: (-scores[index], index))
    true_positives = 0
    precision_sum = 0.0
    for rank, index in enumerate(order, start=1):
        if labels[index] == 1:
            true_positives += 1
            precision_sum += true_positives / rank
    return precision_sum / positive_count


def _bootstrap_paired_rows(
    row_count: int,
    statistic: Callable[[Sequence[int]], float],
    *,
    groups: Sequence[Hashable] | None,
    confidence_level: float,
    n_resamples: int,
    seed: int,
) -> tuple[float, float] | None:
    if groups is not None and len(groups) != row_count:
        raise ValueError("groups must contain one value per evaluated row")
    if groups is None:
        units = [[index] for index in range(row_count)]
    else:
        group_indices: dict[Hashable, list[int]] = defaultdict(list)
        for index, group_id in enumerate(groups):
            group_indices[group_id].append(index)
        units = list(group_indices.values())

    generator = random.Random(seed)
    estimates: list[float] = []
    for _ in range(n_resamples):
        sampled: list[int] = []
        for _ in range(len(units)):
            sampled.extend(generator.choice(units))
        value = statistic(sampled)
        if math.isfinite(value):
            estimates.append(value)
    if len(estimates) < 2:
        return None
    estimates.sort()
    alpha = 1.0 - confidence_level
    return _percentile(estimates, alpha / 2.0), _percentile(estimates, 1.0 - alpha / 2.0)


def evaluate_entity_resolution(
    labels: Sequence[int | bool],
    match_scores: Sequence[float],
    *,
    threshold: float,
    is_synthetic: Sequence[bool],
    groups: Sequence[Hashable] | None = None,
    include_synthetic: bool = False,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = DEFAULT_RANDOM_SEED,
) -> EvaluationResult:
    """Evaluate pair matching while keeping synthetic evidence out of claims."""

    length = _validate_equal_lengths(labels, match_scores, is_synthetic)
    if groups is not None and len(groups) != length:
        raise ValueError("groups must contain one value per row")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between zero and one")
    indices, data_use = _prepare_indices(length, is_synthetic, include_synthetic=include_synthetic)
    filtered_labels = [int(labels[index]) for index in indices]
    if any(label not in (0, 1) for label in filtered_labels):
        raise ValueError("entity-resolution labels must be binary")
    filtered_scores = [float(match_scores[index]) for index in indices]
    if any(not math.isfinite(score) or not 0.0 <= score <= 1.0 for score in filtered_scores):
        raise ValueError("match scores must be finite probabilities")
    filtered_groups = [groups[index] for index in indices] if groups is not None else None
    predictions = [score >= threshold for score in filtered_scores]
    true_positives, false_positives, false_negatives = _binary_counts(filtered_labels, predictions)
    predicted_positives = true_positives + false_positives
    actual_positives = true_positives + false_negatives
    precision = true_positives / predicted_positives if predicted_positives else 0.0
    recall = true_positives / actual_positives if actual_positives else 0.0
    f1_score = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    average_precision = _average_precision(filtered_labels, filtered_scores)
    coverage = predicted_positives / len(filtered_labels)

    def sampled_f1(sampled: Sequence[int]) -> float:
        sampled_labels = [filtered_labels[index] for index in sampled]
        sampled_predictions = [predictions[index] for index in sampled]
        tp_count, fp_count, fn_count = _binary_counts(sampled_labels, sampled_predictions)
        denominator = 2 * tp_count + fp_count + fn_count
        return 2 * tp_count / denominator if denominator else 0.0

    def sampled_average_precision(sampled: Sequence[int]) -> float:
        return _average_precision(
            [filtered_labels[index] for index in sampled],
            [filtered_scores[index] for index in sampled],
        )

    f1_interval = _bootstrap_paired_rows(
        len(indices),
        sampled_f1,
        groups=filtered_groups,
        confidence_level=confidence_level,
        n_resamples=n_resamples,
        seed=seed,
    )
    average_precision_interval = _bootstrap_paired_rows(
        len(indices),
        sampled_average_precision,
        groups=filtered_groups,
        confidence_level=confidence_level,
        n_resamples=n_resamples,
        seed=seed + 1,
    )
    metrics = (
        _metric_with_interval(
            name="entity.precision",
            value=precision,
            sample_count=predicted_positives,
            interval=wilson_confidence_interval(
                true_positives, predicted_positives, confidence_level=confidence_level
            ),
            confidence_level=confidence_level,
            numerator=true_positives,
            denominator=predicted_positives,
        ),
        _metric_with_interval(
            name="entity.recall",
            value=recall,
            sample_count=actual_positives,
            interval=wilson_confidence_interval(
                true_positives, actual_positives, confidence_level=confidence_level
            ),
            confidence_level=confidence_level,
            numerator=true_positives,
            denominator=actual_positives,
        ),
        _metric_with_interval(
            name="entity.f1",
            value=f1_score,
            sample_count=len(filtered_labels),
            interval=f1_interval,
            confidence_level=confidence_level,
        ),
        _metric_with_interval(
            name="entity.average_precision",
            value=average_precision,
            sample_count=len(filtered_labels),
            interval=average_precision_interval,
            confidence_level=confidence_level,
        ),
        _metric_with_interval(
            name="entity.auto_match_coverage",
            value=coverage,
            sample_count=len(filtered_labels),
            interval=wilson_confidence_interval(
                predicted_positives, len(filtered_labels), confidence_level=confidence_level
            ),
            confidence_level=confidence_level,
            numerator=predicted_positives,
            denominator=len(filtered_labels),
        ),
    )
    return EvaluationResult(
        metrics=metrics,
        data_use=data_use,
        metadata={
            "threshold": threshold,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "bootstrap_unit": "group" if groups is not None else "row",
        },
    )


def _mean_absolute_error(actual: Sequence[float], predicted: Sequence[float]) -> float:
    return fmean(abs(left - right) for left, right in zip(actual, predicted, strict=True))


def _root_mean_squared_error(actual: Sequence[float], predicted: Sequence[float]) -> float:
    return math.sqrt(
        fmean((left - right) ** 2 for left, right in zip(actual, predicted, strict=True))
    )


def _r_squared(actual: Sequence[float], predicted: Sequence[float]) -> float:
    actual_mean = fmean(actual)
    total = sum((value - actual_mean) ** 2 for value in actual)
    if total == 0.0:
        return math.nan
    residual = sum((left - right) ** 2 for left, right in zip(actual, predicted, strict=True))
    return 1.0 - residual / total


def _average_ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        average_rank = (cursor + 1 + end) / 2.0
        for ordered_index in order[cursor:end]:
            ranks[ordered_index] = average_rank
        cursor = end
    return ranks


def _pearson(left: Sequence[float], right: Sequence[float]) -> float:
    left_mean = fmean(left)
    right_mean = fmean(right)
    numerator = sum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in zip(left, right, strict=True)
    )
    left_scale = math.sqrt(sum((value - left_mean) ** 2 for value in left))
    right_scale = math.sqrt(sum((value - right_mean) ** 2 for value in right))
    denominator = left_scale * right_scale
    if not denominator:
        return math.nan
    correlation = max(-1.0, min(1.0, numerator / denominator))
    if math.isclose(abs(correlation), 1.0, rel_tol=1e-12, abs_tol=1e-12):
        return math.copysign(1.0, correlation)
    return correlation


def _spearman(actual: Sequence[float], predicted: Sequence[float]) -> float:
    return _pearson(_average_ranks(actual), _average_ranks(predicted))


def evaluate_regression(
    actual: Sequence[float],
    predicted: Sequence[float],
    *,
    is_synthetic: Sequence[bool],
    groups: Sequence[Hashable] | None = None,
    include_synthetic: bool = False,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = DEFAULT_RANDOM_SEED,
) -> EvaluationResult:
    """Evaluate a regressor with row- or group-level paired bootstrap intervals."""

    length = _validate_equal_lengths(actual, predicted, is_synthetic)
    if groups is not None and len(groups) != length:
        raise ValueError("groups must contain one value per row")
    indices, data_use = _prepare_indices(length, is_synthetic, include_synthetic=include_synthetic)
    actual_values = [float(actual[index]) for index in indices]
    predicted_values = [float(predicted[index]) for index in indices]
    if not all(math.isfinite(value) for value in actual_values + predicted_values):
        raise ValueError("regression inputs must be finite")
    if len(actual_values) < 2 or len(set(actual_values)) < 2:
        raise ValueError("regression evaluation requires at least two distinct targets")
    filtered_groups = [groups[index] for index in indices] if groups is not None else None

    statistics: list[tuple[str, Callable[[Sequence[float], Sequence[float]], float], str]] = [
        ("regression.mae", _mean_absolute_error, "target_unit"),
        ("regression.rmse", _root_mean_squared_error, "target_unit"),
        ("regression.r_squared", _r_squared, "ratio"),
        ("regression.spearman", _spearman, "ratio"),
    ]
    if all(value > 0.0 for value in actual_values):
        statistics.append(
            (
                "regression.mape",
                lambda observed, estimate: fmean(
                    abs(left - right) / left for left, right in zip(observed, estimate, strict=True)
                ),
                "ratio",
            )
        )

    metrics: list[MetricEstimate] = []
    omitted_metrics: list[str] = []
    for offset, (name, statistic, unit) in enumerate(statistics):
        point_estimate = statistic(actual_values, predicted_values)
        if not math.isfinite(point_estimate):
            omitted_metrics.append(name)
            continue

        def sampled_statistic(
            sampled: Sequence[int],
            statistic_function: Callable[[Sequence[float], Sequence[float]], float] = statistic,
        ) -> float:
            return statistic_function(
                [actual_values[index] for index in sampled],
                [predicted_values[index] for index in sampled],
            )

        interval = _bootstrap_paired_rows(
            len(indices),
            sampled_statistic,
            groups=filtered_groups,
            confidence_level=confidence_level,
            n_resamples=n_resamples,
            seed=seed + offset,
        )
        metrics.append(
            _metric_with_interval(
                name=name,
                value=point_estimate,
                sample_count=len(indices),
                interval=interval,
                confidence_level=confidence_level,
                unit=unit,
            )
        )
    return EvaluationResult(
        metrics=tuple(metrics),
        data_use=data_use,
        metadata={
            "bootstrap_unit": "group" if groups is not None else "row",
            "omitted_metrics": omitted_metrics,
            "mape_policy": "reported only when every observed target is strictly positive",
        },
    )


def _discounted_cumulative_gain(relevance: Sequence[float], cutoff: int) -> float:
    return float(
        sum(
            (2.0**grade - 1.0) / math.log2(rank + 2.0)
            for rank, grade in enumerate(relevance[:cutoff])
        )
    )


def _query_metrics(
    relevance: Sequence[float],
    scores: Sequence[float],
    *,
    recall_cutoffs: Sequence[int],
    ndcg_cutoff: int,
    reciprocal_rank_cutoff: int,
) -> dict[str, float]:
    order = sorted(range(len(relevance)), key=lambda index: (-scores[index], index))
    ideal_order = sorted(range(len(relevance)), key=lambda index: (-relevance[index], index))
    ranked_relevance = [relevance[index] for index in order]
    ideal_relevance = [relevance[index] for index in ideal_order]
    ideal_dcg = _discounted_cumulative_gain(ideal_relevance, ndcg_cutoff)
    result = {
        f"retrieval.ndcg_at_{ndcg_cutoff}": (
            _discounted_cumulative_gain(ranked_relevance, ndcg_cutoff) / ideal_dcg
            if ideal_dcg
            else 0.0
        )
    }
    relevant_count = sum(grade > 0 for grade in relevance)
    for cutoff in recall_cutoffs:
        retrieved_relevant = sum(grade > 0 for grade in ranked_relevance[:cutoff])
        result[f"retrieval.recall_at_{cutoff}"] = (
            retrieved_relevant / relevant_count if relevant_count else 0.0
        )
    reciprocal_rank = 0.0
    for rank, grade in enumerate(ranked_relevance[:reciprocal_rank_cutoff], start=1):
        if grade > 0:
            reciprocal_rank = 1.0 / rank
            break
    result[f"retrieval.mrr_at_{reciprocal_rank_cutoff}"] = reciprocal_rank
    return result


def _group_query_rows(
    query_ids: Sequence[Hashable],
    relevance: Sequence[float],
    *score_columns: Sequence[float],
) -> list[tuple[Hashable, list[float], list[list[float]]]]:
    query_indices: dict[Hashable, list[int]] = defaultdict(list)
    for index, query_id in enumerate(query_ids):
        query_indices[query_id].append(index)
    result: list[tuple[Hashable, list[float], list[list[float]]]] = []
    for query_id, indices in query_indices.items():
        query_relevance = [relevance[index] for index in indices]
        if not any(grade > 0 for grade in query_relevance):
            continue
        scores = [[column[index] for index in indices] for column in score_columns]
        result.append((query_id, query_relevance, scores))
    return result


def evaluate_retrieval(
    query_ids: Sequence[Hashable],
    relevance: Sequence[int | float],
    scores: Sequence[float],
    *,
    is_synthetic: Sequence[bool],
    recall_cutoffs: Sequence[int] = (20, 50),
    ndcg_cutoff: int = 10,
    reciprocal_rank_cutoff: int = 50,
    include_synthetic: bool = False,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = DEFAULT_RANDOM_SEED,
) -> EvaluationResult:
    """Evaluate one retrieval score using query-level bootstrap intervals."""

    length = _validate_equal_lengths(query_ids, relevance, scores, is_synthetic)
    cutoffs = tuple(dict.fromkeys(int(cutoff) for cutoff in recall_cutoffs))
    if not cutoffs or any(cutoff <= 0 for cutoff in cutoffs):
        raise ValueError("recall cutoffs must be positive")
    if ndcg_cutoff <= 0 or reciprocal_rank_cutoff <= 0:
        raise ValueError("ranking cutoffs must be positive")
    indices, data_use = _prepare_indices(length, is_synthetic, include_synthetic=include_synthetic)
    filtered_query_ids = [query_ids[index] for index in indices]
    filtered_relevance = [float(relevance[index]) for index in indices]
    filtered_scores = [float(scores[index]) for index in indices]
    if any(value < 0.0 or not math.isfinite(value) for value in filtered_relevance):
        raise ValueError("relevance labels must be finite and non-negative")
    if any(not math.isfinite(value) for value in filtered_scores):
        raise ValueError("retrieval scores must be finite")

    query_rows = _group_query_rows(filtered_query_ids, filtered_relevance, filtered_scores)
    if not query_rows:
        raise ValueError("at least one query must contain a relevant judged product")
    per_query: dict[str, list[float]] = defaultdict(list)
    for _, query_relevance, query_score_columns in query_rows:
        query_metric_values = _query_metrics(
            query_relevance,
            query_score_columns[0],
            recall_cutoffs=cutoffs,
            ndcg_cutoff=ndcg_cutoff,
            reciprocal_rank_cutoff=reciprocal_rank_cutoff,
        )
        for name, value in query_metric_values.items():
            per_query[name].append(value)

    metrics: list[MetricEstimate] = []
    for offset, name in enumerate(sorted(per_query)):
        metric_values = per_query[name]
        point_estimate = fmean(metric_values)
        interval = bootstrap_confidence_interval(
            metric_values,
            confidence_level=confidence_level,
            n_resamples=n_resamples,
            seed=seed + offset,
        )
        metrics.append(
            _metric_with_interval(
                name=name,
                value=point_estimate,
                sample_count=len(metric_values),
                interval=interval,
                confidence_level=confidence_level,
            )
        )
    return EvaluationResult(
        metrics=tuple(metrics),
        data_use=data_use,
        metadata={
            "evaluable_query_count": len(query_rows),
            "all_judged_query_count": len(set(filtered_query_ids)),
            "recall_scope": "judged_pool",
            "bootstrap_unit": "query",
        },
    )


def evaluate_ranker_lift(
    query_ids: Sequence[Hashable],
    relevance: Sequence[int | float],
    baseline_scores: Sequence[float],
    candidate_scores: Sequence[float],
    *,
    is_synthetic: Sequence[bool],
    ndcg_cutoff: int = 10,
    include_synthetic: bool = False,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = DEFAULT_RANDOM_SEED,
) -> EvaluationResult:
    """Measure paired LambdaMART lift on an identical frozen candidate set."""

    length = _validate_equal_lengths(
        query_ids, relevance, baseline_scores, candidate_scores, is_synthetic
    )
    if ndcg_cutoff <= 0:
        raise ValueError("ndcg_cutoff must be positive")
    indices, data_use = _prepare_indices(length, is_synthetic, include_synthetic=include_synthetic)
    filtered_query_ids = [query_ids[index] for index in indices]
    filtered_relevance = [float(relevance[index]) for index in indices]
    filtered_baseline = [float(baseline_scores[index]) for index in indices]
    filtered_candidate = [float(candidate_scores[index]) for index in indices]
    if any(value < 0.0 or not math.isfinite(value) for value in filtered_relevance):
        raise ValueError("relevance labels must be finite and non-negative")
    if any(not math.isfinite(value) for value in filtered_baseline + filtered_candidate):
        raise ValueError("ranker scores must be finite")

    query_rows = _group_query_rows(
        filtered_query_ids,
        filtered_relevance,
        filtered_baseline,
        filtered_candidate,
    )
    if not query_rows:
        raise ValueError("at least one query must contain a relevant judged product")
    baseline_ndcg: list[float] = []
    candidate_ndcg: list[float] = []
    for _, query_relevance, score_columns in query_rows:
        baseline_value = _query_metrics(
            query_relevance,
            score_columns[0],
            recall_cutoffs=(ndcg_cutoff,),
            ndcg_cutoff=ndcg_cutoff,
            reciprocal_rank_cutoff=ndcg_cutoff,
        )[f"retrieval.ndcg_at_{ndcg_cutoff}"]
        candidate_value = _query_metrics(
            query_relevance,
            score_columns[1],
            recall_cutoffs=(ndcg_cutoff,),
            ndcg_cutoff=ndcg_cutoff,
            reciprocal_rank_cutoff=ndcg_cutoff,
        )[f"retrieval.ndcg_at_{ndcg_cutoff}"]
        baseline_ndcg.append(baseline_value)
        candidate_ndcg.append(candidate_value)

    baseline_mean = fmean(baseline_ndcg)
    candidate_mean = fmean(candidate_ndcg)
    if baseline_mean <= 0.0:
        raise ValueError("relative ranker lift is undefined when baseline NDCG is zero")
    absolute_lift = candidate_mean - baseline_mean
    relative_lift_percent = absolute_lift / baseline_mean * 100.0
    win_count = sum(
        candidate > baseline
        for candidate, baseline in zip(candidate_ndcg, baseline_ndcg, strict=True)
    )

    generator = random.Random(seed)
    bootstrap_baseline: list[float] = []
    bootstrap_candidate: list[float] = []
    bootstrap_absolute: list[float] = []
    bootstrap_relative: list[float] = []
    for _ in range(n_resamples):
        sampled = [generator.randrange(len(query_rows)) for _ in query_rows]
        sampled_baseline = fmean(baseline_ndcg[index] for index in sampled)
        sampled_candidate = fmean(candidate_ndcg[index] for index in sampled)
        bootstrap_baseline.append(sampled_baseline)
        bootstrap_candidate.append(sampled_candidate)
        bootstrap_absolute.append(sampled_candidate - sampled_baseline)
        if sampled_baseline > 0.0:
            bootstrap_relative.append(
                (sampled_candidate - sampled_baseline) / sampled_baseline * 100.0
            )

    def interval(values: list[float]) -> tuple[float, float]:
        values.sort()
        alpha = 1.0 - confidence_level
        return _percentile(values, alpha / 2.0), _percentile(values, 1.0 - alpha / 2.0)

    metrics = (
        _metric_with_interval(
            name=f"ranker.baseline_ndcg_at_{ndcg_cutoff}",
            value=baseline_mean,
            sample_count=len(query_rows),
            interval=interval(bootstrap_baseline),
            confidence_level=confidence_level,
        ),
        _metric_with_interval(
            name=f"ranker.candidate_ndcg_at_{ndcg_cutoff}",
            value=candidate_mean,
            sample_count=len(query_rows),
            interval=interval(bootstrap_candidate),
            confidence_level=confidence_level,
        ),
        _metric_with_interval(
            name=f"ranker.absolute_ndcg_at_{ndcg_cutoff}_lift",
            value=absolute_lift,
            sample_count=len(query_rows),
            interval=interval(bootstrap_absolute),
            confidence_level=confidence_level,
        ),
        _metric_with_interval(
            name=f"ranker.relative_ndcg_at_{ndcg_cutoff}_lift_percent",
            value=relative_lift_percent,
            sample_count=len(query_rows),
            interval=interval(bootstrap_relative),
            confidence_level=confidence_level,
            unit="percent",
        ),
        _metric_with_interval(
            name="ranker.query_win_rate",
            value=win_count / len(query_rows),
            sample_count=len(query_rows),
            interval=wilson_confidence_interval(
                win_count, len(query_rows), confidence_level=confidence_level
            ),
            confidence_level=confidence_level,
            numerator=win_count,
            denominator=len(query_rows),
        ),
    )
    return EvaluationResult(
        metrics=metrics,
        data_use=data_use,
        metadata={
            "candidate_set_policy": "identical_rows_for_baseline_and_candidate",
            "evaluable_query_count": len(query_rows),
            "bootstrap_unit": "query",
        },
    )
