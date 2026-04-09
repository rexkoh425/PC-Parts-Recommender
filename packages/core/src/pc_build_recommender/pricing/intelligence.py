"""Deterministic descriptive price intelligence.

Historical statistics use one observation per UTC day: the cheapest delivered,
in-stock offer. This avoids overweighting retailers that are polled more often and
matches the price a user could actually have paid on that day.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from datetime import date, datetime, timedelta
from decimal import Decimal
from statistics import fmean, median, pstdev
from typing import Any

from .models import (
    MarketTrend,
    PriceAnomaly,
    PriceIntelligence,
    PriceLabel,
    PriceObservation,
    StockState,
    as_utc,
)

PriceInput = PriceObservation | Mapping[str, Any]


def _coerce_observation(value: PriceInput) -> PriceObservation:
    if isinstance(value, PriceObservation):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("price observations must be PriceObservation instances or mappings")
    required = ("listing_id", "observed_at", "base_price")
    missing = [name for name in required if name not in value]
    if missing:
        raise ValueError(f"price observation is missing fields: {', '.join(missing)}")
    return PriceObservation(
        listing_id=str(value["listing_id"]),
        observed_at=value["observed_at"],
        base_price=value["base_price"],
        shipping_price=value.get("shipping_price", 0),
        stock_status=value.get("stock_status", StockState.UNKNOWN),
        seller_name=str(value.get("seller_name", "")),
        retailer=str(value.get("retailer", "")),
        currency=str(value.get("currency", "SGD")),
        source_url=str(value["source_url"]) if value.get("source_url") else None,
    )


def _daily_lowest_in_stock(
    observations: Sequence[PriceObservation],
) -> tuple[PriceObservation, ...]:
    daily: dict[date, PriceObservation] = {}
    for observation in observations:
        if observation.stock_status is not StockState.IN_STOCK:
            continue
        day = observation.observed_at.date()
        incumbent = daily.get(day)
        key = (observation.delivered_price, observation.listing_id)
        if incumbent is None or key < (incumbent.delivered_price, incumbent.listing_id):
            daily[day] = observation
    return tuple(daily[day] for day in sorted(daily))


def _within_days(
    observations: Sequence[PriceObservation],
    *,
    as_of: datetime,
    days: int,
) -> tuple[PriceObservation, ...]:
    threshold = as_of - timedelta(days=days)
    return tuple(item for item in observations if threshold <= item.observed_at <= as_of)


def _decimal_median(values: Sequence[Decimal]) -> Decimal | None:
    return median(values) if values else None


def _empirical_percentile(current: Decimal, values: Sequence[Decimal]) -> float | None:
    if not values:
        return None
    lower = sum(value < current for value in values)
    equal = sum(value == current for value in values)
    return round(100.0 * (lower + 0.5 * equal) / len(values), 2)


def _coefficient_of_variation(values: Sequence[Decimal]) -> float | None:
    if len(values) < 2:
        return None
    centre = float(median(values))
    if centre == 0:
        return None
    return round(100.0 * pstdev(float(value) for value in values) / centre, 2)


def detect_mad_anomalies(
    observations: Iterable[PriceInput],
    *,
    threshold: float = 3.5,
) -> tuple[PriceAnomaly, ...]:
    """Flag unusual daily market prices using a median absolute deviation score.

    If MAD is zero, any value that differs from the median is flagged without a
    numeric score. Returning ``None`` rather than infinity keeps API output valid JSON.
    """

    if threshold <= 0:
        raise ValueError("threshold must be positive")
    records = tuple(_coerce_observation(item) for item in observations)
    daily = _daily_lowest_in_stock(records)
    if len(daily) < 3:
        return ()
    values = [item.delivered_price for item in daily]
    centre = median(values)
    mad = median([abs(value - centre) for value in values])
    anomalies: list[PriceAnomaly] = []
    for item in daily:
        difference = item.delivered_price - centre
        if mad == 0:
            is_anomaly = difference != 0
            score = None
        else:
            score = round(float(Decimal("0.6745") * difference / mad), 4)
            is_anomaly = abs(score) > threshold
        if is_anomaly:
            anomalies.append(
                PriceAnomaly(
                    observed_at=item.observed_at,
                    listing_id=item.listing_id,
                    delivered_price=item.delivered_price,
                    direction="high" if difference > 0 else "low",
                    modified_z_score=score,
                    source_url=item.source_url,
                )
            )
    return tuple(anomalies)


def _daily_market_state(
    observations: Sequence[PriceObservation],
) -> dict[date, tuple[int, float]]:
    """Return daily in-stock seller count and listing availability ratio."""

    by_day_listing: dict[date, dict[str, PriceObservation]] = defaultdict(dict)
    for item in observations:
        day = item.observed_at.date()
        incumbent = by_day_listing[day].get(item.listing_id)
        if incumbent is None or item.observed_at > incumbent.observed_at:
            by_day_listing[day][item.listing_id] = item

    result: dict[date, tuple[int, float]] = {}
    for day, listing_records in by_day_listing.items():
        records = tuple(listing_records.values())
        in_stock = tuple(item for item in records if item.stock_status is StockState.IN_STOCK)
        sellers = {item.seller_key for item in in_stock}
        result[day] = (len(sellers), len(in_stock) / len(records))
    return result


def _direction(
    previous: Sequence[float],
    recent: Sequence[float],
    *,
    absolute_threshold: float,
    relative_threshold: float = 0.1,
) -> MarketTrend:
    if len(previous) < 2 or len(recent) < 2:
        return MarketTrend.INSUFFICIENT_HISTORY
    previous_average = fmean(previous)
    difference = fmean(recent) - previous_average
    threshold = max(absolute_threshold, abs(previous_average) * relative_threshold)
    if difference > threshold:
        return MarketTrend.INCREASING
    if difference < -threshold:
        return MarketTrend.DECREASING
    return MarketTrend.STABLE


def _market_trends(
    observations: Sequence[PriceObservation],
    *,
    as_of: datetime,
) -> tuple[MarketTrend, MarketTrend]:
    state = _daily_market_state(observations)
    recent_start = (as_of - timedelta(days=7)).date()
    previous_start = (as_of - timedelta(days=14)).date()
    as_of_date = as_of.date()
    previous = [value for day, value in state.items() if previous_start <= day < recent_start]
    recent = [value for day, value in state.items() if recent_start <= day <= as_of_date]
    seller_trend = _direction(
        [float(value[0]) for value in previous],
        [float(value[0]) for value in recent],
        absolute_threshold=0.5,
    )
    stock_trend = _direction(
        [value[1] for value in previous],
        [value[1] for value in recent],
        absolute_threshold=0.1,
        relative_threshold=0.1,
    )
    return seller_trend, stock_trend


def _price_labels(
    *,
    current: Decimal | None,
    median_30d: Decimal | None,
    recent_low: Decimal | None,
    percentile: float | None,
    volatility: float | None,
    sufficient: bool,
    high_volatility_threshold: float,
) -> tuple[PriceLabel, ...]:
    if current is None:
        return (PriceLabel.NO_CURRENT_OFFER,)
    if not sufficient:
        return (PriceLabel.INSUFFICIENT_HISTORY,)

    labels: list[PriceLabel] = []
    if recent_low is not None and current <= recent_low * Decimal("1.03"):
        labels.append(PriceLabel.NEAR_RECENT_LOW)
    if percentile is not None and percentile <= 25:
        labels.append(PriceLabel.GOOD_CURRENT_VALUE)
    if median_30d is not None and current > median_30d * Decimal("1.10"):
        labels.append(PriceLabel.ABOVE_RECENT_AVERAGE)
    if not labels:
        labels.append(PriceLabel.TYPICAL_PRICE)
    if volatility is not None and volatility >= high_volatility_threshold:
        labels.append(PriceLabel.HIGH_VOLATILITY)
    return tuple(labels)


def analyse_product_prices(
    observations: Iterable[PriceInput],
    *,
    as_of: datetime | None = None,
    current_max_age: timedelta | None = timedelta(hours=48),
    minimum_history_days: int = 7,
    anomaly_threshold: float = 3.5,
    high_volatility_threshold: float = 10.0,
) -> PriceIntelligence:
    """Compute delivered-price, history, availability, trend, and anomaly features.

    ``as_of`` defaults to the latest supplied observation, making offline evaluation
    reproducible. Callers serving live traffic should pass the request's data timestamp.
    """

    records = tuple(_coerce_observation(item) for item in observations)
    if not records:
        raise ValueError("at least one price observation is required")
    if minimum_history_days < 1:
        raise ValueError("minimum_history_days must be positive")
    if current_max_age is not None and current_max_age < timedelta(0):
        raise ValueError("current_max_age cannot be negative")
    if high_volatility_threshold <= 0:
        raise ValueError("high_volatility_threshold must be positive")

    currencies = {item.currency for item in records}
    if len(currencies) != 1:
        raise ValueError("price statistics require a single currency; convert before analysis")
    currency = next(iter(currencies))
    analysis_time = max(item.observed_at for item in records) if as_of is None else as_utc(as_of)
    eligible = tuple(item for item in records if item.observed_at <= analysis_time)
    if not eligible:
        raise ValueError("no observations exist on or before as_of")

    latest_by_listing: dict[str, PriceObservation] = {}
    for item in eligible:
        incumbent = latest_by_listing.get(item.listing_id)
        if incumbent is None or item.observed_at > incumbent.observed_at:
            latest_by_listing[item.listing_id] = item
    freshness_cutoff = (
        analysis_time - current_max_age if current_max_age is not None else None
    )
    current_offers = tuple(
        item
        for item in latest_by_listing.values()
        if item.stock_status is StockState.IN_STOCK
        and (freshness_cutoff is None or item.observed_at >= freshness_cutoff)
    )
    cheapest = min(
        current_offers,
        key=lambda item: (item.delivered_price, item.listing_id),
        default=None,
    )
    current = cheapest.delivered_price if cheapest is not None else None

    daily = _daily_lowest_in_stock(eligible)
    daily_30 = _within_days(daily, as_of=analysis_time, days=30)
    daily_90 = _within_days(daily, as_of=analysis_time, days=90)
    values_30 = tuple(item.delivered_price for item in daily_30)
    values_90 = tuple(item.delivered_price for item in daily_90)
    median_30d = _decimal_median(values_30)
    median_90d = _decimal_median(values_90)
    recent_low = min(values_90, default=None)
    sufficient = len(values_90) >= minimum_history_days
    percentile = _empirical_percentile(current, values_90) if current and sufficient else None
    volatility = _coefficient_of_variation(values_90) if sufficient else None
    current_sellers = {item.seller_key for item in current_offers}
    trend_records = _within_days(eligible, as_of=analysis_time, days=14)
    seller_trend, stock_trend = _market_trends(trend_records, as_of=analysis_time)
    anomalies = detect_mad_anomalies(daily_90, threshold=anomaly_threshold)
    labels = _price_labels(
        current=current,
        median_30d=median_30d,
        recent_low=recent_low,
        percentile=percentile,
        volatility=volatility,
        sufficient=sufficient,
        high_volatility_threshold=high_volatility_threshold,
    )

    return PriceIntelligence(
        currency=currency,
        as_of=analysis_time,
        current_delivered_price=current,
        cheapest_in_stock=cheapest,
        median_30d=median_30d,
        median_90d=median_90d,
        percentile_90d=percentile,
        recent_low_90d=recent_low,
        volatility_90d=volatility,
        current_seller_count=len(current_sellers),
        seller_trend=seller_trend,
        stock_trend=stock_trend,
        history_days_30d=len(daily_30),
        history_days_90d=len(daily_90),
        history_sufficient=sufficient,
        labels=labels,
        anomalies=anomalies,
        source_observation_count=len(eligible),
    )


analyze_product_prices = analyse_product_prices
