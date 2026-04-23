"""HTTP-safe projection of descriptive product price history."""

from __future__ import annotations

from collections.abc import Iterable
from heapq import nlargest

from pc_build_recommender.pricing import PriceObservation as HistoricalPriceObservation
from pc_build_recommender.pricing import analyse_product_prices
from services.api.models import PriceHistoryAnomaly, PriceIntelligenceSummary

MAX_PRICE_OBSERVATIONS_ANALYZED = 10_000


def summarize_price_history(
    observations: Iterable[HistoricalPriceObservation],
) -> PriceIntelligenceSummary | None:
    """Summarize the newest bounded observed history without forecasting.

    Callers remain responsible for supplying only evidence whose rights and product
    eligibility permit it to contribute to a market summary.
    """

    newest = nlargest(
        MAX_PRICE_OBSERVATIONS_ANALYZED + 1,
        observations,
        key=lambda item: (item.observed_at, item.listing_id),
    )
    if not newest:
        return None
    truncated = len(newest) > MAX_PRICE_OBSERVATIONS_ANALYZED
    selected = newest[:MAX_PRICE_OBSERVATIONS_ANALYZED]
    result = analyse_product_prices(selected)
    return PriceIntelligenceSummary(
        currency=result.currency,
        as_of=result.as_of,
        current_delivered_price_sgd=(
            float(result.current_delivered_price)
            if result.current_delivered_price is not None
            else None
        ),
        median_30d_sgd=float(result.median_30d) if result.median_30d is not None else None,
        median_90d_sgd=float(result.median_90d) if result.median_90d is not None else None,
        percentile_90d=result.percentile_90d,
        recent_low_90d_sgd=(
            float(result.recent_low_90d) if result.recent_low_90d is not None else None
        ),
        volatility_90d_pct=result.volatility_90d,
        current_seller_count=result.current_seller_count,
        seller_trend=result.seller_trend.value,
        stock_trend=result.stock_trend.value,
        history_days_30d=result.history_days_30d,
        history_days_90d=result.history_days_90d,
        history_sufficient=result.history_sufficient,
        labels=[label.value for label in result.labels],
        anomalies=[
            PriceHistoryAnomaly(
                observed_at=item.observed_at,
                listing_id=item.listing_id,
                delivered_price_sgd=float(item.delivered_price),
                direction=item.direction,
                modified_z_score=item.modified_z_score,
                source_url=item.source_url,
            )
            for item in result.anomalies
        ],
        observations_analyzed=result.source_observation_count,
        analysis_truncated=truncated,
    )
    print("DEBUG", locals())  # noqa
