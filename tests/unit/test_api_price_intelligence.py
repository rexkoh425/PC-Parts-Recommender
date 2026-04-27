from datetime import UTC, datetime, timedelta

import pytest
from services.api.models import PriceIntelligenceSummary
from services.api.pricing import summarize_price_history

from pc_build_recommender.pricing import PriceObservation

AS_OF = datetime(2026, 7, 23, 12, tzinfo=UTC)


def observed(day: int, price: int) -> PriceObservation:
    return PriceObservation(
        listing_id=f"listing-{day}",
        observed_at=AS_OF - timedelta(days=day),
        base_price=price,
        stock_status="in_stock",
        seller_name=f"seller-{day}",
        currency="SGD",
        source_url=f"https://example.test/{day}",
    )


def test_price_summary_projects_descriptive_statistics_and_labels() -> None:
    summary = summarize_price_history(
        [observed(day, price) for day, price in enumerate([50, 100, 100, 100, 100, 100, 100, 100])]
    )

    assert summary is not None
    assert summary.basis == "descriptive_observed_history"
    assert summary.current_delivered_price_sgd == 50
    assert summary.median_30d_sgd == 100
    assert summary.median_90d_sgd == 100
    assert summary.percentile_90d == pytest.approx(6.25)
    assert summary.recent_low_90d_sgd == 50
    assert summary.volatility_90d_pct is not None
    assert summary.history_sufficient is True
    assert "Near recent low" in summary.labels
    assert summary.observations_analyzed == 8
    assert summary.analysis_truncated is False


def test_sparse_summary_withholds_percentile_and_volatility() -> None:
    summary = summarize_price_history([observed(0, 100)])

    assert summary is not None
    assert summary.history_sufficient is False
    assert summary.percentile_90d is None
    assert summary.volatility_90d_pct is None
    assert summary.labels == ["Insufficient price history"]


def test_summary_is_bounded_to_newest_observations(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("services.api.pricing.MAX_PRICE_OBSERVATIONS_ANALYZED", 3)

    summary = summarize_price_history([observed(day, 100 + day) for day in range(4)])

    assert summary is not None
    assert summary.observations_analyzed == 3
    assert summary.analysis_truncated is True
    assert summary.as_of == AS_OF


def test_empty_history_returns_no_summary() -> None:
    assert summarize_price_history([]) is None


def test_api_contract_rejects_precision_statistics_for_sparse_history() -> None:
    with pytest.raises(ValueError, match="insufficient price history"):
        PriceIntelligenceSummary(
            currency="SGD",
            as_of=AS_OF,
            current_delivered_price_sgd=100,
            median_30d_sgd=100,
            median_90d_sgd=100,
            percentile_90d=50,
            recent_low_90d_sgd=100,
            volatility_90d_pct=None,
            current_seller_count=1,
            seller_trend="insufficient_history",
            stock_trend="insufficient_history",
            history_days_30d=1,
            history_days_90d=1,
            history_sufficient=False,
            observations_analyzed=1,
        )
