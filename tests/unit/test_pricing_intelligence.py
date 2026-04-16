from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from pc_build_recommender.pricing import (
    MarketTrend,
    PriceLabel,
    PriceObservation,
    analyse_product_prices,
    detect_mad_anomalies,
)

AS_OF = datetime(2026, 7, 22, 12, tzinfo=UTC)


def observation(
    day: int,
    price: int | str,
    *,
    listing: str = "listing-a",
    shipping: int | str = 0,
    stock: str = "in_stock",
    seller: str = "seller-a",
    currency: str = "SGD",
) -> PriceObservation:
    return PriceObservation(
        listing_id=listing,
        observed_at=AS_OF - timedelta(days=day),
        base_price=price,
        shipping_price=shipping,
        stock_status=stock,
        seller_name=seller,
        retailer=seller,
        currency=currency,
        source_url=f"https://example.test/{listing}/{day}",
    )


def test_delivered_price_and_cheapest_current_offer_include_shipping() -> None:
    observations = [
        observation(0, 100, shipping=20, listing="a", seller="shop-a"),
        observation(0, 110, shipping=0, listing="b", seller="shop-b"),
    ]

    result = analyse_product_prices(observations, as_of=AS_OF, minimum_history_days=1)

    assert result.current_delivered_price == Decimal("110")
    assert result.cheapest_in_stock is not None
    assert result.cheapest_in_stock.listing_id == "b"
    assert result.current_seller_count == 2


def test_daily_low_prevents_frequently_polled_retailer_from_skewing_median() -> None:
    records = [observation(day, 100 + day) for day in range(7)]
    records.extend(
        PriceObservation(
            listing_id=f"expensive-{hour}",
            observed_at=AS_OF - timedelta(days=1, hours=hour),
            base_price=999,
            stock_status="in_stock",
            seller_name=f"seller-{hour}",
        )
        for hour in range(10)
    )

    result = analyse_product_prices(records, as_of=AS_OF)

    assert result.history_days_30d == 7
    assert result.median_30d == Decimal("103")


def test_sparse_history_is_explicit_and_does_not_report_percentile_or_volatility() -> None:
    result = analyse_product_prices([observation(0, 100), observation(1, 110)], as_of=AS_OF)

    assert not result.history_sufficient
    assert result.percentile_90d is None
    assert result.volatility_90d is None
    assert result.labels == (PriceLabel.INSUFFICIENT_HISTORY,)
    assert result.median_30d == Decimal("105")


def test_current_value_recent_low_and_volatility_labels_are_descriptive() -> None:
    prices = [50, 100, 100, 100, 100, 100, 100, 100]
    result = analyse_product_prices(
        [observation(day, price) for day, price in enumerate(prices)],
        as_of=AS_OF,
    )

    assert result.recent_low_90d == Decimal("50")
    assert result.percentile_90d == pytest.approx(6.25)
    assert PriceLabel.NEAR_RECENT_LOW in result.labels
    assert PriceLabel.GOOD_CURRENT_VALUE in result.labels
    assert PriceLabel.HIGH_VOLATILITY in result.labels


def test_latest_out_of_stock_snapshot_removes_listing_from_current_offers() -> None:
    records = [
        observation(1, 100, listing="a", stock="in_stock"),
        observation(0, 100, listing="a", stock="out_of_stock"),
        observation(0, 120, listing="b", stock="in_stock", seller="seller-b"),
    ]

    result = analyse_product_prices(records, as_of=AS_OF, minimum_history_days=1)

    assert result.current_delivered_price == Decimal("120")
    assert result.cheapest_in_stock is not None
    assert result.cheapest_in_stock.listing_id == "b"


def test_stale_offer_is_not_presented_as_current() -> None:
    result = analyse_product_prices([observation(10, 100)], as_of=AS_OF)

    assert result.current_delivered_price is None
    assert result.labels == (PriceLabel.NO_CURRENT_OFFER,)


def test_mad_flags_large_high_and_low_deviations_and_serialises_zero_mad() -> None:
    high = [observation(day, price) for day, price in enumerate([100, 100, 100, 1000])]
    anomaly = detect_mad_anomalies(high)

    assert len(anomaly) == 1
    assert anomaly[0].direction == "high"
    assert anomaly[0].modified_z_score is None
    assert anomaly[0].to_dict()["modified_z_score"] is None


def test_seller_and_stock_trends_compare_two_observed_seven_day_windows() -> None:
    records: list[PriceObservation] = []
    for day in range(7, 14):
        records.append(observation(day, 100, listing="a", seller="seller-a"))
        records.append(
            observation(day, 110, listing="b", seller="seller-b", stock="out_of_stock")
        )
    for day in range(0, 7):
        records.append(observation(day, 100, listing="a", seller="seller-a"))
        records.append(observation(day, 110, listing="b", seller="seller-b"))

    result = analyse_product_prices(records, as_of=AS_OF)

    assert result.seller_trend is MarketTrend.INCREASING
    assert result.stock_trend is MarketTrend.INCREASING


def test_mapping_inputs_are_supported_and_mixed_currency_is_rejected() -> None:
    mapped = {
        "listing_id": "mapped",
        "observed_at": AS_OF,
        "base_price": "123.45",
        "shipping_price": "4.55",
        "stock_status": "available",
    }
    result = analyse_product_prices([mapped], as_of=AS_OF, minimum_history_days=1)
    assert result.current_delivered_price == Decimal("128.00")

    with pytest.raises(ValueError, match="single currency"):
        analyse_product_prices(
            [observation(0, 100), observation(0, 100, listing="usd", currency="USD")],
            as_of=AS_OF,
        )


def test_analysis_defaults_to_latest_observation_for_reproducibility() -> None:
    records = [observation(20, 90), observation(10, 100)]
    result = analyse_product_prices(records)
    assert result.as_of == AS_OF - timedelta(days=10)
    assert result.current_delivered_price == Decimal("100")
