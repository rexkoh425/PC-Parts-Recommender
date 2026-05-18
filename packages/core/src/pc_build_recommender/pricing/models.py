"""Value objects used by price intelligence."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from typing import Any, cast


def as_money(value: Decimal | int | float | str, *, field_name: str) -> Decimal:
    """Convert a numeric input without inheriting binary floating-point artefacts."""

    try:
        amount = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"{field_name} must be a finite decimal amount") from exc
    if not amount.is_finite():
        raise ValueError(f"{field_name} must be a finite decimal amount")
    return amount


def as_utc(value: datetime) -> datetime:
    """Normalise source timestamps; legacy naive values are interpreted as UTC."""

    if not isinstance(value, datetime):
        raise TypeError("observed_at must be a datetime")
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


class StockStatus(StrEnum):
    """Normalised availability states used by pricing and optimisation."""

    IN_STOCK = "in_stock"
    OUT_OF_STOCK = "out_of_stock"
    PREORDER = "preorder"
    UNKNOWN = "unknown"

    @classmethod
    def parse(cls, value: StockStatus | str) -> StockStatus:
        if isinstance(value, cls):
            return value
        normalised = str(value).strip().casefold().replace("-", "_").replace(" ", "_")
        aliases = {
            "available": cls.IN_STOCK,
            "instock": cls.IN_STOCK,
            "in_stock": cls.IN_STOCK,
            "sold_out": cls.OUT_OF_STOCK,
            "outofstock": cls.OUT_OF_STOCK,
            "out_of_stock": cls.OUT_OF_STOCK,
            "pre_order": cls.PREORDER,
            "preorder": cls.PREORDER,
            "unknown": cls.UNKNOWN,
        }
        return aliases.get(normalised, cls.UNKNOWN)


class MarketTrend(StrEnum):
    """Direction of seller availability or stock availability."""

    INCREASING = "increasing"
    STABLE = "stable"
    DECREASING = "decreasing"
    INSUFFICIENT_HISTORY = "insufficient_history"


class PriceLabel(StrEnum):
    """User-facing, descriptive price labels (never forecasts)."""

    GOOD_CURRENT_VALUE = "Good current value"
    NEAR_RECENT_LOW = "Near recent low"
    TYPICAL_PRICE = "Typical price"
    ABOVE_RECENT_AVERAGE = "Above recent average"
    HIGH_VOLATILITY = "High volatility"
    INSUFFICIENT_HISTORY = "Insufficient price history"
    NO_CURRENT_OFFER = "No current in-stock offer"


@dataclass(frozen=True, slots=True)
class PriceObservation:
    """One source listing snapshot.

    ``base_price + shipping_price`` is the delivered price. Promotions are not
    subtracted unless the adapter has already represented them in ``base_price``;
    this prevents unverified coupon text from changing optimisation inputs.
    """

    listing_id: str
    observed_at: datetime
    base_price: Decimal | int | float | str
    shipping_price: Decimal | int | float | str = Decimal("0")
    stock_status: StockStatus | str = StockStatus.UNKNOWN
    seller_name: str = ""
    retailer: str = ""
    currency: str = "SGD"
    source_url: str | None = None

    def __post_init__(self) -> None:
        listing_id = self.listing_id.strip()
        if not listing_id:
            raise ValueError("listing_id must not be empty")
        base_price = as_money(self.base_price, field_name="base_price")
        shipping_price = as_money(self.shipping_price, field_name="shipping_price")
        if base_price <= 0:
            raise ValueError("base_price must be positive")
        if shipping_price < 0:
            raise ValueError("shipping_price cannot be negative")
        currency = self.currency.strip().upper()
        if not currency:
            raise ValueError("currency must not be empty")
        source_url = self.source_url.strip() if self.source_url else None

        object.__setattr__(self, "listing_id", listing_id)
        object.__setattr__(self, "observed_at", as_utc(self.observed_at))
        object.__setattr__(self, "base_price", base_price)
        object.__setattr__(self, "shipping_price", shipping_price)
        object.__setattr__(self, "stock_status", StockStatus.parse(self.stock_status))
        object.__setattr__(self, "seller_name", self.seller_name.strip())
        object.__setattr__(self, "retailer", self.retailer.strip())
        object.__setattr__(self, "currency", currency)
        object.__setattr__(self, "source_url", source_url)

    @property
    def delivered_price(self) -> Decimal:
        return cast(Decimal, self.base_price) + cast(Decimal, self.shipping_price)

    @property
    def seller_key(self) -> str:
        identity = self.seller_name or self.retailer or self.listing_id
        return identity.casefold()


@dataclass(frozen=True, slots=True)
class PriceAnomaly:
    """A robustly unusual daily market-low observation."""

    observed_at: datetime
    listing_id: str
    delivered_price: Decimal
    direction: str
    modified_z_score: float | None
    source_url: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "observed_at": self.observed_at.isoformat(),
            "listing_id": self.listing_id,
            "delivered_price": str(self.delivered_price),
            "direction": self.direction,
            "modified_z_score": self.modified_z_score,
            "source_url": self.source_url,
        }


@dataclass(frozen=True, slots=True)
class PriceIntelligence:
    """Descriptive market-price summary for one canonical product."""

    currency: str
    as_of: datetime
    current_delivered_price: Decimal | None
    cheapest_in_stock: PriceObservation | None
    median_30d: Decimal | None
    median_90d: Decimal | None
    percentile_90d: float | None
    recent_low_90d: Decimal | None
    volatility_90d: float | None
    current_seller_count: int
    seller_trend: MarketTrend
    stock_trend: MarketTrend
    history_days_30d: int
    history_days_90d: int
    history_sufficient: bool
    labels: tuple[PriceLabel, ...]
    anomalies: tuple[PriceAnomaly, ...]
    source_observation_count: int

    @property
    def current_price(self) -> Decimal | None:
        """Compatibility alias for clients that call delivered price current price."""

        return self.current_delivered_price

    def to_dict(self) -> dict[str, Any]:
        offer = self.cheapest_in_stock
        return {
            "currency": self.currency,
            "as_of": self.as_of.isoformat(),
            "current_delivered_price": (
                str(self.current_delivered_price)
                if self.current_delivered_price is not None
                else None
            ),
            "cheapest_in_stock": (
                {
                    "listing_id": offer.listing_id,
                    "retailer": offer.retailer,
                    "seller_name": offer.seller_name,
                    "delivered_price": str(offer.delivered_price),
                    "observed_at": offer.observed_at.isoformat(),
                    "source_url": offer.source_url,
                }
                if offer is not None
                else None
            ),
            "median_30d": str(self.median_30d) if self.median_30d is not None else None,
            "median_90d": str(self.median_90d) if self.median_90d is not None else None,
            "percentile_90d": self.percentile_90d,
            "recent_low_90d": (
                str(self.recent_low_90d) if self.recent_low_90d is not None else None
            ),
            "volatility_90d": self.volatility_90d,
            "current_seller_count": self.current_seller_count,
            "seller_trend": self.seller_trend.value,
            "stock_trend": self.stock_trend.value,
            "history_days_30d": self.history_days_30d,
            "history_days_90d": self.history_days_90d,
            "history_sufficient": self.history_sufficient,
            "labels": [label.value for label in self.labels],
            "anomalies": [anomaly.to_dict() for anomaly in self.anomalies],
            "source_observation_count": self.source_observation_count,
        }
