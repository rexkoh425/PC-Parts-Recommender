"""Price-history analysis for canonical PC products.

The public API deliberately accepts either :class:`PriceObservation` instances or
plain mappings so ingestion and API layers do not need to depend on an ORM type.
"""

from .intelligence import analyse_product_prices, analyze_product_prices, detect_mad_anomalies
from .models import (
    MarketTrend,
    PriceAnomaly,
    PriceIntelligence,
    PriceLabel,
    PriceObservation,
    StockState,
)

__all__ = [
    "MarketTrend",
    "PriceAnomaly",
    "PriceIntelligence",
    "PriceLabel",
    "PriceObservation",
    "StockState",
    "analyse_product_prices",
    "analyze_product_prices",
    "detect_mad_anomalies",
]
