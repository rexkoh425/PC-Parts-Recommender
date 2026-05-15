import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { PriceIntelligencePanel } from "../components/price-intelligence-panel";
import type { PriceIntelligenceSummary } from "../lib/types";

const summary: PriceIntelligenceSummary = {
  basis: "descriptive_observed_history",
  currency: "SGD",
  as_of: "2026-07-23T00:00:00.000Z",
  current_delivered_price_sgd: 899,
  median_30d_sgd: 929,
  median_90d_sgd: 949,
  percentile_90d: 20,
  recent_low_90d_sgd: 879,
  volatility_90d_pct: 4.2,
  current_seller_count: 3,
  seller_trend: "stable",
  stock_trend: "increasing",
  history_days_30d: 20,
  history_days_90d: 45,
  history_sufficient: true,
  labels: ["Good current value"],
  anomalies: [],
  observations_analyzed: 70,
  analysis_truncated: false,
};

function render(intelligence: PriceIntelligenceSummary): string {
  return renderToStaticMarkup(createElement(PriceIntelligencePanel, { intelligence }));
}

describe("PriceIntelligencePanel", () => {
  it("renders observed statistics with an explicit non-forecast boundary", () => {
    const html = render(summary);

    expect(html).toContain("Descriptive history, not a forecast.");
    expect(html).toContain("It does not guarantee a future or live retailer price.");
    expect(html).toContain("30-day median");
    expect(html).toContain("90-day price position");
    expect(html).toContain("Good current value");
  });

  it("does not invent percentile or volatility for sparse history", () => {
    const html = render({
      ...summary,
      percentile_90d: null,
      volatility_90d_pct: null,
      history_days_90d: 1,
      history_sufficient: false,
      labels: ["Insufficient price history"],
    });

    expect(html.match(/Insufficient history/g)?.length).toBeGreaterThanOrEqual(2);
    expect(html).toContain("Insufficient price history");
  });
});
