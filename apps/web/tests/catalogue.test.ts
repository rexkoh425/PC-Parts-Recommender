import { describe, expect, it } from "vitest";
import {
  confidencePresentation,
  formatAttributeValue,
  formatSignedDelta,
  humanizeAttributeKey,
  observedStockLabel,
  priceObservationPresentation,
  summarizeCompatibilityChecks,
} from "../lib/catalogue";
import { getDemoPrices, searchDemoProducts } from "../lib/demo-api";

describe("catalogue evidence presentation", () => {
  it("describes availability as an observation, never a live claim", () => {
    expect(observedStockLabel("in_stock")).toBe("Observed in stock");
    expect(observedStockLabel("out-of-stock")).toBe("Observed out of stock");
    expect(observedStockLabel("demo_only")).toBe("Demo catalogue");
    expect(observedStockLabel(null)).toBe("Availability not reported");
  });

  it("distinguishes used and ineligible price observations", () => {
    expect(
      priceObservationPresentation({ condition: "used", current_offer_eligible: false }),
    ).toEqual({
      conditionLabel: "Condition: Used",
      eligibilityLabel: "Not eligible as a current offer",
    });
    expect(
      priceObservationPresentation({ condition: "new", current_offer_eligible: true }),
    ).toMatchObject({ eligibilityLabel: "Eligible current offer" });
  });

  it("formats structured attributes without inventing values", () => {
    expect(humanizeAttributeKey("maximum_gpu_length_mm")).toBe("Maximum GPU Length mm");
    expect(formatAttributeValue(true)).toBe("Yes");
    expect(formatAttributeValue(["ATX", "mATX"])).toBe("ATX, mATX");
    expect(formatAttributeValue(null)).toBe("Not reported");
  });

  it("keeps confidence bounded and explicitly labels missing confidence", () => {
    expect(confidencePresentation(0.92)).toMatchObject({ tone: "high", percent: 92 });
    expect(confidencePresentation(0.7)).toMatchObject({ tone: "medium", percent: 70 });
    expect(confidencePresentation(undefined).tone).toBe("unknown");
  });

  it("does not imply a delta where the model returned none", () => {
    expect(formatSignedDelta(undefined, " pts")).toBe("Not modelled");
    expect(formatSignedDelta(3.25, " pts")).toBe("+3.3 pts");
  });

  it("summarises rule outcomes with fail-closed precedence", () => {
    const summary = summarizeCompatibilityChecks([
      { rule_id: "socket", status: "pass", message: "Socket matches." },
      { rule_id: "clearance", status: "warning", message: "Clearance is close." },
      { rule_id: "connector", status: "unknown", message: "Connector is not reported." },
    ]);
    expect(summary).toMatchObject({ total: 3, pass: 1, warning: 1, unknown: 1, overall: "unknown" });
  });

  it("keeps controlled demo pagination and coverage explicit", () => {
    const first = searchDemoProducts({ query: "", limit: 1, page_size: 1, page: 1 });
    const second = searchDemoProducts({ query: "", limit: 1, page_size: 1, page: 2 });

    expect(first.products).toHaveLength(1);
    expect(second.products).toHaveLength(1);
    expect(second.products[0]?.product_id).not.toBe(first.products[0]?.product_id);
    expect(first.pagination).toMatchObject({ page: 1, page_size: 1, has_next: true });
    expect(first.coverage?.scope_label).toBe("Parts in this demo");
    expect(first.coverage?.canonical_products).toBeGreaterThan(1);
  });

  it("returns provider category facets independently of the active category filter", () => {
    const response = searchDemoProducts({ query: "", category: "gpu", limit: 12 });
    const facetCategories = response.facets?.categories?.map((facet) => facet.value);

    expect(response.products.every((product) => product.category === "gpu")).toBe(true);
    expect(facetCategories).toContain("gpu");
    expect(facetCategories).toContain("cpu");
  });

  it("labels the one-point controlled demo summary as insufficient history", () => {
    const productId = searchDemoProducts({ query: "", limit: 1 }).products[0]?.product_id;
    expect(productId).toBeTruthy();

    const summary = getDemoPrices(productId as string).price_intelligence;

    expect(summary).toMatchObject({
      basis: "descriptive_observed_history",
      history_days_90d: 1,
      history_sufficient: false,
      percentile_90d: null,
      volatility_90d_pct: null,
      labels: ["Insufficient price history"],
    });
  });
});
