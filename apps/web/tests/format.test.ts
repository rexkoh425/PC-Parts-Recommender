import { describe, expect, it } from "vitest";
import {
  clampScore,
  formatFreshnessSummary,
  formatScore,
  formatSgd,
  humanizeToken,
} from "../lib/format";
import type { FreshnessSummary } from "../lib/types";

describe("presentation formatters", () => {
  it("formats Singapore-dollar values", () => {
    expect(formatSgd(2478)).toContain("2,478");
    expect(formatSgd(2478)).toMatch(/\$|SGD/);
  });

  it("does not let score visualisation exceed its range", () => {
    expect(clampScore(-4)).toBe(0);
    expect(clampScore(107)).toBe(100);
    expect(formatScore(undefined)).toBe("—");
  });

  it("turns API tokens into readable labels", () => {
    expect(humanizeToken("gaming_1440p")).toBe("Gaming 1440p");
  });

  it("does not label fresh prices as stale when only the catalogue is stale", () => {
    const freshness: FreshnessSummary = {
      data_version: "catalog-v1",
      status: "stale",
      catalogue_status: "stale",
      price_status: "fresh",
      last_catalog_update: "2026-08-01T00:00:00Z",
      prices_updated_at: "2026-08-15T00:00:00Z",
      stale_after_hours: 24,
      catalogue_stale_after_hours: 168,
      price_stale_after_hours: 24,
      source_count: 2,
      product_count: 3000,
      listing_count: 10000,
      production_ready: false,
      readiness_blockers: ["Catalogue data is stale."],
    };

    expect(formatFreshnessSummary(freshness)).toContain("Catalogue stale");
    expect(formatFreshnessSummary(freshness)).not.toContain("Prices stale");
  });
});
