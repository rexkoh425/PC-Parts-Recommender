import { describe, expect, it } from "vitest";
import {
  categoryInlineLabels,
  categoryLabels,
  categoryPluralLabels,
  clampScore,
  formatFreshnessSummary,
  formatScore,
  formatSgd,
  formatSuggestedRelaxation,
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

  it("turns suggested-relaxation fields and values into readable actions", () => {
    expect(formatSuggestedRelaxation("budget_sgd", 2200)).toBe("Set new-parts budget to S$2,200");
    expect(formatSuggestedRelaxation("requirements.minimum_gpu_vram_gb", 16)).toBe(
      "Set minimum GPU memory to 16 GB",
    );
    expect(formatSuggestedRelaxation("requirements.case_size", "mini_tower")).toBe(
      "Set case size to Mini tower",
    );
    expect(formatSuggestedRelaxation("requirements.wifi_required", false)).toBe(
      "Allow builds without Wi-Fi",
    );
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

describe("category naming", () => {
  it("spells the irregular plurals correctly", () => {
    expect(categoryPluralLabels.gpu).toBe("Graphics cards");
    expect(categoryPluralLabels.memory).toBe("Memory kits");
    expect(categoryPluralLabels.storage).toBe("Storage drives");
    expect(categoryPluralLabels.psu).toBe("Power supplies");
  });

  it("never emits a doubled or mis-stemmed ending", () => {
    for (const plural of Object.values(categoryPluralLabels)) {
      expect(plural).not.toMatch(/ss$/);
      expect(plural).not.toMatch(/[^aeiou]ys$/);
      expect(plural).not.toMatch(/(Storages|Memorys|Graphicss)/);
    }
  });

  it("keeps acronyms capitalised in mid-sentence labels", () => {
    expect(categoryInlineLabels.cooler).toBe("CPU cooler");
    for (const inline of Object.values(categoryInlineLabels)) {
      expect(inline).not.toMatch(/cpu/);
      expect(inline).not.toMatch(/gpu/);
    }
  });

  it("covers exactly the categories that have singular labels", () => {
    expect(Object.keys(categoryPluralLabels).sort()).toEqual(Object.keys(categoryLabels).sort());
    expect(Object.keys(categoryInlineLabels).sort()).toEqual(Object.keys(categoryLabels).sort());
  });
});
