import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { ObservedStock } from "../components/observed-stock";

describe("observed stock presentation", () => {
  it("uses status-specific library icons without hiding the written status", () => {
    const available = renderToStaticMarkup(createElement(ObservedStock, { status: "in_stock" }));
    const lowStock = renderToStaticMarkup(createElement(ObservedStock, { status: "low_stock" }));
    const unknown = renderToStaticMarkup(createElement(ObservedStock, { status: null }));

    expect(available).toContain("Observed in stock");
    expect(available).toContain("lucide-circle-check");
    expect(lowStock).toContain("Observed low stock");
    expect(lowStock).toContain("lucide-circle-alert");
    expect(unknown).toContain("Availability not reported");
    expect(unknown).toContain("lucide-circle");
  });
});
