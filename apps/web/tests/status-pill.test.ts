import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { StatusPill } from "../components/status-pill";
import type { CompatibilityStatus } from "../lib/types";

const cases: Array<[CompatibilityStatus, string, string]> = [
  ["pass", "Compatible", "lucide-check"],
  ["warning", "Compatible with notes", "lucide-triangle-alert"],
  ["unknown", "Needs verification", "lucide-search"],
  ["fail", "Not compatible", "lucide-x"],
];

describe("compatibility status pills", () => {
  it.each(cases)("uses an icon for %s status", (status, label, iconClass) => {
    const html = renderToStaticMarkup(createElement(StatusPill, { status }));

    expect(html).toContain(label);
    expect(html).toContain(iconClass);
    expect(html).not.toMatch(/[✓!×?]/);
  });
});
