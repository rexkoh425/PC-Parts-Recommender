import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import HowItWorksPage from "../app/how-it-works/page";

describe("how it works presentation", () => {
  it("uses clear evidence copy and a library disclosure icon", () => {
    const html = renderToStaticMarkup(createElement(HowItWorksPage));

    expect(html).toContain("What powers the recommendations.");
    expect(html).toContain("Each part checked against the rest");
    expect(html).toContain("search vectors for matching parts");
    expect(html).toContain("lucide-chevron-down");
    expect(html).toContain("Where these numbers come from");
  });
});
