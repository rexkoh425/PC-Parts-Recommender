import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { CategoryIcon } from "../components/category-icon";
import { componentCategories, type ComponentCategory } from "../lib/types";

const iconClasses: Record<ComponentCategory, string> = {
  cpu: "lucide-cpu",
  gpu: "lucide-gpu",
  motherboard: "lucide-circuit-board",
  memory: "lucide-memory-stick",
  storage: "lucide-hard-drive",
  psu: "lucide-plug-zap",
  cooler: "lucide-fan",
  case: "lucide-computer",
};

describe("category icons", () => {
  it.each(componentCategories)("renders the matching library icon for %s", (category) => {
    const html = renderToStaticMarkup(createElement(CategoryIcon, { category }));

    expect(html).toContain(iconClasses[category]);
    expect(html).toContain('aria-hidden="true"');
    expect(html).toContain('focusable="false"');
  });
});
