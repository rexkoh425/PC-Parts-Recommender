import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { NoFeasibleBuild } from "../components/recommendations-screen";
import type { GenerateBuildsResponse } from "../lib/types";

const infeasibleResponse: GenerateBuildsResponse = {
  request_id: "recommendations-presentation",
  status: "infeasible",
  generated_at: "2026-08-26T00:00:00Z",
  data_version: "test-data-v1",
  ranking_model: "test-ranker-v1",
  rule_version: "test-rules-v1",
  solver_version: "test-solver-v1",
  solver_status: "INFEASIBLE",
  builds: [],
  infeasibility: {
    reasons: [{ code: "budget", message: "Nothing in stock meets every requirement at this budget." }],
    suggested_relaxations: [],
  },
};

describe("recommendations presentation", () => {
  it("uses a library icon for each infeasibility reason", () => {
    const html = renderToStaticMarkup(
      createElement(NoFeasibleBuild, {
        response: infeasibleResponse,
        onRelax: () => undefined,
      }),
    );

    expect(html).toContain("reason-list__bullet");
    expect(html).toContain("lucide-circle");
    expect(html).toContain("Requirements check");
  });
});
