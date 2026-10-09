import { describe, expect, it } from "vitest";
import { relativeDecisionLabel } from "../components/build-detail-screen";
import type { PerformanceDecision, PerformanceSignal } from "../lib/types";

function signal(decision: PerformanceDecision): PerformanceSignal {
  return {
    workload: "gaming",
    metric: "Example score",
    value: 80,
    basis: "relative",
    decision,
  };
}

describe("performance evidence labels", () => {
  it("keeps model-governance caveats clear without exposing internal terminology", () => {
    expect(relativeDecisionLabel(signal("model_not_promotion_eligible")))
      .toBe("Relative score · estimate pending review");
    expect(relativeDecisionLabel(signal("input_outside_training_contract")))
      .toBe("Relative score · outside tested range");
    expect(relativeDecisionLabel(signal("model_not_promotion_eligible_and_input_outside_training_contract")))
      .toBe("Relative score · estimate pending review; outside tested range");
    expect(relativeDecisionLabel(signal("precise_predictions_disabled")))
      .toBe("Relative score · detailed estimate unavailable");
    expect(relativeDecisionLabel(signal("precise_predictions_disabled_and_input_outside_training_contract")))
      .toBe("Relative score · detailed estimate unavailable; outside tested range");
  });
});
