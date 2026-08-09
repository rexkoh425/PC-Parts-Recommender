import { describe, expect, it } from "vitest";
import { oneBasedRank } from "../lib/interactions";
import type { InteractionEvent } from "../lib/types";

describe("interaction event contract", () => {
  it("converts rendered indexes to one-based rank positions", () => {
    expect(oneBasedRank(0)).toBe(1);
    expect(oneBasedRank(4)).toBe(5);
  });

  it("rejects invalid display indexes", () => {
    expect(() => oneBasedRank(-1)).toThrow(RangeError);
    expect(() => oneBasedRank(1.5)).toThrow(RangeError);
  });

  it("supports the complete event and metadata contract", () => {
    const event = {
      event_type: "feedback_submitted",
      session_id: "session-1",
      user_id: "user-1",
      rank_position: oneBasedRank(0),
      rule_version: "compat-v7",
      metadata: { rating: 4, reason: "balanced" },
    } satisfies InteractionEvent;

    expect(event.rank_position).toBe(1);
    expect(event.metadata.rating).toBe(4);
    expect(event.rule_version).toBe("compat-v7");
  });
});
