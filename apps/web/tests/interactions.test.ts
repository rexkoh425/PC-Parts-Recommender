import { afterEach, describe, expect, it, vi } from "vitest";
import { trackInteraction } from "../lib/api";
import { oneBasedRank } from "../lib/interactions";
import type { InteractionEvent } from "../lib/types";

afterEach(() => {
  vi.unstubAllGlobals();
});

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

  it("sends a stable idempotency key only for server-issued impressions", async () => {
    const fetchMock = vi.fn<typeof fetch>(
      async () =>
        new Response(
          JSON.stringify({
            event_id: "evt-1",
            accepted_at: "2026-08-15T10:00:00Z",
            status: "accepted",
            data_version: "catalog-v1",
            rule_version: "compat-v1",
            trust_level: "verified_impression",
            replayed: false,
          }),
          { status: 202, headers: { "Content-Type": "application/json" } },
        ),
    );
    vi.stubGlobal("fetch", fetchMock);
    const event = {
      event_type: "build_saved",
      session_id: "session-1",
      impression_token: "imp_v1.opaque-token",
    } satisfies InteractionEvent;

    await trackInteraction(event);
    await trackInteraction(event);
    await trackInteraction({ event_type: "search_submitted", session_id: "session-1" });

    const firstHeaders = fetchMock.mock.calls[0]?.[1]?.headers as Record<string, string>;
    const secondHeaders = fetchMock.mock.calls[1]?.[1]?.headers as Record<string, string>;
    const unsignedHeaders = fetchMock.mock.calls[2]?.[1]?.headers as
      | Record<string, string>
      | undefined;
    expect(firstHeaders["Idempotency-Key"]).toMatch(/^interaction-/);
    expect(secondHeaders["Idempotency-Key"]).toBe(firstHeaders["Idempotency-Key"]);
    expect(unsignedHeaders?.["Idempotency-Key"]).toBeUndefined();
    expect(JSON.parse(String(fetchMock.mock.calls[0]?.[1]?.body))).toMatchObject({
      impression_token: "imp_v1.opaque-token",
    });
  });
});
