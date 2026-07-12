import { describe, expect, it } from "vitest";
import { parseSavedBuilds } from "../lib/saved-builds";

describe("saved build persistence", () => {
  it("returns an empty list for corrupt browser data", () => {
    expect(parseSavedBuilds("not-json")).toEqual([]);
    expect(parseSavedBuilds('{"build":"wrong-shape"}')).toEqual([]);
  });

  it("keeps only structurally valid saved entries", () => {
    const value = JSON.stringify([
      {
        build: { build_id: "build_1", profile: "best_overall" },
        saved_at: "2026-07-22T01:00:00Z",
      },
      { build: {}, saved_at: "2026-07-22T01:00:00Z" },
    ]);

    expect(parseSavedBuilds(value)).toHaveLength(1);
    expect(parseSavedBuilds(value)[0].build.build_id).toBe("build_1");
  });
});
