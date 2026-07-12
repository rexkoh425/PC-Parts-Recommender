import { describe, expect, it } from "vitest";
import { clampScore, formatScore, formatSgd, humanizeToken } from "../lib/format";

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
});
