import { describe, expect, it } from "vitest";
import {
  DEFAULT_API_REQUEST_TIMEOUT_MS,
  parseApiRequestTimeout,
  resolveApiBaseUrl,
  runtimeCapabilitiesForDataMode,
} from "../lib/runtime";
import { resolveSiteUrl } from "../lib/site-url";

describe("runtime production boundaries", () => {
  it("bounds a configured API timeout", () => {
    expect(parseApiRequestTimeout(undefined)).toBe(DEFAULT_API_REQUEST_TIMEOUT_MS);
    expect(parseApiRequestTimeout("not-a-number")).toBe(DEFAULT_API_REQUEST_TIMEOUT_MS);
    expect(parseApiRequestTimeout("20")).toBe(1_000);
    expect(parseApiRequestTimeout("120000")).toBe(60_000);
    expect(parseApiRequestTimeout("12500.4")).toBe(12_500);
  });

  it("enables unlocked replacement only for the catalogue-backed API runtime", () => {
    expect(runtimeCapabilitiesForDataMode("api").reoptimizeUnlockedReplacement).toBe(true);
    expect(runtimeCapabilitiesForDataMode("demo").reoptimizeUnlockedReplacement).toBe(false);
    expect(runtimeCapabilitiesForDataMode("unexpected").reoptimizeUnlockedReplacement).toBe(false);
  });

  it("supports a same-origin API for a single-domain deployment", () => {
    expect(resolveApiBaseUrl("/")).toBe("");
    expect(resolveApiBaseUrl("https://api.example.test/")).toBe("https://api.example.test");
    expect(resolveApiBaseUrl(undefined)).toBe("http://localhost:8000");
  });

  it("normalises an explicit public site origin", () => {
    expect(resolveSiteUrl("https://example.test/path?q=1#fragment", "production").toString()).toBe(
      "https://example.test/",
    );
  });

  it("rejects invalid and insecure production metadata origins", () => {
    expect(resolveSiteUrl("javascript:alert(1)", "production").protocol).toBe("https:");
    expect(resolveSiteUrl("http://example.test", "production").protocol).toBe("https:");
  });
});
