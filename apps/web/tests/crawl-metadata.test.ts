import { describe, expect, it } from "vitest";
import robots from "../app/robots";
import sitemap from "../app/sitemap";

describe("crawl metadata", () => {
  it("keeps session-specific recommendation routes out of search indexes", () => {
    const metadata = robots();

    expect(metadata.rules).toEqual({
      userAgent: "*",
      allow: "/",
      disallow: ["/builds/", "/recommendations/", "/saved", "/share"],
    });
    expect(metadata.sitemap).toMatch(/\/sitemap\.xml$/);
  });

  it("lists only stable public routes in the sitemap", () => {
    const entries = sitemap();

    expect(entries.map((entry) => new URL(entry.url).pathname)).toEqual(["/", "/catalogue"]);
  });
});
