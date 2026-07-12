import type { MetadataRoute } from "next";
import { siteUrl } from "../lib/site-url";

export default function sitemap(): MetadataRoute.Sitemap {
  return [
    {
      url: new URL("/", siteUrl).toString(),
      changeFrequency: "monthly",
      priority: 1,
    },
    {
      url: new URL("/catalogue", siteUrl).toString(),
      changeFrequency: "weekly",
      priority: 0.8,
    },
  ];
}
