"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { getFreshness, USING_DEMO_DATA } from "@/lib/api";
import { formatFreshness } from "@/lib/format";
import type { FreshnessSummary } from "@/lib/types";

export function SiteHeader() {
  const [freshness, setFreshness] = useState<FreshnessSummary | null>(null);
  const [offline, setOffline] = useState(false);

  useEffect(() => {
    let active = true;
    const controller = new AbortController();
    getFreshness({ signal: controller.signal })
      .then((result) => {
        if (active) setFreshness(result);
      })
      .catch(() => {
        if (active) setOffline(true);
      });
    return () => {
      active = false;
      controller.abort();
    };
  }, []);

  const updatedAt = freshness?.prices_updated_at ?? freshness?.last_catalog_update;
  const freshnessLabel = USING_DEMO_DATA
    ? "Curated demo data"
    : offline
      ? "Data service offline"
      : freshness?.status === "stale"
        ? `Prices stale · ${formatFreshness(updatedAt)}`
        : freshness?.status === "degraded"
          ? "Coverage degraded"
          : freshness
            ? formatFreshness(updatedAt)
            : "Checking market data";
  const freshnessState = USING_DEMO_DATA
    ? "demo"
    : offline
      ? "offline"
      : freshness?.status ?? "checking";

  return (
    <header className="site-header">
      {USING_DEMO_DATA && (
        <div className="demo-banner" role="status">
          <div className="shell demo-banner__inner">
            <strong>Public portfolio demo</strong>
            <span>Curated sample components and illustrative SGD prices. Live retailer stock is not connected.</span>
          </div>
        </div>
      )}
      <div className="site-header__inner shell">
        <Link className="brand" href="/" aria-label="PC Build Recommender home">
          <span className="brand__mark" aria-hidden="true">
            <span />
            <span />
            <span />
          </span>
          <span>
            <strong>BuildSignal</strong>
            <small>PC Build Recommender</small>
          </span>
        </Link>

        <nav className="header-nav" aria-label="Primary navigation">
          <span className={`freshness-chip freshness-chip--${freshnessState}`}>
            <span className="freshness-chip__dot" aria-hidden="true" />
            {freshnessLabel}
          </span>
          <Link className="header-nav__secondary" href="/#method">Method</Link>
          <Link className="header-nav__secondary header-nav__evidence" href="/#evidence">
            Evidence
          </Link>
          <Link className="header-nav__secondary" href="/#data-status">Data status</Link>
          <Link href="/catalogue">Catalogue</Link>
          <Link className="header-nav__comparison" href="/compare">
            Compare
          </Link>
          <Link href="/saved">Saved builds</Link>
        </nav>
      </div>
    </header>
  );
}
