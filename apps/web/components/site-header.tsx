"use client";

import Link from "next/link";
import { USING_DEMO_DATA } from "@/lib/api";

export function SiteHeader() {
  return (
    <header className="site-header">
      {USING_DEMO_DATA && (
        <div className="demo-banner" role="status">
          <div className="shell demo-banner__inner">
            <strong>Portfolio demo</strong>
            <span>Real parts, real prices from August 2026 — not live retailer stock.</span>
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
          <Link className="header-nav__secondary header-nav__evidence" href="/how-it-works">How it works</Link>

          <Link href="/catalogue">Catalogue</Link>
          <Link className="header-nav__comparison" href="/compare">
            Compare
          </Link>
          <Link href="/saved">Saved</Link>
        </nav>
      </div>
    </header>
  );
}
