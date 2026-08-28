import type { Metadata } from "next";
import Link from "next/link";

/*
 * The tab said "BuildSignal - PC builds that fit together" on a page that had
 * failed to find anything, which reads as though the page loaded correctly.
 */
export const metadata: Metadata = {
  title: "Page not found",
  robots: { index: false, follow: true },
};

export default function NotFound() {
  return (
    <main className="shell state-page">
      <div className="state-card">
        <span className="state-card__icon" aria-hidden="true">404</span>
        <p className="eyebrow">Page not found</p>
        {/*
          Was "This route is not in the current build" - developer language on
          the one page a visitor reaches by being lost. "Route" means nothing to
          them, and "build" here meant the deployment, not the PC they came to
          put together.
        */}
        <h1>We could not find that page.</h1>
        <p>It may have moved, or the link may be wrong. Here is the way back.</p>
        <div className="button-row">
          <Link className="button button--primary" href="/">Build a PC</Link>
          <Link className="button button--secondary" href="/catalogue">Browse parts</Link>
          <Link className="button button--secondary" href="/saved">Saved builds</Link>
        </div>
      </div>
    </main>
  );
}
