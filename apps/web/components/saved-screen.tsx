"use client";

import Link from "next/link";
import { BuildCard } from "./build-card";
import { useSavedBuilds } from "@/lib/use-saved-builds";

export function SavedScreen() {
  const { entries, savedIds, toggle, ready } = useSavedBuilds();

  if (!ready) {
    return (
      <main className="shell saved-page" aria-busy="true">
        <div className="skeleton skeleton--eyebrow" />
        <div className="skeleton skeleton--title" />
        <p className="sr-only" role="status">Loading saved builds.</p>
      </main>
    );
  }

  return (
    <main className="shell saved-page">
      <header className="saved-header">
        <div>
          <p className="eyebrow">Stored on this browser</p>
          <h1>Saved builds</h1>
          <p className="lede">
            Keep a shortlist without creating an account. Prices are snapshots from generation time.
          </p>
        </div>
        <Link className="button button--primary" href="/">
          Generate another build
        </Link>
      </header>

      {entries.length === 0 ? (
        <section className="empty-saved" aria-labelledby="empty-saved-title">
          <div className="empty-saved__graphic" aria-hidden="true">
            <span />
            <span />
            <span />
          </div>
          <p className="eyebrow">Your shortlist is clear</p>
          <h2 id="empty-saved-title">No saved builds yet</h2>
          <p>Generate a recommendation set and save the builds you want to compare later.</p>
          <Link className="button button--primary" href="/">
            Start a build brief
          </Link>
        </section>
      ) : (
        <>
          <div className="saved-note" role="note">
            <span aria-hidden="true">i</span>
            <p>
              <strong>Recheck before buying.</strong> Stock and prices may have changed since these
              builds were saved.
            </p>
          </div>
          <section className="results-grid" aria-label="Saved PC builds">
            {entries.map((entry) => (
              <div className="saved-build-wrap" key={entry.build.build_id}>
                <p className="saved-at">
                  Saved {new Date(entry.saved_at).toLocaleDateString("en-SG", { dateStyle: "medium" })}
                </p>
                <BuildCard
                  build={entry.build}
                  saved={savedIds.has(entry.build.build_id)}
                  onToggleSaved={toggle}
                />
              </div>
            ))}
          </section>
        </>
      )}
    </main>
  );
}
