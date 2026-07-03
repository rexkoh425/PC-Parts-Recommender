"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useMemo, useState } from "react";
import { BuildCard } from "./build-card";
import {
  generateBuilds,
  readCachedBuildRequest,
  USING_DEMO_DATA,
} from "@/lib/api";
import { formatScore, formatSgd, profileLabels } from "@/lib/format";
import { useSavedBuilds } from "@/lib/use-saved-builds";
import type { BuildSummary, ComponentCategory } from "@/lib/types";

const maximumComparedBuilds = 3;

function componentName(build: BuildSummary, category: ComponentCategory): string {
  return build.components.find((component) => component.category === category)?.canonical_name ?? "Not reported";
}

export function SavedScreen() {
  const router = useRouter();
  const { entries, savedIds, toggle, ready } = useSavedBuilds();
  const [selectedBuildIds, setSelectedBuildIds] = useState<string[]>([]);
  const [rerunState, setRerunState] = useState<"idle" | "running" | "failed">("idle");
  const selectedEntries = useMemo(
    () => entries.filter((entry) => selectedBuildIds.includes(entry.build.build_id)),
    [entries, selectedBuildIds],
  );
  const selectedIdSet = useMemo(
    () => new Set(selectedEntries.map((entry) => entry.build.build_id)),
    [selectedEntries],
  );
  const rerunRequest =
    !USING_DEMO_DATA && selectedEntries.length === 1 && selectedEntries[0].build.request_id
      ? readCachedBuildRequest(selectedEntries[0].build.request_id)
      : undefined;

  const rerunExplanation = USING_DEMO_DATA
    ? "Current-price reruns are unavailable in the public demo because no live retailer feed is connected. Generate a new controlled-demo shortlist instead."
    : selectedEntries.length === 0
      ? "Select one saved build to check whether its original structured brief is still available in this browser session."
      : selectedEntries.length > 1
        ? "Select exactly one saved build to rerun its original brief."
        : !rerunRequest
          ? "The original brief is no longer available in this browser session. Start a new brief to query current stored observations."
          : "Uses the original structured brief to query current stored observations and re-check every compatibility rule.";

  function toggleComparison(buildId: string) {
    setSelectedBuildIds((current) => {
      const availableIds = new Set(entries.map((entry) => entry.build.build_id));
      const availableSelection = current.filter((candidate) => availableIds.has(candidate));
      if (availableSelection.includes(buildId)) {
        return availableSelection.filter((candidate) => candidate !== buildId);
      }
      if (availableSelection.length >= maximumComparedBuilds) return availableSelection;
      return [...availableSelection, buildId];
    });
  }

  async function rerunSelectedBuild() {
    if (!rerunRequest) return;
    setRerunState("running");
    try {
      const response = await generateBuilds(rerunRequest);
      router.push(`/recommendations/${encodeURIComponent(response.request_id)}`);
    } catch {
      setRerunState("failed");
    }
  }

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

          <section className="saved-comparison" aria-labelledby="saved-comparison-heading">
            <div className="saved-comparison__header">
              <div>
                <p className="eyebrow">Snapshot comparison</p>
                <h2 id="saved-comparison-heading">Compare saved builds</h2>
                <p aria-live="polite">
                  {selectedEntries.length} of {maximumComparedBuilds} selected. Choose at least two
                  saved builds to compare their recorded trade-offs.
                </p>
              </div>
              {selectedEntries.length > 0 && (
                <button
                  className="button button--secondary"
                  type="button"
                  onClick={() => setSelectedBuildIds([])}
                >
                  Clear selection
                </button>
              )}
            </div>

            {selectedEntries.length >= 2 ? (
              <div className="table-scroll" tabIndex={0} aria-label="Scrollable saved build comparison">
                <table className="comparison-table saved-comparison__table" data-testid="saved-build-comparison">
                  <caption className="sr-only">
                    Recorded fields for the selected saved PC build snapshots
                  </caption>
                  <thead>
                    <tr>
                      <th scope="col">Profile</th>
                      <th scope="col">Recorded total</th>
                      <th scope="col">Overall</th>
                      <th scope="col">Value</th>
                      <th scope="col">Upgradeability</th>
                      <th scope="col">Processor</th>
                      <th scope="col">Graphics</th>
                    </tr>
                  </thead>
                  <tbody>
                    {selectedEntries.map(({ build }) => (
                      <tr key={build.build_id}>
                        <th scope="row">
                          <Link href={`/builds/${encodeURIComponent(build.build_id)}`}>
                            {profileLabels[build.profile]}
                          </Link>
                        </th>
                        <td>{formatSgd(build.total_price_sgd)}</td>
                        <td>{formatScore(build.overall_score)}</td>
                        <td>{typeof build.value_score === "number" ? formatScore(build.value_score) : "—"}</td>
                        <td>
                          {typeof build.upgradeability_score === "number"
                            ? formatScore(build.upgradeability_score)
                            : "—"}
                        </td>
                        <td>{componentName(build, "cpu")}</td>
                        <td>{componentName(build, "gpu")}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="saved-comparison__empty" role="status">
                <strong>Select two builds below to reveal the comparison table.</strong>
                <p>Selection never changes or re-ranks the saved generation-time snapshots.</p>
              </div>
            )}

            <div className="saved-refresh" role="note">
              <div>
                <strong>Need current prices?</strong>
                <p id="saved-rerun-explanation">
                  {rerunState === "failed"
                    ? "The rerun could not be completed. The saved snapshot is unchanged; start a fresh brief or try again."
                    : rerunExplanation}
                </p>
              </div>
              <div className="saved-refresh__actions">
                <button
                  className="button button--secondary"
                  type="button"
                  disabled={!rerunRequest || rerunState === "running"}
                  aria-describedby="saved-rerun-explanation"
                  onClick={() => void rerunSelectedBuild()}
                >
                  {rerunState === "running" ? "Re-running…" : "Re-run with current prices"}
                </button>
                <Link className="button button--primary" href="/">
                  Start a fresh brief
                </Link>
              </div>
            </div>
          </section>

          <section className="results-grid" aria-label="Saved PC builds">
            {entries.map((entry) => (
              <div className="saved-build-wrap" key={entry.build.build_id}>
                <div className="saved-build-wrap__controls">
                  <p className="saved-at">
                    Saved {new Date(entry.saved_at).toLocaleDateString("en-SG", { dateStyle: "medium" })}
                  </p>
                  <label>
                    <input
                      type="checkbox"
                      checked={selectedIdSet.has(entry.build.build_id)}
                      disabled={
                        selectedEntries.length >= maximumComparedBuilds &&
                        !selectedIdSet.has(entry.build.build_id)
                      }
                      onChange={() => toggleComparison(entry.build.build_id)}
                      aria-label={`Compare ${profileLabels[entry.build.profile]} saved build`}
                      data-testid="saved-build-select"
                    />
                    Compare
                  </label>
                </div>
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
