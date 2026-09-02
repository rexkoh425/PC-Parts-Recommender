"use client";

import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { useEffect, useMemo, useState } from "react";
import { getBuildShare } from "@/lib/api";
import { categoryLabels, formatFreshness, formatScore, formatSgd, profileLabels } from "@/lib/format";
import { decodeSharedBuild, sharedSnapshotFromApi, type SharedBuildSnapshot } from "@/lib/shared-build";
import { StatusPill } from "./status-pill";

export function SharedBuildScreen() {
  const searchParams = useSearchParams();
  const localSnapshot = useMemo(() => decodeSharedBuild(searchParams.get("build")), [searchParams]);
  const shareId = searchParams.get("share");
  const shareIdIsValid = Boolean(shareId && /^[A-Za-z0-9_-]{1,80}$/.test(shareId));
  const [serverSnapshot, setServerSnapshot] = useState<SharedBuildSnapshot | null>(null);
  const [serverError, setServerError] = useState("");

  useEffect(() => {
    if (!shareIdIsValid || !shareId) return;
    let active = true;
    const controller = new AbortController();
    getBuildShare(shareId, { signal: controller.signal })
      .then((response) => {
        if (active) setServerSnapshot(sharedSnapshotFromApi(response.snapshot));
      })
      .catch((error) => {
        if (active && !controller.signal.aborted) {
          setServerError(error instanceof Error ? error.message : "The shared build could not be loaded.");
        }
      });
    return () => {
      active = false;
      controller.abort();
    };
  }, [shareId, shareIdIsValid]);

  const snapshot = shareId ? serverSnapshot : localSnapshot;
  const verifiedServerRecord = Boolean(shareId && serverSnapshot);
  const effectiveServerError = shareId && !shareIdIsValid
    ? "The public share identifier is malformed."
    : serverError;

  if (shareIdIsValid && !serverSnapshot && !serverError) {
    return (
      <main className="shell state-page shared-page" aria-busy="true">
        <p>Loading shared build snapshot…</p>
      </main>
    );
  }

  if (!snapshot) {
    return (
      <main className="shell state-page shared-page">
        <div className="state-card" role="alert">
          <span className="state-card__icon" aria-hidden="true">?</span>
          <p className="eyebrow">Shared snapshot unavailable</p>
          <h1>This build link is invalid or incomplete.</h1>
          <p>{effectiveServerError || "A shared link carries the build itself and nothing else - no account, no checkout, no live prices."}</p>
          <Link className="button button--primary" href="/">
            Generate a build
          </Link>
        </div>
      </main>
    );
  }

  return (
    <main className="shell detail-page shared-page">
      <nav className="breadcrumbs" aria-label="Breadcrumb">
        <Link href="/">BuildSignal</Link>
        <span aria-hidden="true">/</span>
        <span aria-current="page">Shared build</span>
      </nav>

      <header className="detail-header">
        <div>
          <p className="eyebrow">{verifiedServerRecord ? "Verified build" : "Unverified build"}</p>
          <h1>{profileLabels[snapshot.profile]}</h1>
          <p className="lede">
            {verifiedServerRecord
              ? "This is a server-recorded recommendation snapshot from generation time. Re-run the build before buying to check the current catalogue, stock, and prices."
              : "This link contains editable browser-provided data. Its prices, scores, and claimed compatibility are unverified; generate a new build before relying on it."}
          </p>
        </div>
        <StatusPill
          status={verifiedServerRecord ? snapshot.compatibility_status : "unknown"}
          label={verifiedServerRecord ? undefined : "Unverified build"}
        />
      </header>

      <div className="detail-layout">
        <div className="detail-main">
          <section className="detail-section" aria-labelledby="shared-components-heading">
            <div className="section-heading">
              <div>
                <p className="eyebrow">Frozen component set</p>
                <h2 id="shared-components-heading">Selected components</h2>
              </div>
              <p>{snapshot.components.length} required categories</p>
            </div>
            <div className="component-table-wrap">
              <table className="component-table">
                <thead>
                  <tr><th>Category</th><th>Product</th><th>Selection basis</th><th>Price when saved</th></tr>
                </thead>
                <tbody>
                  {snapshot.components.map((component) => (
                    <tr key={component.category}>
                      <th scope="row" data-label="Category">{categoryLabels[component.category]}</th>
                      <td data-label="Product">
                        <strong>{component.canonical_name}</strong>
                        {component.brand && <span>{component.brand}</span>}
                      </td>
                      <td data-label="Selection basis">{component.selection_reason ?? "Not included in the shared link"}</td>
                      <td data-label="Price when saved"><strong>{component.price_sgd === null ? "Not included" : formatSgd(component.price_sgd)}</strong></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>

          <section className="detail-section" aria-labelledby="shared-performance-heading">
            <div className="section-heading">
              <div><p className="eyebrow">Generation-time scoring</p><h2 id="shared-performance-heading">Workload fit</h2></div>
            </div>
            <div className="performance-grid">
              {Object.entries(snapshot.workload_scores).map(([workload, score]) => (
                <article className="metric-card" key={workload}>
                  <span>{workload.replaceAll("_", " ")}</span>
                  <strong>{score === null ? "Not reported" : formatScore(score)}</strong>
                </article>
              ))}
            </div>
          </section>

          {(snapshot.explanations.length > 0 || snapshot.warnings.length > 0) && (
            <section className="detail-section" aria-labelledby="shared-notes-heading">
              <div className="section-heading"><div><p className="eyebrow">Snapshot notes</p><h2 id="shared-notes-heading">Why it was selected</h2></div></div>
              {snapshot.explanations.length > 0 && (
                <ul className="shared-note-list">
                  {snapshot.explanations.map((item) => <li key={item}>{item}</li>)}
                </ul>
              )}
              {snapshot.warnings.length > 0 && (
                <div className="shared-warning" role="note">
                  <strong>Warnings retained from generation</strong>
                  <ul>{snapshot.warnings.map((item) => <li key={item}>{item}</li>)}</ul>
                </div>
              )}
            </section>
          )}
        </div>

        <aside className="build-summary-card" aria-label="Shared build summary">
          <p className="profile-kicker">{profileLabels[snapshot.profile]}</p>
          <div className="build-summary-card__price"><small>Total for new parts</small><strong>{formatSgd(snapshot.total_price_sgd)}</strong></div>
          <div className="summary-score"><strong>{formatScore(snapshot.overall_score)}</strong><span>Relative fit score</span></div>
          <dl>
            {typeof snapshot.value_score === "number" && <div><dt>Value</dt><dd>{formatScore(snapshot.value_score)}</dd></div>}
            {typeof snapshot.upgradeability_score === "number" && <div><dt>Upgradeability</dt><dd>{formatScore(snapshot.upgradeability_score)}</dd></div>}
            {typeof snapshot.estimated_peak_power_w === "number" && <div><dt>Estimated peak</dt><dd>{Math.round(snapshot.estimated_peak_power_w)} W</dd></div>}
          </dl>
          <StatusPill
            status={verifiedServerRecord ? snapshot.compatibility_status : "unknown"}
            label={verifiedServerRecord ? undefined : "Compatibility claim unverified"}
          />
          <Link className="button button--primary button--large" href="/">Re-run with current data</Link>
          <div className="summary-provenance">
            <div><span>Generated</span><strong>{formatFreshness(snapshot.generated_at).replace("Updated ", "")}</strong></div>
          </div>
        </aside>
      </div>
    </main>
  );
}
