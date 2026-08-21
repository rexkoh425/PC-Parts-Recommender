"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useMemo, useState } from "react";
import {
  createBuildShare,
  getBuild,
  getSessionId,
  productImpressionContext,
  rememberProductImpression,
  revokeBuildShare,
  trackInteraction,
  USING_DEMO_DATA,
} from "@/lib/api";
import { Tabs } from "@/components/tabs";
import { formatSignedDelta, summarizeCompatibilityChecks } from "@/lib/catalogue";
import {
  categoryLabels,
  formatFreshness,
  formatScore,
  formatSgd,
  humanizeToken,
  profileLabels,
  workloadLabels,
} from "@/lib/format";
import {
  summarizeReplacementChange,
  type ReplacementChangeSummary,
} from "@/lib/replacement";
import type {
  BuildComponent,
  BuildShareCreated,
  BuildSummary,
  ComponentCategory,
  PerformanceSignal,
  ReplacementResponse,
  WorkloadName,
} from "@/lib/types";
import { useSavedBuilds } from "@/lib/use-saved-builds";
import { sharedBuildHref } from "@/lib/shared-build";
import { BuildBudgetBreakdown } from "./build-budget-breakdown";
import { ReplacementDrawer } from "./replacement-drawer";
import { ScoreMeter } from "./score-meter";
import { StatusPill } from "./status-pill";

const profileHeadings: Record<BuildSummary["profile"], string> = {
  best_overall: "The strongest balance across workload fit, value, and flexibility.",
  best_value: "The most workload value inside this compatible shortlist.",
  highest_performance: "The highest relative workload performance that fits the brief.",
  most_upgradeable: "A compatible foundation with more room for future upgrades.",
  lowest_power: "The lowest estimated peak load among the compatible options.",
};

function relativeDecisionLabel(signal: PerformanceSignal): string {
  switch (signal.decision) {
    case "model_not_promotion_eligible":
      return "Relative score · model not promoted";
    case "input_outside_training_contract":
      return "Relative score · input outside training range";
    case "model_not_promotion_eligible_and_input_outside_training_contract":
      return "Relative score · model not promoted and input outside range";
    case "precise_predictions_disabled":
      return "Relative score · precise estimate disabled";
    case "precise_predictions_disabled_and_input_outside_training_contract":
      return "Relative score · precise estimate disabled and input outside range";
    case "deterministic_baseline":
      return "Relative score";
    default:
      return "Relative score";
  }
}

function EvidenceBadge({ signal }: { signal?: PerformanceSignal }) {
  if (!signal || signal.basis === "insufficient_data") {
    return <span className="evidence-badge evidence-badge--missing">Evidence limited</span>;
  }
  return (
    <span className={`evidence-badge evidence-badge--${signal.basis}`}>
      {signal.basis === "observed"
        ? "Observed"
        : signal.basis === "predicted"
          ? `Predicted${signal.confidence ? ` · ${signal.confidence}` : ""}`
          : relativeDecisionLabel(signal)}
    </span>
  );
}

function DetailSkeleton() {
  return (
    <main className="shell detail-page" aria-busy="true">
      <div className="skeleton skeleton--eyebrow" />
      <div className="skeleton skeleton--title" />
      <div className="detail-layout">
        <div className="skeleton-card skeleton-card--tall" />
        <div className="skeleton-card" />
      </div>
      <p className="sr-only" role="status">Loading build details.</p>
    </main>
  );
}

function componentReason(component: BuildComponent): string {
  return component.selection_reasons?.[0] ?? "Selected for compatibility and objective fit.";
}

function formatSignedSgd(value: number): string {
  return `${value > 0 ? "+" : ""}${formatSgd(value)}`;
}

function workloadLabel(workload: string): string {
  return workload in workloadLabels
    ? workloadLabels[workload as WorkloadName]
    : humanizeToken(workload);
}

// Only an estimate earns a visible label. A sourced price is just a price;
// badging every row turned the table into a provenance audit rather than a
// parts list someone reads.
const priceBasisLabel = {
  launch_msrp: "",
  manufacturer_list: "",
  street_aug_2026: "",
  estimate: "estimate",
} as const;

// The tooltip carries the detail the badge has no room for: where the number
// came from and when, so a reader can judge it rather than just trust it.
function priceBasisDetail(component: BuildComponent): string {
  const asOf = component.price_as_of ? ` as of ${component.price_as_of}` : "";
  if (component.price_basis === "manufacturer_list") {
    return `Current list price on the manufacturer's own store${component.price_source ? ` (${component.price_source})` : ""}${asOf}, converted at 1 USD = 1.2775 SGD.`;
  }
  if (component.price_basis === "launch_msrp") {
    return `Manufacturer launch MSRP${component.price_source ? ` published by ${component.price_source}` : ""}, converted at 1 USD = 1.2775 SGD.`;
  }
  if (component.price_basis === "street_aug_2026") {
    return `Retail price${component.price_source ? ` from ${component.price_source}` : ""}${asOf}, converted at 1 USD = 1.2775 SGD. Not a live quote.`;
  }
  return `Fixture estimate. Public sources disagreed too widely to cite one${asOf}.`;
}

export function BuildDetailScreen({ buildId }: { buildId: string }) {
  const router = useRouter();
  const { savedIds, toggle } = useSavedBuilds();
  const [build, setBuild] = useState<BuildSummary | null>(null);
  const [error, setError] = useState("");
  const [retryKey, setRetryKey] = useState(0);
  const [replacementCategory, setReplacementCategory] = useState<ComponentCategory | null>(null);
  const [announcement, setAnnouncement] = useState("");
  const [replacementResult, setReplacementResult] = useState<ReplacementChangeSummary | null>(null);
  const [shareState, setShareState] = useState<"idle" | "copied" | "failed" | "revoking" | "revoked">("idle");
  const [activeShare, setActiveShare] = useState<BuildShareCreated | null>(null);

  useEffect(() => {
    let active = true;
    const controller = new AbortController();
    getBuild(buildId, { signal: controller.signal })
      .then((result) => {
        if (!active) return;
        setBuild(result);
        void trackInteraction({
          event_type: "build_viewed",
          session_id: getSessionId(),
          impression_token: result.impression_token ?? undefined,
        });
      })
      .catch((requestError) => {
        if (active) {
          setError(requestError instanceof Error ? requestError.message : "The build could not be loaded.");
        }
      });
    return () => {
      active = false;
      controller.abort();
    };
  }, [buildId, retryKey]);

  const performanceSignals = useMemo(
    () => build?.components.flatMap((component) => component.performance_signals ?? []) ?? [],
    [build],
  );

  function handleReplacement(response: ReplacementResponse) {
    if (!build) return;
    const summary = summarizeReplacementChange(build, response);
    setBuild(response.build);
    setReplacementResult(summary);
    const changed = summary.changedCategories.map((category) => categoryLabels[category]).join(", ");
    const workloadChanges = summary.workloadScoreDeltas
      .map(([workload, delta]) => `${workloadLabel(workload)} ${formatSignedDelta(delta, " points")}`)
      .join(", ");
    setAnnouncement(
      `Replacement applied.${changed ? ` Changed: ${changed}.` : ""} Price ${formatSignedSgd(summary.priceDeltaSgd)}. Peak power ${formatSignedDelta(summary.powerDeltaW, " W")}.${workloadChanges ? ` Workload scores: ${workloadChanges}.` : ""}`,
    );
    void trackInteraction({
      event_type: "component_replaced",
      session_id: getSessionId(),
      impression_token: response.build.impression_token ?? undefined,
    });
    router.replace(`/builds/${encodeURIComponent(response.build.build_id)}`);
  }

  async function revokeCreatedShare(created: BuildShareCreated): Promise<boolean> {
    try {
      await revokeBuildShare(created.share_id, created.revocation_token);
      window.sessionStorage.removeItem(`pcbr:build-share-revocation:${created.share_id}`);
      setActiveShare(null);
      return true;
    } catch {
      window.sessionStorage.setItem(
        `pcbr:build-share-revocation:${created.share_id}`,
        created.revocation_token,
      );
      setActiveShare(created);
      return false;
    }
  }

  async function revokeActiveShare() {
    if (!activeShare) return;
    setShareState("revoking");
    if (await revokeCreatedShare(activeShare)) {
      setShareState("revoked");
      setAnnouncement("The public build snapshot has been revoked.");
    } else {
      setShareState("failed");
      setAnnouncement("The public snapshot could not be revoked. The revoke control remains available so you can try again.");
    }
  }

  async function shareBuildSnapshot() {
    if (!build || typeof window === "undefined") return;
    let url = new URL(sharedBuildHref(build), window.location.origin).toString();
    const shareData = {
      title: `${profileLabels[build.profile]} PC build`,
      text: "A generation-time BuildSignal PC build snapshot.",
      url,
    };
    let createdShare: BuildShareCreated | undefined;
    try {
      if (!USING_DEMO_DATA) {
        const share = activeShare ?? await createBuildShare(build.build_id);
        if (!activeShare) createdShare = share;
        url = new URL(`/share?share=${encodeURIComponent(share.share_id)}`, window.location.origin).toString();
        shareData.url = url;
      }
      if (navigator.share) {
        await navigator.share(shareData);
      } else if (navigator.clipboard?.writeText) {
        await navigator.clipboard.writeText(url);
      } else {
        throw new Error("Sharing is not available in this browser.");
      }
      if (createdShare) {
        window.sessionStorage.setItem(
          `pcbr:build-share-revocation:${createdShare.share_id}`,
          createdShare.revocation_token,
        );
        setActiveShare(createdShare);
      }
      setShareState("copied");
      setAnnouncement("A shareable build snapshot is ready. Re-run it before buying to check current data.");
      void trackInteraction({
        event_type: "build_shared",
        session_id: getSessionId(),
        impression_token: build.impression_token ?? undefined,
        metadata: { share_format: "public_snapshot_v1" },
      });
    } catch (error) {
      const wasCancelled = error instanceof DOMException && error.name === "AbortError";
      const revoked = createdShare ? await revokeCreatedShare(createdShare) : true;
      if (!revoked) {
        setShareState("failed");
        setAnnouncement("Sharing did not finish and automatic cleanup failed. Use Revoke public snapshot to remove the created link.");
        return;
      }
      if (wasCancelled) {
        setShareState("idle");
        setAnnouncement("Sharing was cancelled.");
        return;
      }
      setShareState("failed");
      setAnnouncement("This browser could not create the share link. You can still save the build locally.");
    }
  }

  if (error) {
    return (
      <main className="shell state-page">
        <div className="state-card" role="alert">
          <span className="state-card__icon" aria-hidden="true">?</span>
          <p className="eyebrow">Build unavailable</p>
          <h1>We could not load this build.</h1>
          <p>{error}</p>
          <div className="button-row">
            <button
              className="button button--primary"
              type="button"
              onClick={() => {
                setError("");
                setBuild(null);
                setRetryKey((key) => key + 1);
              }}
            >
              Try again
            </button>
            <Link className="button button--secondary" href="/">
              Start a new build
            </Link>
          </div>
        </div>
      </main>
    );
  }

  if (!build) return <DetailSkeleton />;

  const checks = [...(build.compatibility_checks ?? []), ...(build.warnings ?? [])].filter(
    (check, index, all) => all.findIndex((item) => item.rule_id === check.rule_id) === index,
  );
  const compatibilitySummary = summarizeCompatibilityChecks(checks);
  const saved = savedIds.has(build.build_id);
  const observedCount = performanceSignals.filter((signal) => signal.basis === "observed").length;
  const predictedCount = performanceSignals.filter((signal) => signal.basis === "predicted").length;

  return (
    <main className="shell detail-page">
      <nav className="breadcrumbs" aria-label="Breadcrumb">
        <Link href="/">Build brief</Link>
        <span aria-hidden="true">/</span>
        <Link href={build.request_id ? `/recommendations/${encodeURIComponent(build.request_id)}` : "/"}>
          Recommendations
        </Link>
        <span aria-hidden="true">/</span>
        <span aria-current="page">{profileLabels[build.profile]}</span>
      </nav>

      <header className="detail-header">
        <div>
          <p className="eyebrow">{profileLabels[build.profile]}</p>
          <h1>{profileHeadings[build.profile]}</h1>
          <p className="lede">
            Component-level reasons, benchmark basis, and compatibility checks are kept visible.
          </p>
        </div>
        <StatusPill status={build.compatibility_status} />
      </header>

      <div className="visually-live" aria-live="polite">{announcement}</div>

      {replacementResult && (
        <section
          className="replacement-result"
          aria-labelledby="replacement-result-heading"
          data-testid="replacement-result"
        >
          <div className="replacement-result__header">
            <div>
              <p className="eyebrow">Updated build</p>
              <h2 id="replacement-result-heading">Replacement impact</h2>
            </div>
            <button
              type="button"
              className="text-button"
              onClick={() => setReplacementResult(null)}
              aria-label="Dismiss replacement impact"
            >
              Dismiss
            </button>
          </div>
          <dl className="replacement-result__metrics">
            <div>
              <dt>Changed components</dt>
              <dd>
                {replacementResult.changedCategories.length
                  ? replacementResult.changedCategories
                      .map((category) => categoryLabels[category])
                      .join(", ")
                  : "No category changes reported"}
              </dd>
            </div>
            <div>
              <dt>Build price</dt>
              <dd>{formatSignedSgd(replacementResult.priceDeltaSgd)}</dd>
            </div>
            <div>
              <dt>Peak power</dt>
              <dd>{formatSignedDelta(replacementResult.powerDeltaW, " W")}</dd>
            </div>
          </dl>
          <div className="replacement-result__workloads">
            <strong>Workload scores</strong>
            {replacementResult.workloadScoreDeltas.length ? (
              <ul>
                {replacementResult.workloadScoreDeltas.map(([workload, delta]) => (
                  <li key={workload}>
                    <span>{workloadLabel(workload)}</span>
                    <strong>{formatSignedDelta(delta, " points")}</strong>
                  </li>
                ))}
              </ul>
            ) : (
              <p>No workload-score change was reported.</p>
            )}
          </div>
        </section>
      )}

      <div className="detail-layout">
        <div className="detail-main">
          <BuildBudgetBreakdown build={build} demo={USING_DEMO_DATA} />

          <Tabs
            label="Build detail"
            className="detail-tabs"
            items={[
              {
                id: "components",
                label: "Components",
                hint: String(build.components.length),
                content: (
          <section className="detail-section" aria-labelledby="components-heading">
            <div className="section-heading">
              <div>
                <p className="eyebrow">Complete parts list</p>
                <h2 id="components-heading">Selected components</h2>
              </div>
              <p>{build.components.length} categories · one compatible configuration</p>
            </div>
            <div className="table-scroll component-table-scroll">
              <table className="component-table" data-testid="component-table">
                <thead>
                  <tr>
                    <th scope="col">Category</th>
                    <th scope="col">Product and reason</th>
                    <th scope="col">Evidence</th>
                    <th scope="col">Price</th>
                    <th scope="col"><span className="sr-only">Actions</span></th>
                  </tr>
                </thead>
                <tbody>
                  {build.components.map((component) => {
                    const impressionContext = productImpressionContext({
                      surface: "build_component",
                      sourceId: build.build_id,
                      productId: component.product_id,
                    });
                    const productHref = `/products/${encodeURIComponent(component.product_id)}?impression=${encodeURIComponent(impressionContext)}`;
                    return (
                    <tr key={`${component.category}-${component.product_id}`} data-testid="component-row">
                      <th scope="row" data-label="Category">
                        <span className={`category-icon category-icon--${component.category}`} aria-hidden="true">
                          {component.category.slice(0, 2).toUpperCase()}
                        </span>
                        {categoryLabels[component.category]}
                      </th>
                      <td data-label="Product">
                        <strong>{component.canonical_name}</strong>
                        <span>{componentReason(component)}</span>
                        {component.already_owned ? (
                          <small className="owned-chip">Already owned · excluded from spend</small>
                        ) : component.retailer ? (
                          <small>{component.retailer}</small>
                        ) : null}
                      </td>
                      <td data-label="Evidence">
                        <EvidenceBadge signal={component.performance_signals?.[0]} />
                        <div className="component-evidence-links">
                          <Link
                            href={`${productHref}#prices-heading`}
                            aria-label={`View price history evidence for ${component.canonical_name}`}
                            onClick={() => rememberProductImpression(component, impressionContext)}
                          >
                            Price history
                          </Link>
                          <Link
                            href={`${productHref}#reviews-heading`}
                            aria-label={`View review evidence for ${component.canonical_name}`}
                            onClick={() => rememberProductImpression(component, impressionContext)}
                          >
                            Reviews
                          </Link>
                          {component.listing_url ? (
                            <a
                              href={component.listing_url}
                              target="_blank"
                              rel="noreferrer"
                              aria-label={`Open recorded retailer price for ${component.canonical_name}`}
                              onClick={() =>
                                void trackInteraction({
                                  event_type: "retailer_clicked",
                                  session_id: getSessionId(),
                                  impression_token: component.impression_token ?? undefined,
                                })
                              }
                            >
                              Retailer price ↗
                            </a>
                          ) : component.spec_url ? (
                            <a
                              href={component.spec_url}
                              target="_blank"
                              rel="noreferrer"
                              aria-label={`Open the manufacturer specification page for ${component.canonical_name}`}
                            >
                              Manufacturer spec ↗
                            </a>
                          ) : (
                            <small>
                              {USING_DEMO_DATA
                                ? "No external demo listing"
                                : "No retailer URL recorded"}
                            </small>
                          )}
                        </div>
                      </td>
                      <td data-label="Price">
                        <strong>{component.already_owned ? "Owned" : formatSgd(component.price_sgd)}</strong>
                        {!component.already_owned && component.price_basis && priceBasisLabel[component.price_basis] ? (
                          <span
                            className={`price-basis price-basis--${component.price_basis}`}
                            title={priceBasisDetail(component)}
                          >
                            {priceBasisLabel[component.price_basis]}
                          </span>
                        ) : null}
                      </td>
                      <td data-label="Action">
                        <button
                          type="button"
                          className="text-button"
                          onClick={() => setReplacementCategory(component.category)}
                          aria-label={`Replace ${component.canonical_name}`}
                        >
                          Replace
                        </button>
                      </td>
                    </tr>
                    );
                  })}
                </tbody>
              </table>
              <p className="price-note">
                RAM and SSD prices are high because of the 2026 memory shortage. Figures are
                from August 2026 and converted from USD.
              </p>
            </div>
          </section>
                ),
              },
              {
                id: "performance",
                label: "Performance",
                content: (
          <section className="detail-section" aria-labelledby="performance-heading">
            <div className="section-heading">
              <div>
                <p className="eyebrow">Workload fit</p>
                <h2 id="performance-heading">Performance basis</h2>
              </div>
              <p>{observedCount} observed · {predictedCount} predicted signals</p>
            </div>
            <div className="performance-grid">
              {Object.entries(build.workload_scores ?? {}).map(([workload, score]) => (
                <div className="metric-card" key={workload}>
                  <ScoreMeter label={humanizeToken(workload)} value={score} />
                </div>
              ))}
            </div>
            {performanceSignals.length > 0 ? (
              <ul className="evidence-list">
                {performanceSignals.map((signal, index) => (
                  <li key={`${signal.workload}-${signal.metric}-${index}`}>
                    <div>
                      <EvidenceBadge signal={signal} />
                      <strong>{signal.metric}</strong>
                      <span>{humanizeToken(signal.workload)}</span>
                    </div>
                    <div className="evidence-list__value">
                      <strong>
                        {signal.value === null ? "Not enough evidence" : `${formatScore(signal.value)}${signal.unit ? ` ${signal.unit}` : ""}`}
                      </strong>
                      {signal.model_version && <small>Model {signal.model_version}</small>}
                    </div>
                    {(signal.sources ?? []).length > 0 && (
                      <div className="evidence-list__sources">
                        {(signal.sources ?? []).map((source) => (
                          <a
                            key={source.url}
                            href={source.url}
                            target="_blank"
                            rel="noreferrer"
                            aria-label={`Open ${source.label} source for ${signal.metric}`}
                          >
                            {source.label} ↗
                          </a>
                        ))}
                      </div>
                    )}
                  </li>
                ))}
              </ul>
            ) : (
              <div className="empty-evidence">
                <strong>Detailed benchmark evidence is not attached to this response.</strong>
                <p>The overall relative scores remain available; missing values are not shown as zero.</p>
              </div>
            )}
          </section>
                ),
              },
              {
                id: "compatibility",
                label: "Compatibility",
                hint: String(checks.length),
                content: (
          <section className="detail-section" aria-labelledby="compatibility-heading">
            <div className="section-heading">
              <div>
                <p className="eyebrow">Versioned rule engine</p>
                <h2 id="compatibility-heading">Compatibility checks</h2>
              </div>
              <StatusPill status={build.compatibility_status} />
            </div>
            {checks.length > 0 ? (
              <>
                <div className="compatibility-overview" role="status">
                  <dl>
                    <div><dt>Checks run</dt><dd>{compatibilitySummary.total}</dd></div>
                    <div><dt>Passed</dt><dd>{compatibilitySummary.pass}</dd></div>
                    <div><dt>Warnings</dt><dd>{compatibilitySummary.warning}</dd></div>
                    <div><dt>Unknown</dt><dd>{compatibilitySummary.unknown}</dd></div>
                  </dl>
                  <p>
                    Complete builds cannot contain a hard FAIL or UNKNOWN result. Warnings remain
                    visible and are penalised by the optimiser.
                  </p>
                </div>
                <ul className="check-list">
                  {checks.map((check) => (
                    <li key={check.rule_id}>
                      <StatusPill status={check.status} label={humanizeToken(check.status)} />
                      <div>
                        <strong>{check.message}</strong>
                        <span className="check-list__meta">
                          <small>{check.rule_id}</small>
                          {(check.affected_categories ?? []).length > 0 && (
                            <small>
                              {(check.affected_categories ?? [])
                                .map((category) => categoryLabels[category])
                                .join(" + ")}
                            </small>
                          )}
                          {check.evidence_source && <small>Evidence: {check.evidence_source}</small>}
                        </span>
                      </div>
                    </li>
                  ))}
                </ul>
              </>
            ) : (
              <div className="empty-evidence">
                <strong>Detailed rule outcomes were not returned.</strong>
                <p>
                  The overall build status is {build.compatibility_status}, but an absent check list
                  is never interpreted as evidence that every individual rule passed.
                </p>
              </div>
            )}
          </section>
                ),
              },
            ]}
          />
        </div>

        <aside className="build-summary-card" aria-label="Build summary">
          <p className="profile-kicker">{profileLabels[build.profile]}</p>
          <div className="build-summary-card__price">
            <small>Total for new parts</small>
            <strong>{formatSgd(build.total_price_sgd)}</strong>
          </div>
          <div className="summary-score">
            <strong>{formatScore(build.overall_score)}</strong>
            <span>Relative fit score</span>
          </div>
          <dl>
            {typeof build.value_score === "number" && (
              <div><dt>Value</dt><dd>{formatScore(build.value_score)}</dd></div>
            )}
            {typeof build.upgradeability_score === "number" && (
              <div><dt>Upgradeability</dt><dd>{formatScore(build.upgradeability_score)}</dd></div>
            )}
            {typeof build.estimated_peak_power_w === "number" && (
              <div><dt>Estimated peak</dt><dd>{Math.round(build.estimated_peak_power_w)} W</dd></div>
            )}
          </dl>
          <StatusPill status={build.compatibility_status} />
          <button
            className="button button--primary button--large"
            type="button"
            aria-pressed={saved}
            onClick={() => {
              const nowSaved = toggle(build);
              if (nowSaved) {
                void trackInteraction({
                  event_type: "build_saved",
                  session_id: getSessionId(),
                  impression_token: build.impression_token ?? undefined,
                });
              }
            }}
          >
            {saved ? "Saved to this browser" : "Save this build"}
          </button>
          <button
            className="button button--secondary button--large"
            type="button"
            onClick={() => void shareBuildSnapshot()}
            disabled={shareState === "revoking"}
          >
            {shareState === "copied" ? "Share link ready" : "Share snapshot"}
          </button>
          {activeShare && !USING_DEMO_DATA && (
            <button
              className="text-button"
              type="button"
              onClick={() => void revokeActiveShare()}
              disabled={shareState === "revoking"}
            >
              {shareState === "revoking" ? "Revoking public snapshot…" : "Revoke public snapshot"}
            </button>
          )}
          <p className={`share-status share-status--${shareState}`} role="status">
            {shareState === "copied"
              ? "A public snapshot was shared or copied. It excludes your request, saved builds, and retailer links."
              : shareState === "revoked"
                ? "The previously created public snapshot has been revoked."
              : shareState === "failed"
                ? activeShare
                  ? "A public snapshot exists but sharing or automatic cleanup failed. Use the revoke control above."
                  : "Sharing is unavailable in this browser."
                : "Creates a generation-time public snapshot; it is not a live stock or price quote."}
          </p>
          <div className="summary-provenance">
            <div><span>Generated</span><strong>{formatFreshness(build.generated_at).replace("Updated ", "")}</strong></div>
            <div><span>Data</span><strong>{build.data_version ?? "Versioned"}</strong></div>
            <div><span>Ranker</span><strong>{build.ranking_model ?? "Recorded by API"}</strong></div>
            <div><span>Rules</span><strong>{build.rule_version ?? "Recorded by API"}</strong></div>
            <div><span>Solver</span><strong>{build.solver_version}</strong></div>
          </div>
        </aside>
      </div>

      {replacementCategory && (
        <ReplacementDrawer
          build={build}
          category={replacementCategory}
          onClose={() => setReplacementCategory(null)}
          onReplaced={handleReplacement}
        />
      )}
    </main>
  );
}
