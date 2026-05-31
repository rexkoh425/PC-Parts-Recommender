"use client";

import Link from "next/link";
import { categoryLabels, formatFreshness, formatScore, formatSgd, humanizeToken, profileLabels } from "@/lib/format";
import type { BuildSummary, ComponentCategory } from "@/lib/types";
import { ScoreMeter } from "./score-meter";
import { StatusPill } from "./status-pill";

function componentFor(build: BuildSummary, category: ComponentCategory) {
  return build.components.find((component) => component.category === category);
}

export function BuildCard({
  build,
  budgetSgd,
  saved,
  onToggleSaved,
}: {
  build: BuildSummary;
  budgetSgd?: number;
  saved: boolean;
  onToggleSaved(build: BuildSummary): void;
}) {
  const featured = build.profile === "best_overall";
  const explanation = build.explanation?.[0];
  const explanationText = typeof explanation === "string" ? explanation : explanation?.text;
  const keyCategories: ComponentCategory[] = ["cpu", "gpu", "memory", "storage"];
  const scoreEntries = Object.entries(build.workload_scores ?? {}).slice(0, 2);

  return (
    <article
      className={`build-card ${featured ? "build-card--featured" : ""}`}
      data-testid="build-card"
    >
      <div className="build-card__topline">
        <span className="profile-kicker">{profileLabels[build.profile]}</span>
        {featured && <span className="recommended-chip">Recommended</span>}
      </div>
      <div className="build-card__price-row">
        <div>
          <small>Total for new parts</small>
          <h2>{formatSgd(build.total_price_sgd)}</h2>
        </div>
        <div className="overall-score" aria-label={`Relative fit score ${formatScore(build.overall_score)} out of 100 for this request`}>
          <strong>{formatScore(build.overall_score)}</strong>
          <small>relative fit</small>
        </div>
      </div>
      {typeof budgetSgd === "number" && (
        <p className="budget-note">
          {build.total_price_sgd <= budgetSgd
            ? `${formatSgd(budgetSgd - build.total_price_sgd)} budget headroom`
            : `${formatSgd(build.total_price_sgd - budgetSgd)} above budget`}
        </p>
      )}

      <ul className="component-preview">
        {keyCategories.map((category) => {
          const component = componentFor(build, category);
          if (!component) return null;
          return (
            <li key={category}>
              <span>{categoryLabels[category]}</span>
              <strong>{component.canonical_name}</strong>
            </li>
          );
        })}
      </ul>

      <div className="build-card__scores">
        {scoreEntries.map(([workload, score]) => (
          <ScoreMeter key={workload} compact label={humanizeToken(workload)} value={score} />
        ))}
      </div>

      <div className="build-card__facts">
        <StatusPill status={build.compatibility_status} />
        {typeof build.estimated_peak_power_w === "number" && (
          <span>{Math.round(build.estimated_peak_power_w)} W peak estimate</span>
        )}
      </div>

      {explanationText && <p className="build-card__reason">{explanationText}</p>}

      <div className="build-card__meta">
        <span>Generated {formatFreshness(build.generated_at).replace("Updated ", "")}</span>
        <span>{build.data_version ? `Data ${build.data_version}` : "Versioned evidence"}</span>
      </div>

      <div className="build-card__actions">
        <Link className="button button--primary" href={`/builds/${encodeURIComponent(build.build_id)}`}>
          View build
        </Link>
        <button
          className="button button--secondary button--icon"
          type="button"
          aria-pressed={saved}
          onClick={() => onToggleSaved(build)}
        >
          <span aria-hidden="true">{saved ? "◆" : "◇"}</span>
          {saved ? "Saved" : "Save"}
        </button>
      </div>
    </article>
  );
}
