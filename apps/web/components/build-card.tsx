"use client";

import Link from "next/link";
import type React from "react";
import { categoryLabels, formatScore, formatSgd, profileLabels } from "../lib/format";
import type { BuildSummary, ComponentCategory } from "../lib/types";
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
  const keyCategories: ComponentCategory[] = ["cpu", "gpu"];
  const objectiveScores = [
    ["Value", build.value_score],
    ["Upgradeability", build.upgradeability_score],
  ].filter((entry): entry is [string, number] => typeof entry[1] === "number");

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
        <div
          className="overall-score"
          // The ring reads the value from a custom property so the figure and
          // the arc can never disagree.
          style={{ "--score": build.overall_score } as React.CSSProperties}
          aria-label={`Relative fit score ${formatScore(build.overall_score)} out of 100 for this request`}
        >
          <strong>{formatScore(build.overall_score)}</strong>
          <small>fit</small>
        </div>
      </div>
      {typeof budgetSgd === "number" && (
        <p className="budget-note">
          {build.total_price_sgd <= budgetSgd
            ? `${formatSgd(budgetSgd - build.total_price_sgd)} budget headroom`
            : `${formatSgd(build.total_price_sgd - budgetSgd)} above budget`}
        </p>
      )}

      {objectiveScores.length > 0 && (
        <div
          className="build-card__objective-scores"
          aria-label="Build objective scores"
          data-testid="build-decision-scores"
        >
          {objectiveScores.map(([label, score]) => (
            <ScoreMeter key={label} compact label={label} value={score} />
          ))}
        </div>
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

      <div className="build-card__facts">
        <StatusPill status={build.compatibility_status} />
        {typeof build.estimated_peak_power_w === "number" && (
          <span>{Math.round(build.estimated_peak_power_w)} W peak estimate</span>
        )}
      </div>


      <div className="build-card__actions">
        <Link
          className="button button--primary"
          href={`/builds/${encodeURIComponent(build.build_id)}`}
          aria-label={`View ${profileLabels[build.profile]} build`}
        >
          View build
        </Link>
        <button
          className="button button--secondary button--icon"
          type="button"
          aria-pressed={saved}
          aria-label={`${saved ? "Remove" : "Save"} ${profileLabels[build.profile]} build${saved ? " from saved builds" : ""}`}
          onClick={() => onToggleSaved(build)}
        >
          <span aria-hidden="true">{saved ? "◆" : "◇"}</span>
          {saved ? "Saved" : "Save"}
        </button>
      </div>
    </article>
  );
}
