"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useMemo, useState } from "react";
import { getRequestBuilds, getSessionId, trackInteraction } from "@/lib/api";
import { categoryLabels, formatScore, formatSgd, profileLabels } from "@/lib/format";
import { useSavedBuilds } from "@/lib/use-saved-builds";
import type {
  BuildSummary,
  GenerateBuildsResponse,
  SuggestedRelaxation,
} from "@/lib/types";
import { BuildCard } from "./build-card";

function ResultSkeleton() {
  return (
    <main className="shell results-page" aria-busy="true" aria-label="Loading recommendations">
      <div className="skeleton skeleton--eyebrow" />
      <div className="skeleton skeleton--title" />
      <div className="results-grid">
        {[0, 1, 2].map((index) => (
          <div className="skeleton-card" key={index}>
            <div className="skeleton skeleton--line" />
            <div className="skeleton skeleton--price" />
            <div className="skeleton skeleton--block" />
          </div>
        ))}
      </div>
      <p className="sr-only" role="status">
        Loading ranked builds.
      </p>
    </main>
  );
}

function NoFeasibleBuild({
  response,
  onRelax,
}: {
  response: GenerateBuildsResponse;
  onRelax(relaxation: SuggestedRelaxation): void;
}) {
  const reasons = response.infeasibility?.reasons ?? [
    {
      code: "no_feasible_build",
      message: "Nothing in stock meets every requirement at this budget.",
    },
  ];
  const relaxations = response.infeasibility?.suggested_relaxations ?? [];

  return (
    <section className="infeasible-panel" aria-labelledby="infeasible-title">
      <div className="infeasible-panel__icon" aria-hidden="true">
        ↗
      </div>
      <div>
        <p className="eyebrow">Constraint result</p>
        <h1 id="infeasible-title">No compatible build yet</h1>
        <p className="lede">
          We did not substitute an incompatible part. These requirements eliminated the feasible
          catalogue:
        </p>
        <ul className="reason-list">
          {reasons.map((reason) => (
            <li key={`${reason.code}-${reason.message}`}>{reason.message}</li>
          ))}
        </ul>
        {relaxations.length > 0 && (
          <div className="relaxations">
            <h2>Smallest useful adjustments</h2>
            <div className="relaxation-grid">
              {relaxations.map((relaxation) => (
                <button
                  key={`${relaxation.field_path}-${String(relaxation.proposed_value)}`}
                  type="button"
                  onClick={() => onRelax(relaxation)}
                >
                  <strong>{relaxation.expected_effect}</strong>
                  <span>
                    Change {relaxation.field_path.split(".").at(-1)?.replaceAll("_", " ")} to{" "}
                    {String(relaxation.proposed_value)}
                  </span>
                </button>
              ))}
            </div>
          </div>
        )}
        <Link className="button button--primary" href="/">
          Edit the build brief
        </Link>
      </div>
    </section>
  );
}

export function RecommendationsScreen({ requestId }: { requestId: string }) {
  const router = useRouter();
  const { savedIds, toggle } = useSavedBuilds();
  const [response, setResponse] = useState<GenerateBuildsResponse | null>(null);
  const [error, setError] = useState("");
  const [retryKey, setRetryKey] = useState(0);

  useEffect(() => {
    let active = true;
    const controller = new AbortController();
    getRequestBuilds(requestId, { signal: controller.signal })
      .then((result) => {
        if (!active) return;
        setResponse(result);
      })
      .catch((requestError) => {
        if (active) {
          setError(
            requestError instanceof Error
              ? requestError.message
              : "The recommendation set could not be loaded.",
          );
        }
      });
    return () => {
      active = false;
      controller.abort();
    };
  }, [requestId, retryKey]);

  const builds = useMemo(() => {
    if (!response) return [];
    return response.builds.map((build) => ({
        ...build,
        request_id: build.request_id ?? response.request_id,
        generated_at: build.generated_at ?? response.generated_at,
        data_version: build.data_version ?? response.data_version,
        ranking_model: build.ranking_model ?? response.ranking_model,
        rule_version: build.rule_version ?? response.rule_version,
      }));
  }, [response]);

  function toggleSaved(build: BuildSummary) {
    const nowSaved = toggle(build);
    if (nowSaved && response) {
      void trackInteraction({
        event_type: "build_saved",
        session_id: getSessionId(),
        impression_token: build.impression_token ?? undefined,
      });
    }
  }

  function applyRelaxation(relaxation: SuggestedRelaxation) {
    window.sessionStorage.setItem("pcbr:suggested-relaxation", JSON.stringify(relaxation));
    router.push("/");
  }

  if (error) {
    return (
      <main className="shell state-page">
        <div className="state-card" role="alert">
          <span className="state-card__icon" aria-hidden="true">
            ⟳
          </span>
          <p className="eyebrow">Connection interrupted</p>
          <h1>We could not load this recommendation set.</h1>
          <p>{error}</p>
          <div className="button-row">
            <button
              className="button button--primary"
              type="button"
              onClick={() => {
                setError("");
                setResponse(null);
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

  if (!response) return <ResultSkeleton />;

  if (response.status === "infeasible" || builds.length === 0) {
    return (
      <main className="shell results-page">
        <NoFeasibleBuild response={response} onRelax={applyRelaxation} />
      </main>
    );
  }

  const budget = response.request?.budget_sgd;
  const isPartial = response.status === "partial" || builds.length < 3;

  return (
    <main className="shell results-page">
      <nav className="breadcrumbs" aria-label="Breadcrumb">
        <Link href="/">Build brief</Link>
        <span aria-hidden="true">/</span>
        <span aria-current="page">Recommendations</span>
      </nav>

      <header className="results-header">
        <div>
          <p className="eyebrow">Constraint-checked shortlist</p>
          <h1>
            {builds.length} compatible build{builds.length === 1 ? "" : "s"}
            {budget ? ` under ${formatSgd(budget)}` : ""}
          </h1>
          <p className="lede">
            Each profile uses the same requirements. They differ in what they prioritise.
          </p>
        </div>
        <Link className="button button--secondary" href="/">
          Edit requirements
        </Link>
      </header>

      {isPartial && (
        <div className="notice-banner" role="status">
          <span aria-hidden="true">i</span>
          <p>
            <strong>Only {builds.length} meaningfully distinct build{builds.length === 1 ? " was" : "s were"} feasible.</strong>{" "}
            We avoid returning near-duplicates just to fill the page.
          </p>
        </div>
      )}

      <div className="notice-banner notice-banner--score" role="note">
        <span aria-hidden="true">i</span>
        <p>
          <strong>Scores are relative to this request.</strong>{" "}
          They compare the builds in this shortlist, not products in general.
        </p>
      </div>

      <section className="results-grid" aria-label="Ranked PC builds">
        {builds.map((build) => (
          <BuildCard
            key={build.build_id}
            build={build}
            budgetSgd={budget}
            saved={savedIds.has(build.build_id)}
            onToggleSaved={toggleSaved}
          />
        ))}
      </section>

      <section className="comparison-panel" aria-labelledby="comparison-heading">
        <div className="section-heading">
          <div>
            <p className="eyebrow">At a glance</p>
            <h2 id="comparison-heading">Compare the trade-offs</h2>
          </div>
          <p>All scores are normalised within the current compatible candidate set.</p>
        </div>
        <div className="table-scroll" tabIndex={0} aria-label="Scrollable build comparison">
          <table className="comparison-table">
            <thead>
              <tr>
                <th scope="col">Profile</th>
                <th scope="col">Total</th>
                <th scope="col">Overall</th>
                <th scope="col">Graphics</th>
                <th scope="col">Processor</th>
                <th scope="col">Peak power</th>
              </tr>
            </thead>
            <tbody>
              {builds.map((build) => (
                <tr key={build.build_id}>
                  <th scope="row">
                    <Link href={`/builds/${encodeURIComponent(build.build_id)}`}>
                      {profileLabels[build.profile]}
                    </Link>
                  </th>
                  <td>{formatSgd(build.total_price_sgd)}</td>
                  <td>{formatScore(build.overall_score)}</td>
                  <td>
                    {build.components.find((component) => component.category === "gpu")?.canonical_name ?? "—"}
                  </td>
                  <td>
                    {build.components.find((component) => component.category === "cpu")?.canonical_name ?? "—"}
                  </td>
                  <td>
                    {typeof build.estimated_peak_power_w === "number"
                      ? `${Math.round(build.estimated_peak_power_w)} W`
                      : "—"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <footer className="evidence-footer">
        <p>
          {categoryLabels.gpu} and workload scores distinguish direct benchmark observations from model estimates in build details.
        </p>
      </footer>
    </main>
  );
}
