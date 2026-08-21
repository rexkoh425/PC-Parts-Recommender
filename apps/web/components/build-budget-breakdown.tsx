import { categoryLabels, formatSgd } from "../lib/format";
import type { BuildSummary } from "../lib/types";

export function BuildBudgetBreakdown({
  build,
  demo,
}: {
  build: BuildSummary;
  demo: boolean;
}) {
  const newParts = build.components.filter((component) => !component.already_owned);
  const ownedParts = build.components.filter((component) => component.already_owned);
  const componentSubtotal = newParts.reduce((total, component) => total + component.price_sgd, 0);
  const reportedDifference = build.total_price_sgd - componentSubtotal;
  const hasDifference = Math.abs(reportedDifference) >= 0.01;

  return (
    <section
      className="detail-section budget-breakdown"
      aria-labelledby="budget-breakdown-heading"
      data-testid="budget-breakdown"
    >
      <div className="section-heading">
        <div>
          <p className="eyebrow">Recorded spend</p>
          <h2 id="budget-breakdown-heading">Budget breakdown</h2>
        </div>
        <p>
          {demo
            ? "Prices recorded August 2026."
            : "Generation-time listing observations; verify the retailer before purchase."}
        </p>
      </div>

      <dl className="budget-breakdown__summary">
        <div>
          <dt>New-parts total</dt>
          <dd>{formatSgd(build.total_price_sgd)}</dd>
        </div>
        <div>
          <dt>Priced components</dt>
          <dd>{newParts.length}</dd>
        </div>
        <div>
          <dt>Existing parts excluded</dt>
          <dd>{ownedParts.length}</dd>
        </div>
      </dl>

      <ol className="budget-breakdown__parts" aria-label="Component contribution to new-parts total">
        {build.components.map((component) => {
          const share =
            !component.already_owned && build.total_price_sgd > 0
              ? Math.min(100, Math.max(0, (component.price_sgd / build.total_price_sgd) * 100))
              : 0;
          return (
            <li key={`${component.category}-${component.product_id}`}>
              <div>
                <span>{categoryLabels[component.category]}</span>
                <strong>{component.already_owned ? "Owned" : formatSgd(component.price_sgd)}</strong>
              </div>
              <div
                className="budget-breakdown__track"
                role="meter"
                aria-label={`${categoryLabels[component.category]} share of new-parts total`}
                aria-valuemin={0}
                aria-valuemax={100}
                aria-valuenow={Math.round(share)}
              >
                <span style={{ width: `${share}%` }} />
              </div>
              <small>{component.canonical_name}</small>
            </li>
          );
        })}
      </ol>

      {hasDifference && (
        <p className="budget-breakdown__difference" role="note">
          The parts add up to {formatSgd(componentSubtotal)}, {formatSgd(Math.abs(reportedDifference))} away
          from the build total shown.
        </p>
      )}
    </section>
  );
}
