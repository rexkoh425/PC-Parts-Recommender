"use client";

import { useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { Tabs } from "@/components/tabs";
import { generateBuilds, getSessionId, trackInteraction, USING_DEMO_DATA } from "@/lib/api";
import {
  defaultBuildFormValues,
  MAX_PERFORMANCE_TARGET_LENGTH,
  toBuildRequest,
  validateBuildForm,
  type BuildFormErrors,
  type BuildFormValues,
} from "@/lib/build-request";
import { formatSgd, profileLabels, workloadLabels } from "@/lib/format";
import type { BuildProfile, ExistingProductInput, WorkloadName } from "@/lib/types";
import { ExistingProductPicker } from "./existing-product-picker";

const workloadOptions = Object.entries(workloadLabels) as Array<[WorkloadName, string]>;
const profileOptions = Object.entries(profileLabels) as Array<[BuildProfile, string]>;

function initialBuildFormValues(): BuildFormValues {
  if (typeof window === "undefined") return defaultBuildFormValues;
  const stored = window.sessionStorage.getItem("pcbr:suggested-relaxation");
  if (!stored) return defaultBuildFormValues;
  window.sessionStorage.removeItem("pcbr:suggested-relaxation");
  try {
    const relaxation = JSON.parse(stored) as {
      field_path?: string;
      proposed_value?: unknown;
    };
    const fieldMap: Record<string, keyof BuildFormValues> = {
      budget_sgd: "budget_sgd",
      "requirements.minimum_gpu_vram_gb": "minimum_gpu_vram_gb",
      "requirements.minimum_memory_gb": "minimum_memory_gb",
      "requirements.storage_gb": "storage_gb",
      "requirements.wifi_required": "wifi_required",
      "requirements.case_size": "case_size",
    };
    const field = relaxation.field_path ? fieldMap[relaxation.field_path] : undefined;
    return field && relaxation.proposed_value !== undefined
      ? ({ ...defaultBuildFormValues, [field]: relaxation.proposed_value } as BuildFormValues)
      : defaultBuildFormValues;
  } catch {
    return defaultBuildFormValues;
  }
}

function FieldError({ id, message }: { id: string; message?: string }) {
  if (!message) return null;
  return (
    <p className="field-error" id={id}>
      <span aria-hidden="true">!</span>
      {message}
    </p>
  );
}

/**
 * Which fields belong to which tab, so a validation error surfaces as a count on
 * the tab that owns it rather than hiding inside a collapsed panel.
 */
const STEP_FIELDS = {
  budget: [
    "budget_sgd",
    "performance_target",
    "secondary_workload",
    "primary_weight_percent",
  ],
  requirements: ["minimum_gpu_vram_gb", "minimum_memory_gb", "storage_gb"],
  tuning: ["max_builds", "requested_profiles"],
} as const;

function stepErrorCount(errors: BuildFormErrors, fields: readonly string[]): string {
  const count = fields.filter((field) => Boolean(errors[field as keyof BuildFormErrors])).length;
  return count ? String(count) : "";
}

export function BuildForm() {
  const router = useRouter();
  const errorSummaryRef = useRef<HTMLDivElement>(null);
  const [values, setValues] = useState<BuildFormValues>(initialBuildFormValues);
  const [errors, setErrors] = useState<BuildFormErrors>({});
  const [submitting, setSubmitting] = useState(false);
  const [serverError, setServerError] = useState("");

  function setField<K extends keyof BuildFormValues>(field: K, value: BuildFormValues[K]) {
    setValues((current) => ({ ...current, [field]: value }));
    setErrors((current) => ({ ...current, [field]: undefined, form: undefined }));
    setServerError("");
  }

  function setMaximumBuilds(maxBuilds: number) {
    setValues((current) => ({
      ...current,
      max_builds: maxBuilds,
      requested_profiles: current.requested_profiles.slice(0, maxBuilds),
    }));
    setErrors((current) => ({
      ...current,
      max_builds: undefined,
      requested_profiles: undefined,
      form: undefined,
    }));
    setServerError("");
  }

  function setProfileSelected(profile: BuildProfile, selected: boolean) {
    setValues((current) => {
      const requestedProfiles = selected
        ? current.requested_profiles.includes(profile) ||
          current.requested_profiles.length >= current.max_builds
          ? current.requested_profiles
          : [...current.requested_profiles, profile]
        : current.requested_profiles.filter((candidate) => candidate !== profile);
      return { ...current, requested_profiles: requestedProfiles };
    });
    setErrors((current) => ({ ...current, requested_profiles: undefined, form: undefined }));
    setServerError("");
  }

  async function handleSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const nextErrors = validateBuildForm(values);
    if (Object.keys(nextErrors).length) {
      setErrors(nextErrors);
      window.requestAnimationFrame(() => errorSummaryRef.current?.focus());
      return;
    }

    setSubmitting(true);
    setServerError("");
    const request = toBuildRequest(values);
    try {
      const response = await generateBuilds(request);
      void trackInteraction({
        event_type: "search_submitted",
        session_id: getSessionId(),
      });
      router.push(`/recommendations/${encodeURIComponent(response.request_id)}`);
    } catch (error) {
      setServerError(
        error instanceof Error
          ? error.message
          : "The recommendation service could not generate builds. Try again.",
      );
      window.requestAnimationFrame(() => errorSummaryRef.current?.focus());
      setSubmitting(false);
    }
  }

  const secondaryWeight = values.secondary_workload === "none" ? 0 : 100 - values.primary_weight_percent;
  const errorMessages = Object.entries(errors).filter(([, message]) => Boolean(message));

  return (
    <form className="build-form" onSubmit={handleSubmit} noValidate>
      {(errorMessages.length > 0 || serverError) && (
        <div className="error-summary" role="alert" tabIndex={-1} ref={errorSummaryRef}>
          <strong>{serverError ? "We could not generate your builds." : "Check your build brief."}</strong>
          {serverError ? (
            <p>{serverError}</p>
          ) : (
            <ul>
              {errorMessages.map(([field, message]) => (
                <li key={field}>{message}</li>
              ))}
            </ul>
          )}
        </div>
      )}

      <div className="brief-layout">
        <div className="brief-layout__form">
          <Tabs
            label="Build request steps"
            className="form-tabs"
            items={[
              {
                id: "budget",
                label: "Budget & workload",
                hint: stepErrorCount(errors, STEP_FIELDS.budget) || undefined,
                content: (
          <fieldset className="form-section" disabled={submitting}>
            <legend>
              <span>01</span>
              Set the budget and workload
            </legend>
            <p className="section-intro">
              Every part is chosen to work with the rest of the machine.
            </p>

            <div className="field field--budget">
              <label htmlFor="budget-sgd">Total budget for new parts</label>
              <div className="money-input">
                <span aria-hidden="true">S$</span>
                <input
                  id="budget-sgd"
                  name="budget_sgd"
                  type="number"
                  inputMode="decimal"
                  min="1"
                  step="50"
                  value={values.budget_sgd}
                  onChange={(event) => setField("budget_sgd", Number(event.target.value))}
                  aria-invalid={Boolean(errors.budget_sgd)}
                  aria-describedby={errors.budget_sgd ? "budget-error" : "budget-help"}
                />
              </div>
              <p className="field-help" id="budget-help">
                Prices include the selected listing and shipping when available.
              </p>
              <FieldError id="budget-error" message={errors.budget_sgd} />
            </div>

            <div className="field-grid field-grid--two">
              <div className="field">
                <label htmlFor="primary-workload">Primary workload</label>
                <select
                  id="primary-workload"
                  value={values.primary_workload}
                  onChange={(event) =>
                    setField("primary_workload", event.target.value as WorkloadName)
                  }
                >
                  {workloadOptions.map(([value, label]) => (
                    <option key={value} value={value}>
                      {label}
                    </option>
                  ))}
                </select>
              </div>
              <div className="field">
                <label htmlFor="secondary-workload">Secondary workload</label>
                <select
                  id="secondary-workload"
                  value={values.secondary_workload}
                  onChange={(event) =>
                    setField(
                      "secondary_workload",
                      event.target.value as WorkloadName | "none",
                    )
                  }
                  aria-invalid={Boolean(errors.secondary_workload)}
                  aria-describedby={errors.secondary_workload ? "secondary-workload-error" : undefined}
                >
                  <option value="none">No secondary workload</option>
                  {workloadOptions.map(([value, label]) => (
                    <option key={value} value={value}>
                      {label}
                    </option>
                  ))}
                </select>
                <FieldError id="secondary-workload-error" message={errors.secondary_workload} />
              </div>
            </div>

            {values.secondary_workload !== "none" && (
              <div className="field workload-balance">
                <div className="field-label-row">
                  <label htmlFor="workload-balance">Workload balance</label>
                  <span>
                    {values.primary_weight_percent}% / {secondaryWeight}%
                  </span>
                </div>
                <input
                  id="workload-balance"
                  type="range"
                  min="10"
                  max="90"
                  step="10"
                  value={values.primary_weight_percent}
                  onChange={(event) =>
                    setField("primary_weight_percent", Number(event.target.value))
                  }
                  aria-describedby="workload-balance-help"
                />
                <div className="balance-labels" id="workload-balance-help">
                  <span>{workloadLabels[values.primary_workload]}</span>
                  <span>{workloadLabels[values.secondary_workload]}</span>
                </div>
                <div className="segmented-control" aria-label="Workload balance presets">
                  {[50, 60, 70, 80].map((weight) => (
                    <button
                      key={weight}
                      type="button"
                      className={values.primary_weight_percent === weight ? "is-active" : ""}
                      aria-pressed={values.primary_weight_percent === weight}
                      onClick={() => setField("primary_weight_percent", weight)}
                    >
                      {weight}/{100 - weight}
                    </button>
                  ))}
                </div>
              </div>
            )}

            <div className="field">
              <label htmlFor="performance-target">Performance target (optional)</label>
              <input
                id="performance-target"
                name="performance_target"
                type="text"
                maxLength={MAX_PERFORMANCE_TARGET_LENGTH}
                value={values.performance_target}
                placeholder="e.g. 120 FPS at 1440p high settings"
                onChange={(event) => setField("performance_target", event.target.value)}
                aria-invalid={Boolean(errors.performance_target)}
                aria-describedby={
                  errors.performance_target
                    ? "performance-target-help performance-target-error"
                    : "performance-target-help"
                }
              />
              <p className="field-help" id="performance-target-help">
                Something specific you want to hit, like 100 fps at 1440p. We will show the
                evidence we have for it.
              </p>
              <FieldError
                id="performance-target-error"
                message={errors.performance_target}
              />
            </div>
          </fieldset>
                ),
              },
              {
                id: "requirements",
                label: "Requirements",
                hint: stepErrorCount(errors, STEP_FIELDS.requirements) || undefined,
                content: (
          <fieldset className="form-section" disabled={submitting}>
            <legend>
              <span>02</span>
              Lock requirements
            </legend>
            <p className="section-intro">
              Builds that miss any of these are excluded.
            </p>
            <div className="field-grid field-grid--three">
              <div className="field">
                <label htmlFor="minimum-vram">Minimum GPU memory</label>
                <div className="unit-input">
                  <input
                    id="minimum-vram"
                    type="number"
                    min="0"
                    step="2"
                    value={values.minimum_gpu_vram_gb}
                    onChange={(event) =>
                      setField("minimum_gpu_vram_gb", Number(event.target.value))
                    }
                    aria-invalid={Boolean(errors.minimum_gpu_vram_gb)}
                    aria-describedby={errors.minimum_gpu_vram_gb ? "vram-error" : undefined}
                  />
                  <span>GB</span>
                </div>
                <FieldError id="vram-error" message={errors.minimum_gpu_vram_gb} />
              </div>
              <div className="field">
                <label htmlFor="minimum-memory">System memory</label>
                <div className="unit-input">
                  <input
                    id="minimum-memory"
                    type="number"
                    min="1"
                    step="8"
                    value={values.minimum_memory_gb}
                    onChange={(event) =>
                      setField("minimum_memory_gb", Number(event.target.value))
                    }
                    aria-invalid={Boolean(errors.minimum_memory_gb)}
                    aria-describedby={errors.minimum_memory_gb ? "memory-error" : undefined}
                  />
                  <span>GB</span>
                </div>
                <FieldError id="memory-error" message={errors.minimum_memory_gb} />
              </div>
              <div className="field">
                <label htmlFor="storage">Total storage</label>
                <div className="unit-input">
                  <input
                    id="storage"
                    type="number"
                    min="1"
                    step="500"
                    value={values.storage_gb}
                    onChange={(event) => setField("storage_gb", Number(event.target.value))}
                    aria-invalid={Boolean(errors.storage_gb)}
                    aria-describedby={errors.storage_gb ? "storage-error" : undefined}
                  />
                  <span>GB</span>
                </div>
                <FieldError id="storage-error" message={errors.storage_gb} />
              </div>
            </div>

            <div className="field-grid field-grid--three requirements-row">
              <div className="field">
                <label htmlFor="case-size">Case size</label>
                <select
                  id="case-size"
                  value={values.case_size}
                  onChange={(event) =>
                    setField("case_size", event.target.value as BuildFormValues["case_size"])
                  }
                >
                  <option value="small_form_factor">Small form factor</option>
                  <option value="mini_tower">Mini tower</option>
                  <option value="mid_tower">Mid tower</option>
                  <option value="full_tower">Full tower</option>
                </select>
              </div>
              <label className="check-card" htmlFor="wifi-required">
                <input
                  id="wifi-required"
                  type="checkbox"
                  checked={values.wifi_required}
                  onChange={(event) => setField("wifi_required", event.target.checked)}
                />
                <span className="check-card__mark" aria-hidden="true" />
                <span>
                  <strong>Wi-Fi required</strong>
                  <small>Motherboard or included adapter</small>
                </span>
              </label>
              <label className="check-card" htmlFor="in-stock-only">
                <input
                  id="in-stock-only"
                  type="checkbox"
                  checked={values.in_stock_only}
                  onChange={(event) => setField("in_stock_only", event.target.checked)}
                />
                <span className="check-card__mark" aria-hidden="true" />
                <span>
                  <strong>In-stock offers only</strong>
                  <small>Exclude listings last observed unavailable</small>
                </span>
              </label>
            </div>

            <ExistingProductPicker
              selected={values.existing_products}
              onChange={(products: ExistingProductInput[]) =>
                setField("existing_products", products)
              }
              disabled={submitting}
            />
          </fieldset>
                ),
              },
              {
                id: "tuning",
                label: "Tuning",
                hint: stepErrorCount(errors, STEP_FIELDS.tuning) || undefined,
                content: (
          <fieldset className="form-section" disabled={submitting}>
            <legend>
              <span>03</span>
              Tune the recommendation
            </legend>
            <p className="section-intro">
              Preferences order the results once compatibility is settled.
            </p>
            <div className="field-grid field-grid--three">
              <div className="field">
                <label htmlFor="noise">Noise preference</label>
                <select
                  id="noise"
                  value={values.noise}
                  onChange={(event) =>
                    setField("noise", event.target.value as BuildFormValues["noise"])
                  }
                >
                  <option value="low">Quiet preferred</option>
                  <option value="medium">Balanced</option>
                  <option value="any">No preference</option>
                </select>
              </div>
              <div className="field">
                <label htmlFor="efficiency">Power efficiency</label>
                <select
                  id="efficiency"
                  value={values.power_efficiency}
                  onChange={(event) =>
                    setField(
                      "power_efficiency",
                      event.target.value as BuildFormValues["power_efficiency"],
                    )
                  }
                >
                  <option value="low">Low priority</option>
                  <option value="medium">Balanced</option>
                  <option value="high">High priority</option>
                </select>
              </div>
              <div className="field">
                <label htmlFor="upgradeability">Upgradeability</label>
                <select
                  id="upgradeability"
                  value={values.upgradeability}
                  onChange={(event) =>
                    setField(
                      "upgradeability",
                      event.target.value as BuildFormValues["upgradeability"],
                    )
                  }
                >
                  <option value="low">Low priority</option>
                  <option value="medium">Balanced</option>
                  <option value="high">High priority</option>
                </select>
              </div>
            </div>

            <div className="build-output-controls">
              <div className="field">
                <label htmlFor="max-builds">Maximum build options</label>
                <select
                  id="max-builds"
                  value={values.max_builds}
                  onChange={(event) => setMaximumBuilds(Number(event.target.value))}
                  aria-invalid={Boolean(errors.max_builds)}
                  aria-describedby={errors.max_builds ? "max-builds-error" : "max-builds-help"}
                >
                  <option value="3">3 builds</option>
                  <option value="4">4 builds</option>
                  <option value="5">5 builds</option>
                </select>
                <p className="field-help" id="max-builds-help">
                  You may get fewer if your requirements leave fewer genuinely different builds.
                </p>
                <FieldError id="max-builds-error" message={errors.max_builds} />
              </div>
              <fieldset
                className="profile-picker"
                aria-describedby={errors.requested_profiles ? "profiles-error" : "profiles-help"}
              >
                <legend>Build profiles</legend>
                <p className="field-help" id="profiles-help">
                  Select up to {values.max_builds} objective profiles.
                </p>
                <div className="profile-picker__grid">
                  {profileOptions.map(([profile, label]) => {
                    const selected = values.requested_profiles.includes(profile);
                    const limitReached =
                      !selected && values.requested_profiles.length >= values.max_builds;
                    return (
                      <label key={profile}>
                        <input
                          type="checkbox"
                          value={profile}
                          checked={selected}
                          disabled={limitReached}
                          onChange={(event) => setProfileSelected(profile, event.target.checked)}
                        />
                        <span>{label}</span>
                      </label>
                    );
                  })}
                </div>
                <FieldError id="profiles-error" message={errors.requested_profiles} />
              </fieldset>
            </div>

            <details className="advanced-preferences">
              <summary>Brand preferences</summary>
              <div className="field-grid field-grid--two">
                <div className="field">
                  <label htmlFor="preferred-brands">Preferred brands</label>
                  <input
                    id="preferred-brands"
                    type="text"
                    value={values.preferred_brands}
                    placeholder="e.g. AMD, Fractal Design"
                    onChange={(event) => setField("preferred_brands", event.target.value)}
                  />
                  <p className="field-help">Comma separated. Used as a soft ranking signal.</p>
                </div>
                <div className="field">
                  <label htmlFor="excluded-brands">Excluded brands</label>
                  <input
                    id="excluded-brands"
                    type="text"
                    value={values.excluded_brands}
                    placeholder="e.g. Brand name"
                    onChange={(event) => setField("excluded_brands", event.target.value)}
                  />
                  <p className="field-help">Excluded brands are a hard constraint.</p>
                </div>
              </div>
              <FieldError id="form-error" message={errors.form} />
            </details>
          </fieldset>
                ),
              },
            ]}
          />
        </div>

        <aside className="brief-summary" aria-label="Build brief summary">
          <div className="brief-summary__eyebrow">
            <span aria-hidden="true">◎</span>
            Your build brief
          </div>
          <div className="brief-summary__budget">
            <small>New-parts budget</small>
            <strong>{formatSgd(values.budget_sgd)}</strong>
          </div>
          <dl>
            <div>
              <dt>Primary</dt>
              <dd>
                {workloadLabels[values.primary_workload]}
                {values.secondary_workload !== "none" && ` · ${values.primary_weight_percent}%`}
              </dd>
            </div>
            {values.secondary_workload !== "none" && (
              <div>
                <dt>Secondary</dt>
                <dd>
                  {workloadLabels[values.secondary_workload]} · {secondaryWeight}%
                </dd>
              </div>
            )}
            <div>
              <dt>Minimums</dt>
              <dd>
                {values.minimum_gpu_vram_gb || 0} GB VRAM · {values.minimum_memory_gb} GB RAM
              </dd>
            </div>
            <div>
              <dt>Storage</dt>
              <dd>{values.storage_gb.toLocaleString("en-SG")} GB</dd>
            </div>
            <div>
              <dt>Performance target</dt>
              <dd>{values.performance_target.trim() || "Not specified"}</dd>
            </div>
            <div>
              <dt>Retained parts</dt>
              <dd>{values.existing_products.length || "None"}</dd>
            </div>
            <div>
              <dt>Build options</dt>
              <dd>
                {values.requested_profiles.length} profile{values.requested_profiles.length === 1 ? "" : "s"}
                {` / up to ${values.max_builds}`}
              </dd>
            </div>
            <div>
              <dt>Availability</dt>
              <dd>
                {USING_DEMO_DATA
                  ? "Controlled demo catalogue"
                  : values.in_stock_only
                    ? "Observed in-stock only"
                    : "Include unavailable listings"}
              </dd>
            </div>
          </dl>
          <button
            className="button button--primary button--large generate-button"
            type="submit"
            disabled={submitting}
            data-testid="generate-builds"
          >
            {submitting ? (
              <>
                <span className="button-spinner" aria-hidden="true" />
                Solving the build…
              </>
            ) : (
              <>
                Generate ranked builds
                <span aria-hidden="true">→</span>
              </>
            )}
          </button>
          <p className="brief-summary__note">
            {USING_DEMO_DATA
              ? "Prices are real but dated — this demo isn't wired to live retailer stock."
              : "Compatibility is checked before anything is ranked."}
          </p>
          <div className="generation-status" aria-live="polite">
            {submitting && "Ranking candidates and solving compatibility constraints."}
          </div>
        </aside>
      </div>
    </form>
  );
}
