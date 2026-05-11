"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import {
  apiErrorEvidence,
  apiErrorRequestId,
  replaceComponent,
  searchProducts,
} from "@/lib/api";
import { categoryLabels, formatSgd } from "@/lib/format";
import { formatSignedDelta } from "@/lib/catalogue";
import {
  canApplyReplacementCandidate,
  firstApplicableCandidateId,
  productSearchItemToReplacementCandidate,
  replacementCandidateStatus,
  replacementStatusLabel,
} from "@/lib/replacement";
import { runtimeCapabilities } from "@/lib/runtime";
import type {
  BuildResult,
  ComponentKind,
  ReplacementOption,
  ReplacementRequest,
  ReplacementResponse,
} from "@/lib/types";
import { StatusPill } from "./status-pill";

export function ReplacementDrawer({
  build,
  category,
  onClose,
  onReplaced,
}: {
  build: BuildResult;
  category: ComponentKind;
  onClose(): void;
  onReplaced(response: ReplacementResponse): void;
}) {
  const dialogRef = useRef<HTMLDialogElement>(null);
  const current = build.components.find((component) => component.category === category);
  const seededAlternatives = useMemo(() => {
    const direct = current?.alternatives ?? [];
    const buildWide = (build.alternatives ?? []).filter((candidate) => candidate.category === category);
    return [...direct, ...buildWide].filter(
      (candidate, index, candidates) =>
        candidate.product_id !== current?.product_id &&
        candidates.findIndex((item) => item.product_id === candidate.product_id) === index,
    );
  }, [build.alternatives, category, current]);
  const [candidates, setCandidates] = useState<ReplacementOption[]>(seededAlternatives);
  const [selectedId, setSelectedId] = useState(firstApplicableCandidateId(seededAlternatives));
  const [loading, setLoading] = useState(seededAlternatives.length === 0);
  const [applying, setApplying] = useState(false);
  const [error, setError] = useState("");
  const [errorEvidence, setErrorEvidence] = useState<string[]>([]);
  const [errorRequestId, setErrorRequestId] = useState<string>();
  const [mode, setMode] = useState<ReplacementRequest["mode"]>("lock_other_components");

  useEffect(() => {
    const dialog = dialogRef.current;
    if (dialog && !dialog.open) dialog.showModal();
  }, []);

  useEffect(() => {
    if (seededAlternatives.length > 0) return;
    let active = true;
    const controller = new AbortController();
    searchProducts(
      { query: "", category, compatible_with_build_id: build.build_id, limit: 8 },
      { signal: controller.signal },
    )
      .then((response) => {
        if (!active) return;
        const mapped: ReplacementOption[] = response.products
          .filter((product) => product.product_id !== current?.product_id)
          .map(productSearchItemToReplacementCandidate);
        setCandidates(mapped);
        setSelectedId(firstApplicableCandidateId(mapped));
      })
      .catch(() => {
        if (active) setError("Compatible alternatives could not be loaded right now.");
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
      controller.abort();
    };
  }, [build.build_id, category, current?.product_id, seededAlternatives.length]);

  async function applyReplacement() {
    const selected = candidates.find((candidate) => candidate.product_id === selectedId);
    if (!selected || !canApplyReplacementCandidate(selected)) {
      setError("Choose an alternative that has passed compatibility screening.");
      return;
    }
    setApplying(true);
    setError("");
    setErrorEvidence([]);
    setErrorRequestId(undefined);
    try {
      const response = await replaceComponent(build.build_id, {
        category,
        replacement_product_id: selectedId,
        mode,
      });
      onReplaced(response);
      dialogRef.current?.close();
      onClose();
    } catch (requestError) {
      setErrorEvidence(apiErrorEvidence(requestError));
      setErrorRequestId(apiErrorRequestId(requestError));
      setError(
        requestError instanceof Error
          ? requestError.message
          : "This replacement could not produce a compatible build.",
      );
      setApplying(false);
    }
  }

  return (
    <dialog
      ref={dialogRef}
      className="replacement-dialog"
      aria-labelledby="replacement-title"
      onCancel={(event) => {
        event.preventDefault();
        onClose();
      }}
      onClose={onClose}
      data-testid="replacement-drawer"
    >
      <div className="replacement-dialog__header">
        <div>
          <p className="eyebrow">Compatible alternatives</p>
          <h2 id="replacement-title">Replace {categoryLabels[category].toLowerCase()}</h2>
          <p>
            Current: <strong>{current?.canonical_name ?? "Unknown component"}</strong>
            {current && !current.already_owned && <> · {formatSgd(current.price_sgd)}</>}
            {current?.already_owned && <> · already owned</>}
          </p>
        </div>
        <button
          type="button"
          className="icon-button"
          onClick={() => {
            dialogRef.current?.close();
            onClose();
          }}
          aria-label="Close replacement panel"
        >
          ×
        </button>
      </div>

      <div className="replacement-dialog__body">
        <fieldset className="replacement-mode" disabled={applying}>
          <legend>Replacement mode</legend>
          <label>
            <input
              type="radio"
              name="replacement-mode"
              checked={mode === "lock_other_components"}
              onChange={() => setMode("lock_other_components")}
            />
            <span>
              <strong>Swap this part only</strong>
              <small>Keep every other selected component fixed.</small>
            </span>
          </label>
          <label>
            <input
              type="radio"
              name="replacement-mode"
              checked={mode === "reoptimize_unlocked"}
              disabled={!runtimeCapabilities.reoptimizeUnlockedReplacement}
              onChange={() => setMode("reoptimize_unlocked")}
              aria-describedby="reoptimize-capability-note"
            />
            <span>
              <strong>Re-optimise supporting parts</strong>
              <small id="reoptimize-capability-note">
                {runtimeCapabilities.reoptimizeUnlockedReplacement
                  ? "Allow the optimiser to change other unlocked parts to preserve compatibility and value."
                  : "Unavailable in the controlled demo. Connect the catalogue-backed API to use this mode."}
              </small>
            </span>
          </label>
        </fieldset>

        {loading && (
          <div className="drawer-loading" role="status">
            <span className="button-spinner" aria-hidden="true" />
            Checking compatible catalogue options…
          </div>
        )}

        {!loading && candidates.length > 0 && (
          <fieldset className="candidate-list" disabled={applying}>
            <legend className="sr-only">Choose a replacement</legend>
            {candidates.map((candidate) => (
              (() => {
                const status = replacementCandidateStatus(candidate);
                const selectable = canApplyReplacementCandidate(candidate);
                return (
                  <label
                    className={`candidate-card ${selectedId === candidate.product_id ? "candidate-card--selected" : ""} ${!selectable ? "candidate-card--blocked" : ""}`}
                    key={candidate.product_id}
                  >
                    <input
                      type="radio"
                      name="replacement-product"
                      value={candidate.product_id}
                      checked={selectedId === candidate.product_id}
                      onChange={() => setSelectedId(candidate.product_id)}
                      disabled={!selectable}
                    />
                    <span className="candidate-card__main">
                      <strong>{candidate.canonical_name}</strong>
                      <small>{candidate.retailer ?? "Best observed listing"}</small>
                      {(candidate.reasons ?? []).length > 0 && (
                        <ul className="candidate-card__reasons">
                          {(candidate.reasons ?? []).slice(0, 2).map((reason) => (
                            <li key={reason}>{reason}</li>
                          ))}
                        </ul>
                      )}
                      {!selectable && (
                        <span>
                          {status === "fail"
                            ? "This candidate fails a compatibility rule and cannot be applied."
                            : "Compatibility has not been verified; this candidate cannot be applied."}
                        </span>
                      )}
                    </span>
                    <span className="candidate-card__aside">
                      <strong>
                        {candidate.price_sgd === null
                          ? "Price unavailable"
                          : formatSgd(candidate.price_sgd)}
                      </strong>
                      <StatusPill status={status} label={replacementStatusLabel(status)} />
                    </span>
                    <span className="candidate-card__deltas" aria-label="Replacement changes">
                      <span>
                        <small>Performance</small>
                        <strong>{formatSignedDelta(candidate.performance_delta, " pts")}</strong>
                      </span>
                      <span>
                        <small>Build price</small>
                        <strong>
                          {typeof candidate.price_delta_sgd === "number"
                            ? `${candidate.price_delta_sgd > 0 ? "+" : ""}${formatSgd(candidate.price_delta_sgd)}`
                            : "Not calculated"}
                        </strong>
                      </span>
                      <span>
                        <small>Peak power</small>
                        <strong>{formatSignedDelta(candidate.power_delta_w, " W")}</strong>
                      </span>
                    </span>
                  </label>
                );
              })()
            ))}
          </fieldset>
        )}

        {!loading && candidates.length === 0 && !error && (
          <div className="empty-drawer-state">
            <strong>No compatible alternatives found.</strong>
            <p>Try editing the build requirements or checking again after the catalogue refreshes.</p>
          </div>
        )}

        {error && (
          <div className="inline-error" role="alert">
            <strong>Replacement unavailable</strong>
            <p>{error}</p>
            {errorEvidence.length > 0 && (
              <ul>
                {errorEvidence.map((evidence) => (
                  <li key={evidence}>{evidence}</li>
                ))}
              </ul>
            )}
            {errorRequestId && (
              <small className="error-request-id">Request ID: {errorRequestId}</small>
            )}
          </div>
        )}
      </div>

      <div className="replacement-dialog__footer">
        <button
          type="button"
          className="button button--secondary"
          onClick={() => {
            dialogRef.current?.close();
            onClose();
          }}
          disabled={applying}
        >
          Cancel
        </button>
        <button
          type="button"
          className="button button--primary"
          onClick={applyReplacement}
          disabled={!selectedId || applying}
        >
          {applying ? (
            <>
              <span className="button-spinner" aria-hidden="true" />
              Rechecking build…
            </>
          ) : (
            "Apply replacement"
          )}
        </button>
      </div>
    </dialog>
  );
}
