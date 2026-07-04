"use client";

import { useRef, useState } from "react";
import { useRouter } from "next/navigation";
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
import type { BuildPreset, ExistingProductInput, WorkloadLabel } from "@/lib/types";
import { ExistingProductPicker } from "./existing-product-picker";

const workloadOptions = Object.entries(workloadLabels) as Array<[WorkloadLabel, string]>;
const profileOptions = Object.entries(profileLabels) as Array<[BuildPreset, string]>;

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

// TODO: rest of this component still to come.
