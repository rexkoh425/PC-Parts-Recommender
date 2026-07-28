import type {
  BuildProfile,
  BuildRequest,
  ExistingProductInput,
  WorkloadName,
} from "./types";

export const MAX_PERFORMANCE_TARGET_LENGTH = 200;

export interface BuildFormValues {
  budget_sgd: number;
  performance_target: string;
  primary_workload: WorkloadName;
  secondary_workload: WorkloadName | "none";
  primary_weight_percent: number;
  minimum_gpu_vram_gb: number;
  minimum_memory_gb: number;
  storage_gb: number;
  wifi_required: boolean;
  in_stock_only: boolean;
  case_size: "small_form_factor" | "mini_tower" | "mid_tower" | "full_tower";
  noise: "low" | "medium" | "any";
  upgradeability: "low" | "medium" | "high";
  power_efficiency: "low" | "medium" | "high";
  preferred_brands: string;
  excluded_brands: string;
  existing_products: ExistingProductInput[];
  max_builds: number;
  requested_profiles: BuildProfile[];
}

export type BuildFormErrors = Partial<Record<keyof BuildFormValues | "form", string>>;

export const defaultBuildFormValues: BuildFormValues = {
  // Raised from 2500 in Aug 2026: the DRAM/NAND shortage put roughly S$500
  // of memory and S$470 of storage into a mid-range build, so 2500 no longer
  // clears a complete one in this catalogue.
  budget_sgd: 3500,
  performance_target: "",
  primary_workload: "local_ai",
  secondary_workload: "gaming_1440p",
  primary_weight_percent: 60,
  minimum_gpu_vram_gb: 16,
  minimum_memory_gb: 32,
  storage_gb: 2000,
  wifi_required: true,
  in_stock_only: true,
  case_size: "mid_tower",
  noise: "low",
  upgradeability: "high",
  power_efficiency: "medium",
  preferred_brands: "",
  excluded_brands: "",
  existing_products: [],
  max_builds: 5,
  requested_profiles: [
    "best_overall",
    "best_value",
    "highest_performance",
    "most_upgradeable",
    "lowest_power",
  ],
};

function splitBrands(value: string): string[] {
  return [...new Set(value.split(",").map((brand) => brand.trim()).filter(Boolean))];
}

export function validateBuildForm(values: BuildFormValues): BuildFormErrors {
  const errors: BuildFormErrors = {};

  if (!Number.isFinite(values.budget_sgd) || values.budget_sgd <= 0) {
    errors.budget_sgd = "Enter a budget greater than S$0.";
  }
  if (values.performance_target.trim().length > MAX_PERFORMANCE_TARGET_LENGTH) {
    errors.performance_target = `Keep the performance target to ${MAX_PERFORMANCE_TARGET_LENGTH} characters or fewer.`;
  }
  if (values.secondary_workload !== "none" && values.secondary_workload === values.primary_workload) {
    errors.secondary_workload = "Choose a different secondary workload.";
  }
  if (
    values.secondary_workload !== "none" &&
    (!Number.isFinite(values.primary_weight_percent) ||
      values.primary_weight_percent < 10 ||
      values.primary_weight_percent > 90)
  ) {
    errors.primary_weight_percent = "Choose a primary workload weight from 10% to 90%.";
  }
  if (!Number.isFinite(values.minimum_gpu_vram_gb) || values.minimum_gpu_vram_gb < 0) {
    errors.minimum_gpu_vram_gb = "GPU memory cannot be negative.";
  }
  if (!Number.isFinite(values.minimum_memory_gb) || values.minimum_memory_gb <= 0) {
    errors.minimum_memory_gb = "System memory must be greater than zero.";
  }
  if (!Number.isFinite(values.storage_gb) || values.storage_gb <= 0) {
    errors.storage_gb = "Storage must be greater than zero.";
  }
  if (!Number.isInteger(values.max_builds) || values.max_builds < 1 || values.max_builds > 5) {
    errors.max_builds = "Choose between one and five build options.";
  }
  if (values.requested_profiles.length === 0) {
    errors.requested_profiles = "Choose at least one build profile.";
  } else if (new Set(values.requested_profiles).size !== values.requested_profiles.length) {
    errors.requested_profiles = "Build profiles must be unique.";
  } else if (values.requested_profiles.length > values.max_builds) {
    errors.requested_profiles = "Selected profiles cannot exceed the maximum build count.";
  }

  const preferred = splitBrands(values.preferred_brands).map((brand) => brand.toLowerCase());
  const excluded = new Set(splitBrands(values.excluded_brands).map((brand) => brand.toLowerCase()));
  const overlap = preferred.find((brand) => excluded.has(brand));
  if (overlap) {
    errors.form = `A brand cannot be both preferred and excluded: ${overlap}.`;
  }

  return errors;
}

export function toBuildRequest(values: BuildFormValues): BuildRequest {
  const workloads: BuildRequest["workloads"] = [];
  if (values.secondary_workload === "none") {
    workloads.push({ name: values.primary_workload, weight: 1 });
  } else {
    const primaryWeight = values.primary_weight_percent / 100;
    workloads.push({ name: values.primary_workload, weight: primaryWeight });
    workloads.push({
      name: values.secondary_workload,
      weight: Number((1 - primaryWeight).toFixed(4)),
    });
  }

  return {
    budget_sgd: values.budget_sgd,
    performance_target: values.performance_target.trim() || undefined,
    workloads,
    existing_products: values.existing_products,
    requirements: {
      minimum_gpu_vram_gb: values.minimum_gpu_vram_gb || undefined,
      minimum_memory_gb: values.minimum_memory_gb,
      storage_gb: values.storage_gb,
      wifi_required: values.wifi_required,
      case_size: values.case_size,
      in_stock_only: values.in_stock_only,
    },
    preferences: {
      noise: values.noise,
      upgradeability: values.upgradeability,
      power_efficiency: values.power_efficiency,
      preferred_brands: splitBrands(values.preferred_brands),
      excluded_brands: splitBrands(values.excluded_brands),
    },
    max_builds: values.max_builds,
    requested_profiles: values.requested_profiles,
  };
}
