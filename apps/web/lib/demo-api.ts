import type {
  BenchmarkObservation,
  BuildComponent,
  BuildProfile,
  BuildRequest,
  BuildSummary,
  CompatibilityCheck,
  CompatibilityCheckRequest,
  CompatibilityCheckResponse,
  ComponentCategory,
  FreshnessSummary,
  GenerateBuildsResponse,
  InteractionAccepted,
  InteractionEvent,
  PriceObservation,
  ProductBenchmarksResponse,
  ProductDetail,
  ProductPricesResponse,
  ProductReviewsResponse,
  ProductSearchRequest,
  ProductSearchResponse,
  ReplacementRequest,
  ReplacementResponse,
  WorkloadName,
} from "./types";

export const DEMO_DATA_VERSION = "portfolio-demo-2026-07-22";
export const DEMO_RANKING_MODEL = "deterministic-demo-ranker-v1";
export const DEMO_RULE_VERSION = "compat-demo-v1";
export const DEMO_SOLVER_VERSION = "browser-demo-v1";

interface DemoProduct {
  product_id: string;
  category: ComponentCategory;
  canonical_name: string;
  brand: string;
  model: string;
  price_sgd: number;
  performance: number;
  power_w: number;
  vram_gb?: number;
  memory_gb?: number;
  storage_gb?: number;
  case_size?: "mini_tower" | "mid_tower";
  attributes: Record<string, unknown>;
}

interface DemoTemplate {
  profile: BuildProfile;
  product_ids: string[];
  overall_score: number;
  value_score: number;
  upgradeability_score: number;
  efficiency_score: number;
}

const products: DemoProduct[] = [
  {
    product_id: "cpu-amd-7600",
    category: "cpu",
    canonical_name: "AMD Ryzen 5 7600",
    brand: "AMD",
    model: "Ryzen 5 7600",
    price_sgd: 249,
    performance: 73,
    power_w: 88,
    attributes: { socket: "AM5", cores: 6, threads: 12 },
  },
  {
    product_id: "cpu-amd-7700",
    category: "cpu",
    canonical_name: "AMD Ryzen 7 7700",
    brand: "AMD",
    model: "Ryzen 7 7700",
    price_sgd: 389,
    performance: 84,
    power_w: 115,
    attributes: { socket: "AM5", cores: 8, threads: 16 },
  },
  {
    product_id: "cpu-amd-7900",
    category: "cpu",
    canonical_name: "AMD Ryzen 9 7900",
    brand: "AMD",
    model: "Ryzen 9 7900",
    price_sgd: 529,
    performance: 94,
    power_w: 165,
    attributes: { socket: "AM5", cores: 12, threads: 24 },
  },
  {
    product_id: "gpu-rtx-5060ti-16",
    category: "gpu",
    canonical_name: "NVIDIA GeForce RTX 5060 Ti 16 GB",
    brand: "NVIDIA",
    model: "GeForce RTX 5060 Ti 16 GB",
    price_sgd: 699,
    performance: 78,
    power_w: 180,
    vram_gb: 16,
    attributes: { vram_gb: 16, length_mm: 247, slot_width: 2.5 },
  },
  {
    product_id: "gpu-rx-7800xt-16",
    category: "gpu",
    canonical_name: "AMD Radeon RX 7800 XT 16 GB",
    brand: "AMD",
    model: "Radeon RX 7800 XT 16 GB",
    price_sgd: 749,
    performance: 82,
    power_w: 263,
    vram_gb: 16,
    attributes: { vram_gb: 16, length_mm: 280, slot_width: 2.5 },
  },
  {
    product_id: "gpu-rtx-4070tis-16",
    category: "gpu",
    canonical_name: "NVIDIA GeForce RTX 4070 Ti SUPER 16 GB",
    brand: "NVIDIA",
    model: "GeForce RTX 4070 Ti SUPER 16 GB",
    price_sgd: 1099,
    performance: 94,
    power_w: 285,
    vram_gb: 16,
    attributes: { vram_gb: 16, length_mm: 305, slot_width: 3 },
  },
  {
    product_id: "mb-b650m-wifi",
    category: "motherboard",
    canonical_name: "B650M Wi-Fi DDR5 Motherboard",
    brand: "DemoBoard",
    model: "B650M Wi-Fi",
    price_sgd: 219,
    performance: 72,
    power_w: 45,
    attributes: { socket: "AM5", chipset: "B650", memory_type: "DDR5", wifi: true },
  },
  {
    product_id: "mb-b650-atx-wifi",
    category: "motherboard",
    canonical_name: "B650 ATX Wi-Fi DDR5 Motherboard",
    brand: "DemoBoard",
    model: "B650 ATX Wi-Fi",
    price_sgd: 279,
    performance: 82,
    power_w: 50,
    attributes: { socket: "AM5", chipset: "B650", memory_type: "DDR5", wifi: true },
  },
  {
    product_id: "mb-x670-atx-wifi",
    category: "motherboard",
    canonical_name: "X670 ATX Wi-Fi DDR5 Motherboard",
    brand: "DemoBoard",
    model: "X670 ATX Wi-Fi",
    price_sgd: 389,
    performance: 92,
    power_w: 55,
    attributes: { socket: "AM5", chipset: "X670", memory_type: "DDR5", wifi: true },
  },
  {
    product_id: "mem-ddr5-32-5600",
    category: "memory",
    canonical_name: "32 GB DDR5-5600 Memory Kit",
    brand: "DemoMemory",
    model: "32 GB DDR5-5600",
    price_sgd: 129,
    performance: 74,
    power_w: 8,
    memory_gb: 32,
    attributes: { memory_type: "DDR5", capacity_gb: 32, module_count: 2 },
  },
  {
    product_id: "mem-ddr5-32-6000",
    category: "memory",
    canonical_name: "32 GB DDR5-6000 Low-Latency Memory Kit",
    brand: "DemoMemory",
    model: "32 GB DDR5-6000",
    price_sgd: 149,
    performance: 82,
    power_w: 9,
    memory_gb: 32,
    attributes: { memory_type: "DDR5", capacity_gb: 32, module_count: 2 },
  },
  {
    product_id: "mem-ddr5-64-6000",
    category: "memory",
    canonical_name: "64 GB DDR5-6000 Memory Kit",
    brand: "DemoMemory",
    model: "64 GB DDR5-6000",
    price_sgd: 269,
    performance: 94,
    power_w: 12,
    memory_gb: 64,
    attributes: { memory_type: "DDR5", capacity_gb: 64, module_count: 2 },
  },
  {
    product_id: "ssd-nvme-2tb-value",
    category: "storage",
    canonical_name: "2 TB PCIe 4.0 NVMe SSD",
    brand: "DemoStorage",
    model: "2 TB NVMe Value",
    price_sgd: 139,
    performance: 76,
    power_w: 6,
    storage_gb: 2000,
    attributes: { capacity_gb: 2000, interface: "NVMe PCIe 4.0" },
  },
  {
    product_id: "ssd-nvme-2tb-fast",
    category: "storage",
    canonical_name: "2 TB High-Performance PCIe 4.0 NVMe SSD",
    brand: "DemoStorage",
    model: "2 TB NVMe Performance",
    price_sgd: 169,
    performance: 91,
    power_w: 7,
    storage_gb: 2000,
    attributes: { capacity_gb: 2000, interface: "NVMe PCIe 4.0" },
  },
  {
    product_id: "psu-750-gold",
    category: "psu",
    canonical_name: "750 W 80 Plus Gold Modular PSU",
    brand: "DemoPower",
    model: "750 W Gold",
    price_sgd: 159,
    performance: 81,
    power_w: 0,
    attributes: { wattage: 750, efficiency_rating: "80 Plus Gold" },
  },
  {
    product_id: "psu-850-gold",
    category: "psu",
    canonical_name: "850 W 80 Plus Gold ATX 3.0 Modular PSU",
    brand: "DemoPower",
    model: "850 W Gold ATX 3.0",
    price_sgd: 189,
    performance: 90,
    power_w: 0,
    attributes: { wattage: 850, efficiency_rating: "80 Plus Gold" },
  },
  {
    product_id: "cooler-single-tower",
    category: "cooler",
    canonical_name: "120 mm Single-Tower CPU Cooler",
    brand: "DemoCooling",
    model: "Single Tower 120",
    price_sgd: 59,
    performance: 73,
    power_w: 4,
    attributes: { supported_sockets: ["AM5"], height_mm: 154 },
  },
  {
    product_id: "cooler-dual-tower",
    category: "cooler",
    canonical_name: "120 mm Dual-Tower CPU Cooler",
    brand: "DemoCooling",
    model: "Dual Tower 120",
    price_sgd: 79,
    performance: 88,
    power_w: 6,
    attributes: { supported_sockets: ["AM5"], height_mm: 157 },
  },
  {
    product_id: "case-matx-air",
    category: "case",
    canonical_name: "Airflow Micro-ATX Mini Tower Case",
    brand: "DemoCase",
    model: "mATX Air",
    price_sgd: 99,
    performance: 75,
    power_w: 0,
    case_size: "mini_tower",
    attributes: { maximum_gpu_length_mm: 330, maximum_cooler_height_mm: 165 },
  },
  {
    product_id: "case-atx-air",
    category: "case",
    canonical_name: "Airflow ATX Mid-Tower Case",
    brand: "DemoCase",
    model: "ATX Air",
    price_sgd: 139,
    performance: 84,
    power_w: 0,
    case_size: "mid_tower",
    attributes: { maximum_gpu_length_mm: 380, maximum_cooler_height_mm: 175 },
  },
  {
    product_id: "case-atx-quiet",
    category: "case",
    canonical_name: "Dampened ATX Quiet Mid-Tower Case",
    brand: "DemoCase",
    model: "ATX Quiet",
    price_sgd: 159,
    performance: 82,
    power_w: 0,
    case_size: "mid_tower",
    attributes: { maximum_gpu_length_mm: 360, maximum_cooler_height_mm: 170 },
  },
];

const byId = new Map(products.map((product) => [product.product_id, product]));

const templates: DemoTemplate[] = [
  {
    profile: "best_overall",
    product_ids: [
      "cpu-amd-7700",
      "gpu-rtx-5060ti-16",
      "mb-b650-atx-wifi",
      "mem-ddr5-32-6000",
      "ssd-nvme-2tb-fast",
      "psu-750-gold",
      "cooler-dual-tower",
      "case-atx-quiet",
    ],
    overall_score: 88,
    value_score: 86,
    upgradeability_score: 88,
    efficiency_score: 84,
  },
  {
    profile: "best_value",
    product_ids: [
      "cpu-amd-7600",
      "gpu-rtx-5060ti-16",
      "mb-b650m-wifi",
      "mem-ddr5-32-5600",
      "ssd-nvme-2tb-value",
      "psu-750-gold",
      "cooler-single-tower",
      "case-matx-air",
    ],
    overall_score: 81,
    value_score: 94,
    upgradeability_score: 79,
    efficiency_score: 91,
  },
  {
    profile: "highest_performance",
    product_ids: [
      "cpu-amd-7700",
      "gpu-rtx-4070tis-16",
      "mb-b650m-wifi",
      "mem-ddr5-32-6000",
      "ssd-nvme-2tb-value",
      "psu-850-gold",
      "cooler-dual-tower",
      "case-atx-air",
    ],
    overall_score: 94,
    value_score: 82,
    upgradeability_score: 83,
    efficiency_score: 80,
  },
  {
    profile: "most_upgradeable",
    product_ids: [
      "cpu-amd-7600",
      "gpu-rx-7800xt-16",
      "mb-x670-atx-wifi",
      "mem-ddr5-32-6000",
      "ssd-nvme-2tb-fast",
      "psu-850-gold",
      "cooler-dual-tower",
      "case-atx-air",
    ],
    overall_score: 86,
    value_score: 83,
    upgradeability_score: 96,
    efficiency_score: 78,
  },
  {
    profile: "lowest_power",
    product_ids: [
      "cpu-amd-7600",
      "gpu-rtx-5060ti-16",
      "mb-b650m-wifi",
      "mem-ddr5-32-6000",
      "ssd-nvme-2tb-fast",
      "psu-750-gold",
      "cooler-dual-tower",
      "case-matx-air",
    ],
    overall_score: 83,
    value_score: 89,
    upgradeability_score: 82,
    efficiency_score: 96,
  },
];

const profileExplanation: Record<BuildProfile, string> = {
  best_overall: "Balances workload fit, value, efficiency, and future flexibility.",
  best_value: "Keeps the strongest workload-per-dollar trade-off in this controlled catalogue.",
  highest_performance: "Allocates more of the budget to the highest relative CPU and GPU scores.",
  most_upgradeable: "Prioritises motherboard expansion and power headroom for later upgrades.",
  lowest_power: "Favours the lowest estimated peak load while preserving the hard requirements.",
};

const categoryOrder: ComponentCategory[] = [
  "cpu",
  "gpu",
  "motherboard",
  "memory",
  "storage",
  "psu",
  "cooler",
  "case",
];

function randomId(prefix: string): string {
  const suffix = globalThis.crypto?.randomUUID?.() ?? `${Date.now()}-${Math.random()}`;
  return `${prefix}_${suffix}`;
}

function stableDemoSearchId(request: ProductSearchRequest): string {
  const identity = JSON.stringify({
    query: request.query.trim().toLowerCase().replace(/\s+/g, " "),
    category: request.category ?? null,
    compatible_with_build_id: request.compatible_with_build_id ?? null,
    brand: request.brand?.trim().toLowerCase() ?? null,
    in_stock_only: request.in_stock_only ?? true,
    data_version: DEMO_DATA_VERSION,
    retrieval_model: "controlled-demo-search-v1",
  });
  let first = 0x811c9dc5;
  let second = 0x9e3779b9;
  for (let index = 0; index < identity.length; index += 1) {
    const code = identity.charCodeAt(index);
    first = Math.imul(first ^ code, 0x01000193);
    second = Math.imul(second ^ code, 0x85ebca6b);
  }
  return `search_demo_${(first >>> 0).toString(16).padStart(8, "0")}${(second >>> 0).toString(16).padStart(8, "0")}`;
}

function scoreWorkload(
  workload: WorkloadName,
  selected: Map<ComponentCategory, DemoProduct>,
): number {
  const cpu = selected.get("cpu")?.performance ?? 0;
  const gpu = selected.get("gpu")?.performance ?? 0;
  const storage = selected.get("storage")?.performance ?? 0;
  if (workload.startsWith("gaming")) return 0.8 * gpu + 0.2 * cpu;
  if (workload === "local_ai") return 0.9 * gpu + 0.1 * cpu;
  if (workload === "software_development") return 0.75 * cpu + 0.25 * storage;
  return 0.55 * gpu + 0.45 * cpu;
}

function caseFits(requested: BuildRequest["requirements"]["case_size"], actual?: string): boolean {
  if (!requested) return true;
  const sizes = ["small_form_factor", "mini_tower", "mid_tower", "full_tower"];
  return sizes.indexOf(actual ?? "mid_tower") <= sizes.indexOf(requested);
}

function compatibilityChecks(selected: Map<ComponentCategory, DemoProduct>): CompatibilityCheck[] {
  const gpu = selected.get("gpu");
  const cpu = selected.get("cpu");
  const psu = selected.get("psu");
  const estimated = [...selected.values()].reduce((total, product) => total + product.power_w, 100);
  return [
    {
      rule_id: "compat.cpu_socket.v1",
      status: "pass",
      message: `${cpu?.canonical_name ?? "CPU"} and the motherboard use the AM5 socket.`,
      affected_categories: ["cpu", "motherboard"],
    },
    {
      rule_id: "compat.memory_generation.v1",
      status: "pass",
      message: "The selected memory kit and motherboard both use DDR5.",
      affected_categories: ["memory", "motherboard"],
    },
    {
      rule_id: "compat.gpu_clearance.v1",
      status: "pass",
      message: `${gpu?.canonical_name ?? "GPU"} fits the selected case clearance in the demo fixture.`,
      affected_categories: ["gpu", "case"],
    },
    {
      rule_id: "compat.psu_headroom.v1",
      status: "pass",
      message: `${String(psu?.attributes.wattage ?? "Selected")} W provides headroom over the ${Math.round(estimated)} W estimated peak.`,
      affected_categories: ["psu", "cpu", "gpu"],
    },
    {
      rule_id: "compat.storage_interface.v1",
      status: "pass",
      message: "The NVMe storage interface is supported by the selected motherboard.",
      affected_categories: ["storage", "motherboard"],
    },
  ];
}

function componentFromProduct(
  product: DemoProduct,
  request: BuildRequest,
  alreadyOwned: boolean,
): BuildComponent {
  const alternatives = products
    .filter((candidate) => candidate.category === product.category && candidate.product_id !== product.product_id)
    .slice(0, 3)
    .map((candidate) => ({
      product_id: candidate.product_id,
      canonical_name: candidate.canonical_name,
      category: candidate.category,
      price_sgd: candidate.price_sgd,
      retailer: "Controlled demo catalogue",
      price_delta_sgd: candidate.price_sgd - product.price_sgd,
      performance_delta: candidate.performance - product.performance,
      power_delta_w: candidate.power_w - product.power_w,
      compatibility_status: "pass" as const,
      reasons: ["Screened against the other seven fixed demo components."],
    }));
  const preferred = new Set(request.preferences.preferred_brands.map((brand) => brand.toLowerCase()));
  const reasons = [`Meets the hard ${product.category.replaceAll("_", " ")} requirements.`];
  if (preferred.has(product.brand.toLowerCase())) reasons.push("Matches a preferred brand.");
  return {
    category: product.category,
    product_id: product.product_id,
    listing_id: `demo-listing-${product.product_id}`,
    canonical_name: product.canonical_name,
    brand: product.brand,
    retailer: "Controlled demo catalogue",
    price_sgd: alreadyOwned ? 0 : product.price_sgd,
    already_owned: alreadyOwned,
    component_score: product.performance,
    selection_reasons: reasons,
    performance_signals: request.workloads.map((workload) => ({
      workload: workload.name,
      metric: "deterministic relative component score",
      value: product.performance,
      unit: "relative index",
      basis: "relative",
      confidence: "low",
      decision: "deterministic_baseline",
      model_version: DEMO_RANKING_MODEL,
    })),
    alternatives,
  };
}

function buildFromTemplate(
  template: DemoTemplate,
  request: BuildRequest,
  requestId: string,
  generatedAt: string,
): BuildSummary | undefined {
  const selected = new Map<ComponentCategory, DemoProduct>();
  for (const productId of template.product_ids) {
    const product = byId.get(productId);
    if (product) selected.set(product.category, product);
  }

  const minimumMemory = request.requirements.minimum_memory_gb ?? 0;
  if (minimumMemory > 64 || (request.requirements.minimum_gpu_vram_gb ?? 0) > 16) return undefined;
  if ((request.requirements.storage_gb ?? 0) > 2000) return undefined;
  if (minimumMemory > (selected.get("memory")?.memory_gb ?? 0)) {
    const memory = byId.get("mem-ddr5-64-6000");
    if (memory) selected.set("memory", memory);
  }

  for (const existing of request.existing_products) {
    const retained = byId.get(existing.product_id);
    if (retained) selected.set(existing.category, retained);
  }

  const selectedCase = selected.get("case");
  if (!caseFits(request.requirements.case_size, selectedCase?.case_size)) return undefined;

  const excluded = new Set(request.preferences.excluded_brands.map((brand) => brand.toLowerCase()));
  if ([...selected.values()].some((product) => excluded.has(product.brand.toLowerCase()))) {
    return undefined;
  }

  const retainedIds = new Set(request.existing_products.map((product) => product.product_id));
  const total = [...selected.values()].reduce(
    (sum, product) => sum + (retainedIds.has(product.product_id) ? 0 : product.price_sgd),
    0,
  );
  if (total > request.budget_sgd || selected.size !== categoryOrder.length) return undefined;

  const workloadScores = Object.fromEntries(
    request.workloads.map((workload) => [
      workload.name,
      Number(scoreWorkload(workload.name, selected).toFixed(1)),
    ]),
  );
  const checks = compatibilityChecks(selected);
  const components = categoryOrder.map((category) => {
    const product = selected.get(category);
    if (!product) throw new Error(`Demo template is missing ${category}.`);
    return componentFromProduct(product, request, retainedIds.has(product.product_id));
  });
  const estimatedPeak = [...selected.values()].reduce((sum, product) => sum + product.power_w, 100);
  return {
    build_id: `${requestId}-${template.profile}`,
    request_id: requestId,
    profile: template.profile,
    total_price_sgd: total,
    overall_score: template.overall_score,
    value_score: template.value_score,
    upgradeability_score: template.upgradeability_score,
    efficiency_score: template.efficiency_score,
    estimated_peak_power_w: estimatedPeak,
    workload_scores: workloadScores,
    compatibility_status: "pass",
    components,
    compatibility_checks: checks,
    warnings: [],
    explanation: [
      { kind: "performance", text: profileExplanation[template.profile] },
      {
        kind: "compatibility",
        text: "Every displayed hard rule passed against the controlled demo attributes.",
      },
      {
        kind: "price",
        text: "Prices are illustrative SGD demo values, not live retailer quotes.",
      },
    ],
    generated_at: generatedAt,
    data_version: DEMO_DATA_VERSION,
    ranking_model: DEMO_RANKING_MODEL,
    rule_version: DEMO_RULE_VERSION,
    solver_version: DEMO_SOLVER_VERSION,
    solver_status: "FEASIBLE_DEMO",
  };
}

export function generateDemoBuilds(request: BuildRequest): GenerateBuildsResponse {
  const requestId = randomId("demo-request");
  const generatedAt = new Date().toISOString();
  const requestedTemplates = request.requested_profiles
    ? request.requested_profiles
        .map((profile) => templates.find((template) => template.profile === profile))
        .filter((template): template is DemoTemplate => Boolean(template))
    : templates;
  const builds = requestedTemplates
    .map((template) => buildFromTemplate(template, request, requestId, generatedAt))
    .filter((build): build is BuildSummary => Boolean(build))
    .slice(0, request.max_builds ?? 5);

  if (builds.length === 0) {
    return {
      request_id: requestId,
      status: "infeasible",
      generated_at: generatedAt,
      data_version: DEMO_DATA_VERSION,
      ranking_model: DEMO_RANKING_MODEL,
      rule_version: DEMO_RULE_VERSION,
      solver_version: DEMO_SOLVER_VERSION,
      solver_status: "INFEASIBLE",
      builds: [],
      request,
      infeasibility: {
        reasons: [
          {
            code: "demo_catalogue_exhausted",
            message:
              "No controlled demo template satisfies the budget, memory, storage, brand, and case constraints together.",
          },
        ],
        suggested_relaxations: [
          {
            field_path: "budget_sgd",
            current_value: request.budget_sgd,
            proposed_value: Math.max(2200, request.budget_sgd + 400),
            expected_effect: "Expands the illustrative demo template set.",
          },
        ],
      },
    };
  }

  return {
    request_id: requestId,
    status: builds.length >= 3 ? "complete" : "partial",
    generated_at: generatedAt,
    data_version: DEMO_DATA_VERSION,
    ranking_model: DEMO_RANKING_MODEL,
    rule_version: DEMO_RULE_VERSION,
    solver_version: DEMO_SOLVER_VERSION,
    solver_status: "FEASIBLE_DEMO",
    builds,
    request,
  };
}

export function searchDemoProducts(request: ProductSearchRequest): ProductSearchResponse {
  const tokens = request.query.trim().toLowerCase().split(/\s+/).filter(Boolean);
  const queryMatches = products
    .filter((product) => {
      const haystack = `${product.canonical_name} ${product.brand} ${product.model}`.toLowerCase();
      return tokens.every((token) => haystack.includes(token));
    });
  const categoryMatches = queryMatches.filter(
    (product) => !request.category || product.category === request.category,
  );
  const filtered = categoryMatches.filter(
    (product) => !request.brand || product.brand.toLowerCase() === request.brand.toLowerCase(),
  );
  const pageSize = Math.max(1, Math.min(request.page_size ?? request.limit ?? 20, 100));
  const requestedPage = Number.isFinite(request.page) ? Math.floor(request.page as number) : 1;
  const page = Math.max(1, requestedPage);
  const totalPages = Math.max(1, Math.ceil(filtered.length / pageSize));
  const offset = (page - 1) * pageSize;
  const matches = filtered
    .slice(offset, offset + pageSize)
    .map((product) => ({
      product_id: product.product_id,
      category: product.category,
      canonical_name: product.canonical_name,
      brand: product.brand,
      model: product.model,
      lowest_price_sgd: product.price_sgd,
      stock_status: "demo_only",
      compatibility_status: request.compatible_with_build_id ? ("pass" as const) : null,
    }));
  return {
    query_id: stableDemoSearchId(request),
    products: matches,
    total: filtered.length,
    filtered_incompatible: 0,
    filtered_unknown: 0,
    data_version: DEMO_DATA_VERSION,
    retrieval_model: "controlled-demo-search-v1",
    facets: {
      categories: Array.from(
        queryMatches.reduce((counts, product) => {
          counts.set(product.category, (counts.get(product.category) ?? 0) + 1);
          return counts;
        }, new Map<ComponentCategory, number>()),
      ).map(([value, count]) => ({ value, count })),
      brands: Array.from(
        categoryMatches.reduce((counts, product) => {
          counts.set(product.brand, (counts.get(product.brand) ?? 0) + 1);
          return counts;
        }, new Map<string, number>()),
      )
        .map(([value, count]) => ({ value, count }))
        .sort((left, right) => left.value.localeCompare(right.value)),
    },
    pagination: {
      page,
      page_size: pageSize,
      total_pages: totalPages,
      has_previous: page > 1,
      has_next: page < totalPages,
    },
    coverage: {
      canonical_products: products.length,
      retailer_listings: null,
      source_count: null,
      category_count: new Set(products.map((product) => product.category)).size,
        as_of: "2026-07-22T00:00:00.000Z",
        scope_label: "Controlled illustrative demo",
        source_attributions: [],
    },
  };
}

function requireDemoProduct(productId: string): DemoProduct {
  const product = byId.get(productId);
  if (!product) throw new Error("This product is not present in the controlled public demo.");
  return product;
}

export function getDemoProduct(productId: string): ProductDetail {
  const product = requireDemoProduct(productId);
  return {
    product_id: product.product_id,
    category: product.category,
    canonical_name: product.canonical_name,
    brand: product.brand,
    model: product.model,
    lowest_price_sgd: product.price_sgd,
    stock_status: "demo_only",
    compatibility_status: null,
    manufacturer_part_number: null,
    attributes: product.attributes,
      source_confidence: null,
      source_url: null,
      source_attributions: [],
      updated_at: "2026-07-22T00:00:00.000Z",
    data_version: DEMO_DATA_VERSION,
  };
}

export function getDemoPrices(productId: string): ProductPricesResponse {
  const product = requireDemoProduct(productId);
  const observation: PriceObservation = {
    listing_id: `demo-listing-${product.product_id}`,
    retailer: "Controlled demo catalogue",
    observed_at: "2026-07-22T00:00:00.000Z",
    base_price_sgd: product.price_sgd,
    shipping_price_sgd: 0,
    stock_status: "demo_only",
    condition: "demo_only",
    current_offer_eligible: false,
    listing_url: null,
  };
  return {
    product_id: productId,
    current_lowest_price_sgd: product.price_sgd,
    observations: [observation],
    price_intelligence: {
      basis: "descriptive_observed_history",
      currency: "SGD",
      as_of: observation.observed_at,
      current_delivered_price_sgd: product.price_sgd,
      median_30d_sgd: product.price_sgd,
      median_90d_sgd: product.price_sgd,
      percentile_90d: null,
      recent_low_90d_sgd: product.price_sgd,
      volatility_90d_pct: null,
      current_seller_count: 1,
      seller_trend: "insufficient_history",
      stock_trend: "insufficient_history",
      history_days_30d: 1,
      history_days_90d: 1,
      history_sufficient: false,
      labels: ["Insufficient price history"],
      anomalies: [],
      observations_analyzed: 1,
      analysis_truncated: false,
    },
    data_version: DEMO_DATA_VERSION,
  };
}

export function getDemoBenchmarks(productId: string): ProductBenchmarksResponse {
  const product = requireDemoProduct(productId);
  const benchmark: BenchmarkObservation = {
    benchmark_name: "Controlled relative demo index",
    workload: "portfolio_demo",
    score: product.performance,
    unit: "relative index",
    higher_is_better: true,
    basis: "predicted",
    model_version: DEMO_RANKING_MODEL,
    source_url: null,
    observed_at: null,
  };
  return {
    product_id: productId,
    benchmarks: [benchmark],
    data_version: DEMO_DATA_VERSION,
    performance_model_version: DEMO_RANKING_MODEL,
  };
}

export function getDemoReviews(productId: string): ProductReviewsResponse {
  requireDemoProduct(productId);
  return { product_id: productId, evidence: [], data_version: DEMO_DATA_VERSION };
}

export function checkDemoCompatibility(
  request: CompatibilityCheckRequest,
): CompatibilityCheckResponse {
  void request;
  return {
    status: "unknown",
    is_feasible: false,
    checks: [
      {
        rule_id: "compat.demo.not_authoritative.v1",
        status: "unknown",
        message:
          "Arbitrary compatibility checks require the full rule service; the public demo does not manufacture a pass.",
      },
    ],
    rule_version: DEMO_RULE_VERSION,
    data_version: DEMO_DATA_VERSION,
  };
}

export function replaceDemoComponent(
  build: BuildSummary,
  request: ReplacementRequest,
): ReplacementResponse {
  const replacement = requireDemoProduct(request.replacement_product_id);
  if (replacement.category !== request.category) {
    throw new Error("The replacement category does not match the selected component.");
  }
  const previous = build.components.find((component) => component.category === request.category);
  if (!previous) throw new Error("The selected build does not contain that component category.");
  const nextComponent: BuildComponent = {
    ...previous,
    product_id: replacement.product_id,
    canonical_name: replacement.canonical_name,
    brand: replacement.brand,
    price_sgd: replacement.price_sgd,
    component_score: replacement.performance,
    selection_reasons: ["Applied after compatibility screening against the fixed demo build."],
  };
  const priceDelta = replacement.price_sgd - previous.price_sgd;
  const nextBuild: BuildSummary = {
    ...build,
    build_id: `${build.build_id}-swap-${replacement.product_id}`,
    total_price_sgd: build.total_price_sgd + priceDelta,
    components: build.components.map((component) =>
      component.category === request.category ? nextComponent : component,
    ),
    generated_at: new Date().toISOString(),
  };
  return {
    build: nextBuild,
    changed_categories: [request.category],
    price_delta_sgd: priceDelta,
    workload_score_deltas: {},
    new_warnings: [],
    data_version: DEMO_DATA_VERSION,
    ranking_model: DEMO_RANKING_MODEL,
    rule_version: DEMO_RULE_VERSION,
    solver_version: DEMO_SOLVER_VERSION,
  };
}

export function getDemoFreshness(): FreshnessSummary {
  return {
    data_version: DEMO_DATA_VERSION,
    status: "degraded",
    last_catalog_update: "2026-07-22T00:00:00.000Z",
    prices_updated_at: "2026-07-22T00:00:00.000Z",
    stale_after_hours: 24,
    source_count: 1,
    product_count: products.length,
    listing_count: products.length,
    production_ready: false,
    readiness_blockers: [
      "The public demo is not connected to a measured market-data release.",
    ],
    catalogue_readiness: null,
  };
}

export function acceptDemoInteraction(event: InteractionEvent): InteractionAccepted {
  void event;
  return {
    event_id: randomId("demo-event"),
    accepted_at: new Date().toISOString(),
    status: "accepted",
    data_version: DEMO_DATA_VERSION,
    rule_version: DEMO_RULE_VERSION,
  };
}
