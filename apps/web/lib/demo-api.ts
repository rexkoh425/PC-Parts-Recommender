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

export const DEMO_DATA_VERSION = "portfolio-demo-2026-08-21";
export const DEMO_PRICE_AS_OF = "2026-08-21";
// USD->SGD at the Federal Reserve H.10 rate for 2026-08-18. Recorded so the
// SGD figures can be re-derived rather than taken on trust.
export const DEMO_PRICE_FX_USD_SGD = 1.2775;
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
  price_usd: number;
  // How the figure was arrived at. "launch_msrp" is vendor-announced and exact;
  // "street_aug_2026" is a retailer figure on the date below; "estimate" is the
  // fixture's own number where public sources disagreed too widely to pick one.
  price_basis: "launch_msrp" | "manufacturer_list" | "street_aug_2026" | "estimate";
  price_source: string | null;
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

// First-party manufacturer pages, checked by hand rather than harvested: the
// demo catalogue is a fixture, but every part in it now names a real SKU, so
// each one resolves to its manufacturer page. Manufacturer pages are used
// rather than retailer offers so nothing here implies a live price: these
// URLs describe the part, they do not quote it.
const manufacturerPartNumbers: Record<string, string> = {
  "mem-ddr5-32-5600": "CMK32GX5M2B5600Z36",
  "mem-ddr5-32-6000": "F5-6000J3036F16GX2-TZ5NRW",
  "mem-ddr5-64-6000": "F5-6000J3040G32GX2-TZ5NR",
  "ssd-nvme-2tb-value": "WDS200T2X0E",
  "ssd-nvme-2tb-fast": "MZ-V9P2T0BW",
  "psu-750-gold": "CP-9020248-NA",
  "psu-850-gold": "CP-9020270-NA",
};

const specUrls: Record<string, string> = {
  "cpu-amd-7600":
    "https://www.amd.com/en/products/processors/desktops/ryzen/7000-series/amd-ryzen-5-7600.html",
  "cpu-amd-7700":
    "https://www.amd.com/en/products/processors/desktops/ryzen/7000-series/amd-ryzen-7-7700.html",
  "cpu-amd-7900":
    "https://www.amd.com/en/products/processors/desktops/ryzen/7000-series/amd-ryzen-9-7900.html",
  "gpu-rtx-5060ti-16":
    "https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5060-family/",
  "gpu-rx-7800xt-16":
    "https://www.amd.com/en/products/graphics/desktops/radeon/7000-series/amd-radeon-rx-7800-xt.html",
  "gpu-rtx-4070tis-16":
    "https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4070-family/",
  "mb-b650m-wifi":
    "https://www.gigabyte.com/us/Motherboard/B650M-GAMING-PLUS-WIFI-rev-1x",
  "mb-b650-atx-wifi":
    "https://www.gigabyte.com/us/Motherboard/B650-EAGLE-AX-rev-10-11",
  "mb-x670-atx-wifi":
    "https://rog.asus.com/motherboards/rog-strix/rog-strix-x670e-a-gaming-wifi-model/",
  "mem-ddr5-32-5600":
    "https://www.corsair.com/us/en/p/memory/cmk32gx5m2b5600z36/vengeance-32gb-2x16gb-ddr5-dram-5600mt-s-c36-amd-expo-memory-kit-cmk32gx5m2b5600z36",
  "mem-ddr5-32-6000":
    "https://www.gskill.com/product/165/390/1692584545/F5-6000J3036F16GX2-TZ5NRW",
  "mem-ddr5-64-6000":
    "https://www.gskill.com/product/165/390/1665020865/F5-6000J3040G32GX2-TZ5NR",
  "ssd-nvme-2tb-value":
    "https://www.westerndigital.com/en-us/products/internal-drives/wd-black-sn850x-nvme-ssd?sku=WDS200T2X0E",
  "ssd-nvme-2tb-fast":
    "https://www.samsung.com/sg/memory-storage/nvme-ssd/990-pro-2tb-nvme-pcie-gen-4-mz-v9p2t0bw/",
  "psu-750-gold":
    "https://www.corsair.com/us/en/p/psu/cp-9020248-na/rme-series-rm750e-fully-modular-low-noise-atx-power-supply-cp-9020248-na",
  "psu-850-gold":
    "https://www.corsair.com/us/en/p/psu/cp-9020270-na/rmx-series-rm850x-fully-modular-power-supply-cp-9020270-na",
  "cooler-single-tower":
    "https://www.deepcool.com/products/Cooling/cpuaircoolers/AK400-Performance-CPU-Cooler-1700-AM5/2021/15222.shtml",
  "cooler-dual-tower":
    "https://www.deepcool.com/products/Cooling/cpuaircoolers/AK620-High-Performance-CPU-Cooler-1700-AM5/2021/13067.shtml",
  "case-matx-air":
    "https://www.fractal-design.com/products/cases/pop/pop-mini-air/rgb-black-tg-clear/",
  "case-atx-air":
    "https://lian-li.com/product/lancool-216/",
  "case-atx-quiet":
    "https://www.fractal-design.com/products/cases/define/define-7/black-tg-dark-tint/",
};

const products: DemoProduct[] = [
  {
    product_id: "cpu-amd-7600",
    category: "cpu",
    canonical_name: "AMD Ryzen 5 7600",
    brand: "AMD",
    model: "Ryzen 5 7600",
    price_sgd: 293,
    price_usd: 229.0,
    price_basis: "launch_msrp",
    price_source: "amd.com",
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
    price_sgd: 420,
    price_usd: 329.0,
    price_basis: "launch_msrp",
    price_source: "amd.com",
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
    price_sgd: 548,
    price_usd: 429.0,
    price_basis: "launch_msrp",
    price_source: "amd.com",
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
    price_sgd: 548,
    price_usd: 429.0,
    price_basis: "launch_msrp",
    price_source: "nvidia.com",
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
    price_sgd: 637,
    price_usd: 499.0,
    price_basis: "launch_msrp",
    price_source: "amd.com",
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
    price_sgd: 1021,
    price_usd: 799.0,
    price_basis: "launch_msrp",
    price_source: "nvidia.com",
    performance: 94,
    power_w: 285,
    vram_gb: 16,
    attributes: { vram_gb: 16, length_mm: 305, slot_width: 3 },
  },
  {
    product_id: "mb-b650m-wifi",
    category: "motherboard",
    canonical_name: "GIGABYTE B650M GAMING PLUS WIFI",
    brand: "GIGABYTE",
    model: "B650M GAMING PLUS WIFI",
    price_sgd: 144,
    price_usd: 112.99,
    price_basis: "street_aug_2026",
    price_source: "Newegg",
    performance: 72,
    power_w: 45,
    attributes: { socket: "AM5", chipset: "B650", memory_type: "DDR5", wifi: true, form_factor: "mATX", lan: "2.5 GbE", m2_slots: 2 },
  },
  {
    product_id: "mb-b650-atx-wifi",
    category: "motherboard",
    canonical_name: "GIGABYTE B650 EAGLE AX",
    brand: "GIGABYTE",
    model: "B650 EAGLE AX",
    price_sgd: 258,
    price_usd: 202.0,
    price_basis: "street_aug_2026",
    price_source: "Newegg",
    performance: 82,
    power_w: 50,
    attributes: { socket: "AM5", chipset: "B650", memory_type: "DDR5", wifi: true, form_factor: "ATX", lan: "2.5 GbE", m2_slots: 3, pcie_5_m2: true },
  },
  {
    product_id: "mb-x670-atx-wifi",
    category: "motherboard",
    canonical_name: "ASUS ROG Strix X670E-A Gaming WiFi",
    brand: "ASUS",
    model: "ROG Strix X670E-A Gaming WiFi",
    price_sgd: 420,
    price_usd: 328.68,
    price_basis: "street_aug_2026",
    price_source: "Newegg (Snow colourway)",
    performance: 92,
    power_w: 55,
    attributes: { socket: "AM5", chipset: "X670E", memory_type: "DDR5", wifi: true, form_factor: "ATX", lan: "2.5 GbE", m2_slots: 4, power_stages: "16+2" },
  },
  {
    product_id: "mem-ddr5-32-5600",
    category: "memory",
    canonical_name: "Corsair Vengeance 32 GB DDR5-5600 CL36 EXPO",
    brand: "Corsair",
    model: "CMK32GX5M2B5600Z36",
    price_sgd: 479,
    price_usd: 375.0,
    price_basis: "street_aug_2026",
    price_source: "Tom's Hardware RAM index",
    performance: 74,
    power_w: 8,
    memory_gb: 32,
    attributes: { memory_type: "DDR5", capacity_gb: 32, module_count: 2, speed_mts: 5600, cas_latency: 36, expo: true },
  },
  {
    product_id: "mem-ddr5-32-6000",
    category: "memory",
    canonical_name: "G.Skill Trident Z5 Neo RGB 32 GB DDR5-6000 CL30",
    brand: "G.Skill",
    model: "F5-6000J3036F16GX2-TZ5NRW",
    price_sgd: 501,
    price_usd: 392.0,
    price_basis: "street_aug_2026",
    price_source: "Tom's Hardware RAM index",
    performance: 82,
    power_w: 9,
    memory_gb: 32,
    attributes: { memory_type: "DDR5", capacity_gb: 32, module_count: 2, speed_mts: 6000, cas_latency: 30, expo: true },
  },
  {
    product_id: "mem-ddr5-64-6000",
    category: "memory",
    canonical_name: "G.Skill Trident Z5 Neo RGB 64 GB DDR5-6000 CL30",
    brand: "G.Skill",
    model: "F5-6000J3040G32GX2-TZ5NR",
    price_sgd: 1405,
    price_usd: 1099.99,
    price_basis: "street_aug_2026",
    price_source: "price trackers",
    performance: 94,
    power_w: 12,
    memory_gb: 64,
    attributes: { memory_type: "DDR5", capacity_gb: 64, module_count: 2, speed_mts: 6000, cas_latency: 30, expo: true },
  },
  {
    product_id: "ssd-nvme-2tb-value",
    category: "storage",
    canonical_name: "WD_BLACK SN850X 2 TB PCIe 4.0 NVMe SSD",
    brand: "Western Digital",
    model: "WDS200T2X0E",
    price_sgd: 470,
    price_usd: 367.99,
    price_basis: "street_aug_2026",
    price_source: "price trackers",
    performance: 76,
    power_w: 7,
    storage_gb: 2000,
    attributes: { capacity_gb: 2000, interface: "NVMe PCIe 4.0", sequential_read_mb_s: 7300, sequential_write_mb_s: 6300 },
  },
  {
    product_id: "ssd-nvme-2tb-fast",
    category: "storage",
    canonical_name: "Samsung 990 PRO 2 TB PCIe 4.0 NVMe SSD",
    brand: "Samsung",
    model: "MZ-V9P2T0BW",
    price_sgd: 498,
    price_usd: 389.99,
    price_basis: "street_aug_2026",
    price_source: "price trackers",
    performance: 91,
    power_w: 6,
    storage_gb: 2000,
    attributes: { capacity_gb: 2000, interface: "NVMe PCIe 4.0", sequential_read_mb_s: 7450, sequential_write_mb_s: 6900 },
  },
  {
    product_id: "psu-750-gold",
    category: "psu",
    canonical_name: "Corsair RM750e 750 W 80 Plus Gold Modular PSU",
    brand: "Corsair",
    model: "CP-9020248-NA",
    price_sgd: 115,
    price_usd: 89.99,
    price_basis: "street_aug_2026",
    price_source: "micro center",
    performance: 81,
    power_w: 0,
    attributes: { wattage: 750, efficiency_rating: "80 Plus Gold", atx_standard: "ATX 3.1", length_mm: 140, modular: "full" },
  },
  {
    product_id: "psu-850-gold",
    category: "psu",
    canonical_name: "Corsair RM850x 850 W 80 Plus Gold Modular PSU",
    brand: "Corsair",
    model: "CP-9020270-NA",
    price_sgd: 179,
    price_usd: 139.99,
    price_basis: "manufacturer_list",
    price_source: "corsair.com",
    performance: 90,
    power_w: 0,
    attributes: { wattage: 850, efficiency_rating: "80 Plus Gold", atx_standard: "ATX 3.1", modular: "full" },
  },
  {
    product_id: "cooler-single-tower",
    category: "cooler",
    canonical_name: "DeepCool AK400 Single-Tower CPU Cooler",
    brand: "DeepCool",
    model: "AK400",
    price_sgd: 45,
    price_usd: 34.99,
    price_basis: "launch_msrp",
    price_source: "deepcool.com",
    performance: 73,
    power_w: 4,
    attributes: { supported_sockets: ["AM5", "AM4", "LGA1700"], height_mm: 155, tdp_w: 220, heat_pipes: 4, fans: 1 },
  },
  {
    product_id: "cooler-dual-tower",
    category: "cooler",
    canonical_name: "DeepCool AK620 Dual-Tower CPU Cooler",
    brand: "DeepCool",
    model: "AK620",
    price_sgd: 83,
    price_usd: 64.99,
    price_basis: "street_aug_2026",
    price_source: "multiple retailers",
    performance: 88,
    power_w: 6,
    attributes: { supported_sockets: ["AM5", "AM4", "LGA1700"], height_mm: 160, tdp_w: 260, heat_pipes: 6, fans: 2 },
  },
  {
    product_id: "case-matx-air",
    category: "case",
    canonical_name: "Fractal Design Pop Mini Air Micro-ATX Case",
    brand: "Fractal Design",
    model: "Pop Mini Air",
    price_sgd: 128,
    price_usd: 99.99,
    price_basis: "street_aug_2026",
    price_source: "price trackers",
    performance: 75,
    power_w: 0,
    case_size: "mini_tower",
    attributes: { maximum_gpu_length_mm: 365, maximum_cooler_height_mm: 170, maximum_psu_length_mm: 150, form_factors: ["mATX", "Mini-ITX"], dimensions_mm: "432 x 215 x 393" },
  },
  {
    product_id: "case-atx-air",
    category: "case",
    canonical_name: "Lian Li LANCOOL 216 ATX Mid-Tower Case",
    brand: "Lian Li",
    model: "LANCOOL 216",
    price_sgd: 128,
    price_usd: 99.99,
    price_basis: "street_aug_2026",
    price_source: "price trackers",
    performance: 84,
    power_w: 0,
    case_size: "mid_tower",
    attributes: { maximum_gpu_length_mm: 392, maximum_cooler_height_mm: 180, maximum_psu_length_mm: 220, form_factors: ["E-ATX", "ATX", "mATX", "Mini-ITX"], dimensions_mm: "481 x 235 x 492" },
  },
  {
    product_id: "case-atx-quiet",
    category: "case",
    canonical_name: "Fractal Design Define 7 ATX Mid-Tower Case",
    brand: "Fractal Design",
    model: "Define 7",
    price_sgd: 249,
    price_usd: 194.99,
    price_basis: "street_aug_2026",
    price_source: "Newegg",
    performance: 82,
    power_w: 0,
    case_size: "mid_tower",
    attributes: { maximum_gpu_length_mm: 470, maximum_cooler_height_mm: 185, maximum_psu_length_mm: 250, form_factors: ["E-ATX", "ATX", "mATX", "Mini-ITX"], layout: "open", dimensions_mm: "547 x 240 x 475" },
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
    ...(specUrls[product.product_id] ? { spec_url: specUrls[product.product_id] } : {}),
    price_basis: product.price_basis,
    price_source: product.price_source,
    price_as_of: DEMO_PRICE_AS_OF,
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

/** Every product in the controlled demo, for build-time prerendering. */
export const demoProductIds: string[] = products.map((product) => product.product_id);

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
        as_of: "2026-08-21T00:00:00.000Z",
        scope_label: "Parts in this demo",
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
    manufacturer_part_number: manufacturerPartNumbers[product.product_id] ?? null,
    attributes: product.attributes,
      source_confidence: null,
      source_url: specUrls[product.product_id] ?? null,
      source_attributions: [],
      updated_at: "2026-08-21T00:00:00.000Z",
    data_version: DEMO_DATA_VERSION,
  };
}

export function getDemoPrices(productId: string): ProductPricesResponse {
  const product = requireDemoProduct(productId);
  const observation: PriceObservation = {
    listing_id: `demo-listing-${product.product_id}`,
    retailer: "Controlled demo catalogue",
    observed_at: "2026-08-21T00:00:00.000Z",
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
    // Deliberately null: this benchmark is the demo's own synthetic index, not
    // a figure the manufacturer published. Citing their page here would
    // attribute the number to them.
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
    catalogue_status: "degraded",
    price_status: "degraded",
    last_catalog_update: "2026-08-21T00:00:00.000Z",
    prices_updated_at: "2026-08-21T00:00:00.000Z",
    stale_after_hours: 24,
    catalogue_stale_after_hours: 168,
    price_stale_after_hours: 24,
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
    trust_level: "legacy_untrusted",
    replayed: false,
  };
}
