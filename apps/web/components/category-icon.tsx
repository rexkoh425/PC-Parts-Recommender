import {
  CircuitBoard,
  Computer,
  Cpu,
  Fan,
  Gpu,
  HardDrive,
  MemoryStick,
  PlugZap,
  type LucideIcon,
} from "lucide-react";
import type { ComponentCategory } from "@/lib/types";

const categoryIcons: Record<ComponentCategory, LucideIcon> = {
  cpu: Cpu,
  gpu: Gpu,
  motherboard: CircuitBoard,
  memory: MemoryStick,
  storage: HardDrive,
  psu: PlugZap,
  cooler: Fan,
  case: Computer,
};

export function CategoryIcon({ category }: { category: ComponentCategory }) {
  const Icon = categoryIcons[category];

  return <Icon className="category-icon" aria-hidden="true" focusable="false" strokeWidth={1.7} />;
}
