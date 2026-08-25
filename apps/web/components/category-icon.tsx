import type { ComponentCategory } from "@/lib/types";

/*
 * A mark per category.
 *
 * The catalogue chips were text alone, so a grid of 24 cards had no visual
 * texture and nothing to scan by shape - every card looked the same until you
 * read it. These are drawn on one 24x24 grid with a single 1.6 stroke so they
 * read as one family rather than eight separate drawings.
 *
 * currentColor throughout, so a mark inherits whatever colour its chip has and
 * never needs a second definition for dark surfaces.
 */

const paths: Record<ComponentCategory, React.ReactNode> = {
  // Processor: a die with pins on every side.
  cpu: (
    <>
      <rect x="7" y="7" width="10" height="10" rx="1.5" />
      <path d="M10 4v3M14 4v3M10 17v3M14 17v3M4 10h3M4 14h3M17 10h3M17 14h3" />
    </>
  ),
  // Graphics: a board with a fan.
  gpu: (
    <>
      <rect x="3" y="7" width="18" height="10" rx="1.5" />
      <circle cx="9" cy="12" r="2.6" />
      <path d="M15 10h3M15 14h3" />
    </>
  ),
  // Motherboard: a board with a socket and traces.
  motherboard: (
    <>
      <rect x="3.5" y="3.5" width="17" height="17" rx="1.5" />
      <rect x="7" y="7" width="6" height="6" rx="1" />
      <path d="M16 7v4M16 15h2M7 16h5" />
    </>
  ),
  // Memory: a DIMM with its notch.
  memory: (
    <>
      <rect x="2.5" y="8" width="19" height="8" rx="1" />
      <path d="M6 16v2M10 16v2M14 16v2M18 16v2M11 8v3h2V8" />
    </>
  ),
  // Storage: stacked platters.
  storage: (
    <>
      <ellipse cx="12" cy="6.5" rx="8" ry="3" />
      <path d="M4 6.5v11c0 1.7 3.6 3 8 3s8-1.3 8-3v-11" />
      <path d="M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3" />
    </>
  ),
  // Power supply: a unit with a fan and a socket.
  psu: (
    <>
      <rect x="2.5" y="6" width="19" height="12" rx="1.5" />
      <circle cx="9" cy="12" r="3.2" />
      <path d="M16 10h3M16 14h3" />
    </>
  ),
  // Cooler: a fan blade set in a frame.
  cooler: (
    <>
      <rect x="3.5" y="3.5" width="17" height="17" rx="2" />
      <circle cx="12" cy="12" r="4.2" />
      <path d="M12 7.8V12l3.6 2.1" />
    </>
  ),
  // Case: a tower with a front panel.
  case: (
    <>
      <rect x="5.5" y="2.5" width="13" height="19" rx="1.5" />
      <path d="M9 6h6M9 9h6" />
      <circle cx="12" cy="15" r="2.6" />
    </>
  ),
};

export function CategoryIcon({ category }: { category: ComponentCategory }) {
  return (
    <svg
      className="category-icon"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.6"
      strokeLinecap="round"
      strokeLinejoin="round"
      // Decorative: the chip already names the category in text, so announcing
      // it again would just repeat itself to a screen reader.
      aria-hidden="true"
      focusable="false"
    >
      {paths[category]}
    </svg>
  );
}
