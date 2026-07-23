"use client";

import { useId, useRef, useState } from "react";

export interface TabItem {
  id: string;
  label: string;
  /** Optional short count or status shown after the label. */
  hint?: string;
  content: React.ReactNode;
}

/**
 * Accessible tab set following the WAI-ARIA authoring pattern: arrow keys move
 * between tabs, Home/End jump to the ends, and only the active tab is tabbable.
 * Panels stay mounted so form state and in-page anchors survive switching.
 */
export function Tabs({
  items,
  initialId,
  label,
  className = "",
}: {
  items: TabItem[];
  initialId?: string;
  label: string;
  className?: string;
}) {
  const base = useId();
  const [active, setActive] = useState(initialId ?? items[0]?.id);
  const tabRefs = useRef<Record<string, HTMLButtonElement | null>>({});

  function focusTab(id: string) {
    setActive(id);
    tabRefs.current[id]?.focus();
  }

  function onKeyDown(event: React.KeyboardEvent<HTMLButtonElement>) {
    const index = items.findIndex((item) => item.id === active);
    if (index < 0) return;
    const last = items.length - 1;
    const next = {
      ArrowRight: index === last ? 0 : index + 1,
      ArrowLeft: index === 0 ? last : index - 1,
      Home: 0,
      End: last,
    }[event.key];
    if (next === undefined) return;
    event.preventDefault();
    focusTab(items[next].id);
  }

  return (
    <div className={`tabs ${className}`.trim()}>
      <div className="tabs__list" role="tablist" aria-label={label}>
        {items.map((item) => {
          const selected = item.id === active;
          return (
            <button
              key={item.id}
              ref={(node) => {
                tabRefs.current[item.id] = node;
              }}
              type="button"
              role="tab"
              id={`${base}-tab-${item.id}`}
              aria-controls={`${base}-panel-${item.id}`}
              aria-selected={selected}
              tabIndex={selected ? 0 : -1}
              className="tabs__tab"
              onClick={() => setActive(item.id)}
              onKeyDown={onKeyDown}
            >
              <span className="tabs__label">{item.label}</span>
              {item.hint ? <span className="tabs__hint">{item.hint}</span> : null}
            </button>
          );
        })}
      </div>
      {items.map((item) => (
        <div
          key={item.id}
          role="tabpanel"
          id={`${base}-panel-${item.id}`}
          aria-labelledby={`${base}-tab-${item.id}`}
          className="tabs__panel"
          hidden={item.id !== active}
        >
          {item.content}
        </div>
      ))}
    </div>
  );
}
