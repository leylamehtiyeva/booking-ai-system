import { CircleHelp, ListChecks, ShieldQuestion, SlidersHorizontal, type LucideIcon } from "lucide-react";
import { useRef, useState, type KeyboardEvent } from "react";
import type { ListingCard as Card } from "../../lib/types";
import { ListingCard } from "../demo/ListingCard";

export interface Scenario {
  id: string;
  label: string;
  title: string;
  summary: string;
  points: string[];
  messages: { role: "user" | "assistant"; text: string }[];
  card?: Card;
}

const ICONS: Record<string, { icon: LucideIcon; tile: string }> = {
  search: { icon: ListChecks, tile: "tile-blue" },
  refine: { icon: SlidersHorizontal, tile: "tile-amber" },
  question: { icon: CircleHelp, tile: "tile-indigo" },
  uncertain: { icon: ShieldQuestion, tile: "tile-emerald" },
};

export default function UseCaseTabs({ scenarios }: { scenarios: Scenario[] }) {
  const [active, setActive] = useState(0);
  const tabs = useRef<(HTMLButtonElement | null)[]>([]);
  const scenario = scenarios[active];

  function onKey(event: KeyboardEvent) {
    const delta = event.key === "ArrowRight" ? 1 : event.key === "ArrowLeft" ? -1 : 0;
    if (!delta) return;
    event.preventDefault();
    const next = (active + delta + scenarios.length) % scenarios.length;
    setActive(next);
    tabs.current[next]?.focus();
  }

  const { icon: Icon, tile } = ICONS[scenario.id] ?? ICONS.search;

  return (
    <div className="overflow-hidden rounded-xl border border-line bg-white shadow-sm">
      <div role="tablist" aria-label="Use cases" onKeyDown={onKey} className="flex overflow-x-auto border-b border-line text-sm">
        {scenarios.map((item, index) => {
          const selected = index === active;
          return (
            <button
              key={item.id}
              ref={(el) => {
                tabs.current[index] = el;
              }}
              role="tab"
              id={`tab-${item.id}`}
              aria-selected={selected}
              aria-controls={`panel-${item.id}`}
              tabIndex={selected ? 0 : -1}
              onClick={() => setActive(index)}
              className={`relative shrink-0 whitespace-nowrap px-4 py-3 font-medium transition-colors sm:px-5 ${
                selected ? "text-ink after:absolute after:inset-x-3 after:bottom-0 after:h-0.5 after:rounded-full after:bg-ink" : "text-subtle hover:text-ink-soft"
              }`}
            >
              {item.label}
            </button>
          );
        })}
      </div>

      <div role="tabpanel" id={`panel-${scenario.id}`} aria-labelledby={`tab-${scenario.id}`} className="grid gap-6 bg-paper-sunk p-5 md:grid-cols-[minmax(0,0.95fr)_minmax(0,1.05fr)] md:p-6">
        <div className="min-w-0">
          <div className="flex items-center gap-3">
            <div className={`icon-tile ${tile}`}>
              <Icon size={20} strokeWidth={1.5} aria-hidden="true" />
            </div>
            <div>
              <h3 className="text-base font-semibold text-ink">{scenario.title}</h3>
              <p className="text-sm text-muted">{scenario.summary}</p>
            </div>
          </div>
          <div className="mt-5 space-y-2.5 text-sm leading-relaxed">
            {scenario.messages.map((m, i) =>
              m.role === "user" ? (
                <p key={i} className="ml-auto w-fit max-w-[92%] rounded-xl rounded-br-sm bg-ink px-3.5 py-2 text-white">{m.text}</p>
              ) : (
                <p key={i} className="w-fit max-w-[92%] rounded-xl rounded-bl-sm border border-line bg-white px-3.5 py-2 text-ink-soft">{m.text}</p>
              ),
            )}
          </div>
          <ul className="mt-5 space-y-2 text-sm text-muted">
            {scenario.points.map((p) => (
              <li key={p} className="flex gap-2.5">
                <span className="mt-2 h-1 w-1 shrink-0 rounded-full bg-subtle" aria-hidden="true" />
                {p}
              </li>
            ))}
          </ul>
        </div>
        <div className="min-w-0">
          {scenario.card ? (
            <ListingCard card={scenario.card} />
          ) : (
            <div className="grid h-full min-h-40 place-items-center rounded-xl border border-dashed border-line-strong bg-white p-6 text-center text-sm text-muted">
              No listing cards for this turn: the answer is text, rendered straight from the verdict.
            </div>
          )}
          <p className="mt-2 text-right text-xs text-subtle">Recorded on sample data</p>
        </div>
      </div>
    </div>
  );
}
