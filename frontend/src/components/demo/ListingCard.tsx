import type { ListingCard as Card, ListingCardPoint } from "../../lib/types";
import { STATUS_TEXT, TIER_TEXT, isOptional, isSampleUrl } from "../../lib/present";

type Tone = "ok" | "warn" | "bad";

const TONE: Record<Tone, { dot: string; text: string; title: string }> = {
  ok: { dot: "bg-ok", text: "text-ok", title: "Confirmed" },
  warn: { dot: "bg-warn", text: "text-warn", title: "Not confirmed" },
  bad: { dot: "bg-bad", text: "text-bad", title: "Not satisfied" },
};

function Icon({ tone }: { tone: Tone }) {
  const path = tone === "ok" ? "M5 12.5l4 4 10-10" : tone === "bad" ? "M7 7l10 10M17 7 7 17" : "M12 7v6m0 4h.01";
  return (
    <svg viewBox="0 0 24 24" className={`mt-0.5 h-4 w-4 shrink-0 ${TONE[tone].text}`} fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d={path} />
    </svg>
  );
}

function PointGroup({ tone, points }: { tone: Tone; points: ListingCardPoint[] }) {
  if (points.length === 0) return null;
  return (
    <div>
      <p className={`text-xs font-semibold ${TONE[tone].text}`}>
        {TONE[tone].title} · {points.length}
      </p>
      <ul className="mt-2 space-y-2">
        {points.map((point) => (
          <li key={point.name} className="flex gap-2.5 text-sm">
            <Icon tone={tone} />
            <span className="min-w-0">
              <span className="font-medium text-ink">{point.label}</span>
              {isOptional(point) ? (
                <span className="ml-2 rounded-md bg-gray-100 px-1.5 py-0.5 text-[0.68rem] font-medium text-muted">Optional</span>
              ) : (
                point.reason &&
                point.reason.toLowerCase() !== point.label.toLowerCase() && <span className="block break-words text-[0.8rem] leading-snug text-muted">{point.reason}</span>
              )}
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}

export function ListingCard({ card, onInspect }: { card: Card; onInspect?: (resultId: string) => void }) {
  const sample = isSampleUrl(card.url);
  const status = card.status_label ? (STATUS_TEXT[card.status_label] ?? card.status_label) : null;
  const tier = card.match_tier ? (TIER_TEXT[card.match_tier] ?? card.match_tier) : null;

  return (
    <article className="group rounded-xl border border-line bg-white p-5 transition-[border-color,box-shadow] duration-200 hover:border-line-strong hover:shadow-md">
      <header className="flex items-start gap-4">
        <span className="mt-0.5 grid h-7 w-7 shrink-0 place-items-center rounded-full bg-gray-100 text-xs font-semibold text-muted" aria-label={`Result ${card.position}`}>
          {card.position}
        </span>
        <div className="min-w-0 flex-1">
          <h3 className="text-base font-semibold leading-snug text-ink">{card.title}</h3>
          <div className="mt-1.5 flex flex-wrap items-center gap-x-3 gap-y-1 text-[0.8rem] text-muted">
            {tier && <span className="rounded-md bg-ok-soft px-1.5 py-0.5 font-medium text-ok">{tier}</span>}
            {status && <span>{status}</span>}
          </div>
        </div>
      </header>

      {(card.price_summary || card.budget_summary) && (
        <dl className="mt-4 grid gap-3 rounded-lg border border-line bg-paper-sunk px-4 py-3 text-sm sm:grid-cols-2">
          {card.price_summary && (
            <div>
              <dt className="text-xs text-muted">Price</dt>
              <dd className="mt-0.5 font-medium text-ink">{card.price_summary}</dd>
            </div>
          )}
          {card.budget_summary && (
            <div>
              <dt className="text-xs text-muted">Budget</dt>
              <dd className="mt-0.5 text-ink-soft">{card.budget_summary}</dd>
            </div>
          )}
        </dl>
      )}

      <div className="mt-5 grid gap-5">
        <PointGroup tone="ok" points={card.confirmed} />
        <PointGroup tone="warn" points={card.needs_confirmation} />
        <PointGroup tone="bad" points={card.not_satisfied} />
      </div>

      <footer className="mt-6 flex flex-wrap items-center justify-between gap-3 border-t border-line pt-4 text-sm">
        {sample ? (
          <span className="text-muted">Sample listing · no external page</span>
        ) : (
          <a href={card.url ?? undefined} target="_blank" rel="noopener noreferrer" className="font-medium underline decoration-line-strong underline-offset-4 hover:decoration-accent">
            View listing
          </a>
        )}
        {onInspect && (
          <button type="button" onClick={() => onInspect(card.result_id)} className="font-medium text-ink-soft transition-colors hover:text-accent">
            Why this result? →
          </button>
        )}
      </footer>
    </article>
  );
}
