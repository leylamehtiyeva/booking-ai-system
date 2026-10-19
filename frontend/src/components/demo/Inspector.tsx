import type { ReactNode } from "react";
import type { ConstraintStatus, ResultDecision, TurnResult } from "../../lib/types";
import {
  ACTION_TEXT,
  PRIORITY_TEXT,
  SOURCE_TEXT,
  TIER_TEXT,
  budgetText,
  guestsText,
  stayText,
} from "../../lib/present";

function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="border-t border-line py-5 first:border-t-0 first:pt-0">
      <h3 className="eyebrow">{title}</h3>
      <div className="mt-3 text-sm text-ink-soft">{children}</div>
    </section>
  );
}

function Row({ label, value }: { label: string; value: ReactNode }) {
  if (value == null || value === "") return null;
  return (
    <div className="flex justify-between gap-4 py-1">
      <dt className="text-muted">{label}</dt>
      <dd className="text-right font-medium text-ink">{value}</dd>
    </div>
  );
}

const VERDICT = {
  matched: { label: "Confirmed", className: "bg-ok-soft text-ok" },
  uncertain: { label: "Not confirmed", className: "bg-warn-soft text-warn" },
  failed: { label: "Not satisfied", className: "bg-bad-soft text-bad" },
} as const;

function Verdicts({ items, kind }: { items: ConstraintStatus[]; kind: keyof typeof VERDICT }) {
  return items.map((item) => (
    <li key={`${kind}-${item.name}`} className="flex items-start gap-2.5 py-1">
      <span className={`mt-0.5 shrink-0 rounded px-1.5 py-px text-[0.66rem] font-medium ${VERDICT[kind].className}`}>{VERDICT[kind].label}</span>
      <span className="min-w-0">
        <span className="font-medium text-ink">{item.name}</span>
        {item.constraint?.priority && <span className="ml-1.5 text-[0.72rem] text-muted">({PRIORITY_TEXT[item.constraint.priority] ?? item.constraint.priority})</span>}
        {item.reason && <span className="block break-words font-mono text-[0.72rem] leading-relaxed text-muted">{item.reason}</span>}
      </span>
    </li>
  ));
}

function ResultBlock({ result, highlighted }: { result: ResultDecision; highlighted: boolean }) {
  return (
    <div id={`inspect-${result.result_id}`} className={`rounded-xl border p-4 transition-colors ${highlighted ? "border-accent bg-accent-soft/30" : "border-line"}`}>
      <div className="flex items-start justify-between gap-3">
        <p className="font-semibold text-ink">{result.title}</p>
        <span className="shrink-0 font-mono text-[0.72rem] text-muted">score {result.score}</span>
      </div>
      <p className="mt-1 text-[0.78rem] text-muted">
        {[result.match_tier && (TIER_TEXT[result.match_tier] ?? result.match_tier), result.eligibility_status].filter(Boolean).join(" · ")}
      </p>
      {result.selection_reasons.length > 0 && (
        <p className="mt-2 text-[0.82rem]">Selected because {result.selection_reasons.join("; ")}.</p>
      )}
      {result.blocking_reasons.length > 0 && <p className="mt-1 text-[0.82rem] text-bad">Blocked: {result.blocking_reasons.join("; ")}</p>}
      <ul className="mt-3">
        <Verdicts items={result.matched_constraints} kind="matched" />
        <Verdicts items={result.uncertain_constraints} kind="uncertain" />
        <Verdicts items={result.failed_constraints} kind="failed" />
      </ul>
      {result.constraint_resolution_results.length > 0 && (
        <div className="mt-3 space-y-2 border-t border-line pt-3">
          <p className="eyebrow !text-[0.62rem]">Evidence from listing text</p>
          {result.constraint_resolution_results.map((item) => (
            <div key={item.raw_text} className="text-[0.8rem]">
              <p>
                <span className="font-medium text-ink">{item.raw_text}</span> → {item.decision}
              </p>
              <p className="text-muted">{item.reason}</p>
              {item.evidence.map((e, i) => (
                <blockquote key={i} className="mt-1 border-l-2 border-line-strong pl-2 italic text-ink-soft">“{e.snippet}”</blockquote>
              ))}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export function Inspector({ turn, highlight }: { turn: TurnResult | null; highlight: string | null }) {
  if (!turn) {
    return (
      <div className="text-sm text-muted">
        <p className="text-lg font-semibold text-ink">Decision Inspector</p>
        <p className="mt-3 leading-relaxed">
          Send a message to see how it was handled: the routing decision, the requirements the assistant understood, and the verdict behind every
          recommended listing.
        </p>
      </div>
    );
  }

  const { inspector, telemetry } = turn;
  const search = inspector.current_search;
  const routing = inspector.routing;

  return (
    <div>
      <Section title="Routing">
        <dl>
          <Row label="Action" value={inspector.effective_action ? (ACTION_TEXT[inspector.effective_action] ?? inspector.effective_action) : "None"} />
          <Row label="Decision" value={routing?.decision_status} />
          <Row label="Clarification" value={routing && routing.clarification_reason !== "none" ? routing.clarification_reason : null} />
          <Row label="Outcome" value={turn.outcome_kind.replace("_", " ")} />
          <Row label="Response" value={SOURCE_TEXT[inspector.response_source] ?? inspector.response_source} />
        </dl>
        {turn.failure && (
          <p className="mt-2 rounded-lg bg-bad-soft px-3 py-2 text-[0.8rem] text-bad">
            Stage “{turn.failure.stage}” failed ({turn.failure.code}). The search state was not changed.
          </p>
        )}
      </Section>

      {search && (
        <Section title="Understood request">
          <dl>
            <Row label="City" value={search.city} />
            <Row label="Stay" value={stayText(search)} />
            <Row label="Guests" value={guestsText(search)} />
            <Row label="Budget" value={budgetText(search)} />
            <Row label="Property" value={search.property_types?.join(", ")} />
            <Row label="Bedrooms" value={search.filters?.bedrooms_min != null ? `at least ${search.filters.bedrooms_min}` : null} />
            <Row label="Area" value={search.filters?.area_sqm_min != null ? `at least ${search.filters.area_sqm_min} m²` : null} />
          </dl>
          {search.constraints.length > 0 && (
            <ul className="mt-3 flex flex-wrap gap-1.5">
              {search.constraints.map((c) => (
                <li key={c.id ?? c.normalized_text} className="rounded-full border border-line-strong px-2.5 py-0.5 text-[0.76rem]">
                  {c.normalized_text}
                  <span className="ml-1 text-muted">· {PRIORITY_TEXT[c.priority] ?? c.priority}</span>
                </li>
              ))}
            </ul>
          )}
        </Section>
      )}

      {inspector.results.length > 0 && (
        <Section title={`Recommended results · ${inspector.results.length}`}>
          <div className="space-y-3">
            {inspector.results.map((result) => (
              <ResultBlock key={result.result_id} result={result} highlighted={highlight === result.result_id} />
            ))}
          </div>
        </Section>
      )}

      {inspector.listing_query.length > 0 && (
        <Section title="Answer about shown listings">
          <ul className="space-y-2">
            {inspector.listing_query.map((item) => (
              <li key={item.match.result_id}>
                <p className="font-medium text-ink">
                  #{item.position} {item.title}
                </p>
                {item.match.constraint_results.map((r, i) => (
                  <p key={i} className="text-[0.82rem]">
                    {r.raw_text}: <span className="font-mono">{r.value}</span>
                  </p>
                ))}
              </li>
            ))}
          </ul>
        </Section>
      )}

      {inspector.notes.length > 0 && (
        <Section title="Notes">
          <ul className="list-disc space-y-1 pl-4 text-[0.82rem]">
            {inspector.notes.map((note) => <li key={note}>{note}</li>)}
          </ul>
        </Section>
      )}

      <Section title="This turn">
        <dl>
          <Row label="Latency" value={`${Math.round(telemetry.latency_ms).toLocaleString("en-US")} ms`} />
          <Row label="LLM calls" value={telemetry.failed_llm_calls ? `${telemetry.llm_calls} (${telemetry.failed_llm_calls} failed)` : telemetry.llm_calls} />
          <Row label="Estimated cost" value={`$${telemetry.estimated_cost_usd.toFixed(5)}`} />
        </dl>
      </Section>
    </div>
  );
}
