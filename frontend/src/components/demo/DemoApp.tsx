import { useEffect, useMemo, useRef, useState } from "react";
import { ApiError, MAX_MESSAGE_LENGTH, httpTransport, type ChatTransport } from "../../lib/api";
import type { TurnResult } from "../../lib/types";
import { Inspector } from "./Inspector";
import { ListingCard } from "./ListingCard";
import { MessageText } from "./MessageText";

import { FOLLOW_UP_PROMPTS, STARTER_PROMPTS } from "./prompts";

export { FOLLOW_UP_PROMPTS, STARTER_PROMPTS };

type NewItem =
  | { kind: "user"; text: string; failed?: boolean }
  | { kind: "assistant"; turn: TurnResult }
  | { kind: "notice"; text: string; retry?: { text: string; fresh: boolean } };
type Item = NewItem & { id: number };

interface Props {
  transport?: ChatTransport;
  mock?: boolean;
}

function defaultTransport(): ChatTransport {
  return httpTransport(import.meta.env.PUBLIC_API_BASE_URL ?? "");
}

// With listing cards below it, a long message mostly repeats them: show the
// beginning and let the reader expand the full text (nothing is removed).
function CollapsibleMessage({ text, collapse }: { text: string; collapse: boolean }) {
  const [open, setOpen] = useState(false);
  const long = collapse && text.length > 360;
  return (
    <div>
      <div className={long && !open ? "relative max-h-36 overflow-hidden" : undefined}>
        <MessageText text={text} />
        {long && !open && <div className="absolute inset-x-0 bottom-0 h-14 bg-gradient-to-t from-paper-raised to-transparent" aria-hidden="true" />}
      </div>
      {long && (
        <button type="button" onClick={() => setOpen(!open)} aria-expanded={open} className="mt-2 text-[0.8rem] font-medium text-ink-soft underline decoration-line-strong underline-offset-4 hover:text-ink">
          {open ? "Show less" : "Show full message"}
        </button>
      )}
    </div>
  );
}

export default function DemoApp({ transport, mock = false }: Props) {
  const api = useMemo(() => transport ?? defaultTransport(), [transport]);
  const [items, setItems] = useState<Item[]>([]);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [draft, setDraft] = useState("");
  const [pending, setPending] = useState(false);
  const [selected, setSelected] = useState<number | null>(null);
  const [highlight, setHighlight] = useState<string | null>(null);
  const [inspectorOpen, setInspectorOpen] = useState(false);
  const nextId = useRef(1);
  const threadEnd = useRef<HTMLDivElement>(null);
  const input = useRef<HTMLTextAreaElement>(null);

  const add = (item: NewItem) => {
    const id = nextId.current++;
    setItems((current) => [...current, { ...item, id }]);
    return id;
  };

  // /demo?q=... (from the landing page's example prompts) fills the input;
  // the user still sends it, so opening a link never triggers an LLM call.
  useEffect(() => {
    const prompt = new URLSearchParams(window.location.search).get("q");
    if (prompt) setDraft(prompt.slice(0, MAX_MESSAGE_LENGTH));
  }, []);

  useEffect(() => {
    threadEnd.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [items, pending]);

  const assistantTurns = items.filter((i): i is Extract<Item, { kind: "assistant" }> => i.kind === "assistant");
  const inspected = assistantTurns.find((i) => i.id === selected) ?? assistantTurns.at(-1) ?? null;
  const hasSearched = assistantTurns.some((i) => i.turn.listing_cards.length > 0);

  async function send(raw: string, { fresh = false }: { fresh?: boolean } = {}) {
    const text = raw.trim();
    if (!text || pending || text.length > MAX_MESSAGE_LENGTH) return;
    const session = fresh ? null : sessionId;
    if (fresh) setSessionId(null);

    const userId = add({ kind: "user", text });
    setDraft("");
    setPending(true);
    try {
      const turn = await api.send(text, session);
      setSessionId(turn.session_id);
      const id = add({ kind: "assistant", turn });
      setSelected(id);
      setHighlight(null);
    } catch (error) {
      setItems((current) => current.map((i) => (i.id === userId && i.kind === "user" ? { ...i, failed: true } : i)));
      if (error instanceof ApiError && error.kind === "session_expired") {
        setSessionId(null);
        add({ kind: "notice", text: error.message, retry: { text, fresh: true } });
      } else {
        const message = error instanceof ApiError ? error.message : "Something unexpected happened. Please try again.";
        add({ kind: "notice", text: message, retry: { text, fresh: false } });
      }
    } finally {
      setPending(false);
      input.current?.focus();
    }
  }

  function reset() {
    setItems([]);
    setSessionId(null);
    setSelected(null);
    setHighlight(null);
    setDraft("");
    input.current?.focus();
  }

  function inspect(turnId: number, resultId?: string) {
    setSelected(turnId);
    setHighlight(resultId ?? null);
    setInspectorOpen(true);
    if (resultId) {
      requestAnimationFrame(() => document.getElementById(`inspect-${resultId}`)?.scrollIntoView({ behavior: "smooth", block: "nearest" }));
    }
  }

  const overLimit = draft.length > MAX_MESSAGE_LENGTH;

  return (
    <div className="grid gap-8 lg:grid-cols-[minmax(0,1fr)_380px]">
      <section aria-label="Conversation" className="flex min-h-[70vh] flex-col">
        <div className="mb-5 flex flex-wrap items-center justify-between gap-3">
          <p className="rounded-full border border-line bg-paper-raised px-3 py-1.5 text-[0.78rem] text-muted">
            {mock ? "Mock mode · replaying recorded responses" : "Sample dataset · listings, prices and links are illustrative, not live availability"}
          </p>
          <div className="flex items-center gap-2">
            <button type="button" onClick={() => setInspectorOpen(true)} className="btn btn-ghost !px-3 !py-1.5 text-sm lg:hidden">
              Inspector
            </button>
            {items.length > 0 && (
              <button type="button" onClick={reset} className="btn btn-ghost !px-3 !py-1.5 text-sm">
                New conversation
              </button>
            )}
          </div>
        </div>

        <div className="flex-1 space-y-6" aria-live="polite">
          {items.length === 0 && (
            <div className="rounded-xl border border-line bg-paper-sunk px-5 py-10 text-center sm:px-10">
              <p className="text-2xl font-bold tracking-tight text-ink">Describe your ideal stay.</p>
              <p className="mx-auto mt-3 max-w-md text-sm leading-relaxed text-muted">
                City, dates, budget and what matters to you. Then refine the results or ask about a listing — the conversation keeps your search.
              </p>
              <div className="mt-7 flex flex-col items-stretch gap-2 sm:items-center">
                {STARTER_PROMPTS.map((prompt) => (
                  <button
                    key={prompt}
                    type="button"
                    onClick={() => send(prompt)}
                    className="rounded-lg border border-line bg-white px-4 py-3 text-left text-sm text-ink-soft transition-colors hover:border-line-strong hover:text-ink sm:max-w-xl"
                  >
                    {prompt}
                  </button>
                ))}
              </div>
            </div>
          )}

          {items.map((item) => {
            if (item.kind === "user") {
              return (
                <div key={item.id} className="flex justify-end">
                  <div className="max-w-[85%] rounded-xl rounded-br-md bg-ink px-4 py-3 text-[0.95rem] leading-relaxed text-paper">
                    {item.text}
                    {item.failed && <span className="mt-1 block text-[0.72rem] opacity-70">Not sent</span>}
                  </div>
                </div>
              );
            }
            if (item.kind === "notice") {
              return (
                <div key={item.id} role="alert" className="rounded-xl border border-bad/30 bg-bad-soft px-4 py-3 text-sm text-ink">
                  <p>{item.text}</p>
                  {item.retry && (
                    <button
                      type="button"
                      onClick={() => send(item.retry!.text, { fresh: item.retry!.fresh })}
                      disabled={pending}
                      className="mt-2 font-medium text-bad underline underline-offset-4 disabled:opacity-50"
                    >
                      {item.retry.fresh ? "Start a new conversation with this message" : "Try again"}
                    </button>
                  )}
                </div>
              );
            }
            const { turn } = item;
            const isSelected = inspected?.id === item.id;
            return (
              <div key={item.id} className="space-y-4">
                <div
                  className={`max-w-[92%] rounded-xl rounded-bl-md border px-4 py-3 text-[0.95rem] leading-relaxed ${
                    turn.outcome_kind === "failure" ? "border-warn/40 bg-warn-soft" : "border-line bg-paper-raised"
                  }`}
                >
                  <CollapsibleMessage text={turn.assistant_message} collapse={turn.listing_cards.length > 0} />
                  <button
                    type="button"
                    onClick={() => inspect(item.id)}
                    className={`mt-3 block text-[0.75rem] font-medium transition-colors ${isSelected ? "text-accent" : "text-muted hover:text-ink"}`}
                  >
                    {isSelected ? "Shown in inspector" : "Inspect this answer"}
                  </button>
                </div>
                {turn.listing_cards.length > 0 && (
                  <div className="grid gap-4 xl:grid-cols-2">
                    {turn.listing_cards.map((card) => (
                      <ListingCard key={card.result_id} card={card} onInspect={(resultId) => inspect(item.id, resultId)} />
                    ))}
                  </div>
                )}
              </div>
            );
          })}

          {pending && (
            <div className="flex items-center gap-3 text-sm text-muted" role="status">
              <span className="flex gap-1" aria-hidden="true">
                {[0, 1, 2].map((i) => (
                  <span key={i} className="h-1.5 w-1.5 rounded-full bg-ink-soft" style={{ animation: `pulse-dot 1.2s ${i * 0.15}s infinite ease-in-out` }} />
                ))}
              </span>
              Understanding your request and checking listings…
            </div>
          )}
          <div ref={threadEnd} />
        </div>

        <div className="sticky bottom-0 mt-8 bg-gradient-to-t from-paper via-paper to-paper/0 pb-4 pt-6">
          {hasSearched && !pending && (
            <div className="mb-3 flex flex-wrap gap-2">
              {FOLLOW_UP_PROMPTS.map((prompt) => (
                <button key={prompt} type="button" onClick={() => send(prompt)} className="rounded-full border border-line bg-paper-raised px-3 py-1.5 text-[0.8rem] text-ink-soft transition-colors hover:border-ink hover:text-ink">
                  {prompt}
                </button>
              ))}
            </div>
          )}
          <form
            onSubmit={(event) => {
              event.preventDefault();
              send(draft);
            }}
            className="flex items-end gap-2 rounded-xl border border-line-strong bg-paper-raised p-2 shadow-sm focus-within:border-ink"
          >
            <label htmlFor="message" className="sr-only">
              Message
            </label>
            <textarea
              id="message"
              ref={input}
              value={draft}
              rows={1}
              onChange={(event) => setDraft(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
                  event.preventDefault();
                  send(draft);
                }
              }}
              placeholder={hasSearched ? "Refine the search or ask about a listing…" : "e.g. A quiet apartment in Baku for April 10–15 with a kitchen"}
              className="max-h-40 min-h-[44px] flex-1 resize-none bg-transparent px-3 py-2.5 text-[0.95rem] text-ink outline-none placeholder:text-muted [field-sizing:content]"
            />
            <button type="submit" disabled={pending || !draft.trim() || overLimit} className="btn btn-primary !px-4 !py-2.5 text-sm disabled:cursor-not-allowed disabled:opacity-40 disabled:hover:translate-y-0">
              Send
            </button>
          </form>
          {overLimit && (
            <p className="mt-2 text-[0.78rem] text-bad">
              {draft.length}/{MAX_MESSAGE_LENGTH} characters — please shorten the message.
            </p>
          )}
        </div>
      </section>

      <aside
        aria-label="Decision Inspector"
        className={`fixed inset-x-0 bottom-0 z-50 max-h-[85vh] overflow-y-auto rounded-t-2xl border-t border-line bg-paper p-6 shadow-2xl transition-transform duration-300 lg:sticky lg:top-20 lg:z-auto lg:block lg:max-h-[calc(100vh-7rem)] lg:translate-y-0 lg:rounded-xl lg:border lg:bg-paper-raised lg:shadow-none ${
          inspectorOpen ? "translate-y-0" : "translate-y-full"
        }`}
      >
        <div className="mb-5 flex items-center justify-between">
          <p className="flex items-center gap-2 text-sm font-semibold text-ink"><span className="h-2 w-2 rounded-full bg-accent" aria-hidden="true" />Decision Inspector</p>
          <button type="button" onClick={() => setInspectorOpen(false)} className="text-sm text-muted hover:text-ink lg:hidden">
            Close
          </button>
        </div>
        <Inspector turn={inspected?.turn ?? null} highlight={highlight} />
      </aside>
      {inspectorOpen && <button type="button" aria-label="Close inspector" onClick={() => setInspectorOpen(false)} className="fixed inset-0 z-40 bg-ink/30 lg:hidden" />}
    </div>
  );
}
