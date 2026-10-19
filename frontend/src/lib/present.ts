// Presentation only: wording and formatting of values the API returned.
import type { ListingCardPoint, SearchRequest } from "./types";

export const OPTIONAL_REASON = "Not confirmed (optional)"; // app/logic/normalize_search_response.py

export const STATUS_TEXT: Record<string, string> = {
  fully_confirmed_match: "All requirements confirmed",
  strong_match_with_caveats: "Optional details unconfirmed",
  partially_confirmed_match: "Some requirements unconfirmed",
  relevant_option: "Relevant option",
  not_matched: "Does not meet every requirement",
};

export const TIER_TEXT: Record<string, string> = {
  strong: "Strong match",
  partial: "Partial match",
  weak: "Weak match",
};

export const ACTION_TEXT: Record<string, string> = {
  start_search: "New search",
  update_search: "Refine search",
  listing_question: "Question about a listing",
  general_chat: "General chat",
};

export const SOURCE_TEXT: Record<string, string> = {
  llm: "Worded by the LLM from decided outcomes",
  deterministic: "Rendered deterministically",
  deterministic_fallback: "Deterministic fallback wording",
  direct: "Direct message",
};

export const isOptional = (point: ListingCardPoint) => point.reason === OPTIONAL_REASON;

/** Fixture listings link to example.com placeholders, not real pages. */
export function isSampleUrl(url: string | null): boolean {
  if (!url) return true;
  try {
    return new URL(url).hostname.endsWith("example.com");
  } catch {
    return true;
  }
}

export function formatDate(iso: string | null): string | null {
  if (!iso) return null;
  const date = new Date(`${iso}T00:00:00`);
  return Number.isNaN(date.getTime())
    ? iso
    : date.toLocaleDateString("en-GB", { day: "numeric", month: "short", year: "numeric" });
}

export function stayText(search: SearchRequest): string | null {
  const from = formatDate(search.check_in);
  const to = formatDate(search.check_out);
  if (from && to) return `${from} – ${to}`;
  if (from) return `From ${from}`;
  return null;
}

export function budgetText(search: SearchRequest): string | null {
  const price = search.filters?.price;
  if (!price || (price.max_amount == null && price.min_amount == null)) return null;
  const value = price.max_amount ?? price.min_amount;
  const currency = price.currency ?? "";
  const money = /^[$€£₼]$/.test(currency) ? `${currency}${value}` : `${value} ${currency}`;
  const amount = price.max_amount != null ? `up to ${money}` : `from ${money}`;
  const scope = price.scope === "per_night" ? " per night" : price.scope === "total_stay" ? " for the stay" : "";
  return `${amount.trim()}${scope}`;
}

export function guestsText(search: SearchRequest): string {
  const parts = [`${search.adults} adult${search.adults === 1 ? "" : "s"}`];
  if (search.children) parts.push(`${search.children} child${search.children === 1 ? "" : "ren"}`);
  return parts.join(", ");
}

export const PRIORITY_TEXT: Record<string, string> = { must: "Required", nice: "Optional", forbidden: "Excluded" };
