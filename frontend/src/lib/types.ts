// The public TurnResult contract of POST /api/chat
// (app/schemas/turn.py). Mirrors the backend; no field is derived here.

export type OutcomeKind =
  | "search"
  | "listing_query"
  | "clarification"
  | "informational"
  | "general_chat"
  | "failure";

export type ConversationAction =
  | "start_search"
  | "update_search"
  | "listing_question"
  | "general_chat";

export interface ListingCardPoint {
  name: string;
  label: string;
  reason: string;
}

export interface ListingCard {
  result_id: string;
  position: number;
  title: string;
  url: string | null;
  price_summary: string | null;
  budget_summary: string | null;
  match_tier: string | null;
  status_label: string | null;
  confirmed: ListingCardPoint[];
  needs_confirmation: ListingCardPoint[];
  not_satisfied: ListingCardPoint[];
}

export interface UserConstraint {
  id?: string;
  raw_text: string;
  normalized_text: string;
  priority: "must" | "nice" | "forbidden" | string;
  category: string;
  mapping_status: string;
  mapped_fields: string[];
  evidence_strategy: string;
}

export interface PriceConstraint {
  min_amount: number | null;
  max_amount: number | null;
  currency: string | null;
  scope: "per_night" | "total_stay" | null;
}

export interface SearchFilters {
  bedrooms_min: number | null;
  bedrooms_max: number | null;
  area_sqm_min: number | null;
  area_sqm_max: number | null;
  bathrooms_min: number | null;
  bathrooms_max: number | null;
  price: PriceConstraint | null;
}

export interface SearchRequest {
  city: string | null;
  check_in: string | null;
  check_out: string | null;
  adults: number;
  children: number;
  rooms: number;
  filters: SearchFilters | null;
  property_types: string[] | null;
  occupancy_types: string[] | null;
  constraints: UserConstraint[];
}

export interface ConstraintStatus {
  name: string;
  status: string;
  reason: string | null;
  constraint: UserConstraint | null;
}

export interface ConstraintResolutionItem {
  raw_text: string;
  normalized_text: string;
  resolver_type: string;
  decision: string;
  resolution_status: string;
  confidence: number | null;
  reason: string;
  evidence: { snippet: string; source?: string | null; path?: string | null }[];
}

export interface ResultDecision {
  result_id: string;
  title: string;
  score: number;
  eligibility_status: string | null;
  match_tier: string | null;
  selection_reasons: string[];
  blocking_reasons: string[];
  matched_constraints: ConstraintStatus[];
  uncertain_constraints: ConstraintStatus[];
  failed_constraints: ConstraintStatus[];
  constraint_resolution_results: ConstraintResolutionItem[];
}

export interface ConversationActionDecision {
  action: ConversationAction;
  reason: string;
  decision_status: string;
  clarification_reason: string;
  result_scope: string;
  target_result_id: string;
}

export interface ListingQueryItem {
  position: number;
  title: string | null;
  match: {
    result_id: string;
    constraint_results: {
      raw_text: string;
      field?: string | null;
      value: "YES" | "NO" | "UNCERTAIN" | string;
      reason?: string | null;
      evidence?: unknown[];
    }[];
  };
}

export interface DecisionInspector {
  routing: ConversationActionDecision | null;
  effective_action: ConversationAction | null;
  response_source: "llm" | "deterministic" | "deterministic_fallback" | "direct";
  current_search: SearchRequest | null;
  search_status: string | null;
  results: ResultDecision[];
  listing_query: ListingQueryItem[];
  notes: string[];
}

export interface TurnTelemetry {
  trace_id: string;
  latency_ms: number;
  estimated_cost_usd: number;
  llm_calls: number;
  failed_llm_calls: number;
}

export interface TurnResult {
  session_id: string;
  assistant_message: string;
  outcome_kind: OutcomeKind;
  conversation_action: ConversationAction | null;
  listing_cards: ListingCard[];
  inspector: DecisionInspector;
  failure: { stage: string; code: string } | null;
  telemetry: TurnTelemetry;
}
