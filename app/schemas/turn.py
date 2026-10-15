"""
Typed result of one conversation turn (app.services.conversation_service).

Public, JSON-serializable payload for a frontend:
- assistant_message / outcome_kind / conversation_action / failure;
- listing_cards: the listings produced by this turn's search;
- inspector: the decisions behind the answer (Decision Inspector);
- telemetry: a compact per-turn summary.

Never part of the public payload:
- session (exclude=True): the updated server-side state, returned
  in-process so the caller can persist it - it holds full ListingRaw;
- debug (exclude=True): raw internal dicts for the Streamlit developer
  view, only filled when TurnConfig.include_debug is set.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from app.config.settings import (
    FALLBACK_TOP_K_DEFAULT,
    MAX_ITEMS_HARD_CAP,
    SOURCE_NAME,
    TOP_N_DEFAULT,
)
from app.schemas.conversation_response import ListingQueryResponseItem
from app.schemas.conversation_route import ConversationAction, ConversationActionDecision
from app.schemas.query import SearchRequest
from app.schemas.search_response import ConstraintResolutionItem, ConstraintStatus
from app.schemas.session import ConversationSession

OutcomeKind = Literal[
    "search",
    "listing_query",
    "clarification",
    "informational",
    "general_chat",
    "failure",
]

ResponseSource = Literal["llm", "deterministic", "deterministic_fallback", "direct"]


class TurnConfig(BaseModel):
    """Per-call settings; defaults reproduce the Streamlit app's previous ones."""

    source: Literal["fixtures", "apify"] = SOURCE_NAME
    top_n: int = TOP_N_DEFAULT
    max_items: int = MAX_ITEMS_HARD_CAP
    fallback_top_k: int = FALLBACK_TOP_K_DEFAULT
    history_limit: int = 8
    persist_telemetry: bool = True
    include_debug: bool = False


class TurnFailure(BaseModel):
    """An expected stage failure (provider or retrieval) - never an internal exception."""

    stage: str
    code: str


class ListingCardPoint(BaseModel):
    name: str
    label: str
    reason: str


class ListingCard(BaseModel):
    result_id: str
    position: int
    title: str
    url: str | None = None
    price_summary: str | None = None
    budget_summary: str | None = None
    match_tier: str | None = None
    status_label: str | None = None
    confirmed: list[ListingCardPoint] = Field(default_factory=list)
    needs_confirmation: list[ListingCardPoint] = Field(default_factory=list)
    not_satisfied: list[ListingCardPoint] = Field(default_factory=list)


class ResultDecision(BaseModel):
    """Why one search result was selected - per-constraint verdicts and evidence."""

    result_id: str
    title: str
    score: float
    eligibility_status: str | None = None
    match_tier: str | None = None
    selection_reasons: list[str] = Field(default_factory=list)
    blocking_reasons: list[str] = Field(default_factory=list)
    matched_constraints: list[ConstraintStatus] = Field(default_factory=list)
    uncertain_constraints: list[ConstraintStatus] = Field(default_factory=list)
    failed_constraints: list[ConstraintStatus] = Field(default_factory=list)
    constraint_resolution_results: list[ConstraintResolutionItem] = Field(
        default_factory=list
    )


class DecisionInspector(BaseModel):
    routing: ConversationActionDecision | None = None
    effective_action: ConversationAction | None = None
    response_source: ResponseSource
    current_search: SearchRequest | None = None
    search_status: str | None = None
    results: list[ResultDecision] = Field(default_factory=list)
    listing_query: list[ListingQueryResponseItem] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class TurnTelemetry(BaseModel):
    trace_id: str
    latency_ms: float
    estimated_cost_usd: float
    llm_calls: int
    failed_llm_calls: int


class TurnDebug(BaseModel):
    """Developer view (Streamlit debug expander) - raw internal structures."""

    parsed_intent: dict[str, Any] | None = None
    search_request: dict[str, Any] | None = None
    state_after: dict[str, Any] | None = None
    answer_payload: dict[str, Any] | None = None
    telemetry: dict[str, Any] | None = None
    telemetry_log_info: dict[str, Any] | None = None


class TurnResult(BaseModel):
    session_id: str
    assistant_message: str
    outcome_kind: OutcomeKind
    conversation_action: ConversationAction | None = None
    listing_cards: list[ListingCard] = Field(default_factory=list)
    inspector: DecisionInspector
    failure: TurnFailure | None = None
    telemetry: TurnTelemetry

    # In-process only, never serialized (see module docstring).
    session: ConversationSession = Field(exclude=True)
    debug: TurnDebug | None = Field(default=None, exclude=True)
