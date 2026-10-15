"""
Application service: one user turn, independent of any UI.

    process_turn(session, user_message) -> TurnResult

Orchestrates the existing pieces in the order the Streamlit handler used
to: domain processing (handle_user_message) -> typed response input ->
response wording (LLM, deterministic, or direct passthrough) -> final
telemetry -> new session. No domain decision is made here.

State consistency: the input session is never mutated, and the updated
session is only built after every step above has succeeded. Expected
provider/retrieval failures arrive as typed results from P0-1's handling
inside handle_user_message (previous state, no new snapshot); anything
else raised here is a bug and propagates to the caller unchanged, with no
state committed.
"""

from __future__ import annotations

import logging
from typing import Any

from app.logic.conversation_flow import handle_user_message
from app.logic.conversation_response_adapter import build_conversation_response_input
from app.logic.conversation_response_llm import generate_conversation_response_with_llm
from app.logic.display_answer import build_display_answer, build_result_links
from app.observability.telemetry_logger import save_telemetry_record
from app.observability.trace import RequestTrace
from app.schemas.conversation_response import (
    ConversationMessage,
    ConversationResponseInput,
    ListingQueryConversationOutcome,
    SearchConversationOutcome,
)
from app.schemas.conversation_route import ConversationAction, ConversationActionDecision
from app.schemas.fallback_policy import FallbackPolicy
from app.schemas.query import SearchRequest
from app.schemas.session import MAX_SESSION_MESSAGES, ConversationSession
from app.schemas.shown_result_set import ShownResultSet
from app.schemas.turn import (
    DecisionInspector,
    ListingCard,
    ListingCardPoint,
    OutcomeKind,
    ResponseSource,
    ResultDecision,
    TurnConfig,
    TurnDebug,
    TurnFailure,
    TurnResult,
    TurnTelemetry,
)

logger = logging.getLogger(__name__)


def build_recent_conversation_messages(
    messages: list[dict[str, Any]],
    *,
    limit: int = 8,
) -> list[ConversationMessage]:
    """
    Convert chat messages into the small conversation-history contract
    used by the response layer.

    Debug data and other UI-specific fields are intentionally excluded.
    """
    if limit <= 0:
        return []

    visible_messages: list[ConversationMessage] = []

    for message in messages:
        role = message.get("role")
        content = message.get("content")

        if role not in {"user", "assistant"}:
            continue

        if not isinstance(content, str) or not content.strip():
            continue

        visible_messages.append(
            ConversationMessage(
                role=role,
                content=content,
            )
        )

    return visible_messages[-limit:]


async def process_turn(
    session: ConversationSession,
    user_message: str,
    *,
    config: TurnConfig | None = None,
) -> TurnResult:
    config = config or TurnConfig()

    # History BEFORE the current user message.
    recent_messages = build_recent_conversation_messages(
        [message.model_dump() for message in session.messages],
        limit=config.history_limit,
    )

    # One trace for the whole turn: router -> domain -> response wording.
    trace = RequestTrace()

    result = await handle_user_message(
        user_message=user_message,
        previous_state=session.search_request,
        source=config.source,
        top_n=config.top_n,
        fallback_policy=FallbackPolicy(enabled=True, top_k=config.fallback_top_k),
        max_items=config.max_items,
        shown_result_set=session.shown_result_set,
        trace=trace,
    )

    # Whether a typed response exists at all is decided by the adapter from
    # the result's shape; None only for routing_unavailable (no action).
    response_input = build_conversation_response_input(
        user_message=user_message,
        result=result,
        recent_messages=recent_messages,
    )

    assistant_message, answer_payload = build_display_answer(result)
    response_source: ResponseSource = "direct"

    if response_input is not None:
        generation = await generate_conversation_response_with_llm(
            response_input,
            trace=trace,
        )
        assistant_message = generation.text
        response_source = generation.source

        result_links = build_result_links(answer_payload)
        if result_links:
            assistant_message = f"{assistant_message}\n\n{result_links}"

    # Final telemetry includes the response-generation call, which the
    # snapshot inside handle_user_message's result does not.
    telemetry = trace.summary()
    result["telemetry"] = telemetry

    telemetry_log_info = (
        _persist_telemetry(result, user_message=user_message, config=config)
        if config.persist_telemetry
        else None
    )

    updated_session = _build_updated_session(
        session,
        result=result,
        user_message=user_message,
        assistant_message=assistant_message,
    )

    return TurnResult(
        session_id=updated_session.session_id,
        assistant_message=assistant_message,
        outcome_kind=_outcome_kind(response_input),
        conversation_action=_conversation_action(result),
        listing_cards=_listing_cards(answer_payload, response_input),
        inspector=_build_inspector(
            result,
            response_input=response_input,
            response_source=response_source,
            current_search=updated_session.search_request,
        ),
        failure=_failure(result),
        telemetry=_turn_telemetry(telemetry),
        session=updated_session,
        debug=(
            TurnDebug(
                parsed_intent=result.get("parsed_intent"),
                search_request=result.get("search_request"),
                state_after=result.get("state"),
                answer_payload=answer_payload,
                telemetry=telemetry,
                telemetry_log_info=telemetry_log_info,
            )
            if config.include_debug
            else None
        ),
    )


# --- session ---------------------------------------------------------------


def _build_updated_session(
    session: ConversationSession,
    *,
    result: dict[str, Any],
    user_message: str,
    assistant_message: str,
) -> ConversationSession:
    """
    Commit rules (unchanged from the Streamlit handler):
    - SearchRequest is whatever handle_user_message returned as "state" -
      the previous one for read-only/failed turns, the domain-approved
      partial one for a START_SEARCH clarification;
    - ShownResultSet is replaced only when "shown_result_set" is present,
      i.e. a search actually ran and produced the shown snapshot.
    """
    state = result.get("state")

    shown_result_set = (
        ShownResultSet.model_validate(result["shown_result_set"])
        if "shown_result_set" in result
        else session.shown_result_set
    )

    messages = [
        *session.messages,
        ConversationMessage(role="user", content=user_message),
        ConversationMessage(role="assistant", content=assistant_message),
    ][-MAX_SESSION_MESSAGES:]

    return ConversationSession(
        session_id=session.session_id,
        search_request=SearchRequest.model_validate(state) if state is not None else None,
        shown_result_set=shown_result_set,
        messages=messages,
    )


# --- telemetry -------------------------------------------------------------


def _persist_telemetry(
    result: dict[str, Any],
    *,
    user_message: str,
    config: TurnConfig,
) -> dict[str, Any]:
    return save_telemetry_record(
        telemetry=result["telemetry"],
        user_message=user_message,
        source=config.source,
        top_n=config.top_n,
        max_items=config.max_items,
        result_summary={
            "status": result.get("status"),
            "need_clarification": result.get("need_clarification"),
            "results_count": len(result.get("results") or []),
            "questions": result.get("questions"),
        },
    )


def _turn_telemetry(telemetry: dict[str, Any]) -> TurnTelemetry:
    calls = telemetry["llm"]["calls"]
    return TurnTelemetry(
        trace_id=telemetry["trace_id"],
        latency_ms=telemetry["latency_ms"]["total_observed"],
        estimated_cost_usd=telemetry["cost"]["estimated_total_usd"],
        llm_calls=len(calls),
        failed_llm_calls=sum(1 for call in calls if not call.get("success", True)),
    )


# --- result projection -----------------------------------------------------


def _outcome_kind(response_input: ConversationResponseInput | None) -> OutcomeKind:
    if response_input is None:
        # build_conversation_response_input returns None only for
        # routing_unavailable - a failure with no ConversationAction.
        return "failure"
    return response_input.outcome.kind


def _conversation_action(result: dict[str, Any]) -> ConversationAction | None:
    action = result.get("conversation_action")
    return ConversationAction(action) if action is not None else None


def _failure(result: dict[str, Any]) -> TurnFailure | None:
    response_type = result.get("response_type")

    if response_type == "processing_unavailable":
        return TurnFailure(**result["failure"])

    if response_type == "routing_unavailable":
        router = (result.get("parsed_intent") or {}).get("router") or {}
        return TurnFailure(
            stage="conversation_routing",
            code=router.get("error_code") or "unknown",
        )

    return None


def _routing_decision(result: dict[str, Any]) -> ConversationActionDecision | None:
    router = (result.get("parsed_intent") or {}).get("router") or {}
    if "action" not in router:
        return None  # routing failed - there is no decision to show
    return ConversationActionDecision.model_validate(
        {name: router[name] for name in ConversationActionDecision.model_fields}
    )


def _listing_cards(
    answer_payload: dict[str, Any] | None,
    response_input: ConversationResponseInput | None,
) -> list[ListingCard]:
    if response_input is None or not isinstance(
        response_input.outcome, SearchConversationOutcome
    ):
        return []

    cards: list[ListingCard] = []
    for position, item in enumerate((answer_payload or {}).get("top_results") or [], start=1):
        explanation = item.get("answer_explanation") or {}
        cards.append(
            ListingCard(
                result_id=item["result_id"],
                position=position,
                title=item.get("title") or "",
                url=item.get("url"),
                price_summary=item.get("price_summary"),
                budget_summary=item.get("budget_summary"),
                match_tier=item.get("match_tier"),
                status_label=explanation.get("status_label"),
                confirmed=[ListingCardPoint(**p) for p in explanation.get("confirmed") or []],
                needs_confirmation=[
                    ListingCardPoint(**p) for p in explanation.get("needs_confirmation") or []
                ],
                not_satisfied=[ListingCardPoint(**p) for p in explanation.get("not_satisfied") or []],
            )
        )
    return cards


def _build_inspector(
    result: dict[str, Any],
    *,
    response_input: ConversationResponseInput | None,
    response_source: ResponseSource,
    current_search: SearchRequest | None,
) -> DecisionInspector:
    outcome = response_input.outcome if response_input is not None else None

    search_status = None
    results: list[ResultDecision] = []
    notes: list[str] = []
    if isinstance(outcome, SearchConversationOutcome):
        search_response = outcome.search_response
        search_status = search_response.status.value
        notes = list(search_response.debug_notes or [])
        results = [
            ResultDecision(
                result_id=r.result_id,
                title=r.title,
                score=r.score,
                eligibility_status=r.eligibility_status,
                match_tier=r.match_tier,
                selection_reasons=list(r.selection_reasons),
                blocking_reasons=list(r.blocking_reasons),
                matched_constraints=list(r.matched_constraints),
                uncertain_constraints=list(r.uncertain_constraints),
                failed_constraints=list(r.failed_constraints),
                constraint_resolution_results=list(r.constraint_resolution_results),
            )
            for r in search_response.results
        ]

    listing_query = (
        list(outcome.items) if isinstance(outcome, ListingQueryConversationOutcome) else []
    )

    return DecisionInspector(
        routing=_routing_decision(result),
        effective_action=_conversation_action(result),
        response_source=response_source,
        current_search=current_search,
        search_status=search_status,
        results=results,
        listing_query=listing_query,
        notes=notes,
    )
