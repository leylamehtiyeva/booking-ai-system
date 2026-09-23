from __future__ import annotations

from typing import Any, Dict, Optional
from app.observability.trace import RequestTrace
from app.logic.intent_router import build_search_request_adk_async
from app.logic.intent_update import update_search_state_async
from app.logic.request_resolution import resolve_required_search_context
from app.logic.listing_signals import collect_listing_signals
from app.schemas.query import SearchRequest
from app.tools.orchestrate_search_tool import (
    orchestrate_search_request,
)
from app.logic.constraint_evidence_resolution import (
    ConstraintResolutionRequest,
    resolve_constraint_via_textual_evidence,
)
from app.logic.factual_question_extraction import extract_factual_constraints_async
from app.logic.result_candidate_selection import select_shown_candidates
from app.logic.result_query_matching import (
    ResultQueryMatchOutcome,
    match_result_query_against_candidates,
)
from app.schemas.fallback_policy import FallbackPolicy
from app.schemas.result_query import ResultQuery
from app.config.settings import MAX_ITEMS_HARD_CAP
from app.logic.conversation_router import (
    ConversationRoutingError,
    route_conversation_async,
)
from app.schemas.conversation_route import (
    ClarificationReason,
    ConversationAction,
    ConversationActionDecision,
    ConversationDecisionStatus,
    ResultScope,
    RouterInput,
)
from app.schemas.shown_result_set import ShownResultSet
import logging


logger = logging.getLogger(__name__)


def _build_state_payload(state: SearchRequest | None) -> dict[str, Any] | None:
    if state is None:
        return None
    return state.model_dump(mode="json", exclude_none=True)


def _build_shown_results_router_context(
    shown_result_set: ShownResultSet | None,
) -> dict[str, Any] | None:
    """
    Lightweight, router-facing projection of the latest shown result set.

    Deliberately excludes ListingRaw (facilities/rooms/policies/raw) - the
    router only needs enough to resolve "the second one" / a hotel name to
    a result_id, never enough to answer a factual question itself.
    """
    if shown_result_set is None:
        return None

    return {
        "has_shown_results": bool(shown_result_set.items),
        "shown_results": [
            {
                "position": position,
                "result_id": item.result_id,
                "title": item.listing.name,
            }
            for position, item in enumerate(shown_result_set.items, start=1)
        ],
    }


def _resolve_target_reference(
    decision: ConversationActionDecision,
    shown_result_set: ShownResultSet | None,
) -> tuple[str, str | None, str]:
    """
    Deterministic validation of the router's own (unvalidated,
    decision_status == RESOLVED) reference resolution. The LLM is never
    trusted as the source of truth for identity - a target_result_id that
    does not exist in the actual latest ShownResultSet is rejected
    outright, never guessed or looked up.

    Only called when decision.decision_status == RESOLVED - a router that
    already abstained (NEEDS_CLARIFICATION) is handled entirely by the
    top-level gate in handle_user_message, before this function ever runs.

    Returns (result_scope_value, validated_target_result_id, status), where
    status is one of:
    - "not_applicable": result_scope is NOT_APPLICABLE or CURRENT_RESULTS
      (no single target is expected).
    - "resolved": target_result_id matches a shown item.
    - "invalid_reference": the router claimed RESOLVED for a
      SPECIFIC_RESULT question but either gave no target_result_id, or
      gave one that does not exist in the latest shown result set. Both
      cases are treated identically - the router is not trusted either
      way, and the caller turns this into a clarification turn.
    """
    result_scope_value = decision.result_scope.value

    if decision.result_scope != ResultScope.SPECIFIC_RESULT:
        return result_scope_value, None, "not_applicable"

    target_result_id = decision.target_result_id or None

    if target_result_id is None:
        return result_scope_value, None, "invalid_reference"

    valid_ids = {
        item.result_id for item in (shown_result_set.items if shown_result_set else [])
    }

    if target_result_id not in valid_ids:
        return result_scope_value, None, "invalid_reference"

    return result_scope_value, target_result_id, "resolved"


def _build_clarification_question(
    reason: ClarificationReason,
    shown_result_set: ShownResultSet | None,
) -> str:
    """
    Deterministic (no LLM) clarification text. English-only, matching every
    other canned/deterministic message already in this module and in
    conversation_response_generator.py - a full multilingual
    response-generation layer is a separate, larger piece of scope.
    """
    if reason == ClarificationReason.AMBIGUOUS_REFERENCE:
        titles = [
            item.listing.name
            for item in (shown_result_set.items if shown_result_set else [])
            if item.listing.name
        ]

        if len(titles) >= 2:
            return (
                "Which hotel do you mean — "
                + ", ".join(titles[:-1])
                + f" or {titles[-1]}?"
            )

        return "Which of the shown hotels do you mean?"

    if reason == ClarificationReason.AMBIGUOUS_ACTION:
        return (
            "I'm not sure whether you want me to change the search "
            "criteria, or check the hotels already shown. Could you "
            "clarify — update the search, or look at the current results?"
        )

    return "Could you clarify what you mean?"


def _build_clarification_response(
    *,
    reason: ClarificationReason,
    shown_result_set: ShownResultSet | None,
    previous_state_json: dict[str, Any] | None,
    route_debug: dict[str, Any] | None,
    user_message: str,
) -> Dict[str, Any]:
    """
    Read-only turn: reuses the existing need_clarification contract so it
    flows through build_display_answer/conversation_response_adapter.py
    unchanged (need_clarification is checked before action everywhere).
    Deliberately does not set "shown_result_set" - its absence is what
    keeps chat_handler.py from touching the stored snapshot, exactly like
    the existing GENERAL_CHAT/LISTING_QUESTION responses today.
    """
    return {
        "need_clarification": True,
        "questions": [
            _build_clarification_question(reason, shown_result_set)
        ],
        "state": previous_state_json,
        "parsed_intent": {
            "router": route_debug,
            "user_message": user_message,
            "previous_state": previous_state_json,
            "clarification_reason": reason.value,
        },
        "search_request": previous_state_json,
    }


def _build_no_op_update_response(
    *,
    previous_state_json: dict[str, Any] | None,
    route_debug: dict[str, Any] | None,
    user_message: str,
) -> Dict[str, Any]:
    """
    UPDATE_SEARCH whose patch left SearchRequest semantically unchanged
    (updated_search == previous_search - not merely "patch was empty",
    since a patch that only restates an already-true value collapses to
    the same equality after apply_intent_patch's dedup/merge).

    This is a completed, resolved turn - NOT a clarification question and
    NOT a failure. need_clarification=False (that contract genuinely
    means "the user must supply more information before anything can
    proceed", per its every consumer - reusing it here would make e2e
    eval score this as UNCERTAIN and would have the response LLM phrase
    the informational text as a question to the user, per
    conversation_response_agent.py's clarification-outcome instruction).
    "answer" (not "questions") carries the fixed informational text;
    response_type="update_no_op" is what lets
    conversation_response_adapter.py route it correctly without touching
    conversation_action. Deliberately does not set "shown_result_set" -
    its absence is what keeps chat_handler.py from replacing the stored
    snapshot.
    """
    return {
        "need_clarification": False,
        "response_type": "update_no_op",
        "answer": (
            "Your search hasn't changed, so I didn't run a new search."
        ),
        "state": previous_state_json,
        "parsed_intent": {
            "router": route_debug,
            "user_message": user_message,
            "previous_state": previous_state_json,
            "no_op_reason": "search_request_unchanged",
        },
        "search_request": previous_state_json,
    }


def _build_listing_query_selection_failure_response(
    *,
    status: str,
    previous_state_json: dict[str, Any] | None,
    route_debug: dict[str, Any] | None,
    user_message: str,
) -> Dict[str, Any]:
    """
    Candidate selection did not produce a safe set of listings to check -
    this is a conversation-level problem ("which listing?"/"there is
    nothing shown right now"), not a fact about any listing, so it must
    never reach the matcher and must never be reported as UNCERTAIN.
    Reuses the existing need_clarification contract (same shape as
    resolve_required_search_context's own inline branch below), rather
    than inventing a new outcome shape for what is structurally the same
    "the user must tell me more before I can proceed" situation.
    """
    if status == "no_shown_results":
        question = (
            "There are no current search results to check right now - "
            "could you run a search first?"
        )
    else:
        # "invalid_target" (defensive - _resolve_target_reference already
        # validates target_result_id before this point) or
        # "not_applicable".
        question = "I couldn't tell which listing you mean - could you clarify?"

    return {
        "need_clarification": True,
        "questions": [question],
        "state": previous_state_json,
        "parsed_intent": {
            "router": route_debug,
            "user_message": user_message,
            "previous_state": previous_state_json,
            "candidate_selection_status": status,
        },
        "search_request": previous_state_json,
    }


def _build_listing_query_unsupported_response(
    *,
    result_query: ResultQuery,
    previous_state_json: dict[str, Any] | None,
    route_debug: dict[str, Any] | None,
    user_message: str,
) -> Dict[str, Any]:
    """
    The factual extractor found no checkable constraint at all (e.g. "is
    it romantic?"). Resolved, not a clarification question - the user's
    message was clear, it's just outside what the matcher can check - so
    need_clarification=False, mirroring _build_no_op_update_response's
    reasoning for why a "resolved but different" outcome must not reuse
    the needs-more-info contract.
    """
    return {
        "need_clarification": False,
        "response_type": "listing_query_unsupported",
        "answer": (
            "I can't check that from the information in the shown listings."
        ),
        "state": previous_state_json,
        "parsed_intent": {
            "router": route_debug,
            "user_message": user_message,
            "previous_state": previous_state_json,
            "result_query": result_query.model_dump(mode="json"),
        },
        "search_request": previous_state_json,
    }


def _build_listing_query_result_response(
    *,
    result_query: ResultQuery,
    match_outcome: ResultQueryMatchOutcome,
    shown_result_set: ShownResultSet | None,
    previous_state_json: dict[str, Any] | None,
    route_debug: dict[str, Any] | None,
    user_message: str,
) -> Dict[str, Any]:
    """
    Successful structured factual match - carries ResultQueryMatch[] as-is
    (per-constraint YES/NO/UNCERTAIN + evidence, per candidate), with no
    prose generation and no filtering/collapsing across constraints.

    "presentation" is a lightweight, ListingRaw-free projection
    (result_id/position/title) reused from _build_shown_results_router_context
    - this raw application dict is allowed to carry matches/presentation as
    two parallel collections; conversation_response_adapter.py is what
    performs the one deterministic join into ListingQueryConversationOutcome,
    which turns "listing_query_result" into natural language.
    """
    presentation = (_build_shown_results_router_context(shown_result_set) or {}).get(
        "shown_results", []
    )

    listing_query_payload = {
        "query": result_query.model_dump(mode="json"),
        "matches": [m.model_dump(mode="json") for m in match_outcome.results],
        "presentation": presentation,
    }

    return {
        "need_clarification": False,
        "response_type": "listing_query_result",
        "answer": "I checked the shown listings for that.",
        "listing_query_result": listing_query_payload,
        "state": previous_state_json,
        "parsed_intent": {
            "router": route_debug,
            "user_message": user_message,
            "previous_state": previous_state_json,
            "listing_query_result": listing_query_payload,
        },
        "search_request": previous_state_json,
    }


async def _answer_listing_query(
    *,
    user_message: str,
    result_scope: ResultScope,
    target_result_id: str | None,
    shown_result_set: ShownResultSet | None,
    previous_state_json: dict[str, Any] | None,
    route_debug: dict[str, Any] | None,
    trace: RequestTrace,
) -> Dict[str, Any]:
    """
    Orchestrates the already-tested LISTING_QUESTION pipeline:

        candidate selection -> factual extraction -> existing structured
        matcher + existing textual LLM fallback
        (match_result_query_against_candidates)

    match_result_query_against_candidates already applies the same
    structured-matcher/textual-fallback resolution order used by search
    (structured YES/NO is final; structured UNCERTAIN or an unresolved
    constraint goes through the existing
    resolve_listing_constraints_with_fallback) - this function does not
    duplicate any of that decision logic, it only supplies the already-
    selected candidates and already-extracted constraints.

    Only calls existing, independently-tested components - no extraction
    prompt logic, no candidate-selection logic, and no matching/fallback
    loop is duplicated here. Read-only: never calls apply_intent_patch,
    update_search_state_async, orchestrate_search_request,
    set_search_state, or retrieval - result_scope/target_result_id are
    already decided by the router+_resolve_target_reference before this
    function runs, so this function only selects among and matches
    against the listings already in shown_result_set.

    Candidate selection (free, synchronous) runs before factual
    extraction (a paid LLM call) so an unusable candidate set (empty
    ShownResultSet, invalid target) short-circuits before spending that
    call - not just before running the matcher.
    """
    candidate_selection = select_shown_candidates(
        result_scope=result_scope,
        target_result_id=target_result_id,
        shown_result_set=shown_result_set,
    )

    if candidate_selection.status != "ok":
        return _build_listing_query_selection_failure_response(
            status=candidate_selection.status,
            previous_state_json=previous_state_json,
            route_debug=route_debug,
            user_message=user_message,
        )

    with trace.step("listing_question_factual_extraction"):
        result_query = await extract_factual_constraints_async(
            user_message,
            trace=trace,
        )

    if not result_query.constraints:
        return _build_listing_query_unsupported_response(
            result_query=result_query,
            previous_state_json=previous_state_json,
            route_debug=route_debug,
            user_message=user_message,
        )

    match_outcome = await match_result_query_against_candidates(
        result_query=result_query,
        candidate_selection=candidate_selection,
        trace=trace,
    )

    return _build_listing_query_result_response(
        result_query=result_query,
        match_outcome=match_outcome,
        shown_result_set=shown_result_set,
        previous_state_json=previous_state_json,
        route_debug=route_debug,
        user_message=user_message,
    )


def _finalize_response(
    response: Dict[str, Any],
    *,
    trace: RequestTrace,
    action: ConversationAction | None = None,
) -> Dict[str, Any]:
    """
    Attach request-level metadata to the application response.
    """
    if action is not None:
        response["conversation_action"] = action.value

    response["telemetry"] = trace.summary()
    return response


async def _answer_listing_question(
    *,
    user_message: str,
    shown_listing: dict[str, Any] | None,
    previous_state: SearchRequest | None,
    route_debug: dict[str, Any] | None = None,
) -> Dict[str, Any]:
    previous_state_json = _build_state_payload(previous_state)

    if shown_listing is None:
        return {
            "need_clarification": False,
            "response_type": "listing_question",
            "answer": "I need a specific shown listing to answer that question.",
            "state": previous_state_json,
            "parsed_intent": {
                "router": route_debug,
                "user_message": user_message,
            },
            "search_request": previous_state_json,
        }

    signals = collect_listing_signals(shown_listing)

    request = ConstraintResolutionRequest(
        listing_id=shown_listing.get("id"),
        listing_title=shown_listing.get("name"),
        constraint_id=None,
        raw_text=user_message,
        normalized_text=user_message,
        priority="must",
        category="other",
        mapping_status="unresolved",
        evidence_strategy="textual",
        mapped_fields=[],
        structured_value=None,
        resolver_type="textual",
        listing_evidence=[
            {
                "source": s.source,
                "path": s.path,
                "text": s.raw_text or s.text,
            }
            for s in signals
        ],
    )

    result = await resolve_constraint_via_textual_evidence(request)

    return {
        "need_clarification": False,
        "response_type": "listing_question",
        "answer": result.reason,
        "listing_question_result": result.model_dump(mode="json"),
        "state": previous_state_json,
        "parsed_intent": {
            "router": route_debug,
            "user_message": user_message,
            "listing_question_result": result.model_dump(mode="json"),
        },
        "search_request": previous_state_json,
    }




async def handle_user_message(
    user_message: str,
    previous_state: Optional[SearchRequest] = None,
    *,
    source: str = "fixtures",
    top_n: int = 5,
    fallback_policy: FallbackPolicy | None = None,
    max_items: int = MAX_ITEMS_HARD_CAP,
    shown_listing: dict[str, Any] | None = None,
    shown_result_set: ShownResultSet | None = None,
    latest_result_context: dict[str, Any] | None = None,
    trace: RequestTrace | None = None,
) -> Dict[str, Any]:
    previous_state_json = _build_state_payload(
        previous_state
    )
    if trace is None:
        trace = RequestTrace()

    router_input = RouterInput(
        user_message=user_message,
        current_search=previous_state,
        latest_result_context=(
            latest_result_context
            if latest_result_context is not None
            else _build_shown_results_router_context(shown_result_set)
        ),
    )

    try:
        with trace.step("conversation_routing"):
            decision = await route_conversation_async(
                router_input=router_input,
                trace=trace,
            )

    except ConversationRoutingError as exc:
        logger.exception(
            "Conversation routing failed",
            extra={
                "routing_error_code": exc.code,
            },
        )

        return _finalize_response(
            {
                "need_clarification": False,
                "response_type": "routing_unavailable",
                "answer": (
                    "I couldn't process that message right now. "
                    "Your current search has not been changed. "
                    "Please try again."
                ),
                "state": previous_state_json,
                "parsed_intent": {
                    "router": {
                        "status": "failed",
                        "error_code": exc.code,
                    },
                    "user_message": user_message,
                    "previous_state": previous_state_json,
                },
                "search_request": previous_state_json,
            },
            trace=trace,
        )

    route_debug = decision.model_dump(
        mode="json",
        exclude_none=True,
    )

    effective_action = decision.action

    if (
        effective_action
        == ConversationAction.UPDATE_SEARCH
        and previous_state is None
    ):
        effective_action = (
            ConversationAction.START_SEARCH
        )

        route_debug["effective_action"] = (
            effective_action.value
        )
        route_debug["normalization_reason"] = (
            "update_search cannot be executed "
            "without an existing search"
        )

    # Abstention gate: the router is not required to always resolve
    # action/reference. This must run before any action-specific branch -
    # a genuinely ambiguous turn is read-only (no persistent mutation, no
    # search, no ShownResultSet replacement), regardless of what
    # action/result_scope/target_result_id otherwise contain.
    if (
        decision.decision_status
        == ConversationDecisionStatus.NEEDS_CLARIFICATION
    ):
        clarification_reason = (
            decision.clarification_reason
            if decision.clarification_reason != ClarificationReason.NONE
            else ClarificationReason.AMBIGUOUS_ACTION
        )

        return _finalize_response(
            _build_clarification_response(
                reason=clarification_reason,
                shown_result_set=shown_result_set,
                previous_state_json=previous_state_json,
                route_debug=route_debug,
                user_message=user_message,
            ),
            trace=trace,
            action=effective_action,
        )

    if (
        effective_action
        == ConversationAction.LISTING_QUESTION
    ):
        result_scope, target_result_id, reference_resolution_status = (
            _resolve_target_reference(decision, shown_result_set)
        )

        # Source B: the router claimed RESOLVED, but deterministic
        # validation does not trust it (missing or hallucinated
        # target_result_id). Never substitute the nearest/first result -
        # fall back to the same read-only clarification turn as source A.
        if reference_resolution_status == "invalid_reference":
            return _finalize_response(
                _build_clarification_response(
                    reason=ClarificationReason.AMBIGUOUS_REFERENCE,
                    shown_result_set=shown_result_set,
                    previous_state_json=previous_state_json,
                    route_debug=route_debug,
                    user_message=user_message,
                ),
                trace=trace,
                action=effective_action,
            )

        response = await _answer_listing_query(
            user_message=user_message,
            result_scope=decision.result_scope,
            target_result_id=target_result_id,
            shown_result_set=shown_result_set,
            previous_state_json=previous_state_json,
            route_debug=route_debug,
            trace=trace,
        )

        response["result_scope"] = result_scope
        response["target_result_id"] = target_result_id
        response["reference_resolution_status"] = reference_resolution_status

        return _finalize_response(
            response,
            trace=trace,
            action=effective_action,
        )

    if (
        effective_action
        == ConversationAction.GENERAL_CHAT
    ):
        return _finalize_response(
            {
                "need_clarification": False,
                "response_type": "other",
                "answer": (
                    "Hello! I can help you search for "
                    "accommodation, update an existing search, "
                    "or answer questions about shown options."
                ),
                "state": previous_state_json,
                "parsed_intent": {
                    "router": route_debug,
                    "user_message": user_message,
                    "previous_state": previous_state_json,
                },
                "search_request": previous_state_json,
            },
            trace=trace,
            action=effective_action,
        )

    if (
        effective_action
        == ConversationAction.START_SEARCH
    ):
        with trace.step("search_intent_extraction"):
            state = await build_search_request_adk_async(
                user_message,
                trace=trace,
                step="search_intent_extraction",
            )

    elif (
        effective_action
        == ConversationAction.UPDATE_SEARCH
    ):
        if previous_state is None:
            raise RuntimeError(
                "update_search requires an existing "
                "SearchRequest"
            )

        with trace.step("search_state_update"):
            state = await update_search_state_async(
                previous_state,
                user_message,
                trace=trace,
            )

        if state == previous_state:
            # Semantic no-op guard: the patch (empty, or one that only
            # restates already-true values) left SearchRequest unchanged.
            # Re-running retrieval/matching/ranking here would waste work
            # and silently replace ShownResultSet/result_set_id for no
            # reason - short-circuit before the shared search tail below,
            # the same way the abstention gate above already does.
            return _finalize_response(
                _build_no_op_update_response(
                    previous_state_json=previous_state_json,
                    route_debug=route_debug,
                    user_message=user_message,
                ),
                trace=trace,
                action=effective_action,
            )

    else:
        raise RuntimeError(
            "Unsupported conversation action: "
            f"{effective_action}"
        )

    parsed_intent_debug = {
        "router": route_debug,
        "user_message": user_message,
        "previous_state": previous_state_json,
        "constraint_count": len(
            state.constraints or []
        ),
        "constraints": [
            {
                "normalized_text": (
                    constraint.normalized_text
                ),
                "priority": constraint.priority.value,
                "mapping_status": (
                    constraint.mapping_status.value
                ),
            }
            for constraint in (
                state.constraints or []
            )
        ],
    }

    state_json = _build_state_payload(state)
    assert state_json is not None

    resolved = resolve_required_search_context(
        state
    )

    if resolved.need_clarification:
        return _finalize_response(
            {
                "need_clarification": True,
                "questions": resolved.questions,
                "state": state_json,
                "parsed_intent": parsed_intent_debug,
                "search_request": state_json,
            },
            trace=trace,
            action=effective_action,
        )

    search_response, shown_result_set = await orchestrate_search_request(
        state,
        result_limit=top_n,
        candidate_pool_size=max_items,
        fallback_policy=fallback_policy,
        source=source,
        trace=trace,
    )

    result = search_response.model_dump(
        mode="json",
        exclude_none=True,
    )

    result["state"] = state_json
    result["parsed_intent"] = parsed_intent_debug
    result["search_request"] = state_json
    result["shown_result_set"] = shown_result_set.model_dump(
        mode="json",
        exclude_none=True,
    )

    return _finalize_response(
        result,
        trace=trace,
        action=effective_action,
    )