from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
from app.observability.trace import RequestTrace
from app.logic.clarification_continuation import (
    PendingTransition,
    build_pending_clarification,
    resolve_pending_transition,
)
from app.logic.intent_router import build_search_request_adk_async
from app.logic.intent_update import update_search_state_async
from app.logic.date_normalization import InvalidStayDatesError
from app.llm.call_policy import LlmCallError
from app.retrieval.errors import RetrievalError
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
    PendingAwareConversationActionDecision,
    PendingClarification,
    PendingResolution,
    ResultScope,
    RouterInput,
)
from app.schemas.shown_result_set import ShownResultSet
import logging


logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class _ClarifiedChoice:
    """
    A pending AMBIGUOUS_ACTION clarification that the router judged to be
    answered (and Python validated). original_user_message carries the
    semantic request; clarification_reply is the answer, which may add a
    detail to it.
    """

    original_user_message: str
    clarification_reply: str


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


def _build_clarification_cancelled_response(
    *,
    previous_state_json: dict[str, Any] | None,
    route_debug: dict[str, Any],
    user_message: str,
) -> Dict[str, Any]:
    """
    The user dropped a pending clarification ("never mind"). Resolved and
    read-only: no "pending_clarification" (cleared) and no
    "shown_result_set" (snapshot kept).
    """
    return {
        "need_clarification": False,
        "response_type": "clarification_cancelled",
        "answer": "Okay, I've left your search and results as they were.",
        "state": previous_state_json,
        "parsed_intent": {
            "router": route_debug,
            "user_message": user_message,
            "previous_state": previous_state_json,
        },
        "search_request": previous_state_json,
    }


def _build_stale_clarification_response(
    *,
    previous_state_json: dict[str, Any] | None,
    route_debug: dict[str, Any],
    user_message: str,
) -> Dict[str, Any]:
    """
    The user chose to inspect the shown results, but the result set the
    pending question was asked about is no longer the stored one. Never
    run the old question against other listings - ask again instead.
    """
    return {
        "need_clarification": True,
        "questions": [
            "The results I showed earlier are no longer available. "
            "Could you ask your question again about the current results?"
        ],
        "state": previous_state_json,
        "parsed_intent": {
            "router": route_debug,
            "user_message": user_message,
            "previous_state": previous_state_json,
            "clarification_reason": "stale_result_set",
        },
        "search_request": previous_state_json,
    }


_PROCESSING_UNAVAILABLE_MESSAGE = (
    "I couldn't process that message right now. "
    "Your current search has not been changed. "
    "Please try again."
)

_RETRIEVAL_UNAVAILABLE_MESSAGE = (
    "I couldn't search for listings right now. "
    "Your current search has not been changed. "
    "Please try again."
)

# Retrying cannot help with these, so the answer does not suggest it.
_RETRIEVAL_NOT_CONFIGURED_MESSAGE = (
    "Hotel search isn't available right now. "
    "Your current search has not been changed."
)
_NON_RETRYABLE_RETRIEVAL_CODES = frozenset({"configuration_error", "authentication_error"})


# After a pending clarification was resolved (continued or superseded) the
# user's choice is NOT kept when execution fails, so the answer asks for
# the complete request again instead of inviting a bare "try again".
_REQUEST_NOT_COMPLETED_MESSAGE = (
    "I couldn't complete that request, so your search and results have "
    "not been changed. Please send your full request again, including "
    "any details you added."
)


def _stage_failure_answer(
    stage: str,
    code: str,
    user_safe_message: str | None,
    *,
    restate_request: bool = False,
) -> str:
    if restate_request:
        if stage == "retrieval" and user_safe_message:
            return f"{user_safe_message} {_REQUEST_NOT_COMPLETED_MESSAGE}"
        return _REQUEST_NOT_COMPLETED_MESSAGE
    if stage != "retrieval":
        return _PROCESSING_UNAVAILABLE_MESSAGE
    if user_safe_message:
        return user_safe_message
    if code in _NON_RETRYABLE_RETRIEVAL_CODES:
        return _RETRIEVAL_NOT_CONFIGURED_MESSAGE
    return _RETRIEVAL_UNAVAILABLE_MESSAGE


def _build_stage_failure_response(
    *,
    stage: str,
    code: str,
    previous_state_json: dict[str, Any] | None,
    route_debug: dict[str, Any] | None,
    user_message: str,
    user_safe_message: str | None = None,
    restate_request: bool = False,
) -> Dict[str, Any]:
    """
    A fatal stage failure (LLM provider or retrieval) - the turn did not
    complete. Mirrors routing_unavailable: the previous SearchRequest is
    returned unchanged and "shown_result_set" is left out, so neither the
    persisted search nor the stored snapshot is replaced. The answer is a
    fixed user-safe text - the retriever's own user_message for a search it
    cannot run (e.g. an unsupported destination), otherwise a generic one;
    stage/code are for telemetry and the UI's debug view, never provider
    messages.
    """
    failure = {"stage": stage, "code": code}

    return {
        "need_clarification": False,
        "response_type": "processing_unavailable",
        "answer": _stage_failure_answer(
            stage, code, user_safe_message, restate_request=restate_request
        ),
        "failure": failure,
        "state": previous_state_json,
        "parsed_intent": {
            "router": route_debug,
            "user_message": user_message,
            "previous_state": previous_state_json,
            "failure": failure,
        },
        "search_request": previous_state_json,
    }


def _build_invalid_dates_response(
    exc: InvalidStayDatesError,
    *,
    state_json: dict[str, Any] | None,
    route_debug: dict[str, Any] | None,
    user_message: str,
    previous_state_json: dict[str, Any] | None,
) -> Dict[str, Any]:
    """
    The user's dates cannot form a valid future stay (past, reversed,
    impossible). Same need_clarification shape as the missing-context
    branch of handle_user_message - no search runs and, by leaving out
    "shown_result_set", the stored snapshot is untouched.
    """
    return {
        "need_clarification": True,
        "questions": [exc.question],
        "state": state_json,
        "parsed_intent": {
            "router": route_debug,
            "user_message": user_message,
            "previous_state": previous_state_json,
            "date_issue": exc.issue.value,
        },
        "search_request": state_json,
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
    original_question: str | None = None,
    restate_request: bool = False,
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

    try:
        with trace.step("listing_question_factual_extraction"):
            result_query = await extract_factual_constraints_async(
                user_message,
                trace=trace,
                original_question=original_question,
            )
    except LlmCallError as exc:
        return _build_stage_failure_response(
            stage="listing_question_factual_extraction",
            code=exc.code,
            previous_state_json=previous_state_json,
            route_debug=route_debug,
            user_message=user_message,
            restate_request=restate_request,
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
    pending_clarification: PendingClarification | None = None,
    trace: RequestTrace | None = None,
) -> Dict[str, Any]:
    """
    One domain turn. The returned "pending_clarification" key (a dumped
    PendingClarification) is what the caller stores for the next turn;
    its absence clears any pending clarification.

    A pending clarification is semantic input to the router
    (RouterInput.pending_clarification). The router judges how the message
    relates to it (pending_resolution); clarification_continuation.
    resolve_pending_transition checks that judgement against the pending
    state:

    - CONTINUE  -> the chosen action runs on the ORIGINAL request (a
      listing question only against the same result set, and a specific
      listing only once its reference is valid); cleared;
    - CANCEL    -> read-only acknowledgement; cleared;
    - KEEP      -> the clarification is asked again, nothing runs; kept;
    - SUPERSEDE -> the message is handled on its own and the original
      request is never applied; cleared (or replaced by a new one if the
      message itself is ambiguous).

    Failures: if routing itself fails (no semantic decision was made) the
    pending clarification is kept. Once a decision was made (CONTINUE or
    SUPERSEDE), an execution failure clears it - it is never revived or
    retried with only part of what the user said - and the answer asks
    for the complete request again. SearchRequest/ShownResultSet stay at
    the last accepted state either way.
    """
    if trace is None:
        trace = RequestTrace()

    result = await _process_message(
        user_message,
        previous_state,
        source=source,
        top_n=top_n,
        fallback_policy=fallback_policy,
        max_items=max_items,
        shown_result_set=shown_result_set,
        latest_result_context=latest_result_context,
        pending_clarification=pending_clarification,
        trace=trace,
    )

    if (
        pending_clarification is not None
        and result.get("response_type") == "routing_unavailable"
    ):
        result["pending_clarification"] = pending_clarification.model_dump(mode="json")

    return result


async def _apply_clarified_search_update(
    previous_state: SearchRequest,
    clarified: _ClarifiedChoice,
    *,
    trace: RequestTrace,
) -> SearchRequest:
    """
    Contract between clarification continuation and Intent Update - two
    ordinary updates, never one concatenated message:

    1. the original message, flagged as a confirmed search change (the
       user has just chosen to change the search, so a question-shaped
       "only the ones with parking?" is a requirement);
    2. the clarification reply itself, as a follow-up update on top of 1,
       with the original as read-only context (so "make it optional" can
       resolve "it") and the instruction that it is already applied.
       Intent Update decides what the reply changes: a bare "yes, change
       the search" yields an empty patch, while "...but make it optional"
       restates parking, and the existing restatement rule makes it nice.

    Two patches, not one: within a single patch duplicate constraints
    merge as "stronger priority wins", so must + nice would stay must.
    Only the final state is returned; nothing is persisted in between.
    """
    state = await update_search_state_async(
        previous_state,
        clarified.original_user_message,
        trace=trace,
        confirmed_search_change=True,
    )

    return await update_search_state_async(
        state,
        clarified.clarification_reply,
        trace=trace,
        answered_request=clarified.original_user_message,
    )


def _continuation_decision(
    decision: ConversationActionDecision,
) -> ConversationActionDecision:
    """
    The read-only reading of an ambiguous action is a question about the
    shown set: a listing continuation without a scope gets CURRENT_RESULTS
    (a specific target, if the router gave one, is kept and validated by
    _resolve_target_reference as usual).
    """
    if (
        decision.action == ConversationAction.LISTING_QUESTION
        and decision.result_scope == ResultScope.NOT_APPLICABLE
    ):
        return decision.model_copy(
            update={"result_scope": ResultScope.CURRENT_RESULTS, "target_result_id": ""}
        )
    return decision


def _build_routing_unavailable_response(
    *,
    code: str,
    previous_state_json: dict[str, Any] | None,
    user_message: str,
    trace: RequestTrace,
) -> Dict[str, Any]:
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
                    "error_code": code,
                },
                "user_message": user_message,
                "previous_state": previous_state_json,
            },
            "search_request": previous_state_json,
        },
        trace=trace,
    )


async def _process_message(
    user_message: str,
    previous_state: Optional[SearchRequest],
    *,
    source: str,
    top_n: int,
    fallback_policy: FallbackPolicy | None,
    max_items: int,
    shown_result_set: ShownResultSet | None,
    latest_result_context: dict[str, Any] | None,
    pending_clarification: PendingClarification | None,
    trace: RequestTrace,
) -> Dict[str, Any]:
    previous_state_json = _build_state_payload(
        previous_state
    )

    router_input = RouterInput(
        user_message=user_message,
        current_search=previous_state,
        latest_result_context=(
            latest_result_context
            if latest_result_context is not None
            else _build_shown_results_router_context(shown_result_set)
        ),
        pending_clarification=pending_clarification,
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

        return _build_routing_unavailable_response(
            code=exc.code,
            previous_state_json=previous_state_json,
            user_message=user_message,
            trace=trace,
        )

    route_debug = decision.model_dump(
        mode="json",
        exclude_none=True,
    )

    clarified: _ClarifiedChoice | None = None

    if pending_clarification is not None:
        if not isinstance(decision, PendingAwareConversationActionDecision):
            # Any router (Gemini via the strict output schema, or an
            # injected adapter) must judge the pending clarification. A
            # decision without that judgement is a contract failure, not a
            # new request - nothing runs and the pending request is kept.
            logger.error(
                "Router returned no pending_resolution on a pending turn",
                extra={"routing_error_code": "missing_pending_resolution"},
            )
            return _build_routing_unavailable_response(
                code="missing_pending_resolution",
                previous_state_json=previous_state_json,
                user_message=user_message,
                trace=trace,
            )

        transition = resolve_pending_transition(pending_clarification, decision)
        resolution = decision.pending_resolution
        # Downstream only ever sees the six-field contract.
        decision = decision.to_decision()

        route_debug["pending_clarification"] = pending_clarification.model_dump(mode="json")
        route_debug["pending_transition"] = transition.value

        if transition == PendingTransition.CANCEL:
            return _finalize_response(
                _build_clarification_cancelled_response(
                    previous_state_json=previous_state_json,
                    route_debug=route_debug,
                    user_message=user_message,
                ),
                trace=trace,
                action=ConversationAction.GENERAL_CHAT,
            )

        if transition == PendingTransition.KEEP:
            # The user chose to look at one shown listing but the router
            # could not tell which: ask which listing, not the original
            # question again. The pending request itself is unchanged.
            asks_for_reference = (
                resolution == PendingResolution.CONTINUE_PENDING
                and decision.decision_status == ConversationDecisionStatus.NEEDS_CLARIFICATION
                and decision.action == ConversationAction.LISTING_QUESTION
                and decision.result_scope == ResultScope.SPECIFIC_RESULT
                and decision.clarification_reason == ClarificationReason.AMBIGUOUS_REFERENCE
            )
            response = _build_clarification_response(
                reason=(
                    ClarificationReason.AMBIGUOUS_REFERENCE
                    if asks_for_reference
                    else pending_clarification.kind
                ),
                shown_result_set=shown_result_set,
                previous_state_json=previous_state_json,
                route_debug=route_debug,
                user_message=user_message,
            )
            response["pending_clarification"] = pending_clarification.model_dump(mode="json")
            return _finalize_response(
                response,
                trace=trace,
                action=decision.action,
            )

        if transition == PendingTransition.CONTINUE:
            if decision.action == ConversationAction.LISTING_QUESTION and (
                shown_result_set is None
                or not shown_result_set.items
                or shown_result_set.result_set_id != pending_clarification.result_set_id
            ):
                return _finalize_response(
                    _build_stale_clarification_response(
                        previous_state_json=previous_state_json,
                        route_debug=route_debug,
                        user_message=user_message,
                    ),
                    trace=trace,
                    action=decision.action,
                )

            if (
                decision.action == ConversationAction.LISTING_QUESTION
                and decision.result_scope == ResultScope.SPECIFIC_RESULT
                and _resolve_target_reference(decision, shown_result_set)[2]
                != "resolved"
            ):
                # The user picked one listing but the reference does not
                # match the shown set: never widen it to all results. Ask
                # which one, keeping the original question pending.
                response = _build_clarification_response(
                    reason=ClarificationReason.AMBIGUOUS_REFERENCE,
                    shown_result_set=shown_result_set,
                    previous_state_json=previous_state_json,
                    route_debug=route_debug,
                    user_message=user_message,
                )
                response["pending_clarification"] = pending_clarification.model_dump(
                    mode="json"
                )
                return _finalize_response(
                    response,
                    trace=trace,
                    action=decision.action,
                )

            decision = _continuation_decision(decision)
            clarified = _ClarifiedChoice(
                original_user_message=pending_clarification.original_user_message,
                clarification_reply=user_message,
            )

    return await _execute_decision(
        decision,
        route_debug=route_debug,
        user_message=user_message,
        clarified=clarified,
        after_pending_resolution=pending_clarification is not None,
        previous_state=previous_state,
        previous_state_json=previous_state_json,
        source=source,
        top_n=top_n,
        fallback_policy=fallback_policy,
        max_items=max_items,
        shown_result_set=shown_result_set,
        trace=trace,
    )


async def _execute_decision(
    decision: ConversationActionDecision,
    *,
    route_debug: dict[str, Any],
    user_message: str,
    clarified: _ClarifiedChoice | None,
    after_pending_resolution: bool,
    previous_state: SearchRequest | None,
    previous_state_json: dict[str, Any] | None,
    source: str,
    top_n: int,
    fallback_policy: FallbackPolicy | None,
    max_items: int,
    shown_result_set: ShownResultSet | None,
    trace: RequestTrace,
) -> Dict[str, Any]:
    """
    Guards and dispatch for one routing decision - the same path whether
    or not a pending clarification was involved. clarified: the decision
    continues a pending request (its original message drives extraction).
    after_pending_resolution: a pending clarification was resolved this
    turn, so an execution failure asks for the complete request again.
    """

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

        response = _build_clarification_response(
            reason=clarification_reason,
            shown_result_set=shown_result_set,
            previous_state_json=previous_state_json,
            route_debug=route_debug,
            user_message=user_message,
        )

        # Application state, not model output: the abstained message is
        # kept so the user's answer can be applied to it next turn.
        pending = build_pending_clarification(
            reason=clarification_reason,
            user_message=user_message,
            previous_state=previous_state,
            shown_result_set=shown_result_set,
        )
        if pending is not None:
            response["pending_clarification"] = pending.model_dump(mode="json")

        return _finalize_response(
            response,
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
            original_question=(
                clarified.original_user_message if clarified is not None else None
            ),
            restate_request=after_pending_resolution,
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
        try:
            with trace.step("search_intent_extraction"):
                state = await build_search_request_adk_async(
                    user_message,
                    trace=trace,
                    step="search_intent_extraction",
                )
        except LlmCallError as exc:
            return _finalize_response(
                _build_stage_failure_response(
                    stage="search_intent_extraction",
                    code=exc.code,
                    previous_state_json=previous_state_json,
                    route_debug=route_debug,
                    user_message=user_message,
                    restate_request=after_pending_resolution,
                ),
                trace=trace,
                action=effective_action,
            )
        except InvalidStayDatesError as exc:
            # A new search replaces the old one, so the partial request
            # (everything except the dates) becomes the state to complete.
            return _finalize_response(
                _build_invalid_dates_response(
                    exc,
                    state_json=_build_state_payload(exc.partial_state),
                    route_debug=route_debug,
                    user_message=user_message,
                    previous_state_json=previous_state_json,
                ),
                trace=trace,
                action=effective_action,
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

        try:
            with trace.step("search_state_update"):
                if clarified is not None:
                    state = await _apply_clarified_search_update(
                        previous_state,
                        clarified,
                        trace=trace,
                    )
                else:
                    state = await update_search_state_async(
                        previous_state,
                        user_message,
                        trace=trace,
                    )
        except LlmCallError as exc:
            return _finalize_response(
                _build_stage_failure_response(
                    stage="search_state_update",
                    code=exc.code,
                    previous_state_json=previous_state_json,
                    route_debug=route_debug,
                    user_message=user_message,
                    restate_request=after_pending_resolution,
                ),
                trace=trace,
                action=effective_action,
            )
        except InvalidStayDatesError as exc:
            # The update is rejected as a whole - the previous, valid
            # search stays in place until the user confirms the dates.
            response = _build_invalid_dates_response(
                exc,
                state_json=previous_state_json,
                route_debug=route_debug,
                user_message=user_message,
                previous_state_json=previous_state_json,
            )
            if clarified is not None:
                # The continued request is dropped with the rejected
                # update; say so, or its other parts would vanish silently.
                response["questions"].append(_REQUEST_NOT_COMPLETED_MESSAGE)
            return _finalize_response(
                response,
                trace=trace,
                action=effective_action,
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

    try:
        search_response, shown_result_set = await orchestrate_search_request(
            state,
            result_limit=top_n,
            candidate_pool_size=max_items,
            fallback_policy=fallback_policy,
            source=source,
            trace=trace,
        )
    except RetrievalError as exc:
        # The new SearchRequest was never searched - keep the previous one,
        # so the persisted search always matches the shown results.
        return _finalize_response(
            _build_stage_failure_response(
                stage="retrieval",
                code=exc.code,
                previous_state_json=previous_state_json,
                route_debug=route_debug,
                user_message=user_message,
                user_safe_message=exc.user_message,
                restate_request=after_pending_resolution,
            ),
            trace=trace,
            action=effective_action,
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