from __future__ import annotations

from typing import Any, Dict, Optional
from app.observability.trace import RequestTrace
from app.logic.intent_router import build_search_request_adk_async
from app.logic.intent_update import update_search_state_async
from app.logic.request_resolution import resolve_required_search_context
from app.logic.listing_signals import collect_listing_signals
from app.schemas.query import SearchRequest
from app.tools.orchestrate_search_tool import orchestrate_search
from app.logic.constraint_evidence_resolution import (
    ConstraintResolutionRequest,
    resolve_constraint_via_textual_evidence,
)
from app.schemas.fallback_policy import FallbackPolicy
from app.config.settings import MAX_ITEMS_HARD_CAP
from app.logic.conversation_router import (
    ConversationRoutingError,
    route_conversation_async,
)
from app.schemas.conversation_route import (
    ConversationAction,
    RouterInput,
)
import logging


logger = logging.getLogger(__name__)


def _build_state_payload(state: SearchRequest | None) -> dict[str, Any] | None:
    if state is None:
        return None
    return state.model_dump(mode="json", exclude_none=True)


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


def _build_orchestrate_intent_payload(state: SearchRequest) -> dict[str, Any]:
    """
    Serialize the canonical constraint-centric search state for orchestrate_search.
    """
    payload = _build_state_payload(state)
    assert payload is not None
    return payload


async def handle_user_message(
    user_message: str,
    previous_state: Optional[SearchRequest] = None,
    *,
    source: str = "fixtures",
    top_n: int = 5,
    fallback_policy: FallbackPolicy | None = None,
    max_items: int = MAX_ITEMS_HARD_CAP,
    shown_listing: dict[str, Any] | None = None,
    latest_result_context: dict[str, Any] | None = None,
) -> Dict[str, Any]:
    previous_state_json = _build_state_payload(
        previous_state
    )
    trace = RequestTrace()

    router_input = RouterInput(
        user_message=user_message,
        current_search=previous_state,
        latest_result_context=latest_result_context,
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

    if (
        effective_action
        == ConversationAction.LISTING_QUESTION
    ):
        response = await _answer_listing_question(
            user_message=user_message,
            shown_listing=shown_listing,
            previous_state=previous_state,
            route_debug=route_debug,
        )

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

    state_json = _build_orchestrate_intent_payload(
        state
    )

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

    result = await orchestrate_search(
        user_text=user_message,
        intent=state_json,
        top_n=top_n,
        fallback_policy=fallback_policy,
        max_items=max_items,
        source=source,
        trace=trace,
    )

    result["state"] = state_json
    result["parsed_intent"] = parsed_intent_debug
    result["search_request"] = state_json

    return _finalize_response(
        result,
        trace=trace,
        action=effective_action,
    )