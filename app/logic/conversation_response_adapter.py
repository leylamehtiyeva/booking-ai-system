from __future__ import annotations

from typing import Any

from app.schemas.conversation_response import (
    ClarificationConversationOutcome,
    ConversationMessage,
    ConversationResponseInput,
    GeneralChatConversationOutcome,
    SearchConversationOutcome,
)
from app.schemas.conversation_route import ConversationAction
from app.schemas.query import SearchRequest
from app.schemas.search_response import NormalizedSearchResponse


def build_conversation_response_input(
    *,
    user_message: str,
    result: dict[str, Any],
    recent_messages: list[ConversationMessage] | None = None,
) -> ConversationResponseInput:
    """
    Convert the application result produced by handle_user_message()
    into the typed contract consumed by the conversational response layer.

    This function does not call an LLM and does not mutate domain state.
    """

    action_raw = result.get("conversation_action")

    if action_raw is None:
        raise ValueError(
            "Cannot build ConversationResponseInput "
            "without conversation_action"
        )

    action = ConversationAction(action_raw)

    current_search = _build_current_search(
        result.get("state")
    )

    if result.get("need_clarification"):
        outcome = ClarificationConversationOutcome(
            questions=result.get("questions", [])
        )

    elif action == ConversationAction.GENERAL_CHAT:
        outcome = GeneralChatConversationOutcome()

    elif action in {
        ConversationAction.START_SEARCH,
        ConversationAction.UPDATE_SEARCH,
    }:
        outcome = SearchConversationOutcome(
            search_response=(
                NormalizedSearchResponse.model_validate(
                    result
                )
            )
        )

    elif action == ConversationAction.LISTING_QUESTION:
        raise ValueError(
            "listing_question is not yet supported by "
            "Conversation Response Generator"
        )

    else:
        raise ValueError(
            f"Unsupported conversation action: {action}"
        )

    return ConversationResponseInput(
        user_message=user_message,
        action=action,
        recent_messages=recent_messages or [],
        current_search=current_search,
        outcome=outcome,
    )


def _build_current_search(
    state: Any,
) -> SearchRequest | None:
    if state is None:
        return None

    if isinstance(state, SearchRequest):
        return state

    return SearchRequest.model_validate(state)