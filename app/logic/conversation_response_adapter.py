from __future__ import annotations

from typing import Any

from app.schemas.conversation_response import (
    ClarificationConversationOutcome,
    ConversationFailureOutcome,
    ConversationMessage,
    ConversationResponseInput,
    GeneralChatConversationOutcome,
    InformationalConversationOutcome,
    ListingQueryConversationOutcome,
    ListingQueryResponseItem,
    SearchConversationOutcome,
)
from app.schemas.conversation_route import ConversationAction
from app.schemas.query import SearchRequest
from app.schemas.result_query_match import ResultQueryMatch
from app.schemas.search_response import NormalizedSearchResponse

_INFORMATIONAL_RESPONSE_TYPES = {"update_no_op", "listing_query_unsupported"}


def build_conversation_response_input(
    *,
    user_message: str,
    result: dict[str, Any],
    recent_messages: list[ConversationMessage] | None = None,
) -> ConversationResponseInput | None:
    """
    Convert the application result produced by handle_user_message()
    into the typed contract consumed by the conversational response layer.

    This function does not call an LLM and does not mutate domain state.

    Returns None only for the one known pre-domain technical failure
    shape (routing itself failed, so conversation_action was never set -
    there is no ConversationAction to type this as, and inventing one
    would assert a decision that never happened). Any other result
    missing conversation_action is a genuine bug and still raises.
    """

    action_raw = result.get("conversation_action")

    if action_raw is None:
        if result.get("response_type") == "routing_unavailable":
            return None
        raise ValueError(
            "Cannot build ConversationResponseInput "
            "without conversation_action"
        )

    action = ConversationAction(action_raw)

    current_search = _build_current_search(
        result.get("state")
    )

    response_type = result.get("response_type")

    if result.get("need_clarification"):
        outcome = ClarificationConversationOutcome(
            questions=result.get("questions", [])
        )

    elif response_type in _INFORMATIONAL_RESPONSE_TYPES:
        # UPDATE_SEARCH no-op and unsupported listing questions are both
        # resolved, non-clarification, non-failure turns whose text is
        # fully determined by this code - not a pre-built message, see
        # InformationalConversationOutcome's own docstring for why a
        # single outcome kind covers both.
        outcome = InformationalConversationOutcome(code=response_type)

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
        if response_type != "listing_query_result":
            raise ValueError(
                "Unsupported LISTING_QUESTION response_type: "
                f"{response_type!r}"
            )

        listing_query_result = result.get("listing_query_result")
        if listing_query_result is None:
            raise ValueError(
                "listing_query_result response is missing its "
                "'listing_query_result' payload"
            )

        outcome = _build_listing_query_outcome(listing_query_result)

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


def _build_listing_query_outcome(
    listing_query_result: dict[str, Any],
) -> ListingQueryConversationOutcome:
    """
    Deterministic join of ResultQueryMatch[] (the factual source of
    truth) with the presentation projection (result_id/position/title),
    both carried as parallel collections in the raw application dict.
    This is the one place that join happens - the typed
    ListingQueryConversationOutcome never carries two collections that
    could drift apart.

    Fails loudly (ValueError) rather than masking a broken contract: a
    match without presentation, or any duplicate result_id on either
    side, is a bug upstream, never guessed at or defaulted to the
    first/nearest item.
    """
    matches = [
        ResultQueryMatch.model_validate(raw)
        for raw in listing_query_result.get("matches") or []
    ]
    presentation_entries = listing_query_result.get("presentation") or []

    presentation_by_id: dict[str, dict[str, Any]] = {}
    for entry in presentation_entries:
        result_id = entry.get("result_id")
        if result_id in presentation_by_id:
            raise ValueError(
                f"Duplicate presentation entry for result_id={result_id!r}"
            )
        presentation_by_id[result_id] = entry

    seen_result_ids: set[str] = set()
    items: list[ListingQueryResponseItem] = []

    for match in matches:
        if match.result_id in seen_result_ids:
            raise ValueError(
                f"Duplicate ResultQueryMatch for result_id={match.result_id!r}"
            )
        seen_result_ids.add(match.result_id)

        presentation = presentation_by_id.get(match.result_id)
        if presentation is None:
            raise ValueError(
                "Missing presentation metadata for "
                f"result_id={match.result_id!r} - cannot render a "
                "listing-query response without it."
            )

        items.append(
            ListingQueryResponseItem(
                position=presentation["position"],
                title=presentation.get("title"),
                match=match,
            )
        )

    items.sort(key=lambda item: item.position)

    return ListingQueryConversationOutcome(items=items)


def _build_current_search(
    state: Any,
) -> SearchRequest | None:
    if state is None:
        return None

    if isinstance(state, SearchRequest):
        return state

    return SearchRequest.model_validate(state)
