from __future__ import annotations

from app.logic.answer_generation import build_user_answer
from app.logic.build_answer_payload import build_answer_payload
from app.logic.listing_query_response import render_listing_query_response
from app.schemas.conversation_response import (
    ClarificationConversationOutcome,
    ConversationFailureOutcome,
    ConversationResponseInput,
    GeneralChatConversationOutcome,
    InformationalConversationOutcome,
    ListingQueryConversationOutcome,
    SearchConversationOutcome,
)

_INFORMATIONAL_MESSAGES = {
    "update_no_op": "Your search hasn't changed, so I didn't run a new search.",
    "listing_query_unsupported": (
        "I can't check that from the information in the shown listings."
    ),
}


def generate_deterministic_conversation_response(
    response_input: ConversationResponseInput,
    *,
    top_k: int | None = None,
) -> str:
    """
    Generate a safe deterministic user-facing response from the
    typed conversation response contract.

    This function:
    - does not call an LLM;
    - does not mutate SearchRequest;
    - does not perform routing;
    - does not perform retrieval or matching;
    - does not read telemetry or debug data.
    """

    outcome = response_input.outcome

    if isinstance(outcome, GeneralChatConversationOutcome):
        return (
            "Hello! I can help you search for accommodation, "
            "update an existing search, or answer questions "
            "about shown options."
        )

    if isinstance(outcome, ClarificationConversationOutcome):
        payload = {
            "need_clarification": True,
            "questions": outcome.questions,
            "debug_notes": [],
        }

        return build_user_answer(payload)

    if isinstance(outcome, SearchConversationOutcome):
        # outcome.search_response.results is already the final shown list
        # (top_n applied upstream in orchestrate_search_request/
        # select_ranked_items) - default to using all of it rather than
        # independently re-truncating; an explicit top_k still overrides.
        effective_top_k = (
            top_k
            if top_k is not None
            else len(outcome.search_response.results)
        )

        payload = build_answer_payload(
            outcome.search_response,
            latest_user_query=response_input.user_message,
            top_k=effective_top_k,
        )

        # SearchRequest is the canonical domain state.
        # Prefer it over request_summary when constructing the
        # user-facing response context.
        if response_input.current_search is not None:
            payload["active_intent"] = (
                response_input.current_search.model_dump(
                    mode="json",
                    exclude_none=True,
                )
            )

        return build_user_answer(payload)

    if isinstance(outcome, ListingQueryConversationOutcome):
        return render_listing_query_response(outcome)

    if isinstance(outcome, InformationalConversationOutcome):
        return _INFORMATIONAL_MESSAGES[outcome.code]

    if isinstance(outcome, ConversationFailureOutcome):
        return outcome.user_safe_message

    raise TypeError(
        f"Unsupported conversation outcome: "
        f"{type(outcome).__name__}"
    )