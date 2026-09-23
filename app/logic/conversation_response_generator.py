from __future__ import annotations

from app.logic.answer_generation import build_user_answer
from app.logic.build_answer_payload import build_answer_payload
from app.schemas.conversation_response import (
    ClarificationConversationOutcome,
    ConversationFailureOutcome,
    ConversationResponseInput,
    GeneralChatConversationOutcome,
    SearchConversationOutcome,
)


def generate_deterministic_conversation_response(
    response_input: ConversationResponseInput,
    *,
    top_k: int = 3,
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
        payload = build_answer_payload(
            outcome.search_response,
            latest_user_query=response_input.user_message,
            top_k=top_k,
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

    if isinstance(outcome, ConversationFailureOutcome):
        return outcome.user_safe_message

    raise TypeError(
        f"Unsupported conversation outcome: "
        f"{type(outcome).__name__}"
    )