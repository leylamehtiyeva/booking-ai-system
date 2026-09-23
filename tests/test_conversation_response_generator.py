from datetime import date

from app.logic.conversation_response_generator import (
    generate_deterministic_conversation_response,
)
from app.schemas.conversation_response import (
    ClarificationConversationOutcome,
    ConversationFailureOutcome,
    ConversationResponseInput,
    GeneralChatConversationOutcome,
    SearchConversationOutcome,
)
from app.schemas.conversation_route import ConversationAction
from app.schemas.query import SearchRequest
from app.schemas.search_response import (
    NormalizedRequestSummary,
    NormalizedSearchResponse,
)


def test_generates_general_chat_response():
    response_input = ConversationResponseInput(
        user_message="Hello",
        action=ConversationAction.GENERAL_CHAT,
        outcome=GeneralChatConversationOutcome(),
    )

    answer = generate_deterministic_conversation_response(
        response_input
    )

    assert "help you search for accommodation" in answer


def test_generates_clarification_from_domain_questions():
    response_input = ConversationResponseInput(
        user_message="Find me somewhere nice",
        action=ConversationAction.START_SEARCH,
        outcome=ClarificationConversationOutcome(
            questions=[
                "Which city would you like to stay in?"
            ]
        ),
    )

    answer = generate_deterministic_conversation_response(
        response_input
    )

    assert answer == "Which city would you like to stay in?"


def test_search_response_uses_current_search_as_domain_truth():
    current_search = SearchRequest(
        city="Baku",
        check_in=date(2026, 9, 10),
        check_out=date(2026, 9, 15),
    )

    search_response = NormalizedSearchResponse(
        need_clarification=False,
        request_summary=NormalizedRequestSummary(
            city="Paris",
            check_in="2026-09-10",
            check_out="2026-09-15",
        ),
        results=[],
    )

    response_input = ConversationResponseInput(
        user_message="Find me something in Baku",
        action=ConversationAction.START_SEARCH,
        current_search=current_search,
        outcome=SearchConversationOutcome(
            search_response=search_response
        ),
    )

    answer = generate_deterministic_conversation_response(
        response_input
    )

    assert "Baku" in answer
    assert "Paris" not in answer


def test_generates_safe_failure_message_without_rewriting_it():
    response_input = ConversationResponseInput(
        user_message="Make it cheaper",
        action=ConversationAction.UPDATE_SEARCH,
        outcome=ConversationFailureOutcome(
            user_safe_message=(
                "I couldn't complete that request right now."
            )
        ),
    )

    answer = generate_deterministic_conversation_response(
        response_input
    )

    assert (
        answer
        == "I couldn't complete that request right now."
    )