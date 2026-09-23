import pytest
from pydantic import TypeAdapter, ValidationError

from app.schemas.conversation_response import (
    ClarificationConversationOutcome,
    ConversationMessage,
    ConversationOutcome,
    ConversationResponseInput,
    GeneralChatConversationOutcome,
    SearchConversationOutcome,
)
from app.schemas.conversation_route import ConversationAction
from app.schemas.query import SearchRequest
from app.schemas.search_response import NormalizedSearchResponse


def test_conversation_message_accepts_user_and_assistant_roles():
    user_message = ConversationMessage(
        role="user",
        content="Hello",
    )
    assistant_message = ConversationMessage(
        role="assistant",
        content="Hi!",
    )

    assert user_message.role == "user"
    assert assistant_message.role == "assistant"


def test_conversation_message_rejects_unknown_role():
    with pytest.raises(ValidationError):
        ConversationMessage(
            role="system",
            content="Internal instruction",
        )


def test_general_chat_response_input_does_not_require_search_state():
    response_input = ConversationResponseInput(
        user_message="Hello",
        action=ConversationAction.GENERAL_CHAT,
        outcome=GeneralChatConversationOutcome(),
    )

    assert response_input.current_search is None
    assert response_input.recent_messages == []
    assert response_input.outcome.kind == "general_chat"


def test_search_outcome_requires_normalized_search_response():
    search_response = NormalizedSearchResponse(
        need_clarification=False,
        results=[],
    )

    outcome = SearchConversationOutcome(
        search_response=search_response,
    )

    assert outcome.kind == "search"
    assert outcome.search_response is search_response


def test_clarification_outcome_contains_domain_questions():
    outcome = ClarificationConversationOutcome(
        questions=[
            "Which city would you like to stay in?"
        ],
    )

    assert outcome.kind == "clarification"
    assert outcome.questions == [
        "Which city would you like to stay in?"
    ]


def test_response_input_can_contain_current_search():
    current_search = SearchRequest(
        city="Baku",
        budget_max=120,
    )

    response_input = ConversationResponseInput(
        user_message="Make it cheaper",
        action=ConversationAction.UPDATE_SEARCH,
        current_search=current_search,
        outcome=SearchConversationOutcome(
            search_response=NormalizedSearchResponse(
                need_clarification=False,
                results=[],
            )
        ),
    )

    assert response_input.current_search == current_search
    assert response_input.action == ConversationAction.UPDATE_SEARCH


def test_conversation_outcome_rejects_unknown_kind():
    adapter = TypeAdapter(ConversationOutcome)

    with pytest.raises(ValidationError):
        adapter.validate_python(
            {
                "kind": "something_else",
            }
        )


def test_search_outcome_without_search_response_is_invalid():
    adapter = TypeAdapter(ConversationOutcome)

    with pytest.raises(ValidationError):
        adapter.validate_python(
            {
                "kind": "search",
            }
        )