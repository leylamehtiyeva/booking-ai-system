import pytest
from pydantic import TypeAdapter, ValidationError

from app.schemas.conversation_response import (
    ClarificationConversationOutcome,
    ConversationMessage,
    ConversationOutcome,
    ConversationResponseInput,
    GeneralChatConversationOutcome,
    InformationalConversationOutcome,
    ListingQueryConversationOutcome,
    ListingQueryResponseItem,
    SearchConversationOutcome,
)
from app.schemas.conversation_route import ConversationAction
from app.schemas.match import Ternary
from app.schemas.query import SearchRequest
from app.schemas.result_query_match import ConstraintMatchResult, ResultQueryMatch
from app.schemas.search_response import NormalizedSearchResponse, SearchStatus


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
        status=SearchStatus.NO_RESULTS,
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
                status=SearchStatus.NO_RESULTS,
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


def test_listing_query_outcome_carries_result_query_match_unchanged():
    match = ResultQueryMatch(
        result_id="H1",
        constraint_results=[
            ConstraintMatchResult(
                constraint_id="c1",
                raw_text="parking",
                field="parking",
                value=Ternary.YES,
                evidence=[],
            )
        ],
    )

    outcome = ListingQueryConversationOutcome(
        items=[ListingQueryResponseItem(position=1, title="Hilton Baku", match=match)]
    )

    assert outcome.kind == "listing_query"
    assert outcome.items[0].match is match


def test_informational_outcome_requires_a_known_code():
    with pytest.raises(ValidationError):
        InformationalConversationOutcome(code="not_a_real_code")

    outcome = InformationalConversationOutcome(code="update_no_op")
    assert outcome.kind == "informational"


def test_conversation_outcome_union_round_trips_listing_query_by_discriminator():
    adapter = TypeAdapter(ConversationOutcome)

    payload = {
        "kind": "listing_query",
        "items": [
            {
                "position": 1,
                "title": "Hilton Baku",
                "match": {
                    "result_id": "H1",
                    "constraint_results": [
                        {
                            "constraint_id": "c1",
                            "raw_text": "parking",
                            "field": "parking",
                            "value": "YES",
                            "evidence": [],
                        }
                    ],
                },
            }
        ],
    }

    outcome = adapter.validate_python(payload)

    assert isinstance(outcome, ListingQueryConversationOutcome)
    assert outcome.items[0].match.result_id == "H1"


def test_search_outcome_without_search_response_is_invalid():
    adapter = TypeAdapter(ConversationOutcome)

    with pytest.raises(ValidationError):
        adapter.validate_python(
            {
                "kind": "search",
            }
        )