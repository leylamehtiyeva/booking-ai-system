import pytest

from app.logic.conversation_response_adapter import (
    build_conversation_response_input,
)
from app.schemas.conversation_response import (
    ClarificationConversationOutcome,
    ConversationMessage,
    GeneralChatConversationOutcome,
    InformationalConversationOutcome,
    ListingQueryConversationOutcome,
    SearchConversationOutcome,
)
from app.schemas.conversation_route import ConversationAction
from app.schemas.match import Ternary


def test_builds_general_chat_response_input():
    result = {
        "conversation_action": "general_chat",
        "need_clarification": False,
        "response_type": "other",
        "answer": "Hello!",
        "state": None,
    }

    response_input = build_conversation_response_input(
        user_message="Hello",
        result=result,
    )

    assert (
        response_input.action
        == ConversationAction.GENERAL_CHAT
    )
    assert response_input.current_search is None
    assert isinstance(
        response_input.outcome,
        GeneralChatConversationOutcome,
    )
    
    
def test_builds_clarification_outcome_before_search_outcome():
    result = {
        "conversation_action": "start_search",
        "need_clarification": True,
        "questions": [
            "Which city would you like to stay in?"
        ],
        "state": {
            "adults": 2,
            "children": 0,
            "rooms": 1,
            "currency": "USD",
            "constraints": [],
        },
    }

    response_input = build_conversation_response_input(
        user_message="Find me somewhere nice",
        result=result,
    )

    assert (
        response_input.action
        == ConversationAction.START_SEARCH
    )
    assert isinstance(
        response_input.outcome,
        ClarificationConversationOutcome,
    )
    assert response_input.outcome.questions == [
        "Which city would you like to stay in?"
    ]
    
    
def test_builds_search_outcome():
    result = {
        "conversation_action": "start_search",
        "need_clarification": False,
        "questions": [],
        "status": "results",
        "request_summary": {
            "city": "Baku",
        },
        "results": [],
        "debug_notes": [],
        "state": {
            "city": "Baku",
            "adults": 2,
            "children": 0,
            "rooms": 1,
            "currency": "USD",
            "constraints": [],
        },
    }

    response_input = build_conversation_response_input(
        user_message="Find an apartment in Baku",
        result=result,
    )

    assert (
        response_input.action
        == ConversationAction.START_SEARCH
    )

    assert isinstance(
        response_input.outcome,
        SearchConversationOutcome,
    )

    assert response_input.current_search is not None
    assert response_input.current_search.city == "Baku"

    assert (
        response_input.outcome
        .search_response
        .request_summary
        .city
        == "Baku"
    )
    
    
def test_preserves_recent_conversation_messages():
    recent_messages = [
        ConversationMessage(
            role="user",
            content="Hello",
        ),
        ConversationMessage(
            role="assistant",
            content="Hi! How can I help?",
        ),
    ]

    result = {
        "conversation_action": "general_chat",
        "need_clarification": False,
        "response_type": "other",
        "answer": "You're welcome!",
        "state": None,
    }

    response_input = build_conversation_response_input(
        user_message="Thanks",
        result=result,
        recent_messages=recent_messages,
    )

    assert response_input.recent_messages == recent_messages
    
    
def test_missing_conversation_action_still_raises_for_unknown_shapes():
    result = {
        "need_clarification": False,
        "response_type": "some_future_response_type",
        "answer": "Please try again.",
        "state": None,
    }

    with pytest.raises(
        ValueError,
        match="without conversation_action",
    ):
        build_conversation_response_input(
            user_message="Make it cheaper",
            result=result,
        )


def test_routing_unavailable_without_conversation_action_returns_none():
    result = {
        "need_clarification": False,
        "response_type": "routing_unavailable",
        "answer": "Please try again.",
        "state": None,
    }

    response_input = build_conversation_response_input(
        user_message="Make it cheaper",
        result=result,
    )

    assert response_input is None


def _constraint_match(result_id: str, value: str, field: str = "parking") -> dict:
    return {
        "result_id": result_id,
        "constraint_results": [
            {
                "constraint_id": f"c-{result_id}",
                "raw_text": "parking",
                "field": field,
                "value": value,
                "evidence": [],
                "reason": "",
            }
        ],
    }


def _presentation(result_id: str, position: int, title: str | None = None) -> dict:
    return {"result_id": result_id, "position": position, "title": title}


def _listing_question_result(*, matches: list[dict], presentation: list[dict]) -> dict:
    return {
        "conversation_action": "listing_question",
        "need_clarification": False,
        "response_type": "listing_query_result",
        "answer": "I checked the shown listings for that.",
        "listing_query_result": {
            "query": {"constraints": []},
            "matches": matches,
            "presentation": presentation,
        },
        "state": None,
    }


def test_listing_query_result_builds_listing_query_outcome():
    result = _listing_question_result(
        matches=[_constraint_match("H1", "YES")],
        presentation=[_presentation("H1", 1, "Hilton Baku")],
    )

    response_input = build_conversation_response_input(
        user_message="Does it have parking?",
        result=result,
    )

    assert isinstance(response_input.outcome, ListingQueryConversationOutcome)
    assert len(response_input.outcome.items) == 1

    item = response_input.outcome.items[0]
    assert item.position == 1
    assert item.title == "Hilton Baku"
    assert item.match.result_id == "H1"
    assert item.match.constraint_results[0].value == Ternary.YES


def test_listing_query_result_orders_items_by_position():
    result = _listing_question_result(
        matches=[
            _constraint_match("H2", "NO"),
            _constraint_match("H1", "YES"),
        ],
        presentation=[
            _presentation("H1", 1, "Hotel A"),
            _presentation("H2", 2, "Hotel B"),
        ],
    )

    response_input = build_conversation_response_input(
        user_message="Which of these have parking?",
        result=result,
    )

    assert [item.match.result_id for item in response_input.outcome.items] == ["H1", "H2"]


def test_listing_query_result_missing_presentation_fails_loudly():
    result = _listing_question_result(
        matches=[_constraint_match("H1", "YES")],
        presentation=[],
    )

    with pytest.raises(ValueError, match="Missing presentation metadata"):
        build_conversation_response_input(
            user_message="Does it have parking?",
            result=result,
        )


def test_listing_query_result_duplicate_match_result_id_fails_loudly():
    result = _listing_question_result(
        matches=[
            _constraint_match("H1", "YES"),
            _constraint_match("H1", "NO"),
        ],
        presentation=[_presentation("H1", 1, "Hotel A")],
    )

    with pytest.raises(ValueError, match="Duplicate ResultQueryMatch"):
        build_conversation_response_input(
            user_message="Does it have parking?",
            result=result,
        )


def test_listing_query_result_duplicate_presentation_entry_fails_loudly():
    result = _listing_question_result(
        matches=[_constraint_match("H1", "YES")],
        presentation=[
            _presentation("H1", 1, "Hotel A"),
            _presentation("H1", 1, "Hotel A (dup)"),
        ],
    )

    with pytest.raises(ValueError, match="Duplicate presentation entry"):
        build_conversation_response_input(
            user_message="Does it have parking?",
            result=result,
        )


def test_listing_query_unresolved_response_type_still_raises():
    result = {
        "conversation_action": "listing_question",
        "need_clarification": False,
        "response_type": "listing_question",
        "answer": "Yes, it has parking.",
        "state": None,
    }

    with pytest.raises(ValueError, match="Unsupported LISTING_QUESTION response_type"):
        build_conversation_response_input(
            user_message="Does it have parking?",
            result=result,
        )


def test_update_no_op_builds_informational_outcome_not_failure():
    result = {
        "conversation_action": "update_search",
        "need_clarification": False,
        "response_type": "update_no_op",
        "answer": "Your search hasn't changed, so I didn't run a new search.",
        "state": None,
    }

    response_input = build_conversation_response_input(
        user_message="No changes",
        result=result,
    )

    assert isinstance(response_input.outcome, InformationalConversationOutcome)
    assert response_input.outcome.code == "update_no_op"


def test_listing_query_unsupported_builds_informational_outcome():
    result = {
        "conversation_action": "listing_question",
        "need_clarification": False,
        "response_type": "listing_query_unsupported",
        "answer": "I can't check that from the information in the shown listings.",
        "state": None,
    }

    response_input = build_conversation_response_input(
        user_message="Is it romantic?",
        result=result,
    )

    assert isinstance(response_input.outcome, InformationalConversationOutcome)
    assert response_input.outcome.code == "listing_query_unsupported"