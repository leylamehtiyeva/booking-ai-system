import pytest

from app.logic.conversation_response_adapter import (
    build_conversation_response_input,
)
from app.schemas.conversation_response import (
    ClarificationConversationOutcome,
    ConversationMessage,
    GeneralChatConversationOutcome,
    SearchConversationOutcome,
)
from app.schemas.conversation_route import ConversationAction


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
    
    
def test_rejects_result_without_conversation_action():
    result = {
        "need_clarification": False,
        "response_type": "routing_unavailable",
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
        
        
def test_listing_question_is_not_supported_yet():
    result = {
        "conversation_action": "listing_question",
        "need_clarification": False,
        "response_type": "listing_question",
        "answer": "Yes, it has parking.",
        "state": None,
    }

    with pytest.raises(
        ValueError,
        match="listing_question is not yet supported",
    ):
        build_conversation_response_input(
            user_message="Does it have parking?",
            result=result,
        )