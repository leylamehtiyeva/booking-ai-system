from app.logic.conversation_response_llm import (
    _build_llm_payload,
)
from app.schemas.conversation_response import (
    ConversationResponseInput,
    GeneralChatConversationOutcome,
)
from app.schemas.conversation_route import (
    ConversationAction,
)


def test_general_chat_llm_payload_contains_only_conversation_context():
    response_input = ConversationResponseInput(
        user_message="Thanks",
        action=ConversationAction.GENERAL_CHAT,
        outcome=GeneralChatConversationOutcome(),
    )

    payload = _build_llm_payload(
        response_input
    )

    assert payload["current_user_message"] == "Thanks"
    assert payload["action"] == "general_chat"

    assert payload["outcome"] == {
        "kind": "general_chat"
    }
    
    
from app.schemas.conversation_response import (
    ConversationMessage,
)


def test_llm_payload_contains_recent_visible_history():
    response_input = ConversationResponseInput(
        user_message="Thanks",
        action=ConversationAction.GENERAL_CHAT,
        recent_messages=[
            ConversationMessage(
                role="user",
                content="Find an apartment in Baku",
            ),
            ConversationMessage(
                role="assistant",
                content="Which dates?",
            ),
        ],
        outcome=GeneralChatConversationOutcome(),
    )

    payload = _build_llm_payload(
        response_input
    )

    assert payload["recent_messages"] == [
        {
            "role": "user",
            "content": "Find an apartment in Baku",
        },
        {
            "role": "assistant",
            "content": "Which dates?",
        },
    ]