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

import asyncio

from app.logic.conversation_response_generator import (
    generate_deterministic_conversation_response,
)
from app.logic.conversation_response_llm import (
    generate_conversation_response_with_llm,
)
from app.observability.trace import RequestTrace
from app.schemas.conversation_response import (
    ConversationResponseInput,
    GeneralChatConversationOutcome,
)
from app.schemas.conversation_route import ConversationAction


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
    
    
def test_conversation_response_llm_failure_uses_deterministic_fallback(
    monkeypatch,
):
    response_input = ConversationResponseInput(
        user_message="hello",
        action=ConversationAction.GENERAL_CHAT,
        recent_messages=[],
        current_search=None,
        outcome=GeneralChatConversationOutcome(),
    )

    trace = RequestTrace()

    expected_fallback = (
        generate_deterministic_conversation_response(
            response_input
        )
    )

    def fail_build_adk_model(*args, **kwargs):
        raise RuntimeError(
            "simulated responder failure"
        )

    monkeypatch.setattr(
        "app.logic.conversation_response_llm.build_adk_model",
        fail_build_adk_model,
    )

    result = asyncio.run(
        generate_conversation_response_with_llm(
            response_input,
            trace=trace,
        )
    )

    assert result.source == "deterministic_fallback"
    assert result.text == expected_fallback

    telemetry = trace.summary()

    assert (
        telemetry["scenario"][
            "used_conversation_response"
        ]
        is True
    )

    assert (
        telemetry["scenario"][
            "used_conversation_response_fallback"
        ]
        is True
    )

    assert telemetry["llm"]["calls_count"] == 1

    llm_call = telemetry["llm"]["calls"][0]

    assert (
        llm_call["step"]
        == "conversation_response_generation"
    )
    assert llm_call["success"] is False
    assert (
        "simulated responder failure"
        in llm_call["error"]
    )