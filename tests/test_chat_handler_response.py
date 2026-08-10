from ui.services.chat_handler import build_assistant_response
from ui.services.chat_handler import (
    build_assistant_response,
    build_recent_conversation_messages,
)

from app.logic.conversation_response_llm import (
    ConversationResponseGenerationResult,
)


def test_general_chat_uses_conversation_response_layer():
    result = {
        "conversation_action": "general_chat",
        "need_clarification": False,
        "response_type": "other",
        "answer": "OLD DIRECT ANSWER",
        "state": None,
    }

    answer, payload = build_assistant_response(
        user_message="Hello",
        result=result,
    )

    assert "help you search for accommodation" in answer
    assert answer != "OLD DIRECT ANSWER"
    assert payload is None
    
    
def test_routing_failure_keeps_direct_safe_response():
    result = {
        "need_clarification": False,
        "response_type": "routing_unavailable",
        "answer": (
            "I couldn't process that message right now. "
            "Please try again."
        ),
        "state": None,
    }

    answer, payload = build_assistant_response(
        user_message="Make it cheaper",
        result=result,
    )

    assert answer == (
        "I couldn't process that message right now. "
        "Please try again."
    )
    assert payload is None
    
    
    
def test_listing_question_keeps_existing_direct_response():
    result = {
        "conversation_action": "listing_question",
        "need_clarification": False,
        "response_type": "listing_question",
        "answer": "Yes, this property has parking.",
        "state": None,
    }

    answer, payload = build_assistant_response(
        user_message="Does it have parking?",
        result=result,
    )

    assert answer == "Yes, this property has parking."
    assert payload is None
    
    
def test_clarification_uses_conversation_response_layer():
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

    answer, payload = build_assistant_response(
        user_message="Find me somewhere nice",
        result=result,
    )

    assert answer == "Which city would you like to stay in?"
    
    
def test_build_recent_conversation_messages_strips_ui_debug_data():
    messages = [
        {
            "role": "user",
            "content": "Find an apartment in Baku",
        },
        {
            "role": "assistant",
            "content": "Which dates?",
            "debug_data": {
                "telemetry": {"secret": "debug"},
                "parsed_intent": {"something": "internal"},
            },
        },
    ]

    history = build_recent_conversation_messages(messages)

    assert len(history) == 2

    assert history[0].role == "user"
    assert history[0].content == "Find an apartment in Baku"

    assert history[1].role == "assistant"
    assert history[1].content == "Which dates?"

    assert not hasattr(history[1], "debug_data")
    
    
def test_build_recent_conversation_messages_keeps_only_latest_messages():
    messages = [
        {
            "role": "user",
            "content": f"message-{index}",
        }
        for index in range(10)
    ]

    history = build_recent_conversation_messages(
        messages,
        limit=4,
    )

    assert [message.content for message in history] == [
        "message-6",
        "message-7",
        "message-8",
        "message-9",
    ]
    
    
def test_build_recent_conversation_messages_ignores_invalid_messages():
    messages = [
        {
            "role": "system",
            "content": "internal prompt",
        },
        {
            "role": "assistant",
            "content": "",
        },
        {
            "role": "user",
            "content": "Hello",
        },
    ]

    history = build_recent_conversation_messages(messages)

    assert len(history) == 1
    assert history[0].role == "user"
    assert history[0].content == "Hello"
    
    
from contextlib import nullcontext

from ui.services import chat_handler


def test_process_user_message_passes_previous_history_to_response_input(
    monkeypatch,
):
    existing_messages = [
        {
            "role": "user",
            "content": "Find an apartment in Baku",
        },
        {
            "role": "assistant",
            "content": "Which dates would you like?",
            "debug_data": {"internal": "should not leak"},
        },
    ]

    result = {
        "conversation_action": "general_chat",
        "need_clarification": False,
        "response_type": "other",
        "answer": "Old answer",
        "state": None,
    }

    captured = {}

    monkeypatch.setattr(
        chat_handler,
        "get_messages",
        lambda: existing_messages,
    )

    monkeypatch.setattr(
        chat_handler,
        "append_message",
        lambda *args, **kwargs: None,
    )

    monkeypatch.setattr(
        chat_handler,
        "get_search_state",
        lambda: None,
    )

    async def fake_handle_user_message(**kwargs):
        return result

    async def fake_generate_conversation_response_with_llm(
        response_input,
        *,
        trace=None,
    ):
        return ConversationResponseGenerationResult(
            text="Generated answer",
            source="llm",
        )

    def fake_run_async(awaitable):
        import asyncio

        return asyncio.run(awaitable)

    monkeypatch.setattr(
        chat_handler,
        "handle_user_message",
        fake_handle_user_message,
    )

    monkeypatch.setattr(
        chat_handler,
        "run_async",
        fake_run_async,
    )

    monkeypatch.setattr(
        chat_handler.st,
        "spinner",
        lambda *_args, **_kwargs: nullcontext(),
    )

    def fake_build_conversation_response_input(
        *,
        user_message,
        result,
        recent_messages,
    ):
        captured["user_message"] = user_message
        captured["recent_messages"] = recent_messages
        return object()

    monkeypatch.setattr(
        chat_handler,
        "build_conversation_response_input",
        fake_build_conversation_response_input,
    )

    monkeypatch.setattr(
        chat_handler,
        "generate_conversation_response_with_llm",
        fake_generate_conversation_response_with_llm,
    )

    monkeypatch.setattr(
        chat_handler,
        "build_display_answer",
        lambda _: ("ignored", None),
    )

    monkeypatch.setattr(
        chat_handler,
        "save_telemetry_record",
        lambda **kwargs: {
            "log_file": "fake.jsonl",
            "attempt_number": 1,
            "timestamp": "2026-01-01T00:00:00+00:00",
        },
    )

    monkeypatch.setattr(
        chat_handler,
        "set_search_state",
        lambda _: None,
    )

    chat_handler.process_user_message("Thanks")

    assert captured["user_message"] == "Thanks"

    assert [
        message.content
        for message in captured["recent_messages"]
    ] == [
        "Find an apartment in Baku",
        "Which dates would you like?",
    ]

    assert all(
        message.content != "Thanks"
        for message in captured["recent_messages"]
    )