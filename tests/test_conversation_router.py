from collections.abc import AsyncIterator
from types import SimpleNamespace

import pytest
from unittest.mock import Mock

from app.logic import conversation_router
from app.observability.trace import RequestTrace

from app.logic.conversation_router import (
    _collect_final_response_text,
)

from app.logic.conversation_router import (
    ConversationRoutingError,
    _collect_final_response_text,
)


class FakeEvent:
    def __init__(
        self,
        *,
        text_parts: list[str | None],
        is_final: bool,
    ) -> None:
        self.content = SimpleNamespace(
            parts=[
                SimpleNamespace(text=text)
                for text in text_parts
            ]
        )
        self._is_final = is_final

    def is_final_response(self) -> bool:
        return self._is_final


async def _event_stream(
    *events: FakeEvent,
) -> AsyncIterator[FakeEvent]:
    for event in events:
        yield event


@pytest.mark.asyncio
async def test_collect_final_response_ignores_partial_events():
    partial_event = FakeEvent(
        text_parts=[
            '{"route":"search_update",',
            '"reason":"partial"}',
        ],
        is_final=False,
    )

    final_event = FakeEvent(
        text_parts=[
            '{"route":"other",',
            '"reason":"greeting"}',
        ],
        is_final=True,
    )

    result = await _collect_final_response_text(
        _event_stream(
            partial_event,
            final_event,
        )
    )

    assert result == (
        '{"route":"other",'
        '"reason":"greeting"}'
    )
    
    
@pytest.mark.asyncio
async def test_collect_final_response_returns_none_without_final_text():
    partial_event = FakeEvent(
        text_parts=[
            '{"route":"search_update"}',
        ],
        is_final=False,
    )

    empty_final_event = FakeEvent(
        text_parts=[],
        is_final=True,
    )

    result = await _collect_final_response_text(
        _event_stream(
            partial_event,
            empty_final_event,
        )
    )

    assert result is None
    
    
class FakeSessionService:
    async def create_session(self, **kwargs):
        return None


class FakeRunner:
    def __init__(
        self,
        *,
        agent,
        app_name,
        session_service,
    ):
        pass

    def run_async(self, **kwargs):
        return _event_stream(
            FakeEvent(
                text_parts=[
                    '{"route":"other",',
                    '"reason":"greeting"}',
                ],
                is_final=True,
            )
        )


@pytest.mark.asyncio
async def test_router_records_successful_llm_call_once(
    monkeypatch,
):
    record_mock = Mock()

    monkeypatch.setattr(
        conversation_router,
        "_ensure_gemini_key",
        lambda: None,
    )
    monkeypatch.setattr(
        conversation_router,
        "build_conversation_router_agent",
        lambda: object(),
    )
    monkeypatch.setattr(
        conversation_router,
        "InMemorySessionService",
        FakeSessionService,
    )
    monkeypatch.setattr(
        conversation_router,
        "Runner",
        FakeRunner,
    )
    monkeypatch.setattr(
        conversation_router,
        "get_gemini_model",
        lambda: "test-model",
    )
    monkeypatch.setattr(
        conversation_router,
        "record_llm_call_estimated",
        record_mock,
    )

    trace = RequestTrace()

    decision = await conversation_router.route_conversation_async(
        user_message="hello",
        previous_state=None,
        trace=trace,
    )

    assert decision.route == "other"

    record_mock.assert_called_once()

    recorded = record_mock.call_args.kwargs

    assert recorded["step"] == "conversation_routing"
    assert recorded["model"] == "test-model"
    assert recorded["success"] is True
    assert recorded["error"] is None
    assert recorded["response_text"] == (
        '{"route":"other",'
        '"reason":"greeting"}'
    )
    assert "You are a conversation router" in recorded["prompt_text"]
    assert "Latest user message:" in recorded["prompt_text"]
    
    
    
class InvalidResponseRunner:
    def __init__(
        self,
        *,
        agent,
        app_name,
        session_service,
    ):
        pass

    def run_async(self, **kwargs):
        return _event_stream(
            FakeEvent(
                text_parts=["not valid json"],
                is_final=True,
            )
        )
        
        

@pytest.mark.asyncio
async def test_router_records_invalid_response_failure(
    monkeypatch,
):
    record_mock = Mock()

    monkeypatch.setattr(
        conversation_router,
        "_ensure_gemini_key",
        lambda: None,
    )
    monkeypatch.setattr(
        conversation_router,
        "build_conversation_router_agent",
        lambda: object(),
    )
    monkeypatch.setattr(
        conversation_router,
        "InMemorySessionService",
        FakeSessionService,
    )
    monkeypatch.setattr(
        conversation_router,
        "Runner",
        InvalidResponseRunner,
    )
    monkeypatch.setattr(
        conversation_router,
        "get_gemini_model",
        lambda: "test-model",
    )
    monkeypatch.setattr(
        conversation_router,
        "record_llm_call_estimated",
        record_mock,
    )

    trace = RequestTrace()

    with pytest.raises(ConversationRoutingError) as exc_info:
        await conversation_router.route_conversation_async(
            user_message="hello",
            previous_state=None,
            trace=trace,
        )

    assert exc_info.value.code == "invalid_response"

    record_mock.assert_called_once()

    recorded = record_mock.call_args.kwargs

    assert recorded["success"] is False
    assert recorded["error"] == "invalid_response"
    assert recorded["response_text"] == "not valid json"