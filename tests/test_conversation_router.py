import asyncio
from collections.abc import AsyncIterator
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from app.logic import conversation_router
from app.logic.conversation_router import (
    ConversationRoutingError,
    _collect_final_response_text,
)
from app.observability.trace import RequestTrace
from app.schemas.conversation_route import (
    ConversationAction,
    RouterInput,
)


def _patch_router_model_layer(
    monkeypatch,
):
    profile = SimpleNamespace(
        model="test-model",
    )
    adk_model = object()

    get_profile_mock = Mock(
        return_value=profile,
    )
    build_model_mock = Mock(
        return_value=adk_model,
    )
    build_agent_mock = Mock(
        return_value=object(),
    )

    monkeypatch.setattr(
        conversation_router,
        "ROUTER_LLM_PROFILE",
        "test-profile",
    )
    monkeypatch.setattr(
        conversation_router,
        "get_llm_profile",
        get_profile_mock,
    )
    monkeypatch.setattr(
        conversation_router,
        "build_adk_model",
        build_model_mock,
    )
    monkeypatch.setattr(
        conversation_router,
        "build_conversation_router_agent",
        build_agent_mock,
    )

    return (
        profile,
        adk_model,
        get_profile_mock,
        build_model_mock,
        build_agent_mock,
    )


class FakeAPIError(Exception):
    def __init__(self, code: int):
        super().__init__(f"API error: {code}")
        self.code = code


class FakeEvent:
    def __init__(
        self,
        *,
        text_parts: list[str | None],
        is_final: bool,
    ) -> None:
        self.content = SimpleNamespace(
            parts=[SimpleNamespace(text=text) for text in text_parts]
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
            '{"action":"update_search",',
            '"reason":"partial"}',
        ],
        is_final=False,
    )

    final_event = FakeEvent(
        text_parts=[
            '{"action":"general_chat","reason":"greeting"}',
        ],
        is_final=True,
    )

    result = await _collect_final_response_text(
        _event_stream(
            partial_event,
            final_event,
        )
    )

    assert result == ('{"action":"general_chat","reason":"greeting"}')


@pytest.mark.asyncio
async def test_collect_final_response_returns_none_without_final_text():
    partial_event = FakeEvent(
        text_parts=[
            '{"action":"update_search","reason":"partial"}',
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
                    '{"action":"general_chat",',
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

    (
    profile,
    adk_model,
    get_profile_mock,
    build_model_mock,
    build_agent_mock,
    ) = _patch_router_model_layer(
        monkeypatch,
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
        "record_llm_call_estimated",
        record_mock,
    )

    trace = RequestTrace()

    decision = await conversation_router.route_conversation_async(
        router_input=RouterInput(
            user_message="hello",
        ),
        trace=trace,
    )

    assert decision.action == ConversationAction.GENERAL_CHAT

    get_profile_mock.assert_called_once_with(
        "test-profile",
    )

    build_model_mock.assert_called_once_with(
        profile,
    )
    build_agent_mock.assert_called_once_with(
        model=adk_model,
    )

    record_mock.assert_called_once()

    recorded = record_mock.call_args.kwargs

    assert recorded["step"] == "conversation_routing"
    assert recorded["model"] == "test-model"
    assert recorded["success"] is True
    assert recorded["error"] is None
    assert recorded["response_text"] == (
        '{"action":"general_chat","reason":"greeting"}'
    )
    assert "You are a conversation classifier" in recorded["prompt_text"]
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
        "InMemorySessionService",
        FakeSessionService,
    )
    monkeypatch.setattr(
        conversation_router,
        "Runner",
        InvalidResponseRunner,
    )
    _patch_router_model_layer(
        monkeypatch,
    )
    monkeypatch.setattr(
        conversation_router,
        "record_llm_call_estimated",
        record_mock,
    )

    trace = RequestTrace()

    with pytest.raises(ConversationRoutingError) as exc_info:
        await conversation_router.route_conversation_async(
            router_input=RouterInput(
                user_message="hello",
            ),
            trace=trace,
        )

    assert exc_info.value.code == "invalid_response"

    record_mock.assert_called_once()

    recorded = record_mock.call_args.kwargs

    assert recorded["success"] is False
    assert recorded["error"] == "invalid_response"
    assert recorded["response_text"] == "not valid json"


class SlowRunner:
    def __init__(
        self,
        *,
        agent,
        app_name,
        session_service,
    ):
        pass

    async def _slow_event_stream(self):
        await asyncio.sleep(1)

        yield FakeEvent(
            text_parts=[
                '{"action":"general_chat","reason":"late"}',
            ],
            is_final=True,
        )

    def run_async(self, **kwargs):
        return self._slow_event_stream()


@pytest.mark.asyncio
async def test_router_timeout_is_recorded_as_failure(
    monkeypatch,
):
    record_mock = Mock()

    monkeypatch.setattr(
        conversation_router,
        "InMemorySessionService",
        FakeSessionService,
    )
    monkeypatch.setattr(
        conversation_router,
        "Runner",
        SlowRunner,
    )
    _patch_router_model_layer(
        monkeypatch,
    )
    monkeypatch.setattr(
        conversation_router,
        "record_llm_call_estimated",
        record_mock,
    )
    monkeypatch.setattr(
        conversation_router,
        "CONVERSATION_ROUTER_TIMEOUT_SECONDS",
        0.01,
    )

    trace = RequestTrace()

    with pytest.raises(ConversationRoutingError) as exc_info:
        await conversation_router.route_conversation_async(
            router_input=RouterInput(
                user_message="hello",
            ),
            trace=trace,
        )

    assert exc_info.value.code == "timeout"

    record_mock.assert_called_once()

    recorded = record_mock.call_args.kwargs

    assert recorded["success"] is False
    assert recorded["error"] == "timeout"
    assert recorded["response_text"] is None


class APIErrorRunner:
    def __init__(
        self,
        *,
        agent,
        app_name,
        session_service,
    ):
        pass

    async def _failing_event_stream(self):
        if False:
            yield None

        raise FakeAPIError(429)

    def run_async(self, **kwargs):
        return self._failing_event_stream()


@pytest.mark.asyncio
async def test_router_classifies_api_error(
    monkeypatch,
):
    record_mock = Mock()

    monkeypatch.setattr(
        conversation_router,
        "CONVERSATION_ROUTER_MAX_ATTEMPTS",
        1,
    )
    monkeypatch.setattr(
        conversation_router,
        "CONVERSATION_ROUTER_RETRY_DELAY_SECONDS",
        0,
    )

    monkeypatch.setattr(
        conversation_router,
        "InMemorySessionService",
        FakeSessionService,
    )
    monkeypatch.setattr(
        conversation_router,
        "Runner",
        APIErrorRunner,
    )
    monkeypatch.setattr(
        conversation_router,
        "APIError",
        FakeAPIError,
    )
    _patch_router_model_layer(
        monkeypatch,
    )
    monkeypatch.setattr(
        conversation_router,
        "record_llm_call_estimated",
        record_mock,
    )

    trace = RequestTrace()

    with pytest.raises(ConversationRoutingError) as exc_info:
        await conversation_router.route_conversation_async(
            router_input=RouterInput(
                user_message="hello",
            ),
            trace=trace,
        )

    assert exc_info.value.code == "rate_limited"

    record_mock.assert_called_once()

    recorded = record_mock.call_args.kwargs

    assert recorded["success"] is False
    assert recorded["error"] == "rate_limited"
    assert recorded["response_text"] is None


class UnexpectedErrorRunner:
    def __init__(
        self,
        *,
        agent,
        app_name,
        session_service,
    ):
        pass

    def run_async(self, **kwargs):
        return object()


@pytest.mark.asyncio
async def test_router_does_not_mask_unexpected_error(
    monkeypatch,
):
    record_mock = Mock()

    async def _broken_collector(events):
        raise AttributeError("broken event processing")

    monkeypatch.setattr(
        conversation_router,
        "InMemorySessionService",
        FakeSessionService,
    )
    monkeypatch.setattr(
        conversation_router,
        "Runner",
        UnexpectedErrorRunner,
    )
    monkeypatch.setattr(
        conversation_router,
        "_collect_final_response_text",
        _broken_collector,
    )
    _patch_router_model_layer(
        monkeypatch,
    )
    monkeypatch.setattr(
        conversation_router,
        "record_llm_call_estimated",
        record_mock,
    )

    trace = RequestTrace()

    with pytest.raises(
        AttributeError,
        match="broken event processing",
    ):
        await conversation_router.route_conversation_async(
            router_input=RouterInput(
                user_message="hello",
            ),
            trace=trace,
        )

    record_mock.assert_called_once()

    recorded = record_mock.call_args.kwargs

    assert recorded["success"] is False
    assert recorded["error"] == "unexpected_error"


class RetryThenSuccessRunner:
    def __init__(
        self,
        *,
        agent,
        app_name,
        session_service,
    ):
        self.run_calls = 0

    async def _api_error_stream(self):
        if False:
            yield None

        raise FakeAPIError(500)

    def run_async(self, **kwargs):
        self.run_calls += 1

        if self.run_calls == 1:
            return self._api_error_stream()

        return _event_stream(
            FakeEvent(
                text_parts=[
                    '{"action":"general_chat",',
                    '"reason":"greeting"}',
                ],
                is_final=True,
            )
        )


@pytest.mark.asyncio
async def test_router_retries_retryable_api_error_once(
    monkeypatch,
):
    record_mock = Mock()

    runner = RetryThenSuccessRunner(
        agent=object(),
        app_name="test",
        session_service=object(),
    )

    monkeypatch.setattr(
        conversation_router,
        "InMemorySessionService",
        FakeSessionService,
    )
    monkeypatch.setattr(
        conversation_router,
        "Runner",
        lambda **kwargs: runner,
    )
    monkeypatch.setattr(
        conversation_router,
        "APIError",
        FakeAPIError,
    )
    _patch_router_model_layer(
        monkeypatch,
    )
    monkeypatch.setattr(
        conversation_router,
        "record_llm_call_estimated",
        record_mock,
    )
    monkeypatch.setattr(
        conversation_router,
        "CONVERSATION_ROUTER_MAX_ATTEMPTS",
        2,
    )
    monkeypatch.setattr(
        conversation_router,
        "CONVERSATION_ROUTER_RETRY_DELAY_SECONDS",
        0,
    )

    trace = RequestTrace()

    decision = await conversation_router.route_conversation_async(
        router_input=RouterInput(
            user_message="hello",
        ),
        trace=trace,
    )

    assert decision.action == ConversationAction.GENERAL_CHAT
    assert runner.run_calls == 2

    assert record_mock.call_count == 2

    first_attempt = record_mock.call_args_list[0].kwargs
    second_attempt = record_mock.call_args_list[1].kwargs

    assert first_attempt["success"] is False
    assert first_attempt["error"] == "provider_unavailable"

    assert second_attempt["success"] is True
    assert second_attempt["error"] is None


class AuthenticationErrorRunner:
    def __init__(
        self,
        *,
        agent,
        app_name,
        session_service,
    ):
        self.run_calls = 0

    async def _error_stream(self):
        if False:
            yield None

        raise FakeAPIError(401)

    def run_async(self, **kwargs):
        self.run_calls += 1
        return self._error_stream()


@pytest.mark.asyncio
async def test_router_does_not_retry_authentication_error(
    monkeypatch,
):
    record_mock = Mock()

    runner = AuthenticationErrorRunner(
        agent=object(),
        app_name="test",
        session_service=object(),
    )

    monkeypatch.setattr(
        conversation_router,
        "InMemorySessionService",
        FakeSessionService,
    )
    monkeypatch.setattr(
        conversation_router,
        "Runner",
        lambda **kwargs: runner,
    )
    monkeypatch.setattr(
        conversation_router,
        "APIError",
        FakeAPIError,
    )
    _patch_router_model_layer(
        monkeypatch,
    )
    monkeypatch.setattr(
        conversation_router,
        "record_llm_call_estimated",
        record_mock,
    )
    monkeypatch.setattr(
        conversation_router,
        "CONVERSATION_ROUTER_MAX_ATTEMPTS",
        2,
    )
    monkeypatch.setattr(
        conversation_router,
        "CONVERSATION_ROUTER_RETRY_DELAY_SECONDS",
        0,
    )

    trace = RequestTrace()

    with pytest.raises(ConversationRoutingError) as exc_info:
        await conversation_router.route_conversation_async(
            router_input=RouterInput(
                user_message="hello",
            ),
            trace=trace,
        )

    assert exc_info.value.code == "authentication_error"
    assert runner.run_calls == 1
    assert record_mock.call_count == 1

    recorded = record_mock.call_args.kwargs

    assert recorded["success"] is False
    assert recorded["error"] == "authentication_error"
