"""
The Streamlit handler is a thin adapter over process_turn: session in,
session out, plus the display transcript. Runs against st.session_state
in bare mode (no Streamlit server).
"""

from __future__ import annotations

from contextlib import nullcontext

import pytest
import streamlit as st

from app.schemas.conversation_response import ConversationMessage
from app.schemas.session import ConversationSession
from app.schemas.turn import DecisionInspector, TurnDebug, TurnResult, TurnTelemetry
from ui import state
from ui.services import chat_handler


@pytest.fixture(autouse=True)
def fresh_session_state(monkeypatch):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    state.init_session_state()
    monkeypatch.setattr(chat_handler.st, "spinner", lambda *a, **k: nullcontext())
    yield
    for key in list(st.session_state.keys()):
        del st.session_state[key]


def _turn(session: ConversationSession, message: str) -> TurnResult:
    return TurnResult(
        session_id=session.session_id,
        assistant_message=message,
        outcome_kind="general_chat",
        inspector=DecisionInspector(response_source="llm"),
        telemetry=TurnTelemetry(
            trace_id="t", latency_ms=1.0, estimated_cost_usd=0.0, llm_calls=1, failed_llm_calls=0
        ),
        session=session,
        debug=TurnDebug(parsed_intent={"router": {"action": "general_chat"}}),
    )


def test_adapter_passes_stored_session_and_stores_the_returned_one(monkeypatch):
    stored = state.get_conversation_session()
    updated = ConversationSession(session_id=stored.session_id, messages=[])
    captured = {}

    async def fake_process_turn(session, user_message, *, config):
        captured["session"] = session
        captured["config"] = config
        return _turn(updated, "Hello!")

    monkeypatch.setattr(chat_handler, "process_turn", fake_process_turn)

    chat_handler.process_user_message("hi")

    assert captured["session"] is stored
    assert captured["config"].include_debug is True
    assert state.get_conversation_session() is updated
    assert [(m["role"], m["content"]) for m in state.get_messages()] == [
        ("user", "hi"),
        ("assistant", "Hello!"),
    ]
    debug = state.get_messages()[-1]["debug_data"]
    assert debug["parsed_intent"] == {"router": {"action": "general_chat"}}
    assert debug["conversation_response_source"] == "llm"


def test_unexpected_error_is_shown_safely_and_session_is_not_committed(monkeypatch):
    stored = state.get_conversation_session()
    before = stored.model_dump()

    async def broken_process_turn(session, user_message, *, config):
        raise KeyError("bug")

    monkeypatch.setattr(chat_handler, "process_turn", broken_process_turn)

    chat_handler.process_user_message("hi")

    assert state.get_conversation_session() is stored
    assert stored.model_dump() == before
    assert state.get_messages()[-1] == {
        "role": "assistant",
        "content": chat_handler.UNEXPECTED_ERROR_MESSAGE,
    }


def test_turn_after_unexpected_error_uses_last_valid_session(monkeypatch):
    """
    The display transcript keeps the failed exchange (so the user sees what
    happened), but the backend history and state come only from the last
    committed ConversationSession - the failed user message never reaches
    the router, the response layer, or the stored search.
    """
    valid = ConversationSession.model_validate(
        {
            "search_request": {"city": "Baku"},
            "messages": [
                {"role": "user", "content": "apartment in Baku"},
                {"role": "assistant", "content": "Which dates?"},
            ],
        }
    )
    state.set_conversation_session(valid)
    calls = []

    async def flaky_process_turn(session, user_message, *, config):
        calls.append((session, user_message))
        if len(calls) == 1:
            raise KeyError("bug")
        return _turn(
            session.model_copy(
                update={
                    "messages": [
                        *session.messages,
                        ConversationMessage(role="user", content=user_message),
                        ConversationMessage(role="assistant", content="OK"),
                    ]
                }
            ),
            "OK",
        )

    monkeypatch.setattr(chat_handler, "process_turn", flaky_process_turn)

    chat_handler.process_user_message("first, crashes")
    chat_handler.process_user_message("April 10-15")

    second_session, _ = calls[1]
    assert second_session is valid
    assert [m.content for m in second_session.messages] == ["apartment in Baku", "Which dates?"]
    assert second_session.search_request.city == "Baku"
    assert [m.content for m in state.get_conversation_session().messages] == [
        "apartment in Baku",
        "Which dates?",
        "April 10-15",
        "OK",
    ]
    assert [m["content"] for m in state.get_messages()] == [
        "first, crashes",
        chat_handler.UNEXPECTED_ERROR_MESSAGE,
        "April 10-15",
        "OK",
    ]


def test_streamlit_app_handles_a_full_search_turn(monkeypatch):
    """
    Real app script + real process_turn (fixtures, matching, ranking,
    deterministic rendering); only LLM boundaries and telemetry writes are
    faked. Covers more than the initial render.
    """
    from streamlit.testing.v1 import AppTest

    from app.agents.intent_router_agent import IntentRoute
    from app.llm.call_policy import LlmCallError
    from app.logic import conversation_flow, intent_router
    from app.schemas.conversation_route import (
        ClarificationReason,
        ConversationAction,
        ConversationActionDecision,
        ConversationDecisionStatus,
        ResultScope,
    )
    from app.services import conversation_service

    async def fake_route(**kwargs):
        return ConversationActionDecision(
            action=ConversationAction.START_SEARCH,
            reason="test",
            decision_status=ConversationDecisionStatus.RESOLVED,
            clarification_reason=ClarificationReason.NONE,
            result_scope=ResultScope.NOT_APPLICABLE,
            target_result_id="",
        )

    async def fake_intent(user_text, **kwargs):
        return IntentRoute(
            city="Baku",
            check_in="2026-04-10",
            check_out="2026-04-15",
            property_types=["apartment"],
        )

    def no_response_model(*args, **kwargs):
        raise LlmCallError("configuration_error", stage="conversation_response_generation")

    monkeypatch.setattr(conversation_flow, "route_conversation_async", fake_route)
    monkeypatch.setattr(intent_router, "route_intent_adk_async", fake_intent)
    monkeypatch.setattr("app.logic.conversation_response_llm.build_adk_model", no_response_model)
    monkeypatch.setattr(conversation_service, "save_telemetry_record", lambda **kwargs: {"log_file": "fake"})

    app = AppTest.from_file("ui/streamlit_app.py", default_timeout=60).run()
    app.chat_input[0].set_value("apartment in Baku for April 10-15").run()

    assert not app.exception
    session = app.session_state["conversation_session"]
    assert session.search_request.city == "Baku"
    assert [item.result_id for item in session.shown_result_set.items]
    assert [m.role for m in session.messages] == ["user", "assistant"]

    transcript = app.session_state["messages"]
    assert [m["role"] for m in transcript] == ["user", "assistant"]
    assert transcript[-1]["content"] == session.messages[-1].content
    assert transcript[-1]["debug_data"]["conversation_response_source"] == "deterministic_fallback"

    rendered = " ".join(block.value for block in app.markdown)
    first_title = session.shown_result_set.items[0].listing.name
    assert first_title in rendered
