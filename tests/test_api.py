"""
FastAPI MVP: the API is a thin adapter over process_turn.

The real process_turn, fixture retrieval, matching and session commit run
here; only the LLM boundaries are scripted (as in the multi-turn tests).
"""

from __future__ import annotations

import asyncio
from datetime import date

import httpx
import pytest
from fastapi.testclient import TestClient

from app.agents.intent_router_agent import IntentRoute
from app.api import main as api_main
from app.api.main import create_app
from app.api.session_store import InMemorySessionStore, SessionNotFoundError
from app.api.usage_limits import UsageLimitConfig
from app.config import clock
from app.llm.call_policy import LlmCallError
from app.logic import conversation_flow, intent_router, intent_update
from app.schemas.conversation_route import (
    ClarificationReason,
    ConversationAction,
    ConversationActionDecision,
    ConversationDecisionStatus,
    ResultScope,
)
from app.schemas.filters import PriceConstraint, SearchFilters
from app.schemas.intent_patch import SearchIntentPatch
from app.schemas.turn import TurnConfig

TODAY = date(2026, 3, 1)


def _decision(action: ConversationAction) -> ConversationActionDecision:
    return ConversationActionDecision(
        action=action,
        reason="t",
        decision_status=ConversationDecisionStatus.RESOLVED,
        clarification_reason=ClarificationReason.NONE,
        result_scope=ResultScope.NOT_APPLICABLE,
        target_result_id="",
    )


@pytest.fixture
def scripted(monkeypatch):
    queues = {"route": [], "intent": [], "patch": []}

    async def route(**kwargs):
        return _decision(queues["route"].pop(0))

    async def intent(user_text, **kwargs):
        return queues["intent"].pop(0)

    async def patch(previous_state, user_message, **kwargs):
        return queues["patch"].pop(0)

    monkeypatch.setattr(conversation_flow, "route_conversation_async", route)
    monkeypatch.setattr(intent_router, "route_intent_adk_async", intent)
    monkeypatch.setattr(intent_update, "route_intent_update_patch_async", patch)
    monkeypatch.setattr("app.logic.conversation_response_llm.build_adk_model", _no_response_model)
    yield queues
    assert all(not queue for queue in queues.values()), queues


def _limits(tmp_path) -> UsageLimitConfig:
    return UsageLimitConfig(db_path=str(tmp_path / "usage.sqlite3"))


@pytest.fixture
def client(tmp_path):
    app = create_app(turn_config=TurnConfig(persist_telemetry=False), usage_config=_limits(tmp_path))
    with clock.use_reference_date(TODAY), TestClient(app) as test_client:
        yield test_client


def _search_in(city: str, queues, *, budget: PriceConstraint | None = None) -> None:
    queues["route"].append(ConversationAction.START_SEARCH)
    queues["intent"].append(
        IntentRoute(
            city=city,
            check_in="2026-04-10",
            check_out="2026-04-15",
            filters=SearchFilters(price=budget) if budget else SearchFilters(),
        )
    )


def _chat(client: TestClient, message: str, session_id: str | None = None) -> httpx.Response:
    body = {"message": message} if session_id is None else {"message": message, "session_id": session_id}
    return client.post("/api/chat", json=body)


def test_health(client):
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_first_message_starts_a_session_and_returns_the_public_turn(client, scripted):
    _search_in("Baku", scripted)

    response = _chat(client, "Apartment in Baku, April 10 to 15")

    assert response.status_code == 200
    body = response.json()
    assert body["session_id"]
    assert body["outcome_kind"] == "search"
    assert body["listing_cards"]
    assert set(body) == {
        "session_id", "assistant_message", "outcome_kind", "conversation_action",
        "listing_cards", "inspector", "failure", "telemetry",
    }
    # Server state (full ListingRaw, history) never reaches the browser.
    assert "shown_result_set" not in response.text
    assert "raw_payload" not in response.text


def test_messages_in_one_session_build_on_each_other(client, scripted):
    _search_in("Baku", scripted, budget=PriceConstraint(max_amount=200, currency="$"))
    scripted["route"].append(ConversationAction.UPDATE_SEARCH)
    scripted["patch"].append(SearchIntentPatch(set_filters=SearchFilters(price=PriceConstraint(scope="per_night"))))

    first = _chat(client, "Baku, April 10 to 15, $200 max").json()
    second = _chat(client, "per night", session_id=first["session_id"]).json()

    assert first["outcome_kind"] == "clarification"
    assert second["session_id"] == first["session_id"]
    assert second["outcome_kind"] == "search"
    price = second["inspector"]["current_search"]["filters"]["price"]
    assert (price["max_amount"], price["scope"]) == (200, "per_night")


def test_sessions_are_independent(client, scripted):
    _search_in("Baku", scripted)
    _search_in("Tbilisi", scripted)

    baku = _chat(client, "Baku please").json()
    tbilisi = _chat(client, "Tbilisi please").json()

    assert baku["session_id"] != tbilisi["session_id"]
    assert baku["inspector"]["current_search"]["city"] == "Baku"
    assert tbilisi["inspector"]["current_search"]["city"] == "Tbilisi"
    stored = client.app.state.sessions
    assert len(stored) == 2


def test_unknown_session_is_404_and_creates_nothing(client):
    response = _chat(client, "hello", session_id="does-not-exist")

    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "session_not_found"
    assert len(client.app.state.sessions) == 0


@pytest.mark.parametrize("message", ["", "   ", "x" * 2001])
def test_invalid_message_is_rejected(client, message):
    assert _chat(client, message).status_code == 422


def test_unexpected_error_is_500_and_keeps_the_stored_session(client, scripted, monkeypatch):
    _search_in("Baku", scripted)
    first = _chat(client, "Baku please").json()
    sessions = client.app.state.sessions
    before = sessions._entries[first["session_id"]].session.model_dump()

    async def broken(*args, **kwargs):
        raise RuntimeError("bug")

    monkeypatch.setattr(api_main, "process_turn", broken)
    response = _chat(client, "cheaper", session_id=first["session_id"])

    assert response.status_code == 500
    assert response.json()["detail"]["code"] == "internal_error"
    assert "bug" not in response.text
    assert sessions._entries[first["session_id"]].session.model_dump() == before


async def test_concurrent_messages_in_one_session_do_not_lose_updates(monkeypatch, tmp_path):
    """Two requests in the same session run in order; the second sees the first."""
    from app.services import conversation_service

    real = conversation_service.process_turn
    seen_history: list[int] = []

    async def slow_turn(session, message, *, config=None):
        seen_history.append(len(session.messages))
        await asyncio.sleep(0.05)
        return await real(session, message, config=config)

    monkeypatch.setattr(api_main, "process_turn", slow_turn)
    monkeypatch.setattr(conversation_flow, "route_conversation_async", lambda **kw: _general_chat())
    monkeypatch.setattr("app.logic.conversation_response_llm.build_adk_model", _no_response_model)

    app = create_app(turn_config=TurnConfig(persist_telemetry=False), usage_config=_limits(tmp_path))
    session_id = app.state.sessions.create()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
        responses = await asyncio.gather(
            *(http.post("/api/chat", json={"message": f"hi {i}", "session_id": session_id}) for i in range(2))
        )

    assert [r.status_code for r in responses] == [200, 200]
    assert seen_history == [0, 2]
    assert len(app.state.sessions._entries[session_id].session.messages) == 4


def _no_response_model(*args, **kwargs):
    raise LlmCallError("configuration_error", stage="conversation_response_generation")


async def _general_chat():
    return _decision(ConversationAction.GENERAL_CHAT)


# --- session store ---------------------------------------------------------------------


async def test_store_expires_idle_sessions():
    now = [0.0]
    store = InMemorySessionStore(ttl_seconds=10, clock=lambda: now[0])
    session_id = store.create()

    now[0] = 11
    with pytest.raises(SessionNotFoundError):
        async with store.checkout(session_id):
            pass


async def test_store_drops_the_least_recently_used_session_when_full():
    store = InMemorySessionStore(max_sessions=2)
    first, second = store.create(), store.create()
    async with store.checkout(first):
        pass  # first is now the most recently used

    store.create()

    assert len(store) == 2
    with pytest.raises(SessionNotFoundError):
        async with store.checkout(second):
            pass
