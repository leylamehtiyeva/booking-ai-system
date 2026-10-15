"""
Date correctness fixes found by the 2026-10-15 e2e baseline.

- 0032: "from April 2nd, 2026 until the end of the year" is a valid
  273-night stay. The 90-night guard exists to catch a check-out that was
  rolled into the next year because it was really before check-in
  ("April 15 to April 10"); without a year rollover it must not fire.
- 0028: "long stay starting April 1st" gives no check-out. The system
  must ask for it instead of silently searching one night, and keep the
  rest of the request so the next message can complete it.
"""

from __future__ import annotations

from datetime import date

import pytest

from app.agents.intent_router_agent import IntentRoute
from app.config import clock
from app.llm.call_policy import LlmCallError
from app.logic import conversation_flow, intent_router, intent_update
from app.logic.date_normalization import (
    InvalidStayDatesError,
    StayDatesIssue,
    resolve_stay_dates,
)
from app.logic.request_resolution import resolve_required_search_context
from app.schemas.conversation_route import (
    ClarificationReason,
    ConversationAction,
    ConversationActionDecision,
    ConversationDecisionStatus,
    ResultScope,
)
from app.schemas.intent_patch import SearchIntentPatch
from app.schemas.query import SearchRequest
from app.schemas.session import ConversationSession
from app.schemas.turn import TurnConfig
from app.services.conversation_service import process_turn

TODAY = date(2026, 3, 1)


def _resolve(check_in, check_out, *, nights=None, text=""):
    return resolve_stay_dates(check_in=check_in, check_out=check_out, nights=nights, user_text=text, today=TODAY)


# --- 0032: long stays without a year rollover --------------------------------------


def test_until_end_of_year_with_the_year_written_once():
    stay = _resolve("2026-04-02", "2026-12-31", text="from April 2nd, 2026 until the end of the year")

    assert (stay.check_in, stay.check_out) == (date(2026, 4, 2), date(2026, 12, 31))


def test_until_end_of_year_without_any_year():
    stay = _resolve("2026-04-02", "2026-12-31", text="from April 2nd until the end of the year")

    assert (stay.check_in, stay.check_out) == (date(2026, 4, 2), date(2026, 12, 31))


def test_long_stay_with_both_years_written():
    stay = _resolve("2026-11-01", "2027-03-01", text="from Nov 1 2026 to March 1 2027")

    assert (stay.check_out - stay.check_in).days == 120


def test_check_out_before_check_in_is_still_caught():
    """The guard's real job: April 15 -> April 10 would roll check-out into next year."""
    with pytest.raises(InvalidStayDatesError) as exc_info:
        _resolve("2026-04-15", "2026-04-10", text="from April 15 to April 10")

    assert exc_info.value.issue == StayDatesIssue.CHECK_OUT_NOT_AFTER_CHECK_IN


def test_cross_year_range_still_resolves():
    stay = _resolve("2026-12-28", "2026-01-03", text="Dec 28 to Jan 3")

    assert (stay.check_in, stay.check_out) == (date(2026, 12, 28), date(2027, 1, 3))


# --- 0028: no automatic one-night stay ---------------------------------------------


def test_check_in_only_leaves_check_out_open():
    stay = _resolve("2026-04-01", None, text="long stay starting April 1st")

    assert (stay.check_in, stay.check_out) == (date(2026, 4, 1), None)


def test_explicit_single_night():
    stay = _resolve("2026-04-01", None, nights=1, text="April 1st for one night")

    assert stay.check_out == date(2026, 4, 2)


def test_explicit_long_stay_in_nights():
    stay = _resolve("2026-04-01", None, nights=60, text="April 1st for 60 nights")

    assert stay.check_out == date(2026, 5, 31)


def test_required_context_asks_for_check_out_when_only_check_in_is_known():
    resolved = resolve_required_search_context(
        SearchRequest(city="Singapore", check_in=date(2026, 4, 1))
    )

    assert resolved.need_clarification
    assert resolved.check_out is None
    assert any("check-out" in question.lower() for question in resolved.questions)


def test_required_context_still_computes_check_out_from_nights():
    resolved = resolve_required_search_context(
        IntentRoute(city="Baku", check_in="2026-04-20", nights=1)
    )

    assert not resolved.need_clarification
    assert resolved.check_out == date(2026, 4, 21)


# --- flow: clarification, then completion through an update ------------------------


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
    """Scripted LLM boundaries over the real flow, fixture retrieval and session commit."""
    queues = {"route": [], "intent": [], "patch": []}

    async def route(**kwargs):
        return _decision(queues["route"].pop(0))

    async def intent(user_text, **kwargs):
        return queues["intent"].pop(0)

    async def patch(previous_state, user_message, **kwargs):
        return queues["patch"].pop(0)

    def no_response_model(*args, **kwargs):
        raise LlmCallError("configuration_error", stage="conversation_response_generation")

    monkeypatch.setattr(conversation_flow, "route_conversation_async", route)
    monkeypatch.setattr(intent_router, "route_intent_adk_async", intent)
    monkeypatch.setattr(intent_update, "route_intent_update_patch_async", patch)
    monkeypatch.setattr("app.logic.conversation_response_llm.build_adk_model", no_response_model)
    yield queues
    assert all(not queue for queue in queues.values()), queues


async def _turn(session: ConversationSession, message: str):
    before = session.model_dump(mode="json")
    with clock.use_reference_date(TODAY):
        turn = await process_turn(session, message, config=TurnConfig(persist_telemetry=False))
    assert session.model_dump(mode="json") == before, "process_turn mutated its input session"
    return turn


async def test_long_stay_without_check_out_asks_and_keeps_the_request(scripted):
    scripted["route"].append(ConversationAction.START_SEARCH)
    scripted["intent"].append(
        IntentRoute(city="Baku", check_in="2026-04-01", property_types=["apartment"])
    )

    turn = await _turn(ConversationSession(), "Long stay in Baku starting April 1st, apartment please")

    assert turn.outcome_kind == "clarification"
    assert "check-out" in turn.assistant_message.lower()
    state = turn.session.search_request
    assert (state.city, state.check_in, state.check_out) == ("Baku", date(2026, 4, 1), None)
    assert [p.value for p in state.property_types] == ["apartment"]
    assert turn.session.shown_result_set is None
    assert turn.listing_cards == []


async def test_clarification_then_check_out_update_runs_the_search(scripted):
    scripted["route"] += [ConversationAction.START_SEARCH, ConversationAction.UPDATE_SEARCH]
    scripted["intent"].append(IntentRoute(city="Baku", check_in="2026-04-01", property_types=["apartment"]))
    scripted["patch"].append(SearchIntentPatch(set_check_out="2026-04-20"))

    clarification = await _turn(ConversationSession(), "Long stay in Baku starting April 1st")
    search = await _turn(clarification.session, "Until April 20th")

    assert search.outcome_kind == "search"
    state = search.session.search_request
    assert (state.check_in, state.check_out) == (date(2026, 4, 1), date(2026, 4, 20))
    assert search.session.shown_result_set is not None
    assert search.listing_cards


async def test_explicit_one_night_searches_without_asking(scripted):
    scripted["route"].append(ConversationAction.START_SEARCH)
    scripted["intent"].append(IntentRoute(city="Baku", check_in="2026-04-10", nights=1))

    turn = await _turn(ConversationSession(), "Baku, April 10th for one night")

    assert turn.outcome_kind == "search"
    assert turn.session.search_request.check_out == date(2026, 4, 11)


async def test_explicit_long_stay_until_end_of_year_is_not_rejected(scripted):
    scripted["route"].append(ConversationAction.START_SEARCH)
    scripted["intent"].append(IntentRoute(city="Tbilisi", check_in="2026-04-02", check_out="2026-12-31"))

    turn = await _turn(ConversationSession(), "Tbilisi from April 2nd, 2026 until the end of the year")

    # No fixture covers 273 nights - a plain search result, not a date clarification.
    assert turn.outcome_kind == "search"
    state = turn.session.search_request
    assert (state.check_in, state.check_out) == (date(2026, 4, 2), date(2026, 12, 31))


def test_update_prompt_handles_a_check_out_only_answer():
    """Reply to the check-out question must update check-out, not become a new check-in."""
    from app.agents.intent_update_agent import build_intent_update_agent
    from app.logic.intent_update import _build_update_prompt

    rule = 'set_check_out only - keep the existing check-in'
    prompt = _build_update_prompt(SearchRequest(city="Baku", check_in=date(2026, 4, 1)), "Until April 20th")

    assert rule in prompt
    assert rule in build_intent_update_agent().instruction
