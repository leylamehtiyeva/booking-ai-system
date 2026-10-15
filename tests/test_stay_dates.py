"""
Regression tests for stay-date interpretation (P0-2).

Reference date for every case is 2026-10-15 unless stated otherwise, the
day the bugs below were confirmed:
- "April 10-15" was forced to the current year -> a stay in the past;
- "Dec 28 - Jan 3" normalized each date independently -> check_out before
  check_in -> uncaught ValidationError out of handle_user_message;
- an UPDATE_SEARCH patch could produce check_out <= check_in, which then
  crashed the response layer on SearchRequest.model_validate.
"""

from __future__ import annotations

from datetime import date

import pytest

from app.agents.intent_router_agent import IntentRoute, build_intent_router_agent
from app.config import clock
from app.logic import conversation_flow, intent_router, intent_update
from app.logic.conversation_response_adapter import build_conversation_response_input
from app.logic.date_normalization import (
    InvalidStayDatesError,
    StayDatesIssue,
    resolve_stay_dates,
)
from app.schemas.conversation_route import (
    ClarificationReason,
    ConversationAction,
    ConversationActionDecision,
    ConversationDecisionStatus,
    ResultScope,
)
from app.schemas.fallback_policy import FallbackPolicy
from app.schemas.intent_patch import SearchIntentPatch
from app.schemas.query import SearchRequest
from app.schemas.search_response import NormalizedSearchResponse, SearchStatus
from app.schemas.shown_result_set import ShownResultSet

TODAY = date(2026, 10, 15)


def _resolve(check_in=None, check_out=None, nights=None, text="", **kwargs):
    return resolve_stay_dates(
        check_in=check_in,
        check_out=check_out,
        nights=nights,
        user_text=text,
        today=TODAY,
        **kwargs,
    )


# --- year inference -------------------------------------------------------


def test_date_without_year_already_passed_this_year_moves_to_next_year():
    stay = _resolve("2026-04-10", "2026-04-15", text="Baku April 10-15")

    assert (stay.check_in, stay.check_out) == (date(2027, 4, 10), date(2027, 4, 15))


def test_date_without_year_later_this_year_stays_in_this_year():
    stay = _resolve("2026-11-03", "2026-11-07", text="Baku Nov 3 to Nov 7")

    assert (stay.check_in, stay.check_out) == (date(2026, 11, 3), date(2026, 11, 7))


def test_check_in_today_is_allowed():
    stay = _resolve("2026-10-15", "2026-10-17", text="from Oct 15 to Oct 17")

    assert stay.check_in == TODAY


def test_cross_year_range_without_year():
    stay = _resolve("2026-12-28", "2026-01-03", text="Dec 28 to Jan 3")

    assert (stay.check_in, stay.check_out) == (date(2026, 12, 28), date(2027, 1, 3))


def test_llm_year_is_ignored_when_user_gave_no_year():
    stay = _resolve("2024-12-28", "2024-01-03", text="Dec 28 to Jan 3")

    assert (stay.check_in, stay.check_out) == (date(2026, 12, 28), date(2027, 1, 3))


def test_relative_dates_resolved_by_llm_against_today_are_kept():
    stay = _resolve("2026-10-17", "2026-10-19", text="next weekend")

    assert (stay.check_in, stay.check_out) == (date(2026, 10, 17), date(2026, 10, 19))


# --- explicit years -------------------------------------------------------


def test_explicit_future_year_is_kept():
    stay = _resolve("2027-04-10", "2027-04-15", text="April 10-15 2027")

    assert (stay.check_in, stay.check_out) == (date(2027, 4, 10), date(2027, 4, 15))


def test_explicit_past_dates_are_not_silently_moved_to_the_future():
    with pytest.raises(InvalidStayDatesError) as exc_info:
        _resolve("2025-04-10", "2025-04-15", text="April 10-15 2025")

    assert exc_info.value.issue == StayDatesIssue.IN_THE_PAST
    assert "past" in exc_info.value.question


def test_explicit_reversed_range_is_rejected():
    with pytest.raises(InvalidStayDatesError) as exc_info:
        _resolve("2027-04-15", "2027-04-10", text="from 15 to 10 April 2027")

    assert exc_info.value.issue == StayDatesIssue.CHECK_OUT_NOT_AFTER_CHECK_IN


# --- invalid ranges and dates ---------------------------------------------


def test_check_out_before_check_in_without_year_is_rejected_not_rolled_years_ahead():
    with pytest.raises(InvalidStayDatesError) as exc_info:
        _resolve("2026-04-15", "2026-04-10", text="from April 15 to April 10")

    assert exc_info.value.issue == StayDatesIssue.CHECK_OUT_NOT_AFTER_CHECK_IN


def test_same_day_check_in_and_check_out_is_rejected():
    with pytest.raises(InvalidStayDatesError) as exc_info:
        _resolve("2026-11-03", "2026-11-03", text="Nov 3 to Nov 3")

    assert exc_info.value.issue == StayDatesIssue.CHECK_OUT_NOT_AFTER_CHECK_IN


@pytest.mark.parametrize("raw", ["2027-02-30", "next friday", "2027-13-01"])
def test_unparseable_or_impossible_dates_are_rejected(raw):
    with pytest.raises(InvalidStayDatesError) as exc_info:
        _resolve(raw, None, text="Baku 2027")

    assert exc_info.value.issue == StayDatesIssue.INVALID_DATE


def test_non_positive_nights_is_rejected():
    with pytest.raises(InvalidStayDatesError) as exc_info:
        _resolve("2026-11-03", None, nights=0, text="Nov 3 for 0 nights")

    assert exc_info.value.issue == StayDatesIssue.NON_POSITIVE_NIGHTS


# --- leap years -----------------------------------------------------------


def test_feb_29_without_year_resolves_to_next_leap_year():
    stay = _resolve("2026-02-29", "2026-03-02", text="Feb 29 to March 2")

    assert (stay.check_in, stay.check_out) == (date(2028, 2, 29), date(2028, 3, 2))


def test_feb_29_with_explicit_non_leap_year_is_invalid():
    with pytest.raises(InvalidStayDatesError) as exc_info:
        _resolve("2027-02-29", "2027-03-02", text="Feb 29 2027 to March 2")

    assert exc_info.value.issue == StayDatesIssue.INVALID_DATE


# --- nights / single date / partial update --------------------------------


def test_check_in_with_nights_computes_check_out():
    stay = _resolve("2026-04-20", None, nights=6, text="April 20 for 6 nights")

    assert (stay.check_in, stay.check_out) == (date(2027, 4, 20), date(2027, 4, 26))


def test_single_date_defaults_to_one_night_when_requested():
    stay = _resolve("2026-11-03", None, text="Nov 3", default_one_night=True)

    assert stay.check_out == date(2026, 11, 4)


def test_partial_update_check_out_is_anchored_to_existing_check_in():
    stay = _resolve(None, "2026-01-03", text="until Jan 3", anchor_check_in=date(2026, 12, 28))

    assert (stay.check_in, stay.check_out) == (None, date(2027, 1, 3))


# --- reference clock ------------------------------------------------------


def test_clock_context_override_wins_over_environment(monkeypatch):
    monkeypatch.setenv(clock.REFERENCE_DATE_ENV_VAR, "2026-01-01")

    with clock.use_reference_date(TODAY):
        assert clock.today() == TODAY

    assert clock.today() == date(2026, 1, 1)


def test_clock_uses_real_today_without_override(monkeypatch):
    monkeypatch.delenv(clock.REFERENCE_DATE_ENV_VAR, raising=False)

    assert clock.today() == date.today()


def test_clock_rejects_malformed_environment_value(monkeypatch):
    monkeypatch.setenv(clock.REFERENCE_DATE_ENV_VAR, "15.10.2026")

    with pytest.raises(ValueError):
        clock.today()


def test_resolve_stay_dates_defaults_to_reference_clock():
    with clock.use_reference_date(TODAY):
        stay = resolve_stay_dates(
            check_in="2026-04-10",
            check_out="2026-04-15",
            nights=None,
            user_text="April 10-15",
        )

    assert stay.check_in == date(2027, 4, 10)


# --- prompts carry the reference date --------------------------------------


def test_intent_agent_instruction_contains_reference_date():
    with clock.use_reference_date(TODAY):
        agent = build_intent_router_agent()

    assert "2026-10-15" in agent.instruction


def test_update_prompt_contains_reference_date():
    previous = SearchRequest(city="Baku", check_in=date(2027, 4, 10), check_out=date(2027, 4, 15))

    with clock.use_reference_date(TODAY):
        prompt = intent_update._build_update_prompt(previous, "move to May")

    assert "2026-10-15" in prompt


# --- conversation flow: no uncaught exceptions ------------------------------


def _decision(action: ConversationAction) -> ConversationActionDecision:
    return ConversationActionDecision(
        action=action,
        reason="test",
        decision_status=ConversationDecisionStatus.RESOLVED,
        clarification_reason=ClarificationReason.NONE,
        result_scope=ResultScope.NOT_APPLICABLE,
        target_result_id="",
    )


@pytest.fixture
def flow(monkeypatch):
    """Mocks every LLM/retrieval boundary; records the searched state."""
    searched: list[SearchRequest] = []

    def set_route(action: ConversationAction) -> None:
        async def fake_route(**kwargs):
            return _decision(action)

        monkeypatch.setattr(conversation_flow, "route_conversation_async", fake_route)

    def set_intent(intent: IntentRoute) -> None:
        async def fake_intent(user_text, **kwargs):
            return intent

        monkeypatch.setattr(intent_router, "route_intent_adk_async", fake_intent)

    def set_patch(patch: SearchIntentPatch) -> None:
        async def fake_patch(previous_state, user_message, **kwargs):
            return patch

        monkeypatch.setattr(intent_update, "route_intent_update_patch_async", fake_patch)

    async def fake_orchestrate(state, **kwargs):
        searched.append(state)
        return (
            NormalizedSearchResponse(status=SearchStatus.NO_RESULTS, results=[]),
            ShownResultSet(result_set_id="rs-test", items=[]),
        )

    monkeypatch.setattr(conversation_flow, "orchestrate_search_request", fake_orchestrate)

    class Flow:
        pass

    f = Flow()
    f.searched = searched
    f.set_route = set_route
    f.set_intent = set_intent
    f.set_patch = set_patch
    return f


def _assert_response_layer_accepts(message: str, result: dict) -> None:
    # Used to raise ValidationError on an invalid persisted state.
    build_conversation_response_input(user_message=message, result=result)


async def test_start_search_cross_year_range_searches_correct_dates(flow):
    flow.set_route(ConversationAction.START_SEARCH)
    flow.set_intent(IntentRoute(city="Baku", check_in="2026-12-28", check_out="2026-01-03"))

    with clock.use_reference_date(TODAY):
        result = await conversation_flow.handle_user_message("Baku from Dec 28 to Jan 3")

    assert not result.get("need_clarification")
    assert [(s.check_in, s.check_out) for s in flow.searched] == [
        (date(2026, 12, 28), date(2027, 1, 3))
    ]
    _assert_response_layer_accepts("Baku from Dec 28 to Jan 3", result)


async def test_start_search_without_year_searches_next_occurrence(flow):
    flow.set_route(ConversationAction.START_SEARCH)
    flow.set_intent(IntentRoute(city="Baku", check_in="2026-04-10", check_out="2026-04-15"))

    with clock.use_reference_date(TODAY):
        await conversation_flow.handle_user_message("Baku April 10-15")

    assert flow.searched[0].check_in == date(2027, 4, 10)


async def test_start_search_with_past_dates_asks_and_keeps_rest_of_request(flow):
    flow.set_route(ConversationAction.START_SEARCH)
    flow.set_intent(IntentRoute(city="Baku", check_in="2025-04-10", check_out="2025-04-15"))

    with clock.use_reference_date(TODAY):
        result = await conversation_flow.handle_user_message("Baku April 10-15 2025")

    assert result["need_clarification"] is True
    assert "past" in result["questions"][0]
    assert result["state"]["city"] == "Baku"
    assert "check_in" not in result["state"]
    assert "shown_result_set" not in result
    assert flow.searched == []
    _assert_response_layer_accepts("Baku April 10-15 2025", result)


async def test_start_search_with_reversed_explicit_range_asks_instead_of_crashing(flow):
    flow.set_route(ConversationAction.START_SEARCH)
    flow.set_intent(IntentRoute(city="Baku", check_in="2027-04-15", check_out="2027-04-10"))

    with clock.use_reference_date(TODAY):
        result = await conversation_flow.handle_user_message("Baku 15 to 10 April 2027")

    assert result["need_clarification"] is True
    assert flow.searched == []
    _assert_response_layer_accepts("Baku 15 to 10 April 2027", result)


async def test_update_check_out_before_check_in_asks_and_keeps_previous_state(flow):
    previous = SearchRequest(city="Baku", check_in=date(2027, 4, 10), check_out=date(2027, 4, 15))
    flow.set_route(ConversationAction.UPDATE_SEARCH)
    flow.set_patch(SearchIntentPatch(set_check_out="2027-04-08"))

    with clock.use_reference_date(TODAY):
        result = await conversation_flow.handle_user_message(
            "check out on April 8 2027", previous_state=previous
        )

    assert result["need_clarification"] is True
    assert SearchRequest.model_validate(result["state"]) == previous
    assert "shown_result_set" not in result
    assert flow.searched == []
    _assert_response_layer_accepts("check out on April 8 2027", result)


async def test_update_check_out_across_new_year_is_anchored_to_check_in(flow):
    previous = SearchRequest(city="Baku", check_in=date(2026, 12, 28), check_out=date(2026, 12, 31))
    flow.set_route(ConversationAction.UPDATE_SEARCH)
    flow.set_patch(SearchIntentPatch(set_check_out="2026-01-03"))

    with clock.use_reference_date(TODAY):
        result = await conversation_flow.handle_user_message(
            "stay until Jan 3", previous_state=previous
        )

    assert not result.get("need_clarification")
    assert flow.searched[0].check_out == date(2027, 1, 3)


async def test_update_dates_without_year_follow_start_search_rules(flow):
    previous = SearchRequest(city="Baku", check_in=date(2026, 11, 3), check_out=date(2026, 11, 7))
    flow.set_route(ConversationAction.UPDATE_SEARCH)
    flow.set_patch(SearchIntentPatch(set_check_in="2026-04-10", set_check_out="2026-04-15"))

    with clock.use_reference_date(TODAY):
        await conversation_flow.handle_user_message(
            "change dates to April 10-15", previous_state=previous
        )

    assert (flow.searched[0].check_in, flow.searched[0].check_out) == (
        date(2027, 4, 10),
        date(2027, 4, 15),
    )


async def test_update_with_past_explicit_dates_keeps_previous_state(flow):
    previous = SearchRequest(city="Baku", check_in=date(2026, 11, 3), check_out=date(2026, 11, 7))
    flow.set_route(ConversationAction.UPDATE_SEARCH)
    flow.set_patch(SearchIntentPatch(set_check_in="2025-04-10", set_check_out="2025-04-15"))

    with clock.use_reference_date(TODAY):
        result = await conversation_flow.handle_user_message(
            "change dates to April 10-15 2025", previous_state=previous
        )

    assert result["need_clarification"] is True
    assert SearchRequest.model_validate(result["state"]) == previous
    assert flow.searched == []


# --- demo reproducibility ---------------------------------------------------


@pytest.mark.parametrize(
    "reference_date",
    [date(2026, 3, 1), date(2026, 10, 15), date(2027, 6, 1), date(2028, 1, 20)],
)
async def test_readme_demo_search_is_stable_regardless_of_today(monkeypatch, reference_date):
    """
    README example "apartment in Baku, April 10-15" through the real
    fixture retrieval, matching and ranking - only the LLM boundaries are
    mocked (and the LLM fallback disabled). The requested year moves with
    the reference date; the shown listings must not.
    """

    async def fake_route(**kwargs):
        return _decision(ConversationAction.START_SEARCH)

    async def fake_intent(user_text, **kwargs):
        return IntentRoute(
            city="Baku",
            check_in="2026-04-10",
            check_out="2026-04-15",
            property_types=["apartment"],
        )

    monkeypatch.setattr(conversation_flow, "route_conversation_async", fake_route)
    monkeypatch.setattr(intent_router, "route_intent_adk_async", fake_intent)

    with clock.use_reference_date(reference_date):
        result = await conversation_flow.handle_user_message(
            "I am looking for an apartment in Baku for April 10-15",
            fallback_policy=FallbackPolicy(enabled=False),
        )

    result_ids = [item["result_id"] for item in result["results"]]

    assert result["status"] == "results"
    assert result["state"]["check_in"].endswith("-04-10")
    assert result_ids == EXPECTED_DEMO_RESULT_IDS


EXPECTED_DEMO_RESULT_IDS = [
    "apt-small",
    "apt-expensive",
    "baku-apt-1",
    "baku-apt-2",
    "baku-apt-3",
]
