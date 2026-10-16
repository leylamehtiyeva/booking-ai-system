"""
Budget scope clarification (baseline cases 0005, 0011, 0012, 0025, 0045).

A budget without "per night" / "for the whole stay" is ambiguous for a
multi-night stay; the extraction model used to pick a scope, and an
unknown scope was matched as a total, producing false NO results. Now:

- extraction keeps an unstated scope unknown (filters.price.scope = None);
- required-context resolution asks one question before searching;
- the answer is applied through the existing UPDATE_SEARCH patch;
- a one-night stay, an explicit scope or no budget never asks;
- the price matcher never turns an unknown multi-night scope into NO.
"""

from __future__ import annotations

from datetime import date

import pytest

from app.agents.intent_router_agent import IntentRoute, build_intent_router_agent
from app.agents.intent_update_agent import build_intent_update_agent
from app.config import clock
from app.llm.call_policy import LlmCallError
from app.logic import conversation_flow, intent_router, intent_update
from app.logic.intent_update import _build_update_prompt
from app.logic.numeric_filters import match_price_filters
from app.logic.request_resolution import resolve_required_search_context
from app.schemas.constraints import (
    ConstraintCategory,
    ConstraintMappingStatus,
    ConstraintPriority,
    EvidenceStrategy,
    UserConstraint,
)
from app.schemas.conversation_route import (
    ClarificationReason,
    ConversationAction,
    ConversationActionDecision,
    ConversationDecisionStatus,
    ResultScope,
)
from app.schemas.filters import PriceConstraint, SearchFilters
from app.schemas.fields import Field
from app.schemas.intent_patch import SearchIntentPatch
from app.schemas.match import Ternary
from app.schemas.query import SearchRequest
from app.schemas.session import ConversationSession
from app.schemas.turn import TurnConfig
from app.services.conversation_service import process_turn

TODAY = date(2026, 3, 1)


def _budget(amount: float, scope=None, currency: str | None = "USD") -> SearchFilters:
    return SearchFilters(price=PriceConstraint(max_amount=amount, currency=currency, scope=scope))


def _request(check_out: date = date(2026, 4, 30), **kwargs) -> SearchRequest:
    return SearchRequest(city="Baku", check_in=date(2026, 4, 1), check_out=check_out, **kwargs)


# --- required-context resolution ---------------------------------------------------------


def test_ambiguous_budget_for_several_nights_asks_one_question():
    resolved = resolve_required_search_context(_request(filters=_budget(200)))

    assert resolved.questions == ["Do you mean $200 per night or $200 for the whole stay?"]


@pytest.mark.parametrize("scope", ["per_night", "total_stay"])
def test_explicit_scope_does_not_ask(scope):
    assert resolve_required_search_context(_request(filters=_budget(200, scope))).questions == []


def test_one_night_stay_does_not_ask():
    resolved = resolve_required_search_context(_request(check_out=date(2026, 4, 2), filters=_budget(200)))

    assert resolved.questions == []


def test_no_budget_does_not_ask():
    assert resolve_required_search_context(_request()).questions == []


@pytest.mark.parametrize(
    ("currency", "expected"),
    [
        ("$", "Do you mean $200 per night or $200 for the whole stay?"),
        (None, "Do you mean 200 per night or 200 for the whole stay?"),
        ("MYR", "Do you mean 350 MYR per night or 350 MYR for the whole stay?"),
    ],
)
def test_question_names_the_amount_and_currency(currency, expected):
    amount = 350 if currency == "MYR" else 200
    resolved = resolve_required_search_context(_request(filters=_budget(amount, currency=currency)))

    assert resolved.questions == [expected]


def test_budget_question_waits_until_city_and_dates_are_known():
    resolved = resolve_required_search_context(SearchRequest(city="Baku", filters=_budget(200)))

    assert len(resolved.questions) == 1
    assert "dates" in resolved.questions[0].lower()


def test_minimum_only_budget_is_also_ambiguous():
    filters = SearchFilters(price=PriceConstraint(min_amount=100, currency="USD"))

    assert resolve_required_search_context(_request(filters=filters)).questions == [
        "Do you mean $100 per night or $100 for the whole stay?"
    ]


# --- matcher: no premature price verdict ------------------------------------------------------


def test_unknown_scope_for_several_nights_is_never_a_price_no():
    result = match_price_filters(
        5000.0,  # far above both readings of the budget
        "USD",
        _budget(200),
        check_in=date(2026, 4, 1),
        check_out=date(2026, 4, 30),
    )

    assert result.value == Ternary.UNCERTAIN
    assert "scope" in result.why.lower()


def test_unknown_scope_for_one_night_is_matched_as_before():
    result = match_price_filters(250.0, "USD", _budget(200), check_in=date(2026, 4, 1), check_out=date(2026, 4, 2))

    assert result.value == Ternary.NO


# --- prompts: an unstated scope stays unknown --------------------------------------------------


def test_intent_prompt_keeps_an_unstated_scope_unknown():
    instruction = build_intent_router_agent().instruction

    assert "leave filters.price.scope null" in instruction
    assert '"$200 max"' in instruction
    assert "per month" in instruction


def test_update_prompt_maps_a_scope_answer_to_a_scope_patch():
    prompt = _build_update_prompt(_request(filters=_budget(200)), "per night")

    for text in (prompt, build_intent_update_agent().instruction):
        assert '{"set_filters":{"price":{"scope":"per_night"}}}' in text
        assert '{"set_filters":{"price":{"scope":"total_stay"}}}' in text


# --- conversation flow --------------------------------------------------------------------


def _decision(action: ConversationAction) -> ConversationActionDecision:
    return ConversationActionDecision(
        action=action,
        reason="t",
        decision_status=ConversationDecisionStatus.RESOLVED,
        clarification_reason=ClarificationReason.NONE,
        result_scope=ResultScope.NOT_APPLICABLE,
        target_result_id="",
    )


WIFI = UserConstraint(
    raw_text="WiFi",
    normalized_text="wifi",
    priority=ConstraintPriority.MUST,
    category=ConstraintCategory.AMENITY,
    mapping_status=ConstraintMappingStatus.KNOWN,
    mapped_fields=[Field.WIFI],
    evidence_strategy=EvidenceStrategy.STRUCTURED,
)


@pytest.fixture
def scripted(monkeypatch):
    """Scripted LLM boundaries over the real flow, retrieval, matching and session commit."""
    queues = {"route": [], "intent": [], "patch": []}
    searches = []

    async def route(**kwargs):
        return _decision(queues["route"].pop(0))

    async def intent(user_text, **kwargs):
        return queues["intent"].pop(0)

    async def patch(previous_state, user_message, **kwargs):
        return queues["patch"].pop(0)

    real_orchestrate = conversation_flow.orchestrate_search_request

    async def spy_orchestrate(state, **kwargs):
        searches.append(state)
        return await real_orchestrate(state, **kwargs)

    def no_response_model(*args, **kwargs):
        raise LlmCallError("configuration_error", stage="conversation_response_generation")

    monkeypatch.setattr(conversation_flow, "route_conversation_async", route)
    monkeypatch.setattr(intent_router, "route_intent_adk_async", intent)
    monkeypatch.setattr(intent_update, "route_intent_update_patch_async", patch)
    monkeypatch.setattr(conversation_flow, "orchestrate_search_request", spy_orchestrate)
    monkeypatch.setattr("app.logic.conversation_response_llm.build_adk_model", no_response_model)
    yield queues, searches
    assert all(not q for q in queues.values()), queues


async def _turn(session: ConversationSession, message: str):
    before = session.model_dump(mode="json")
    with clock.use_reference_date(TODAY):
        turn = await process_turn(session, message, config=TurnConfig(persist_telemetry=False))
    assert session.model_dump(mode="json") == before, "process_turn mutated its input session"
    return turn


def _start(queues, scope=None, check_out="2026-04-30", budget=True):
    queues["route"].append(ConversationAction.START_SEARCH)
    queues["intent"].append(
        IntentRoute(
            city="Baku",
            check_in="2026-04-01",
            check_out=check_out,
            property_types=["apartment"],
            constraints=[WIFI],
            filters=_budget(200, scope, currency="$") if budget else SearchFilters(),
        )
    )


async def test_ambiguous_budget_asks_before_searching_and_keeps_the_request(scripted):
    queues, searches = scripted
    _start(queues)

    turn = await _turn(ConversationSession(), "Apartment in Baku for April with WiFi, $200 max")

    assert turn.outcome_kind == "clarification"
    assert "per night or $200 for the whole stay" in turn.assistant_message
    assert searches == []  # no search, so no premature price filtering
    state = turn.session.search_request
    assert (state.city, state.check_in, state.check_out) == ("Baku", date(2026, 4, 1), date(2026, 4, 30))
    assert [c.normalized_text for c in state.constraints] == ["wifi"]
    assert (state.filters.price.max_amount, state.filters.price.scope) == (200, None)
    assert turn.session.shown_result_set is None


@pytest.mark.parametrize("scope", ["per_night", "total_stay"])
async def test_explicit_scope_searches_directly(scripted, scope):
    queues, searches = scripted
    _start(queues, scope=scope)

    turn = await _turn(ConversationSession(), "Apartment in Baku for April with WiFi, $200")

    assert turn.outcome_kind == "search"
    assert len(searches) == 1


@pytest.mark.parametrize(
    ("answer", "scope"),
    [("per night", "per_night"), ("for the whole stay", "total_stay")],
)
async def test_answer_to_the_budget_question_updates_the_search(scripted, answer, scope):
    queues, searches = scripted
    _start(queues)
    queues["route"].append(ConversationAction.UPDATE_SEARCH)
    queues["patch"].append(SearchIntentPatch(set_filters=SearchFilters(price=PriceConstraint(scope=scope))))

    clarification = await _turn(ConversationSession(), "Apartment in Baku for April with WiFi, $200 max")
    search = await _turn(clarification.session, answer)

    assert search.outcome_kind == "search"
    assert len(searches) == 1
    state = search.session.search_request
    assert (state.filters.price.max_amount, state.filters.price.currency, state.filters.price.scope) == (200, "$", scope)
    assert (state.city, state.check_in, state.check_out) == ("Baku", date(2026, 4, 1), date(2026, 4, 30))
    assert [c.normalized_text for c in state.constraints] == ["wifi"]
    assert search.session.shown_result_set is not None


async def test_per_night_answer_finds_what_a_total_reading_would_have_rejected(scripted):
    """The false-NO mechanism: $200/night for 29 nights admits listings a $200 total never would."""
    queues, searches = scripted
    _start(queues)
    queues["route"].append(ConversationAction.UPDATE_SEARCH)
    queues["patch"].append(SearchIntentPatch(set_filters=SearchFilters(price=PriceConstraint(scope="per_night"))))

    clarification = await _turn(ConversationSession(), "Apartment in Baku for April with WiFi, $200 max")
    search = await _turn(clarification.session, "per night")

    assert search.inspector.search_status == "results"
    assert search.listing_cards


async def test_one_night_stay_with_a_budget_searches_without_asking(scripted):
    queues, searches = scripted
    _start(queues, check_out="2026-04-02")

    turn = await _turn(ConversationSession(), "Apartment in Baku on April 1st for one night with WiFi, $200 max")

    assert turn.outcome_kind == "search"
    assert len(searches) == 1


async def test_no_budget_never_asks_about_budget(scripted):
    queues, searches = scripted
    _start(queues, budget=False)

    turn = await _turn(ConversationSession(), "Apartment in Baku for April with WiFi")

    assert turn.outcome_kind == "search"
    assert "per night" not in turn.assistant_message


def test_scope_only_patch_survives_validation_and_merges_into_the_budget():
    """The validator used to drop a price without an amount, losing the answer."""
    from app.logic.apply_intent_patch import apply_intent_patch

    patch = SearchIntentPatch.model_validate_json('{"set_filters":{"price":{"scope":"per_night"}}}')
    updated = apply_intent_patch(_request(filters=_budget(200, currency="$")), patch)

    assert patch.set_filters.price.scope == "per_night"
    assert (updated.filters.price.max_amount, updated.filters.price.currency, updated.filters.price.scope) == (200, "$", "per_night")


# --- monthly / weekly budgets: an explicit unit the schema cannot express ------------------


def _period_budget(raw: str = "$900 per month", normalized: str = "budget per month") -> UserConstraint:
    """How extraction keeps a monthly or weekly rate: as a requirement, never as a price filter."""
    return UserConstraint(
        raw_text=raw,
        normalized_text=normalized,
        priority=ConstraintPriority.MUST,
        category=ConstraintCategory.NUMERIC,
        mapping_status=ConstraintMappingStatus.KNOWN,
        mapped_fields=[],
        evidence_strategy=EvidenceStrategy.NONE,
    )


def test_monthly_budget_asks_for_a_supported_budget_without_guessing():
    resolved = resolve_required_search_context(_request(constraints=[_period_budget()]))

    assert resolved.questions == [
        "You mentioned $900 per month. I can filter by a nightly or total-stay budget. "
        "What budget would you like me to use for your dates?"
    ]


def test_weekly_budget_asks_the_same_way_even_for_one_night():
    resolved = resolve_required_search_context(
        _request(check_out=date(2026, 4, 2), constraints=[_period_budget("€300 a week", "budget per week")])
    )

    assert resolved.questions[0].startswith("You mentioned €300 a week.")


def test_monthly_budget_marker_survives_validation_and_creates_no_price_filter():
    marker = UserConstraint.model_validate(_period_budget().model_dump())

    assert marker.mapping_status == ConstraintMappingStatus.KNOWN  # numeric is slot-backed, left as is
    assert _request(constraints=[marker]).filters is None


def test_monthly_budget_is_not_asked_again_once_a_supported_budget_exists():
    state = _request(constraints=[_period_budget()], filters=_budget(40, "per_night"))

    assert resolve_required_search_context(state).questions == []


def test_intent_and_update_prompts_keep_period_rates_out_of_the_price_filter():
    instructions = [build_intent_router_agent().instruction, build_intent_update_agent().instruction]
    instructions.append(_build_update_prompt(_request(), "$900 per month"))

    for text in instructions:
        assert '"budget per month"' in text
        assert "never put it in filters.price" in text.lower()


async def test_monthly_budget_flow_asks_then_applies_the_new_amount_and_scope(scripted):
    queues, searches = scripted
    queues["route"] += [ConversationAction.START_SEARCH, ConversationAction.UPDATE_SEARCH]
    queues["intent"].append(
        IntentRoute(
            city="Baku",
            check_in="2026-04-01",
            check_out="2026-04-30",
            property_types=["apartment"],
            constraints=[WIFI, _period_budget()],
        )
    )
    # The answer sets amount AND scope, and drops the monthly marker.
    queues["patch"].append(
        SearchIntentPatch(
            set_filters=SearchFilters(price=PriceConstraint(max_amount=40, currency="$", scope="per_night")),
            remove_constraint_texts=["$900 per month"],
        )
    )

    clarification = await _turn(ConversationSession(), "Apartment in Baku for April with WiFi, my budget is $900 per month")

    assert clarification.outcome_kind == "clarification"
    assert clarification.assistant_message.startswith("You mentioned $900 per month.")
    assert searches == []
    state = clarification.session.search_request
    assert (state.city, state.check_in, state.check_out) == ("Baku", date(2026, 4, 1), date(2026, 4, 30))
    assert state.filters is None  # no price filter from a monthly rate
    assert {c.normalized_text for c in state.constraints} == {"wifi", "budget per month"}

    search = await _turn(clarification.session, "$40 per night")

    assert search.outcome_kind == "search"
    assert len(searches) == 1
    price = search.session.search_request.filters.price
    assert (price.max_amount, price.scope) == (40, "per_night")
    assert [c.normalized_text for c in search.session.search_request.constraints] == ["wifi"]


async def test_scope_only_answer_to_a_monthly_budget_does_not_reuse_the_monthly_amount(scripted):
    queues, searches = scripted
    queues["route"] += [ConversationAction.START_SEARCH, ConversationAction.UPDATE_SEARCH]
    queues["intent"].append(
        IntentRoute(city="Baku", check_in="2026-04-01", check_out="2026-04-30", constraints=[_period_budget()])
    )
    queues["patch"].append(SearchIntentPatch(set_filters=SearchFilters(price=PriceConstraint(scope="per_night"))))

    clarification = await _turn(ConversationSession(), "Baku for April, $900 per month")
    again = await _turn(clarification.session, "per night")

    # No amount was given, so nothing is filtered and the question stands.
    assert again.outcome_kind == "clarification"
    assert again.assistant_message.startswith("You mentioned $900 per month.")
    assert searches == []
    assert again.session.search_request.filters.price.max_amount is None


# --- price validator after the scope-only fix --------------------------------------------------


def test_scope_only_price_creates_no_active_price_filter():
    filters = SearchFilters(price=PriceConstraint(scope="per_night"))

    assert filters.price is not None
    assert match_price_filters(5000.0, "USD", filters, check_in=date(2026, 4, 1), check_out=date(2026, 4, 30)) is None
    assert resolve_required_search_context(_request(filters=filters)).questions == []


def test_completely_empty_price_is_still_removed():
    assert SearchFilters(price=PriceConstraint()).price is None


@pytest.mark.parametrize(
    "price",
    [
        PriceConstraint(max_amount=200, currency="USD", scope="per_night"),
        PriceConstraint(min_amount=50, max_amount=120, currency="EUR", scope="total_stay"),
        PriceConstraint(max_amount=80),
    ],
)
def test_valid_price_objects_are_unchanged(price):
    assert SearchFilters(price=price).price == price


async def test_no_scope_warning_reaches_the_user_after_a_correct_update(scripted):
    queues, _ = scripted
    _start(queues)
    queues["route"].append(ConversationAction.UPDATE_SEARCH)
    queues["patch"].append(SearchIntentPatch(set_filters=SearchFilters(price=PriceConstraint(scope="per_night"))))

    clarification = await _turn(ConversationSession(), "Apartment in Baku for April with WiFi, $200 max")
    search = await _turn(clarification.session, "per night")

    public = search.model_dump_json()
    assert "scope (per night or whole stay) not specified" not in public
    assert "per night or $200 for the whole stay" not in search.assistant_message


# --- monthly budget: extraction mistakes are caught from the user's own words ----------------
# Adversarial: the model ignores the rule and returns the monthly amount as a price filter.


def _misread_monthly(amount: float = 900, scope=None, constraints=None) -> IntentRoute:
    return IntentRoute(
        city="Baku",
        check_in="2026-04-01",
        check_out="2026-04-30",
        constraints=constraints or [],
        filters=_budget(amount, scope, currency="$"),
    )


async def test_misread_monthly_budget_then_per_night_never_becomes_900_a_night(scripted):
    queues, searches = scripted
    queues["route"] += [ConversationAction.START_SEARCH, ConversationAction.UPDATE_SEARCH]
    queues["intent"].append(_misread_monthly())
    queues["patch"].append(SearchIntentPatch(set_filters=SearchFilters(price=PriceConstraint(scope="per_night"))))

    first = await _turn(ConversationSession(), "Baku for April. My budget is $900 per month.")

    assert first.assistant_message.startswith("You mentioned $900 per month.")
    assert "$900 per night" not in first.assistant_message
    assert first.session.search_request.filters is None or first.session.search_request.filters.price is None

    second = await _turn(first.session, "per night")

    assert second.outcome_kind == "clarification"
    assert searches == []
    price = second.session.search_request.filters.price
    assert price.max_amount is None  # $900 was never confirmed as a nightly amount


@pytest.mark.parametrize(
    "message",
    [
        "My budget is $900 per month.",
        "I can pay 900 USD a month",
        "budget: $900/month",
        "around $900 monthly",
        "My monthly budget is $900",
        "I spend €250 per week on rent",
    ],
)
async def test_period_wording_is_detected_without_relying_on_the_model_marker(scripted, message):
    queues, searches = scripted
    queues["route"].append(ConversationAction.START_SEARCH)
    amount = 250 if "250" in message else 900
    queues["intent"].append(_misread_monthly(amount))

    turn = await _turn(ConversationSession(), message)

    assert turn.outcome_kind == "clarification"
    assert turn.assistant_message.startswith("You mentioned")
    assert searches == []


async def test_a_silently_converted_monthly_amount_is_also_dropped(scripted):
    """$900/month -> $30/night is an interpretation the user never stated."""
    queues, searches = scripted
    queues["route"].append(ConversationAction.START_SEARCH)
    queues["intent"].append(_misread_monthly(30, scope="per_night"))

    turn = await _turn(ConversationSession(), "Baku for April, $900 per month")

    assert turn.assistant_message.startswith("You mentioned $900 per month.")
    assert searches == []


async def test_a_separately_stated_nightly_budget_is_kept(scripted):
    queues, searches = scripted
    queues["route"].append(ConversationAction.START_SEARCH)
    queues["intent"].append(_misread_monthly(40, scope="per_night"))

    turn = await _turn(ConversationSession(), "I usually pay $900 per month at home, here $40 per night max")

    assert turn.outcome_kind == "search"
    assert turn.session.search_request.filters.price.max_amount == 40


async def test_a_stay_length_is_not_a_monthly_rate(scripted):
    queues, searches = scripted
    queues["route"].append(ConversationAction.START_SEARCH)
    queues["intent"].append(_misread_monthly(900, scope="total_stay"))

    turn = await _turn(ConversationSession(), "Baku for a month, $900 in total for the whole month")

    assert turn.outcome_kind == "search"
    assert turn.session.search_request.filters.price.scope == "total_stay"


async def test_monthly_budget_given_in_an_update_is_guarded_too(scripted):
    queues, searches = scripted
    _start(queues, scope="per_night")
    queues["route"].append(ConversationAction.UPDATE_SEARCH)
    queues["patch"].append(SearchIntentPatch(set_filters=_budget(900, currency="$")))

    first = await _turn(ConversationSession(), "Apartment in Baku for April with WiFi, $200 per night")
    second = await _turn(first.session, "actually make it $900 per month")

    assert second.outcome_kind == "clarification"
    assert second.assistant_message.startswith("You mentioned $900 per month.")
    assert len(searches) == 1
    assert second.session.search_request.filters is None or second.session.search_request.filters.price is None
