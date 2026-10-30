"""
Hermetic multi-turn conversations over LiteAPI data (source="liteapi").

Real: process_turn, conversation flow, LiteApiRetriever and its client,
normalization, structured matching, budget filtering, ranking/selection,
reference validation, deterministic verdicts and rendering, session commit.

Scripted: the LLM boundaries (ScriptedLlm from test_multi_turn_scenarios,
response wording falls back to the deterministic renderer) and LiteAPI
itself - an httpx.MockTransport serving synthetic payloads in LiteAPI's
shape. No provider data is stored here; internal amounts are distinctive
sentinels so any leak is caught by a substring check.
"""

from __future__ import annotations

import json
from datetime import date
from types import SimpleNamespace

import httpx
import pytest
from test_multi_turn_scenarios import ScriptedLlm, _amenity, ask_about, route

from app.agents.intent_router_agent import IntentRoute
from app.api.main import create_app
from app.config import clock, settings
from app.logic.conversation_response_llm import _build_llm_payload
from app.retrieval.liteapi_client import LiteApiClient
from app.schemas.constraints import ConstraintPriority
from app.schemas.conversation_response import SearchConversationOutcome
from app.schemas.conversation_route import ConversationAction
from app.schemas.fields import Field
from app.schemas.filters import PriceConstraint, SearchFilters
from app.schemas.intent_patch import SearchIntentPatch
from app.schemas.result_query import ResultQuery
from app.schemas.session import ConversationSession
from app.schemas.turn import TurnConfig, TurnResult
from app.services.conversation_service import process_turn

REFERENCE_DATE = date(2026, 10, 29)
CONFIG = TurnConfig(source="liteapi", persist_telemetry=False, include_debug=True)

NEW_SEARCH = "Find me a hotel in Barcelona, December 10-13, 2026, for two adults, with free Wi-Fi, under $300 per night."
REFINE = "Actually, I'd prefer somewhere with parking."
INSPECT = "Does the first hotel have a balcony?"

# Internal amounts (never shown): one distinct sentinel per field kind.
RETAIL_TOTAL, INITIAL_PRICE, COMMISSION = 517.43, 611.29, 41.17
PROVIDER_COMMISSION, TAX, CITY_TAX, PENALTY = 13.71, 37.19, 11.83, 233.47
CONFIDENTIAL = [RETAIL_TOTAL, INITIAL_PRICE, COMMISSION, PROVIDER_COMMISSION, TAX, CITY_TAX, PENALTY]


# --- synthetic LiteAPI payloads -------------------------------------------------


def _money(amount: float) -> dict:
    return {"amount": amount, "currency": "USD"}


def _offer(room_name: str, ssp: float) -> dict:
    rate = {
        "rateId": f"rate-{room_name}",
        "name": room_name,
        "maxOccupancy": 2,
        "boardType": "RO",
        "boardName": "Room Only",
        "commission": [_money(COMMISSION)],
        "providerCommission": _money(PROVIDER_COMMISSION),
        "retailRate": {
            "total": [_money(RETAIL_TOTAL)],
            "suggestedSellingPrice": [{**_money(ssp), "source": "providerDirect"}],
            "initialPrice": [_money(INITIAL_PRICE)],
            "taxesAndFees": [
                {"included": True, "description": "VAT", **_money(TAX)},
                {"included": False, "description": "City tax", **_money(CITY_TAX)},
            ],
        },
        "cancellationPolicies": {
            "cancelPolicyInfos": [{"cancelTime": "2026-12-01 10:00:00", "timezone": "GMT", **_money(PENALTY)}],
            "refundableTag": "RFN",
        },
    }
    return {
        "offerId": f"offer-secret-{room_name}",
        "rates": [rate],
        "offerRetailRate": _money(RETAIL_TOTAL),
        "offerInitialPrice": _money(INITIAL_PRICE),
        "suggestedSellingPrice": {**_money(ssp), "source": "providerDirect"},
    }


def _hotel(hotel_id: str, name: str, facilities: list[str], room_amenities: list[str]) -> dict:
    return {
        "id": hotel_id,
        "name": name,
        "hotelDescription": f"<p><strong>{name}</strong></p><p>A hotel in central Barcelona.</p>",
        "hotelType": "Hotels",
        "starRating": 4,
        "rating": 9.0,
        "reviewCount": 1000,
        "city": "Barcelona",
        "address": "Example street",
        "location": {"latitude": 41.39, "longitude": 2.17},
        "facilities": [{"facilityId": i, "name": f} for i, f in enumerate(facilities)],
        "rooms": [
            {
                "roomName": "Double Room",
                "roomSizeSquare": 20,
                "roomSizeUnit": "m2",
                "maxOccupancy": 2,
                "roomAmenities": [{"amenitiesId": i, "name": a} for i, a in enumerate(room_amenities)],
            }
        ],
        "sentiment_analysis": {"pros": [f"{name} pro"], "cons": [f"{name} con"], "categories": []},
        "sentiment_updated_at": "2026-01-01T00:00:00Z",
    }


# A: WiFi, nothing said about parking or a balcony.  B: WiFi, parking, balcony room.
# C: WiFi and parking, but over the $300/night budget (3 nights -> $900 total).
HOTELS = {
    "lp-a": (_hotel("lp-a", "Arc Hotel", ["Free WiFi"], ["Desk"]), 600.0),
    "lp-b": (_hotel("lp-b", "Bay Hotel", ["Free WiFi", "Parking"], ["Balcony", "Desk"]), 750.0),
    "lp-c": (_hotel("lp-c", "Crown Hotel", ["Free WiFi", "Parking"], ["Desk"]), 1200.0),
}


class FakeLiteApi:
    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.rates_status = 200

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.url.path.endswith("/hotels/rates"):
            if self.rates_status != 200:
                return httpx.Response(self.rates_status)
            data = [{"hotelId": hid, "roomTypes": [_offer("Double Room", ssp)]} for hid, (_, ssp) in HOTELS.items()]
            return httpx.Response(200, json={"sandbox": True, "data": data})
        hotel, _ = HOTELS[request.url.params["hotelId"]]
        return httpx.Response(200, json={"data": hotel})

    def rates_bodies(self) -> list[dict]:
        return [json.loads(r.content) for r in self.requests if r.url.path.endswith("/hotels/rates")]


@pytest.fixture
def liteapi(monkeypatch) -> FakeLiteApi:
    fake = FakeLiteApi()

    def from_env(cls, **kwargs):
        return LiteApiClient("sand_test-key", transport=httpx.MockTransport(fake), retry_delay_seconds=0)

    monkeypatch.setattr(LiteApiClient, "from_env", classmethod(from_env))
    return fake


class CapturingLlm(ScriptedLlm):
    """ScriptedLlm that also keeps every textual-fallback prompt."""

    def __init__(self, monkeypatch) -> None:
        self.fallback_prompts: list[str] = []
        super().__init__(monkeypatch)

    def _genai_client(self):
        llm = self

        class Models:
            def generate_content(self, **kwargs):
                llm.fallback_calls += 1
                llm.fallback_prompts.append(
                    json.dumps([part.text for content in kwargs["contents"] for part in content.parts])
                )
                return SimpleNamespace(
                    text=json.dumps({"signals": [], "evidence_missing": True, "reason": "No evidence."}),
                    usage_metadata=None,
                )

        return SimpleNamespace(models=Models())


@pytest.fixture
def llm(monkeypatch):
    scripted = CapturingLlm(monkeypatch)
    yield scripted
    scripted.assert_consumed()


class Conversation:
    def __init__(self) -> None:
        self.session = ConversationSession()
        self.turns: list[TurnResult] = []

    async def say(self, message: str) -> TurnResult:
        with clock.use_reference_date(REFERENCE_DATE):
            turn = await process_turn(self.session, message, config=CONFIG)
        self.session = turn.session
        self.turns.append(turn)
        return turn


def WIFI():
    return _amenity(Field.WIFI, "free Wi-Fi")


def PARKING_NICE():
    return _amenity(Field.PARKING, "parking").model_copy(update={"priority": ConstraintPriority.NICE})


def BALCONY():
    return _amenity(Field.BALCONY, "balcony")


def _new_search_intent(**overrides) -> IntentRoute:
    values = dict(
        city="Barcelona",
        check_in="2026-12-10",
        check_out="2026-12-13",
        adults=2,
        rooms=1,
        constraints=[WIFI()],
        filters=SearchFilters(price=PriceConstraint(max_amount=300, currency="USD", scope="per_night")),
    )
    values.update(overrides)
    return IntentRoute(**values)


def _shown(turn: TurnResult) -> list[str]:
    return [item.result_id for item in turn.session.shown_result_set.items]


def _assert_no_confidential_amounts(text: str) -> None:
    for amount in CONFIDENTIAL:
        assert str(amount) not in text, f"internal amount {amount} leaked"
    assert "offer-secret" not in text
    assert "commission" not in text.lower()


# --- A -> B -> C on live-shaped data -------------------------------------------------


async def test_new_search_refine_and_inspect_on_liteapi_data(llm, liteapi):
    chat = Conversation()

    # A. NEW_SEARCH
    llm.router_steps.append(route(ConversationAction.START_SEARCH))
    llm.intents.append(_new_search_intent())
    search = await chat.say(NEW_SEARCH)

    assert search.conversation_action == ConversationAction.START_SEARCH
    assert search.outcome_kind == "search" and search.failure is None
    assert search.inspector.search_status == "results"
    first_request = search.session.search_request
    assert (first_request.city, first_request.check_in, first_request.check_out) == (
        "Barcelona", date(2026, 12, 10), date(2026, 12, 13),
    )
    assert first_request.filters.price.max_amount == 300
    # Crown Hotel is over budget ($1200 for 3 nights); the others are shown.
    assert sorted(_shown(search)) == ["lp-a", "lp-b"]
    assert [card.result_id for card in search.listing_cards] == _shown(search)
    for item in search.session.shown_result_set.items:
        assert item.listing.provenance.provider == "liteapi"
        assert item.listing.price == HOTELS[item.result_id][1]  # the public SSP, nothing else
    rates = liteapi.rates_bodies()
    assert rates == [{
        "cityName": "Barcelona", "countryCode": "ES", "checkin": "2026-12-10", "checkout": "2026-12-13",
        "occupancies": [{"adults": 2, "children": []}], "currency": "USD", "guestNationality": "US",
        "maxRatesPerHotel": 3, "limit": 20, "timeout": 10,
    }]
    assert len(liteapi.requests) == 1 + len(HOTELS)

    # B. REFINE_SEARCH: a new preference; city, dates, budget and WiFi are kept.
    llm.router_steps.append(route(ConversationAction.UPDATE_SEARCH))
    llm.patches.append(SearchIntentPatch(add_constraints=[PARKING_NICE()]))
    refine = await chat.say(REFINE)

    assert refine.conversation_action == ConversationAction.UPDATE_SEARCH
    assert refine.outcome_kind == "search" and refine.failure is None
    refined = refine.session.search_request
    assert (refined.city, refined.check_in, refined.check_out, refined.filters) == (
        first_request.city, first_request.check_in, first_request.check_out, first_request.filters,
    )
    assert [(c.normalized_text, c.priority.value) for c in refined.constraints] == [
        ("free wi-fi", "must"), ("parking", "nice"),
    ]
    assert refine.session.shown_result_set.result_set_id != search.session.shown_result_set.result_set_id
    assert _shown(refine)[0] == "lp-b"  # the one hotel that states parking ranks first
    assert len(liteapi.rates_bodies()) == 2

    # C. INSPECT_RESULT: "the first hotel" is the first of the refined results.
    requests_before_question = len(liteapi.requests)
    llm.router_steps.append(ask_about(1))
    llm.factual.append(ResultQuery(constraints=[BALCONY()]))
    question = await chat.say(INSPECT)

    assert question.conversation_action == ConversationAction.LISTING_QUESTION
    assert question.outcome_kind == "listing_query"
    assert question.inspector.routing.target_result_id == "lp-b"
    [item] = question.inspector.listing_query
    assert item.match.result_id == "lp-b"
    [verdict] = item.match.constraint_results
    assert verdict.value.value == "YES"
    assert any("balcony" in (e.snippet or "").lower() for e in verdict.evidence)
    assert "Bay Hotel" in question.assistant_message
    # Read-only turn: no LiteAPI call, state untouched.
    assert len(liteapi.requests) == requests_before_question
    assert question.session.search_request == refine.session.search_request
    assert question.session.shown_result_set == refine.session.shown_result_set

    # Nothing confidential anywhere: public result, session, debug view, LLM inputs.
    for turn in chat.turns:
        _assert_no_confidential_amounts(turn.model_dump_json())
        _assert_no_confidential_amounts(turn.session.model_dump_json())
        _assert_no_confidential_amounts(turn.debug.model_dump_json())
    for router_input in llm.router_inputs:
        _assert_no_confidential_amounts(router_input.model_dump_json())
    search_inputs = [r for r in llm.response_inputs if isinstance(r.outcome, SearchConversationOutcome)]
    assert len(search_inputs) == 2
    for response_input in search_inputs:
        _assert_no_confidential_amounts(json.dumps(_build_llm_payload(response_input), default=str))
    for prompt in llm.fallback_prompts:
        _assert_no_confidential_amounts(prompt)


async def test_unconfirmed_fact_about_a_shown_hotel_is_uncertain_not_no(llm, liteapi):
    chat = Conversation()

    llm.router_steps.append(route(ConversationAction.START_SEARCH))
    llm.intents.append(_new_search_intent())
    search = await chat.say(NEW_SEARCH)
    arc_position = _shown(search).index("lp-a") + 1

    llm.router_steps.append(ask_about(arc_position))
    llm.factual.append(ResultQuery(constraints=[BALCONY()]))
    question = await chat.say("Does that hotel have a balcony?")

    [item] = question.inspector.listing_query
    assert item.match.result_id == "lp-a"
    assert item.match.constraint_results[0].value.value == "UNCERTAIN"
    # The textual check ran on Arc Hotel's own data and saw no internal amounts.
    assert llm.fallback_calls == 1
    assert "Arc Hotel" in llm.fallback_prompts[0]
    _assert_no_confidential_amounts(llm.fallback_prompts[0])


async def test_free_cancellation_question_on_a_refundable_rate_is_uncertain(llm, liteapi):
    chat = Conversation()

    llm.router_steps.append(route(ConversationAction.START_SEARCH))
    llm.intents.append(_new_search_intent())
    search = await chat.say(NEW_SEARCH)
    first = _shown(search)[0]
    [offer] = next(i.listing for i in search.session.shown_result_set.items if i.result_id == first).public_offers
    assert offer.refundable is True  # every synthetic rate here is RFN

    llm.router_steps.append(ask_about(1))
    llm.factual.append(ResultQuery(constraints=[_amenity(Field.FREE_CANCELLATION, "free cancellation")]))
    question = await chat.say("Can I cancel the first hotel for free?")

    [item] = question.inspector.listing_query
    assert item.match.result_id == first
    assert item.match.constraint_results[0].value.value == "UNCERTAIN"
    # The textual check saw no cancellation or refund wording to quote.
    [prompt] = llm.fallback_prompts
    evidence = json.loads(json.loads(prompt)[0])["listing_evidence"]
    assert not any("cancel" in e["text"].lower() or "refund" in e["text"].lower() for e in evidence)


# --- failures keep the conversation state --------------------------------------------


async def test_provider_failure_keeps_state_then_recovers(llm, liteapi):
    chat = Conversation()

    llm.router_steps.append(route(ConversationAction.START_SEARCH))
    llm.intents.append(_new_search_intent())
    search = await chat.say(NEW_SEARCH)

    liteapi.rates_status = 503
    llm.router_steps.append(route(ConversationAction.UPDATE_SEARCH))
    llm.patches.append(SearchIntentPatch(add_constraints=[PARKING_NICE()]))
    failed = await chat.say(REFINE)

    assert failed.outcome_kind == "failure"
    assert failed.failure.model_dump() == {"stage": "retrieval", "code": "provider_unavailable"}
    assert "Please try again" in failed.assistant_message
    assert "has not been changed" in failed.assistant_message
    assert failed.listing_cards == []
    assert failed.session.search_request == search.session.search_request
    assert failed.session.shown_result_set == search.session.shown_result_set
    assert len(liteapi.rates_bodies()) == 1 + 2  # first search + both attempts of the failed one

    liteapi.rates_status = 200
    llm.router_steps.append(route(ConversationAction.UPDATE_SEARCH))
    llm.patches.append(SearchIntentPatch(add_constraints=[PARKING_NICE()]))
    retried = await chat.say(REFINE)

    assert retried.outcome_kind == "search" and retried.failure is None
    assert [c.normalized_text for c in retried.session.search_request.constraints] == ["free wi-fi", "parking"]


@pytest.mark.parametrize(
    ("intent_overrides", "code", "message"),
    [
        pytest.param(
            {"city": "Madrid"}, "unsupported_destination",
            "Live hotel search currently covers only Barcelona.", id="other-city",
        ),
        pytest.param(
            {"adults": 3}, "unsupported_request",
            "Live hotel search currently supports one room for 2 adults, without children.", id="three-adults",
        ),
        pytest.param(
            {"children": 1}, "unsupported_request",
            "Live hotel search currently supports one room for 2 adults, without children.", id="children",
        ),
    ],
)
async def test_unsupported_searches_are_refused_explicitly_without_calling_liteapi(
    llm, liteapi, intent_overrides, code, message
):
    chat = Conversation()

    llm.router_steps.append(route(ConversationAction.START_SEARCH))
    llm.intents.append(_new_search_intent(**intent_overrides))
    turn = await chat.say(NEW_SEARCH)

    assert turn.outcome_kind == "failure"
    assert turn.failure.model_dump() == {"stage": "retrieval", "code": code}
    assert turn.assistant_message == f"{message} Your current search has not been changed."
    assert turn.inspector.response_source == "deterministic"
    assert liteapi.requests == []
    assert turn.session.search_request is None
    assert turn.session.shown_result_set is None


async def test_unsupported_city_mid_conversation_keeps_the_previous_search(llm, liteapi):
    chat = Conversation()

    llm.router_steps.append(route(ConversationAction.START_SEARCH))
    llm.intents.append(_new_search_intent())
    search = await chat.say(NEW_SEARCH)
    requests_after_search = len(liteapi.requests)

    llm.router_steps.append(route(ConversationAction.UPDATE_SEARCH))
    llm.patches.append(SearchIntentPatch(set_city="Madrid"))
    moved = await chat.say("Let's look in Madrid instead")

    assert moved.failure.code == "unsupported_destination"
    assert len(liteapi.requests) == requests_after_search
    assert moved.session.search_request == search.session.search_request
    assert moved.session.shown_result_set == search.session.shown_result_set


async def test_missing_sandbox_key_is_a_configuration_failure_without_retry_advice(llm):
    # No LiteAPI fake: the client is built from the (placeholder) environment.
    chat = Conversation()

    llm.router_steps.append(route(ConversationAction.START_SEARCH))
    llm.intents.append(_new_search_intent())
    turn = await chat.say(NEW_SEARCH)

    assert turn.failure.model_dump() == {"stage": "retrieval", "code": "configuration_error"}
    assert turn.assistant_message == "Hotel search isn't available right now. Your current search has not been changed."
    assert turn.session.search_request is None


# --- fixtures stay the default ---------------------------------------------------------


def test_fixtures_remain_the_default_source(monkeypatch):
    assert TurnConfig().source == "fixtures"
    assert settings.RETRIEVAL_SOURCE == "fixtures"

    from ui.services.chat_handler import STREAMLIT_TURN_CONFIG

    assert STREAMLIT_TURN_CONFIG.source == "fixtures"

    # The public API ignores the local flag.
    monkeypatch.setenv("BOOKING_RETRIEVAL_SOURCE", "liteapi")
    assert create_app().state.turn_config.source == "fixtures"
