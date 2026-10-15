"""
P0-8: fixture listings store `price` as the TOTAL for their whole
`available_dates` window (rate x window nights). Every consumer downstream
(price filter, evidence, facts, listing cards) reads ListingRaw.price as
the total for the REQUESTED stay - which is what Apify returns. The
fixture retriever therefore converts:

    requested_total = window_total * requested_nights / window_nights

using the original window length, so the per-night rate is preserved for
recurring (other-year), cross-year and leap-narrowed windows alike.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from app.agents.intent_router_agent import IntentRoute
from app.config import clock
from app.llm.call_policy import LlmCallError
from app.logic import conversation_flow, intent_router
from app.retrieval.fixtures import FIXTURES_PATH, FixturesRetriever
from app.schemas.conversation_route import (
    ClarificationReason,
    ConversationAction,
    ConversationActionDecision,
    ConversationDecisionStatus,
    ResultScope,
)
from app.schemas.fallback_policy import FallbackPolicy
from app.schemas.filters import PriceConstraint, SearchFilters
from app.schemas.query import SearchRequest
from app.schemas.session import ConversationSession
from app.schemas.turn import TurnConfig
from app.services.conversation_service import process_turn
from app.tools.orchestrate_search_tool import orchestrate_search_request

# apt-large-good: Baku apartment, 3150 USD for 2026-08-01..2026-09-15
# (45 nights = 70 USD/night); the only Baku listing available in August.
LISTING_ID = "apt-large-good"


def _request(check_in: date, check_out: date, *, price: PriceConstraint | None = None) -> SearchRequest:
    return SearchRequest(
        city="Baku",
        check_in=check_in,
        check_out=check_out,
        property_types=["apartment"],
        filters=SearchFilters(price=price) if price else None,
        constraints=[],
    )


async def _price(request: SearchRequest, retriever: FixturesRetriever | None = None) -> float | None:
    listings = await (retriever or FixturesRetriever()).get_candidates(request, max_items=40)
    return next(listing.price for listing in listings if listing.id == LISTING_ID)


# --- retriever -----------------------------------------------------------------


async def test_short_stay_is_priced_for_the_requested_nights():
    assert await _price(_request(date(2026, 8, 10), date(2026, 8, 15))) == 350.0


async def test_whole_window_keeps_the_original_total():
    assert await _price(_request(date(2026, 8, 1), date(2026, 9, 15))) == 3150.0


async def test_recurring_window_in_another_year_uses_the_same_rate():
    assert await _price(_request(date(2027, 8, 10), date(2027, 8, 15))) == 350.0


def _write(path: Path, listing: dict) -> FixturesRetriever:
    path.write_text(json.dumps([{"id": LISTING_ID, "name": "x", "city": "Baku", **listing}]), encoding="utf-8")
    return FixturesRetriever(path=path)


async def test_cross_year_window(tmp_path):
    # 2026-12-20..2027-01-10 = 21 nights for 2100 -> 100/night.
    retriever = _write(
        tmp_path / "l.json",
        {"price": 2100, "available_dates": {"check_in": "2026-12-20", "check_out": "2027-01-10"}},
    )

    price = await _price(_request(date(2027, 12, 28), date(2028, 1, 3)), retriever)

    assert price == 600.0


async def test_leap_narrowed_window_keeps_the_original_rate(tmp_path):
    # Feb 2028 window, 28 nights for 2800 -> 100/night; in 2027 the window
    # is narrowed to Feb 28, but the rate must stay 100.
    retriever = _write(
        tmp_path / "l.json",
        {"price": 2800, "available_dates": {"check_in": "2028-02-01", "check_out": "2028-02-29"}},
    )

    price = await _price(_request(date(2027, 2, 10), date(2027, 2, 15)), retriever)

    assert price == 500.0


@pytest.mark.parametrize(
    "listing",
    [
        {"price": 1000},  # no availability window -> price covers an unknown period
        {"price": 1000, "available_dates": {"check_in": "soon", "check_out": "later"}},
    ],
    ids=["no-window", "unparseable-window"],
)
async def test_unknown_window_length_gives_no_price_rather_than_an_invented_one(tmp_path, listing):
    retriever = _write(tmp_path / "l.json", listing)

    assert await _price(_request(date(2026, 8, 10), date(2026, 8, 15)), retriever) is None


async def test_fixture_file_is_not_modified():
    before = FIXTURES_PATH.read_text(encoding="utf-8")

    await _price(_request(date(2026, 8, 10), date(2026, 8, 15)))

    assert FIXTURES_PATH.read_text(encoding="utf-8") == before
    stored = next(item for item in json.loads(before) if item["id"] == LISTING_ID)
    assert stored["price"] == 3150


# --- downstream: matching, evidence, facts, shown snapshot ----------------------


async def _search(price: PriceConstraint):
    response, shown = await orchestrate_search_request(
        _request(date(2026, 8, 10), date(2026, 8, 15), price=price),
        result_limit=5,
        candidate_pool_size=40,
        fallback_policy=FallbackPolicy(enabled=False),
    )
    result = next(r for r in response.results if r.result_id == LISTING_ID)
    return result, shown


def _price_status(result):
    statuses = [
        c
        for c in (*result.matched_constraints, *result.uncertain_constraints, *result.failed_constraints)
        if "price" in c.name
    ]
    assert len(statuses) == 1, [c.name for c in statuses]
    return statuses[0]


@pytest.mark.parametrize(
    "budget",
    [
        PriceConstraint(max_amount=100, currency="USD", scope="per_night"),
        PriceConstraint(max_amount=500, currency="USD", scope="total_stay"),
    ],
    ids=["100-per-night", "500-total"],
)
async def test_budget_check_uses_the_stay_price(budget):
    result, shown = await _search(budget)

    status = _price_status(result)
    assert status.status == "matched", status.reason
    assert "350" in status.reason
    facts = {fact.key: fact.value for fact in result.facts}
    assert facts["listing_price_total"] == 350.0
    assert facts["listing_price_per_night_derived"] == 70.0
    shown_listing = next(item.listing for item in shown.items if item.result_id == LISTING_ID)
    assert shown_listing.price == 350.0


async def test_budget_below_the_nightly_rate_is_still_rejected():
    response, _ = await orchestrate_search_request(
        _request(date(2026, 8, 10), date(2026, 8, 15), price=PriceConstraint(max_amount=60, currency="USD", scope="per_night")),
        result_limit=5,
        candidate_pool_size=40,
        fallback_policy=FallbackPolicy(enabled=False),
    )

    assert LISTING_ID not in [r.result_id for r in response.results]


async def test_listing_card_and_answer_show_the_stay_price(monkeypatch):
    async def route(**kwargs):
        return ConversationActionDecision(
            action=ConversationAction.START_SEARCH,
            reason="t",
            decision_status=ConversationDecisionStatus.RESOLVED,
            clarification_reason=ClarificationReason.NONE,
            result_scope=ResultScope.NOT_APPLICABLE,
            target_result_id="",
        )

    async def intent(user_text, **kwargs):
        return IntentRoute(
            city="Baku",
            check_in="2026-08-10",
            check_out="2026-08-15",
            property_types=["apartment"],
            filters=SearchFilters(price=PriceConstraint(max_amount=100, currency="USD", scope="per_night")),
        )

    def no_model(*args, **kwargs):
        raise LlmCallError("configuration_error", stage="conversation_response_generation")

    monkeypatch.setattr(conversation_flow, "route_conversation_async", route)
    monkeypatch.setattr(intent_router, "route_intent_adk_async", intent)
    monkeypatch.setattr("app.logic.conversation_response_llm.build_adk_model", no_model)

    with clock.use_reference_date(date(2026, 3, 1)):
        turn = await process_turn(
            ConversationSession(),
            "apartment in Baku Aug 10-15 under $100 a night",
            config=TurnConfig(persist_telemetry=False),
        )

    card = next(c for c in turn.listing_cards if c.result_id == LISTING_ID)
    assert card.price_summary == "350.0 USD total for 5 night(s)"
    assert "630" not in turn.assistant_message
    assert "3150" not in json.dumps(turn.model_dump(mode="json"))
