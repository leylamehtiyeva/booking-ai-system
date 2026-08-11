from datetime import date

from app.agents.intent_router_agent import IntentRoute
from app.logic.request_resolution import resolve_required_search_context


def test_missing_city_and_dates_requires_clarification():
    intent = IntentRoute(
        city=None,
        check_in=None,
        check_out=None,
        nights=None,
        constraints=[],
        filters={},
        property_types=[],
        occupancy_types=[],
    )

    resolved = resolve_required_search_context(intent)

    assert resolved.need_clarification is True
    assert any(
        "city" in question.lower()
        for question in resolved.questions
    )
    assert any(
        "date" in question.lower()
        or "travel dates" in question.lower()
        for question in resolved.questions
    )


def test_single_date_with_one_night_resolves_checkout():
    intent = IntentRoute(
        city="Baku",
        check_in="2026-04-20",
        check_out=None,
        nights=1,
        constraints=[],
        filters={},
        property_types=[],
        occupancy_types=[],
    )

    resolved = resolve_required_search_context(intent)

    assert resolved.need_clarification is False
    assert resolved.check_in == date(2026, 4, 20)
    assert resolved.check_out == date(2026, 4, 21)


def test_checkin_and_nights_resolve_checkout():
    intent = IntentRoute(
        city="Baku",
        check_in="2026-04-20",
        check_out=None,
        nights=6,
        constraints=[],
        filters={},
        property_types=[],
        occupancy_types=[],
    )

    resolved = resolve_required_search_context(intent)

    assert resolved.need_clarification is False
    assert resolved.check_in == date(2026, 4, 20)
    assert resolved.check_out == date(2026, 4, 26)