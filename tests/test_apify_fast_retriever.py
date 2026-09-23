from datetime import date
from urllib.error import URLError

import pytest

from app.observability.trace import RequestTrace
from app.retrieval import apify_fast
from app.retrieval.apify_fast import (
    ApifyFastRetriever,
    build_fast_search_input,
)
from app.schemas.query import SearchRequest


def make_request(
    **overrides,
) -> SearchRequest:
    data = {
        "city": "Baku",
        "check_in": date(
            2026,
            9,
            15,
        ),
        "check_out": date(
            2026,
            9,
            18,
        ),
        "adults": 2,
        "children": 0,
        "rooms": 1,
        "currency": "USD",
        "constraints": [],
    }

    data.update(overrides)

    return SearchRequest.model_validate(
        data
    )


def test_build_fast_search_input():
    req = make_request(
        adults=3,
        children=1,
        rooms=2,
        property_types=[
            "apartment"
        ],
    )

    actor_input = (
        build_fast_search_input(
            req,
            max_items=20,
            language="en-gb",
        )
    )

    assert actor_input == {
        "search": "Baku",
        "maxItems": 20,
        "currency": "USD",
        "language": "en-gb",
        "checkIn": "2026-09-15",
        "checkOut": "2026-09-18",
        "adults": 3,
        "children": 1,
        "rooms": 2,
        "propertyType": "Apartments",
    }


def test_multiple_property_types_are_not_pushed_as_one_type():
    req = make_request(
        property_types=[
            "hotel",
            "apartment",
        ],
    )

    actor_input = (
        build_fast_search_input(
            req,
            max_items=20,
        )
    )

    assert (
        "propertyType"
        not in actor_input
    )


def test_unsupported_property_type_is_not_pushed():
    req = make_request(
        property_types=[
            "ryokan"
        ],
    )

    actor_input = (
        build_fast_search_input(
            req,
            max_items=20,
        )
    )

    assert (
        "propertyType"
        not in actor_input
    )


@pytest.mark.asyncio
async def test_fast_retriever_calls_actor_and_normalizes_results(
    monkeypatch,
):
    monkeypatch.setenv(
        "APIFY_TOKEN",
        "test-token",
    )

    monkeypatch.delenv(
        "APIFY_FAST_COST_PER_RESULT_USD",
        raising=False,
    )

    captured = {}

    def fake_post_json_sync(
        url,
        payload,
        timeout,
    ):
        captured["url"] = url
        captured["payload"] = payload
        captured["timeout"] = timeout

        return [
            {
                "name": "Apartment One",
                "url": (
                    "https://example.com/one"
                ),
                "price": 100,
                "currency": "USD",
                "rating": 8.5,
                "roomType": (
                    "One-Bedroom Apartment"
                ),
                "persons": 2,
                "address": (
                    "Baku, Azerbaijan"
                ),
            },
            {
                "name": "Apartment Two",
                "url": (
                    "https://example.com/two"
                ),
                "price": 120,
                "currency": "USD",
                "rating": 9.0,
                "roomType": "Studio",
                "persons": 3,
                "address": (
                    "Baku, Azerbaijan"
                ),
            },
        ]

    monkeypatch.setattr(
        apify_fast,
        "_post_json_sync",
        fake_post_json_sync,
    )

    trace = RequestTrace()

    retriever = ApifyFastRetriever()

    listings = await retriever.get_candidates(
        make_request(),
        max_items=10,
        trace=trace,
    )

    assert len(listings) == 2

    assert (
        listings[0].name
        == "Apartment One"
    )

    assert (
        listings[0].max_occupancy
        == 2
    )

    assert (
        listings[0].address
        == "Baku, Azerbaijan"
    )

    assert (
        listings[0].city
        is None
    )

    assert (
        "/voyager~fast-booking-scraper/"
        in captured["url"]
    )

    assert (
        "run-sync-get-dataset-items"
        in captured["url"]
    )

    assert (
        "token=test-token"
        in captured["url"]
    )

    assert (
        captured["payload"]["search"]
        == "Baku"
    )

    assert (
        captured["payload"]["maxItems"]
        == 10
    )

    assert captured["timeout"] == 180

    assert len(
        trace.external_calls
    ) == 1

    external_call = (
        trace.external_calls[0]
    )

    assert (
        external_call.step
        == "apify_booking_fast_search"
    )

    assert external_call.success is True

    assert (
        external_call.metadata[
            "retrieval_mode"
        ]
        == "fast"
    )

    assert (
        external_call.metadata[
            "returned_items"
        ]
        == 2
    )

    assert (
        external_call.estimated_cost_usd
        == pytest.approx(0.004)
    )


@pytest.mark.asyncio
async def test_fast_retriever_records_failed_external_call(
    monkeypatch,
):
    monkeypatch.setenv(
        "APIFY_TOKEN",
        "test-token",
    )

    def fake_post_json_sync(
        url,
        payload,
        timeout,
    ):
        raise URLError(
            "provider unavailable"
        )

    monkeypatch.setattr(
        apify_fast,
        "_post_json_sync",
        fake_post_json_sync,
    )

    trace = RequestTrace()

    retriever = ApifyFastRetriever()

    with pytest.raises(
        RuntimeError,
        match="Apify URLError",
    ):
        await retriever.get_candidates(
            make_request(),
            max_items=10,
            trace=trace,
        )

    assert len(
        trace.external_calls
    ) == 1

    external_call = (
        trace.external_calls[0]
    )

    assert external_call.success is False

    assert (
        external_call.step
        == "apify_booking_fast_search"
    )

    assert (
        "provider unavailable"
        in external_call.error
    )
    
    
def test_per_night_max_price_is_pushed_down():
    req = make_request(
        filters={
            "price": {
                "max_amount": 80,
                "currency": "USD",
                "scope": "per_night",
            }
        }
    )

    actor_input = build_fast_search_input(
        req,
        max_items=20,
    )

    assert (
        actor_input["minMaxPrice"]
        == "0-80"
    )

    assert (
        actor_input["currency"]
        == "USD"
    )


def test_per_night_price_range_is_pushed_down():
    req = make_request(
        filters={
            "price": {
                "min_amount": 50,
                "max_amount": 120,
                "currency": "EUR",
                "scope": "per_night",
            }
        }
    )

    actor_input = build_fast_search_input(
        req,
        max_items=20,
    )

    assert (
        actor_input["minMaxPrice"]
        == "50-120"
    )

    assert (
        actor_input["currency"]
        == "EUR"
    )


def test_per_night_min_price_is_pushed_down():
    req = make_request(
        filters={
            "price": {
                "min_amount": 100,
                "currency": "USD",
                "scope": "per_night",
            }
        }
    )

    actor_input = build_fast_search_input(
        req,
        max_items=20,
    )

    assert (
        actor_input["minMaxPrice"]
        == "100+"
    )


def test_total_stay_price_is_not_pushed_down():
    req = make_request(
        filters={
            "price": {
                "max_amount": 500,
                "currency": "USD",
                "scope": "total_stay",
            }
        }
    )

    actor_input = build_fast_search_input(
        req,
        max_items=20,
    )

    assert (
        "minMaxPrice"
        not in actor_input
    )


def test_price_with_unspecified_scope_is_not_pushed_down():
    req = make_request(
        filters={
            "price": {
                "max_amount": 500,
                "currency": "USD",
                "scope": None,
            }
        }
    )

    actor_input = build_fast_search_input(
        req,
        max_items=20,
    )

    assert (
        "minMaxPrice"
        not in actor_input
    )


def test_per_night_price_currency_overrides_display_currency():
    req = make_request(
        currency="USD",
        filters={
            "price": {
                "max_amount": 80,
                "currency": "EUR",
                "scope": "per_night",
            }
        },
    )

    actor_input = build_fast_search_input(
        req,
        max_items=20,
    )

    assert (
        actor_input["currency"]
        == "EUR"
    )

    assert (
        actor_input["minMaxPrice"]
        == "0-80"
    )


def test_min_guest_rating_is_pushed_down():
    req = make_request(
        min_guest_rating=8.5,
    )

    actor_input = build_fast_search_input(
        req,
        max_items=20,
    )

    assert (
        actor_input["minScore"]
        == "8.5"
    )


def test_missing_rating_is_not_pushed_down():
    req = make_request(
        min_guest_rating=None,
    )

    actor_input = build_fast_search_input(
        req,
        max_items=20,
    )

    assert (
        "minScore"
        not in actor_input
    )


def test_unrelated_numeric_filters_are_not_pushed_down():
    req = make_request(
        filters={
            "bedrooms_min": 2,
            "bathrooms_min": 2,
            "area_sqm_min": 80,
        }
    )

    actor_input = build_fast_search_input(
        req,
        max_items=20,
    )

    assert (
        "bedrooms_min"
        not in actor_input
    )

    assert (
        "bathrooms_min"
        not in actor_input
    )

    assert (
        "area_sqm_min"
        not in actor_input
    )

    assert (
        "minMaxPrice"
        not in actor_input
    )