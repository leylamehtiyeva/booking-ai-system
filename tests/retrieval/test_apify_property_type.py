from app.schemas.property_semantics import PropertyType
from app.retrieval.apify import (
    ApifyRetriever,
    _apify_property_type,
    _apify_property_types,
    _split_max_items,
    _deduplicate_listings,
)
from app.retrieval.apify import _save_apify_debug_payload
import asyncio
from app.retrieval.apify import ApifyRetriever
from app.schemas.query import SearchRequest
import pytest
from app.schemas.listing import ListingRaw
def test_apify_debug_payloads_do_not_overwrite(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)

    actor_input = {
        "search": "Baku",
        "propertyType": "Apartments",
    }

    items = []

    _save_apify_debug_payload(
        actor_input,
        items,
    )

    _save_apify_debug_payload(
        actor_input,
        items,
    )

    files = list(
        (tmp_path / "logs" / "apify_raw")
        .glob("apify_raw_*.json")
    )

    assert len(files) == 2

def test_deduplicate_listings_by_url_when_id_missing():
    listings = [
        ListingRaw(
            name="Apartment A",
            url="https://example.com/a",
        ),
        ListingRaw(
            name="Apartment A duplicate",
            url="https://example.com/a",
        ),
    ]

    result = _deduplicate_listings(
        listings
    )

    assert len(result) == 1

def test_deduplicate_listings_by_id():
    listings = [
        ListingRaw(
            id="123",
            name="Apartment A",
        ),
        ListingRaw(
            id="123",
            name="Apartment A duplicate",
        ),
        ListingRaw(
            id="456",
            name="Villa B",
        ),
    ]

    result = _deduplicate_listings(
        listings
    )

    assert len(result) == 2

    assert [
        listing.id
        for listing in result
    ] == [
        "123",
        "456",
    ]

@pytest.mark.asyncio
async def test_multiple_property_types_run_in_parallel(
    monkeypatch,
):
    active_calls = 0
    max_active_calls = 0

    async def fake_get_candidates_for_property_type(
        self,
        req,
        property_type,
        max_items,
        trace=None,
    ):
        nonlocal active_calls
        nonlocal max_active_calls

        active_calls += 1

        max_active_calls = max(
            max_active_calls,
            active_calls,
        )

        await asyncio.sleep(0.01)

        active_calls -= 1

        return []

    monkeypatch.setattr(
        ApifyRetriever,
        "_get_candidates_for_property_type",
        fake_get_candidates_for_property_type,
    )

    req = SearchRequest(
        city="Baku",
        property_types=[
            PropertyType.APARTMENT,
            PropertyType.VILLA,
        ],
    )

    retriever = ApifyRetriever()

    await retriever.get_candidates(
        req,
        max_items=20,
    )

    assert max_active_calls == 2




def test_split_max_items_two_property_types():
    assert _split_max_items(
        20,
        2,
    ) == [10, 10]


def test_split_max_items_three_property_types():
    assert _split_max_items(
        20,
        3,
    ) == [7, 7, 6]


def test_split_max_items_uneven():
    assert _split_max_items(
        5,
        2,
    ) == [3, 2]


def test_split_max_items_no_property_types():
    assert _split_max_items(
        20,
        0,
    ) == []

def test_apify_property_type_apartment():
    assert (
        _apify_property_type(
            PropertyType.APARTMENT
        )
        == "Apartments"
    )


def test_apify_property_type_villa():
    assert (
        _apify_property_type(
            PropertyType.VILLA
        )
        == "Villas"
    )


def test_apify_property_type_unsupported():
    assert (
        _apify_property_type(
            PropertyType.RYOKAN
        )
        == "none"
    )


def test_apify_property_type_missing():
    assert (
        _apify_property_type(None)
        == "none"
    )
    
    
def test_apify_property_types_multiple():
    assert _apify_property_types(
        [
            PropertyType.APARTMENT,
            PropertyType.VILLA,
        ]
    ) == [
        "Apartments",
        "Villas",
    ]


def test_apify_property_types_with_unsupported():
    assert _apify_property_types(
        [
            PropertyType.APARTMENT,
            PropertyType.RYOKAN,
        ]
    ) == [
        "Apartments",
        "none",
    ]


def test_apify_property_types_missing():
    assert (
        _apify_property_types(None)
        == ["none"]
    )


def test_apify_property_types_deduplicates():
    assert _apify_property_types(
        [
            PropertyType.APARTMENT,
            PropertyType.APARTMENT,
        ]
    ) == [
        "Apartments"
    ]