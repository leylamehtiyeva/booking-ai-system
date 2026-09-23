from datetime import date

import pytest

from app.retrieval import apify
from app.retrieval.apify import ApifyRetriever
from app.schemas.listing import Facility
from app.schemas.query import SearchRequest


@pytest.mark.asyncio
async def test_apify_retriever_returns_normalized_listings(
    monkeypatch,
):
    monkeypatch.setenv(
        "APIFY_TOKEN",
        "test-token",
    )

    raw_items = [
        {
            "hotelId": 16578874,
            "name": (
                "Modern Apartment In Nizami"
            ),
            "url": (
                "https://example.com/"
                "modern-apartment"
            ),
            "type": "apartment",
            "price": 352.94,
            "currency": "US$",
            "rating": 10,
            "address": {
                "full": (
                    "63 Nizami Street, "
                    "1005 Baku, Azerbaijan"
                ),
                "city": "Baku",
            },
            "description": (
                "Spacious apartment."
            ),
            "facilities": [
                {
                    "name": "Parking",
                    "overview": (
                        "No parking available."
                    ),
                    "facilities": [],
                }
            ],
            "rooms": [
                {
                    "roomType": (
                        "Apartment with Sea View"
                    ),
                    "persons": 3,
                    "facilities": [
                        "Entire apartment",
                        "Private kitchen",
                        "Private bathroom",
                        "2153 feet²",
                    ],
                    "options": [
                        {
                            "price": 352.94,
                            "currency": "US$",
                            "persons": 3,
                            "freeCancellation": True,
                            "cancellationType": (
                                "free_cancellation"
                            ),
                            "yourChoices": [
                                "Free cancellation"
                            ],
                        }
                    ],
                }
            ],
        }
    ]

    def fake_post_json_sync(
        url,
        payload,
        timeout,
    ):
        return raw_items

    monkeypatch.setattr(
        apify,
        "_post_json_sync",
        fake_post_json_sync,
    )

    # Do not write logs/apify_raw during test.
    monkeypatch.setattr(
        apify,
        "_save_apify_debug_payload",
        lambda actor_input, items: None,
    )

    req = SearchRequest(
        city="Baku",
        check_in=date(2026, 9, 10),
        check_out=date(2026, 9, 12),
        adults=3,
        children=2,
        rooms=2,
        currency="USD",
    )

    listings = await (
        ApifyRetriever().get_candidates(
            req,
            max_items=3,
        )
    )

    assert len(listings) == 1

    listing = listings[0]

    assert listing.id == "16578874"
    assert listing.city == "Baku"

    assert listing.address == (
        "63 Nizami Street, "
        "1005 Baku, Azerbaijan"
    )

    assert (
        listing.property_type
        == "apartment"
    )

    assert len(listing.rooms) == 1

    room = listing.rooms[0]

    assert (
        room.name
        == "Apartment with Sea View"
    )

    assert (
        room.options[0].choices
        == ["Free cancellation"]
    )

    parking = listing.facilities[0]

    assert isinstance(
        parking,
        Facility,
    )

    assert (
        parking.overview
        == "No parking available."
    )

    assert listing.raw == raw_items[0]