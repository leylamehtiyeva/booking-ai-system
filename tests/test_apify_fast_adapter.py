from app.retrieval.apify_fast import (
    normalize_fast_listing,
)


def test_normalize_fast_listing_maps_provider_fields():
    raw_item = {
        "name": "Central Baku Apartment",
        "url": "https://www.booking.com/example",
        "price": 240.5,
        "currency": "US$",
        "rating": 8.7,
        "stars": 4,
        "roomType": "One-Bedroom Apartment",
        "persons": 3,
        "address": "Baku, Azerbaijan",
        "checkIn": "2026-09-15",
        "checkOut": "2026-09-18",
        "timeOfScrapeISO": (
            "2026-08-11T10:00:00Z"
        ),
    }

    listing = normalize_fast_listing(
        raw_item
    )

    # Search scope is not treated as
    # verified listing location.
    assert listing.city is None

    assert (
        listing.address
        == "Baku, Azerbaijan"
    )

    assert (
        listing.name
        == "Central Baku Apartment"
    )

    assert (
        listing.url
        == "https://www.booking.com/example"
    )

    assert listing.price == 240.5
    assert listing.currency == "US$"

    assert listing.rating == 8.7
    assert listing.stars == 4

    assert (
        listing.room_type
        == "One-Bedroom Apartment"
    )

    assert listing.max_occupancy == 3

    assert listing.raw == raw_item


def test_normalize_fast_listing_does_not_invent_rich_evidence():
    raw_item = {
        "name": "Fast Result",
        "url": (
            "https://www.booking.com/"
            "fast-result"
        ),
        "price": 100,
        "currency": "USD",
        "roomType": "Double Room",
        "persons": 2,
    }

    listing = normalize_fast_listing(
        raw_item
    )

    assert listing.description is None
    assert listing.facilities == []
    assert listing.rooms == []


def test_normalize_fast_listing_parses_numeric_strings():
    raw_item = {
        "name": "Fast Result",
        "price": "1,234.50",
        "rating": "9.1",
        "stars": "5",
        "persons": "4",
    }

    listing = normalize_fast_listing(
        raw_item
    )

    assert listing.price == 1234.50
    assert listing.rating == 9.1
    assert listing.stars == 5
    assert listing.max_occupancy == 4


def test_normalize_fast_listing_invalid_numeric_values_become_unknown():
    raw_item = {
        "name": "Fast Result",
        "price": "unknown",
        "rating": "not-rated",
        "stars": "",
        "persons": "unknown",
    }

    listing = normalize_fast_listing(
        raw_item
    )

    assert listing.price is None
    assert listing.rating is None
    assert listing.stars is None
    assert listing.max_occupancy is None


def test_normalize_fast_listing_preserves_original_payload():
    raw_item = {
        "name": "Fast Result",
        "address": (
            "Some provider-specific address"
        ),
        "checkIn": "2026-09-15",
        "checkOut": "2026-09-18",
        "customProviderField": {
            "some": "value",
        },
    }

    listing = normalize_fast_listing(
        raw_item
    )

    assert listing.raw == raw_item

    assert (
        listing.raw[
            "customProviderField"
        ]["some"]
        == "value"
    )