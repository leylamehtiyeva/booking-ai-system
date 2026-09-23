from app.retrieval.apify import (
    normalize_apify_listing,
)
from app.schemas.listing import Facility


def test_normalize_apify_listing_core_fields():
    raw = {
        "hotelId": 16578874,
        "name": "Modern Apartment In Nizami",
        "url": "https://example.com/listing",
        "type": "apartment",
        "price": 352.94,
        "currency": "US$",
        "rating": 10,
        "stars": None,
        "description": "Spacious apartment.",
        "address": {
            "full": (
                "63 Nizami Street, "
                "1005 Baku, Azerbaijan"
            ),
            "country": "az",
            "city": "Baku",
        },
    }

    listing = normalize_apify_listing(
        raw
    )

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

    assert (
        listing.name
        == "Modern Apartment In Nizami"
    )

    assert listing.price == 352.94
    assert listing.currency == "US$"
    assert listing.rating == 10

    assert listing.raw == raw


def test_normalize_apify_preserves_negative_facility_overview():
    raw = {
        "name": "Test apartment",
        "facilities": [
            {
                "name": "Parking",
                "overview": (
                    "No parking available."
                ),
                "facilities": [],
            }
        ],
    }

    listing = normalize_apify_listing(
        raw
    )

    assert len(listing.facilities) == 1

    parking = listing.facilities[0]

    assert isinstance(
        parking,
        Facility,
    )

    assert parking.name == "Parking"
    assert parking.category == "Parking"

    assert (
        parking.overview
        == "No parking available."
    )


def test_normalize_apify_room_evidence():
    raw = {
        "name": "Test apartment",
        "rooms": [
            {
                "available": True,
                "roomType": (
                    "Apartment with Sea View"
                ),
                "persons": 3,
                "bedTypes": [
                    {
                        "room": "Bedroom 1",
                        "beds": [
                            "1 large double bed"
                        ],
                    }
                ],
                "facilities": [
                    "Entire apartment",
                    "Private kitchen",
                    "Private bathroom",
                    "Free WiFi",
                ],
                "options": [
                    {
                        "price": 130.58,
                        "currency": "US$",
                        "persons": 3,
                        "cancellationType": (
                            "free_cancellation"
                        ),
                        "freeCancellation": True,
                        "yourChoices": [
                            (
                                "Free cancellation "
                                "before arrival"
                            )
                        ],
                    }
                ],
            }
        ],
    }

    listing = normalize_apify_listing(
        raw
    )

    assert len(listing.rooms) == 1

    room = listing.rooms[0]

    assert (
        room.name
        == "Apartment with Sea View"
    )
    assert room.available is True
    assert room.persons == 3

    assert (
        room.bed_types[0].room
        == "Bedroom 1"
    )

    assert room.bed_types[0].beds == [
        "1 large double bed"
    ]

    facility_names = [
        facility.name
        for facility in room.facilities
        if isinstance(
            facility,
            Facility,
        )
    ]

    assert "Private kitchen" in (
        facility_names
    )
    assert "Private bathroom" in (
        facility_names
    )

    option = room.options[0]

    assert option.persons == 3
    assert option.price == 130.58

    assert (
        option.cancellation_type
        == "free_cancellation"
    )

    assert (
        option.free_cancellation
        is True
    )

    assert option.choices == [
        "Free cancellation before arrival"
    ]
    
    
def test_normalize_apify_preserves_textual_evidence():
    raw = {
        "name": "Test apartment",
        "finePrint": (
            "This property will not "
            "accommodate parties."
        ),
        "policies": [
            {
                "title": "Pets",
                "content": (
                    "Pets are allowed on request."
                ),
            },
            {
                "title": "Smoking",
                "content": (
                    "Smoking is not allowed."
                ),
            },
        ],
        "highlights": [
            {
                "header": "Apartments with:",
                "contents": [
                    "Sea view",
                    "City view",
                ],
            }
        ],
    }

    listing = normalize_apify_listing(raw)

    assert listing.fine_print == (
        "This property will not "
        "accommodate parties."
    )

    assert len(listing.policies) == 2

    assert (
        listing.policies[0].title
        == "Pets"
    )

    assert (
        listing.policies[0].content
        == "Pets are allowed on request."
    )

    assert len(listing.highlights) == 1

    assert (
        listing.highlights[0].header
        == "Apartments with:"
    )

    assert listing.highlights[0].contents == [
        "Sea view",
        "City view",
    ]