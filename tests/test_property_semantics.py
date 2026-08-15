from app.logic.property_semantics import (
    detect_occupancy_type,
    detect_property_type,
    match_occupancy_types,
    match_property_types,
)
from app.schemas.listing import ListingRaw
from app.schemas.match import Ternary
from app.schemas.property_semantics import OccupancyType, PropertyType

from app.schemas.listing import (
    Facility,
    ListingRaw,
    Room,
)

def test_detect_occupancy_type_from_canonical_room_facility():
    listing = ListingRaw(
        id="occupancy-1",
        name="Nice stay",
        rooms=[
            Room(
                name="Apartment",
                facilities=[
                    Facility(
                        name="Entire apartment"
                    )
                ],
            )
        ],
    )

    detected, evidence = (
        detect_occupancy_type(listing)
    )

    assert (
        detected
        == OccupancyType.ENTIRE_PLACE
    )

    assert evidence

    assert (
        "facilities"
        in evidence[0].path
    )


def test_detect_property_type_prefers_canonical_field():
    listing = ListingRaw(
        id="canonical-1",
        name="Hotel-looking accommodation",
        property_type="apartment",
        rooms=[],
    )

    detected, evidence = (
        detect_property_type(listing)
    )

    assert (
        detected
        == PropertyType.APARTMENT
    )

    assert evidence

    assert (
        evidence[0].path
        == "listing.property_type"
    )

def test_detect_property_type_apartment():
    listing = ListingRaw(
        id="p1",
        name="Beautiful apartment in Baku",
        rooms=[],
    )

    detected, evidence = detect_property_type(listing)

    assert detected == PropertyType.APARTMENT
    assert evidence


def test_detect_occupancy_type_entire_place():
    listing = ListingRaw(
        id="o1",
        name="Nice stay",
        description="Entire apartment with balcony and kitchen.",
        rooms=[],
    )

    detected, evidence = detect_occupancy_type(listing)

    assert detected == OccupancyType.ENTIRE_PLACE
    assert evidence


def test_match_property_type_yes():
    listing = ListingRaw(
        id="m1",
        name="Modern apartment in city center",
        rooms=[],
    )

    out = match_property_types(listing, [PropertyType.APARTMENT])

    assert out is not None
    assert out.value == Ternary.YES


def test_match_property_type_no():
    listing = ListingRaw(
        id="m2",
        name="Comfort hotel in Baku",
        rooms=[],
    )

    out = match_property_types(listing, [PropertyType.APARTMENT])

    assert out is not None
    assert out.value == Ternary.NO


def test_match_occupancy_type_yes():
    listing = ListingRaw(
        id="m3",
        description="Private room with shared kitchen.",
        rooms=[],
    )

    out = match_occupancy_types(listing, [OccupancyType.PRIVATE_ROOM])

    assert out is not None
    assert out.value == Ternary.YES


def test_match_occupancy_type_uncertain():
    listing = ListingRaw(
        id="m4",
        name="Nice accommodation",
        rooms=[],
    )

    out = match_occupancy_types(listing, [OccupancyType.ENTIRE_PLACE])

    assert out is not None
    assert out.value == Ternary.UNCERTAIN