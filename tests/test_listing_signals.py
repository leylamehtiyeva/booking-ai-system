from app.logic.listing_signals import collect_listing_signals
from app.schemas.listing import (
    BedGroup,
    Facility,
    Highlight,
    ListingRaw,
    Policy,
    Room,
    RoomOption,
)


def test_collect_listing_signals_from_listing_and_rooms():
    listing = ListingRaw(
        id="x1",
        name="CHINAR Apartment",
        property_type="apartment",
        description=(
            "Spacious apartment in Baku center"
        ),
        facilities=[
            Facility(name="Free WiFi"),
            Facility(
                name="Air conditioning"
            ),
        ],
        rooms=[
            Room(
                name=(
                    "Three-Bedroom Apartment "
                    "with View"
                ),
                facilities=[
                    Facility(
                        name="Private kitchen"
                    ),
                    Facility(
                        name="Private bathroom"
                    ),
                ],
                bed_types=[
                    BedGroup(
                        room="Bedroom 1",
                        beds=[
                            "3 single beds",
                            "1 sofa bed",
                        ],
                    )
                ],
                options=[
                    RoomOption(
                        name="Deluxe rate",
                        choices=[
                            "Free cancellation",
                            (
                                "No prepayment "
                                "needed"
                            ),
                        ],
                    )
                ],
            )
        ],
    )

    signals = collect_listing_signals(
        listing
    )

    texts = [
        signal.text
        for signal in signals
    ]

    paths = [
        signal.path
        for signal in signals
    ]

    assert "chinar apartment" in texts
    assert "apartment" in texts

    assert (
        "spacious apartment in baku center"
        in texts
    )

    assert "free wifi" in texts
    assert "air conditioning" in texts

    assert (
        "three-bedroom apartment with view"
        in texts
    )

    assert "private kitchen" in texts
    assert "private bathroom" in texts
    assert "deluxe rate" in texts
    assert "free cancellation" in texts

    assert (
        "no prepayment needed"
        in texts
    )

    assert "3 single beds" in texts

    assert "listing.name" in paths
    assert "listing.property_type" in paths

    assert any(
        path.startswith(
            "listing.facilities"
        )
        for path in paths
    )

    assert "rooms[0].name" in paths

    assert any(
        path.startswith(
            "rooms[0].bed_types"
        )
        for path in paths
    )

    assert any(
        path.startswith(
            "rooms[0].facilities"
        )
        for path in paths
    )

    assert (
        "rooms[0].options[0].choices"
        in paths
    )


def test_collect_listing_signals_from_highlights_and_policies():
    listing = ListingRaw(
        id="x2",
        name="Nice stay",
        highlights=[
            Highlight(
                header="Great for your stay",
                contents=[
                    "Balcony",
                    "City view",
                ],
            )
        ],
        policies=[
            Policy(
                title="Pets",
                content=(
                    "Pets are allowed "
                    "on request."
                ),
            ),
            Policy(
                title="Smoking",
                content=(
                    "Non-smoking throughout."
                ),
            ),
        ],
    )

    signals = collect_listing_signals(
        listing
    )

    texts = [
        signal.text
        for signal in signals
    ]

    assert "great for your stay" in texts
    assert "balcony" in texts
    assert "city view" in texts
    assert "pets" in texts

    assert (
        "pets are allowed on request."
        in texts
    )

    assert "smoking" in texts

    assert (
        "non-smoking throughout."
        in texts
    )
    
    
def test_collect_listing_signals_preserves_facility_overview():
    listing = ListingRaw(
        id="parking-1",
        facilities=[
            Facility(
                name="Parking",
                category="Parking",
                overview=(
                    "No parking available."
                ),
            )
        ],
    )

    signals = collect_listing_signals(
        listing
    )

    texts = [
        signal.text
        for signal in signals
    ]

    assert "parking" in texts

    assert (
        "no parking available."
        in texts
    )
    
