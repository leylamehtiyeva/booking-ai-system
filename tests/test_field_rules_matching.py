from app.logic.field_rules import FIELD_RULES
from app.logic.listing_signals import collect_listing_signals, find_best_signal_match
from app.schemas.fields import Field
from app.schemas.listing import ListingRaw, Room, RoomOption


def test_find_best_signal_match_prefers_room_facilities_for_kitchen():
    listing = ListingRaw(
        id="x1",
        name="Apartment",
        description="Nice place with a shared kitchen downstairs",
        facilities=[{"name": "Kitchen"}],
        rooms=[
            Room(
                name="Room 1",
                facilities=["Private kitchen", "Private bathroom"],
            )
        ],
    )

    signals = collect_listing_signals(listing)
    rule = FIELD_RULES[Field.KITCHEN]

    best = find_best_signal_match(
        signals=signals,
        aliases=rule.aliases,
        preferred_path_prefixes=rule.preferred_path_prefixes,
    )

    assert best is not None
    assert best.path == "rooms[0].facilities"
    assert best.raw_text == "Private kitchen"


def test_find_best_signal_match_for_free_cancellation():
    listing = ListingRaw(
        id="x2",
        name="Stay",
        rooms=[
            Room(
                name="Room",
                options=[
                    RoomOption(
                        name="Flexible rate",
                        choices=[
                            "Free cancellation",
                            "No prepayment needed",
                        ],
                    )
                ],
            )
        ],
    )

    signals = collect_listing_signals(listing)
    rule = FIELD_RULES[Field.FREE_CANCELLATION]

    best = find_best_signal_match(
        signals=signals,
        aliases=rule.aliases,
        preferred_path_prefixes=rule.preferred_path_prefixes,
    )

    assert best is not None
    assert best.path == "rooms[0].options[0].choices"
    assert best.raw_text == "Free cancellation"


