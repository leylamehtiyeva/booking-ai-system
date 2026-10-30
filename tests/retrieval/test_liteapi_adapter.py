"""
LiteAPI adapter: offers -> PublicOffer, hotel content -> ListingRaw, retriever.

All payloads are synthetic, written for these tests in the shape of LiteAPI
responses - no provider data is stored in the repository. Internal amounts
use distinctive sentinel values so a leak into any serialized output is
caught by a plain substring check.
"""

from __future__ import annotations

import json
from datetime import date

import httpx
import pytest
from pydantic import ValidationError

from app.logic.constraint_evidence_resolution import _prepare_listing_evidence, _snippet_found
from app.logic.listing_signals import collect_listing_signals
from app.logic.matcher_structured import match_listing_structured
from app.logic.numeric_filters import extract_area_candidates, extract_total_price
from app.retrieval.errors import RetrievalError
from app.retrieval.liteapi import (
    LiteApiRetriever,
    clean_html_text,
    normalize_liteapi_listing,
    parse_public_offer,
    parse_rates_response,
)
from app.retrieval.liteapi_client import LiteApiClient
from app.schemas.constraints import (
    ConstraintCategory,
    ConstraintMappingStatus,
    ConstraintPriority,
    EvidenceStrategy,
    UserConstraint,
)
from app.schemas.fields import Field
from app.schemas.public_offer import PublicOffer
from app.schemas.query import SearchRequest
from app.schemas.shown_result_set import ShownResult, ShownResultSet

SSP = 450.0
# Internal amounts - none of them may appear in any output.
RETAIL_TOTAL = 391.37
INITIAL_PRICE = 512.73
COMMISSION = 23.71
PROVIDER_COMMISSION = 17.29
TAX_AMOUNT = 41.53
CITY_TAX_AMOUNT = 13.67
PENALTY_AMOUNT = 287.19
CONFIDENTIAL = [RETAIL_TOTAL, INITIAL_PRICE, COMMISSION, PROVIDER_COMMISSION, TAX_AMOUNT, CITY_TAX_AMOUNT, PENALTY_AMOUNT]


def _money(amount: float, currency: str = "USD") -> dict:
    return {"amount": amount, "currency": currency}


def make_rate(
    *,
    name: str = "Double Room",
    ssp: float | None = SSP,
    ssp_currency: str = "USD",
    board_type: str = "RO",
    board_name: str = "Room Only",
    refundable_tag: str = "RFN",
) -> dict:
    retail = {
        "total": [_money(RETAIL_TOTAL)],
        "initialPrice": [_money(INITIAL_PRICE)],
        "taxesAndFees": [
            {"included": True, "description": "VAT", **_money(TAX_AMOUNT)},
            {"included": False, "description": "City tax", **_money(CITY_TAX_AMOUNT)},
        ],
    }
    if ssp is not None:
        retail["suggestedSellingPrice"] = [{**_money(ssp, ssp_currency), "source": "providerDirect"}]
    return {
        "rateId": "rate-1",
        "name": name,
        "maxOccupancy": 2,
        "adultCount": 2,
        "boardType": board_type,
        "boardName": board_name,
        "priceType": "commission",
        "commission": [_money(COMMISSION)],
        "providerCommission": _money(PROVIDER_COMMISSION),
        "retailRate": retail,
        "cancellationPolicies": {
            "cancelPolicyInfos": [
                {"cancelTime": "2026-11-20 10:00:00", "timezone": "GMT", "type": "amount", **_money(PENALTY_AMOUNT)}
            ],
            "refundableTag": refundable_tag,
        },
        "paymentTypes": ["NUITEE_PAY"],
    }


def make_offer(*, ssp: float | None = SSP, ssp_currency: str = "USD", rates: list[dict] | None = None, **rate_kwargs) -> dict:
    offer = {
        "roomTypeId": "rt-1",
        "offerId": "opaque-offer-id",
        "rates": rates if rates is not None else [make_rate(ssp=ssp, ssp_currency=ssp_currency, **rate_kwargs)],
        "offerRetailRate": _money(RETAIL_TOTAL),
        "offerInitialPrice": _money(INITIAL_PRICE),
        "priceType": "commission",
    }
    if ssp is not None:
        offer["suggestedSellingPrice"] = {**_money(ssp, ssp_currency), "source": "providerDirect"}
    return offer


def make_hotel(hotel_id: str = "lp-test-1", **overrides) -> dict:
    hotel = {
        "id": hotel_id,
        "name": "Test Harbour Hotel",
        "hotelDescription": (
            "<p><strong>Rooftop Pool &amp; Views</strong><br>The hotel offers an outdoor pool "
            "with city views.</p><p>Guests can work from a quiet co-working lounge.</p>"
        ),
        "hotelImportantInformation": "<p>Guests must show a photo ID at check-in.</p>",
        "hotelType": "Hotels",
        "hotelTypeId": 204,
        "starRating": 4,
        "rating": 9.1,
        "reviewCount": 1234,
        "city": "Barcelona",
        "country": "es",
        "address": "Example Street 1",
        "location": {"latitude": 41.39, "longitude": 2.17},
        "facilities": [
            {"facilityId": 1, "name": "Free WiFi"},
            {"facilityId": 2, "name": "Swimming pool"},
            {"facilityId": 3, "name": "Breakfast takeaway containers"},
        ],
        "hotelFacilities": ["Free WiFi", "Swimming pool", "Breakfast takeaway containers"],
        "rooms": [
            {
                "roomName": "Double Room",
                "description": "<p>183 sq feet</p>",
                "roomSizeSquare": 17,
                "roomSizeUnit": "m2",
                "maxOccupancy": 2,
                "bedTypes": [{"quantity": 1, "bedType": "Double bed", "bedSize": "131-150 cm wide"}],
                "roomAmenities": [{"amenitiesId": 1, "name": "Desk"}, {"amenitiesId": 2, "name": "Shower"}],
                "views": [{"view": "Sea View", "id": 10}],
                "attributes": {"laptopFriendlyWorkplace": True, "kitchen": None, "airConditioner": False},
            },
            {
                "roomName": "Family Suite",
                "roomSizeSquare": 38,
                "roomSizeUnit": "sqm",
                "maxOccupancy": 4,
                "roomAmenities": [{"amenitiesId": 3, "name": "Sofa bed"}],
            },
            {"roomName": "Unknown Size Room", "roomSizeSquare": None, "roomSizeUnit": "m2", "maxOccupancy": 2},
        ],
        "checkinCheckoutTimes": {"checkin_start": "03:00 PM", "checkin_end": "", "checkout": "12:00 PM"},
        "parking": "PAID",
        "petsAllowed": False,
        "childAllowed": True,
        "policies": [],
        "policyAttributes": {
            "wifi": {"inRoom": True, "free": True},
            "parking": {"selfParking": True, "free": False, "chargeMode": "charges_apply", "evCharger": True},
            "pets": {"allowed": None},
            "children": {"allowed": True, "cribAvailable": True},
            "facilities": {"pool": True, "poolType": "outdoor", "fitnessCenter": False, "elevator": True, "sauna": None},
            "foodAndDrink": {"restaurant": True, "bar": True, "breakfastAvailable": False, "breakfastIncluded": None},
        },
        "poi": [{"name": "Old Town", "category": "district", "distanceKm": 1.5, "importance": "major"}],
        "sentiment_analysis": {
            "pros": ["Great location", "Friendly staff"],
            "cons": ["Street noise at night"],
            "categories": [{"name": "Cleanliness", "rating": 9.1, "description": "Rooms are often called spotless."}],
        },
        "sentiment_updated_at": "2026-01-07T15:51:32Z",
        "phone": "+00 000 000",
        "hotelImages": [{"url": "https://example.invalid/image.jpg"}],
    }
    hotel.update(overrides)
    return hotel


def _assert_no_confidential_amounts(serialized: str) -> None:
    for amount in CONFIDENTIAL:
        assert str(amount) not in serialized, f"internal amount {amount} leaked"


# --- PublicOffer -------------------------------------------------------------


def test_public_offer_uses_the_suggested_selling_price_only():
    offer = parse_public_offer(make_offer(), requested_currency="USD")

    assert offer.public_total == SSP
    assert offer.currency == "USD"
    assert offer.price_source == "suggested_selling_price"
    assert offer.price_basis == "total_stay"
    assert offer.board_name == "Room Only"
    assert offer.refundable is True
    assert offer.cancel_policy_times == ["2026-11-20 10:00:00 GMT"]  # kept as given
    assert offer.excluded_taxes_and_fees == ["City tax"]
    _assert_no_confidential_amounts(offer.model_dump_json())


def test_public_offer_contract_rejects_internal_price_fields():
    with pytest.raises(ValidationError):
        PublicOffer(public_total=SSP, currency="USD", retail_total=RETAIL_TOTAL)
    with pytest.raises(ValidationError):
        PublicOffer(public_total=SSP)  # a price without currency


@pytest.mark.parametrize(
    "offer",
    [
        pytest.param(make_offer(ssp=None), id="ssp-missing"),
        pytest.param(make_offer(ssp=0), id="ssp-zero"),
        pytest.param(make_offer(ssp_currency="EUR"), id="other-currency"),
        pytest.param({**make_offer(), "suggestedSellingPrice": _money(SSP + 50)}, id="offer-and-rate-ssp-disagree"),
        pytest.param({**make_offer(), "rates": []}, id="no-rates"),
    ],
)
def test_ambiguous_public_price_is_not_shown_and_never_replaced(offer):
    parsed = parse_public_offer(offer, requested_currency="USD")

    assert parsed.public_total is None
    assert parsed.currency is None

    listing = normalize_liteapi_listing(make_hotel(), [parsed])
    assert listing.price is None
    assert listing.currency is None
    assert all(option.price is None for room in listing.rooms for option in room.options)
    # Downstream price extraction finds nothing to filter on (-> UNCERTAIN), not a substitute.
    assert extract_total_price(listing)[0] is None
    _assert_no_confidential_amounts(listing.model_dump_json())


def test_mixed_refundability_and_boards_are_not_merged():
    offer = parse_public_offer(
        make_offer(
            rates=[
                make_rate(refundable_tag="RFN", board_name="Room Only"),
                make_rate(refundable_tag="NRFN", board_name="Breakfast Included"),
            ],
            ssp=SSP * 2,
        ),
        requested_currency="USD",
    )

    assert offer.public_total == SSP * 2  # offer SSP == sum of the rate SSPs
    assert offer.refundable is None  # RFN and NRFN in one offer: not stated either way
    assert offer.board_name is None
    assert offer.max_occupancy is None


def test_parse_rates_response_keeps_provider_order_and_skips_malformed_hotels():
    payload = {
        "sandbox": True,
        "data": [
            {"hotelId": "b", "roomTypes": [make_offer()]},
            {"roomTypes": [make_offer()]},
            {"hotelId": "a", "roomTypes": [make_offer(), make_offer(ssp=None)]},
            {"hotelId": "c", "roomTypes": []},
        ],
    }

    offers = parse_rates_response(payload, requested_currency="USD")

    assert list(offers) == ["b", "a"]
    assert [offer.public_total for offer in offers["a"]] == [SSP, None]


# --- normalization -----------------------------------------------------------


def test_clean_html_text_keeps_meaning_and_sentence_boundaries():
    text = clean_html_text(
        "<p><strong>Rooftop Pool &amp; Views</strong><br>The pool is open in summer</p>"
        "<script>alert(1)</script><ul><li>Free WiFi</li><li>Bar</li></ul>"
    )

    assert text == "Rooftop Pool & Views. The pool is open in summer. Free WiFi. Bar."
    assert clean_html_text("<p> </p>") is None
    assert clean_html_text(None) is None


def test_core_fields_and_provenance():
    listing = normalize_liteapi_listing(
        make_hotel(), [parse_public_offer(make_offer(), requested_currency="USD")], environment="sandbox"
    )

    assert listing.id == "lp-test-1"
    assert listing.name == "Test Harbour Hotel"
    assert listing.city == "Barcelona"
    assert listing.property_type == "hotel"
    assert listing.stars == 4
    assert listing.rating == 9.1
    assert listing.review_count == 1234
    assert (listing.latitude, listing.longitude) == (41.39, 2.17)
    assert listing.price == SSP and listing.currency == "USD"
    assert listing.url is None
    assert listing.raw is None
    assert listing.provenance.provider == "liteapi"
    assert listing.provenance.environment == "sandbox"
    assert listing.fine_print == "Guests must show a photo ID at check-in."


def test_unknown_hotel_type_is_left_unset():
    listing = normalize_liteapi_listing(make_hotel(hotelType="Something new"), [])
    assert listing.property_type is None


def test_description_quotes_survive_evidence_verification():
    listing = normalize_liteapi_listing(make_hotel(), [])

    assert "<" not in listing.description
    assert "Rooftop Pool & Views." in listing.description

    evidence = _prepare_listing_evidence(listing)
    assert _snippet_found("Guests can work from a quiet co-working lounge", evidence)
    assert _snippet_found("outdoor pool with city views", evidence)


def _offers(*room_names: str, **kwargs) -> list[PublicOffer]:
    return [parse_public_offer(make_offer(name=name, **kwargs), requested_currency="USD") for name in room_names]


def test_hotel_and_room_amenities_stay_separate():
    listing = normalize_liteapi_listing(make_hotel(), _offers("Double Room"))

    hotel_names = {facility.name for facility in listing.facilities}
    double = next(room for room in listing.rooms if room.name == "Double Room")
    room_names = {facility.name for facility in double.facilities}

    assert {"Free WiFi", "Swimming pool"} <= hotel_names
    assert {"Desk", "Shower", "Sea View", "Laptop-friendly workspace", "Room size: 17 m²"} <= room_names
    assert not hotel_names & {"Desk", "Shower", "Sea View", "Laptop-friendly workspace"}
    assert not room_names & {"Free WiFi", "Swimming pool"}
    assert double.persons == 2
    assert double.bed_types[0].beds == ["1 x Double bed (131-150 cm wide)"]


def test_room_sizes_are_unit_scoped_and_missing_sizes_skipped():
    listing = normalize_liteapi_listing(make_hotel(), _offers("Double Room", "Family Suite", "Unknown Size Room"))

    areas, _, reason = extract_area_candidates(listing)

    assert areas == [17.0, 38.0]
    assert reason is None
    unknown = next(room for room in listing.rooms if room.name == "Unknown Size Room")
    assert not any("Room size" in facility.name for facility in unknown.facilities)


def test_false_or_missing_flags_never_become_a_stated_no():
    # The offered Double Room carries airConditioner=false and kitchen=null.
    listing = normalize_liteapi_listing(make_hotel(), _offers("Double Room"))
    texts = " | ".join(signal.raw_text for signal in collect_listing_signals(listing)).lower()

    # breakfastAvailable false, petsAllowed false, fitnessCenter false, airConditioner false,
    # kitchen null: nothing is emitted, neither as a fact nor as a negative.
    assert "breakfast available" not in texts
    assert "pets" not in texts
    assert "fitness" not in texts
    assert "air conditioning" not in texts
    assert "kitchen" not in texts
    assert "no " not in texts.replace("non-", "")
    # Positive flags are kept.
    assert "outdoor swimming pool" in texts
    assert "elevator" in texts
    assert "parking charges apply" in texts
    assert "children allowed" in texts
    # A provider facility name is passed through unchanged, not reinterpreted.
    assert "breakfast takeaway containers" in texts


def test_policies_and_nearby_places():
    listing = normalize_liteapi_listing(make_hotel(), [])
    policies = {policy.title: policy.content for policy in listing.policies}

    assert policies["Check-in"] == "Check-in from 03:00 PM"
    assert policies["Check-out"] == "Check-out until 12:00 PM"
    assert policies["Parking"] == "Self parking available; Parking charges apply; EV charger available"
    assert policies["WiFi"] == "WiFi in rooms; Free WiFi"
    assert "Pets" not in policies
    assert listing.highlights[0].header == "Nearby places"
    assert listing.highlights[0].contents == ["Old Town (district): 1.5 km away"]


def test_review_sentiment_is_kept_with_origin_but_not_used_as_evidence():
    listing = normalize_liteapi_listing(make_hotel(), [])

    summary = listing.review_summary
    assert summary.source == "liteapi.sentiment_analysis"
    assert summary.updated_at == "2026-01-07T15:51:32Z"
    assert summary.pros == ["Great location", "Friendly staff"]
    assert summary.cons == ["Street noise at night"]
    assert summary.categories[0].name == "Cleanliness" and summary.categories[0].rating == 9.1

    texts = {signal.raw_text for signal in collect_listing_signals(listing)}
    assert not texts & {"Great location", "Friendly staff", "Street noise at night"}
    assert not any("spotless" in text for text in texts)


def test_no_review_summary_without_sentiment():
    listing = normalize_liteapi_listing(make_hotel(sentiment_analysis=None), [])
    assert listing.review_summary is None


def test_offers_attach_to_matching_rooms_only():
    matching = parse_public_offer(make_offer(name="Double Room", board_name="Breakfast Included"), requested_currency="USD")
    other = parse_public_offer(make_offer(name="Deluxe King", refundable_tag="NRFN", ssp=520.0), requested_currency="USD")

    listing = normalize_liteapi_listing(make_hotel(), [matching, other])
    rooms = {room.name: room for room in listing.rooms}

    double = rooms["Double Room"]
    assert double.available is True
    assert [option.price for option in double.options] == [SSP]
    assert double.options[0].choices == ["Breakfast Included"]
    assert double.options[0].cancellation_type == "Refundable rate"
    assert double.options[0].free_cancellation is None

    deluxe = rooms["Deluxe King"]
    assert deluxe.facilities == []  # offer-only room: no amenities invented
    assert deluxe.options[0].choices == ["Room Only"]
    assert deluxe.options[0].cancellation_type == "Non-refundable rate"
    assert deluxe.options[0].free_cancellation is False

    assert set(rooms) == {"Double Room", "Deluxe King"}  # catalog rooms without an offer are left out
    assert listing.price == SSP  # lowest public offer


# --- which rooms reach matching (only offered ones, unambiguous catalog links) --


def _catalog_room(name: str, *, amenities: list[str] = (), size: int | None = None, max_occupancy: int = 2) -> dict:
    return {
        "roomName": name,
        "roomSizeSquare": size,
        "roomSizeUnit": "m2",
        "maxOccupancy": max_occupancy,
        "roomAmenities": [{"amenitiesId": i, "name": a} for i, a in enumerate(amenities)],
    }


def _structured(listing, field: Field) -> str:
    constraint = UserConstraint(
        raw_text=field.value,
        normalized_text=field.value,
        priority=ConstraintPriority.MUST,
        category=ConstraintCategory.AMENITY,
        mapping_status=ConstraintMappingStatus.KNOWN,
        mapped_fields=[field],
        evidence_strategy=EvidenceStrategy.STRUCTURED,
    )
    request = SearchRequest(city="Barcelona", constraints=[constraint])
    return match_listing_structured(listing, request).matches[field].value.value


def test_an_unoffered_catalog_room_does_not_confirm_room_features():
    hotel = make_hotel(rooms=[
        _catalog_room("Standard Room", amenities=["Desk"], size=18),
        _catalog_room("Suite", amenities=["Balcony", "Desk"], size=45, max_occupancy=4),
    ])

    listing = normalize_liteapi_listing(hotel, _offers("Standard Room"))

    assert [room.name for room in listing.rooms] == ["Standard Room"]
    assert _structured(listing, Field.BALCONY) != "YES"
    assert extract_area_candidates(listing)[0] == [18.0]  # not the suite's 45 m²
    assert listing.max_occupancy == 2  # not the suite's 4


def test_an_ambiguous_room_name_gets_a_minimal_room_not_an_arbitrary_one():
    hotel = make_hotel(rooms=[
        _catalog_room("Double Room", amenities=["Balcony"], size=30),
        _catalog_room("Double Room", amenities=["Desk"], size=15),
    ])

    listing = normalize_liteapi_listing(hotel, _offers("Double Room") + _offers("Double Room", board_name="Breakfast Included"))

    [room] = listing.rooms  # one room for the name, no duplicates
    assert room.name == "Double Room"
    assert room.facilities == [] and room.bed_types == []
    assert room.persons == 2  # from the offer itself
    assert room.available is True
    assert [option.choices[0] for option in room.options] == ["Room Only", "Breakfast Included"]
    assert _structured(listing, Field.BALCONY) != "YES"
    assert extract_area_candidates(listing)[0] == []


def test_an_unmatched_offer_keeps_only_what_the_offer_states():
    listing = normalize_liteapi_listing(make_hotel(), _offers("Deluxe King"))

    [room] = listing.rooms
    assert room.name == "Deluxe King"
    assert room.facilities == [] and room.bed_types == []
    assert room.available is True
    assert room.options[0].price == SSP
    assert room.options[0].choices == ["Room Only"]
    assert room.options[0].cancellation_type == "Refundable rate"


def test_a_unique_name_match_keeps_the_catalog_description():
    listing = normalize_liteapi_listing(make_hotel(), _offers("Double Room"))

    [room] = listing.rooms
    names = {facility.name for facility in room.facilities}
    assert {"Desk", "Shower", "Sea View", "Laptop-friendly workspace", "Room size: 17 m²"} <= names
    assert room.bed_types[0].beds == ["1 x Double bed (131-150 cm wide)"]
    assert room.available is True and len(room.options) == 1
    assert _structured(listing, Field.BALCONY) != "YES"  # the Double Room has a sea view, not a balcony


def test_hotel_level_facilities_do_not_depend_on_rooms():
    for offers in ([], _offers("Deluxe King"), _offers("Double Room")):
        listing = normalize_liteapi_listing(make_hotel(), offers)
        assert {"Free WiFi", "Swimming pool"} <= {facility.name for facility in listing.facilities}
        assert _structured(listing, Field.WIFI) == "YES"


def test_every_offer_survives_normalization_without_fake_rooms():
    offers = (
        _offers("Double Room")
        + _offers("Double Room", board_name="Breakfast Included", ssp=SSP + 60)
        + _offers("Deluxe King", ssp=SSP + 120)
        + _offers("Deluxe King", refundable_tag="NRFN", ssp=SSP + 90)
    )

    listing = normalize_liteapi_listing(make_hotel(), offers)

    assert listing.public_offers == offers
    assert sum(len(room.options) for room in listing.rooms) == len(offers)
    assert [room.name for room in listing.rooms] == ["Double Room", "Deluxe King"]
    assert all(room.available is True and room.options for room in listing.rooms)
    assert sorted(option.price for room in listing.rooms for option in room.options) == sorted(
        offer.public_total for offer in offers
    )
    assert listing.price == SSP


# --- refundability: kept per offer, never free cancellation, never hotel evidence --


def _evidence_texts(listing) -> str:
    signals = " | ".join(signal.raw_text for signal in collect_listing_signals(listing))
    return (signals + " | " + json.dumps(_prepare_listing_evidence(listing))).lower()


def _assert_no_cancellation_claims(listing) -> None:
    texts = _evidence_texts(listing)
    for phrase in ("free cancellation", "refundable", "non-refundable", "cancel", "2026-11-20"):
        assert phrase not in texts, f"{phrase!r} reached listing evidence"


def test_rfn_is_a_refundable_rate_never_free_cancellation():
    [offer] = _offers("Double Room")
    listing = normalize_liteapi_listing(make_hotel(), [offer])

    assert offer.refundable is True and offer.refundability_label == "Refundable rate"
    [option] = listing.rooms[0].options
    assert option.cancellation_type == "Refundable rate"
    assert option.free_cancellation is None
    assert _structured(listing, Field.FREE_CANCELLATION) == "UNCERTAIN"
    _assert_no_cancellation_claims(listing)


def test_nrfn_is_a_non_refundable_rate_without_a_hotel_level_no():
    offers = _offers("Double Room", "Deluxe King", refundable_tag="NRFN")
    listing = normalize_liteapi_listing(make_hotel(), offers)

    assert all(offer.refundable is False and offer.refundability_label == "Non-refundable rate" for offer in offers)
    options = [option for room in listing.rooms for option in room.options]
    assert all(option.cancellation_type == "Non-refundable rate" and option.free_cancellation is False for option in options)
    # Only a few of the hotel's rates are fetched (maxRatesPerHotel): their
    # being non-refundable says nothing about the hotel as a whole.
    assert _structured(listing, Field.FREE_CANCELLATION) == "UNCERTAIN"
    _assert_no_cancellation_claims(listing)


def test_missing_refundability_is_unknown():
    [offer] = _offers("Double Room", refundable_tag=None)
    listing = normalize_liteapi_listing(make_hotel(), [offer])

    assert offer.refundable is None and offer.refundability_label == "Unknown"
    [option] = listing.rooms[0].options
    assert option.cancellation_type == "Unknown" and option.free_cancellation is None
    assert _structured(listing, Field.FREE_CANCELLATION) == "UNCERTAIN"


def test_rfn_and_nrfn_inside_one_offer_is_unknown_not_false():
    offer = parse_public_offer(
        make_offer(rates=[make_rate(refundable_tag="RFN"), make_rate(refundable_tag="NRFN")], ssp=SSP * 2),
        requested_currency="USD",
    )
    listing = normalize_liteapi_listing(make_hotel(), [offer])

    assert offer.refundable is None and offer.refundability_label == "Unknown"
    [option] = listing.rooms[0].options
    assert option.free_cancellation is None
    assert option.price == SSP * 2


def test_mixed_rates_in_one_hotel_keep_per_offer_status_and_no_hotel_verdict():
    refundable = _offers("Double Room")
    non_refundable = _offers("Double Room", refundable_tag="NRFN", ssp=SSP - 50)
    listing = normalize_liteapi_listing(make_hotel(), refundable + non_refundable)

    [room] = listing.rooms
    assert [(o.cancellation_type, o.free_cancellation, o.price) for o in room.options] == [
        ("Refundable rate", None, SSP),
        ("Non-refundable rate", False, SSP - 50),
    ]
    assert listing.price == SSP - 50  # the cheapest offer is the non-refundable one
    assert _structured(listing, Field.FREE_CANCELLATION) == "UNCERTAIN"
    _assert_no_cancellation_claims(listing)


def test_free_cancellation_request_without_proof_is_never_confirmed():
    listing = normalize_liteapi_listing(make_hotel(), _offers("Double Room"))
    constraint = UserConstraint(
        raw_text="free cancellation",
        normalized_text="free cancellation",
        priority=ConstraintPriority.MUST,
        category=ConstraintCategory.POLICY,
        mapping_status=ConstraintMappingStatus.KNOWN,
        mapped_fields=[Field.FREE_CANCELLATION],
        evidence_strategy=EvidenceStrategy.STRUCTURED,
    )
    report = match_listing_structured(listing, SearchRequest(city="Barcelona", constraints=[constraint]))

    assert report.matches[Field.FREE_CANCELLATION].value.value == "UNCERTAIN"
    assert report.hard_fail_fields == []  # not rejected either
    # The textual fallback would get no cancellation wording to quote.
    assert not any("cancel" in item["text"].lower() or "refund" in item["text"].lower()
                   for item in _prepare_listing_evidence(listing))


def test_offer_level_cancellation_data_and_ssp_are_preserved():
    offers = _offers("Double Room") + _offers("Deluxe King", refundable_tag="NRFN", ssp=SSP + 70)
    listing = normalize_liteapi_listing(make_hotel(), offers)

    assert listing.public_offers == offers
    assert [(o.refundable, o.public_total, o.cancel_policy_times) for o in listing.public_offers] == [
        (True, SSP, ["2026-11-20 10:00:00 GMT"]),
        (False, SSP + 70, ["2026-11-20 10:00:00 GMT"]),
    ]
    assert listing.price == SSP
    _assert_no_confidential_amounts(listing.model_dump_json())


def test_no_confidential_amounts_in_any_serialized_form():
    offers = [parse_public_offer(make_offer(), requested_currency="USD")]
    listing = normalize_liteapi_listing(make_hotel(), offers)
    shown = ShownResultSet(result_set_id="rs-1", items=[ShownResult(result_id="r-1", listing=listing)])

    serialized = [
        listing.model_dump_json(),
        json.dumps(listing.model_dump(mode="json")),
        shown.model_dump_json(),
        repr(listing),
        " ".join(signal.raw_text for signal in collect_listing_signals(listing)),
        json.dumps(_prepare_listing_evidence(listing)),
    ]
    for text in serialized:
        _assert_no_confidential_amounts(text)
        assert "opaque-offer-id" not in text
        assert "commission" not in text.lower()


# --- retriever ---------------------------------------------------------------


class _FakeLiteApi:
    """MockTransport handler: records requests, serves synthetic payloads."""

    def __init__(self, rates_payload: dict, hotels: dict[str, dict | int]) -> None:
        self.rates_payload = rates_payload
        self.hotels = hotels
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.url.path.endswith("/hotels/rates"):
            return httpx.Response(200, json=self.rates_payload)
        hotel = self.hotels.get(request.url.params["hotelId"])
        if isinstance(hotel, int):
            return httpx.Response(hotel, json={"error": {"message": "synthetic"}})
        return httpx.Response(200, json={"data": hotel} if hotel else {})


def _request(**overrides) -> SearchRequest:
    values = {"city": "Barcelona", "check_in": date(2026, 11, 28), "check_out": date(2026, 12, 1), "adults": 2}
    values.update(overrides)
    return SearchRequest(**values)


def _retriever(fake: _FakeLiteApi, **kwargs) -> LiteApiRetriever:
    client = LiteApiClient("sand_test-key", transport=httpx.MockTransport(fake), retry_delay_seconds=0)
    return LiteApiRetriever(client, country_code="ES", **kwargs)


def _rates(*hotel_ids: str) -> dict:
    return {"sandbox": True, "data": [{"hotelId": hotel_id, "roomTypes": [make_offer()]} for hotel_id in hotel_ids]}


async def test_retriever_builds_listings_from_rates_and_details():
    fake = _FakeLiteApi(_rates("h1", "h2"), {"h1": make_hotel("h1"), "h2": make_hotel("h2", name="Second Hotel")})

    listings = await _retriever(fake).get_candidates(_request(), max_items=10)

    assert [listing.id for listing in listings] == ["h1", "h2"]
    assert all(listing.price == SSP for listing in listings)

    rates_request = fake.requests[0]
    body = json.loads(rates_request.content)
    assert rates_request.method == "POST"
    assert body["cityName"] == "Barcelona" and body["countryCode"] == "ES"
    assert body["checkin"] == "2026-11-28" and body["checkout"] == "2026-12-01"
    assert body["occupancies"] == [{"adults": 2, "children": []}]
    assert body["currency"] == "USD"
    assert "aiSearch" not in body
    assert {request.url.params["hotelId"] for request in fake.requests[1:]} == {"h1", "h2"}


async def test_retriever_limits_detail_calls():
    hotel_ids = [f"h{i}" for i in range(8)]
    fake = _FakeLiteApi(_rates(*hotel_ids), {hotel_id: make_hotel(hotel_id) for hotel_id in hotel_ids})

    listings = await _retriever(fake, max_detail_calls=3).get_candidates(_request(), max_items=40)

    assert [listing.id for listing in listings] == ["h0", "h1", "h2"]
    assert len(fake.requests) == 1 + 3


async def test_retriever_skips_hotels_without_content_or_failed_details():
    fake = _FakeLiteApi(_rates("h1", "h2", "h3"), {"h1": None, "h2": 404, "h3": make_hotel("h3")})

    listings = await _retriever(fake).get_candidates(_request(), max_items=10)

    assert [listing.id for listing in listings] == ["h3"]


async def test_retriever_raises_when_every_detail_call_fails():
    fake = _FakeLiteApi(_rates("h1", "h2"), {"h1": 503, "h2": 503})

    with pytest.raises(RetrievalError) as exc:
        await _retriever(fake).get_candidates(_request(), max_items=10)

    assert exc.value.code == "provider_unavailable"


async def test_retriever_returns_nothing_without_availability():
    fake = _FakeLiteApi({"sandbox": True, "data": []}, {})

    assert await _retriever(fake).get_candidates(_request(), max_items=10) == []
    assert len(fake.requests) == 1


async def test_retriever_treats_an_error_object_as_provider_error():
    fake = _FakeLiteApi({"error": {"code": 9999, "message": "synthetic"}}, {})

    with pytest.raises(RetrievalError) as exc:
        await _retriever(fake).get_candidates(_request(), max_items=10)

    assert exc.value.code == "provider_error"


@pytest.mark.parametrize(
    "overrides",
    [pytest.param({"children": 1}, id="children-without-ages"), pytest.param({"rooms": 2}, id="several-rooms")],
)
async def test_retriever_refuses_occupancies_it_cannot_express(overrides):
    fake = _FakeLiteApi(_rates("h1"), {"h1": make_hotel("h1")})

    with pytest.raises(RetrievalError) as exc:
        await _retriever(fake).get_candidates(_request(**overrides), max_items=10)

    assert exc.value.code == "unsupported_request"
    assert fake.requests == []


async def test_retriever_requires_a_country():
    client = LiteApiClient("sand_test-key", transport=httpx.MockTransport(_FakeLiteApi(_rates(), {})))

    with pytest.raises(RetrievalError) as exc:
        await LiteApiRetriever(client).get_candidates(_request(), max_items=10)

    assert exc.value.code == "configuration_error"
