from app.logic.numeric_filters import (
    extract_area_sqm,
    extract_bathroom_count,
    extract_bedroom_count,
    extract_total_price,
    match_area_filters,
    match_bathroom_filters,
    match_bedrooms_filters,
    match_price_filters,
)
from datetime import date
from app.schemas.filters import PriceConstraint, SearchFilters
from app.schemas.listing import ListingRaw, Room
from app.schemas.match import Ternary


def test_extract_bedroom_count_from_room_name():
    listing = ListingRaw(
        id="x1",
        name="Sea View Apartment",
        description="Spacious apartment in Baku",
        rooms=[
            Room(name="Three-Bedroom Apartment with Balcony", facilities=[]),
        ],
    )

    value, evidence = extract_bedroom_count(listing)

    assert value == 3
    assert evidence
    assert evidence[0].path == "rooms[0].name"


def test_extract_bedroom_count_from_studio():
    listing = ListingRaw(
        id="x2",
        name="Cozy studio in city center",
        description="Nice and compact option",
        rooms=[],
    )

    value, evidence = extract_bedroom_count(listing)

    assert value == 0
    assert evidence


def test_extract_area_sqm_from_sqm():
    listing = ListingRaw(
        id="x3",
        name="Apartment 85 sqm",
        description="Bright apartment with private bathroom",
        rooms=[],
    )

    value, evidence = extract_area_sqm(listing)

    assert value == 85.0
    assert evidence
    assert evidence[0].path == "listing.name"


def test_extract_area_sqm_from_square_feet():
    listing = ListingRaw(
        id="x4",
        name="Large apartment",
        description="Huge unit with 2196 feet² and private entrance.",
        rooms=[],
    )

    value, evidence = extract_area_sqm(listing)

    assert value is not None
    assert 203.9 <= value <= 204.1
    assert evidence
    assert evidence[0].path == "listing.description"


def test_match_bedrooms_filters_yes():
    filters = SearchFilters(bedrooms_min=2)

    out = match_bedrooms_filters(3, filters)

    assert out is not None
    assert out.value == Ternary.YES
    assert out.actual_value == 3
    assert "3 >= required 2" in out.why


def test_match_bedrooms_filters_no():
    filters = SearchFilters(bedrooms_min=2)

    out = match_bedrooms_filters(1, filters)

    assert out is not None
    assert out.value == Ternary.NO
    assert "1 < required min 2" in out.why


def test_match_bedrooms_filters_uncertain_when_missing():
    filters = SearchFilters(bedrooms_min=2)

    out = match_bedrooms_filters(None, filters)

    assert out is not None
    assert out.value == Ternary.UNCERTAIN
    assert out.actual_value is None


def test_match_area_filters_yes():
    filters = SearchFilters(area_sqm_min=80)

    out = match_area_filters(204.0, filters)

    assert out is not None
    assert out.value == Ternary.YES
    assert "204 sqm >= required 80" in out.why


def test_match_area_filters_no():
    filters = SearchFilters(area_sqm_min=80)

    out = match_area_filters(45.0, filters)

    assert out is not None
    assert out.value == Ternary.NO
    assert "45 sqm < required min 80" in out.why


def test_match_area_filters_uncertain_when_missing():
    filters = SearchFilters(area_sqm_min=80)

    out = match_area_filters(None, filters)

    assert out is not None
    assert out.value == Ternary.UNCERTAIN
    
    
def test_extract_total_price_from_listing_top_level():
    listing = ListingRaw(
        id="p1",
        name="Apartment STEL",
        price=700.0,
        currency="US$",
        rooms=[],
    )

    total_price, currency, evidence = extract_total_price(listing)

    assert total_price == 700.0
    assert currency == "USD"
    assert evidence
    assert evidence[0].path == "listing.price"


def test_match_price_filters_per_night_yes():
    filters = SearchFilters(
        price=PriceConstraint(
            max_amount=120,
            currency="USD",
            scope="per_night",
        )
    )

    out = match_price_filters(
        total_price=700.0,
        listing_currency="US$",
        filters=filters,
        check_in=date(2026, 4, 8),
        check_out=date(2026, 4, 15),  # 7 nights
    )

    assert out is not None
    assert out.value == Ternary.YES
    assert "PRICE:" in out.why


def test_match_price_filters_per_night_no():
    filters = SearchFilters(
        price=PriceConstraint(
            max_amount=50,
            currency="USD",
            scope="per_night",
        )
    )

    out = match_price_filters(
        total_price=700.0,
        listing_currency="US$",
        filters=filters,
        check_in=date(2026, 4, 8),
        check_out=date(2026, 4, 15),  # 7 nights => max total 350
    )

    assert out is not None
    assert out.value == Ternary.NO
    assert "allowed max total 350" in out.why


def test_match_price_filters_total_stay_yes():
    filters = SearchFilters(
        price=PriceConstraint(
            max_amount=750,
            currency="USD",
            scope="total_stay",
        )
    )

    out = match_price_filters(
        total_price=700.0,
        listing_currency="USD",
        filters=filters,
        check_in=date(2026, 4, 8),
        check_out=date(2026, 4, 15),
    )

    assert out is not None
    assert out.value == Ternary.YES


def test_match_price_filters_converts_non_usd_request_budget(monkeypatch):
    filters = SearchFilters(
        price=PriceConstraint(
            max_amount=500,
            currency="AZN",
            scope="total_stay",
        )
    )

    def _fake_convert(amount, currency):
        assert currency == "AZN"
        return amount / 2.0, None

    monkeypatch.setattr("app.logic.numeric_filters.convert_amount_to_usd", _fake_convert)

    out = match_price_filters(
        total_price=700.0,
        listing_currency="USD",
        filters=filters,
        check_in=date(2026, 4, 8),
        check_out=date(2026, 4, 15),
    )

    assert out is not None
    assert out.value == Ternary.NO
    assert "allowed max total 250.0" in out.why.lower()
    
def test_extract_bathroom_count_from_description_numeric():
    listing = ListingRaw(
        id="b1",
        name="Spacious apartment",
        description="Beautiful apartment with 2 bathrooms and a balcony.",
        rooms=[],
    )

    bathroom_count, evidence = extract_bathroom_count(listing)

    assert bathroom_count == 2.0
    assert evidence
    assert "2 bathrooms" in evidence[0].snippet.lower()


def test_extract_bathroom_count_from_description_decimal():
    listing = ListingRaw(
        id="b2",
        name="Family stay",
        description="Includes 1.5 bathrooms and a private kitchen.",
        rooms=[],
    )

    bathroom_count, evidence = extract_bathroom_count(listing)

    assert bathroom_count == 1.5
    assert evidence
    assert "1.5 bathrooms" in evidence[0].snippet.lower()


def test_extract_bathroom_count_from_description_word_number():
    listing = ListingRaw(
        id="b3",
        name="Large house",
        description="A large house with two bathrooms and garden view.",
        rooms=[],
    )

    bathroom_count, evidence = extract_bathroom_count(listing)

    assert bathroom_count == 2.0
    assert evidence
    assert "two bathrooms" in evidence[0].snippet.lower()


def test_match_bathroom_filters_yes():
    filters = SearchFilters(bathrooms_min=2)

    out = match_bathroom_filters(
        bathroom_count=2.0,
        filters=filters,
    )

    assert out is not None
    assert out.value == Ternary.YES
    assert "BATHROOMS:" in out.why


def test_match_bathroom_filters_no():
    filters = SearchFilters(bathrooms_min=2)

    out = match_bathroom_filters(
        bathroom_count=1.0,
        filters=filters,
    )

    assert out is not None
    assert out.value == Ternary.NO
    assert "required min 2" in out.why


def test_match_bathroom_filters_uncertain_when_missing():
    filters = SearchFilters(bathrooms_min=2)

    out = match_bathroom_filters(
        bathroom_count=None,
        filters=filters,
    )

    assert out is not None
    assert out.value == Ternary.UNCERTAIN
    
    
    
def test_match_price_filters_non_usd_request_budget_uncertain_when_fx_unavailable(monkeypatch):
    filters = SearchFilters(
        price=PriceConstraint(
            max_amount=500,
            currency="EUR",
            scope="total_stay",
        )
    )

    monkeypatch.setattr(
        "app.logic.numeric_filters.convert_amount_to_usd",
        lambda amount, currency: (None, None),
    )

    out = match_price_filters(
        total_price=700.0,
        listing_currency="USD",
        filters=filters,
        check_in=date(2026, 4, 8),
        check_out=date(2026, 4, 15),
    )

    assert out is not None
    assert out.value == Ternary.UNCERTAIN
    assert "could not convert request currency eur to usd" in out.why.lower()