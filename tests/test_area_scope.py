"""
Area measurements must describe the unit the user would book.

Found in the 2026-10-15 baseline: "Hotel in Baku with private bathroom and
120 square meters." was read as the size of a room. A building, plot or
terrace size never confirms (or rejects) a room or apartment size; an area
whose scope is unclear is UNCERTAIN, never a guessed YES or a false NO.

Fixture listings are used where they exist; every other listing below is
synthetic and named so.
"""

from __future__ import annotations

import json
from datetime import date
from types import SimpleNamespace

import pytest

from app.logic import constraint_evidence_resolution
from app.logic.numeric_filters import evaluate_numeric_filters, extract_area_sqm
from app.retrieval.fixtures import FIXTURES_PATH
from app.schemas.constraints import (
    ConstraintCategory,
    ConstraintMappingStatus,
    ConstraintPriority,
    EvidenceStrategy,
    UserConstraint,
)
from app.schemas.fallback_policy import FallbackPolicy
from app.schemas.filters import PriceConstraint, SearchFilters
from app.schemas.listing import Facility, ListingRaw, Room
from app.schemas.match import Ternary


def _fixture(listing_id: str) -> ListingRaw:
    data = json.loads(FIXTURES_PATH.read_text(encoding="utf-8"))
    items = data if isinstance(data, list) else data["listings"]
    return ListingRaw.model_validate(next(item for item in items if item["id"] == listing_id))


def _synthetic(description: str | None = None, *, name: str = "Synthetic listing", rooms: list[Room] = ()) -> ListingRaw:
    return ListingRaw(id="synthetic", name=name, city="Baku", description=description, rooms=list(rooms))


def _room(name: str, *facilities: str) -> Room:
    return Room(name=name, facilities=[Facility(name=f) for f in facilities])


def _area(listing: ListingRaw, minimum: float = 30):
    (result,) = evaluate_numeric_filters(listing, SearchFilters(area_sqm_min=minimum))
    return result


# --- deterministic numeric path ------------------------------------------------------------


@pytest.mark.parametrize(
    ("listing", "expected"),
    [
        (_synthetic(rooms=[_room("Deluxe Double Room", "Room area: 35 sqm")]), 35.0),  # synthetic
        (_synthetic("Each room is 35 sqm, with a desk and a rain shower."), 35.0),  # synthetic
        (_synthetic("Apartment: 85 sqm, two bedrooms and a kitchen."), 85.0),  # synthetic
        (_synthetic("Rooms: 25 sqm, rooftop terrace: 200 sqm."), 25.0),  # synthetic
        (_synthetic("A 35 sqm room with a 10 sqm balcony."), 35.0),  # synthetic
        (_synthetic("Spacious 30 m² rooms in a hotel building of 1200 m²."), 30.0),  # synthetic
    ],
)
def test_unit_area_is_taken_from_the_bookable_unit(listing, expected):
    value, evidence = extract_area_sqm(listing)

    assert value == expected
    assert evidence


@pytest.mark.parametrize(
    "listing",
    [
        _synthetic("Hotel building: 120 sqm."),  # synthetic
        _synthetic("Rooftop terrace: 200 sqm with city views."),  # synthetic
        _synthetic("Set on a 900 sqm plot with a garden."),  # synthetic
    ],
)
def test_non_unit_area_does_not_confirm_or_reject_a_unit_size(listing):
    assert extract_area_sqm(listing)[0] is None
    for minimum in (30, 500):
        assert _area(listing, minimum).value == Ternary.UNCERTAIN


@pytest.mark.parametrize("listing_id", ["hotel-large", "tbilisi-hotel-large"])
def test_fixture_hotel_size_is_not_a_room_size(listing_id):
    """The baseline bug: '... hotel in baku with private bathroom and 120 square meters.'"""
    result = _area(_fixture(listing_id), minimum=80)

    assert result.value == Ternary.UNCERTAIN
    assert result.actual_value is None


@pytest.mark.parametrize(
    ("listing_id", "expected"),
    [("apt-small", 45.0), ("baku-hotel-1", 30.0), ("tbilisi-hotel-1", 60.0), ("baku-apt-12", 130.0), ("phuket-apt-1", 65.0)],
)
def test_fixture_unit_areas_are_unchanged(listing_id, expected):
    assert extract_area_sqm(_fixture(listing_id))[0] == expected


def test_different_room_sizes_are_not_resolved_to_the_largest():
    listing = _synthetic(  # synthetic
        rooms=[_room("Standard Room", "20 m²"), _room("Junior Suite", "45 m²")],
    )

    assert extract_area_sqm(listing)[0] is None
    assert _area(listing, minimum=30).value == Ternary.UNCERTAIN  # depends on which room is booked
    assert _area(listing, minimum=15).value == Ternary.YES  # every room qualifies
    assert _area(listing, minimum=60).value == Ternary.NO  # no room qualifies


def test_a_size_range_is_not_resolved_to_its_upper_end():
    listing = _synthetic("Rooms from 20 to 45 sqm.")  # synthetic

    assert _area(listing, minimum=30).value == Ternary.UNCERTAIN


def test_missing_area_is_uncertain():
    result = _area(_synthetic("Bright apartment near the park."))  # synthetic

    assert result.value == Ternary.UNCERTAIN
    assert result.why == "AREA: could not extract area"


def test_unit_area_still_matches_yes_and_no():
    listing = _synthetic("Apartment: 85 sqm.")  # synthetic

    assert _area(listing, minimum=80).value == Ternary.YES
    assert _area(listing, minimum=100).value == Ternary.NO


def test_other_numeric_filters_are_unaffected():
    listing = _fixture("hotel-large")
    filters = SearchFilters(
        bedrooms_min=1,
        price=PriceConstraint(max_amount=100000, currency="USD", scope="total_stay"),
    )

    results = evaluate_numeric_filters(listing, filters, check_in=date(2026, 4, 1), check_out=date(2026, 4, 5))

    assert {r.attribute for r in results} >= {"bedrooms", "price_total"}
    assert all(r.attribute != "area_sqm" for r in results)


# --- textual fallback ---------------------------------------------------------------------


def _constraint(text: str) -> UserConstraint:
    return UserConstraint(
        raw_text=text,
        normalized_text=text,
        priority=ConstraintPriority.MUST,
        category=ConstraintCategory.LAYOUT,
        mapping_status=ConstraintMappingStatus.UNRESOLVED,
        mapped_fields=[],
        evidence_strategy=EvidenceStrategy.TEXTUAL,
    )


def _analysis(relation: str, snippet: str, *, evidence_type: str = "measurement") -> dict:
    support = relation == "supports"
    return {
        "signals": [
            {"relation": relation, "strength": "strong", "snippet": snippet, "source": "description", "evidence_type": evidence_type}
        ],
        "has_direct_support": support,
        "has_direct_contradiction": not support,
        "has_only_weak_or_indirect_evidence": False,
        "has_conflicting_evidence": False,
        "evidence_missing": False,
        "condition_or_extra_requirement_present": False,
        "confidence": 0.9,
        "reason": "scripted",
    }


@pytest.fixture
def llm_answers(monkeypatch):
    answers: list[dict] = []

    class Models:
        def generate_content(self, **kwargs):
            return SimpleNamespace(text=json.dumps(answers.pop(0)), usage_metadata=None)

    monkeypatch.setattr(constraint_evidence_resolution, "_gemini_client", lambda: SimpleNamespace(models=Models()))
    yield answers
    assert not answers


async def _decision(text: str, listing: ListingRaw):
    (result,) = await constraint_evidence_resolution.resolve_listing_constraints_with_fallback(
        listing=listing,
        constraints=[_constraint(text)],
        structured_matches_by_field={},
        policy=FallbackPolicy(enabled=True, must_only=False),
    )
    return result


async def test_fallback_hotel_size_does_not_confirm_a_spacious_room(llm_answers):
    listing = _fixture("hotel-large")
    llm_answers.append(_analysis("supports", "hotel in baku with private bathroom and 120 square meters."))

    result = await _decision("spacious room", listing)

    assert result.decision == "UNCERTAIN"
    assert "another part of the property" in result.reason


async def test_fallback_terrace_size_does_not_reject_a_spacious_room(llm_answers):
    listing = _synthetic("Cosy studio. Shared rooftop terrace of 15 sqm.")  # synthetic
    llm_answers.append(_analysis("contradicts", "Shared rooftop terrace of 15 sqm."))

    assert (await _decision("spacious room", listing)).decision == "UNCERTAIN"


async def test_fallback_unit_measurement_still_confirms(llm_answers):
    listing = _fixture("baku-apt-12")
    llm_answers.append(_analysis("supports", "large 130 sqm apartment with sea view"))

    assert (await _decision("spacious", listing)).decision == "YES"


async def test_fallback_terrace_size_confirms_a_terrace_requirement(llm_answers):
    listing = _synthetic("Apartment with a private terrace of 40 sqm.")  # synthetic
    llm_answers.append(_analysis("supports", "a private terrace of 40 sqm"))

    assert (await _decision("large terrace", listing)).decision == "YES"


async def test_fallback_factual_confirmation_without_an_area_is_unchanged(llm_answers):
    listing = _synthetic("Hotel with a quiet garden courtyard.")  # synthetic
    llm_answers.append(_analysis("supports", "Hotel with a quiet garden courtyard.", evidence_type="direct_statement"))

    assert (await _decision("garden", listing)).decision == "YES"


def test_evidence_prompt_states_the_measurement_scope_rule():
    prompt = constraint_evidence_resolution._build_system_prompt()

    assert "bookable unit" in prompt
    assert "building, plot or shared terrace" in prompt
