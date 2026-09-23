"""
Step 3 of the LISTING_QUESTION implementation: direct unit coverage for
the two V1 example fields (parking, balcony) across all three Ternary
outcomes, via the real match_listing_structured/_match_field_via_rules
path - not a fake/mocked matcher. Per the prior audit, `parking` had a
FieldRule but zero direct test coverage at all, and `balcony` had only a
YES case. This file does not change field_rules.py.

FIXED (previously a confirmed bug, documented via xfail; now a passing
regression test): _match_field_via_rules used to check positive aliases
first and unconditionally return YES on any match, before ever consulting
negative_aliases. PARKING's positive aliases include the bare word
"parking" (and "parking available"); BALCONY's include the bare word
"balcony". Both are literal substrings of their own negative_aliases'
natural phrasing ("no parking available on site" contains both "parking"
and "parking available"; "rooms have no balcony" contains "balcony").
_match_field_via_rules now computes both the best positive and best
negative signal match before deciding: if the SAME signal matched both
(the negative phrase textually nests the positive one), the negative,
more specific read wins (NO); if positive and negative evidence come from
different signals, that is a genuine conflict and resolves to UNCERTAIN
rather than guessing. See matcher_structured.py for the full logic.
"""

from __future__ import annotations

from app.logic.matcher_structured import match_listing_structured
from app.schemas.constraints import (
    ConstraintCategory,
    ConstraintMappingStatus,
    ConstraintPriority,
    EvidenceStrategy,
    UserConstraint,
)
from app.schemas.fields import Field
from app.schemas.listing import ListingRaw, Room
from app.schemas.match import Ternary
from app.schemas.query import SearchRequest


def _must_constraint(field: Field, text: str) -> UserConstraint:
    return UserConstraint(
        raw_text=text,
        normalized_text=text,
        priority=ConstraintPriority.MUST,
        category=ConstraintCategory.AMENITY,
        mapping_status=ConstraintMappingStatus.KNOWN,
        mapped_fields=[field],
        evidence_strategy=EvidenceStrategy.STRUCTURED,
    )


def _match(listing: ListingRaw, field: Field, text: str):
    req = SearchRequest(constraints=[_must_constraint(field, text)])
    report = match_listing_structured(listing, req)
    return report.matches[field]


# --- parking ---------------------------------------------------------


def test_parking_explicit_positive_evidence_is_yes():
    listing = ListingRaw(
        id="p-yes",
        name="Test Listing",
        facilities=[{"name": "Free parking"}],
    )

    fm = _match(listing, Field.PARKING, "parking")

    assert fm.value == Ternary.YES
    assert fm.evidence


def test_parking_explicit_negative_evidence_is_no():
    listing = ListingRaw(
        id="p-no",
        name="Test Listing",
        description="No parking available on site.",
    )

    fm = _match(listing, Field.PARKING, "parking")

    assert fm.value == Ternary.NO


def test_parking_missing_evidence_is_uncertain_not_no():
    listing = ListingRaw(
        id="p-unknown",
        name="Test Listing",
        description="A cozy apartment in the city center with a kitchen.",
        facilities=[{"name": "Free WiFi"}],
    )

    fm = _match(listing, Field.PARKING, "parking")

    assert fm.value == Ternary.UNCERTAIN
    assert fm.value != Ternary.NO


# --- balcony -----------------------------------------------------------


def test_balcony_explicit_positive_evidence_is_yes():
    listing = ListingRaw(
        id="b-yes",
        name="Test Listing",
        rooms=[Room(name="Deluxe Room", facilities=["Private balcony"])],
    )

    fm = _match(listing, Field.BALCONY, "balcony")

    assert fm.value == Ternary.YES
    assert fm.evidence


def test_balcony_explicit_negative_evidence_is_no():
    listing = ListingRaw(
        id="b-no",
        name="Test Listing",
        description="Rooms have no balcony.",
    )

    fm = _match(listing, Field.BALCONY, "balcony")

    assert fm.value == Ternary.NO


def test_balcony_missing_evidence_is_uncertain_not_no():
    listing = ListingRaw(
        id="b-unknown",
        name="Test Listing",
        description="A cozy apartment in the city center with a kitchen.",
        facilities=[{"name": "Free WiFi"}],
    )

    fm = _match(listing, Field.BALCONY, "balcony")

    assert fm.value == Ternary.UNCERTAIN
    assert fm.value != Ternary.NO


# --- conflicting evidence across independent signals --------------------
#
# Step 7 of the narrow matcher fix: when positive and negative evidence
# come from two DIFFERENT signals (e.g. a facility claims parking exists
# but the description says it doesn't), that's a genuine conflict we
# cannot safely resolve deterministically - it must be UNCERTAIN, not a
# guess. This is distinct from the same-signal case above (e.g. "No
# parking available on site."), where the negative phrase textually nests
# the positive alias and is resolved as NO.


def test_parking_conflicting_evidence_across_signals_is_uncertain():
    listing = ListingRaw(
        id="p-conflict",
        name="Test Listing",
        facilities=[{"name": "Private parking"}],
        description="No parking available on site.",
    )

    fm = _match(listing, Field.PARKING, "parking")

    assert fm.value == Ternary.UNCERTAIN
    assert len(fm.evidence) == 2


def test_balcony_conflicting_evidence_across_signals_is_uncertain():
    listing = ListingRaw(
        id="b-conflict",
        name="Test Listing",
        rooms=[Room(name="Deluxe Room", facilities=["Private balcony"])],
        description="No balcony available in this room.",
    )

    fm = _match(listing, Field.BALCONY, "balcony")

    assert fm.value == Ternary.UNCERTAIN
    assert len(fm.evidence) == 2


def test_parking_independent_positive_evidence_in_multiple_signals_is_yes():
    listing = ListingRaw(
        id="p-multi-yes",
        name="Test Listing",
        facilities=[{"name": "Private parking"}],
        description="Parking is available for guests.",
    )

    fm = _match(listing, Field.PARKING, "parking")

    assert fm.value == Ternary.YES


def test_parking_independent_negative_evidence_in_multiple_signals_is_no():
    listing = ListingRaw(
        id="p-multi-no",
        name="Test Listing",
        description="No parking available on site.",
        policies=[{"title": "Parking", "content": "Parking not available."}],
    )

    fm = _match(listing, Field.PARKING, "parking")

    assert fm.value == Ternary.NO
