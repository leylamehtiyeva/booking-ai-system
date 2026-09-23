from app.logic.result_selection import classify_ranked_item, select_ranked_items
from app.schemas.listing import ListingRaw
from app.schemas.property_semantics import PropertyType


def test_classify_ranked_item_as_strong_when_all_required_constraints_are_confirmed():
    item = {
        "score": 25.0,
        "matched_must_total": 2,
        "matched_must_count": 2,
        "constraint_resolution_results": [],
        "matches": {},
        "numeric_results": [],
    }

    classified = classify_ranked_item(item)

    assert classified["eligibility_status"] == "eligible"
    assert classified["match_tier"] == "strong"
    assert "all required constraints are confirmed" in classified["selection_reasons"]


def test_classify_ranked_item_as_partial_when_required_constraints_are_uncertain():
    item = {
        "score": 20.0,
        "matched_must_total": 2,
        "matched_must_count": 1,
        "constraint_resolution_results": [
            {
                "normalized_text": "satellite TV",
                "resolution_status": "uncertain",
                "explicit_negative": False,
            }
        ],
        "matches": {},
        "numeric_results": [],
    }

    classified = classify_ranked_item(item)

    assert classified["eligibility_status"] == "eligible"
    assert classified["match_tier"] == "partial"
    assert "some requested constraints are not fully confirmed" in classified["selection_reasons"]


def test_classify_ranked_item_as_ineligible_when_required_constraints_failed():
    item = {
        "score": 18.0,
        "matched_must_total": 1,
        "matched_must_count": 0,
        "constraint_resolution_results": [
            {
                "normalized_text": "satellite TV",
                "resolution_status": "failed",
                "explicit_negative": True,
            }
        ],
        "matches": {},
        "numeric_results": [],
    }

    classified = classify_ranked_item(item)

    assert classified["eligibility_status"] == "ineligible"
    assert classified["match_tier"] == "weak"
    assert "failed required constraints" in classified["blocking_reasons"]


def test_select_ranked_items_prefers_strong_matches_before_partial():
    strong_item = {
        "listing_name": "Strong listing",
        "score": 30.0,
        "matched_must_total": 1,
        "matched_must_count": 1,
        "constraint_resolution_results": [],
        "matches": {},
        "numeric_results": [],
    }

    partial_item = {
        "listing_name": "Partial listing",
        "score": 40.0,
        "matched_must_total": 2,
        "matched_must_count": 1,
        "constraint_resolution_results": [
            {
                "normalized_text": "satellite TV",
                "resolution_status": "uncertain",
                "explicit_negative": False,
            }
        ],
        "matches": {},
        "numeric_results": [],
    }

    selected = select_ranked_items([partial_item, strong_item], top_n=2)

    assert len(selected) == 2
    assert selected[0]["listing_name"] == "Strong listing"
    assert selected[0]["match_tier"] == "strong"
    assert selected[1]["listing_name"] == "Partial listing"
    assert selected[1]["match_tier"] == "partial"


def _base_item(**overrides):
    item = {
        "listing_name": "Test listing",
        "score": 10.0,
        "matched_must_total": 0,
        "matched_must_count": 0,
        "constraint_resolution_results": [],
        "matches": {},
        "numeric_results": [],
    }
    item.update(overrides)
    return item


def test_eligible_strong_when_all_must_constraints_confirmed():
    item = _base_item(
        matched_must_total=2,
        matched_must_count=2,
    )

    classified = classify_ranked_item(item)

    assert classified["eligibility_status"] == "eligible"
    assert classified["match_tier"] == "strong"
    assert "all required constraints are confirmed" in classified["selection_reasons"]


def test_eligible_partial_when_must_constraint_is_uncertain():
    item = _base_item(
        matched_must_total=1,
        matched_must_count=0,
        uncertain_constraints=[
            {
                "name": "Ryokan",
                "status": "uncertain",
                "reason": "Could not confirm property type.",
            }
        ],
    )

    classified = classify_ranked_item(item)

    assert classified["eligibility_status"] == "eligible"
    assert classified["match_tier"] == "weak"
    assert "no required constraints are confirmed" in classified["selection_reasons"]

def test_ineligible_when_required_constraint_failed():
    item = _base_item(
        matched_must_total=1,
        matched_must_count=0,
        failed_constraints=[
            {
                "name": "Pets allowed",
                "status": "failed",
                "reason": "Pets are not allowed.",
            }
        ],
    )

    classified = classify_ranked_item(item)

    assert classified["eligibility_status"] == "ineligible"
    assert classified["match_tier"] == "weak"
    assert "failed required constraints" in classified["blocking_reasons"]


def test_ineligible_when_explicit_negative_evidence_found():
    item = _base_item(
        matched_must_total=1,
        matched_must_count=0,
        constraint_resolution_results=[
            {
                "normalized_text": "pet friendly",
                "resolution_status": "failed",
                "explicit_negative": True,
                "reason": "Pets are not allowed.",
            }
        ],
    )

    classified = classify_ranked_item(item)

    assert classified["eligibility_status"] == "ineligible"
    assert classified["match_tier"] == "weak"
    assert "explicit negative evidence for requested constraints" in classified["blocking_reasons"]


def test_select_ranked_items_diversifies_equal_quality_requested_property_types():
    """
    Property type for diversification comes straight from item["listing"].property_type
    (the Apify-provided field) — not from a match/comparison result, since retrieval
    already filters by requested property type.
    """
    apartment_a = _base_item(
        listing_name="Apartment A",
        score=10.0,
        listing=ListingRaw(property_type="apartment"),
    )

    apartment_b = _base_item(
        listing_name="Apartment B",
        score=10.0,
        listing=ListingRaw(property_type="apartment"),
    )

    hotel_a = _base_item(
        listing_name="Hotel A",
        score=10.0,
        listing=ListingRaw(property_type="hotel"),
    )

    hotel_b = _base_item(
        listing_name="Hotel B",
        score=10.0,
        listing=ListingRaw(property_type="hotel"),
    )

    selected = select_ranked_items(
        [
            apartment_a,
            apartment_b,
            hotel_a,
            hotel_b,
        ],
        top_n=2,
        requested_property_types=[
            PropertyType.APARTMENT,
            PropertyType.HOTEL,
        ],
    )

    assert [
        item["listing_name"]
        for item in selected
    ] == [
        "Apartment A",
        "Hotel A",
    ]


def test_property_type_mismatch_makes_item_ineligible():
    """
    Safety net for when retrieval's own property-type filter misses (e.g.
    Apify's own classification disagrees with what we requested).
    """
    item = _base_item(
        matched_must_total=1,
        matched_must_count=1,
        listing=ListingRaw(property_type="hotel"),
    )

    classified = classify_ranked_item(
        item,
        requested_property_types=[PropertyType.APARTMENT],
    )

    assert classified["eligibility_status"] == "ineligible"
    assert "property type does not match requested type" in classified["blocking_reasons"]


def test_property_type_match_stays_eligible():
    item = _base_item(
        matched_must_total=1,
        matched_must_count=1,
        listing=ListingRaw(property_type="apartment"),
    )

    classified = classify_ranked_item(
        item,
        requested_property_types=[PropertyType.APARTMENT],
    )

    assert classified["eligibility_status"] == "eligible"
    assert classified["blocking_reasons"] == []


def test_unknown_property_type_is_not_blocked():
    """
    A listing without a structured property_type (e.g. the fast provider,
    which doesn't set this field) is not penalized — this is an exclusion
    safety net, not a confirmation requirement.
    """
    item = _base_item(
        matched_must_total=1,
        matched_must_count=1,
        listing=ListingRaw(property_type=None),
    )

    classified = classify_ranked_item(
        item,
        requested_property_types=[PropertyType.APARTMENT],
    )

    assert classified["eligibility_status"] == "eligible"
    assert classified["blocking_reasons"] == []


def test_property_type_alone_makes_strong_when_no_other_musts():
    """
    A confirmed property_type counts as its own must slot, so a query with
    no other must-constraints ("just find me a ryokan") can still reach
    "strong" — via the same all-musts-confirmed rule, not a separate
    OR-branch that could bypass an unconfirmed must elsewhere.
    """
    item = _base_item(
        matched_must_total=0,
        matched_must_count=0,
        listing=ListingRaw(property_type="ryokan"),
    )

    classified = classify_ranked_item(
        item,
        requested_property_types=[PropertyType.RYOKAN],
    )

    assert classified["eligibility_status"] == "eligible"
    assert classified["match_tier"] == "strong"
    assert "all required constraints are confirmed" in classified["selection_reasons"]


def test_unconfirmed_must_still_blocks_strong_even_when_property_type_matches():
    """
    The bug this replaces: property_type used to be an independent OR
    signal that could push a listing to "strong" even when a real must
    (e.g. wifi) was not confirmed. Now it's just one more slot in the same
    all-or-nothing must count.
    """
    item = _base_item(
        matched_must_total=1,  # e.g. "wifi", not confirmed
        matched_must_count=0,
        listing=ListingRaw(property_type="apartment"),
    )

    classified = classify_ranked_item(
        item,
        requested_property_types=[PropertyType.APARTMENT],
    )

    assert classified["match_tier"] != "strong"


def test_unknown_property_type_prevents_strong_even_with_musts_confirmed():
    item = _base_item(
        matched_must_total=1,
        matched_must_count=1,
        listing=ListingRaw(property_type=None),
    )

    classified = classify_ranked_item(
        item,
        requested_property_types=[PropertyType.APARTMENT],
    )

    assert classified["eligibility_status"] == "eligible"
    assert classified["match_tier"] != "strong"


def test_no_requested_property_types_never_blocks():
    item = _base_item(
        matched_must_total=1,
        matched_must_count=1,
        listing=ListingRaw(property_type="hotel"),
    )

    classified = classify_ranked_item(item, requested_property_types=None)

    assert classified["eligibility_status"] == "eligible"
    assert classified["blocking_reasons"] == []


def test_select_ranked_items_excludes_property_type_mismatch():
    matching = _base_item(
        listing_name="Kyoto Ryokan",
        score=10.0,
        matched_must_total=1,
        matched_must_count=1,
        listing=ListingRaw(property_type="ryokan"),
    )

    mismatched = _base_item(
        listing_name="Kyoto Apartment",
        score=100.0,
        matched_must_total=1,
        matched_must_count=1,
        listing=ListingRaw(property_type="apartment"),
    )

    selected = select_ranked_items(
        [mismatched, matching],
        top_n=2,
        requested_property_types=[PropertyType.RYOKAN],
    )

    assert len(selected) == 1
    assert selected[0]["listing_name"] == "Kyoto Ryokan"
