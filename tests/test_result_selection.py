from app.logic.result_selection import classify_ranked_item, select_ranked_items


def test_classify_ranked_item_as_strong_when_all_required_constraints_are_confirmed():
    item = {
        "score": 25.0,
        "matched_must_total": 2,
        "matched_must_count": 2,
        "constraint_resolution_results": [],
        "matches": {},
        "numeric_results": [],
        "property_result": None,
        "occupancy_result": None,
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
        "property_result": None,
        "occupancy_result": None,
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
        "property_result": None,
        "occupancy_result": None,
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
        "property_result": None,
        "occupancy_result": None,
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
        "property_result": None,
        "occupancy_result": None,
    }

    selected = select_ranked_items([partial_item, strong_item], top_n=2)

    assert len(selected) == 2
    assert selected[0]["listing_name"] == "Strong listing"
    assert selected[0]["match_tier"] == "strong"
    assert selected[1]["listing_name"] == "Partial listing"
    assert selected[1]["match_tier"] == "partial"
    
    
    
from app.logic.property_semantics import SemanticMatchResult
from app.schemas.match import Ternary


def _base_item(**overrides):
    item = {
        "listing_name": "Test listing",
        "score": 10.0,
        "matched_must_total": 0,
        "matched_must_count": 0,
        "constraint_resolution_results": [],
        "matches": {},
        "numeric_results": [],
        "property_result": None,
        "occupancy_result": None,
    }
    item.update(overrides)
    return item


def _property_result(value: Ternary, actual_value: str | None = None):
    return SemanticMatchResult(
        attribute="property_type",
        value=value,
        actual_value=actual_value,
        evidence=[],
        why=f"PROPERTY_TYPE: {value.value}",
    )


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


def test_property_type_yes_makes_core_request_strong():
    item = _base_item(
        property_result=_property_result(Ternary.YES, actual_value="ryokan"),
    )

    classified = classify_ranked_item(item)

    assert classified["eligibility_status"] == "eligible"
    assert classified["match_tier"] == "strong"
    assert "core request is confirmed" in classified["selection_reasons"]


def test_property_type_uncertain_makes_partial_match():
    item = _base_item(
        property_result=_property_result(Ternary.UNCERTAIN, actual_value=None),
    )

    classified = classify_ranked_item(item)

    assert classified["eligibility_status"] == "eligible"
    assert classified["match_tier"] == "partial"


def test_property_type_no_makes_ineligible():
    item = _base_item(
        property_result=_property_result(Ternary.NO, actual_value="apartment"),
    )

    classified = classify_ranked_item(item)

    assert classified["eligibility_status"] == "ineligible"
    assert classified["match_tier"] == "weak"


def test_select_ranked_items_excludes_ineligible_items():
    eligible_item = _base_item(
        listing_name="Kyoto Ryokan",
        score=10.0,
        property_result=_property_result(Ternary.YES, actual_value="ryokan"),
    )

    ineligible_item = _base_item(
        listing_name="Kyoto Apartment",
        score=100.0,
        property_result=_property_result(Ternary.NO, actual_value="apartment"),
    )

    selected = select_ranked_items([ineligible_item, eligible_item], top_n=2)

    assert len(selected) == 1
    assert selected[0]["listing_name"] == "Kyoto Ryokan"
    assert selected[0]["eligibility_status"] == "eligible"
    assert selected[0]["match_tier"] == "strong"
    
    
def test_occupancy_yes_makes_strong():
    item = _base_item(
        occupancy_result=_property_result(Ternary.YES, actual_value="entire_place"),
    )

    classified = classify_ranked_item(item)

    assert classified["eligibility_status"] == "eligible"
    assert classified["match_tier"] == "strong"


def test_occupancy_uncertain_makes_partial():
    item = _base_item(
        occupancy_result=_property_result(Ternary.UNCERTAIN),
    )

    classified = classify_ranked_item(item)

    assert classified["eligibility_status"] == "eligible"
    assert classified["match_tier"] == "partial"


def test_occupancy_no_makes_ineligible():
    item = _base_item(
        occupancy_result=_property_result(Ternary.NO, actual_value="shared_room"),
    )

    classified = classify_ranked_item(item)

    assert classified["eligibility_status"] == "ineligible"
    assert classified["match_tier"] == "weak"


def test_property_yes_but_other_must_uncertain_results_in_weak():
    item = _base_item(
        matched_must_total=1,
        matched_must_count=0,
        constraint_resolution_results=[
            {
                "normalized_text": "city center",
                "resolution_status": "uncertain",
                "explicit_negative": False,
            }
        ],
        property_result=_property_result(Ternary.YES, actual_value="ryokan"),
    )

    classified = classify_ranked_item(item)

    assert classified["eligibility_status"] == "eligible"
    assert classified["match_tier"] == "weak"


def test_property_yes_but_explicit_negative_still_ineligible():
    item = _base_item(
        property_result=_property_result(Ternary.YES, actual_value="ryokan"),
        constraint_resolution_results=[
            {
                "normalized_text": "pet friendly",
                "resolution_status": "failed",
                "explicit_negative": True,
            }
        ],
    )

    classified = classify_ranked_item(item)

    assert classified["eligibility_status"] == "ineligible"
    assert classified["match_tier"] == "weak"