from app.logic.build_answer_payload import build_answer_payload

from app.schemas.search_response import (
    ConstraintResolutionEvidence,
    ConstraintResolutionItem,
    ConstraintStatus,
    NormalizedRequestSummary,
    NormalizedSearchResponse,
    NormalizedSearchResult,
    ResultFact,
)

from app.schemas.search_response import SearchStatus


def test_build_answer_payload_uses_actual_property_type_in_explanation():
    response = NormalizedSearchResponse(
        status=SearchStatus.RESULTS,
        request_summary=NormalizedRequestSummary(
            city="Baku",
            check_in="2026-08-21",
            check_out="2026-09-04",
            constraints=[],
            property_types=[
                "apartment",
                "hotel",
            ],
            occupancy_types=[],
            filters={},
        ),
        results=[
            NormalizedSearchResult(
                result_id="hotel-1",
                title="Moxen Luxury Hotel",
                url="https://example.com/hotel",
                score=0.0,
                matched_constraints=[
                    ConstraintStatus(
                        name="property_type",
                        status="matched",
                        reason="PROPERTY_TYPE: matched hotel",
                    )
                ],
                uncertain_constraints=[],
                failed_constraints=[],
                facts=[
                    ResultFact(
                        key="property_type",
                        value="hotel",
                        source="property_semantics",
                    )
                ],
                why=[
                    "PROPERTY_TYPE: matched hotel"
                ],
            )
        ],
        debug_notes=[],
    )

    payload = build_answer_payload(
        response,
        top_k=3,
    )

    explanation = (
        payload["top_results"][0]
        ["answer_explanation"]
    )

    assert (
        explanation["confirmed"][0]["reason"]
        == "Matches the requested hotel type"
    )


def test_build_answer_payload_for_normal_results():
    response = NormalizedSearchResponse(
        need_clarification=False,
        questions=[],
        request_summary=NormalizedRequestSummary(
            city="Baku",
            check_in="2026-04-08",
            check_out="2026-04-15",
            constraints=[
                {
                    "raw_text": "with a kitchen",
                    "normalized_text": "kitchen",
                    "priority": "must",
                    "category": "amenity",
                    "mapping_status": "known",
                    "mapped_fields": ["kitchen"],
                    "evidence_strategy": "structured",
                },
                {
                    "raw_text": "balcony",
                    "normalized_text": "balcony",
                    "priority": "nice",
                    "category": "amenity",
                    "mapping_status": "known",
                    "mapped_fields": ["balcony"],
                    "evidence_strategy": "structured",
                },
            ],
            property_types=["apartment"],
            occupancy_types=[],
            filters={
                "bedrooms_min": 2,
                "price": {
                    "max_amount": 120,
                    "currency": "USD",
                    "scope": "per_night",
                },
            },
        ),
        results=[
            NormalizedSearchResult(
                result_id="abc123",
                title="Apartment STEL",
                url="https://example.com/stel",
                score=23.0,
                matched_constraints=[
                    ConstraintStatus(
                        name="kitchen",
                        status="matched",
                        reason="Private kitchen",
                    ),
                    ConstraintStatus(
                        name="property_type",
                        status="matched",
                        reason="PROPERTY_TYPE: matched apartment",
                    ),
                ],
                uncertain_constraints=[
                    ConstraintStatus(
                        name="price_total",
                        status="uncertain",
                        reason="PRICE: currency mismatch listing=USD, request=AZN",
                    )
                ],
                failed_constraints=[],
                facts=[
                    ResultFact(key="property_type", value="apartment", source="property_semantics"),
                    ResultFact(key="listing_price_total", value=493.39, source="listing"),
                    ResultFact(key="listing_currency", value="USD", source="listing"),
                    ResultFact(key="bedrooms", value=2, source="numeric_filters"),
                ],
                why=[
                    "KITCHEN: Private kitchen",
                    "PRICE: currency mismatch listing=USD, request=AZN",
                ],
            )
        ],
        debug_notes=[],
    )

    payload = build_answer_payload(
        response,
        latest_user_query="I want an apartment in Baku with a kitchen",
        top_k=3,
    )

    assert payload["need_clarification"] is False
    assert payload["results_count"] == 1
    assert payload["request_summary"]["city"] == "Baku"
    assert payload["active_intent"]["city"] == "Baku"
    assert payload["latest_user_query"] == "I want an apartment in Baku with a kitchen"

    first = payload["top_results"][0]
    assert first["result_id"] == "abc123"
    assert first["title"] == "Apartment STEL"

    assert "kitchen" in first["matched_constraint_names"]
    assert "price_total" in first["uncertain_constraint_names"]

    assert first["key_facts"]["property_type"] == "apartment"
    assert first["key_facts"]["listing_currency"] == "USD"
    assert first["key_facts"]["bedrooms"] == 2

    assert first["fit_summary"]
    assert "uncertain" in first["fit_summary"].lower()

    assert first["why_match"]
    assert "Private kitchen" in first["why_match"]

    assert first["uncertain_points"]
    assert "PRICE: currency mismatch listing=USD, request=AZN" in first["uncertain_points"]

    assert first["tradeoffs"] == []

    assert first["price_summary"] == "493.39 USD total"
    assert first["budget_summary"] is None
    assert first["budget_status"] is None

    assert first["key_facts_summary"]
    assert "type: apartment" in first["key_facts_summary"]
    assert "2 bedroom(s)" in first["key_facts_summary"]
    
    assert "answer_explanation" in first

    explanation = first["answer_explanation"]
    assert explanation["status_label"] in {
        "fully_confirmed_match",
        "partially_confirmed_match",
        "strong_match_with_caveats",
        "relevant_option",
    }
    assert explanation["status_text"]
    assert explanation["decision_summary"]

    assert explanation["confirmed"]
    assert explanation["confirmed"][0]["label"]

    assert explanation["needs_confirmation"]
    assert explanation["needs_confirmation"][0]["label"] == "Budget"


def test_build_answer_payload_for_clarification():
    response = NormalizedSearchResponse(
        need_clarification=True,
        questions=["В каком городе искать?"],
        request_summary=None,
        results=[],
        debug_notes=[],
    )

    payload = build_answer_payload(response)

    assert payload["need_clarification"] is True
    assert payload["questions"] == ["В каком городе искать?"]
    assert payload["results_count"] == 0
    assert payload["top_results"] == []


def test_build_answer_payload_uses_uncertain_reason_instead_of_field_name():
    response = NormalizedSearchResponse(
        need_clarification=False,
        questions=[],
        request_summary=NormalizedRequestSummary(
            city="Baku",
            check_in="2026-04-08",
            check_out="2026-04-15",
            constraints=[
                {
                    "raw_text": "pet friendly",
                    "normalized_text": "pet_friendly",
                    "priority": "must",
                    "category": "policy",
                    "mapping_status": "known",
                    "mapped_fields": ["pet_friendly"],
                    "evidence_strategy": "structured",
                },
                {
                    "raw_text": "kitchen",
                    "normalized_text": "kitchen",
                    "priority": "must",
                    "category": "amenity",
                    "mapping_status": "known",
                    "mapped_fields": ["kitchen"],
                    "evidence_strategy": "structured",
                },
            ],
            property_types=["apartment"],
            occupancy_types=[],
            filters={},
        ),
        results=[
            NormalizedSearchResult(
                result_id="apt1",
                title="Test Apartment",
                url="https://example.com/test",
                score=10.0,
                matched_constraints=[
                    ConstraintStatus(
                        name="kitchen",
                        status="matched",
                        reason="Kitchen",
                    )
                ],
                uncertain_constraints=[
                    ConstraintStatus(
                        name="pet_friendly",
                        status="uncertain",
                        reason="Pet policy is not explicitly confirmed in the listing.",
                    )
                ],
                failed_constraints=[],
                facts=[],
                why=["PET_FRIENDLY: maybe (needs check)", "KITCHEN: Kitchen"],
            )
        ],
        debug_notes=[],
    )

    payload = build_answer_payload(response, top_k=3)
    first = payload["top_results"][0]

    assert first["uncertain_points"] == [
        "Pet policy is not explicitly confirmed in the listing."
    ]


def test_build_answer_payload_includes_constraint_resolution_results():
    response = NormalizedSearchResponse(
        need_clarification=False,
        questions=[],
        request_summary=NormalizedRequestSummary(
            city="Baku",
            check_in="2026-04-08",
            check_out="2026-04-15",
            constraints=[
                {
                    "raw_text": "iron",
                    "normalized_text": "iron",
                    "priority": "must",
                    "category": "amenity",
                    "mapping_status": "known",
                    "mapped_fields": ["iron"],
                    "evidence_strategy": "structured",
                },
                {
                    "raw_text": "satellite TV",
                    "normalized_text": "satellite TV",
                    "priority": "must",
                    "category": "amenity",
                    "mapping_status": "unresolved",
                    "mapped_fields": [],
                    "evidence_strategy": "textual",
                },
            ],
            property_types=["apartment"],
            occupancy_types=[],
            filters={},
        ),
        results=[
            NormalizedSearchResult(
                result_id="apt1",
                title="Compact Apartment",
                url="https://example.com/compact",
                score=10.0,
                matched_constraints=[
                    ConstraintStatus(
                        name="iron",
                        status="matched",
                        reason="Ironing facilities",
                    )
                ],
                uncertain_constraints=[],
                failed_constraints=[],
                constraint_resolution_results=[
                    ConstraintResolutionItem(
                        listing_id="listing-1",
                        listing_title="Example listing",
                        constraint_id="c1",
                        raw_text="satellite TV",
                        normalized_text="satellite TV",
                        resolver_type="textual",
                        decision="YES",
                        resolution_status="matched",
                        reason="Satellite TV is mentioned.",
                        evidence=[
                            ConstraintResolutionEvidence(
                                source="description",
                                path="description",
                                snippet="Satellite TV available",
                            )
                        ],
                    )
                ],
                facts=[],
                why=[],
            )
        ],
        debug_notes=[],
    )

    payload = build_answer_payload(response, top_k=3)
    first = payload["top_results"][0]

    assert "constraint_resolution_results" in first
    assert first["constraint_resolution_results"]
    assert first["constraint_resolution_results"][0]["normalized_text"] == "satellite TV"


def test_build_answer_payload_includes_ranking_reasons_and_standout_reason():
    response = NormalizedSearchResponse(
        need_clarification=False,
        questions=[],
        request_summary=NormalizedRequestSummary(
            city="Baku",
            check_in="2026-04-08",
            check_out="2026-04-15",
            constraints=[
                {
                    "raw_text": "iron",
                    "normalized_text": "iron",
                    "priority": "must",
                    "category": "amenity",
                    "mapping_status": "known",
                    "mapped_fields": ["iron"],
                    "evidence_strategy": "structured",
                },
                {
                    "raw_text": "satellite TV",
                    "normalized_text": "satellite TV",
                    "priority": "must",
                    "category": "amenity",
                    "mapping_status": "unresolved",
                    "mapped_fields": [],
                    "evidence_strategy": "textual",
                },
            ],
            property_types=["apartment"],
            occupancy_types=[],
            filters={},
        ),
        results=[
            NormalizedSearchResult(
                result_id="apt1",
                title="Compact Apartment",
                url="https://example.com/compact",
                score=10.0,
                matched_constraints=[
                    ConstraintStatus(
                        name="iron",
                        status="matched",
                        reason="Ironing facilities",
                    )
                ],
                uncertain_constraints=[],
                failed_constraints=[],
                constraint_resolution_results=[
                    ConstraintResolutionItem(
                        listing_id="listing-1",
                        listing_title="Example listing",
                        constraint_id="c1",
                        raw_text="satellite TV",
                        normalized_text="satellite TV",
                        resolver_type="textual",
                        decision="YES",
                        resolution_status="matched",
                        reason="Satellite TV is mentioned.",
                        evidence=[
                            ConstraintResolutionEvidence(
                                source="description",
                                path="description",
                                snippet="Satellite TV available",
                            )
                        ],
                    )
                ],
                facts=[],
                why=[
                    "CONSTRAINT_RESOLUTION: satellite TV confirmed",
                    "PROPERTY_TYPE: matched apartment",
                ],
            )
        ],
        debug_notes=[],
    )

    payload = build_answer_payload(response, top_k=3)
    first = payload["top_results"][0]

    assert first["ranking_reasons"]
    assert "matched apartment" in first["ranking_reasons"][0]
    assert first["standout_reason"] is None