from app.logic.answer_generation import build_user_answer


def test_build_user_answer_for_clarification():
    payload = {
        "need_clarification": True,
        "questions": ["В каком городе искать?"],
        "request_summary": None,
        "results_count": 0,
        "top_results": [],
    }

    out = build_user_answer(payload)

    assert "В каком городе искать?" in out


def test_build_user_answer_for_results():
    payload = {
        "need_clarification": False,
        "questions": [],
        "request_summary": {
            "city": "Baku",
            "check_in": "2026-04-08",
            "check_out": "2026-04-15",
            "constraints": [
                {
                    "raw_text": "kitchen",
                    "normalized_text": "kitchen",
                    "priority": "must",
                    "category": "amenity",
                    "mapping_status": "known",
                    "mapped_fields": ["kitchen"],
                    "evidence_strategy": "structured",
                }
            ],
            "property_types": ["apartment"],
            "occupancy_types": [],
            "filters": {},
        },
        "results_count": 1,
        "top_results": [
            {
                "result_id": "abc123",
                "title": "Apartment STEL",
                "url": "https://example.com/stel",
                "score": 23.0,
                "answer_explanation": {
                    "status_label": "partially_confirmed_match",
                    "status_text": "Partially confirmed match",
                    "confirmed": [
                        {
                            "name": "kitchen",
                            "label": "Kitchen",
                            "reason": "Private kitchen",
                        }
                    ],
                    "needs_confirmation": [
                        {
                            "name": "price_total",
                            "label": "Budget",
                            "reason": "PRICE: currency mismatch listing=USD, request=AZN",
                        }
                    ],
                    "not_satisfied": [],
                },
                "price_summary": None,
            }
        ],
        "debug_notes": [],
    }

    out = build_user_answer(payload)

    assert "I found 1 option(s) that match your requirements in Baku for 2026-04-08 to 2026-04-15." in out
    assert "Apartment STEL" in out
    assert "Partially confirmed match" in out
    assert "Matches:" in out
    assert "- Kitchen — Private kitchen" in out
    assert "Budget" not in out
    assert "Summary:" not in out
    assert "Confirmed:" not in out
    assert "Why it matches:" not in out
    assert "Ranking signals:" not in out

    assert "Link: https://example.com/stel" in out


def test_build_user_answer_for_multiple_questions():
    payload = {
        "need_clarification": True,
        "questions": ["В каком городе искать?", "Какие даты заезда и выезда?"],
        "request_summary": None,
        "results_count": 0,
        "top_results": [],
    }

    out = build_user_answer(payload)

    assert "I need a few more details:" in out
    assert "В каком городе искать?" in out
    assert "Какие даты заезда и выезда?" in out


def test_build_user_answer_omits_raw_constraint_resolution_details():
    payload = {
        "need_clarification": False,
        "questions": [],
        "active_intent": {
            "city": "Baku",
            "check_in": "2026-04-08",
            "check_out": "2026-04-15",
            "constraints": [
                {
                    "raw_text": "ironing facilities",
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
            "property_types": ["apartment"],
            "occupancy_types": [],
            "filters": {},
        },
        "results_count": 1,
        "top_results": [
            {
                "title": "Compact Apartment",
                "url": "https://example.com/compact",
                "fit_summary": "Matches all required criteria.",
                "price_summary": None,
                "budget_summary": None,
                "key_facts_summary": "type: apartment",
                "why_match": ["Ironing facilities"],
                "tradeoffs": [],
                "uncertain_points": [],
                "constraint_resolution_results": [
                    {
                        "normalized_text": "satellite TV",
                        "decision": "YES",
                        "resolution_status": "matched",
                        "reason": "Found in description",
                    }
                ],
            }
        ],
    }

    out = build_user_answer(payload)

    assert "Compact Apartment" in out
    assert "satellite tv" not in out.lower()
    assert "match your requirements" in out.lower()


def test_build_user_answer_uses_clean_summary_without_internal_ranking_details():
    payload = {
        "need_clarification": False,
        "questions": [],
        "active_intent": {
            "city": "Baku",
            "check_in": "2026-04-08",
            "check_out": "2026-04-15",
            "constraints": [
                {
                    "raw_text": "ironing facilities",
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
            "property_types": ["apartment"],
            "occupancy_types": [],
            "filters": {},
        },
        "results_count": 1,
        "top_results": [
            {
                "title": "Compact Apartment",
                "url": "https://example.com/compact",
                "fit_summary": "Strong overall match.",
                "price_summary": None,
                "budget_summary": None,
                "key_facts_summary": "type: apartment",
                "why_match": ["Ironing facilities"],
                "tradeoffs": [],
                "uncertain_points": [],
                "constraint_resolution_results": [
                    {
                        "normalized_text": "satellite TV",
                        "decision": "YES",
                        "resolution_status": "matched",
                        "reason": "Found in description",
                    }
                ],
                "ranking_reasons": ["satellite TV found"],
                "standout_reason": "Explicitly matches your requested detail: satellite TV",
            }
        ],
    }

    out = build_user_answer(payload)

    assert "Compact Apartment" in out
    assert "satellite tv" not in out.lower()
    assert "ranking" not in out.lower()
    assert "standout" not in out.lower()