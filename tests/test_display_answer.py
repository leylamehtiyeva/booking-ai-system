from app.logic.display_answer import build_display_answer


def test_general_chat_answer_is_not_formatted_as_search_result():
    result = {
        "need_clarification": False,
        "response_type": "other",
        "answer": "Hello! How can I help?",
        "state": None,
    }

    answer, payload = build_display_answer(result)

    assert answer == "Hello! How can I help?"
    assert payload is None
    
    
def test_listing_question_answer_is_not_formatted_as_search_result():
    result = {
        "need_clarification": False,
        "response_type": "listing_question",
        "answer": "Yes, parking is listed for this property.",
    }

    answer, payload = build_display_answer(result)

    assert answer == "Yes, parking is listed for this property."
    assert payload is None
    
    
def test_routing_failure_answer_is_not_formatted_as_search_result():
    result = {
        "need_clarification": False,
        "response_type": "routing_unavailable",
        "answer": "I couldn't process that message right now.",
    }

    answer, payload = build_display_answer(result)

    assert answer == "I couldn't process that message right now."
    assert payload is None

def test_update_no_op_answer_is_passed_through_not_parsed_as_search():
    result = {
        "need_clarification": False,
        "response_type": "update_no_op",
        "answer": "Your search hasn't changed, so I didn't run a new search.",
        "state": {"city": "Baku"},
    }

    answer, payload = build_display_answer(result)

    assert answer == "Your search hasn't changed, so I didn't run a new search."
    assert payload is None
