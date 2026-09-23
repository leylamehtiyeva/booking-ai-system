from ui.formatters import build_display_answer


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