from datetime import date

from app.logic import conversation_response_generator
from app.logic.conversation_response_generator import (
    generate_deterministic_conversation_response,
)
from app.schemas.conversation_response import (
    ClarificationConversationOutcome,
    ConversationFailureOutcome,
    ConversationResponseInput,
    GeneralChatConversationOutcome,
    InformationalConversationOutcome,
    ListingQueryConversationOutcome,
    ListingQueryResponseItem,
    SearchConversationOutcome,
)
from app.schemas.conversation_route import ConversationAction
from app.schemas.query import SearchRequest
from app.schemas.result_query_match import ConstraintMatchResult, ResultQueryMatch
from app.schemas.match import Ternary
from app.schemas.search_response import (
    NormalizedRequestSummary,
    NormalizedSearchResponse,
    NormalizedSearchResult,
    SearchStatus,
)


def _item(result_id: str, position: int, title: str | None, value: Ternary, field: str = "parking") -> ListingQueryResponseItem:
    return ListingQueryResponseItem(
        position=position,
        title=title,
        match=ResultQueryMatch(
            result_id=result_id,
            constraint_results=[
                ConstraintMatchResult(
                    constraint_id=f"c-{result_id}",
                    raw_text="parking",
                    field=field,
                    value=value,
                    evidence=[],
                )
            ],
        ),
    )


def _multi_constraint_item(result_id: str, position: int, title: str | None, values: dict[str, Ternary]) -> ListingQueryResponseItem:
    return ListingQueryResponseItem(
        position=position,
        title=title,
        match=ResultQueryMatch(
            result_id=result_id,
            constraint_results=[
                ConstraintMatchResult(
                    constraint_id=f"c-{result_id}-{field}",
                    raw_text=field,
                    field=field,
                    value=value,
                    evidence=[],
                )
                for field, value in values.items()
            ],
        ),
    )


def test_listing_query_yes_is_affirmative():
    response_input = ConversationResponseInput(
        user_message="Does it have parking?",
        action=ConversationAction.LISTING_QUESTION,
        outcome=ListingQueryConversationOutcome(
            items=[_item("H1", 1, "Hilton Baku", Ternary.YES)]
        ),
    )

    answer = generate_deterministic_conversation_response(response_input)

    assert answer == "Hilton Baku has parking."


def test_listing_query_no_is_explicit_negative():
    response_input = ConversationResponseInput(
        user_message="Does it have parking?",
        action=ConversationAction.LISTING_QUESTION,
        outcome=ListingQueryConversationOutcome(
            items=[_item("H1", 1, "Hilton Baku", Ternary.NO)]
        ),
    )

    answer = generate_deterministic_conversation_response(response_input)

    assert answer == "Hilton Baku does not have parking."
    assert "has parking" not in answer


def test_listing_query_uncertain_is_inability_to_verify_not_no():
    response_input = ConversationResponseInput(
        user_message="Does it have parking?",
        action=ConversationAction.LISTING_QUESTION,
        outcome=ListingQueryConversationOutcome(
            items=[_item("H1", 1, "Hilton Baku", Ternary.UNCERTAIN)]
        ),
    )

    answer = generate_deterministic_conversation_response(response_input)

    assert answer == "I couldn't verify parking for Hilton Baku from the listing information."
    assert "does not have" not in answer
    assert "has parking" not in answer


def test_listing_query_current_results_preserves_shown_order():
    response_input = ConversationResponseInput(
        user_message="Which of these have parking?",
        action=ConversationAction.LISTING_QUESTION,
        outcome=ListingQueryConversationOutcome(
            items=[
                _item("H1", 1, None, Ternary.YES),
                _item("H2", 2, None, Ternary.NO),
                _item("H3", 3, None, Ternary.UNCERTAIN),
            ]
        ),
    )

    answer = generate_deterministic_conversation_response(response_input)
    lines = answer.split("\n")

    assert lines == [
        "the first hotel has parking.",
        "the second hotel does not have parking.",
        "I couldn't verify parking for the third hotel from the listing information.",
    ]


def test_listing_query_multiple_constraints_rendered_independently_no_aggregation():
    response_input = ConversationResponseInput(
        user_message="Which of these have parking and wifi?",
        action=ConversationAction.LISTING_QUESTION,
        outcome=ListingQueryConversationOutcome(
            items=[
                _multi_constraint_item(
                    "H1", 1, "Hotel A", {"parking": Ternary.YES, "wifi": Ternary.YES}
                ),
                _multi_constraint_item(
                    "H2", 2, "Hotel B", {"parking": Ternary.NO, "wifi": Ternary.YES}
                ),
            ]
        ),
    )

    answer = generate_deterministic_conversation_response(response_input)

    assert "- parking: yes" in answer
    assert "- wifi: yes" in answer
    assert "- parking: no" in answer
    for forbidden in ("overall", "only", "all requirements", "matches your requirements"):
        assert forbidden not in answer.lower()


def test_informational_outcome_update_no_op_text():
    response_input = ConversationResponseInput(
        user_message="No changes",
        action=ConversationAction.UPDATE_SEARCH,
        outcome=InformationalConversationOutcome(code="update_no_op"),
    )

    answer = generate_deterministic_conversation_response(response_input)

    assert answer == "Your search hasn't changed, so I didn't run a new search."


def test_informational_outcome_listing_query_unsupported_text():
    response_input = ConversationResponseInput(
        user_message="Is it romantic?",
        action=ConversationAction.LISTING_QUESTION,
        outcome=InformationalConversationOutcome(code="listing_query_unsupported"),
    )

    answer = generate_deterministic_conversation_response(response_input)

    assert answer == "I can't check that from the information in the shown listings."


def test_generates_general_chat_response():
    response_input = ConversationResponseInput(
        user_message="Hello",
        action=ConversationAction.GENERAL_CHAT,
        outcome=GeneralChatConversationOutcome(),
    )

    answer = generate_deterministic_conversation_response(
        response_input
    )

    assert "help you search for accommodation" in answer


def test_generates_clarification_from_domain_questions():
    response_input = ConversationResponseInput(
        user_message="Find me somewhere nice",
        action=ConversationAction.START_SEARCH,
        outcome=ClarificationConversationOutcome(
            questions=[
                "Which city would you like to stay in?"
            ]
        ),
    )

    answer = generate_deterministic_conversation_response(
        response_input
    )

    assert answer == "Which city would you like to stay in?"


def test_search_response_uses_current_search_as_domain_truth():
    current_search = SearchRequest(
        city="Baku",
        check_in=date(2026, 9, 10),
        check_out=date(2026, 9, 15),
    )

    search_response = NormalizedSearchResponse(
        status=SearchStatus.NO_RESULTS,
        request_summary=NormalizedRequestSummary(
            city="Paris",
            check_in="2026-09-10",
            check_out="2026-09-15",
        ),
        results=[],
    )

    response_input = ConversationResponseInput(
        user_message="Find me something in Baku",
        action=ConversationAction.START_SEARCH,
        current_search=current_search,
        outcome=SearchConversationOutcome(
            search_response=search_response
        ),
    )

    answer = generate_deterministic_conversation_response(
        response_input
    )

    assert "Baku" in answer
    assert "Paris" not in answer


def test_generates_safe_failure_message_without_rewriting_it():
    response_input = ConversationResponseInput(
        user_message="Make it cheaper",
        action=ConversationAction.UPDATE_SEARCH,
        outcome=ConversationFailureOutcome(
            user_safe_message=(
                "I couldn't complete that request right now."
            )
        ),
    )

    answer = generate_deterministic_conversation_response(
        response_input
    )

    assert (
        answer
        == "I couldn't complete that request right now."
    )


def _search_response_with_n_results(n: int) -> NormalizedSearchResponse:
    return NormalizedSearchResponse(
        status=SearchStatus.RESULTS,
        results=[
            NormalizedSearchResult(
                result_id=f"listing-{i}",
                title=f"Listing {i}",
                score=0.0,
            )
            for i in range(n)
        ],
    )


def test_deterministic_response_uses_all_results_by_default(
    monkeypatch,
):
    """
    NormalizedSearchResponse.results is already the final shown list -
    the deterministic fallback must not independently re-truncate it to
    its old hardcoded default of 3.
    """
    search_response = _search_response_with_n_results(5)

    response_input = ConversationResponseInput(
        user_message="Find me somewhere nice",
        action=ConversationAction.START_SEARCH,
        outcome=SearchConversationOutcome(
            search_response=search_response
        ),
    )

    captured_top_k = {}

    def fake_build_answer_payload(response, *, latest_user_query, top_k):
        captured_top_k["value"] = top_k
        return {
            "need_clarification": False,
            "questions": [],
            "top_results": [],
            "results_count": len(response.results),
            "active_intent": None,
        }

    monkeypatch.setattr(
        conversation_response_generator,
        "build_answer_payload",
        fake_build_answer_payload,
    )

    generate_deterministic_conversation_response(response_input)

    assert captured_top_k["value"] == len(search_response.results)
    assert captured_top_k["value"] != 3


def test_deterministic_response_explicit_top_k_still_overrides(
    monkeypatch,
):
    search_response = _search_response_with_n_results(5)

    response_input = ConversationResponseInput(
        user_message="Find me somewhere nice",
        action=ConversationAction.START_SEARCH,
        outcome=SearchConversationOutcome(
            search_response=search_response
        ),
    )

    captured_top_k = {}

    def fake_build_answer_payload(response, *, latest_user_query, top_k):
        captured_top_k["value"] = top_k
        return {
            "need_clarification": False,
            "questions": [],
            "top_results": [],
            "results_count": len(response.results),
            "active_intent": None,
        }

    monkeypatch.setattr(
        conversation_response_generator,
        "build_answer_payload",
        fake_build_answer_payload,
    )

    generate_deterministic_conversation_response(
        response_input,
        top_k=2,
    )

    assert captured_top_k["value"] == 2