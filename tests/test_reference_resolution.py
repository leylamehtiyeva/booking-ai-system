"""
Tests for the deterministic layer of the reference-resolution /
abstention experiment: turning the router's own (unvalidated)
decision_status / result_scope / target_result_id into either an
executed action or a read-only clarification turn.

These tests mock route_conversation_async and inject a crafted
ConversationActionDecision directly - they exercise the deterministic
CODE, not the LLM's own language understanding (English vs. Russian
phrasing produces identical code behavior once the decision object is
the same; real router language quality belongs in a live-LLM eval, not
here).
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from app.logic import conversation_flow
from app.schemas.conversation_route import (
    ClarificationReason,
    ConversationAction,
    ConversationActionDecision,
    ConversationDecisionStatus,
    ResultScope,
)
from app.schemas.listing import ListingRaw
from app.schemas.query import SearchRequest
from app.schemas.shown_result_set import ShownResult, ShownResultSet


def make_shown_result_set(*listings: tuple[str, str]) -> ShownResultSet:
    """listings: sequence of (result_id, title) pairs, in display order."""
    return ShownResultSet(
        result_set_id="rs-test",
        items=[
            ShownResult(
                result_id=result_id,
                listing=ListingRaw(id=result_id, name=title, city="Baku"),
            )
            for result_id, title in listings
        ],
    )


def fake_route(decision: ConversationActionDecision):
    async def _fake_route(**kwargs):
        return decision

    return _fake_route


def make_decision(
    *,
    action: ConversationAction,
    reason: str,
    decision_status: ConversationDecisionStatus = ConversationDecisionStatus.RESOLVED,
    clarification_reason: ClarificationReason = ClarificationReason.NONE,
    result_scope: ResultScope = ResultScope.NOT_APPLICABLE,
    target_result_id: str = "",
) -> ConversationActionDecision:
    """Every ConversationActionDecision field is required (no Python
    default) so it can pass Groq's strict structured-output schema - this
    factory fills in the "nothing special happened" values by default."""
    return ConversationActionDecision(
        action=action,
        reason=reason,
        decision_status=decision_status,
        clarification_reason=clarification_reason,
        result_scope=result_scope,
        target_result_id=target_result_id,
    )


@pytest.mark.asyncio
async def test_update_search_response_has_no_reference_fields(monkeypatch):
    """
    "Only with parking" against an existing search must stay a plain
    UPDATE_SEARCH - no result_scope/target_result_id leak into the
    response for the persistent-refinement path.
    """
    previous_state = SearchRequest(
        city="Baku",
        check_in=None,
        check_out=None,
        constraints=[],
    )

    async def fake_update(prev_state, msg, *, trace=None):
        return prev_state

    monkeypatch.setattr(
        conversation_flow,
        "route_conversation_async",
        fake_route(
            make_decision(
                action=ConversationAction.UPDATE_SEARCH,
                reason="The user adds a search requirement.",
            )
        ),
    )
    monkeypatch.setattr(
        conversation_flow,
        "update_search_state_async",
        fake_update,
    )

    async def fake_orchestrate(*args, **kwargs):
        from app.schemas.search_response import (
            NormalizedSearchResponse,
            SearchStatus,
        )

        return (
            NormalizedSearchResponse(
                status=SearchStatus.NO_RESULTS,
                results=[],
            ),
            ShownResultSet(result_set_id="rs-x", items=[]),
        )

    monkeypatch.setattr(
        conversation_flow,
        "orchestrate_search_request",
        fake_orchestrate,
    )

    result = await conversation_flow.handle_user_message(
        "Only with parking",
        previous_state=previous_state,
        shown_result_set=make_shown_result_set(
            ("H1", "Hilton Baku"), ("H2", "Marriott Baku")
        ),
    )

    assert "result_scope" not in result
    assert "target_result_id" not in result


@pytest.mark.asyncio
async def test_listing_question_current_results_has_no_target(monkeypatch):
    """
    "Which of these have parking?" / "В каких из этих есть парковка?"
    -> LISTING_QUESTION + CURRENT_RESULTS, no single target expected.
    """
    monkeypatch.setattr(
        conversation_flow,
        "route_conversation_async",
        fake_route(
            make_decision(
                action=ConversationAction.LISTING_QUESTION,
                reason="The user asks about the whole shown set.",
                result_scope=ResultScope.CURRENT_RESULTS,
            )
        ),
    )

    result = await conversation_flow.handle_user_message(
        "Which of these have parking?",
        shown_result_set=make_shown_result_set(
            ("H1", "Hilton Baku"), ("H2", "Marriott Baku")
        ),
    )

    assert result["result_scope"] == "current_results"
    assert result["target_result_id"] is None
    assert result["reference_resolution_status"] == "not_applicable"


@pytest.mark.asyncio
async def test_listing_question_specific_result_resolves_valid_id(
    monkeypatch,
):
    """
    Covers both "Does the second one have parking?" / "А во втором есть
    парковка?" (ordinal) and "What about Marriott?" / "А у Marriott есть
    балкон?" (name) - at the validation layer both simply mean the
    router already resolved to Marriott's result_id "H2".
    """
    monkeypatch.setattr(
        conversation_flow,
        "route_conversation_async",
        fake_route(
            make_decision(
                action=ConversationAction.LISTING_QUESTION,
                reason="The user asks about the second shown listing.",
                result_scope=ResultScope.SPECIFIC_RESULT,
                target_result_id="H2",
            )
        ),
    )

    result = await conversation_flow.handle_user_message(
        "Does the second one have parking?",
        shown_result_set=make_shown_result_set(
            ("H1", "Hilton Baku"), ("H2", "Marriott Baku")
        ),
    )

    assert result["result_scope"] == "specific_result"
    assert result["target_result_id"] == "H2"
    assert result["reference_resolution_status"] == "resolved"


@pytest.mark.asyncio
async def test_router_abstains_on_ambiguous_reference(monkeypatch):
    """
    "Does it have parking?" with two shown results and no other context -
    the router itself abstains (decision_status=needs_clarification)
    instead of guessing. Must produce a read-only clarification turn
    naming both shown hotels, not a listing_question answer.
    """
    monkeypatch.setattr(
        conversation_flow,
        "route_conversation_async",
        fake_route(
            make_decision(
                action=ConversationAction.LISTING_QUESTION,
                reason="Pronoun reference is ambiguous among two shown listings.",
                decision_status=ConversationDecisionStatus.NEEDS_CLARIFICATION,
                clarification_reason=ClarificationReason.AMBIGUOUS_REFERENCE,
                result_scope=ResultScope.SPECIFIC_RESULT,
            )
        ),
    )

    result = await conversation_flow.handle_user_message(
        "Does it have parking?",
        shown_result_set=make_shown_result_set(
            ("H1", "Hilton Baku"), ("H2", "Marriott Baku")
        ),
    )

    assert result["need_clarification"] is True
    assert "Hilton Baku" in result["questions"][0]
    assert "Marriott Baku" in result["questions"][0]
    assert "target_result_id" not in result
    assert "result_scope" not in result


@pytest.mark.asyncio
async def test_router_abstains_on_ambiguous_action(monkeypatch):
    """
    "Only the ones with parking?" could mean update_search or
    listing_question/current_results - the router must abstain rather
    than silently guess the persistent mutation.
    """
    previous_state = SearchRequest(city="Baku", constraints=[])

    update_mock = AsyncMock()
    orchestrate_mock = AsyncMock()

    monkeypatch.setattr(
        conversation_flow,
        "route_conversation_async",
        fake_route(
            make_decision(
                action=ConversationAction.UPDATE_SEARCH,
                reason="Could be a persistent filter or a question about shown results.",
                decision_status=ConversationDecisionStatus.NEEDS_CLARIFICATION,
                clarification_reason=ClarificationReason.AMBIGUOUS_ACTION,
            )
        ),
    )
    monkeypatch.setattr(
        conversation_flow, "update_search_state_async", update_mock
    )
    monkeypatch.setattr(
        conversation_flow, "orchestrate_search_request", orchestrate_mock
    )

    result = await conversation_flow.handle_user_message(
        "Only the ones with parking?",
        previous_state=previous_state,
        shown_result_set=make_shown_result_set(
            ("H1", "Hilton Baku"), ("H2", "Marriott Baku")
        ),
    )

    assert result["need_clarification"] is True
    assert result["questions"]

    update_mock.assert_not_awaited()
    orchestrate_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_abstention_is_read_only(monkeypatch):
    """
    Architectural invariant: a clarification turn must never call
    build_search_request_adk_async, update_search_state_async or
    orchestrate_search_request, must not mutate SearchRequest, and must
    not include a "shown_result_set" key (so chat_handler.py leaves the
    stored snapshot untouched).
    """
    previous_state = SearchRequest(city="Baku", constraints=[])
    previous_shown = make_shown_result_set(
        ("H1", "Hilton Baku"), ("H2", "Marriott Baku")
    )

    build_search_mock = AsyncMock()
    update_mock = AsyncMock()
    orchestrate_mock = AsyncMock()

    monkeypatch.setattr(
        conversation_flow,
        "route_conversation_async",
        fake_route(
            make_decision(
                action=ConversationAction.LISTING_QUESTION,
                reason="Ambiguous pronoun.",
                decision_status=ConversationDecisionStatus.NEEDS_CLARIFICATION,
                clarification_reason=ClarificationReason.AMBIGUOUS_REFERENCE,
                result_scope=ResultScope.SPECIFIC_RESULT,
            )
        ),
    )
    monkeypatch.setattr(
        conversation_flow, "build_search_request_adk_async", build_search_mock
    )
    monkeypatch.setattr(
        conversation_flow, "update_search_state_async", update_mock
    )
    monkeypatch.setattr(
        conversation_flow, "orchestrate_search_request", orchestrate_mock
    )

    result = await conversation_flow.handle_user_message(
        "Does it have parking?",
        previous_state=previous_state,
        shown_result_set=previous_shown,
    )

    build_search_mock.assert_not_awaited()
    update_mock.assert_not_awaited()
    orchestrate_mock.assert_not_awaited()

    assert result["need_clarification"] is True
    assert result["state"] == previous_state.model_dump(
        mode="json", exclude_none=True
    )
    assert "shown_result_set" not in result


@pytest.mark.asyncio
async def test_listing_question_specific_result_single_shown_result(
    monkeypatch,
):
    """
    "Does it have parking?" with exactly one shown result - the router
    is expected to resolve the pronoun to it (decision_status stays
    resolved); the deterministic layer just validates that the returned
    id matches, it does not special-case count == 1 itself.
    """
    monkeypatch.setattr(
        conversation_flow,
        "route_conversation_async",
        fake_route(
            make_decision(
                action=ConversationAction.LISTING_QUESTION,
                reason="Only one listing was shown.",
                result_scope=ResultScope.SPECIFIC_RESULT,
                target_result_id="H1",
            )
        ),
    )

    result = await conversation_flow.handle_user_message(
        "Does it have parking?",
        shown_result_set=make_shown_result_set(("H1", "Hilton Baku")),
    )

    assert result["target_result_id"] == "H1"
    assert result["reference_resolution_status"] == "resolved"


@pytest.mark.asyncio
async def test_listing_question_hallucinated_id_triggers_clarification(
    monkeypatch,
):
    """
    LLM is not the source of truth for identity - a target_result_id
    that does not exist in the actual latest ShownResultSet must be
    rejected outright (never guessed or looked up) and must fall back to
    a read-only clarification turn, not a listing_question answer.
    """
    monkeypatch.setattr(
        conversation_flow,
        "route_conversation_async",
        fake_route(
            make_decision(
                action=ConversationAction.LISTING_QUESTION,
                reason="Hallucinated reference.",
                result_scope=ResultScope.SPECIFIC_RESULT,
                target_result_id="H999",
            )
        ),
    )

    result = await conversation_flow.handle_user_message(
        "Does the second one have parking?",
        shown_result_set=make_shown_result_set(
            ("H1", "Hilton Baku"), ("H2", "Marriott Baku")
        ),
    )

    assert result["need_clarification"] is True
    assert "target_result_id" not in result


@pytest.mark.asyncio
async def test_listing_question_reference_without_any_shown_result_set(
    monkeypatch,
):
    """
    Router hallucinates a target_result_id even though no ShownResultSet
    was ever passed in (e.g. nothing has been searched yet this session).
    """
    monkeypatch.setattr(
        conversation_flow,
        "route_conversation_async",
        fake_route(
            make_decision(
                action=ConversationAction.LISTING_QUESTION,
                reason="Hallucinated reference with no shown results.",
                result_scope=ResultScope.SPECIFIC_RESULT,
                target_result_id="H1",
            )
        ),
    )

    result = await conversation_flow.handle_user_message(
        "Does the second one have parking?",
        shown_result_set=None,
    )

    assert result["need_clarification"] is True
    assert "target_result_id" not in result


def test_shown_results_router_context_excludes_listing_details():
    """
    Critical invariant: the router must never see facilities/rooms/
    policies/raw - only position, result_id and title.
    """
    shown = ShownResultSet(
        result_set_id="rs-1",
        items=[
            ShownResult(
                result_id="H1",
                listing=ListingRaw(
                    id="H1",
                    name="Hilton Baku",
                    city="Baku",
                    description="A very nice hotel with a rooftop pool.",
                    facilities=[{"name": "Free WiFi"}],
                    policies=[
                        {"title": "Smoking", "content": "Not allowed."}
                    ],
                    rooms=[{"name": "Deluxe Room", "facilities": ["Balcony"]}],
                ),
            )
        ],
    )

    context = conversation_flow._build_shown_results_router_context(shown)

    assert context == {
        "has_shown_results": True,
        "shown_results": [
            {"position": 1, "result_id": "H1", "title": "Hilton Baku"}
        ],
    }

    serialized = json.dumps(context)
    assert "WiFi" not in serialized
    assert "Balcony" not in serialized
    assert "rooftop pool" not in serialized
    assert "Smoking" not in serialized
