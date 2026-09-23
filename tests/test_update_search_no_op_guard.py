"""
Tests for the UPDATE_SEARCH no-op guard: if applying the patch leaves
SearchRequest semantically unchanged (updated_search == previous_search),
the search pipeline (retrieval/matching/ranking/normalization) must not
run again, and the existing ShownResultSet must not be replaced.

The invariant is equality of the resulting SearchRequest, not "the patch
was empty" - a patch that only restates an already-true value collapses
to the same equality after apply_intent_patch's dedup/merge, and must be
caught the same way.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from app.logic import conversation_flow
from app.schemas.constraints import (
    ConstraintCategory,
    ConstraintMappingStatus,
    ConstraintPriority,
    EvidenceStrategy,
    UserConstraint,
)
from app.schemas.conversation_route import (
    ClarificationReason,
    ConversationAction,
    ConversationActionDecision,
    ConversationDecisionStatus,
    ResultScope,
)
from app.schemas.query import SearchRequest


def _resolved_kwargs() -> dict:
    return {
        "decision_status": ConversationDecisionStatus.RESOLVED,
        "clarification_reason": ClarificationReason.NONE,
        "result_scope": ResultScope.NOT_APPLICABLE,
        "target_result_id": "",
    }


def _fake_update_search_route():
    async def _fake_route(**kwargs):
        return ConversationActionDecision(
            action=ConversationAction.UPDATE_SEARCH,
            reason="The user updates the active search.",
            **_resolved_kwargs(),
        )

    return _fake_route


def _mock_search_pipeline(monkeypatch):
    """Patches build_search_request_adk_async/orchestrate_search_request
    as AsyncMocks so tests can assert whether the shared search tail ran
    at all, without needing a real retrieval/matching pipeline."""
    build_search_mock = AsyncMock()
    orchestrate_mock = AsyncMock()

    monkeypatch.setattr(
        conversation_flow, "build_search_request_adk_async", build_search_mock
    )
    monkeypatch.setattr(
        conversation_flow, "orchestrate_search_request", orchestrate_mock
    )

    return build_search_mock, orchestrate_mock


@pytest.mark.asyncio
async def test_empty_patch_does_not_run_search(monkeypatch):
    """
    previous SearchRequest + SearchIntentPatch() -> same SearchRequest
    -> orchestrate_search_request NOT called.
    """
    previous_state = SearchRequest(
        city="Baku",
        constraints=[],
    )

    async def fake_update(prev_state, msg, *, trace=None):
        # Mirrors what apply_intent_patch actually returns for an empty
        # patch: a new object, equal field values.
        return prev_state.model_copy(deep=True)

    monkeypatch.setattr(
        conversation_flow, "route_conversation_async", _fake_update_search_route()
    )
    monkeypatch.setattr(
        conversation_flow, "update_search_state_async", fake_update
    )
    _build_mock, orchestrate_mock = _mock_search_pipeline(monkeypatch)

    result = await conversation_flow.handle_user_message(
        "thanks, that's fine",
        previous_state=previous_state,
    )

    orchestrate_mock.assert_not_awaited()

    assert result["need_clarification"] is False
    assert result["response_type"] == "update_no_op"
    assert not result.get("questions")
    assert result["answer"]
    assert result["state"] == previous_state.model_dump(
        mode="json", exclude_none=True
    )


@pytest.mark.asyncio
async def test_semantically_identical_update_does_not_run_search(
    monkeypatch,
):
    """
    The patch restates an already-true requirement (parking=True when the
    search already requires parking=True) - apply_intent_patch's own
    dedup/merge collapses this back to an equal SearchRequest, which must
    be treated exactly like an empty patch.
    """
    parking_constraint = UserConstraint(
        raw_text="parking",
        normalized_text="parking",
        priority=ConstraintPriority.MUST,
        category=ConstraintCategory.AMENITY,
        mapping_status=ConstraintMappingStatus.UNRESOLVED,
        evidence_strategy=EvidenceStrategy.TEXTUAL,
    )

    previous_state = SearchRequest(
        city="Baku",
        constraints=[parking_constraint],
    )

    async def fake_update(prev_state, msg, *, trace=None):
        # Same constraint object/id, same every field - exactly what
        # apply_intent_patch produces when add_constraints repeats an
        # already-present requirement with identical wording.
        return SearchRequest(
            city=prev_state.city,
            constraints=[parking_constraint],
        )

    monkeypatch.setattr(
        conversation_flow, "route_conversation_async", _fake_update_search_route()
    )
    monkeypatch.setattr(
        conversation_flow, "update_search_state_async", fake_update
    )
    _build_mock, orchestrate_mock = _mock_search_pipeline(monkeypatch)

    result = await conversation_flow.handle_user_message(
        "still want parking",
        previous_state=previous_state,
    )

    orchestrate_mock.assert_not_awaited()
    assert result["need_clarification"] is False
    assert result["response_type"] == "update_no_op"


@pytest.mark.asyncio
async def test_real_update_still_runs_search(monkeypatch):
    """
    A genuine change (parking=False/absent -> parking=True) must still
    execute the search pipeline exactly as before this guard existed.
    """
    from app.schemas.search_response import (
        NormalizedSearchResponse,
        NormalizedSearchResult,
        SearchStatus,
    )
    from datetime import date

    from app.schemas.shown_result_set import ShownResultSet

    previous_state = SearchRequest(
        city="Baku",
        check_in=date(2026, 4, 10),
        check_out=date(2026, 4, 15),
        constraints=[],
    )

    parking_constraint = UserConstraint(
        raw_text="parking",
        normalized_text="parking",
        priority=ConstraintPriority.MUST,
        category=ConstraintCategory.AMENITY,
        mapping_status=ConstraintMappingStatus.UNRESOLVED,
        evidence_strategy=EvidenceStrategy.TEXTUAL,
    )

    async def fake_update(prev_state, msg, *, trace=None):
        return SearchRequest(
            city=prev_state.city,
            check_in=prev_state.check_in,
            check_out=prev_state.check_out,
            constraints=[parking_constraint],
        )

    async def fake_orchestrate(req, **kwargs):
        return (
            NormalizedSearchResponse(
                status=SearchStatus.RESULTS,
                results=[
                    NormalizedSearchResult(
                        result_id="ok-1", title="OK", score=0.0
                    )
                ],
            ),
            ShownResultSet(result_set_id="rs-new", items=[]),
        )

    monkeypatch.setattr(
        conversation_flow, "route_conversation_async", _fake_update_search_route()
    )
    monkeypatch.setattr(
        conversation_flow, "update_search_state_async", fake_update
    )
    monkeypatch.setattr(
        conversation_flow, "orchestrate_search_request", fake_orchestrate
    )

    result = await conversation_flow.handle_user_message(
        "only with parking",
        previous_state=previous_state,
    )

    assert result.get("need_clarification") is not True
    assert result["shown_result_set"]["result_set_id"] == "rs-new"
    assert result["state"]["constraints"]


@pytest.mark.asyncio
async def test_no_op_update_does_not_replace_shown_result_set(monkeypatch):
    """
    On a no-op update, the response must not carry a "shown_result_set"
    key at all - its absence is what keeps chat_handler.py from touching
    (replacing or clearing) the already-stored snapshot.
    """
    previous_state = SearchRequest(city="Baku", constraints=[])

    async def fake_update(prev_state, msg, *, trace=None):
        return prev_state.model_copy(deep=True)

    monkeypatch.setattr(
        conversation_flow, "route_conversation_async", _fake_update_search_route()
    )
    monkeypatch.setattr(
        conversation_flow, "update_search_state_async", fake_update
    )
    _build_mock, orchestrate_mock = _mock_search_pipeline(monkeypatch)

    result = await conversation_flow.handle_user_message(
        "ok cool",
        previous_state=previous_state,
    )

    orchestrate_mock.assert_not_awaited()
    assert "shown_result_set" not in result


def test_adapter_routes_no_op_to_informational_outcome_not_clarification():
    """
    conversation_response_adapter.py must not send an update_no_op result
    into ClarificationConversationOutcome (would be phrased as a question
    by the response LLM) or attempt NormalizedSearchResponse validation
    (would crash - there is no search_response shape). It also must not
    be typed as ConversationFailureOutcome - nothing failed here, the
    turn resolved successfully with no change - so it goes through
    InformationalConversationOutcome instead (see the response-layer
    unification task for why ConversationFailureOutcome was a semantic
    misuse for this case).
    """
    from app.logic.conversation_response_adapter import (
        build_conversation_response_input,
    )
    from app.logic.conversation_response_generator import (
        generate_deterministic_conversation_response,
    )
    from app.schemas.conversation_response import (
        ClarificationConversationOutcome,
        ConversationFailureOutcome,
        InformationalConversationOutcome,
    )

    no_op_result = {
        "conversation_action": "update_search",
        "need_clarification": False,
        "response_type": "update_no_op",
        "answer": "Your search hasn't changed, so I didn't run a new search.",
        "state": {"city": "Baku"},
    }

    response_input = build_conversation_response_input(
        user_message="thanks, that's fine",
        result=no_op_result,
    )

    assert isinstance(response_input.outcome, InformationalConversationOutcome)
    assert not isinstance(
        response_input.outcome, ClarificationConversationOutcome
    )
    assert not isinstance(
        response_input.outcome, ConversationFailureOutcome
    )
    assert response_input.outcome.code == "update_no_op"
    assert generate_deterministic_conversation_response(response_input) == (
        "Your search hasn't changed, so I didn't run a new search."
    )


def test_adapter_still_routes_real_clarification_normally():
    """
    Protects the pre-existing contract: a genuine missing-city/date
    clarification (need_clarification=True) must still produce
    ClarificationConversationOutcome, unaffected by the update_no_op path.
    """
    from app.logic.conversation_response_adapter import (
        build_conversation_response_input,
    )
    from app.schemas.conversation_response import (
        ClarificationConversationOutcome,
    )

    clarification_result = {
        "conversation_action": "update_search",
        "need_clarification": True,
        "questions": ["Which city should I search in?"],
        "state": None,
    }

    response_input = build_conversation_response_input(
        user_message="find me a hotel",
        result=clarification_result,
    )

    assert isinstance(
        response_input.outcome, ClarificationConversationOutcome
    )
    assert response_input.outcome.questions == [
        "Which city should I search in?"
    ]


def test_deterministic_response_returns_no_op_answer_verbatim():
    """
    generate_deterministic_conversation_response must return the fixed
    no-op text unchanged, not a generic canned message.
    """
    from app.logic.conversation_response_generator import (
        generate_deterministic_conversation_response,
    )
    from app.schemas.conversation_response import (
        ConversationFailureOutcome,
        ConversationResponseInput,
    )
    from app.schemas.conversation_route import ConversationAction

    response_input = ConversationResponseInput(
        user_message="thanks, that's fine",
        action=ConversationAction.UPDATE_SEARCH,
        outcome=ConversationFailureOutcome(
            user_safe_message=(
                "Your search hasn't changed, so I didn't run a new search."
            )
        ),
    )

    answer = generate_deterministic_conversation_response(response_input)

    assert answer == (
        "Your search hasn't changed, so I didn't run a new search."
    )


def test_e2e_infer_decision_no_longer_reports_uncertain_for_no_op():
    """
    The end-to-end eval harness's _infer_decision used to treat any
    need_clarification=True response as UNCERTAIN. With need_clarification
    now False for a no-op update, it must not be scored as UNCERTAIN.
    Only checks the existing pure function - does not touch or extend the
    e2e harness itself.
    """
    from evaluation.tasks.end_to_end.adapter import _infer_decision

    no_op_response = {
        "need_clarification": False,
        "response_type": "update_no_op",
        "answer": "Your search hasn't changed, so I didn't run a new search.",
        "results": [],
    }

    assert _infer_decision(no_op_response) != "UNCERTAIN"
