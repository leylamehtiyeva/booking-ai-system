from datetime import date

import pytest

from app.schemas.fields import Field
from app.schemas.query import SearchRequest
from app.schemas.constraints import (
    UserConstraint,
    ConstraintPriority,
    ConstraintCategory,
    ConstraintMappingStatus,
    EvidenceStrategy,
)
from app.schemas.conversation_route import ConversationRouteDecision


def kitchen_constraint():
    return UserConstraint(
        raw_text="kitchen",
        normalized_text="kitchen",
        priority=ConstraintPriority.MUST,
        category=ConstraintCategory.AMENITY,
        mapping_status=ConstraintMappingStatus.KNOWN,
        mapped_fields=[Field.KITCHEN],
        evidence_strategy=EvidenceStrategy.STRUCTURED,
    )


def beds_constraint():
    return UserConstraint(
        raw_text="2 beds",
        normalized_text="2 beds",
        priority=ConstraintPriority.MUST,
        category=ConstraintCategory.LAYOUT,
        mapping_status=ConstraintMappingStatus.UNRESOLVED,
        mapped_fields=[],
        evidence_strategy=EvidenceStrategy.TEXTUAL,
    )


@pytest.mark.asyncio
async def test_conversation_flow_first_turn_builds_state_and_searches(monkeypatch):
    async def _fake_build_search_request(user_message: str, trace=None, step=None) -> SearchRequest:
        return SearchRequest(
    city="Baku",
    check_in=date(2026, 4, 20),
    check_out=date(2026, 4, 25),
    constraints=[kitchen_constraint(), beds_constraint()],
)

    async def _fake_orchestrate_search(**kwargs):
        return {
            "need_clarification": False,
            "results": [{"title": "Large Family Apartment"}],
        }

    monkeypatch.setattr(
        "app.logic.conversation_flow.build_search_request_adk_async",
        _fake_build_search_request,
    )
    monkeypatch.setattr(
        "app.logic.conversation_flow.orchestrate_search",
        _fake_orchestrate_search,
    )

    from app.logic.conversation_flow import handle_user_message

    out = await handle_user_message("any query")

    assert out["need_clarification"] is False
    assert out["results"][0]["title"] == "Large Family Apartment"
    assert out["state"]["city"] == "Baku"
    assert out["state"]["constraints"]


@pytest.mark.asyncio
async def test_conversation_flow_followup_updates_existing_state(monkeypatch):
    previous_state = SearchRequest(
        city="Baku",
        check_in=date(2026, 4, 20),
        check_out=date(2026, 4, 26),
        constraints=[kitchen_constraint()],
    )

    async def _fake_route(**kwargs):
        return ConversationRouteDecision(route="search_update")

    async def _fake_update(prev_state, msg, trace=None):
        return SearchRequest(
            city=prev_state.city,
            check_in=prev_state.check_in,
            check_out=prev_state.check_out,
            constraints=[kitchen_constraint()],
        )

    async def _fake_orchestrate_search(**kwargs):
        return {"need_clarification": False, "results": [{"title": "OK"}]}

    monkeypatch.setattr("app.logic.conversation_flow.route_conversation_async", _fake_route)
    monkeypatch.setattr("app.logic.conversation_flow.update_search_state_async", _fake_update)
    monkeypatch.setattr("app.logic.conversation_flow.orchestrate_search", _fake_orchestrate_search)

    from app.logic.conversation_flow import handle_user_message

    out = await handle_user_message("update", previous_state=previous_state)

    assert out["state"]["city"] == "Baku"
    assert out["state"]["constraints"]


@pytest.mark.asyncio
async def test_conversation_flow_listing_question_does_not_mutate_state(monkeypatch):
    previous_state = SearchRequest(
        city="Baku",
        constraints=[beds_constraint()],
    )

    async def _fake_route(**kwargs):
        return ConversationRouteDecision(route="listing_question")

    async def _fake_answer(**kwargs):
        return {
            "need_clarification": False,
            "response_type": "listing_question",
            "state": previous_state.model_dump(mode="json"),
        }

    monkeypatch.setattr("app.logic.conversation_flow.route_conversation_async", _fake_route)
    monkeypatch.setattr("app.logic.conversation_flow._answer_listing_question", _fake_answer)

    from app.logic.conversation_flow import handle_user_message

    out = await handle_user_message("question", previous_state=previous_state)

    assert out["response_type"] == "listing_question"
    assert out["state"]["constraints"]


@pytest.mark.asyncio
async def test_conversation_flow_new_search_rebuilds_state(monkeypatch):
    previous_state = SearchRequest(city="Baku")

    async def _fake_route(**kwargs):
        return ConversationRouteDecision(route="new_search")

    async def _fake_build(msg, trace=None, step=None):
        return SearchRequest(city="Paris", constraints=[])

    async def _fake_orchestrate(**kwargs):
        return {"need_clarification": False, "results": []}

    monkeypatch.setattr("app.logic.conversation_flow.route_conversation_async", _fake_route)
    monkeypatch.setattr("app.logic.conversation_flow.build_search_request_adk_async", _fake_build)
    monkeypatch.setattr("app.logic.conversation_flow.orchestrate_search", _fake_orchestrate)

    from app.logic.conversation_flow import handle_user_message

    out = await handle_user_message("new", previous_state=previous_state)

    assert out["state"]["city"] == "Paris"


@pytest.mark.asyncio
async def test_conversation_flow_other_route_returns_previous_state(monkeypatch):
    previous_state = SearchRequest(
        city="Baku",
        constraints=[beds_constraint()],
    )

    async def _fake_route(**kwargs):
        return ConversationRouteDecision(route="other")

    monkeypatch.setattr("app.logic.conversation_flow.route_conversation_async", _fake_route)

    from app.logic.conversation_flow import handle_user_message

    out = await handle_user_message("thanks", previous_state=previous_state)

    assert out["response_type"] == "other"
    assert out["state"]["constraints"]
    
    
from unittest.mock import AsyncMock

import pytest

from app.logic import conversation_flow
from app.logic.conversation_router import ConversationRoutingError


@pytest.mark.asyncio
async def test_routing_failure_does_not_change_search_state(
    monkeypatch,
):

    previous_state = SearchRequest(
        city="Baku",
        constraints=[beds_constraint()],
    )

    route_mock = AsyncMock(
        side_effect=ConversationRoutingError(
            code="invalid_response",
        )
    )
    update_mock = AsyncMock()
    search_mock = AsyncMock()

    monkeypatch.setattr(
        conversation_flow,
        "route_conversation_async",
        route_mock,
    )
    monkeypatch.setattr(
        conversation_flow,
        "update_search_state_async",
        update_mock,
    )
    monkeypatch.setattr(
        conversation_flow,
        "orchestrate_search",
        search_mock,
    )

    result = await conversation_flow.handle_user_message(
        user_message="Does this hotel allow pets?",
        previous_state=previous_state,
    )

    expected_state = previous_state.model_dump(
        mode="json",
        exclude_none=True,
    )

    assert result["response_type"] == "routing_unavailable"
    assert result["state"] == expected_state
    assert result["search_request"] == expected_state

    update_mock.assert_not_awaited()
    search_mock.assert_not_awaited()