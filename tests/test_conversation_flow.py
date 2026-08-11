from datetime import date
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.logic import conversation_flow
from app.logic.conversation_router import ConversationRoutingError
from app.schemas.constraints import (
    ConstraintCategory,
    ConstraintMappingStatus,
    ConstraintPriority,
    EvidenceStrategy,
    UserConstraint,
)
from app.schemas.conversation_route import (
    ConversationAction,
    ConversationActionDecision,
)
from app.schemas.fields import Field
from app.schemas.query import SearchRequest


def kitchen_constraint() -> UserConstraint:
    return UserConstraint(
        raw_text="kitchen",
        normalized_text="kitchen",
        priority=ConstraintPriority.MUST,
        category=ConstraintCategory.AMENITY,
        mapping_status=ConstraintMappingStatus.KNOWN,
        mapped_fields=[Field.KITCHEN],
        evidence_strategy=EvidenceStrategy.STRUCTURED,
    )


def beds_constraint() -> UserConstraint:
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
async def test_conversation_flow_first_turn_builds_state_and_searches(
    monkeypatch,
):
    async def _fake_build_search_request(
        user_message: str,
        trace=None,
        step=None,
    ) -> SearchRequest:
        return SearchRequest(
            city="Baku",
            check_in=date(2026, 4, 20),
            check_out=date(2026, 4, 25),
            constraints=[
                kitchen_constraint(),
                beds_constraint(),
            ],
        )

    async def _fake_orchestrate_search(
        req: SearchRequest,
        **kwargs,
    ):
        assert isinstance(req, SearchRequest)
        assert req.city == "Baku"

        return {
            "need_clarification": False,
            "results": [
                {
                    "title": "Large Family Apartment",
                }
            ],
        }

    async def _fake_route(**kwargs):
        return ConversationActionDecision(
            action=ConversationAction.START_SEARCH,
            reason="The user starts a search.",
        )

    monkeypatch.setattr(
        conversation_flow,
        "build_search_request_adk_async",
        _fake_build_search_request,
    )
    monkeypatch.setattr(
        conversation_flow,
        "route_conversation_async",
        _fake_route,
    )
    monkeypatch.setattr(
        conversation_flow,
        "orchestrate_search_request",
        _fake_orchestrate_search,
    )

    out = await conversation_flow.handle_user_message(
        "any query"
    )

    assert out["need_clarification"] is False
    assert (
        out["results"][0]["title"]
        == "Large Family Apartment"
    )
    assert out["state"]["city"] == "Baku"
    assert out["state"]["constraints"]


@pytest.mark.asyncio
async def test_conversation_flow_followup_updates_existing_state(
    monkeypatch,
):
    previous_state = SearchRequest(
        city="Baku",
        check_in=date(2026, 4, 20),
        check_out=date(2026, 4, 26),
        constraints=[kitchen_constraint()],
    )

    async def _fake_route(**kwargs):
        return ConversationActionDecision(
            action=ConversationAction.UPDATE_SEARCH,
            reason="The user updates the active search.",
        )

    async def _fake_update(
        prev_state,
        msg,
        trace=None,
    ):
        return SearchRequest(
            city=prev_state.city,
            check_in=prev_state.check_in,
            check_out=prev_state.check_out,
            constraints=[kitchen_constraint()],
        )

    async def _fake_orchestrate_search(
        req: SearchRequest,
        **kwargs,
    ):
        assert isinstance(req, SearchRequest)

        return {
            "need_clarification": False,
            "results": [{"title": "OK"}],
        }

    monkeypatch.setattr(
        conversation_flow,
        "route_conversation_async",
        _fake_route,
    )
    monkeypatch.setattr(
        conversation_flow,
        "update_search_state_async",
        _fake_update,
    )
    monkeypatch.setattr(
        conversation_flow,
        "orchestrate_search_request",
        _fake_orchestrate_search,
    )

    out = await conversation_flow.handle_user_message(
        "update",
        previous_state=previous_state,
    )

    assert out["state"]["city"] == "Baku"
    assert out["state"]["constraints"]


@pytest.mark.asyncio
async def test_conversation_flow_listing_question_does_not_mutate_state(
    monkeypatch,
):
    previous_state = SearchRequest(
        city="Baku",
        constraints=[beds_constraint()],
    )

    async def _fake_route(**kwargs):
        return ConversationActionDecision(
            action=ConversationAction.LISTING_QUESTION,
            reason="The user asks about a shown listing.",
        )

    async def _fake_answer(**kwargs):
        return {
            "need_clarification": False,
            "response_type": "listing_question",
            "state": previous_state.model_dump(
                mode="json"
            ),
        }

    monkeypatch.setattr(
        conversation_flow,
        "route_conversation_async",
        _fake_route,
    )
    monkeypatch.setattr(
        conversation_flow,
        "_answer_listing_question",
        _fake_answer,
    )

    out = await conversation_flow.handle_user_message(
        "question",
        previous_state=previous_state,
    )

    assert out["response_type"] == "listing_question"
    assert out["state"]["constraints"]
    assert "telemetry" in out
    assert out["telemetry"] is not None


@pytest.mark.asyncio
async def test_conversation_flow_new_search_rebuilds_state(
    monkeypatch,
):
    previous_state = SearchRequest(city="Baku")

    async def _fake_route(**kwargs):
        return ConversationActionDecision(
            action=ConversationAction.START_SEARCH,
            reason="The user explicitly starts a new search.",
        )

    async def _fake_build(
        msg,
        trace=None,
        step=None,
    ):
        return SearchRequest(
            city="Paris",
            constraints=[],
        )

    monkeypatch.setattr(
        conversation_flow,
        "route_conversation_async",
        _fake_route,
    )
    monkeypatch.setattr(
        conversation_flow,
        "build_search_request_adk_async",
        _fake_build,
    )

    out = await conversation_flow.handle_user_message(
        "new",
        previous_state=previous_state,
    )

    assert out["state"]["city"] == "Paris"
    assert (
        out["conversation_action"]
        == ConversationAction.START_SEARCH.value
    )


@pytest.mark.asyncio
async def test_conversation_flow_general_chat_returns_previous_state(
    monkeypatch,
):
    previous_state = SearchRequest(
        city="Baku",
        constraints=[beds_constraint()],
    )

    async def _fake_route(**kwargs):
        return ConversationActionDecision(
            action=ConversationAction.GENERAL_CHAT,
            reason="The user thanks the assistant.",
        )

    monkeypatch.setattr(
        conversation_flow,
        "route_conversation_async",
        _fake_route,
    )

    out = await conversation_flow.handle_user_message(
        "thanks",
        previous_state=previous_state,
    )

    assert out["response_type"] == "other"
    assert out["state"]["constraints"]
    assert "telemetry" in out
    assert out["telemetry"] is not None
    assert (
        out["conversation_action"]
        == ConversationAction.GENERAL_CHAT.value
    )


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
        "orchestrate_search_request",
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
    assert "telemetry" in result
    assert result["telemetry"] is not None
    assert "conversation_action" not in result

    update_mock.assert_not_awaited()
    search_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_conversation_flow_clarification_contains_telemetry(
    monkeypatch,
):
    async def _fake_build_search_request(
        user_message: str,
        trace=None,
        step=None,
    ) -> SearchRequest:
        return SearchRequest(
            city=None,
            constraints=[kitchen_constraint()],
        )

    def _fake_resolve_required_search_context(state):
        return SimpleNamespace(
            need_clarification=True,
            questions=[
                "Which city should I search in?"
            ],
        )

    async def _fake_route(**kwargs):
        return ConversationActionDecision(
            action=ConversationAction.START_SEARCH,
            reason="The user starts a search.",
        )

    search_mock = AsyncMock()

    monkeypatch.setattr(
        conversation_flow,
        "build_search_request_adk_async",
        _fake_build_search_request,
    )
    monkeypatch.setattr(
        conversation_flow,
        "route_conversation_async",
        _fake_route,
    )
    monkeypatch.setattr(
        conversation_flow,
        "resolve_required_search_context",
        _fake_resolve_required_search_context,
    )
    monkeypatch.setattr(
        conversation_flow,
        "orchestrate_search_request",
        search_mock,
    )

    result = await conversation_flow.handle_user_message(
        user_message=(
            "Find me an apartment with a kitchen"
        ),
    )

    assert result["need_clarification"] is True
    assert result["questions"] == [
        "Which city should I search in?"
    ]
    assert "telemetry" in result
    assert result["telemetry"] is not None

    search_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_first_turn_general_chat_does_not_search(
    monkeypatch,
):
    async def _fake_route(**kwargs):
        return ConversationActionDecision(
            action=ConversationAction.GENERAL_CHAT,
            reason="The user is greeting the assistant.",
        )

    build_mock = AsyncMock()
    search_mock = AsyncMock()

    monkeypatch.setattr(
        conversation_flow,
        "route_conversation_async",
        _fake_route,
    )
    monkeypatch.setattr(
        conversation_flow,
        "build_search_request_adk_async",
        build_mock,
    )
    monkeypatch.setattr(
        conversation_flow,
        "orchestrate_search_request",
        search_mock,
    )

    result = await conversation_flow.handle_user_message(
        user_message="Hello",
    )

    assert result["response_type"] == "other"
    assert result["state"] is None

    build_mock.assert_not_awaited()
    search_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_update_without_existing_state_starts_search(
    monkeypatch,
):
    async def _fake_route(**kwargs):
        return ConversationActionDecision(
            action=ConversationAction.UPDATE_SEARCH,
            reason=(
                "The model interpreted the message "
                "as an update."
            ),
        )

    build_mock = AsyncMock(
        return_value=SearchRequest(
            city="Baku",
            check_in=date(2026, 4, 20),
            check_out=date(2026, 4, 25),
            constraints=[],
        )
    )

    search_mock = AsyncMock(
        return_value={
            "need_clarification": False,
            "results": [],
        }
    )

    update_mock = AsyncMock()

    monkeypatch.setattr(
        conversation_flow,
        "route_conversation_async",
        _fake_route,
    )
    monkeypatch.setattr(
        conversation_flow,
        "build_search_request_adk_async",
        build_mock,
    )
    monkeypatch.setattr(
        conversation_flow,
        "update_search_state_async",
        update_mock,
    )
    monkeypatch.setattr(
        conversation_flow,
        "orchestrate_search_request",
        search_mock,
    )

    result = await conversation_flow.handle_user_message(
        user_message="Find an apartment in Baku",
    )

    build_mock.assert_awaited_once()
    update_mock.assert_not_awaited()
    search_mock.assert_awaited_once()

    search_args = search_mock.await_args.args
    assert search_args
    assert isinstance(search_args[0], SearchRequest)

    assert (
        result["parsed_intent"]["router"][
            "effective_action"
        ]
        == "start_search"
    )

    assert (
        result["conversation_action"]
        == ConversationAction.START_SEARCH.value
    )