from datetime import date
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.logic import conversation_flow
from app.logic import constraint_evidence_resolution
from app.logic.conversation_router import ConversationRoutingError
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
from app.schemas.fields import Field
from app.schemas.listing import ListingRaw
from app.schemas.query import SearchRequest
from app.schemas.result_query import ResultQuery
from app.schemas.search_response import (
    NormalizedSearchResponse,
    NormalizedSearchResult,
    SearchStatus,
)
from app.schemas.shown_result_set import ShownResult, ShownResultSet


@pytest.fixture(autouse=True)
def _stub_textual_fallback(monkeypatch):
    """
    LISTING_QUESTION's structured-UNCERTAIN -> textual LLM fallback is
    exercised with dedicated mocks in tests/test_result_query_matching.py
    (including real is_constraint_fallback_eligible behavior). These
    conversation_flow.py tests only verify orchestration wiring, so the
    actual Gemini call is stubbed out here - kept UNCERTAIN, matching the
    final value every existing test here already expected before the
    fallback layer existed - to keep this file fast/deterministic and
    avoid real API calls on every test run.
    """

    async def _fake(req, *, model=None, trace=None):
        return constraint_evidence_resolution.ConstraintResolutionResult(
            listing_id=req.listing_id,
            listing_title=req.listing_title,
            constraint_id=req.constraint_id,
            raw_text=req.raw_text,
            normalized_text=req.normalized_text,
            resolver_type="textual",
            priority=req.priority,
            mapped_fields=req.mapped_fields,
            decision="UNCERTAIN",
            resolution_status="uncertain",
            confidence=0.0,
            reason="No relevant evidence in the listing text.",
            evidence=[],
        )

    monkeypatch.setattr(
        constraint_evidence_resolution,
        "resolve_constraint_via_textual_evidence",
        _fake,
    )


def _resolved_kwargs() -> dict:
    """Every ConversationActionDecision field is required (no Python
    default) so it can pass Groq's strict structured-output schema - this
    fills in the "nothing special happened" values for tests that only
    care about action/reason."""
    return {
        "decision_status": ConversationDecisionStatus.RESOLVED,
        "clarification_reason": ClarificationReason.NONE,
        "result_scope": ResultScope.NOT_APPLICABLE,
        "target_result_id": "",
    }


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


def _known_constraint(field: Field, text: str) -> UserConstraint:
    return UserConstraint(
        raw_text=text,
        normalized_text=text,
        priority=ConstraintPriority.MUST,
        category=ConstraintCategory.AMENITY,
        mapping_status=ConstraintMappingStatus.KNOWN,
        mapped_fields=[field],
        evidence_strategy=EvidenceStrategy.STRUCTURED,
    )


def unresolved_constraint(text: str) -> UserConstraint:
    return UserConstraint(
        raw_text=text,
        normalized_text=text,
        priority=ConstraintPriority.MUST,
        category=ConstraintCategory.OTHER,
        mapping_status=ConstraintMappingStatus.UNRESOLVED,
        mapped_fields=[],
        evidence_strategy=EvidenceStrategy.NONE,
    )


def _shown_result(result_id: str, **listing_kwargs) -> ShownResult:
    return ShownResult(
        result_id=result_id,
        listing=ListingRaw(id=result_id, name=f"Hotel {result_id}", **listing_kwargs),
    )


def three_hotel_shown_result_set() -> ShownResultSet:
    return ShownResultSet(
        result_set_id="rs-parking",
        items=[
            _shown_result("H1", facilities=[{"name": "Private parking"}]),
            _shown_result("H2", description="No parking available on site."),
            _shown_result("H3", description="A cozy hotel near the center."),
        ],
    )


def _listing_question_decision(
    *,
    result_scope: ResultScope = ResultScope.CURRENT_RESULTS,
    target_result_id: str = "",
) -> ConversationActionDecision:
    return ConversationActionDecision(
        action=ConversationAction.LISTING_QUESTION,
        reason="The user asks about shown listings.",
        decision_status=ConversationDecisionStatus.RESOLVED,
        clarification_reason=ClarificationReason.NONE,
        result_scope=result_scope,
        target_result_id=target_result_id,
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

        return (
            NormalizedSearchResponse(
                status=SearchStatus.RESULTS,
                results=[
                    NormalizedSearchResult(
                        result_id="apt-1",
                        title="Large Family Apartment",
                        score=0.0,
                    )
                ],
            ),
            ShownResultSet(result_set_id="rs-1", items=[]),
        )

    async def _fake_route(**kwargs):
        return ConversationActionDecision(
            action=ConversationAction.START_SEARCH,
            reason="The user starts a search.",
            **_resolved_kwargs(),
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

    assert out["status"] == "results"
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
            **_resolved_kwargs(),
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

        return (
            NormalizedSearchResponse(
                status=SearchStatus.RESULTS,
                results=[
                    NormalizedSearchResult(
                        result_id="ok-1",
                        title="OK",
                        score=0.0,
                    )
                ],
            ),
            ShownResultSet(result_set_id="rs-2", items=[]),
        )

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


def _patch_route(monkeypatch, decision: ConversationActionDecision) -> None:
    async def _fake_route(**kwargs):
        return decision

    monkeypatch.setattr(conversation_flow, "route_conversation_async", _fake_route)


def _patch_extraction(monkeypatch, result_query: ResultQuery, spy: list | None = None) -> None:
    async def _fake_extract(user_message, *, trace=None):
        if spy is not None:
            spy.append(user_message)
        return result_query

    monkeypatch.setattr(
        conversation_flow,
        "extract_factual_constraints_async",
        _fake_extract,
    )


@pytest.mark.asyncio
async def test_conversation_flow_listing_question_does_not_mutate_state(
    monkeypatch,
):
    previous_state = SearchRequest(
        city="Baku",
        constraints=[beds_constraint()],
    )
    shown_result_set = three_hotel_shown_result_set()

    _patch_route(monkeypatch, _listing_question_decision())
    _patch_extraction(
        monkeypatch,
        ResultQuery(constraints=[_known_constraint(Field.PARKING, "parking")]),
    )

    out = await conversation_flow.handle_user_message(
        "Which of these have parking?",
        previous_state=previous_state,
        shown_result_set=shown_result_set,
    )

    assert out["response_type"] == "listing_query_result"
    assert out["state"] == previous_state.model_dump(mode="json", exclude_none=True)
    assert out["state"]["constraints"]
    assert "telemetry" in out
    assert out["telemetry"] is not None
    # Invariant: LISTING_QUESTION never replaces the stored snapshot -
    # chat_handler.py only calls set_shown_result_set when this key is
    # present, and no new result_set_id is ever minted here.
    assert "shown_result_set" not in out


@pytest.mark.asyncio
async def test_listing_question_current_results_yes_no_uncertain(monkeypatch):
    shown_result_set = three_hotel_shown_result_set()

    _patch_route(monkeypatch, _listing_question_decision(result_scope=ResultScope.CURRENT_RESULTS))
    _patch_extraction(
        monkeypatch,
        ResultQuery(constraints=[_known_constraint(Field.PARKING, "parking")]),
    )

    out = await conversation_flow.handle_user_message(
        "Which of these have parking?",
        previous_state=None,
        shown_result_set=shown_result_set,
    )

    assert out["response_type"] == "listing_query_result"
    assert out["need_clarification"] is False

    matches = out["listing_query_result"]["matches"]
    by_id = {m["result_id"]: m["constraint_results"][0]["value"] for m in matches}
    assert by_id == {"H1": "YES", "H2": "NO", "H3": "UNCERTAIN"}

    presentation = out["listing_query_result"]["presentation"]
    assert [(p["result_id"], p["position"]) for p in presentation] == [
        ("H1", 1), ("H2", 2), ("H3", 3),
    ]
    assert all(p["title"] == "Hotel " + p["result_id"] for p in presentation)


@pytest.mark.asyncio
async def test_listing_question_specific_result_only_matches_target(monkeypatch):
    shown_result_set = ShownResultSet(
        result_set_id="rs-balcony",
        items=[
            _shown_result("H1", description="No balcony in this room."),
            _shown_result("H2", description="The room has a private balcony."),
        ],
    )

    _patch_route(
        monkeypatch,
        _listing_question_decision(
            result_scope=ResultScope.SPECIFIC_RESULT,
            target_result_id="H2",
        ),
    )
    _patch_extraction(
        monkeypatch,
        ResultQuery(constraints=[_known_constraint(Field.BALCONY, "balcony")]),
    )

    out = await conversation_flow.handle_user_message(
        "Does the second one have a balcony?",
        previous_state=None,
        shown_result_set=shown_result_set,
    )

    matches = out["listing_query_result"]["matches"]
    assert len(matches) == 1
    assert matches[0]["result_id"] == "H2"
    assert matches[0]["constraint_results"][0]["value"] == "YES"
    assert out["target_result_id"] == "H2"


@pytest.mark.asyncio
async def test_listing_question_multiple_constraints_kept_separate(monkeypatch):
    shown_result_set = ShownResultSet(
        result_set_id="rs-multi",
        items=[
            _shown_result(
                "H1",
                facilities=[{"name": "Private parking"}, {"name": "Free WiFi"}],
            ),
            _shown_result(
                "H2",
                description="No parking available on site. Free WiFi included.",
            ),
        ],
    )

    _patch_route(monkeypatch, _listing_question_decision())
    _patch_extraction(
        monkeypatch,
        ResultQuery(
            constraints=[
                _known_constraint(Field.PARKING, "parking"),
                _known_constraint(Field.WIFI, "wifi"),
            ]
        ),
    )

    out = await conversation_flow.handle_user_message(
        "Which of these have parking and Wi-Fi?",
        previous_state=None,
        shown_result_set=shown_result_set,
    )

    matches = out["listing_query_result"]["matches"]
    by_id = {m["result_id"]: m["constraint_results"] for m in matches}

    h1_by_field = {cr["field"]: cr["value"] for cr in by_id["H1"]}
    h2_by_field = {cr["field"]: cr["value"] for cr in by_id["H2"]}

    assert h1_by_field == {"parking": "YES", "wifi": "YES"}
    assert h2_by_field == {"parking": "NO", "wifi": "YES"}


@pytest.mark.asyncio
async def test_listing_question_uncertain_is_a_successful_result_not_clarification(
    monkeypatch,
):
    shown_result_set = ShownResultSet(
        result_set_id="rs-uncertain",
        items=[_shown_result("H3", description="A cozy hotel near the center.")],
    )

    _patch_route(monkeypatch, _listing_question_decision())
    _patch_extraction(
        monkeypatch,
        ResultQuery(constraints=[_known_constraint(Field.PARKING, "parking")]),
    )

    out = await conversation_flow.handle_user_message(
        "Does it have parking?",
        previous_state=None,
        shown_result_set=shown_result_set,
    )

    assert out["response_type"] == "listing_query_result"
    assert out["need_clarification"] is False
    assert out["listing_query_result"]["matches"][0]["constraint_results"][0]["value"] == "UNCERTAIN"


@pytest.mark.asyncio
async def test_listing_question_structured_uncertain_resolved_via_textual_fallback_end_to_end(
    monkeypatch,
):
    """
    End-to-end proof (through the real handle_user_message wiring, not
    just the result_query_matching.py unit tests) that a structured
    UNCERTAIN verdict for LISTING_QUESTION really is replaced by the
    existing textual LLM fallback's decision, and that ResultQueryMatch
    carries only the FINAL verdict.
    """
    shown_result_set = ShownResultSet(
        result_set_id="rs-fallback",
        items=[_shown_result("H3", description="A cozy hotel near the center.")],
    )

    async def _fake_fallback_yes(req, *, model=None, trace=None):
        return constraint_evidence_resolution.ConstraintResolutionResult(
            listing_id=req.listing_id,
            listing_title=req.listing_title,
            constraint_id=req.constraint_id,
            raw_text=req.raw_text,
            normalized_text=req.normalized_text,
            resolver_type="textual",
            priority=req.priority,
            mapped_fields=req.mapped_fields,
            decision="YES",
            resolution_status="matched",
            confidence=0.9,
            reason="Free parking is mentioned in the fine print.",
            evidence=[
                constraint_evidence_resolution.ConstraintEvidence(
                    snippet="Free parking available on request.",
                    source="fine_print",
                    path="listing.fine_print",
                )
            ],
        )

    monkeypatch.setattr(
        constraint_evidence_resolution,
        "resolve_constraint_via_textual_evidence",
        _fake_fallback_yes,
    )

    _patch_route(monkeypatch, _listing_question_decision())
    _patch_extraction(
        monkeypatch,
        ResultQuery(constraints=[_known_constraint(Field.PARKING, "parking")]),
    )

    out = await conversation_flow.handle_user_message(
        "Does it have parking?",
        previous_state=None,
        shown_result_set=shown_result_set,
    )

    constraint_result = out["listing_query_result"]["matches"][0]["constraint_results"][0]
    assert constraint_result["value"] == "YES"
    assert constraint_result["reason"] == "fallback_resolved"
    assert constraint_result["evidence"][0]["snippet"] == "Free parking available on request."


@pytest.mark.asyncio
async def test_listing_question_empty_shown_results_skips_matcher(monkeypatch):
    extraction_calls: list[str] = []

    _patch_route(monkeypatch, _listing_question_decision(result_scope=ResultScope.CURRENT_RESULTS))
    _patch_extraction(
        monkeypatch,
        ResultQuery(constraints=[_known_constraint(Field.PARKING, "parking")]),
        spy=extraction_calls,
    )

    out = await conversation_flow.handle_user_message(
        "Which of these have parking?",
        previous_state=None,
        shown_result_set=ShownResultSet(result_set_id="rs-empty", items=[]),
    )

    assert out["need_clarification"] is True
    assert out["questions"]
    assert "listing_query_result" not in out
    # The paid factual-extraction LLM call must never run when there is
    # nothing to check against.
    assert extraction_calls == []


@pytest.mark.asyncio
async def test_listing_question_zero_constraints_skips_matcher(monkeypatch):
    shown_result_set = ShownResultSet(
        result_set_id="rs-romantic",
        items=[_shown_result("H1", description="A charming romantic hideaway.")],
    )

    _patch_route(monkeypatch, _listing_question_decision())
    _patch_extraction(monkeypatch, ResultQuery(constraints=[]))

    out = await conversation_flow.handle_user_message(
        "Is it romantic?",
        previous_state=None,
        shown_result_set=shown_result_set,
    )

    assert out["need_clarification"] is False
    assert out["response_type"] == "listing_query_unsupported"
    assert "listing_query_result" not in out


@pytest.mark.asyncio
async def test_listing_question_never_calls_search_orchestration(monkeypatch):
    """
    STEP 4/11 invariant: LISTING_QUESTION must never touch persistent
    state or trigger retrieval. orchestrate_search_request/
    update_search_state_async are monkeypatched to hard-fail if called at
    all - not just asserted "not called" after the fact.
    """
    shown_result_set = three_hotel_shown_result_set()

    async def _must_not_be_called(*args, **kwargs):
        raise AssertionError("must not be called for LISTING_QUESTION")

    monkeypatch.setattr(
        conversation_flow,
        "orchestrate_search_request",
        _must_not_be_called,
    )
    monkeypatch.setattr(
        conversation_flow,
        "update_search_state_async",
        _must_not_be_called,
    )

    _patch_route(monkeypatch, _listing_question_decision())
    _patch_extraction(
        monkeypatch,
        ResultQuery(constraints=[_known_constraint(Field.PARKING, "parking")]),
    )

    out = await conversation_flow.handle_user_message(
        "Which of these have parking?",
        previous_state=SearchRequest(city="Baku"),
        shown_result_set=shown_result_set,
    )

    assert out["response_type"] == "listing_query_result"


@pytest.mark.asyncio
async def test_conversation_flow_new_search_rebuilds_state(
    monkeypatch,
):
    previous_state = SearchRequest(city="Baku")

    async def _fake_route(**kwargs):
        return ConversationActionDecision(
            action=ConversationAction.START_SEARCH,
            reason="The user explicitly starts a new search.",
            **_resolved_kwargs(),
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
            **_resolved_kwargs(),
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
            **_resolved_kwargs(),
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

    # Search never executed -> no shown_result_set key -> chat_handler
    # will leave whatever snapshot was already stored untouched.
    assert "shown_result_set" not in result

    search_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_first_turn_general_chat_does_not_search(
    monkeypatch,
):
    async def _fake_route(**kwargs):
        return ConversationActionDecision(
            action=ConversationAction.GENERAL_CHAT,
            reason="The user is greeting the assistant.",
            **_resolved_kwargs(),
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
    assert "shown_result_set" not in result

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
            **_resolved_kwargs(),
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
        return_value=(
            NormalizedSearchResponse(
                status=SearchStatus.NO_RESULTS,
                results=[],
            ),
            ShownResultSet(result_set_id="rs-3", items=[]),
        )
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

    # Search actually executed (NO_RESULTS) -> shown_result_set key is
    # present, and empty, replacing whatever snapshot existed before.
    assert result["shown_result_set"] == {
        "result_set_id": "rs-3",
        "items": [],
    }

    assert (
        result["conversation_action"]
        == ConversationAction.START_SEARCH.value
    )