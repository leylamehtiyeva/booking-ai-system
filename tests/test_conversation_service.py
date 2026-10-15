"""
Hermetic integration tests for the application service (P0-4):
process_turn(session, message) -> TurnResult.

Real fixture retrieval, matching, ranking and deterministic rendering;
only the LLM boundaries are faked (router, intent extraction/update,
factual extraction, response wording). Single turns only - multi-turn
chains are P0-6.
"""

from __future__ import annotations

import ast
import json
from datetime import date
from pathlib import Path

import pytest

from app.agents.intent_router_agent import IntentRoute
from app.config import clock
from app.llm.call_policy import LlmCallError
from app.logic import conversation_flow, intent_router, intent_update
from app.logic.conversation_response_llm import ConversationResponseGenerationResult
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
from app.schemas.intent_patch import SearchIntentPatch
from app.schemas.result_query import ResultQuery
from app.schemas.session import ConversationSession
from app.schemas.turn import TurnConfig, TurnResult
from app.services import conversation_service
from app.services.conversation_service import (
    build_recent_conversation_messages,
    process_turn,
)

CONFIG = TurnConfig(persist_telemetry=False, include_debug=True)
TODAY = date(2026, 3, 1)


# --- fakes -------------------------------------------------------------------


def _decision(action, *, scope=ResultScope.NOT_APPLICABLE, target="", status=None, reason=None):
    return ConversationActionDecision(
        action=action,
        reason="test",
        decision_status=status or ConversationDecisionStatus.RESOLVED,
        clarification_reason=reason or ClarificationReason.NONE,
        result_scope=scope,
        target_result_id=target,
    )


def _wifi() -> UserConstraint:
    return UserConstraint(
        raw_text="WiFi",
        normalized_text="wifi",
        priority=ConstraintPriority.MUST,
        category=ConstraintCategory.AMENITY,
        mapping_status=ConstraintMappingStatus.KNOWN,
        mapped_fields=[Field.WIFI],
        evidence_strategy=EvidenceStrategy.STRUCTURED,
    )


@pytest.fixture
def fake_llm(monkeypatch):
    """Scriptable LLM boundaries; response wording falls back by default."""

    class FakeLlm:
        def route(self, decision_or_exc):
            async def fake(**kwargs):
                if isinstance(decision_or_exc, BaseException):
                    raise decision_or_exc
                return decision_or_exc

            monkeypatch.setattr(conversation_flow, "route_conversation_async", fake)

        def intent(self, intent_or_exc):
            async def fake(user_text, **kwargs):
                if isinstance(intent_or_exc, BaseException):
                    raise intent_or_exc
                return intent_or_exc

            monkeypatch.setattr(intent_router, "route_intent_adk_async", fake)

        def patch(self, patch_or_exc):
            async def fake(previous_state, user_message, **kwargs):
                if isinstance(patch_or_exc, BaseException):
                    raise patch_or_exc
                return patch_or_exc

            monkeypatch.setattr(intent_update, "route_intent_update_patch_async", fake)

        def factual(self, query: ResultQuery):
            async def fake(user_message, **kwargs):
                return query

            monkeypatch.setattr(conversation_flow, "extract_factual_constraints_async", fake)

        def response_llm(self, text: str):
            async def fake(response_input, *, trace=None):
                return ConversationResponseGenerationResult(text=text, source="llm")

            monkeypatch.setattr(conversation_service, "generate_conversation_response_with_llm", fake)

    # Default: the real response layer, whose model build fails offline ->
    # deterministic_fallback (exercises the real fallback path).
    def no_model(*args, **kwargs):
        raise LlmCallError("configuration_error", stage="conversation_response_generation")

    monkeypatch.setattr("app.logic.conversation_response_llm.build_adk_model", no_model)
    return FakeLlm()


async def _turn(session: ConversationSession, message: str) -> TurnResult:
    with clock.use_reference_date(TODAY):
        return await process_turn(session, message, config=CONFIG)


async def _searched_session(fake_llm) -> ConversationSession:
    fake_llm.route(_decision(ConversationAction.START_SEARCH))
    fake_llm.intent(
        IntentRoute(
            city="Baku",
            check_in="2026-04-10",
            check_out="2026-04-15",
            property_types=["apartment"],
            constraints=[_wifi()],
        )
    )
    turn = await _turn(ConversationSession(), "apartment in Baku April 10-15 with WiFi")
    return turn.session


# Keys that only exist on ListingRaw, server session or debug records.
# ("rooms" is excluded: SearchRequest.rooms is the requested room count.)
INTERNAL_KEYS = {
    "raw", "facilities", "policies", "highlights", "fine_print", "description",
    "available_dates", "address", "max_occupancy",
    "session", "debug", "shown_result_set", "messages",
    "parsed_intent", "answer_payload", "debug_selection", "telemetry_log_info",
    "llm", "external", "calls",
}


def _all_keys(value) -> set[str]:
    if isinstance(value, dict):
        return set(value) | set().union(*(_all_keys(v) for v in value.values()))
    if isinstance(value, list):
        return set().union(*(_all_keys(v) for v in value))
    return set()


def _assert_public_payload_is_clean(turn: TurnResult) -> None:
    payload = turn.model_dump(mode="json")
    assert _all_keys(payload) & INTERNAL_KEYS == set()
    serialized = json.dumps(payload)
    for secret_like in ("Traceback", "offline-test-placeholder", "token="):
        assert secret_like not in serialized


# --- scenarios ---------------------------------------------------------------


async def test_start_search(fake_llm):
    fake_llm.route(_decision(ConversationAction.START_SEARCH))
    fake_llm.intent(
        IntentRoute(
            city="Baku",
            check_in="2026-04-10",
            check_out="2026-04-15",
            property_types=["apartment"],
            constraints=[_wifi()],
        )
    )
    session = ConversationSession()

    turn = await _turn(session, "apartment in Baku April 10-15 with WiFi")

    assert turn.outcome_kind == "search"
    assert turn.conversation_action == ConversationAction.START_SEARCH
    assert turn.failure is None
    assert turn.inspector.response_source == "deterministic_fallback"
    assert turn.inspector.search_status == "results"
    assert turn.listing_cards
    assert [c.position for c in turn.listing_cards] == list(range(1, len(turn.listing_cards) + 1))
    assert [c.result_id for c in turn.listing_cards] == [r.result_id for r in turn.inspector.results]
    assert [i.result_id for i in turn.session.shown_result_set.items] == [
        c.result_id for c in turn.listing_cards
    ]
    assert turn.session.search_request.city == "Baku"
    assert turn.session.search_request.check_in == date(2026, 4, 10)
    assert [m.role for m in turn.session.messages] == ["user", "assistant"]
    assert turn.session.messages[-1].content == turn.assistant_message
    assert turn.session.session_id == session.session_id == turn.session_id
    assert turn.telemetry.trace_id
    _assert_public_payload_is_clean(turn)


async def test_search_answer_from_llm_gets_result_links(fake_llm):
    fake_llm.response_llm("Here are two apartments.")
    fake_llm.route(_decision(ConversationAction.START_SEARCH))
    fake_llm.intent(IntentRoute(city="Baku", check_in="2026-04-10", check_out="2026-04-15"))

    turn = await _turn(ConversationSession(), "Baku April 10-15")

    assert turn.inspector.response_source == "llm"
    first, _, links = turn.assistant_message.partition("\n\n")
    assert first == "Here are two apartments."
    assert links.splitlines()[0].startswith("1. [")


async def test_update_search_replaces_state_and_snapshot(fake_llm):
    session = await _searched_session(fake_llm)
    fake_llm.route(_decision(ConversationAction.UPDATE_SEARCH))
    fake_llm.patch(SearchIntentPatch(set_check_in="2026-04-11", set_check_out="2026-04-14"))

    turn = await _turn(session, "make it April 11-14")

    assert turn.outcome_kind == "search"
    assert turn.conversation_action == ConversationAction.UPDATE_SEARCH
    assert turn.session.search_request.check_in == date(2026, 4, 11)
    assert turn.session.search_request.constraints == session.search_request.constraints
    assert turn.session.shown_result_set.result_set_id != session.shown_result_set.result_set_id
    assert len(turn.session.messages) == 4


async def test_update_no_op_keeps_snapshot(fake_llm):
    session = await _searched_session(fake_llm)
    fake_llm.route(_decision(ConversationAction.UPDATE_SEARCH))
    fake_llm.patch(SearchIntentPatch())

    turn = await _turn(session, "with WiFi please")

    assert turn.outcome_kind == "informational"
    assert turn.inspector.response_source == "deterministic"
    assert turn.session.search_request == session.search_request
    assert turn.session.shown_result_set == session.shown_result_set
    assert turn.listing_cards == []


async def test_listing_question_is_read_only(fake_llm):
    session = await _searched_session(fake_llm)
    target = session.shown_result_set.items[1]
    fake_llm.route(
        _decision(
            ConversationAction.LISTING_QUESTION,
            scope=ResultScope.SPECIFIC_RESULT,
            target=target.result_id,
        )
    )
    fake_llm.factual(ResultQuery(constraints=[_wifi()]))

    turn = await _turn(session, "does the second one have WiFi?")

    assert turn.outcome_kind == "listing_query"
    assert turn.inspector.response_source == "deterministic"
    assert [i.match.result_id for i in turn.inspector.listing_query] == [target.result_id]
    assert turn.inspector.routing.target_result_id == target.result_id
    assert target.listing.name in turn.assistant_message
    assert turn.listing_cards == []
    assert turn.session.search_request == session.search_request
    assert turn.session.shown_result_set == session.shown_result_set
    _assert_public_payload_is_clean(turn)


async def test_clarification_keeps_domain_approved_partial_state(fake_llm):
    fake_llm.route(_decision(ConversationAction.START_SEARCH))
    fake_llm.intent(IntentRoute(city="Baku"))

    turn = await _turn(ConversationSession(), "apartment in Baku")

    assert turn.outcome_kind == "clarification"
    assert "dates" in turn.assistant_message.lower()
    assert turn.session.search_request.city == "Baku"
    assert turn.session.shown_result_set is None


async def test_router_abstention_is_clarification_and_read_only(fake_llm):
    session = await _searched_session(fake_llm)
    fake_llm.route(
        _decision(
            ConversationAction.LISTING_QUESTION,
            status=ConversationDecisionStatus.NEEDS_CLARIFICATION,
            reason=ClarificationReason.AMBIGUOUS_REFERENCE,
        )
    )

    turn = await _turn(session, "what about that one?")

    assert turn.outcome_kind == "clarification"
    assert turn.session.search_request == session.search_request
    assert turn.session.shown_result_set == session.shown_result_set


async def test_general_chat(fake_llm):
    session = await _searched_session(fake_llm)
    fake_llm.route(_decision(ConversationAction.GENERAL_CHAT))

    turn = await _turn(session, "thanks!")

    assert turn.outcome_kind == "general_chat"
    assert "help you search" in turn.assistant_message
    assert turn.session.search_request == session.search_request
    assert turn.session.shown_result_set == session.shown_result_set


# --- failures ------------------------------------------------------------------


async def test_provider_failure_is_typed_and_preserves_state(fake_llm):
    session = await _searched_session(fake_llm)
    fake_llm.route(_decision(ConversationAction.UPDATE_SEARCH))
    fake_llm.patch(LlmCallError("rate_limited", stage="intent_update"))

    turn = await _turn(session, "cheaper please")

    assert turn.outcome_kind == "failure"
    assert turn.conversation_action == ConversationAction.UPDATE_SEARCH
    assert turn.failure.model_dump() == {"stage": "search_state_update", "code": "rate_limited"}
    assert turn.inspector.response_source == "deterministic"
    assert turn.session.search_request == session.search_request
    assert turn.session.shown_result_set == session.shown_result_set
    _assert_public_payload_is_clean(turn)


async def test_routing_failure_has_no_invented_action(fake_llm):
    session = await _searched_session(fake_llm)
    fake_llm.route(ConversationRoutingError("timeout"))

    turn = await _turn(session, "hello?")

    assert turn.outcome_kind == "failure"
    assert turn.conversation_action is None
    assert turn.inspector.routing is None
    assert turn.inspector.effective_action is None
    assert turn.inspector.response_source == "direct"
    assert turn.failure.model_dump() == {"stage": "conversation_routing", "code": "timeout"}
    assert "has not been changed" in turn.assistant_message
    assert turn.session.search_request == session.search_request
    assert turn.session.shown_result_set == session.shown_result_set
    _assert_public_payload_is_clean(turn)


async def test_response_generation_failure_falls_back_deterministically(fake_llm):
    fake_llm.route(_decision(ConversationAction.GENERAL_CHAT))

    turn = await _turn(ConversationSession(), "hi")

    assert turn.inspector.response_source == "deterministic_fallback"
    assert turn.telemetry.failed_llm_calls == 1
    assert turn.assistant_message.startswith("Hello!")


async def test_programming_error_propagates_without_committing(fake_llm):
    session = await _searched_session(fake_llm)
    before = session.model_dump(mode="json")
    fake_llm.route(_decision(ConversationAction.START_SEARCH))
    fake_llm.intent(KeyError("bug"))

    with pytest.raises(KeyError):
        await _turn(session, "new search")

    assert session.model_dump(mode="json") == before


async def test_input_session_is_never_mutated(fake_llm):
    session = await _searched_session(fake_llm)
    before = session.model_dump(mode="json")
    fake_llm.route(_decision(ConversationAction.UPDATE_SEARCH))
    fake_llm.patch(SearchIntentPatch(set_check_in="2026-04-11", set_check_out="2026-04-14"))

    turn = await _turn(session, "April 11-14")

    assert session.model_dump(mode="json") == before
    assert turn.session is not session


# --- history and telemetry --------------------------------------------------------


async def test_previous_history_reaches_response_layer_without_current_message(fake_llm, monkeypatch):
    captured = {}

    async def capture(response_input, *, trace=None):
        captured["history"] = [m.content for m in response_input.recent_messages]
        return ConversationResponseGenerationResult(text="ok", source="llm")

    monkeypatch.setattr(conversation_service, "generate_conversation_response_with_llm", capture)
    fake_llm.route(_decision(ConversationAction.GENERAL_CHAT))
    session = ConversationSession.model_validate(
        {
            "messages": [
                {"role": "user", "content": "Find an apartment in Baku"},
                {"role": "assistant", "content": "Which dates would you like?"},
            ]
        }
    )

    await _turn(session, "Thanks")

    assert captured["history"] == ["Find an apartment in Baku", "Which dates would you like?"]


async def test_session_history_is_bounded(fake_llm):
    fake_llm.route(_decision(ConversationAction.GENERAL_CHAT))
    session = ConversationSession.model_validate(
        {"messages": [{"role": "user", "content": f"m{i}"} for i in range(20)]}
    )

    turn = await _turn(session, "hi")

    assert len(turn.session.messages) == 20
    assert turn.session.messages[-2].content == "hi"


async def test_telemetry_is_persisted_with_result_summary(fake_llm, monkeypatch):
    captured = {}

    def fake_save(**kwargs):
        captured.update(kwargs)
        return {"log_file": "fake.jsonl", "attempt_number": 1}

    monkeypatch.setattr(conversation_service, "save_telemetry_record", fake_save)
    fake_llm.route(_decision(ConversationAction.START_SEARCH))
    fake_llm.intent(IntentRoute(city="Baku", check_in="2026-04-10", check_out="2026-04-15"))

    with clock.use_reference_date(TODAY):
        turn = await process_turn(
            ConversationSession(),
            "Baku April 10-15",
            config=TurnConfig(persist_telemetry=True, include_debug=True),
        )

    assert captured["result_summary"]["status"] == "results"
    assert captured["result_summary"]["results_count"] == len(turn.listing_cards)
    # The persisted trace includes the response-generation step.
    steps = [s["name"] for s in captured["telemetry"]["latency_ms"]["steps"]]
    assert "conversation_response_generation" in steps
    assert turn.debug.telemetry_log_info == {"log_file": "fake.jsonl", "attempt_number": 1}


async def test_debug_is_only_filled_when_requested(fake_llm):
    fake_llm.route(_decision(ConversationAction.GENERAL_CHAT))

    with clock.use_reference_date(TODAY):
        turn = await process_turn(ConversationSession(), "hi", config=TurnConfig(persist_telemetry=False))

    assert turn.debug is None


# --- API readiness ---------------------------------------------------------------


def test_contracts_produce_json_schemas():
    from app.schemas.session import ConversationSession as Session

    session_schema = Session.model_json_schema()
    turn_schema = TurnResult.model_json_schema(mode="serialization")

    assert {"session_id", "search_request", "shown_result_set", "messages"} <= set(
        session_schema["properties"]
    )
    assert "session" not in turn_schema["properties"]
    assert "debug" not in turn_schema["properties"]
    assert {
        "session_id",
        "assistant_message",
        "outcome_kind",
        "conversation_action",
        "listing_cards",
        "inspector",
        "failure",
        "telemetry",
    } <= set(turn_schema["properties"])
    assert "ListingRaw" not in turn_schema["$defs"]
    assert "ShownResultSet" not in turn_schema["$defs"]


async def test_session_round_trips_through_json(fake_llm):
    session = await _searched_session(fake_llm)

    restored = ConversationSession.model_validate_json(session.model_dump_json())

    assert restored == session


def test_application_layer_does_not_import_streamlit():
    offenders = []
    for path in Path("app").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            if any(n == "streamlit" or n.startswith(("streamlit.", "ui.")) or n == "ui" for n in names):
                offenders.append(str(path))
    assert offenders == []


# --- history helper (moved from ui/services/chat_handler.py) ---------------------


def test_build_recent_conversation_messages_strips_ui_debug_data():
    history = build_recent_conversation_messages(
        [
            {"role": "user", "content": "Find an apartment in Baku"},
            {
                "role": "assistant",
                "content": "Which dates?",
                "debug_data": {"telemetry": {"secret": "debug"}},
            },
        ]
    )

    assert [(m.role, m.content) for m in history] == [
        ("user", "Find an apartment in Baku"),
        ("assistant", "Which dates?"),
    ]
    assert not hasattr(history[1], "debug_data")


def test_build_recent_conversation_messages_keeps_only_latest_messages():
    history = build_recent_conversation_messages(
        [{"role": "user", "content": f"message-{i}"} for i in range(10)],
        limit=4,
    )

    assert [m.content for m in history] == ["message-6", "message-7", "message-8", "message-9"]


def test_build_recent_conversation_messages_ignores_invalid_messages():
    history = build_recent_conversation_messages(
        [
            {"role": "system", "content": "internal prompt"},
            {"role": "assistant", "content": ""},
            {"role": "user", "content": "Hello"},
        ]
    )

    assert [(m.role, m.content) for m in history] == [("user", "Hello")]


async def test_invalid_fallback_response_never_reaches_the_public_payload(fake_llm, monkeypatch):
    """Raw LLM output from an unparseable textual-fallback answer stays internal."""
    from types import SimpleNamespace

    from app.logic import constraint_evidence_resolution

    class Models:
        def generate_content(self, **kwargs):
            return SimpleNamespace(text="not json RAW-PROVIDER-OUTPUT-123", usage_metadata=None)

    monkeypatch.setattr(
        constraint_evidence_resolution,
        "_gemini_client",
        lambda: SimpleNamespace(models=Models()),
    )
    fake_llm.route(_decision(ConversationAction.START_SEARCH))
    fake_llm.intent(
        IntentRoute(
            city="Baku",
            check_in="2026-04-10",
            check_out="2026-04-15",
            constraints=[
                UserConstraint(
                    raw_text="quiet street",
                    normalized_text="quiet street",
                    priority=ConstraintPriority.MUST,
                    category=ConstraintCategory.OTHER,
                    mapping_status=ConstraintMappingStatus.UNRESOLVED,
                    mapped_fields=[],
                    evidence_strategy=EvidenceStrategy.TEXTUAL,
                )
            ],
        )
    )

    turn = await _turn(ConversationSession(), "quiet apartment in Baku April 10-15")

    resolutions = [
        item
        for result in turn.inspector.results
        for item in result.constraint_resolution_results
    ]
    assert resolutions and {item.decision for item in resolutions} == {"UNCERTAIN"}
    assert "RAW-PROVIDER-OUTPUT" not in json.dumps(turn.model_dump(mode="json"))
    _assert_public_payload_is_clean(turn)
